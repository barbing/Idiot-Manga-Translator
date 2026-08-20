# -*- coding: utf-8 -*-
"""Page-local cached rerender service for GUI-owned effective edits."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import io
import os
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Protocol
import uuid

from app.io.project_edit_store import ProjectEditCommitReceipt, ProjectEditStore
from app.project_edits.contracts import ProjectEdit, freeze_json, thaw_json
from app.render.typesetting_contracts import RenderLayerPlan

from .effective_render_plan import (
    EffectiveRenderLayerPlan,
    EffectiveRenderPlanError,
    MissingCleanedPageBaseError,
    project_effective_render_layers,
    render_layer_plan_payload,
)
from .fingerprints import canonical_sha256
from .layer_cache import (
    ParentLayerArtifact,
    ParentLayerCache,
    ParentLayerCacheError,
    ParentLayerCacheInputs,
    file_sha256,
    parent_layer_cache_root,
)
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    ProjectionIssueKind,
)
from .renderer_adapter import EffectiveLayerRendererAdapter

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


PAGE_RERENDER_SERVICE_VERSION = "page_rerender_service_v1"
PARENT_LAYER_REVISION_VERSION = "parent_layer_revision_v1"
RENDERED_PAGE_REVISION_VERSION = "rendered_page_revision_v1"


class RerenderMode(str, Enum):
    PREVIEW = "preview"
    COMMIT = "commit"


class RerenderStage(str, Enum):
    VALIDATING = "validating"
    PROJECTING = "projecting"
    CACHE_LOOKUP = "cache_lookup"
    RENDERING_PARENT = "rendering_parent"
    COMPOSING = "composing"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RerenderStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RerenderAvailabilityCode(str, Enum):
    READY = "ready"
    MISSING_BASE = "missing_base"
    BLOCKED = "blocked"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class PageRerenderPreflight:
    page_id: str
    code: RerenderAvailabilityCode
    ready: bool
    message: str
    canvas_size: tuple[int, int] = ()
    parent_ids: tuple[str, ...] = ()
    issue_kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CacheRejection:
    parent_id: str
    cache_key: str
    reason_code: str
    detail: str


class PageRerenderError(RuntimeError):
    """The rerender request failed before publishing project state."""


class CleanedPageBaseUnavailableError(PageRerenderError):
    """The selected immutable page substrate is unavailable or invalid."""


class CancellationProbe(Protocol):
    def is_cancelled(self) -> bool: ...


class PageRerenderCancellationToken:
    """Thread-safe cooperative cancellation at GUI-owned safe points."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class RerenderProgress:
    page_id: str
    stage: RerenderStage
    completed_parents: int
    total_parents: int
    parent_id: str = ""
    cache_hit: bool | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class PageRerenderRequest:
    snapshot: EffectivePageSnapshot
    automatic_parent_bundles: tuple[Any, ...]
    mode: RerenderMode = RerenderMode.PREVIEW
    pending_edits: tuple[ProjectEdit, ...] = ()
    expected_page_head_sha256: str = ""
    expected_global_head_sha256: str = ""
    output_revision_id: str = ""
    transaction_id: str = ""


@dataclass(frozen=True, slots=True)
class ParentLayerRevision:
    revision_id: str
    page_id: str
    parent_id: str
    cache_key: str
    asset: str
    content_sha256: str
    cache_manifest: str
    cache_manifest_sha256: str
    cleaned_base_revision_id: str
    cleaned_base_sha256: str
    effective_plan_fingerprint: str
    renderer_contract_fingerprint: str
    font_asset_fingerprint: str
    canvas_size: tuple[int, int]
    alpha_bounds: tuple[int, ...]
    cache_hit: bool
    render_audit: Any

    def to_artifact_record(self) -> dict[str, Any]:
        return {
            "catalog": "parent_layers",
            "parent_layer_revision_version": PARENT_LAYER_REVISION_VERSION,
            "revision_id": self.revision_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "provenance": "user_effective_rerender",
            "valid": True,
            "asset": self.asset,
            "content_sha256": self.content_sha256,
            "cache_key": self.cache_key,
            "cache_manifest": self.cache_manifest,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cleaned_base_revision_id": self.cleaned_base_revision_id,
            "cleaned_base_sha256": self.cleaned_base_sha256,
            "effective_plan_fingerprint": self.effective_plan_fingerprint,
            "renderer_contract_fingerprint": self.renderer_contract_fingerprint,
            "font_asset_fingerprint": self.font_asset_fingerprint,
            "canvas_size": list(self.canvas_size),
            "image_mode": "RGBA",
            "alpha_bounds": list(self.alpha_bounds),
            "render_audit": thaw_json(self.render_audit),
        }


@dataclass(frozen=True, slots=True)
class RenderedPageRevision:
    revision_id: str
    page_id: str
    asset: str
    content_sha256: str
    cleaned_base_revision_id: str
    cleaned_base_sha256: str
    effective_page_fingerprint: str
    parent_layer_revision_ids: tuple[str, ...]
    renderer_contract_fingerprint: str
    font_asset_fingerprint: str
    canvas_size: tuple[int, int]

    def to_artifact_record(self) -> dict[str, Any]:
        return {
            "catalog": "rendered_pages",
            "rendered_page_revision_version": RENDERED_PAGE_REVISION_VERSION,
            "revision_id": self.revision_id,
            "page_id": self.page_id,
            "provenance": "user_effective_rerender",
            "valid": True,
            "asset": self.asset,
            "content_sha256": self.content_sha256,
            "cleaned_base_revision_id": self.cleaned_base_revision_id,
            "cleaned_base_sha256": self.cleaned_base_sha256,
            "effective_page_fingerprint": self.effective_page_fingerprint,
            "parent_layer_revision_ids": list(self.parent_layer_revision_ids),
            "renderer_contract_fingerprint": self.renderer_contract_fingerprint,
            "font_asset_fingerprint": self.font_asset_fingerprint,
            "canvas_size": list(self.canvas_size),
            "image_mode": "RGBA",
        }


@dataclass(frozen=True, slots=True)
class PageRerenderReceipt:
    page_id: str
    mode: RerenderMode
    status: RerenderStatus
    output_path: str
    output_sha256: str
    rendered_page_revision: RenderedPageRevision | None
    parent_layer_revisions: tuple[ParentLayerRevision, ...]
    cache_hits: int
    cache_misses: int
    cache_rejections: tuple[CacheRejection, ...]
    renderer_ms: float
    composition_ms: float
    persistence_ms: float
    elapsed_ms: float
    commit_receipt: ProjectEditCommitReceipt | None
    cancellation_stage: str = ""


ProgressCallback = Callable[[RerenderProgress], None]


class _Cancelled(RuntimeError):
    def __init__(self, stage: RerenderStage) -> None:
        super().__init__(stage.value)
        self.stage = stage


class PageRerenderService:
    """Render effective edits without invoking any upstream pipeline owner."""

    def __init__(
        self,
        *,
        project_path: str,
        edit_store: ProjectEditStore | None = None,
        cache: ParentLayerCache | None = None,
        renderer_adapter: EffectiveLayerRendererAdapter | None = None,
        artifact_root: str | None = None,
    ) -> None:
        raw_project_path = str(project_path or "").strip()
        if not raw_project_path:
            raise ValueError("project_path is required")
        self.project_path = os.path.abspath(raw_project_path)
        self.project_directory = os.path.dirname(self.project_path) or os.getcwd()
        self.edit_store = edit_store
        self.cache = cache or ParentLayerCache(
            parent_layer_cache_root(self.project_path)
        )
        self._renderer_adapter = renderer_adapter
        self._renderer_adapter_lock = threading.RLock()
        self.artifact_root = os.path.abspath(
            artifact_root or rerender_artifact_root(self.project_path)
        )
        self._preview_lock = threading.RLock()
        self._preview_paths: dict[str, str] = {}

    @property
    def renderer_adapter(self) -> EffectiveLayerRendererAdapter:
        with self._renderer_adapter_lock:
            if self._renderer_adapter is None:
                self._renderer_adapter = EffectiveLayerRendererAdapter()
            return self._renderer_adapter

    def preflight(
        self,
        snapshot: EffectivePageSnapshot,
        automatic_parent_bundles: tuple[Any, ...],
    ) -> PageRerenderPreflight:
        """Return typed page-local availability without initializing fonts."""

        issue_kinds = tuple(sorted({issue.kind.value for issue in snapshot.issues}))
        if any(issue.kind is ProjectionIssueKind.CONFLICT for issue in snapshot.issues):
            return PageRerenderPreflight(
                page_id=snapshot.page_id,
                code=RerenderAvailabilityCode.CONFLICT,
                ready=False,
                message="Resolve conflicting edits before previewing this page.",
                issue_kinds=issue_kinds,
            )
        cleanup_issue = any(
            issue.domain == "cleanup"
            and issue.kind
            in {
                ProjectionIssueKind.MISSING_DEPENDENCY,
                ProjectionIssueKind.STALE_DEPENDENCY,
            }
            for issue in snapshot.issues
        )
        if cleanup_issue:
            return PageRerenderPreflight(
                page_id=snapshot.page_id,
                code=RerenderAvailabilityCode.MISSING_BASE,
                ready=False,
                message=(
                    "A compatible CleanedPageBase is required before rerendering."
                ),
                issue_kinds=issue_kinds,
            )
        if snapshot.issues:
            return PageRerenderPreflight(
                page_id=snapshot.page_id,
                code=RerenderAvailabilityCode.BLOCKED,
                ready=False,
                message="Resolve page edit issues before rerendering.",
                issue_kinds=issue_kinds,
            )
        try:
            _, _, canvas_size = self._validated_clean_base(snapshot)
            layers = project_effective_render_layers(
                snapshot,
                automatic_parent_bundles,
            )
        except (CleanedPageBaseUnavailableError, MissingCleanedPageBaseError):
            return PageRerenderPreflight(
                page_id=snapshot.page_id,
                code=RerenderAvailabilityCode.MISSING_BASE,
                ready=False,
                message=(
                    "The selected CleanedPageBase is missing, invalid, or incompatible."
                ),
                issue_kinds=issue_kinds,
            )
        except EffectiveRenderPlanError:
            return PageRerenderPreflight(
                page_id=snapshot.page_id,
                code=RerenderAvailabilityCode.BLOCKED,
                ready=False,
                message="The effective page cannot form a render plan.",
                issue_kinds=issue_kinds,
            )
        return PageRerenderPreflight(
            page_id=snapshot.page_id,
            code=RerenderAvailabilityCode.READY,
            ready=True,
            message="Ready to preview.",
            canvas_size=canvas_size,
            parent_ids=tuple(layer.parent_id for layer in layers),
            issue_kinds=issue_kinds,
        )

    def rerender(
        self,
        request: PageRerenderRequest,
        *,
        cancellation: CancellationProbe | None = None,
        progress: ProgressCallback | None = None,
    ) -> PageRerenderReceipt:
        started = time.perf_counter()
        output_path = ""
        output_sha256 = ""
        layer_revisions: list[ParentLayerRevision] = []
        cache_hits = 0
        cache_misses = 0
        cache_rejections: list[CacheRejection] = []
        renderer_ms = 0.0
        composition_ms = 0.0
        persistence_ms = 0.0
        rendered_revision: RenderedPageRevision | None = None
        total = 0
        try:
            self._emit(
                progress,
                request.snapshot.page_id,
                RerenderStage.VALIDATING,
                0,
                0,
                message="Validating selected CleanedPageBase",
            )
            self._check_cancel(cancellation, RerenderStage.VALIDATING)
            cleaned_path, cleaned_sha256, canvas_size = self._validated_clean_base(
                request.snapshot
            )
            self._validate_request(request)

            self._emit(
                progress,
                request.snapshot.page_id,
                RerenderStage.PROJECTING,
                0,
                0,
                message="Projecting effective render layers",
            )
            effective_layers = project_effective_render_layers(
                request.snapshot,
                request.automatic_parent_bundles,
            )
            parent_snapshots = {
                parent.parent_id: parent for parent in request.snapshot.parents
            }
            ordered_layers = sorted(
                effective_layers,
                key=lambda layer: (
                    layer.to_render_layer_plan().draw_order,
                    layer.layer_id,
                ),
            )
            total = len(ordered_layers)
            self._check_cancel(cancellation, RerenderStage.PROJECTING)

            with Image.open(cleaned_path) as source:
                composed_page = source.convert("RGBA")
            if composed_page.size != canvas_size:
                raise PageRerenderError("CleanedPageBase dimensions changed during render")

            for index, effective_layer in enumerate(ordered_layers):
                parent = parent_snapshots.get(effective_layer.parent_id)
                if parent is None:
                    raise PageRerenderError(
                        f"effective parent is unavailable: {effective_layer.parent_id}"
                    )
                plan = effective_layer.to_render_layer_plan()
                inputs = self._cache_inputs(
                    request.snapshot,
                    parent,
                    effective_layer,
                    plan,
                    cleaned_sha256,
                )
                self._check_cancel(cancellation, RerenderStage.CACHE_LOOKUP)
                self._emit(
                    progress,
                    request.snapshot.page_id,
                    RerenderStage.CACHE_LOOKUP,
                    index,
                    total,
                    parent_id=effective_layer.parent_id,
                    message="Checking parent-layer cache",
                )
                repair_corrupt = False
                try:
                    artifact = self.cache.load(
                        inputs.cache_key,
                        expected_canvas_size=canvas_size,
                    )
                except ParentLayerCacheError as exc:
                    artifact = None
                    repair_corrupt = True
                    cache_rejections.append(
                        CacheRejection(
                            parent_id=effective_layer.parent_id,
                            cache_key=inputs.cache_key,
                            reason_code=str(
                                getattr(exc, "code", "cache_contract_invalid")
                            ),
                            detail=str(exc),
                        )
                    )
                cache_hit = artifact is not None
                if cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    self._emit(
                        progress,
                        request.snapshot.page_id,
                        RerenderStage.RENDERING_PARENT,
                        index,
                        total,
                        parent_id=effective_layer.parent_id,
                        cache_hit=False,
                        message="Rendering changed parent",
                    )
                    self._check_cancel(cancellation, RerenderStage.RENDERING_PARENT)
                    rendered = self.renderer_adapter.render_isolated_layer(
                        cleaned_page_base_path=cleaned_path,
                        plan=plan,
                        working_directory=self.cache.root,
                    )
                    renderer_ms += float(rendered.elapsed_ms)
                    self._check_cancel(cancellation, RerenderStage.RENDERING_PARENT)
                    authoritative_audit = dict(rendered.authoritative_audit)
                    authoritative_audit["isolated_layer_extraction"] = dict(
                        rendered.extraction_audit
                    )
                    artifact = self.cache.store(
                        inputs=inputs,
                        surface=rendered.surface,
                        plan=rendered.plan,
                        layout=rendered.layout,
                        fit_report=rendered.fit_report,
                        layer_audit=authoritative_audit,
                        elapsed_ms=rendered.elapsed_ms,
                        repair_corrupt=repair_corrupt,
                    )

                if artifact is None:  # pragma: no cover - defensive
                    raise PageRerenderError("parent-layer cache returned no artifact")
                if artifact.inputs != inputs:
                    raise PageRerenderError(
                        "parent-layer cache returned a different input contract"
                    )
                composition_started = time.perf_counter()
                with Image.open(artifact.image_path) as layer_image:
                    layer = layer_image.convert("RGBA")
                if layer.size != composed_page.size:
                    raise PageRerenderError("parent layer has the wrong page canvas")
                composed_page.alpha_composite(layer)
                composition_ms += (time.perf_counter() - composition_started) * 1000.0
                layer_revision = self._layer_revision(
                    request.snapshot,
                    effective_layer,
                    artifact,
                    cache_hit=cache_hit,
                )
                layer_revisions.append(layer_revision)
                self._emit(
                    progress,
                    request.snapshot.page_id,
                    RerenderStage.COMPOSING,
                    index + 1,
                    total,
                    parent_id=effective_layer.parent_id,
                    cache_hit=cache_hit,
                    message="Composed current parent layer",
                )

            self._check_cancel(cancellation, RerenderStage.COMPOSING)
            output_bytes = _png_bytes(composed_page)
            output_sha256 = hashlib.sha256(output_bytes).hexdigest()
            output_path = self._publish_output(
                request,
                output_bytes,
                output_sha256,
            )
            revision_id = _output_revision_id(request, output_sha256)
            rendered_revision = RenderedPageRevision(
                revision_id=revision_id,
                page_id=request.snapshot.page_id,
                asset=_portable_asset_path(output_path, self.project_directory),
                content_sha256=output_sha256,
                cleaned_base_revision_id=request.snapshot.cleaned_base_revision_id,
                cleaned_base_sha256=cleaned_sha256,
                effective_page_fingerprint=request.snapshot.effective_fingerprint,
                parent_layer_revision_ids=tuple(
                    value.revision_id for value in layer_revisions
                ),
                renderer_contract_fingerprint=(
                    self.renderer_adapter.renderer_contract_fingerprint
                ),
                font_asset_fingerprint=self.renderer_adapter.font_asset_fingerprint,
                canvas_size=canvas_size,
            )

            commit_receipt: ProjectEditCommitReceipt | None = None
            if request.mode is RerenderMode.COMMIT:
                self._check_cancel(cancellation, RerenderStage.PERSISTING)
                self._emit(
                    progress,
                    request.snapshot.page_id,
                    RerenderStage.PERSISTING,
                    total,
                    total,
                    message="Publishing edits and artifacts atomically",
                )
                persistence_started = time.perf_counter()
                commit_receipt = self._commit(
                    request,
                    tuple(layer_revisions),
                    rendered_revision,
                )
                persistence_ms = (time.perf_counter() - persistence_started) * 1000.0

            self._emit(
                progress,
                request.snapshot.page_id,
                RerenderStage.COMPLETED,
                total,
                total,
                message="Rerender completed",
            )
            return PageRerenderReceipt(
                page_id=request.snapshot.page_id,
                mode=request.mode,
                status=RerenderStatus.COMPLETED,
                output_path=output_path,
                output_sha256=output_sha256,
                rendered_page_revision=rendered_revision,
                parent_layer_revisions=tuple(layer_revisions),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_rejections=tuple(cache_rejections),
                renderer_ms=round(renderer_ms, 6),
                composition_ms=round(composition_ms, 6),
                persistence_ms=round(persistence_ms, 6),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 6),
                commit_receipt=commit_receipt,
            )
        except _Cancelled as cancelled:
            self._emit(
                progress,
                request.snapshot.page_id,
                RerenderStage.CANCELLED,
                len(layer_revisions),
                total,
                message="Rerender cancelled at a safe boundary",
            )
            return PageRerenderReceipt(
                page_id=request.snapshot.page_id,
                mode=request.mode,
                status=RerenderStatus.CANCELLED,
                output_path=output_path,
                output_sha256=output_sha256,
                rendered_page_revision=rendered_revision,
                parent_layer_revisions=tuple(layer_revisions),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                cache_rejections=tuple(cache_rejections),
                renderer_ms=round(renderer_ms, 6),
                composition_ms=round(composition_ms, 6),
                persistence_ms=round(persistence_ms, 6),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 6),
                commit_receipt=None,
                cancellation_stage=cancelled.stage.value,
            )

    def _validated_clean_base(
        self,
        snapshot: EffectivePageSnapshot,
    ) -> tuple[str, str, tuple[int, int]]:
        cleaned = thaw_json(snapshot.cleaned_page_base)
        if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
            raise CleanedPageBaseUnavailableError(
                "selected CleanedPageBase revision is invalid"
            )
        page_id = str(cleaned.get("page_id") or "").strip()
        if page_id != snapshot.page_id:
            raise CleanedPageBaseUnavailableError(
                "CleanedPageBase page identity is invalid"
            )
        descriptor_revision_id = str(cleaned.get("revision_id") or "").strip()
        if (
            descriptor_revision_id
            and descriptor_revision_id != snapshot.cleaned_base_revision_id
        ):
            raise CleanedPageBaseUnavailableError(
                "CleanedPageBase revision identity is invalid"
            )
        asset = str(cleaned.get("asset") or "").strip()
        expected_sha256 = _require_sha256(
            cleaned.get("content_sha256"),
            "CleanedPageBase content_sha256",
        )
        path = _resolve_asset_path(asset, self.project_directory)
        if not os.path.isfile(path):
            raise CleanedPageBaseUnavailableError(
                "selected CleanedPageBase asset is missing"
            )
        try:
            with open(path, "rb") as stream:
                payload = stream.read()
        except OSError as exc:
            raise CleanedPageBaseUnavailableError(
                "selected CleanedPageBase asset cannot be read"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise CleanedPageBaseUnavailableError(
                "selected CleanedPageBase hash is invalid"
            )
        if Image is None:
            raise RuntimeError("Pillow is not installed")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                if image.width <= 0 or image.height <= 0:
                    raise CleanedPageBaseUnavailableError(
                        "CleanedPageBase dimensions are invalid"
                    )
                canvas_size = (int(image.width), int(image.height))
        except CleanedPageBaseUnavailableError:
            raise
        except Exception as exc:
            raise CleanedPageBaseUnavailableError(
                "CleanedPageBase cannot be decoded"
            ) from exc
        pinned_path = os.path.join(
            self.artifact_root,
            "inputs",
            f"cleaned-base-{expected_sha256}.png",
        )
        _write_once_atomic(
            pinned_path,
            payload,
            expected_sha256=expected_sha256,
        )
        return pinned_path, expected_sha256, canvas_size

    def _validate_request(self, request: PageRerenderRequest) -> None:
        if not isinstance(request.mode, RerenderMode):
            raise PageRerenderError("rerender mode is invalid")
        snapshot = request.snapshot
        if not snapshot.page_id or not snapshot.project_id:
            raise PageRerenderError("effective page identity is missing")
        pending_ids = {edit.edit_id for edit in request.pending_edits}
        if len(pending_ids) != len(request.pending_edits):
            raise PageRerenderError("pending edit identity is duplicated")
        for edit in request.pending_edits:
            if edit.project_id != snapshot.project_id or edit.page_id != snapshot.page_id:
                raise PageRerenderError("pending edit does not belong to the effective page")
        if request.mode is RerenderMode.COMMIT:
            if self.edit_store is None:
                raise PageRerenderError("commit rerender requires a project edit store")
            if self.edit_store.project_id != snapshot.project_id:
                raise PageRerenderError("project edit store identity mismatch")
            if os.path.normcase(os.path.abspath(self.edit_store.project_path)) != (
                os.path.normcase(self.project_path)
            ):
                raise PageRerenderError("project edit store path mismatch")
            _require_sha256(
                request.expected_page_head_sha256,
                "expected_page_head_sha256",
            )
            _require_sha256(
                request.expected_global_head_sha256,
                "expected_global_head_sha256",
            )
            try:
                persisted = self.edit_store.load_ledger()
                persisted_record_ids = {edit.edit_id for edit in persisted.edits}
                if pending_ids & persisted_record_ids:
                    raise PageRerenderError(
                        "pending edit is already present in the project ledger"
                    )
                candidate = persisted
                for edit in request.pending_edits:
                    candidate = candidate.append(edit)
            except PageRerenderError:
                raise
            except Exception as exc:
                raise PageRerenderError(
                    "pending edit delta is invalid"
                ) from exc
            try:
                # Re-run the sole projection owner over the exact candidate
                # ledger.  Raw active-ledger membership is not equivalent to
                # an effective page: superseded edits, control records, and
                # project-scoped glossary edits all have typed projection
                # semantics that only ``project_effective_page`` owns.
                from app.io.project import load_project_for_editing
                from .projection import project_effective_page

                project = load_project_for_editing(self.project_path)
                project = self.edit_store.materialize_project(project)
                candidate_snapshot = project_effective_page(
                    project,
                    candidate,
                    page_id=snapshot.page_id,
                )
            except PageRerenderError:
                raise
            except Exception as exc:
                raise PageRerenderError(
                    "candidate edit delta cannot be projected"
                ) from exc
            if (
                candidate_snapshot.effective_fingerprint
                != snapshot.effective_fingerprint
                or candidate_snapshot.applied_edit_ids
                != snapshot.applied_edit_ids
            ):
                raise PageRerenderError(
                    "effective projection does not match the exact persisted edit delta"
                )

    def _cache_inputs(
        self,
        snapshot: EffectivePageSnapshot,
        parent: EffectiveParentSnapshot,
        effective_layer: EffectiveRenderLayerPlan,
        plan: RenderLayerPlan,
        cleaned_sha256: str,
    ) -> ParentLayerCacheInputs:
        return ParentLayerCacheInputs(
            page_id=snapshot.page_id,
            parent_id=parent.parent_id,
            automatic_parent_fingerprint=parent.automatic_fingerprint,
            effective_target_text_fingerprint=canonical_sha256(
                {"parent_id": parent.parent_id, "target_text": parent.target_text}
            ),
            automated_resolved_style_fingerprint=canonical_sha256(
                thaw_json(parent.automatic_render_style)
            ),
            render_override_fingerprint=effective_layer.override_fingerprint,
            effective_render_geometry_fingerprint=canonical_sha256(
                {
                    "target_box": list(plan.target_box),
                    "hard_bounds": list(plan.hard_bounds),
                    "clipping_region_ref": dict(plan.clipping_region_ref or {}),
                    "draw_order": int(plan.draw_order),
                    "role": str(plan.role or ""),
                    "canvas_dependency": cleaned_sha256,
                }
            ),
            writing_and_break_fingerprint=canonical_sha256(
                {
                    "writing_mode": str(plan.writing_mode),
                    "line_height": dict(plan.resolved_render_style or {}).get(
                        "line_height"
                    ),
                    "break_hints": dict(
                        thaw_json(parent.render_layout_overrides)
                    ).get(
                        "break_hints", []
                    ),
                }
            ),
            placement_context_fingerprint=canonical_sha256(
                {
                    "layout_policy": "parent_local",
                    "occupied_sibling_bounds": "ignored_by_renderer_contract",
                }
            ),
            render_plan_payload_fingerprint=canonical_sha256(
                render_layer_plan_payload(plan)
            ),
            renderer_contract_fingerprint=(
                self.renderer_adapter.renderer_contract_fingerprint
            ),
            font_asset_fingerprint=self.renderer_adapter.font_asset_fingerprint,
            cleaned_page_base_fingerprint=cleaned_sha256,
        )

    def _layer_revision(
        self,
        snapshot: EffectivePageSnapshot,
        effective_layer: EffectiveRenderLayerPlan,
        artifact: ParentLayerArtifact,
        *,
        cache_hit: bool,
    ) -> ParentLayerRevision:
        return ParentLayerRevision(
            revision_id=(
                f"parent-layer:{snapshot.page_id}:{artifact.cache_key[:32]}"
            ),
            page_id=snapshot.page_id,
            parent_id=artifact.parent_id,
            cache_key=artifact.cache_key,
            asset=_portable_asset_path(artifact.image_path, self.project_directory),
            content_sha256=artifact.image_sha256,
            cache_manifest=_portable_asset_path(
                artifact.manifest_path,
                self.project_directory,
            ),
            cache_manifest_sha256=artifact.manifest_sha256,
            cleaned_base_revision_id=snapshot.cleaned_base_revision_id,
            cleaned_base_sha256=artifact.inputs.cleaned_page_base_fingerprint,
            effective_plan_fingerprint=effective_layer.effective_plan_fingerprint,
            renderer_contract_fingerprint=(
                artifact.inputs.renderer_contract_fingerprint
            ),
            font_asset_fingerprint=artifact.inputs.font_asset_fingerprint,
            canvas_size=artifact.canvas_size,
            alpha_bounds=artifact.alpha_bounds,
            cache_hit=cache_hit,
            render_audit=freeze_json(
                thaw_json(artifact.layer_audit),
                field_name="parent_layer_revision.render_audit",
            ),
        )

    def _publish_output(
        self,
        request: PageRerenderRequest,
        payload: bytes,
        content_sha256: str,
    ) -> str:
        mode_directory = "previews" if request.mode is RerenderMode.PREVIEW else "rendered"
        directory = os.path.join(self.artifact_root, mode_directory)
        page_fragment = canonical_sha256({"page_id": request.snapshot.page_id})[:16]
        suffix = (
            uuid.uuid4().hex[:16]
            if request.mode is RerenderMode.PREVIEW
            else content_sha256[:24]
        )
        path = os.path.join(directory, f"page-{page_fragment}-{suffix}.png")
        _write_once_atomic(path, payload, expected_sha256=content_sha256)
        if request.mode is RerenderMode.PREVIEW:
            with self._preview_lock:
                previous = self._preview_paths.get(request.snapshot.page_id, "")
                self._preview_paths[request.snapshot.page_id] = path
            _discard_page_preview_files(
                directory,
                page_fragment,
                keep_path=path,
            )
            if previous and previous != path and os.path.isfile(previous):
                # The directory sweep above normally removes this.  Retain the
                # explicit prior-path cleanup for a caller-supplied artifact
                # root whose filesystem view changes between scans.
                try:
                    os.unlink(previous)
                except OSError:
                    pass
        return path

    def discard_preview(self, page_id: str) -> None:
        """Remove this service's temporary preview for one page."""

        with self._preview_lock:
            path = self._preview_paths.pop(str(page_id or ""), "")
        if path:
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except OSError:
                pass
        directory = os.path.join(self.artifact_root, "previews")
        page_fragment = canonical_sha256({"page_id": str(page_id or "")})[:16]
        _discard_page_preview_files(directory, page_fragment)

    def _commit(
        self,
        request: PageRerenderRequest,
        layers: tuple[ParentLayerRevision, ...],
        rendered: RenderedPageRevision,
    ) -> ProjectEditCommitReceipt | None:
        if self.edit_store is None:  # pragma: no cover - validated earlier
            raise PageRerenderError("project edit store is unavailable")
        proposed = [layer.to_artifact_record() for layer in layers]
        proposed.append(rendered.to_artifact_record())
        existing = {
            str(value.get("revision_id") or ""): dict(value)
            for value in self.edit_store.load_artifact_revisions()
        }
        pending_artifacts: list[dict[str, Any]] = []
        for artifact in proposed:
            revision_id = str(artifact.get("revision_id") or "")
            prior = existing.get(revision_id)
            if prior is None:
                pending_artifacts.append(artifact)
                continue
            if prior != artifact:
                raise PageRerenderError(
                    f"artifact revision identity conflict: {revision_id}"
                )
        if not request.pending_edits and not pending_artifacts:
            if (
                self.edit_store.page_head(request.snapshot.page_id)
                != request.expected_page_head_sha256
            ):
                raise PageRerenderError(
                    "page edit head changed before the transaction committed"
                )
            if self.edit_store.global_head() != request.expected_global_head_sha256:
                raise PageRerenderError(
                    "project edit head changed before the transaction committed"
                )
            return None
        return self.edit_store.commit_page_edits(
            request.pending_edits,
            automatic_page_sha256=request.snapshot.automatic_fingerprint,
            expected_page_head_sha256=request.expected_page_head_sha256,
            expected_global_head_sha256=request.expected_global_head_sha256,
            artifact_revisions=pending_artifacts,
            transaction_id=request.transaction_id or f"gui3-{uuid.uuid4().hex}",
        )

    @staticmethod
    def _check_cancel(
        cancellation: CancellationProbe | None,
        stage: RerenderStage,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            raise _Cancelled(stage)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        page_id: str,
        stage: RerenderStage,
        completed: int,
        total: int,
        *,
        parent_id: str = "",
        cache_hit: bool | None = None,
        message: str = "",
    ) -> None:
        if callback is None:
            return
        try:
            callback(
                RerenderProgress(
                    page_id=page_id,
                    stage=stage,
                    completed_parents=int(completed),
                    total_parents=int(total),
                    parent_id=parent_id,
                    cache_hit=cache_hit,
                    message=message,
                )
            )
        except Exception:
            # Progress is an observer surface.  It must never alter whether a
            # page-local transaction commits or how its result is classified.
            return


def rerender_artifact_root(project_path: str) -> str:
    raw = str(project_path or "").strip()
    if not raw:
        raise ValueError("project_path is required")
    absolute = os.path.abspath(raw)
    parent = os.path.dirname(absolute) or os.getcwd()
    return os.path.join(parent, f".{os.path.basename(absolute)}.gui-render-artifacts")


def _output_revision_id(
    request: PageRerenderRequest,
    output_sha256: str,
) -> str:
    supplied = str(request.output_revision_id or "").strip()
    if supplied:
        return supplied
    return (
        f"rendered:{request.snapshot.page_id}:"
        f"{request.snapshot.effective_fingerprint[:16]}:{output_sha256[:16]}"
    )


def _resolve_asset_path(asset: str, project_directory: str) -> str:
    raw = str(asset or "").strip()
    if not raw:
        raise PageRerenderError("CleanedPageBase asset path is missing")
    return os.path.abspath(
        raw if os.path.isabs(raw) else os.path.join(project_directory, raw)
    )


def _portable_asset_path(path: str, project_directory: str) -> str:
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, project_directory)
    except ValueError:
        return absolute.replace("\\", "/")
    return relative.replace("\\", "/")


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PageRerenderError(f"{field_name} must be a SHA-256 digest")
    return text


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_once_atomic(path: str, payload: bytes, *, expected_sha256: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PageRerenderError("output payload hash is invalid")
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        if file_sha256(path) != expected_sha256:
            raise PageRerenderError("content-addressed output path already differs")
        return
    handle, temp_path = tempfile.mkstemp(
        prefix=".rerender-",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _discard_page_preview_files(
    directory: str,
    page_fragment: str,
    *,
    keep_path: str = "",
) -> None:
    """Remove only superseded GUI preview PNGs for one exact page."""

    if not os.path.isdir(directory):
        return
    prefix = f"page-{page_fragment}-"
    keep = os.path.abspath(keep_path) if keep_path else ""
    try:
        entries = tuple(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        if (
            not entry.is_file(follow_symlinks=False)
            or not entry.name.startswith(prefix)
            or not entry.name.endswith(".png")
        ):
            continue
        candidate = os.path.abspath(entry.path)
        if keep and candidate == keep:
            continue
        try:
            os.unlink(candidate)
        except OSError:
            pass
