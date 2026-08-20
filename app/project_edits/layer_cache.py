# -*- coding: utf-8 -*-
"""Content-addressed parent-layer cache for page-local GUI rerendering."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import os
import tempfile
import threading
from typing import Any, Mapping

from app.render.typesetting_contracts import FitReport, RenderLayerPlan, TypesetLayout

from .effective_render_plan import (
    render_layer_plan_from_payload,
    render_layer_plan_payload,
)
from .contracts import freeze_json, thaw_json
from .fingerprints import canonical_sha256

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


PARENT_LAYER_CACHE_VERSION = "parent_layer_cache_v2"


class ParentLayerCacheError(RuntimeError):
    """A cached artifact is malformed, corrupt, or outside its cache root."""

    def __init__(self, message: str, *, code: str = "cache_contract_invalid") -> None:
        super().__init__(message)
        self.code = str(code or "cache_contract_invalid")


@dataclass(frozen=True, slots=True)
class ParentLayerCacheInputs:
    page_id: str
    parent_id: str
    automatic_parent_fingerprint: str
    effective_target_text_fingerprint: str
    automated_resolved_style_fingerprint: str
    render_override_fingerprint: str
    effective_render_geometry_fingerprint: str
    writing_and_break_fingerprint: str
    placement_context_fingerprint: str
    render_plan_payload_fingerprint: str
    renderer_contract_fingerprint: str
    font_asset_fingerprint: str
    cleaned_page_base_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("page_id", "parent_id"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} is required")
        for field_name in self.__dataclass_fields__:
            if field_name in {"page_id", "parent_id"}:
                continue
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "automatic_parent_fingerprint": self.automatic_parent_fingerprint,
            "effective_target_text_fingerprint": self.effective_target_text_fingerprint,
            "automated_resolved_style_fingerprint": self.automated_resolved_style_fingerprint,
            "render_override_fingerprint": self.render_override_fingerprint,
            "effective_render_geometry_fingerprint": self.effective_render_geometry_fingerprint,
            "writing_and_break_fingerprint": self.writing_and_break_fingerprint,
            "placement_context_fingerprint": self.placement_context_fingerprint,
            "render_plan_payload_fingerprint": self.render_plan_payload_fingerprint,
            "renderer_contract_fingerprint": self.renderer_contract_fingerprint,
            "font_asset_fingerprint": self.font_asset_fingerprint,
            "cleaned_page_base_fingerprint": self.cleaned_page_base_fingerprint,
        }

    @property
    def cache_key(self) -> str:
        return canonical_sha256(
            {
                "parent_layer_cache_version": PARENT_LAYER_CACHE_VERSION,
                **self.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class ParentLayerArtifact:
    cache_key: str
    page_id: str
    parent_id: str
    image_path: str
    manifest_path: str
    manifest_sha256: str
    image_sha256: str
    canvas_size: tuple[int, int]
    image_mode: str
    alpha_bounds: tuple[int, ...]
    adjusted_plan_payload_fingerprint: str
    plan_payload: Mapping[str, Any]
    layout_payload: Mapping[str, Any]
    fit_report_payload: Mapping[str, Any]
    layer_audit: Mapping[str, Any]
    inputs: ParentLayerCacheInputs
    elapsed_ms: float

    def to_render_values(
        self,
    ) -> tuple[RenderLayerPlan, TypesetLayout, FitReport]:
        return (
            render_layer_plan_from_payload(thaw_json(self.plan_payload)),
            typeset_layout_from_payload(thaw_json(self.layout_payload)),
            fit_report_from_payload(thaw_json(self.fit_report_payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_layer_cache_version": PARENT_LAYER_CACHE_VERSION,
            "cache_key": self.cache_key,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "image_file": _cache_image_filename(
                self.cache_key,
                self.image_sha256,
            ),
            "image_sha256": self.image_sha256,
            "canvas_size": list(self.canvas_size),
            "image_mode": self.image_mode,
            "alpha_bounds": list(self.alpha_bounds),
            "adjusted_plan_payload_fingerprint": (
                self.adjusted_plan_payload_fingerprint
            ),
            "plan": thaw_json(self.plan_payload),
            "layout": thaw_json(self.layout_payload),
            "fit_report": thaw_json(self.fit_report_payload),
            "layer_audit": thaw_json(self.layer_audit),
            "inputs": self.inputs.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        image_path: str,
        manifest_path: str,
        manifest_sha256: str,
    ) -> "ParentLayerArtifact":
        required_keys = {
            "parent_layer_cache_version",
            "cache_key",
            "page_id",
            "parent_id",
            "image_file",
            "image_sha256",
            "canvas_size",
            "image_mode",
            "alpha_bounds",
            "adjusted_plan_payload_fingerprint",
            "plan",
            "layout",
            "fit_report",
            "layer_audit",
            "inputs",
        }
        if set(value) != required_keys:
            raise ParentLayerCacheError("parent-layer metadata schema is invalid")
        if value.get("parent_layer_cache_version") != PARENT_LAYER_CACHE_VERSION:
            raise ParentLayerCacheError("unsupported parent-layer cache version")
        inputs_value = value.get("inputs")
        if not isinstance(inputs_value, Mapping):
            raise ParentLayerCacheError("parent-layer cache inputs are missing")
        if set(inputs_value) != set(ParentLayerCacheInputs.__dataclass_fields__):
            raise ParentLayerCacheError("parent-layer cache input schema is invalid")
        try:
            inputs = ParentLayerCacheInputs(
                **{
                    field: str(inputs_value.get(field) or "")
                    for field in ParentLayerCacheInputs.__dataclass_fields__
                }
            )
        except TypeError as exc:
            raise ParentLayerCacheError("parent-layer cache inputs are invalid") from exc
        cache_key = str(value.get("cache_key") or "")
        if cache_key != inputs.cache_key:
            raise ParentLayerCacheError("parent-layer cache key does not match its inputs")
        for field in ("plan", "layout", "fit_report", "layer_audit"):
            if not isinstance(value.get(field), Mapping):
                raise ParentLayerCacheError(f"parent-layer {field} is invalid")
        page_id = str(value.get("page_id") or "")
        parent_id = str(value.get("parent_id") or "")
        if page_id != inputs.page_id or parent_id != inputs.parent_id:
            raise ParentLayerCacheError("parent-layer identity does not match its inputs")
        if str(value.get("image_file") or "") != _cache_image_filename(
            cache_key,
            value.get("image_sha256"),
        ):
            raise ParentLayerCacheError("parent-layer image filename is invalid")
        canvas_size = _integer_tuple(value.get("canvas_size"), 2, "canvas_size")
        if any(item <= 0 for item in canvas_size):
            raise ParentLayerCacheError("parent-layer canvas size is invalid")
        image_mode = str(value.get("image_mode") or "")
        if image_mode != "RGBA":
            raise ParentLayerCacheError("parent-layer image mode is invalid")
        alpha_value = value.get("alpha_bounds")
        alpha_bounds = (
            _integer_tuple(alpha_value, 4, "alpha_bounds")
            if alpha_value
            else ()
        )
        artifact = cls(
            cache_key=cache_key,
            page_id=page_id,
            parent_id=parent_id,
            image_path=image_path,
            manifest_path=manifest_path,
            manifest_sha256=_require_sha256(
                manifest_sha256,
                "manifest_sha256",
            ),
            image_sha256=_require_sha256(value.get("image_sha256"), "image_sha256"),
            canvas_size=(int(canvas_size[0]), int(canvas_size[1])),
            image_mode=image_mode,
            alpha_bounds=tuple(int(item) for item in alpha_bounds),
            adjusted_plan_payload_fingerprint=_require_sha256(
                value.get("adjusted_plan_payload_fingerprint"),
                "adjusted_plan_payload_fingerprint",
            ),
            plan_payload=freeze_json(value["plan"], field_name="cache.plan"),
            layout_payload=freeze_json(value["layout"], field_name="cache.layout"),
            fit_report_payload=freeze_json(value["fit_report"], field_name="cache.fit_report"),
            layer_audit=freeze_json(value["layer_audit"], field_name="cache.layer_audit"),
            inputs=inputs,
            # Render timing is receipt-only telemetry.  It deliberately does
            # not participate in the immutable cache manifest.
            elapsed_ms=0.0,
        )
        _validate_artifact_contracts(artifact)
        return artifact


class ParentLayerCache:
    """Atomic, validation-first layer storage with no eviction policy."""

    def __init__(self, root: str) -> None:
        raw_root = str(root or "").strip()
        if not raw_root:
            raise ValueError("parent-layer cache root is required")
        self.root = os.path.abspath(raw_root)
        self._lock = threading.RLock()

    def load(
        self,
        cache_key: str,
        *,
        expected_canvas_size: tuple[int, int] | None = None,
    ) -> ParentLayerArtifact | None:
        key = _require_sha256(cache_key, "cache_key")
        try:
            artifact = self._load_manifest(key, require_immutable=True)
            if artifact is None:
                return None
            if not os.path.isfile(artifact.image_path):
                raise ParentLayerCacheError(
                    "parent-layer image is missing",
                    code="image_missing",
                )
            if file_sha256(artifact.image_path) != artifact.image_sha256:
                raise ParentLayerCacheError(
                    "parent-layer image hash does not match metadata",
                    code="image_hash_mismatch",
                )
            _verify_layer_image(
                artifact.image_path,
                expected_canvas_size=expected_canvas_size or artifact.canvas_size,
                expected_alpha_bounds=artifact.alpha_bounds,
            )
            if expected_canvas_size and artifact.canvas_size != expected_canvas_size:
                raise ParentLayerCacheError("parent-layer canvas does not match the page")
            return artifact
        except ParentLayerCacheError:
            raise
        except Exception as exc:
            # Corrupt cache data is always a typed cache rejection.  Callers
            # may safely rerender; malformed nested renderer contracts must
            # never escape as arbitrary ValueError/TypeError exceptions.
            raise ParentLayerCacheError(
                "parent-layer cache entry cannot be decoded",
                code="schema_decode_failed",
            ) from exc

    def store(
        self,
        *,
        inputs: ParentLayerCacheInputs,
        surface: Any,
        plan: RenderLayerPlan,
        layout: TypesetLayout,
        fit_report: FitReport,
        layer_audit: Mapping[str, Any],
        elapsed_ms: float,
        repair_corrupt: bool = False,
    ) -> ParentLayerArtifact:
        if Image is None:
            raise RuntimeError("Pillow is not installed")
        if not isinstance(surface, Image.Image):
            raise TypeError("parent-layer surface must be a Pillow image")
        if not math.isfinite(float(elapsed_ms)) or float(elapsed_ms) < 0.0:
            raise ValueError("parent-layer elapsed time must be finite and nonnegative")
        key = inputs.cache_key
        os.makedirs(self.root, exist_ok=True)
        rgba_surface = surface.convert("RGBA")
        image_bytes = _png_bytes(rgba_surface)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        image_path = self._file_path(
            _cache_image_filename(key, image_sha256)
        )
        alpha_bounds = rgba_surface.getchannel("A").getbbox() or ()
        artifact = ParentLayerArtifact(
            cache_key=key,
            page_id=inputs.page_id,
            parent_id=inputs.parent_id,
            image_path=image_path,
            manifest_path="",
            manifest_sha256="0" * 64,
            image_sha256=image_sha256,
            canvas_size=(int(rgba_surface.width), int(rgba_surface.height)),
            image_mode="RGBA",
            alpha_bounds=tuple(int(item) for item in alpha_bounds),
            adjusted_plan_payload_fingerprint=canonical_sha256(
                render_layer_plan_payload(plan)
            ),
            plan_payload=freeze_json(
                render_layer_plan_payload(plan),
                field_name="cache.plan",
            ),
            layout_payload=freeze_json(
                typeset_layout_payload(layout),
                field_name="cache.layout",
            ),
            fit_report_payload=freeze_json(
                fit_report_payload(fit_report),
                field_name="cache.fit_report",
            ),
            layer_audit=freeze_json(
                dict(layer_audit),
                field_name="cache.layer_audit",
            ),
            inputs=inputs,
            elapsed_ms=float(elapsed_ms),
        )
        manifest_payload = _json_bytes(artifact.to_dict())
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        artifact = replace(
            artifact,
            manifest_path=self._artifact_manifest_path(key, manifest_sha256),
            manifest_sha256=manifest_sha256,
        )
        _validate_artifact_contracts(artifact)
        with self._lock:
            try:
                existing = self.load(key)
            except ParentLayerCacheError:
                if not repair_corrupt:
                    raise
                existing = None
            if existing is not None:
                if (
                    existing.image_sha256 != image_sha256
                    or _artifact_contract_fingerprint(existing)
                    != _artifact_contract_fingerprint(artifact)
                ):
                    raise ParentLayerCacheError(
                        "deterministic parent-layer cache key produced different output"
                    )
                return existing
            metadata_path = self.manifest_path(key)
            existing_manifest: ParentLayerArtifact | None = (
                self._load_immutable_manifest(key)
            )
            if repair_corrupt and os.path.exists(metadata_path):
                try:
                    pointer_manifest = self._load_manifest(
                        key,
                        require_immutable=False,
                    )
                except ParentLayerCacheError:
                    pointer_manifest = None
                if pointer_manifest is not None:
                    if (
                        existing_manifest is not None
                        and pointer_manifest.manifest_sha256
                        != existing_manifest.manifest_sha256
                    ):
                        raise ParentLayerCacheError(
                            "cache pointer disagrees with immutable manifest",
                            code="determinism_conflict",
                        )
                    existing_manifest = pointer_manifest
            if existing_manifest is not None:
                if (
                    existing_manifest.image_sha256 != image_sha256
                    or _artifact_contract_fingerprint(existing_manifest)
                    != _artifact_contract_fingerprint(artifact)
                ):
                    raise ParentLayerCacheError(
                        "corrupt cache repair differs from its immutable manifest"
                    )
                _write_bytes_once(
                    existing_manifest.image_path,
                    image_bytes,
                    repair_corrupt=True,
                )
                self._publish_immutable_manifest(existing_manifest)
                _write_json_atomic(metadata_path, existing_manifest.to_dict())
                repaired = self.load(key)
                if repaired is None:  # pragma: no cover - defensive
                    raise ParentLayerCacheError(
                        "parent-layer cache repair disappeared"
                    )
                return repaired

            _write_bytes_once(
                image_path,
                image_bytes,
                repair_corrupt=repair_corrupt,
            )
            self._publish_immutable_manifest(artifact)
            if repair_corrupt and os.path.exists(metadata_path):
                # Only an unreadable pointer may be replaced.  A parseable
                # manifest above is an immutable historical contract.
                _write_json_atomic(metadata_path, artifact.to_dict())
            else:
                try:
                    _write_json_once(metadata_path, artifact.to_dict())
                except FileExistsError:
                    # A second cache instance or process published this key
                    # first.  The already-visible immutable entry remains
                    # authoritative.
                    existing = self.load(key)
                    if existing is None:  # pragma: no cover - defensive race guard
                        raise ParentLayerCacheError(
                            "parent-layer cache publication disappeared"
                        )
                    if (
                        existing.image_sha256 != image_sha256
                        or _artifact_contract_fingerprint(existing)
                        != _artifact_contract_fingerprint(artifact)
                    ):
                        raise ParentLayerCacheError(
                            "deterministic parent-layer cache key produced different output"
                        )
                    return existing
            stored = self.load(key)
            if stored is None:  # pragma: no cover - defensive
                raise ParentLayerCacheError("parent-layer cache publication failed")
            return stored

    def _load_manifest(
        self,
        key: str,
        *,
        require_immutable: bool,
    ) -> ParentLayerArtifact | None:
        metadata_path = self.manifest_path(key)
        if not os.path.isfile(metadata_path):
            return None
        try:
            with open(metadata_path, "rb") as stream:
                payload = stream.read()
            value = json.loads(
                payload.decode("utf-8"),
                parse_constant=lambda value: _raise_invalid_json_constant(value),
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ParentLayerCacheError(
                "parent-layer metadata cannot be read",
                code="manifest_unreadable",
            ) from exc
        if not isinstance(value, Mapping):
            raise ParentLayerCacheError("parent-layer metadata is invalid")
        artifact = self._artifact_from_manifest_payload(key, payload, value)
        if require_immutable:
            if not os.path.isfile(artifact.manifest_path):
                raise ParentLayerCacheError(
                    "immutable parent-layer manifest is missing",
                    code="immutable_manifest_missing",
                )
            if file_sha256(artifact.manifest_path) != artifact.manifest_sha256:
                raise ParentLayerCacheError(
                    "immutable parent-layer manifest hash is invalid",
                    code="immutable_manifest_hash_mismatch",
                )
        return artifact

    def _artifact_from_manifest_payload(
        self,
        key: str,
        payload: bytes,
        value: Mapping[str, Any],
    ) -> ParentLayerArtifact:
        if payload != _json_bytes(value):
            raise ParentLayerCacheError(
                "parent-layer metadata is not canonical"
            )
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        immutable_path = self._artifact_manifest_path(key, manifest_sha256)
        image_sha256 = _require_sha256(value.get("image_sha256"), "image_sha256")
        return ParentLayerArtifact.from_dict(
            value,
            image_path=self._file_path(_cache_image_filename(key, image_sha256)),
            manifest_path=immutable_path,
            manifest_sha256=manifest_sha256,
        )

    def _load_immutable_manifest(self, key: str) -> ParentLayerArtifact | None:
        prefix = f".artifact-{key[:12]}-"
        suffix = ".json"
        candidates: list[ParentLayerArtifact] = []
        try:
            filenames = os.listdir(self.root)
        except FileNotFoundError:
            return None
        for filename in filenames:
            if not filename.startswith(prefix) or not filename.endswith(suffix):
                continue
            hash_text = filename[len(prefix) : -len(suffix)]
            try:
                if len(hash_text) != 24 or any(
                    character not in "0123456789abcdef" for character in hash_text
                ):
                    continue
                path = self._file_path(filename)
                with open(path, "rb") as stream:
                    payload = stream.read()
                value = json.loads(
                    payload.decode("utf-8"),
                    parse_constant=lambda value: _raise_invalid_json_constant(value),
                )
                if not isinstance(value, Mapping):
                    raise ParentLayerCacheError(
                        "immutable parent-layer manifest is invalid"
                    )
                if str(value.get("cache_key") or "") != key:
                    continue
                artifact = self._artifact_from_manifest_payload(key, payload, value)
                if not artifact.manifest_sha256.startswith(hash_text):
                    raise ParentLayerCacheError(
                        "immutable parent-layer manifest filename is invalid"
                    )
                candidates.append(artifact)
            except ParentLayerCacheError:
                raise
            except Exception as exc:
                raise ParentLayerCacheError(
                    "immutable parent-layer manifest cannot be decoded",
                    code="immutable_manifest_unreadable",
                ) from exc
        if len(candidates) > 1:
            raise ParentLayerCacheError(
                "multiple immutable manifests exist for one parent-layer key",
                code="determinism_conflict",
            )
        return candidates[0] if candidates else None

    def _publish_immutable_manifest(self, artifact: ParentLayerArtifact) -> None:
        payload = _json_bytes(artifact.to_dict())
        if hashlib.sha256(payload).hexdigest() != artifact.manifest_sha256:
            raise ParentLayerCacheError(
                "parent-layer immutable manifest fingerprint is invalid"
            )
        try:
            _write_json_once(artifact.manifest_path, artifact.to_dict())
        except FileExistsError:
            if file_sha256(artifact.manifest_path) != artifact.manifest_sha256:
                raise ParentLayerCacheError(
                    "immutable parent-layer manifest already differs"
                )

    def _path(self, key: str, suffix: str) -> str:
        path = os.path.abspath(os.path.join(self.root, f"{key}{suffix}"))
        if os.path.commonpath((self.root, path)) != self.root:
            raise ParentLayerCacheError("parent-layer cache path escaped its root")
        return path

    def _file_path(self, filename: str) -> str:
        path = os.path.abspath(os.path.join(self.root, filename))
        if os.path.commonpath((self.root, path)) != self.root:
            raise ParentLayerCacheError("parent-layer cache path escaped its root")
        return path

    def manifest_path(self, cache_key: str) -> str:
        return self._path(_require_sha256(cache_key, "cache_key"), ".json")

    def _artifact_manifest_path(self, cache_key: str, manifest_sha256: str) -> str:
        key = _require_sha256(cache_key, "cache_key")
        manifest_hash = _require_sha256(manifest_sha256, "manifest_sha256")
        # Keep Windows paths comfortably below the legacy MAX_PATH boundary.
        # The full SHA-256 remains inside and validates the immutable manifest;
        # a truncated-name collision fails closed in _publish_immutable_manifest.
        return self._file_path(
            f".artifact-{key[:12]}-{manifest_hash[:24]}.json"
        )


def parent_layer_cache_root(project_path: str) -> str:
    raw = str(project_path or "").strip()
    if not raw:
        raise ValueError("project_path is required")
    absolute = os.path.abspath(raw)
    parent = os.path.dirname(absolute) or os.getcwd()
    return os.path.join(parent, f".{os.path.basename(absolute)}.gui-render-cache")


def typeset_layout_payload(value: TypesetLayout) -> dict[str, Any]:
    payload = value.to_audit_dict()
    payload.pop("typeset_layout_version", None)
    return payload


def typeset_layout_from_payload(value: Mapping[str, Any]) -> TypesetLayout:
    payload = dict(value)
    payload.pop("typeset_layout_version", None)
    return TypesetLayout(**payload)


def fit_report_payload(value: FitReport) -> dict[str, Any]:
    payload = value.to_audit_dict()
    payload.pop("fit_report_version", None)
    return payload


def fit_report_from_payload(value: Mapping[str, Any]) -> FitReport:
    payload = dict(value)
    payload.pop("fit_report_version", None)
    return FitReport(**payload)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ParentLayerCacheError(f"{field_name} must be a SHA-256 digest")
    return text


def _integer_tuple(value: Any, length: int, field_name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ParentLayerCacheError(
            f"parent-layer {field_name} must contain {length} integers"
        )
    return tuple(int(item) for item in value)


def _raise_invalid_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _validate_artifact_contracts(artifact: ParentLayerArtifact) -> None:
    plan, layout, report = artifact.to_render_values()
    if artifact.adjusted_plan_payload_fingerprint != canonical_sha256(
        render_layer_plan_payload(plan)
    ):
        raise ParentLayerCacheError(
            "parent-layer adjusted render plan fingerprint is inconsistent"
        )
    identities = (
        (plan.page_id, plan.parent_id),
        (layout.page_id, layout.parent_id),
        (report.page_id, report.parent_id),
        (
            str(artifact.layer_audit.get("page_id") or ""),
            str(artifact.layer_audit.get("parent_id") or ""),
        ),
    )
    if any(
        page_id != artifact.page_id or parent_id != artifact.parent_id
        for page_id, parent_id in identities
    ):
        raise ParentLayerCacheError(
            "parent-layer plan, layout, report, or audit identity is inconsistent"
        )
    layer_ids = {
        str(plan.layer_id or ""),
        str(layout.layer_id or ""),
        str(report.layer_id or ""),
        str(artifact.layer_audit.get("layer_id") or ""),
    }
    if len(layer_ids) != 1 or not next(iter(layer_ids), ""):
        raise ParentLayerCacheError("parent-layer layer identity is inconsistent")
    audit = artifact.layer_audit
    has_visible_text = bool(str(plan.translated_text or ""))
    if has_visible_text and not bool(audit.get("drawn")):
        raise ParentLayerCacheError("non-empty parent-layer audit is not drawn")
    if has_visible_text and not bool(audit.get("text_placement_complete")):
        raise ParentLayerCacheError("parent-layer audit has incomplete target text")
    transaction = audit.get("page_transaction")
    if not isinstance(transaction, Mapping) or not bool(
        transaction.get("output_committed")
    ):
        raise ParentLayerCacheError(
            "parent-layer audit is not bound to committed renderer output"
        )
    extraction = audit.get("isolated_layer_extraction")
    if not isinstance(extraction, Mapping):
        raise ParentLayerCacheError(
            "parent-layer audit lacks isolated-layer extraction proof"
        )
    if not bool(extraction.get("pixel_parity")):
        raise ParentLayerCacheError(
            "parent-layer extraction is not pixel-identical"
        )
    if not bool(extraction.get("same_executor_compositor_used")):
        raise ParentLayerCacheError(
            "parent-layer extraction did not use the renderer compositor"
        )
    if bool(extraction.get("pixel_difference_used_for_layer_extraction")):
        raise ParentLayerCacheError(
            "parent-layer extraction used pixel differencing"
        )
    if not bool(extraction.get("pixel_difference_used_for_parity_validation")):
        raise ParentLayerCacheError(
            "parent-layer extraction lacks pixel-parity validation"
        )
    if str(extraction.get("renderer_contract_fingerprint") or "") != (
        artifact.inputs.renderer_contract_fingerprint
    ):
        raise ParentLayerCacheError(
            "parent-layer extraction renderer contract is inconsistent"
        )
    if str(extraction.get("font_asset_fingerprint") or "") != (
        artifact.inputs.font_asset_fingerprint
    ):
        raise ParentLayerCacheError(
            "parent-layer extraction font contract is inconsistent"
        )
    if str(extraction.get("requested_plan_fingerprint") or "") != (
        artifact.inputs.render_plan_payload_fingerprint
    ):
        raise ParentLayerCacheError(
            "parent-layer extraction requested plan is inconsistent"
        )
    if str(extraction.get("adjusted_plan_fingerprint") or "") != (
        artifact.adjusted_plan_payload_fingerprint
    ):
        raise ParentLayerCacheError(
            "parent-layer extraction adjusted plan is inconsistent"
        )
    extraction_canvas = _integer_tuple(
        extraction.get("canvas_size"),
        2,
        "isolated_layer_extraction.canvas_size",
    )
    if extraction_canvas != artifact.canvas_size:
        raise ParentLayerCacheError(
            "parent-layer extraction canvas is inconsistent"
        )
    if artifact.canvas_size[0] <= 0 or artifact.canvas_size[1] <= 0:
        raise ParentLayerCacheError("parent-layer canvas size is invalid")
    if artifact.image_mode != "RGBA":
        raise ParentLayerCacheError("parent-layer image mode is invalid")


def _artifact_contract_fingerprint(artifact: ParentLayerArtifact) -> str:
    return canonical_sha256(
        {
            "page_id": artifact.page_id,
            "parent_id": artifact.parent_id,
            "canvas_size": list(artifact.canvas_size),
            "image_mode": artifact.image_mode,
            "alpha_bounds": list(artifact.alpha_bounds),
            "adjusted_plan_payload_fingerprint": (
                artifact.adjusted_plan_payload_fingerprint
            ),
            "plan": thaw_json(artifact.plan_payload),
            "layout": thaw_json(artifact.layout_payload),
            "fit_report": thaw_json(artifact.fit_report_payload),
            "layer_audit": thaw_json(artifact.layer_audit),
            "inputs": artifact.inputs.to_dict(),
        }
    )


def _verify_layer_image(
    path: str,
    *,
    expected_canvas_size: tuple[int, int],
    expected_alpha_bounds: tuple[int, ...],
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is not installed")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != "RGBA":
                raise ParentLayerCacheError("parent-layer image must be RGBA")
            if image.width <= 0 or image.height <= 0:
                raise ParentLayerCacheError("parent-layer image dimensions are invalid")
            if image.size != expected_canvas_size:
                raise ParentLayerCacheError("parent-layer image canvas size is invalid")
            actual_alpha_bounds = image.getchannel("A").getbbox() or ()
            if tuple(actual_alpha_bounds) != tuple(expected_alpha_bounds):
                raise ParentLayerCacheError("parent-layer alpha bounds are invalid")
    except ParentLayerCacheError:
        raise
    except Exception as exc:
        raise ParentLayerCacheError(
            "parent-layer image cannot be decoded",
            code="image_decode_failed",
        ) from exc


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _cache_image_filename(cache_key: Any, image_sha256: Any) -> str:
    key = _require_sha256(cache_key, "cache_key")
    image_hash = _require_sha256(image_sha256, "image_sha256")
    return f"{key}.{image_hash}.png"


def _write_bytes_once(
    path: str,
    payload: bytes,
    *,
    repair_corrupt: bool = False,
) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(
        prefix=".layer-publish-",
        suffix=".png",
        dir=directory,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)
            _fsync_directory(directory)
            return
        except FileExistsError:
            if file_sha256(path) == hashlib.sha256(payload).hexdigest():
                return
            if not repair_corrupt:
                raise ParentLayerCacheError(
                    "content-addressed parent-layer image already differs"
                )
            _write_bytes_atomic(path, payload)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_bytes_atomic(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(
        prefix=".layer-repair-",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_json_once(path: str, value: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = _json_bytes(value)
    handle, temp_path = tempfile.mkstemp(
        prefix=".layer-publish-",
        suffix=".json",
        dir=directory,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_json_atomic(path: str, value: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = _json_bytes(value)
    handle, temp_path = tempfile.mkstemp(prefix=".layer-", suffix=".json", dir=directory)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _fsync_directory(path: str) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
