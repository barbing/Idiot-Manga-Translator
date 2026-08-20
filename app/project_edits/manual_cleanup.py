# -*- coding: utf-8 -*-
"""GUI-owned manual cleanup previews and immutable clean-base revisions.

This module deliberately stops at the existing fixed cleanup-backend seam.  It
does not change automatic cleanup planning, masks, proofs, or backend policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from PIL import Image, ImageChops, ImageFilter

from app.io.project_edit_store import (
    ProjectEditCommitReceipt,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
    inspect_project_edit_store,
)
from app.pipeline.hierarchy_revision_contracts import (
    EFFECTIVE_HIERARCHY_REVISION_PREFIX,
    ParentOrigin,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
    validate_user_parent_identity_pair,
)
from app.pipeline.ocr_revision_contracts import (
    OcrRevisionError,
    OcrSourceRevisionArtifact,
    OriginalPageAssetBinding,
)
from app.pipeline.translation_revision_contracts import (
    TranslationRevisionArtifact,
)
from app.project_edits.contracts import (
    EditDomain,
    EditTarget,
    EditTargetKind,
    create_project_edit,
    thaw_json,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    canonical_sha256,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import (
    EffectivePageSnapshot,
    ProjectionIssueKind,
    cleaned_base_automatic_lineage,
    effective_source_fingerprint,
    field_base_fingerprint,
    project_effective_page,
)


MANUAL_CLEANUP_SERVICE_VERSION = "manual_cleanup_service_v1"
MANUAL_CLEANUP_RECEIPT_VERSION = "manual_cleanup_receipt_v1"
MANUAL_CLEANED_BASE_REVISION_VERSION = "manual_cleaned_base_revision_v1"
USER_PARENT_CLEANUP_COVERAGE_TARGET_VERSION = (
    "user_parent_cleanup_coverage_target_v1"
)
DEFAULT_MANUAL_CLEANUP_BACKEND_ID = "iopaint_anime_manga_big_lama"
_MAX_MASK_BYTES = 64 * 1024 * 1024
_MAX_CANVAS_PIXELS = 50_000_000
_COVERAGE_CONFLICT_MESSAGE = (
    "Another user parent already has the current user-confirmed cleanup base. "
    "Revoke that cleanup selection in History before preparing this parent."
)


def _coverage_identity(value: Any, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} is required")
    return candidate


def _coverage_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "").lower()
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return candidate


def _coverage_canvas(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError("canvas_size must contain exact integer width and height")
    width, height = (int(item) for item in value)
    if width <= 0 or height <= 0 or width * height > _MAX_CANVAS_PIXELS:
        raise ValueError("canvas_size is outside the page safety limit")
    return width, height


def _coverage_bbox(
    value: Any,
    *,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(
            "workflow_area_bbox must contain exact integer x, y, width, height"
        )
    x, y, width, height = (int(item) for item in value)
    page_width, page_height = canvas_size
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > page_width
        or y + height > page_height
    ):
        raise ValueError("workflow_area_bbox must remain inside the original page")
    return x, y, width, height


class ManualCleanupStage(str, Enum):
    VALIDATING = "validating"
    PREPARING_MASKS = "preparing_masks"
    INPAINTING = "inpainting"
    PUBLISHING_PREVIEW = "publishing_preview"
    COMMITTING = "committing"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ManualCleanupStatus(str, Enum):
    PREVIEW_READY = "preview_ready"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


class ManualCleanupAvailabilityCode(str, Enum):
    READY = "ready"
    MISSING_BASE = "missing_base"
    STALE_BASE = "stale_base"
    INVALID_MASK = "invalid_mask"
    INVALID_COVERAGE_TARGET = "invalid_coverage_target"
    STALE_COVERAGE_TARGET = "stale_coverage_target"
    COVERAGE_CONFLICT = "coverage_conflict"
    ORIGINAL_ASSET_MISMATCH = "original_asset_mismatch"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BLOCKED = "blocked"


class ManualCleanupFailureCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    COVERAGE_TARGET_INVALID = "coverage_target_invalid"
    COVERAGE_TARGET_STALE = "coverage_target_stale"
    COVERAGE_CONFLICT = "coverage_conflict"
    ORIGINAL_ASSET_MISMATCH = "original_asset_mismatch"
    MISSING_BASE = "missing_base"
    STALE_BASE = "stale_base"
    INVALID_MASK = "invalid_mask"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_FAILED = "backend_failed"
    PREVIEW_STALE = "preview_stale"
    STORE_UNAVAILABLE = "store_unavailable"
    COMMIT_STALE = "commit_stale"
    ARTIFACT_INVALID = "artifact_invalid"


class ManualCleanupFailure(RuntimeError):
    """Typed fail-closed manual-cleanup error."""

    def __init__(
        self,
        code: ManualCleanupFailureCode,
        message: str,
        *,
        stage: ManualCleanupStage = ManualCleanupStage.VALIDATING,
    ) -> None:
        super().__init__(str(message))
        self.code = ManualCleanupFailureCode(code)
        self.stage = ManualCleanupStage(stage)
        self.message = str(message)


class CancellationProbe(Protocol):
    def is_cancelled(self) -> bool: ...


class ManualCleanupCancellationToken:
    """Thread-safe cooperative cancellation at GUI-owned boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class ManualCleanupParameters:
    grow_px: int = 0
    feather_px: int = 0
    backend_id: str = DEFAULT_MANUAL_CLEANUP_BACKEND_ID
    use_gpu: bool = False

    def __post_init__(self) -> None:
        for field_name in ("grow_px", "feather_px"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not 0 <= value <= 64:
                raise ValueError(f"{field_name} must be between 0 and 64")
        backend_id = str(self.backend_id or "").strip()
        if not backend_id:
            raise ValueError("backend_id is required")
        object.__setattr__(self, "backend_id", backend_id)
        if type(self.use_gpu) is not bool:
            raise TypeError("use_gpu must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grow_px": self.grow_px,
            "feather_px": self.feather_px,
            "backend_id": self.backend_id,
            "use_gpu": self.use_gpu,
        }


@dataclass(frozen=True, slots=True)
class UserParentCleanupCoverageTargetV1:
    """Exact immutable dependency binding for one user-parent cleanup preview.

    ``workflow_area_bbox`` is lineage and guide geometry only.  It is never
    interpreted as an erase mask; the page-bounded user mask remains the sole
    cleanup authority.
    """

    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    workflow_area_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    original_page_asset_id: str
    original_page_asset_reference: str
    original_page_content_sha256: str
    input_cleaned_base_revision_id: str
    input_cleaned_base_content_sha256: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    source_artifact_sha256: str
    source_fingerprint: str
    translation_revision_id: str
    translation_selection_edit_id: str
    translation_artifact_sha256: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "parent_role",
            "original_page_asset_id",
            "original_page_asset_reference",
            "input_cleaned_base_revision_id",
            "hierarchy_revision_id",
            "source_revision_id",
            "source_selection_edit_id",
            "translation_revision_id",
            "translation_selection_edit_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _coverage_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        if not self.hierarchy_revision_id.startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("hierarchy_revision_id is invalid")
        canvas_size = _coverage_canvas(self.canvas_size)
        object.__setattr__(self, "canvas_size", canvas_size)
        object.__setattr__(
            self,
            "workflow_area_bbox",
            _coverage_bbox(self.workflow_area_bbox, canvas_size=canvas_size),
        )
        for field_name in (
            "original_page_content_sha256",
            "input_cleaned_base_content_sha256",
            "hierarchy_fingerprint",
            "source_artifact_sha256",
            "source_fingerprint",
            "translation_artifact_sha256",
            "effective_page_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _coverage_sha256(getattr(self, field_name), field_name),
            )
        expected_asset_id = "original-page-v1-" + canonical_sha256(
            {
                "page_id": self.page_id,
                "asset_reference": self.original_page_asset_reference,
            }
        )
        if self.original_page_asset_id != expected_asset_id:
            raise ValueError("original_page_asset_id does not match the page binding")

    def _fingerprint_body(self) -> dict[str, Any]:
        return {
            "schema_version": USER_PARENT_CLEANUP_COVERAGE_TARGET_VERSION,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "parent_role": self.parent_role,
            "workflow_area_bbox": list(self.workflow_area_bbox),
            "canvas_size": list(self.canvas_size),
            "original_page_asset_id": self.original_page_asset_id,
            "original_page_asset_reference": self.original_page_asset_reference,
            "original_page_content_sha256": self.original_page_content_sha256,
            "input_cleaned_base_revision_id": self.input_cleaned_base_revision_id,
            "input_cleaned_base_content_sha256": (
                self.input_cleaned_base_content_sha256
            ),
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
            "source_revision_id": self.source_revision_id,
            "source_selection_edit_id": self.source_selection_edit_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_fingerprint": self.source_fingerprint,
            "translation_revision_id": self.translation_revision_id,
            "translation_selection_edit_id": self.translation_selection_edit_id,
            "translation_artifact_sha256": self.translation_artifact_sha256,
            "effective_page_fingerprint": self.effective_page_fingerprint,
        }

    @property
    def coverage_dependency_fingerprint(self) -> str:
        return canonical_sha256(self._fingerprint_body())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._fingerprint_body(),
            "coverage_dependency_fingerprint": (
                self.coverage_dependency_fingerprint
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "UserParentCleanupCoverageTargetV1":
        if not isinstance(value, Mapping):
            raise TypeError("user-parent cleanup coverage target must be a mapping")
        expected = {
            "schema_version",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "parent_role",
            "workflow_area_bbox",
            "canvas_size",
            "original_page_asset_id",
            "original_page_asset_reference",
            "original_page_content_sha256",
            "input_cleaned_base_revision_id",
            "input_cleaned_base_content_sha256",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
            "source_revision_id",
            "source_selection_edit_id",
            "source_artifact_sha256",
            "source_fingerprint",
            "translation_revision_id",
            "translation_selection_edit_id",
            "translation_artifact_sha256",
            "effective_page_fingerprint",
            "coverage_dependency_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("user-parent cleanup coverage target fields are invalid")
        if value.get("schema_version") != USER_PARENT_CLEANUP_COVERAGE_TARGET_VERSION:
            raise ValueError("user-parent cleanup coverage target version is invalid")
        target = cls(
            project_id=value["project_id"],
            page_id=value["page_id"],
            parent_id=value["parent_id"],
            root_id=value["root_id"],
            parent_authored_edit_id=value["parent_authored_edit_id"],
            parent_role=value["parent_role"],
            workflow_area_bbox=tuple(value["workflow_area_bbox"]),
            canvas_size=tuple(value["canvas_size"]),
            original_page_asset_id=value["original_page_asset_id"],
            original_page_asset_reference=value["original_page_asset_reference"],
            original_page_content_sha256=value["original_page_content_sha256"],
            input_cleaned_base_revision_id=value[
                "input_cleaned_base_revision_id"
            ],
            input_cleaned_base_content_sha256=value[
                "input_cleaned_base_content_sha256"
            ],
            hierarchy_revision_id=value["hierarchy_revision_id"],
            hierarchy_fingerprint=value["hierarchy_fingerprint"],
            source_revision_id=value["source_revision_id"],
            source_selection_edit_id=value["source_selection_edit_id"],
            source_artifact_sha256=value["source_artifact_sha256"],
            source_fingerprint=value["source_fingerprint"],
            translation_revision_id=value["translation_revision_id"],
            translation_selection_edit_id=value[
                "translation_selection_edit_id"
            ],
            translation_artifact_sha256=value["translation_artifact_sha256"],
            effective_page_fingerprint=value["effective_page_fingerprint"],
        )
        observed = _coverage_sha256(
            value["coverage_dependency_fingerprint"],
            "coverage_dependency_fingerprint",
        )
        if observed != target.coverage_dependency_fingerprint:
            raise ValueError("coverage dependency fingerprint does not match the target")
        return target


def _metadata_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {str(key): thaw_json(item) for key, item in value}


def _stage_requirement(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
    stage: RevisionStage,
) -> Any:
    parent = next(
        (candidate for candidate in snapshot.parents if candidate.parent_id == parent_id),
        None,
    )
    if parent is None:
        return None
    matches = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is stage
    )
    return matches[0] if len(matches) == 1 else None


def _other_current_user_parent_cleanup_coverage_ids(
    snapshot: EffectivePageSnapshot,
    selected_parent_id: str,
) -> tuple[str, ...]:
    current_parent_ids: list[str] = []
    for parent in snapshot.parents:
        if (
            parent.parent_id == selected_parent_id
            or parent.origin is not ParentOrigin.USER
            or parent.excluded
        ):
            continue
        cleanup_requirements = tuple(
            requirement
            for requirement in parent.stage_requirements
            if requirement.stage is RevisionStage.CLEANUP_BASE
        )
        if (
            len(cleanup_requirements) == 1
            and cleanup_requirements[0].state
            is RevisionStageState.CURRENT
            and cleanup_requirements[0].required_action
            is RevisionRequiredAction.NONE
        ):
            current_parent_ids.append(parent.parent_id)
    return tuple(sorted(current_parent_ids))


def user_parent_cleanup_coverage_target_from_snapshot(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
    *,
    original_page: OriginalPageAssetBinding,
) -> UserParentCleanupCoverageTargetV1:
    """Bind one current user parent without deriving cleanup authority."""

    if not isinstance(snapshot, EffectivePageSnapshot):
        raise TypeError("snapshot must be an EffectivePageSnapshot")
    if not isinstance(original_page, OriginalPageAssetBinding):
        raise TypeError("original_page must be an OriginalPageAssetBinding")
    parent_id = _coverage_identity(parent_id, "parent_id")
    matches = tuple(
        parent for parent in snapshot.parents if parent.parent_id == parent_id
    )
    if len(matches) != 1:
        raise ValueError("the selected user parent is not exact")
    parent = matches[0]
    if _other_current_user_parent_cleanup_coverage_ids(snapshot, parent_id):
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.COVERAGE_CONFLICT,
            _COVERAGE_CONFLICT_MESSAGE,
        )
    lineage = parent.lineage
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or parent.excluded
        or parent.root_id != lineage.root_id
        or parent.role != lineage.role
        or lineage.authored_edit_id not in parent.applied_edit_ids
    ):
        raise ValueError("the selected parent has no current user-authored lineage")
    if tuple(lineage.canvas_size) != original_page.canvas_size:
        raise ValueError("the original page canvas does not match parent lineage")
    for stage in (RevisionStage.SOURCE, RevisionStage.TRANSLATION):
        requirement = _stage_requirement(snapshot, parent_id, stage)
        if not (
            requirement is not None
            and requirement.state is RevisionStageState.CURRENT
            and requirement.required_action is RevisionRequiredAction.NONE
        ):
            raise ValueError(f"the selected parent {stage.value} revision is not current")
    cleanup_requirement = _stage_requirement(
        snapshot,
        parent_id,
        RevisionStage.CLEANUP_BASE,
    )
    if not (
        cleanup_requirement is not None
        and cleanup_requirement.state is RevisionStageState.STALE
        and cleanup_requirement.required_action is RevisionRequiredAction.REBUILD
    ):
        raise ValueError("the selected parent does not require cleanup-base coverage")
    cleaned = thaw_json(snapshot.cleaned_page_base)
    if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
        raise ValueError("the selected input CleanedPageBase is unavailable")
    input_sha256 = _coverage_sha256(
        cleaned.get("content_sha256"),
        "input_cleaned_base_content_sha256",
    )
    if str(cleaned.get("revision_id") or "") != snapshot.cleaned_base_revision_id:
        raise ValueError("the selected input CleanedPageBase identity is inconsistent")
    source = _metadata_dict(parent.source_revision_metadata)
    translation = _metadata_dict(parent.target_revision_metadata)
    try:
        source_artifact = OcrSourceRevisionArtifact.from_record(source)
        translation_artifact = TranslationRevisionArtifact.from_record(
            translation
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source and translation revision artifacts are invalid"
        ) from exc
    source_revision_id = _coverage_identity(
        source_artifact.revision_id,
        "source_revision_id",
    )
    source_selection_edit_id = _coverage_identity(
        source_artifact.selection_edit_id,
        "source_selection_edit_id",
    )
    translation_revision_id = _coverage_identity(
        translation_artifact.revision_id,
        "translation_revision_id",
    )
    translation_selection_edit_id = _coverage_identity(
        translation_artifact.selection_edit_id,
        "translation_selection_edit_id",
    )
    source_fingerprint = effective_source_fingerprint(
        parent.parent_id,
        parent.source_text,
    )
    if (
        parent.source_revision_id != source_revision_id
        or source_artifact.page_id != snapshot.page_id
        or source_artifact.parent_id != parent.parent_id
        or source_artifact.root_id != lineage.root_id
        or source_artifact.parent_authored_edit_id != lineage.authored_edit_id
        or source_artifact.original_page != original_page
        or source_artifact.sampling_bbox
        != tuple(lineage.workflow_area_bbox)
        or source_artifact.hierarchy_revision_id
        != snapshot.hierarchy.revision_id
        or source_artifact.hierarchy_fingerprint
        != snapshot.hierarchy.fingerprint
        or source_artifact.source_text != parent.source_text
        or source_selection_edit_id not in parent.applied_edit_ids
        or parent.target_revision_id != translation_revision_id
        or translation_artifact.project_id != snapshot.project_id
        or translation_artifact.page_id != snapshot.page_id
        or translation_artifact.parent_id != parent.parent_id
        or translation_artifact.root_id != lineage.root_id
        or translation_artifact.parent_authored_edit_id
        != lineage.authored_edit_id
        or translation_artifact.parent_role != parent.role
        or translation_artifact.bubble_local_nested_speech
        or translation_artifact.target_text != parent.target_text
        or translation_artifact.source_text != parent.source_text
        or translation_artifact.source_authority != parent.source_authority
        or translation_artifact.source_revision_id != source_revision_id
        or translation_artifact.source_selection_edit_id
        != source_selection_edit_id
        or translation_artifact.source_fingerprint != source_fingerprint
        or translation_artifact.hierarchy_revision_id
        != snapshot.hierarchy.revision_id
        or translation_artifact.hierarchy_fingerprint
        != snapshot.hierarchy.fingerprint
        or translation_selection_edit_id not in parent.applied_edit_ids
    ):
        raise ValueError("source and translation revision ancestry is inconsistent")
    return UserParentCleanupCoverageTargetV1(
        project_id=snapshot.project_id,
        page_id=snapshot.page_id,
        parent_id=parent.parent_id,
        root_id=parent.root_id,
        parent_authored_edit_id=lineage.authored_edit_id,
        parent_role=parent.role,
        workflow_area_bbox=tuple(lineage.workflow_area_bbox),
        canvas_size=tuple(lineage.canvas_size),
        original_page_asset_id=original_page.asset_id,
        original_page_asset_reference=original_page.asset_reference,
        original_page_content_sha256=original_page.content_sha256,
        input_cleaned_base_revision_id=snapshot.cleaned_base_revision_id,
        input_cleaned_base_content_sha256=input_sha256,
        hierarchy_revision_id=snapshot.hierarchy.revision_id,
        hierarchy_fingerprint=snapshot.hierarchy.fingerprint,
        source_revision_id=source_revision_id,
        source_selection_edit_id=source_selection_edit_id,
        source_artifact_sha256=canonical_sha256(source),
        source_fingerprint=source_fingerprint,
        translation_revision_id=translation_revision_id,
        translation_selection_edit_id=translation_selection_edit_id,
        translation_artifact_sha256=canonical_sha256(translation),
        effective_page_fingerprint=snapshot.effective_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class ManualCleanupRebaseReview:
    """Validated saved mask set bound to one newly selected clean base.

    This contract deliberately carries no prior result pixels.  It is only an
    immutable review input for an explicit new preview against ``current_*``.
    """

    page_id: str
    stale_selection_edit_id: str
    stale_selection_edit_ids: tuple[str, ...]
    stale_revision_id: str
    stale_operation_id: str
    stale_effective_fingerprint: str
    stale_input_base_revision_id: str
    stale_input_base_sha256: str
    current_base_revision_id: str
    current_base_sha256: str
    current_base_path: str
    current_effective_fingerprint: str
    source_image_path: str
    canvas_size: tuple[int, int]
    erase_mask_png: bytes
    erase_mask_sha256: str
    protect_mask_png: bytes
    protect_mask_sha256: str
    effective_mask_png: bytes
    effective_mask_sha256: str
    parameters: ManualCleanupParameters

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": "manual_cleanup_rebase_review_v1",
                "page_id": self.page_id,
                "stale_selection_edit_id": self.stale_selection_edit_id,
                "stale_selection_edit_ids": list(self.stale_selection_edit_ids),
                "stale_revision_id": self.stale_revision_id,
                "stale_operation_id": self.stale_operation_id,
                "stale_effective_fingerprint": self.stale_effective_fingerprint,
                "stale_input_base_revision_id": self.stale_input_base_revision_id,
                "stale_input_base_sha256": self.stale_input_base_sha256,
                "current_base_revision_id": self.current_base_revision_id,
                "current_base_sha256": self.current_base_sha256,
                "current_effective_fingerprint": self.current_effective_fingerprint,
                "canvas_size": list(self.canvas_size),
                "erase_mask_sha256": self.erase_mask_sha256,
                "protect_mask_sha256": self.protect_mask_sha256,
                "effective_mask_sha256": self.effective_mask_sha256,
                "parameters": self.parameters.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class ManualCleanupPreflight:
    page_id: str
    code: ManualCleanupAvailabilityCode
    ready: bool
    message: str
    canvas_size: tuple[int, int] = ()
    input_base_revision_id: str = ""
    input_base_sha256: str = ""
    effective_mask_pixels: int = 0
    protected_pixels: int = 0
    backend_id: str = ""
    selected_base_path: str = ""
    source_image_path: str = ""


@dataclass(frozen=True, slots=True)
class ManualCleanupContext:
    page_id: str
    ready: bool
    code: ManualCleanupAvailabilityCode
    message: str
    canvas_size: tuple[int, int] = ()
    input_base_revision_id: str = ""
    input_base_sha256: str = ""
    selected_base_path: str = ""
    source_image_path: str = ""
    rebase_review: ManualCleanupRebaseReview | None = None


@dataclass(frozen=True, slots=True)
class ManualCleanupProgress:
    page_id: str
    stage: ManualCleanupStage
    completed_steps: int
    total_steps: int
    message: str = ""


def _normalized_reviewed_stale_binding(
    selection_edit_ids: tuple[str, ...],
    effective_fingerprint: str,
) -> tuple[tuple[str, ...], str]:
    if not isinstance(selection_edit_ids, tuple):
        raise TypeError("reviewed stale selection edit IDs must be a tuple")
    normalized_ids = tuple(str(value or "").strip() for value in selection_edit_ids)
    if any(not value for value in normalized_ids):
        raise ValueError("reviewed stale selection edit IDs must be non-empty")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("reviewed stale selection edit IDs must be unique")
    fingerprint = str(effective_fingerprint or "").strip().lower()
    if bool(normalized_ids) != bool(fingerprint):
        raise ValueError(
            "reviewed stale selection IDs and effective fingerprint must be paired"
        )
    if fingerprint and (
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            "reviewed stale effective fingerprint must be a SHA-256 digest"
        )
    return normalized_ids, fingerprint


@dataclass(frozen=True, slots=True)
class ManualCleanupRequest:
    snapshot: EffectivePageSnapshot
    erase_mask_png: bytes
    protect_mask_png: bytes | None = None
    parameters: ManualCleanupParameters = ManualCleanupParameters()
    expected_page_head_sha256: str = ""
    expected_global_head_sha256: str = ""
    reviewed_stale_selection_edit_ids: tuple[str, ...] = ()
    reviewed_stale_effective_fingerprint: str = ""
    operation_id: str = ""
    transaction_id: str = ""
    coverage_target: UserParentCleanupCoverageTargetV1 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EffectivePageSnapshot):
            raise TypeError("snapshot must be an EffectivePageSnapshot")
        if not isinstance(self.erase_mask_png, bytes):
            raise TypeError("erase_mask_png must be bytes")
        if self.protect_mask_png is not None and not isinstance(
            self.protect_mask_png, bytes
        ):
            raise TypeError("protect_mask_png must be bytes or None")
        if not isinstance(self.parameters, ManualCleanupParameters):
            raise TypeError("parameters must be ManualCleanupParameters")
        if self.coverage_target is not None and not isinstance(
            self.coverage_target,
            UserParentCleanupCoverageTargetV1,
        ):
            raise TypeError(
                "coverage_target must be a UserParentCleanupCoverageTargetV1 or None"
            )
        edit_ids, fingerprint = _normalized_reviewed_stale_binding(
            self.reviewed_stale_selection_edit_ids,
            self.reviewed_stale_effective_fingerprint,
        )
        object.__setattr__(self, "reviewed_stale_selection_edit_ids", edit_ids)
        object.__setattr__(
            self,
            "reviewed_stale_effective_fingerprint",
            fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ManualCleanupPreviewLease:
    operation_id: str
    project_id: str
    page_id: str
    input_base_revision_id: str
    input_base_sha256: str
    input_descriptor_sha256: str
    automatic_page_sha256: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str
    transaction_id: str
    canvas_size: tuple[int, int]
    image_mode: str
    result_path: str
    result_sha256: str
    erase_mask_path: str
    erase_mask_sha256: str
    protect_mask_path: str
    protect_mask_sha256: str
    effective_mask_path: str
    effective_mask_sha256: str
    effective_mask_pixels: int
    protected_pixels: int
    parameters: ManualCleanupParameters
    backend_id: str
    backend_name: str
    backend_family: str
    backend_model_path: str
    backend_adapter_path: str
    backend_version: str
    backend_runtime_ms: float
    created_at: str
    reviewed_stale_selection_edit_ids: tuple[str, ...] = ()
    reviewed_stale_effective_fingerprint: str = ""
    coverage_target: UserParentCleanupCoverageTargetV1 | None = None

    def __post_init__(self) -> None:
        edit_ids, fingerprint = _normalized_reviewed_stale_binding(
            self.reviewed_stale_selection_edit_ids,
            self.reviewed_stale_effective_fingerprint,
        )
        object.__setattr__(self, "reviewed_stale_selection_edit_ids", edit_ids)
        object.__setattr__(
            self,
            "reviewed_stale_effective_fingerprint",
            fingerprint,
        )
        if self.coverage_target is not None and not isinstance(
            self.coverage_target,
            UserParentCleanupCoverageTargetV1,
        ):
            raise TypeError(
                "coverage_target must be a UserParentCleanupCoverageTargetV1 or None"
            )


@dataclass(frozen=True, slots=True)
class ManualCleanupReceipt:
    operation_id: str
    page_id: str
    provenance: str
    status: ManualCleanupStatus
    input_base_revision_id: str
    input_base_sha256: str
    erase_mask_sha256: str
    protect_mask_sha256: str
    effective_mask_sha256: str
    result_sha256: str
    canvas_size: tuple[int, int]
    parameters: ManualCleanupParameters
    backend_id: str
    backend_name: str
    backend_family: str
    backend_model_path: str
    backend_adapter_path: str
    backend_version: str
    backend_runtime_ms: float
    effective_mask_pixels: int
    protected_pixels: int
    page_bounds_validated: bool
    started_at: str
    completed_at: str
    reviewed_stale_selection_edit_ids: tuple[str, ...] = ()
    reviewed_stale_effective_fingerprint: str = ""
    output_path: str = ""
    revision_id: str = ""
    selection_edit_id: str = ""
    cancellation_stage: str = ""
    preview_lease: ManualCleanupPreviewLease | None = None
    commit_receipt: ProjectEditCommitReceipt | None = None
    coverage_target: UserParentCleanupCoverageTargetV1 | None = None

    def __post_init__(self) -> None:
        if self.coverage_target is not None and not isinstance(
            self.coverage_target,
            UserParentCleanupCoverageTargetV1,
        ):
            raise TypeError(
                "coverage_target must be a UserParentCleanupCoverageTargetV1 or None"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable receipt body stored with a committed base."""

        result = {
            "manual_cleanup_receipt_version": MANUAL_CLEANUP_RECEIPT_VERSION,
            "operation_id": self.operation_id,
            "page_id": self.page_id,
            "provenance": self.provenance,
            "status": self.status.value,
            "input_base_revision_id": self.input_base_revision_id,
            "input_base_sha256": self.input_base_sha256,
            "erase_mask_sha256": self.erase_mask_sha256,
            "protect_mask_sha256": self.protect_mask_sha256,
            "effective_mask_sha256": self.effective_mask_sha256,
            "preview_sha256": self.result_sha256,
            "result_sha256": self.result_sha256,
            "canvas_size": list(self.canvas_size),
            "parameters": self.parameters.to_dict(),
            "backend": {
                "candidate_id": self.backend_id,
                "name": self.backend_name,
                "family": self.backend_family,
                "model_path": self.backend_model_path,
                "adapter_path": self.backend_adapter_path,
                "version": self.backend_version,
                "runtime_ms": self.backend_runtime_ms,
            },
            "effective_mask_pixels": self.effective_mask_pixels,
            "protected_pixels": self.protected_pixels,
            "page_bounds_validated": self.page_bounds_validated,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "reviewed_stale_selection_edit_ids": list(
                self.reviewed_stale_selection_edit_ids
            ),
            "reviewed_stale_effective_fingerprint": (
                self.reviewed_stale_effective_fingerprint
            ),
            "result_asset": self.output_path,
            "revision_id": self.revision_id,
            "selection_edit_id": self.selection_edit_id,
            "cancellation_stage": self.cancellation_stage,
        }
        if self.coverage_target is not None:
            result["user_parent_cleanup_coverage_target"] = (
                self.coverage_target.to_dict()
            )
        return result


ProgressCallback = Callable[[ManualCleanupProgress], None]
BackendInventory = Callable[[], Sequence[Any]]
BackendRunner = Callable[..., Any]


class _Cancelled(RuntimeError):
    def __init__(self, stage: ManualCleanupStage) -> None:
        super().__init__(stage.value)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class _PinnedBase:
    path: str
    payload: bytes
    content_sha256: str
    canvas_size: tuple[int, int]
    image_mode: str
    image: Image.Image


@dataclass(frozen=True, slots=True)
class _PreparedMasks:
    erase: Image.Image
    protect: Image.Image
    effective: Image.Image
    erase_png: bytes
    protect_png: bytes
    effective_png: bytes
    erase_sha256: str
    protect_sha256: str
    effective_sha256: str
    effective_pixels: int
    protected_pixels: int


class ManualCleanupService:
    """Create user cleanup revisions without touching automatic proof state."""

    def __init__(
        self,
        *,
        project_path: str,
        edit_store: ProjectEditStore | None = None,
        artifact_root: str | None = None,
        backend_inventory: BackendInventory | None = None,
        backend_runner: BackendRunner | None = None,
    ) -> None:
        raw = str(project_path or "").strip()
        if not raw:
            raise ValueError("project_path is required")
        self.project_path = os.path.abspath(raw)
        self.project_directory = os.path.dirname(self.project_path) or os.getcwd()
        self.edit_store = edit_store
        self.artifact_root = os.path.abspath(
            artifact_root or manual_cleanup_artifact_root(self.project_path)
        )
        self._backend_inventory = backend_inventory or _fixed_backend_inventory
        self._backend_runner = backend_runner or _fixed_backend_runner
        self._preview_lock = threading.RLock()
        self._preview_directories: dict[str, str] = {}

    def rebase_review(
        self,
        project: Mapping[str, Any],
        ledger: ProjectEditLedger,
        snapshot: EffectivePageSnapshot,
    ) -> ManualCleanupRebaseReview | None:
        """Discover one active stale cleanup selection and validate its masks.

        The returned bytes are the saved user masks only.  The prior inpainted
        result is neither opened nor exposed, so a caller must run a new
        preview against the currently selected clean base.
        """

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(ledger, ProjectEditLedger):
            raise TypeError("ledger must be a ProjectEditLedger")
        if not isinstance(snapshot, EffectivePageSnapshot):
            raise TypeError("snapshot must be an EffectivePageSnapshot")
        if not _has_cleanup_rebase_issue(snapshot):
            return None
        selection_chain = _active_cleanup_selection_chain(ledger, snapshot.page_id)
        if not selection_chain:
            return None
        selection = selection_chain[-1]
        revision_id = str(selection.payload.get("revision_id") or "").strip()
        if not revision_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale cleanup selection has no revision identity.",
            )
        current_snapshot = _cleanup_rebase_base_snapshot(project, ledger, snapshot)
        if any(
            issue.domain == EditDomain.CLEANUP.value
            for issue in current_snapshot.issues
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.STALE_BASE,
                "A rebuilt automatic/current CleanedPageBase compatible with the "
                "effective hierarchy is required before rebase.",
            )
        if revision_id == current_snapshot.cleaned_base_revision_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.STALE_BASE,
                "A distinct rebuilt automatic/current CleanedPageBase is required "
                "before rebase.",
            )
        artifact = _manual_cleanup_revision(project, snapshot.page_id, revision_id)
        return self._validated_rebase_review(
            selection,
            selection_chain,
            artifact,
            snapshot,
            current_snapshot,
        )

    def rebase_snapshot(
        self,
        project: Mapping[str, Any],
        ledger: ProjectEditLedger,
        stale_snapshot: EffectivePageSnapshot,
        review: ManualCleanupRebaseReview,
    ) -> EffectivePageSnapshot:
        """Resolve and revalidate the compatible automatic/current substrate."""

        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(ledger, ProjectEditLedger):
            raise TypeError("ledger must be a ProjectEditLedger")
        if not isinstance(stale_snapshot, EffectivePageSnapshot):
            raise TypeError("stale_snapshot must be an EffectivePageSnapshot")
        if not isinstance(review, ManualCleanupRebaseReview):
            raise TypeError("review must be a ManualCleanupRebaseReview")
        if (
            stale_snapshot.page_id != review.page_id
            or stale_snapshot.effective_fingerprint
            != review.stale_effective_fingerprint
            or tuple(
                edit.edit_id
                for edit in _active_cleanup_selection_chain(
                    ledger,
                    stale_snapshot.page_id,
                )
            )
            != review.stale_selection_edit_ids
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The stale cleanup selection changed; reload the editor.",
            )
        current_snapshot = _cleanup_rebase_base_snapshot(
            project,
            ledger,
            stale_snapshot,
        )
        self._validate_rebase_binding(current_snapshot, review)
        return current_snapshot

    def context(
        self,
        snapshot: EffectivePageSnapshot,
        *,
        rebase_review: ManualCleanupRebaseReview | None = None,
        coverage_target: UserParentCleanupCoverageTargetV1 | None = None,
    ) -> ManualCleanupContext:
        """Return mask-free read-only editor context for one selected base."""

        if not isinstance(snapshot, EffectivePageSnapshot):
            raise TypeError("snapshot must be an EffectivePageSnapshot")
        cleanup_issues = tuple(
            issue for issue in snapshot.issues if issue.domain == EditDomain.CLEANUP.value
        )
        if cleanup_issues and rebase_review is None:
            missing = any(
                issue.kind is ProjectionIssueKind.MISSING_DEPENDENCY
                for issue in cleanup_issues
            )
            stale = any(
                issue.kind is ProjectionIssueKind.STALE_DEPENDENCY
                for issue in cleanup_issues
            )
            code = (
                ManualCleanupAvailabilityCode.MISSING_BASE
                if missing
                else ManualCleanupAvailabilityCode.STALE_BASE
                if stale
                else ManualCleanupAvailabilityCode.BLOCKED
            )
            return ManualCleanupContext(
                page_id=snapshot.page_id,
                ready=False,
                code=code,
                message="The selected CleanedPageBase is unavailable or stale.",
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                source_image_path=self._source_image_path(snapshot),
            )
        try:
            if rebase_review is not None:
                self._validate_rebase_binding(snapshot, rebase_review)
            pinned = self._validated_clean_base(snapshot, publish_pin=False)
            self._validate_coverage_target(
                snapshot,
                coverage_target,
                pinned=pinned,
            )
        except ManualCleanupFailure as exc:
            return ManualCleanupContext(
                page_id=snapshot.page_id,
                ready=False,
                code=_availability_code_for_failure(exc.code),
                message=exc.message,
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                source_image_path=self._source_image_path(snapshot),
            )
        return ManualCleanupContext(
            page_id=snapshot.page_id,
            ready=True,
            code=ManualCleanupAvailabilityCode.READY,
            message=(
                "Saved cleanup mask is ready on the rebuilt current base."
                if rebase_review is not None
                else "Manual cleanup editor is ready."
            ),
            canvas_size=pinned.canvas_size,
            input_base_revision_id=snapshot.cleaned_base_revision_id,
            input_base_sha256=pinned.content_sha256,
            selected_base_path=pinned.path,
            source_image_path=self._source_image_path(snapshot),
            rebase_review=rebase_review,
        )

    def preflight_rebase(
        self,
        snapshot: EffectivePageSnapshot,
        review: ManualCleanupRebaseReview,
    ) -> ManualCleanupPreflight:
        """Preflight the exact saved masks against the review's current base."""

        self._validate_rebase_binding(snapshot, review)
        return self.preflight(
            _without_cleanup_issues(snapshot),
            review.erase_mask_png,
            review.protect_mask_png,
            parameters=review.parameters,
        )

    def preflight(
        self,
        snapshot: EffectivePageSnapshot,
        erase_mask_png: bytes,
        protect_mask_png: bytes | None = None,
        *,
        parameters: ManualCleanupParameters | None = None,
        coverage_target: UserParentCleanupCoverageTargetV1 | None = None,
    ) -> ManualCleanupPreflight:
        parameters = parameters or ManualCleanupParameters()
        if not isinstance(snapshot, EffectivePageSnapshot):
            raise TypeError("snapshot must be an EffectivePageSnapshot")
        cleanup_issues = tuple(
            issue for issue in snapshot.issues if issue.domain == EditDomain.CLEANUP.value
        )
        if cleanup_issues:
            missing = any(
                issue.kind is ProjectionIssueKind.MISSING_DEPENDENCY
                for issue in cleanup_issues
            )
            stale = any(
                issue.kind is ProjectionIssueKind.STALE_DEPENDENCY
                for issue in cleanup_issues
            )
            return ManualCleanupPreflight(
                page_id=snapshot.page_id,
                code=(
                    ManualCleanupAvailabilityCode.MISSING_BASE
                    if missing
                    else ManualCleanupAvailabilityCode.STALE_BASE
                    if stale
                    else ManualCleanupAvailabilityCode.BLOCKED
                ),
                ready=False,
                message="The selected CleanedPageBase is unavailable or stale.",
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                backend_id=parameters.backend_id,
            )
        try:
            pinned = self._validated_clean_base(snapshot, publish_pin=False)
            self._validate_coverage_target(
                snapshot,
                coverage_target,
                pinned=pinned,
            )
        except ManualCleanupFailure as exc:
            return ManualCleanupPreflight(
                page_id=snapshot.page_id,
                code=_availability_code_for_failure(exc.code),
                ready=False,
                message=exc.message,
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                backend_id=parameters.backend_id,
            )
        try:
            masks = _prepare_masks(
                erase_mask_png,
                protect_mask_png,
                canvas_size=pinned.canvas_size,
                parameters=parameters,
            )
        except ManualCleanupFailure as exc:
            return ManualCleanupPreflight(
                page_id=snapshot.page_id,
                code=ManualCleanupAvailabilityCode.INVALID_MASK,
                ready=False,
                message=exc.message,
                canvas_size=pinned.canvas_size,
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                input_base_sha256=pinned.content_sha256,
                backend_id=parameters.backend_id,
            )
        candidate = self._candidate(parameters.backend_id)
        if candidate is None or not bool(getattr(candidate, "available", False)):
            return ManualCleanupPreflight(
                page_id=snapshot.page_id,
                code=ManualCleanupAvailabilityCode.BACKEND_UNAVAILABLE,
                ready=False,
                message="The selected fixed cleanup backend is unavailable.",
                canvas_size=pinned.canvas_size,
                input_base_revision_id=snapshot.cleaned_base_revision_id,
                input_base_sha256=pinned.content_sha256,
                effective_mask_pixels=masks.effective_pixels,
                protected_pixels=masks.protected_pixels,
                backend_id=parameters.backend_id,
                selected_base_path=pinned.path,
                source_image_path=self._source_image_path(snapshot),
            )
        return ManualCleanupPreflight(
            page_id=snapshot.page_id,
            code=ManualCleanupAvailabilityCode.READY,
            ready=True,
            message="Ready to preview manual cleanup.",
            canvas_size=pinned.canvas_size,
            input_base_revision_id=snapshot.cleaned_base_revision_id,
            input_base_sha256=pinned.content_sha256,
            effective_mask_pixels=masks.effective_pixels,
            protected_pixels=masks.protected_pixels,
            backend_id=parameters.backend_id,
            selected_base_path=pinned.path,
            source_image_path=self._source_image_path(snapshot),
        )

    def preview(
        self,
        request: ManualCleanupRequest,
        *,
        cancellation: CancellationProbe | None = None,
        progress: ProgressCallback | None = None,
    ) -> ManualCleanupReceipt:
        if not isinstance(request, ManualCleanupRequest):
            raise TypeError("request must be ManualCleanupRequest")
        started_at = _utc_now()
        started = time.perf_counter()
        operation_id = _operation_id(request.operation_id)
        try:
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.VALIDATING,
                0,
                "Validating selected CleanedPageBase",
            )
            self._check_cancel(cancellation, ManualCleanupStage.VALIDATING)
            self._validate_coverage_preview_heads(request)
            preflight = self.preflight(
                request.snapshot,
                request.erase_mask_png,
                request.protect_mask_png,
                parameters=request.parameters,
                coverage_target=request.coverage_target,
            )
            if not preflight.ready:
                raise ManualCleanupFailure(
                    _failure_code_for_preflight(preflight.code),
                    preflight.message,
                    stage=ManualCleanupStage.VALIDATING,
                )
            pinned = self._validated_clean_base(
                request.snapshot,
                publish_pin=True,
            )
            self._check_cancel(cancellation, ManualCleanupStage.PREPARING_MASKS)
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.PREPARING_MASKS,
                1,
                "Preparing bounded erase and protect masks",
            )
            masks = _prepare_masks(
                request.erase_mask_png,
                request.protect_mask_png,
                canvas_size=pinned.canvas_size,
                parameters=request.parameters,
            )
            candidate = self._candidate(request.parameters.backend_id)
            if candidate is None or not bool(getattr(candidate, "available", False)):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.BACKEND_UNAVAILABLE,
                    "The selected fixed cleanup backend is unavailable.",
                    stage=ManualCleanupStage.INPAINTING,
                )
            self._check_cancel(cancellation, ManualCleanupStage.INPAINTING)
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.INPAINTING,
                2,
                "Running the existing fixed cleanup backend",
            )
            backend_started = time.perf_counter()
            try:
                execution = self._backend_runner(
                    image=pinned.image.copy(),
                    mask=masks.effective.copy(),
                    candidate=candidate,
                    use_gpu=request.parameters.use_gpu,
                )
            except Exception as exc:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.BACKEND_FAILED,
                    "Manual cleanup backend execution failed: "
                    f"{type(exc).__name__}: {exc}",
                    stage=ManualCleanupStage.INPAINTING,
                ) from exc
            backend_runtime_ms = round(
                float(getattr(execution, "runtime_ms", 0.0))
                or (time.perf_counter() - backend_started) * 1000.0,
                6,
            )
            status = str(getattr(execution, "status", "") or "")
            backend_image = getattr(execution, "cleaned_image", None)
            if status != "completed" or not isinstance(backend_image, Image.Image):
                detail = str(getattr(execution, "detail", "") or status or "unknown")
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.BACKEND_FAILED,
                    f"Manual cleanup backend did not complete: {detail}",
                    stage=ManualCleanupStage.INPAINTING,
                )
            if backend_image.size != pinned.canvas_size:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.BACKEND_FAILED,
                    "Manual cleanup backend returned the wrong page dimensions.",
                    stage=ManualCleanupStage.INPAINTING,
                )
            self._check_cancel(cancellation, ManualCleanupStage.PUBLISHING_PREVIEW)
            result = _protected_composite(
                pinned.image,
                backend_image,
                masks.effective,
                masks.protect,
                output_mode=pinned.image_mode,
            )
            result_png = _png_bytes(result)
            result_sha256 = _sha256(result_png)
            self._validate_coverage_preview_heads(request)
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.PUBLISHING_PREVIEW,
                3,
                "Publishing managed preview artifacts",
            )
            paths = self._publish_preview(
                page_id=request.snapshot.page_id,
                operation_id=operation_id,
                result_png=result_png,
                masks=masks,
            )
            self._check_cancel(cancellation, ManualCleanupStage.PUBLISHING_PREVIEW)
            lease = ManualCleanupPreviewLease(
                operation_id=operation_id,
                project_id=request.snapshot.project_id,
                page_id=request.snapshot.page_id,
                input_base_revision_id=request.snapshot.cleaned_base_revision_id,
                input_base_sha256=pinned.content_sha256,
                input_descriptor_sha256=canonical_sha256(
                    thaw_json(request.snapshot.cleaned_page_base)
                ),
                automatic_page_sha256=request.snapshot.automatic_fingerprint,
                expected_page_head_sha256=str(
                    request.expected_page_head_sha256 or ""
                ).lower(),
                expected_global_head_sha256=str(
                    request.expected_global_head_sha256 or ""
                ).lower(),
                transaction_id=str(request.transaction_id or "").strip(),
                canvas_size=pinned.canvas_size,
                image_mode=pinned.image_mode,
                result_path=paths["result"],
                result_sha256=result_sha256,
                erase_mask_path=paths["erase"],
                erase_mask_sha256=masks.erase_sha256,
                protect_mask_path=paths["protect"],
                protect_mask_sha256=masks.protect_sha256,
                effective_mask_path=paths["effective"],
                effective_mask_sha256=masks.effective_sha256,
                effective_mask_pixels=masks.effective_pixels,
                protected_pixels=masks.protected_pixels,
                parameters=request.parameters,
                backend_id=str(getattr(candidate, "candidate_id", "") or ""),
                backend_name=str(
                    getattr(execution, "backend_name", "")
                    or getattr(candidate, "candidate_id", "")
                ),
                backend_family=str(
                    getattr(execution, "backend_family", "")
                    or getattr(candidate, "backend_family", "")
                ),
                backend_model_path=str(
                    getattr(execution, "model_path", "")
                    or getattr(candidate, "model_path", "")
                ),
                backend_adapter_path=str(
                    getattr(execution, "adapter_path", "")
                    or getattr(candidate, "adapter_path", "")
                ),
                backend_version=_backend_version_fingerprint(candidate, execution),
                backend_runtime_ms=backend_runtime_ms,
                created_at=_utc_now(),
                reviewed_stale_selection_edit_ids=(
                    request.reviewed_stale_selection_edit_ids
                ),
                reviewed_stale_effective_fingerprint=(
                    request.reviewed_stale_effective_fingerprint
                ),
                coverage_target=request.coverage_target,
            )
            completed_at = _utc_now()
            receipt = _receipt_from_lease(
                lease,
                status=ManualCleanupStatus.PREVIEW_READY,
                started_at=started_at,
                completed_at=completed_at,
                output_path=lease.result_path,
                preview_lease=lease,
            )
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.COMPLETED,
                4,
                f"Preview ready in {(time.perf_counter() - started) * 1000.0:.1f} ms",
            )
            return receipt
        except _Cancelled as cancelled:
            self.discard_preview(request.snapshot.page_id)
            self._emit(
                progress,
                request.snapshot.page_id,
                ManualCleanupStage.CANCELLED,
                0,
                "Manual cleanup preview cancelled",
            )
            return _cancelled_receipt(
                operation_id,
                request,
                started_at,
                cancelled.stage,
            )

    def preview_rebase(
        self,
        request: ManualCleanupRequest,
        review: ManualCleanupRebaseReview,
        *,
        cancellation: CancellationProbe | None = None,
        progress: ProgressCallback | None = None,
    ) -> ManualCleanupReceipt:
        """Run a fresh preview from saved masks on the newly selected base."""

        if not isinstance(request, ManualCleanupRequest):
            raise TypeError("request must be ManualCleanupRequest")
        self._validate_rebase_binding(request.snapshot, review)
        if (
            request.erase_mask_png != review.erase_mask_png
            or request.protect_mask_png != review.protect_mask_png
            or request.parameters != review.parameters
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The saved rebase masks or parameters changed before preview.",
                stage=ManualCleanupStage.VALIDATING,
            )
        rebound = replace(
            request,
            snapshot=_without_cleanup_issues(request.snapshot),
            reviewed_stale_selection_edit_ids=(
                review.stale_selection_edit_ids
            ),
            reviewed_stale_effective_fingerprint=(
                review.stale_effective_fingerprint
            ),
        )
        return self.preview(
            rebound,
            cancellation=cancellation,
            progress=progress,
        )

    def commit_preview(
        self,
        lease: ManualCleanupPreviewLease,
        *,
        cancellation: CancellationProbe | None = None,
        progress: ProgressCallback | None = None,
        transaction_id: str = "",
    ) -> ManualCleanupReceipt:
        """Commit the exact preview bytes; the backend is never called here."""

        if not isinstance(lease, ManualCleanupPreviewLease):
            raise TypeError("lease must be a ManualCleanupPreviewLease")
        started_at = lease.created_at
        try:
            self._check_cancel(cancellation, ManualCleanupStage.COMMITTING)
            if self.edit_store is None:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "Manual cleanup commit requires a project edit store.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            if self.edit_store.project_id != lease.project_id:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "Project edit store identity does not match the preview.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            if os.path.normcase(os.path.abspath(self.edit_store.project_path)) != os.path.normcase(
                self.project_path
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "Project edit store path does not match the preview.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            expected_page_head = _require_sha256(
                lease.expected_page_head_sha256,
                "expected_page_head_sha256",
                failure_code=ManualCleanupFailureCode.COMMIT_STALE,
            )
            expected_global_head = _require_sha256(
                lease.expected_global_head_sha256,
                "expected_global_head_sha256",
                failure_code=ManualCleanupFailureCode.COMMIT_STALE,
            )
            if (
                self.edit_store.page_head(lease.page_id) != expected_page_head
                or self.edit_store.global_head() != expected_global_head
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.COMMIT_STALE,
                    "Project edit heads changed after the cleanup preview.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            current_project, current_ledger, projected_snapshot = self._current_snapshot(
                lease.page_id
            )
            rebase_selection_chain: tuple[Any, ...] = ()
            current_snapshot = projected_snapshot
            has_rebase_issue = _has_cleanup_rebase_issue(projected_snapshot)
            if lease.reviewed_stale_selection_edit_ids:
                current_reviewed_chain = _active_cleanup_selection_chain(
                    current_ledger,
                    lease.page_id,
                )
                if (
                    not has_rebase_issue
                    or tuple(
                        edit.edit_id for edit in current_reviewed_chain
                    )
                    != lease.reviewed_stale_selection_edit_ids
                    or projected_snapshot.effective_fingerprint
                    != lease.reviewed_stale_effective_fingerprint
                ):
                    raise ManualCleanupFailure(
                        ManualCleanupFailureCode.COMMIT_STALE,
                        "The reviewed stale cleanup selection changed after "
                        "preview.",
                        stage=ManualCleanupStage.COMMITTING,
                    )
                rebase_selection_chain = current_reviewed_chain
                current_snapshot = _cleanup_rebase_base_snapshot(
                    current_project,
                    current_ledger,
                    projected_snapshot,
                )
                if any(
                    issue.domain == EditDomain.CLEANUP.value
                    for issue in current_snapshot.issues
                ):
                    raise ManualCleanupFailure(
                        ManualCleanupFailureCode.STALE_BASE,
                        "The rebuilt automatic/current CleanedPageBase is no "
                        "longer compatible with the effective hierarchy.",
                        stage=ManualCleanupStage.COMMITTING,
                    )
            elif has_rebase_issue:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.COMMIT_STALE,
                    "The cleanup preview was not bound to the stale selection "
                    "that requires settlement.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            selected = thaw_json(current_snapshot.cleaned_page_base)
            if (
                current_snapshot.project_id != lease.project_id
                or current_snapshot.automatic_fingerprint
                != lease.automatic_page_sha256
                or current_snapshot.cleaned_base_revision_id
                != lease.input_base_revision_id
                or not isinstance(selected, Mapping)
                or canonical_sha256(selected) != lease.input_descriptor_sha256
                or str(selected.get("content_sha256") or "").lower()
                != lease.input_base_sha256
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STALE_BASE,
                    "The selected CleanedPageBase changed after the preview.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            # Revalidate the actual selected asset, not just its catalog record.
            pinned = self._validated_clean_base(
                current_snapshot,
                publish_pin=False,
            )
            if (
                pinned.content_sha256 != lease.input_base_sha256
                or pinned.canvas_size != lease.canvas_size
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STALE_BASE,
                    "The selected CleanedPageBase bytes changed after the preview.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            self._validate_coverage_target(
                current_snapshot,
                lease.coverage_target,
                pinned=pinned,
                project=current_project,
            )
            preview_payloads = self._validated_preview_payloads(lease)
            self._check_cancel(cancellation, ManualCleanupStage.COMMITTING)
            self._emit(
                progress,
                lease.page_id,
                ManualCleanupStage.COMMITTING,
                0,
                "Promoting the exact preview and masks",
            )
            durable, created = self._promote_preview(lease, preview_payloads)
            try:
                lineage = cleaned_base_automatic_lineage(selected)
                if lineage is None:
                    raise ManualCleanupFailure(
                        ManualCleanupFailureCode.ARTIFACT_INVALID,
                        "The selected CleanedPageBase lacks automatic erasure lineage.",
                        stage=ManualCleanupStage.COMMITTING,
                    )
                lineage_sha256 = canonical_sha256(lineage)
                revision_identity = canonical_sha256(
                    {
                        "operation_id": lease.operation_id,
                        "page_id": lease.page_id,
                        "input_base_revision_id": lease.input_base_revision_id,
                        "input_base_sha256": lease.input_base_sha256,
                        "result_sha256": lease.result_sha256,
                        "erase_mask_sha256": lease.erase_mask_sha256,
                        "protect_mask_sha256": lease.protect_mask_sha256,
                        "effective_mask_sha256": lease.effective_mask_sha256,
                        "parameters": lease.parameters.to_dict(),
                        "backend_id": lease.backend_id,
                        "coverage_dependency_fingerprint": (
                            lease.coverage_target.coverage_dependency_fingerprint
                            if lease.coverage_target is not None
                            else ""
                        ),
                    }
                )
                revision_id = (
                    f"manual-cleaned:{lease.page_id}:{revision_identity[:32]}"
                )
                active_selection_chain = (
                    rebase_selection_chain
                    or _active_cleanup_selection_chain(
                        current_ledger,
                        lease.page_id,
                    )
                )
                active_selection = (
                    active_selection_chain[-1]
                    if active_selection_chain
                    else None
                )
                target = EditTarget(EditTargetKind.PAGE)
                payload = {"revision_id": revision_id}
                page = _find_page(current_project, lease.page_id)
                base_fingerprint = field_base_fingerprint(
                    project=current_project,
                    page=page,
                    target=target,
                    domain=EditDomain.CLEANUP,
                    operation="select_revision",
                    payload=payload,
                )
                if base_fingerprint is None:
                    raise ManualCleanupFailure(
                        ManualCleanupFailureCode.ARTIFACT_INVALID,
                        "The automatic cleanup selection base cannot be fingerprinted.",
                        stage=ManualCleanupStage.COMMITTING,
                    )
                selection = create_project_edit(
                    project_id=lease.project_id,
                    page_id=lease.page_id,
                    target=target,
                    domain=EditDomain.CLEANUP,
                    operation="select_revision",
                    payload=payload,
                    base_revision_id=lease.input_base_revision_id,
                    base_fingerprint=base_fingerprint,
                    supersedes_edit_id=(
                        active_selection.edit_id
                        if active_selection is not None
                        else None
                    ),
                    edit_id=f"manual-cleanup-select-{uuid.uuid4().hex}",
                    created_at=_utc_now(),
                )
                # Projection validates edit eligibility before it resolves
                # supersession.  Every ledger-active ancestor in a rejected
                # cleanup chain must therefore be settled atomically; revoking
                # only the latest head lets an older stale selection resurface.
                stale_selection_controls: list[Any] = []
                controlled_ledger = current_ledger
                for stale_selection in rebase_selection_chain:
                    controlled_ledger = controlled_ledger.revoke(
                        stale_selection.edit_id,
                        event_id=(
                            "manual-cleanup-rebase-revoke-"
                            f"{uuid.uuid4().hex}"
                        ),
                        created_at=_utc_now(),
                    )
                    stale_selection_controls.append(controlled_ledger.edits[-1])
                completed_at = _utc_now()
                portable_result = _portable_asset_path(
                    durable["result"], self.project_directory
                )
                committed_receipt = _receipt_from_lease(
                    lease,
                    status=ManualCleanupStatus.COMMITTED,
                    started_at=started_at,
                    completed_at=completed_at,
                    output_path=portable_result,
                    revision_id=revision_id,
                    selection_edit_id=selection.edit_id,
                )
                nested = selected.get("cleaned_page_base")
                artifact = {
                    "catalog": "cleaned_page_bases",
                    "manual_cleaned_base_revision_version": (
                        MANUAL_CLEANED_BASE_REVISION_VERSION
                    ),
                    "revision_id": revision_id,
                    "page_id": lease.page_id,
                    "provenance": "user_manual_cleanup",
                    "current": False,
                    "valid": True,
                    "state": "manual_cleanup_committed",
                    "asset": portable_result,
                    "content_sha256": lease.result_sha256,
                    "canvas_size": list(lease.canvas_size),
                    "image_mode": lease.image_mode,
                    "input_base_revision_id": lease.input_base_revision_id,
                    "input_base_sha256": lease.input_base_sha256,
                    "erase_mask_asset": _portable_asset_path(
                        durable["erase"], self.project_directory
                    ),
                    "erase_mask_sha256": lease.erase_mask_sha256,
                    "protect_mask_asset": _portable_asset_path(
                        durable["protect"], self.project_directory
                    ),
                    "protect_mask_sha256": lease.protect_mask_sha256,
                    "effective_mask_asset": _portable_asset_path(
                        durable["effective"], self.project_directory
                    ),
                    "effective_mask_sha256": lease.effective_mask_sha256,
                    "automatic_cleanup_lineage": lineage,
                    "automatic_cleanup_lineage_sha256": lineage_sha256,
                    "cleaned_page_base": (
                        dict(nested) if isinstance(nested, Mapping) else {}
                    ),
                    "manual_cleanup_receipt": committed_receipt.to_dict(),
                }
                if lease.coverage_target is not None:
                    artifact["user_parent_cleanup_coverage_target"] = (
                        lease.coverage_target.to_dict()
                    )
                self._emit(
                    progress,
                    lease.page_id,
                    ManualCleanupStage.PERSISTING,
                    1,
                    "Persisting the cleanup selection and revision atomically",
                )
                self._check_cancel(cancellation, ManualCleanupStage.PERSISTING)
                store_receipt = self.edit_store.commit_page_edits(
                    (*stale_selection_controls, selection),
                    automatic_page_sha256=lease.automatic_page_sha256,
                    expected_page_head_sha256=expected_page_head,
                    expected_global_head_sha256=expected_global_head,
                    artifact_revisions=(artifact,),
                    transaction_id=(
                        str(transaction_id or "").strip()
                        or lease.transaction_id
                        or f"gui4-{uuid.uuid4().hex}"
                    ),
                )
            except (StalePageEditHeadError, StaleProjectEditHeadError) as exc:
                _remove_created_files(created)
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.COMMIT_STALE,
                    str(exc),
                    stage=ManualCleanupStage.COMMITTING,
                ) from exc
            except Exception:
                _remove_created_files(created)
                raise
            self.discard_preview(lease.page_id)
            completed = replace(
                committed_receipt,
                commit_receipt=store_receipt,
            )
            self._emit(
                progress,
                lease.page_id,
                ManualCleanupStage.COMPLETED,
                1,
                "Manual cleanup revision committed",
            )
            return completed
        except _Cancelled as cancelled:
            self.discard_preview(lease.page_id)
            self._emit(
                progress,
                lease.page_id,
                ManualCleanupStage.CANCELLED,
                0,
                "Manual cleanup commit cancelled",
            )
            return _receipt_from_lease(
                lease,
                status=ManualCleanupStatus.CANCELLED,
                started_at=started_at,
                completed_at=_utc_now(),
                cancellation_stage=cancelled.stage.value,
            )

    def discard_preview(self, page_id: str) -> None:
        page_id = str(page_id or "")
        with self._preview_lock:
            directory = self._preview_directories.pop(page_id, "")
        if directory:
            _safe_remove_preview_directory(directory, self.artifact_root)
        previews = os.path.join(self.artifact_root, "previews")
        fragment = canonical_sha256({"page_id": page_id})[:16]
        if os.path.isdir(previews):
            for name in os.listdir(previews):
                path = os.path.join(previews, name)
                if name.startswith(f"page-{fragment}-") and os.path.isdir(path):
                    _safe_remove_preview_directory(path, self.artifact_root)

    def _validated_clean_base(
        self,
        snapshot: EffectivePageSnapshot,
        *,
        publish_pin: bool,
    ) -> _PinnedBase:
        cleaned = thaw_json(snapshot.cleaned_page_base)
        if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase revision is invalid.",
            )
        if str(cleaned.get("page_id") or "").strip() != snapshot.page_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase has the wrong page identity.",
            )
        descriptor_revision = str(cleaned.get("revision_id") or "").strip()
        if descriptor_revision and descriptor_revision != snapshot.cleaned_base_revision_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase has the wrong revision identity.",
            )
        expected_sha256 = _require_sha256(
            cleaned.get("content_sha256"),
            "CleanedPageBase content_sha256",
            failure_code=ManualCleanupFailureCode.MISSING_BASE,
        )
        asset = str(cleaned.get("asset") or "").strip()
        if not asset:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase has no asset.",
            )
        path = _resolve_asset_path(asset, self.project_directory)
        try:
            with open(path, "rb") as stream:
                payload = stream.read()
        except OSError as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase asset cannot be read.",
            ) from exc
        if _sha256(payload) != expected_sha256:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase hash does not match its catalog.",
            )
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format != "PNG":
                    raise ValueError("not PNG")
                width, height = int(opened.width), int(opened.height)
                if width <= 0 or height <= 0 or width * height > _MAX_CANVAS_PIXELS:
                    raise ValueError("unsafe dimensions")
                if opened.mode not in {"RGB", "RGBA"}:
                    raise ValueError("unsupported mode")
                image_mode = opened.mode
                image = opened.copy()
        except Exception as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.MISSING_BASE,
                "The selected CleanedPageBase must be a safe RGB/RGBA PNG.",
            ) from exc
        pinned_path = path
        if publish_pin:
            pinned_path = os.path.join(
                self.artifact_root,
                "inputs",
                f"cleaned-base-{expected_sha256}.png",
            )
            _write_once_atomic(pinned_path, payload, expected_sha256)
        return _PinnedBase(
            path=pinned_path,
            payload=payload,
            content_sha256=expected_sha256,
            canvas_size=(width, height),
            image_mode=image_mode,
            image=image,
        )

    def _validate_coverage_target(
        self,
        snapshot: EffectivePageSnapshot,
        target: UserParentCleanupCoverageTargetV1 | None,
        *,
        pinned: _PinnedBase,
        project: Mapping[str, Any] | None = None,
    ) -> None:
        if target is None:
            return
        if not isinstance(target, UserParentCleanupCoverageTargetV1):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.COVERAGE_TARGET_INVALID,
                "The user-parent cleanup coverage target is not typed.",
            )
        current_snapshot = snapshot
        current_project = project
        if current_project is None:
            try:
                (
                    current_project,
                    _,
                    current_snapshot,
                    _,
                    _,
                ) = self._coverage_current_read(
                    target.page_id,
                    expected_project_id=target.project_id,
                )
            except ManualCleanupFailure:
                raise
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.COVERAGE_TARGET_STALE,
                    "The current user-parent cleanup coverage state is unavailable.",
                ) from exc
        if _other_current_user_parent_cleanup_coverage_ids(
            current_snapshot,
            target.parent_id,
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.COVERAGE_CONFLICT,
                _COVERAGE_CONFLICT_MESSAGE,
            )
        try:
            from app.project_edits.ocr_revision_service import (
                resolve_original_page_asset_binding,
            )

            original_page = resolve_original_page_asset_binding(
                current_project,
                page_id=target.page_id,
                project_path=self.project_path,
            )
        except (OcrRevisionError, TypeError, ValueError) as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ORIGINAL_ASSET_MISMATCH,
                "The official original-page asset is unavailable or invalid.",
            ) from exc
        if (
            target.original_page_asset_id != original_page.asset_id
            or target.original_page_asset_reference
            != original_page.asset_reference
            or target.original_page_content_sha256
            != original_page.content_sha256
            or target.canvas_size != original_page.canvas_size
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ORIGINAL_ASSET_MISMATCH,
                "The cleanup coverage target does not bind the official "
                "original-page asset.",
            )
        try:
            expected = user_parent_cleanup_coverage_target_from_snapshot(
                current_snapshot,
                target.parent_id,
                original_page=original_page,
            )
        except (TypeError, ValueError) as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.COVERAGE_TARGET_STALE,
                f"The selected user-parent cleanup coverage target is stale: {exc}",
            ) from exc
        if expected != target:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.COVERAGE_TARGET_STALE,
                "The selected user-parent cleanup coverage dependencies changed.",
            )
        if (
            pinned.content_sha256 != target.input_cleaned_base_content_sha256
            or pinned.canvas_size != target.canvas_size
            or current_snapshot.cleaned_base_revision_id
            != target.input_cleaned_base_revision_id
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.COVERAGE_TARGET_STALE,
                "The selected input CleanedPageBase changed after target binding.",
            )

    def _validate_coverage_preview_heads(
        self,
        request: ManualCleanupRequest,
    ) -> None:
        if request.coverage_target is None:
            return
        expected_page_head = _require_sha256(
            request.expected_page_head_sha256,
            "expected_page_head_sha256",
            failure_code=ManualCleanupFailureCode.PREVIEW_STALE,
        )
        expected_global_head = _require_sha256(
            request.expected_global_head_sha256,
            "expected_global_head_sha256",
            failure_code=ManualCleanupFailureCode.PREVIEW_STALE,
        )
        if self.edit_store is None:
            (
                _,
                _,
                _,
                current_page_head,
                current_global_head,
            ) = self._coverage_current_read(
                request.snapshot.page_id,
                expected_project_id=request.snapshot.project_id,
            )
        else:
            if (
                self.edit_store.project_id != request.snapshot.project_id
                or os.path.normcase(os.path.abspath(self.edit_store.project_path))
                != os.path.normcase(self.project_path)
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "User-parent cleanup preview requires the current project edit store.",
                )
            current_page_head = self.edit_store.page_head(request.snapshot.page_id)
            current_global_head = self.edit_store.global_head()
        if (
            current_page_head != expected_page_head
            or current_global_head != expected_global_head
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "Project edit heads changed before cleanup preview publication.",
            )

    def _validated_rebase_review(
        self,
        selection: Any,
        selection_chain: tuple[Any, ...],
        artifact: Mapping[str, Any],
        stale_snapshot: EffectivePageSnapshot,
        current_snapshot: EffectivePageSnapshot,
    ) -> ManualCleanupRebaseReview:
        receipt = artifact.get("manual_cleanup_receipt")
        if not isinstance(receipt, Mapping):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision has no cleanup receipt.",
            )
        revision_id = str(artifact.get("revision_id") or "").strip()
        selection_edit_id = str(selection.edit_id or "").strip()
        if (
            str(artifact.get("manual_cleaned_base_revision_version") or "")
            != MANUAL_CLEANED_BASE_REVISION_VERSION
            or str(artifact.get("page_id") or "") != stale_snapshot.page_id
            or str(artifact.get("provenance") or "") != "user_manual_cleanup"
            or bool(artifact.get("current"))
            or not bool(artifact.get("valid"))
            or str(artifact.get("state") or "") != "manual_cleanup_committed"
            or str(receipt.get("manual_cleanup_receipt_version") or "")
            != MANUAL_CLEANUP_RECEIPT_VERSION
            or str(receipt.get("page_id") or "") != stale_snapshot.page_id
            or str(receipt.get("provenance") or "") != "user"
            or str(receipt.get("status") or "") != ManualCleanupStatus.COMMITTED.value
            or str(receipt.get("revision_id") or "") != revision_id
            or str(receipt.get("selection_edit_id") or "") != selection_edit_id
            or not bool(receipt.get("page_bounds_validated"))
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision or receipt identity is invalid.",
            )
        result_sha256 = _require_sha256(
            artifact.get("content_sha256"),
            "manual cleanup result hash",
            failure_code=ManualCleanupFailureCode.ARTIFACT_INVALID,
        )
        if (
            str(receipt.get("result_sha256") or "").lower() != result_sha256
            or str(receipt.get("result_asset") or "")
            != str(artifact.get("asset") or "")
            or str(receipt.get("input_base_revision_id") or "")
            != str(artifact.get("input_base_revision_id") or "")
            or str(receipt.get("input_base_sha256") or "").lower()
            != str(artifact.get("input_base_sha256") or "").lower()
            or cleaned_base_automatic_lineage(artifact) is None
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision lineage is invalid.",
            )
        canvas_value = receipt.get("canvas_size")
        if (
            not isinstance(canvas_value, (list, tuple))
            or len(canvas_value) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in canvas_value
            )
            or list(canvas_value) != list(artifact.get("canvas_size") or ())
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision canvas is invalid.",
            )
        canvas_size = (int(canvas_value[0]), int(canvas_value[1]))
        parameters_value = receipt.get("parameters")
        if not isinstance(parameters_value, Mapping) or set(parameters_value) != {
            "grow_px",
            "feather_px",
            "backend_id",
            "use_gpu",
        }:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision parameters are invalid.",
            )
        try:
            parameters = ManualCleanupParameters(**dict(parameters_value))
        except (TypeError, ValueError) as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The stale manual revision parameters are invalid.",
            ) from exc
        payloads: dict[str, bytes] = {}
        hashes: dict[str, str] = {}
        for name in ("erase", "protect", "effective"):
            hash_field = f"{name}_mask_sha256"
            asset_field = f"{name}_mask_asset"
            expected = _require_sha256(
                artifact.get(hash_field),
                f"{name} mask hash",
                failure_code=ManualCleanupFailureCode.ARTIFACT_INVALID,
            )
            if str(receipt.get(hash_field) or "").lower() != expected:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.ARTIFACT_INVALID,
                    f"The stale {name} mask receipt hash is invalid.",
                )
            payloads[name] = self._read_rebase_mask_asset(
                str(artifact.get(asset_field) or ""),
                expected,
                canvas_size,
                name,
            )
            hashes[name] = expected
        try:
            prepared = _prepare_masks(
                payloads["erase"],
                payloads["protect"],
                canvas_size=canvas_size,
                parameters=parameters,
            )
        except ManualCleanupFailure as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                f"The saved rebase masks are invalid: {exc.message}",
            ) from exc
        if (
            prepared.erase_png != payloads["erase"]
            or prepared.protect_png != payloads["protect"]
            or prepared.effective_png != payloads["effective"]
            or prepared.erase_sha256 != hashes["erase"]
            or prepared.protect_sha256 != hashes["protect"]
            or prepared.effective_sha256 != hashes["effective"]
            or int(receipt.get("effective_mask_pixels") or -1)
            != prepared.effective_pixels
            or int(receipt.get("protected_pixels") or -1) != prepared.protected_pixels
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "The saved rebase masks do not match their committed receipt.",
            )
        pinned = self._validated_clean_base(current_snapshot, publish_pin=False)
        if pinned.canvas_size != canvas_size:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.STALE_BASE,
                "The saved mask dimensions differ from the newly selected clean base.",
            )
        return ManualCleanupRebaseReview(
            page_id=stale_snapshot.page_id,
            stale_selection_edit_id=selection_edit_id,
            stale_selection_edit_ids=tuple(
                value.edit_id for value in selection_chain
            ),
            stale_revision_id=revision_id,
            stale_operation_id=str(receipt.get("operation_id") or "").strip(),
            stale_effective_fingerprint=stale_snapshot.effective_fingerprint,
            stale_input_base_revision_id=str(
                receipt.get("input_base_revision_id") or ""
            ).strip(),
            stale_input_base_sha256=str(
                receipt.get("input_base_sha256") or ""
            ).lower(),
            current_base_revision_id=current_snapshot.cleaned_base_revision_id,
            current_base_sha256=pinned.content_sha256,
            current_base_path=pinned.path,
            current_effective_fingerprint=current_snapshot.effective_fingerprint,
            source_image_path=self._source_image_path(current_snapshot),
            canvas_size=canvas_size,
            erase_mask_png=payloads["erase"],
            erase_mask_sha256=hashes["erase"],
            protect_mask_png=payloads["protect"],
            protect_mask_sha256=hashes["protect"],
            effective_mask_png=payloads["effective"],
            effective_mask_sha256=hashes["effective"],
            parameters=parameters,
        )

    def _read_rebase_mask_asset(
        self,
        asset: str,
        expected_sha256: str,
        canvas_size: tuple[int, int],
        name: str,
    ) -> bytes:
        path = _resolve_asset_path(asset, self.project_directory)
        mask_root = os.path.join(self.artifact_root, "masks")
        if not asset or not _path_is_within(path, mask_root):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                f"The stale {name} mask path is outside managed artifacts.",
            )
        try:
            with open(path, "rb") as stream:
                payload = stream.read(_MAX_MASK_BYTES + 1)
        except OSError as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                f"The stale {name} mask cannot be read.",
            ) from exc
        if len(payload) > _MAX_MASK_BYTES or _sha256(payload) != expected_sha256:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                f"The stale {name} mask bytes do not match their hash.",
            )
        _decode_mask_png(payload, canvas_size, f"saved {name} mask")
        return payload

    def _validate_rebase_binding(
        self,
        snapshot: EffectivePageSnapshot,
        review: ManualCleanupRebaseReview,
    ) -> None:
        if not isinstance(review, ManualCleanupRebaseReview):
            raise TypeError("review must be a ManualCleanupRebaseReview")
        if review.page_id != snapshot.page_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The rebase review belongs to another page.",
            )
        if any(
            issue.domain == EditDomain.CLEANUP.value
            for issue in snapshot.issues
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The rebuilt automatic/current CleanedPageBase is not compatible "
                "with the effective hierarchy.",
            )
        pinned = self._validated_clean_base(snapshot, publish_pin=False)
        if (
            snapshot.cleaned_base_revision_id != review.current_base_revision_id
            or snapshot.effective_fingerprint
            != review.current_effective_fingerprint
            or pinned.content_sha256 != review.current_base_sha256
            or pinned.canvas_size != review.canvas_size
            or os.path.normcase(os.path.abspath(pinned.path))
            != os.path.normcase(os.path.abspath(review.current_base_path))
            or snapshot.cleaned_base_revision_id == review.stale_revision_id
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The clean base changed after the rebase review was prepared.",
            )
        prepared = _prepare_masks(
            review.erase_mask_png,
            review.protect_mask_png,
            canvas_size=review.canvas_size,
            parameters=review.parameters,
        )
        if (
            prepared.erase_sha256 != review.erase_mask_sha256
            or prepared.protect_sha256 != review.protect_mask_sha256
            or prepared.effective_sha256 != review.effective_mask_sha256
            or prepared.effective_png != review.effective_mask_png
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.PREVIEW_STALE,
                "The saved masks changed after the rebase review was prepared.",
            )

    def _candidate(self, candidate_id: str) -> Any | None:
        candidates = self._backend_inventory()
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(
            candidates, Sequence
        ):
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.BACKEND_UNAVAILABLE,
                "Cleanup backend inventory is invalid.",
            )
        matches = [
            candidate
            for candidate in candidates
            if str(getattr(candidate, "candidate_id", "") or "") == candidate_id
        ]
        if len(matches) > 1:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.BACKEND_UNAVAILABLE,
                "Cleanup backend inventory contains duplicate identities.",
            )
        return matches[0] if matches else None

    def _source_image_path(self, snapshot: EffectivePageSnapshot) -> str:
        """Resolve the immutable source image for comparison only.

        This path is never accepted as an inpaint substrate or fallback.
        """

        cleaned = thaw_json(snapshot.cleaned_page_base)
        nested = (
            cleaned.get("cleaned_page_base")
            if isinstance(cleaned, Mapping)
            else None
        )
        raw = str(
            nested.get("source_image_path")
            if isinstance(nested, Mapping)
            else ""
        ).strip()
        if not raw:
            return ""
        path = _resolve_asset_path(raw, self.project_directory)
        return path if os.path.isfile(path) else ""

    def _publish_preview(
        self,
        *,
        page_id: str,
        operation_id: str,
        result_png: bytes,
        masks: _PreparedMasks,
    ) -> dict[str, str]:
        self.discard_preview(page_id)
        root = os.path.join(self.artifact_root, "previews")
        os.makedirs(root, exist_ok=True)
        fragment = canonical_sha256({"page_id": page_id})[:16]
        directory = tempfile.mkdtemp(
            prefix=f"page-{fragment}-{operation_id[:12]}-",
            dir=root,
        )
        paths = {
            "result": os.path.join(directory, "result.png"),
            "erase": os.path.join(directory, "erase-mask.png"),
            "protect": os.path.join(directory, "protect-mask.png"),
            "effective": os.path.join(directory, "effective-mask.png"),
        }
        try:
            for key, payload, digest in (
                ("result", result_png, _sha256(result_png)),
                ("erase", masks.erase_png, masks.erase_sha256),
                ("protect", masks.protect_png, masks.protect_sha256),
                ("effective", masks.effective_png, masks.effective_sha256),
            ):
                _write_once_atomic(paths[key], payload, digest)
        except Exception:
            _safe_remove_preview_directory(directory, self.artifact_root)
            raise
        with self._preview_lock:
            self._preview_directories[page_id] = directory
        return paths

    def _validated_preview_payloads(
        self,
        lease: ManualCleanupPreviewLease,
    ) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for key, path, digest in (
            ("result", lease.result_path, lease.result_sha256),
            ("erase", lease.erase_mask_path, lease.erase_mask_sha256),
            ("protect", lease.protect_mask_path, lease.protect_mask_sha256),
            ("effective", lease.effective_mask_path, lease.effective_mask_sha256),
        ):
            if not _path_is_within(path, os.path.join(self.artifact_root, "previews")):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.PREVIEW_STALE,
                    "Manual cleanup preview lease is outside managed storage.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            try:
                with open(path, "rb") as stream:
                    payload = stream.read()
            except OSError as exc:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.PREVIEW_STALE,
                    "Manual cleanup preview artifact is unavailable.",
                    stage=ManualCleanupStage.COMMITTING,
                ) from exc
            if _sha256(payload) != digest:
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.PREVIEW_STALE,
                    "Manual cleanup preview artifact changed before commit.",
                    stage=ManualCleanupStage.COMMITTING,
                )
            result[key] = payload
        return result

    def _promote_preview(
        self,
        lease: ManualCleanupPreviewLease,
        payloads: Mapping[str, bytes],
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        targets = {
            "result": os.path.join(
                self.artifact_root, "cleaned", f"{lease.result_sha256}.png"
            ),
            "erase": os.path.join(
                self.artifact_root, "masks", f"{lease.erase_mask_sha256}.png"
            ),
            "protect": os.path.join(
                self.artifact_root, "masks", f"{lease.protect_mask_sha256}.png"
            ),
            "effective": os.path.join(
                self.artifact_root, "masks", f"{lease.effective_mask_sha256}.png"
            ),
        }
        created: list[str] = []
        try:
            for key, target in targets.items():
                existed = os.path.isfile(target)
                _write_once_atomic(target, payloads[key], _sha256(payloads[key]))
                if not existed:
                    created.append(target)
        except Exception:
            _remove_created_files(created)
            raise
        return targets, tuple(created)

    def _current_snapshot(
        self,
        page_id: str,
    ) -> tuple[Mapping[str, Any], Any, EffectivePageSnapshot]:
        from app.io.project import load_project_for_editing

        if self.edit_store is None:  # pragma: no cover - checked by caller
            raise RuntimeError("edit store is unavailable")
        project = load_project_for_editing(self.project_path)
        project = self.edit_store.materialize_project(project)
        ledger = self.edit_store.load_ledger()
        snapshot = project_effective_page(project, ledger, page_id=page_id)
        return project, ledger, snapshot

    def _coverage_current_read(
        self,
        page_id: str,
        *,
        expected_project_id: str,
    ) -> tuple[
        Mapping[str, Any],
        ProjectEditLedger,
        EffectivePageSnapshot,
        str,
        str,
    ]:
        """Read one exact coverage view without retaining a preview store."""

        page_id = str(page_id or "").strip()
        expected_project_id = str(expected_project_id or "").strip()
        if not page_id or not expected_project_id:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.STORE_UNAVAILABLE,
                "User-parent cleanup preview requires an exact project edit store.",
            )
        if self.edit_store is not None:
            if (
                self.edit_store.project_id != expected_project_id
                or os.path.normcase(os.path.abspath(self.edit_store.project_path))
                != os.path.normcase(self.project_path)
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "User-parent cleanup preview requires an exact project edit store.",
                )
            project, ledger, snapshot = self._current_snapshot(page_id)
            return (
                project,
                ledger,
                snapshot,
                self.edit_store.page_head(page_id),
                self.edit_store.global_head(),
            )

        from app.io.project import load_project_for_editing

        transient_store: ProjectEditStore | None = None
        try:
            project = load_project_for_editing(self.project_path)
            project_id = project_id_for(project)
            metadata = inspect_project_edit_store(self.project_path)
            if (
                metadata is None
                or project_id != expected_project_id
                or str(metadata.get("project_id") or "") != project_id
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "User-parent cleanup preview requires the exact existing "
                    "project edit store.",
                )
            transient_store = ProjectEditStore(
                project_path=self.project_path,
                project_id=project_id,
                project_origin_sha256=project_origin_fingerprint(project),
                automated_state_sha256=automated_state_fingerprint(project),
                base_ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
                base_artifact_revisions=project["artifact_revisions"],
            )
            read_snapshot = transient_store.materialize_project_snapshot(
                project,
                page_id=page_id,
            )
            snapshot = project_effective_page(
                read_snapshot.project,
                read_snapshot.ledger,
                page_id=page_id,
            )
            if (
                snapshot.project_id != expected_project_id
                or snapshot.page_id != page_id
            ):
                raise ManualCleanupFailure(
                    ManualCleanupFailureCode.STORE_UNAVAILABLE,
                    "The materialized cleanup coverage view changed project or page.",
                )
            return (
                read_snapshot.project,
                read_snapshot.ledger,
                snapshot,
                read_snapshot.page_head_sha256,
                read_snapshot.global_head_sha256,
            )
        except ManualCleanupFailure:
            raise
        except Exception as exc:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.STORE_UNAVAILABLE,
                "The exact current project edit store is unavailable.",
            ) from exc
        finally:
            if transient_store is not None:
                transient_store.close()

    @staticmethod
    def _check_cancel(
        cancellation: CancellationProbe | None,
        stage: ManualCleanupStage,
    ) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            raise _Cancelled(stage)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        page_id: str,
        stage: ManualCleanupStage,
        completed: int,
        message: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(
                ManualCleanupProgress(
                    page_id=page_id,
                    stage=stage,
                    completed_steps=int(completed),
                    total_steps=4,
                    message=message,
                )
            )
        except Exception:
            return


def manual_cleanup_artifact_root(project_path: str) -> str:
    raw = str(project_path or "").strip()
    if not raw:
        raise ValueError("project_path is required")
    absolute = os.path.abspath(raw)
    parent = os.path.dirname(absolute) or os.getcwd()
    return os.path.join(
        parent,
        f".{os.path.basename(absolute)}.gui-cleanup-artifacts",
    )


def _fixed_backend_inventory() -> Sequence[Any]:
    from app.pipeline.cleanup_backend_runner import inventory_local_cleanup_backends

    return tuple(inventory_local_cleanup_backends())


def _fixed_backend_runner(**kwargs: Any) -> Any:
    from app.pipeline.cleanup_backend_runner import run_cleanup_backend_candidate

    return run_cleanup_backend_candidate(**kwargs)


def _backend_version_fingerprint(candidate: Any, execution: Any) -> str:
    """Return an explicit backend version or a truthful local identity hash."""

    explicit = str(
        getattr(execution, "backend_version", "")
        or getattr(candidate, "version", "")
        or ""
    ).strip()
    if explicit:
        return explicit

    def path_identity(value: Any) -> Mapping[str, Any]:
        raw = str(value or "").strip()
        if not raw:
            return {"path": "", "exists": False}
        absolute = os.path.abspath(raw)
        result: dict[str, Any] = {
            "path": os.path.normcase(absolute),
            "exists": os.path.exists(absolute),
        }
        try:
            stat = os.stat(absolute)
        except OSError:
            return result
        result.update(
            {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_file": os.path.isfile(absolute),
            }
        )
        return result

    identity = {
        "schema_version": "manual_cleanup_backend_identity_v1",
        "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
        "backend_name": str(getattr(execution, "backend_name", "") or ""),
        "backend_family": str(
            getattr(execution, "backend_family", "")
            or getattr(candidate, "backend_family", "")
            or ""
        ),
        "model": path_identity(
            getattr(execution, "model_path", "")
            or getattr(candidate, "model_path", "")
        ),
        "adapter": path_identity(
            getattr(execution, "adapter_path", "")
            or getattr(candidate, "adapter_path", "")
        ),
    }
    return f"backend-identity-v1:{canonical_sha256(identity)}"


def _prepare_masks(
    erase_mask_png: bytes,
    protect_mask_png: bytes | None,
    *,
    canvas_size: tuple[int, int],
    parameters: ManualCleanupParameters,
) -> _PreparedMasks:
    erase = _decode_mask_png(erase_mask_png, canvas_size, "erase mask")
    protect = (
        _decode_mask_png(protect_mask_png, canvas_size, "protect mask")
        if protect_mask_png is not None
        else Image.new("L", canvas_size, 0)
    )
    # Canonical binary user masks keep mask semantics independent of PNG mode
    # and encoder metadata. Grow then feather is the declared deterministic
    # order; protect is subtracted last so every protected pixel remains exact.
    erase = erase.point(lambda value: 255 if value else 0, mode="L")
    protect = protect.point(lambda value: 255 if value else 0, mode="L")
    effective = erase
    if parameters.grow_px:
        effective = effective.filter(
            ImageFilter.MaxFilter(parameters.grow_px * 2 + 1)
        )
    if parameters.feather_px:
        effective = effective.filter(
            ImageFilter.GaussianBlur(radius=float(parameters.feather_px))
        )
    effective = ImageChops.subtract(effective, protect)
    effective_pixels = int(sum(effective.histogram()[1:]))
    protected_pixels = int(sum(protect.histogram()[1:]))
    if effective_pixels <= 0:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.INVALID_MASK,
            "The effective erase mask is empty after protection.",
            stage=ManualCleanupStage.PREPARING_MASKS,
        )
    erase_png = _png_bytes(erase)
    protect_png = _png_bytes(protect)
    effective_png = _png_bytes(effective)
    return _PreparedMasks(
        erase=erase,
        protect=protect,
        effective=effective,
        erase_png=erase_png,
        protect_png=protect_png,
        effective_png=effective_png,
        erase_sha256=_sha256(erase_png),
        protect_sha256=_sha256(protect_png),
        effective_sha256=_sha256(effective_png),
        effective_pixels=effective_pixels,
        protected_pixels=protected_pixels,
    )


def _decode_mask_png(
    payload: bytes | None,
    canvas_size: tuple[int, int],
    field_name: str,
) -> Image.Image:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_MASK_BYTES:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.INVALID_MASK,
            f"{field_name} must be a bounded PNG byte payload.",
            stage=ManualCleanupStage.PREPARING_MASKS,
        )
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != "PNG" or opened.mode not in {"1", "L"}:
                raise ValueError("mask must be a one-channel PNG")
            if opened.size != canvas_size:
                raise ValueError("mask dimensions differ from the page")
            return opened.convert("L").copy()
    except ManualCleanupFailure:
        raise
    except Exception as exc:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.INVALID_MASK,
            f"{field_name} must be a page-sized one-channel PNG.",
            stage=ManualCleanupStage.PREPARING_MASKS,
        ) from exc


def _protected_composite(
    base: Image.Image,
    backend: Image.Image,
    effective_mask: Image.Image,
    protect_mask: Image.Image,
    *,
    output_mode: str,
) -> Image.Image:
    base_rgba = base.convert("RGBA")
    backend_rgba = backend.convert("RGBA")
    blended_rgb = Image.composite(
        backend_rgba.convert("RGB"),
        base_rgba.convert("RGB"),
        effective_mask,
    )
    # Backend alpha is not an authority. Preserve the immutable substrate's
    # alpha channel everywhere, then restore protected RGBA pixels explicitly.
    result = Image.merge("RGBA", (*blended_rgb.split(), base_rgba.getchannel("A")))
    result.paste(base_rgba, (0, 0), protect_mask)
    return result if output_mode == "RGBA" else result.convert("RGB")


def _receipt_from_lease(
    lease: ManualCleanupPreviewLease,
    *,
    status: ManualCleanupStatus,
    started_at: str,
    completed_at: str,
    output_path: str = "",
    revision_id: str = "",
    selection_edit_id: str = "",
    cancellation_stage: str = "",
    preview_lease: ManualCleanupPreviewLease | None = None,
) -> ManualCleanupReceipt:
    return ManualCleanupReceipt(
        operation_id=lease.operation_id,
        page_id=lease.page_id,
        provenance="user",
        status=status,
        input_base_revision_id=lease.input_base_revision_id,
        input_base_sha256=lease.input_base_sha256,
        erase_mask_sha256=lease.erase_mask_sha256,
        protect_mask_sha256=lease.protect_mask_sha256,
        effective_mask_sha256=lease.effective_mask_sha256,
        result_sha256=lease.result_sha256,
        canvas_size=lease.canvas_size,
        parameters=lease.parameters,
        backend_id=lease.backend_id,
        backend_name=lease.backend_name,
        backend_family=lease.backend_family,
        backend_model_path=lease.backend_model_path,
        backend_adapter_path=lease.backend_adapter_path,
        backend_version=lease.backend_version,
        backend_runtime_ms=lease.backend_runtime_ms,
        effective_mask_pixels=lease.effective_mask_pixels,
        protected_pixels=lease.protected_pixels,
        page_bounds_validated=True,
        started_at=started_at,
        completed_at=completed_at,
        reviewed_stale_selection_edit_ids=(
            lease.reviewed_stale_selection_edit_ids
        ),
        reviewed_stale_effective_fingerprint=(
            lease.reviewed_stale_effective_fingerprint
        ),
        output_path=output_path,
        revision_id=revision_id,
        selection_edit_id=selection_edit_id,
        cancellation_stage=cancellation_stage,
        preview_lease=preview_lease,
        coverage_target=lease.coverage_target,
    )


def _cancelled_receipt(
    operation_id: str,
    request: ManualCleanupRequest,
    started_at: str,
    stage: ManualCleanupStage,
) -> ManualCleanupReceipt:
    return ManualCleanupReceipt(
        operation_id=operation_id,
        page_id=request.snapshot.page_id,
        provenance="user",
        status=ManualCleanupStatus.CANCELLED,
        input_base_revision_id=request.snapshot.cleaned_base_revision_id,
        input_base_sha256="",
        erase_mask_sha256="",
        protect_mask_sha256="",
        effective_mask_sha256="",
        result_sha256="",
        canvas_size=(),
        parameters=request.parameters,
        backend_id=request.parameters.backend_id,
        backend_name="",
        backend_family="",
        backend_model_path="",
        backend_adapter_path="",
        backend_version=MANUAL_CLEANUP_SERVICE_VERSION,
        backend_runtime_ms=0.0,
        effective_mask_pixels=0,
        protected_pixels=0,
        page_bounds_validated=False,
        started_at=started_at,
        completed_at=_utc_now(),
        reviewed_stale_selection_edit_ids=(
            request.reviewed_stale_selection_edit_ids
        ),
        reviewed_stale_effective_fingerprint=(
            request.reviewed_stale_effective_fingerprint
        ),
        cancellation_stage=stage.value,
        coverage_target=request.coverage_target,
    )


def _failure_code_for_preflight(
    code: ManualCleanupAvailabilityCode,
) -> ManualCleanupFailureCode:
    return {
        ManualCleanupAvailabilityCode.MISSING_BASE: ManualCleanupFailureCode.MISSING_BASE,
        ManualCleanupAvailabilityCode.STALE_BASE: ManualCleanupFailureCode.STALE_BASE,
        ManualCleanupAvailabilityCode.INVALID_MASK: ManualCleanupFailureCode.INVALID_MASK,
        ManualCleanupAvailabilityCode.INVALID_COVERAGE_TARGET: (
            ManualCleanupFailureCode.COVERAGE_TARGET_INVALID
        ),
        ManualCleanupAvailabilityCode.STALE_COVERAGE_TARGET: (
            ManualCleanupFailureCode.COVERAGE_TARGET_STALE
        ),
        ManualCleanupAvailabilityCode.COVERAGE_CONFLICT: (
            ManualCleanupFailureCode.COVERAGE_CONFLICT
        ),
        ManualCleanupAvailabilityCode.ORIGINAL_ASSET_MISMATCH: (
            ManualCleanupFailureCode.ORIGINAL_ASSET_MISMATCH
        ),
        ManualCleanupAvailabilityCode.BACKEND_UNAVAILABLE: ManualCleanupFailureCode.BACKEND_UNAVAILABLE,
    }.get(code, ManualCleanupFailureCode.INVALID_REQUEST)


def _availability_code_for_failure(
    code: ManualCleanupFailureCode,
) -> ManualCleanupAvailabilityCode:
    return {
        ManualCleanupFailureCode.MISSING_BASE: (
            ManualCleanupAvailabilityCode.MISSING_BASE
        ),
        ManualCleanupFailureCode.STALE_BASE: (
            ManualCleanupAvailabilityCode.STALE_BASE
        ),
        ManualCleanupFailureCode.COVERAGE_TARGET_INVALID: (
            ManualCleanupAvailabilityCode.INVALID_COVERAGE_TARGET
        ),
        ManualCleanupFailureCode.COVERAGE_TARGET_STALE: (
            ManualCleanupAvailabilityCode.STALE_COVERAGE_TARGET
        ),
        ManualCleanupFailureCode.COVERAGE_CONFLICT: (
            ManualCleanupAvailabilityCode.COVERAGE_CONFLICT
        ),
        ManualCleanupFailureCode.ORIGINAL_ASSET_MISMATCH: (
            ManualCleanupAvailabilityCode.ORIGINAL_ASSET_MISMATCH
        ),
    }.get(code, ManualCleanupAvailabilityCode.BLOCKED)


def _active_cleanup_selection_chain(ledger: Any, page_id: str) -> tuple[Any, ...]:
    """Return one active cleanup supersession chain from root through head."""

    matches = [
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.CLEANUP
        and edit.operation == "select_revision"
        and edit.target.kind is EditTargetKind.PAGE
    ]
    match_ids = {edit.edit_id for edit in matches}
    superseded = {
        edit.supersedes_edit_id
        for edit in matches
        if edit.supersedes_edit_id in match_ids
    }
    heads = [edit for edit in matches if edit.edit_id not in superseded]
    if len(heads) > 1:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.COMMIT_STALE,
            "The active cleanup selection is ambiguous.",
            stage=ManualCleanupStage.COMMITTING,
        )
    if not heads:
        return ()
    by_id = {edit.edit_id: edit for edit in matches}
    reversed_chain: list[Any] = []
    current = heads[0]
    while current is not None:
        reversed_chain.append(current)
        parent_id = current.supersedes_edit_id
        current = by_id.get(parent_id) if parent_id else None
    chain = tuple(reversed(reversed_chain))
    if len(chain) != len(matches):
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.COMMIT_STALE,
            "The active cleanup selection chain is disconnected.",
            stage=ManualCleanupStage.COMMITTING,
        )
    return chain


def _active_cleanup_selection_head(ledger: Any, page_id: str) -> Any | None:
    chain = _active_cleanup_selection_chain(ledger, page_id)
    return chain[-1] if chain else None


def _active_cleanup_selection(ledger: Any, snapshot: EffectivePageSnapshot) -> Any | None:
    return _active_cleanup_selection_head(ledger, snapshot.page_id)


def _manual_cleanup_revision(
    project: Mapping[str, Any],
    page_id: str,
    revision_id: str,
) -> Mapping[str, Any]:
    catalogs = project.get("artifact_revisions")
    values = catalogs.get("cleaned_page_bases") if isinstance(catalogs, Mapping) else None
    matches = [
        value
        for value in values or ()
        if isinstance(value, Mapping)
        and str(value.get("revision_id") or "").strip() == revision_id
        and str(value.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.ARTIFACT_INVALID,
            "The stale manual cleanup revision is unavailable or ambiguous.",
        )
    return matches[0]


def _without_cleanup_issues(snapshot: EffectivePageSnapshot) -> EffectivePageSnapshot:
    return replace(
        snapshot,
        issues=tuple(
            issue
            for issue in snapshot.issues
            if issue.domain != EditDomain.CLEANUP.value
        ),
    )


def _cleanup_rebase_base_snapshot(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    stale_snapshot: EffectivePageSnapshot,
) -> EffectivePageSnapshot:
    """Project the effective page with cleanup selections removed.

    Structural, text, glossary, review, and renderer edits remain active.  The
    central projector therefore decides whether the current automatic clean
    base is actually compatible with that effective hierarchy; this helper
    never strips or bypasses a resulting cleanup issue.
    """

    cleanup_edit_ids = {
        edit.edit_id
        for edit in ledger.edits
        if not edit.is_control
        and edit.domain is EditDomain.CLEANUP
        and edit.operation == "select_revision"
        and edit.target.kind is EditTargetKind.PAGE
        and edit.page_id == stale_snapshot.page_id
    }
    retained = tuple(
        edit
        for edit in ledger.edits
        if edit.edit_id not in cleanup_edit_ids
        and not (
            edit.is_control
            and edit.target.edit_id in cleanup_edit_ids
        )
    )
    cleanup_free_ledger = ProjectEditLedger(
        retained,
        project_id=ledger.project_id,
        schema_version=ledger.schema_version,
    )
    current = project_effective_page(
        project,
        cleanup_free_ledger,
        page_id=stale_snapshot.page_id,
    )
    if (
        current.project_id != stale_snapshot.project_id
        or current.page_id != stale_snapshot.page_id
        or current.automatic_fingerprint != stale_snapshot.automatic_fingerprint
    ):
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.PREVIEW_STALE,
            "The automatic page changed while preparing rebase review.",
        )
    return current


def _has_cleanup_rebase_issue(snapshot: EffectivePageSnapshot) -> bool:
    """Return whether projection rejected cleanup due to a changed base.

    A forward cleanup result normally changes the automatic cleanup field
    fingerprint, so projection falls back to the automatic base with a
    ``STALE_EDIT_BASE`` issue.  Hierarchy changes can instead retain the
    selection long enough to produce ``STALE_DEPENDENCY``.  Both are explicit
    rebase-review cases; orphaned or conflicting selections are not.
    """

    return any(
        issue.domain == EditDomain.CLEANUP.value
        and issue.kind
        in {
            ProjectionIssueKind.STALE_EDIT_BASE,
            ProjectionIssueKind.STALE_DEPENDENCY,
        }
        for issue in snapshot.issues
    )


def _find_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.STALE_BASE,
            "The preview page is no longer uniquely available.",
            stage=ManualCleanupStage.COMMITTING,
        )
    return matches[0]


def _operation_id(value: str) -> str:
    candidate = str(value or "").strip() or uuid.uuid4().hex
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in candidate
    ):
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.INVALID_REQUEST,
            "operation_id must be path-safe.",
        )
    return candidate


def _require_sha256(
    value: Any,
    field_name: str,
    *,
    failure_code: ManualCleanupFailureCode,
) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ManualCleanupFailure(
            failure_code,
            f"{field_name} must be a SHA-256 digest.",
        )
    return text


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


def _write_once_atomic(path: str, payload: bytes, expected_sha256: str) -> None:
    if _sha256(payload) != expected_sha256:
        raise ManualCleanupFailure(
            ManualCleanupFailureCode.ARTIFACT_INVALID,
            "Artifact payload hash does not match its identity.",
        )
    parent = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    if os.path.isfile(path):
        with open(path, "rb") as stream:
            existing = stream.read()
        if _sha256(existing) != expected_sha256:
            raise ManualCleanupFailure(
                ManualCleanupFailureCode.ARTIFACT_INVALID,
                "A content-addressed artifact has conflicting bytes.",
            )
        return
    handle, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            handle = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if handle >= 0:
            os.close(handle)
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_asset_path(asset: str, project_directory: str) -> str:
    return os.path.abspath(
        asset if os.path.isabs(asset) else os.path.join(project_directory, asset)
    )


def _portable_asset_path(path: str, project_directory: str) -> str:
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, project_directory)
    except ValueError:
        return absolute.replace("\\", "/")
    return relative.replace("\\", "/")


def _path_is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            (os.path.abspath(path), os.path.abspath(root))
        ) == os.path.abspath(root)
    except ValueError:
        return False


def _safe_remove_preview_directory(path: str, artifact_root: str) -> None:
    preview_root = os.path.join(os.path.abspath(artifact_root), "previews")
    if not _path_is_within(path, preview_root) or os.path.abspath(path) == preview_root:
        return
    shutil.rmtree(path, ignore_errors=True)


def _remove_created_files(paths: Sequence[str]) -> None:
    for path in paths:
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
