# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent editor commands.

This module owns only GUI draft and command-presentation state.  Durable edit
identity, validation, projection, invalidation, and persistence remain owned by
``app.project_edits`` and ``ProjectEditStore``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    AddUserParentCommandErrorCode,
    AddUserParentCommandReceipt,
    AddUserParentOperation,
    ParentGeometryCommandErrorCode,
    ParentGeometryCommandReceipt,
    ParentGeometryOperation,
    ParentMembershipCommandErrorCode,
    ParentMembershipCommandReceipt,
    ParentMembershipOperation,
    MergePipelineParentsCommandErrorCode,
    MergePipelineParentsCommandReceipt,
    MergePipelineParentsOperation,
    ReadingOrderCommandErrorCode,
    ReadingOrderCommandReceipt,
    ReadingOrderOperation,
    RenderLayoutLineHeightCommandErrorCode,
    RenderLayoutLineHeightCommandReceipt,
    RenderLayoutLineHeightOperation,
    RenderLayoutRotationCommandErrorCode,
    RenderLayoutRotationCommandReceipt,
    RenderLayoutRotationOperation,
    RenderStyleFillColorCommandErrorCode,
    RenderStyleFillColorCommandReceipt,
    RenderStyleFillColorOperation,
    RenderLayoutWritingModeCommandErrorCode,
    RenderLayoutWritingModeCommandReceipt,
    RenderLayoutWritingModeOperation,
    SourceTextCommandErrorCode,
    SourceTextCommandReceipt,
    SourceTextOperation,
    SplitUserParentCommandErrorCode,
    SplitUserParentCommandReceipt,
    SplitUserParentOperation,
    SplitUserParentOrientation,
    TargetTextCommandErrorCode,
    TargetTextCommandReceipt,
    TargetTextOperation,
    create_user_parent_identity,
)
from app.project_edits.contracts import (
    CANONICAL_WRITING_MODES,
    EditDomain,
    ParentSourceEvidenceMappingV1,
    TargetTextRevisionBaseV1,
    canonical_render_line_height,
    canonical_render_rotation,
    canonical_render_fill_color,
    thaw_json,
    validate_user_parent_identity_pair,
)
from app.project_edits.projection import (
    TargetFreshness,
    automatic_ordered_parent_ids_for_page,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


_STALE_COMMAND_CODES = frozenset(
    {
        TargetTextCommandErrorCode.STALE_EFFECTIVE_PAGE,
        TargetTextCommandErrorCode.STALE_PAGE_HEAD,
        TargetTextCommandErrorCode.STALE_GLOBAL_HEAD,
        TargetTextCommandErrorCode.REVISION_ID_MISMATCH,
        TargetTextCommandErrorCode.REVISION_SELECTION_MISMATCH,
        TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH,
        TargetTextCommandErrorCode.REVISION_SOURCE_MISMATCH,
        TargetTextCommandErrorCode.REVISION_HIERARCHY_MISMATCH,
        TargetTextCommandErrorCode.MAPPED_BASE_MISMATCH,
        TargetTextCommandErrorCode.PARENT_LINEAGE_MISMATCH,
    }
)
_MEMBERSHIP_STALE_COMMAND_CODES = frozenset(
    {
        ParentMembershipCommandErrorCode.STALE_EFFECTIVE_PAGE,
        ParentMembershipCommandErrorCode.STALE_PAGE_HEAD,
        ParentMembershipCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_GEOMETRY_STALE_COMMAND_CODES = frozenset(
    {
        ParentGeometryCommandErrorCode.STALE_EFFECTIVE_PAGE,
        ParentGeometryCommandErrorCode.STALE_PAGE_HEAD,
        ParentGeometryCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_SOURCE_STALE_COMMAND_CODES = frozenset(
    {
        SourceTextCommandErrorCode.STALE_EFFECTIVE_PAGE,
        SourceTextCommandErrorCode.STALE_PAGE_HEAD,
        SourceTextCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_WRITING_MODE_STALE_COMMAND_CODES = frozenset(
    {
        RenderLayoutWritingModeCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderLayoutWritingModeCommandErrorCode.STALE_PAGE_HEAD,
        RenderLayoutWritingModeCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_LINE_HEIGHT_STALE_COMMAND_CODES = frozenset(
    {
        RenderLayoutLineHeightCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderLayoutLineHeightCommandErrorCode.STALE_PAGE_HEAD,
        RenderLayoutLineHeightCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_ROTATION_STALE_COMMAND_CODES = frozenset(
    {
        RenderLayoutRotationCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderLayoutRotationCommandErrorCode.STALE_PAGE_HEAD,
        RenderLayoutRotationCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_FILL_COLOR_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleFillColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleFillColorCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleFillColorCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_READING_ORDER_STALE_COMMAND_CODES = frozenset(
    {
        ReadingOrderCommandErrorCode.STALE_EFFECTIVE_PAGE,
        ReadingOrderCommandErrorCode.STALE_PAGE_HEAD,
        ReadingOrderCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_ADD_USER_PARENT_STALE_COMMAND_CODES = frozenset(
    {
        AddUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
        AddUserParentCommandErrorCode.STALE_PAGE_HEAD,
        AddUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_SPLIT_USER_PARENT_STALE_COMMAND_CODES = frozenset(
    {
        SplitUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
        SplitUserParentCommandErrorCode.STALE_PAGE_HEAD,
        SplitUserParentCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_MERGE_PIPELINE_PARENTS_STALE_COMMAND_CODES = frozenset(
    {
        MergePipelineParentsCommandErrorCode.STALE_EFFECTIVE_PAGE,
        MergePipelineParentsCommandErrorCode.STALE_PAGE_HEAD,
        MergePipelineParentsCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _required_identity(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty exact identifier")
    return value


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a SHA-256 hex digest")
    candidate = value.lower()
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return candidate


def _exact_bbox_components(
    value: Any,
    field_name: str,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in value
        )
    ):
        raise ValueError(f"{field_name} must contain four exact integers")
    return tuple(int(component) for component in value)


def _partial_bbox_components(
    value: Any,
    field_name: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            component is not None
            and (isinstance(component, bool) or not isinstance(component, int))
            for component in value
        )
    ):
        raise ValueError(
            f"{field_name} must contain four exact integers or empty components"
        )
    return tuple(
        None if component is None else int(component) for component in value
    )


def _complete_partial_bbox(
    value: tuple[int | None, int | None, int | None, int | None] | None,
) -> tuple[int, int, int, int] | None:
    if value is None or any(component is None for component in value):
        return None
    return tuple(int(component) for component in value)


def _exact_canvas_size(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(component, bool)
            or not isinstance(component, int)
            or component <= 0
            for component in value
        )
    ):
        raise ValueError("canvas_size must contain two positive exact integers")
    canvas_size = tuple(int(component) for component in value)
    if canvas_size[0] * canvas_size[1] > 50_000_000:
        raise ValueError("canvas_size exceeds the geometry safety limit")
    return canvas_size


def _bbox_validation_problem(
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> str | None:
    x, y, width, height = bbox
    if x < 0 or y < 0:
        return "Geometry X and Y must be zero or greater."
    if width <= 0 or height <= 0:
        return "Geometry width and height must be greater than zero."
    if x + width > canvas_size[0] or y + height > canvas_size[1]:
        return "Geometry must remain fully inside the page canvas."
    return None


def _validated_page_bbox(
    value: Any,
    field_name: str,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    bbox = _exact_bbox_components(value, field_name)
    problem = _bbox_validation_problem(bbox, canvas_size)
    if problem is not None:
        raise ValueError(f"{field_name}: {problem}")
    return bbox


class TargetTextEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class TargetTextWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class TargetTextWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SNAPSHOT_STALE = "snapshot_stale"
    TARGET_SLOT_CONFLICT = "target_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class TargetTextSelection:
    """Exact selected-parent state from one immutable UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_target_text: str | None
    effective_target_text: str
    target_authority: str
    effective_page_fingerprint: str
    selected_model_target_text: str | None = None
    revision_base: TargetTextRevisionBaseV1 | None = None
    mapped_pipeline_target_text: str | None = None
    source_evidence_base: ParentSourceEvidenceMappingV1 | None = None
    target_freshness: TargetFreshness = TargetFreshness.CURRENT

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self, "page_id", _required_identity(self.page_id, "page_id")
        )
        object.__setattr__(
            self, "parent_id", _required_identity(self.parent_id, "parent_id")
        )
        if (
            self.automatic_target_text is not None
            and not isinstance(self.automatic_target_text, str)
        ):
            raise TypeError("automatic_target_text must be a string or None")
        if (
            self.selected_model_target_text is not None
            and not isinstance(self.selected_model_target_text, str)
        ):
            raise TypeError("selected_model_target_text must be a string or None")
        if (
            self.mapped_pipeline_target_text is not None
            and not isinstance(self.mapped_pipeline_target_text, str)
        ):
            raise TypeError("mapped_pipeline_target_text must be a string or None")
        if not isinstance(self.effective_target_text, str):
            raise TypeError("effective_target_text must be a string")
        authority = str(self.target_authority or "").strip()
        if authority not in {
            "automatic",
            "translation_revision",
            "mapped_automatic",
            "user",
        }:
            raise ValueError(
                "target_authority must be automatic, translation_revision, "
                "mapped_automatic, or user"
            )
        object.__setattr__(self, "target_authority", authority)
        revision_base = self.revision_base
        if revision_base is not None and not isinstance(
            revision_base,
            TargetTextRevisionBaseV1,
        ):
            raise TypeError("revision_base must be a TargetTextRevisionBaseV1")
        source_evidence_base = self.source_evidence_base
        if source_evidence_base is not None and not isinstance(
            source_evidence_base,
            ParentSourceEvidenceMappingV1,
        ):
            raise TypeError(
                "source_evidence_base must be a ParentSourceEvidenceMappingV1"
            )
        if revision_base is not None and source_evidence_base is not None:
            raise ValueError("target selection may carry only one immutable base")
        if revision_base is None and source_evidence_base is None:
            if self.automatic_target_text is None:
                raise ValueError("automatic target topology requires automatic text")
            if self.selected_model_target_text is not None:
                raise ValueError(
                    "automatic target topology cannot carry selected model text"
                )
            if self.mapped_pipeline_target_text is not None:
                raise ValueError(
                    "automatic target topology cannot carry mapped pipeline text"
                )
            if authority not in {"automatic", "user"}:
                raise ValueError(
                    "automatic target topology supports automatic or user authority"
                )
        elif revision_base is not None:
            if self.automatic_target_text is not None:
                raise ValueError(
                    "revision-backed target topology has no Automatic target"
                )
            if self.selected_model_target_text is None:
                raise ValueError(
                    "revision-backed target topology requires selected model text"
                )
            if self.mapped_pipeline_target_text is not None:
                raise ValueError(
                    "revision-backed target topology cannot carry mapped pipeline text"
                )
            if authority not in {"translation_revision", "user"}:
                raise ValueError(
                    "revision-backed target topology supports model or user authority"
                )
            if (
                authority == "translation_revision"
                and self.effective_target_text != self.selected_model_target_text
            ):
                raise ValueError(
                    "selected model authority must expose the selected model text"
                )
        else:
            assert source_evidence_base is not None
            if self.automatic_target_text is not None:
                raise ValueError(
                    "mapped target topology has no standalone Automatic target"
                )
            if self.selected_model_target_text is not None:
                raise ValueError(
                    "mapped target topology cannot carry selected model text"
                )
            if (
                self.mapped_pipeline_target_text is None
                or self.mapped_pipeline_target_text
                != source_evidence_base.target_text
            ):
                raise ValueError(
                    "mapped target topology requires its exact pipeline text"
                )
            if authority not in {"mapped_automatic", "user"}:
                raise ValueError(
                    "mapped target topology supports mapped pipeline or user authority"
                )
            if (
                authority == "mapped_automatic"
                and self.effective_target_text
                != self.mapped_pipeline_target_text
            ):
                raise ValueError(
                    "mapped pipeline authority must expose the mapped pipeline text"
                )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "target_freshness",
            TargetFreshness(self.target_freshness),
        )


@dataclass(frozen=True, slots=True)
class TargetTextWorkerCommand:
    """UI-owned command carrier; it deliberately contains no CAS heads.

    The worker captures page/global edit heads from its thread-owned store.
    User text is never stripped, normalized, or rewritten.
    """

    project_path: str
    page_id: str
    parent_id: str
    operation: TargetTextOperation
    expected_effective_page_fingerprint: str
    text: str = ""
    command_id: str = ""
    revision_base: TargetTextRevisionBaseV1 | None = None
    source_evidence_base: ParentSourceEvidenceMappingV1 | None = None

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self, "page_id", _required_identity(self.page_id, "page_id")
        )
        object.__setattr__(
            self, "parent_id", _required_identity(self.parent_id, "parent_id")
        )
        operation = TargetTextOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if operation in {
            TargetTextOperation.RESTORE_AUTOMATIC,
            TargetTextOperation.RESTORE_SELECTED_REVISION,
            TargetTextOperation.RESTORE_MAPPED_PIPELINE,
        } and self.text != "":
            raise ValueError(f"{operation.value} must not carry replacement text")
        revision_base = self.revision_base
        if revision_base is not None and not isinstance(
            revision_base,
            TargetTextRevisionBaseV1,
        ):
            raise TypeError("revision_base must be a TargetTextRevisionBaseV1")
        if (
            operation is TargetTextOperation.RESTORE_SELECTED_REVISION
            and revision_base is None
        ):
            raise ValueError("restore_selected_revision requires revision_base")
        if (
            operation is TargetTextOperation.RESTORE_AUTOMATIC
            and revision_base is not None
        ):
            raise ValueError("restore_automatic must not carry revision_base")
        source_evidence_base = self.source_evidence_base
        if source_evidence_base is not None and not isinstance(
            source_evidence_base,
            ParentSourceEvidenceMappingV1,
        ):
            raise TypeError(
                "source_evidence_base must be a ParentSourceEvidenceMappingV1"
            )
        if revision_base is not None and source_evidence_base is not None:
            raise ValueError("target command may carry only one immutable base")
        if (
            operation is TargetTextOperation.RESTORE_MAPPED_PIPELINE
            and source_evidence_base is None
        ):
            raise ValueError("restore_mapped_pipeline requires source_evidence_base")
        if (
            operation
            in {
                TargetTextOperation.RESTORE_AUTOMATIC,
                TargetTextOperation.RESTORE_SELECTED_REVISION,
            }
            and source_evidence_base is not None
        ):
            raise ValueError(
                f"{operation.value} must not carry source_evidence_base"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )
        command_id = self.command_id or uuid.uuid4().hex
        command_id = _required_identity(command_id, "command_id")
        if _PATH_SAFE_ID.fullmatch(command_id) is None:
            raise ValueError("command_id must be path-safe")
        object.__setattr__(self, "command_id", command_id)


@dataclass(frozen=True, slots=True)
class TargetTextWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: TargetTextWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class TargetTextCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class TargetTextCancelledReceipt:
    command_id: str
    project_path: str
    page_id: str
    parent_id: str
    stage: TargetTextWorkerStage
    message: str = "Target-text update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class TargetTextWorkerFailure:
    code: TargetTextWorkerFailureCode
    stage: TargetTextWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    command_id: str
    message: str
    exception_type: str = ""
    command_error_code: TargetTextCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: TargetTextCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return (
            self.code is TargetTextWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class TargetTextWorkerReceipt:
    """Atomic shell-refresh payload produced entirely in the worker thread."""

    command_receipt: TargetTextCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, TargetTextCommandReceipt):
            raise TypeError("command_receipt must be TargetTextCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        # Local import avoids the shell package's eager MainWindow import while
        # the worker module itself is still being initialized.
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        edit = self.command_receipt.edit
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != self.command_receipt.after_effective_page_fingerprint
        ):
            raise ValueError("worker projection is not the committed effective page")
        parent = page.parent(edit.target.parent_id)
        receipt_parents = tuple(
            candidate
            for candidate in self.command_receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("target-text receipt has no exact effective parent")
        if (
            parent.effective.target_text != self.command_receipt.after_target_text
            or parent.effective.target_authority
            != self.command_receipt.after_target_authority
            or parent.effective.target_freshness
            is not receipt_parents[0].target_freshness
        ):
            raise ValueError(
                "worker projection does not contain the committed target state"
            )


@dataclass(frozen=True, slots=True)
class TargetTextEditorState:
    selection: TargetTextSelection
    phase: TargetTextEditorPhase
    draft_text: str
    message: str = ""
    worker_command: TargetTextWorkerCommand | None = None
    busy_state: TargetTextWorkerBusyState | None = None
    receipt: TargetTextWorkerReceipt | None = None
    failure: TargetTextWorkerFailure | None = None
    cancelled: TargetTextCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_text != self.selection.effective_target_text

    @property
    def busy(self) -> bool:
        return self.phase is TargetTextEditorPhase.COMMITTING

    @property
    def editing_enabled(self) -> bool:
        return not self.busy and self.phase is not TargetTextEditorPhase.STALE

    @property
    def apply_enabled(self) -> bool:
        return self.editing_enabled and self.dirty

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return (
            self.editing_enabled
            and not self.dirty
            and self.selection.target_authority == "user"
        )

    @property
    def keep_existing_enabled(self) -> bool:
        return (
            self.editing_enabled
            and not self.dirty
            and self.selection.target_freshness is TargetFreshness.STALE
            and self.worker_command is None
        )

    @property
    def stable_for_run(self) -> bool:
        """True only when no draft, stale projection, or command is pending."""

        return (
            not self.dirty
            and not self.busy
            and self.phase is not TargetTextEditorPhase.STALE
            and self.worker_command is None
            and self.selection.target_freshness is not TargetFreshness.STALE
        )

    @property
    def revision_backed(self) -> bool:
        return self.selection.revision_base is not None

    @property
    def mapped_pipeline_backed(self) -> bool:
        return self.selection.source_evidence_base is not None


class TargetTextEditorModel:
    """UI-thread reducer for exact target-text draft and worker events."""

    def __init__(self, selection: TargetTextSelection) -> None:
        if not isinstance(selection, TargetTextSelection):
            raise TypeError("selection must be TargetTextSelection")
        self._state = TargetTextEditorState(
            selection=selection,
            phase=TargetTextEditorPhase.READY,
            draft_text=selection.effective_target_text,
            message="Edit the exact effective target text, then choose Replace.",
        )

    @property
    def state(self) -> TargetTextEditorState:
        return self._state

    def set_draft(self, text: str) -> TargetTextEditorState:
        if not isinstance(text, str):
            raise TypeError("target-text draft must be a string")
        if not self._state.editing_enabled:
            raise RuntimeError("target-text draft is not editable")
        phase = (
            TargetTextEditorPhase.DIRTY
            if text != self._state.selection.effective_target_text
            else TargetTextEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_text=text,
            message=("Target text has unapplied changes." if phase is TargetTextEditorPhase.DIRTY else "No target-text changes."),
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> TargetTextEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard a draft while it is committing")
        phase = (
            TargetTextEditorPhase.STALE
            if self._state.phase is TargetTextEditorPhase.STALE
            else TargetTextEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_text=self._state.selection.effective_target_text,
            message=(
                "Reload the current parent before editing."
                if phase is TargetTextEditorPhase.STALE
                else "Draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(self._state.failure if phase is TargetTextEditorPhase.STALE else None),
            cancelled=None,
        )
        return self._state

    def begin_replace(self, *, command_id: str = "") -> TargetTextWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable target-text draft")
        command = self._command(
            TargetTextOperation.REPLACE,
            text=self._state.draft_text,
            command_id=command_id,
        )
        self._begin(command, "Applying target-text edit...")
        return command

    def begin_restore(self, *, command_id: str = "") -> TargetTextWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("the selected target base is already effective")
        revision_backed = self._state.revision_backed
        mapped_pipeline_backed = self._state.mapped_pipeline_backed
        command = self._command(
            (
                TargetTextOperation.RESTORE_SELECTED_REVISION
                if revision_backed
                else (
                    TargetTextOperation.RESTORE_MAPPED_PIPELINE
                    if mapped_pipeline_backed
                    else TargetTextOperation.RESTORE_AUTOMATIC
                )
            ),
            text="",
            command_id=command_id,
        )
        self._begin(
            command,
            (
                "Restoring the selected model translation..."
                if revision_backed
                else (
                    "Restoring the mapped pipeline translation..."
                    if mapped_pipeline_backed
                    else "Restoring automatic target text..."
                )
            ),
        )
        return command

    def begin_keep_existing(
        self,
        *,
        command_id: str = "",
    ) -> TargetTextWorkerCommand:
        if not self._state.keep_existing_enabled:
            raise RuntimeError("the current target text does not require acknowledgement")
        command = self._command(
            TargetTextOperation.REPLACE,
            text=self._state.selection.effective_target_text,
            command_id=command_id,
        )
        self._begin(command, "Keeping the existing target text explicitly...")
        return command

    def accept_busy(
        self,
        value: TargetTextWorkerBusyState,
    ) -> TargetTextEditorState:
        if not isinstance(value, TargetTextWorkerBusyState):
            raise TypeError("value must be TargetTextWorkerBusyState")
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(TargetTextEditorPhase.COMMITTING if value.busy else self._state.phase),
            message=(value.message or self._state.message),
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: TargetTextWorkerReceipt,
    ) -> TargetTextEditorState:
        if not isinstance(value, TargetTextWorkerReceipt):
            raise TypeError("value must be TargetTextWorkerReceipt")
        command = self._require_active_command(value.command_receipt.command_id)
        edit = value.command_receipt.edit
        if edit.page_id != command.page_id or edit.target.parent_id != command.parent_id:
            raise ValueError("target-text receipt belongs to another selection")
        if edit.operation != command.operation.value:
            raise ValueError("target-text receipt operation does not match the command")
        payload = dict(edit.payload)
        expected_revision_base = (
            command.revision_base.to_dict()
            if command.revision_base is not None
            else None
        )
        if payload.get("revision_base") != expected_revision_base:
            if expected_revision_base is not None or "revision_base" in payload:
                raise ValueError(
                    "target-text receipt revision base does not match the command"
                )
        expected_source_evidence_base = (
            command.source_evidence_base.to_dict()
            if command.source_evidence_base is not None
            else None
        )
        if payload.get("source_evidence_base") != expected_source_evidence_base:
            if (
                expected_source_evidence_base is not None
                or "source_evidence_base" in payload
            ):
                raise ValueError(
                    "target-text receipt mapped pipeline base does not match the command"
                )
        if (
            command.operation is TargetTextOperation.REPLACE
            and payload.get("text") != command.text
        ):
            raise ValueError("target-text receipt does not preserve exact text")
        if (
            command.operation is TargetTextOperation.RESTORE_SELECTED_REVISION
            and value.command_receipt.after_target_text
            != self._state.selection.selected_model_target_text
        ):
            raise ValueError(
                "target-text restore did not re-expose the selected model text"
            )
        if (
            command.operation is TargetTextOperation.RESTORE_MAPPED_PIPELINE
            and value.command_receipt.after_target_text
            != self._state.selection.mapped_pipeline_target_text
        ):
            raise ValueError(
                "target-text restore did not re-expose the mapped pipeline text"
            )
        selection = replace(
            self._state.selection,
            effective_target_text=value.command_receipt.after_target_text,
            target_authority=value.command_receipt.after_target_authority,
            effective_page_fingerprint=(
                value.command_receipt.after_effective_page_fingerprint
            ),
            target_freshness=(
                value.projection.page(edit.page_id)
                .parent(edit.target.parent_id)
                .effective.target_freshness
            ),
        )
        self._state = TargetTextEditorState(
            selection=selection,
            phase=TargetTextEditorPhase.COMMITTED,
            draft_text=value.command_receipt.after_target_text,
            message=(
                "Target text saved. Preview or rerender remains explicit."
                if selection.target_freshness is not TargetFreshness.STALE
                else "Automatic target restored, but it remains stale for the current source text."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: TargetTextWorkerFailure,
    ) -> TargetTextEditorState:
        if not isinstance(value, TargetTextWorkerFailure):
            raise TypeError("value must be TargetTextWorkerFailure")
        self._require_active_command(value.command_id)
        phase = (
            TargetTextEditorPhase.STALE
            if value.stale
            else TargetTextEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: TargetTextWorkerFailure,
    ) -> TargetTextEditorState:
        if not value.stale:
            raise ValueError("target-text failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: TargetTextCancelledReceipt,
    ) -> TargetTextEditorState:
        if not isinstance(value, TargetTextCancelledReceipt):
            raise TypeError("value must be TargetTextCancelledReceipt")
        self._require_active_command(value.command_id)
        self._state = replace(
            self._state,
            phase=TargetTextEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: TargetTextSelection) -> TargetTextEditorState:
        if not isinstance(selection, TargetTextSelection):
            raise TypeError("selection must be TargetTextSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a commit is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = self._state.draft_text if preserve_draft else selection.effective_target_text
        phase = (
            TargetTextEditorPhase.DIRTY
            if draft != selection.effective_target_text
            else TargetTextEditorPhase.READY
        )
        self._state = TargetTextEditorState(
            selection=selection,
            phase=phase,
            draft_text=draft,
            message=(
                "Current state changed; review the preserved draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied draft was preserved."
                if preserve_draft
                else "Current target text loaded."
            ),
        )
        return self._state

    def _command(
        self,
        operation: TargetTextOperation,
        *,
        text: str,
        command_id: str,
    ) -> TargetTextWorkerCommand:
        selection = self._state.selection
        return TargetTextWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
            text=text,
            command_id=command_id,
            revision_base=selection.revision_base,
            source_evidence_base=selection.source_evidence_base,
        )

    def _begin(self, command: TargetTextWorkerCommand, message: str) -> None:
        self._state = replace(
            self._state,
            phase=TargetTextEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_command(self, command_id: str) -> TargetTextWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no target-text worker command is active")
        if command.command_id != str(command_id or ""):
            raise ValueError("worker event belongs to another target-text command")
        return command

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> TargetTextWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no target-text worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command


class ParentMembershipEditorPhase(str, Enum):
    READY = "ready"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class ParentMembershipWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class ParentMembershipWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SNAPSHOT_STALE = "snapshot_stale"
    MEMBERSHIP_SLOT_CONFLICT = "membership_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class ParentMembershipSelection:
    """Selected effective membership from one immutable UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    excluded: bool
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self, "page_id", _required_identity(self.page_id, "page_id")
        )
        object.__setattr__(
            self, "parent_id", _required_identity(self.parent_id, "parent_id")
        )
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class ParentMembershipWorkerCommand:
    """UI command carrier with no persistence or CAS implementation state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: ParentMembershipOperation
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self, "page_id", _required_identity(self.page_id, "page_id")
        )
        object.__setattr__(
            self, "parent_id", _required_identity(self.parent_id, "parent_id")
        )
        object.__setattr__(
            self,
            "operation",
            ParentMembershipOperation(self.operation),
        )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )

    @property
    def desired_excluded(self) -> bool:
        return self.operation is ParentMembershipOperation.EXCLUDE


@dataclass(frozen=True, slots=True)
class ParentMembershipWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: ParentMembershipWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class ParentMembershipCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class ParentMembershipCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: ParentMembershipOperation
    stage: ParentMembershipWorkerStage
    message: str = "Parent membership update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class ParentMembershipWorkerFailure:
    code: ParentMembershipWorkerFailureCode
    stage: ParentMembershipWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: ParentMembershipOperation
    message: str
    exception_type: str = ""
    command_error_code: ParentMembershipCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: ParentMembershipCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return (
            self.code is ParentMembershipWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _MEMBERSHIP_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class ParentMembershipWorkerReceipt:
    """Atomic shell-refresh payload produced entirely in the worker thread."""

    command_receipt: ParentMembershipCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            ParentMembershipCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be ParentMembershipCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        edit = self.command_receipt.edit
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != self.command_receipt.after_effective_page_fingerprint
        ):
            raise ValueError(
                "worker projection is not the committed effective page"
            )
        parent = page.parent(edit.target.parent_id)
        if parent.effective.excluded != self.command_receipt.after_excluded:
            raise ValueError(
                "worker projection does not contain the committed membership"
            )


@dataclass(frozen=True, slots=True)
class ParentMembershipEditorState:
    selection: ParentMembershipSelection
    phase: ParentMembershipEditorPhase
    message: str = ""
    worker_command: ParentMembershipWorkerCommand | None = None
    busy_state: ParentMembershipWorkerBusyState | None = None
    receipt: ParentMembershipWorkerReceipt | None = None
    failure: ParentMembershipWorkerFailure | None = None
    cancelled: ParentMembershipCancelledReceipt | None = None

    @property
    def excluded(self) -> bool:
        return self.selection.excluded

    @property
    def busy(self) -> bool:
        return self.phase is ParentMembershipEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is ParentMembershipEditorPhase.STALE

    @property
    def desired_excluded(self) -> bool:
        return not self.selection.excluded

    @property
    def command_label(self) -> str:
        return "Restore Parent" if self.selection.excluded else "Exclude Parent"

    @property
    def command_enabled(self) -> bool:
        return (
            not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return not self.busy and not self.stale and self.worker_command is None

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        return {
            ParentMembershipEditorPhase.READY: "muted",
            ParentMembershipEditorPhase.COMMITTING: "editing",
            ParentMembershipEditorPhase.COMMITTED: "ready",
            ParentMembershipEditorPhase.CANCELLED: "muted",
            ParentMembershipEditorPhase.STALE: "warning",
            ParentMembershipEditorPhase.FAILED: "error",
        }[self.phase]


class ParentMembershipEditorModel:
    """UI-thread reducer for selected-parent exclude/restore commands."""

    def __init__(self, selection: ParentMembershipSelection) -> None:
        if not isinstance(selection, ParentMembershipSelection):
            raise TypeError("selection must be ParentMembershipSelection")
        self._state = ParentMembershipEditorState(
            selection=selection,
            phase=ParentMembershipEditorPhase.READY,
            message=self._ready_message(selection.excluded),
        )

    @property
    def state(self) -> ParentMembershipEditorState:
        return self._state

    def begin_set_excluded(
        self,
        desired_excluded: bool,
    ) -> ParentMembershipWorkerCommand:
        if not isinstance(desired_excluded, bool):
            raise TypeError("desired_excluded must be a boolean")
        if not self._state.command_enabled:
            raise RuntimeError("parent membership command is not available")
        if desired_excluded == self._state.selection.excluded:
            raise RuntimeError("requested parent membership is already effective")
        operation = (
            ParentMembershipOperation.EXCLUDE
            if desired_excluded
            else ParentMembershipOperation.RESTORE
        )
        selection = self._state.selection
        command = ParentMembershipWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )
        self._state = replace(
            self._state,
            phase=ParentMembershipEditorPhase.COMMITTING,
            message=(
                "Excluding the selected parent..."
                if desired_excluded
                else "Restoring the selected parent..."
            ),
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(
        self,
        value: ParentMembershipWorkerBusyState,
    ) -> ParentMembershipEditorState:
        if not isinstance(value, ParentMembershipWorkerBusyState):
            raise TypeError("value must be ParentMembershipWorkerBusyState")
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                ParentMembershipEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: ParentMembershipWorkerReceipt,
    ) -> ParentMembershipEditorState:
        if not isinstance(value, ParentMembershipWorkerReceipt):
            raise TypeError("value must be ParentMembershipWorkerReceipt")
        command = self._require_active_target(
            value.command_receipt.edit.page_id,
            value.command_receipt.edit.target.parent_id,
        )
        if value.command_receipt.edit.operation != command.operation.value:
            raise ValueError("membership receipt has another operation")
        if value.command_receipt.after_excluded != command.desired_excluded:
            raise ValueError("membership receipt has another effective state")
        selection = replace(
            self._state.selection,
            excluded=value.command_receipt.after_excluded,
            effective_page_fingerprint=(
                value.command_receipt.after_effective_page_fingerprint
            ),
        )
        self._state = ParentMembershipEditorState(
            selection=selection,
            phase=ParentMembershipEditorPhase.COMMITTED,
            message=(
                "Parent excluded. Preview this page remains explicit."
                if selection.excluded
                else "Automatic parent restored. Preview this page remains explicit."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: ParentMembershipWorkerFailure,
    ) -> ParentMembershipEditorState:
        if not isinstance(value, ParentMembershipWorkerFailure):
            raise TypeError("value must be ParentMembershipWorkerFailure")
        self._require_active_target(value.page_id, value.parent_id)
        phase = (
            ParentMembershipEditorPhase.STALE
            if value.stale
            else ParentMembershipEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: ParentMembershipWorkerFailure,
    ) -> ParentMembershipEditorState:
        if not value.stale:
            raise ValueError("parent-membership failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: ParentMembershipCancelledReceipt,
    ) -> ParentMembershipEditorState:
        if not isinstance(value, ParentMembershipCancelledReceipt):
            raise TypeError("value must be ParentMembershipCancelledReceipt")
        command = self._require_active_target(value.page_id, value.parent_id)
        if value.operation is not command.operation:
            raise ValueError("cancelled membership event has another operation")
        self._state = replace(
            self._state,
            phase=ParentMembershipEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: ParentMembershipSelection,
    ) -> ParentMembershipEditorState:
        if not isinstance(selection, ParentMembershipSelection):
            raise TypeError("selection must be ParentMembershipSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        same_effective_state = (
            same_target
            and selection.excluded == self._state.selection.excluded
            and selection.effective_page_fingerprint
            == self._state.selection.effective_page_fingerprint
        )
        if same_effective_state:
            self._state = replace(self._state, selection=selection)
            return self._state
        self._state = ParentMembershipEditorState(
            selection=selection,
            phase=ParentMembershipEditorPhase.READY,
            message=self._ready_message(selection.excluded),
        )
        return self._state

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> ParentMembershipWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no parent-membership worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    @staticmethod
    def _ready_message(excluded: bool) -> str:
        return (
            "This parent is excluded; Restore reveals the automatic parent."
            if excluded
            else "Exclude removes this parent from the effective page only."
        )


class ParentGeometryEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class ParentGeometryWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class ParentGeometryWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_GEOMETRY = "invalid_geometry"
    GEOMETRY_OUT_OF_BOUNDS = "geometry_out_of_bounds"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    GEOMETRY_SLOT_CONFLICT = "geometry_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class ParentGeometrySelection:
    """Selected automatic/effective geometry from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_bbox: tuple[int, int, int, int]
    effective_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    effective_page_fingerprint: str
    excluded: bool

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        canvas_size = _exact_canvas_size(self.canvas_size)
        object.__setattr__(self, "canvas_size", canvas_size)
        object.__setattr__(
            self,
            "automatic_bbox",
            _validated_page_bbox(
                self.automatic_bbox,
                "automatic_bbox",
                canvas_size,
            ),
        )
        object.__setattr__(
            self,
            "effective_bbox",
            _validated_page_bbox(
                self.effective_bbox,
                "effective_bbox",
                canvas_size,
            ),
        )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")


@dataclass(frozen=True, slots=True)
class ParentGeometryWorkerCommand:
    """UI carrier with no persistence identity or compare-and-swap heads."""

    project_path: str
    page_id: str
    parent_id: str
    bbox: tuple[int, int, int, int]
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        bbox = _exact_bbox_components(self.bbox, "bbox")
        if bbox[0] < 0 or bbox[1] < 0:
            raise ValueError("bbox origin must not be negative")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class ParentGeometryWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: ParentGeometryWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class ParentGeometryCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class ParentGeometryCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    bbox: tuple[int, int, int, int]
    stage: ParentGeometryWorkerStage
    message: str = "Parent geometry update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class ParentGeometryWorkerFailure:
    code: ParentGeometryWorkerFailureCode
    stage: ParentGeometryWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    bbox: tuple[int, int, int, int]
    message: str
    exception_type: str = ""
    command_error_code: ParentGeometryCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: ParentGeometryCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return (
            self.code is ParentGeometryWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _GEOMETRY_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class ParentGeometryWorkerReceipt:
    """Atomic shell-refresh payload produced in the worker thread."""

    command_receipt: ParentGeometryCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, ParentGeometryCommandReceipt):
            raise TypeError(
                "command_receipt must be ParentGeometryCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.project_edits.contracts import thaw_json
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        edit = self.command_receipt.edit
        if edit.operation != ParentGeometryOperation.SET_GEOMETRY.value:
            raise ValueError("worker receipt is not a geometry edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != self.command_receipt.after_effective_page_fingerprint
        ):
            raise ValueError(
                "worker projection is not the committed effective page"
            )
        parent = page.parent(edit.target.parent_id)
        projected_bbox = _exact_bbox_components(
            thaw_json(parent.effective.geometry),
            "projected parent geometry",
        )
        if projected_bbox != self.command_receipt.after_bbox:
            raise ValueError(
                "worker projection does not contain the committed geometry"
            )


@dataclass(frozen=True, slots=True)
class ParentGeometryEditorState:
    selection: ParentGeometrySelection
    phase: ParentGeometryEditorPhase
    draft_bbox: tuple[int, int, int, int]
    message: str = ""
    worker_command: ParentGeometryWorkerCommand | None = None
    busy_state: ParentGeometryWorkerBusyState | None = None
    receipt: ParentGeometryWorkerReceipt | None = None
    failure: ParentGeometryWorkerFailure | None = None
    cancelled: ParentGeometryCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_bbox != self.selection.effective_bbox

    @property
    def valid(self) -> bool:
        return (
            _bbox_validation_problem(
                self.draft_bbox,
                self.selection.canvas_size,
            )
            is None
        )

    @property
    def busy(self) -> bool:
        return self.phase is ParentGeometryEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is ParentGeometryEditorPhase.STALE

    @property
    def editing_enabled(self) -> bool:
        return not self.busy and not self.stale and not self.selection.excluded

    @property
    def apply_enabled(self) -> bool:
        return (
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return (
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        if self.phase is ParentGeometryEditorPhase.FAILED:
            return "error"
        if self.phase is ParentGeometryEditorPhase.STALE:
            return "warning"
        if self.phase is ParentGeometryEditorPhase.COMMITTING:
            return "editing"
        if self.phase is ParentGeometryEditorPhase.COMMITTED:
            return "ready"
        if self.dirty and not self.valid:
            return "warning"
        if self.dirty:
            return "editing"
        return "muted"


class ParentGeometryEditorModel:
    """UI-thread reducer for one selected-parent geometry draft."""

    def __init__(self, selection: ParentGeometrySelection) -> None:
        if not isinstance(selection, ParentGeometrySelection):
            raise TypeError("selection must be ParentGeometrySelection")
        self._state = ParentGeometryEditorState(
            selection=selection,
            phase=ParentGeometryEditorPhase.READY,
            draft_bbox=selection.effective_bbox,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> ParentGeometryEditorState:
        return self._state

    def set_draft_bbox(
        self,
        bbox: tuple[int, int, int, int],
    ) -> ParentGeometryEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("parent geometry draft is not editable")
        draft = _exact_bbox_components(bbox, "draft_bbox")
        dirty = draft != self._state.selection.effective_bbox
        problem = _bbox_validation_problem(
            draft,
            self._state.selection.canvas_size,
        )
        if not dirty:
            phase = ParentGeometryEditorPhase.READY
            message = "No geometry changes."
        elif problem is not None:
            phase = ParentGeometryEditorPhase.DIRTY
            message = problem
        else:
            phase = ParentGeometryEditorPhase.DIRTY
            message = "Parent geometry has unapplied changes."
        self._state = replace(
            self._state,
            phase=phase,
            draft_bbox=draft,
            message=message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> ParentGeometryEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard geometry while it is committing")
        phase = (
            ParentGeometryEditorPhase.STALE
            if self._state.stale
            else ParentGeometryEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_bbox=self._state.selection.effective_bbox,
            message=(
                "Reload the selected parent before editing geometry."
                if phase is ParentGeometryEditorPhase.STALE
                else "Geometry draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is ParentGeometryEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_apply(self) -> ParentGeometryWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no valid applicable geometry draft")
        selection = self._state.selection
        command = ParentGeometryWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            bbox=self._state.draft_bbox,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )
        self._state = replace(
            self._state,
            phase=ParentGeometryEditorPhase.COMMITTING,
            message="Applying selected-parent geometry...",
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(
        self,
        value: ParentGeometryWorkerBusyState,
    ) -> ParentGeometryEditorState:
        if not isinstance(value, ParentGeometryWorkerBusyState):
            raise TypeError("value must be ParentGeometryWorkerBusyState")
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                ParentGeometryEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: ParentGeometryWorkerReceipt,
    ) -> ParentGeometryEditorState:
        if not isinstance(value, ParentGeometryWorkerReceipt):
            raise TypeError("value must be ParentGeometryWorkerReceipt")
        edit = value.command_receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        if edit.operation != ParentGeometryOperation.SET_GEOMETRY.value:
            raise ValueError("geometry receipt has another operation")
        if value.command_receipt.after_bbox != command.bbox:
            raise ValueError("geometry receipt has another effective bbox")
        selection = replace(
            self._state.selection,
            effective_bbox=value.command_receipt.after_bbox,
            canvas_size=value.command_receipt.canvas_size,
            effective_page_fingerprint=(
                value.command_receipt.after_effective_page_fingerprint
            ),
        )
        self._state = ParentGeometryEditorState(
            selection=selection,
            phase=ParentGeometryEditorPhase.COMMITTED,
            draft_bbox=value.command_receipt.after_bbox,
            message=(
                "Parent geometry saved. Upstream revalidation and page preview remain explicit."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: ParentGeometryWorkerFailure,
    ) -> ParentGeometryEditorState:
        if not isinstance(value, ParentGeometryWorkerFailure):
            raise TypeError("value must be ParentGeometryWorkerFailure")
        command = self._require_active_target(value.page_id, value.parent_id)
        if value.bbox != command.bbox:
            raise ValueError("geometry failure belongs to another draft")
        phase = (
            ParentGeometryEditorPhase.STALE
            if value.stale
            else ParentGeometryEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: ParentGeometryWorkerFailure,
    ) -> ParentGeometryEditorState:
        if not value.stale:
            raise ValueError("parent-geometry failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: ParentGeometryCancelledReceipt,
    ) -> ParentGeometryEditorState:
        if not isinstance(value, ParentGeometryCancelledReceipt):
            raise TypeError("value must be ParentGeometryCancelledReceipt")
        command = self._require_active_target(value.page_id, value.parent_id)
        if value.bbox != command.bbox:
            raise ValueError("cancelled geometry event belongs to another draft")
        self._state = replace(
            self._state,
            phase=ParentGeometryEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: ParentGeometrySelection,
    ) -> ParentGeometryEditorState:
        if not isinstance(selection, ParentGeometrySelection):
            raise TypeError("selection must be ParentGeometrySelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = self._state.draft_bbox if preserve_draft else selection.effective_bbox
        dirty = draft != selection.effective_bbox
        problem = _bbox_validation_problem(draft, selection.canvas_size)
        phase = (
            ParentGeometryEditorPhase.DIRTY
            if dirty
            else ParentGeometryEditorPhase.READY
        )
        if selection.excluded:
            message = "Restore this parent before editing its geometry."
        elif dirty and problem is not None:
            message = problem
        elif preserve_draft and fingerprint_changed:
            message = "Current state changed; review the preserved geometry draft before applying."
        elif preserve_draft:
            message = "Selection refreshed; the unapplied geometry draft was preserved."
        else:
            message = "Current parent geometry loaded."
        self._state = ParentGeometryEditorState(
            selection=selection,
            phase=phase,
            draft_bbox=draft,
            message=message,
        )
        return self._state

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> ParentGeometryWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no parent-geometry worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    @staticmethod
    def _ready_message(selection: ParentGeometrySelection) -> str:
        if selection.excluded:
            return "Restore this parent before editing its geometry."
        if selection.effective_bbox == selection.automatic_bbox:
            return "Automatic parent geometry is effective."
        return "User-authored parent geometry is effective."


class SourceTextEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class SourceTextWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class SourceTextWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SNAPSHOT_STALE = "snapshot_stale"
    SOURCE_SLOT_CONFLICT = "source_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class SourceTextSelection:
    """Exact selected-parent source state from one immutable projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_source_text: str
    effective_source_text: str
    source_authority: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        if not isinstance(self.automatic_source_text, str):
            raise TypeError("automatic_source_text must be a string")
        if not isinstance(self.effective_source_text, str):
            raise TypeError("effective_source_text must be a string")
        authority = str(self.source_authority or "").strip()
        if authority not in {"automatic", "user"}:
            raise ValueError("source_authority must be automatic or user")
        object.__setattr__(self, "source_authority", authority)
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceTextWorkerCommand:
    """UI carrier with exact source text and no persistence/CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: SourceTextOperation
    text: str
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        operation = SourceTextOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if operation is SourceTextOperation.RESTORE_AUTOMATIC and self.text != "":
            raise ValueError("restore_automatic must not carry source text")
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceTextWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: SourceTextWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class SourceTextCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class SourceTextCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: SourceTextOperation
    text: str
    stage: SourceTextWorkerStage
    message: str = "Source-text update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class SourceTextWorkerFailure:
    code: SourceTextWorkerFailureCode
    stage: SourceTextWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: SourceTextOperation
    text: str
    message: str
    exception_type: str = ""
    command_error_code: SourceTextCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: SourceTextCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return (
            self.code is SourceTextWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SOURCE_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class SourceTextWorkerReceipt:
    """Atomic source and target-freshness refresh payload."""

    command_receipt: SourceTextCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, SourceTextCommandReceipt):
            raise TypeError("command_receipt must be SourceTextCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        edit = self.command_receipt.edit
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != self.command_receipt.after_effective_page_fingerprint
        ):
            raise ValueError(
                "worker projection is not the committed effective page"
            )
        parent = page.parent(edit.target.parent_id).effective
        if (
            parent.source_text != self.command_receipt.after_source_text
            or parent.source_authority
            != self.command_receipt.after_source_authority
            or parent.target_text != self.command_receipt.after_target_text
            or parent.target_authority
            != self.command_receipt.after_target_authority
            or parent.target_freshness
            is not self.command_receipt.after_target_freshness
        ):
            raise ValueError(
                "worker projection does not contain the committed source and target state"
            )


@dataclass(frozen=True, slots=True)
class SourceTextEditorState:
    selection: SourceTextSelection
    phase: SourceTextEditorPhase
    draft_text: str
    message: str = ""
    worker_command: SourceTextWorkerCommand | None = None
    busy_state: SourceTextWorkerBusyState | None = None
    receipt: SourceTextWorkerReceipt | None = None
    failure: SourceTextWorkerFailure | None = None
    cancelled: SourceTextCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_text != self.selection.effective_source_text

    @property
    def busy(self) -> bool:
        return self.phase is SourceTextEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is SourceTextEditorPhase.STALE

    @property
    def editing_enabled(self) -> bool:
        return not self.busy and not self.stale

    @property
    def apply_enabled(self) -> bool:
        return self.editing_enabled and self.dirty and self.worker_command is None

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return (
            self.editing_enabled
            and not self.dirty
            and self.selection.source_authority == "user"
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return (
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        return {
            SourceTextEditorPhase.READY: "muted",
            SourceTextEditorPhase.DIRTY: "editing",
            SourceTextEditorPhase.COMMITTING: "editing",
            SourceTextEditorPhase.COMMITTED: "ready",
            SourceTextEditorPhase.CANCELLED: "muted",
            SourceTextEditorPhase.STALE: "warning",
            SourceTextEditorPhase.FAILED: "error",
        }[self.phase]


class SourceTextEditorModel:
    """UI-thread reducer for exact selected-parent source text."""

    def __init__(self, selection: SourceTextSelection) -> None:
        if not isinstance(selection, SourceTextSelection):
            raise TypeError("selection must be SourceTextSelection")
        self._state = SourceTextEditorState(
            selection=selection,
            phase=SourceTextEditorPhase.READY,
            draft_text=selection.effective_source_text,
            message="Edit the effective source text, then choose Apply.",
        )

    @property
    def state(self) -> SourceTextEditorState:
        return self._state

    def set_draft(self, text: str) -> SourceTextEditorState:
        if not isinstance(text, str):
            raise TypeError("source-text draft must be a string")
        if not self._state.editing_enabled:
            raise RuntimeError("source-text draft is not editable")
        phase = (
            SourceTextEditorPhase.DIRTY
            if text != self._state.selection.effective_source_text
            else SourceTextEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_text=text,
            message=(
                "Source text has unapplied changes."
                if phase is SourceTextEditorPhase.DIRTY
                else "No source-text changes."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> SourceTextEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard source text while it is committing")
        phase = (
            SourceTextEditorPhase.STALE
            if self._state.stale
            else SourceTextEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_text=self._state.selection.effective_source_text,
            message=(
                "Reload the selected parent before editing source text."
                if phase is SourceTextEditorPhase.STALE
                else "Source-text draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is SourceTextEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_replace(self) -> SourceTextWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable source-text draft")
        command = self._command(
            SourceTextOperation.REPLACE,
            text=self._state.draft_text,
        )
        self._begin(command, "Applying source-text edit...")
        return command

    def begin_restore(self) -> SourceTextWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic source text is already effective")
        command = self._command(
            SourceTextOperation.RESTORE_AUTOMATIC,
            text="",
        )
        self._begin(command, "Restoring automatic source text...")
        return command

    def accept_busy(
        self,
        value: SourceTextWorkerBusyState,
    ) -> SourceTextEditorState:
        if not isinstance(value, SourceTextWorkerBusyState):
            raise TypeError("value must be SourceTextWorkerBusyState")
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                SourceTextEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: SourceTextWorkerReceipt,
    ) -> SourceTextEditorState:
        if not isinstance(value, SourceTextWorkerReceipt):
            raise TypeError("value must be SourceTextWorkerReceipt")
        edit = value.command_receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        if edit.operation != command.operation.value:
            raise ValueError("source-text receipt has another operation")
        if (
            command.operation is SourceTextOperation.REPLACE
            and value.command_receipt.after_source_text != command.text
        ):
            raise ValueError("source-text receipt has another effective text")
        if (
            command.expected_effective_page_fingerprint
            != value.command_receipt.before_effective_page_fingerprint
        ):
            raise ValueError("source-text receipt has another base revision")
        selection = replace(
            self._state.selection,
            effective_source_text=value.command_receipt.after_source_text,
            source_authority=value.command_receipt.after_source_authority,
            effective_page_fingerprint=(
                value.command_receipt.after_effective_page_fingerprint
            ),
        )
        target_is_stale = (
            value.command_receipt.after_target_freshness
            is TargetFreshness.STALE
        )
        self._state = SourceTextEditorState(
            selection=selection,
            phase=SourceTextEditorPhase.COMMITTED,
            draft_text=value.command_receipt.after_source_text,
            message=(
                "Source text saved. Resolve the stale target explicitly before Preview or Start."
                if target_is_stale
                else "Source text saved. Preview remains explicit."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: SourceTextWorkerFailure,
    ) -> SourceTextEditorState:
        if not isinstance(value, SourceTextWorkerFailure):
            raise TypeError("value must be SourceTextWorkerFailure")
        command = self._require_active_target(value.page_id, value.parent_id)
        if value.operation is not command.operation or value.text != command.text:
            raise ValueError("source-text failure belongs to another command")
        phase = (
            SourceTextEditorPhase.STALE
            if value.stale
            else SourceTextEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: SourceTextWorkerFailure,
    ) -> SourceTextEditorState:
        if not value.stale:
            raise ValueError("source-text failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: SourceTextCancelledReceipt,
    ) -> SourceTextEditorState:
        if not isinstance(value, SourceTextCancelledReceipt):
            raise TypeError("value must be SourceTextCancelledReceipt")
        command = self._require_active_target(value.page_id, value.parent_id)
        if value.operation is not command.operation or value.text != command.text:
            raise ValueError("cancelled source-text event has another command")
        self._state = replace(
            self._state,
            phase=SourceTextEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: SourceTextSelection) -> SourceTextEditorState:
        if not isinstance(selection, SourceTextSelection):
            raise TypeError("selection must be SourceTextSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = (
            self._state.draft_text
            if preserve_draft
            else selection.effective_source_text
        )
        phase = (
            SourceTextEditorPhase.DIRTY
            if draft != selection.effective_source_text
            else SourceTextEditorPhase.READY
        )
        self._state = SourceTextEditorState(
            selection=selection,
            phase=phase,
            draft_text=draft,
            message=(
                "Current state changed; review the preserved source-text draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied source-text draft was preserved."
                if preserve_draft
                else "Current source text loaded."
            ),
        )
        return self._state

    def _command(
        self,
        operation: SourceTextOperation,
        *,
        text: str,
    ) -> SourceTextWorkerCommand:
        selection = self._state.selection
        return SourceTextWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            text=text,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(self, command: SourceTextWorkerCommand, message: str) -> None:
        self._state = replace(
            self._state,
            phase=SourceTextEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> SourceTextWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no source-text worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command


class RenderLayoutWritingModeEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderLayoutWritingModeWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class RenderLayoutWritingModeWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_WRITING_MODE_UNAVAILABLE = "automatic_writing_mode_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    WRITING_MODE_SLOT_CONFLICT = "writing_mode_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_writing_mode(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    if value not in CANONICAL_WRITING_MODES:
        raise ValueError(
            f"{field_name} must be exactly 'horizontal' or 'vertical'"
        )
    return value


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeSelection:
    """Exact selected-parent writing-mode state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_writing_mode: str | None
    user_writing_mode: str | None
    effective_writing_mode: str | None
    writing_mode_authority: str
    render_required: bool
    excluded: bool
    unavailable_reason: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        for field_name in (
            "automatic_writing_mode",
            "user_writing_mode",
            "effective_writing_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_writing_mode(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.writing_mode_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "writing_mode_authority must be automatic or user"
            )
        object.__setattr__(self, "writing_mode_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_writing_mode is not None:
                raise ValueError(
                    "automatic writing-mode authority cannot carry a user value"
                )
            if self.effective_writing_mode != self.automatic_writing_mode:
                raise ValueError(
                    "automatic authority must expose the automatic effective mode"
                )
        else:
            if self.user_writing_mode is None:
                raise ValueError("user writing-mode authority requires a user value")
            if self.effective_writing_mode != self.user_writing_mode:
                raise ValueError(
                    "user authority must expose the user effective mode"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_writing_mode is not None
            and self.effective_writing_mode is not None
        )
        if eligible and reason:
            raise ValueError(
                "available writing-mode selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable writing-mode selection requires an unavailable reason"
            )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )

    @property
    def available(self) -> bool:
        return bool(
            not self.excluded
            and self.render_required
            and self.automatic_writing_mode is not None
            and self.effective_writing_mode is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeWorkerCommand:
    """UI carrier with exact mode and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutWritingModeOperation
    writing_mode: str
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        operation = RenderLayoutWritingModeOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.writing_mode, str):
            raise TypeError("writing_mode must be a string")
        if operation is RenderLayoutWritingModeOperation.SET:
            if self.writing_mode not in CANONICAL_WRITING_MODES:
                raise ValueError(
                    "writing_mode must be exactly 'horizontal' or 'vertical'"
                )
        elif self.writing_mode != "":
            raise ValueError(
                "restore_automatic must not carry a writing_mode value"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderLayoutWritingModeWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutWritingModeOperation
    writing_mode: str
    stage: RenderLayoutWritingModeWorkerStage
    message: str = "Writing-mode update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeWorkerFailure:
    code: RenderLayoutWritingModeWorkerFailureCode
    stage: RenderLayoutWritingModeWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutWritingModeOperation
    writing_mode: str
    message: str
    exception_type: str = ""
    command_error_code: RenderLayoutWritingModeCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderLayoutWritingModeCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderLayoutWritingModeWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _WRITING_MODE_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeWorkerReceipt:
    """Atomic committed project and UI-projection refresh payload."""

    command_receipt: RenderLayoutWritingModeCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderLayoutWritingModeCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderLayoutWritingModeCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        from app.project_edits.fingerprints import canonical_sha256

        if (
            canonical_sha256(self.project)
            != self.projection.source_project_fingerprint
        ):
            raise ValueError(
                "worker project mapping does not match the projected project"
            )
        receipt = self.command_receipt
        edit = receipt.edit
        if receipt.command_id != edit.edit_id:
            raise ValueError("reading-order command and edit identities disagree")
        if edit.domain is not EditDomain.RENDER_LAYOUT:
            raise ValueError("worker receipt is not a render-layout edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
        ):
            raise ValueError("worker projection is not the committed effective page")
        parent = page.parent(edit.target.parent_id).effective
        overrides = dict(parent.render_layout_overrides)
        projected_mode = overrides.get(
            "writing_mode",
            receipt.automatic_writing_mode,
        )
        projected_authority = (
            "user" if "writing_mode" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("writing-mode receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_layout_overrides)
        receipt_mode = receipt_overrides.get(
            "writing_mode",
            receipt.automatic_writing_mode,
        )
        receipt_authority = (
            "user" if "writing_mode" in receipt_overrides else "automatic"
        )
        if (
            projected_mode != receipt.after_writing_mode
            or projected_authority != receipt.after_writing_mode_authority
            or receipt_mode != receipt.after_writing_mode
            or receipt_authority != receipt.after_writing_mode_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed writing mode"
            )


@dataclass(frozen=True, slots=True)
class RenderLayoutWritingModeEditorState:
    selection: RenderLayoutWritingModeSelection
    phase: RenderLayoutWritingModeEditorPhase
    draft_writing_mode: str | None
    message: str = ""
    worker_command: RenderLayoutWritingModeWorkerCommand | None = None
    busy_state: RenderLayoutWritingModeWorkerBusyState | None = None
    receipt: RenderLayoutWritingModeWorkerReceipt | None = None
    failure: RenderLayoutWritingModeWorkerFailure | None = None
    cancelled: RenderLayoutWritingModeCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_writing_mode != self.selection.effective_writing_mode

    @property
    def valid(self) -> bool:
        return self.draft_writing_mode in CANONICAL_WRITING_MODES

    @property
    def busy(self) -> bool:
        return self.phase is RenderLayoutWritingModeEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderLayoutWritingModeEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return self.available and not self.busy and not self.stale

    @property
    def apply_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and not self.dirty
            and self.selection.writing_mode_authority == "user"
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        if (
            not self.available
            and self.phase
            in {
                RenderLayoutWritingModeEditorPhase.READY,
                RenderLayoutWritingModeEditorPhase.COMMITTED,
                RenderLayoutWritingModeEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderLayoutWritingModeEditorPhase.READY: "muted",
            RenderLayoutWritingModeEditorPhase.DIRTY: "editing",
            RenderLayoutWritingModeEditorPhase.COMMITTING: "editing",
            RenderLayoutWritingModeEditorPhase.COMMITTED: "ready",
            RenderLayoutWritingModeEditorPhase.CANCELLED: "muted",
            RenderLayoutWritingModeEditorPhase.STALE: "warning",
            RenderLayoutWritingModeEditorPhase.FAILED: "error",
        }[self.phase]


class RenderLayoutWritingModeEditorModel:
    """UI-thread reducer for one canonical selected-parent writing mode."""

    def __init__(self, selection: RenderLayoutWritingModeSelection) -> None:
        if not isinstance(selection, RenderLayoutWritingModeSelection):
            raise TypeError(
                "selection must be RenderLayoutWritingModeSelection"
            )
        self._state = RenderLayoutWritingModeEditorState(
            selection=selection,
            phase=RenderLayoutWritingModeEditorPhase.READY,
            draft_writing_mode=selection.effective_writing_mode,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderLayoutWritingModeEditorState:
        return self._state

    def set_draft_writing_mode(
        self,
        value: str,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(value, str):
            raise TypeError("writing-mode draft must be a string")
        if value not in CANONICAL_WRITING_MODES:
            raise ValueError(
                "writing-mode draft must be exactly 'horizontal' or 'vertical'"
            )
        if not self._state.editing_enabled:
            raise RuntimeError("writing-mode draft is not editable")
        phase = (
            RenderLayoutWritingModeEditorPhase.DIRTY
            if value != self._state.selection.effective_writing_mode
            else RenderLayoutWritingModeEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_writing_mode=value,
            message=(
                "Writing mode has an unapplied change."
                if phase is RenderLayoutWritingModeEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderLayoutWritingModeEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard writing mode while it is committing")
        phase = (
            RenderLayoutWritingModeEditorPhase.STALE
            if self._state.stale
            else RenderLayoutWritingModeEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_writing_mode=self._state.selection.effective_writing_mode,
            message=(
                "Reload the selected parent before editing writing mode."
                if phase is RenderLayoutWritingModeEditorPhase.STALE
                else "Writing-mode draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderLayoutWritingModeEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderLayoutWritingModeWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable writing-mode draft")
        writing_mode = self._state.draft_writing_mode
        if writing_mode is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("writing-mode draft is unavailable")
        command = self._command(
            RenderLayoutWritingModeOperation.SET,
            writing_mode=writing_mode,
        )
        self._begin(command, "Applying writing-mode edit...")
        return command

    def begin_restore(self) -> RenderLayoutWritingModeWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic writing mode is already effective")
        command = self._command(
            RenderLayoutWritingModeOperation.RESTORE_AUTOMATIC,
            writing_mode="",
        )
        self._begin(command, "Restoring automatic writing mode...")
        return command

    def accept_busy(
        self,
        value: RenderLayoutWritingModeWorkerBusyState,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(value, RenderLayoutWritingModeWorkerBusyState):
            raise TypeError(
                "value must be RenderLayoutWritingModeWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderLayoutWritingModeEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderLayoutWritingModeWorkerReceipt,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(value, RenderLayoutWritingModeWorkerReceipt):
            raise TypeError(
                "value must be RenderLayoutWritingModeWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderLayoutWritingModeOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("writing-mode receipt has another operation")
        if command.operation is RenderLayoutWritingModeOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"writing_mode": command.writing_mode}
                or receipt.after_writing_mode != command.writing_mode
            ):
                raise ValueError("writing-mode receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("writing_mode",)
                or receipt.after_writing_mode != receipt.automatic_writing_mode
            ):
                raise ValueError("writing-mode receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("writing-mode receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_writing_mode != selection.automatic_writing_mode
            or receipt.before_writing_mode != selection.effective_writing_mode
            or receipt.before_writing_mode_authority
            != selection.writing_mode_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("writing-mode receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_writing_mode=(
                receipt.after_writing_mode
                if receipt.after_writing_mode_authority == "user"
                else None
            ),
            effective_writing_mode=receipt.after_writing_mode,
            writing_mode_authority=receipt.after_writing_mode_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderLayoutWritingModeEditorState(
            selection=updated_selection,
            phase=RenderLayoutWritingModeEditorPhase.COMMITTED,
            draft_writing_mode=receipt.after_writing_mode,
            message="Writing mode saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderLayoutWritingModeWorkerFailure,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(value, RenderLayoutWritingModeWorkerFailure):
            raise TypeError(
                "value must be RenderLayoutWritingModeWorkerFailure"
            )
        command = self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.writing_mode,
        )
        del command
        phase = (
            RenderLayoutWritingModeEditorPhase.STALE
            if value.stale
            else RenderLayoutWritingModeEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: RenderLayoutWritingModeWorkerFailure,
    ) -> RenderLayoutWritingModeEditorState:
        if not value.stale:
            raise ValueError("writing-mode failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderLayoutWritingModeCancelledReceipt,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(value, RenderLayoutWritingModeCancelledReceipt):
            raise TypeError(
                "value must be RenderLayoutWritingModeCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.writing_mode,
        )
        self._state = replace(
            self._state,
            phase=RenderLayoutWritingModeEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: RenderLayoutWritingModeSelection,
    ) -> RenderLayoutWritingModeEditorState:
        if not isinstance(selection, RenderLayoutWritingModeSelection):
            raise TypeError(
                "selection must be RenderLayoutWritingModeSelection"
            )
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = (
            self._state.draft_writing_mode
            if preserve_draft
            else selection.effective_writing_mode
        )
        phase = (
            RenderLayoutWritingModeEditorPhase.DIRTY
            if draft != selection.effective_writing_mode
            else RenderLayoutWritingModeEditorPhase.READY
        )
        self._state = RenderLayoutWritingModeEditorState(
            selection=selection,
            phase=phase,
            draft_writing_mode=draft,
            message=(
                "Current state changed; review the preserved writing-mode draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied writing-mode draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderLayoutWritingModeOperation,
        *,
        writing_mode: str,
    ) -> RenderLayoutWritingModeWorkerCommand:
        selection = self._state.selection
        return RenderLayoutWritingModeWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            writing_mode=writing_mode,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderLayoutWritingModeWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderLayoutWritingModeEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> RenderLayoutWritingModeWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no writing-mode worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderLayoutWritingModeOperation,
        writing_mode: str,
    ) -> RenderLayoutWritingModeWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.writing_mode != writing_mode
        ):
            raise ValueError("worker event belongs to another writing-mode command")
        return command

    @staticmethod
    def _ready_message(selection: RenderLayoutWritingModeSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.writing_mode_authority == "user":
            return "User writing mode is effective."
        return "Automatic writing mode is effective."


class RenderLayoutLineHeightEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderLayoutLineHeightWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class RenderLayoutLineHeightWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_LINE_HEIGHT_UNAVAILABLE = "automatic_line_height_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    LINE_HEIGHT_SLOT_CONFLICT = "line_height_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_line_height(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_render_line_height(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightSelection:
    """Exact selected-parent line-height state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_line_height: float | None
    user_line_height: float | None
    effective_line_height: float | None
    line_height_authority: str
    render_required: bool
    excluded: bool
    unavailable_reason: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        for field_name in (
            "automatic_line_height",
            "user_line_height",
            "effective_line_height",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_line_height(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.line_height_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "line_height_authority must be automatic or user"
            )
        object.__setattr__(self, "line_height_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_line_height is not None:
                raise ValueError(
                    "automatic line-height authority cannot carry a user value"
                )
            if self.effective_line_height != self.automatic_line_height:
                raise ValueError(
                    "automatic authority must expose the automatic effective line height"
                )
        else:
            if self.user_line_height is None:
                raise ValueError(
                    "user line-height authority requires a user value"
                )
            if self.effective_line_height != self.user_line_height:
                raise ValueError(
                    "user authority must expose the user effective line height"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_line_height is not None
            and self.effective_line_height is not None
        )
        if eligible and reason:
            raise ValueError(
                "available line-height selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable line-height selection requires an unavailable reason"
            )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )

    @property
    def available(self) -> bool:
        return bool(
            not self.excluded
            and self.render_required
            and self.automatic_line_height is not None
            and self.effective_line_height is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightWorkerCommand:
    """UI carrier with one exact ratio and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutLineHeightOperation
    line_height: float | None
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        operation = RenderLayoutLineHeightOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutLineHeightOperation.SET:
            object.__setattr__(
                self,
                "line_height",
                canonical_render_line_height(self.line_height),
            )
        elif self.line_height is not None:
            raise ValueError(
                "restore_automatic must not carry a line_height value"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderLayoutLineHeightWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutLineHeightOperation
    line_height: float | None
    stage: RenderLayoutLineHeightWorkerStage
    message: str = "Line-height update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightWorkerFailure:
    code: RenderLayoutLineHeightWorkerFailureCode
    stage: RenderLayoutLineHeightWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutLineHeightOperation
    line_height: float | None
    message: str
    exception_type: str = ""
    command_error_code: RenderLayoutLineHeightCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderLayoutLineHeightCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderLayoutLineHeightWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _LINE_HEIGHT_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderLayoutLineHeightCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderLayoutLineHeightCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderLayoutLineHeightCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        from app.project_edits.fingerprints import canonical_sha256
        from app.project_edits.ledger import ProjectEditLedger

        if (
            canonical_sha256(self.project)
            != self.projection.source_project_fingerprint
        ):
            raise ValueError(
                "worker project mapping does not match the projected project"
            )
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        automatic_line_height = canonical_render_line_height(
            receipt.automatic_line_height,
            field_name="receipt automatic_line_height",
        )
        before_line_height = canonical_render_line_height(
            receipt.before_line_height,
            field_name="receipt before_line_height",
        )
        after_line_height = canonical_render_line_height(
            receipt.after_line_height,
            field_name="receipt after_line_height",
        )
        if receipt.before_line_height_authority not in {"automatic", "user"}:
            raise ValueError("line-height before authority is invalid")
        if receipt.after_line_height_authority not in {"automatic", "user"}:
            raise ValueError("line-height after authority is invalid")
        if (
            receipt.before_line_height_authority == "automatic"
            and before_line_height != automatic_line_height
        ):
            raise ValueError("automatic line-height before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("line-height command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("line-height supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("line-height commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_LAYOUT:
            raise ValueError("worker receipt is not a render-layout edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"line_height": after_line_height}
                or receipt.after_line_height_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed line-height set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("line_height",)
                or after_line_height != automatic_line_height
                or receipt.after_line_height_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed line-height restore"
                )
        else:
            raise ValueError("worker receipt has another render-layout operation")
        ledger = ProjectEditLedger.from_dict(self.project["edit_ledger"])
        if ledger.get(edit.edit_id) != edit:
            raise ValueError("worker project does not contain the committed edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if page.effective != receipt.effective_page:
            raise ValueError("worker projection is not the committed effective page")
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
        ):
            raise ValueError("worker projection has another effective fingerprint")
        parent = page.parent(edit.target.parent_id).effective
        overrides = dict(parent.render_layout_overrides)
        projected_height = canonical_render_line_height(
            overrides.get("line_height", automatic_line_height),
            field_name="projected line_height",
        )
        projected_authority = (
            "user" if "line_height" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("line-height receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_layout_overrides)
        receipt_height = canonical_render_line_height(
            receipt_overrides.get(
                "line_height",
                automatic_line_height,
            ),
            field_name="receipt line_height",
        )
        receipt_authority = (
            "user" if "line_height" in receipt_overrides else "automatic"
        )
        if (
            projected_height != after_line_height
            or projected_authority != receipt.after_line_height_authority
            or receipt_height != after_line_height
            or receipt_authority != receipt.after_line_height_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed line height"
            )


@dataclass(frozen=True, slots=True)
class RenderLayoutLineHeightEditorState:
    selection: RenderLayoutLineHeightSelection
    phase: RenderLayoutLineHeightEditorPhase
    draft_line_height: float | None
    message: str = ""
    worker_command: RenderLayoutLineHeightWorkerCommand | None = None
    busy_state: RenderLayoutLineHeightWorkerBusyState | None = None
    receipt: RenderLayoutLineHeightWorkerReceipt | None = None
    failure: RenderLayoutLineHeightWorkerFailure | None = None
    cancelled: RenderLayoutLineHeightCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_line_height != self.selection.effective_line_height

    @property
    def valid(self) -> bool:
        if self.draft_line_height is None:
            return False
        try:
            canonical_render_line_height(self.draft_line_height)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderLayoutLineHeightEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderLayoutLineHeightEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return self.available and not self.busy and not self.stale

    @property
    def apply_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and not self.dirty
            and self.selection.line_height_authority == "user"
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        if (
            not self.available
            and self.phase
            in {
                RenderLayoutLineHeightEditorPhase.READY,
                RenderLayoutLineHeightEditorPhase.COMMITTED,
                RenderLayoutLineHeightEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderLayoutLineHeightEditorPhase.READY: "muted",
            RenderLayoutLineHeightEditorPhase.DIRTY: "editing",
            RenderLayoutLineHeightEditorPhase.COMMITTING: "editing",
            RenderLayoutLineHeightEditorPhase.COMMITTED: "ready",
            RenderLayoutLineHeightEditorPhase.CANCELLED: "muted",
            RenderLayoutLineHeightEditorPhase.STALE: "warning",
            RenderLayoutLineHeightEditorPhase.FAILED: "error",
        }[self.phase]


class RenderLayoutLineHeightEditorModel:
    """UI-thread reducer for one exact selected-parent line-height ratio."""

    def __init__(self, selection: RenderLayoutLineHeightSelection) -> None:
        if not isinstance(selection, RenderLayoutLineHeightSelection):
            raise TypeError(
                "selection must be RenderLayoutLineHeightSelection"
            )
        self._state = RenderLayoutLineHeightEditorState(
            selection=selection,
            phase=RenderLayoutLineHeightEditorPhase.READY,
            draft_line_height=selection.effective_line_height,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderLayoutLineHeightEditorState:
        return self._state

    def set_draft_line_height(
        self,
        value: float,
    ) -> RenderLayoutLineHeightEditorState:
        line_height = canonical_render_line_height(
            value,
            field_name="line-height draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("line-height draft is not editable")
        phase = (
            RenderLayoutLineHeightEditorPhase.DIRTY
            if line_height != self._state.selection.effective_line_height
            else RenderLayoutLineHeightEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_line_height=line_height,
            message=(
                "Line height has an unapplied change."
                if phase is RenderLayoutLineHeightEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderLayoutLineHeightEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard line height while it is committing")
        phase = (
            RenderLayoutLineHeightEditorPhase.STALE
            if self._state.stale
            else RenderLayoutLineHeightEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_line_height=self._state.selection.effective_line_height,
            message=(
                "Reload the selected parent before editing line height."
                if phase is RenderLayoutLineHeightEditorPhase.STALE
                else "Line-height draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderLayoutLineHeightEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderLayoutLineHeightWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable line-height draft")
        line_height = self._state.draft_line_height
        if line_height is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("line-height draft is unavailable")
        command = self._command(
            RenderLayoutLineHeightOperation.SET,
            line_height=line_height,
        )
        self._begin(command, "Applying line-height edit...")
        return command

    def begin_restore(self) -> RenderLayoutLineHeightWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic line height is already effective")
        command = self._command(
            RenderLayoutLineHeightOperation.RESTORE_AUTOMATIC,
            line_height=None,
        )
        self._begin(command, "Restoring automatic line height...")
        return command

    def accept_busy(
        self,
        value: RenderLayoutLineHeightWorkerBusyState,
    ) -> RenderLayoutLineHeightEditorState:
        if not isinstance(value, RenderLayoutLineHeightWorkerBusyState):
            raise TypeError(
                "value must be RenderLayoutLineHeightWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderLayoutLineHeightEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderLayoutLineHeightWorkerReceipt,
    ) -> RenderLayoutLineHeightEditorState:
        if not isinstance(value, RenderLayoutLineHeightWorkerReceipt):
            raise TypeError(
                "value must be RenderLayoutLineHeightWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderLayoutLineHeightOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("line-height receipt has another operation")
        if command.operation is RenderLayoutLineHeightOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"line_height": command.line_height}
                or receipt.after_line_height != command.line_height
            ):
                raise ValueError("line-height receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("line_height",)
                or receipt.after_line_height != receipt.automatic_line_height
            ):
                raise ValueError("line-height receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("line-height receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_line_height != selection.automatic_line_height
            or receipt.before_line_height != selection.effective_line_height
            or receipt.before_line_height_authority
            != selection.line_height_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("line-height receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_line_height=(
                receipt.after_line_height
                if receipt.after_line_height_authority == "user"
                else None
            ),
            effective_line_height=receipt.after_line_height,
            line_height_authority=receipt.after_line_height_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderLayoutLineHeightEditorState(
            selection=updated_selection,
            phase=RenderLayoutLineHeightEditorPhase.COMMITTED,
            draft_line_height=receipt.after_line_height,
            message="Line height saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderLayoutLineHeightWorkerFailure,
    ) -> RenderLayoutLineHeightEditorState:
        if not isinstance(value, RenderLayoutLineHeightWorkerFailure):
            raise TypeError(
                "value must be RenderLayoutLineHeightWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.line_height,
        )
        phase = (
            RenderLayoutLineHeightEditorPhase.STALE
            if value.stale
            else RenderLayoutLineHeightEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: RenderLayoutLineHeightWorkerFailure,
    ) -> RenderLayoutLineHeightEditorState:
        if not value.stale:
            raise ValueError("line-height failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderLayoutLineHeightCancelledReceipt,
    ) -> RenderLayoutLineHeightEditorState:
        if not isinstance(value, RenderLayoutLineHeightCancelledReceipt):
            raise TypeError(
                "value must be RenderLayoutLineHeightCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.line_height,
        )
        self._state = replace(
            self._state,
            phase=RenderLayoutLineHeightEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: RenderLayoutLineHeightSelection,
    ) -> RenderLayoutLineHeightEditorState:
        if not isinstance(selection, RenderLayoutLineHeightSelection):
            raise TypeError(
                "selection must be RenderLayoutLineHeightSelection"
            )
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = (
            self._state.draft_line_height
            if preserve_draft
            else selection.effective_line_height
        )
        phase = (
            RenderLayoutLineHeightEditorPhase.DIRTY
            if draft != selection.effective_line_height
            else RenderLayoutLineHeightEditorPhase.READY
        )
        self._state = RenderLayoutLineHeightEditorState(
            selection=selection,
            phase=phase,
            draft_line_height=draft,
            message=(
                "Current state changed; review the preserved line-height draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied line-height draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderLayoutLineHeightOperation,
        *,
        line_height: float | None,
    ) -> RenderLayoutLineHeightWorkerCommand:
        selection = self._state.selection
        return RenderLayoutLineHeightWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            line_height=line_height,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderLayoutLineHeightWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderLayoutLineHeightEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> RenderLayoutLineHeightWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no line-height worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderLayoutLineHeightOperation,
        line_height: float | None,
    ) -> RenderLayoutLineHeightWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.line_height != line_height
        ):
            raise ValueError("worker event belongs to another line-height command")
        return command

    @staticmethod
    def _ready_message(selection: RenderLayoutLineHeightSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.line_height_authority == "user":
            return "User line height is effective."
        return "Automatic line height is effective."


class RenderLayoutRotationEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderLayoutRotationWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class RenderLayoutRotationWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_ROTATION_UNAVAILABLE = "automatic_rotation_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    ROTATION_SLOT_CONFLICT = "rotation_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_rotation(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_render_rotation(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationSelection:
    """Exact selected-parent rotation state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_rotation: float | None
    user_rotation: float | None
    effective_rotation: float | None
    rotation_authority: str
    render_required: bool
    excluded: bool
    unavailable_reason: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        for field_name in (
            "automatic_rotation",
            "user_rotation",
            "effective_rotation",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_rotation(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.rotation_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "rotation_authority must be automatic or user"
            )
        object.__setattr__(self, "rotation_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_rotation is not None:
                raise ValueError(
                    "automatic rotation authority cannot carry a user value"
                )
            if self.effective_rotation != self.automatic_rotation:
                raise ValueError(
                    "automatic authority must expose the automatic effective rotation"
                )
        else:
            if self.user_rotation is None:
                raise ValueError(
                    "user rotation authority requires a user value"
                )
            if self.effective_rotation != self.user_rotation:
                raise ValueError(
                    "user authority must expose the user effective rotation"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_rotation is not None
            and self.effective_rotation is not None
        )
        if eligible and reason:
            raise ValueError(
                "available rotation selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable rotation selection requires an unavailable reason"
            )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )

    @property
    def available(self) -> bool:
        return bool(
            not self.excluded
            and self.render_required
            and self.automatic_rotation is not None
            and self.effective_rotation is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationWorkerCommand:
    """UI carrier with one exact clockwise rotation and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRotationOperation
    rotation: float | None
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        operation = RenderLayoutRotationOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutRotationOperation.SET:
            object.__setattr__(
                self,
                "rotation",
                canonical_render_rotation(self.rotation),
            )
        elif self.rotation is not None:
            raise ValueError(
                "restore_automatic must not carry a rotation value"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderLayoutRotationWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRotationOperation
    rotation: float | None
    stage: RenderLayoutRotationWorkerStage
    message: str = "Rotation update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationWorkerFailure:
    code: RenderLayoutRotationWorkerFailureCode
    stage: RenderLayoutRotationWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRotationOperation
    rotation: float | None
    message: str
    exception_type: str = ""
    command_error_code: RenderLayoutRotationCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderLayoutRotationCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderLayoutRotationWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _ROTATION_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderLayoutRotationCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderLayoutRotationCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderLayoutRotationCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        from app.project_edits.fingerprints import canonical_sha256
        from app.project_edits.ledger import ProjectEditLedger

        if (
            canonical_sha256(self.project)
            != self.projection.source_project_fingerprint
        ):
            raise ValueError(
                "worker project mapping does not match the projected project"
            )
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        automatic_rotation = canonical_render_rotation(
            receipt.automatic_rotation,
            field_name="receipt automatic_rotation",
        )
        before_rotation = canonical_render_rotation(
            receipt.before_rotation,
            field_name="receipt before_rotation",
        )
        after_rotation = canonical_render_rotation(
            receipt.after_rotation,
            field_name="receipt after_rotation",
        )
        if receipt.before_rotation_authority not in {"automatic", "user"}:
            raise ValueError("rotation before authority is invalid")
        if receipt.after_rotation_authority not in {"automatic", "user"}:
            raise ValueError("rotation after authority is invalid")
        if (
            receipt.before_rotation_authority == "automatic"
            and before_rotation != automatic_rotation
        ):
            raise ValueError("automatic rotation before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("rotation command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("rotation supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("rotation commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_LAYOUT:
            raise ValueError("worker receipt is not a render-layout edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"rotation": after_rotation}
                or receipt.after_rotation_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed rotation set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("rotation",)
                or after_rotation != automatic_rotation
                or receipt.after_rotation_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed rotation restore"
                )
        else:
            raise ValueError("worker receipt has another render-layout operation")
        ledger = ProjectEditLedger.from_dict(self.project["edit_ledger"])
        if ledger.get(edit.edit_id) != edit:
            raise ValueError("worker project does not contain the committed edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if page.effective != receipt.effective_page:
            raise ValueError("worker projection is not the committed effective page")
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
        ):
            raise ValueError("worker projection has another effective fingerprint")
        parent = page.parent(edit.target.parent_id).effective
        overrides = dict(parent.render_layout_overrides)
        projected_rotation = canonical_render_rotation(
            overrides.get("rotation", automatic_rotation),
            field_name="projected rotation",
        )
        projected_authority = (
            "user" if "rotation" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("rotation receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_layout_overrides)
        receipt_rotation = canonical_render_rotation(
            receipt_overrides.get(
                "rotation",
                automatic_rotation,
            ),
            field_name="receipt rotation",
        )
        receipt_authority = (
            "user" if "rotation" in receipt_overrides else "automatic"
        )
        if (
            projected_rotation != after_rotation
            or projected_authority != receipt.after_rotation_authority
            or receipt_rotation != after_rotation
            or receipt_authority != receipt.after_rotation_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed rotation"
            )


@dataclass(frozen=True, slots=True)
class RenderLayoutRotationEditorState:
    selection: RenderLayoutRotationSelection
    phase: RenderLayoutRotationEditorPhase
    draft_rotation: float | None
    message: str = ""
    worker_command: RenderLayoutRotationWorkerCommand | None = None
    busy_state: RenderLayoutRotationWorkerBusyState | None = None
    receipt: RenderLayoutRotationWorkerReceipt | None = None
    failure: RenderLayoutRotationWorkerFailure | None = None
    cancelled: RenderLayoutRotationCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_rotation != self.selection.effective_rotation

    @property
    def valid(self) -> bool:
        if self.draft_rotation is None:
            return False
        try:
            canonical_render_rotation(self.draft_rotation)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderLayoutRotationEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderLayoutRotationEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return self.available and not self.busy and not self.stale

    @property
    def apply_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and not self.dirty
            and self.selection.rotation_authority == "user"
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        if (
            not self.available
            and self.phase
            in {
                RenderLayoutRotationEditorPhase.READY,
                RenderLayoutRotationEditorPhase.COMMITTED,
                RenderLayoutRotationEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderLayoutRotationEditorPhase.READY: "muted",
            RenderLayoutRotationEditorPhase.DIRTY: "editing",
            RenderLayoutRotationEditorPhase.COMMITTING: "editing",
            RenderLayoutRotationEditorPhase.COMMITTED: "ready",
            RenderLayoutRotationEditorPhase.CANCELLED: "muted",
            RenderLayoutRotationEditorPhase.STALE: "warning",
            RenderLayoutRotationEditorPhase.FAILED: "error",
        }[self.phase]


class RenderLayoutRotationEditorModel:
    """UI-thread reducer for one exact selected-parent clockwise rotation."""

    def __init__(self, selection: RenderLayoutRotationSelection) -> None:
        if not isinstance(selection, RenderLayoutRotationSelection):
            raise TypeError(
                "selection must be RenderLayoutRotationSelection"
            )
        self._state = RenderLayoutRotationEditorState(
            selection=selection,
            phase=RenderLayoutRotationEditorPhase.READY,
            draft_rotation=selection.effective_rotation,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderLayoutRotationEditorState:
        return self._state

    def set_draft_rotation(
        self,
        value: float,
    ) -> RenderLayoutRotationEditorState:
        rotation = canonical_render_rotation(
            value,
            field_name="rotation draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("rotation draft is not editable")
        phase = (
            RenderLayoutRotationEditorPhase.DIRTY
            if rotation != self._state.selection.effective_rotation
            else RenderLayoutRotationEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_rotation=rotation,
            message=(
                "Rotation has an unapplied change."
                if phase is RenderLayoutRotationEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderLayoutRotationEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard rotation while it is committing")
        phase = (
            RenderLayoutRotationEditorPhase.STALE
            if self._state.stale
            else RenderLayoutRotationEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_rotation=self._state.selection.effective_rotation,
            message=(
                "Reload the selected parent before editing rotation."
                if phase is RenderLayoutRotationEditorPhase.STALE
                else "Rotation draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderLayoutRotationEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderLayoutRotationWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable rotation draft")
        rotation = self._state.draft_rotation
        if rotation is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("rotation draft is unavailable")
        command = self._command(
            RenderLayoutRotationOperation.SET,
            rotation=rotation,
        )
        self._begin(command, "Applying rotation edit...")
        return command

    def begin_restore(self) -> RenderLayoutRotationWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic rotation is already effective")
        command = self._command(
            RenderLayoutRotationOperation.RESTORE_AUTOMATIC,
            rotation=None,
        )
        self._begin(command, "Restoring automatic rotation...")
        return command

    def accept_busy(
        self,
        value: RenderLayoutRotationWorkerBusyState,
    ) -> RenderLayoutRotationEditorState:
        if not isinstance(value, RenderLayoutRotationWorkerBusyState):
            raise TypeError(
                "value must be RenderLayoutRotationWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderLayoutRotationEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderLayoutRotationWorkerReceipt,
    ) -> RenderLayoutRotationEditorState:
        if not isinstance(value, RenderLayoutRotationWorkerReceipt):
            raise TypeError(
                "value must be RenderLayoutRotationWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderLayoutRotationOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("rotation receipt has another operation")
        if command.operation is RenderLayoutRotationOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"rotation": command.rotation}
                or receipt.after_rotation != command.rotation
            ):
                raise ValueError("rotation receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("rotation",)
                or receipt.after_rotation != receipt.automatic_rotation
            ):
                raise ValueError("rotation receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("rotation receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_rotation != selection.automatic_rotation
            or receipt.before_rotation != selection.effective_rotation
            or receipt.before_rotation_authority
            != selection.rotation_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("rotation receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_rotation=(
                receipt.after_rotation
                if receipt.after_rotation_authority == "user"
                else None
            ),
            effective_rotation=receipt.after_rotation,
            rotation_authority=receipt.after_rotation_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderLayoutRotationEditorState(
            selection=updated_selection,
            phase=RenderLayoutRotationEditorPhase.COMMITTED,
            draft_rotation=receipt.after_rotation,
            message="Rotation saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderLayoutRotationWorkerFailure,
    ) -> RenderLayoutRotationEditorState:
        if not isinstance(value, RenderLayoutRotationWorkerFailure):
            raise TypeError(
                "value must be RenderLayoutRotationWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.rotation,
        )
        phase = (
            RenderLayoutRotationEditorPhase.STALE
            if value.stale
            else RenderLayoutRotationEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: RenderLayoutRotationWorkerFailure,
    ) -> RenderLayoutRotationEditorState:
        if not value.stale:
            raise ValueError("rotation failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderLayoutRotationCancelledReceipt,
    ) -> RenderLayoutRotationEditorState:
        if not isinstance(value, RenderLayoutRotationCancelledReceipt):
            raise TypeError(
                "value must be RenderLayoutRotationCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.rotation,
        )
        self._state = replace(
            self._state,
            phase=RenderLayoutRotationEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: RenderLayoutRotationSelection,
    ) -> RenderLayoutRotationEditorState:
        if not isinstance(selection, RenderLayoutRotationSelection):
            raise TypeError(
                "selection must be RenderLayoutRotationSelection"
            )
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = (
            self._state.draft_rotation
            if preserve_draft
            else selection.effective_rotation
        )
        phase = (
            RenderLayoutRotationEditorPhase.DIRTY
            if draft != selection.effective_rotation
            else RenderLayoutRotationEditorPhase.READY
        )
        self._state = RenderLayoutRotationEditorState(
            selection=selection,
            phase=phase,
            draft_rotation=draft,
            message=(
                "Current state changed; review the preserved rotation draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied rotation draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderLayoutRotationOperation,
        *,
        rotation: float | None,
    ) -> RenderLayoutRotationWorkerCommand:
        selection = self._state.selection
        return RenderLayoutRotationWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            rotation=rotation,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderLayoutRotationWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderLayoutRotationEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> RenderLayoutRotationWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no rotation worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderLayoutRotationOperation,
        rotation: float | None,
    ) -> RenderLayoutRotationWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.rotation != rotation
        ):
            raise ValueError("worker event belongs to another rotation command")
        return command

    @staticmethod
    def _ready_message(selection: RenderLayoutRotationSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.rotation_authority == "user":
            return "User rotation is effective."
        return "Automatic rotation is effective."


class RenderStyleFillColorEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleFillColorWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class RenderStyleFillColorWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_FILL_COLOR_UNAVAILABLE = "automatic_fill_color_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    FILL_COLOR_SLOT_CONFLICT = "fill_color_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_fill_color(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return canonical_render_fill_color(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorSelection:
    """Canonical selected-parent fill color state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_fill_color: str | None
    user_fill_color: str | None
    effective_fill_color: str | None
    fill_color_authority: str
    render_required: bool
    excluded: bool
    unavailable_reason: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        for field_name in (
            "automatic_fill_color",
            "user_fill_color",
            "effective_fill_color",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_fill_color(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.fill_color_authority or "")
        if authority not in {"automatic", "user", "unresolved"}:
            raise ValueError(
                "fill_color_authority must be automatic, user, or unresolved"
            )
        object.__setattr__(self, "fill_color_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_fill_color is not None:
                raise ValueError(
                    "automatic fill_color authority cannot carry a user value"
                )
            if self.effective_fill_color != self.automatic_fill_color:
                raise ValueError(
                    "automatic authority must expose the automatic effective fill_color"
                )
        elif authority == "user":
            if self.user_fill_color is None:
                raise ValueError(
                    "user fill_color authority requires a user value"
                )
            if self.effective_fill_color != self.user_fill_color:
                raise ValueError(
                    "user authority must expose the user effective fill_color"
                )
        else:
            if (
                self.user_fill_color is not None
                or self.effective_fill_color is not None
            ):
                raise ValueError(
                    "unresolved fill_color authority cannot expose a canonical user or effective value"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_fill_color is not None
            and self.effective_fill_color is not None
        )
        if eligible and reason:
            raise ValueError(
                "available fill_color selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable fill_color selection requires an unavailable reason"
            )
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )

    @property
    def available(self) -> bool:
        return bool(
            not self.excluded
            and self.render_required
            and self.automatic_fill_color is not None
            and self.effective_fill_color is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorWorkerCommand:
    """UI carrier with one opaque fill color and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFillColorOperation
    fill_color: str | None
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        operation = RenderStyleFillColorOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleFillColorOperation.SET:
            object.__setattr__(
                self,
                "fill_color",
                canonical_render_fill_color(self.fill_color),
            )
        elif self.fill_color is not None:
            raise ValueError(
                "restore_automatic must not carry a fill_color value"
            )
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleFillColorWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFillColorOperation
    fill_color: str | None
    stage: RenderStyleFillColorWorkerStage
    message: str = "Fill color update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorWorkerFailure:
    code: RenderStyleFillColorWorkerFailureCode
    stage: RenderStyleFillColorWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFillColorOperation
    fill_color: str | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleFillColorCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleFillColorCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleFillColorWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _FILL_COLOR_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleFillColorCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleFillColorCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleFillColorCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        from app.project_edits.fingerprints import canonical_sha256
        from app.project_edits.ledger import ProjectEditLedger

        if (
            canonical_sha256(self.project)
            != self.projection.source_project_fingerprint
        ):
            raise ValueError(
                "worker project mapping does not match the projected project"
            )
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        automatic_fill_color = canonical_render_fill_color(
            receipt.automatic_fill_color,
            field_name="receipt automatic_fill_color",
        )
        before_fill_color = canonical_render_fill_color(
            receipt.before_fill_color,
            field_name="receipt before_fill_color",
        )
        after_fill_color = canonical_render_fill_color(
            receipt.after_fill_color,
            field_name="receipt after_fill_color",
        )
        if receipt.before_fill_color_authority not in {"automatic", "user"}:
            raise ValueError("fill_color before authority is invalid")
        if receipt.after_fill_color_authority not in {"automatic", "user"}:
            raise ValueError("fill_color after authority is invalid")
        if (
            receipt.before_fill_color_authority == "automatic"
            and before_fill_color != automatic_fill_color
        ):
            raise ValueError("automatic fill_color before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("fill_color command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("fill_color supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("fill_color commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"fill_color": after_fill_color}
                or receipt.after_fill_color_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed fill_color set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("fill_color",)
                or after_fill_color != automatic_fill_color
                or receipt.after_fill_color_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed fill_color restore"
                )
        else:
            raise ValueError("worker receipt has another render-style operation")
        ledger = ProjectEditLedger.from_dict(self.project["edit_ledger"])
        if ledger.get(edit.edit_id) != edit:
            raise ValueError("worker project does not contain the committed edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if page.effective != receipt.effective_page:
            raise ValueError("worker projection is not the committed effective page")
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
        ):
            raise ValueError("worker projection has another effective fingerprint")
        parent = page.parent(edit.target.parent_id).effective
        overrides = dict(parent.render_style_overrides)
        projected_fill_color = canonical_render_fill_color(
            overrides.get("fill_color", automatic_fill_color),
            field_name="projected fill_color",
        )
        projected_authority = (
            "user" if "fill_color" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("fill_color receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_fill_color = canonical_render_fill_color(
            receipt_overrides.get(
                "fill_color",
                automatic_fill_color,
            ),
            field_name="receipt fill_color",
        )
        receipt_authority = (
            "user" if "fill_color" in receipt_overrides else "automatic"
        )
        if (
            projected_fill_color != after_fill_color
            or projected_authority != receipt.after_fill_color_authority
            or receipt_fill_color != after_fill_color
            or receipt_authority != receipt.after_fill_color_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed fill_color"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleFillColorEditorState:
    selection: RenderStyleFillColorSelection
    phase: RenderStyleFillColorEditorPhase
    draft_fill_color: str | None
    message: str = ""
    worker_command: RenderStyleFillColorWorkerCommand | None = None
    busy_state: RenderStyleFillColorWorkerBusyState | None = None
    receipt: RenderStyleFillColorWorkerReceipt | None = None
    failure: RenderStyleFillColorWorkerFailure | None = None
    cancelled: RenderStyleFillColorCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        if self.draft_fill_color is None:
            return self.selection.effective_fill_color is not None
        try:
            canonical = canonical_render_fill_color(self.draft_fill_color)
        except (TypeError, ValueError):
            return True
        baseline = (
            self.selection.automatic_fill_color
            if self.selection.fill_color_authority == "unresolved"
            else self.selection.effective_fill_color
        )
        return canonical != baseline

    @property
    def valid(self) -> bool:
        if self.draft_fill_color is None:
            return False
        try:
            canonical_render_fill_color(self.draft_fill_color)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleFillColorEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleFillColorEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return bool(
            not self.selection.excluded
            and self.selection.render_required
            and self.selection.automatic_fill_color is not None
            and not self.busy
            and not self.stale
        )

    @property
    def apply_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def restore_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and not self.dirty
            and self.selection.fill_color_authority in {"user", "unresolved"}
            and self.worker_command is None
        )

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
            and self.selection.fill_color_authority != "unresolved"
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        if (
            not self.available
            and self.phase
            in {
                RenderStyleFillColorEditorPhase.READY,
                RenderStyleFillColorEditorPhase.COMMITTED,
                RenderStyleFillColorEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleFillColorEditorPhase.READY: "muted",
            RenderStyleFillColorEditorPhase.DIRTY: "editing",
            RenderStyleFillColorEditorPhase.COMMITTING: "editing",
            RenderStyleFillColorEditorPhase.COMMITTED: "ready",
            RenderStyleFillColorEditorPhase.CANCELLED: "muted",
            RenderStyleFillColorEditorPhase.STALE: "warning",
            RenderStyleFillColorEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleFillColorEditorModel:
    """UI-thread reducer for one exact selected-parent opaque fill color."""

    def __init__(self, selection: RenderStyleFillColorSelection) -> None:
        if not isinstance(selection, RenderStyleFillColorSelection):
            raise TypeError(
                "selection must be RenderStyleFillColorSelection"
            )
        self._state = RenderStyleFillColorEditorState(
            selection=selection,
            phase=RenderStyleFillColorEditorPhase.READY,
            draft_fill_color=selection.effective_fill_color,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleFillColorEditorState:
        return self._state

    def set_draft_fill_color(
        self,
        value: str,
    ) -> RenderStyleFillColorEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("fill_color draft is not editable")
        if not isinstance(value, str):
            raise TypeError("fill_color draft must be a string")
        try:
            canonical = canonical_render_fill_color(
                value,
                field_name="fill_color draft",
            )
        except (TypeError, ValueError):
            canonical = None
        baseline = (
            self._state.selection.automatic_fill_color
            if self._state.selection.fill_color_authority == "unresolved"
            else self._state.selection.effective_fill_color
        )
        dirty = canonical != baseline if canonical is not None else True
        phase = (
            RenderStyleFillColorEditorPhase.DIRTY
            if dirty
            else RenderStyleFillColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_fill_color=value,
            message=(
                "Fill color must use exactly #RRGGBB."
                if canonical is None
                else "Fill color has an unapplied change."
                if phase is RenderStyleFillColorEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleFillColorEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard fill_color while it is committing")
        phase = (
            RenderStyleFillColorEditorPhase.STALE
            if self._state.stale
            else RenderStyleFillColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_fill_color=self._state.selection.effective_fill_color,
            message=(
                "Reload the selected parent before editing fill_color."
                if phase is RenderStyleFillColorEditorPhase.STALE
                else "Fill color draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleFillColorEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleFillColorWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable fill_color draft")
        fill_color = self._state.draft_fill_color
        if fill_color is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("fill_color draft is unavailable")
        fill_color = canonical_render_fill_color(
            fill_color,
            field_name="fill_color draft",
        )
        command = self._command(
            RenderStyleFillColorOperation.SET,
            fill_color=fill_color,
        )
        self._begin(command, "Applying fill_color edit...")
        return command

    def begin_restore(self) -> RenderStyleFillColorWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic fill_color is already effective")
        command = self._command(
            RenderStyleFillColorOperation.RESTORE_AUTOMATIC,
            fill_color=None,
        )
        self._begin(command, "Restoring automatic fill_color...")
        return command

    def accept_busy(
        self,
        value: RenderStyleFillColorWorkerBusyState,
    ) -> RenderStyleFillColorEditorState:
        if not isinstance(value, RenderStyleFillColorWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleFillColorWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleFillColorEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleFillColorWorkerReceipt,
    ) -> RenderStyleFillColorEditorState:
        if not isinstance(value, RenderStyleFillColorWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleFillColorWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleFillColorOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("fill_color receipt has another operation")
        if command.operation is RenderStyleFillColorOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"fill_color": command.fill_color}
                or receipt.after_fill_color != command.fill_color
            ):
                raise ValueError("fill_color receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("fill_color",)
                or receipt.after_fill_color != receipt.automatic_fill_color
            ):
                raise ValueError("fill_color receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("fill_color receipt has another base revision")
        selection = self._state.selection
        if selection.fill_color_authority == "unresolved":
            before_matches_selection = bool(
                receipt.before_fill_color == selection.automatic_fill_color
                and receipt.before_fill_color_authority == "automatic"
            )
        else:
            before_matches_selection = bool(
                receipt.before_fill_color == selection.effective_fill_color
                and receipt.before_fill_color_authority
                == selection.fill_color_authority
            )
        if (
            receipt.automatic_fill_color != selection.automatic_fill_color
            or not before_matches_selection
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("fill_color receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_fill_color=(
                receipt.after_fill_color
                if receipt.after_fill_color_authority == "user"
                else None
            ),
            effective_fill_color=receipt.after_fill_color,
            fill_color_authority=receipt.after_fill_color_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleFillColorEditorState(
            selection=updated_selection,
            phase=RenderStyleFillColorEditorPhase.COMMITTED,
            draft_fill_color=receipt.after_fill_color,
            message="Fill color saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleFillColorWorkerFailure,
    ) -> RenderStyleFillColorEditorState:
        if not isinstance(value, RenderStyleFillColorWorkerFailure):
            raise TypeError(
                "value must be RenderStyleFillColorWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.fill_color,
        )
        phase = (
            RenderStyleFillColorEditorPhase.STALE
            if value.stale
            else RenderStyleFillColorEditorPhase.FAILED
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: RenderStyleFillColorWorkerFailure,
    ) -> RenderStyleFillColorEditorState:
        if not value.stale:
            raise ValueError("fill_color failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleFillColorCancelledReceipt,
    ) -> RenderStyleFillColorEditorState:
        if not isinstance(value, RenderStyleFillColorCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleFillColorCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.fill_color,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleFillColorEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: RenderStyleFillColorSelection,
    ) -> RenderStyleFillColorEditorState:
        if not isinstance(selection, RenderStyleFillColorSelection):
            raise TypeError(
                "selection must be RenderStyleFillColorSelection"
            )
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve_draft = same_target and self._state.dirty
        fingerprint_changed = (
            selection.effective_page_fingerprint
            != self._state.selection.effective_page_fingerprint
        )
        draft = (
            self._state.draft_fill_color
            if preserve_draft
            else selection.effective_fill_color
        )
        if draft is None:
            draft_dirty = selection.effective_fill_color is not None
        else:
            try:
                canonical_draft = canonical_render_fill_color(draft)
            except (TypeError, ValueError):
                draft_dirty = True
            else:
                baseline = (
                    selection.automatic_fill_color
                    if selection.fill_color_authority == "unresolved"
                    else selection.effective_fill_color
                )
                draft_dirty = canonical_draft != baseline
        phase = (
            RenderStyleFillColorEditorPhase.DIRTY
            if draft_dirty
            else RenderStyleFillColorEditorPhase.READY
        )
        self._state = RenderStyleFillColorEditorState(
            selection=selection,
            phase=phase,
            draft_fill_color=draft,
            message=(
                "Current state changed; review the preserved fill_color draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied fill_color draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleFillColorOperation,
        *,
        fill_color: str | None,
    ) -> RenderStyleFillColorWorkerCommand:
        selection = self._state.selection
        return RenderStyleFillColorWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            fill_color=fill_color,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleFillColorWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleFillColorEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> RenderStyleFillColorWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no fill_color worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleFillColorOperation,
        fill_color: str | None,
    ) -> RenderStyleFillColorWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.fill_color != fill_color
        ):
            raise ValueError("worker event belongs to another fill_color command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleFillColorSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.fill_color_authority == "user":
            return "User fill color is effective."
        return "Automatic fill color is effective."




class ReadingOrderEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class ReadingOrderWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class ReadingOrderWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    AUTOMATIC_ORDER_UNAVAILABLE = "automatic_order_unavailable"
    INVALID_ORDER = "invalid_order"
    EXCLUDED_SLOT_MOVED = "excluded_slot_moved"
    MULTIPLE_PARENTS_MOVED = "multiple_parents_moved"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    READING_ORDER_SLOT_CONFLICT = "reading_order_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _exact_identity_order(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    order = tuple(_required_identity(item, f"{field_name} item") for item in value)
    if len(order) != len(set(order)):
        raise ValueError(f"{field_name} must not contain duplicate IDs")
    return order


@dataclass(frozen=True, slots=True)
class ReadingOrderSelection:
    """Exact page-wide order with one active parent selected for movement."""

    project_path: str
    page_id: str
    selected_parent_id: str
    automatic_ordered_parent_ids: tuple[str, ...]
    effective_ordered_parent_ids: tuple[str, ...]
    excluded_parent_ids: tuple[str, ...]
    effective_page_fingerprint: str
    merge_retained_automatic_parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "selected_parent_id",
            _required_identity(self.selected_parent_id, "selected_parent_id"),
        )
        automatic = _exact_identity_order(
            self.automatic_ordered_parent_ids,
            "automatic_ordered_parent_ids",
        )
        effective = _exact_identity_order(
            self.effective_ordered_parent_ids,
            "effective_ordered_parent_ids",
        )
        if not isinstance(self.merge_retained_automatic_parent_ids, tuple):
            raise TypeError("merge_retained_automatic_parent_ids must be a tuple")
        retained = tuple(
            _required_identity(value, "merge_retained_automatic_parent_ids item")
            for value in self.merge_retained_automatic_parent_ids
        )
        if len(retained) != len(set(retained)):
            raise ValueError("merge-retained automatic parents must be unique")
        if (
            not frozenset(retained).issubset(automatic)
            or frozenset(retained).intersection(effective)
            or not frozenset(automatic).issubset((*effective, *retained))
        ):
            raise ValueError(
                "automatic order must contain only effective or merge-retained parents"
            )
        object.__setattr__(self, "automatic_ordered_parent_ids", automatic)
        object.__setattr__(self, "effective_ordered_parent_ids", effective)
        object.__setattr__(self, "merge_retained_automatic_parent_ids", retained)
        if not isinstance(self.excluded_parent_ids, tuple):
            raise TypeError("excluded_parent_ids must be a tuple")
        excluded = tuple(
            _required_identity(value, "excluded_parent_ids item")
            for value in self.excluded_parent_ids
        )
        if len(excluded) != len(set(excluded)):
            raise ValueError("excluded_parent_ids must not contain duplicates")
        if not set(excluded).issubset(effective):
            raise ValueError("excluded_parent_ids must belong to the page order")
        canonical_excluded = tuple(
            parent_id for parent_id in effective if parent_id in set(excluded)
        )
        if excluded != canonical_excluded:
            raise ValueError("excluded_parent_ids must follow effective page order")
        object.__setattr__(self, "excluded_parent_ids", excluded)
        if self.selected_parent_id not in effective:
            raise ValueError("selected parent must belong to the page order")
        if self.selected_parent_id in excluded:
            raise ValueError("selected parent must be active")
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReadingOrderWorkerCommand:
    """GUI carrier with no persistence-owned command ID or CAS heads."""

    project_path: str
    page_id: str
    selected_parent_id: str
    ordered_parent_ids: tuple[str, ...]
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(
            self,
            "selected_parent_id",
            _required_identity(self.selected_parent_id, "selected_parent_id"),
        )
        order = _exact_identity_order(self.ordered_parent_ids, "ordered_parent_ids")
        if self.selected_parent_id not in order:
            raise ValueError("ordered_parent_ids must include the selected parent")
        object.__setattr__(self, "ordered_parent_ids", order)
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReadingOrderWorkerBusyState:
    page_id: str
    selected_parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: ReadingOrderWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class ReadingOrderCancellationState:
    page_id: str
    selected_parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class ReadingOrderCancelledReceipt:
    project_path: str
    page_id: str
    selected_parent_id: str
    ordered_parent_ids: tuple[str, ...]
    stage: ReadingOrderWorkerStage
    message: str = "Reading-order update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class ReadingOrderWorkerFailure:
    code: ReadingOrderWorkerFailureCode
    stage: ReadingOrderWorkerStage
    project_path: str
    page_id: str
    selected_parent_id: str
    ordered_parent_ids: tuple[str, ...]
    message: str
    exception_type: str = ""
    command_error_code: ReadingOrderCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: ReadingOrderCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is ReadingOrderWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _READING_ORDER_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class ReadingOrderWorkerReceipt:
    """Atomic committed project and UI-projection refresh payload."""

    command_receipt: ReadingOrderCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, ReadingOrderCommandReceipt):
            raise TypeError("command_receipt must be ReadingOrderCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        from app.project_edits.fingerprints import canonical_sha256

        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError(
                "worker project mapping does not match the projected project"
            )
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        if not (
            receipt.command_id == edit.edit_id == commit.transaction_id
            and commit.page_id == edit.page_id
            and commit.edit_ids == (edit.edit_id,)
            and commit.artifact_revision_ids == ()
        ):
            raise ValueError(
                "reading-order command, edit, and commit identities disagree"
            )
        if edit.domain is not EditDomain.STRUCTURAL:
            raise ValueError("worker receipt is not a structural edit")
        if edit.operation != ReadingOrderOperation.SET.value:
            raise ValueError("worker receipt is not a reading-order edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        if edit.target.kind.value != "page":
            raise ValueError("reading-order edit must target the page")
        if edit.target.parent_id or edit.target.artifact_id or edit.target.edit_id:
            raise ValueError("reading-order page target must not carry another identity")
        payload_order = edit.payload.get("ordered_parent_ids")
        if (
            edit.payload.get("selected_parent_id") != receipt.selected_parent_id
            or not isinstance(payload_order, tuple)
            or payload_order != receipt.after_ordered_parent_ids
        ):
            raise ValueError("reading-order edit payload and receipt disagree")
        pages = self.project.get("pages")
        if not isinstance(pages, list):
            raise ValueError("worker project has no exact page collection")
        project_pages = tuple(
            candidate
            for candidate in pages
            if isinstance(candidate, Mapping)
            and str(candidate.get("page_id") or "").strip() == edit.page_id
        )
        if (
            len(project_pages) != 1
            or automatic_ordered_parent_ids_for_page(project_pages[0])
            != receipt.automatic_ordered_parent_ids
        ):
            raise ValueError("worker project and automatic page order disagree")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or tuple(page.effective.hierarchy.ordered_parent_ids)
            != receipt.after_ordered_parent_ids
            or receipt.effective_page.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or tuple(receipt.effective_page.hierarchy.ordered_parent_ids)
            != receipt.after_ordered_parent_ids
        ):
            raise ValueError("worker projection is not the committed page order")


@dataclass(frozen=True, slots=True)
class ReadingOrderEditorState:
    selection: ReadingOrderSelection
    phase: ReadingOrderEditorPhase
    draft_ordered_parent_ids: tuple[str, ...]
    message: str = ""
    worker_command: ReadingOrderWorkerCommand | None = None
    busy_state: ReadingOrderWorkerBusyState | None = None
    receipt: ReadingOrderWorkerReceipt | None = None
    failure: ReadingOrderWorkerFailure | None = None
    cancelled: ReadingOrderCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_ordered_parent_ids != self.selection.effective_ordered_parent_ids

    @property
    def busy(self) -> bool:
        return self.phase is ReadingOrderEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is ReadingOrderEditorPhase.STALE

    @property
    def valid(self) -> bool:
        draft = self.draft_ordered_parent_ids
        effective = self.selection.effective_ordered_parent_ids
        if len(draft) != len(effective) or frozenset(draft) != frozenset(effective):
            return False
        excluded = set(self.selection.excluded_parent_ids)
        return all(
            draft[index] == parent_id
            for index, parent_id in enumerate(effective)
            if parent_id in excluded
        )

    @property
    def active_draft_parent_ids(self) -> tuple[str, ...]:
        excluded = set(self.selection.excluded_parent_ids)
        return tuple(
            parent_id
            for parent_id in self.draft_ordered_parent_ids
            if parent_id not in excluded
        )

    @property
    def editing_enabled(self) -> bool:
        return not self.busy and not self.stale

    @property
    def can_move_earlier(self) -> bool:
        active = self.active_draft_parent_ids
        return bool(
            self.editing_enabled
            and self.selection.selected_parent_id in active
            and active.index(self.selection.selected_parent_id) > 0
        )

    @property
    def can_move_later(self) -> bool:
        active = self.active_draft_parent_ids
        return bool(
            self.editing_enabled
            and self.selection.selected_parent_id in active
            and active.index(self.selection.selected_parent_id) < len(active) - 1
        )

    @property
    def apply_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.dirty
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return not self.busy and self.dirty

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            not self.dirty
            and not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        return {
            ReadingOrderEditorPhase.READY: "muted",
            ReadingOrderEditorPhase.DIRTY: "editing",
            ReadingOrderEditorPhase.COMMITTING: "editing",
            ReadingOrderEditorPhase.COMMITTED: "ready",
            ReadingOrderEditorPhase.CANCELLED: "muted",
            ReadingOrderEditorPhase.STALE: "warning",
            ReadingOrderEditorPhase.FAILED: "error",
        }[self.phase]


class ReadingOrderEditorModel:
    """UI-thread reducer for one page-wide reading-order permutation."""

    def __init__(self, selection: ReadingOrderSelection) -> None:
        if not isinstance(selection, ReadingOrderSelection):
            raise TypeError("selection must be ReadingOrderSelection")
        self._state = ReadingOrderEditorState(
            selection=selection,
            phase=ReadingOrderEditorPhase.READY,
            draft_ordered_parent_ids=selection.effective_ordered_parent_ids,
            message="Move the selected parent earlier or later, then choose Apply.",
        )

    @property
    def state(self) -> ReadingOrderEditorState:
        return self._state

    def move_earlier(self) -> ReadingOrderEditorState:
        if not self._state.can_move_earlier:
            raise RuntimeError("selected parent cannot move earlier")
        return self._move(-1)

    def move_later(self) -> ReadingOrderEditorState:
        if not self._state.can_move_later:
            raise RuntimeError("selected parent cannot move later")
        return self._move(1)

    def cancel_draft(self) -> ReadingOrderEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard reading order while it is committing")
        phase = (
            ReadingOrderEditorPhase.STALE
            if self._state.stale
            else ReadingOrderEditorPhase.CANCELLED
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_ordered_parent_ids=(
                self._state.selection.effective_ordered_parent_ids
            ),
            message=(
                "Reload the selected page before editing reading order."
                if phase is ReadingOrderEditorPhase.STALE
                else "Reading-order draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(self._state.failure if phase is ReadingOrderEditorPhase.STALE else None),
            cancelled=None,
        )
        return self._state

    def begin_apply(self) -> ReadingOrderWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable reading-order draft")
        selection = self._state.selection
        command = ReadingOrderWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            selected_parent_id=selection.selected_parent_id,
            ordered_parent_ids=self._state.draft_ordered_parent_ids,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )
        self._state = replace(
            self._state,
            phase=ReadingOrderEditorPhase.COMMITTING,
            message="Applying page reading order...",
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(self, value: ReadingOrderWorkerBusyState) -> ReadingOrderEditorState:
        if not isinstance(value, ReadingOrderWorkerBusyState):
            raise TypeError("value must be ReadingOrderWorkerBusyState")
        self._require_active_target(value.page_id, value.selected_parent_id)
        self._state = replace(
            self._state,
            phase=(ReadingOrderEditorPhase.COMMITTING if value.busy else self._state.phase),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(self, value: ReadingOrderWorkerReceipt) -> ReadingOrderEditorState:
        if not isinstance(value, ReadingOrderWorkerReceipt):
            raise TypeError("value must be ReadingOrderWorkerReceipt")
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(edit.page_id, receipt.selected_parent_id)
        if (
            command.ordered_parent_ids != receipt.after_ordered_parent_ids
            or command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("reading-order receipt belongs to another draft")
        selection = self._state.selection
        if (
            receipt.automatic_ordered_parent_ids
            != selection.automatic_ordered_parent_ids
            or receipt.before_ordered_parent_ids
            != selection.effective_ordered_parent_ids
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("reading-order receipt belongs to another selection")
        updated_selection = replace(
            selection,
            effective_ordered_parent_ids=receipt.after_ordered_parent_ids,
            effective_page_fingerprint=receipt.after_effective_page_fingerprint,
        )
        self._state = ReadingOrderEditorState(
            selection=updated_selection,
            phase=ReadingOrderEditorPhase.COMMITTED,
            draft_ordered_parent_ids=receipt.after_ordered_parent_ids,
            message="Reading order saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(self, value: ReadingOrderWorkerFailure) -> ReadingOrderEditorState:
        if not isinstance(value, ReadingOrderWorkerFailure):
            raise TypeError("value must be ReadingOrderWorkerFailure")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=(
                ReadingOrderEditorPhase.STALE
                if value.stale
                else ReadingOrderEditorPhase.FAILED
            ),
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(self, value: ReadingOrderWorkerFailure) -> ReadingOrderEditorState:
        if not value.stale:
            raise ValueError("reading-order failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: ReadingOrderCancelledReceipt,
    ) -> ReadingOrderEditorState:
        if not isinstance(value, ReadingOrderCancelledReceipt):
            raise TypeError("value must be ReadingOrderCancelledReceipt")
        command = self._require_active_event(value)
        del command
        self._state = replace(
            self._state,
            phase=ReadingOrderEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: ReadingOrderSelection) -> ReadingOrderEditorState:
        if not isinstance(selection, ReadingOrderSelection):
            raise TypeError("selection must be ReadingOrderSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_page_and_parent = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.selected_parent_id
            == self._state.selection.selected_parent_id
        )
        preserve_draft = same_page_and_parent and self._state.dirty
        draft = (
            self._state.draft_ordered_parent_ids
            if preserve_draft
            and frozenset(self._state.draft_ordered_parent_ids)
            == frozenset(selection.effective_ordered_parent_ids)
            else selection.effective_ordered_parent_ids
        )
        candidate = ReadingOrderEditorState(
            selection=selection,
            phase=(
                ReadingOrderEditorPhase.DIRTY
                if draft != selection.effective_ordered_parent_ids
                else ReadingOrderEditorPhase.READY
            ),
            draft_ordered_parent_ids=draft,
            message=(
                "Current page changed; review the preserved reading-order draft."
                if preserve_draft and draft != selection.effective_ordered_parent_ids
                else "Current page reading order loaded."
            ),
        )
        if not candidate.valid:
            candidate = replace(
                candidate,
                phase=ReadingOrderEditorPhase.READY,
                draft_ordered_parent_ids=selection.effective_ordered_parent_ids,
                message="Current page reading order changed; draft was discarded.",
            )
        self._state = candidate
        return self._state

    def _move(self, active_offset: int) -> ReadingOrderEditorState:
        active = self._state.active_draft_parent_ids
        selected = self._state.selection.selected_parent_id
        selected_active_index = active.index(selected)
        other = active[selected_active_index + active_offset]
        draft = list(self._state.draft_ordered_parent_ids)
        selected_index = draft.index(selected)
        other_index = draft.index(other)
        draft[selected_index], draft[other_index] = draft[other_index], draft[selected_index]
        proposed = tuple(draft)
        phase = (
            ReadingOrderEditorPhase.DIRTY
            if proposed != self._state.selection.effective_ordered_parent_ids
            else ReadingOrderEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_ordered_parent_ids=proposed,
            message=(
                "Reading order has an unapplied page-wide change."
                if phase is ReadingOrderEditorPhase.DIRTY
                else "Effective reading order restored in the draft."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        if not self._state.valid:  # pragma: no cover - invariant defense
            raise RuntimeError("reading-order move violated page permutation rules")
        return self._state

    def _require_active_target(
        self,
        page_id: str,
        selected_parent_id: str,
    ) -> ReadingOrderWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no reading-order worker command is active")
        if (
            command.page_id != page_id
            or command.selected_parent_id != selected_parent_id
        ):
            raise ValueError("worker event belongs to another reading-order draft")
        return command

    def _require_active_event(
        self,
        value: ReadingOrderWorkerFailure | ReadingOrderCancelledReceipt,
    ) -> ReadingOrderWorkerCommand:
        command = self._require_active_target(
            value.page_id,
            value.selected_parent_id,
        )
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(value.project_path))
            or command.ordered_parent_ids != value.ordered_parent_ids
        ):
            raise ValueError("worker event belongs to another reading-order command")
        return command


class AddUserParentEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    PENDING = "pending"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class AddUserParentWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class AddUserParentWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_WORKFLOW_AREA = "invalid_workflow_area"
    WORKFLOW_AREA_OUT_OF_BOUNDS = "workflow_area_out_of_bounds"
    IDENTITY_COLLISION = "identity_collision"
    SNAPSHOT_STALE = "snapshot_stale"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class AddUserParentSelection:
    """Page-bound evidence from which one Add Parent draft may start."""

    project_path: str
    page_id: str
    canvas_size: tuple[int, int]
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    available: bool = True
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        object.__setattr__(self, "canvas_size", _exact_canvas_size(self.canvas_size))
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "hierarchy_revision_id",
            _required_identity(self.hierarchy_revision_id, "hierarchy_revision_id"),
        )
        object.__setattr__(
            self,
            "hierarchy_fingerprint",
            _required_sha256(
                self.hierarchy_fingerprint,
                "hierarchy_fingerprint",
            ),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        if self.available and reason:
            raise ValueError("available Add Parent state cannot have an unavailable reason")
        if not self.available and not reason:
            raise ValueError("unavailable Add Parent state requires a reason")
        object.__setattr__(self, "unavailable_reason", reason)


@dataclass(frozen=True, slots=True)
class AddUserParentWorkerCommand:
    """GUI carrier with fixed user IDs but no store, ledger, or CAS heads."""

    project_path: str
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    expected_effective_page_fingerprint: str
    expected_hierarchy_revision_id: str
    expected_hierarchy_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(
            self,
            "page_id",
            _required_identity(self.page_id, "page_id"),
        )
        parent_id = _required_identity(self.parent_id, "parent_id")
        root_id = _required_identity(self.root_id, "root_id")
        validate_user_parent_identity_pair(parent_id, root_id)
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "root_id", root_id)
        if self.role not in {"speech", "caption"}:
            raise ValueError("role must be speech or caption")
        object.__setattr__(
            self,
            "workflow_area_bbox",
            _exact_bbox_components(
                self.workflow_area_bbox,
                "workflow_area_bbox",
            ),
        )
        if self.workflow_area_bbox[0] < 0 or self.workflow_area_bbox[1] < 0:
            raise ValueError("workflow_area_bbox origin must not be negative")
        if self.workflow_area_bbox[2] <= 0 or self.workflow_area_bbox[3] <= 0:
            raise ValueError("workflow_area_bbox width and height must be positive")
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "expected_hierarchy_revision_id",
            _required_identity(
                self.expected_hierarchy_revision_id,
                "expected_hierarchy_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "expected_hierarchy_fingerprint",
            _required_sha256(
                self.expected_hierarchy_fingerprint,
                "expected_hierarchy_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class AddUserParentWorkerBusyState:
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: AddUserParentWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class AddUserParentCancellationState:
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class AddUserParentCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    stage: AddUserParentWorkerStage
    message: str = "Add Parent cancelled before persistence."


@dataclass(frozen=True, slots=True)
class AddUserParentWorkerFailure:
    code: AddUserParentWorkerFailureCode
    stage: AddUserParentWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    root_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    message: str
    exception_type: str = ""
    command_error_code: AddUserParentCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: AddUserParentCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is AddUserParentWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _ADD_USER_PARENT_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class AddUserParentWorkerReceipt:
    """Atomic, identity-bound committed Add Parent shell refresh."""

    command_receipt: AddUserParentCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, AddUserParentCommandReceipt):
            raise TypeError("command_receipt must be AddUserParentCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.project_edits.fingerprints import canonical_sha256
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project mapping does not match the projection")
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        if not (
            receipt.command_id == edit.edit_id == commit.transaction_id
            and commit.page_id == edit.page_id
            and commit.edit_ids == (edit.edit_id,)
            and commit.artifact_revision_ids == ()
        ):
            raise ValueError("Add Parent command, edit, and commit identities disagree")
        if (
            edit.domain is not EditDomain.STRUCTURAL
            or edit.operation != AddUserParentOperation.ADD.value
            or edit.target.kind.value != "parent"
            or edit.target.parent_id != receipt.parent_id
        ):
            raise ValueError("worker receipt is not one exact Add Parent edit")
        payload = edit.payload
        if set(payload) != {
            "identity_namespace",
            "root_id",
            "root_identity_namespace",
            "role",
            "workflow_area_bbox",
            "canvas_size",
            "order_policy",
        }:
            raise ValueError("Add Parent receipt payload has unsupported fields")
        if (
            payload.get("identity_namespace") != "user_parent_v1"
            or payload.get("root_id") != receipt.root_id
            or payload.get("root_identity_namespace") != "user_root_v1"
            or payload.get("role") != receipt.role
            or tuple(thaw_json(payload.get("workflow_area_bbox")))
            != receipt.workflow_area_bbox
            or tuple(thaw_json(payload.get("canvas_size"))) != receipt.canvas_size
            or payload.get("order_policy") != "append"
        ):
            raise ValueError("Add Parent receipt payload and command facts disagree")
        validate_user_parent_identity_pair(receipt.parent_id, receipt.root_id)
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if (
            page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or page.effective.hierarchy.revision_id
            != receipt.after_hierarchy_revision_id
            or page.effective.hierarchy.fingerprint
            != receipt.after_hierarchy_fingerprint
            or receipt.effective_page.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or receipt.effective_page.hierarchy.revision_id
            != receipt.after_hierarchy_revision_id
            or receipt.effective_page.hierarchy.fingerprint
            != receipt.after_hierarchy_fingerprint
        ):
            raise ValueError("worker projection is not the committed hierarchy revision")
        projected_parent = page.parent(receipt.parent_id)
        effective_parent = projected_parent.effective
        if (
            effective_parent.root_id != receipt.root_id
            or effective_parent.role != receipt.role
            or effective_parent.origin.value != "user"
            or effective_parent.bundle_id is not None
            or effective_parent.automatic_fingerprint is not None
            or effective_parent.automatic_geometry is not None
            or effective_parent.geometry is not None
            or tuple(thaw_json(effective_parent.workflow_area_bbox))
            != receipt.workflow_area_bbox
            or effective_parent.source_text is not None
            or effective_parent.target_text is not None
            or tuple(effective_parent.stage_requirements)
            != tuple(receipt.stage_requirements)
            or projected_parent.execution_ready
            or page.execution_ready
        ):
            raise ValueError("worker projection fabricated or lost pending parent state")
        roots = tuple(
            root for root in page.user_roots if root.root_id == receipt.root_id
        )
        lineage = effective_parent.lineage
        if (
            len(roots) != 1
            or roots[0].evidence_kind.value != "workflow_area_only"
            or roots[0].authored_edit_id != edit.edit_id
            or lineage is None
            or lineage.parent_id != receipt.parent_id
            or lineage.root_id != receipt.root_id
            or lineage.authored_edit_id != edit.edit_id
        ):
            raise ValueError("worker projection has no exact standalone user root")


@dataclass(frozen=True, slots=True)
class AddUserParentEditorState:
    selection: AddUserParentSelection
    phase: AddUserParentEditorPhase
    draft_role: str | None = None
    draft_workflow_area_bbox: (
        tuple[int | None, int | None, int | None, int | None] | None
    ) = None
    parent_id: str = ""
    root_id: str = ""
    message: str = ""
    worker_command: AddUserParentWorkerCommand | None = None
    busy_state: AddUserParentWorkerBusyState | None = None
    receipt: AddUserParentWorkerReceipt | None = None
    failure: AddUserParentWorkerFailure | None = None
    cancelled: AddUserParentCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        draft_present = bool(
            self.draft_role is not None
            or self.draft_workflow_area_bbox is not None
        )
        return bool(
            draft_present
            and not self.pending
            and self.receipt is None
            and (
                self.failure is None
                or not self.failure.persistence_committed
            )
        )

    @property
    def valid(self) -> bool:
        bbox = _complete_partial_bbox(self.draft_workflow_area_bbox)
        return bool(
            self.selection.available
            and self.draft_role in {"speech", "caption"}
            and bbox is not None
            and _bbox_validation_problem(
                bbox,
                self.selection.canvas_size,
            )
            is None
        )

    @property
    def busy(self) -> bool:
        return self.phase is AddUserParentEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is AddUserParentEditorPhase.STALE

    @property
    def pending(self) -> bool:
        return self.phase is AddUserParentEditorPhase.PENDING

    @property
    def unavailable(self) -> bool:
        return self.phase is AddUserParentEditorPhase.UNAVAILABLE

    @property
    def editing_enabled(self) -> bool:
        return bool(
            self.selection.available
            and not self.busy
            and not self.stale
            and not self.pending
        )

    @property
    def add_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and self.valid
            and self.worker_command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return bool(self.editing_enabled and self.dirty)

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(
            self.selection.available
            and not self.dirty
            and not self.busy
            and not self.stale
            and not self.pending
            and self.worker_command is None
        )

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        return {
            AddUserParentEditorPhase.READY: "muted",
            AddUserParentEditorPhase.DIRTY: "editing" if self.valid else "warning",
            AddUserParentEditorPhase.COMMITTING: "editing",
            AddUserParentEditorPhase.PENDING: "warning",
            AddUserParentEditorPhase.CANCELLED: "muted",
            AddUserParentEditorPhase.STALE: "warning",
            AddUserParentEditorPhase.FAILED: "error",
            AddUserParentEditorPhase.UNAVAILABLE: "warning",
        }[self.phase]


class AddUserParentEditorModel:
    """UI-thread reducer for a standalone, topology-only user parent draft."""

    def __init__(self, selection: AddUserParentSelection) -> None:
        if not isinstance(selection, AddUserParentSelection):
            raise TypeError("selection must be AddUserParentSelection")
        unavailable = not selection.available
        self._state = AddUserParentEditorState(
            selection=selection,
            phase=(
                AddUserParentEditorPhase.UNAVAILABLE
                if unavailable
                else AddUserParentEditorPhase.READY
            ),
            message=(
                selection.unavailable_reason
                if unavailable
                else "Choose Dialogue or Caption and enter a workflow area."
            ),
        )

    @property
    def state(self) -> AddUserParentEditorState:
        return self._state

    def set_draft_role(self, value: str | None) -> AddUserParentEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("Add Parent draft is not editable")
        if value is not None and value not in {"speech", "caption"}:
            raise ValueError("role must be speech, caption, or None")
        self._state = replace(
            self._state,
            draft_role=value,
            phase=AddUserParentEditorPhase.DIRTY,
            message=self._draft_message(
                value,
                self._state.draft_workflow_area_bbox,
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def set_draft_workflow_area(
        self,
        value: (
            tuple[int | None, int | None, int | None, int | None] | None
        ),
    ) -> AddUserParentEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("Add Parent draft is not editable")
        bbox = (
            None
            if value is None
            else _partial_bbox_components(value, "draft_workflow_area_bbox")
        )
        self._state = replace(
            self._state,
            draft_workflow_area_bbox=bbox,
            phase=AddUserParentEditorPhase.DIRTY,
            message=self._draft_message(self._state.draft_role, bbox),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> AddUserParentEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard Add Parent while it is committing")
        if self._state.pending:
            raise RuntimeError("a persisted user parent must be removed through History")
        phase = (
            AddUserParentEditorPhase.UNAVAILABLE
            if not self._state.selection.available
            else AddUserParentEditorPhase.READY
        )
        self._state = AddUserParentEditorState(
            selection=self._state.selection,
            phase=phase,
            message=(
                self._state.selection.unavailable_reason
                if phase is AddUserParentEditorPhase.UNAVAILABLE
                else "Add Parent draft discarded; project state was not changed."
            ),
        )
        return self._state

    def begin_add(
        self,
        parent_id: str | None = None,
        root_id: str | None = None,
    ) -> AddUserParentWorkerCommand:
        if not self._state.add_enabled:
            raise RuntimeError("there is no valid Add Parent draft")
        if (parent_id is None) != (root_id is None):
            raise ValueError("parent_id and root_id must be supplied together")
        fixed_parent_id = self._state.parent_id
        fixed_root_id = self._state.root_id
        if fixed_parent_id or fixed_root_id:
            validate_user_parent_identity_pair(fixed_parent_id, fixed_root_id)
            if parent_id is not None and (
                parent_id != fixed_parent_id or root_id != fixed_root_id
            ):
                raise ValueError("Add Parent retry must preserve its fixed identities")
        elif parent_id is None:
            fixed_parent_id, fixed_root_id = create_user_parent_identity()
        else:
            fixed_parent_id = str(parent_id)
            fixed_root_id = str(root_id)
            validate_user_parent_identity_pair(fixed_parent_id, fixed_root_id)
        selection = self._state.selection
        assert self._state.draft_role is not None
        workflow_area_bbox = _complete_partial_bbox(
            self._state.draft_workflow_area_bbox
        )
        if workflow_area_bbox is None:  # pragma: no cover - add_enabled invariant
            raise RuntimeError("Add Parent workflow area is incomplete")
        command = AddUserParentWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=fixed_parent_id,
            root_id=fixed_root_id,
            role=self._state.draft_role,
            workflow_area_bbox=workflow_area_bbox,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
            expected_hierarchy_revision_id=selection.hierarchy_revision_id,
            expected_hierarchy_fingerprint=selection.hierarchy_fingerprint,
        )
        self._state = replace(
            self._state,
            phase=AddUserParentEditorPhase.COMMITTING,
            parent_id=fixed_parent_id,
            root_id=fixed_root_id,
            message="Adding a standalone pending user parent...",
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(
        self,
        value: AddUserParentWorkerBusyState,
    ) -> AddUserParentEditorState:
        if not isinstance(value, AddUserParentWorkerBusyState):
            raise TypeError("value must be AddUserParentWorkerBusyState")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=(
                AddUserParentEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: AddUserParentWorkerReceipt,
    ) -> AddUserParentEditorState:
        if not isinstance(value, AddUserParentWorkerReceipt):
            raise TypeError("value must be AddUserParentWorkerReceipt")
        receipt = value.command_receipt
        command = self._require_active_identity(
            receipt.edit.page_id,
            receipt.parent_id,
            receipt.root_id,
            receipt.role,
            receipt.workflow_area_bbox,
        )
        if (
            receipt.before_effective_page_fingerprint
            != command.expected_effective_page_fingerprint
            or receipt.before_hierarchy_revision_id
            != command.expected_hierarchy_revision_id
            or receipt.before_hierarchy_fingerprint
            != command.expected_hierarchy_fingerprint
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("Add Parent receipt belongs to another page revision")
        selection = replace(
            self._state.selection,
            canvas_size=receipt.canvas_size,
            effective_page_fingerprint=receipt.after_effective_page_fingerprint,
            hierarchy_revision_id=receipt.after_hierarchy_revision_id,
            hierarchy_fingerprint=receipt.after_hierarchy_fingerprint,
        )
        self._state = AddUserParentEditorState(
            selection=selection,
            phase=AddUserParentEditorPhase.PENDING,
            draft_role=receipt.role,
            draft_workflow_area_bbox=receipt.workflow_area_bbox,
            parent_id=receipt.parent_id,
            root_id=receipt.root_id,
            message=(
                "User parent added as pending. Forward stage revisions are required; "
                "Start and Preview remain unavailable."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: AddUserParentWorkerFailure,
    ) -> AddUserParentEditorState:
        if not isinstance(value, AddUserParentWorkerFailure):
            raise TypeError("value must be AddUserParentWorkerFailure")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=(
                AddUserParentEditorPhase.STALE
                if value.stale
                else AddUserParentEditorPhase.FAILED
            ),
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: AddUserParentWorkerFailure,
    ) -> AddUserParentEditorState:
        if not value.stale:
            raise ValueError("Add Parent failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: AddUserParentCancelledReceipt,
    ) -> AddUserParentEditorState:
        if not isinstance(value, AddUserParentCancelledReceipt):
            raise TypeError("value must be AddUserParentCancelledReceipt")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=AddUserParentEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: AddUserParentSelection,
    ) -> AddUserParentEditorState:
        if not isinstance(selection, AddUserParentSelection):
            raise TypeError("selection must be AddUserParentSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace Add Parent page while committing")
        same_page = bool(
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
        )
        if (
            same_page
            and self._state.pending
            and self._state.receipt is not None
            and selection.effective_page_fingerprint
            == self._state.receipt.command_receipt.after_effective_page_fingerprint
            and selection.hierarchy_revision_id
            == self._state.receipt.command_receipt.after_hierarchy_revision_id
            and selection.hierarchy_fingerprint
            == self._state.receipt.command_receipt.after_hierarchy_fingerprint
        ):
            self._state = replace(self._state, selection=selection)
            return self._state
        preserve_draft = same_page and self._state.dirty and selection.available
        draft_role = self._state.draft_role if preserve_draft else None
        draft_bbox = (
            self._state.draft_workflow_area_bbox if preserve_draft else None
        )
        phase = (
            AddUserParentEditorPhase.UNAVAILABLE
            if not selection.available
            else (
                AddUserParentEditorPhase.DIRTY
                if preserve_draft
                else AddUserParentEditorPhase.READY
            )
        )
        self._state = AddUserParentEditorState(
            selection=selection,
            phase=phase,
            draft_role=draft_role,
            draft_workflow_area_bbox=draft_bbox,
            parent_id=self._state.parent_id if preserve_draft else "",
            root_id=self._state.root_id if preserve_draft else "",
            message=(
                selection.unavailable_reason
                if phase is AddUserParentEditorPhase.UNAVAILABLE
                else (
                    self._draft_message(draft_role, draft_bbox)
                    if preserve_draft
                    else "Choose Dialogue or Caption and enter a workflow area."
                )
            ),
        )
        return self._state

    def _require_active_identity(
        self,
        page_id: str,
        parent_id: str,
        root_id: str,
        role: str,
        workflow_area_bbox: tuple[int, int, int, int],
    ) -> AddUserParentWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no Add Parent worker command is active")
        if (
            command.page_id != page_id
            or command.parent_id != parent_id
            or command.root_id != root_id
            or command.role != role
            or command.workflow_area_bbox != workflow_area_bbox
        ):
            raise ValueError("worker event belongs to another Add Parent command")
        return command

    def _require_active_event(
        self,
        value: (
            AddUserParentWorkerBusyState
            | AddUserParentCancelledReceipt
            | AddUserParentWorkerFailure
        ),
    ) -> AddUserParentWorkerCommand:
        command = self._require_active_identity(
            value.page_id,
            value.parent_id,
            value.root_id,
            value.role,
            value.workflow_area_bbox,
        )
        project_path = getattr(value, "project_path", command.project_path)
        if os.path.normcase(os.path.abspath(project_path)) != os.path.normcase(
            os.path.abspath(command.project_path)
        ):
            raise ValueError("worker event belongs to another Add Parent project")
        return command

    def _draft_message(
        self,
        role: str | None,
        bbox: tuple[int | None, int | None, int | None, int | None] | None,
    ) -> str:
        if role is None and bbox is None:
            return "Choose Dialogue or Caption and enter a workflow area."
        if role is None:
            return "Choose Dialogue or Caption."
        if bbox is None:
            return "Enter integer X, Y, width, and height for the workflow area."
        complete_bbox = _complete_partial_bbox(bbox)
        if complete_bbox is None:
            return "Enter all four integer workflow-area values."
        problem = _bbox_validation_problem(
            complete_bbox,
            self._state.selection.canvas_size,
        )
        if problem is not None:
            return problem.replace("Geometry", "Workflow area")
        return "Add Parent draft is valid and has not changed the project."


class MergePipelineParentsEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class MergePipelineParentsWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class MergePipelineParentsWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    SOURCE_PARENT_NOT_FOUND = "source_parent_not_found"
    SOURCE_PARENT_NOT_AUTOMATIC = "source_parent_not_automatic"
    SOURCE_PARENT_EXCLUDED = "source_parent_excluded"
    SOURCE_PARENT_EDITED = "source_parent_edited"
    SOURCE_EVIDENCE_UNAVAILABLE = "source_evidence_unavailable"
    ROLE_MISMATCH = "role_mismatch"
    SOURCES_NOT_CONSECUTIVE = "sources_not_consecutive"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    IDENTITY_COLLISION = "identity_collision"
    MERGE_SLOT_CONFLICT = "merge_slot_conflict"
    SNAPSHOT_STALE = "snapshot_stale"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class MergePipelineParentCandidate:
    parent_id: str
    root_id: str
    label: str
    role: str
    bbox: tuple[int, int, int, int]
    source_text: str
    order_index: int

    def __post_init__(self) -> None:
        for field_name in ("parent_id", "root_id", "label"):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        if self.role not in {"speech", "caption"}:
            raise ValueError("role must be speech or caption")
        bbox = _exact_bbox_components(self.bbox, "bbox")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox is invalid")
        object.__setattr__(self, "bbox", bbox)
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source_text must contain exact non-empty OCR text")
        if isinstance(self.order_index, bool) or not isinstance(self.order_index, int):
            raise TypeError("order_index must be an integer")
        if self.order_index < 0:
            raise ValueError("order_index must be non-negative")


@dataclass(frozen=True, slots=True)
class MergePipelineParentsSelection:
    project_path: str
    page_id: str
    source: MergePipelineParentCandidate
    candidates: tuple[MergePipelineParentCandidate, ...]
    canvas_size: tuple[int, int]
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    available: bool = True
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(self, "page_id", _required_identity(self.page_id, "page_id"))
        if not isinstance(self.source, MergePipelineParentCandidate):
            raise TypeError("source must be a MergePipelineParentCandidate")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(value, MergePipelineParentCandidate)
            for value in self.candidates
        ):
            raise TypeError("candidates must contain typed merge candidates")
        candidate_ids = tuple(value.parent_id for value in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("merge candidates must have unique identities")
        if self.source.parent_id in set(candidate_ids):
            raise ValueError("source parent cannot also be a merge candidate")
        if any(value.role != self.source.role for value in self.candidates):
            raise ValueError("merge candidates must share the source role")
        if len({self.source.order_index, *(value.order_index for value in self.candidates)}) != len(self.candidates) + 1:
            raise ValueError("merge source and candidates must have unique order indexes")
        if any(abs(value.order_index - self.source.order_index) != 1 for value in self.candidates):
            raise ValueError("merge candidates must be adjacent to the source")
        canvas = _exact_canvas_size(self.canvas_size)
        for value in (self.source, *self.candidates):
            bbox = value.bbox
            if bbox[0] + bbox[2] > canvas[0] or bbox[1] + bbox[3] > canvas[1]:
                raise ValueError("merge candidate bbox must remain inside canvas")
        object.__setattr__(self, "canvas_size", canvas)
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "hierarchy_revision_id",
            _required_identity(self.hierarchy_revision_id, "hierarchy_revision_id"),
        )
        object.__setattr__(
            self,
            "hierarchy_fingerprint",
            _required_sha256(self.hierarchy_fingerprint, "hierarchy_fingerprint"),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        reason = str(self.unavailable_reason or "").strip()
        if self.available == bool(reason):
            raise ValueError("available merge selection and unavailable_reason disagree")
        if self.available and not self.candidates:
            raise ValueError("available merge selection requires a compatible candidate")
        object.__setattr__(self, "unavailable_reason", reason)


@dataclass(frozen=True, slots=True)
class MergePipelineParentsWorkerCommand:
    project_path: str
    page_id: str
    source_parent_ids: tuple[str, str]
    merged_parent_id: str
    merged_root_id: str
    expected_effective_page_fingerprint: str
    expected_hierarchy_revision_id: str
    expected_hierarchy_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(self, "page_id", _required_identity(self.page_id, "page_id"))
        if not isinstance(self.source_parent_ids, tuple) or len(self.source_parent_ids) != 2:
            raise ValueError("source_parent_ids must contain exactly two identities")
        source_ids = tuple(
            _required_identity(value, f"source_parent_ids[{index}]")
            for index, value in enumerate(self.source_parent_ids)
        )
        if len(set(source_ids)) != 2:
            raise ValueError("source_parent_ids must contain two unique identities")
        object.__setattr__(self, "source_parent_ids", source_ids)
        object.__setattr__(self, "merged_parent_id", _required_identity(self.merged_parent_id, "merged_parent_id"))
        object.__setattr__(self, "merged_root_id", _required_identity(self.merged_root_id, "merged_root_id"))
        validate_user_parent_identity_pair(self.merged_parent_id, self.merged_root_id)
        if self.merged_parent_id in set(source_ids):
            raise ValueError("merged parent identity must differ from source parents")
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(self.expected_effective_page_fingerprint, "expected_effective_page_fingerprint"),
        )
        object.__setattr__(
            self,
            "expected_hierarchy_revision_id",
            _required_identity(self.expected_hierarchy_revision_id, "expected_hierarchy_revision_id"),
        )
        object.__setattr__(
            self,
            "expected_hierarchy_fingerprint",
            _required_sha256(self.expected_hierarchy_fingerprint, "expected_hierarchy_fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class MergePipelineParentsWorkerBusyState:
    page_id: str
    source_parent_ids: tuple[str, str]
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: MergePipelineParentsWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class MergePipelineParentsCancellationState:
    page_id: str
    source_parent_ids: tuple[str, str]
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class MergePipelineParentsCancelledReceipt:
    project_path: str
    page_id: str
    source_parent_ids: tuple[str, str]
    stage: MergePipelineParentsWorkerStage
    message: str = "Merge Parent cancelled before persistence."


@dataclass(frozen=True, slots=True)
class MergePipelineParentsWorkerFailure:
    code: MergePipelineParentsWorkerFailureCode
    stage: MergePipelineParentsWorkerStage
    project_path: str
    page_id: str
    source_parent_ids: tuple[str, str]
    message: str
    exception_type: str = ""
    command_error_code: MergePipelineParentsCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: MergePipelineParentsCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is MergePipelineParentsWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _MERGE_PIPELINE_PARENTS_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class MergePipelineParentsWorkerReceipt:
    command_receipt: MergePipelineParentsCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, MergePipelineParentsCommandReceipt):
            raise TypeError("command_receipt must be MergePipelineParentsCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.project_edits.fingerprints import canonical_sha256
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project mapping does not match the projection")
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        if not (
            receipt.command_id == edit.edit_id == commit.transaction_id
            and commit.page_id == edit.page_id
            and commit.edit_ids == (edit.edit_id,)
            and commit.artifact_revision_ids == ()
            and edit.domain is EditDomain.STRUCTURAL
            and edit.operation == MergePipelineParentsOperation.MERGE.value
            and edit.target.parent_id == receipt.merged_parent_id
        ):
            raise ValueError("Merge Parent command, edit, and commit identities disagree")
        page = self.projection.page(edit.page_id)
        projected = page.parent(receipt.merged_parent_id).effective
        if (
            self.projection.metadata.project_id != edit.project_id
            or page.effective.effective_fingerprint != receipt.after_effective_page_fingerprint
            or page.effective.hierarchy.revision_id != receipt.after_hierarchy_revision_id
            or page.effective.hierarchy.fingerprint != receipt.after_hierarchy_fingerprint
            or any(parent_id in page.effective.hierarchy.ordered_parent_ids for parent_id in receipt.source_parent_ids)
            or projected.source_text != receipt.merged_source_text
            or tuple(thaw_json(projected.geometry)) != receipt.merged_workflow_area_bbox
            or tuple(projected.stage_requirements) != tuple(receipt.stage_requirements)
            or projected.lineage is None
            or projected.lineage.order_policy != "replace_sources"
            or projected.lineage.source_parent_ids != receipt.source_parent_ids
        ):
            raise ValueError("worker projection is not the committed merged revision")


@dataclass(frozen=True, slots=True)
class MergePipelineParentsEditorState:
    selection: MergePipelineParentsSelection
    phase: MergePipelineParentsEditorPhase
    partner_parent_id: str = ""
    merged_parent_id: str = ""
    merged_root_id: str = ""
    message: str = ""
    worker_command: MergePipelineParentsWorkerCommand | None = None
    busy_state: MergePipelineParentsWorkerBusyState | None = None
    receipt: MergePipelineParentsWorkerReceipt | None = None
    failure: MergePipelineParentsWorkerFailure | None = None
    cancelled: MergePipelineParentsCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return bool(self.phase is MergePipelineParentsEditorPhase.DIRTY and self.partner_parent_id)

    @property
    def busy(self) -> bool:
        return self.phase is MergePipelineParentsEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is MergePipelineParentsEditorPhase.STALE

    @property
    def editing_enabled(self) -> bool:
        return bool(self.selection.available and not self.busy and not self.stale)

    @property
    def merge_enabled(self) -> bool:
        return bool(self.editing_enabled and self.dirty and self.worker_command is None)

    @property
    def cancel_enabled(self) -> bool:
        return bool(self.editing_enabled and self.dirty)

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(not self.dirty and not self.busy and not self.stale)

    @property
    def status_tone(self) -> str:
        return {
            MergePipelineParentsEditorPhase.READY: "muted",
            MergePipelineParentsEditorPhase.DIRTY: "editing",
            MergePipelineParentsEditorPhase.COMMITTING: "editing",
            MergePipelineParentsEditorPhase.COMMITTED: "ready",
            MergePipelineParentsEditorPhase.CANCELLED: "muted",
            MergePipelineParentsEditorPhase.STALE: "warning",
            MergePipelineParentsEditorPhase.FAILED: "error",
            MergePipelineParentsEditorPhase.UNAVAILABLE: "muted",
        }[self.phase]


class MergePipelineParentsEditorModel:
    def __init__(self, selection: MergePipelineParentsSelection) -> None:
        if not isinstance(selection, MergePipelineParentsSelection):
            raise TypeError("selection must be MergePipelineParentsSelection")
        self._state = self._initial_state(selection)

    @property
    def state(self) -> MergePipelineParentsEditorState:
        return self._state

    def select_partner(self, parent_id: str | None) -> MergePipelineParentsEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("Merge Parent draft is not editable")
        candidate_id = str(parent_id or "")
        if candidate_id and candidate_id not in {
            value.parent_id for value in self._state.selection.candidates
        }:
            raise ValueError("selected Merge Parent candidate is unavailable")
        self._state = replace(
            self._state,
            phase=(MergePipelineParentsEditorPhase.DIRTY if candidate_id else MergePipelineParentsEditorPhase.READY),
            partner_parent_id=candidate_id,
            message=(
                "Unapplied merge combines both pipeline bboxes and OCR text. No project state has changed."
                if candidate_id
                else "Choose the adjacent compatible pipeline parent."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> MergePipelineParentsEditorState:
        if self._state.busy:
            raise RuntimeError("cannot cancel Merge Parent while it is committing")
        self._state = self._initial_state(
            self._state.selection,
            message="Merge Parent draft discarded; project state was not changed.",
        )
        return self._state

    def begin_merge(self) -> MergePipelineParentsWorkerCommand:
        if not self._state.merge_enabled:
            raise RuntimeError("there is no valid Merge Parent draft")
        merged_parent_id, merged_root_id = (
            (self._state.merged_parent_id, self._state.merged_root_id)
            if self._state.merged_parent_id
            else create_user_parent_identity()
        )
        selection = self._state.selection
        command = MergePipelineParentsWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            source_parent_ids=tuple(
                value.parent_id
                for value in sorted(
                    (
                        selection.source,
                        next(
                            candidate
                            for candidate in selection.candidates
                            if candidate.parent_id == self._state.partner_parent_id
                        ),
                    ),
                    key=lambda value: value.order_index,
                )
            ),
            merged_parent_id=merged_parent_id,
            merged_root_id=merged_root_id,
            expected_effective_page_fingerprint=selection.effective_page_fingerprint,
            expected_hierarchy_revision_id=selection.hierarchy_revision_id,
            expected_hierarchy_fingerprint=selection.hierarchy_fingerprint,
        )
        self._state = replace(
            self._state,
            phase=MergePipelineParentsEditorPhase.COMMITTING,
            merged_parent_id=merged_parent_id,
            merged_root_id=merged_root_id,
            message="Merging the two selected pipeline parents...",
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(self, value: MergePipelineParentsWorkerBusyState) -> MergePipelineParentsEditorState:
        self._require_event(value.page_id, value.source_parent_ids)
        self._state = replace(self._state, message=value.message, busy_state=value)
        return self._state

    def accept_receipt(self, value: MergePipelineParentsWorkerReceipt) -> MergePipelineParentsEditorState:
        if not isinstance(value, MergePipelineParentsWorkerReceipt):
            raise TypeError("value must be MergePipelineParentsWorkerReceipt")
        receipt = value.command_receipt
        command = self._require_event(receipt.edit.page_id, receipt.source_parent_ids)
        if (
            receipt.before_effective_page_fingerprint != command.expected_effective_page_fingerprint
            or receipt.before_hierarchy_revision_id != command.expected_hierarchy_revision_id
            or receipt.before_hierarchy_fingerprint != command.expected_hierarchy_fingerprint
        ):
            raise ValueError("Merge Parent receipt belongs to another revision")
        self._state = replace(
            self._state,
            phase=MergePipelineParentsEditorPhase.COMMITTED,
            message="Merge saved. Combined OCR is current; translation and later owners remain explicit.",
            worker_command=None,
            busy_state=None,
            receipt=value,
            failure=None,
            cancelled=None,
        )
        return self._state

    def accept_failure(self, value: MergePipelineParentsWorkerFailure) -> MergePipelineParentsEditorState:
        self._require_event(value.page_id, value.source_parent_ids, project_path=value.project_path)
        self._state = replace(
            self._state,
            phase=(MergePipelineParentsEditorPhase.STALE if value.stale else MergePipelineParentsEditorPhase.FAILED),
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(self, value: MergePipelineParentsWorkerFailure) -> MergePipelineParentsEditorState:
        if not value.stale:
            raise ValueError("Merge Parent failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(self, value: MergePipelineParentsCancelledReceipt) -> MergePipelineParentsEditorState:
        self._require_event(value.page_id, value.source_parent_ids, project_path=value.project_path)
        self._state = replace(
            self._state,
            phase=MergePipelineParentsEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: MergePipelineParentsSelection) -> MergePipelineParentsEditorState:
        if self._state.busy:
            raise RuntimeError("cannot replace Merge Parent selection while committing")
        same = bool(
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.source.parent_id == self._state.selection.source.parent_id
        )
        if same and self._state.dirty and selection.available and self._state.partner_parent_id in {value.parent_id for value in selection.candidates}:
            self._state = replace(self._state, selection=selection)
            return self._state
        self._state = self._initial_state(selection)
        return self._state

    def _initial_state(self, selection: MergePipelineParentsSelection, *, message: str | None = None) -> MergePipelineParentsEditorState:
        return MergePipelineParentsEditorState(
            selection=selection,
            phase=(MergePipelineParentsEditorPhase.READY if selection.available else MergePipelineParentsEditorPhase.UNAVAILABLE),
            message=(message if message is not None else ("Choose the adjacent compatible pipeline parent." if selection.available else selection.unavailable_reason)),
        )

    def _require_event(self, page_id: str, source_parent_ids: tuple[str, str], *, project_path: str | None = None) -> MergePipelineParentsWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no Merge Parent worker command is active")
        if command.page_id != page_id or command.source_parent_ids != tuple(source_parent_ids):
            raise ValueError("worker event belongs to another Merge Parent command")
        if project_path is not None and os.path.normcase(os.path.abspath(project_path)) != os.path.normcase(os.path.abspath(command.project_path)):
            raise ValueError("worker event belongs to another Merge Parent project")
        return command


class SplitUserParentEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class SplitUserParentWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    PROJECTING = "projecting"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class SplitUserParentWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    SOURCE_PARENT_NOT_FOUND = "source_parent_not_found"
    SOURCE_PARENT_NOT_STANDALONE = "source_parent_not_standalone"
    SOURCE_PARENT_EXCLUDED = "source_parent_excluded"
    CANVAS_UNAVAILABLE = "canvas_unavailable"
    INVALID_SPLIT_OFFSET = "invalid_split_offset"
    IDENTITY_COLLISION = "identity_collision"
    SPLIT_SLOT_CONFLICT = "split_slot_conflict"
    SNAPSHOT_STALE = "snapshot_stale"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class SplitUserParentSelection:
    project_path: str
    page_id: str
    source_parent_id: str
    source_root_id: str
    source_role: str
    source_workflow_area_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    available: bool = True
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in ("page_id", "source_parent_id", "source_root_id"):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(
            self.source_parent_id,
            self.source_root_id,
        )
        if self.source_role not in {"speech", "caption"}:
            raise ValueError("source_role must be speech or caption")
        bbox = _exact_bbox_components(
            self.source_workflow_area_bbox,
            "source_workflow_area_bbox",
        )
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("source_workflow_area_bbox is invalid")
        object.__setattr__(self, "source_workflow_area_bbox", bbox)
        canvas = _exact_canvas_size(self.canvas_size)
        if bbox[0] + bbox[2] > canvas[0] or bbox[1] + bbox[3] > canvas[1]:
            raise ValueError("source_workflow_area_bbox must remain inside canvas")
        object.__setattr__(self, "canvas_size", canvas)
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "hierarchy_revision_id",
            _required_identity(
                self.hierarchy_revision_id,
                "hierarchy_revision_id",
            ),
        )
        object.__setattr__(
            self,
            "hierarchy_fingerprint",
            _required_sha256(self.hierarchy_fingerprint, "hierarchy_fingerprint"),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        reason = str(self.unavailable_reason or "").strip()
        if self.available == bool(reason):
            raise ValueError(
                "available split selection and unavailable_reason disagree"
            )
        object.__setattr__(self, "unavailable_reason", reason)


@dataclass(frozen=True, slots=True)
class SplitUserParentWorkerCommand:
    project_path: str
    page_id: str
    source_parent_id: str
    first_parent_id: str
    first_root_id: str
    second_parent_id: str
    second_root_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    expected_effective_page_fingerprint: str
    expected_hierarchy_revision_id: str
    expected_hierarchy_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in (
            "page_id",
            "source_parent_id",
            "first_parent_id",
            "first_root_id",
            "second_parent_id",
            "second_root_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.first_parent_id, self.first_root_id)
        validate_user_parent_identity_pair(self.second_parent_id, self.second_root_id)
        if len(
            {self.source_parent_id, self.first_parent_id, self.second_parent_id}
        ) != 3:
            raise ValueError("source and child parent identities must be unique")
        object.__setattr__(
            self,
            "orientation",
            SplitUserParentOrientation(self.orientation),
        )
        if (
            isinstance(self.split_offset, bool)
            or not isinstance(self.split_offset, int)
            or self.split_offset <= 0
        ):
            raise ValueError("split_offset must be a positive exact integer")
        for field_name in (
            "expected_effective_page_fingerprint",
            "expected_hierarchy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "expected_hierarchy_revision_id",
            _required_identity(
                self.expected_hierarchy_revision_id,
                "expected_hierarchy_revision_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class SplitUserParentWorkerBusyState:
    page_id: str
    source_parent_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: SplitUserParentWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class SplitUserParentCancellationState:
    page_id: str
    source_parent_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class SplitUserParentCancelledReceipt:
    project_path: str
    page_id: str
    source_parent_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    stage: SplitUserParentWorkerStage
    message: str = "Split Parent cancelled before persistence."


@dataclass(frozen=True, slots=True)
class SplitUserParentWorkerFailure:
    code: SplitUserParentWorkerFailureCode
    stage: SplitUserParentWorkerStage
    project_path: str
    page_id: str
    source_parent_id: str
    orientation: SplitUserParentOrientation
    split_offset: int
    message: str
    exception_type: str = ""
    command_error_code: SplitUserParentCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: SplitUserParentCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is SplitUserParentWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SPLIT_USER_PARENT_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class SplitUserParentWorkerReceipt:
    command_receipt: SplitUserParentCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, SplitUserParentCommandReceipt):
            raise TypeError("command_receipt must be SplitUserParentCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.project_edits.fingerprints import canonical_sha256
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project mapping does not match the projection")
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        if not (
            receipt.command_id == edit.edit_id == commit.transaction_id
            and commit.page_id == edit.page_id
            and commit.edit_ids == (edit.edit_id,)
            and commit.artifact_revision_ids == ()
            and edit.domain is EditDomain.STRUCTURAL
            and edit.operation == SplitUserParentOperation.SPLIT.value
            and edit.target.parent_id == receipt.source_parent_id
        ):
            raise ValueError("Split Parent command, edit, and commit identities disagree")
        page = self.projection.page(edit.page_id)
        if (
            self.projection.metadata.project_id != edit.project_id
            or page.effective.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or page.effective.hierarchy.revision_id
            != receipt.after_hierarchy_revision_id
            or page.effective.hierarchy.fingerprint
            != receipt.after_hierarchy_fingerprint
            or receipt.source_parent_id
            in page.effective.hierarchy.ordered_parent_ids
        ):
            raise ValueError("worker projection is not the committed split revision")
        for index, child_parent_id in enumerate(receipt.child_parent_ids):
            projected = page.parent(child_parent_id).effective
            lineage = projected.lineage
            if (
                projected.root_id != receipt.child_root_ids[index]
                or projected.origin.value != "user"
                or projected.source_text is not None
                or projected.target_text is not None
                or projected.bundle_id is not None
                or tuple(thaw_json(projected.workflow_area_bbox))
                != receipt.child_workflow_area_bboxes[index]
                or tuple(projected.stage_requirements)
                != tuple(receipt.child_stage_requirements[index])
                or lineage is None
                or lineage.order_policy != "replace_source"
                or lineage.source_parent_id != receipt.source_parent_id
                or lineage.split_orientation != receipt.orientation.value
                or lineage.split_ordinal != index
            ):
                raise ValueError("worker projection fabricated or lost split child state")


@dataclass(frozen=True, slots=True)
class SplitUserParentEditorState:
    selection: SplitUserParentSelection
    phase: SplitUserParentEditorPhase
    draft_orientation: SplitUserParentOrientation | None = None
    draft_split_offset: int | None = None
    first_parent_id: str = ""
    first_root_id: str = ""
    second_parent_id: str = ""
    second_root_id: str = ""
    message: str = ""
    worker_command: SplitUserParentWorkerCommand | None = None
    busy_state: SplitUserParentWorkerBusyState | None = None
    receipt: SplitUserParentWorkerReceipt | None = None
    failure: SplitUserParentWorkerFailure | None = None
    cancelled: SplitUserParentCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return bool(
            self.phase is SplitUserParentEditorPhase.DIRTY
            and self.draft_orientation is not None
            and self.draft_split_offset is not None
        )

    @property
    def busy(self) -> bool:
        return self.phase is SplitUserParentEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is SplitUserParentEditorPhase.STALE

    @property
    def valid(self) -> bool:
        if not self.selection.available or not self.dirty:
            return False
        _, _, width, height = self.selection.source_workflow_area_bbox
        limit = (
            width
            if self.draft_orientation is SplitUserParentOrientation.VERTICAL
            else height
        )
        return bool(0 < int(self.draft_split_offset or 0) < limit)

    @property
    def editing_enabled(self) -> bool:
        return bool(self.selection.available and not self.busy and not self.stale)

    @property
    def split_enabled(self) -> bool:
        return bool(self.editing_enabled and self.valid and self.worker_command is None)

    @property
    def cancel_enabled(self) -> bool:
        return bool(self.editing_enabled and self.dirty)

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return bool(not self.dirty and not self.busy and not self.stale)

    @property
    def status_text(self) -> str:
        return self.message

    @property
    def status_tone(self) -> str:
        return {
            SplitUserParentEditorPhase.READY: "muted",
            SplitUserParentEditorPhase.DIRTY: "editing" if self.valid else "warning",
            SplitUserParentEditorPhase.COMMITTING: "editing",
            SplitUserParentEditorPhase.COMMITTED: "ready",
            SplitUserParentEditorPhase.CANCELLED: "muted",
            SplitUserParentEditorPhase.STALE: "warning",
            SplitUserParentEditorPhase.FAILED: "error",
            SplitUserParentEditorPhase.UNAVAILABLE: "muted",
        }[self.phase]


class SplitUserParentEditorModel:
    def __init__(self, selection: SplitUserParentSelection) -> None:
        if not isinstance(selection, SplitUserParentSelection):
            raise TypeError("selection must be SplitUserParentSelection")
        self._state = self._initial_state(selection)

    @property
    def state(self) -> SplitUserParentEditorState:
        return self._state

    def set_orientation(
        self,
        value: SplitUserParentOrientation | str | None,
    ) -> SplitUserParentEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("Split Parent draft is not editable")
        orientation = None if value is None else SplitUserParentOrientation(value)
        offset = None
        if orientation is not None:
            _, _, width, height = self._state.selection.source_workflow_area_bbox
            offset = max(
                1,
                (width if orientation is SplitUserParentOrientation.VERTICAL else height)
                // 2,
            )
        self._state = replace(
            self._state,
            phase=(
                SplitUserParentEditorPhase.READY
                if orientation is None
                else SplitUserParentEditorPhase.DIRTY
            ),
            draft_orientation=orientation,
            draft_split_offset=offset,
            message=self._draft_message(orientation, offset),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def set_split_offset(self, value: int) -> SplitUserParentEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("Split Parent draft is not editable")
        if self._state.draft_orientation is None:
            raise RuntimeError("choose a Split Parent direction first")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("split offset must be an exact integer")
        self._state = replace(
            self._state,
            phase=SplitUserParentEditorPhase.DIRTY,
            draft_split_offset=value,
            message=self._draft_message(self._state.draft_orientation, value),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> SplitUserParentEditorState:
        if self._state.busy:
            raise RuntimeError("cannot cancel Split Parent while it is committing")
        self._state = self._initial_state(
            self._state.selection,
            message="Split Parent draft discarded; project state was not changed.",
        )
        return self._state

    def begin_split(self) -> SplitUserParentWorkerCommand:
        if not self._state.split_enabled:
            raise RuntimeError("there is no valid Split Parent draft")
        first_parent_id, first_root_id = (
            (self._state.first_parent_id, self._state.first_root_id)
            if self._state.first_parent_id
            else create_user_parent_identity()
        )
        second_parent_id, second_root_id = (
            (self._state.second_parent_id, self._state.second_root_id)
            if self._state.second_parent_id
            else create_user_parent_identity()
        )
        selection = self._state.selection
        assert self._state.draft_orientation is not None
        assert self._state.draft_split_offset is not None
        command = SplitUserParentWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            source_parent_id=selection.source_parent_id,
            first_parent_id=first_parent_id,
            first_root_id=first_root_id,
            second_parent_id=second_parent_id,
            second_root_id=second_root_id,
            orientation=self._state.draft_orientation,
            split_offset=self._state.draft_split_offset,
            expected_effective_page_fingerprint=selection.effective_page_fingerprint,
            expected_hierarchy_revision_id=selection.hierarchy_revision_id,
            expected_hierarchy_fingerprint=selection.hierarchy_fingerprint,
        )
        self._state = replace(
            self._state,
            phase=SplitUserParentEditorPhase.COMMITTING,
            first_parent_id=first_parent_id,
            first_root_id=first_root_id,
            second_parent_id=second_parent_id,
            second_root_id=second_root_id,
            message="Splitting the selected user parent...",
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(
        self,
        value: SplitUserParentWorkerBusyState,
    ) -> SplitUserParentEditorState:
        self._require_active_event(value)
        self._state = replace(
            self._state,
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: SplitUserParentWorkerReceipt,
    ) -> SplitUserParentEditorState:
        if not isinstance(value, SplitUserParentWorkerReceipt):
            raise TypeError("value must be SplitUserParentWorkerReceipt")
        receipt = value.command_receipt
        command = self._require_active_identity(
            receipt.edit.page_id,
            receipt.source_parent_id,
            receipt.orientation,
            receipt.split_offset,
        )
        if (
            receipt.before_effective_page_fingerprint
            != command.expected_effective_page_fingerprint
            or receipt.before_hierarchy_revision_id
            != command.expected_hierarchy_revision_id
            or receipt.before_hierarchy_fingerprint
            != command.expected_hierarchy_fingerprint
        ):
            raise ValueError("Split Parent receipt belongs to another revision")
        self._state = replace(
            self._state,
            phase=SplitUserParentEditorPhase.COMMITTED,
            message=(
                "Split saved. Both new parents require explicit OCR, translation, "
                "cleanup, style, eligibility, layout, and rendering revisions."
            ),
            worker_command=None,
            busy_state=None,
            receipt=value,
            failure=None,
            cancelled=None,
        )
        return self._state

    def accept_failure(
        self,
        value: SplitUserParentWorkerFailure,
    ) -> SplitUserParentEditorState:
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=(
                SplitUserParentEditorPhase.STALE
                if value.stale
                else SplitUserParentEditorPhase.FAILED
            ),
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(
        self,
        value: SplitUserParentWorkerFailure,
    ) -> SplitUserParentEditorState:
        if not value.stale:
            raise ValueError("Split Parent failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: SplitUserParentCancelledReceipt,
    ) -> SplitUserParentEditorState:
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=SplitUserParentEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: SplitUserParentSelection,
    ) -> SplitUserParentEditorState:
        if self._state.busy:
            raise RuntimeError("cannot replace Split Parent selection while committing")
        same = bool(
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.source_parent_id
            == self._state.selection.source_parent_id
        )
        if same and self._state.dirty and selection.available:
            self._state = replace(self._state, selection=selection)
            return self._state
        self._state = self._initial_state(selection)
        return self._state

    def _initial_state(
        self,
        selection: SplitUserParentSelection,
        *,
        message: str | None = None,
    ) -> SplitUserParentEditorState:
        available = selection.available
        role_label = "Dialogue" if selection.source_role == "speech" else "Caption"
        return SplitUserParentEditorState(
            selection=selection,
            phase=(
                SplitUserParentEditorPhase.READY
                if available
                else SplitUserParentEditorPhase.UNAVAILABLE
            ),
            message=(
                message
                if message is not None
                else (
                    f"{role_label} user parent is eligible. Choose a split direction."
                    if available
                    else selection.unavailable_reason
                )
            ),
        )

    def _draft_message(
        self,
        orientation: SplitUserParentOrientation | None,
        offset: int | None,
    ) -> str:
        if orientation is None or offset is None:
            return "Choose Vertical (left/right) or Horizontal (top/bottom)."
        _, _, width, height = self._state.selection.source_workflow_area_bbox
        limit = width if orientation is SplitUserParentOrientation.VERTICAL else height
        if offset <= 0 or offset >= limit:
            return f"Divider must be between 1 and {limit - 1} page pixels."
        order = "left then right" if orientation is SplitUserParentOrientation.VERTICAL else "top then bottom"
        return f"Unapplied exact partition: {order}. No project state has changed."

    def _require_active_identity(
        self,
        page_id: str,
        source_parent_id: str,
        orientation: SplitUserParentOrientation,
        split_offset: int,
    ) -> SplitUserParentWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no Split Parent worker command is active")
        if (
            command.page_id != page_id
            or command.source_parent_id != source_parent_id
            or command.orientation is not SplitUserParentOrientation(orientation)
            or command.split_offset != split_offset
        ):
            raise ValueError("worker event belongs to another Split Parent command")
        return command

    def _require_active_event(
        self,
        value: (
            SplitUserParentWorkerBusyState
            | SplitUserParentCancelledReceipt
            | SplitUserParentWorkerFailure
        ),
    ) -> SplitUserParentWorkerCommand:
        command = self._require_active_identity(
            value.page_id,
            value.source_parent_id,
            value.orientation,
            value.split_offset,
        )
        project_path = getattr(value, "project_path", command.project_path)
        if os.path.normcase(os.path.abspath(project_path)) != os.path.normcase(
            os.path.abspath(command.project_path)
        ):
            raise ValueError("worker event belongs to another Split Parent project")
        return command


__all__ = [
    "AddUserParentCancellationState",
    "AddUserParentCancelledReceipt",
    "AddUserParentEditorModel",
    "AddUserParentEditorPhase",
    "AddUserParentEditorState",
    "AddUserParentSelection",
    "AddUserParentWorkerBusyState",
    "AddUserParentWorkerCommand",
    "AddUserParentWorkerFailure",
    "AddUserParentWorkerFailureCode",
    "AddUserParentWorkerReceipt",
    "AddUserParentWorkerStage",
    "ParentGeometryCancellationState",
    "ParentGeometryCancelledReceipt",
    "ParentGeometryEditorModel",
    "ParentGeometryEditorPhase",
    "ParentGeometryEditorState",
    "ParentGeometrySelection",
    "ParentGeometryWorkerBusyState",
    "ParentGeometryWorkerCommand",
    "ParentGeometryWorkerFailure",
    "ParentGeometryWorkerFailureCode",
    "ParentGeometryWorkerReceipt",
    "ParentGeometryWorkerStage",
    "ParentMembershipCancellationState",
    "ParentMembershipCancelledReceipt",
    "ParentMembershipEditorModel",
    "ParentMembershipEditorPhase",
    "ParentMembershipEditorState",
    "ParentMembershipSelection",
    "ParentMembershipWorkerBusyState",
    "ParentMembershipWorkerCommand",
    "ParentMembershipWorkerFailure",
    "ParentMembershipWorkerFailureCode",
    "ParentMembershipWorkerReceipt",
    "ParentMembershipWorkerStage",
    "RenderLayoutLineHeightCancellationState",
    "RenderLayoutLineHeightCancelledReceipt",
    "RenderLayoutLineHeightEditorModel",
    "RenderLayoutLineHeightEditorPhase",
    "RenderLayoutLineHeightEditorState",
    "RenderLayoutLineHeightSelection",
    "RenderLayoutLineHeightWorkerBusyState",
    "RenderLayoutLineHeightWorkerCommand",
    "RenderLayoutLineHeightWorkerFailure",
    "RenderLayoutLineHeightWorkerFailureCode",
    "RenderLayoutLineHeightWorkerReceipt",
    "RenderLayoutLineHeightWorkerStage",
    "RenderLayoutRotationCancellationState",
    "RenderLayoutRotationCancelledReceipt",
    "RenderLayoutRotationEditorModel",
    "RenderLayoutRotationEditorPhase",
    "RenderLayoutRotationEditorState",
    "RenderLayoutRotationSelection",
    "RenderLayoutRotationWorkerBusyState",
    "RenderLayoutRotationWorkerCommand",
    "RenderLayoutRotationWorkerFailure",
    "RenderLayoutRotationWorkerFailureCode",
    "RenderLayoutRotationWorkerReceipt",
    "RenderLayoutRotationWorkerStage",
    "RenderStyleFillColorCancellationState",
    "RenderStyleFillColorCancelledReceipt",
    "RenderStyleFillColorEditorModel",
    "RenderStyleFillColorEditorPhase",
    "RenderStyleFillColorEditorState",
    "RenderStyleFillColorSelection",
    "RenderStyleFillColorWorkerBusyState",
    "RenderStyleFillColorWorkerCommand",
    "RenderStyleFillColorWorkerFailure",
    "RenderStyleFillColorWorkerFailureCode",
    "RenderStyleFillColorWorkerReceipt",
    "RenderStyleFillColorWorkerStage",
    "RenderLayoutWritingModeCancellationState",
    "RenderLayoutWritingModeCancelledReceipt",
    "RenderLayoutWritingModeEditorModel",
    "RenderLayoutWritingModeEditorPhase",
    "RenderLayoutWritingModeEditorState",
    "RenderLayoutWritingModeSelection",
    "RenderLayoutWritingModeWorkerBusyState",
    "RenderLayoutWritingModeWorkerCommand",
    "RenderLayoutWritingModeWorkerFailure",
    "RenderLayoutWritingModeWorkerFailureCode",
    "RenderLayoutWritingModeWorkerReceipt",
    "RenderLayoutWritingModeWorkerStage",
    "ReadingOrderCancellationState",
    "ReadingOrderCancelledReceipt",
    "ReadingOrderEditorModel",
    "ReadingOrderEditorPhase",
    "ReadingOrderEditorState",
    "ReadingOrderSelection",
    "ReadingOrderWorkerBusyState",
    "ReadingOrderWorkerCommand",
    "ReadingOrderWorkerFailure",
    "ReadingOrderWorkerFailureCode",
    "ReadingOrderWorkerReceipt",
    "ReadingOrderWorkerStage",
    "MergePipelineParentCandidate",
    "MergePipelineParentsCancellationState",
    "MergePipelineParentsCancelledReceipt",
    "MergePipelineParentsEditorModel",
    "MergePipelineParentsEditorPhase",
    "MergePipelineParentsEditorState",
    "MergePipelineParentsSelection",
    "MergePipelineParentsWorkerBusyState",
    "MergePipelineParentsWorkerCommand",
    "MergePipelineParentsWorkerFailure",
    "MergePipelineParentsWorkerFailureCode",
    "MergePipelineParentsWorkerReceipt",
    "MergePipelineParentsWorkerStage",
    "SplitUserParentCancellationState",
    "SplitUserParentCancelledReceipt",
    "SplitUserParentEditorModel",
    "SplitUserParentEditorPhase",
    "SplitUserParentEditorState",
    "SplitUserParentSelection",
    "SplitUserParentWorkerBusyState",
    "SplitUserParentWorkerCommand",
    "SplitUserParentWorkerFailure",
    "SplitUserParentWorkerFailureCode",
    "SplitUserParentWorkerReceipt",
    "SplitUserParentWorkerStage",
    "SourceTextCancellationState",
    "SourceTextCancelledReceipt",
    "SourceTextEditorModel",
    "SourceTextEditorPhase",
    "SourceTextEditorState",
    "SourceTextSelection",
    "SourceTextWorkerBusyState",
    "SourceTextWorkerCommand",
    "SourceTextWorkerFailure",
    "SourceTextWorkerFailureCode",
    "SourceTextWorkerReceipt",
    "SourceTextWorkerStage",
    "TargetTextCancellationState",
    "TargetTextCancelledReceipt",
    "TargetTextEditorModel",
    "TargetTextEditorPhase",
    "TargetTextEditorState",
    "TargetTextSelection",
    "TargetTextWorkerBusyState",
    "TargetTextWorkerCommand",
    "TargetTextWorkerFailure",
    "TargetTextWorkerFailureCode",
    "TargetTextWorkerReceipt",
    "TargetTextWorkerStage",
]
