# -*- coding: utf-8 -*-
"""Framework-neutral UI state for one explicit parent OCR revision.

The model keeps four provenance lanes separate:

* Automatic source belongs to immutable automatic-parent evidence.
* A user-triggered OCR result is a selected model revision.
* A manual source replacement remains a user edit.
* Effective source is the deterministic projected value.

The explicit OCR result must never be relabelled as either Automatic source or
Your edit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import Any, Mapping, TYPE_CHECKING
import uuid

from app.config.settings_contracts import RunSettingsSnapshot
from app.pipeline.hierarchy_revision_contracts import (
    ParentStageRequirement,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
    validate_user_parent_identity_pair,
)
from app.pipeline.ocr_revision_contracts import (
    ExplicitOcrRevisionReceipt,
    OriginalPageAssetBinding,
    SUPPORTED_OCR_ENGINES,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _required_identity(value: Any, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} is required")
    return candidate


def _required_path_safe_identity(value: Any, field_name: str) -> str:
    candidate = _required_identity(value, field_name)
    if _PATH_SAFE_ID.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be path-safe")
    return candidate


def _required_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "").lower()
    if _SHA256.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return candidate


def _require_exact_bbox(
    value: Any,
    *,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError("sampling_bbox must contain four exact integers")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("sampling_bbox must be a positive page-bounded rectangle")
    if x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise ValueError("sampling_bbox must remain inside the original page")
    return value


def _stage_requirement(
    requirements: tuple[ParentStageRequirement, ...],
    stage: RevisionStage,
) -> ParentStageRequirement:
    matches = tuple(value for value in requirements if value.stage is stage)
    if len(matches) != 1:
        raise ValueError(f"selection requires one exact {stage.value} requirement")
    return matches[0]


class OcrRevisionPhase(str, Enum):
    READY = "ready"
    RUNNING = "running"
    CANCEL_DEFERRED = "cancel_deferred"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class OcrRevisionWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    VALIDATING_SELECTION = "validating_selection"
    PREPARING_REQUEST = "preparing_request"
    INITIALIZING_OWNER = "initializing_owner"
    RECOGNIZING = "recognizing"
    DISCARDING_CANCELLED_RESULT = "discarding_cancelled_result"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class OcrRevisionCancellationMode(str, Enum):
    AVAILABLE = "available"
    REQUESTED_DEFERRED = "requested_deferred"
    LOCKED = "locked"
    UNAVAILABLE = "unavailable"


class OcrRevisionFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SNAPSHOT_STALE = "snapshot_stale"
    SETTINGS_STALE = "settings_stale"
    ORIGINAL_ASSET_UNAVAILABLE = "original_asset_unavailable"
    ORIGINAL_ASSET_MISMATCH = "original_asset_mismatch"
    SOURCE_NOT_RUNNABLE = "source_not_runnable"
    RECOGNITION_FAILED = "recognition_failed"
    NON_AUTHORITATIVE_RESULT = "non_authoritative_result"
    EMPTY_RESULT = "empty_result"
    PERSISTENCE_REJECTED = "persistence_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    COMMITTED_STALE = "committed_stale"
    WORKER_REUSED = "worker_reused"
    COMMAND_REJECTED = "command_rejected"


@dataclass(frozen=True, slots=True)
class OcrRevisionSelection:
    """Exact selected-parent state from which one OCR revision may start."""

    project_path: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    original_page: OriginalPageAssetBinding
    sampling_bbox: tuple[int, int, int, int]
    run_settings_snapshot: RunSettingsSnapshot
    selected_ocr_engine: str
    stage_requirements: tuple[ParentStageRequirement, ...]
    model_source_revision_id: str | None = None
    model_source_text: str | None = None
    model_source_engine: str | None = None
    user_source_text: str | None = None
    effective_source_text: str | None = None
    effective_source_authority: str = "unavailable"
    available: bool = True
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "hierarchy_revision_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        for field_name in (
            "effective_page_fingerprint",
            "hierarchy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if not isinstance(self.original_page, OriginalPageAssetBinding):
            raise TypeError("original_page must be OriginalPageAssetBinding")
        object.__setattr__(
            self,
            "sampling_bbox",
            _require_exact_bbox(
                self.sampling_bbox,
                canvas_size=self.original_page.canvas_size,
            ),
        )
        if not isinstance(self.run_settings_snapshot, RunSettingsSnapshot):
            raise TypeError("run_settings_snapshot must be RunSettingsSnapshot")
        if self.run_settings_snapshot.project_id != self.project_id:
            raise ValueError("run settings belong to another project")
        selected = _required_identity(self.selected_ocr_engine, "selected_ocr_engine")
        if selected not in SUPPORTED_OCR_ENGINES:
            raise ValueError("selected_ocr_engine is unsupported")
        snapshot_selected = str(
            self.run_settings_snapshot.pipeline_values.get("ocr_engine") or ""
        )
        if snapshot_selected != selected:
            raise ValueError("selected OCR engine differs from the run snapshot")
        object.__setattr__(self, "selected_ocr_engine", selected)
        requirements = tuple(self.stage_requirements)
        if any(not isinstance(value, ParentStageRequirement) for value in requirements):
            raise TypeError("stage_requirements must contain ParentStageRequirement")
        if any(value.parent_id != self.parent_id for value in requirements):
            raise ValueError("stage requirements belong to another parent")
        _stage_requirement(requirements, RevisionStage.SOURCE)
        _stage_requirement(requirements, RevisionStage.TRANSLATION)
        object.__setattr__(self, "stage_requirements", requirements)

        optional_ids = ("model_source_revision_id", "model_source_engine")
        for field_name in optional_ids:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required_identity(value, field_name),
                )
        for field_name in (
            "model_source_text",
            "user_source_text",
            "effective_source_text",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if (self.model_source_revision_id is None) != (self.model_source_text is None):
            raise ValueError("model source revision identity and text must travel together")
        if self.model_source_revision_id is None and self.model_source_engine is not None:
            raise ValueError("model source engine requires a selected model revision")
        if self.model_source_revision_id is not None:
            if not self.model_source_text or not self.model_source_text.strip():
                raise ValueError("selected model OCR revision must contain source text")
            if self.model_source_engine is None:
                raise ValueError("selected model OCR revision requires engine provenance")
        if self.effective_source_authority not in {"model", "user", "unavailable"}:
            raise ValueError(
                "effective_source_authority must be model, user, or unavailable"
            )
        if self.effective_source_authority == "unavailable":
            if self.effective_source_text is not None:
                raise ValueError("unavailable effective source cannot carry text")
        elif self.effective_source_authority == "model":
            if self.model_source_text is None:
                raise ValueError("model source authority requires a model revision")
            if self.effective_source_text != self.model_source_text:
                raise ValueError("effective model source must equal the selected revision")
        else:
            if self.user_source_text is None:
                raise ValueError("user source authority requires a user edit")
            if self.effective_source_text != self.user_source_text:
                raise ValueError("effective user source must equal the user edit")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        if self.available and reason:
            raise ValueError("available OCR revision state cannot have a reason")
        if not self.available and not reason:
            raise ValueError("unavailable OCR revision state requires a reason")
        object.__setattr__(self, "unavailable_reason", reason)

    @property
    def run_settings_fingerprint(self) -> str:
        return self.run_settings_snapshot.settings_fingerprint

    @property
    def source_requirement(self) -> ParentStageRequirement:
        return _stage_requirement(self.stage_requirements, RevisionStage.SOURCE)

    @property
    def translation_requirement(self) -> ParentStageRequirement:
        return _stage_requirement(self.stage_requirements, RevisionStage.TRANSLATION)

    @property
    def source_current(self) -> bool:
        value = self.source_requirement
        return bool(
            value.state is RevisionStageState.CURRENT
            and value.required_action is RevisionRequiredAction.NONE
        )

    @property
    def source_runnable(self) -> bool:
        value = self.source_requirement
        return bool(
            value.state is RevisionStageState.MISSING
            and value.required_action is RevisionRequiredAction.EXPLICIT_RUN
        )

    @property
    def translation_required(self) -> bool:
        value = self.translation_requirement
        return not (
            value.state is RevisionStageState.CURRENT
            and value.required_action is RevisionRequiredAction.NONE
        )


@dataclass(frozen=True, slots=True)
class OcrRevisionCommandIdentity:
    operation_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    run_settings_fingerprint: str
    selected_ocr_engine: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_path_safe_identity(self.operation_id, "operation_id"),
        )
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "hierarchy_revision_id",
            "selected_ocr_engine",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        for field_name in (
            "effective_page_fingerprint",
            "hierarchy_fingerprint",
            "run_settings_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )

    @classmethod
    def from_selection(
        cls,
        selection: OcrRevisionSelection,
        *,
        operation_id: str | None = None,
    ) -> "OcrRevisionCommandIdentity":
        return cls(
            operation_id=operation_id or uuid.uuid4().hex,
            project_id=selection.project_id,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            root_id=selection.root_id,
            parent_authored_edit_id=selection.parent_authored_edit_id,
            effective_page_fingerprint=selection.effective_page_fingerprint,
            hierarchy_revision_id=selection.hierarchy_revision_id,
            hierarchy_fingerprint=selection.hierarchy_fingerprint,
            run_settings_fingerprint=selection.run_settings_fingerprint,
            selected_ocr_engine=selection.selected_ocr_engine,
        )


@dataclass(frozen=True, slots=True)
class OcrRevisionWorkerCommand:
    identity: OcrRevisionCommandIdentity
    selection: OcrRevisionSelection

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OcrRevisionCommandIdentity):
            raise TypeError("identity must be OcrRevisionCommandIdentity")
        if not isinstance(self.selection, OcrRevisionSelection):
            raise TypeError("selection must be OcrRevisionSelection")
        expected = OcrRevisionCommandIdentity.from_selection(
            self.selection,
            operation_id=self.identity.operation_id,
        )
        if self.identity != expected:
            raise ValueError("worker command identity differs from its selection")


@dataclass(frozen=True, slots=True)
class OcrRevisionWorkerBusyState:
    identity: OcrRevisionCommandIdentity
    busy: bool
    stage: OcrRevisionWorkerStage
    cancellation_mode: OcrRevisionCancellationMode
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class OcrRevisionCancellationState:
    identity: OcrRevisionCommandIdentity
    mode: OcrRevisionCancellationMode
    stage: OcrRevisionWorkerStage
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class OcrRevisionCancelledReceipt:
    identity: OcrRevisionCommandIdentity
    stage: OcrRevisionWorkerStage
    inference_completed: bool
    message: str = (
        "OCR revision cancelled; no source revision or selection edit was published."
    )


@dataclass(frozen=True, slots=True)
class OcrRevisionWorkerFailure:
    identity: OcrRevisionCommandIdentity
    code: OcrRevisionFailureCode
    stage: OcrRevisionWorkerStage
    message: str
    exception_type: str = ""
    persistence_committed: bool = False
    core_receipt: ExplicitOcrRevisionReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code
            in {
                OcrRevisionFailureCode.SNAPSHOT_STALE,
                OcrRevisionFailureCode.SETTINGS_STALE,
                OcrRevisionFailureCode.POST_COMMIT_PROJECTION_FAILED,
                OcrRevisionFailureCode.COMMITTED_STALE,
            }
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class OcrRevisionWorkerReceipt:
    """Identity-bound, atomic shell refresh after a committed OCR revision."""

    identity: OcrRevisionCommandIdentity
    core_receipt: ExplicitOcrRevisionReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"
    selection: OcrRevisionSelection

    def __post_init__(self) -> None:
        if not isinstance(self.identity, OcrRevisionCommandIdentity):
            raise TypeError("identity must be OcrRevisionCommandIdentity")
        if not isinstance(self.core_receipt, ExplicitOcrRevisionReceipt):
            raise TypeError("core_receipt must be ExplicitOcrRevisionReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(self.selection, OcrRevisionSelection):
            raise TypeError("selection must be OcrRevisionSelection")

        from app.project_edits.fingerprints import canonical_sha256
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project mapping does not match its projection")
        receipt = self.core_receipt
        if (
            receipt.command_id != self.identity.operation_id
            or receipt.project_id != self.identity.project_id
            or receipt.page_id != self.identity.page_id
            or receipt.parent_id != self.identity.parent_id
            or receipt.root_id != self.identity.root_id
            or receipt.parent_authored_edit_id
            != self.identity.parent_authored_edit_id
            or receipt.before_effective_page_fingerprint
            != self.identity.effective_page_fingerprint
            or receipt.hierarchy_revision_id != self.identity.hierarchy_revision_id
            or receipt.hierarchy_fingerprint != self.identity.hierarchy_fingerprint
            or receipt.run_settings_fingerprint
            != self.identity.run_settings_fingerprint
            or receipt.selected_ocr_engine != self.identity.selected_ocr_engine
            or receipt.original_page != self.selection.original_page
            or receipt.sampling_bbox != self.selection.sampling_bbox
        ):
            raise ValueError("OCR receipt belongs to another selected-parent command")
        after = self.selection
        if (
            after.project_id != receipt.project_id
            or after.page_id != receipt.page_id
            or after.parent_id != receipt.parent_id
            or after.root_id != self.identity.root_id
            or after.parent_authored_edit_id != self.identity.parent_authored_edit_id
            or after.effective_page_fingerprint
            != receipt.after_effective_page_fingerprint
            or after.hierarchy_revision_id != receipt.hierarchy_revision_id
            or after.hierarchy_fingerprint != receipt.hierarchy_fingerprint
            or after.run_settings_fingerprint != receipt.run_settings_fingerprint
            or after.model_source_revision_id != receipt.source_revision_id
            or after.model_source_text != receipt.source_text
            or after.model_source_engine != receipt.selected_ocr_engine
            or not after.source_current
            or not after.translation_required
        ):
            raise ValueError("worker projection is not the committed OCR revision")
        if self.projection.metadata.project_id != receipt.project_id:
            raise ValueError("worker projection belongs to another project")


@dataclass(frozen=True, slots=True)
class OcrRevisionState:
    selection: OcrRevisionSelection
    phase: OcrRevisionPhase
    message: str
    command: OcrRevisionWorkerCommand | None = None
    busy_state: OcrRevisionWorkerBusyState | None = None
    cancellation_state: OcrRevisionCancellationState | None = None
    receipt: OcrRevisionWorkerReceipt | None = None
    failure: OcrRevisionWorkerFailure | None = None
    cancelled: OcrRevisionCancelledReceipt | None = None

    @property
    def busy(self) -> bool:
        return self.phase in {
            OcrRevisionPhase.RUNNING,
            OcrRevisionPhase.CANCEL_DEFERRED,
        }

    @property
    def stale(self) -> bool:
        return self.phase is OcrRevisionPhase.STALE

    @property
    def rerun_enabled(self) -> bool:
        return bool(
            self.selection.available
            and self.selection.source_runnable
            and not self.busy
            and not self.stale
            and self.command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return bool(
            self.busy
            and self.cancellation_state is not None
            and self.cancellation_state.mode is OcrRevisionCancellationMode.AVAILABLE
        )

    @property
    def deferred_cancel(self) -> bool:
        return self.phase is OcrRevisionPhase.CANCEL_DEFERRED

    @property
    def blocks_navigation(self) -> bool:
        return bool(self.busy or self.stale)

    @property
    def blocks_start_preview(self) -> bool:
        # This slice publishes only source. Translation and later owner
        # revisions still prevent execution even after OCR succeeds.
        return True

    @property
    def status_tone(self) -> str:
        return {
            OcrRevisionPhase.READY: "warning",
            OcrRevisionPhase.RUNNING: "editing",
            OcrRevisionPhase.CANCEL_DEFERRED: "warning",
            OcrRevisionPhase.SUCCEEDED: "ready",
            OcrRevisionPhase.CANCELLED: "muted",
            OcrRevisionPhase.STALE: "warning",
            OcrRevisionPhase.FAILED: "error",
            OcrRevisionPhase.UNAVAILABLE: "warning",
        }[self.phase]


class OcrRevisionModel:
    """UI-thread reducer for one explicit selected model OCR revision."""

    def __init__(self, selection: OcrRevisionSelection) -> None:
        if not isinstance(selection, OcrRevisionSelection):
            raise TypeError("selection must be OcrRevisionSelection")
        phase = (
            OcrRevisionPhase.READY
            if selection.available
            else OcrRevisionPhase.UNAVAILABLE
        )
        self._state = OcrRevisionState(
            selection=selection,
            phase=phase,
            message=(
                self._ready_message(selection)
                if selection.available
                else selection.unavailable_reason
            ),
        )

    @property
    def state(self) -> OcrRevisionState:
        return self._state

    def begin(self, operation_id: str | None = None) -> OcrRevisionWorkerCommand:
        if not self._state.rerun_enabled:
            raise RuntimeError("Rerun OCR is not available for this selection")
        identity = OcrRevisionCommandIdentity.from_selection(
            self._state.selection,
            operation_id=operation_id,
        )
        command = OcrRevisionWorkerCommand(
            identity=identity,
            selection=self._state.selection,
        )
        cancellation = OcrRevisionCancellationState(
            identity=identity,
            mode=OcrRevisionCancellationMode.AVAILABLE,
            stage=OcrRevisionWorkerStage.LOADING_PROJECT,
            persistence_started=False,
            message="OCR can be cancelled before persistence.",
        )
        self._state = replace(
            self._state,
            phase=OcrRevisionPhase.RUNNING,
            message="Preparing the selected model OCR revision...",
            command=command,
            busy_state=None,
            cancellation_state=cancellation,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(self, value: OcrRevisionWorkerBusyState) -> OcrRevisionState:
        if not isinstance(value, OcrRevisionWorkerBusyState):
            raise TypeError("value must be OcrRevisionWorkerBusyState")
        self._require_identity(value.identity)
        deferred_cancel = self._state.phase is OcrRevisionPhase.CANCEL_DEFERRED
        phase = self._state.phase
        if value.busy:
            phase = (
                OcrRevisionPhase.CANCEL_DEFERRED
                if deferred_cancel
                or value.cancellation_mode
                is OcrRevisionCancellationMode.REQUESTED_DEFERRED
                else OcrRevisionPhase.RUNNING
            )
        self._state = replace(
            self._state,
            phase=phase,
            message=(
                self._state.message
                if deferred_cancel
                else value.message or self._state.message
            ),
            busy_state=value,
        )
        return self._state

    def accept_cancellation(
        self,
        value: OcrRevisionCancellationState,
    ) -> OcrRevisionState:
        if not isinstance(value, OcrRevisionCancellationState):
            raise TypeError("value must be OcrRevisionCancellationState")
        self._require_identity(value.identity)
        deferred_cancel = self._state.phase is OcrRevisionPhase.CANCEL_DEFERRED
        cancellation_state = value
        if deferred_cancel and self._state.cancellation_state is not None:
            progress = {
                OcrRevisionCancellationMode.AVAILABLE: 0,
                OcrRevisionCancellationMode.REQUESTED_DEFERRED: 1,
                OcrRevisionCancellationMode.LOCKED: 2,
                OcrRevisionCancellationMode.UNAVAILABLE: 3,
            }
            if progress[value.mode] < progress[self._state.cancellation_state.mode]:
                cancellation_state = self._state.cancellation_state
        phase = (
            OcrRevisionPhase.CANCEL_DEFERRED
            if deferred_cancel
            or value.mode is OcrRevisionCancellationMode.REQUESTED_DEFERRED
            else self._state.phase
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=(
                self._state.message
                if deferred_cancel
                else value.message or self._state.message
            ),
            cancellation_state=cancellation_state,
        )
        return self._state

    def mark_cancel_requested(self) -> OcrRevisionState:
        command = self._require_command()
        stage = (
            self._state.busy_state.stage
            if self._state.busy_state is not None
            else OcrRevisionWorkerStage.LOADING_PROJECT
        )
        cancellation = OcrRevisionCancellationState(
            identity=command.identity,
            mode=OcrRevisionCancellationMode.REQUESTED_DEFERRED,
            stage=stage,
            persistence_started=False,
            message=(
                "Cancellation requested. If OCR inference is active, it will finish "
                "before its result is discarded; nothing will be published."
            ),
        )
        return self.accept_cancellation(cancellation)

    def accept_receipt(self, value: OcrRevisionWorkerReceipt) -> OcrRevisionState:
        if not isinstance(value, OcrRevisionWorkerReceipt):
            raise TypeError("value must be OcrRevisionWorkerReceipt")
        self._require_identity(value.identity)
        self._state = OcrRevisionState(
            selection=value.selection,
            phase=OcrRevisionPhase.SUCCEEDED,
            message=(
                "Selected model OCR revision is current. Translation is required "
                "before Start or Preview can continue."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(self, value: OcrRevisionWorkerFailure) -> OcrRevisionState:
        if not isinstance(value, OcrRevisionWorkerFailure):
            raise TypeError("value must be OcrRevisionWorkerFailure")
        self._require_identity(value.identity)
        self._state = replace(
            self._state,
            phase=(OcrRevisionPhase.STALE if value.stale else OcrRevisionPhase.FAILED),
            message=value.message,
            command=None,
            busy_state=None,
            cancellation_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_cancelled(
        self,
        value: OcrRevisionCancelledReceipt,
    ) -> OcrRevisionState:
        if not isinstance(value, OcrRevisionCancelledReceipt):
            raise TypeError("value must be OcrRevisionCancelledReceipt")
        self._require_identity(value.identity)
        self._state = replace(
            self._state,
            phase=OcrRevisionPhase.CANCELLED,
            message=value.message,
            command=None,
            busy_state=None,
            cancellation_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: OcrRevisionSelection) -> OcrRevisionState:
        if not isinstance(selection, OcrRevisionSelection):
            raise TypeError("selection must be OcrRevisionSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace the OCR selection while it is active")
        phase = (
            OcrRevisionPhase.READY
            if selection.available
            else OcrRevisionPhase.UNAVAILABLE
        )
        self._state = OcrRevisionState(
            selection=selection,
            phase=phase,
            message=(
                self._ready_message(selection)
                if selection.available
                else selection.unavailable_reason
            ),
        )
        return self._state

    def _require_command(self) -> OcrRevisionWorkerCommand:
        command = self._state.command
        if command is None:
            raise RuntimeError("no OCR revision command is active")
        return command

    def _require_identity(
        self,
        identity: OcrRevisionCommandIdentity,
    ) -> OcrRevisionWorkerCommand:
        if not isinstance(identity, OcrRevisionCommandIdentity):
            raise TypeError("event identity must be OcrRevisionCommandIdentity")
        command = self._require_command()
        if command.identity != identity:
            raise ValueError("worker event belongs to another OCR revision command")
        return command

    @staticmethod
    def _ready_message(selection: OcrRevisionSelection) -> str:
        if selection.source_current:
            if not selection.translation_required:
                return (
                    "Selected model OCR revision and translation revision are current. "
                    "Later owner revisions remain explicit."
                )
            return (
                "A selected model OCR revision is current. Translation is now the "
                "next required owner; this source-only command is complete."
            )
        return (
            "Source is missing or stale. Rerun OCR will sample the exact workflow "
            "area with the selected OCR engine."
        )


def ocr_revision_selection_from_projection(
    projection: "ProjectUiProjection",
    *,
    page_id: str,
    parent_id: str,
    run_settings_snapshot: RunSettingsSnapshot,
) -> OcrRevisionSelection:
    """Build the exact UI selection from one immutable background projection."""

    from app.pipeline.hierarchy_revision_contracts import ParentOrigin
    from app.ui.shell.project_projection import ProjectUiProjection

    if not isinstance(projection, ProjectUiProjection):
        raise TypeError("projection must be ProjectUiProjection")
    if not isinstance(run_settings_snapshot, RunSettingsSnapshot):
        raise TypeError("run_settings_snapshot must be RunSettingsSnapshot")
    page = projection.page(page_id)
    parent = page.parent(parent_id)
    effective = parent.effective
    lineage = effective.lineage
    if effective.origin is not ParentOrigin.USER or lineage is None:
        raise ValueError("explicit OCR revision requires one user-authored parent")
    original = page.original_page_binding
    if original is None:
        raise ValueError(
            page.original_page_binding_problem
            or "The original page is unavailable for OCR."
        )
    selected_engine = str(
        run_settings_snapshot.pipeline_values.get("ocr_engine") or ""
    )
    artifact = parent.selected_model_source_revision
    if artifact is not None:
        model_revision_id = artifact.revision_id
        model_source_text = artifact.source_text
        model_source_engine = artifact.selected_ocr_engine
    else:
        model_revision_id = None
        model_source_text = None
        model_source_engine = None
    authority = {
        "unavailable": "unavailable",
        "ocr_revision": "model",
        "user": "user",
    }.get(effective.source_authority)
    if authority is None:
        raise ValueError("projected user-parent source authority is unsupported")
    source_requirement = _stage_requirement(
        tuple(effective.stage_requirements),
        RevisionStage.SOURCE,
    )
    source_runnable = bool(
        source_requirement.state is RevisionStageState.MISSING
        and source_requirement.required_action is RevisionRequiredAction.EXPLICIT_RUN
    )
    translation_requirement = _stage_requirement(
        tuple(effective.stage_requirements),
        RevisionStage.TRANSLATION,
    )
    translation_required = not (
        translation_requirement.state is RevisionStageState.CURRENT
        and translation_requirement.required_action is RevisionRequiredAction.NONE
    )
    if run_settings_snapshot.unresolved_requirements:
        available = False
        reason = (
            "Resolve the current run settings before Rerun OCR: "
            + "; ".join(run_settings_snapshot.unresolved_requirements)
        )
    elif not source_runnable:
        available = False
        reason = (
            (
                "Selected model OCR revision is current. Translation is the next "
                "required owner."
                if translation_required
                else (
                    "Selected model OCR revision and translation revision are "
                    "current. Later owner revisions remain explicit."
                )
            )
            if artifact is not None
            else "The selected parent does not require an explicit OCR revision."
        )
    else:
        available = True
        reason = ""
    return OcrRevisionSelection(
        project_path=projection.metadata.project_path,
        project_id=projection.metadata.project_id,
        page_id=page.effective.page_id,
        parent_id=effective.parent_id,
        root_id=effective.root_id,
        parent_authored_edit_id=lineage.authored_edit_id,
        effective_page_fingerprint=page.effective.effective_fingerprint,
        hierarchy_revision_id=page.effective.hierarchy.revision_id,
        hierarchy_fingerprint=page.effective.hierarchy.fingerprint,
        original_page=original,
        sampling_bbox=tuple(lineage.workflow_area_bbox),
        run_settings_snapshot=run_settings_snapshot,
        selected_ocr_engine=selected_engine,
        stage_requirements=tuple(effective.stage_requirements),
        model_source_revision_id=model_revision_id,
        model_source_text=model_source_text,
        model_source_engine=model_source_engine,
        user_source_text=parent.user_source_text,
        effective_source_text=effective.source_text,
        effective_source_authority=authority,
        available=available,
        unavailable_reason=reason,
    )


__all__ = [
    "OcrRevisionCancellationMode",
    "OcrRevisionCancellationState",
    "OcrRevisionCancelledReceipt",
    "OcrRevisionCommandIdentity",
    "OcrRevisionFailureCode",
    "OcrRevisionModel",
    "OcrRevisionPhase",
    "OcrRevisionSelection",
    "OcrRevisionState",
    "OcrRevisionWorkerBusyState",
    "OcrRevisionWorkerCommand",
    "OcrRevisionWorkerFailure",
    "OcrRevisionWorkerReceipt",
    "OcrRevisionWorkerStage",
    "ocr_revision_selection_from_projection",
]
