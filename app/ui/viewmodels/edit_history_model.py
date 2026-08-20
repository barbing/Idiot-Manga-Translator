# -*- coding: utf-8 -*-
"""Framework-neutral state for durable selected-page edit history.

The model exposes append-only revoke/reapply intent only.  It never carries an
edit payload, clones a prior edit, or treats Qt-local history as durable state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import TYPE_CHECKING, Any, Mapping

from app.project_edits.commands import (
    EditHistoryCommandErrorCode,
    EditHistoryCommandReceipt,
    EditHistoryOperation,
)
from app.project_edits.contracts import EditDomain, EditTargetKind
from app.project_edits.projection import ProjectionIssueKind

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


_STALE_COMMAND_CODES = frozenset(
    {
        EditHistoryCommandErrorCode.STALE_EFFECTIVE_PAGE,
        EditHistoryCommandErrorCode.STALE_PAGE_HEAD,
        EditHistoryCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)


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


class EditHistoryEditorPhase(str, Enum):
    READY = "ready"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class EditHistoryWorkerStage(str, Enum):
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


class EditHistoryWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    TARGET_EDIT_NOT_FOUND = "target_edit_not_found"
    TARGET_EDIT_PAGE_MISMATCH = "target_edit_page_mismatch"
    TARGET_FORBIDDEN = "target_forbidden"
    ALREADY_ACTIVE = "already_active"
    ALREADY_REVOKED = "already_revoked"
    ACTIVE_DEPENDENT_EDIT = "active_dependent_edit"
    SNAPSHOT_STALE = "snapshot_stale"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


@dataclass(frozen=True, slots=True)
class EditHistorySelection:
    """Safe history evidence for one persisted record on the selected page."""

    project_path: str
    page_id: str
    target_edit_id: str
    domain: EditDomain
    operation: str
    target_kind: EditTargetKind
    parent_id: str
    active: bool
    effective: bool
    is_control: bool
    issue_kinds: tuple[ProjectionIssueKind, ...]
    effective_page_fingerprint: str
    field_name: str = ""

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
            "target_edit_id",
            _required_identity(self.target_edit_id, "target_edit_id"),
        )
        object.__setattr__(self, "domain", EditDomain(self.domain))
        object.__setattr__(
            self,
            "operation",
            _required_identity(self.operation, "operation"),
        )
        target_kind = EditTargetKind(self.target_kind)
        object.__setattr__(self, "target_kind", target_kind)
        if not isinstance(self.parent_id, str):
            raise TypeError("parent_id must be a string")
        parent_id = self.parent_id.strip()
        if target_kind is EditTargetKind.PARENT:
            parent_id = _required_identity(parent_id, "parent_id")
        object.__setattr__(self, "parent_id", parent_id)
        for field_name in ("active", "effective", "is_control"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if self.effective and not self.active:
            raise ValueError("an effective history edit must be active")
        expected_control = self.domain is EditDomain.LEDGER_CONTROL
        if self.is_control != expected_control:
            raise ValueError("history control flag and domain disagree")
        if self.is_control and (
            self.target_kind is not EditTargetKind.EDIT
            or self.active
            or self.effective
        ):
            raise ValueError("ledger-control history must remain read-only")
        if not isinstance(self.issue_kinds, tuple):
            raise TypeError("issue_kinds must be a tuple")
        issue_kinds = tuple(ProjectionIssueKind(value) for value in self.issue_kinds)
        if len(issue_kinds) != len(set(issue_kinds)):
            raise ValueError("issue_kinds must not contain duplicates")
        object.__setattr__(self, "issue_kinds", issue_kinds)
        object.__setattr__(
            self,
            "effective_page_fingerprint",
            _required_sha256(
                self.effective_page_fingerprint,
                "effective_page_fingerprint",
            ),
        )
        if not isinstance(self.field_name, str):
            raise TypeError("field_name must be a string")
        object.__setattr__(self, "field_name", self.field_name.strip())

    @property
    def eligible_operation(self) -> EditHistoryOperation | None:
        """Return the sole durable action supported by exact ledger state."""

        if (
            self.is_control
            or self.domain is EditDomain.LEDGER_CONTROL
            or self.target_kind in {EditTargetKind.ARTIFACT, EditTargetKind.EDIT}
        ):
            return None
        if self.domain is EditDomain.STRUCTURAL:
            structural_target_supported = bool(
                (
                    self.operation
                    in {
                        "set_geometry",
                        "add_user_parent",
                        "split_user_parent",
                        "merge_pipeline_parents",
                    }
                    and self.target_kind is EditTargetKind.PARENT
                )
                or (
                    self.operation == "set_reading_order"
                    and self.target_kind is EditTargetKind.PAGE
                )
            )
            if not structural_target_supported:
                return None
        if self.domain is EditDomain.CLEANUP and self.target_kind is not EditTargetKind.PAGE:
            return None
        if self.domain is EditDomain.GLOSSARY and (
            self.target_kind is not EditTargetKind.PROJECT
            or self.operation not in {"set_entry", "remove_entry"}
        ):
            return None
        if (
            self.domain is EditDomain.RENDER_LAYOUT
            and self.field_name
            not in {"writing_mode", "line_height", "rotation", "render_box"}
        ):
            return None
        if (
            self.domain is EditDomain.RENDER_STYLE
            and self.field_name not in {
                "fill_color",
                "font_role",
                "font_weight_tier",
                "outline_color",
                "outline_width",
                "preferred_size",
                "shadow_blur",
                "shadow_color",
                "shadow_offset",
                "shadow_enabled",
            }
        ):
            return None
        if self.domain not in {
            EditDomain.STRUCTURAL,
            EditDomain.SOURCE_TEXT,
            EditDomain.TARGET_TEXT,
            EditDomain.CLEANUP,
            EditDomain.RENDER_STYLE,
            EditDomain.RENDER_LAYOUT,
            EditDomain.REVIEW_METADATA,
            EditDomain.GLOSSARY,
        }:
            return None
        return (
            EditHistoryOperation.REVOKE
            if self.active
            else EditHistoryOperation.REAPPLY
        )

    @property
    def actionable(self) -> bool:
        return self.eligible_operation is not None

    @property
    def ineligibility_reason(self) -> str:
        if self.eligible_operation is not None:
            return ""
        if self.is_control or self.target_kind is EditTargetKind.EDIT:
            return "History control records are read-only evidence."
        if self.target_kind is EditTargetKind.ARTIFACT:
            return "Artifact revision history is read-only evidence."
        if self.domain is EditDomain.STRUCTURAL:
            return (
                "This structural edit cannot be reversed until all exact "
                "invalidation facts are available."
            )
        if self.domain is EditDomain.RENDER_STYLE:
            return (
                "Only registered font-role and opaque fill-color render-style "
                "history is actionable in this slice."
            )
        if self.domain is EditDomain.RENDER_LAYOUT:
            return (
                "Only writing-mode, line-height, rotation, and render-box layout "
                "history are actionable in this slice."
            )
        return "This history record has no supported durable action."


@dataclass(frozen=True, slots=True)
class EditHistoryWorkerCommand:
    """UI carrier with no persistence-owned command identity or CAS heads."""

    project_path: str
    page_id: str
    target_edit_id: str
    operation: EditHistoryOperation
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
            "target_edit_id",
            _required_identity(self.target_edit_id, "target_edit_id"),
        )
        object.__setattr__(self, "operation", EditHistoryOperation(self.operation))
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class EditHistoryWorkerBusyState:
    page_id: str
    target_edit_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: EditHistoryWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class EditHistoryCancellationState:
    page_id: str
    target_edit_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class EditHistoryCancelledReceipt:
    project_path: str
    page_id: str
    target_edit_id: str
    operation: EditHistoryOperation
    stage: EditHistoryWorkerStage
    message: str = "History action cancelled before persistence."


@dataclass(frozen=True, slots=True)
class EditHistoryWorkerFailure:
    code: EditHistoryWorkerFailureCode
    stage: EditHistoryWorkerStage
    project_path: str
    page_id: str
    target_edit_id: str
    operation: EditHistoryOperation
    message: str
    exception_type: str = ""
    command_error_code: EditHistoryCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: EditHistoryCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is EditHistoryWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class EditHistoryWorkerReceipt:
    """Atomic, project-fingerprint-bound history refresh payload."""

    command_receipt: EditHistoryCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(self.command_receipt, EditHistoryCommandReceipt):
            raise TypeError("command_receipt must be EditHistoryCommandReceipt")
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
        target = receipt.target_edit
        control = receipt.control_edit
        if self.projection.metadata.project_id != target.project_id:
            raise ValueError("worker projection belongs to another project")
        if control.project_id != target.project_id or control.page_id != target.page_id:
            raise ValueError("history control and target identities disagree")
        if control.target.edit_id != target.edit_id:
            raise ValueError("history control targets another edit")
        page = self.projection.page(target.page_id)
        if page.effective.effective_fingerprint != receipt.after_effective_page_fingerprint:
            raise ValueError("worker projection is not the committed effective page")
        if (
            receipt.effective_page.page_id != target.page_id
            or receipt.effective_page.effective_fingerprint
            != receipt.after_effective_page_fingerprint
            or tuple(receipt.effective_page.issues) != tuple(receipt.after_issues)
            or tuple(page.effective.issues) != tuple(receipt.after_issues)
        ):
            raise ValueError("history receipt and projected page state disagree")
        history = (
            self.projection.glossary_history
            if target.domain is EditDomain.GLOSSARY
            else page.edit_history
        )
        target_rows = tuple(
            item for item in history if item.record_id == target.edit_id
        )
        control_rows = tuple(
            item for item in history if item.record_id == control.edit_id
        )
        if len(target_rows) != 1 or target_rows[0].is_control:
            raise ValueError("worker projection has no exact target history edit")
        if target_rows[0].active is not receipt.after_active:
            raise ValueError("worker projection has another target active state")
        expected_effective = target.edit_id in receipt.effective_page.applied_edit_ids
        if target_rows[0].effective is not expected_effective:
            raise ValueError("worker projection has another target effective state")
        if len(control_rows) != 1 or not control_rows[0].is_control:
            raise ValueError("worker projection has no exact history control")


@dataclass(frozen=True, slots=True)
class EditHistoryEditorState:
    selection: EditHistorySelection
    phase: EditHistoryEditorPhase
    message: str = ""
    worker_command: EditHistoryWorkerCommand | None = None
    busy_state: EditHistoryWorkerBusyState | None = None
    receipt: EditHistoryWorkerReceipt | None = None
    failure: EditHistoryWorkerFailure | None = None
    cancelled: EditHistoryCancelledReceipt | None = None

    @property
    def busy(self) -> bool:
        return self.phase is EditHistoryEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is EditHistoryEditorPhase.STALE

    @property
    def eligible_operation(self) -> EditHistoryOperation | None:
        return self.selection.eligible_operation

    @property
    def action_enabled(self) -> bool:
        return bool(
            self.eligible_operation is not None
            and not self.busy
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
        return bool(
            not self.busy
            and not self.stale
            and self.worker_command is None
        )

    @property
    def status_tone(self) -> str:
        return {
            EditHistoryEditorPhase.READY: "muted",
            EditHistoryEditorPhase.COMMITTING: "editing",
            EditHistoryEditorPhase.COMMITTED: "ready",
            EditHistoryEditorPhase.CANCELLED: "muted",
            EditHistoryEditorPhase.STALE: "warning",
            EditHistoryEditorPhase.FAILED: "error",
        }[self.phase]


class EditHistoryEditorModel:
    """UI-thread reducer for one persisted history record."""

    def __init__(self, selection: EditHistorySelection) -> None:
        if not isinstance(selection, EditHistorySelection):
            raise TypeError("selection must be EditHistorySelection")
        self._state = EditHistoryEditorState(
            selection=selection,
            phase=EditHistoryEditorPhase.READY,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> EditHistoryEditorState:
        return self._state

    def begin(self) -> EditHistoryWorkerCommand:
        operation = self._state.eligible_operation
        if operation is None or not self._state.action_enabled:
            raise RuntimeError("selected history record has no durable action")
        command = self._command(operation)
        self._begin(command)
        return command

    def begin_revoke(self) -> EditHistoryWorkerCommand:
        if self._state.eligible_operation is not EditHistoryOperation.REVOKE:
            raise RuntimeError("selected history edit is not revocable")
        return self.begin()

    def begin_reapply(self) -> EditHistoryWorkerCommand:
        if self._state.eligible_operation is not EditHistoryOperation.REAPPLY:
            raise RuntimeError("selected history edit is not reapplicable")
        return self.begin()

    def accept_busy(self, value: EditHistoryWorkerBusyState) -> EditHistoryEditorState:
        if not isinstance(value, EditHistoryWorkerBusyState):
            raise TypeError("value must be EditHistoryWorkerBusyState")
        self._require_active_target(value.page_id, value.target_edit_id)
        self._state = replace(
            self._state,
            phase=(EditHistoryEditorPhase.COMMITTING if value.busy else self._state.phase),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(self, value: EditHistoryWorkerReceipt) -> EditHistoryEditorState:
        if not isinstance(value, EditHistoryWorkerReceipt):
            raise TypeError("value must be EditHistoryWorkerReceipt")
        receipt = value.command_receipt
        command = self._require_active_target(
            receipt.target_edit.page_id,
            receipt.target_edit.edit_id,
        )
        if receipt.command_id != receipt.control_edit.edit_id:
            raise ValueError("history receipt command identity is inconsistent")
        if receipt.control_edit.operation != command.operation.value:
            raise ValueError("history receipt has another operation")
        if receipt.before_active is not self._state.selection.active:
            raise ValueError("history receipt has another starting active state")
        expected_after_active = command.operation is EditHistoryOperation.REAPPLY
        if receipt.after_active is not expected_after_active:
            raise ValueError("history receipt has another resulting active state")
        if (
            receipt.before_effective_page_fingerprint
            != command.expected_effective_page_fingerprint
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("history receipt belongs to another selection")
        page = value.projection.page(command.page_id)
        target_edit = receipt.target_edit
        history = (
            value.projection.glossary_history
            if target_edit.domain is EditDomain.GLOSSARY
            else page.edit_history
        )
        reference = next(
            item
            for item in history
            if item.record_id == command.target_edit_id
        )
        issue_kinds = tuple(
            dict.fromkeys(
                issue.kind
                for issue in receipt.after_issues
                if command.target_edit_id in issue.edit_ids
            )
        )
        selection = replace(
            self._state.selection,
            active=reference.active,
            effective=reference.effective,
            issue_kinds=issue_kinds,
            effective_page_fingerprint=receipt.after_effective_page_fingerprint,
        )
        issue_message = (
            " The reapplied edit needs review because projection reported an issue."
            if issue_kinds
            else ""
        )
        self._state = EditHistoryEditorState(
            selection=selection,
            phase=EditHistoryEditorPhase.COMMITTED,
            message=(
                "Edit reapplied."
                if command.operation is EditHistoryOperation.REAPPLY
                else "Edit revoked."
            )
            + issue_message,
            receipt=value,
        )
        return self._state

    def accept_failure(self, value: EditHistoryWorkerFailure) -> EditHistoryEditorState:
        if not isinstance(value, EditHistoryWorkerFailure):
            raise TypeError("value must be EditHistoryWorkerFailure")
        command = self._require_active_event(value)
        del command
        self._state = replace(
            self._state,
            phase=(
                EditHistoryEditorPhase.STALE
                if value.stale
                else EditHistoryEditorPhase.FAILED
            ),
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_stale(self, value: EditHistoryWorkerFailure) -> EditHistoryEditorState:
        if not value.stale:
            raise ValueError("history failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: EditHistoryCancelledReceipt,
    ) -> EditHistoryEditorState:
        if not isinstance(value, EditHistoryCancelledReceipt):
            raise TypeError("value must be EditHistoryCancelledReceipt")
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no history worker command is active")
        if (
            os.path.normcase(value.project_path)
            != os.path.normcase(command.project_path)
            or value.page_id != command.page_id
            or value.target_edit_id != command.target_edit_id
            or value.operation is not command.operation
        ):
            raise ValueError("cancelled receipt belongs to another history action")
        self._state = replace(
            self._state,
            phase=EditHistoryEditorPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(self, selection: EditHistorySelection) -> EditHistoryEditorState:
        if not isinstance(selection, EditHistorySelection):
            raise TypeError("selection must be EditHistorySelection")
        if self._state.busy:
            raise RuntimeError("cannot replace history selection while committing")
        self._state = EditHistoryEditorState(
            selection=selection,
            phase=EditHistoryEditorPhase.READY,
            message=self._ready_message(selection),
        )
        return self._state

    def _command(self, operation: EditHistoryOperation) -> EditHistoryWorkerCommand:
        selection = self._state.selection
        return EditHistoryWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            target_edit_id=selection.target_edit_id,
            operation=operation,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(self, command: EditHistoryWorkerCommand) -> None:
        self._state = replace(
            self._state,
            phase=EditHistoryEditorPhase.COMMITTING,
            message=(
                "Reapplying selected edit..."
                if command.operation is EditHistoryOperation.REAPPLY
                else "Revoking selected edit..."
            ),
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )

    def _require_active_target(
        self,
        page_id: str,
        target_edit_id: str,
    ) -> EditHistoryWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no history worker command is active")
        if command.page_id != page_id or command.target_edit_id != target_edit_id:
            raise ValueError("worker event belongs to another history record")
        return command

    def _require_active_event(
        self,
        value: EditHistoryWorkerFailure,
    ) -> EditHistoryWorkerCommand:
        command = self._require_active_target(value.page_id, value.target_edit_id)
        if (
            os.path.normcase(value.project_path)
            != os.path.normcase(command.project_path)
            or value.operation is not command.operation
        ):
            raise ValueError("worker event belongs to another history action")
        return command

    @staticmethod
    def _ready_message(selection: EditHistorySelection) -> str:
        operation = selection.eligible_operation
        if operation is EditHistoryOperation.REVOKE:
            return "Selected edit can be revoked from durable history."
        if operation is EditHistoryOperation.REAPPLY:
            return "Selected edit can be reapplied from durable history."
        return selection.ineligibility_reason


__all__ = [
    "EditHistoryCancellationState",
    "EditHistoryCancelledReceipt",
    "EditHistoryEditorModel",
    "EditHistoryEditorPhase",
    "EditHistoryEditorState",
    "EditHistorySelection",
    "EditHistoryWorkerBusyState",
    "EditHistoryWorkerCommand",
    "EditHistoryWorkerFailure",
    "EditHistoryWorkerFailureCode",
    "EditHistoryWorkerReceipt",
    "EditHistoryWorkerStage",
]
