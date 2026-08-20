# -*- coding: utf-8 -*-
"""GUI state for one selected-parent registered font-role override."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import TYPE_CHECKING, Any, Mapping

from app.project_edits.commands import (
    RenderStyleFontRoleCommandErrorCode,
    RenderStyleFontRoleCommandReceipt,
    RenderStyleFontRoleOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_font_role,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


class RenderStyleFontRoleEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleFontRoleWorkerStage(str, Enum):
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


class RenderStyleFontRoleWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_FONT_ROLE_UNAVAILABLE = "automatic_font_role_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    FONT_ROLE_SLOT_CONFLICT = "font_role_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleFontRoleCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleFontRoleCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleFontRoleCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)


def _required_identity(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = _required_identity(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return text


def _optional_role(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return canonical_render_font_role(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleSelection:
    project_path: str
    page_id: str
    parent_id: str
    automatic_font_role: str | None
    user_font_role: str | None
    effective_font_role: str | None
    font_role_authority: str
    render_required: bool
    excluded: bool
    unavailable_reason: str
    effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in ("page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        for field_name in (
            "automatic_font_role",
            "user_font_role",
            "effective_font_role",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_role(getattr(self, field_name), field_name),
            )
        authority = str(self.font_role_authority or "")
        if authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "font_role_authority must be automatic, user, or unavailable"
            )
        object.__setattr__(self, "font_role_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        reason = str(self.unavailable_reason or "").strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_font_role is not None:
                raise ValueError("automatic authority cannot carry a user role")
            if self.effective_font_role != self.automatic_font_role:
                raise ValueError("automatic role must be effective")
        elif authority == "user":
            if self.user_font_role is None:
                raise ValueError("user authority requires a user role")
            if self.effective_font_role != self.user_font_role:
                raise ValueError("user role must be effective")
        elif any(
            value is not None
            for value in (
                self.automatic_font_role,
                self.user_font_role,
                self.effective_font_role,
            )
        ):
            raise ValueError("unavailable authority cannot carry font roles")
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_font_role is not None
            and self.effective_font_role is not None
        )
        if eligible == bool(reason):
            raise ValueError(
                "font-role availability and unavailable reason disagree"
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
            and self.automatic_font_role is not None
            and self.effective_font_role is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleWorkerCommand:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontRoleOperation
    font_role: str | None
    expected_effective_page_fingerprint: str

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in ("page_id", "parent_id"):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        operation = RenderStyleFontRoleOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleFontRoleOperation.SET:
            object.__setattr__(
                self,
                "font_role",
                canonical_render_font_role(self.font_role),
            )
        elif self.font_role is not None:
            raise ValueError("restore_automatic must not carry font_role")
        object.__setattr__(
            self,
            "expected_effective_page_fingerprint",
            _required_sha256(
                self.expected_effective_page_fingerprint,
                "expected_effective_page_fingerprint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleFontRoleWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontRoleOperation
    font_role: str | None
    stage: RenderStyleFontRoleWorkerStage
    message: str = "Font-role update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleWorkerFailure:
    code: RenderStyleFontRoleWorkerFailureCode
    stage: RenderStyleFontRoleWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontRoleOperation
    font_role: str | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleFontRoleCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleFontRoleCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleFontRoleWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleWorkerReceipt:
    command_receipt: RenderStyleFontRoleCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleFontRoleCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleFontRoleCommandReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        from app.project_edits.fingerprints import canonical_sha256
        from app.project_edits.ledger import ProjectEditLedger
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project does not match its projection")
        receipt = self.command_receipt
        edit = receipt.edit
        commit = receipt.commit_receipt
        automatic = canonical_render_font_role(receipt.automatic_font_role)
        before = canonical_render_font_role(receipt.before_font_role)
        after = canonical_render_font_role(receipt.after_font_role)
        if receipt.before_font_role_authority not in {"automatic", "user"}:
            raise ValueError("font-role before authority is invalid")
        if receipt.after_font_role_authority not in {"automatic", "user"}:
            raise ValueError("font-role after authority is invalid")
        if receipt.before_font_role_authority == "automatic" and before != automatic:
            raise ValueError("automatic font-role before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("font-role command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("font-role supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("font-role commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"font_role": after}
                or receipt.after_font_role_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed role set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(fields, tuple)
                or fields != ("font_role",)
                or after != automatic
                or receipt.after_font_role_authority != "automatic"
            ):
                raise ValueError("worker receipt is not the committed role restore")
        else:
            raise ValueError("worker receipt has another render-style operation")
        ledger = ProjectEditLedger.from_dict(self.project["edit_ledger"])
        if ledger.get(edit.edit_id) != edit:
            raise ValueError("worker project lacks the committed font-role edit")
        if self.projection.metadata.project_id != edit.project_id:
            raise ValueError("worker projection belongs to another project")
        page = self.projection.page(edit.page_id)
        if page.effective != receipt.effective_page:
            raise ValueError("worker projection is not the committed effective page")
        projected = page.parent(edit.target.parent_id)
        if (
            projected.automatic_font_role != automatic
            or projected.effective_font_role != after
            or projected.font_role_authority != receipt.after_font_role_authority
        ):
            raise ValueError("worker projection lacks the committed font role")


@dataclass(frozen=True, slots=True)
class RenderStyleFontRoleEditorState:
    selection: RenderStyleFontRoleSelection
    phase: RenderStyleFontRoleEditorPhase
    draft_font_role: str | None
    message: str = ""
    worker_command: RenderStyleFontRoleWorkerCommand | None = None
    busy_state: RenderStyleFontRoleWorkerBusyState | None = None
    receipt: RenderStyleFontRoleWorkerReceipt | None = None
    failure: RenderStyleFontRoleWorkerFailure | None = None
    cancelled: RenderStyleFontRoleCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_font_role != self.selection.effective_font_role

    @property
    def valid(self) -> bool:
        try:
            canonical_render_font_role(self.draft_font_role)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleFontRoleEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleFontRoleEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return bool(self.available and not self.busy and not self.stale)

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
        return bool(not self.busy and self.dirty)

    @property
    def restore_enabled(self) -> bool:
        return bool(
            self.editing_enabled
            and not self.dirty
            and self.selection.font_role_authority == "user"
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
        if not self.available and self.phase in {
            RenderStyleFontRoleEditorPhase.READY,
            RenderStyleFontRoleEditorPhase.COMMITTED,
            RenderStyleFontRoleEditorPhase.CANCELLED,
        }:
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleFontRoleEditorPhase.READY: "muted",
            RenderStyleFontRoleEditorPhase.DIRTY: "editing",
            RenderStyleFontRoleEditorPhase.COMMITTING: "editing",
            RenderStyleFontRoleEditorPhase.COMMITTED: "ready",
            RenderStyleFontRoleEditorPhase.CANCELLED: "muted",
            RenderStyleFontRoleEditorPhase.STALE: "warning",
            RenderStyleFontRoleEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleFontRoleEditorModel:
    def __init__(self, selection: RenderStyleFontRoleSelection) -> None:
        if not isinstance(selection, RenderStyleFontRoleSelection):
            raise TypeError("selection must be RenderStyleFontRoleSelection")
        self._state = RenderStyleFontRoleEditorState(
            selection=selection,
            phase=RenderStyleFontRoleEditorPhase.READY,
            draft_font_role=selection.effective_font_role,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleFontRoleEditorState:
        return self._state

    def set_draft_font_role(self, value: str) -> RenderStyleFontRoleEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("font-role draft is not editable")
        role = canonical_render_font_role(value, field_name="font-role draft")
        phase = (
            RenderStyleFontRoleEditorPhase.DIRTY
            if role != self._state.selection.effective_font_role
            else RenderStyleFontRoleEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_font_role=role,
            message=(
                "Font role has an unapplied change."
                if phase is RenderStyleFontRoleEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleFontRoleEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard font role while it is committing")
        phase = (
            RenderStyleFontRoleEditorPhase.STALE
            if self._state.stale
            else RenderStyleFontRoleEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_font_role=self._state.selection.effective_font_role,
            message=(
                "Reload the selected parent before editing font role."
                if phase is RenderStyleFontRoleEditorPhase.STALE
                else "Font-role draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleFontRoleWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable font-role draft")
        command = self._command(
            RenderStyleFontRoleOperation.SET,
            font_role=canonical_render_font_role(self._state.draft_font_role),
        )
        return self._begin(command, "Applying registered font-role edit...")

    def begin_restore(self) -> RenderStyleFontRoleWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic font role is already effective")
        command = self._command(
            RenderStyleFontRoleOperation.RESTORE_AUTOMATIC,
            font_role=None,
        )
        return self._begin(command, "Restoring automatic font role...")

    def accept_busy(
        self,
        value: RenderStyleFontRoleWorkerBusyState,
    ) -> RenderStyleFontRoleEditorState:
        if not isinstance(value, RenderStyleFontRoleWorkerBusyState):
            raise TypeError("value must be RenderStyleFontRoleWorkerBusyState")
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleFontRoleEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleFontRoleWorkerReceipt,
    ) -> RenderStyleFontRoleEditorState:
        if not isinstance(value, RenderStyleFontRoleWorkerReceipt):
            raise TypeError("value must be RenderStyleFontRoleWorkerReceipt")
        receipt = value.command_receipt
        command = self._require_active_target(
            receipt.edit.page_id,
            receipt.edit.target.parent_id,
        )
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
            or receipt.automatic_font_role
            != self._state.selection.automatic_font_role
            or receipt.before_font_role
            != self._state.selection.effective_font_role
            or receipt.before_font_role_authority
            != self._state.selection.font_role_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(value.projection.metadata.project_path))
        ):
            raise ValueError("font-role receipt belongs to another selection")
        expected_after = (
            command.font_role
            if command.operation is RenderStyleFontRoleOperation.SET
            else receipt.automatic_font_role
        )
        expected_authority = (
            "user"
            if command.operation is RenderStyleFontRoleOperation.SET
            else "automatic"
        )
        if (
            receipt.after_font_role != expected_after
            or receipt.after_font_role_authority != expected_authority
        ):
            raise ValueError("font-role receipt has another effective value")
        updated = replace(
            self._state.selection,
            user_font_role=(expected_after if expected_authority == "user" else None),
            effective_font_role=expected_after,
            font_role_authority=expected_authority,
            unavailable_reason="",
            effective_page_fingerprint=receipt.after_effective_page_fingerprint,
        )
        self._state = RenderStyleFontRoleEditorState(
            selection=updated,
            phase=RenderStyleFontRoleEditorPhase.COMMITTED,
            draft_font_role=expected_after,
            message="Font role saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleFontRoleWorkerFailure,
    ) -> RenderStyleFontRoleEditorState:
        if not isinstance(value, RenderStyleFontRoleWorkerFailure):
            raise TypeError("value must be RenderStyleFontRoleWorkerFailure")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleFontRoleEditorPhase.STALE
                if value.stale
                else RenderStyleFontRoleEditorPhase.FAILED
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
        value: RenderStyleFontRoleWorkerFailure,
    ) -> RenderStyleFontRoleEditorState:
        if not value.stale:
            raise ValueError("font-role failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleFontRoleCancelledReceipt,
    ) -> RenderStyleFontRoleEditorState:
        if not isinstance(value, RenderStyleFontRoleCancelledReceipt):
            raise TypeError("value must be RenderStyleFontRoleCancelledReceipt")
        self._require_active_event(value)
        self._state = replace(
            self._state,
            phase=RenderStyleFontRoleEditorPhase.CANCELLED,
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
        selection: RenderStyleFontRoleSelection,
    ) -> RenderStyleFontRoleEditorState:
        if not isinstance(selection, RenderStyleFontRoleSelection):
            raise TypeError("selection must be RenderStyleFontRoleSelection")
        if self._state.busy:
            raise RuntimeError("cannot replace selection while a command is active")
        same_target = (
            selection.project_path == self._state.selection.project_path
            and selection.page_id == self._state.selection.page_id
            and selection.parent_id == self._state.selection.parent_id
        )
        preserve = same_target and self._state.dirty
        draft = self._state.draft_font_role if preserve else selection.effective_font_role
        phase = (
            RenderStyleFontRoleEditorPhase.DIRTY
            if draft != selection.effective_font_role
            else RenderStyleFontRoleEditorPhase.READY
        )
        self._state = RenderStyleFontRoleEditorState(
            selection=selection,
            phase=phase,
            draft_font_role=draft,
            message=(
                "Selection refreshed; the unapplied font-role draft was preserved."
                if preserve
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleFontRoleOperation,
        *,
        font_role: str | None,
    ) -> RenderStyleFontRoleWorkerCommand:
        selection = self._state.selection
        return RenderStyleFontRoleWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            font_role=font_role,
            expected_effective_page_fingerprint=selection.effective_page_fingerprint,
        )

    def _begin(
        self,
        command: RenderStyleFontRoleWorkerCommand,
        message: str,
    ) -> RenderStyleFontRoleWorkerCommand:
        self._state = replace(
            self._state,
            phase=RenderStyleFontRoleEditorPhase.COMMITTING,
            message=message,
            worker_command=command,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def _require_active_target(
        self,
        page_id: str,
        parent_id: str,
    ) -> RenderStyleFontRoleWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no font-role worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        value: RenderStyleFontRoleWorkerFailure | RenderStyleFontRoleCancelledReceipt,
    ) -> RenderStyleFontRoleWorkerCommand:
        command = self._require_active_target(value.page_id, value.parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(value.project_path))
            or command.operation is not value.operation
            or command.font_role != value.font_role
        ):
            raise ValueError("worker event belongs to another font-role command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleFontRoleSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.font_role_authority == "user":
            return "User font role is effective."
        return "Automatic font role is effective."


__all__ = [
    "RenderStyleFontRoleCancellationState",
    "RenderStyleFontRoleCancelledReceipt",
    "RenderStyleFontRoleEditorModel",
    "RenderStyleFontRoleEditorPhase",
    "RenderStyleFontRoleEditorState",
    "RenderStyleFontRoleSelection",
    "RenderStyleFontRoleWorkerBusyState",
    "RenderStyleFontRoleWorkerCommand",
    "RenderStyleFontRoleWorkerFailure",
    "RenderStyleFontRoleWorkerFailureCode",
    "RenderStyleFontRoleWorkerReceipt",
    "RenderStyleFontRoleWorkerStage",
]
