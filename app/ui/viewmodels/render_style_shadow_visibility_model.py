# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent shadow-visibility commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    RenderStyleShadowVisibilityCommandErrorCode,
    RenderStyleShadowVisibilityCommandReceipt,
    RenderStyleShadowVisibilityOperation,
)
from app.project_edits.contracts import EditDomain

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHADOW_VISIBILITY_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleShadowVisibilityCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleShadowVisibilityCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleShadowVisibilityCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)


def _required_identity(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or _PATH_SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a path-safe identity")
    return text


def _required_sha256(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return text

class RenderStyleShadowVisibilityEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleShadowVisibilityWorkerStage(str, Enum):
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


class RenderStyleShadowVisibilityWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    SHADOW_VISIBILITY_SLOT_CONFLICT = "shadow_visibility_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_shadow_enabled(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean or None")
    return value


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilitySelection:
    """Exact selected-parent shadow-visibility state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_shadow_enabled: bool | None
    user_shadow_enabled: bool | None
    effective_shadow_enabled: bool | None
    shadow_enabled_authority: str
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
            "automatic_shadow_enabled",
            "user_shadow_enabled",
            "effective_shadow_enabled",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_shadow_enabled(getattr(self, field_name), field_name),
            )
        authority = str(self.shadow_enabled_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "shadow_enabled_authority must be automatic or user"
            )
        object.__setattr__(self, "shadow_enabled_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_shadow_enabled is not None:
                raise ValueError(
                    "automatic shadow-visibility authority cannot carry a user value"
                )
            if self.effective_shadow_enabled != self.automatic_shadow_enabled:
                raise ValueError(
                    "automatic authority must expose the automatic effective shadow visibility"
                )
        else:
            if self.user_shadow_enabled is not False:
                raise ValueError(
                    "user shadow-visibility authority requires exactly false"
                )
            if self.effective_shadow_enabled != self.user_shadow_enabled:
                raise ValueError(
                    "user authority must expose the user effective shadow visibility"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_shadow_enabled is not None
            and self.effective_shadow_enabled is not None
        )
        if eligible and reason:
            raise ValueError(
                "available shadow-visibility selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable shadow-visibility selection requires an unavailable reason"
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
            and self.automatic_shadow_enabled is not None
            and self.effective_shadow_enabled is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityWorkerCommand:
    """UI carrier with one exact shadow visibility and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowVisibilityOperation
    shadow_enabled: bool | None
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
        operation = RenderStyleShadowVisibilityOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowVisibilityOperation.HIDE:
            if self.shadow_enabled is not False:
                if not isinstance(self.shadow_enabled, bool):
                    raise TypeError("shadow_enabled must be the boolean false")
                raise ValueError("Hide must carry only shadow_enabled=false")
        elif self.shadow_enabled is not None:
            raise ValueError(
                "restore_automatic must not carry a shadow_enabled value"
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
class RenderStyleShadowVisibilityWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleShadowVisibilityWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowVisibilityOperation
    shadow_enabled: bool | None
    stage: RenderStyleShadowVisibilityWorkerStage
    message: str = "Shadow-visibility update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityWorkerFailure:
    code: RenderStyleShadowVisibilityWorkerFailureCode
    stage: RenderStyleShadowVisibilityWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowVisibilityOperation
    shadow_enabled: bool | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleShadowVisibilityCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleShadowVisibilityCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleShadowVisibilityWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SHADOW_VISIBILITY_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleShadowVisibilityCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleShadowVisibilityCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleShadowVisibilityCommandReceipt"
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
        automatic_shadow_enabled = _optional_shadow_enabled(
            receipt.automatic_shadow_enabled, "receipt automatic_shadow_enabled"
        )
        before_shadow_enabled = _optional_shadow_enabled(
            receipt.before_shadow_enabled, "receipt before_shadow_enabled"
        )
        after_shadow_enabled = _optional_shadow_enabled(
            receipt.after_shadow_enabled, "receipt after_shadow_enabled"
        )
        if automatic_shadow_enabled is not True:
            raise ValueError("automatic shadow must be visible")
        if receipt.before_shadow_enabled_authority not in {"automatic", "user"}:
            raise ValueError("shadow-visibility before authority is invalid")
        if receipt.after_shadow_enabled_authority not in {"automatic", "user"}:
            raise ValueError("shadow-visibility after authority is invalid")
        if (
            receipt.before_shadow_enabled_authority == "automatic"
            and before_shadow_enabled != automatic_shadow_enabled
        ):
            raise ValueError("automatic shadow-visibility before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("shadow-visibility command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("shadow-visibility supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("shadow-visibility commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"shadow_enabled": False}
                or after_shadow_enabled is not False
                or receipt.after_shadow_enabled_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed shadow-visibility set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("shadow_enabled",)
                or after_shadow_enabled != automatic_shadow_enabled
                or receipt.after_shadow_enabled_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed shadow-visibility restore"
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
        projected_value = _optional_shadow_enabled(
            overrides.get("shadow_enabled", automatic_shadow_enabled),
            "projected shadow_enabled",
        )
        projected_authority = (
            "user" if "shadow_enabled" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("shadow-visibility receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_value = _optional_shadow_enabled(
            receipt_overrides.get(
                "shadow_enabled",
                automatic_shadow_enabled,
            ),
            "receipt shadow_enabled",
        )
        receipt_authority = (
            "user" if "shadow_enabled" in receipt_overrides else "automatic"
        )
        if (
            projected_value != after_shadow_enabled
            or projected_authority != receipt.after_shadow_enabled_authority
            or receipt_value != after_shadow_enabled
            or receipt_authority != receipt.after_shadow_enabled_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed shadow visibility"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowVisibilityEditorState:
    selection: RenderStyleShadowVisibilitySelection
    phase: RenderStyleShadowVisibilityEditorPhase
    draft_shadow_enabled: bool | None
    message: str = ""
    worker_command: RenderStyleShadowVisibilityWorkerCommand | None = None
    busy_state: RenderStyleShadowVisibilityWorkerBusyState | None = None
    receipt: RenderStyleShadowVisibilityWorkerReceipt | None = None
    failure: RenderStyleShadowVisibilityWorkerFailure | None = None
    cancelled: RenderStyleShadowVisibilityCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_shadow_enabled != self.selection.effective_shadow_enabled

    @property
    def valid(self) -> bool:
        return self.draft_shadow_enabled is False

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleShadowVisibilityEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleShadowVisibilityEditorPhase.STALE

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
            and self.selection.shadow_enabled_authority == "user"
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
                RenderStyleShadowVisibilityEditorPhase.READY,
                RenderStyleShadowVisibilityEditorPhase.COMMITTED,
                RenderStyleShadowVisibilityEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleShadowVisibilityEditorPhase.READY: "muted",
            RenderStyleShadowVisibilityEditorPhase.DIRTY: "editing",
            RenderStyleShadowVisibilityEditorPhase.COMMITTING: "editing",
            RenderStyleShadowVisibilityEditorPhase.COMMITTED: "ready",
            RenderStyleShadowVisibilityEditorPhase.CANCELLED: "muted",
            RenderStyleShadowVisibilityEditorPhase.STALE: "warning",
            RenderStyleShadowVisibilityEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleShadowVisibilityEditorModel:
    """UI-thread reducer for one exact selected-parent shadow-visibility ratio."""

    def __init__(self, selection: RenderStyleShadowVisibilitySelection) -> None:
        if not isinstance(selection, RenderStyleShadowVisibilitySelection):
            raise TypeError(
                "selection must be RenderStyleShadowVisibilitySelection"
            )
        self._state = RenderStyleShadowVisibilityEditorState(
            selection=selection,
            phase=RenderStyleShadowVisibilityEditorPhase.READY,
            draft_shadow_enabled=selection.effective_shadow_enabled,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleShadowVisibilityEditorState:
        return self._state

    def set_draft_shadow_enabled(
        self,
        value: bool,
    ) -> RenderStyleShadowVisibilityEditorState:
        if value is not False:
            if not isinstance(value, bool):
                raise TypeError("shadow-visibility draft must be the boolean false")
            raise ValueError("shadow visibility can only draft Hidden")
        shadow_enabled = False
        if not self._state.editing_enabled:
            raise RuntimeError("shadow-visibility draft is not editable")
        phase = (
            RenderStyleShadowVisibilityEditorPhase.DIRTY
            if shadow_enabled != self._state.selection.effective_shadow_enabled
            else RenderStyleShadowVisibilityEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_enabled=shadow_enabled,
            message=(
                "Shadow visibility has an unapplied change."
                if phase is RenderStyleShadowVisibilityEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleShadowVisibilityEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard shadow visibility while it is committing")
        phase = (
            RenderStyleShadowVisibilityEditorPhase.STALE
            if self._state.stale
            else RenderStyleShadowVisibilityEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_enabled=self._state.selection.effective_shadow_enabled,
            message=(
                "Reload the selected parent before editing shadow visibility."
                if phase is RenderStyleShadowVisibilityEditorPhase.STALE
                else "Shadow-visibility draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleShadowVisibilityEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleShadowVisibilityWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable shadow-visibility draft")
        shadow_enabled = self._state.draft_shadow_enabled
        if shadow_enabled is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("shadow-visibility draft is unavailable")
        command = self._command(
            RenderStyleShadowVisibilityOperation.HIDE,
            shadow_enabled=shadow_enabled,
        )
        self._begin(command, "Applying shadow-visibility edit...")
        return command

    def begin_restore(self) -> RenderStyleShadowVisibilityWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic shadow visibility is already effective")
        command = self._command(
            RenderStyleShadowVisibilityOperation.RESTORE_AUTOMATIC,
            shadow_enabled=None,
        )
        self._begin(command, "Restoring automatic shadow visibility...")
        return command

    def accept_busy(
        self,
        value: RenderStyleShadowVisibilityWorkerBusyState,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not isinstance(value, RenderStyleShadowVisibilityWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleShadowVisibilityWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleShadowVisibilityEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleShadowVisibilityWorkerReceipt,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not isinstance(value, RenderStyleShadowVisibilityWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleShadowVisibilityWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleShadowVisibilityOperation.HIDE
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("shadow-visibility receipt has another operation")
        if command.operation is RenderStyleShadowVisibilityOperation.HIDE:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"shadow_enabled": command.shadow_enabled}
                or receipt.after_shadow_enabled != command.shadow_enabled
            ):
                raise ValueError("shadow-visibility receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("shadow_enabled",)
                or receipt.after_shadow_enabled != receipt.automatic_shadow_enabled
            ):
                raise ValueError("shadow-visibility receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("shadow-visibility receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_shadow_enabled != selection.automatic_shadow_enabled
            or receipt.before_shadow_enabled != selection.effective_shadow_enabled
            or receipt.before_shadow_enabled_authority
            != selection.shadow_enabled_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("shadow-visibility receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_shadow_enabled=(
                receipt.after_shadow_enabled
                if receipt.after_shadow_enabled_authority == "user"
                else None
            ),
            effective_shadow_enabled=receipt.after_shadow_enabled,
            shadow_enabled_authority=receipt.after_shadow_enabled_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleShadowVisibilityEditorState(
            selection=updated_selection,
            phase=RenderStyleShadowVisibilityEditorPhase.COMMITTED,
            draft_shadow_enabled=receipt.after_shadow_enabled,
            message="Shadow visibility saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleShadowVisibilityWorkerFailure,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not isinstance(value, RenderStyleShadowVisibilityWorkerFailure):
            raise TypeError(
                "value must be RenderStyleShadowVisibilityWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_enabled,
        )
        phase = (
            RenderStyleShadowVisibilityEditorPhase.STALE
            if value.stale
            else RenderStyleShadowVisibilityEditorPhase.FAILED
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
        value: RenderStyleShadowVisibilityWorkerFailure,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not value.stale:
            raise ValueError("shadow-visibility failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleShadowVisibilityCancelledReceipt,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not isinstance(value, RenderStyleShadowVisibilityCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleShadowVisibilityCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_enabled,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleShadowVisibilityEditorPhase.CANCELLED,
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
        selection: RenderStyleShadowVisibilitySelection,
    ) -> RenderStyleShadowVisibilityEditorState:
        if not isinstance(selection, RenderStyleShadowVisibilitySelection):
            raise TypeError(
                "selection must be RenderStyleShadowVisibilitySelection"
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
            self._state.draft_shadow_enabled
            if preserve_draft
            else selection.effective_shadow_enabled
        )
        phase = (
            RenderStyleShadowVisibilityEditorPhase.DIRTY
            if draft != selection.effective_shadow_enabled
            else RenderStyleShadowVisibilityEditorPhase.READY
        )
        self._state = RenderStyleShadowVisibilityEditorState(
            selection=selection,
            phase=phase,
            draft_shadow_enabled=draft,
            message=(
                "Current state changed; review the preserved shadow-visibility draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied shadow-visibility draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleShadowVisibilityOperation,
        *,
        shadow_enabled: bool | None,
    ) -> RenderStyleShadowVisibilityWorkerCommand:
        selection = self._state.selection
        return RenderStyleShadowVisibilityWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            shadow_enabled=shadow_enabled,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleShadowVisibilityWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleShadowVisibilityEditorPhase.COMMITTING,
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
    ) -> RenderStyleShadowVisibilityWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no shadow-visibility worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleShadowVisibilityOperation,
        shadow_enabled: bool | None,
    ) -> RenderStyleShadowVisibilityWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.shadow_enabled != shadow_enabled
        ):
            raise ValueError("worker event belongs to another shadow-visibility command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleShadowVisibilitySelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.shadow_enabled_authority == "user":
            return "Your edit hides the automatic shadow."
        return "Automatic visible shadow is effective."

__all__ = [
    "RenderStyleShadowVisibilityCancellationState",
    "RenderStyleShadowVisibilityCancelledReceipt",
    "RenderStyleShadowVisibilityEditorModel",
    "RenderStyleShadowVisibilityEditorPhase",
    "RenderStyleShadowVisibilityEditorState",
    "RenderStyleShadowVisibilitySelection",
    "RenderStyleShadowVisibilityWorkerBusyState",
    "RenderStyleShadowVisibilityWorkerCommand",
    "RenderStyleShadowVisibilityWorkerFailure",
    "RenderStyleShadowVisibilityWorkerFailureCode",
    "RenderStyleShadowVisibilityWorkerReceipt",
    "RenderStyleShadowVisibilityWorkerStage",
]
