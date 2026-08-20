# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent shadow-offset commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    RenderStyleShadowOffsetCommandErrorCode,
    RenderStyleShadowOffsetCommandReceipt,
    RenderStyleShadowOffsetOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_shadow_offset,
    thaw_json,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHADOW_OFFSET_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleShadowOffsetCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleShadowOffsetCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleShadowOffsetCommandErrorCode.STALE_GLOBAL_HEAD,
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

class RenderStyleShadowOffsetEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleShadowOffsetWorkerStage(str, Enum):
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


class RenderStyleShadowOffsetWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    SHADOW_OFFSET_SLOT_CONFLICT = "shadow_offset_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_shadow_offset(
    value: Any,
    field_name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    return canonical_render_shadow_offset(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetSelection:
    """Exact selected-parent shadow-offset state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_shadow_offset: tuple[float, float] | None
    user_shadow_offset: tuple[float, float] | None
    effective_shadow_offset: tuple[float, float] | None
    shadow_offset_authority: str
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
            "automatic_shadow_offset",
            "user_shadow_offset",
            "effective_shadow_offset",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_shadow_offset(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.shadow_offset_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "shadow_offset_authority must be automatic or user"
            )
        object.__setattr__(self, "shadow_offset_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_shadow_offset is not None:
                raise ValueError(
                    "automatic shadow-offset authority cannot carry a user value"
                )
            if self.effective_shadow_offset != self.automatic_shadow_offset:
                raise ValueError(
                    "automatic authority must expose the automatic effective shadow offset"
                )
        else:
            if self.user_shadow_offset is None:
                raise ValueError(
                    "user shadow-offset authority requires a user value"
                )
            if self.effective_shadow_offset != self.user_shadow_offset:
                raise ValueError(
                    "user authority must expose the user effective shadow offset"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_shadow_offset is not None
            and self.effective_shadow_offset is not None
        )
        if eligible and reason:
            raise ValueError(
                "available shadow-offset selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable shadow-offset selection requires an unavailable reason"
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
            and self.automatic_shadow_offset is not None
            and self.effective_shadow_offset is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetWorkerCommand:
    """UI carrier with one exact pixel width and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowOffsetOperation
    shadow_offset: tuple[float, float] | None
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
        operation = RenderStyleShadowOffsetOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowOffsetOperation.SET:
            object.__setattr__(
                self,
                "shadow_offset",
                canonical_render_shadow_offset(self.shadow_offset),
            )
        elif self.shadow_offset is not None:
            raise ValueError(
                "restore_automatic must not carry a shadow_offset value"
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
class RenderStyleShadowOffsetWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleShadowOffsetWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowOffsetOperation
    shadow_offset: tuple[float, float] | None
    stage: RenderStyleShadowOffsetWorkerStage
    message: str = "Shadow-offset update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetWorkerFailure:
    code: RenderStyleShadowOffsetWorkerFailureCode
    stage: RenderStyleShadowOffsetWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowOffsetOperation
    shadow_offset: tuple[float, float] | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleShadowOffsetCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleShadowOffsetCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleShadowOffsetWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SHADOW_OFFSET_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleShadowOffsetCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleShadowOffsetCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleShadowOffsetCommandReceipt"
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
        automatic_shadow_offset = canonical_render_shadow_offset(
            receipt.automatic_shadow_offset,
            field_name="receipt automatic_shadow_offset",
        )
        before_shadow_offset = canonical_render_shadow_offset(
            receipt.before_shadow_offset,
            field_name="receipt before_shadow_offset",
        )
        after_shadow_offset = canonical_render_shadow_offset(
            receipt.after_shadow_offset,
            field_name="receipt after_shadow_offset",
        )
        if receipt.before_shadow_offset_authority not in {"automatic", "user"}:
            raise ValueError("shadow-offset before authority is invalid")
        if receipt.after_shadow_offset_authority not in {"automatic", "user"}:
            raise ValueError("shadow-offset after authority is invalid")
        if (
            receipt.before_shadow_offset_authority == "automatic"
            and before_shadow_offset != automatic_shadow_offset
        ):
            raise ValueError("automatic shadow-offset before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("shadow-offset command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("shadow-offset supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("shadow-offset commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"shadow_offset": after_shadow_offset}
                or receipt.after_shadow_offset_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed shadow-offset set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("shadow_offset",)
                or after_shadow_offset != automatic_shadow_offset
                or receipt.after_shadow_offset_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed shadow-offset restore"
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
        projected_offset = canonical_render_shadow_offset(
            overrides.get("shadow_offset", automatic_shadow_offset),
            field_name="projected shadow_offset",
        )
        projected_authority = (
            "user" if "shadow_offset" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("shadow-offset receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_offset = canonical_render_shadow_offset(
            receipt_overrides.get(
                "shadow_offset",
                automatic_shadow_offset,
            ),
            field_name="receipt shadow_offset",
        )
        receipt_authority = (
            "user" if "shadow_offset" in receipt_overrides else "automatic"
        )
        if (
            projected_offset != after_shadow_offset
            or projected_authority != receipt.after_shadow_offset_authority
            or receipt_offset != after_shadow_offset
            or receipt_authority != receipt.after_shadow_offset_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed shadow offset"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetEditorState:
    selection: RenderStyleShadowOffsetSelection
    phase: RenderStyleShadowOffsetEditorPhase
    draft_shadow_offset: tuple[float, float] | None
    message: str = ""
    worker_command: RenderStyleShadowOffsetWorkerCommand | None = None
    busy_state: RenderStyleShadowOffsetWorkerBusyState | None = None
    receipt: RenderStyleShadowOffsetWorkerReceipt | None = None
    failure: RenderStyleShadowOffsetWorkerFailure | None = None
    cancelled: RenderStyleShadowOffsetCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_shadow_offset != self.selection.effective_shadow_offset

    @property
    def valid(self) -> bool:
        if self.draft_shadow_offset is None:
            return False
        try:
            canonical_render_shadow_offset(self.draft_shadow_offset)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleShadowOffsetEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleShadowOffsetEditorPhase.STALE

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
            and self.selection.shadow_offset_authority == "user"
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
                RenderStyleShadowOffsetEditorPhase.READY,
                RenderStyleShadowOffsetEditorPhase.COMMITTED,
                RenderStyleShadowOffsetEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleShadowOffsetEditorPhase.READY: "muted",
            RenderStyleShadowOffsetEditorPhase.DIRTY: "editing",
            RenderStyleShadowOffsetEditorPhase.COMMITTING: "editing",
            RenderStyleShadowOffsetEditorPhase.COMMITTED: "ready",
            RenderStyleShadowOffsetEditorPhase.CANCELLED: "muted",
            RenderStyleShadowOffsetEditorPhase.STALE: "warning",
            RenderStyleShadowOffsetEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleShadowOffsetEditorModel:
    """UI-thread reducer for one exact selected-parent X/Y shadow offset."""

    def __init__(self, selection: RenderStyleShadowOffsetSelection) -> None:
        if not isinstance(selection, RenderStyleShadowOffsetSelection):
            raise TypeError(
                "selection must be RenderStyleShadowOffsetSelection"
            )
        self._state = RenderStyleShadowOffsetEditorState(
            selection=selection,
            phase=RenderStyleShadowOffsetEditorPhase.READY,
            draft_shadow_offset=selection.effective_shadow_offset,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleShadowOffsetEditorState:
        return self._state

    def set_draft_shadow_offset(
        self,
        value: tuple[float, float] | list[float],
    ) -> RenderStyleShadowOffsetEditorState:
        shadow_offset = canonical_render_shadow_offset(
            value,
            field_name="shadow-offset draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("shadow-offset draft is not editable")
        phase = (
            RenderStyleShadowOffsetEditorPhase.DIRTY
            if shadow_offset != self._state.selection.effective_shadow_offset
            else RenderStyleShadowOffsetEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_offset=shadow_offset,
            message=(
                "Shadow offset has an unapplied change."
                if phase is RenderStyleShadowOffsetEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleShadowOffsetEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard shadow offset while it is committing")
        phase = (
            RenderStyleShadowOffsetEditorPhase.STALE
            if self._state.stale
            else RenderStyleShadowOffsetEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_offset=self._state.selection.effective_shadow_offset,
            message=(
                "Reload the selected parent before editing shadow offset."
                if phase is RenderStyleShadowOffsetEditorPhase.STALE
                else "Shadow-offset draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleShadowOffsetEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleShadowOffsetWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable shadow-offset draft")
        shadow_offset = self._state.draft_shadow_offset
        if shadow_offset is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("shadow-offset draft is unavailable")
        command = self._command(
            RenderStyleShadowOffsetOperation.SET,
            shadow_offset=shadow_offset,
        )
        self._begin(command, "Applying shadow-offset edit...")
        return command

    def begin_restore(self) -> RenderStyleShadowOffsetWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic shadow offset is already effective")
        command = self._command(
            RenderStyleShadowOffsetOperation.RESTORE_AUTOMATIC,
            shadow_offset=None,
        )
        self._begin(command, "Restoring automatic shadow offset...")
        return command

    def accept_busy(
        self,
        value: RenderStyleShadowOffsetWorkerBusyState,
    ) -> RenderStyleShadowOffsetEditorState:
        if not isinstance(value, RenderStyleShadowOffsetWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleShadowOffsetWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleShadowOffsetEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleShadowOffsetWorkerReceipt,
    ) -> RenderStyleShadowOffsetEditorState:
        if not isinstance(value, RenderStyleShadowOffsetWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleShadowOffsetWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleShadowOffsetOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("shadow-offset receipt has another operation")
        if command.operation is RenderStyleShadowOffsetOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"shadow_offset": command.shadow_offset}
                or receipt.after_shadow_offset != command.shadow_offset
            ):
                raise ValueError("shadow-offset receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("shadow_offset",)
                or receipt.after_shadow_offset != receipt.automatic_shadow_offset
            ):
                raise ValueError("shadow-offset receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("shadow-offset receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_shadow_offset != selection.automatic_shadow_offset
            or receipt.before_shadow_offset != selection.effective_shadow_offset
            or receipt.before_shadow_offset_authority
            != selection.shadow_offset_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("shadow-offset receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_shadow_offset=(
                receipt.after_shadow_offset
                if receipt.after_shadow_offset_authority == "user"
                else None
            ),
            effective_shadow_offset=receipt.after_shadow_offset,
            shadow_offset_authority=receipt.after_shadow_offset_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleShadowOffsetEditorState(
            selection=updated_selection,
            phase=RenderStyleShadowOffsetEditorPhase.COMMITTED,
            draft_shadow_offset=receipt.after_shadow_offset,
            message="Shadow offset saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleShadowOffsetWorkerFailure,
    ) -> RenderStyleShadowOffsetEditorState:
        if not isinstance(value, RenderStyleShadowOffsetWorkerFailure):
            raise TypeError(
                "value must be RenderStyleShadowOffsetWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_offset,
        )
        phase = (
            RenderStyleShadowOffsetEditorPhase.STALE
            if value.stale
            else RenderStyleShadowOffsetEditorPhase.FAILED
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
        value: RenderStyleShadowOffsetWorkerFailure,
    ) -> RenderStyleShadowOffsetEditorState:
        if not value.stale:
            raise ValueError("shadow-offset failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleShadowOffsetCancelledReceipt,
    ) -> RenderStyleShadowOffsetEditorState:
        if not isinstance(value, RenderStyleShadowOffsetCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleShadowOffsetCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_offset,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleShadowOffsetEditorPhase.CANCELLED,
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
        selection: RenderStyleShadowOffsetSelection,
    ) -> RenderStyleShadowOffsetEditorState:
        if not isinstance(selection, RenderStyleShadowOffsetSelection):
            raise TypeError(
                "selection must be RenderStyleShadowOffsetSelection"
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
            self._state.draft_shadow_offset
            if preserve_draft
            else selection.effective_shadow_offset
        )
        phase = (
            RenderStyleShadowOffsetEditorPhase.DIRTY
            if draft != selection.effective_shadow_offset
            else RenderStyleShadowOffsetEditorPhase.READY
        )
        self._state = RenderStyleShadowOffsetEditorState(
            selection=selection,
            phase=phase,
            draft_shadow_offset=draft,
            message=(
                "Current state changed; review the preserved shadow-offset draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied shadow-offset draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleShadowOffsetOperation,
        *,
        shadow_offset: tuple[float, float] | None,
    ) -> RenderStyleShadowOffsetWorkerCommand:
        selection = self._state.selection
        return RenderStyleShadowOffsetWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            shadow_offset=shadow_offset,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleShadowOffsetWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleShadowOffsetEditorPhase.COMMITTING,
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
    ) -> RenderStyleShadowOffsetWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no shadow-offset worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleShadowOffsetOperation,
        shadow_offset: tuple[float, float] | None,
    ) -> RenderStyleShadowOffsetWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.shadow_offset != shadow_offset
        ):
            raise ValueError("worker event belongs to another shadow-offset command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleShadowOffsetSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.shadow_offset_authority == "user":
            return "User shadow offset is effective."
        return "Automatic shadow offset is effective."

__all__ = [
    "RenderStyleShadowOffsetCancellationState",
    "RenderStyleShadowOffsetCancelledReceipt",
    "RenderStyleShadowOffsetEditorModel",
    "RenderStyleShadowOffsetEditorPhase",
    "RenderStyleShadowOffsetEditorState",
    "RenderStyleShadowOffsetSelection",
    "RenderStyleShadowOffsetWorkerBusyState",
    "RenderStyleShadowOffsetWorkerCommand",
    "RenderStyleShadowOffsetWorkerFailure",
    "RenderStyleShadowOffsetWorkerFailureCode",
    "RenderStyleShadowOffsetWorkerReceipt",
    "RenderStyleShadowOffsetWorkerStage",
]
