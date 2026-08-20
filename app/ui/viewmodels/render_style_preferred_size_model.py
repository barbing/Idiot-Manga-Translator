# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent preferred-size commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    RenderStylePreferredSizeCommandErrorCode,
    RenderStylePreferredSizeCommandReceipt,
    RenderStylePreferredSizeOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_preferred_size,
    thaw_json,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_PREFERRED_SIZE_STALE_COMMAND_CODES = frozenset(
    {
        RenderStylePreferredSizeCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStylePreferredSizeCommandErrorCode.STALE_PAGE_HEAD,
        RenderStylePreferredSizeCommandErrorCode.STALE_GLOBAL_HEAD,
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

class RenderStylePreferredSizeEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStylePreferredSizeWorkerStage(str, Enum):
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


class RenderStylePreferredSizeWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE = "automatic_preferred_size_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    PREFERRED_SIZE_SLOT_CONFLICT = "preferred_size_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_preferred_size(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_render_preferred_size(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeSelection:
    """Exact selected-parent preferred-size state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_preferred_size: float | None
    user_preferred_size: float | None
    effective_preferred_size: float | None
    preferred_size_authority: str
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
            "automatic_preferred_size",
            "user_preferred_size",
            "effective_preferred_size",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_preferred_size(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.preferred_size_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "preferred_size_authority must be automatic or user"
            )
        object.__setattr__(self, "preferred_size_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_preferred_size is not None:
                raise ValueError(
                    "automatic preferred-size authority cannot carry a user value"
                )
            if self.effective_preferred_size != self.automatic_preferred_size:
                raise ValueError(
                    "automatic authority must expose the automatic effective preferred size"
                )
        else:
            if self.user_preferred_size is None:
                raise ValueError(
                    "user preferred-size authority requires a user value"
                )
            if self.effective_preferred_size != self.user_preferred_size:
                raise ValueError(
                    "user authority must expose the user effective preferred size"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_preferred_size is not None
            and self.effective_preferred_size is not None
        )
        if eligible and reason:
            raise ValueError(
                "available preferred-size selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable preferred-size selection requires an unavailable reason"
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
            and self.automatic_preferred_size is not None
            and self.effective_preferred_size is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeWorkerCommand:
    """UI carrier with one exact pixel width and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStylePreferredSizeOperation
    preferred_size: float | None
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
        operation = RenderStylePreferredSizeOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStylePreferredSizeOperation.SET:
            object.__setattr__(
                self,
                "preferred_size",
                canonical_render_preferred_size(self.preferred_size),
            )
        elif self.preferred_size is not None:
            raise ValueError(
                "restore_automatic must not carry a preferred_size value"
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
class RenderStylePreferredSizeWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStylePreferredSizeWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStylePreferredSizeOperation
    preferred_size: float | None
    stage: RenderStylePreferredSizeWorkerStage
    message: str = "Preferred-size update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeWorkerFailure:
    code: RenderStylePreferredSizeWorkerFailureCode
    stage: RenderStylePreferredSizeWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStylePreferredSizeOperation
    preferred_size: float | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStylePreferredSizeCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStylePreferredSizeCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStylePreferredSizeWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _PREFERRED_SIZE_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStylePreferredSizeCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStylePreferredSizeCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStylePreferredSizeCommandReceipt"
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
        automatic_preferred_size = canonical_render_preferred_size(
            receipt.automatic_preferred_size,
            field_name="receipt automatic_preferred_size",
        )
        before_preferred_size = canonical_render_preferred_size(
            receipt.before_preferred_size,
            field_name="receipt before_preferred_size",
        )
        after_preferred_size = canonical_render_preferred_size(
            receipt.after_preferred_size,
            field_name="receipt after_preferred_size",
        )
        if receipt.before_preferred_size_authority not in {"automatic", "user"}:
            raise ValueError("preferred-size before authority is invalid")
        if receipt.after_preferred_size_authority not in {"automatic", "user"}:
            raise ValueError("preferred-size after authority is invalid")
        if (
            receipt.before_preferred_size_authority == "automatic"
            and before_preferred_size != automatic_preferred_size
        ):
            raise ValueError("automatic preferred-size before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("preferred-size command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("preferred-size supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("preferred-size commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"preferred_size": after_preferred_size}
                or receipt.after_preferred_size_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed preferred-size set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("preferred_size",)
                or after_preferred_size != automatic_preferred_size
                or receipt.after_preferred_size_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed preferred-size restore"
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
        projected_size = canonical_render_preferred_size(
            overrides.get("preferred_size", automatic_preferred_size),
            field_name="projected preferred_size",
        )
        projected_authority = (
            "user" if "preferred_size" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("preferred-size receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_size = canonical_render_preferred_size(
            receipt_overrides.get(
                "preferred_size",
                automatic_preferred_size,
            ),
            field_name="receipt preferred_size",
        )
        receipt_authority = (
            "user" if "preferred_size" in receipt_overrides else "automatic"
        )
        if (
            projected_size != after_preferred_size
            or projected_authority != receipt.after_preferred_size_authority
            or receipt_size != after_preferred_size
            or receipt_authority != receipt.after_preferred_size_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed preferred size"
            )


@dataclass(frozen=True, slots=True)
class RenderStylePreferredSizeEditorState:
    selection: RenderStylePreferredSizeSelection
    phase: RenderStylePreferredSizeEditorPhase
    draft_preferred_size: float | None
    message: str = ""
    worker_command: RenderStylePreferredSizeWorkerCommand | None = None
    busy_state: RenderStylePreferredSizeWorkerBusyState | None = None
    receipt: RenderStylePreferredSizeWorkerReceipt | None = None
    failure: RenderStylePreferredSizeWorkerFailure | None = None
    cancelled: RenderStylePreferredSizeCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_preferred_size != self.selection.effective_preferred_size

    @property
    def valid(self) -> bool:
        if self.draft_preferred_size is None:
            return False
        try:
            canonical_render_preferred_size(self.draft_preferred_size)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStylePreferredSizeEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStylePreferredSizeEditorPhase.STALE

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
            and self.selection.preferred_size_authority == "user"
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
                RenderStylePreferredSizeEditorPhase.READY,
                RenderStylePreferredSizeEditorPhase.COMMITTED,
                RenderStylePreferredSizeEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStylePreferredSizeEditorPhase.READY: "muted",
            RenderStylePreferredSizeEditorPhase.DIRTY: "editing",
            RenderStylePreferredSizeEditorPhase.COMMITTING: "editing",
            RenderStylePreferredSizeEditorPhase.COMMITTED: "ready",
            RenderStylePreferredSizeEditorPhase.CANCELLED: "muted",
            RenderStylePreferredSizeEditorPhase.STALE: "warning",
            RenderStylePreferredSizeEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStylePreferredSizeEditorModel:
    """UI-thread reducer for one exact selected-parent preferred-size ratio."""

    def __init__(self, selection: RenderStylePreferredSizeSelection) -> None:
        if not isinstance(selection, RenderStylePreferredSizeSelection):
            raise TypeError(
                "selection must be RenderStylePreferredSizeSelection"
            )
        self._state = RenderStylePreferredSizeEditorState(
            selection=selection,
            phase=RenderStylePreferredSizeEditorPhase.READY,
            draft_preferred_size=selection.effective_preferred_size,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStylePreferredSizeEditorState:
        return self._state

    def set_draft_preferred_size(
        self,
        value: float,
    ) -> RenderStylePreferredSizeEditorState:
        preferred_size = canonical_render_preferred_size(
            value,
            field_name="preferred-size draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("preferred-size draft is not editable")
        phase = (
            RenderStylePreferredSizeEditorPhase.DIRTY
            if preferred_size != self._state.selection.effective_preferred_size
            else RenderStylePreferredSizeEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_preferred_size=preferred_size,
            message=(
                "Preferred size has an unapplied change."
                if phase is RenderStylePreferredSizeEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStylePreferredSizeEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard preferred size while it is committing")
        phase = (
            RenderStylePreferredSizeEditorPhase.STALE
            if self._state.stale
            else RenderStylePreferredSizeEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_preferred_size=self._state.selection.effective_preferred_size,
            message=(
                "Reload the selected parent before editing preferred size."
                if phase is RenderStylePreferredSizeEditorPhase.STALE
                else "Preferred-size draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStylePreferredSizeEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStylePreferredSizeWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable preferred-size draft")
        preferred_size = self._state.draft_preferred_size
        if preferred_size is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("preferred-size draft is unavailable")
        command = self._command(
            RenderStylePreferredSizeOperation.SET,
            preferred_size=preferred_size,
        )
        self._begin(command, "Applying preferred-size edit...")
        return command

    def begin_restore(self) -> RenderStylePreferredSizeWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic preferred size is already effective")
        command = self._command(
            RenderStylePreferredSizeOperation.RESTORE_AUTOMATIC,
            preferred_size=None,
        )
        self._begin(command, "Restoring automatic preferred size...")
        return command

    def accept_busy(
        self,
        value: RenderStylePreferredSizeWorkerBusyState,
    ) -> RenderStylePreferredSizeEditorState:
        if not isinstance(value, RenderStylePreferredSizeWorkerBusyState):
            raise TypeError(
                "value must be RenderStylePreferredSizeWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStylePreferredSizeEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStylePreferredSizeWorkerReceipt,
    ) -> RenderStylePreferredSizeEditorState:
        if not isinstance(value, RenderStylePreferredSizeWorkerReceipt):
            raise TypeError(
                "value must be RenderStylePreferredSizeWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStylePreferredSizeOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("preferred-size receipt has another operation")
        if command.operation is RenderStylePreferredSizeOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"preferred_size": command.preferred_size}
                or receipt.after_preferred_size != command.preferred_size
            ):
                raise ValueError("preferred-size receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("preferred_size",)
                or receipt.after_preferred_size != receipt.automatic_preferred_size
            ):
                raise ValueError("preferred-size receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("preferred-size receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_preferred_size != selection.automatic_preferred_size
            or receipt.before_preferred_size != selection.effective_preferred_size
            or receipt.before_preferred_size_authority
            != selection.preferred_size_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("preferred-size receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_preferred_size=(
                receipt.after_preferred_size
                if receipt.after_preferred_size_authority == "user"
                else None
            ),
            effective_preferred_size=receipt.after_preferred_size,
            preferred_size_authority=receipt.after_preferred_size_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStylePreferredSizeEditorState(
            selection=updated_selection,
            phase=RenderStylePreferredSizeEditorPhase.COMMITTED,
            draft_preferred_size=receipt.after_preferred_size,
            message="Preferred size saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStylePreferredSizeWorkerFailure,
    ) -> RenderStylePreferredSizeEditorState:
        if not isinstance(value, RenderStylePreferredSizeWorkerFailure):
            raise TypeError(
                "value must be RenderStylePreferredSizeWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.preferred_size,
        )
        phase = (
            RenderStylePreferredSizeEditorPhase.STALE
            if value.stale
            else RenderStylePreferredSizeEditorPhase.FAILED
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
        value: RenderStylePreferredSizeWorkerFailure,
    ) -> RenderStylePreferredSizeEditorState:
        if not value.stale:
            raise ValueError("preferred-size failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStylePreferredSizeCancelledReceipt,
    ) -> RenderStylePreferredSizeEditorState:
        if not isinstance(value, RenderStylePreferredSizeCancelledReceipt):
            raise TypeError(
                "value must be RenderStylePreferredSizeCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.preferred_size,
        )
        self._state = replace(
            self._state,
            phase=RenderStylePreferredSizeEditorPhase.CANCELLED,
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
        selection: RenderStylePreferredSizeSelection,
    ) -> RenderStylePreferredSizeEditorState:
        if not isinstance(selection, RenderStylePreferredSizeSelection):
            raise TypeError(
                "selection must be RenderStylePreferredSizeSelection"
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
            self._state.draft_preferred_size
            if preserve_draft
            else selection.effective_preferred_size
        )
        phase = (
            RenderStylePreferredSizeEditorPhase.DIRTY
            if draft != selection.effective_preferred_size
            else RenderStylePreferredSizeEditorPhase.READY
        )
        self._state = RenderStylePreferredSizeEditorState(
            selection=selection,
            phase=phase,
            draft_preferred_size=draft,
            message=(
                "Current state changed; review the preserved preferred-size draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied preferred-size draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStylePreferredSizeOperation,
        *,
        preferred_size: float | None,
    ) -> RenderStylePreferredSizeWorkerCommand:
        selection = self._state.selection
        return RenderStylePreferredSizeWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            preferred_size=preferred_size,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStylePreferredSizeWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStylePreferredSizeEditorPhase.COMMITTING,
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
    ) -> RenderStylePreferredSizeWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no preferred-size worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStylePreferredSizeOperation,
        preferred_size: float | None,
    ) -> RenderStylePreferredSizeWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.preferred_size != preferred_size
        ):
            raise ValueError("worker event belongs to another preferred-size command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStylePreferredSizeSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.preferred_size_authority == "user":
            return "User preferred size is effective."
        return "Automatic preferred size is effective."

__all__ = [
    "RenderStylePreferredSizeCancellationState",
    "RenderStylePreferredSizeCancelledReceipt",
    "RenderStylePreferredSizeEditorModel",
    "RenderStylePreferredSizeEditorPhase",
    "RenderStylePreferredSizeEditorState",
    "RenderStylePreferredSizeSelection",
    "RenderStylePreferredSizeWorkerBusyState",
    "RenderStylePreferredSizeWorkerCommand",
    "RenderStylePreferredSizeWorkerFailure",
    "RenderStylePreferredSizeWorkerFailureCode",
    "RenderStylePreferredSizeWorkerReceipt",
    "RenderStylePreferredSizeWorkerStage",
]
