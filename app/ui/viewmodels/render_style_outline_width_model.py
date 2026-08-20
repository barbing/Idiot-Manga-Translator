# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent outline-width commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    RenderStyleOutlineWidthCommandErrorCode,
    RenderStyleOutlineWidthCommandReceipt,
    RenderStyleOutlineWidthOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_outline_width,
    thaw_json,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_OUTLINE_WIDTH_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleOutlineWidthCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleOutlineWidthCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleOutlineWidthCommandErrorCode.STALE_GLOBAL_HEAD,
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

class RenderStyleOutlineWidthEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleOutlineWidthWorkerStage(str, Enum):
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


class RenderStyleOutlineWidthWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE = "automatic_outline_width_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    OUTLINE_WIDTH_SLOT_CONFLICT = "outline_width_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_outline_width(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_render_outline_width(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthSelection:
    """Exact selected-parent outline-width state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_outline_width: float | None
    user_outline_width: float | None
    effective_outline_width: float | None
    outline_width_authority: str
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
            "automatic_outline_width",
            "user_outline_width",
            "effective_outline_width",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_outline_width(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.outline_width_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "outline_width_authority must be automatic or user"
            )
        object.__setattr__(self, "outline_width_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_outline_width is not None:
                raise ValueError(
                    "automatic outline-width authority cannot carry a user value"
                )
            if self.effective_outline_width != self.automatic_outline_width:
                raise ValueError(
                    "automatic authority must expose the automatic effective outline width"
                )
        else:
            if self.user_outline_width is None:
                raise ValueError(
                    "user outline-width authority requires a user value"
                )
            if self.effective_outline_width != self.user_outline_width:
                raise ValueError(
                    "user authority must expose the user effective outline width"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_outline_width is not None
            and self.effective_outline_width is not None
        )
        if eligible and reason:
            raise ValueError(
                "available outline-width selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable outline-width selection requires an unavailable reason"
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
            and self.automatic_outline_width is not None
            and self.effective_outline_width is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthWorkerCommand:
    """UI carrier with one exact pixel width and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineWidthOperation
    outline_width: float | None
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
        operation = RenderStyleOutlineWidthOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleOutlineWidthOperation.SET:
            object.__setattr__(
                self,
                "outline_width",
                canonical_render_outline_width(self.outline_width),
            )
        elif self.outline_width is not None:
            raise ValueError(
                "restore_automatic must not carry an outline_width value"
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
class RenderStyleOutlineWidthWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleOutlineWidthWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineWidthOperation
    outline_width: float | None
    stage: RenderStyleOutlineWidthWorkerStage
    message: str = "Outline-width update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthWorkerFailure:
    code: RenderStyleOutlineWidthWorkerFailureCode
    stage: RenderStyleOutlineWidthWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineWidthOperation
    outline_width: float | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleOutlineWidthCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleOutlineWidthCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleOutlineWidthWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _OUTLINE_WIDTH_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleOutlineWidthCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleOutlineWidthCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleOutlineWidthCommandReceipt"
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
        automatic_outline_width = canonical_render_outline_width(
            receipt.automatic_outline_width,
            field_name="receipt automatic_outline_width",
        )
        before_outline_width = canonical_render_outline_width(
            receipt.before_outline_width,
            field_name="receipt before_outline_width",
        )
        after_outline_width = canonical_render_outline_width(
            receipt.after_outline_width,
            field_name="receipt after_outline_width",
        )
        if receipt.before_outline_width_authority not in {"automatic", "user"}:
            raise ValueError("outline-width before authority is invalid")
        if receipt.after_outline_width_authority not in {"automatic", "user"}:
            raise ValueError("outline-width after authority is invalid")
        if (
            receipt.before_outline_width_authority == "automatic"
            and before_outline_width != automatic_outline_width
        ):
            raise ValueError("automatic outline-width before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("outline-width command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("outline-width supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("outline-width commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"outline_width": after_outline_width}
                or receipt.after_outline_width_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed outline-width set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("outline_width",)
                or after_outline_width != automatic_outline_width
                or receipt.after_outline_width_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed outline-width restore"
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
        projected_width = canonical_render_outline_width(
            overrides.get("outline_width", automatic_outline_width),
            field_name="projected outline_width",
        )
        projected_authority = (
            "user" if "outline_width" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("outline-width receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_width = canonical_render_outline_width(
            receipt_overrides.get(
                "outline_width",
                automatic_outline_width,
            ),
            field_name="receipt outline_width",
        )
        receipt_authority = (
            "user" if "outline_width" in receipt_overrides else "automatic"
        )
        if (
            projected_width != after_outline_width
            or projected_authority != receipt.after_outline_width_authority
            or receipt_width != after_outline_width
            or receipt_authority != receipt.after_outline_width_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed outline width"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineWidthEditorState:
    selection: RenderStyleOutlineWidthSelection
    phase: RenderStyleOutlineWidthEditorPhase
    draft_outline_width: float | None
    message: str = ""
    worker_command: RenderStyleOutlineWidthWorkerCommand | None = None
    busy_state: RenderStyleOutlineWidthWorkerBusyState | None = None
    receipt: RenderStyleOutlineWidthWorkerReceipt | None = None
    failure: RenderStyleOutlineWidthWorkerFailure | None = None
    cancelled: RenderStyleOutlineWidthCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_outline_width != self.selection.effective_outline_width

    @property
    def valid(self) -> bool:
        if self.draft_outline_width is None:
            return False
        try:
            canonical_render_outline_width(self.draft_outline_width)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleOutlineWidthEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleOutlineWidthEditorPhase.STALE

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
            and self.selection.outline_width_authority == "user"
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
                RenderStyleOutlineWidthEditorPhase.READY,
                RenderStyleOutlineWidthEditorPhase.COMMITTED,
                RenderStyleOutlineWidthEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleOutlineWidthEditorPhase.READY: "muted",
            RenderStyleOutlineWidthEditorPhase.DIRTY: "editing",
            RenderStyleOutlineWidthEditorPhase.COMMITTING: "editing",
            RenderStyleOutlineWidthEditorPhase.COMMITTED: "ready",
            RenderStyleOutlineWidthEditorPhase.CANCELLED: "muted",
            RenderStyleOutlineWidthEditorPhase.STALE: "warning",
            RenderStyleOutlineWidthEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleOutlineWidthEditorModel:
    """UI-thread reducer for one exact selected-parent outline-width ratio."""

    def __init__(self, selection: RenderStyleOutlineWidthSelection) -> None:
        if not isinstance(selection, RenderStyleOutlineWidthSelection):
            raise TypeError(
                "selection must be RenderStyleOutlineWidthSelection"
            )
        self._state = RenderStyleOutlineWidthEditorState(
            selection=selection,
            phase=RenderStyleOutlineWidthEditorPhase.READY,
            draft_outline_width=selection.effective_outline_width,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleOutlineWidthEditorState:
        return self._state

    def set_draft_outline_width(
        self,
        value: float,
    ) -> RenderStyleOutlineWidthEditorState:
        outline_width = canonical_render_outline_width(
            value,
            field_name="outline-width draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("outline-width draft is not editable")
        phase = (
            RenderStyleOutlineWidthEditorPhase.DIRTY
            if outline_width != self._state.selection.effective_outline_width
            else RenderStyleOutlineWidthEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_outline_width=outline_width,
            message=(
                "Outline width has an unapplied change."
                if phase is RenderStyleOutlineWidthEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleOutlineWidthEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard outline width while it is committing")
        phase = (
            RenderStyleOutlineWidthEditorPhase.STALE
            if self._state.stale
            else RenderStyleOutlineWidthEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_outline_width=self._state.selection.effective_outline_width,
            message=(
                "Reload the selected parent before editing outline width."
                if phase is RenderStyleOutlineWidthEditorPhase.STALE
                else "Outline-width draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleOutlineWidthEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleOutlineWidthWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable outline-width draft")
        outline_width = self._state.draft_outline_width
        if outline_width is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("outline-width draft is unavailable")
        command = self._command(
            RenderStyleOutlineWidthOperation.SET,
            outline_width=outline_width,
        )
        self._begin(command, "Applying outline-width edit...")
        return command

    def begin_restore(self) -> RenderStyleOutlineWidthWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic outline width is already effective")
        command = self._command(
            RenderStyleOutlineWidthOperation.RESTORE_AUTOMATIC,
            outline_width=None,
        )
        self._begin(command, "Restoring automatic outline width...")
        return command

    def accept_busy(
        self,
        value: RenderStyleOutlineWidthWorkerBusyState,
    ) -> RenderStyleOutlineWidthEditorState:
        if not isinstance(value, RenderStyleOutlineWidthWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleOutlineWidthWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleOutlineWidthEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleOutlineWidthWorkerReceipt,
    ) -> RenderStyleOutlineWidthEditorState:
        if not isinstance(value, RenderStyleOutlineWidthWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleOutlineWidthWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleOutlineWidthOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("outline-width receipt has another operation")
        if command.operation is RenderStyleOutlineWidthOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"outline_width": command.outline_width}
                or receipt.after_outline_width != command.outline_width
            ):
                raise ValueError("outline-width receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("outline_width",)
                or receipt.after_outline_width != receipt.automatic_outline_width
            ):
                raise ValueError("outline-width receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("outline-width receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_outline_width != selection.automatic_outline_width
            or receipt.before_outline_width != selection.effective_outline_width
            or receipt.before_outline_width_authority
            != selection.outline_width_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("outline-width receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_outline_width=(
                receipt.after_outline_width
                if receipt.after_outline_width_authority == "user"
                else None
            ),
            effective_outline_width=receipt.after_outline_width,
            outline_width_authority=receipt.after_outline_width_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleOutlineWidthEditorState(
            selection=updated_selection,
            phase=RenderStyleOutlineWidthEditorPhase.COMMITTED,
            draft_outline_width=receipt.after_outline_width,
            message="Outline width saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleOutlineWidthWorkerFailure,
    ) -> RenderStyleOutlineWidthEditorState:
        if not isinstance(value, RenderStyleOutlineWidthWorkerFailure):
            raise TypeError(
                "value must be RenderStyleOutlineWidthWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.outline_width,
        )
        phase = (
            RenderStyleOutlineWidthEditorPhase.STALE
            if value.stale
            else RenderStyleOutlineWidthEditorPhase.FAILED
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
        value: RenderStyleOutlineWidthWorkerFailure,
    ) -> RenderStyleOutlineWidthEditorState:
        if not value.stale:
            raise ValueError("outline-width failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleOutlineWidthCancelledReceipt,
    ) -> RenderStyleOutlineWidthEditorState:
        if not isinstance(value, RenderStyleOutlineWidthCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleOutlineWidthCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.outline_width,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleOutlineWidthEditorPhase.CANCELLED,
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
        selection: RenderStyleOutlineWidthSelection,
    ) -> RenderStyleOutlineWidthEditorState:
        if not isinstance(selection, RenderStyleOutlineWidthSelection):
            raise TypeError(
                "selection must be RenderStyleOutlineWidthSelection"
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
            self._state.draft_outline_width
            if preserve_draft
            else selection.effective_outline_width
        )
        phase = (
            RenderStyleOutlineWidthEditorPhase.DIRTY
            if draft != selection.effective_outline_width
            else RenderStyleOutlineWidthEditorPhase.READY
        )
        self._state = RenderStyleOutlineWidthEditorState(
            selection=selection,
            phase=phase,
            draft_outline_width=draft,
            message=(
                "Current state changed; review the preserved outline-width draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied outline-width draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleOutlineWidthOperation,
        *,
        outline_width: float | None,
    ) -> RenderStyleOutlineWidthWorkerCommand:
        selection = self._state.selection
        return RenderStyleOutlineWidthWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            outline_width=outline_width,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleOutlineWidthWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleOutlineWidthEditorPhase.COMMITTING,
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
    ) -> RenderStyleOutlineWidthWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no outline-width worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleOutlineWidthOperation,
        outline_width: float | None,
    ) -> RenderStyleOutlineWidthWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.outline_width != outline_width
        ):
            raise ValueError("worker event belongs to another outline-width command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleOutlineWidthSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.outline_width_authority == "user":
            return "User outline width is effective."
        return "Automatic outline width is effective."

__all__ = [
    "RenderStyleOutlineWidthCancellationState",
    "RenderStyleOutlineWidthCancelledReceipt",
    "RenderStyleOutlineWidthEditorModel",
    "RenderStyleOutlineWidthEditorPhase",
    "RenderStyleOutlineWidthEditorState",
    "RenderStyleOutlineWidthSelection",
    "RenderStyleOutlineWidthWorkerBusyState",
    "RenderStyleOutlineWidthWorkerCommand",
    "RenderStyleOutlineWidthWorkerFailure",
    "RenderStyleOutlineWidthWorkerFailureCode",
    "RenderStyleOutlineWidthWorkerReceipt",
    "RenderStyleOutlineWidthWorkerStage",
]
