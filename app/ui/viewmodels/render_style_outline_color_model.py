# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent opaque outline-color edits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Any, Mapping

from app.project_edits.commands import (
    RenderStyleOutlineColorCommandErrorCode,
    RenderStyleOutlineColorCommandReceipt,
    RenderStyleOutlineColorOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_outline_color,
)
from app.ui.viewmodels.editor_command_model import (
    _required_identity,
    _required_sha256,
)

_OUTLINE_COLOR_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleOutlineColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleOutlineColorCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleOutlineColorCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)

class RenderStyleOutlineColorEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleOutlineColorWorkerStage(str, Enum):
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


class RenderStyleOutlineColorWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_OUTLINE_COLOR_UNAVAILABLE = "automatic_outline_color_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    OUTLINE_COLOR_SLOT_CONFLICT = "outline_color_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_outline_color(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return canonical_render_outline_color(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorSelection:
    """Canonical selected-parent outline color state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_outline_color: str | None
    user_outline_color: str | None
    effective_outline_color: str | None
    outline_color_authority: str
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
            "automatic_outline_color",
            "user_outline_color",
            "effective_outline_color",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_outline_color(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.outline_color_authority or "")
        if authority not in {"automatic", "user", "unresolved"}:
            raise ValueError(
                "outline_color_authority must be automatic, user, or unresolved"
            )
        object.__setattr__(self, "outline_color_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_outline_color is not None:
                raise ValueError(
                    "automatic outline_color authority cannot carry a user value"
                )
            if self.effective_outline_color != self.automatic_outline_color:
                raise ValueError(
                    "automatic authority must expose the automatic effective outline_color"
                )
        elif authority == "user":
            if self.user_outline_color is None:
                raise ValueError(
                    "user outline_color authority requires a user value"
                )
            if self.effective_outline_color != self.user_outline_color:
                raise ValueError(
                    "user authority must expose the user effective outline_color"
                )
        else:
            if (
                self.user_outline_color is not None
                or self.effective_outline_color is not None
            ):
                raise ValueError(
                    "unresolved outline_color authority cannot expose a canonical user or effective value"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_outline_color is not None
            and self.effective_outline_color is not None
        )
        if eligible and reason:
            raise ValueError(
                "available outline_color selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable outline_color selection requires an unavailable reason"
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
            and self.automatic_outline_color is not None
            and self.effective_outline_color is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorWorkerCommand:
    """UI carrier with one opaque outline color and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineColorOperation
    outline_color: str | None
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
        operation = RenderStyleOutlineColorOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleOutlineColorOperation.SET:
            object.__setattr__(
                self,
                "outline_color",
                canonical_render_outline_color(self.outline_color),
            )
        elif self.outline_color is not None:
            raise ValueError(
                "restore_automatic must not carry a outline_color value"
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
class RenderStyleOutlineColorWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleOutlineColorWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineColorOperation
    outline_color: str | None
    stage: RenderStyleOutlineColorWorkerStage
    message: str = "Outline color update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorWorkerFailure:
    code: RenderStyleOutlineColorWorkerFailureCode
    stage: RenderStyleOutlineColorWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleOutlineColorOperation
    outline_color: str | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleOutlineColorCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleOutlineColorCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleOutlineColorWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _OUTLINE_COLOR_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleOutlineColorCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleOutlineColorCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleOutlineColorCommandReceipt"
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
        automatic_outline_color = canonical_render_outline_color(
            receipt.automatic_outline_color,
            field_name="receipt automatic_outline_color",
        )
        before_outline_color = canonical_render_outline_color(
            receipt.before_outline_color,
            field_name="receipt before_outline_color",
        )
        after_outline_color = canonical_render_outline_color(
            receipt.after_outline_color,
            field_name="receipt after_outline_color",
        )
        if receipt.before_outline_color_authority not in {"automatic", "user"}:
            raise ValueError("outline_color before authority is invalid")
        if receipt.after_outline_color_authority not in {"automatic", "user"}:
            raise ValueError("outline_color after authority is invalid")
        if (
            receipt.before_outline_color_authority == "automatic"
            and before_outline_color != automatic_outline_color
        ):
            raise ValueError("automatic outline_color before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("outline_color command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("outline_color supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("outline_color commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"outline_color": after_outline_color}
                or receipt.after_outline_color_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed outline_color set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("outline_color",)
                or after_outline_color != automatic_outline_color
                or receipt.after_outline_color_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed outline_color restore"
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
        projected_outline_color = canonical_render_outline_color(
            overrides.get("outline_color", automatic_outline_color),
            field_name="projected outline_color",
        )
        projected_authority = (
            "user" if "outline_color" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("outline_color receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_outline_color = canonical_render_outline_color(
            receipt_overrides.get(
                "outline_color",
                automatic_outline_color,
            ),
            field_name="receipt outline_color",
        )
        receipt_authority = (
            "user" if "outline_color" in receipt_overrides else "automatic"
        )
        if (
            projected_outline_color != after_outline_color
            or projected_authority != receipt.after_outline_color_authority
            or receipt_outline_color != after_outline_color
            or receipt_authority != receipt.after_outline_color_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed outline_color"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleOutlineColorEditorState:
    selection: RenderStyleOutlineColorSelection
    phase: RenderStyleOutlineColorEditorPhase
    draft_outline_color: str | None
    message: str = ""
    worker_command: RenderStyleOutlineColorWorkerCommand | None = None
    busy_state: RenderStyleOutlineColorWorkerBusyState | None = None
    receipt: RenderStyleOutlineColorWorkerReceipt | None = None
    failure: RenderStyleOutlineColorWorkerFailure | None = None
    cancelled: RenderStyleOutlineColorCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        if self.draft_outline_color is None:
            return self.selection.effective_outline_color is not None
        try:
            canonical = canonical_render_outline_color(self.draft_outline_color)
        except (TypeError, ValueError):
            return True
        baseline = (
            self.selection.automatic_outline_color
            if self.selection.outline_color_authority == "unresolved"
            else self.selection.effective_outline_color
        )
        return canonical != baseline

    @property
    def valid(self) -> bool:
        if self.draft_outline_color is None:
            return False
        try:
            canonical_render_outline_color(self.draft_outline_color)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleOutlineColorEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleOutlineColorEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return bool(
            not self.selection.excluded
            and self.selection.render_required
            and self.selection.automatic_outline_color is not None
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
            and self.selection.outline_color_authority in {"user", "unresolved"}
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
            and self.selection.outline_color_authority != "unresolved"
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
                RenderStyleOutlineColorEditorPhase.READY,
                RenderStyleOutlineColorEditorPhase.COMMITTED,
                RenderStyleOutlineColorEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleOutlineColorEditorPhase.READY: "muted",
            RenderStyleOutlineColorEditorPhase.DIRTY: "editing",
            RenderStyleOutlineColorEditorPhase.COMMITTING: "editing",
            RenderStyleOutlineColorEditorPhase.COMMITTED: "ready",
            RenderStyleOutlineColorEditorPhase.CANCELLED: "muted",
            RenderStyleOutlineColorEditorPhase.STALE: "warning",
            RenderStyleOutlineColorEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleOutlineColorEditorModel:
    """UI-thread reducer for one exact selected-parent opaque outline color."""

    def __init__(self, selection: RenderStyleOutlineColorSelection) -> None:
        if not isinstance(selection, RenderStyleOutlineColorSelection):
            raise TypeError(
                "selection must be RenderStyleOutlineColorSelection"
            )
        self._state = RenderStyleOutlineColorEditorState(
            selection=selection,
            phase=RenderStyleOutlineColorEditorPhase.READY,
            draft_outline_color=selection.effective_outline_color,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleOutlineColorEditorState:
        return self._state

    def set_draft_outline_color(
        self,
        value: str,
    ) -> RenderStyleOutlineColorEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("outline_color draft is not editable")
        if not isinstance(value, str):
            raise TypeError("outline_color draft must be a string")
        try:
            canonical = canonical_render_outline_color(
                value,
                field_name="outline_color draft",
            )
        except (TypeError, ValueError):
            canonical = None
        baseline = (
            self._state.selection.automatic_outline_color
            if self._state.selection.outline_color_authority == "unresolved"
            else self._state.selection.effective_outline_color
        )
        dirty = canonical != baseline if canonical is not None else True
        phase = (
            RenderStyleOutlineColorEditorPhase.DIRTY
            if dirty
            else RenderStyleOutlineColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_outline_color=value,
            message=(
                "Outline color must use exactly #RRGGBB."
                if canonical is None
                else "Outline color has an unapplied change."
                if phase is RenderStyleOutlineColorEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleOutlineColorEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard outline_color while it is committing")
        phase = (
            RenderStyleOutlineColorEditorPhase.STALE
            if self._state.stale
            else RenderStyleOutlineColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_outline_color=self._state.selection.effective_outline_color,
            message=(
                "Reload the selected parent before editing outline_color."
                if phase is RenderStyleOutlineColorEditorPhase.STALE
                else "Outline color draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleOutlineColorEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleOutlineColorWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable outline_color draft")
        outline_color = self._state.draft_outline_color
        if outline_color is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("outline_color draft is unavailable")
        outline_color = canonical_render_outline_color(
            outline_color,
            field_name="outline_color draft",
        )
        command = self._command(
            RenderStyleOutlineColorOperation.SET,
            outline_color=outline_color,
        )
        self._begin(command, "Applying outline_color edit...")
        return command

    def begin_restore(self) -> RenderStyleOutlineColorWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic outline_color is already effective")
        command = self._command(
            RenderStyleOutlineColorOperation.RESTORE_AUTOMATIC,
            outline_color=None,
        )
        self._begin(command, "Restoring automatic outline_color...")
        return command

    def accept_busy(
        self,
        value: RenderStyleOutlineColorWorkerBusyState,
    ) -> RenderStyleOutlineColorEditorState:
        if not isinstance(value, RenderStyleOutlineColorWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleOutlineColorWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleOutlineColorEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleOutlineColorWorkerReceipt,
    ) -> RenderStyleOutlineColorEditorState:
        if not isinstance(value, RenderStyleOutlineColorWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleOutlineColorWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleOutlineColorOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("outline_color receipt has another operation")
        if command.operation is RenderStyleOutlineColorOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"outline_color": command.outline_color}
                or receipt.after_outline_color != command.outline_color
            ):
                raise ValueError("outline_color receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("outline_color",)
                or receipt.after_outline_color != receipt.automatic_outline_color
            ):
                raise ValueError("outline_color receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("outline_color receipt has another base revision")
        selection = self._state.selection
        if selection.outline_color_authority == "unresolved":
            before_matches_selection = bool(
                receipt.before_outline_color == selection.automatic_outline_color
                and receipt.before_outline_color_authority == "automatic"
            )
        else:
            before_matches_selection = bool(
                receipt.before_outline_color == selection.effective_outline_color
                and receipt.before_outline_color_authority
                == selection.outline_color_authority
            )
        if (
            receipt.automatic_outline_color != selection.automatic_outline_color
            or not before_matches_selection
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("outline_color receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_outline_color=(
                receipt.after_outline_color
                if receipt.after_outline_color_authority == "user"
                else None
            ),
            effective_outline_color=receipt.after_outline_color,
            outline_color_authority=receipt.after_outline_color_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleOutlineColorEditorState(
            selection=updated_selection,
            phase=RenderStyleOutlineColorEditorPhase.COMMITTED,
            draft_outline_color=receipt.after_outline_color,
            message="Outline color saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleOutlineColorWorkerFailure,
    ) -> RenderStyleOutlineColorEditorState:
        if not isinstance(value, RenderStyleOutlineColorWorkerFailure):
            raise TypeError(
                "value must be RenderStyleOutlineColorWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.outline_color,
        )
        phase = (
            RenderStyleOutlineColorEditorPhase.STALE
            if value.stale
            else RenderStyleOutlineColorEditorPhase.FAILED
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
        value: RenderStyleOutlineColorWorkerFailure,
    ) -> RenderStyleOutlineColorEditorState:
        if not value.stale:
            raise ValueError("outline_color failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleOutlineColorCancelledReceipt,
    ) -> RenderStyleOutlineColorEditorState:
        if not isinstance(value, RenderStyleOutlineColorCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleOutlineColorCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.outline_color,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleOutlineColorEditorPhase.CANCELLED,
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
        selection: RenderStyleOutlineColorSelection,
    ) -> RenderStyleOutlineColorEditorState:
        if not isinstance(selection, RenderStyleOutlineColorSelection):
            raise TypeError(
                "selection must be RenderStyleOutlineColorSelection"
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
            self._state.draft_outline_color
            if preserve_draft
            else selection.effective_outline_color
        )
        if draft is None:
            draft_dirty = selection.effective_outline_color is not None
        else:
            try:
                canonical_draft = canonical_render_outline_color(draft)
            except (TypeError, ValueError):
                draft_dirty = True
            else:
                baseline = (
                    selection.automatic_outline_color
                    if selection.outline_color_authority == "unresolved"
                    else selection.effective_outline_color
                )
                draft_dirty = canonical_draft != baseline
        phase = (
            RenderStyleOutlineColorEditorPhase.DIRTY
            if draft_dirty
            else RenderStyleOutlineColorEditorPhase.READY
        )
        self._state = RenderStyleOutlineColorEditorState(
            selection=selection,
            phase=phase,
            draft_outline_color=draft,
            message=(
                "Current state changed; review the preserved outline_color draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied outline_color draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleOutlineColorOperation,
        *,
        outline_color: str | None,
    ) -> RenderStyleOutlineColorWorkerCommand:
        selection = self._state.selection
        return RenderStyleOutlineColorWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            outline_color=outline_color,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleOutlineColorWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleOutlineColorEditorPhase.COMMITTING,
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
    ) -> RenderStyleOutlineColorWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no outline_color worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleOutlineColorOperation,
        outline_color: str | None,
    ) -> RenderStyleOutlineColorWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.outline_color != outline_color
        ):
            raise ValueError("worker event belongs to another outline_color command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleOutlineColorSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.outline_color_authority == "user":
            return "User outline color is effective."
        return "Automatic outline color is effective."


__all__ = [
    "RenderStyleOutlineColorCancellationState",
    "RenderStyleOutlineColorCancelledReceipt",
    "RenderStyleOutlineColorEditorModel",
    "RenderStyleOutlineColorEditorPhase",
    "RenderStyleOutlineColorEditorState",
    "RenderStyleOutlineColorSelection",
    "RenderStyleOutlineColorWorkerBusyState",
    "RenderStyleOutlineColorWorkerCommand",
    "RenderStyleOutlineColorWorkerFailure",
    "RenderStyleOutlineColorWorkerFailureCode",
    "RenderStyleOutlineColorWorkerReceipt",
    "RenderStyleOutlineColorWorkerStage",
]
