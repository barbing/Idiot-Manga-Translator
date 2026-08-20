# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent RGBA shadow-color edits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Any, Mapping

from app.project_edits.commands import (
    RenderStyleShadowColorCommandErrorCode,
    RenderStyleShadowColorCommandReceipt,
    RenderStyleShadowColorOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_shadow_color,
)
from app.ui.viewmodels.editor_command_model import (
    _required_identity,
    _required_sha256,
)

_SHADOW_COLOR_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleShadowColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleShadowColorCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleShadowColorCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)

class RenderStyleShadowColorEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleShadowColorWorkerStage(str, Enum):
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


class RenderStyleShadowColorWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_SHADOW_COLOR_UNAVAILABLE = "automatic_shadow_color_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    SHADOW_COLOR_SLOT_CONFLICT = "shadow_color_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_shadow_color(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return canonical_render_shadow_color(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorSelection:
    """Canonical selected-parent shadow color state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_shadow_color: str | None
    user_shadow_color: str | None
    effective_shadow_color: str | None
    shadow_color_authority: str
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
            "automatic_shadow_color",
            "user_shadow_color",
            "effective_shadow_color",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_shadow_color(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.shadow_color_authority or "")
        if authority not in {"automatic", "user", "unresolved"}:
            raise ValueError(
                "shadow_color_authority must be automatic, user, or unresolved"
            )
        object.__setattr__(self, "shadow_color_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_shadow_color is not None:
                raise ValueError(
                    "automatic shadow_color authority cannot carry a user value"
                )
            if self.effective_shadow_color != self.automatic_shadow_color:
                raise ValueError(
                    "automatic authority must expose the automatic effective shadow_color"
                )
        elif authority == "user":
            if self.user_shadow_color is None:
                raise ValueError(
                    "user shadow_color authority requires a user value"
                )
            if self.effective_shadow_color != self.user_shadow_color:
                raise ValueError(
                    "user authority must expose the user effective shadow_color"
                )
        else:
            if (
                self.user_shadow_color is not None
                or self.effective_shadow_color is not None
            ):
                raise ValueError(
                    "unresolved shadow_color authority cannot expose a canonical user or effective value"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_shadow_color is not None
            and self.effective_shadow_color is not None
        )
        if eligible and reason:
            raise ValueError(
                "available shadow_color selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable shadow_color selection requires an unavailable reason"
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
            and self.automatic_shadow_color is not None
            and self.effective_shadow_color is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorWorkerCommand:
    """UI carrier with one RGBA shadow color and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowColorOperation
    shadow_color: str | None
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
        operation = RenderStyleShadowColorOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowColorOperation.SET:
            object.__setattr__(
                self,
                "shadow_color",
                canonical_render_shadow_color(self.shadow_color),
            )
        elif self.shadow_color is not None:
            raise ValueError(
                "restore_automatic must not carry a shadow_color value"
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
class RenderStyleShadowColorWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleShadowColorWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowColorOperation
    shadow_color: str | None
    stage: RenderStyleShadowColorWorkerStage
    message: str = "Shadow color update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorWorkerFailure:
    code: RenderStyleShadowColorWorkerFailureCode
    stage: RenderStyleShadowColorWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowColorOperation
    shadow_color: str | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleShadowColorCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleShadowColorCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleShadowColorWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SHADOW_COLOR_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleShadowColorCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleShadowColorCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleShadowColorCommandReceipt"
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
        automatic_shadow_color = canonical_render_shadow_color(
            receipt.automatic_shadow_color,
            field_name="receipt automatic_shadow_color",
        )
        before_shadow_color = canonical_render_shadow_color(
            receipt.before_shadow_color,
            field_name="receipt before_shadow_color",
        )
        after_shadow_color = canonical_render_shadow_color(
            receipt.after_shadow_color,
            field_name="receipt after_shadow_color",
        )
        if receipt.before_shadow_color_authority not in {"automatic", "user"}:
            raise ValueError("shadow_color before authority is invalid")
        if receipt.after_shadow_color_authority not in {"automatic", "user"}:
            raise ValueError("shadow_color after authority is invalid")
        if (
            receipt.before_shadow_color_authority == "automatic"
            and before_shadow_color != automatic_shadow_color
        ):
            raise ValueError("automatic shadow_color before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("shadow_color command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("shadow_color supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("shadow_color commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"shadow_color": after_shadow_color}
                or receipt.after_shadow_color_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed shadow_color set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("shadow_color",)
                or after_shadow_color != automatic_shadow_color
                or receipt.after_shadow_color_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed shadow_color restore"
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
        projected_shadow_color = canonical_render_shadow_color(
            overrides.get("shadow_color", automatic_shadow_color),
            field_name="projected shadow_color",
        )
        projected_authority = (
            "user" if "shadow_color" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("shadow_color receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_shadow_color = canonical_render_shadow_color(
            receipt_overrides.get(
                "shadow_color",
                automatic_shadow_color,
            ),
            field_name="receipt shadow_color",
        )
        receipt_authority = (
            "user" if "shadow_color" in receipt_overrides else "automatic"
        )
        if (
            projected_shadow_color != after_shadow_color
            or projected_authority != receipt.after_shadow_color_authority
            or receipt_shadow_color != after_shadow_color
            or receipt_authority != receipt.after_shadow_color_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed shadow_color"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowColorEditorState:
    selection: RenderStyleShadowColorSelection
    phase: RenderStyleShadowColorEditorPhase
    draft_shadow_color: str | None
    message: str = ""
    worker_command: RenderStyleShadowColorWorkerCommand | None = None
    busy_state: RenderStyleShadowColorWorkerBusyState | None = None
    receipt: RenderStyleShadowColorWorkerReceipt | None = None
    failure: RenderStyleShadowColorWorkerFailure | None = None
    cancelled: RenderStyleShadowColorCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        if self.draft_shadow_color is None:
            return self.selection.effective_shadow_color is not None
        try:
            canonical = canonical_render_shadow_color(self.draft_shadow_color)
        except (TypeError, ValueError):
            return True
        baseline = (
            self.selection.automatic_shadow_color
            if self.selection.shadow_color_authority == "unresolved"
            else self.selection.effective_shadow_color
        )
        return canonical != baseline

    @property
    def valid(self) -> bool:
        if self.draft_shadow_color is None:
            return False
        try:
            canonical_render_shadow_color(self.draft_shadow_color)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleShadowColorEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleShadowColorEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return bool(
            not self.selection.excluded
            and self.selection.render_required
            and self.selection.automatic_shadow_color is not None
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
            and self.selection.shadow_color_authority in {"user", "unresolved"}
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
            and self.selection.shadow_color_authority != "unresolved"
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
                RenderStyleShadowColorEditorPhase.READY,
                RenderStyleShadowColorEditorPhase.COMMITTED,
                RenderStyleShadowColorEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleShadowColorEditorPhase.READY: "muted",
            RenderStyleShadowColorEditorPhase.DIRTY: "editing",
            RenderStyleShadowColorEditorPhase.COMMITTING: "editing",
            RenderStyleShadowColorEditorPhase.COMMITTED: "ready",
            RenderStyleShadowColorEditorPhase.CANCELLED: "muted",
            RenderStyleShadowColorEditorPhase.STALE: "warning",
            RenderStyleShadowColorEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleShadowColorEditorModel:
    """UI-thread reducer for one exact selected-parent RGBA shadow color."""

    def __init__(self, selection: RenderStyleShadowColorSelection) -> None:
        if not isinstance(selection, RenderStyleShadowColorSelection):
            raise TypeError(
                "selection must be RenderStyleShadowColorSelection"
            )
        self._state = RenderStyleShadowColorEditorState(
            selection=selection,
            phase=RenderStyleShadowColorEditorPhase.READY,
            draft_shadow_color=selection.effective_shadow_color,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleShadowColorEditorState:
        return self._state

    def set_draft_shadow_color(
        self,
        value: str,
    ) -> RenderStyleShadowColorEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("shadow_color draft is not editable")
        if not isinstance(value, str):
            raise TypeError("shadow_color draft must be a string")
        try:
            canonical = canonical_render_shadow_color(
                value,
                field_name="shadow_color draft",
            )
        except (TypeError, ValueError):
            canonical = None
        baseline = (
            self._state.selection.automatic_shadow_color
            if self._state.selection.shadow_color_authority == "unresolved"
            else self._state.selection.effective_shadow_color
        )
        dirty = canonical != baseline if canonical is not None else True
        phase = (
            RenderStyleShadowColorEditorPhase.DIRTY
            if dirty
            else RenderStyleShadowColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_color=value,
            message=(
                "Shadow color must use exactly #RRGGBB or #RRGGBBAA."
                if canonical is None
                else "Shadow color has an unapplied change."
                if phase is RenderStyleShadowColorEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleShadowColorEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard shadow_color while it is committing")
        phase = (
            RenderStyleShadowColorEditorPhase.STALE
            if self._state.stale
            else RenderStyleShadowColorEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_color=self._state.selection.effective_shadow_color,
            message=(
                "Reload the selected parent before editing shadow_color."
                if phase is RenderStyleShadowColorEditorPhase.STALE
                else "Shadow color draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleShadowColorEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleShadowColorWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable shadow_color draft")
        shadow_color = self._state.draft_shadow_color
        if shadow_color is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("shadow_color draft is unavailable")
        shadow_color = canonical_render_shadow_color(
            shadow_color,
            field_name="shadow_color draft",
        )
        command = self._command(
            RenderStyleShadowColorOperation.SET,
            shadow_color=shadow_color,
        )
        self._begin(command, "Applying shadow_color edit...")
        return command

    def begin_restore(self) -> RenderStyleShadowColorWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic shadow_color is already effective")
        command = self._command(
            RenderStyleShadowColorOperation.RESTORE_AUTOMATIC,
            shadow_color=None,
        )
        self._begin(command, "Restoring automatic shadow_color...")
        return command

    def accept_busy(
        self,
        value: RenderStyleShadowColorWorkerBusyState,
    ) -> RenderStyleShadowColorEditorState:
        if not isinstance(value, RenderStyleShadowColorWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleShadowColorWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleShadowColorEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleShadowColorWorkerReceipt,
    ) -> RenderStyleShadowColorEditorState:
        if not isinstance(value, RenderStyleShadowColorWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleShadowColorWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleShadowColorOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("shadow_color receipt has another operation")
        if command.operation is RenderStyleShadowColorOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"shadow_color": command.shadow_color}
                or receipt.after_shadow_color != command.shadow_color
            ):
                raise ValueError("shadow_color receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("shadow_color",)
                or receipt.after_shadow_color != receipt.automatic_shadow_color
            ):
                raise ValueError("shadow_color receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("shadow_color receipt has another base revision")
        selection = self._state.selection
        if selection.shadow_color_authority == "unresolved":
            before_matches_selection = bool(
                receipt.before_shadow_color == selection.automatic_shadow_color
                and receipt.before_shadow_color_authority == "automatic"
            )
        else:
            before_matches_selection = bool(
                receipt.before_shadow_color == selection.effective_shadow_color
                and receipt.before_shadow_color_authority
                == selection.shadow_color_authority
            )
        if (
            receipt.automatic_shadow_color != selection.automatic_shadow_color
            or not before_matches_selection
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("shadow_color receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_shadow_color=(
                receipt.after_shadow_color
                if receipt.after_shadow_color_authority == "user"
                else None
            ),
            effective_shadow_color=receipt.after_shadow_color,
            shadow_color_authority=receipt.after_shadow_color_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleShadowColorEditorState(
            selection=updated_selection,
            phase=RenderStyleShadowColorEditorPhase.COMMITTED,
            draft_shadow_color=receipt.after_shadow_color,
            message="Shadow color saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleShadowColorWorkerFailure,
    ) -> RenderStyleShadowColorEditorState:
        if not isinstance(value, RenderStyleShadowColorWorkerFailure):
            raise TypeError(
                "value must be RenderStyleShadowColorWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_color,
        )
        phase = (
            RenderStyleShadowColorEditorPhase.STALE
            if value.stale
            else RenderStyleShadowColorEditorPhase.FAILED
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
        value: RenderStyleShadowColorWorkerFailure,
    ) -> RenderStyleShadowColorEditorState:
        if not value.stale:
            raise ValueError("shadow_color failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleShadowColorCancelledReceipt,
    ) -> RenderStyleShadowColorEditorState:
        if not isinstance(value, RenderStyleShadowColorCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleShadowColorCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_color,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleShadowColorEditorPhase.CANCELLED,
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
        selection: RenderStyleShadowColorSelection,
    ) -> RenderStyleShadowColorEditorState:
        if not isinstance(selection, RenderStyleShadowColorSelection):
            raise TypeError(
                "selection must be RenderStyleShadowColorSelection"
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
            self._state.draft_shadow_color
            if preserve_draft
            else selection.effective_shadow_color
        )
        if draft is None:
            draft_dirty = selection.effective_shadow_color is not None
        else:
            try:
                canonical_draft = canonical_render_shadow_color(draft)
            except (TypeError, ValueError):
                draft_dirty = True
            else:
                baseline = (
                    selection.automatic_shadow_color
                    if selection.shadow_color_authority == "unresolved"
                    else selection.effective_shadow_color
                )
                draft_dirty = canonical_draft != baseline
        phase = (
            RenderStyleShadowColorEditorPhase.DIRTY
            if draft_dirty
            else RenderStyleShadowColorEditorPhase.READY
        )
        self._state = RenderStyleShadowColorEditorState(
            selection=selection,
            phase=phase,
            draft_shadow_color=draft,
            message=(
                "Current state changed; review the preserved shadow_color draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied shadow_color draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleShadowColorOperation,
        *,
        shadow_color: str | None,
    ) -> RenderStyleShadowColorWorkerCommand:
        selection = self._state.selection
        return RenderStyleShadowColorWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            shadow_color=shadow_color,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleShadowColorWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleShadowColorEditorPhase.COMMITTING,
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
    ) -> RenderStyleShadowColorWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no shadow_color worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleShadowColorOperation,
        shadow_color: str | None,
    ) -> RenderStyleShadowColorWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.shadow_color != shadow_color
        ):
            raise ValueError("worker event belongs to another shadow_color command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleShadowColorSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.shadow_color_authority == "user":
            return "User shadow color is effective."
        return "Automatic shadow color is effective."


__all__ = [
    "RenderStyleShadowColorCancellationState",
    "RenderStyleShadowColorCancelledReceipt",
    "RenderStyleShadowColorEditorModel",
    "RenderStyleShadowColorEditorPhase",
    "RenderStyleShadowColorEditorState",
    "RenderStyleShadowColorSelection",
    "RenderStyleShadowColorWorkerBusyState",
    "RenderStyleShadowColorWorkerCommand",
    "RenderStyleShadowColorWorkerFailure",
    "RenderStyleShadowColorWorkerFailureCode",
    "RenderStyleShadowColorWorkerReceipt",
    "RenderStyleShadowColorWorkerStage",
]
