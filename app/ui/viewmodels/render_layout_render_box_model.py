# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent render-box edits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Any, Mapping

from app.project_edits.commands import (
    RenderLayoutRenderBoxCommandErrorCode,
    RenderLayoutRenderBoxCommandReceipt,
    RenderLayoutRenderBoxOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_box,
)
from app.ui.viewmodels.editor_command_model import (
    _required_identity,
    _required_sha256,
)

_RENDER_BOX_STALE_COMMAND_CODES = frozenset(
    {
        RenderLayoutRenderBoxCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderLayoutRenderBoxCommandErrorCode.STALE_PAGE_HEAD,
        RenderLayoutRenderBoxCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)

class RenderLayoutRenderBoxEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderLayoutRenderBoxWorkerStage(str, Enum):
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


class RenderLayoutRenderBoxWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_RENDER_BOX_UNAVAILABLE = "automatic_render_box_unavailable"
    RENDER_BOX_OUTSIDE_HARD_BOUNDS = "render_box_outside_hard_bounds"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    RENDER_BOX_SLOT_CONFLICT = "render_box_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_render_box(
    value: Any,
    field_name: str,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    return canonical_render_box(value, field_name=field_name)


def _contains_xywh(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return bool(
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[0] + inner[2] <= outer[0] + outer[2]
        and inner[1] + inner[3] <= outer[1] + outer[3]
    )


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxSelection:
    """Canonical selected-parent render box state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_render_box: tuple[int, int, int, int] | None
    automatic_hard_bounds: tuple[int, int, int, int] | None
    user_render_box: tuple[int, int, int, int] | None
    effective_render_box: tuple[int, int, int, int] | None
    render_box_authority: str
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
            "automatic_render_box",
            "automatic_hard_bounds",
            "user_render_box",
            "effective_render_box",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_render_box(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.render_box_authority or "")
        if authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "render_box_authority must be automatic, user, or unavailable"
            )
        object.__setattr__(self, "render_box_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_render_box is not None:
                raise ValueError(
                    "automatic render_box authority cannot carry a user value"
                )
            if self.effective_render_box != self.automatic_render_box:
                raise ValueError(
                    "automatic authority must expose the automatic effective render_box"
                )
        elif authority == "user":
            if self.user_render_box is None:
                raise ValueError(
                    "user render_box authority requires a user value"
                )
            if self.effective_render_box != self.user_render_box:
                raise ValueError(
                    "user authority must expose the user effective render_box"
                )
        else:
            if (
                self.user_render_box is not None
                or self.effective_render_box is not None
            ):
                raise ValueError(
                    "unavailable render_box authority cannot expose a canonical user or effective value"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_render_box is not None
            and self.automatic_hard_bounds is not None
            and self.effective_render_box is not None
        )
        if eligible and (
            not _contains_xywh(self.automatic_hard_bounds, self.automatic_render_box)
            or not _contains_xywh(self.automatic_hard_bounds, self.effective_render_box)
        ):
            raise ValueError("render box must stay inside automatic hard bounds")
        if eligible and reason:
            raise ValueError(
                "available render_box selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable render_box selection requires an unavailable reason"
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
            and self.automatic_render_box is not None
            and self.automatic_hard_bounds is not None
            and self.effective_render_box is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxWorkerCommand:
    """UI carrier with one render box and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRenderBoxOperation
    render_box: tuple[int, int, int, int] | None
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
        operation = RenderLayoutRenderBoxOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutRenderBoxOperation.SET:
            object.__setattr__(
                self,
                "render_box",
                canonical_render_box(self.render_box),
            )
        elif self.render_box is not None:
            raise ValueError(
                "restore_automatic must not carry a render_box value"
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
class RenderLayoutRenderBoxWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderLayoutRenderBoxWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRenderBoxOperation
    render_box: tuple[int, int, int, int] | None
    stage: RenderLayoutRenderBoxWorkerStage
    message: str = "Render box update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxWorkerFailure:
    code: RenderLayoutRenderBoxWorkerFailureCode
    stage: RenderLayoutRenderBoxWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRenderBoxOperation
    render_box: tuple[int, int, int, int] | None
    message: str
    exception_type: str = ""
    command_error_code: RenderLayoutRenderBoxCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderLayoutRenderBoxCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderLayoutRenderBoxWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _RENDER_BOX_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderLayoutRenderBoxCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderLayoutRenderBoxCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderLayoutRenderBoxCommandReceipt"
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
        automatic_render_box = canonical_render_box(
            receipt.automatic_render_box,
            field_name="receipt automatic_render_box",
        )
        automatic_hard_bounds = canonical_render_box(
            receipt.automatic_hard_bounds,
            field_name="receipt automatic_hard_bounds",
        )
        before_render_box = canonical_render_box(
            receipt.before_render_box,
            field_name="receipt before_render_box",
        )
        after_render_box = canonical_render_box(
            receipt.after_render_box,
            field_name="receipt after_render_box",
        )
        if receipt.before_render_box_authority not in {"automatic", "user"}:
            raise ValueError("render_box before authority is invalid")
        if receipt.after_render_box_authority not in {"automatic", "user"}:
            raise ValueError("render_box after authority is invalid")
        if (
            receipt.before_render_box_authority == "automatic"
            and before_render_box != automatic_render_box
        ):
            raise ValueError("automatic render_box before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("render_box command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("render_box supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("render_box commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_LAYOUT:
            raise ValueError("worker receipt is not a render-layout edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"render_box": after_render_box}
                or receipt.after_render_box_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed render_box set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("render_box",)
                or after_render_box != automatic_render_box
                or receipt.after_render_box_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed render_box restore"
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
        projected_render_box = canonical_render_box(
            overrides.get("render_box", automatic_render_box),
            field_name="projected render_box",
        )
        projected_authority = (
            "user" if "render_box" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("render_box receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_layout_overrides)
        receipt_render_box = canonical_render_box(
            receipt_overrides.get(
                "render_box",
                automatic_render_box,
            ),
            field_name="receipt render_box",
        )
        receipt_authority = (
            "user" if "render_box" in receipt_overrides else "automatic"
        )
        if (
            projected_render_box != after_render_box
            or not _contains_xywh(automatic_hard_bounds, after_render_box)
            or projected_authority != receipt.after_render_box_authority
            or receipt_render_box != after_render_box
            or receipt_authority != receipt.after_render_box_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed render_box"
            )


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxEditorState:
    selection: RenderLayoutRenderBoxSelection
    phase: RenderLayoutRenderBoxEditorPhase
    draft_render_box: tuple[int, int, int, int] | None
    message: str = ""
    worker_command: RenderLayoutRenderBoxWorkerCommand | None = None
    busy_state: RenderLayoutRenderBoxWorkerBusyState | None = None
    receipt: RenderLayoutRenderBoxWorkerReceipt | None = None
    failure: RenderLayoutRenderBoxWorkerFailure | None = None
    cancelled: RenderLayoutRenderBoxCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        if self.draft_render_box is None:
            return self.selection.effective_render_box is not None
        try:
            canonical = canonical_render_box(self.draft_render_box)
        except (TypeError, ValueError):
            return True
        baseline = self.selection.effective_render_box
        return canonical != baseline

    @property
    def valid(self) -> bool:
        if self.draft_render_box is None:
            return False
        try:
            canonical = canonical_render_box(self.draft_render_box)
        except (TypeError, ValueError):
            return False
        hard_bounds = self.selection.automatic_hard_bounds
        return bool(
            hard_bounds is not None and _contains_xywh(hard_bounds, canonical)
        )

    @property
    def busy(self) -> bool:
        return self.phase is RenderLayoutRenderBoxEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderLayoutRenderBoxEditorPhase.STALE

    @property
    def available(self) -> bool:
        return self.selection.available

    @property
    def editing_enabled(self) -> bool:
        return bool(
            self.available
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
            and self.selection.render_box_authority == "user"
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
                RenderLayoutRenderBoxEditorPhase.READY,
                RenderLayoutRenderBoxEditorPhase.COMMITTED,
                RenderLayoutRenderBoxEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderLayoutRenderBoxEditorPhase.READY: "muted",
            RenderLayoutRenderBoxEditorPhase.DIRTY: "editing",
            RenderLayoutRenderBoxEditorPhase.COMMITTING: "editing",
            RenderLayoutRenderBoxEditorPhase.COMMITTED: "ready",
            RenderLayoutRenderBoxEditorPhase.CANCELLED: "muted",
            RenderLayoutRenderBoxEditorPhase.STALE: "warning",
            RenderLayoutRenderBoxEditorPhase.FAILED: "error",
        }[self.phase]


class RenderLayoutRenderBoxEditorModel:
    """UI-thread reducer for one exact selected-parent render box."""

    def __init__(self, selection: RenderLayoutRenderBoxSelection) -> None:
        if not isinstance(selection, RenderLayoutRenderBoxSelection):
            raise TypeError(
                "selection must be RenderLayoutRenderBoxSelection"
            )
        self._state = RenderLayoutRenderBoxEditorState(
            selection=selection,
            phase=RenderLayoutRenderBoxEditorPhase.READY,
            draft_render_box=selection.effective_render_box,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderLayoutRenderBoxEditorState:
        return self._state

    def set_draft_render_box(
        self,
        value: tuple[int, int, int, int],
    ) -> RenderLayoutRenderBoxEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("render_box draft is not editable")
        canonical = canonical_render_box(value, field_name="render_box draft")
        hard_bounds = self._state.selection.automatic_hard_bounds
        baseline = self._state.selection.effective_render_box
        dirty = canonical != baseline
        contained = bool(
            hard_bounds is not None and _contains_xywh(hard_bounds, canonical)
        )
        phase = (
            RenderLayoutRenderBoxEditorPhase.DIRTY
            if dirty
            else RenderLayoutRenderBoxEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_render_box=canonical,
            message=(
                "Render box must stay inside automatic hard bounds."
                if not contained
                else
                "Render box has an unapplied change."
                if phase is RenderLayoutRenderBoxEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderLayoutRenderBoxEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard render_box while it is committing")
        phase = (
            RenderLayoutRenderBoxEditorPhase.STALE
            if self._state.stale
            else RenderLayoutRenderBoxEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_render_box=self._state.selection.effective_render_box,
            message=(
                "Reload the selected parent before editing render_box."
                if phase is RenderLayoutRenderBoxEditorPhase.STALE
                else "Render box draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderLayoutRenderBoxEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderLayoutRenderBoxWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable render_box draft")
        render_box = self._state.draft_render_box
        if render_box is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("render_box draft is unavailable")
        render_box = canonical_render_box(
            render_box,
            field_name="render_box draft",
        )
        command = self._command(
            RenderLayoutRenderBoxOperation.SET,
            render_box=render_box,
        )
        self._begin(command, "Applying render box edit...")
        return command

    def begin_restore(self) -> RenderLayoutRenderBoxWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic render_box is already effective")
        command = self._command(
            RenderLayoutRenderBoxOperation.RESTORE_AUTOMATIC,
            render_box=None,
        )
        self._begin(command, "Restoring automatic render box...")
        return command

    def accept_busy(
        self,
        value: RenderLayoutRenderBoxWorkerBusyState,
    ) -> RenderLayoutRenderBoxEditorState:
        if not isinstance(value, RenderLayoutRenderBoxWorkerBusyState):
            raise TypeError(
                "value must be RenderLayoutRenderBoxWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderLayoutRenderBoxEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderLayoutRenderBoxWorkerReceipt,
    ) -> RenderLayoutRenderBoxEditorState:
        if not isinstance(value, RenderLayoutRenderBoxWorkerReceipt):
            raise TypeError(
                "value must be RenderLayoutRenderBoxWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderLayoutRenderBoxOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("render_box receipt has another operation")
        if command.operation is RenderLayoutRenderBoxOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"render_box": command.render_box}
                or receipt.after_render_box != command.render_box
            ):
                raise ValueError("render_box receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("render_box",)
                or receipt.after_render_box != receipt.automatic_render_box
            ):
                raise ValueError("render_box receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("render_box receipt has another base revision")
        selection = self._state.selection
        before_matches_selection = bool(
            receipt.before_render_box == selection.effective_render_box
            and receipt.before_render_box_authority
            == selection.render_box_authority
        )
        if (
            receipt.automatic_render_box != selection.automatic_render_box
            or receipt.automatic_hard_bounds != selection.automatic_hard_bounds
            or not before_matches_selection
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("render_box receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_render_box=(
                receipt.after_render_box
                if receipt.after_render_box_authority == "user"
                else None
            ),
            effective_render_box=receipt.after_render_box,
            render_box_authority=receipt.after_render_box_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderLayoutRenderBoxEditorState(
            selection=updated_selection,
            phase=RenderLayoutRenderBoxEditorPhase.COMMITTED,
            draft_render_box=receipt.after_render_box,
            message=(
                "Render box restored to Automatic. Preview remains explicit."
                if command.operation
                is RenderLayoutRenderBoxOperation.RESTORE_AUTOMATIC
                else "Render box saved. Preview remains explicit."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderLayoutRenderBoxWorkerFailure,
    ) -> RenderLayoutRenderBoxEditorState:
        if not isinstance(value, RenderLayoutRenderBoxWorkerFailure):
            raise TypeError(
                "value must be RenderLayoutRenderBoxWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.render_box,
        )
        phase = (
            RenderLayoutRenderBoxEditorPhase.STALE
            if value.stale
            else RenderLayoutRenderBoxEditorPhase.FAILED
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
        value: RenderLayoutRenderBoxWorkerFailure,
    ) -> RenderLayoutRenderBoxEditorState:
        if not value.stale:
            raise ValueError("render_box failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderLayoutRenderBoxCancelledReceipt,
    ) -> RenderLayoutRenderBoxEditorState:
        if not isinstance(value, RenderLayoutRenderBoxCancelledReceipt):
            raise TypeError(
                "value must be RenderLayoutRenderBoxCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.render_box,
        )
        self._state = replace(
            self._state,
            phase=RenderLayoutRenderBoxEditorPhase.CANCELLED,
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
        selection: RenderLayoutRenderBoxSelection,
    ) -> RenderLayoutRenderBoxEditorState:
        if not isinstance(selection, RenderLayoutRenderBoxSelection):
            raise TypeError(
                "selection must be RenderLayoutRenderBoxSelection"
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
            self._state.draft_render_box
            if preserve_draft
            else selection.effective_render_box
        )
        if draft is None:
            draft_dirty = selection.effective_render_box is not None
        else:
            try:
                canonical_draft = canonical_render_box(draft)
            except (TypeError, ValueError):
                draft_dirty = True
            else:
                baseline = (
                    selection.effective_render_box
                )
                draft_dirty = canonical_draft != baseline
        phase = (
            RenderLayoutRenderBoxEditorPhase.DIRTY
            if draft_dirty
            else RenderLayoutRenderBoxEditorPhase.READY
        )
        self._state = RenderLayoutRenderBoxEditorState(
            selection=selection,
            phase=phase,
            draft_render_box=draft,
            message=(
                "Current state changed; review the preserved render_box draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied render_box draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderLayoutRenderBoxOperation,
        *,
        render_box: tuple[int, int, int, int] | None,
    ) -> RenderLayoutRenderBoxWorkerCommand:
        selection = self._state.selection
        return RenderLayoutRenderBoxWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            render_box=render_box,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderLayoutRenderBoxWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderLayoutRenderBoxEditorPhase.COMMITTING,
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
    ) -> RenderLayoutRenderBoxWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no render_box worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderLayoutRenderBoxOperation,
        render_box: tuple[int, int, int, int] | None,
    ) -> RenderLayoutRenderBoxWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.render_box != render_box
        ):
            raise ValueError("worker event belongs to another render_box command")
        return command

    @staticmethod
    def _ready_message(selection: RenderLayoutRenderBoxSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.render_box_authority == "user":
            return "User render box is effective."
        return "Automatic render box is effective."


__all__ = [
    "RenderLayoutRenderBoxCancellationState",
    "RenderLayoutRenderBoxCancelledReceipt",
    "RenderLayoutRenderBoxEditorModel",
    "RenderLayoutRenderBoxEditorPhase",
    "RenderLayoutRenderBoxEditorState",
    "RenderLayoutRenderBoxSelection",
    "RenderLayoutRenderBoxWorkerBusyState",
    "RenderLayoutRenderBoxWorkerCommand",
    "RenderLayoutRenderBoxWorkerFailure",
    "RenderLayoutRenderBoxWorkerFailureCode",
    "RenderLayoutRenderBoxWorkerReceipt",
    "RenderLayoutRenderBoxWorkerStage",
]
