# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent shadow-blur commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from app.project_edits.commands import (
    RenderStyleShadowBlurCommandErrorCode,
    RenderStyleShadowBlurCommandReceipt,
    RenderStyleShadowBlurOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_shadow_blur,
    thaw_json,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection

_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SHADOW_BLUR_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleShadowBlurCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleShadowBlurCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleShadowBlurCommandErrorCode.STALE_GLOBAL_HEAD,
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

class RenderStyleShadowBlurEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleShadowBlurWorkerStage(str, Enum):
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


class RenderStyleShadowBlurWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    SHADOW_BLUR_SLOT_CONFLICT = "shadow_blur_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_shadow_blur(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return canonical_render_shadow_blur(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurSelection:
    """Exact selected-parent shadow-blur state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_shadow_blur: float | None
    user_shadow_blur: float | None
    effective_shadow_blur: float | None
    shadow_blur_authority: str
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
            "automatic_shadow_blur",
            "user_shadow_blur",
            "effective_shadow_blur",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_shadow_blur(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.shadow_blur_authority or "")
        if authority not in {"automatic", "user"}:
            raise ValueError(
                "shadow_blur_authority must be automatic or user"
            )
        object.__setattr__(self, "shadow_blur_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_shadow_blur is not None:
                raise ValueError(
                    "automatic shadow-blur authority cannot carry a user value"
                )
            if self.effective_shadow_blur != self.automatic_shadow_blur:
                raise ValueError(
                    "automatic authority must expose the automatic effective shadow blur"
                )
        else:
            if self.user_shadow_blur is None:
                raise ValueError(
                    "user shadow-blur authority requires a user value"
                )
            if self.effective_shadow_blur != self.user_shadow_blur:
                raise ValueError(
                    "user authority must expose the user effective shadow blur"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_shadow_blur is not None
            and self.effective_shadow_blur is not None
        )
        if eligible and reason:
            raise ValueError(
                "available shadow-blur selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable shadow-blur selection requires an unavailable reason"
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
            and self.automatic_shadow_blur is not None
            and self.effective_shadow_blur is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurWorkerCommand:
    """UI carrier with one exact pixel width and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowBlurOperation
    shadow_blur: float | None
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
        operation = RenderStyleShadowBlurOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowBlurOperation.SET:
            object.__setattr__(
                self,
                "shadow_blur",
                canonical_render_shadow_blur(self.shadow_blur),
            )
        elif self.shadow_blur is not None:
            raise ValueError(
                "restore_automatic must not carry a shadow_blur value"
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
class RenderStyleShadowBlurWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleShadowBlurWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowBlurOperation
    shadow_blur: float | None
    stage: RenderStyleShadowBlurWorkerStage
    message: str = "Shadow-blur update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurWorkerFailure:
    code: RenderStyleShadowBlurWorkerFailureCode
    stage: RenderStyleShadowBlurWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowBlurOperation
    shadow_blur: float | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleShadowBlurCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleShadowBlurCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleShadowBlurWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _SHADOW_BLUR_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleShadowBlurCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleShadowBlurCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleShadowBlurCommandReceipt"
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
        automatic_shadow_blur = canonical_render_shadow_blur(
            receipt.automatic_shadow_blur,
            field_name="receipt automatic_shadow_blur",
        )
        before_shadow_blur = canonical_render_shadow_blur(
            receipt.before_shadow_blur,
            field_name="receipt before_shadow_blur",
        )
        after_shadow_blur = canonical_render_shadow_blur(
            receipt.after_shadow_blur,
            field_name="receipt after_shadow_blur",
        )
        if receipt.before_shadow_blur_authority not in {"automatic", "user"}:
            raise ValueError("shadow-blur before authority is invalid")
        if receipt.after_shadow_blur_authority not in {"automatic", "user"}:
            raise ValueError("shadow-blur after authority is invalid")
        if (
            receipt.before_shadow_blur_authority == "automatic"
            and before_shadow_blur != automatic_shadow_blur
        ):
            raise ValueError("automatic shadow-blur before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("shadow-blur command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("shadow-blur supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("shadow-blur commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"shadow_blur": after_shadow_blur}
                or receipt.after_shadow_blur_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed shadow-blur set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("shadow_blur",)
                or after_shadow_blur != automatic_shadow_blur
                or receipt.after_shadow_blur_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed shadow-blur restore"
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
        projected_size = canonical_render_shadow_blur(
            overrides.get("shadow_blur", automatic_shadow_blur),
            field_name="projected shadow_blur",
        )
        projected_authority = (
            "user" if "shadow_blur" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("shadow-blur receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_size = canonical_render_shadow_blur(
            receipt_overrides.get(
                "shadow_blur",
                automatic_shadow_blur,
            ),
            field_name="receipt shadow_blur",
        )
        receipt_authority = (
            "user" if "shadow_blur" in receipt_overrides else "automatic"
        )
        if (
            projected_size != after_shadow_blur
            or projected_authority != receipt.after_shadow_blur_authority
            or receipt_size != after_shadow_blur
            or receipt_authority != receipt.after_shadow_blur_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed shadow blur"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleShadowBlurEditorState:
    selection: RenderStyleShadowBlurSelection
    phase: RenderStyleShadowBlurEditorPhase
    draft_shadow_blur: float | None
    message: str = ""
    worker_command: RenderStyleShadowBlurWorkerCommand | None = None
    busy_state: RenderStyleShadowBlurWorkerBusyState | None = None
    receipt: RenderStyleShadowBlurWorkerReceipt | None = None
    failure: RenderStyleShadowBlurWorkerFailure | None = None
    cancelled: RenderStyleShadowBlurCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        return self.draft_shadow_blur != self.selection.effective_shadow_blur

    @property
    def valid(self) -> bool:
        if self.draft_shadow_blur is None:
            return False
        try:
            canonical_render_shadow_blur(self.draft_shadow_blur)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleShadowBlurEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleShadowBlurEditorPhase.STALE

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
            and self.selection.shadow_blur_authority == "user"
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
                RenderStyleShadowBlurEditorPhase.READY,
                RenderStyleShadowBlurEditorPhase.COMMITTED,
                RenderStyleShadowBlurEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleShadowBlurEditorPhase.READY: "muted",
            RenderStyleShadowBlurEditorPhase.DIRTY: "editing",
            RenderStyleShadowBlurEditorPhase.COMMITTING: "editing",
            RenderStyleShadowBlurEditorPhase.COMMITTED: "ready",
            RenderStyleShadowBlurEditorPhase.CANCELLED: "muted",
            RenderStyleShadowBlurEditorPhase.STALE: "warning",
            RenderStyleShadowBlurEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleShadowBlurEditorModel:
    """UI-thread reducer for one exact selected-parent shadow-blur ratio."""

    def __init__(self, selection: RenderStyleShadowBlurSelection) -> None:
        if not isinstance(selection, RenderStyleShadowBlurSelection):
            raise TypeError(
                "selection must be RenderStyleShadowBlurSelection"
            )
        self._state = RenderStyleShadowBlurEditorState(
            selection=selection,
            phase=RenderStyleShadowBlurEditorPhase.READY,
            draft_shadow_blur=selection.effective_shadow_blur,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleShadowBlurEditorState:
        return self._state

    def set_draft_shadow_blur(
        self,
        value: float,
    ) -> RenderStyleShadowBlurEditorState:
        shadow_blur = canonical_render_shadow_blur(
            value,
            field_name="shadow-blur draft",
        )
        if not self._state.editing_enabled:
            raise RuntimeError("shadow-blur draft is not editable")
        phase = (
            RenderStyleShadowBlurEditorPhase.DIRTY
            if shadow_blur != self._state.selection.effective_shadow_blur
            else RenderStyleShadowBlurEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_blur=shadow_blur,
            message=(
                "Shadow blur has an unapplied change."
                if phase is RenderStyleShadowBlurEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleShadowBlurEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard shadow blur while it is committing")
        phase = (
            RenderStyleShadowBlurEditorPhase.STALE
            if self._state.stale
            else RenderStyleShadowBlurEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_shadow_blur=self._state.selection.effective_shadow_blur,
            message=(
                "Reload the selected parent before editing shadow blur."
                if phase is RenderStyleShadowBlurEditorPhase.STALE
                else "Shadow-blur draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleShadowBlurEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleShadowBlurWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable shadow-blur draft")
        shadow_blur = self._state.draft_shadow_blur
        if shadow_blur is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("shadow-blur draft is unavailable")
        command = self._command(
            RenderStyleShadowBlurOperation.SET,
            shadow_blur=shadow_blur,
        )
        self._begin(command, "Applying shadow-blur edit...")
        return command

    def begin_restore(self) -> RenderStyleShadowBlurWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic shadow blur is already effective")
        command = self._command(
            RenderStyleShadowBlurOperation.RESTORE_AUTOMATIC,
            shadow_blur=None,
        )
        self._begin(command, "Restoring automatic shadow blur...")
        return command

    def accept_busy(
        self,
        value: RenderStyleShadowBlurWorkerBusyState,
    ) -> RenderStyleShadowBlurEditorState:
        if not isinstance(value, RenderStyleShadowBlurWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleShadowBlurWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleShadowBlurEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleShadowBlurWorkerReceipt,
    ) -> RenderStyleShadowBlurEditorState:
        if not isinstance(value, RenderStyleShadowBlurWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleShadowBlurWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleShadowBlurOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("shadow-blur receipt has another operation")
        if command.operation is RenderStyleShadowBlurOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"shadow_blur": command.shadow_blur}
                or receipt.after_shadow_blur != command.shadow_blur
            ):
                raise ValueError("shadow-blur receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("shadow_blur",)
                or receipt.after_shadow_blur != receipt.automatic_shadow_blur
            ):
                raise ValueError("shadow-blur receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("shadow-blur receipt has another base revision")
        selection = self._state.selection
        if (
            receipt.automatic_shadow_blur != selection.automatic_shadow_blur
            or receipt.before_shadow_blur != selection.effective_shadow_blur
            or receipt.before_shadow_blur_authority
            != selection.shadow_blur_authority
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("shadow-blur receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_shadow_blur=(
                receipt.after_shadow_blur
                if receipt.after_shadow_blur_authority == "user"
                else None
            ),
            effective_shadow_blur=receipt.after_shadow_blur,
            shadow_blur_authority=receipt.after_shadow_blur_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleShadowBlurEditorState(
            selection=updated_selection,
            phase=RenderStyleShadowBlurEditorPhase.COMMITTED,
            draft_shadow_blur=receipt.after_shadow_blur,
            message="Shadow blur saved. Preview remains explicit.",
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleShadowBlurWorkerFailure,
    ) -> RenderStyleShadowBlurEditorState:
        if not isinstance(value, RenderStyleShadowBlurWorkerFailure):
            raise TypeError(
                "value must be RenderStyleShadowBlurWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_blur,
        )
        phase = (
            RenderStyleShadowBlurEditorPhase.STALE
            if value.stale
            else RenderStyleShadowBlurEditorPhase.FAILED
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
        value: RenderStyleShadowBlurWorkerFailure,
    ) -> RenderStyleShadowBlurEditorState:
        if not value.stale:
            raise ValueError("shadow-blur failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleShadowBlurCancelledReceipt,
    ) -> RenderStyleShadowBlurEditorState:
        if not isinstance(value, RenderStyleShadowBlurCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleShadowBlurCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.shadow_blur,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleShadowBlurEditorPhase.CANCELLED,
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
        selection: RenderStyleShadowBlurSelection,
    ) -> RenderStyleShadowBlurEditorState:
        if not isinstance(selection, RenderStyleShadowBlurSelection):
            raise TypeError(
                "selection must be RenderStyleShadowBlurSelection"
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
            self._state.draft_shadow_blur
            if preserve_draft
            else selection.effective_shadow_blur
        )
        phase = (
            RenderStyleShadowBlurEditorPhase.DIRTY
            if draft != selection.effective_shadow_blur
            else RenderStyleShadowBlurEditorPhase.READY
        )
        self._state = RenderStyleShadowBlurEditorState(
            selection=selection,
            phase=phase,
            draft_shadow_blur=draft,
            message=(
                "Current state changed; review the preserved shadow-blur draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied shadow-blur draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleShadowBlurOperation,
        *,
        shadow_blur: float | None,
    ) -> RenderStyleShadowBlurWorkerCommand:
        selection = self._state.selection
        return RenderStyleShadowBlurWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            shadow_blur=shadow_blur,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleShadowBlurWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleShadowBlurEditorPhase.COMMITTING,
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
    ) -> RenderStyleShadowBlurWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no shadow-blur worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleShadowBlurOperation,
        shadow_blur: float | None,
    ) -> RenderStyleShadowBlurWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.shadow_blur != shadow_blur
        ):
            raise ValueError("worker event belongs to another shadow-blur command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleShadowBlurSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.shadow_blur_authority == "user":
            return "User shadow blur is effective."
        return "Automatic shadow blur is effective."

__all__ = [
    "RenderStyleShadowBlurCancellationState",
    "RenderStyleShadowBlurCancelledReceipt",
    "RenderStyleShadowBlurEditorModel",
    "RenderStyleShadowBlurEditorPhase",
    "RenderStyleShadowBlurEditorState",
    "RenderStyleShadowBlurSelection",
    "RenderStyleShadowBlurWorkerBusyState",
    "RenderStyleShadowBlurWorkerCommand",
    "RenderStyleShadowBlurWorkerFailure",
    "RenderStyleShadowBlurWorkerFailureCode",
    "RenderStyleShadowBlurWorkerReceipt",
    "RenderStyleShadowBlurWorkerStage",
]
