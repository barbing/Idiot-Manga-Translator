# -*- coding: utf-8 -*-
"""Framework-neutral state for selected-parent font-weight-tier edits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import Any, Mapping

from app.project_edits.commands import (
    RenderStyleFontWeightTierCommandErrorCode,
    RenderStyleFontWeightTierCommandReceipt,
    RenderStyleFontWeightTierOperation,
)
from app.project_edits.contracts import (
    EditDomain,
    canonical_render_font_weight_tier,
)
from app.ui.viewmodels.editor_command_model import (
    _required_identity,
    _required_sha256,
)

_FONT_WEIGHT_TIER_STALE_COMMAND_CODES = frozenset(
    {
        RenderStyleFontWeightTierCommandErrorCode.STALE_EFFECTIVE_PAGE,
        RenderStyleFontWeightTierCommandErrorCode.STALE_PAGE_HEAD,
        RenderStyleFontWeightTierCommandErrorCode.STALE_GLOBAL_HEAD,
    }
)

class RenderStyleFontWeightTierEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderStyleFontWeightTierWorkerStage(str, Enum):
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


class RenderStyleFontWeightTierWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_UNAVAILABLE = "parent_unavailable"
    AUTOMATIC_FONT_WEIGHT_TIER_UNAVAILABLE = "automatic_font_weight_tier_unavailable"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    FONT_WEIGHT_TIER_SLOT_CONFLICT = "font_weight_tier_slot_conflict"
    DUPLICATE_COMMAND = "duplicate_command"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _optional_canonical_font_weight_tier(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return canonical_render_font_weight_tier(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierSelection:
    """Canonical selected-parent font weight tier state from one UI projection."""

    project_path: str
    page_id: str
    parent_id: str
    automatic_font_weight_tier: str | None
    user_font_weight_tier: str | None
    effective_font_weight_tier: str | None
    font_weight_tier_authority: str
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
            "automatic_font_weight_tier",
            "user_font_weight_tier",
            "effective_font_weight_tier",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_canonical_font_weight_tier(
                    getattr(self, field_name),
                    field_name,
                ),
            )
        authority = str(self.font_weight_tier_authority or "")
        if authority not in {"automatic", "user", "unavailable"}:
            raise ValueError(
                "font_weight_tier_authority must be automatic, user, or unavailable"
            )
        object.__setattr__(self, "font_weight_tier_authority", authority)
        if not isinstance(self.render_required, bool):
            raise TypeError("render_required must be a boolean")
        if not isinstance(self.excluded, bool):
            raise TypeError("excluded must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        object.__setattr__(self, "unavailable_reason", reason)
        if authority == "automatic":
            if self.user_font_weight_tier is not None:
                raise ValueError(
                    "automatic font_weight_tier authority cannot carry a user value"
                )
            if self.effective_font_weight_tier != self.automatic_font_weight_tier:
                raise ValueError(
                    "automatic authority must expose the automatic effective font_weight_tier"
                )
        elif authority == "user":
            if self.user_font_weight_tier is None:
                raise ValueError(
                    "user font_weight_tier authority requires a user value"
                )
            if self.effective_font_weight_tier != self.user_font_weight_tier:
                raise ValueError(
                    "user authority must expose the user effective font_weight_tier"
                )
        else:
            if (
                self.user_font_weight_tier is not None
                or self.effective_font_weight_tier is not None
            ):
                raise ValueError(
                    "unavailable font_weight_tier authority cannot expose a canonical user or effective value"
                )
        eligible = bool(
            not self.excluded
            and self.render_required
            and self.automatic_font_weight_tier is not None
            and self.effective_font_weight_tier is not None
        )
        if eligible and reason:
            raise ValueError(
                "available font_weight_tier selection cannot carry an unavailable reason"
            )
        if not eligible and not reason:
            raise ValueError(
                "unavailable font_weight_tier selection requires an unavailable reason"
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
            and self.automatic_font_weight_tier is not None
            and self.effective_font_weight_tier is not None
            and not self.unavailable_reason
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierWorkerCommand:
    """UI carrier with one font weight tier and no persistence-owned CAS state."""

    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontWeightTierOperation
    font_weight_tier: str | None
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
        operation = RenderStyleFontWeightTierOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleFontWeightTierOperation.SET:
            object.__setattr__(
                self,
                "font_weight_tier",
                canonical_render_font_weight_tier(self.font_weight_tier),
            )
        elif self.font_weight_tier is not None:
            raise ValueError(
                "restore_automatic must not carry a font_weight_tier value"
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
class RenderStyleFontWeightTierWorkerBusyState:
    page_id: str
    parent_id: str
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    stage: RenderStyleFontWeightTierWorkerStage
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierCancellationState:
    page_id: str
    parent_id: str
    enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierCancelledReceipt:
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontWeightTierOperation
    font_weight_tier: str | None
    stage: RenderStyleFontWeightTierWorkerStage
    message: str = "Font weight tier update cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierWorkerFailure:
    code: RenderStyleFontWeightTierWorkerFailureCode
    stage: RenderStyleFontWeightTierWorkerStage
    project_path: str
    page_id: str
    parent_id: str
    operation: RenderStyleFontWeightTierOperation
    font_weight_tier: str | None
    message: str
    exception_type: str = ""
    command_error_code: RenderStyleFontWeightTierCommandErrorCode | None = None
    persistence_committed: bool = False
    command_receipt: RenderStyleFontWeightTierCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code is RenderStyleFontWeightTierWorkerFailureCode.SNAPSHOT_STALE
            or self.command_error_code in _FONT_WEIGHT_TIER_STALE_COMMAND_CODES
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierWorkerReceipt:
    """Committed command bound to one materialized project and projection."""

    command_receipt: RenderStyleFontWeightTierCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        if not isinstance(
            self.command_receipt,
            RenderStyleFontWeightTierCommandReceipt,
        ):
            raise TypeError(
                "command_receipt must be RenderStyleFontWeightTierCommandReceipt"
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
        automatic_font_weight_tier = canonical_render_font_weight_tier(
            receipt.automatic_font_weight_tier,
            field_name="receipt automatic_font_weight_tier",
        )
        before_font_weight_tier = canonical_render_font_weight_tier(
            receipt.before_font_weight_tier,
            field_name="receipt before_font_weight_tier",
        )
        after_font_weight_tier = canonical_render_font_weight_tier(
            receipt.after_font_weight_tier,
            field_name="receipt after_font_weight_tier",
        )
        if receipt.before_font_weight_tier_authority not in {"automatic", "user"}:
            raise ValueError("font_weight_tier before authority is invalid")
        if receipt.after_font_weight_tier_authority not in {"automatic", "user"}:
            raise ValueError("font_weight_tier after authority is invalid")
        if (
            receipt.before_font_weight_tier_authority == "automatic"
            and before_font_weight_tier != automatic_font_weight_tier
        ):
            raise ValueError("automatic font_weight_tier before state is inconsistent")
        if receipt.command_id != edit.edit_id:
            raise ValueError("font_weight_tier command and edit identities disagree")
        if receipt.superseded_edit_id != edit.supersedes_edit_id:
            raise ValueError("font_weight_tier supersession identities disagree")
        if (
            commit.transaction_id != receipt.command_id
            or commit.page_id != edit.page_id
            or commit.edit_ids != (edit.edit_id,)
            or commit.artifact_revision_ids
        ):
            raise ValueError("font_weight_tier commit receipt is not command-bound")
        if edit.domain is not EditDomain.RENDER_STYLE:
            raise ValueError("worker receipt is not a render-style edit")
        edit_fields = edit.payload.get("fields")
        if edit.operation == "set_fields":
            if (
                not isinstance(edit_fields, Mapping)
                or dict(edit_fields) != {"font_weight_tier": after_font_weight_tier}
                or receipt.after_font_weight_tier_authority != "user"
            ):
                raise ValueError("worker receipt is not the committed font_weight_tier set")
        elif edit.operation == "restore_automatic":
            if (
                not isinstance(edit_fields, tuple)
                or edit_fields != ("font_weight_tier",)
                or after_font_weight_tier != automatic_font_weight_tier
                or receipt.after_font_weight_tier_authority != "automatic"
            ):
                raise ValueError(
                    "worker receipt is not the committed font_weight_tier restore"
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
        projected_font_weight_tier = canonical_render_font_weight_tier(
            overrides.get("font_weight_tier", automatic_font_weight_tier),
            field_name="projected font_weight_tier",
        )
        projected_authority = (
            "user" if "font_weight_tier" in overrides else "automatic"
        )
        receipt_parents = tuple(
            candidate
            for candidate in receipt.effective_page.parents
            if candidate.parent_id == edit.target.parent_id
        )
        if len(receipt_parents) != 1:
            raise ValueError("font_weight_tier receipt has no exact effective parent")
        receipt_overrides = dict(receipt_parents[0].render_style_overrides)
        receipt_font_weight_tier = canonical_render_font_weight_tier(
            receipt_overrides.get(
                "font_weight_tier",
                automatic_font_weight_tier,
            ),
            field_name="receipt font_weight_tier",
        )
        receipt_authority = (
            "user" if "font_weight_tier" in receipt_overrides else "automatic"
        )
        if (
            projected_font_weight_tier != after_font_weight_tier
            or projected_authority != receipt.after_font_weight_tier_authority
            or receipt_font_weight_tier != after_font_weight_tier
            or receipt_authority != receipt.after_font_weight_tier_authority
        ):
            raise ValueError(
                "worker projection does not contain the committed font_weight_tier"
            )


@dataclass(frozen=True, slots=True)
class RenderStyleFontWeightTierEditorState:
    selection: RenderStyleFontWeightTierSelection
    phase: RenderStyleFontWeightTierEditorPhase
    draft_font_weight_tier: str | None
    message: str = ""
    worker_command: RenderStyleFontWeightTierWorkerCommand | None = None
    busy_state: RenderStyleFontWeightTierWorkerBusyState | None = None
    receipt: RenderStyleFontWeightTierWorkerReceipt | None = None
    failure: RenderStyleFontWeightTierWorkerFailure | None = None
    cancelled: RenderStyleFontWeightTierCancelledReceipt | None = None

    @property
    def dirty(self) -> bool:
        if self.draft_font_weight_tier is None:
            return self.selection.effective_font_weight_tier is not None
        try:
            canonical = canonical_render_font_weight_tier(self.draft_font_weight_tier)
        except (TypeError, ValueError):
            return True
        baseline = self.selection.effective_font_weight_tier
        return canonical != baseline

    @property
    def valid(self) -> bool:
        if self.draft_font_weight_tier is None:
            return False
        try:
            canonical_render_font_weight_tier(self.draft_font_weight_tier)
        except (TypeError, ValueError):
            return False
        return True

    @property
    def busy(self) -> bool:
        return self.phase is RenderStyleFontWeightTierEditorPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderStyleFontWeightTierEditorPhase.STALE

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
            and self.selection.font_weight_tier_authority == "user"
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
                RenderStyleFontWeightTierEditorPhase.READY,
                RenderStyleFontWeightTierEditorPhase.COMMITTED,
                RenderStyleFontWeightTierEditorPhase.CANCELLED,
            }
        ):
            return "muted" if self.selection.excluded else "warning"
        return {
            RenderStyleFontWeightTierEditorPhase.READY: "muted",
            RenderStyleFontWeightTierEditorPhase.DIRTY: "editing",
            RenderStyleFontWeightTierEditorPhase.COMMITTING: "editing",
            RenderStyleFontWeightTierEditorPhase.COMMITTED: "ready",
            RenderStyleFontWeightTierEditorPhase.CANCELLED: "muted",
            RenderStyleFontWeightTierEditorPhase.STALE: "warning",
            RenderStyleFontWeightTierEditorPhase.FAILED: "error",
        }[self.phase]


class RenderStyleFontWeightTierEditorModel:
    """UI-thread reducer for one exact selected-parent font weight tier."""

    def __init__(self, selection: RenderStyleFontWeightTierSelection) -> None:
        if not isinstance(selection, RenderStyleFontWeightTierSelection):
            raise TypeError(
                "selection must be RenderStyleFontWeightTierSelection"
            )
        self._state = RenderStyleFontWeightTierEditorState(
            selection=selection,
            phase=RenderStyleFontWeightTierEditorPhase.READY,
            draft_font_weight_tier=selection.effective_font_weight_tier,
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> RenderStyleFontWeightTierEditorState:
        return self._state

    def set_draft_font_weight_tier(
        self,
        value: str,
    ) -> RenderStyleFontWeightTierEditorState:
        if not self._state.editing_enabled:
            raise RuntimeError("font_weight_tier draft is not editable")
        if not isinstance(value, str):
            raise TypeError("font_weight_tier draft must be a string")
        try:
            canonical = canonical_render_font_weight_tier(
                value,
                field_name="font_weight_tier draft",
            )
        except (TypeError, ValueError):
            canonical = None
        baseline = (
            self._state.selection.effective_font_weight_tier
        )
        dirty = canonical != baseline if canonical is not None else True
        phase = (
            RenderStyleFontWeightTierEditorPhase.DIRTY
            if dirty
            else RenderStyleFontWeightTierEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_font_weight_tier=value,
            message=(
                "Font weight tier must be Slender, Base, Emphasis, or Heavy."
                if canonical is None
                else "Font weight tier has an unapplied change."
                if phase is RenderStyleFontWeightTierEditorPhase.DIRTY
                else self._ready_message(self._state.selection)
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return self._state

    def cancel_draft(self) -> RenderStyleFontWeightTierEditorState:
        if self._state.busy:
            raise RuntimeError("cannot discard font_weight_tier while it is committing")
        phase = (
            RenderStyleFontWeightTierEditorPhase.STALE
            if self._state.stale
            else RenderStyleFontWeightTierEditorPhase.READY
        )
        self._state = replace(
            self._state,
            phase=phase,
            draft_font_weight_tier=self._state.selection.effective_font_weight_tier,
            message=(
                "Reload the selected parent before editing font_weight_tier."
                if phase is RenderStyleFontWeightTierEditorPhase.STALE
                else "Font weight tier draft discarded; project state was not changed."
            ),
            worker_command=None,
            busy_state=None,
            receipt=None,
            failure=(
                self._state.failure
                if phase is RenderStyleFontWeightTierEditorPhase.STALE
                else None
            ),
            cancelled=None,
        )
        return self._state

    def begin_set(self) -> RenderStyleFontWeightTierWorkerCommand:
        if not self._state.apply_enabled:
            raise RuntimeError("there is no applicable font_weight_tier draft")
        font_weight_tier = self._state.draft_font_weight_tier
        if font_weight_tier is None:  # pragma: no cover - guarded by valid
            raise RuntimeError("font_weight_tier draft is unavailable")
        font_weight_tier = canonical_render_font_weight_tier(
            font_weight_tier,
            field_name="font_weight_tier draft",
        )
        command = self._command(
            RenderStyleFontWeightTierOperation.SET,
            font_weight_tier=font_weight_tier,
        )
        self._begin(command, "Applying font weight tier edit...")
        return command

    def begin_restore(self) -> RenderStyleFontWeightTierWorkerCommand:
        if not self._state.restore_enabled:
            raise RuntimeError("automatic font_weight_tier is already effective")
        command = self._command(
            RenderStyleFontWeightTierOperation.RESTORE_AUTOMATIC,
            font_weight_tier=None,
        )
        self._begin(command, "Restoring automatic font weight tier...")
        return command

    def accept_busy(
        self,
        value: RenderStyleFontWeightTierWorkerBusyState,
    ) -> RenderStyleFontWeightTierEditorState:
        if not isinstance(value, RenderStyleFontWeightTierWorkerBusyState):
            raise TypeError(
                "value must be RenderStyleFontWeightTierWorkerBusyState"
            )
        self._require_active_target(value.page_id, value.parent_id)
        self._state = replace(
            self._state,
            phase=(
                RenderStyleFontWeightTierEditorPhase.COMMITTING
                if value.busy
                else self._state.phase
            ),
            message=value.message or self._state.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderStyleFontWeightTierWorkerReceipt,
    ) -> RenderStyleFontWeightTierEditorState:
        if not isinstance(value, RenderStyleFontWeightTierWorkerReceipt):
            raise TypeError(
                "value must be RenderStyleFontWeightTierWorkerReceipt"
            )
        receipt = value.command_receipt
        edit = receipt.edit
        command = self._require_active_target(
            edit.page_id,
            edit.target.parent_id,
        )
        expected_edit_operation = (
            "set_fields"
            if command.operation is RenderStyleFontWeightTierOperation.SET
            else "restore_automatic"
        )
        if edit.operation != expected_edit_operation:
            raise ValueError("font_weight_tier receipt has another operation")
        if command.operation is RenderStyleFontWeightTierOperation.SET:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, Mapping)
                or dict(fields) != {"font_weight_tier": command.font_weight_tier}
                or receipt.after_font_weight_tier != command.font_weight_tier
            ):
                raise ValueError("font_weight_tier receipt has another effective value")
        else:
            fields = edit.payload.get("fields")
            if (
                not isinstance(fields, tuple)
                or fields != ("font_weight_tier",)
                or receipt.after_font_weight_tier != receipt.automatic_font_weight_tier
            ):
                raise ValueError("font_weight_tier receipt has another restore value")
        if (
            command.expected_effective_page_fingerprint
            != receipt.before_effective_page_fingerprint
        ):
            raise ValueError("font_weight_tier receipt has another base revision")
        selection = self._state.selection
        before_matches_selection = bool(
            receipt.before_font_weight_tier == selection.effective_font_weight_tier
            and receipt.before_font_weight_tier_authority
            == selection.font_weight_tier_authority
        )
        if (
            receipt.automatic_font_weight_tier != selection.automatic_font_weight_tier
            or not before_matches_selection
            or os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(
                os.path.abspath(value.projection.metadata.project_path)
            )
        ):
            raise ValueError("font_weight_tier receipt belongs to another selection")
        updated_selection = replace(
            selection,
            user_font_weight_tier=(
                receipt.after_font_weight_tier
                if receipt.after_font_weight_tier_authority == "user"
                else None
            ),
            effective_font_weight_tier=receipt.after_font_weight_tier,
            font_weight_tier_authority=receipt.after_font_weight_tier_authority,
            unavailable_reason="",
            effective_page_fingerprint=(
                receipt.after_effective_page_fingerprint
            ),
        )
        self._state = RenderStyleFontWeightTierEditorState(
            selection=updated_selection,
            phase=RenderStyleFontWeightTierEditorPhase.COMMITTED,
            draft_font_weight_tier=receipt.after_font_weight_tier,
            message=(
                "Font weight restored to Automatic. Preview remains explicit."
                if command.operation
                is RenderStyleFontWeightTierOperation.RESTORE_AUTOMATIC
                else "Font weight tier saved. Preview remains explicit."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderStyleFontWeightTierWorkerFailure,
    ) -> RenderStyleFontWeightTierEditorState:
        if not isinstance(value, RenderStyleFontWeightTierWorkerFailure):
            raise TypeError(
                "value must be RenderStyleFontWeightTierWorkerFailure"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.font_weight_tier,
        )
        phase = (
            RenderStyleFontWeightTierEditorPhase.STALE
            if value.stale
            else RenderStyleFontWeightTierEditorPhase.FAILED
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
        value: RenderStyleFontWeightTierWorkerFailure,
    ) -> RenderStyleFontWeightTierEditorState:
        if not value.stale:
            raise ValueError("font_weight_tier failure is not stale")
        return self.accept_failure(value)

    def accept_cancelled(
        self,
        value: RenderStyleFontWeightTierCancelledReceipt,
    ) -> RenderStyleFontWeightTierEditorState:
        if not isinstance(value, RenderStyleFontWeightTierCancelledReceipt):
            raise TypeError(
                "value must be RenderStyleFontWeightTierCancelledReceipt"
            )
        self._require_active_event(
            value.project_path,
            value.page_id,
            value.parent_id,
            value.operation,
            value.font_weight_tier,
        )
        self._state = replace(
            self._state,
            phase=RenderStyleFontWeightTierEditorPhase.CANCELLED,
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
        selection: RenderStyleFontWeightTierSelection,
    ) -> RenderStyleFontWeightTierEditorState:
        if not isinstance(selection, RenderStyleFontWeightTierSelection):
            raise TypeError(
                "selection must be RenderStyleFontWeightTierSelection"
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
            self._state.draft_font_weight_tier
            if preserve_draft
            else selection.effective_font_weight_tier
        )
        if draft is None:
            draft_dirty = selection.effective_font_weight_tier is not None
        else:
            try:
                canonical_draft = canonical_render_font_weight_tier(draft)
            except (TypeError, ValueError):
                draft_dirty = True
            else:
                baseline = (
                    selection.effective_font_weight_tier
                )
                draft_dirty = canonical_draft != baseline
        phase = (
            RenderStyleFontWeightTierEditorPhase.DIRTY
            if draft_dirty
            else RenderStyleFontWeightTierEditorPhase.READY
        )
        self._state = RenderStyleFontWeightTierEditorState(
            selection=selection,
            phase=phase,
            draft_font_weight_tier=draft,
            message=(
                "Current state changed; review the preserved font_weight_tier draft before applying."
                if preserve_draft and fingerprint_changed
                else "Selection refreshed; the unapplied font_weight_tier draft was preserved."
                if preserve_draft
                else self._ready_message(selection)
            ),
        )
        return self._state

    def _command(
        self,
        operation: RenderStyleFontWeightTierOperation,
        *,
        font_weight_tier: str | None,
    ) -> RenderStyleFontWeightTierWorkerCommand:
        selection = self._state.selection
        return RenderStyleFontWeightTierWorkerCommand(
            project_path=selection.project_path,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            operation=operation,
            font_weight_tier=font_weight_tier,
            expected_effective_page_fingerprint=(
                selection.effective_page_fingerprint
            ),
        )

    def _begin(
        self,
        command: RenderStyleFontWeightTierWorkerCommand,
        message: str,
    ) -> None:
        self._state = replace(
            self._state,
            phase=RenderStyleFontWeightTierEditorPhase.COMMITTING,
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
    ) -> RenderStyleFontWeightTierWorkerCommand:
        command = self._state.worker_command
        if command is None:
            raise RuntimeError("no font_weight_tier worker command is active")
        if command.page_id != page_id or command.parent_id != parent_id:
            raise ValueError("worker event belongs to another selected parent")
        return command

    def _require_active_event(
        self,
        project_path: str,
        page_id: str,
        parent_id: str,
        operation: RenderStyleFontWeightTierOperation,
        font_weight_tier: str | None,
    ) -> RenderStyleFontWeightTierWorkerCommand:
        command = self._require_active_target(page_id, parent_id)
        if (
            os.path.normcase(os.path.abspath(command.project_path))
            != os.path.normcase(os.path.abspath(project_path))
            or command.operation is not operation
            or command.font_weight_tier != font_weight_tier
        ):
            raise ValueError("worker event belongs to another font_weight_tier command")
        return command

    @staticmethod
    def _ready_message(selection: RenderStyleFontWeightTierSelection) -> str:
        if not selection.available:
            return selection.unavailable_reason
        if selection.font_weight_tier_authority == "user":
            return "User font weight tier is effective."
        return "Automatic font weight tier is effective."


__all__ = [
    "RenderStyleFontWeightTierCancellationState",
    "RenderStyleFontWeightTierCancelledReceipt",
    "RenderStyleFontWeightTierEditorModel",
    "RenderStyleFontWeightTierEditorPhase",
    "RenderStyleFontWeightTierEditorState",
    "RenderStyleFontWeightTierSelection",
    "RenderStyleFontWeightTierWorkerBusyState",
    "RenderStyleFontWeightTierWorkerCommand",
    "RenderStyleFontWeightTierWorkerFailure",
    "RenderStyleFontWeightTierWorkerFailureCode",
    "RenderStyleFontWeightTierWorkerReceipt",
    "RenderStyleFontWeightTierWorkerStage",
]
