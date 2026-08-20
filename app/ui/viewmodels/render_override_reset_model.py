"""Typed UI state for reversible render-override reset commands."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
from typing import TYPE_CHECKING, Any, Mapping

from app.project_edits.fingerprints import canonical_sha256, project_id_for
from app.project_edits.render_override_reset_commands import (
    RESETTABLE_RENDER_LAYOUT_FIELDS,
    RESETTABLE_RENDER_STYLE_FIELDS,
    RenderOverrideResetCommandReceipt,
    RenderOverrideResetFieldGroup,
    RenderOverrideResetScope,
    RenderOverrideResetSlot,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


class RenderOverrideResetPhase(str, Enum):
    READY = "ready"
    COMMITTING = "committing"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class RenderOverrideResetWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    VALIDATING_INVENTORY = "validating_inventory"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class RenderOverrideResetWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    NO_OP = "no_op"
    SNAPSHOT_STALE = "snapshot_stale"
    SLOT_CONFLICT = "slot_conflict"
    AUTOMATIC_BASE_UNAVAILABLE = "automatic_base_unavailable"
    COMMAND_REJECTED = "command_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _required_text(value: object, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    return result


def _required_sha256(value: object, field_name: str) -> str:
    result = _required_text(value, field_name).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return result


@dataclass(frozen=True, slots=True)
class RenderOverrideResetSelection:
    project_path: str
    project_id: str
    selected_page_id: str
    selected_parent_id: str
    project_fingerprint: str
    slots: tuple[RenderOverrideResetSlot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_path",
            os.path.abspath(_required_text(self.project_path, "project_path")),
        )
        for field_name in ("project_id", "selected_page_id", "selected_parent_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "project_fingerprint",
            _required_sha256(self.project_fingerprint, "project_fingerprint"),
        )
        slots = tuple(self.slots)
        if any(not isinstance(slot, RenderOverrideResetSlot) for slot in slots):
            raise TypeError("slots must contain RenderOverrideResetSlot values")
        if slots != tuple(sorted(slots)) or len(slots) != len(set(slots)):
            raise ValueError("reset slots must be unique and sorted")
        object.__setattr__(self, "slots", slots)


def render_override_reset_selection_from_projection(
    project_path: str,
    projection: "ProjectUiProjection",
    *,
    selected_page_id: str,
    selected_parent_id: str,
) -> RenderOverrideResetSelection:
    from app.ui.shell.project_projection import ProjectUiProjection

    if not isinstance(projection, ProjectUiProjection):
        raise TypeError("projection must be ProjectUiProjection")
    page = projection.page(selected_page_id)
    page.parent(selected_parent_id)
    slots: list[RenderOverrideResetSlot] = []
    style_fields = frozenset(RESETTABLE_RENDER_STYLE_FIELDS)
    layout_fields = frozenset(RESETTABLE_RENDER_LAYOUT_FIELDS)
    for projected_page in projection.pages:
        for parent in projected_page.parents:
            for field_name in dict(parent.effective.render_style_overrides):
                if field_name in style_fields:
                    slots.append(
                        RenderOverrideResetSlot(
                            page_id=projected_page.effective.page_id,
                            parent_id=parent.effective.parent_id,
                            domain="render_style",
                            field_name=field_name,
                        )
                    )
            for field_name in dict(parent.effective.render_layout_overrides):
                if field_name in layout_fields:
                    slots.append(
                        RenderOverrideResetSlot(
                            page_id=projected_page.effective.page_id,
                            parent_id=parent.effective.parent_id,
                            domain="render_layout",
                            field_name=field_name,
                        )
                    )
    return RenderOverrideResetSelection(
        project_path=project_path,
        project_id=projection.metadata.project_id,
        selected_page_id=page.effective.page_id,
        selected_parent_id=selected_parent_id,
        project_fingerprint=projection.source_project_fingerprint,
        slots=tuple(sorted(slots)),
    )


def filtered_render_override_reset_slots(
    selection: RenderOverrideResetSelection,
    *,
    scope: RenderOverrideResetScope,
    field_group: RenderOverrideResetFieldGroup,
) -> tuple[RenderOverrideResetSlot, ...]:
    if not isinstance(selection, RenderOverrideResetSelection):
        raise TypeError("selection must be RenderOverrideResetSelection")
    scope = RenderOverrideResetScope(scope)
    field_group = RenderOverrideResetFieldGroup(field_group)
    allowed_domains = {
        RenderOverrideResetFieldGroup.STYLE: frozenset({"render_style"}),
        RenderOverrideResetFieldGroup.LAYOUT: frozenset({"render_layout"}),
        RenderOverrideResetFieldGroup.STYLE_AND_LAYOUT: frozenset(
            {"render_style", "render_layout"}
        ),
    }[field_group]
    return tuple(
        slot
        for slot in selection.slots
        if slot.domain in allowed_domains
        and (
            scope is RenderOverrideResetScope.ENTIRE_PROJECT
            or slot.page_id == selection.selected_page_id
        )
        and (
            scope is not RenderOverrideResetScope.SELECTED_PARENT
            or slot.parent_id == selection.selected_parent_id
        )
    )


@dataclass(frozen=True, slots=True)
class RenderOverrideResetWorkerCommand:
    project_path: str
    project_id: str
    scope: RenderOverrideResetScope
    field_group: RenderOverrideResetFieldGroup
    selected_page_id: str
    selected_parent_id: str
    expected_project_fingerprint: str
    expected_slots: tuple[RenderOverrideResetSlot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_path",
            os.path.abspath(_required_text(self.project_path, "project_path")),
        )
        for field_name in ("project_id", "selected_page_id", "selected_parent_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "scope", RenderOverrideResetScope(self.scope))
        object.__setattr__(
            self,
            "field_group",
            RenderOverrideResetFieldGroup(self.field_group),
        )
        object.__setattr__(
            self,
            "expected_project_fingerprint",
            _required_sha256(
                self.expected_project_fingerprint,
                "expected_project_fingerprint",
            ),
        )
        slots = tuple(self.expected_slots)
        if not slots or any(not isinstance(slot, RenderOverrideResetSlot) for slot in slots):
            raise ValueError("expected_slots must contain reset slots")
        if slots != tuple(sorted(slots)) or len(slots) != len(set(slots)):
            raise ValueError("expected reset slots must be unique and sorted")
        object.__setattr__(self, "expected_slots", slots)


@dataclass(frozen=True, slots=True)
class RenderOverrideResetWorkerBusyState:
    command: RenderOverrideResetWorkerCommand
    stage: RenderOverrideResetWorkerStage
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenderOverrideResetCancelledReceipt:
    command: RenderOverrideResetWorkerCommand
    stage: RenderOverrideResetWorkerStage
    message: str = "Render-override reset cancelled before persistence."


@dataclass(frozen=True, slots=True)
class RenderOverrideResetWorkerFailure:
    code: RenderOverrideResetWorkerFailureCode
    stage: RenderOverrideResetWorkerStage
    command: RenderOverrideResetWorkerCommand
    message: str
    exception_type: str = ""
    persistence_committed: bool = False
    command_receipt: RenderOverrideResetCommandReceipt | None = None

    @property
    def stale(self) -> bool:
        return self.code in {
            RenderOverrideResetWorkerFailureCode.SNAPSHOT_STALE,
            RenderOverrideResetWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        } or self.persistence_committed


@dataclass(frozen=True, slots=True)
class RenderOverrideResetWorkerReceipt:
    command: RenderOverrideResetWorkerCommand
    command_receipt: RenderOverrideResetCommandReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"

    def __post_init__(self) -> None:
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.command, RenderOverrideResetWorkerCommand):
            raise TypeError("command must be RenderOverrideResetWorkerCommand")
        if not isinstance(self.command_receipt, RenderOverrideResetCommandReceipt):
            raise TypeError("command_receipt must be RenderOverrideResetCommandReceipt")
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        receipt = self.command_receipt
        if (
            receipt.scope is not self.command.scope
            or receipt.field_group is not self.command.field_group
            or receipt.slots != self.command.expected_slots
        ):
            raise ValueError("reset receipt does not match the worker command")
        if project_id_for(self.project) != self.command.project_id:
            raise ValueError("reset project identity changed")
        if self.projection.metadata.project_id != self.command.project_id:
            raise ValueError("reset projection belongs to another project")
        if canonical_sha256(self.project) != receipt.after_project_fingerprint:
            raise ValueError("reset project is not the committed materialized state")
        if self.projection.source_project_fingerprint != receipt.after_project_fingerprint:
            raise ValueError("reset projection is not the committed project state")
        committed_edit_ids = tuple(
            edit_id
            for commit in receipt.commit_receipts
            for edit_id in commit.edit_ids
        )
        if committed_edit_ids != tuple(edit.edit_id for edit in receipt.edits):
            raise ValueError("reset commit linkage does not match the edits")
        if any(commit.artifact_revision_ids for commit in receipt.commit_receipts):
            raise ValueError("render reset must not publish artifacts")
        for slot in receipt.slots:
            parent = self.projection.page(slot.page_id).parent(slot.parent_id)
            overrides = dict(
                parent.effective.render_style_overrides
                if slot.domain == "render_style"
                else parent.effective.render_layout_overrides
            )
            if slot.field_name in overrides:
                raise ValueError("reset projection retained a selected override")


@dataclass(frozen=True, slots=True)
class RenderOverrideResetState:
    selection: RenderOverrideResetSelection
    scope: RenderOverrideResetScope
    field_group: RenderOverrideResetFieldGroup
    phase: RenderOverrideResetPhase
    message: str
    worker_command: RenderOverrideResetWorkerCommand | None = None
    busy_state: RenderOverrideResetWorkerBusyState | None = None

    @property
    def slots(self) -> tuple[RenderOverrideResetSlot, ...]:
        return filtered_render_override_reset_slots(
            self.selection,
            scope=self.scope,
            field_group=self.field_group,
        )

    @property
    def busy(self) -> bool:
        return self.phase is RenderOverrideResetPhase.COMMITTING

    @property
    def stale(self) -> bool:
        return self.phase is RenderOverrideResetPhase.STALE

    @property
    def reset_enabled(self) -> bool:
        return bool(self.slots) and not self.busy and not self.stale

    @property
    def cancellation_enabled(self) -> bool:
        return bool(
            self.busy_state is not None
            and self.busy_state.cancellation_enabled
        )

    @property
    def stable_for_run(self) -> bool:
        return not self.busy and not self.stale and self.worker_command is None

    @property
    def status_tone(self) -> str:
        if self.phase is RenderOverrideResetPhase.FAILED:
            return "error"
        if self.phase is RenderOverrideResetPhase.STALE:
            return "warning"
        if self.phase is RenderOverrideResetPhase.COMMITTING:
            return "editing"
        if self.phase is RenderOverrideResetPhase.COMMITTED:
            return "ready"
        return "muted"


class RenderOverrideResetModel:
    def __init__(self, selection: RenderOverrideResetSelection) -> None:
        if not isinstance(selection, RenderOverrideResetSelection):
            raise TypeError("selection must be RenderOverrideResetSelection")
        self._state = RenderOverrideResetState(
            selection=selection,
            scope=RenderOverrideResetScope.SELECTED_PARENT,
            field_group=RenderOverrideResetFieldGroup.STYLE_AND_LAYOUT,
            phase=RenderOverrideResetPhase.READY,
            message="Choose a scope and field group to reset.",
        )

    @property
    def state(self) -> RenderOverrideResetState:
        return self._state

    def set_scope(self, value: RenderOverrideResetScope) -> RenderOverrideResetState:
        if self._state.busy:
            raise RuntimeError("cannot change reset scope while a worker is active")
        self._state = replace(
            self._state,
            scope=RenderOverrideResetScope(value),
            phase=RenderOverrideResetPhase.READY,
            message="Reset scope updated; project state is unchanged.",
            worker_command=None,
            busy_state=None,
        )
        return self._state

    def set_field_group(
        self,
        value: RenderOverrideResetFieldGroup,
    ) -> RenderOverrideResetState:
        if self._state.busy:
            raise RuntimeError("cannot change reset fields while a worker is active")
        self._state = replace(
            self._state,
            field_group=RenderOverrideResetFieldGroup(value),
            phase=RenderOverrideResetPhase.READY,
            message="Reset field group updated; project state is unchanged.",
            worker_command=None,
            busy_state=None,
        )
        return self._state

    def begin_reset(self) -> RenderOverrideResetWorkerCommand:
        if not self._state.reset_enabled:
            raise RuntimeError("the selected reset has no applicable render overrides")
        selection = self._state.selection
        command = RenderOverrideResetWorkerCommand(
            project_path=selection.project_path,
            project_id=selection.project_id,
            scope=self._state.scope,
            field_group=self._state.field_group,
            selected_page_id=selection.selected_page_id,
            selected_parent_id=selection.selected_parent_id,
            expected_project_fingerprint=selection.project_fingerprint,
            expected_slots=self._state.slots,
        )
        self._state = replace(
            self._state,
            phase=RenderOverrideResetPhase.COMMITTING,
            message="Resetting selected render overrides...",
            worker_command=command,
            busy_state=None,
        )
        return command

    def accept_busy(
        self,
        value: RenderOverrideResetWorkerBusyState,
    ) -> RenderOverrideResetState:
        self._require_command(value.command)
        self._state = replace(
            self._state,
            message=value.message,
            busy_state=value,
        )
        return self._state

    def accept_receipt(
        self,
        value: RenderOverrideResetWorkerReceipt,
    ) -> RenderOverrideResetState:
        self._require_command(value.command)
        self._state = replace(
            self._state,
            phase=RenderOverrideResetPhase.COMMITTED,
            message=(
                f"Reset {len(value.command_receipt.slots)} render overrides. "
                "Preview remains explicit."
            ),
            worker_command=None,
            busy_state=None,
        )
        return self._state

    def accept_failure(
        self,
        value: RenderOverrideResetWorkerFailure,
    ) -> RenderOverrideResetState:
        self._require_command(value.command)
        self._state = replace(
            self._state,
            phase=(
                RenderOverrideResetPhase.STALE
                if value.stale
                else RenderOverrideResetPhase.FAILED
            ),
            message=value.message,
            worker_command=None,
            busy_state=None,
        )
        return self._state

    def accept_cancelled(
        self,
        value: RenderOverrideResetCancelledReceipt,
    ) -> RenderOverrideResetState:
        self._require_command(value.command)
        self._state = replace(
            self._state,
            phase=RenderOverrideResetPhase.CANCELLED,
            message=value.message,
            worker_command=None,
            busy_state=None,
        )
        return self._state

    def rebind(
        self,
        selection: RenderOverrideResetSelection,
    ) -> RenderOverrideResetState:
        if self._state.busy:
            raise RuntimeError("cannot replace reset selection while a worker is active")
        self._state = RenderOverrideResetState(
            selection=selection,
            scope=self._state.scope,
            field_group=self._state.field_group,
            phase=RenderOverrideResetPhase.READY,
            message="Choose a scope and field group to reset.",
        )
        return self._state

    def _require_command(
        self,
        command: RenderOverrideResetWorkerCommand,
    ) -> RenderOverrideResetWorkerCommand:
        active = self._state.worker_command
        if active is None:
            raise RuntimeError("no render-reset worker command is active")
        if command != active:
            raise ValueError("render-reset worker event belongs to another command")
        return active


__all__ = [
    "RenderOverrideResetCancelledReceipt",
    "RenderOverrideResetModel",
    "RenderOverrideResetPhase",
    "RenderOverrideResetSelection",
    "RenderOverrideResetState",
    "RenderOverrideResetWorkerBusyState",
    "RenderOverrideResetWorkerCommand",
    "RenderOverrideResetWorkerFailure",
    "RenderOverrideResetWorkerFailureCode",
    "RenderOverrideResetWorkerReceipt",
    "RenderOverrideResetWorkerStage",
    "filtered_render_override_reset_slots",
    "render_override_reset_selection_from_projection",
]
