# -*- coding: utf-8 -*-
"""Reversible parent/page/project reset of GUI-supported render overrides."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from app.io.project_edit_store import (
    ProjectEditCommitReceipt,
    ProjectEditMultiPageReadSnapshot,
    ProjectEditPageBatch,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
)

from .contracts import (
    EditDomain,
    EditTarget,
    EditTargetKind,
    ProjectEdit,
    create_project_edit,
    thaw_json,
)
from .fingerprints import canonical_sha256, project_id_for
from .invalidation import InvalidationResult, invalidation_for_edit
from .ledger import ProjectEditLedger
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    field_base_fingerprint,
    project_effective_page,
)


RESETTABLE_RENDER_STYLE_FIELDS = (
    "fill_color",
    "font_role",
    "font_weight_tier",
    "outline_color",
    "outline_width",
    "preferred_size",
    "shadow_blur",
    "shadow_color",
    "shadow_enabled",
    "shadow_offset",
)
RESETTABLE_RENDER_LAYOUT_FIELDS = (
    "line_height",
    "render_box",
    "rotation",
    "writing_mode",
)


class RenderOverrideResetScope(str, Enum):
    SELECTED_PARENT = "selected_parent"
    CURRENT_PAGE = "current_page"
    ENTIRE_PROJECT = "entire_project"


class RenderOverrideResetFieldGroup(str, Enum):
    STYLE = "style"
    LAYOUT = "layout"
    STYLE_AND_LAYOUT = "style_and_layout"


class RenderOverrideResetCommandErrorCode(str, Enum):
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STALE_PROJECT = "stale_project"
    STALE_SLOT_INVENTORY = "stale_slot_inventory"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SLOT_CONFLICT = "slot_conflict"
    AUTOMATIC_BASE_UNAVAILABLE = "automatic_base_unavailable"
    DUPLICATE_COMMAND = "duplicate_command"
    NO_OP = "no_op"
    PROJECTION_REJECTED = "projection_rejected"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"


class RenderOverrideResetCommandError(RuntimeError):
    def __init__(
        self,
        code: RenderOverrideResetCommandErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = RenderOverrideResetCommandErrorCode(code)


def _required_identity(value: object, field_name: str) -> str:
    result = str(value or "").strip()
    if not result or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
        for character in result
    ):
        raise ValueError(f"{field_name} must be a path-safe identity")
    return result


def _require_sha256(value: object, field_name: str) -> str:
    result = str(value or "").strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return result


@dataclass(frozen=True, slots=True, order=True)
class RenderOverrideResetSlot:
    page_id: str
    parent_id: str
    domain: str
    field_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_id", _required_identity(self.page_id, "page_id"))
        object.__setattr__(
            self,
            "parent_id",
            _required_identity(self.parent_id, "parent_id"),
        )
        domain = str(self.domain or "").strip()
        allowed = {
            EditDomain.RENDER_STYLE.value: frozenset(RESETTABLE_RENDER_STYLE_FIELDS),
            EditDomain.RENDER_LAYOUT.value: frozenset(RESETTABLE_RENDER_LAYOUT_FIELDS),
        }
        if domain not in allowed:
            raise ValueError("reset slot domain is unsupported")
        field_name = str(self.field_name or "").strip()
        if field_name not in allowed[domain]:
            raise ValueError("reset slot field is unsupported")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "field_name", field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "domain": self.domain,
            "field_name": self.field_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderOverrideResetSlot":
        if not isinstance(value, Mapping):
            raise TypeError("reset slot must be a mapping")
        if frozenset(value) != {
            "page_id",
            "parent_id",
            "domain",
            "field_name",
        }:
            raise ValueError("reset slot has unsupported fields")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class RenderOverrideResetCommand:
    command_id: str
    project_id: str
    scope: RenderOverrideResetScope
    field_group: RenderOverrideResetFieldGroup
    selected_page_id: str
    selected_parent_id: str
    expected_project_fingerprint: str
    expected_slots: tuple[RenderOverrideResetSlot, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "command_id",
            "project_id",
            "selected_page_id",
            "selected_parent_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
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
            _require_sha256(
                self.expected_project_fingerprint,
                "expected_project_fingerprint",
            ),
        )
        slots = tuple(self.expected_slots)
        if any(not isinstance(slot, RenderOverrideResetSlot) for slot in slots):
            raise TypeError("expected_slots must contain reset slots")
        if slots != tuple(sorted(slots)) or len(slots) != len(set(slots)):
            raise ValueError("expected reset slots must be unique and sorted")
        object.__setattr__(self, "expected_slots", slots)


@dataclass(frozen=True, slots=True)
class RenderOverrideResetCommandReceipt:
    command_id: str
    scope: RenderOverrideResetScope
    field_group: RenderOverrideResetFieldGroup
    slots: tuple[RenderOverrideResetSlot, ...]
    edits: tuple[ProjectEdit, ...]
    invalidations: tuple[InvalidationResult, ...]
    commit_receipts: tuple[ProjectEditCommitReceipt, ...]
    affected_page_ids: tuple[str, ...]
    affected_parent_ids: tuple[str, ...]
    before_project_fingerprint: str
    after_project_fingerprint: str


def _page_by_id(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is unavailable: {page_id}",
        )
    return matches[0]


def _parent_by_id(
    page: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = tuple(parent for parent in page.parents if parent.parent_id == parent_id)
    if len(matches) != 1:
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is unavailable: {parent_id}",
        )
    return matches[0]


def _field_for_edit(edit: ProjectEdit) -> str | None:
    if edit.domain not in {EditDomain.RENDER_STYLE, EditDomain.RENDER_LAYOUT}:
        return None
    payload = thaw_json(edit.payload)
    fields = payload.get("fields") if isinstance(payload, Mapping) else None
    if isinstance(fields, Mapping) and len(fields) == 1:
        return str(next(iter(fields)))
    if (
        isinstance(fields, (list, tuple))
        and len(fields) == 1
        and isinstance(fields[0], str)
    ):
        return fields[0]
    return None


def _active_slot_head(
    ledger: ProjectEditLedger,
    slot: RenderOverrideResetSlot,
) -> ProjectEdit:
    domain = EditDomain(slot.domain)
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=slot.page_id)
        if edit.domain is domain
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == slot.parent_id
        and _field_for_edit(edit) == slot.field_name
    )
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(edit for edit in candidates if edit.edit_id not in superseded)
    if len(heads) != 1:
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.SLOT_CONFLICT,
            "A selected render override has competing or missing active heads.",
        )
    return heads[0]


def _scope_pages_and_parents(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    scope: RenderOverrideResetScope,
    selected_page_id: str,
    selected_parent_id: str,
) -> tuple[tuple[EffectivePageSnapshot, tuple[EffectiveParentSnapshot, ...]], ...]:
    raw_pages = project.get("pages")
    if not isinstance(raw_pages, list):
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    page_ids = tuple(
        str(page.get("page_id") or "").strip()
        for page in raw_pages
        if isinstance(page, Mapping)
    )
    if selected_page_id not in page_ids:
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PAGE_NOT_FOUND,
            "The selected page is no longer available.",
        )
    selected_page = project_effective_page(
        project,
        ledger,
        page_id=selected_page_id,
    )
    _parent_by_id(selected_page, selected_parent_id)
    target_page_ids = (
        tuple(sorted(page_ids))
        if scope is RenderOverrideResetScope.ENTIRE_PROJECT
        else (selected_page_id,)
    )
    result: list[
        tuple[EffectivePageSnapshot, tuple[EffectiveParentSnapshot, ...]]
    ] = []
    for page_id in target_page_ids:
        effective = (
            selected_page
            if page_id == selected_page_id
            else project_effective_page(project, ledger, page_id=page_id)
        )
        parents = (
            (_parent_by_id(effective, selected_parent_id),)
            if scope is RenderOverrideResetScope.SELECTED_PARENT
            else tuple(effective.parents)
        )
        result.append((effective, parents))
    return tuple(result)


def render_override_reset_slots(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    scope: RenderOverrideResetScope,
    field_group: RenderOverrideResetFieldGroup,
    selected_page_id: str,
    selected_parent_id: str,
) -> tuple[RenderOverrideResetSlot, ...]:
    """Return the exact currently effective supported override inventory."""

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    if not isinstance(ledger, ProjectEditLedger):
        raise TypeError("ledger must be a ProjectEditLedger")
    scope = RenderOverrideResetScope(scope)
    field_group = RenderOverrideResetFieldGroup(field_group)
    selected_page_id = _required_identity(selected_page_id, "selected_page_id")
    selected_parent_id = _required_identity(
        selected_parent_id,
        "selected_parent_id",
    )
    include_style = field_group in {
        RenderOverrideResetFieldGroup.STYLE,
        RenderOverrideResetFieldGroup.STYLE_AND_LAYOUT,
    }
    include_layout = field_group in {
        RenderOverrideResetFieldGroup.LAYOUT,
        RenderOverrideResetFieldGroup.STYLE_AND_LAYOUT,
    }
    slots: list[RenderOverrideResetSlot] = []
    for page, parents in _scope_pages_and_parents(
        project,
        ledger,
        scope=scope,
        selected_page_id=selected_page_id,
        selected_parent_id=selected_parent_id,
    ):
        for parent in parents:
            if include_style:
                for field_name in sorted(
                    set(dict(parent.render_style_overrides))
                    & set(RESETTABLE_RENDER_STYLE_FIELDS)
                ):
                    slots.append(
                        RenderOverrideResetSlot(
                            page_id=page.page_id,
                            parent_id=parent.parent_id,
                            domain=EditDomain.RENDER_STYLE.value,
                            field_name=field_name,
                        )
                    )
            if include_layout:
                for field_name in sorted(
                    set(dict(parent.render_layout_overrides))
                    & set(RESETTABLE_RENDER_LAYOUT_FIELDS)
                ):
                    slots.append(
                        RenderOverrideResetSlot(
                            page_id=page.page_id,
                            parent_id=parent.parent_id,
                            domain=EditDomain.RENDER_LAYOUT.value,
                            field_name=field_name,
                        )
                    )
    return tuple(sorted(slots))


def _assert_reset_projection(
    *,
    before_pages: Mapping[str, EffectivePageSnapshot],
    after_pages: Mapping[str, EffectivePageSnapshot],
    slots: tuple[RenderOverrideResetSlot, ...],
) -> None:
    selected = {
        (slot.page_id, slot.parent_id, slot.domain, slot.field_name)
        for slot in slots
    }
    if set(before_pages) != set(after_pages):
        raise RenderOverrideResetCommandError(
            RenderOverrideResetCommandErrorCode.PROJECTION_REJECTED,
            "Reset projection changed the project page set.",
        )
    for page_id in sorted(before_pages):
        before = before_pages[page_id]
        after = after_pages[page_id]
        if (
            before.automatic_fingerprint != after.automatic_fingerprint
            or tuple(parent.parent_id for parent in before.parents)
            != tuple(parent.parent_id for parent in after.parents)
        ):
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.PROJECTION_REJECTED,
                "Reset projection changed immutable page or parent evidence.",
            )
        after_by_id = {parent.parent_id: parent for parent in after.parents}
        for before_parent in before.parents:
            after_parent = after_by_id[before_parent.parent_id]
            expected_style = dict(before_parent.render_style_overrides)
            expected_layout = dict(before_parent.render_layout_overrides)
            for candidate in tuple(expected_style):
                if (
                    page_id,
                    before_parent.parent_id,
                    EditDomain.RENDER_STYLE.value,
                    candidate,
                ) in selected:
                    del expected_style[candidate]
            for candidate in tuple(expected_layout):
                if (
                    page_id,
                    before_parent.parent_id,
                    EditDomain.RENDER_LAYOUT.value,
                    candidate,
                ) in selected:
                    del expected_layout[candidate]
            if (
                dict(after_parent.render_style_overrides) != expected_style
                or dict(after_parent.render_layout_overrides) != expected_layout
            ):
                raise RenderOverrideResetCommandError(
                    RenderOverrideResetCommandErrorCode.PROJECTION_REJECTED,
                    "Reset projection changed an unselected render override.",
                )


class RenderOverrideResetCommandService:
    """Persist one atomic reset command without invoking a pipeline owner."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditMultiPageReadSnapshot,
        command: RenderOverrideResetCommand,
    ) -> RenderOverrideResetCommandReceipt:
        if not isinstance(snapshot, ProjectEditMultiPageReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditMultiPageReadSnapshot")
        if not isinstance(command, RenderOverrideResetCommand):
            raise TypeError("command must be a RenderOverrideResetCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        if (
            project_id_for(snapshot.project) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        before_project_fingerprint = canonical_sha256(snapshot.project)
        if before_project_fingerprint != command.expected_project_fingerprint:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.STALE_PROJECT,
                "Project state changed after the reset was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.DUPLICATE_COMMAND,
                "The reset command was already recorded.",
            )
        slots = render_override_reset_slots(
            snapshot.project,
            snapshot.ledger,
            scope=command.scope,
            field_group=command.field_group,
            selected_page_id=command.selected_page_id,
            selected_parent_id=command.selected_parent_id,
        )
        if slots != command.expected_slots:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.STALE_SLOT_INVENTORY,
                "Render overrides changed after the reset inventory was prepared.",
            )
        if not slots:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.NO_OP,
                "The selected scope has no effective supported render overrides.",
            )

        page_ids = tuple(sorted({slot.page_id for slot in slots}))
        before_pages = {
            page_id: project_effective_page(
                snapshot.project,
                snapshot.ledger,
                page_id=page_id,
            )
            for page_id in page_ids
        }
        raw_pages = {
            page_id: _page_by_id(snapshot.project, page_id) for page_id in page_ids
        }
        edits: list[ProjectEdit] = []
        invalidations: list[InvalidationResult] = []
        for index, slot in enumerate(slots):
            before_parent = _parent_by_id(
                before_pages[slot.page_id],
                slot.parent_id,
            )
            slot_head = _active_slot_head(snapshot.ledger, slot)
            domain = EditDomain(slot.domain)
            payload = {"fields": (slot.field_name,)}
            target = EditTarget(EditTargetKind.PARENT, parent_id=slot.parent_id)
            base_fingerprint = field_base_fingerprint(
                project=snapshot.project,
                page=raw_pages[slot.page_id],
                target=target,
                domain=domain,
                operation="restore_automatic",
                payload=payload,
            )
            if base_fingerprint is None:
                raise RenderOverrideResetCommandError(
                    RenderOverrideResetCommandErrorCode.AUTOMATIC_BASE_UNAVAILABLE,
                    "An automatic render value is unavailable for reset.",
                )
            edit = create_project_edit(
                project_id=command.project_id,
                page_id=slot.page_id,
                target=target,
                domain=domain,
                operation="restore_automatic",
                payload=payload,
                base_revision_id=before_parent.base_revision_id,
                base_fingerprint=base_fingerprint,
                supersedes_edit_id=slot_head.edit_id,
                edit_id=f"{command.command_id}-{index:04d}",
            )
            invalidation = invalidation_for_edit(edit)
            if invalidation.unresolved_facts:
                raise RenderOverrideResetCommandError(
                    RenderOverrideResetCommandErrorCode.INVALIDATION_UNRESOLVED,
                    "A selected reset has unresolved invalidation facts.",
                )
            edits.append(edit)
            invalidations.append(invalidation)

        candidate_ledger = snapshot.ledger
        try:
            for edit in edits:
                candidate_ledger = candidate_ledger.append(edit)
            after_pages = {
                page_id: project_effective_page(
                    snapshot.project,
                    candidate_ledger,
                    page_id=page_id,
                )
                for page_id in page_ids
            }
        except (TypeError, ValueError) as exc:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the reset edits.",
            ) from exc
        _assert_reset_projection(
            before_pages=before_pages,
            after_pages=after_pages,
            slots=slots,
        )

        edits_by_page = {
            page_id: tuple(edit for edit in edits if edit.page_id == page_id)
            for page_id in page_ids
        }
        try:
            commit_receipts = self._edit_store.commit_multi_page_edits(
                tuple(
                    ProjectEditPageBatch(
                        page_id=page_id,
                        edits=edits_by_page[page_id],
                        automatic_page_sha256=(
                            before_pages[page_id].automatic_fingerprint
                        ),
                        expected_page_head_sha256=snapshot.page_head(page_id),
                        transaction_id=f"{command.command_id}-p{index:04d}",
                    )
                    for index, page_id in enumerate(page_ids)
                ),
                expected_global_head_sha256=snapshot.global_head_sha256,
            )
        except StalePageEditHeadError as exc:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.STALE_PAGE_HEAD,
                "A page changed before the reset was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderOverrideResetCommandError(
                RenderOverrideResetCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the reset was committed.",
            ) from exc

        after_project = dict(snapshot.project)
        after_project["edit_ledger"] = candidate_ledger.to_dict()
        return RenderOverrideResetCommandReceipt(
            command_id=command.command_id,
            scope=command.scope,
            field_group=command.field_group,
            slots=slots,
            edits=tuple(edits),
            invalidations=tuple(invalidations),
            commit_receipts=commit_receipts,
            affected_page_ids=page_ids,
            affected_parent_ids=tuple(sorted({slot.parent_id for slot in slots})),
            before_project_fingerprint=before_project_fingerprint,
            after_project_fingerprint=canonical_sha256(after_project),
        )
