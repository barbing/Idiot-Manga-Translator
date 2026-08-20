# -*- coding: utf-8 -*-
"""Typed GUI command for one renderer-supported shadow-offset override."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping

from app.io.project_edit_store import (
    ProjectEditCommitReceipt,
    ProjectEditReadSnapshot,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
)

from .contracts import (
    EditDomain,
    EditTarget,
    EditTargetKind,
    ProjectEdit,
    canonical_render_shadow_offset,
    create_project_edit,
    thaw_json,
)
from .fingerprints import project_id_for
from .invalidation import (
    Dependency,
    InvalidationAction,
    InvalidationResult,
    InvalidationScope,
    invalidation_for_edit,
)
from .ledger import ProjectEditLedger
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    automatic_render_shadow_offset,
    field_base_fingerprint,
    project_effective_page,
)


_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class RenderStyleShadowOffsetOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderStyleShadowOffsetCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_SHADOW_UNAVAILABLE = "automatic_shadow_unavailable"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SHADOW_OFFSET_SLOT_CONFLICT = "shadow_offset_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderStyleShadowOffsetCommandError(RuntimeError):
    def __init__(
        self,
        code: RenderStyleShadowOffsetCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderStyleShadowOffsetCommandErrorCode(code)
        super().__init__(str(message or self.code.value))


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetCommand:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderStyleShadowOffsetOperation
    shadow_offset: tuple[float, float] | list[float] | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        for name in ("command_id", "project_id", "page_id", "parent_id"):
            object.__setattr__(self, name, _required_id(getattr(self, name), name))
        operation = RenderStyleShadowOffsetOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderStyleShadowOffsetOperation.SET:
            object.__setattr__(
                self,
                "shadow_offset",
                canonical_render_shadow_offset(self.shadow_offset),
            )
        elif self.shadow_offset is not None:
            raise ValueError("restore_automatic must not carry a shadow_offset value")
        for name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(self, name, _required_sha256(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class RenderStyleShadowOffsetCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_shadow_offset: tuple[float, float]
    before_shadow_offset: tuple[float, float]
    after_shadow_offset: tuple[float, float]
    before_shadow_offset_authority: str
    after_shadow_offset_authority: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    invalidation: InvalidationResult
    effective_page: EffectivePageSnapshot
    commit_receipt: ProjectEditCommitReceipt


def _required_id(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result or not _PATH_SAFE_ID.fullmatch(result):
        raise ValueError(f"{name} must be a path-safe identity")
    return result


def _required_sha256(value: Any, name: str) -> str:
    result = str(value or "").strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return result


def _project_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _automatic_parent(
    page: Mapping[str, Any],
    *,
    page_id: str,
    parent_id: str,
) -> Mapping[str, Any]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1 or str(matches[0].get("page_id") or "").strip() != page_id:
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is unavailable: {parent_id}",
        )
    return matches[0]


def _effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is unavailable: {parent_id}",
        )
    return matches[0]


def _edit_field(edit: ProjectEdit) -> str | None:
    if edit.domain is not EditDomain.RENDER_STYLE:
        return None
    payload = thaw_json(edit.payload)
    fields = payload.get("fields") if isinstance(payload, Mapping) else None
    if isinstance(fields, Mapping) and len(fields) == 1:
        return str(next(iter(fields)))
    if isinstance(fields, (list, tuple)) and len(fields) == 1:
        return str(fields[0])
    return None


def _active_slot_head(
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    parent_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits(page_id=page_id)
        if edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _edit_field(edit) == "shadow_offset"
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded_ids = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(edit for edit in candidates if edit.edit_id not in superseded_ids)
    if len(heads) != 1:
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.SHADOW_OFFSET_SLOT_CONFLICT,
            "Shadow offset has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_shadow_offset: tuple[float, float],
) -> tuple[tuple[float, float], str]:
    overrides = dict(parent.render_style_overrides)
    if "shadow_offset" not in overrides:
        return automatic_shadow_offset, "automatic"
    try:
        return canonical_render_shadow_offset(overrides["shadow_offset"]), "user"
    except (TypeError, ValueError) as exc:
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid shadow offset.",
        ) from exc


def _require_exact_invalidation(
    invalidation: InvalidationResult,
    *,
    parent_id: str,
) -> None:
    actual = tuple(
        (effect.dependency, effect.action, effect.scope, effect.target_ids, effect.reason)
        for effect in invalidation.effects
    )
    expected = (
        (
            Dependency.LAYOUT_RENDER,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PARENT,
            (parent_id,),
            "render_style_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_style_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderStyleShadowOffsetCommandError(
            RenderStyleShadowOffsetCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Shadow-offset invalidation must affect only parent layout and page output.",
        )


class RenderStyleShadowOffsetCommandService:
    """Persist one exact X/Y shadow offset through GUI-owned owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderStyleShadowOffsetCommand,
    ) -> RenderStyleShadowOffsetCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: RenderStyleShadowOffsetCommand,
    ) -> RenderStyleShadowOffsetCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderStyleShadowOffsetCommand):
            raise TypeError("command must be a RenderStyleShadowOffsetCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the shadow-offset command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the shadow-offset command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.DUPLICATE_COMMAND,
                "The shadow-offset command was already recorded.",
            )
        page = _project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before shadow-offset editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _effective_parent(before_page, command.parent_id)
        automatic_parent = _automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PARENT_EXCLUDED,
                "Shadow offset is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Shadow offset is available only for render-required parents.",
            )
        automatic_offset = automatic_render_shadow_offset(automatic_parent)
        if automatic_offset is None:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic parent does not have one valid visible shadow.",
            )
        slot_head = _active_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_offset, before_authority = _effective_state(
            before_parent,
            automatic_shadow_offset=automatic_offset,
        )
        if command.operation is RenderStyleShadowOffsetOperation.SET:
            if command.shadow_offset == before_offset:
                raise RenderStyleShadowOffsetCommandError(
                    RenderStyleShadowOffsetCommandErrorCode.NO_OP,
                    "The requested shadow offset is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"shadow_offset": list(command.shadow_offset or ())}
            }
        else:
            if (
                before_authority == "automatic" and slot_head is None
            ) or (
                slot_head is not None and slot_head.operation == "restore_automatic"
            ):
                raise RenderStyleShadowOffsetCommandError(
                    RenderStyleShadowOffsetCommandErrorCode.NO_OP,
                    "Shadow offset already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("shadow_offset",)}
        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
        )
        if base_fingerprint is None:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.AUTOMATIC_SHADOW_UNAVAILABLE,
                "The automatic shadow is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_STYLE,
            operation=operation,
            payload=payload,
            base_revision_id=before_parent.base_revision_id,
            base_fingerprint=base_fingerprint,
            supersedes_edit_id=(slot_head.edit_id if slot_head is not None else None),
            edit_id=command.command_id,
        )
        invalidation = invalidation_for_edit(edit)
        _require_exact_invalidation(invalidation, parent_id=command.parent_id)
        try:
            candidate_ledger = snapshot.ledger.append(edit)
            after_page = project_effective_page(
                materialized,
                candidate_ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the shadow-offset edit.",
            ) from exc
        after_parent = _effective_parent(after_page, command.parent_id)
        after_offset, after_authority = _effective_state(
            after_parent,
            automatic_shadow_offset=automatic_offset,
        )
        accepted = (
            after_offset == command.shadow_offset and after_authority == "user"
            if command.operation is RenderStyleShadowOffsetOperation.SET
            else after_offset == automatic_offset and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested shadow offset.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                (edit,),
                automatic_page_sha256=before_page.automatic_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before shadow offset was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderStyleShadowOffsetCommandError(
                RenderStyleShadowOffsetCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before shadow offset was committed.",
            ) from exc
        return RenderStyleShadowOffsetCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_shadow_offset=automatic_offset,
            before_shadow_offset=before_offset,
            after_shadow_offset=after_offset,
            before_shadow_offset_authority=before_authority,
            after_shadow_offset_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


__all__ = [
    "RenderStyleShadowOffsetCommand",
    "RenderStyleShadowOffsetCommandError",
    "RenderStyleShadowOffsetCommandErrorCode",
    "RenderStyleShadowOffsetCommandReceipt",
    "RenderStyleShadowOffsetCommandService",
    "RenderStyleShadowOffsetOperation",
]
