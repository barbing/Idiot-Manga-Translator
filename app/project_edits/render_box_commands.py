# -*- coding: utf-8 -*-
"""Typed GUI command for one exact renderer target-box override."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

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
    canonical_render_box,
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
    automatic_render_box,
    automatic_render_hard_bounds,
    field_base_fingerprint,
    project_effective_page,
)


_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class RenderLayoutRenderBoxOperation(str, Enum):
    SET = "set"
    RESTORE_AUTOMATIC = "restore_automatic"


class RenderLayoutRenderBoxCommandErrorCode(str, Enum):
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_EXCLUDED = "parent_excluded"
    PARENT_NOT_RENDER_REQUIRED = "parent_not_render_required"
    AUTOMATIC_RENDER_BOX_UNAVAILABLE = "automatic_render_box_unavailable"
    RENDER_BOX_OUTSIDE_HARD_BOUNDS = "render_box_outside_hard_bounds"
    NO_OP = "no_op"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    RENDER_BOX_SLOT_CONFLICT = "render_box_slot_conflict"
    INVALIDATION_UNRESOLVED = "invalidation_unresolved"
    DUPLICATE_COMMAND = "duplicate_command"
    PROJECTION_REJECTED = "projection_rejected"


class RenderLayoutRenderBoxCommandError(RuntimeError):
    def __init__(
        self,
        code: RenderLayoutRenderBoxCommandErrorCode,
        message: str,
    ) -> None:
        self.code = RenderLayoutRenderBoxCommandErrorCode(code)
        super().__init__(str(message or self.code.value))


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxCommand:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    operation: RenderLayoutRenderBoxOperation
    render_box: tuple[int, int, int, int] | None
    expected_effective_page_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _required_id(self.command_id, "command_id"),
        )
        object.__setattr__(
            self,
            "project_id",
            _required_identity(self.project_id, "project_id"),
        )
        for name in ("page_id", "parent_id"):
            object.__setattr__(self, name, _required_id(getattr(self, name), name))
        operation = RenderLayoutRenderBoxOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        if operation is RenderLayoutRenderBoxOperation.SET:
            object.__setattr__(self, "render_box", canonical_render_box(self.render_box))
        elif self.render_box is not None:
            raise ValueError("restore_automatic must not carry a render_box value")
        for name in (
            "expected_effective_page_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(self, name, _required_sha256(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class RenderLayoutRenderBoxCommandReceipt:
    command_id: str
    edit: ProjectEdit
    superseded_edit_id: str | None
    automatic_render_box: tuple[int, int, int, int]
    automatic_hard_bounds: tuple[int, int, int, int]
    before_render_box: tuple[int, int, int, int]
    after_render_box: tuple[int, int, int, int]
    before_render_box_authority: str
    after_render_box_authority: str
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


def _required_identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact identifier")
    return value


def _required_sha256(value: Any, name: str) -> str:
    result = str(value or "").strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return result


def _contains_xywh(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return bool(
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[0] + inner[2] <= outer[0] + outer[2]
        and inner[1] + inner[3] <= outer[1] + outer[3]
    )


def _project_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is unavailable: {page_id}",
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
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PARENT_NOT_FOUND,
            "Automatic parent records are unavailable.",
        )
    matches = [
        parent
        for parent in parents
        if isinstance(parent, Mapping)
        and str(parent.get("parent_id") or "").strip() == parent_id
    ]
    if len(matches) != 1 or str(matches[0].get("page_id") or "").strip() != page_id:
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PARENT_NOT_FOUND,
            f"Automatic parent identity is unavailable: {parent_id}",
        )
    return matches[0]


def _effective_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = [parent for parent in snapshot.parents if parent.parent_id == parent_id]
    if len(matches) != 1:
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PARENT_NOT_FOUND,
            f"Effective parent identity is unavailable: {parent_id}",
        )
    return matches[0]


def _edit_field(edit: ProjectEdit) -> str | None:
    if edit.domain is not EditDomain.RENDER_LAYOUT:
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
        if edit.domain is EditDomain.RENDER_LAYOUT
        and edit.target.kind is EditTargetKind.PARENT
        and edit.target.parent_id == parent_id
        and _edit_field(edit) == "render_box"
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
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.RENDER_BOX_SLOT_CONFLICT,
            "Render box has competing active edits; resolve the conflict first.",
        )
    return heads[0]


def _effective_state(
    parent: EffectiveParentSnapshot,
    *,
    automatic_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], str]:
    overrides = dict(parent.render_layout_overrides)
    if "render_box" not in overrides:
        return automatic_box, "automatic"
    try:
        return canonical_render_box(overrides["render_box"]), "user"
    except (TypeError, ValueError) as exc:
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.PROJECTION_REJECTED,
            "The effective projector produced an invalid render box.",
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
            "render_layout_override",
        ),
        (
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            (parent_id,),
            "render_layout_override",
        ),
    )
    if invalidation.unresolved_facts or actual != expected:
        raise RenderLayoutRenderBoxCommandError(
            RenderLayoutRenderBoxCommandErrorCode.INVALIDATION_UNRESOLVED,
            "Render-box invalidation must affect only parent layout and page output.",
        )


class RenderLayoutRenderBoxCommandService:
    """Persist one exact target box through GUI-owned edit owners only."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: RenderLayoutRenderBoxCommand,
    ) -> RenderLayoutRenderBoxCommandReceipt:
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
        command: RenderLayoutRenderBoxCommand,
    ) -> RenderLayoutRenderBoxCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, RenderLayoutRenderBoxCommand):
            raise TypeError("command must be a RenderLayoutRenderBoxCommand")
        if self._edit_store.project_id != command.project_id:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the command.",
            )
        materialized = snapshot.project
        if (
            project_id_for(materialized) != command.project_id
            or snapshot.ledger.project_id != command.project_id
        ):
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Materialized project identity does not match the command.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed after the render-box command was prepared.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after the render-box command was prepared.",
            )
        if snapshot.ledger.get(command.command_id) is not None:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.DUPLICATE_COMMAND,
                "The render-box command was already recorded.",
            )
        page = _project_page(materialized, command.page_id)
        try:
            before_page = project_effective_page(
                materialized,
                snapshot.ledger,
                page_id=command.page_id,
            )
        except (TypeError, ValueError) as exc:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PROJECTION_REJECTED,
                "The effective page could not be projected before render-box editing.",
            ) from exc
        if before_page.effective_fingerprint != command.expected_effective_page_fingerprint:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STALE_EFFECTIVE_PAGE,
                "Effective page state changed after the command was prepared.",
            )
        before_parent = _effective_parent(before_page, command.parent_id)
        automatic_parent = _automatic_parent(
            page,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        if before_parent.excluded:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PARENT_EXCLUDED,
                "Render box is unavailable while the parent is excluded.",
            )
        if automatic_parent.get("render_required") is not True:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PARENT_NOT_RENDER_REQUIRED,
                "Render box is available only for render-required parents.",
            )
        automatic_box = automatic_render_box(automatic_parent)
        hard_bounds = automatic_render_hard_bounds(automatic_parent)
        if automatic_box is None or hard_bounds is None:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.AUTOMATIC_RENDER_BOX_UNAVAILABLE,
                "The automatic target box or hard bounds are unavailable.",
            )
        slot_head = _active_slot_head(
            snapshot.ledger,
            page_id=command.page_id,
            parent_id=command.parent_id,
        )
        before_box, before_authority = _effective_state(
            before_parent,
            automatic_box=automatic_box,
        )
        if command.operation is RenderLayoutRenderBoxOperation.SET:
            assert command.render_box is not None
            if not _contains_xywh(hard_bounds, command.render_box):
                raise RenderLayoutRenderBoxCommandError(
                    RenderLayoutRenderBoxCommandErrorCode.RENDER_BOX_OUTSIDE_HARD_BOUNDS,
                    "The requested render box must stay inside automatic hard bounds.",
                )
            if command.render_box == before_box:
                raise RenderLayoutRenderBoxCommandError(
                    RenderLayoutRenderBoxCommandErrorCode.NO_OP,
                    "The requested render box is already effective.",
                )
            operation = "set_fields"
            payload: Mapping[str, Any] = {
                "fields": {"render_box": command.render_box}
            }
        else:
            if (
                before_authority == "automatic"
                or slot_head is None
                or slot_head.operation == "restore_automatic"
            ):
                raise RenderLayoutRenderBoxCommandError(
                    RenderLayoutRenderBoxCommandErrorCode.NO_OP,
                    "Render box already uses the automatic value.",
                )
            operation = "restore_automatic"
            payload = {"fields": ("render_box",)}
        target = EditTarget(EditTargetKind.PARENT, parent_id=command.parent_id)
        base_fingerprint = field_base_fingerprint(
            project=materialized,
            page=page,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
            operation=operation,
            payload=payload,
        )
        if base_fingerprint is None:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.AUTOMATIC_RENDER_BOX_UNAVAILABLE,
                "The automatic target box is unavailable for fingerprinting.",
            )
        edit = create_project_edit(
            project_id=command.project_id,
            page_id=command.page_id,
            target=target,
            domain=EditDomain.RENDER_LAYOUT,
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
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector rejected the render-box edit.",
            ) from exc
        after_parent = _effective_parent(after_page, command.parent_id)
        after_box, after_authority = _effective_state(
            after_parent,
            automatic_box=automatic_box,
        )
        accepted = (
            after_box == command.render_box and after_authority == "user"
            if command.operation is RenderLayoutRenderBoxOperation.SET
            else after_box == automatic_box and after_authority == "automatic"
        )
        if edit.edit_id not in after_page.applied_edit_ids or not accepted:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.PROJECTION_REJECTED,
                "The effective projector did not produce the requested render box.",
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
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STALE_PAGE_HEAD,
                "Page edits changed before the render box was committed.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise RenderLayoutRenderBoxCommandError(
                RenderLayoutRenderBoxCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before the render box was committed.",
            ) from exc
        return RenderLayoutRenderBoxCommandReceipt(
            command_id=command.command_id,
            edit=edit,
            superseded_edit_id=(slot_head.edit_id if slot_head is not None else None),
            automatic_render_box=automatic_box,
            automatic_hard_bounds=hard_bounds,
            before_render_box=before_box,
            after_render_box=after_box,
            before_render_box_authority=before_authority,
            after_render_box_authority=after_authority,
            before_effective_page_fingerprint=before_page.effective_fingerprint,
            after_effective_page_fingerprint=after_page.effective_fingerprint,
            invalidation=invalidation,
            effective_page=after_page,
            commit_receipt=commit_receipt,
        )


__all__ = [
    "RenderLayoutRenderBoxCommand",
    "RenderLayoutRenderBoxCommandError",
    "RenderLayoutRenderBoxCommandErrorCode",
    "RenderLayoutRenderBoxCommandReceipt",
    "RenderLayoutRenderBoxCommandService",
    "RenderLayoutRenderBoxOperation",
]
