# -*- coding: utf-8 -*-
"""Pure contracts for user-authored effective hierarchy revisions.

This module is an application-facing seam only.  It deliberately contains no
adapter, scheduler, controller, or pipeline-owner implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol, runtime_checkable


EFFECTIVE_HIERARCHY_REVISION_PREFIX = "effective-hierarchy-v1:"
USER_PARENT_ID_PREFIX = "user-parent-v1-"
USER_ROOT_ID_PREFIX = "user-root-v1-"
USER_PARENT_IDENTITY_NAMESPACE = "user_parent_v1"
USER_ROOT_IDENTITY_NAMESPACE = "user_root_v1"

_USER_PARENT_ID = re.compile(r"^user-parent-v1-([0-9a-f]{32})$")
_USER_ROOT_ID = re.compile(r"^user-root-v1-([0-9a-f]{32})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ParentIdentityNamespace(str, Enum):
    AUTOMATIC = "automatic"
    USER_PARENT_V1 = USER_PARENT_IDENTITY_NAMESPACE


class RootIdentityNamespace(str, Enum):
    AUTOMATIC = "automatic"
    USER_ROOT_V1 = USER_ROOT_IDENTITY_NAMESPACE


class ParentOrigin(str, Enum):
    AUTOMATIC = "automatic"
    USER = "user"


class RootEvidenceKind(str, Enum):
    AUTOMATIC = "automatic"
    WORKFLOW_AREA_ONLY = "workflow_area_only"


class RevisionStage(str, Enum):
    HIERARCHY = "hierarchy"
    SOURCE = "source"
    TRANSLATION = "translation"
    CLEANUP_BASE = "cleanup_base"
    SOURCE_STYLE = "source_style"
    RENDER_ELIGIBILITY = "render_eligibility"
    LAYOUT_RENDER = "layout_render"
    PAGE_OUTPUT = "page_output"


class RevisionStageState(str, Enum):
    CURRENT = "current"
    MISSING = "missing"
    STALE = "stale"
    BLOCKED = "blocked"


class RevisionRequiredAction(str, Enum):
    NONE = "none"
    EXPLICIT_RUN = "explicit_run"
    REBUILD = "rebuild"
    RECOMPUTE = "recompute"
    WAIT_FOR_SOURCE = "wait_for_source"
    WAIT_FOR_PREREQUISITES = "wait_for_prerequisites"


class RevisionScope(str, Enum):
    PARENT = "parent"
    PAGE = "page"
    STYLE_CACHE_PREFIX = "style_cache_prefix"


def _require_identity(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty exact identifier")
    return value


def _require_bbox(value: tuple[int, int, int, int], field_name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field_name} must contain four exact integers")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"{field_name} must be a positive page-area bbox")
    return value


def _require_canvas(value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError("canvas_size must contain two positive exact integers")
    if value[0] * value[1] > 50_000_000:
        raise ValueError("canvas_size exceeds the safety limit")
    return value


def user_parent_identity_suffix(parent_id: str) -> str:
    match = _USER_PARENT_ID.fullmatch(str(parent_id or ""))
    if match is None:
        raise ValueError(
            "parent_id must use user-parent-v1-<32 lowercase hex>"
        )
    return match.group(1)


def user_root_identity_suffix(root_id: str) -> str:
    match = _USER_ROOT_ID.fullmatch(str(root_id or ""))
    if match is None:
        raise ValueError("root_id must use user-root-v1-<32 lowercase hex>")
    return match.group(1)


def validate_user_parent_identity_pair(parent_id: str, root_id: str) -> str:
    parent_suffix = user_parent_identity_suffix(parent_id)
    root_suffix = user_root_identity_suffix(root_id)
    if parent_suffix != root_suffix:
        raise ValueError("user parent and root IDs must use the same identity suffix")
    return parent_suffix


@dataclass(frozen=True)
class EffectiveParentLineage:
    parent_id: str
    identity_namespace: ParentIdentityNamespace
    origin: ParentOrigin
    root_id: str
    root_identity_namespace: RootIdentityNamespace
    authored_edit_id: str
    base_revision_id: str
    role: str
    workflow_area_bbox: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    order_policy: str = "append"
    source_parent_id: str | None = None
    source_root_id: str | None = None
    source_authored_edit_id: str | None = None
    split_orientation: str | None = None
    split_ordinal: int | None = None
    source_parent_ids: tuple[str, ...] = ()
    source_root_ids: tuple[str, ...] = ()
    source_automatic_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        object.__setattr__(
            self, "identity_namespace", ParentIdentityNamespace(self.identity_namespace)
        )
        object.__setattr__(self, "origin", ParentOrigin(self.origin))
        object.__setattr__(
            self,
            "root_identity_namespace",
            RootIdentityNamespace(self.root_identity_namespace),
        )
        if self.identity_namespace is not ParentIdentityNamespace.USER_PARENT_V1:
            raise ValueError("user parent lineage requires the user_parent_v1 namespace")
        if self.root_identity_namespace is not RootIdentityNamespace.USER_ROOT_V1:
            raise ValueError("user parent lineage requires the user_root_v1 namespace")
        if self.origin is not ParentOrigin.USER:
            raise ValueError("user parent lineage requires user origin")
        _require_identity(self.authored_edit_id, "authored_edit_id")
        _require_identity(self.base_revision_id, "base_revision_id")
        if self.role not in {"speech", "caption"}:
            raise ValueError("user parent role must be speech or caption")
        bbox = _require_bbox(self.workflow_area_bbox, "workflow_area_bbox")
        canvas = _require_canvas(self.canvas_size)
        if bbox[0] + bbox[2] > canvas[0] or bbox[1] + bbox[3] > canvas[1]:
            raise ValueError("workflow_area_bbox must remain inside canvas_size")
        if self.order_policy == "append":
            if any(
                value is not None
                for value in (
                    self.source_parent_id,
                    self.source_root_id,
                    self.source_authored_edit_id,
                    self.split_orientation,
                    self.split_ordinal,
                )
            ) or any(
                (
                    self.source_parent_ids,
                    self.source_root_ids,
                    self.source_automatic_fingerprints,
                )
            ):
                raise ValueError("appended user parent lineage cannot carry ancestry")
        elif self.order_policy == "replace_source":
            source_parent_id = _require_identity(
                self.source_parent_id,
                "source_parent_id",
            )
            source_root_id = _require_identity(self.source_root_id, "source_root_id")
            validate_user_parent_identity_pair(source_parent_id, source_root_id)
            if source_parent_id == self.parent_id or source_root_id == self.root_id:
                raise ValueError("split child identities must differ from the source")
            _require_identity(self.source_authored_edit_id, "source_authored_edit_id")
            if self.split_orientation not in {"vertical", "horizontal"}:
                raise ValueError("split_orientation must be vertical or horizontal")
            if (
                isinstance(self.split_ordinal, bool)
                or not isinstance(self.split_ordinal, int)
                or self.split_ordinal not in {0, 1}
            ):
                raise ValueError("split_ordinal must be 0 or 1")
            if any(
                (
                    self.source_parent_ids,
                    self.source_root_ids,
                    self.source_automatic_fingerprints,
                )
            ):
                raise ValueError("split child lineage cannot carry merge ancestry")
        elif self.order_policy == "replace_sources":
            if any(
                value is not None
                for value in (
                    self.source_parent_id,
                    self.source_root_id,
                    self.source_authored_edit_id,
                    self.split_orientation,
                    self.split_ordinal,
                )
            ):
                raise ValueError("merged parent lineage cannot carry split ancestry")
            if (
                len(self.source_parent_ids) != 2
                or len(set(self.source_parent_ids)) != 2
                or len(self.source_root_ids) != 2
                or len(self.source_automatic_fingerprints) != 2
            ):
                raise ValueError(
                    "merged parent lineage requires two exact pipeline ancestors"
                )
            for index, value in enumerate(self.source_parent_ids):
                _require_identity(value, f"source_parent_ids[{index}]")
            for index, value in enumerate(self.source_root_ids):
                _require_identity(value, f"source_root_ids[{index}]")
            if self.parent_id in set(self.source_parent_ids):
                raise ValueError("merged parent identity must differ from its sources")
            if self.root_id in set(self.source_root_ids):
                raise ValueError("merged root identity must differ from its sources")
            for index, fingerprint in enumerate(
                self.source_automatic_fingerprints
            ):
                if _SHA256.fullmatch(str(fingerprint or "")) is None:
                    raise ValueError(
                        f"source_automatic_fingerprints[{index}] must be SHA-256"
                    )
        else:
            raise ValueError(
                "user parent order_policy must be append, replace_source, or replace_sources"
            )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "parent_id": self.parent_id,
            "identity_namespace": self.identity_namespace.value,
            "origin": self.origin.value,
            "root_id": self.root_id,
            "root_identity_namespace": self.root_identity_namespace.value,
            "authored_edit_id": self.authored_edit_id,
            "base_revision_id": self.base_revision_id,
            "role": self.role,
            "workflow_area_bbox": list(self.workflow_area_bbox),
            "canvas_size": list(self.canvas_size),
            "order_policy": self.order_policy,
        }
        if self.order_policy == "replace_source":
            result.update(
                {
                    "source_parent_id": self.source_parent_id,
                    "source_root_id": self.source_root_id,
                    "source_authored_edit_id": self.source_authored_edit_id,
                    "split_orientation": self.split_orientation,
                    "split_ordinal": self.split_ordinal,
                }
            )
        elif self.order_policy == "replace_sources":
            result.update(
                {
                    "source_parent_ids": list(self.source_parent_ids),
                    "source_root_ids": list(self.source_root_ids),
                    "source_automatic_fingerprints": list(
                        self.source_automatic_fingerprints
                    ),
                }
            )
        return result


@dataclass(frozen=True)
class EffectiveUserRootSnapshot:
    root_id: str
    identity_namespace: RootIdentityNamespace
    origin: ParentOrigin
    evidence_kind: RootEvidenceKind
    workflow_area_bbox: tuple[int, int, int, int]
    authored_edit_id: str

    def __post_init__(self) -> None:
        user_root_identity_suffix(self.root_id)
        object.__setattr__(
            self, "identity_namespace", RootIdentityNamespace(self.identity_namespace)
        )
        object.__setattr__(self, "origin", ParentOrigin(self.origin))
        object.__setattr__(
            self, "evidence_kind", RootEvidenceKind(self.evidence_kind)
        )
        if self.identity_namespace is not RootIdentityNamespace.USER_ROOT_V1:
            raise ValueError("user root requires the user_root_v1 namespace")
        if self.origin is not ParentOrigin.USER:
            raise ValueError("user root requires user origin")
        if self.evidence_kind is not RootEvidenceKind.WORKFLOW_AREA_ONLY:
            raise ValueError("user root must remain workflow-area-only evidence")
        _require_bbox(self.workflow_area_bbox, "workflow_area_bbox")
        _require_identity(self.authored_edit_id, "authored_edit_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "identity_namespace": self.identity_namespace.value,
            "origin": self.origin.value,
            "evidence_kind": self.evidence_kind.value,
            "workflow_area_bbox": list(self.workflow_area_bbox),
            "authored_edit_id": self.authored_edit_id,
        }


@dataclass(frozen=True)
class ParentStageRequirement:
    parent_id: str
    stage: RevisionStage
    state: RevisionStageState
    required_action: RevisionRequiredAction
    scope: RevisionScope
    subject_id: str
    reason: str
    depends_on: tuple[RevisionStage, ...] = ()

    def __post_init__(self) -> None:
        _require_identity(self.parent_id, "parent_id")
        object.__setattr__(self, "stage", RevisionStage(self.stage))
        object.__setattr__(self, "state", RevisionStageState(self.state))
        object.__setattr__(
            self,
            "required_action",
            RevisionRequiredAction(self.required_action),
        )
        object.__setattr__(self, "scope", RevisionScope(self.scope))
        _require_identity(self.subject_id, "subject_id")
        _require_identity(self.reason, "reason")
        object.__setattr__(
            self,
            "depends_on",
            tuple(RevisionStage(stage) for stage in self.depends_on),
        )
        if self.state is RevisionStageState.CURRENT:
            if self.required_action is not RevisionRequiredAction.NONE:
                raise ValueError("a current stage cannot require an action")
        elif self.required_action is RevisionRequiredAction.NONE:
            raise ValueError("a non-current stage must require an explicit action")

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "stage": self.stage.value,
            "state": self.state.value,
            "required_action": self.required_action.value,
            "scope": self.scope.value,
            "subject_id": self.subject_id,
            "reason": self.reason,
            "depends_on": [stage.value for stage in self.depends_on],
        }


@dataclass(frozen=True)
class HierarchyRevisionDescriptor:
    page_id: str
    automatic_base_revision_id: str
    topology_fingerprint: str
    ordered_parent_ids: tuple[str, ...]
    excluded_parent_ids: tuple[str, ...]
    active_structural_edit_ids: tuple[str, ...]
    user_roots: tuple[EffectiveUserRootSnapshot, ...]
    user_parent_lineage: tuple[EffectiveParentLineage, ...]
    stage_requirements: tuple[ParentStageRequirement, ...]
    fingerprint: str
    revision_id: str

    def __post_init__(self) -> None:
        _require_identity(self.page_id, "page_id")
        _require_identity(self.automatic_base_revision_id, "automatic_base_revision_id")
        for field_name in ("topology_fingerprint", "fingerprint"):
            value = str(getattr(self, field_name) or "").lower()
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
            object.__setattr__(self, field_name, value)
        if self.revision_id != EFFECTIVE_HIERARCHY_REVISION_PREFIX + self.fingerprint:
            raise ValueError("revision_id must bind the exact hierarchy fingerprint")
        if len(set(self.ordered_parent_ids)) != len(self.ordered_parent_ids):
            raise ValueError("ordered_parent_ids must be unique")
        if not set(self.excluded_parent_ids).issubset(self.ordered_parent_ids):
            raise ValueError("excluded_parent_ids must belong to the effective hierarchy")
        if len({root.root_id for root in self.user_roots}) != len(self.user_roots):
            raise ValueError("user root identities must be unique")
        if len({item.parent_id for item in self.user_parent_lineage}) != len(
            self.user_parent_lineage
        ):
            raise ValueError("user parent identities must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "page_id": self.page_id,
            "automatic_base_revision_id": self.automatic_base_revision_id,
            "topology_fingerprint": self.topology_fingerprint,
            "ordered_parent_ids": list(self.ordered_parent_ids),
            "excluded_parent_ids": list(self.excluded_parent_ids),
            "active_structural_edit_ids": list(self.active_structural_edit_ids),
            "user_roots": [root.to_dict() for root in self.user_roots],
            "user_parent_lineage": [item.to_dict() for item in self.user_parent_lineage],
            "stage_requirements": [item.to_dict() for item in self.stage_requirements],
            "fingerprint": self.fingerprint,
            "revision_id": self.revision_id,
        }


@dataclass(frozen=True)
class ExplicitHierarchyRevisionCommand:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    expected_hierarchy_revision_id: str
    requested_stages: tuple[RevisionStage, ...]

    def __post_init__(self) -> None:
        for field_name in ("command_id", "project_id", "page_id", "parent_id"):
            _require_identity(getattr(self, field_name), field_name)
        if not self.expected_hierarchy_revision_id.startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("expected_hierarchy_revision_id is invalid")
        stages = tuple(RevisionStage(stage) for stage in self.requested_stages)
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("requested_stages must be a non-empty unique tuple")
        object.__setattr__(self, "requested_stages", stages)


@dataclass(frozen=True)
class ExplicitHierarchyRevisionReceipt:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    previous_hierarchy_revision_id: str
    hierarchy_revision: HierarchyRevisionDescriptor
    stage_requirements: tuple[ParentStageRequirement, ...]


@runtime_checkable
class HierarchyRevisionApplicationPort(Protocol):
    """Unimplemented future owner port for explicit stage revision publication."""

    def apply_hierarchy_revision(
        self,
        command: ExplicitHierarchyRevisionCommand,
    ) -> ExplicitHierarchyRevisionReceipt:
        ...


__all__ = [
    "EFFECTIVE_HIERARCHY_REVISION_PREFIX",
    "USER_PARENT_ID_PREFIX",
    "USER_ROOT_ID_PREFIX",
    "USER_PARENT_IDENTITY_NAMESPACE",
    "USER_ROOT_IDENTITY_NAMESPACE",
    "EffectiveParentLineage",
    "EffectiveUserRootSnapshot",
    "ExplicitHierarchyRevisionCommand",
    "ExplicitHierarchyRevisionReceipt",
    "HierarchyRevisionApplicationPort",
    "HierarchyRevisionDescriptor",
    "ParentIdentityNamespace",
    "ParentOrigin",
    "ParentStageRequirement",
    "RevisionRequiredAction",
    "RevisionScope",
    "RevisionStage",
    "RevisionStageState",
    "RootEvidenceKind",
    "RootIdentityNamespace",
    "user_parent_identity_suffix",
    "user_root_identity_suffix",
    "validate_user_parent_identity_pair",
]
