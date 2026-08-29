# -*- coding: utf-8 -*-
"""Sole deterministic projection of automated records and active user edits."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

from app.pipeline.hierarchy_revision_contracts import (
    EFFECTIVE_HIERARCHY_REVISION_PREFIX,
    USER_PARENT_IDENTITY_NAMESPACE,
    USER_ROOT_IDENTITY_NAMESPACE,
    EffectiveParentLineage,
    EffectiveUserRootSnapshot,
    HierarchyRevisionDescriptor,
    ParentIdentityNamespace,
    ParentOrigin,
    ParentStageRequirement,
    RevisionRequiredAction,
    RevisionScope,
    RevisionStage,
    RevisionStageState,
    RootEvidenceKind,
    RootIdentityNamespace,
    validate_user_parent_identity_pair,
)
from app.pipeline.ocr_revision_contracts import OcrSourceRevisionArtifact
from app.pipeline.parent_execution_bundle import validate_resolved_render_style
from app.pipeline.translation_revision_contracts import TranslationRevisionArtifact
from app.render.parent_layer_effects import resolve_parent_layer_effects
from app.render.typesetting_contracts import bbox_from_value

from .contracts import (
    CANONICAL_WRITING_MODES,
    EditDomain,
    EditTarget,
    EditTargetKind,
    ParentSourceEvidenceMappingV1,
    ProjectEdit,
    SourceTextRevisionBaseV1,
    TargetTextRevisionBaseV1,
    canonical_render_box,
    canonical_render_fill_color,
    canonical_render_outline_color,
    canonical_render_shadow_color,
    canonical_render_outline_width,
    canonical_render_preferred_size,
    canonical_render_shadow_blur,
    canonical_render_shadow_offset,
    canonical_render_font_role,
    canonical_render_font_weight_tier,
    canonical_render_line_height,
    canonical_render_rotation,
    freeze_json,
    thaw_json,
)
from .fingerprints import (
    automated_state_fingerprint,
    automatic_revision_id,
    canonical_sha256,
    project_id_for,
)
from .ledger import ProjectEditLedger


class ProjectionIssueKind(str, Enum):
    STALE_EDIT_BASE = "stale_edit_base"
    STALE_DEPENDENCY = "stale_dependency"
    ORPHANED = "orphaned"
    CONFLICT = "conflict"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_EFFECTIVE_VALUE = "invalid_effective_value"


class TargetFreshness(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ProjectionIssue:
    kind: ProjectionIssueKind
    page_id: str
    target_kind: str
    target_id: str
    domain: str
    edit_ids: tuple[str, ...]
    reason: str
    expected_fingerprint: str = ""
    observed_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "page_id": self.page_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "domain": self.domain,
            "edit_ids": list(self.edit_ids),
            "reason": self.reason,
            "expected_fingerprint": self.expected_fingerprint,
            "observed_fingerprint": self.observed_fingerprint,
        }


@dataclass(frozen=True)
class EffectiveParentSnapshot:
    parent_id: str
    bundle_id: str | None
    root_id: str
    automatic_fingerprint: str | None
    base_revision_id: str
    automatic_geometry: Any
    geometry: Any
    render_allowed_area: Any
    root_bbox: Any
    role: str
    reading_order: int
    source_text: str | None
    target_text: str | None
    source_authority: str
    target_authority: str
    target_freshness: TargetFreshness
    excluded: bool
    automatic_render_style: tuple[tuple[str, Any], ...]
    render_style_overrides: tuple[tuple[str, Any], ...]
    automatic_render_layout: tuple[tuple[str, Any], ...]
    render_layout_overrides: tuple[tuple[str, Any], ...]
    review_metadata: tuple[tuple[str, Any], ...]
    applied_edit_ids: tuple[str, ...]
    render_override_edit_ids: tuple[str, ...]
    issues: tuple[ProjectionIssue, ...]
    origin: ParentOrigin = ParentOrigin.AUTOMATIC
    identity_namespace: ParentIdentityNamespace = ParentIdentityNamespace.AUTOMATIC
    root_identity_namespace: RootIdentityNamespace = RootIdentityNamespace.AUTOMATIC
    workflow_area_bbox: Any = None
    lineage: EffectiveParentLineage | None = None
    stage_requirements: tuple[ParentStageRequirement, ...] = ()
    source_revision_id: str | None = None
    source_revision_metadata: tuple[tuple[str, Any], ...] = ()
    target_revision_id: str | None = None
    target_revision_metadata: tuple[tuple[str, Any], ...] = ()
    source_evidence_mapping: ParentSourceEvidenceMappingV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "bundle_id": self.bundle_id,
            "root_id": self.root_id,
            "automatic_fingerprint": self.automatic_fingerprint,
            "base_revision_id": self.base_revision_id,
            "automatic_geometry": thaw_json(self.automatic_geometry),
            "geometry": thaw_json(self.geometry),
            "render_allowed_area": thaw_json(self.render_allowed_area),
            "root_bbox": thaw_json(self.root_bbox),
            "role": self.role,
            "reading_order": self.reading_order,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "authority": {
                "source_text": self.source_authority,
                "target_text": self.target_authority,
            },
            "target_freshness": self.target_freshness.value,
            "excluded": self.excluded,
            "automatic_render_style": _mapping_to_dict(
                self.automatic_render_style
            ),
            "render_style_overrides": _mapping_to_dict(
                self.render_style_overrides
            ),
            "automatic_render_layout": _mapping_to_dict(
                self.automatic_render_layout
            ),
            "render_layout_overrides": _mapping_to_dict(
                self.render_layout_overrides
            ),
            "review_metadata": _mapping_to_dict(self.review_metadata),
            "applied_edit_ids": list(self.applied_edit_ids),
            "render_override_edit_ids": list(self.render_override_edit_ids),
            "issues": [issue.to_dict() for issue in self.issues],
            "origin": self.origin.value,
            "identity_namespace": self.identity_namespace.value,
            "root_identity_namespace": self.root_identity_namespace.value,
            "workflow_area_bbox": thaw_json(self.workflow_area_bbox),
            "lineage": self.lineage.to_dict() if self.lineage is not None else None,
            "stage_requirements": [
                requirement.to_dict() for requirement in self.stage_requirements
            ],
            "source_revision_id": self.source_revision_id,
            "source_revision_metadata": _mapping_to_dict(
                self.source_revision_metadata
            ),
            "target_revision_id": self.target_revision_id,
            "target_revision_metadata": _mapping_to_dict(
                self.target_revision_metadata
            ),
            "source_evidence_mapping": (
                self.source_evidence_mapping.to_dict()
                if self.source_evidence_mapping is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EffectiveHierarchySnapshot:
    ordered_parent_ids: tuple[str, ...]
    excluded_parent_ids: tuple[str, ...]
    fingerprint: str
    revision_id: str = ""
    descriptor: HierarchyRevisionDescriptor | None = None
    user_roots: tuple[EffectiveUserRootSnapshot, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordered_parent_ids": list(self.ordered_parent_ids),
            "excluded_parent_ids": list(self.excluded_parent_ids),
            "fingerprint": self.fingerprint,
            "revision_id": self.revision_id,
            "descriptor": (
                self.descriptor.to_dict() if self.descriptor is not None else None
            ),
            "user_roots": [root.to_dict() for root in self.user_roots],
        }


@dataclass(frozen=True)
class EffectivePageSnapshot:
    project_id: str
    page_id: str
    automatic_fingerprint: str
    base_revision_id: str
    cleaned_base_revision_id: str
    cleaned_page_base: Any
    cleaned_base_provenance: str
    hierarchy: EffectiveHierarchySnapshot
    parents: tuple[EffectiveParentSnapshot, ...]
    effective_glossary: tuple[tuple[str, Any], ...]
    applied_edit_ids: tuple[str, ...]
    issues: tuple[ProjectionIssue, ...]
    effective_fingerprint: str
    stage_requirements: tuple[ParentStageRequirement, ...] = ()

    @property
    def is_current(self) -> bool:
        return not self.issues

    @property
    def execution_ready(self) -> bool:
        return not self.issues and all(
            requirement.state is RevisionStageState.CURRENT
            for requirement in self.stage_requirements
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "project_id": self.project_id,
            "page_id": self.page_id,
            "automatic_fingerprint": self.automatic_fingerprint,
            "base_revision_id": self.base_revision_id,
            "cleaned_base_revision_id": self.cleaned_base_revision_id,
            "cleaned_page_base": thaw_json(self.cleaned_page_base),
            "cleaned_base_provenance": self.cleaned_base_provenance,
            "hierarchy": self.hierarchy.to_dict(),
            "parents": [parent.to_dict() for parent in self.parents],
            "effective_glossary": _mapping_to_dict(self.effective_glossary),
            "applied_edit_ids": list(self.applied_edit_ids),
            "issues": [issue.to_dict() for issue in self.issues],
            "stage_requirements": [
                requirement.to_dict() for requirement in self.stage_requirements
            ],
        }
        if include_fingerprint:
            result["effective_fingerprint"] = self.effective_fingerprint
        return result


def _page_id(page: Mapping[str, Any]) -> str:
    return str(page.get("page_id") or "").strip()


def _find_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    if not isinstance(pages, (list, tuple)):
        raise ValueError("project pages must be a list")
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping) and _page_id(page) == page_id
    ]
    if not matches:
        raise KeyError(f"project page is missing: {page_id}")
    if len(matches) != 1:
        raise ValueError(f"project page identity is duplicated: {page_id}")
    return matches[0]


def automatic_page_fingerprint(page: Mapping[str, Any]) -> str:
    return canonical_sha256(page)


def automatic_parent_fingerprint(parent: Mapping[str, Any]) -> str:
    return canonical_sha256(parent)


def effective_source_fingerprint(parent_id: str, text: str) -> str:
    return canonical_sha256({"parent_id": str(parent_id), "text": text})


def _parent_id(parent: Mapping[str, Any]) -> str:
    return str(parent.get("parent_id") or "").strip()


def _automatic_parents(
    page: Mapping[str, Any],
    *,
    expected_page_id: str,
) -> tuple[Mapping[str, Any], ...]:
    parents = page.get("parent_execution_bundles") or ()
    if not isinstance(parents, (list, tuple)):
        raise ValueError("parent_execution_bundles must be a list")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for parent in parents:
        if not isinstance(parent, Mapping):
            raise ValueError("parent execution bundles must be mappings")
        parent_id = _parent_id(parent)
        if not parent_id:
            raise ValueError("parent execution bundle parent_id is missing")
        if parent_id in seen:
            raise ValueError(
                f"parent execution bundle identity is duplicated: {parent_id}"
            )
        bundle_page_id = str(parent.get("page_id") or "").strip()
        if bundle_page_id != expected_page_id:
            raise ValueError(
                "parent execution bundle page identity does not match its page"
            )
        seen.add(parent_id)
        result.append(parent)
    return tuple(result)


def automatic_ordered_parent_ids_for_page(
    page: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the immutable automatic parent order for one exact page.

    Automatic ordinals are read without coercion, fallback, or identity
    tie-breaking, while user-authored permutations remain separate from the
    immutable bundles.
    """

    page_id = _page_id(page)
    parents = _automatic_parents(page, expected_page_id=page_id)
    ordered: list[tuple[int, str]] = []
    seen_ordinals: set[int] = set()
    for parent in parents:
        raw_ordinal = parent.get(
            "reading_order_index",
            parent.get("reading_order"),
        )
        if (
            isinstance(raw_ordinal, bool)
            or not isinstance(raw_ordinal, int)
            or raw_ordinal < 0
        ):
            raise ValueError(
                "automatic reading-order ordinals must be non-negative exact integers"
            )
        ordinal = int(raw_ordinal)
        if ordinal in seen_ordinals:
            raise ValueError(
                "automatic reading-order ordinals must be unique"
            )
        seen_ordinals.add(ordinal)
        ordered.append((ordinal, _parent_id(parent)))
    return tuple(
        parent_id for _, parent_id in sorted(ordered, key=lambda item: item[0])
    )


def _reading_order_lineage_base(
    edit: ProjectEdit,
    ledger: ProjectEditLedger,
    automatic_order: tuple[str, ...],
) -> tuple[str, ...]:
    """Reconstruct the exact order against which one active edit was authored."""

    lineage: list[ProjectEdit] = []
    seen: set[str] = {edit.edit_id}
    supersedes_edit_id = edit.supersedes_edit_id
    while supersedes_edit_id:
        if supersedes_edit_id in seen:
            raise ValueError("reading-order supersession lineage contains a cycle")
        seen.add(supersedes_edit_id)
        predecessor = ledger.get(supersedes_edit_id)
        if (
            predecessor is None
            or predecessor.project_id != edit.project_id
            or predecessor.page_id != edit.page_id
            or predecessor.target.kind is not EditTargetKind.PAGE
            or predecessor.domain is not EditDomain.STRUCTURAL
            or predecessor.operation != "set_reading_order"
        ):
            raise ValueError("reading-order supersession lineage is invalid")
        lineage.append(predecessor)
        supersedes_edit_id = predecessor.supersedes_edit_id

    before_order = automatic_order
    expected_parent_ids = frozenset(automatic_order)
    for predecessor in reversed(lineage):
        proposed_order = tuple(
            str(parent_id)
            for parent_id in predecessor.payload["ordered_parent_ids"]
        )
        selected_parent_id = str(predecessor.payload["selected_parent_id"])
        if (
            len(proposed_order) != len(automatic_order)
            or len(set(proposed_order)) != len(proposed_order)
            or frozenset(proposed_order) != expected_parent_ids
            or selected_parent_id not in expected_parent_ids
            or proposed_order == before_order
        ):
            raise ValueError("reading-order supersession lineage is invalid")
        excluded_parent_ids = _excluded_parent_ids_before_edit(
            ledger,
            predecessor,
        )
        if selected_parent_id in excluded_parent_ids:
            raise ValueError("reading-order lineage selected an excluded parent")
        if any(
            parent_id in excluded_parent_ids
            and proposed_order[index] != parent_id
            for index, parent_id in enumerate(before_order)
        ):
            raise ValueError("reading-order lineage moved an excluded parent")
        before_other = tuple(
            parent_id
            for parent_id in before_order
            if parent_id != selected_parent_id
            and parent_id not in excluded_parent_ids
        )
        proposed_other = tuple(
            parent_id
            for parent_id in proposed_order
            if parent_id != selected_parent_id
            and parent_id not in excluded_parent_ids
        )
        if before_other != proposed_other:
            raise ValueError("reading-order supersession lineage moves multiple parents")
        before_order = proposed_order
    return before_order


def _excluded_parent_ids_before_edit(
    ledger: ProjectEditLedger,
    edit: ProjectEdit,
) -> frozenset[str]:
    try:
        edit_index = next(
            index
            for index, record in enumerate(ledger.edits)
            if record.edit_id == edit.edit_id
        )
    except StopIteration as exc:
        raise ValueError("reading-order lineage edit is unavailable") from exc
    prefix = ProjectEditLedger(
        ledger.edits[:edit_index],
        project_id=ledger.project_id,
    )
    membership_candidates = tuple(
        candidate
        for candidate in prefix.active_edits(page_id=edit.page_id)
        if candidate.domain is EditDomain.STRUCTURAL
        and candidate.target.kind is EditTargetKind.PARENT
        and candidate.operation in {"exclude", "restore"}
    )
    resolved, conflicts = _resolved_edits(membership_candidates)
    if conflicts:
        raise ValueError("reading-order lineage membership state conflicts")
    return frozenset(
        candidate.target.parent_id
        for candidate in resolved
        if candidate.operation == "exclude"
    )


def _first_mapping(parent: Mapping[str, Any], keys: tuple[str, ...]) -> Mapping[str, Any]:
    for key in keys:
        value = parent.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def automatic_render_writing_mode(parent: Mapping[str, Any]) -> str | None:
    """Return the canonical automatic renderer-owned writing mode, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    value = render_style.get("writing_mode")
    if not isinstance(value, str) or value not in CANONICAL_WRITING_MODES:
        return None
    return value


def automatic_render_line_height(parent: Mapping[str, Any]) -> float | None:
    """Return the immutable renderer-owned line-height ratio, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        return canonical_render_line_height(
            render_style.get("line_height"),
            field_name="render_style.line_height",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_rotation(parent: Mapping[str, Any]) -> float | None:
    """Return the immutable renderer-owned clockwise rotation, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if render_style is not None and not isinstance(render_style, Mapping):
        return None
    try:
        resolution = resolve_parent_layer_effects(render_style)
    except (TypeError, ValueError):
        return None
    if resolution.status not in {"resolved", "unavailable"}:
        return None
    if resolution.rotation.availability == "unavailable":
        return 0.0
    if resolution.rotation.availability != "resolved":
        return None
    try:
        return canonical_render_rotation(
            resolution.rotation.degrees_clockwise,
            field_name="render_style.parent_layer_effects.rotation.degrees_clockwise",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_fill_color(parent: Mapping[str, Any]) -> str | None:
    """Return the immutable nested automatic opaque fill color, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    fill = render_style.get("fill")
    if not isinstance(fill, Mapping):
        return None
    try:
        return canonical_render_fill_color(
            fill.get("color"),
            field_name="render_style.fill.color",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_outline_color(parent: Mapping[str, Any]) -> str | None:
    """Return the immutable nested automatic opaque outline color, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    outline = render_style.get("outline")
    if not isinstance(outline, Mapping):
        return None
    try:
        return canonical_render_outline_color(
            outline.get("color"),
            field_name="render_style.outline.color",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_outline_width(parent: Mapping[str, Any]) -> float | None:
    """Return the immutable nested automatic outline width in pixels, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    outline = render_style.get("outline")
    if not isinstance(outline, Mapping):
        return None
    try:
        return canonical_render_outline_width(
            outline.get("target_width_px"),
            field_name="render_style.outline.target_width_px",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_preferred_size(parent: Mapping[str, Any]) -> float | None:
    """Return the immutable automatic preferred em target in pixels, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        return canonical_render_preferred_size(
            render_style.get("target_preferred_em_px"),
            field_name="render_style.target_preferred_em_px",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_shadow_enabled(parent: Mapping[str, Any]) -> bool | None:
    """Return True only for one strict, visible automatic shadow contract."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        resolution = resolve_parent_layer_effects(render_style)
    except (TypeError, ValueError):
        return None
    if resolution.status != "resolved" or resolution.issues:
        return None
    if resolution.shadow.availability != "resolved" or not resolution.shadow.visible:
        return None
    return True


def automatic_render_shadow_color(parent: Mapping[str, Any]) -> str | None:
    """Return one strict visible automatic shadow RGBA color, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        resolution = resolve_parent_layer_effects(render_style)
        value = canonical_render_shadow_color(
            resolution.shadow.color,
            field_name="render_style.parent_layer_effects.shadow.color",
        )
    except (TypeError, ValueError):
        return None
    if resolution.status != "resolved" or resolution.issues:
        return None
    if resolution.shadow.availability != "resolved" or not resolution.shadow.visible:
        return None
    return value


def automatic_render_shadow_blur(parent: Mapping[str, Any]) -> float | None:
    """Return one strict visible automatic shadow blur radius, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        resolution = resolve_parent_layer_effects(render_style)
        value = canonical_render_shadow_blur(
            resolution.shadow.blur_radius_px,
            field_name="render_style.parent_layer_effects.shadow.blur_radius_px",
        )
    except (TypeError, ValueError):
        return None
    if resolution.status != "resolved" or resolution.issues:
        return None
    if resolution.shadow.availability != "resolved" or not resolution.shadow.visible:
        return None
    return value


def automatic_render_shadow_offset(
    parent: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return one strict visible automatic shadow offset, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        resolution = resolve_parent_layer_effects(render_style)
        value = canonical_render_shadow_offset(
            resolution.shadow.offset_px,
            field_name="render_style.parent_layer_effects.shadow.offset_px",
        )
    except (TypeError, ValueError):
        return None
    if resolution.status != "resolved" or resolution.issues:
        return None
    if resolution.shadow.availability != "resolved" or not resolution.shadow.visible:
        return None
    return value


def automatic_render_font_role(parent: Mapping[str, Any]) -> str | None:
    """Return the immutable registered automatic primary font role, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        return canonical_render_font_role(
            render_style.get("primary_font_role"),
            field_name="render_style.primary_font_role",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_font_weight_tier(
    parent: Mapping[str, Any],
) -> str | None:
    """Return the immutable automatic registered font-weight tier, if usable."""

    if not isinstance(parent, Mapping):
        return None
    render_style = parent.get("render_style")
    if not isinstance(render_style, Mapping):
        return None
    try:
        return canonical_render_font_weight_tier(
            render_style.get("font_weight_tier"),
            field_name="render_style.font_weight_tier",
        )
    except (TypeError, ValueError):
        return None


def automatic_render_box(
    parent: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    """Return the adapter-equivalent immutable automatic target box."""

    if not isinstance(parent, Mapping):
        return None
    for field_name in ("render_allowed_area", "parent_bbox"):
        value = bbox_from_value(parent.get(field_name) or ())
        if not value:
            continue
        try:
            return canonical_render_box(
                value,
                field_name=f"automatic.{field_name}",
            )
        except (TypeError, ValueError):
            return None
    return None


def automatic_render_hard_bounds(
    parent: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    """Return the evidence-backed maximum bounds for effective box edits."""

    if isinstance(parent, Mapping):
        domain = parent.get("render_layout_domain")
        if isinstance(domain, Mapping):
            editable = bbox_from_value(domain.get("editable_bounds") or ())
            if editable:
                try:
                    return canonical_render_box(
                        editable,
                        field_name="automatic.render_layout_domain.editable_bounds",
                    )
                except (TypeError, ValueError):
                    return None
    return automatic_render_box(parent)


def _bundle_bbox(parent: Mapping[str, Any], field_name: str) -> Any:
    value = parent.get(field_name) or []
    if not isinstance(value, (list, tuple)) or len(value) not in {0, 4}:
        raise ValueError(f"{field_name} must contain x, y, width, height")
    if value:
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError(f"{field_name} values must be numeric")
        if float(value[2]) <= 0 or float(value[3]) <= 0:
            raise ValueError(f"{field_name} width and height must be positive")
    return freeze_json(value, field_name=field_name)


def _automatic_geometry(parent: Mapping[str, Any]) -> Any:
    return _bundle_bbox(parent, "parent_bbox")


def _mapping_tuple(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (
                str(key),
                freeze_json(item, field_name=f"mapping.{key}"),
            )
            for key, item in value.items()
        )
    )


def _mapping_to_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: thaw_json(item) for key, item in value}


def _source_text(parent: Mapping[str, Any]) -> str:
    value = parent.get("source_text")
    if value is None:
        value = parent.get("ocr_text")
    return str(value or "")


def _target_text(parent: Mapping[str, Any]) -> str:
    value = parent.get("translated_text")
    if value is None:
        value = parent.get("translation")
    return str(value or "")


def _snapshot_parent(
    parent: Mapping[str, Any],
    *,
    page_id: str = "",
    stage_outcomes: Iterable[Mapping[str, Any]] = (),
    cleanup_current: bool = False,
    page_output_current: bool = False,
) -> EffectiveParentSnapshot:
    fingerprint = automatic_parent_fingerprint(parent)
    geometry = _automatic_geometry(parent)
    style = _first_mapping(parent, ("render_style", "resolved_style", "style"))
    layout = _first_mapping(
        parent,
        (
            "render_layout_summary",
            "render_layer_plan",
            "render_plan",
            "render_layout",
            "layout",
        ),
    )
    reading_order = parent.get("reading_order_index", parent.get("reading_order"))
    if (
        isinstance(reading_order, bool)
        or not isinstance(reading_order, int)
        or reading_order < 0
    ):
        raise ValueError(
            "automatic reading-order ordinals must be non-negative exact integers"
        )
    return EffectiveParentSnapshot(
        parent_id=_parent_id(parent),
        bundle_id=str(parent.get("bundle_id") or ""),
        root_id=str(parent.get("root_id") or parent.get("text_block_root_id") or ""),
        automatic_fingerprint=fingerprint,
        base_revision_id=f"automatic-parent:{fingerprint}",
        automatic_geometry=geometry,
        geometry=geometry,
        render_allowed_area=_bundle_bbox(parent, "render_allowed_area"),
        root_bbox=_bundle_bbox(parent, "root_bbox"),
        role=str(parent.get("role") or parent.get("region_type") or ""),
        reading_order=reading_order,
        source_text=_source_text(parent),
        target_text=_target_text(parent),
        source_authority="automatic",
        target_authority="automatic",
        target_freshness=TargetFreshness.CURRENT,
        excluded=False,
        automatic_render_style=_mapping_tuple(style),
        render_style_overrides=(),
        automatic_render_layout=_mapping_tuple(layout),
        render_layout_overrides=(),
        review_metadata=(),
        applied_edit_ids=(),
        render_override_edit_ids=(),
        issues=(),
        stage_requirements=_automatic_parent_stage_requirements(
            page_id=str(page_id or parent.get("page_id") or ""),
            parent=parent,
            outcomes=tuple(stage_outcomes),
            cleanup_current=cleanup_current,
            page_output_current=page_output_current,
        ),
    )


def _automatic_parent_stage_requirements(
    *,
    page_id: str,
    parent: Mapping[str, Any],
    outcomes: tuple[Mapping[str, Any], ...],
    cleanup_current: bool,
    page_output_current: bool,
) -> tuple[ParentStageRequirement, ...]:
    parent_id = _parent_id(parent)
    relevant: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        outcome_parent_ids = {
            str(value) for value in outcome.get("parent_ids") or [] if str(value)
        }
        if outcome_parent_ids and parent_id not in outcome_parent_ids:
            continue
        stage = str(outcome.get("stage") or "")
        if stage:
            relevant[stage] = outcome

    source_current = bool(_source_text(parent))
    translation_current = bool(_target_text(parent))
    style = _first_mapping(parent, ("render_style", "resolved_style", "style"))
    style_current = bool(validate_resolved_render_style(style).accepted)
    render_eligibility_current = bool(
        parent.get("render_decision_id") or cleanup_current
    )
    layout_current = bool(
        parent.get("render_layout_summary")
        or parent.get("renderer_audit_id")
        or page_output_current
    )

    inferred = {
        RevisionStage.HIERARCHY: True,
        RevisionStage.SOURCE: source_current,
        RevisionStage.TRANSLATION: translation_current,
        RevisionStage.CLEANUP_BASE: cleanup_current,
        RevisionStage.SOURCE_STYLE: style_current,
        RevisionStage.RENDER_ELIGIBILITY: render_eligibility_current,
        RevisionStage.LAYOUT_RENDER: layout_current,
        RevisionStage.PAGE_OUTPUT: page_output_current,
    }
    pipeline_stage = {
        RevisionStage.HIERARCHY: "hierarchy",
        RevisionStage.SOURCE: "ocr",
        RevisionStage.TRANSLATION: "translation",
        RevisionStage.CLEANUP_BASE: "cleanup",
        RevisionStage.SOURCE_STYLE: "style",
        RevisionStage.RENDER_ELIGIBILITY: "cleanup",
        RevisionStage.LAYOUT_RENDER: "rendering",
        RevisionStage.PAGE_OUTPUT: "persistence",
    }
    dependencies = {
        RevisionStage.TRANSLATION: (RevisionStage.SOURCE,),
        RevisionStage.CLEANUP_BASE: (RevisionStage.TRANSLATION,),
        RevisionStage.SOURCE_STYLE: (RevisionStage.CLEANUP_BASE,),
        RevisionStage.RENDER_ELIGIBILITY: (RevisionStage.CLEANUP_BASE,),
        RevisionStage.LAYOUT_RENDER: (
            RevisionStage.TRANSLATION,
            RevisionStage.CLEANUP_BASE,
            RevisionStage.SOURCE_STYLE,
        ),
        RevisionStage.PAGE_OUTPUT: (RevisionStage.LAYOUT_RENDER,),
    }
    requirements: list[ParentStageRequirement] = []
    prior_blocked = False
    for stage in RevisionStage:
        outcome = relevant.get(pipeline_stage[stage])
        outcome_state = str((outcome or {}).get("state") or "")
        if outcome_state == "technical_failure":
            state = RevisionStageState.BLOCKED
            prior_blocked = True
        elif prior_blocked:
            state = RevisionStageState.BLOCKED
        elif outcome_state in {"valid", "valid_with_diagnostics"} or inferred[stage]:
            state = RevisionStageState.CURRENT
        else:
            state = RevisionStageState.MISSING
        if state is RevisionStageState.CURRENT:
            action = RevisionRequiredAction.NONE
        elif state is RevisionStageState.BLOCKED:
            action = RevisionRequiredAction.WAIT_FOR_PREREQUISITES
        elif stage is RevisionStage.CLEANUP_BASE:
            action = RevisionRequiredAction.REBUILD
        elif stage in {RevisionStage.LAYOUT_RENDER, RevisionStage.PAGE_OUTPUT}:
            action = RevisionRequiredAction.RECOMPUTE
        else:
            action = RevisionRequiredAction.EXPLICIT_RUN
        scope = (
            RevisionScope.PAGE
            if stage in {RevisionStage.CLEANUP_BASE, RevisionStage.PAGE_OUTPUT}
            else RevisionScope.STYLE_CACHE_PREFIX
            if stage is RevisionStage.SOURCE_STYLE
            else RevisionScope.PARENT
        )
        requirements.append(
            ParentStageRequirement(
                parent_id=parent_id,
                stage=stage,
                state=state,
                required_action=action,
                scope=scope,
                subject_id=page_id if scope is not RevisionScope.PARENT else parent_id,
                reason=(
                    str((outcome or {}).get("error_code") or "technical_stage_failure")
                    if state is RevisionStageState.BLOCKED
                    else "automatic_stage_artifact_current"
                    if state is RevisionStageState.CURRENT
                    else "automatic_stage_artifact_missing"
                ),
                depends_on=dependencies.get(stage, ()),
            )
        )
    return tuple(requirements)


def _stage_outcomes_for_page(
    project: Mapping[str, Any],
    page: Mapping[str, Any],
    page_id: str,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for source in (project.get("stage_outcomes") or [], page.get("stage_outcomes") or []):
        for outcome in source if isinstance(source, (list, tuple)) else []:
            if not isinstance(outcome, Mapping):
                continue
            if str(outcome.get("page_id") or "") != page_id:
                continue
            identity = str(outcome.get("outcome_id") or "") or canonical_sha256(outcome)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(outcome)
    return tuple(records)


def _user_parent_stage_requirements(
    *,
    page_id: str,
    parent_id: str,
    source_current: bool = False,
    translation_current: bool = False,
    cleanup_current: bool = False,
    source_style_current: bool = False,
    render_eligibility_current: bool = False,
) -> tuple[ParentStageRequirement, ...]:
    layout_prerequisites_current = bool(
        source_current
        and translation_current
        and cleanup_current
        and source_style_current
        and render_eligibility_current
    )
    return (
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.HIERARCHY,
            state=RevisionStageState.CURRENT,
            required_action=RevisionRequiredAction.NONE,
            scope=RevisionScope.PAGE,
            subject_id=page_id,
            reason="user_parent_topology_materialized",
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.SOURCE,
            state=(
                RevisionStageState.CURRENT
                if source_current
                else RevisionStageState.MISSING
            ),
            required_action=(
                RevisionRequiredAction.NONE
                if source_current
                else RevisionRequiredAction.EXPLICIT_RUN
            ),
            scope=RevisionScope.PARENT,
            subject_id=parent_id,
            reason=(
                "source_revision_selected"
                if source_current
                else "source_revision_not_published"
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.TRANSLATION,
            state=(
                RevisionStageState.CURRENT
                if translation_current
                else (
                    RevisionStageState.MISSING
                    if source_current
                    else RevisionStageState.BLOCKED
                )
            ),
            required_action=(
                RevisionRequiredAction.NONE
                if translation_current
                else (
                    RevisionRequiredAction.EXPLICIT_RUN
                    if source_current
                    else RevisionRequiredAction.WAIT_FOR_SOURCE
                )
            ),
            scope=RevisionScope.PARENT,
            subject_id=parent_id,
            reason=(
                "translation_revision_selected"
                if translation_current
                else (
                    "translation_revision_not_published"
                    if source_current
                    else "translation_waits_for_source_revision"
                )
            ),
            depends_on=(
                ()
                if source_current or translation_current
                else (RevisionStage.SOURCE,)
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.CLEANUP_BASE,
            state=(
                RevisionStageState.CURRENT
                if cleanup_current
                else RevisionStageState.STALE
            ),
            required_action=(
                RevisionRequiredAction.NONE
                if cleanup_current
                else RevisionRequiredAction.REBUILD
            ),
            scope=RevisionScope.PAGE,
            subject_id=page_id,
            reason=(
                "user_parent_cleanup_coverage_selected"
                if cleanup_current
                else "cleanup_base_does_not_cover_user_parent_topology"
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.SOURCE_STYLE,
            state=(
                RevisionStageState.CURRENT
                if source_style_current
                else RevisionStageState.MISSING
            ),
            required_action=(
                RevisionRequiredAction.NONE
                if source_style_current
                else RevisionRequiredAction.EXPLICIT_RUN
            ),
            scope=RevisionScope.STYLE_CACHE_PREFIX,
            subject_id=page_id,
            reason=(
                "source_style_mapped_from_automatic_evidence"
                if source_style_current
                else "source_style_revision_not_published"
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.RENDER_ELIGIBILITY,
            state=(
                RevisionStageState.CURRENT
                if render_eligibility_current
                else RevisionStageState.MISSING
            ),
            required_action=(
                RevisionRequiredAction.NONE
                if render_eligibility_current
                else RevisionRequiredAction.EXPLICIT_RUN
            ),
            scope=RevisionScope.PARENT,
            subject_id=parent_id,
            reason=(
                "render_eligibility_mapped_from_automatic_evidence"
                if render_eligibility_current
                else "render_eligibility_revision_not_published"
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.LAYOUT_RENDER,
            state=(
                RevisionStageState.STALE
                if layout_prerequisites_current
                else RevisionStageState.BLOCKED
            ),
            required_action=(
                RevisionRequiredAction.RECOMPUTE
                if layout_prerequisites_current
                else RevisionRequiredAction.WAIT_FOR_PREREQUISITES
            ),
            scope=RevisionScope.PARENT,
            subject_id=parent_id,
            reason=(
                "layout_render_ready_for_explicit_preview"
                if layout_prerequisites_current
                else "layout_render_waits_for_owner_revisions"
            ),
            depends_on=(
                RevisionStage.SOURCE,
                RevisionStage.TRANSLATION,
                RevisionStage.CLEANUP_BASE,
                RevisionStage.SOURCE_STYLE,
                RevisionStage.RENDER_ELIGIBILITY,
            ),
        ),
        ParentStageRequirement(
            parent_id=parent_id,
            stage=RevisionStage.PAGE_OUTPUT,
            state=RevisionStageState.STALE,
            required_action=RevisionRequiredAction.RECOMPUTE,
            scope=RevisionScope.PAGE,
            subject_id=page_id,
            reason="page_output_waits_for_layout_render",
            depends_on=(RevisionStage.LAYOUT_RENDER,),
        ),
    )


def _with_current_user_parent_cleanup_coverage(
    parent: EffectiveParentSnapshot,
) -> EffectiveParentSnapshot | None:
    cleanup_requirements = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is RevisionStage.CLEANUP_BASE
    )
    if len(cleanup_requirements) != 1:
        return None
    cleanup_requirement = cleanup_requirements[0]
    current_cleanup = replace(
        cleanup_requirement,
        state=RevisionStageState.CURRENT,
        required_action=RevisionRequiredAction.NONE,
        reason="user_parent_cleanup_coverage_selected",
    )
    return replace(
        parent,
        stage_requirements=tuple(
            current_cleanup
            if requirement.stage is RevisionStage.CLEANUP_BASE
            else requirement
            for requirement in parent.stage_requirements
        ),
    )


def _user_parent_requirements_after_source_override(
    parent: EffectiveParentSnapshot,
    *,
    page_id: str,
) -> tuple[ParentStageRequirement, ...]:
    """Keep independent owner evidence while making translation explicit."""

    current_stages = {
        requirement.stage
        for requirement in parent.stage_requirements
        if requirement.state is RevisionStageState.CURRENT
    }
    return _user_parent_stage_requirements(
        page_id=page_id,
        parent_id=parent.parent_id,
        source_current=True,
        translation_current=False,
        cleanup_current=RevisionStage.CLEANUP_BASE in current_stages,
        source_style_current=RevisionStage.SOURCE_STYLE in current_stages,
        render_eligibility_current=(
            RevisionStage.RENDER_ELIGIBILITY in current_stages
        ),
    )


def _snapshot_user_parent(
    edit: ProjectEdit,
    *,
    reading_order: int,
) -> EffectiveParentSnapshot:
    payload = thaw_json(edit.payload)
    workflow_area_bbox = tuple(int(value) for value in payload["workflow_area_bbox"])
    canvas_size = tuple(int(value) for value in payload["canvas_size"])
    root_id = str(payload["root_id"])
    lineage = EffectiveParentLineage(
        parent_id=edit.target.parent_id,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        origin=ParentOrigin.USER,
        root_id=root_id,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        authored_edit_id=edit.edit_id,
        base_revision_id=edit.base_revision_id,
        role=str(payload["role"]),
        workflow_area_bbox=workflow_area_bbox,
        canvas_size=canvas_size,
        order_policy=str(payload["order_policy"]),
    )
    requirements = _user_parent_stage_requirements(
        page_id=edit.page_id,
        parent_id=edit.target.parent_id,
    )
    return EffectiveParentSnapshot(
        parent_id=edit.target.parent_id,
        bundle_id=None,
        root_id=root_id,
        automatic_fingerprint=None,
        base_revision_id=edit.base_revision_id,
        automatic_geometry=freeze_json(None, field_name="automatic_geometry"),
        geometry=freeze_json(None, field_name="geometry"),
        render_allowed_area=freeze_json(None, field_name="render_allowed_area"),
        root_bbox=freeze_json(None, field_name="root_bbox"),
        role=str(payload["role"]),
        reading_order=reading_order,
        source_text=None,
        target_text=None,
        source_authority="unavailable",
        target_authority="unavailable",
        target_freshness=TargetFreshness.UNAVAILABLE,
        excluded=False,
        automatic_render_style=(),
        render_style_overrides=(),
        automatic_render_layout=(),
        render_layout_overrides=(),
        review_metadata=(),
        applied_edit_ids=(),
        render_override_edit_ids=(),
        issues=(),
        origin=ParentOrigin.USER,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        workflow_area_bbox=freeze_json(
            workflow_area_bbox,
            field_name="workflow_area_bbox",
        ),
        lineage=lineage,
        stage_requirements=requirements,
    )


def _snapshot_split_user_parent(
    edit: ProjectEdit,
    *,
    child_index: int,
    reading_order: int,
) -> EffectiveParentSnapshot:
    payload = thaw_json(edit.payload)
    child_parent_ids = tuple(str(value) for value in payload["child_parent_ids"])
    child_root_ids = tuple(str(value) for value in payload["child_root_ids"])
    child_bboxes = tuple(
        tuple(int(item) for item in bbox)
        for bbox in payload["child_workflow_area_bboxes"]
    )
    child_mapping_values = payload.get("child_source_evidence_mappings")
    child_source_mapping = (
        ParentSourceEvidenceMappingV1.from_dict(child_mapping_values[child_index])
        if child_mapping_values is not None
        else None
    )
    if child_index not in {0, 1}:
        raise ValueError("split child index must be 0 or 1")
    parent_id = child_parent_ids[child_index]
    root_id = child_root_ids[child_index]
    workflow_area_bbox = child_bboxes[child_index]
    canvas_size = tuple(int(value) for value in payload["canvas_size"])
    lineage = EffectiveParentLineage(
        parent_id=parent_id,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        origin=ParentOrigin.USER,
        root_id=root_id,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        authored_edit_id=edit.edit_id,
        base_revision_id=edit.base_revision_id,
        role=str(payload["source_role"]),
        workflow_area_bbox=workflow_area_bbox,
        canvas_size=canvas_size,
        order_policy="replace_source",
        source_parent_id=edit.target.parent_id,
        source_root_id=str(payload["source_root_id"]),
        source_authored_edit_id=str(payload["source_authored_edit_id"]),
        split_orientation=str(payload["orientation"]),
        split_ordinal=child_index,
    )
    requirements = _user_parent_stage_requirements(
        page_id=edit.page_id,
        parent_id=parent_id,
        source_current=child_source_mapping is not None,
        translation_current=(
            child_source_mapping is not None
            and child_source_mapping.target_text is not None
        ),
    )
    return EffectiveParentSnapshot(
        parent_id=parent_id,
        bundle_id=None,
        root_id=root_id,
        automatic_fingerprint=None,
        base_revision_id=edit.base_revision_id,
        automatic_geometry=freeze_json(None, field_name="automatic_geometry"),
        geometry=freeze_json(
            list(workflow_area_bbox) if child_source_mapping is not None else None,
            field_name="geometry",
        ),
        render_allowed_area=freeze_json(None, field_name="render_allowed_area"),
        root_bbox=freeze_json(None, field_name="root_bbox"),
        role=str(payload["source_role"]),
        reading_order=reading_order,
        source_text=(
            child_source_mapping.source_text
            if child_source_mapping is not None
            else None
        ),
        target_text=(
            child_source_mapping.target_text
            if child_source_mapping is not None
            else None
        ),
        source_authority=("user" if child_source_mapping is not None else "unavailable"),
        target_authority=(
            "mapped_automatic"
            if child_source_mapping is not None
            and child_source_mapping.target_text is not None
            else "unavailable"
        ),
        target_freshness=(
            TargetFreshness.CURRENT
            if child_source_mapping is not None
            and child_source_mapping.target_text is not None
            else TargetFreshness.UNAVAILABLE
        ),
        excluded=False,
        automatic_render_style=(),
        render_style_overrides=(),
        automatic_render_layout=(),
        render_layout_overrides=(),
        review_metadata=(),
        applied_edit_ids=(edit.edit_id,),
        render_override_edit_ids=(),
        issues=(),
        origin=ParentOrigin.USER,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        workflow_area_bbox=freeze_json(
            workflow_area_bbox,
            field_name="workflow_area_bbox",
        ),
        lineage=lineage,
        stage_requirements=requirements,
        source_evidence_mapping=child_source_mapping,
    )


def _snapshot_merged_pipeline_parent(
    edit: ProjectEdit,
    *,
    reading_order: int,
    source_evidence_mapping: ParentSourceEvidenceMappingV1,
) -> EffectiveParentSnapshot:
    payload = thaw_json(edit.payload)
    parent_id = edit.target.parent_id
    root_id = str(payload["merged_root_id"])
    merged_bbox = tuple(
        int(value) for value in payload["merged_workflow_area_bbox"]
    )
    canvas_size = tuple(int(value) for value in payload["canvas_size"])
    source_parent_ids = tuple(
        str(value) for value in payload["source_parent_ids"]
    )
    source_root_ids = tuple(str(value) for value in payload["source_root_ids"])
    source_automatic_fingerprints = tuple(
        str(value) for value in payload["source_automatic_fingerprints"]
    )
    lineage = EffectiveParentLineage(
        parent_id=parent_id,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        origin=ParentOrigin.USER,
        root_id=root_id,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        authored_edit_id=edit.edit_id,
        base_revision_id=edit.base_revision_id,
        role=str(payload["source_role"]),
        workflow_area_bbox=merged_bbox,
        canvas_size=canvas_size,
        order_policy="replace_sources",
        source_parent_ids=source_parent_ids,
        source_root_ids=source_root_ids,
        source_automatic_fingerprints=source_automatic_fingerprints,
    )
    requirements = _user_parent_stage_requirements(
        page_id=edit.page_id,
        parent_id=parent_id,
        source_current=True,
        translation_current=source_evidence_mapping.target_text is not None,
    )
    return EffectiveParentSnapshot(
        parent_id=parent_id,
        bundle_id=None,
        root_id=root_id,
        automatic_fingerprint=None,
        base_revision_id=edit.base_revision_id,
        automatic_geometry=freeze_json(None, field_name="automatic_geometry"),
        geometry=freeze_json(list(merged_bbox), field_name="geometry"),
        render_allowed_area=freeze_json(None, field_name="render_allowed_area"),
        root_bbox=freeze_json(None, field_name="root_bbox"),
        role=str(payload["source_role"]),
        reading_order=reading_order,
        source_text=str(payload["merged_source_text"]),
        target_text=source_evidence_mapping.target_text,
        source_authority="user",
        target_authority=(
            "mapped_automatic"
            if source_evidence_mapping.target_text is not None
            else "unavailable"
        ),
        target_freshness=(
            TargetFreshness.CURRENT
            if source_evidence_mapping.target_text is not None
            else TargetFreshness.UNAVAILABLE
        ),
        excluded=False,
        automatic_render_style=(),
        render_style_overrides=(),
        automatic_render_layout=(),
        render_layout_overrides=(),
        review_metadata=(),
        applied_edit_ids=(edit.edit_id,),
        render_override_edit_ids=(),
        issues=(),
        origin=ParentOrigin.USER,
        identity_namespace=ParentIdentityNamespace.USER_PARENT_V1,
        root_identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
        workflow_area_bbox=freeze_json(
            list(merged_bbox),
            field_name="workflow_area_bbox",
        ),
        lineage=lineage,
        stage_requirements=requirements,
        source_evidence_mapping=source_evidence_mapping,
    )


def _ledger_prefix_before_edit(
    ledger: ProjectEditLedger,
    edit: ProjectEdit,
) -> ProjectEditLedger:
    try:
        edit_index = next(
            index
            for index, record in enumerate(ledger.edits)
            if record.edit_id == edit.edit_id
        )
    except StopIteration as exc:
        raise ValueError("edit is unavailable in its ledger") from exc
    return ProjectEditLedger(
        ledger.edits[:edit_index],
        project_id=ledger.project_id,
    )


def _target_slot_has_ancestor(
    ledger: ProjectEditLedger,
    descendant: ProjectEdit,
    ancestor_edit_id: str,
) -> bool:
    """Follow one exact field-local supersession chain to its selected base."""

    cursor: ProjectEdit | None = descendant
    visited: set[str] = set()
    while cursor is not None and cursor.edit_id not in visited:
        visited.add(cursor.edit_id)
        if cursor.edit_id == ancestor_edit_id:
            return True
        predecessor_id = cursor.supersedes_edit_id
        if predecessor_id is None:
            return False
        predecessor = ledger.get(predecessor_id)
        if (
            predecessor is None
            or predecessor.is_control
            or predecessor.page_id != descendant.page_id
            or predecessor.domain is not EditDomain.TARGET_TEXT
            or predecessor.target != descendant.target
        ):
            return False
        cursor = predecessor
    return False


def _source_slot_has_ancestor(
    ledger: ProjectEditLedger,
    descendant: ProjectEdit,
    ancestor_edit_id: str,
) -> bool:
    """Follow one exact source-text supersession chain to its OCR base."""

    cursor: ProjectEdit | None = descendant
    visited: set[str] = set()
    while cursor is not None and cursor.edit_id not in visited:
        visited.add(cursor.edit_id)
        if cursor.edit_id == ancestor_edit_id:
            return True
        predecessor_id = cursor.supersedes_edit_id
        if predecessor_id is None:
            return False
        predecessor = ledger.get(predecessor_id)
        if (
            predecessor is None
            or predecessor.is_control
            or predecessor.page_id != descendant.page_id
            or predecessor.domain is not EditDomain.SOURCE_TEXT
            or predecessor.target != descendant.target
        ):
            return False
        cursor = predecessor
    return False


def _target_id(edit: ProjectEdit) -> str:
    if edit.target.kind is EditTargetKind.PARENT:
        return edit.target.parent_id
    if edit.target.kind is EditTargetKind.ARTIFACT:
        return edit.target.artifact_id
    if edit.target.kind is EditTargetKind.PAGE:
        return edit.page_id
    if edit.target.kind is EditTargetKind.PROJECT:
        return edit.project_id
    return edit.target.edit_id


def _edit_fields(edit: ProjectEdit) -> tuple[str, ...]:
    if edit.domain in {
        EditDomain.RENDER_STYLE,
        EditDomain.RENDER_LAYOUT,
        EditDomain.REVIEW_METADATA,
    }:
        fields = edit.payload.get("fields")
        if isinstance(fields, Mapping):
            return tuple(sorted(str(field) for field in fields))
        if isinstance(fields, tuple):
            return tuple(sorted(str(field) for field in fields))
        return ("*",)
    if edit.domain is EditDomain.STRUCTURAL:
        return {
            "add_user_parent": ("add_user_parent",),
            "split_user_parent": ("split_user_parent",),
            "merge_pipeline_parents": ("merge_pipeline_parents",),
            "exclude": ("excluded",),
            "restore": ("excluded",),
            "set_geometry": ("geometry",),
            "set_reading_order": ("reading_order",),
            "set_role": ("role",),
        }.get(edit.operation, (edit.operation,))
    if edit.domain is EditDomain.GLOSSARY:
        entry = edit.payload.get("entry")
        if isinstance(entry, Mapping):
            return (str(entry.get("entry_id") or ""),)
        return (str(edit.payload.get("entry_id") or ""),)
    return (edit.domain.value,)


def _issue(
    kind: ProjectionIssueKind,
    edit: ProjectEdit,
    reason: str,
    *,
    edit_ids: Iterable[str] | None = None,
    expected_fingerprint: str = "",
    observed_fingerprint: str = "",
) -> ProjectionIssue:
    return ProjectionIssue(
        kind=kind,
        page_id=edit.page_id,
        target_kind=edit.target.kind.value,
        target_id=_target_id(edit),
        domain=edit.domain.value,
        edit_ids=tuple(edit_ids or (edit.edit_id,)),
        reason=reason,
        expected_fingerprint=expected_fingerprint,
        observed_fingerprint=observed_fingerprint,
    )


def _artifact_index(
    project: Mapping[str, Any],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    catalogs = project.get("artifact_revisions")
    if not isinstance(catalogs, Mapping):
        return result
    for catalog, values in catalogs.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            revision_id = str(value.get("revision_id") or "")
            if revision_id:
                if revision_id in result:
                    raise ValueError(
                        f"artifact revision identity is duplicated: {revision_id}"
                    )
                result[revision_id] = (str(catalog), value)
    return result


def _source_revision_for_edit(
    artifact_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    edit: ProjectEdit,
) -> OcrSourceRevisionArtifact | None:
    if edit.domain is not EditDomain.SOURCE_TEXT or edit.operation != "select_revision":
        return None
    revision_id = str(edit.payload.get("revision_id") or "")
    indexed = artifact_index.get(revision_id)
    if indexed is None or indexed[0] != "source_revisions":
        return None
    try:
        artifact = OcrSourceRevisionArtifact.from_record(indexed[1])
    except (TypeError, ValueError):
        return None
    if (
        artifact.revision_id != revision_id
        or artifact.page_id != edit.page_id
        or artifact.parent_id != edit.target.parent_id
        or artifact.selection_edit_id != edit.edit_id
    ):
        return None
    return artifact


def _source_revision_for_base(
    artifact_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    edit: ProjectEdit,
    revision_base: SourceTextRevisionBaseV1,
) -> OcrSourceRevisionArtifact | None:
    indexed = artifact_index.get(revision_base.source_revision_id)
    if indexed is None or indexed[0] != "source_revisions":
        return None
    try:
        artifact = OcrSourceRevisionArtifact.from_record(indexed[1])
    except (TypeError, ValueError):
        return None
    if (
        artifact.revision_id != revision_base.source_revision_id
        or artifact.page_id != edit.page_id
        or artifact.parent_id != edit.target.parent_id
        or artifact.selection_edit_id != revision_base.selection_edit_id
        or canonical_sha256(artifact.to_record())
        != revision_base.artifact_sha256
        or effective_source_fingerprint(
            artifact.parent_id,
            artifact.source_text,
        )
        != revision_base.source_fingerprint
        or artifact.hierarchy_revision_id
        != revision_base.hierarchy_revision_id
        or artifact.hierarchy_fingerprint
        != revision_base.hierarchy_fingerprint
    ):
        return None
    return artifact


def _translation_revision_for_edit(
    artifact_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    edit: ProjectEdit,
) -> TranslationRevisionArtifact | None:
    if edit.domain is not EditDomain.TARGET_TEXT or edit.operation != "select_revision":
        return None
    revision_id = str(edit.payload.get("revision_id") or "")
    indexed = artifact_index.get(revision_id)
    if indexed is None or indexed[0] != "translation_revisions":
        return None
    try:
        artifact = TranslationRevisionArtifact.from_record(indexed[1])
    except (TypeError, ValueError):
        return None
    if (
        artifact.revision_id != revision_id
        or artifact.page_id != edit.page_id
        or artifact.parent_id != edit.target.parent_id
        or artifact.selection_edit_id != edit.edit_id
    ):
        return None
    return artifact


def _translation_revision_for_base(
    artifact_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    edit: ProjectEdit,
    revision_base: TargetTextRevisionBaseV1,
) -> TranslationRevisionArtifact | None:
    indexed = artifact_index.get(revision_base.translation_revision_id)
    if indexed is None or indexed[0] != "translation_revisions":
        return None
    try:
        artifact = TranslationRevisionArtifact.from_record(indexed[1])
    except (TypeError, ValueError):
        return None
    if (
        artifact.revision_id != revision_base.translation_revision_id
        or artifact.page_id != edit.page_id
        or artifact.parent_id != edit.target.parent_id
        or artifact.selection_edit_id != revision_base.selection_edit_id
        or canonical_sha256(artifact.to_record())
        != revision_base.artifact_sha256
        or artifact.source_fingerprint != revision_base.source_fingerprint
        or artifact.source_revision_id != revision_base.source_revision_id
        or artifact.source_selection_edit_id
        != revision_base.source_selection_edit_id
        or artifact.hierarchy_revision_id != revision_base.hierarchy_revision_id
        or artifact.hierarchy_fingerprint != revision_base.hierarchy_fingerprint
    ):
        return None
    return artifact


def target_text_revision_base_for_parent(
    parent: EffectiveParentSnapshot,
) -> TargetTextRevisionBaseV1 | None:
    """Return the exact immutable model base retained by one target lane."""

    if not isinstance(parent, EffectiveParentSnapshot):
        raise TypeError("parent must be an EffectiveParentSnapshot")
    if parent.target_revision_id is None and not parent.target_revision_metadata:
        return None
    if parent.target_revision_id is None or not parent.target_revision_metadata:
        raise ValueError("selected translation revision metadata is incomplete")
    artifact = TranslationRevisionArtifact.from_record(
        dict(parent.target_revision_metadata)
    )
    if artifact.revision_id != parent.target_revision_id:
        raise ValueError("selected translation revision identity is inconsistent")
    return TargetTextRevisionBaseV1(
        translation_revision_id=artifact.revision_id,
        selection_edit_id=artifact.selection_edit_id,
        artifact_sha256=canonical_sha256(artifact.to_record()),
        source_fingerprint=artifact.source_fingerprint,
        source_revision_id=artifact.source_revision_id,
        source_selection_edit_id=artifact.source_selection_edit_id,
        hierarchy_revision_id=artifact.hierarchy_revision_id,
        hierarchy_fingerprint=artifact.hierarchy_fingerprint,
    )


def source_text_revision_base_for_parent(
    parent: EffectiveParentSnapshot,
) -> SourceTextRevisionBaseV1 | None:
    """Return the immutable OCR base retained by one source override lane."""

    if not isinstance(parent, EffectiveParentSnapshot):
        raise TypeError("parent must be an EffectiveParentSnapshot")
    if parent.source_revision_id is None and not parent.source_revision_metadata:
        return None
    if parent.source_revision_id is None or not parent.source_revision_metadata:
        raise ValueError("selected OCR revision metadata is incomplete")
    artifact = OcrSourceRevisionArtifact.from_record(
        dict(parent.source_revision_metadata)
    )
    if artifact.revision_id != parent.source_revision_id:
        raise ValueError("selected OCR revision identity is inconsistent")
    return SourceTextRevisionBaseV1(
        source_revision_id=artifact.revision_id,
        selection_edit_id=artifact.selection_edit_id,
        artifact_sha256=canonical_sha256(artifact.to_record()),
        source_fingerprint=effective_source_fingerprint(
            artifact.parent_id,
            artifact.source_text,
        ),
        hierarchy_revision_id=artifact.hierarchy_revision_id,
        hierarchy_fingerprint=artifact.hierarchy_fingerprint,
    )


def _valid_cleaned_record(value: Mapping[str, Any]) -> bool:
    asset = str(value.get("asset") or "").strip()
    content_sha256 = str(value.get("content_sha256") or "").lower()
    return bool(
        value.get("valid")
        and asset
        and len(content_sha256) == 64
        and all(character in "0123456789abcdef" for character in content_sha256)
    )


def _cleaned_base_state(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    state = str(value.get("state") or "").strip()
    nested = value.get("cleaned_page_base")
    if not state and isinstance(nested, Mapping):
        state = str(nested.get("state") or "").strip()
    return state


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


_MANUAL_CLEANUP_LINEAGE_VERSION = "manual_cleanup_automatic_lineage_v1"
_MANUAL_CLEANUP_RECEIPT_VERSION = "manual_cleanup_receipt_v1"
_USER_PARENT_CLEANUP_COVERAGE_TARGET_VERSION = (
    "user_parent_cleanup_coverage_target_v1"
)
_USER_PARENT_CLEANUP_COVERAGE_FIELDS = frozenset(
    {
        "schema_version",
        "project_id",
        "page_id",
        "parent_id",
        "root_id",
        "parent_authored_edit_id",
        "parent_role",
        "workflow_area_bbox",
        "canvas_size",
        "original_page_asset_id",
        "original_page_asset_reference",
        "original_page_content_sha256",
        "input_cleaned_base_revision_id",
        "input_cleaned_base_content_sha256",
        "hierarchy_revision_id",
        "hierarchy_fingerprint",
        "source_revision_id",
        "source_selection_edit_id",
        "source_artifact_sha256",
        "source_fingerprint",
        "translation_revision_id",
        "translation_selection_edit_id",
        "translation_artifact_sha256",
        "effective_page_fingerprint",
        "coverage_dependency_fingerprint",
    }
)


def _user_parent_cleanup_coverage_target(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != set(
        _USER_PARENT_CLEANUP_COVERAGE_FIELDS
    ):
        return None
    target = thaw_json(
        freeze_json(value, field_name="user_parent_cleanup_coverage_target")
    )
    if target.get("schema_version") != _USER_PARENT_CLEANUP_COVERAGE_TARGET_VERSION:
        return None
    for field_name in (
        "project_id",
        "page_id",
        "parent_id",
        "root_id",
        "parent_authored_edit_id",
        "parent_role",
        "original_page_asset_id",
        "original_page_asset_reference",
        "input_cleaned_base_revision_id",
        "hierarchy_revision_id",
        "source_revision_id",
        "source_selection_edit_id",
        "translation_revision_id",
        "translation_selection_edit_id",
    ):
        if not str(target.get(field_name) or "").strip():
            return None
    try:
        validate_user_parent_identity_pair(
            str(target["parent_id"]),
            str(target["root_id"]),
        )
    except (TypeError, ValueError):
        return None
    if not str(target["hierarchy_revision_id"]).startswith(
        EFFECTIVE_HIERARCHY_REVISION_PREFIX
    ):
        return None
    canvas_size = target.get("canvas_size")
    bbox = target.get("workflow_area_bbox")
    if (
        not isinstance(canvas_size, list)
        or len(canvas_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in canvas_size)
        or any(item <= 0 for item in canvas_size)
        or canvas_size[0] * canvas_size[1] > 50_000_000
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in bbox)
    ):
        return None
    x, y, width, height = bbox
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > canvas_size[0]
        or y + height > canvas_size[1]
    ):
        return None
    for field_name in (
        "original_page_content_sha256",
        "input_cleaned_base_content_sha256",
        "hierarchy_fingerprint",
        "source_artifact_sha256",
        "source_fingerprint",
        "translation_artifact_sha256",
        "effective_page_fingerprint",
        "coverage_dependency_fingerprint",
    ):
        if not _is_sha256(target.get(field_name)):
            return None
    expected_asset_id = "original-page-v1-" + canonical_sha256(
        {
            "page_id": target["page_id"],
            "asset_reference": target["original_page_asset_reference"],
        }
    )
    if target["original_page_asset_id"] != expected_asset_id:
        return None
    body = dict(target)
    observed = str(body.pop("coverage_dependency_fingerprint") or "").lower()
    if canonical_sha256(body) != observed:
        return None
    return target


def _page_original_asset_reference(page: Mapping[str, Any]) -> str:
    cleaned = page.get("cleaned_page_base")
    if not isinstance(cleaned, Mapping):
        cleaned = {}
    nested = cleaned.get("cleaned_page_base")
    if not isinstance(nested, Mapping):
        nested = {}
    return str(
        page.get("image_path")
        or page.get("source_image_path")
        or cleaned.get("source_image_path")
        or nested.get("source_image_path")
        or ""
    ).strip()


def _user_parent_cleanup_target_matches_snapshot(
    target: Mapping[str, Any],
    snapshot: EffectivePageSnapshot,
    *,
    page: Mapping[str, Any],
) -> bool:
    if (
        target.get("project_id") != snapshot.project_id
        or target.get("page_id") != snapshot.page_id
        or target.get("hierarchy_revision_id")
        != snapshot.hierarchy.revision_id
        or target.get("hierarchy_fingerprint")
        != snapshot.hierarchy.fingerprint
        or target.get("effective_page_fingerprint")
        != snapshot.effective_fingerprint
    ):
        return False
    parent_id = str(target.get("parent_id") or "")
    matches = tuple(
        parent for parent in snapshot.parents if parent.parent_id == parent_id
    )
    if len(matches) != 1:
        return False
    parent = matches[0]
    lineage = parent.lineage
    if (
        parent.origin is not ParentOrigin.USER
        or lineage is None
        or parent.excluded
        or parent.root_id != target.get("root_id")
        or lineage.root_id != target.get("root_id")
        or lineage.authored_edit_id != target.get("parent_authored_edit_id")
        or lineage.authored_edit_id not in parent.applied_edit_ids
        or parent.role != target.get("parent_role")
        or lineage.role != target.get("parent_role")
        or list(lineage.workflow_area_bbox)
        != target.get("workflow_area_bbox")
        or list(lineage.canvas_size) != target.get("canvas_size")
    ):
        return False
    declared_canvas = page.get("canvas_size")
    if isinstance(declared_canvas, (list, tuple)) and len(declared_canvas) == 2:
        if list(declared_canvas) != target.get("canvas_size"):
            return False
    source_requirements = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is RevisionStage.SOURCE
    )
    translation_requirements = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is RevisionStage.TRANSLATION
    )
    cleanup_requirements = tuple(
        requirement
        for requirement in parent.stage_requirements
        if requirement.stage is RevisionStage.CLEANUP_BASE
    )
    if not (
        len(source_requirements) == 1
        and source_requirements[0].state is RevisionStageState.CURRENT
        and source_requirements[0].required_action is RevisionRequiredAction.NONE
        and len(translation_requirements) == 1
        and translation_requirements[0].state is RevisionStageState.CURRENT
        and translation_requirements[0].required_action
        is RevisionRequiredAction.NONE
        and len(cleanup_requirements) == 1
        and cleanup_requirements[0].state is RevisionStageState.STALE
        and cleanup_requirements[0].required_action
        is RevisionRequiredAction.REBUILD
    ):
        return False
    cleaned = thaw_json(snapshot.cleaned_page_base)
    if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
        return False
    if (
        snapshot.cleaned_base_revision_id
        != target.get("input_cleaned_base_revision_id")
        or str(cleaned.get("revision_id") or "")
        != snapshot.cleaned_base_revision_id
        or str(cleaned.get("content_sha256") or "").lower()
        != target.get("input_cleaned_base_content_sha256")
    ):
        return False
    source = _mapping_to_dict(parent.source_revision_metadata)
    translation = _mapping_to_dict(parent.target_revision_metadata)
    try:
        source_artifact = OcrSourceRevisionArtifact.from_record(source)
        translation_artifact = TranslationRevisionArtifact.from_record(
            translation
        )
    except (TypeError, ValueError):
        return False
    source_fingerprint = effective_source_fingerprint(
        parent.parent_id,
        parent.source_text,
    )
    if (
        parent.source_revision_id != source_artifact.revision_id
        or source_artifact.page_id != snapshot.page_id
        or source_artifact.parent_id != parent.parent_id
        or source_artifact.root_id != lineage.root_id
        or source_artifact.parent_authored_edit_id
        != lineage.authored_edit_id
        or source_artifact.original_page.asset_id
        != target.get("original_page_asset_id")
        or source_artifact.original_page.asset_reference
        != target.get("original_page_asset_reference")
        or source_artifact.original_page.content_sha256
        != target.get("original_page_content_sha256")
        or list(source_artifact.original_page.canvas_size)
        != target.get("canvas_size")
        or list(source_artifact.sampling_bbox)
        != target.get("workflow_area_bbox")
        or source_artifact.hierarchy_revision_id
        != snapshot.hierarchy.revision_id
        or source_artifact.hierarchy_fingerprint
        != snapshot.hierarchy.fingerprint
        or source_artifact.source_text != parent.source_text
        or source_artifact.revision_id != target.get("source_revision_id")
        or source_artifact.selection_edit_id
        != target.get("source_selection_edit_id")
        or source_artifact.selection_edit_id not in parent.applied_edit_ids
        or canonical_sha256(source) != target.get("source_artifact_sha256")
        or source_fingerprint != target.get("source_fingerprint")
        or parent.target_revision_id != translation_artifact.revision_id
        or translation_artifact.project_id != snapshot.project_id
        or translation_artifact.page_id != snapshot.page_id
        or translation_artifact.parent_id != parent.parent_id
        or translation_artifact.root_id != lineage.root_id
        or translation_artifact.parent_authored_edit_id
        != lineage.authored_edit_id
        or translation_artifact.parent_role != parent.role
        or translation_artifact.bubble_local_nested_speech
        or translation_artifact.target_text != parent.target_text
        or translation_artifact.source_text != parent.source_text
        or translation_artifact.source_authority != parent.source_authority
        or translation_artifact.hierarchy_revision_id
        != snapshot.hierarchy.revision_id
        or translation_artifact.hierarchy_fingerprint
        != snapshot.hierarchy.fingerprint
        or translation_artifact.revision_id
        != target.get("translation_revision_id")
        or translation_artifact.selection_edit_id
        != target.get("translation_selection_edit_id")
        or translation_artifact.selection_edit_id
        not in parent.applied_edit_ids
        or canonical_sha256(translation)
        != target.get("translation_artifact_sha256")
        or translation_artifact.source_revision_id
        != source_artifact.revision_id
        or translation_artifact.source_selection_edit_id
        != source_artifact.selection_edit_id
        or translation_artifact.source_fingerprint != source_fingerprint
    ):
        return False
    original_reference = _page_original_asset_reference(page)
    if not original_reference:
        return False
    expected_asset_id = "original-page-v1-" + canonical_sha256(
        {
            "page_id": snapshot.page_id,
            "asset_reference": original_reference,
        }
    )
    if (
        target.get("original_page_asset_reference") != original_reference
        or target.get("original_page_asset_id") != expected_asset_id
    ):
        return False
    automatic_lineage = cleaned_base_automatic_lineage(cleaned)
    lineage_source_sha256 = (
        str(automatic_lineage.get("source_sha256") or "").lower()
        if automatic_lineage is not None
        else ""
    )
    if lineage_source_sha256 and (
        not _is_sha256(lineage_source_sha256)
        or lineage_source_sha256
        != target.get("original_page_content_sha256")
    ):
        return False
    return True


def cleaned_base_automatic_lineage(cleaned: Any) -> dict[str, Any] | None:
    """Return immutable automatic erasure facts carried by one clean base.

    A user cleanup revision may change page pixels, but it never acquires
    automatic cleanup authority.  Every such revision therefore copies this
    compact lineage record unchanged from its input base.
    """

    if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
        return None
    provenance = str(cleaned.get("provenance") or "").strip()
    if provenance == "user_manual_cleanup":
        lineage = cleaned.get("automatic_cleanup_lineage")
        if not isinstance(lineage, Mapping):
            return None
        value = thaw_json(freeze_json(lineage, field_name="automatic_cleanup_lineage"))
        if value.get("lineage_schema_version") != _MANUAL_CLEANUP_LINEAGE_VERSION:
            return None
        if not _is_sha256(cleaned.get("automatic_cleanup_lineage_sha256")):
            return None
        if canonical_sha256(value) != str(
            cleaned.get("automatic_cleanup_lineage_sha256") or ""
        ).lower():
            return None
        return value

    nested = cleaned.get("cleaned_page_base")
    if not isinstance(nested, Mapping):
        return None
    revision_id = str(cleaned.get("revision_id") or "").strip()
    content_sha256 = str(cleaned.get("content_sha256") or "").lower()
    if not revision_id or not _is_sha256(content_sha256):
        return None
    try:
        committed_count = int(nested.get("cleanup_committed_count") or 0)
        blocked_count = int(nested.get("cleanup_blocked_count") or 0)
    except (TypeError, ValueError):
        return None
    return {
        "lineage_schema_version": _MANUAL_CLEANUP_LINEAGE_VERSION,
        "origin_base_revision_id": revision_id,
        "origin_base_content_sha256": content_sha256,
        "origin_provenance": provenance,
        "automatic_state": str(
            cleaned.get("state") or nested.get("state") or ""
        ),
        "automatic_valid": bool(nested.get("valid")),
        "cleanup_required": bool(nested.get("cleanup_required")),
        "cleanup_committed_count": committed_count,
        "cleanup_blocked_count": blocked_count,
        "cleanup_committed_region_ids": _string_list(
            nested.get("cleanup_committed_region_ids")
        ),
        "cleanup_blocked_region_ids": _string_list(
            nested.get("cleanup_blocked_region_ids")
        ),
        "cleanup_commit_record_ids": _string_list(
            nested.get("cleanup_commit_record_ids")
        ),
        "cleanup_proof_ids": _string_list(nested.get("cleanup_proof_ids")),
        "parent_execution_bundle_ids": _string_list(
            nested.get("parent_execution_bundle_ids")
        ),
        "parent_execution_signature": str(
            nested.get("parent_execution_signature") or ""
        ),
        "source_sha256": str(nested.get("source_sha256") or "").lower(),
        "errors": _string_list(nested.get("errors")),
    }


def _manual_cleaned_base_lineage_is_valid(
    cleaned: Mapping[str, Any],
    artifact_index: Mapping[str, tuple[str, Mapping[str, Any]]],
    *,
    page_id: str,
    visited: frozenset[str] = frozenset(),
) -> bool:
    if str(cleaned.get("provenance") or "") != "user_manual_cleanup":
        return True
    revision_id = str(cleaned.get("revision_id") or "").strip()
    if not revision_id or revision_id in visited:
        return False
    receipt = cleaned.get("manual_cleanup_receipt")
    if not isinstance(receipt, Mapping):
        return False
    if (
        str(cleaned.get("manual_cleaned_base_revision_version") or "")
        != "manual_cleaned_base_revision_v1"
        or not str(receipt.get("operation_id") or "").strip()
        or str(receipt.get("manual_cleanup_receipt_version") or "")
        != _MANUAL_CLEANUP_RECEIPT_VERSION
        or str(receipt.get("status") or "") != "committed"
        or str(receipt.get("provenance") or "") != "user"
        or str(receipt.get("page_id") or "") != page_id
        or str(receipt.get("revision_id") or "") != revision_id
        or str(receipt.get("result_sha256") or "").lower()
        != str(cleaned.get("content_sha256") or "").lower()
        or str(receipt.get("result_asset") or "")
        != str(cleaned.get("asset") or "")
        or str(receipt.get("input_base_revision_id") or "")
        != str(cleaned.get("input_base_revision_id") or "")
        or str(receipt.get("input_base_sha256") or "").lower()
        != str(cleaned.get("input_base_sha256") or "").lower()
        or not bool(receipt.get("page_bounds_validated"))
    ):
        return False
    artifact_coverage_value = cleaned.get(
        "user_parent_cleanup_coverage_target"
    )
    receipt_coverage_value = receipt.get(
        "user_parent_cleanup_coverage_target"
    )
    if artifact_coverage_value is not None or receipt_coverage_value is not None:
        artifact_coverage = _user_parent_cleanup_coverage_target(
            artifact_coverage_value
        )
        receipt_coverage = _user_parent_cleanup_coverage_target(
            receipt_coverage_value
        )
        if (
            artifact_coverage is None
            or receipt_coverage is None
            or artifact_coverage != receipt_coverage
            or artifact_coverage["page_id"] != page_id
            or artifact_coverage["input_cleaned_base_revision_id"]
            != str(cleaned.get("input_base_revision_id") or "")
            or artifact_coverage["input_cleaned_base_content_sha256"]
            != str(cleaned.get("input_base_sha256") or "").lower()
        ):
            return False
    for field in (
        "input_base_sha256",
        "erase_mask_sha256",
        "protect_mask_sha256",
        "effective_mask_sha256",
        "result_sha256",
    ):
        if not _is_sha256(receipt.get(field)):
            return False
    for receipt_field, artifact_field in (
        ("erase_mask_sha256", "erase_mask_sha256"),
        ("protect_mask_sha256", "protect_mask_sha256"),
        ("effective_mask_sha256", "effective_mask_sha256"),
    ):
        if str(receipt.get(receipt_field) or "").lower() != str(
            cleaned.get(artifact_field) or ""
        ).lower():
            return False
    canvas_size = receipt.get("canvas_size")
    if (
        not isinstance(canvas_size, (list, tuple))
        or len(canvas_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in canvas_size)
        or list(canvas_size) != list(cleaned.get("canvas_size") or ())
    ):
        return False
    input_revision_id = str(receipt.get("input_base_revision_id") or "").strip()
    indexed = artifact_index.get(input_revision_id)
    if (
        not input_revision_id
        or input_revision_id == revision_id
        or indexed is None
        or indexed[0] != "cleaned_page_bases"
        or str(indexed[1].get("page_id") or "") != page_id
        or not _valid_cleaned_record(indexed[1])
        or str(indexed[1].get("content_sha256") or "").lower()
        != str(receipt.get("input_base_sha256") or "").lower()
    ):
        return False
    lineage = cleaned_base_automatic_lineage(cleaned)
    input_lineage = cleaned_base_automatic_lineage(indexed[1])
    if lineage is None or input_lineage is None or lineage != input_lineage:
        return False
    if str(indexed[1].get("provenance") or "") == "user_manual_cleanup":
        return _manual_cleaned_base_lineage_is_valid(
            indexed[1],
            artifact_index,
            page_id=page_id,
            visited=visited | {revision_id},
        )
    return True


def cleaned_base_erasure_membership(
    cleaned: Any,
    automatic_parent: Mapping[str, Any],
) -> bool | None:
    """Return whether the selected base proves erasure for one parent.

    ``None`` means the artifact does not carry enough producer lineage to make
    a structural-membership decision.  This is intentionally fail-closed for
    exclude/restore edits; GUI-4 manual-cleanup receipts will later provide the
    corresponding proof for user-authored bases.
    """

    if not isinstance(cleaned, Mapping) or not bool(cleaned.get("valid")):
        return None
    nested = cleaned.get("cleaned_page_base")
    if not isinstance(nested, Mapping):
        return None
    state = str(cleaned.get("state") or nested.get("state") or "").strip()
    content_sha256 = str(cleaned.get("content_sha256") or "").lower()
    if state == "source_noop":
        source_sha256 = str(nested.get("source_sha256") or "").lower()
        cleaned_sha256 = str(
            nested.get("cleaned_page_base_sha256") or ""
        ).lower()
        try:
            committed_count = int(nested.get("cleanup_committed_count") or 0)
            blocked_count = int(nested.get("cleanup_blocked_count") or 0)
        except (TypeError, ValueError):
            return None
        source_noop_proven = bool(
            nested.get("valid")
            and not bool(nested.get("cleanup_required"))
            and committed_count == 0
            and blocked_count == 0
            and not list(nested.get("cleanup_committed_region_ids") or ())
            and not list(nested.get("errors") or ())
            and _is_sha256(content_sha256)
            and content_sha256 == source_sha256 == cleaned_sha256
        )
        return False if source_noop_proven else None

    provenance = str(cleaned.get("provenance") or "")
    if provenance == "user_manual_cleanup":
        lineage = cleaned_base_automatic_lineage(cleaned)
        if lineage is None:
            return None
        parent_id = str(automatic_parent.get("parent_id") or "").strip()
        bundle_id = str(automatic_parent.get("bundle_id") or "").strip()
        identities = {value for value in (parent_id, bundle_id) if value}
        producer_parent_ids = {
            str(value)
            for value in lineage.get("parent_execution_bundle_ids") or ()
            if str(value)
        }
        if not identities or not identities.intersection(producer_parent_ids):
            return None
        committed_ids = {
            str(value)
            for value in lineage.get("cleanup_committed_region_ids") or ()
            if str(value)
        }
        return bool(identities.intersection(committed_ids))
    if provenance != "automatic":
        return None
    parent_id = str(automatic_parent.get("parent_id") or "").strip()
    bundle_id = str(automatic_parent.get("bundle_id") or "").strip()
    identities = {value for value in (parent_id, bundle_id) if value}
    producer_parent_ids = {
        str(value)
        for value in nested.get("parent_execution_bundle_ids") or ()
        if str(value)
    }
    if not identities or not identities.intersection(producer_parent_ids):
        return None
    committed_ids = {
        str(value)
        for value in nested.get("cleanup_committed_region_ids") or ()
        if str(value)
    }
    return bool(identities.intersection(committed_ids))


def cleaned_base_parent_signature(
    automatic_parents: Iterable[Mapping[str, Any]],
) -> str:
    """Recompute the public CleanedPageBase producer-parent signature."""

    payload: list[dict[str, Any]] = []
    for record in automatic_parents:
        payload.append(
            {
                "bundle_id": str(record.get("bundle_id") or ""),
                "parent_id": str(record.get("parent_id") or ""),
                "root_id": str(record.get("root_id") or ""),
                "state": str(record.get("state") or ""),
                "role": str(record.get("role") or ""),
                "cleanup_required": bool(record.get("cleanup_required")),
                "source_text": str(record.get("source_text") or ""),
                "source_region_ids": _string_list(record.get("source_region_ids")),
                "source_contract_region_id": str(
                    record.get("source_contract_region_id") or ""
                ),
                "source_contract_bbox": _integer_list(
                    record.get("source_contract_bbox")
                ),
                "parent_bbox": _integer_list(record.get("parent_bbox")),
                "cleanup_target_bbox": _integer_list(
                    record.get("cleanup_target_bbox")
                ),
                "render_allowed_area": _integer_list(
                    record.get("render_allowed_area")
                ),
                "source_glyph_mask_ids": _string_list(
                    record.get("source_glyph_mask_ids")
                ),
                "cleanup_job_ids": _string_list(record.get("cleanup_job_ids")),
                "cleanup_mask_ids": _string_list(record.get("cleanup_mask_ids")),
                "render_decision_id": str(record.get("render_decision_id") or ""),
                "semantic_class": str(record.get("semantic_class") or ""),
                "route_intent": str(record.get("route_intent") or ""),
                "reading_order_index": int(record.get("reading_order_index") or 0),
            }
        )
    ordered = sorted(
        payload,
        key=lambda item: (
            item.get("reading_order_index", 0),
            item.get("bundle_id", ""),
        ),
    )
    return canonical_sha256(ordered)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)] if str(value) else []


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        values = list(value)
    except TypeError:
        return []
    result: list[int] = []
    for item in values:
        try:
            result.append(int(round(float(item))))
        except (TypeError, ValueError):
            continue
    return result


def _automatic_cleaned_base(
    project: Mapping[str, Any],
    page: Mapping[str, Any],
) -> tuple[str, Any, str]:
    catalogs = project.get("artifact_revisions")
    if isinstance(catalogs, Mapping):
        values = catalogs.get("cleaned_page_bases")
        if isinstance(values, list):
            current = [
                value
                for value in values
                if isinstance(value, Mapping)
                and str(value.get("page_id") or "") == _page_id(page)
                and bool(value.get("current"))
                and _valid_cleaned_record(value)
            ]
            if len(current) > 1:
                raise ValueError(
                    f"multiple current CleanedPageBase revisions for page {_page_id(page)}"
                )
            if current:
                selected = current[0]
                return str(selected.get("revision_id") or ""), freeze_json(
                    selected,
                    field_name="cleaned_page_base",
                ), str(selected.get("provenance") or "automatic")
    cleaned = page.get("cleaned_page_base")
    if isinstance(cleaned, Mapping) and bool(cleaned.get("valid")):
        descriptor = {
            "page_id": _page_id(page),
            "valid": True,
            "current": True,
            "state": str(cleaned.get("state") or ""),
            "asset": str(
                cleaned.get("image_path") or cleaned.get("cache_path") or ""
            ),
            "content_sha256": str(
                cleaned.get("cleaned_page_base_sha256")
                or cleaned.get("source_sha256")
                or ""
            ),
            "provenance": "automatic",
            "cleaned_page_base": dict(cleaned),
        }
        if not _valid_cleaned_record(descriptor):
            return "", freeze_json(None, field_name="cleaned_page_base"), ""
        return automatic_revision_id(
            cleaned,
            prefix="automatic-cleaned-base",
        ), freeze_json(descriptor, field_name="cleaned_page_base"), "automatic"
    return "", freeze_json(None, field_name="cleaned_page_base"), ""


def field_base_fingerprint(
    *,
    project: Mapping[str, Any],
    page: Mapping[str, Any],
    target: EditTarget,
    domain: EditDomain,
    operation: str,
    payload: Mapping[str, Any],
) -> str | None:
    """Fingerprint only the automated field owned by one prospective edit."""

    parent: Mapping[str, Any] | None = None
    if target.kind is EditTargetKind.PARENT:
        matches = [
            candidate
            for candidate in _automatic_parents(
                page,
                expected_page_id=_page_id(page),
            )
            if _parent_id(candidate) == target.parent_id
        ]
        if len(matches) != 1:
            return None
        parent = matches[0]
    if domain is EditDomain.SOURCE_TEXT and parent is not None:
        value = {"parent_id": target.parent_id, "source_text": _source_text(parent)}
    elif domain is EditDomain.TARGET_TEXT and parent is not None:
        value = {"parent_id": target.parent_id, "target_text": _target_text(parent)}
    elif domain is EditDomain.STRUCTURAL and parent is not None:
        if operation in {"exclude", "restore"}:
            value = {"parent_id": target.parent_id, "automatic_exists": True}
        elif operation == "set_geometry":
            value = {"parent_id": target.parent_id, "geometry": thaw_json(_automatic_geometry(parent))}
        elif operation == "set_reading_order":
            return None
        elif operation == "set_role":
            value = {
                "parent_id": target.parent_id,
                "role": parent.get("role", parent.get("region_type")),
            }
        else:
            value = {
                "parent_ids": sorted(
                    _parent_id(item)
                    for item in _automatic_parents(
                        page,
                        expected_page_id=_page_id(page),
                    )
                ),
                "operation": operation,
            }
    elif domain is EditDomain.RENDER_STYLE and parent is not None:
        fields = payload.get("fields")
        field = next(iter(fields), None) if isinstance(fields, Mapping) else (
            fields[0] if isinstance(fields, tuple) and fields else None
        )
        if field == "fill_color":
            automatic_value = automatic_render_fill_color(parent)
            if automatic_value is None:
                return None
        elif field == "outline_color":
            automatic_value = automatic_render_outline_color(parent)
            if automatic_value is None:
                return None
        elif field == "outline_width":
            automatic_value = automatic_render_outline_width(parent)
            if automatic_value is None:
                return None
        elif field == "preferred_size":
            automatic_value = automatic_render_preferred_size(parent)
            if automatic_value is None:
                return None
        elif field == "shadow_enabled":
            automatic_value = automatic_render_shadow_enabled(parent)
            if automatic_value is None:
                return None
            resolution = resolve_parent_layer_effects(parent.get("render_style"))
            automatic_value = {
                "enabled": True,
                "color": resolution.shadow.color.upper(),
                "offset_px": [float(item) for item in resolution.shadow.offset_px],
                "blur_radius_px": float(resolution.shadow.blur_radius_px),
            }
        elif field == "shadow_blur":
            automatic_value = automatic_render_shadow_blur(parent)
            if automatic_value is None:
                return None
            resolution = resolve_parent_layer_effects(parent.get("render_style"))
            automatic_value = {
                "selected_shadow_blur": automatic_value,
                "parent_layer_effects": resolution.to_audit_dict(),
            }
        elif field == "shadow_color":
            automatic_value = automatic_render_shadow_color(parent)
            if automatic_value is None:
                return None
            resolution = resolve_parent_layer_effects(parent.get("render_style"))
            automatic_value = {
                "selected_shadow_color": automatic_value,
                "parent_layer_effects": resolution.to_audit_dict(),
            }
        elif field == "shadow_offset":
            automatic_value = automatic_render_shadow_offset(parent)
            if automatic_value is None:
                return None
            resolution = resolve_parent_layer_effects(parent.get("render_style"))
            automatic_value = {
                "selected_shadow_offset": list(automatic_value),
                "parent_layer_effects": resolution.to_audit_dict(),
            }
        elif field == "font_role":
            automatic_value = automatic_render_font_role(parent)
            if automatic_value is None:
                return None
        elif field == "font_weight_tier":
            automatic_value = automatic_render_font_weight_tier(parent)
            render_style = parent.get("render_style")
            if automatic_value is None or not isinstance(render_style, Mapping):
                return None
            family_role = str(render_style.get("font_family_role") or "")
            primary_role = automatic_render_font_role(parent)
            role_status = str(render_style.get("primary_font_role_status") or "")
            if (
                family_role not in {"sans", "serif"}
                or primary_role is None
                or role_status not in {
                    "registered_role",
                    "degraded_registered_role",
                    "fallback_registered_role",
                }
            ):
                return None
            automatic_value = {
                "selected_font_weight_tier": automatic_value,
                "font_family_role": family_role,
                "primary_font_role": primary_role,
                "primary_font_role_status": role_status,
            }
        else:
            automatic = _first_mapping(
                parent,
                ("render_style", "resolved_style", "style"),
            )
            automatic_value = automatic.get(field)
        value = {
            "parent_id": target.parent_id,
            "field": field,
            "value": automatic_value,
        }
    elif domain is EditDomain.RENDER_LAYOUT and parent is not None:
        fields = payload.get("fields")
        field = next(iter(fields), None) if isinstance(fields, Mapping) else (
            fields[0] if isinstance(fields, (list, tuple)) and fields else None
        )
        if field == "writing_mode":
            automatic_value = automatic_render_writing_mode(parent)
            if automatic_value is None:
                return None
        elif field == "line_height":
            automatic_value = automatic_render_line_height(parent)
            if automatic_value is None:
                return None
        elif field == "rotation":
            automatic_value = automatic_render_rotation(parent)
            if automatic_value is None:
                return None
        elif field == "render_box":
            automatic_box = automatic_render_box(parent)
            hard_bounds = automatic_render_hard_bounds(parent)
            if automatic_box is None or hard_bounds is None:
                return None
            automatic_value = {
                "selected_render_box": list(automatic_box),
                "hard_bounds": list(hard_bounds),
            }
        else:
            automatic = _first_mapping(
                parent,
                (
                    "render_layout_summary",
                    "render_layer_plan",
                    "render_plan",
                    "render_layout",
                    "layout",
                ),
            )
            automatic_value = automatic.get(field)
        value = {
            "parent_id": target.parent_id,
            "field": field,
            "value": automatic_value,
        }
    elif domain is EditDomain.REVIEW_METADATA and parent is not None:
        fields = payload.get("fields")
        field = next(iter(fields), None) if isinstance(fields, Mapping) else None
        automatic = _first_mapping(parent, ("review_metadata",))
        value = {"parent_id": target.parent_id, "field": field, "value": automatic.get(field)}
    elif domain is EditDomain.CLEANUP:
        revision_id, cleaned, _ = _automatic_cleaned_base(project, page)
        value = {"revision_id": revision_id, "cleaned_page_base": thaw_json(cleaned)}
    elif domain is EditDomain.GLOSSARY:
        value = project.get("glossary") or {}
    elif domain is EditDomain.STRUCTURAL:
        if operation == "set_reading_order" and target.kind is EditTargetKind.PAGE:
            value = {
                "page_id": _page_id(page),
                "automatic_ordered_parent_ids": list(
                    automatic_ordered_parent_ids_for_page(page)
                ),
            }
        else:
            value = {
                "parent_ids": sorted(
                    _parent_id(item)
                    for item in _automatic_parents(
                        page,
                        expected_page_id=_page_id(page),
                    )
                ),
                "operation": operation,
            }
    else:
        return None
    return canonical_sha256(value)


def expected_base_fingerprint(
    project: Mapping[str, Any],
    page: Mapping[str, Any],
    edit: ProjectEdit,
) -> str | None:
    return field_base_fingerprint(
        project=project,
        page=page,
        target=edit.target,
        domain=edit.domain,
        operation=edit.operation,
        payload=edit.payload,
    )


def _resolved_edits(
    edits: tuple[ProjectEdit, ...],
) -> tuple[tuple[ProjectEdit, ...], tuple[ProjectionIssue, ...]]:
    groups: dict[tuple[str, str, str, str], list[ProjectEdit]] = {}
    for edit in edits:
        for field in _edit_fields(edit):
            groups.setdefault(
                (
                    edit.target.kind.value,
                    _target_id(edit),
                    edit.domain.value,
                    field,
                ),
                [],
            ).append(edit)
    chosen_ids: set[str] = set()
    issues: list[ProjectionIssue] = []
    for candidates in groups.values():
        candidate_ids = {candidate.edit_id for candidate in candidates}
        superseded = {
            candidate.supersedes_edit_id
            for candidate in candidates
            if candidate.supersedes_edit_id in candidate_ids
        }
        heads = [
            candidate for candidate in candidates if candidate.edit_id not in superseded
        ]
        if len(heads) != 1:
            first = candidates[0]
            issues.append(
                _issue(
                    ProjectionIssueKind.CONFLICT,
                    first,
                    "competing active edits require explicit supersession",
                    edit_ids=tuple(sorted(candidate.edit_id for candidate in candidates)),
                )
            )
            continue
        chosen_ids.add(heads[0].edit_id)
    return (
        tuple(edit for edit in edits if edit.edit_id in chosen_ids),
        tuple(issues),
    )


def _append_applied(parent: EffectiveParentSnapshot, edit_id: str) -> EffectiveParentSnapshot:
    if edit_id in parent.applied_edit_ids:
        return parent
    return replace(parent, applied_edit_ids=(*parent.applied_edit_ids, edit_id))


def _apply_nontext_parent_edit(
    parent: EffectiveParentSnapshot,
    edit: ProjectEdit,
) -> EffectiveParentSnapshot:
    payload = thaw_json(edit.payload)
    if edit.domain is EditDomain.STRUCTURAL:
        if edit.operation in {"exclude", "restore"}:
            parent = replace(parent, excluded=edit.operation == "exclude")
        elif edit.operation == "set_geometry":
            parent = replace(parent, geometry=freeze_json(payload["bbox"], field_name="geometry"))
        elif edit.operation == "set_role":
            parent = replace(parent, role=str(payload["role"]))
    elif edit.domain is EditDomain.RENDER_STYLE:
        fields = dict(parent.render_style_overrides)
        field = _edit_fields(edit)[0]
        if edit.operation == "set_fields":
            value = edit.payload["fields"][field]
            if field == "fill_color":
                value = canonical_render_fill_color(value)
            elif field == "outline_color":
                value = canonical_render_outline_color(value)
            elif field == "outline_width":
                value = canonical_render_outline_width(value)
            elif field == "preferred_size":
                value = canonical_render_preferred_size(value)
            elif field == "font_weight_tier":
                value = canonical_render_font_weight_tier(value)
            elif field == "shadow_enabled" and value is not False:
                raise ValueError(
                    "render_style.shadow_enabled supports only the boolean false"
                )
            fields[field] = value
        else:
            fields.pop(field, None)
        parent = replace(parent, render_style_overrides=_mapping_tuple(fields))
    elif edit.domain is EditDomain.RENDER_LAYOUT:
        fields = dict(parent.render_layout_overrides)
        field = _edit_fields(edit)[0]
        if edit.operation == "set_fields":
            value = edit.payload["fields"][field]
            if field == "render_box":
                value = canonical_render_box(value)
            fields[field] = value
        else:
            fields.pop(field, None)
        parent = replace(parent, render_layout_overrides=_mapping_tuple(fields))
    elif edit.domain is EditDomain.REVIEW_METADATA:
        fields = dict(parent.review_metadata)
        field = _edit_fields(edit)[0]
        fields[field] = edit.payload["fields"][field]
        parent = replace(parent, review_metadata=_mapping_tuple(fields))
    if edit.domain in {EditDomain.RENDER_STYLE, EditDomain.RENDER_LAYOUT}:
        parent = replace(
            parent,
            render_override_edit_ids=tuple(
                dict.fromkeys((*parent.render_override_edit_ids, edit.edit_id))
            ),
        )
    return _append_applied(parent, edit.edit_id)


def _sort_issues(issues: Iterable[ProjectionIssue]) -> tuple[ProjectionIssue, ...]:
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.kind.value,
                issue.target_kind,
                issue.target_id,
                issue.domain,
                issue.edit_ids,
                issue.reason,
            ),
        )
    )


def project_effective_page(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
    *,
    page_id: str,
    _memo: dict[tuple[str, str], EffectivePageSnapshot] | None = None,
) -> EffectivePageSnapshot:
    """Project one immutable page without invoking any pipeline owner."""

    if _memo is None:
        _memo = {}
    memo_key = (str(page_id), ledger.fingerprint())
    cached = _memo.get(memo_key)
    if cached is not None:
        return cached

    project_id = project_id_for(project)
    if ledger.project_id and ledger.project_id != project_id:
        raise ValueError("edit ledger project identity does not match project")
    page = _find_page(project, page_id)
    page_fingerprint = automatic_page_fingerprint(page)
    base_revision_id = automatic_revision_id(page, prefix="automatic-page")
    automatic_parents = _automatic_parents(page, expected_page_id=page_id)
    page_stage_outcomes = _stage_outcomes_for_page(project, page, page_id)
    cleaned_record = page.get("cleaned_page_base")
    cleanup_current = bool(
        isinstance(cleaned_record, Mapping) and cleaned_record.get("valid")
    )
    processing_state = str(page.get("processing_state") or "")
    page_output_current = bool(
        str(page.get("output_path") or "")
        and processing_state != "failed"
    )
    automatic_parent_order = automatic_ordered_parent_ids_for_page(page)
    automatic_parent_by_id = {
        _parent_id(parent): parent for parent in automatic_parents
    }
    parent_map = {
        parent_id: _snapshot_parent(
            automatic_parent_by_id[parent_id],
            page_id=page_id,
            stage_outcomes=page_stage_outcomes,
            cleanup_current=cleanup_current,
            page_output_current=page_output_current,
        )
        for parent_id in automatic_parent_order
    }
    artifact_index = _artifact_index(project)
    issues: list[ProjectionIssue] = []
    eligible: list[ProjectEdit] = []
    page_candidates = list(ledger.active_edits(page_id=page_id))
    page_candidate_ids = {edit.edit_id for edit in page_candidates}
    page_candidates.extend(
        edit
        for edit in ledger.active_edits()
        if edit.edit_id not in page_candidate_ids
        and edit.domain is EditDomain.GLOSSARY
        and edit.target.kind is EditTargetKind.PROJECT
    )
    all_user_add_records = tuple(
        edit
        for edit in ledger.edits
        if not edit.is_control
        and edit.domain is EditDomain.STRUCTURAL
        and edit.operation == "add_user_parent"
        and edit.target.kind is EditTargetKind.PARENT
    )
    all_user_split_records = tuple(
        edit
        for edit in ledger.edits
        if not edit.is_control
        and edit.domain is EditDomain.STRUCTURAL
        and edit.operation == "split_user_parent"
        and edit.target.kind is EditTargetKind.PARENT
    )
    all_user_merge_records = tuple(
        edit
        for edit in ledger.edits
        if not edit.is_control
        and edit.domain is EditDomain.STRUCTURAL
        and edit.operation == "merge_pipeline_parents"
        and edit.target.kind is EditTargetKind.PARENT
    )
    automatic_parent_ids_project: set[str] = set()
    automatic_root_ids_project: set[str] = set()
    for project_page in project.get("pages") or ():
        if not isinstance(project_page, Mapping):
            continue
        project_page_id = _page_id(project_page)
        for automatic_parent in _automatic_parents(
            project_page,
            expected_page_id=project_page_id,
        ):
            automatic_parent_ids_project.add(_parent_id(automatic_parent))
            automatic_root_id = str(
                automatic_parent.get("root_id")
                or automatic_parent.get("text_block_root_id")
                or ""
            ).strip()
            if automatic_root_id:
                automatic_root_ids_project.add(automatic_root_id)
    user_parent_counts: dict[str, int] = {}
    user_root_counts: dict[str, int] = {}
    for add_edit in all_user_add_records:
        user_parent_counts[add_edit.target.parent_id] = (
            user_parent_counts.get(add_edit.target.parent_id, 0) + 1
        )
        root_id = str(add_edit.payload.get("root_id") or "")
        user_root_counts[root_id] = user_root_counts.get(root_id, 0) + 1
    for split_edit in all_user_split_records:
        for child_parent_id in split_edit.payload.get("child_parent_ids") or ():
            child_parent_id = str(child_parent_id)
            user_parent_counts[child_parent_id] = (
                user_parent_counts.get(child_parent_id, 0) + 1
            )
        for child_root_id in split_edit.payload.get("child_root_ids") or ():
            child_root_id = str(child_root_id)
            user_root_counts[child_root_id] = (
                user_root_counts.get(child_root_id, 0) + 1
            )
    for merge_edit in all_user_merge_records:
        user_parent_counts[merge_edit.target.parent_id] = (
            user_parent_counts.get(merge_edit.target.parent_id, 0) + 1
        )
        root_id = str(merge_edit.payload.get("merged_root_id") or "")
        user_root_counts[root_id] = user_root_counts.get(root_id, 0) + 1

    handled_add_ids: set[str] = set()
    valid_add_edits: list[ProjectEdit] = []
    for edit in tuple(page_candidates):
        if (
            edit.domain is not EditDomain.STRUCTURAL
            or edit.operation != "add_user_parent"
            or edit.target.kind is not EditTargetKind.PARENT
        ):
            continue
        handled_add_ids.add(edit.edit_id)
        root_id = str(edit.payload.get("root_id") or "")
        if (
            edit.target.parent_id in automatic_parent_ids_project
            or root_id in automatic_root_ids_project
            or user_parent_counts.get(edit.target.parent_id, 0) != 1
            or user_root_counts.get(root_id, 0) != 1
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "user_parent_identity_collision",
                )
            )
            continue
        try:
            predecessor_ledger = _ledger_prefix_before_edit(ledger, edit)
            predecessor = project_effective_page(
                project,
                predecessor_ledger,
                page_id=edit.page_id,
                _memo=_memo,
            )
        except (KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "user_parent_predecessor_topology_is_invalid",
                )
            )
            continue
        if (
            edit.base_revision_id != predecessor.hierarchy.revision_id
            or edit.base_fingerprint != predecessor.hierarchy.fingerprint
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.STALE_EDIT_BASE,
                    edit,
                    "add_user_parent_base_hierarchy_mismatch",
                    expected_fingerprint=predecessor.hierarchy.fingerprint,
                    observed_fingerprint=edit.base_fingerprint,
                )
            )
            continue
        parent_map[edit.target.parent_id] = _snapshot_user_parent(
            edit,
            reading_order=len(parent_map),
        )
        valid_add_edits.append(edit)
    handled_split_ids: set[str] = set()
    valid_split_edits: list[ProjectEdit] = []
    split_consumed_edit_ids: set[str] = set()
    ledger_index_by_id = {
        record.edit_id: index for index, record in enumerate(ledger.edits)
    }
    active_id_set = set(ledger.state().active_edit_ids)
    for edit in tuple(page_candidates):
        if (
            edit.domain is not EditDomain.STRUCTURAL
            or edit.operation != "split_user_parent"
            or edit.target.kind is not EditTargetKind.PARENT
        ):
            continue
        handled_split_ids.add(edit.edit_id)
        payload = thaw_json(edit.payload)
        child_parent_ids = tuple(
            str(value) for value in payload["child_parent_ids"]
        )
        child_root_ids = tuple(str(value) for value in payload["child_root_ids"])
        child_mapping_values = payload.get("child_source_evidence_mappings")
        try:
            child_source_mappings = (
                tuple(
                    ParentSourceEvidenceMappingV1.from_dict(value)
                    for value in child_mapping_values
                )
                if child_mapping_values is not None
                else None
            )
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "split_user_parent_source_evidence_is_invalid",
                )
            )
            continue
        identity_collision = bool(
            any(
                child_parent_id in automatic_parent_ids_project
                or user_parent_counts.get(child_parent_id, 0) != 1
                for child_parent_id in child_parent_ids
            )
            or any(
                child_root_id in automatic_root_ids_project
                or user_root_counts.get(child_root_id, 0) != 1
                for child_root_id in child_root_ids
            )
        )
        if identity_collision:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "split_user_parent_identity_collision",
                )
            )
            continue
        try:
            predecessor_ledger = _ledger_prefix_before_edit(ledger, edit)
            predecessor = project_effective_page(
                project,
                predecessor_ledger,
                page_id=edit.page_id,
                _memo=_memo,
            )
        except (KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "split_user_parent_predecessor_topology_is_invalid",
                )
            )
            continue
        predecessor_parent = next(
            (
                parent
                for parent in predecessor.parents
                if parent.parent_id == edit.target.parent_id
            ),
            None,
        )
        lineage = (
            predecessor_parent.lineage
            if predecessor_parent is not None
            else None
        )
        current_source = parent_map.get(edit.target.parent_id)
        source_bbox = tuple(
            int(value) for value in payload["source_workflow_area_bbox"]
        )
        source_canvas = tuple(int(value) for value in payload["canvas_size"])
        evidence_source_mapping = (
            predecessor_parent.source_evidence_mapping
            if predecessor_parent is not None
            else None
        )
        evidence_backed_split = child_source_mappings is not None
        source_authored_edit = (
            predecessor_ledger.get(lineage.authored_edit_id)
            if lineage is not None
            else None
        )
        mapped_sources_are_exact = False
        expected_child_source_mappings = None
        if evidence_backed_split and evidence_source_mapping is not None:
            try:
                child_bboxes = tuple(
                    tuple(int(item) for item in bbox)
                    for bbox in payload["child_workflow_area_bboxes"]
                )
                expected_child_source_mappings = evidence_source_mapping.partition(
                    child_bboxes
                )
            except (TypeError, ValueError):
                expected_child_source_mappings = None
            current_mapped_sources = tuple(
                parent_map.get(parent_id)
                for parent_id in evidence_source_mapping.source_parent_ids
            )
            mapped_sources_are_exact = bool(
                expected_child_source_mappings is not None
                and child_source_mappings == expected_child_source_mappings
                and evidence_source_mapping.page_id == edit.page_id
                and evidence_source_mapping.source_parent_ids
                == lineage.source_parent_ids
                and evidence_source_mapping.source_root_ids
                == lineage.source_root_ids
                and evidence_source_mapping.source_automatic_fingerprints
                == lineage.source_automatic_fingerprints
                and source_authored_edit is not None
                and source_authored_edit.domain is EditDomain.STRUCTURAL
                and source_authored_edit.operation == "merge_pipeline_parents"
                and source_authored_edit.target.parent_id == edit.target.parent_id
                and all(source is not None for source in current_mapped_sources)
                and all(
                    source is not None
                    and source.origin is ParentOrigin.AUTOMATIC
                    and source.bundle_id
                    == evidence_source_mapping.source_bundle_ids[index]
                    and source.root_id
                    == evidence_source_mapping.source_root_ids[index]
                    and source.automatic_fingerprint
                    == evidence_source_mapping.source_automatic_fingerprints[index]
                    and tuple(thaw_json(source.geometry))
                    == evidence_source_mapping.source_bboxes[index]
                    and source.source_text
                    == evidence_source_mapping.source_texts[index]
                    for index, source in enumerate(current_mapped_sources)
                )
            )
        legacy_split = child_source_mappings is None
        if (
            edit.base_revision_id != predecessor.hierarchy.revision_id
            or edit.base_fingerprint != predecessor.hierarchy.fingerprint
            or predecessor_parent is None
            or predecessor_parent.origin is not ParentOrigin.USER
            or predecessor_parent.excluded
            or lineage is None
            or not (
                (
                    legacy_split
                    and current_source is not None
                    and lineage.order_policy == "append"
                    and evidence_source_mapping is None
                )
                or (
                    evidence_backed_split
                    and lineage.order_policy == "replace_sources"
                    and mapped_sources_are_exact
                )
            )
            or lineage.root_id != str(payload["source_root_id"])
            or lineage.authored_edit_id
            != str(payload["source_authored_edit_id"])
            or predecessor_parent.role != str(payload["source_role"])
            or tuple(lineage.workflow_area_bbox) != source_bbox
            or tuple(lineage.canvas_size) != source_canvas
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.STALE_EDIT_BASE,
                    edit,
                    "split_user_parent_source_or_hierarchy_mismatch",
                    expected_fingerprint=predecessor.hierarchy.fingerprint,
                    observed_fingerprint=edit.base_fingerprint,
                )
            )
            continue
        predecessor_order = predecessor.hierarchy.ordered_parent_ids
        if edit.target.parent_id not in predecessor_order:
            issues.append(
                _issue(
                    ProjectionIssueKind.ORPHANED,
                    edit,
                    "split_user_parent_source_is_unavailable",
                )
            )
            continue
        replacement_order: list[str] = []
        for parent_id in predecessor_order:
            if parent_id == edit.target.parent_id:
                replacement_order.extend(child_parent_ids)
            else:
                replacement_order.append(parent_id)
        current_order = tuple(
            parent.parent_id
            for parent in sorted(
                parent_map.values(),
                key=lambda parent: (parent.reading_order, parent.parent_id),
            )
        )
        later_parent_ids = tuple(
            parent_id
            for parent_id in current_order
            if parent_id not in predecessor_order
            and parent_id != edit.target.parent_id
            and parent_id not in child_parent_ids
            and (
                evidence_source_mapping is None
                or parent_id not in evidence_source_mapping.source_parent_ids
            )
        )
        effective_order = (*replacement_order, *later_parent_ids)
        replaced_parent_ids = (
            set(evidence_source_mapping.source_parent_ids)
            if evidence_backed_split and evidence_source_mapping is not None
            else {edit.target.parent_id}
        )
        expected_parent_ids = (
            set(parent_map) - replaced_parent_ids
        ) | set(child_parent_ids)
        if (
            len(effective_order) != len(expected_parent_ids)
            or set(effective_order) != expected_parent_ids
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "split_user_parent_effective_topology_is_invalid",
                )
            )
            continue
        for replaced_parent_id in replaced_parent_ids:
            del parent_map[replaced_parent_id]
        for child_index, child_parent_id in enumerate(child_parent_ids):
            parent_map[child_parent_id] = _snapshot_split_user_parent(
                edit,
                child_index=child_index,
                reading_order=effective_order.index(child_parent_id),
            )
        for reading_order, parent_id in enumerate(effective_order):
            parent_map[parent_id] = replace(
                parent_map[parent_id],
                reading_order=reading_order,
            )
        split_index = ledger_index_by_id[edit.edit_id]
        for prior in ledger.edits[:split_index]:
            if prior.is_control or prior.edit_id not in active_id_set:
                continue
            if (
                prior.operation == "add_user_parent"
                and prior.edit_id == lineage.authored_edit_id
            ):
                continue
            if (
                prior.target.kind is EditTargetKind.PARENT
                and prior.target.parent_id == edit.target.parent_id
            ) or (
                prior.domain is EditDomain.STRUCTURAL
                and prior.operation == "set_reading_order"
                and edit.target.parent_id
                in tuple(
                    str(value)
                    for value in prior.payload.get("ordered_parent_ids") or ()
                )
            ):
                split_consumed_edit_ids.add(prior.edit_id)
        valid_split_edits.append(edit)
    handled_merge_ids: set[str] = set()
    valid_merge_edits: list[ProjectEdit] = []
    merge_consumed_edit_ids: set[str] = set()
    for edit in tuple(page_candidates):
        if (
            edit.domain is not EditDomain.STRUCTURAL
            or edit.operation != "merge_pipeline_parents"
            or edit.target.kind is not EditTargetKind.PARENT
        ):
            continue
        handled_merge_ids.add(edit.edit_id)
        if edit.edit_id in split_consumed_edit_ids:
            valid_merge_edits.append(edit)
            continue
        payload = thaw_json(edit.payload)
        source_parent_ids = tuple(
            str(value) for value in payload["source_parent_ids"]
        )
        source_root_ids = tuple(
            str(value) for value in payload["source_root_ids"]
        )
        source_automatic_fingerprints = tuple(
            str(value) for value in payload["source_automatic_fingerprints"]
        )
        source_bboxes = tuple(
            tuple(int(item) for item in bbox)
            for bbox in payload["source_bboxes"]
        )
        source_texts = tuple(str(value) for value in payload["source_texts"])
        source_text_fingerprints = tuple(
            str(value) for value in payload["source_text_fingerprints"]
        )
        merged_root_id = str(payload["merged_root_id"])
        identity_collision = bool(
            edit.target.parent_id in automatic_parent_ids_project
            or merged_root_id in automatic_root_ids_project
            or user_parent_counts.get(edit.target.parent_id, 0) != 1
            or user_root_counts.get(merged_root_id, 0) != 1
        )
        if identity_collision:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "merge_pipeline_parent_identity_collision",
                )
            )
            continue
        try:
            predecessor_ledger = _ledger_prefix_before_edit(ledger, edit)
            predecessor = project_effective_page(
                project,
                predecessor_ledger,
                page_id=edit.page_id,
                _memo=_memo,
            )
        except (KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "merge_pipeline_parent_predecessor_topology_is_invalid",
                )
            )
            continue
        predecessor_by_id = {
            parent.parent_id: parent for parent in predecessor.parents
        }
        predecessor_sources = tuple(
            predecessor_by_id.get(parent_id) for parent_id in source_parent_ids
        )
        predecessor_order = predecessor.hierarchy.ordered_parent_ids
        payload_predecessor_order = tuple(
            str(value) for value in payload["predecessor_ordered_parent_ids"]
        )
        try:
            first_index = predecessor_order.index(source_parent_ids[0])
        except ValueError:
            first_index = -1
        sources_are_exact = bool(
            edit.base_revision_id == predecessor.hierarchy.revision_id
            and edit.base_fingerprint == predecessor.hierarchy.fingerprint
            and predecessor_order == payload_predecessor_order
            and first_index >= 0
            and first_index + 1 < len(predecessor_order)
            and predecessor_order[first_index + 1] == source_parent_ids[1]
            and all(parent is not None for parent in predecessor_sources)
        )
        if sources_are_exact:
            for index, source in enumerate(predecessor_sources):
                assert source is not None
                parent_local_edit_ids = tuple(
                    edit_id
                    for edit_id in source.applied_edit_ids
                    if (
                        (record := predecessor_ledger.get(edit_id)) is not None
                        and record.target.kind is EditTargetKind.PARENT
                        and record.target.parent_id == source.parent_id
                    )
                )
                if (
                    source.origin is not ParentOrigin.AUTOMATIC
                    or source.excluded
                    or source.lineage is not None
                    or not source.bundle_id
                    or source.root_id != source_root_ids[index]
                    or source.automatic_fingerprint
                    != source_automatic_fingerprints[index]
                    or tuple(thaw_json(source.geometry)) != source_bboxes[index]
                    or source.source_text != source_texts[index]
                    or effective_source_fingerprint(
                        source.parent_id,
                        str(source.source_text),
                    )
                    != source_text_fingerprints[index]
                    or source.role != str(payload["source_role"])
                    or parent_local_edit_ids
                    or source.render_override_edit_ids
                ):
                    sources_are_exact = False
                    break
        merged_bbox = tuple(
            int(value) for value in payload["merged_workflow_area_bbox"]
        )
        if (
            not sources_are_exact
            or merged_bbox
            != (
                min(bbox[0] for bbox in source_bboxes),
                min(bbox[1] for bbox in source_bboxes),
                max(bbox[0] + bbox[2] for bbox in source_bboxes)
                - min(bbox[0] for bbox in source_bboxes),
                max(bbox[1] + bbox[3] for bbox in source_bboxes)
                - min(bbox[1] for bbox in source_bboxes),
            )
            or str(payload["merged_source_text"]) != "".join(source_texts)
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.STALE_EDIT_BASE,
                    edit,
                    "merge_pipeline_parent_source_or_hierarchy_mismatch",
                    expected_fingerprint=predecessor.hierarchy.fingerprint,
                    observed_fingerprint=edit.base_fingerprint,
                )
            )
            continue
        current_sources = tuple(parent_map.get(value) for value in source_parent_ids)
        if any(source is None for source in current_sources):
            issues.append(
                _issue(
                    ProjectionIssueKind.ORPHANED,
                    edit,
                    "merge_pipeline_parent_source_is_unavailable",
                )
            )
            continue
        replacement_order = (
            *predecessor_order[:first_index],
            edit.target.parent_id,
            *predecessor_order[first_index + 2 :],
        )
        current_order = tuple(
            parent.parent_id
            for parent in sorted(
                parent_map.values(),
                key=lambda parent: (parent.reading_order, parent.parent_id),
            )
        )
        later_parent_ids = tuple(
            parent_id
            for parent_id in current_order
            if parent_id not in predecessor_order
            and parent_id != edit.target.parent_id
        )
        effective_order = (*replacement_order, *later_parent_ids)
        expected_parent_ids = (
            set(parent_map) - set(source_parent_ids)
        ) | {edit.target.parent_id}
        if (
            len(effective_order) != len(expected_parent_ids)
            or set(effective_order) != expected_parent_ids
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "merge_pipeline_parent_effective_topology_is_invalid",
                )
            )
            continue
        for source_parent_id in source_parent_ids:
            del parent_map[source_parent_id]
        source_evidence_mapping = ParentSourceEvidenceMappingV1(
            page_id=edit.page_id,
            source_parent_ids=source_parent_ids,
            source_root_ids=source_root_ids,
            source_bundle_ids=tuple(
                str(source.bundle_id)
                for source in predecessor_sources
                if source is not None
            ),
            source_automatic_fingerprints=source_automatic_fingerprints,
            source_bboxes=source_bboxes,
            source_texts=source_texts,
            source_text_fingerprints=source_text_fingerprints,
            source_target_texts=tuple(
                str(source.target_text or "")
                for source in predecessor_sources
                if source is not None
            ),
            source_target_text_fingerprints=tuple(
                canonical_sha256(
                    {
                        "parent_id": source.parent_id,
                        "target_text": str(source.target_text or ""),
                    }
                )
                for source in predecessor_sources
                if source is not None
            ),
            source_reading_orders=tuple(
                source.reading_order
                for source in predecessor_sources
                if source is not None
            ),
            source_roles=tuple(str(payload["source_role"]) for _ in source_parent_ids),
            primary_source_parent_id=source_parent_ids[0],
        )
        parent_map[edit.target.parent_id] = _snapshot_merged_pipeline_parent(
            edit,
            reading_order=effective_order.index(edit.target.parent_id),
            source_evidence_mapping=source_evidence_mapping,
        )
        for reading_order, parent_id in enumerate(effective_order):
            parent_map[parent_id] = replace(
                parent_map[parent_id],
                reading_order=reading_order,
            )
        merge_index = ledger_index_by_id[edit.edit_id]
        for prior in ledger.edits[:merge_index]:
            if prior.is_control or prior.edit_id not in active_id_set:
                continue
            if (
                prior.domain is EditDomain.STRUCTURAL
                and prior.operation == "set_reading_order"
                and set(source_parent_ids).issubset(
                    str(value)
                    for value in prior.payload.get("ordered_parent_ids") or ()
                )
            ):
                merge_consumed_edit_ids.add(prior.edit_id)
        valid_merge_edits.append(edit)
    active_opaque_color_superseded_slots = {
        (
            edit.supersedes_edit_id,
            edit.page_id,
            edit.target.parent_id,
            _edit_fields(edit)[0],
        )
        for edit in page_candidates
        if edit.supersedes_edit_id is not None
        and edit.domain is EditDomain.RENDER_STYLE
        and edit.target.kind is EditTargetKind.PARENT
        and _edit_fields(edit) in {("fill_color",), ("outline_color",)}
    }
    for edit in page_candidates:
        if edit.project_id != project_id:
            raise ValueError("active edit project identity does not match project")
        if edit.edit_id in handled_add_ids:
            if edit in valid_add_edits:
                eligible.append(edit)
            continue
        if edit.edit_id in handled_split_ids:
            continue
        if edit.edit_id in handled_merge_ids:
            continue
        if edit.edit_id in split_consumed_edit_ids:
            continue
        if edit.edit_id in merge_consumed_edit_ids:
            continue
        if (
            edit.target.kind is EditTargetKind.PARENT
            and edit.target.parent_id not in parent_map
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.ORPHANED,
                    edit,
                    "automatic parent identity is unavailable",
                )
            )
            continue
        if edit.target.kind is EditTargetKind.ARTIFACT:
            indexed = artifact_index.get(edit.target.artifact_id)
            if indexed is None or str(indexed[1].get("page_id") or "") != page_id:
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "artifact revision identity is unavailable for this page",
                    )
                )
                continue
        if (
            edit.domain is EditDomain.RENDER_STYLE
            and edit.operation == "set_fields"
            and _edit_fields(edit) in {("fill_color",), ("outline_color",)}
        ):
            field = _edit_fields(edit)[0]
            try:
                canonicalizer = (
                    canonical_render_fill_color
                    if field == "fill_color"
                    else canonical_render_outline_color
                )
                canonicalizer(edit.payload["fields"][field])
            except (KeyError, TypeError, ValueError):
                if (
                    edit.edit_id,
                    edit.page_id,
                    edit.target.parent_id,
                    field,
                ) not in active_opaque_color_superseded_slots:
                    issues.append(
                        _issue(
                            ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                            edit,
                            f"render_style.{field}_requires_opaque_rgb",
                        )
                    )
                continue
        if (
            edit.domain is EditDomain.STRUCTURAL
            and edit.operation == "set_reading_order"
            and edit.target.kind is EditTargetKind.PAGE
        ):
            try:
                predecessor = project_effective_page(
                    project,
                    _ledger_prefix_before_edit(ledger, edit),
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "reading_order_predecessor_topology_is_invalid",
                    )
                )
                continue
            legacy_expected = expected_base_fingerprint(project, page, edit)
            payload_parent_ids = frozenset(
                str(parent_id)
                for parent_id in edit.payload["ordered_parent_ids"]
            )
            effective_base_matches = bool(
                edit.base_revision_id == predecessor.hierarchy.revision_id
                and edit.base_fingerprint == predecessor.hierarchy.fingerprint
            )
            legacy_base_matches = bool(
                not any(
                    parent_id.startswith("user-parent-v1-")
                    for parent_id in payload_parent_ids
                )
                and legacy_expected is not None
                and edit.base_fingerprint == legacy_expected
            )
            if not effective_base_matches and not legacy_base_matches:
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "reading_order_base_hierarchy_mismatch",
                        expected_fingerprint=predecessor.hierarchy.fingerprint,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            eligible.append(edit)
            continue
        if (
            edit.domain is EditDomain.SOURCE_TEXT
            and edit.operation == "select_revision"
            and edit.target.kind is EditTargetKind.PARENT
        ):
            artifact = _source_revision_for_edit(artifact_index, edit)
            if artifact is None:
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "selected OCR source revision is unavailable",
                    )
                )
                continue
            try:
                predecessor = project_effective_page(
                    project,
                    _ledger_prefix_before_edit(ledger, edit),
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_predecessor_is_invalid",
                    )
                )
                continue
            predecessor_parent = next(
                (
                    candidate
                    for candidate in predecessor.parents
                    if candidate.parent_id == edit.target.parent_id
                ),
                None,
            )
            lineage = (
                predecessor_parent.lineage
                if predecessor_parent is not None
                else None
            )
            if (
                predecessor_parent is None
                or predecessor_parent.origin is not ParentOrigin.USER
                or lineage is None
                or artifact.root_id != predecessor_parent.root_id
                or artifact.parent_authored_edit_id != lineage.authored_edit_id
                or artifact.sampling_bbox != tuple(lineage.workflow_area_bbox)
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_user_parent_lineage_mismatch",
                    )
                )
                continue
            if (
                edit.base_revision_id != predecessor.hierarchy.revision_id
                or edit.base_fingerprint != predecessor.effective_fingerprint
                or artifact.hierarchy_revision_id
                != predecessor.hierarchy.revision_id
                or artifact.hierarchy_fingerprint
                != predecessor.hierarchy.fingerprint
                or artifact.input_effective_page_fingerprint
                != predecessor.effective_fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "source_revision_effective_input_mismatch",
                        expected_fingerprint=predecessor.effective_fingerprint,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            eligible.append(edit)
            continue
        if (
            edit.domain is EditDomain.SOURCE_TEXT
            and edit.operation in {"replace", "restore_selected_revision"}
            and "revision_base" in edit.payload
            and edit.target.kind is EditTargetKind.PARENT
        ):
            try:
                revision_base = SourceTextRevisionBaseV1.from_dict(
                    edit.payload["revision_base"]
                )
            except (TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_base_is_invalid",
                    )
                )
                continue
            artifact = _source_revision_for_base(
                artifact_index,
                edit,
                revision_base,
            )
            if artifact is None:
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "source revision base artifact is unavailable",
                    )
                )
                continue
            prefix = _ledger_prefix_before_edit(ledger, edit)
            selection_edit = prefix.get(revision_base.selection_edit_id)
            if (
                selection_edit is None
                or selection_edit.domain is not EditDomain.SOURCE_TEXT
                or selection_edit.operation != "select_revision"
                or selection_edit.target != edit.target
                or str(selection_edit.payload.get("revision_id") or "")
                != revision_base.source_revision_id
                or not _source_slot_has_ancestor(
                    ledger,
                    edit,
                    revision_base.selection_edit_id,
                )
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_selection_ancestry_mismatch",
                    )
                )
                continue
            try:
                predecessor = project_effective_page(
                    project,
                    prefix,
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_predecessor_is_invalid",
                    )
                )
                continue
            predecessor_parent = next(
                (
                    candidate
                    for candidate in predecessor.parents
                    if candidate.parent_id == edit.target.parent_id
                ),
                None,
            )
            try:
                predecessor_base = (
                    source_text_revision_base_for_parent(predecessor_parent)
                    if predecessor_parent is not None
                    else None
                )
            except (TypeError, ValueError):
                predecessor_base = None
            lineage = (
                predecessor_parent.lineage
                if predecessor_parent is not None
                else None
            )
            if (
                predecessor_parent is None
                or predecessor_parent.origin is not ParentOrigin.USER
                or lineage is None
                or artifact.root_id != predecessor_parent.root_id
                or artifact.parent_authored_edit_id != lineage.authored_edit_id
                or predecessor_base != revision_base
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "source_revision_user_parent_lineage_mismatch",
                    )
                )
                continue
            if (
                edit.base_revision_id != revision_base.source_revision_id
                or edit.base_fingerprint != revision_base.artifact_sha256
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "source_revision_artifact_base_mismatch",
                        expected_fingerprint=revision_base.artifact_sha256,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            eligible.append(edit)
            continue
        if (
            edit.domain is EditDomain.TARGET_TEXT
            and edit.operation == "select_revision"
            and edit.target.kind is EditTargetKind.PARENT
        ):
            artifact = _translation_revision_for_edit(artifact_index, edit)
            if artifact is None:
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "selected translation revision is unavailable",
                    )
                )
                continue
            try:
                predecessor = project_effective_page(
                    project,
                    _ledger_prefix_before_edit(ledger, edit),
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "translation_revision_predecessor_is_invalid",
                    )
                )
                continue
            predecessor_parent = next(
                (
                    candidate
                    for candidate in predecessor.parents
                    if candidate.parent_id == edit.target.parent_id
                ),
                None,
            )
            lineage = (
                predecessor_parent.lineage
                if predecessor_parent is not None
                else None
            )
            predecessor_source_fingerprint = (
                effective_source_fingerprint(
                    edit.target.parent_id,
                    predecessor_parent.source_text,
                )
                if predecessor_parent is not None
                else ""
            )
            try:
                predecessor_source_base = (
                    source_text_revision_base_for_parent(predecessor_parent)
                    if predecessor_parent is not None
                    else None
                )
            except (TypeError, ValueError):
                predecessor_source_base = None
            if (
                predecessor_parent is None
                or predecessor_parent.origin is not ParentOrigin.USER
                or lineage is None
                or artifact.root_id != predecessor_parent.root_id
                or artifact.parent_authored_edit_id != lineage.authored_edit_id
                or artifact.parent_role != predecessor_parent.role
                or artifact.bubble_local_nested_speech
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "translation_revision_user_parent_lineage_mismatch",
                    )
                )
                continue
            if (
                artifact.source_text != predecessor_parent.source_text
                or artifact.source_authority != predecessor_parent.source_authority
                or artifact.source_fingerprint != predecessor_source_fingerprint
                or artifact.source_revision_id
                != predecessor_parent.source_revision_id
                or predecessor_source_base is None
                or artifact.source_selection_edit_id
                != predecessor_source_base.selection_edit_id
                or str(edit.payload.get("source_fingerprint") or "")
                != predecessor_source_fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_DEPENDENCY,
                        edit,
                        "translation_revision_source_binding_mismatch",
                        expected_fingerprint=predecessor_source_fingerprint,
                        observed_fingerprint=str(
                            edit.payload.get("source_fingerprint") or ""
                        ),
                    )
                )
                continue
            if (
                edit.base_revision_id != predecessor.hierarchy.revision_id
                or edit.base_fingerprint != predecessor.effective_fingerprint
                or artifact.hierarchy_revision_id
                != predecessor.hierarchy.revision_id
                or artifact.hierarchy_fingerprint
                != predecessor.hierarchy.fingerprint
                or artifact.input_effective_page_fingerprint
                != predecessor.effective_fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "translation_revision_effective_input_mismatch",
                        expected_fingerprint=predecessor.effective_fingerprint,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            eligible.append(edit)
            continue
        if (
            edit.domain is EditDomain.TARGET_TEXT
            and edit.operation in {"replace", "restore_mapped_pipeline"}
            and "source_evidence_base" in edit.payload
            and edit.target.kind is EditTargetKind.PARENT
        ):
            try:
                source_evidence_base = ParentSourceEvidenceMappingV1.from_dict(
                    edit.payload["source_evidence_base"]
                )
                prefix = _ledger_prefix_before_edit(ledger, edit)
                predecessor = project_effective_page(
                    project,
                    prefix,
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_mapped_pipeline_base_is_invalid",
                    )
                )
                continue
            predecessor_parent = next(
                (
                    candidate
                    for candidate in predecessor.parents
                    if candidate.parent_id == edit.target.parent_id
                ),
                None,
            )
            lineage = (
                predecessor_parent.lineage
                if predecessor_parent is not None
                else None
            )
            mapped_target_text = source_evidence_base.target_text
            predecessor_source_fingerprint = (
                effective_source_fingerprint(
                    edit.target.parent_id,
                    predecessor_parent.source_text,
                )
                if predecessor_parent is not None
                and predecessor_parent.source_text is not None
                else ""
            )
            if (
                predecessor_parent is None
                or predecessor_parent.origin is not ParentOrigin.USER
                or lineage is None
                or predecessor_parent.source_evidence_mapping
                != source_evidence_base
                or source_evidence_base.page_id != edit.page_id
                or mapped_target_text is None
                or predecessor_parent.source_text
                != source_evidence_base.source_text
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_mapped_pipeline_parent_or_evidence_mismatch",
                    )
                )
                continue
            if (
                edit.base_revision_id != lineage.authored_edit_id
                or edit.base_fingerprint != source_evidence_base.fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "target_mapped_pipeline_topology_base_mismatch",
                        expected_fingerprint=source_evidence_base.fingerprint,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            if edit.operation == "replace" and (
                str(edit.payload.get("source_fingerprint") or "")
                != predecessor_source_fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_DEPENDENCY,
                        edit,
                        "target_mapped_pipeline_source_binding_mismatch",
                        expected_fingerprint=predecessor_source_fingerprint,
                        observed_fingerprint=str(
                            edit.payload.get("source_fingerprint") or ""
                        ),
                    )
                )
                continue
            if edit.supersedes_edit_id is not None:
                predecessor_edit = prefix.get(edit.supersedes_edit_id)
                try:
                    predecessor_mapping = (
                        ParentSourceEvidenceMappingV1.from_dict(
                            predecessor_edit.payload["source_evidence_base"]
                        )
                        if predecessor_edit is not None
                        else None
                    )
                except (KeyError, TypeError, ValueError):
                    predecessor_mapping = None
                if (
                    predecessor_edit is None
                    or predecessor_edit.is_control
                    or predecessor_edit.domain is not EditDomain.TARGET_TEXT
                    or predecessor_edit.target != edit.target
                    or predecessor_mapping != source_evidence_base
                ):
                    issues.append(
                        _issue(
                            ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                            edit,
                            "target_mapped_pipeline_slot_ancestry_mismatch",
                        )
                    )
                    continue
            eligible.append(edit)
            continue
        if (
            edit.domain is EditDomain.TARGET_TEXT
            and edit.operation in {"replace", "restore_selected_revision"}
            and "revision_base" in edit.payload
            and edit.target.kind is EditTargetKind.PARENT
        ):
            try:
                revision_base = TargetTextRevisionBaseV1.from_dict(
                    edit.payload["revision_base"]
                )
            except (TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_revision_base_is_invalid",
                    )
                )
                continue
            artifact = _translation_revision_for_base(
                artifact_index,
                edit,
                revision_base,
            )
            if artifact is None:
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "target revision base artifact is unavailable",
                    )
                )
                continue
            prefix = _ledger_prefix_before_edit(ledger, edit)
            selection_edit = prefix.get(revision_base.selection_edit_id)
            if (
                selection_edit is None
                or selection_edit.domain is not EditDomain.TARGET_TEXT
                or selection_edit.operation != "select_revision"
                or selection_edit.target != edit.target
                or str(selection_edit.payload.get("revision_id") or "")
                != revision_base.translation_revision_id
                or str(selection_edit.payload.get("source_fingerprint") or "")
                != revision_base.source_fingerprint
                or not _target_slot_has_ancestor(
                    ledger,
                    edit,
                    revision_base.selection_edit_id,
                )
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_revision_selection_ancestry_mismatch",
                    )
                )
                continue
            try:
                predecessor = project_effective_page(
                    project,
                    prefix,
                    page_id=edit.page_id,
                    _memo=_memo,
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_revision_predecessor_is_invalid",
                    )
                )
                continue
            predecessor_parent = next(
                (
                    candidate
                    for candidate in predecessor.parents
                    if candidate.parent_id == edit.target.parent_id
                ),
                None,
            )
            try:
                predecessor_base = (
                    target_text_revision_base_for_parent(predecessor_parent)
                    if predecessor_parent is not None
                    else None
                )
            except (TypeError, ValueError):
                predecessor_base = None
            lineage = (
                predecessor_parent.lineage
                if predecessor_parent is not None
                else None
            )
            if (
                predecessor_parent is None
                or predecessor_parent.origin is not ParentOrigin.USER
                or lineage is None
                or artifact.root_id != predecessor_parent.root_id
                or artifact.parent_authored_edit_id != lineage.authored_edit_id
                or artifact.parent_role != predecessor_parent.role
                or predecessor_base != revision_base
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "target_revision_user_parent_lineage_mismatch",
                    )
                )
                continue
            predecessor_source_fingerprint = effective_source_fingerprint(
                edit.target.parent_id,
                predecessor_parent.source_text,
            )
            try:
                predecessor_source_base = source_text_revision_base_for_parent(
                    predecessor_parent
                )
            except (TypeError, ValueError):
                predecessor_source_base = None
            if (
                artifact.source_text != predecessor_parent.source_text
                or artifact.source_authority
                != predecessor_parent.source_authority
                or artifact.source_fingerprint
                != predecessor_source_fingerprint
                or artifact.source_revision_id
                != predecessor_parent.source_revision_id
                or predecessor_source_base is None
                or artifact.source_selection_edit_id
                != predecessor_source_base.selection_edit_id
                or revision_base.source_fingerprint
                != predecessor_source_fingerprint
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_DEPENDENCY,
                        edit,
                        "target_revision_source_binding_mismatch",
                        expected_fingerprint=predecessor_source_fingerprint,
                        observed_fingerprint=revision_base.source_fingerprint,
                    )
                )
                continue
            if (
                edit.base_revision_id
                != revision_base.translation_revision_id
                or edit.base_fingerprint != revision_base.artifact_sha256
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.STALE_EDIT_BASE,
                        edit,
                        "target_revision_hierarchy_or_base_mismatch",
                        expected_fingerprint=revision_base.artifact_sha256,
                        observed_fingerprint=edit.base_fingerprint,
                    )
                )
                continue
            eligible.append(edit)
            continue
        expected = expected_base_fingerprint(project, page, edit)
        if expected is None or edit.base_fingerprint != expected:
            issues.append(
                _issue(
                    ProjectionIssueKind.STALE_EDIT_BASE,
                    edit,
                    "edit base fingerprint does not match its automatic field",
                    expected_fingerprint=str(expected or ""),
                    observed_fingerprint=edit.base_fingerprint,
                )
            )
            continue
        eligible.append(edit)
    resolved, conflicts = _resolved_edits(tuple(eligible))
    issues.extend(conflicts)

    edits_by_parent: dict[str, list[ProjectEdit]] = {}
    page_edits: list[ProjectEdit] = []
    for edit in resolved:
        if edit.target.kind is EditTargetKind.PARENT:
            edits_by_parent.setdefault(edit.target.parent_id, []).append(edit)
        else:
            page_edits.append(edit)

    applied_ids: list[str] = [
        *(edit.edit_id for edit in valid_add_edits),
        *(edit.edit_id for edit in valid_split_edits),
        *(edit.edit_id for edit in valid_merge_edits),
    ]
    for parent_id, parent in tuple(parent_map.items()):
        edits = edits_by_parent.get(parent_id, [])
        for edit in edits:
            if edit.domain is EditDomain.STRUCTURAL:
                parent = _apply_nontext_parent_edit(parent, edit)
                applied_ids.append(edit.edit_id)
        for edit in edits:
            if edit.domain is EditDomain.SOURCE_TEXT:
                if edit.operation == "replace":
                    revision_base_value = edit.payload.get("revision_base")
                    if revision_base_value is not None:
                        revision_base = SourceTextRevisionBaseV1.from_dict(
                            revision_base_value
                        )
                        artifact = _source_revision_for_base(
                            artifact_index,
                            edit,
                            revision_base,
                        )
                        if artifact is None:  # eligible edits were checked above
                            continue
                        parent = replace(
                            parent,
                            source_text=str(edit.payload["text"]),
                            source_authority="user",
                            source_revision_id=artifact.revision_id,
                            source_revision_metadata=_mapping_tuple(
                                artifact.to_record()
                            ),
                            stage_requirements=(
                                _user_parent_requirements_after_source_override(
                                    parent,
                                    page_id=page_id,
                                )
                            ),
                        )
                    else:
                        parent = replace(
                            parent,
                            source_text=str(edit.payload["text"]),
                            source_authority="user",
                        )
                elif edit.operation == "select_revision":
                    artifact = _source_revision_for_edit(artifact_index, edit)
                    if artifact is None:
                        continue
                    parent = replace(
                        parent,
                        source_text=artifact.source_text,
                        source_authority="ocr_revision",
                        source_revision_id=artifact.revision_id,
                        source_revision_metadata=_mapping_tuple(
                            artifact.to_record()
                        ),
                        stage_requirements=_user_parent_stage_requirements(
                            page_id=page_id,
                            parent_id=parent_id,
                            source_current=True,
                        ),
                    )
                elif edit.operation == "restore_selected_revision":
                    revision_base = SourceTextRevisionBaseV1.from_dict(
                        edit.payload["revision_base"]
                    )
                    artifact = _source_revision_for_base(
                        artifact_index,
                        edit,
                        revision_base,
                    )
                    if artifact is None:  # eligible edits were checked above
                        continue
                    parent = replace(
                        parent,
                        source_text=artifact.source_text,
                        source_authority="ocr_revision",
                        source_revision_id=artifact.revision_id,
                        source_revision_metadata=_mapping_tuple(
                            artifact.to_record()
                        ),
                        stage_requirements=(
                            _user_parent_requirements_after_source_override(
                                parent,
                                page_id=page_id,
                            )
                        ),
                    )
                else:
                    automatic = next(
                        value for value in automatic_parents if _parent_id(value) == parent_id
                    )
                    parent = replace(
                        parent,
                        source_text=_source_text(automatic),
                        source_authority="automatic",
                        source_revision_id=None,
                        source_revision_metadata=(),
                    )
                parent = _append_applied(parent, edit.edit_id)
                applied_ids.append(edit.edit_id)
        source_fingerprint = effective_source_fingerprint(parent_id, parent.source_text)
        target_edit_applied = False
        for edit in edits:
            if edit.domain is not EditDomain.TARGET_TEXT:
                continue
            if edit.operation == "replace":
                acknowledged = str(edit.payload["source_fingerprint"])
                if acknowledged != source_fingerprint:
                    issues.append(
                        _issue(
                            ProjectionIssueKind.STALE_DEPENDENCY,
                            edit,
                            "target edit does not acknowledge the effective source",
                            expected_fingerprint=source_fingerprint,
                            observed_fingerprint=acknowledged,
                        )
                    )
                    continue
                revision_base_value = edit.payload.get("revision_base")
                source_evidence_base_value = edit.payload.get(
                    "source_evidence_base"
                )
                if revision_base_value is not None:
                    revision_base = TargetTextRevisionBaseV1.from_dict(
                        revision_base_value
                    )
                    artifact = _translation_revision_for_base(
                        artifact_index,
                        edit,
                        revision_base,
                    )
                    if artifact is None:  # eligible edits were checked above
                        continue
                    target_revision_id = artifact.revision_id
                    target_revision_metadata = _mapping_tuple(
                        artifact.to_record()
                    )
                elif source_evidence_base_value is not None:
                    source_evidence_base = (
                        ParentSourceEvidenceMappingV1.from_dict(
                            source_evidence_base_value
                        )
                    )
                    if parent.source_evidence_mapping != source_evidence_base:
                        continue
                    target_revision_id = None
                    target_revision_metadata = ()
                else:
                    target_revision_id = None
                    target_revision_metadata = ()
                parent = replace(
                    parent,
                    target_text=str(edit.payload["text"]),
                    target_authority="user",
                    target_freshness=TargetFreshness.CURRENT,
                    target_revision_id=target_revision_id,
                    target_revision_metadata=target_revision_metadata,
                    stage_requirements=(
                        _user_parent_stage_requirements(
                            page_id=page_id,
                            parent_id=parent_id,
                            source_current=True,
                            translation_current=True,
                        )
                        if revision_base_value is not None
                        or source_evidence_base_value is not None
                        else parent.stage_requirements
                    ),
                )
                target_edit_applied = True
            elif edit.operation == "select_revision":
                artifact = _translation_revision_for_edit(artifact_index, edit)
                if artifact is None:
                    continue
                if (
                    artifact.source_text != parent.source_text
                    or artifact.source_authority != parent.source_authority
                    or artifact.source_fingerprint != source_fingerprint
                ):
                    issues.append(
                        _issue(
                            ProjectionIssueKind.STALE_DEPENDENCY,
                            edit,
                            "translation revision is historical after a source edit",
                            expected_fingerprint=source_fingerprint,
                            observed_fingerprint=artifact.source_fingerprint,
                        )
                    )
                    continue
                parent = replace(
                    parent,
                    target_text=artifact.target_text,
                    target_authority="translation_revision",
                    target_freshness=TargetFreshness.CURRENT,
                    target_revision_id=artifact.revision_id,
                    target_revision_metadata=_mapping_tuple(
                        artifact.to_record()
                    ),
                    stage_requirements=_user_parent_stage_requirements(
                        page_id=page_id,
                        parent_id=parent_id,
                        source_current=True,
                        translation_current=True,
                    ),
                )
                target_edit_applied = True
            elif edit.operation == "restore_selected_revision":
                revision_base = TargetTextRevisionBaseV1.from_dict(
                    edit.payload["revision_base"]
                )
                artifact = _translation_revision_for_base(
                    artifact_index,
                    edit,
                    revision_base,
                )
                if artifact is None:  # eligible edits were checked above
                    continue
                if (
                    artifact.source_text != parent.source_text
                    or artifact.source_authority != parent.source_authority
                    or artifact.source_fingerprint != source_fingerprint
                ):
                    issues.append(
                        _issue(
                            ProjectionIssueKind.STALE_DEPENDENCY,
                            edit,
                            "restored translation revision is historical after a source edit",
                            expected_fingerprint=source_fingerprint,
                            observed_fingerprint=artifact.source_fingerprint,
                        )
                    )
                    continue
                parent = replace(
                    parent,
                    target_text=artifact.target_text,
                    target_authority="translation_revision",
                    target_freshness=TargetFreshness.CURRENT,
                    target_revision_id=artifact.revision_id,
                    target_revision_metadata=_mapping_tuple(
                        artifact.to_record()
                    ),
                    stage_requirements=_user_parent_stage_requirements(
                        page_id=page_id,
                        parent_id=parent_id,
                        source_current=True,
                        translation_current=True,
                    ),
                )
                target_edit_applied = True
            elif edit.operation == "restore_mapped_pipeline":
                source_evidence_base = ParentSourceEvidenceMappingV1.from_dict(
                    edit.payload["source_evidence_base"]
                )
                mapped_target_text = source_evidence_base.target_text
                if (
                    mapped_target_text is None
                    or parent.source_evidence_mapping != source_evidence_base
                ):
                    continue
                parent = replace(
                    parent,
                    target_text=mapped_target_text,
                    target_authority="mapped_automatic",
                    target_freshness=TargetFreshness.CURRENT,
                    target_revision_id=None,
                    target_revision_metadata=(),
                    stage_requirements=_user_parent_stage_requirements(
                        page_id=page_id,
                        parent_id=parent_id,
                        source_current=True,
                        translation_current=True,
                    ),
                )
                target_edit_applied = True
            else:
                automatic = next(
                    value for value in automatic_parents if _parent_id(value) == parent_id
                )
                parent = replace(
                    parent,
                    target_text=_target_text(automatic),
                    target_authority="automatic",
                    target_freshness=(
                        TargetFreshness.CURRENT
                        if parent.source_authority == "automatic"
                        else TargetFreshness.STALE
                    ),
                    target_revision_id=None,
                    target_revision_metadata=(),
                )
                target_edit_applied = parent.source_authority == "automatic"
            parent = _append_applied(parent, edit.edit_id)
            applied_ids.append(edit.edit_id)
        source_edit_ids = tuple(
            edit.edit_id for edit in edits if edit.domain is EditDomain.SOURCE_TEXT
        )
        if (
            parent.source_authority == "user"
            and not target_edit_applied
            and (
                bool(source_edit_ids)
                or parent.target_authority
                not in {"unavailable", "mapped_automatic"}
            )
        ):
            parent = replace(parent, target_freshness=TargetFreshness.STALE)
            issues.append(
                ProjectionIssue(
                    kind=ProjectionIssueKind.STALE_DEPENDENCY,
                    page_id=page_id,
                    target_kind=EditTargetKind.PARENT.value,
                    target_id=parent_id,
                    domain=EditDomain.TARGET_TEXT.value,
                    edit_ids=source_edit_ids,
                    reason="automatic target is historical after a source edit",
                    expected_fingerprint=source_fingerprint,
                )
            )
        for edit in edits:
            if edit.domain in {
                EditDomain.RENDER_STYLE,
                EditDomain.RENDER_LAYOUT,
                EditDomain.REVIEW_METADATA,
            }:
                parent = _apply_nontext_parent_edit(parent, edit)
                applied_ids.append(edit.edit_id)
        if parent.excluded:
            parent = replace(parent, target_freshness=TargetFreshness.EXCLUDED)
        parent_map[parent_id] = parent

    for edit in page_edits:
        if (
            edit.domain is not EditDomain.STRUCTURAL
            or edit.operation != "set_reading_order"
        ):
            continue
        ordered_parent_ids = tuple(
            str(parent_id) for parent_id in edit.payload["ordered_parent_ids"]
        )
        selected_parent_id = str(edit.payload["selected_parent_id"])
        try:
            predecessor = project_effective_page(
                project,
                _ledger_prefix_before_edit(ledger, edit),
                page_id=edit.page_id,
                _memo=_memo,
            )
        except (KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "reading_order_supersession_lineage_is_invalid",
                )
            )
            continue
        before_ordered_parent_ids = predecessor.hierarchy.ordered_parent_ids
        lineage_edit_ids: set[str] = set()
        lineage_edit_id = edit.supersedes_edit_id
        while lineage_edit_id:
            lineage_edit_ids.add(lineage_edit_id)
            lineage_edit = ledger.get(lineage_edit_id)
            lineage_edit_id = (
                lineage_edit.supersedes_edit_id
                if lineage_edit is not None
                else None
            )
        if lineage_edit_ids and any(
            issue.domain == EditDomain.STRUCTURAL.value
            and issue.target_kind == EditTargetKind.PAGE.value
            and bool(lineage_edit_ids.intersection(issue.edit_ids))
            for issue in predecessor.issues
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "reading_order_supersession_lineage_is_invalid",
                )
            )
            continue
        authored_parent_ids = frozenset(before_ordered_parent_ids)
        order_is_complete_at_ledger_position = bool(
            len(ordered_parent_ids) == len(before_ordered_parent_ids)
            and len(set(ordered_parent_ids)) == len(ordered_parent_ids)
            and frozenset(ordered_parent_ids) == authored_parent_ids
        )
        if not order_is_complete_at_ledger_position:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "reading_order_is_not_a_complete_page_permutation",
                )
            )
            continue
        predecessor_parent_by_id = {
            parent.parent_id: parent for parent in predecessor.parents
        }
        selected_parent = predecessor_parent_by_id.get(selected_parent_id)
        if selected_parent is None:
            issues.append(
                _issue(
                    ProjectionIssueKind.ORPHANED,
                    edit,
                    "selected_reading_order_parent_is_unavailable",
                )
            )
            continue
        if selected_parent.excluded:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "selected_reading_order_parent_is_excluded",
                )
            )
            continue
        if ordered_parent_ids == before_ordered_parent_ids:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "reading_order_is_no_op",
                )
            )
            continue
        excluded_parent_moved = any(
            predecessor_parent_by_id[parent_id].excluded
            and ordered_parent_ids[index] != parent_id
            for index, parent_id in enumerate(before_ordered_parent_ids)
        )
        if excluded_parent_moved:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "excluded_reading_order_parent_moved",
                )
            )
            continue
        before_other_active = tuple(
            parent_id
            for parent_id in before_ordered_parent_ids
            if parent_id != selected_parent_id
            and not predecessor_parent_by_id[parent_id].excluded
        )
        proposed_other_active = tuple(
            parent_id
            for parent_id in ordered_parent_ids
            if parent_id != selected_parent_id
            and not predecessor_parent_by_id[parent_id].excluded
        )
        if before_other_active != proposed_other_active:
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "multiple_reading_order_parents_moved",
                )
            )
            continue
        if not authored_parent_ids.issubset(parent_map):
            issues.append(
                _issue(
                    ProjectionIssueKind.ORPHANED,
                    edit,
                    "reading_order_authored_parent_is_unavailable",
                )
            )
            continue
        later_active_user_parent_ids = tuple(
            parent.parent_id
            for parent in sorted(
                parent_map.values(),
                key=lambda candidate: (
                    candidate.reading_order,
                    candidate.parent_id,
                ),
            )
            if parent.parent_id not in authored_parent_ids
        )
        effective_order = (*ordered_parent_ids, *later_active_user_parent_ids)
        if (
            len(effective_order) != len(parent_map)
            or frozenset(effective_order) != frozenset(parent_map)
        ):
            issues.append(
                _issue(
                    ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                    edit,
                    "reading_order_effective_topology_is_invalid",
                )
            )
            continue
        for reading_order, parent_id in enumerate(effective_order):
            parent = replace(
                parent_map[parent_id],
                reading_order=reading_order,
            )
            if parent_id == selected_parent_id:
                parent = _append_applied(parent, edit.edit_id)
            parent_map[parent_id] = parent
        applied_ids.append(edit.edit_id)

    (
        cleaned_base_revision_id,
        cleaned_page_base,
        cleaned_base_provenance,
    ) = _automatic_cleaned_base(project, page)
    glossary_value = project.get("glossary")
    effective_glossary = (
        dict(glossary_value) if isinstance(glossary_value, Mapping) else {}
    )
    for edit in page_edits:
        if edit.domain is EditDomain.CLEANUP and edit.operation == "select_revision":
            revision_id = str(edit.payload["revision_id"])
            indexed = artifact_index.get(revision_id)
            if (
                indexed is None
                or indexed[0] != "cleaned_page_bases"
                or str(indexed[1].get("page_id") or "") != page_id
                or not _valid_cleaned_record(indexed[1])
            ):
                issues.append(
                    _issue(
                        ProjectionIssueKind.ORPHANED,
                        edit,
                        "selected CleanedPageBase revision is unavailable",
                    )
                )
                continue
            cleaned_base_revision_id = revision_id
            cleaned_page_base = freeze_json(
                indexed[1],
                field_name="cleaned_page_base",
            )
            cleaned_base_provenance = str(
                indexed[1].get("provenance") or "user"
            )
            applied_ids.append(edit.edit_id)
        elif edit.domain is EditDomain.GLOSSARY:
            if edit.operation == "set_entry":
                entry = thaw_json(edit.payload["entry"])
                effective_glossary[str(entry["entry_id"])] = entry
            elif edit.operation == "remove_entry":
                effective_glossary.pop(str(edit.payload["entry_id"]), None)
            else:
                issues.append(
                    _issue(
                        ProjectionIssueKind.INVALID_EFFECTIVE_VALUE,
                        edit,
                        "unsupported_glossary_operation",
                    )
                )
                continue
            applied_ids.append(edit.edit_id)

    if not cleaned_base_revision_id:
        issues.append(
            ProjectionIssue(
                kind=ProjectionIssueKind.MISSING_DEPENDENCY,
                page_id=page_id,
                target_kind=EditTargetKind.PAGE.value,
                target_id=page_id,
                domain=EditDomain.CLEANUP.value,
                edit_ids=(),
                reason="valid CleanedPageBase revision is unavailable",
            )
        )

    membership_edits = tuple(
        edit
        for edits in edits_by_parent.values()
        for edit in edits
        if edit.domain is EditDomain.STRUCTURAL
        and edit.operation in {"exclude", "restore"}
        and edit.edit_id in set(applied_ids)
    )
    membership_edits_by_parent = {
        edit.target.parent_id: edit for edit in membership_edits
    }
    incompatible_membership_edit_ids: set[str] = set()
    selected_cleaned = thaw_json(cleaned_page_base)
    selected_cleanup_edit_ids = tuple(
        edit.edit_id
        for edit in page_edits
        if edit.domain is EditDomain.CLEANUP
        and edit.operation == "select_revision"
        and edit.edit_id in set(applied_ids)
    )
    selected_coverage_target = (
        _user_parent_cleanup_coverage_target(
            selected_cleaned.get("user_parent_cleanup_coverage_target")
        )
        if isinstance(selected_cleaned, Mapping)
        and str(selected_cleaned.get("provenance") or "")
        == "user_manual_cleanup"
        else None
    )
    manual_lineage_valid = bool(
        not isinstance(selected_cleaned, Mapping)
        or str(selected_cleaned.get("provenance") or "")
        != "user_manual_cleanup"
        or _manual_cleaned_base_lineage_is_valid(
            selected_cleaned,
            artifact_index,
            page_id=page_id,
        )
    )
    if (
        manual_lineage_valid
        and isinstance(selected_cleaned, Mapping)
        and str(selected_cleaned.get("provenance") or "")
        == "user_manual_cleanup"
    ):
        receipt = selected_cleaned.get("manual_cleanup_receipt")
        manual_lineage_valid = bool(
            isinstance(receipt, Mapping)
            and len(selected_cleanup_edit_ids) == 1
            and str(receipt.get("selection_edit_id") or "")
            == selected_cleanup_edit_ids[0]
        )
    if not manual_lineage_valid:
        issues.append(
            ProjectionIssue(
                kind=ProjectionIssueKind.STALE_DEPENDENCY,
                page_id=page_id,
                target_kind=EditTargetKind.PAGE.value,
                target_id=page_id,
                domain=EditDomain.CLEANUP.value,
                edit_ids=selected_cleanup_edit_ids,
                reason="manual_cleanup_lineage_invalid",
            )
        )
    automatic_by_parent = {
        _parent_id(parent): parent for parent in automatic_parents
    }
    selected_nested = (
        selected_cleaned.get("cleaned_page_base")
        if isinstance(selected_cleaned, Mapping)
        else None
    )
    # CleanedPageBase v1 signs the live bundles at cleanup time, before later
    # automatic render metadata is finalized in the saved bundles.  That full
    # signature is therefore not replayable from a schema-1 project.  The
    # cleanup compatibility boundary is the exact producer bundle set plus
    # per-parent committed-erasure membership, which is also the dependency
    # described by the GUI invalidation matrix.
    current_bundle_ids = {
        str(parent.get("bundle_id") or parent.get("parent_id") or "").strip()
        for parent in automatic_parents
        if str(parent.get("bundle_id") or parent.get("parent_id") or "").strip()
    }
    producer_bundle_ids = (
        {
            str(value).strip()
            for value in selected_nested.get("parent_execution_bundle_ids") or ()
            if str(value).strip()
        }
        if isinstance(selected_nested, Mapping)
        else set()
    )
    cleaned_base_is_compatible = bool(
        cleaned_base_revision_id
        and manual_lineage_valid
        and isinstance(selected_nested, Mapping)
        and producer_bundle_ids == current_bundle_ids
    )
    geometry_edit_ids = {
        edit.edit_id
        for parent_id, edits in edits_by_parent.items()
        for edit in edits
        if edit.domain is EditDomain.STRUCTURAL
        and edit.operation == "set_geometry"
        and edit.edit_id in set(applied_ids)
        and (
            parent_map.get(parent_id) is None
            or parent_map[parent_id].source_evidence_mapping is None
        )
    }
    if geometry_edit_ids:
        cleaned_base_is_compatible = False
        incompatible_membership_edit_ids.update(geometry_edit_ids)
    mapped_parent_by_source_id: dict[str, EffectiveParentSnapshot] = {}
    for candidate in parent_map.values():
        if candidate.source_evidence_mapping is None:
            continue
        for source_parent_id in candidate.source_evidence_mapping.source_parent_ids:
            if source_parent_id in mapped_parent_by_source_id:
                cleaned_base_is_compatible = False
                continue
            mapped_parent_by_source_id[source_parent_id] = candidate
    for parent_id, automatic_parent in automatic_by_parent.items():
        effective_parent = parent_map.get(parent_id) or mapped_parent_by_source_id.get(
            parent_id
        )
        expected_erasure = bool(automatic_parent.get("cleanup_required")) and bool(
            effective_parent is not None and not effective_parent.excluded
        )
        actual_erasure = cleaned_base_erasure_membership(
            selected_cleaned,
            automatic_parent,
        )
        if effective_parent is None or actual_erasure is None or actual_erasure != expected_erasure:
            cleaned_base_is_compatible = False
            membership_edit = membership_edits_by_parent.get(parent_id)
            if membership_edit is not None:
                incompatible_membership_edit_ids.add(membership_edit.edit_id)
    if cleaned_base_revision_id and not cleaned_base_is_compatible:
        issues.append(
            ProjectionIssue(
                kind=ProjectionIssueKind.STALE_DEPENDENCY,
                page_id=page_id,
                target_kind=EditTargetKind.PAGE.value,
                target_id=page_id,
                domain=EditDomain.CLEANUP.value,
                edit_ids=tuple(sorted(incompatible_membership_edit_ids)),
                reason="cleaned_page_base_incompatible_with_effective_hierarchy",
                expected_fingerprint=canonical_sha256(sorted(current_bundle_ids)),
                observed_fingerprint=canonical_sha256(sorted(producer_bundle_ids)),
            )
        )

    if selected_coverage_target is not None:
        coverage_baseline: EffectivePageSnapshot | None = None
        coverage_binding_valid = False
        if (
            manual_lineage_valid
            and len(selected_cleanup_edit_ids) == 1
        ):
            selection_edit = ledger.get(selected_cleanup_edit_ids[0])
            if (
                selection_edit is not None
                and not selection_edit.is_control
                and selection_edit.domain is EditDomain.CLEANUP
                and selection_edit.operation == "select_revision"
                and selection_edit.target.kind is EditTargetKind.PAGE
                and str(selection_edit.payload.get("revision_id") or "")
                == cleaned_base_revision_id
            ):
                try:
                    coverage_baseline = project_effective_page(
                        project,
                        _ledger_prefix_before_edit(ledger, selection_edit),
                        page_id=page_id,
                        _memo=_memo,
                    )
                    coverage_binding_valid = (
                        _user_parent_cleanup_target_matches_snapshot(
                            selected_coverage_target,
                            coverage_baseline,
                            page=page,
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    coverage_binding_valid = False
        coverage_parent_id = str(selected_coverage_target["parent_id"])
        coverage_parent = parent_map.get(coverage_parent_id)
        covered_parent = (
            _with_current_user_parent_cleanup_coverage(coverage_parent)
            if coverage_parent is not None
            else None
        )
        if (
            coverage_binding_valid
            and cleaned_base_is_compatible
            and covered_parent is not None
        ):
            parent_map[coverage_parent_id] = _append_applied(
                covered_parent,
                selected_cleanup_edit_ids[0],
            )
        elif not coverage_binding_valid or covered_parent is None:
            issues.append(
                ProjectionIssue(
                    kind=ProjectionIssueKind.STALE_DEPENDENCY,
                    page_id=page_id,
                    target_kind=EditTargetKind.PARENT.value,
                    target_id=coverage_parent_id,
                    domain=EditDomain.CLEANUP.value,
                    edit_ids=selected_cleanup_edit_ids,
                    reason=(
                        "user_parent_cleanup_coverage_dependency_mismatch"
                    ),
                    expected_fingerprint=str(
                        selected_coverage_target[
                            "effective_page_fingerprint"
                        ]
                    ),
                    observed_fingerprint=(
                        coverage_baseline.effective_fingerprint
                        if coverage_baseline is not None
                        else ""
                    ),
                )
            )

    for parent_id, parent in tuple(parent_map.items()):
        mapping = parent.source_evidence_mapping
        if mapping is None:
            continue
        mapped_records = tuple(
            automatic_by_parent.get(source_parent_id)
            for source_parent_id in mapping.source_parent_ids
        )
        source_style_current = bool(
            all(record is not None for record in mapped_records)
            and all(
                validate_resolved_render_style(record.get("render_style") or {}).accepted
                for record in mapped_records
                if record is not None
            )
            and len(
                {
                    canonical_sha256(record.get("render_style") or {})
                    for record in mapped_records
                    if record is not None
                }
            )
            == 1
        )
        render_eligibility_current = bool(
            source_style_current
            and all(
                bool(record.get("render_required"))
                for record in mapped_records
                if record is not None
            )
        )
        parent_map[parent_id] = replace(
            parent,
            stage_requirements=_user_parent_stage_requirements(
                page_id=page_id,
                parent_id=parent_id,
                source_current=bool(parent.source_text),
                translation_current=(
                    parent.target_freshness is TargetFreshness.CURRENT
                    and bool(parent.target_text)
                ),
                cleanup_current=cleaned_base_is_compatible,
                source_style_current=source_style_current,
                render_eligibility_current=render_eligibility_current,
            ),
        )

    sorted_issues = _sort_issues(issues)
    parents_without_issues = tuple(
        sorted(
            parent_map.values(),
            key=lambda parent: (parent.reading_order, parent.parent_id),
        )
    )
    parents = tuple(
        replace(
            parent,
            issues=tuple(
                issue
                for issue in sorted_issues
                if issue.target_kind == EditTargetKind.PARENT.value
                and issue.target_id == parent.parent_id
            ),
        )
        for parent in parents_without_issues
    )
    ordered_parent_ids = tuple(parent.parent_id for parent in parents)
    excluded_parent_ids = tuple(
        parent.parent_id for parent in parents if parent.excluded
    )
    applied_id_set = set(applied_ids)
    active_structural_edit_ids = tuple(
        edit.edit_id
        for edit in ledger.active_edits(page_id=page_id)
        if edit.edit_id in applied_id_set
        and edit.domain is EditDomain.STRUCTURAL
    )
    user_parent_lineage = tuple(
        parent.lineage
        for parent in parents
        if parent.lineage is not None
    )
    user_roots = tuple(
        EffectiveUserRootSnapshot(
            root_id=lineage.root_id,
            identity_namespace=RootIdentityNamespace.USER_ROOT_V1,
            origin=ParentOrigin.USER,
            evidence_kind=RootEvidenceKind.WORKFLOW_AREA_ONLY,
            workflow_area_bbox=lineage.workflow_area_bbox,
            authored_edit_id=lineage.authored_edit_id,
        )
        for lineage in user_parent_lineage
    )
    effective_stage_requirements = tuple(
        requirement
        for parent in parents
        for requirement in parent.stage_requirements
    )
    hierarchy_stage_requirements = tuple(
        requirement
        for parent in parents
        if parent.lineage is not None
        for requirement in _user_parent_stage_requirements(
            page_id=page_id,
            parent_id=parent.parent_id,
        )
    )
    topology_body = {
        "automatic_base_revision_id": base_revision_id,
        "parents": [
            {
                "parent_id": parent.parent_id,
                "root_id": parent.root_id,
                "origin": parent.origin.value,
                "identity_namespace": parent.identity_namespace.value,
                "root_identity_namespace": parent.root_identity_namespace.value,
                "role": parent.role,
                "automatic_geometry": thaw_json(parent.automatic_geometry),
                "effective_geometry": thaw_json(parent.geometry),
                "workflow_area_bbox": thaw_json(parent.workflow_area_bbox),
                "reading_order": parent.reading_order,
                "excluded": parent.excluded,
            }
            for parent in parents
        ],
    }
    topology_fingerprint = canonical_sha256(topology_body)
    descriptor_body = {
        "page_id": page_id,
        "automatic_base_revision_id": base_revision_id,
        "topology_fingerprint": topology_fingerprint,
        "ordered_parent_ids": list(ordered_parent_ids),
        "excluded_parent_ids": list(excluded_parent_ids),
        "active_structural_edit_ids": list(active_structural_edit_ids),
        "user_roots": [root.to_dict() for root in user_roots],
        "user_parent_lineage": [
            lineage.to_dict() for lineage in user_parent_lineage
        ],
        "stage_requirements": [
            requirement.to_dict()
            for requirement in hierarchy_stage_requirements
        ],
    }
    hierarchy_fingerprint = canonical_sha256(descriptor_body)
    hierarchy_revision_id = (
        EFFECTIVE_HIERARCHY_REVISION_PREFIX + hierarchy_fingerprint
    )
    hierarchy_descriptor = HierarchyRevisionDescriptor(
        page_id=page_id,
        automatic_base_revision_id=base_revision_id,
        topology_fingerprint=topology_fingerprint,
        ordered_parent_ids=ordered_parent_ids,
        excluded_parent_ids=excluded_parent_ids,
        active_structural_edit_ids=active_structural_edit_ids,
        user_roots=user_roots,
        user_parent_lineage=user_parent_lineage,
        stage_requirements=hierarchy_stage_requirements,
        fingerprint=hierarchy_fingerprint,
        revision_id=hierarchy_revision_id,
    )
    hierarchy = EffectiveHierarchySnapshot(
        ordered_parent_ids=ordered_parent_ids,
        excluded_parent_ids=excluded_parent_ids,
        fingerprint=hierarchy_fingerprint,
        revision_id=hierarchy_revision_id,
        descriptor=hierarchy_descriptor,
        user_roots=user_roots,
    )
    provisional = EffectivePageSnapshot(
        project_id=project_id,
        page_id=page_id,
        automatic_fingerprint=page_fingerprint,
        base_revision_id=base_revision_id,
        cleaned_base_revision_id=cleaned_base_revision_id,
        cleaned_page_base=cleaned_page_base,
        cleaned_base_provenance=cleaned_base_provenance,
        hierarchy=hierarchy,
        parents=parents,
        effective_glossary=_mapping_tuple(effective_glossary),
        applied_edit_ids=tuple(dict.fromkeys(applied_ids)),
        issues=sorted_issues,
        effective_fingerprint="",
        stage_requirements=effective_stage_requirements,
    )
    result = replace(
        provisional,
        effective_fingerprint=canonical_sha256(
            provisional.to_dict(include_fingerprint=False)
        ),
    )
    _memo[memo_key] = result
    return result
