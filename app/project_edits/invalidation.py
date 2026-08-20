# -*- coding: utf-8 -*-
"""Target-scoped invalidation policy from the GUI architecture matrix."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping

from .contracts import EditDomain, EditTargetKind, ProjectEdit
from .fingerprints import canonical_sha256
from .projection import (
    EffectivePageSnapshot,
    EffectiveParentSnapshot,
    ProjectionIssueKind,
    TargetFreshness,
)


class Dependency(str, Enum):
    HIERARCHY = "hierarchy"
    SOURCE = "source"
    TRANSLATION = "translation"
    CLEANUP_BASE = "cleanup_base"
    STYLE_CACHE = "style_cache"
    RENDER_ELIGIBILITY = "render_eligibility"
    LAYOUT_RENDER = "layout_render"
    PAGE_OUTPUT = "page_output"


class InvalidationAction(str, Enum):
    KEEP = "keep"
    USER_CURRENT = "user_current"
    NEW_REVISION = "new_revision"
    STALE = "stale"
    RERUN = "rerun"
    REBUILD = "rebuild"
    RECOMPUTE = "recompute"
    EXCLUDED = "excluded"
    REQUIRES_FACT = "requires_fact"


class InvalidationScope(str, Enum):
    NONE = "none"
    PARENT = "parent"
    PAGE = "page"
    PROJECT = "project"
    STYLE_CACHE_PREFIX = "style_cache_prefix"


@dataclass(frozen=True)
class InvalidationFacts:
    cleanup_contains_target_pixels: bool | None = None
    source_crop_changed: bool | None = None
    style_donor_set_changed: bool | None = None


@dataclass(frozen=True)
class DependencyEffect:
    dependency: Dependency
    action: InvalidationAction
    scope: InvalidationScope
    target_ids: tuple[str, ...]
    reason: str
    required_fact: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency": self.dependency.value,
            "action": self.action.value,
            "scope": self.scope.value,
            "target_ids": list(self.target_ids),
            "reason": self.reason,
            "required_fact": self.required_fact,
        }


@dataclass(frozen=True)
class InvalidationResult:
    effects: tuple[DependencyEffect, ...]
    unresolved_facts: tuple[str, ...]
    fingerprint: str = ""

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "effects": [effect.to_dict() for effect in self.effects],
            "unresolved_facts": list(self.unresolved_facts),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def action_for(
        self,
        dependency: Dependency,
        *,
        target_id: str,
    ) -> InvalidationAction:
        matches = [
            effect.action
            for effect in self.effects
            if effect.dependency is dependency and target_id in effect.target_ids
        ]
        if not matches:
            return InvalidationAction.KEEP
        return max(matches, key=lambda action: _SEVERITY[action])


def _target_id(edit: ProjectEdit) -> str:
    if edit.target.kind is EditTargetKind.PARENT:
        return edit.target.parent_id
    if edit.target.kind is EditTargetKind.ARTIFACT:
        return edit.target.artifact_id
    return edit.page_id


def _effect(
    dependency: Dependency,
    action: InvalidationAction,
    scope: InvalidationScope,
    target_id: str,
    reason: str,
    *,
    required_fact: str = "",
) -> DependencyEffect:
    return DependencyEffect(
        dependency=dependency,
        action=action,
        scope=scope,
        target_ids=(target_id,),
        reason=reason,
        required_fact=required_fact,
    )


def _result(effects: Iterable[DependencyEffect]) -> InvalidationResult:
    ordered = tuple(
        sorted(
            effects,
            key=lambda effect: (
                effect.dependency.value,
                effect.scope.value,
                effect.target_ids,
                effect.action.value,
                effect.reason,
            ),
        )
    )
    unresolved = tuple(
        sorted(
            {
                effect.required_fact
                for effect in ordered
                if effect.required_fact
            }
        )
    )
    provisional = InvalidationResult(ordered, unresolved)
    return replace(
        provisional,
        fingerprint=canonical_sha256(
            provisional.to_dict(include_fingerprint=False)
        ),
    )


def _conditional_effect(
    *,
    dependency: Dependency,
    value: bool | None,
    when_true: InvalidationAction,
    when_false: InvalidationAction,
    scope: InvalidationScope,
    target_id: str,
    reason: str,
    fact_name: str,
) -> DependencyEffect:
    if value is None:
        return _effect(
            dependency,
            InvalidationAction.REQUIRES_FACT,
            scope,
            target_id,
            reason,
            required_fact=fact_name,
        )
    return _effect(
        dependency,
        when_true if value else when_false,
        scope,
        target_id,
        reason,
    )


def invalidation_for_edit(
    edit: ProjectEdit,
    *,
    facts: InvalidationFacts = InvalidationFacts(),
) -> InvalidationResult:
    if edit.is_control:
        raise ValueError("resolve a ledger control against its target edit first")
    parent_id = _target_id(edit)
    page_id = edit.page_id
    domain = edit.domain

    if domain is EditDomain.REVIEW_METADATA:
        return _result(())
    if domain is EditDomain.TARGET_TEXT:
        translation_action = (
            InvalidationAction.NEW_REVISION
            if edit.operation
            in {
                "select_revision",
                "restore_selected_revision",
                "restore_mapped_pipeline",
            }
            else InvalidationAction.USER_CURRENT
        )
        reason = (
            "restored_mapped_pipeline_translation"
            if edit.operation == "restore_mapped_pipeline"
            else (
                "selected_translation_revision"
                if edit.operation
                in {"select_revision", "restore_selected_revision"}
                else "user_target_current"
            )
        )
        return _result(
            (
                _effect(Dependency.TRANSLATION, translation_action, InvalidationScope.PARENT, parent_id, reason),
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "target_text_changed"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "target_text_changed"),
            )
        )
    if domain is EditDomain.SOURCE_TEXT:
        if edit.operation == "select_revision":
            return _result(
                (
                    _effect(
                        Dependency.SOURCE,
                        InvalidationAction.NEW_REVISION,
                        InvalidationScope.PARENT,
                        parent_id,
                        "selected_ocr_source_revision",
                    ),
                    _effect(
                        Dependency.TRANSLATION,
                        InvalidationAction.RERUN,
                        InvalidationScope.PARENT,
                        parent_id,
                        "source_revision_changed",
                    ),
                )
            )
        return _result(
            (
                _effect(Dependency.SOURCE, InvalidationAction.USER_CURRENT, InvalidationScope.PARENT, parent_id, "user_source_current"),
                _effect(Dependency.TRANSLATION, InvalidationAction.STALE, InvalidationScope.PARENT, parent_id, "effective_source_changed"),
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.STALE, InvalidationScope.PARENT, parent_id, "target_not_current"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.STALE, InvalidationScope.PAGE, parent_id, "target_not_current"),
            )
        )
    if domain is EditDomain.RENDER_STYLE:
        return _result(
            (
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "render_style_override"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "render_style_override"),
            )
        )
    if domain is EditDomain.RENDER_LAYOUT:
        return _result(
            (
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "render_layout_override"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "render_layout_override"),
            )
        )
    if domain is EditDomain.CLEANUP:
        return _result(
            (
                _effect(Dependency.CLEANUP_BASE, InvalidationAction.USER_CURRENT, InvalidationScope.PAGE, page_id, "selected_cleanup_revision"),
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, page_id, "cleaned_base_changed"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, page_id, "cleaned_base_changed"),
            )
        )
    if domain is EditDomain.STRUCTURAL:
        if edit.operation == "add_user_parent":
            return _user_parent_topology_invalidation(
                page_id=page_id,
                parent_id=parent_id,
                reason="user_parent_added",
            )
        if edit.operation == "split_user_parent":
            child_parent_ids = tuple(
                str(value) for value in edit.payload["child_parent_ids"]
            )
            return _user_parent_split_invalidation(
                page_id=page_id,
                child_parent_ids=child_parent_ids,
                reason="user_parent_split",
            )
        if edit.operation == "merge_pipeline_parents":
            return _pipeline_parent_merge_invalidation(
                page_id=page_id,
                merged_parent_id=parent_id,
                reason="pipeline_parents_merged",
            )
        if edit.operation == "set_reading_order":
            return _result(
                (
                    _effect(
                        Dependency.HIERARCHY,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                    _effect(
                        Dependency.LAYOUT_RENDER,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                    _effect(
                        Dependency.PAGE_OUTPUT,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                )
            )
        common = [
            _effect(Dependency.HIERARCHY, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, page_id, "effective_hierarchy_changed"),
            _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "parent_structure_changed"),
            _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "parent_structure_changed"),
        ]
        if edit.operation in {"exclude", "restore"}:
            common.extend(
                (
                    _effect(
                        Dependency.TRANSLATION,
                        InvalidationAction.EXCLUDED if edit.operation == "exclude" else InvalidationAction.KEEP,
                        InvalidationScope.PARENT,
                        parent_id,
                        "parent_workflow_membership_changed",
                    ),
                    _conditional_effect(
                        dependency=Dependency.CLEANUP_BASE,
                        value=facts.cleanup_contains_target_pixels,
                        when_true=InvalidationAction.REBUILD,
                        when_false=InvalidationAction.KEEP,
                        scope=InvalidationScope.PARENT,
                        target_id=parent_id,
                        reason="prior_cleanup_may_contain_parent_pixels",
                        fact_name="cleanup_contains_target_pixels",
                    ),
                    _conditional_effect(
                        dependency=Dependency.STYLE_CACHE,
                        value=facts.style_donor_set_changed,
                        when_true=InvalidationAction.STALE,
                        when_false=InvalidationAction.KEEP,
                        scope=InvalidationScope.STYLE_CACHE_PREFIX,
                        target_id=page_id,
                        reason="style_donor_set_may_change",
                        fact_name="style_donor_set_changed",
                    ),
                )
            )
            return _result(common)
        if edit.operation == "set_geometry":
            common.extend(
                (
                    _effect(
                        Dependency.SOURCE,
                        InvalidationAction.RERUN,
                        InvalidationScope.PARENT,
                        parent_id,
                        "effective_source_crop_changed",
                    ),
                    _effect(
                        Dependency.TRANSLATION,
                        InvalidationAction.STALE,
                        InvalidationScope.PARENT,
                        parent_id,
                        "effective_source_crop_changed",
                    ),
                    _effect(
                        Dependency.CLEANUP_BASE,
                        InvalidationAction.REBUILD,
                        InvalidationScope.PARENT,
                        parent_id,
                        "effective_cleanup_geometry_changed",
                    ),
                    _effect(
                        Dependency.STYLE_CACHE,
                        InvalidationAction.RERUN,
                        InvalidationScope.STYLE_CACHE_PREFIX,
                        page_id,
                        "effective_style_observation_crop_changed",
                    ),
                )
            )
            return _result(common)
        common.extend(
            (
                _conditional_effect(
                    dependency=Dependency.SOURCE,
                    value=facts.source_crop_changed,
                    when_true=InvalidationAction.RERUN,
                    when_false=InvalidationAction.KEEP,
                    scope=InvalidationScope.PARENT,
                    target_id=parent_id,
                    reason="source_crop_may_change",
                    fact_name="source_crop_changed",
                ),
                _conditional_effect(
                    dependency=Dependency.CLEANUP_BASE,
                    value=facts.cleanup_contains_target_pixels,
                    when_true=InvalidationAction.REBUILD,
                    when_false=InvalidationAction.KEEP,
                    scope=InvalidationScope.PARENT,
                    target_id=parent_id,
                    reason="cleanup_may_need_rebuild",
                    fact_name="cleanup_contains_target_pixels",
                ),
                _conditional_effect(
                    dependency=Dependency.STYLE_CACHE,
                    value=facts.style_donor_set_changed,
                    when_true=InvalidationAction.RERUN,
                    when_false=InvalidationAction.KEEP,
                    scope=InvalidationScope.STYLE_CACHE_PREFIX,
                    target_id=page_id,
                    reason="style_observation_may_change",
                    fact_name="style_donor_set_changed",
                ),
            )
        )
        return _result(common)
    if domain is EditDomain.GLOSSARY:
        project_id = edit.project_id
        return _result(
            (
                _effect(Dependency.TRANSLATION, InvalidationAction.STALE, InvalidationScope.PROJECT, project_id, "glossary_changed"),
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.STALE, InvalidationScope.PROJECT, project_id, "glossary_changed"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.STALE, InvalidationScope.PROJECT, project_id, "glossary_changed"),
            )
        )
    raise ValueError(f"unsupported invalidation domain: {domain.value}")


_SEVERITY = {
    InvalidationAction.KEEP: 0,
    InvalidationAction.USER_CURRENT: 1,
    InvalidationAction.NEW_REVISION: 2,
    InvalidationAction.RECOMPUTE: 3,
    InvalidationAction.STALE: 4,
    InvalidationAction.RERUN: 5,
    InvalidationAction.REBUILD: 6,
    InvalidationAction.REQUIRES_FACT: 7,
    InvalidationAction.EXCLUDED: 8,
}


def combine_invalidation(
    results: Iterable[InvalidationResult],
    *,
    effective_target_current_by_parent: Mapping[str, bool],
) -> InvalidationResult:
    grouped: dict[
        tuple[Dependency, InvalidationScope, tuple[str, ...]],
        list[DependencyEffect],
    ] = {}
    for result in results:
        for effect in result.effects:
            grouped.setdefault(
                (effect.dependency, effect.scope, effect.target_ids),
                [],
            ).append(effect)
    combined: list[DependencyEffect] = []
    for key, candidates in grouped.items():
        dependency, scope, target_ids = key
        selected = max(candidates, key=lambda effect: _SEVERITY[effect.action])
        parent_id = target_ids[0] if target_ids else ""
        if effective_target_current_by_parent.get(parent_id) is True:
            if dependency is Dependency.TRANSLATION and selected.action is InvalidationAction.STALE:
                selected = replace(selected, action=InvalidationAction.USER_CURRENT, reason="effective_target_current")
            elif dependency in {Dependency.LAYOUT_RENDER, Dependency.PAGE_OUTPUT} and selected.action is InvalidationAction.STALE:
                selected = replace(selected, action=InvalidationAction.RECOMPUTE, reason="effective_target_current")
        combined.append(selected)
    return _result(combined)


def invalidation_for_control(
    control: ProjectEdit,
    *,
    target_edit: ProjectEdit,
    before_effective_page: EffectivePageSnapshot,
    after_effective_page: EffectivePageSnapshot,
) -> InvalidationResult:
    """Derive one ledger control's effects from exact effective-state change.

    Revoke and reapply are opposite ledger transitions, so the target edit's
    forward invalidation is not a valid control receipt.  This boundary instead
    compares the immutable effective page immediately before and after the
    control.  It never asks a downstream owner for a fact and therefore cannot
    return ``REQUIRES_FACT``.
    """

    if not control.is_control:
        raise ValueError("control record is required")
    if control.target.edit_id != target_edit.edit_id:
        raise ValueError("control and target edit identities differ")
    if control.operation not in {"revoke", "reapply"}:
        raise ValueError("unsupported ledger control operation")
    if target_edit.is_control:
        raise ValueError("a ledger control cannot target another control")
    if (
        control.project_id != target_edit.project_id
        or control.page_id != target_edit.page_id
    ):
        raise ValueError("control and target edit scope differ")
    for snapshot, label in (
        (before_effective_page, "before"),
        (after_effective_page, "after"),
    ):
        if not isinstance(snapshot, EffectivePageSnapshot):
            raise TypeError(f"{label}_effective_page must be an EffectivePageSnapshot")
        if (
            snapshot.project_id != target_edit.project_id
            or snapshot.page_id != target_edit.page_id
        ):
            raise ValueError(f"{label} effective-page scope differs from the edit")

    page_id = target_edit.page_id
    domain = target_edit.domain
    if domain is EditDomain.GLOSSARY:
        before_glossary = before_effective_page.effective_glossary
        after_glossary = after_effective_page.effective_glossary
        if before_glossary == after_glossary:
            return _result(())
        return _result(
            (
                _effect(
                    Dependency.TRANSLATION,
                    InvalidationAction.STALE,
                    InvalidationScope.PROJECT,
                    target_edit.project_id,
                    "effective_glossary_changed",
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.STALE,
                    InvalidationScope.PROJECT,
                    target_edit.project_id,
                    "effective_glossary_changed",
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.STALE,
                    InvalidationScope.PROJECT,
                    target_edit.project_id,
                    "effective_glossary_changed",
                ),
            )
        )
    if (
        domain is EditDomain.STRUCTURAL
        and target_edit.operation == "add_user_parent"
    ):
        if target_edit.target.kind is not EditTargetKind.PARENT:
            raise ValueError("add_user_parent control target must be a parent")
        before_ids = set(before_effective_page.hierarchy.ordered_parent_ids)
        after_ids = set(after_effective_page.hierarchy.ordered_parent_ids)
        parent_id = target_edit.target.parent_id
        before_present = parent_id in before_ids
        after_present = parent_id in after_ids
        if before_present == after_present:
            return _result(())
        expected_after_present = control.operation == "reapply"
        if after_present != expected_after_present:
            raise ValueError(
                "add_user_parent control direction differs from effective state"
            )
        if not after_present:
            return _user_parent_removal_invalidation(
                page_id=page_id,
                reason="user_parent_removed",
            )
        return _user_parent_topology_invalidation(
            page_id=page_id,
            parent_id=parent_id,
            reason="user_parent_reapplied",
        )
    if (
        domain is EditDomain.STRUCTURAL
        and target_edit.operation == "split_user_parent"
    ):
        if target_edit.target.kind is not EditTargetKind.PARENT:
            raise ValueError("split_user_parent control target must be a parent")
        source_parent_id = target_edit.target.parent_id
        child_parent_ids = tuple(
            str(value) for value in target_edit.payload["child_parent_ids"]
        )
        before_ids = set(before_effective_page.hierarchy.ordered_parent_ids)
        after_ids = set(after_effective_page.hierarchy.ordered_parent_ids)
        before_source = source_parent_id in before_ids
        after_source = source_parent_id in after_ids
        before_children = tuple(parent_id in before_ids for parent_id in child_parent_ids)
        after_children = tuple(parent_id in after_ids for parent_id in child_parent_ids)
        expected_reapplied = control.operation == "reapply"
        if expected_reapplied:
            direction_valid = bool(
                before_source
                and not any(before_children)
                and not after_source
                and all(after_children)
            )
        else:
            direction_valid = bool(
                not before_source
                and all(before_children)
                and after_source
                and not any(after_children)
            )
        if not direction_valid:
            raise ValueError(
                "split_user_parent control direction differs from effective state"
            )
        if expected_reapplied:
            return _user_parent_split_invalidation(
                page_id=page_id,
                child_parent_ids=child_parent_ids,
                reason="user_parent_split_reapplied",
            )
        restored_parent = _control_parent(
            after_effective_page,
            source_parent_id,
        )
        return _user_parent_split_restore_invalidation(
            page_id=page_id,
            parent=restored_parent,
            reason="user_parent_split_revoked",
        )
    if (
        domain is EditDomain.STRUCTURAL
        and target_edit.operation == "merge_pipeline_parents"
    ):
        if target_edit.target.kind is not EditTargetKind.PARENT:
            raise ValueError("merge_pipeline_parents control target must be a parent")
        merged_parent_id = target_edit.target.parent_id
        source_parent_ids = tuple(
            str(value) for value in target_edit.payload["source_parent_ids"]
        )
        before_ids = set(before_effective_page.hierarchy.ordered_parent_ids)
        after_ids = set(after_effective_page.hierarchy.ordered_parent_ids)
        before_merged = merged_parent_id in before_ids
        after_merged = merged_parent_id in after_ids
        before_sources = tuple(parent_id in before_ids for parent_id in source_parent_ids)
        after_sources = tuple(parent_id in after_ids for parent_id in source_parent_ids)
        expected_reapplied = control.operation == "reapply"
        if expected_reapplied:
            direction_valid = bool(
                not before_merged
                and all(before_sources)
                and after_merged
                and not any(after_sources)
            )
        else:
            direction_valid = bool(
                before_merged
                and not any(before_sources)
                and not after_merged
                and all(after_sources)
            )
        if not direction_valid:
            raise ValueError(
                "merge_pipeline_parents control direction differs from effective state"
            )
        if expected_reapplied:
            return _pipeline_parent_merge_invalidation(
                page_id=page_id,
                merged_parent_id=merged_parent_id,
                reason="pipeline_parent_merge_reapplied",
            )
        restored_parents = tuple(
            _control_parent(after_effective_page, parent_id)
            for parent_id in source_parent_ids
        )
        return _pipeline_parent_merge_restore_invalidation(
            page_id=page_id,
            parents=restored_parents,
            reason="pipeline_parent_merge_revoked",
        )
    if target_edit.target.kind is EditTargetKind.PARENT:
        parent_id = target_edit.target.parent_id
        before_parent = _control_parent(before_effective_page, parent_id)
        after_parent = _control_parent(after_effective_page, parent_id)
    else:
        parent_id = ""
        before_parent = None
        after_parent = None

    if domain is EditDomain.REVIEW_METADATA:
        return _result(())

    if domain is EditDomain.TARGET_TEXT:
        assert before_parent is not None and after_parent is not None
        before_state = (
            before_parent.target_text,
            before_parent.target_authority,
            before_parent.target_freshness,
            before_parent.target_revision_id,
        )
        after_state = (
            after_parent.target_text,
            after_parent.target_authority,
            after_parent.target_freshness,
            after_parent.target_revision_id,
        )
        if before_state == after_state:
            return _result(())
        return _result(
            (
                _effect(
                    Dependency.TRANSLATION,
                    _translation_action_for_effective_parent(after_parent),
                    InvalidationScope.PARENT,
                    parent_id,
                    "effective_target_state_changed",
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    parent_id,
                    "effective_target_state_changed",
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    parent_id,
                    "effective_target_state_changed",
                ),
            )
        )

    if domain is EditDomain.SOURCE_TEXT:
        assert before_parent is not None and after_parent is not None
        before_state = (
            before_parent.source_text,
            before_parent.source_authority,
            before_parent.target_text,
            before_parent.target_authority,
            before_parent.target_freshness,
        )
        after_state = (
            after_parent.source_text,
            after_parent.source_authority,
            after_parent.target_text,
            after_parent.target_authority,
            after_parent.target_freshness,
        )
        if before_state == after_state:
            return _result(())
        if target_edit.operation == "select_revision":
            return _result(
                (
                    _effect(
                        Dependency.SOURCE,
                        (
                            InvalidationAction.NEW_REVISION
                            if after_parent.source_revision_id
                            else InvalidationAction.RERUN
                        ),
                        InvalidationScope.PARENT,
                        parent_id,
                        "effective_ocr_source_revision_changed",
                    ),
                    _effect(
                        Dependency.TRANSLATION,
                        InvalidationAction.RERUN,
                        InvalidationScope.PARENT,
                        parent_id,
                        "effective_ocr_source_revision_changed",
                    ),
                )
            )
        target_action = _translation_action_for_effective_parent(after_parent)
        render_action = (
            InvalidationAction.STALE
            if target_action is InvalidationAction.STALE
            else InvalidationAction.RECOMPUTE
        )
        return _result(
            (
                _effect(
                    Dependency.SOURCE,
                    (
                        InvalidationAction.USER_CURRENT
                        if after_parent.source_authority == "user"
                        else InvalidationAction.KEEP
                    ),
                    InvalidationScope.PARENT,
                    parent_id,
                    "effective_source_state_changed",
                ),
                _effect(
                    Dependency.TRANSLATION,
                    target_action,
                    InvalidationScope.PARENT,
                    parent_id,
                    "effective_source_state_changed",
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    render_action,
                    InvalidationScope.PARENT,
                    parent_id,
                    "effective_source_state_changed",
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    render_action,
                    InvalidationScope.PAGE,
                    parent_id,
                    "effective_source_state_changed",
                ),
            )
        )

    if domain is EditDomain.RENDER_STYLE:
        assert before_parent is not None and after_parent is not None
        before_overrides = before_parent.render_style_overrides
        after_overrides = after_parent.render_style_overrides
        if before_overrides == after_overrides:
            return _result(())
        reason = "effective_render_style_changed"
        return _result(
            (
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    parent_id,
                    reason,
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    parent_id,
                    reason,
                ),
            )
        )

    if domain is EditDomain.RENDER_LAYOUT:
        assert before_parent is not None and after_parent is not None
        before_overrides = before_parent.render_layout_overrides
        after_overrides = after_parent.render_layout_overrides
        if before_overrides == after_overrides:
            return _result(())
        reason = "effective_render_layout_changed"
        return _result(
            (
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    parent_id,
                    reason,
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    parent_id,
                    reason,
                ),
            )
        )

    if domain is EditDomain.CLEANUP:
        before_state = (
            before_effective_page.cleaned_base_revision_id,
            before_effective_page.cleaned_page_base,
            before_effective_page.cleaned_base_provenance,
        )
        after_state = (
            after_effective_page.cleaned_base_revision_id,
            after_effective_page.cleaned_page_base,
            after_effective_page.cleaned_base_provenance,
        )
        if before_state == after_state:
            return _result(())
        cleanup_action = (
            InvalidationAction.KEEP
            if after_effective_page.cleaned_base_provenance == "automatic"
            else InvalidationAction.USER_CURRENT
        )
        return _result(
            (
                _effect(
                    Dependency.CLEANUP_BASE,
                    cleanup_action,
                    InvalidationScope.PAGE,
                    page_id,
                    "effective_cleaned_base_changed",
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    page_id,
                    "effective_cleaned_base_changed",
                ),
                _effect(
                    Dependency.PAGE_OUTPUT,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PAGE,
                    page_id,
                    "effective_cleaned_base_changed",
                ),
            )
        )

    if domain is EditDomain.STRUCTURAL:
        if target_edit.operation == "set_reading_order":
            if target_edit.target.kind is not EditTargetKind.PAGE:
                raise ValueError("reading-order control target must be a page")
            if (
                before_effective_page.hierarchy.ordered_parent_ids
                == after_effective_page.hierarchy.ordered_parent_ids
            ):
                return _result(())
            return _result(
                (
                    _effect(
                        Dependency.HIERARCHY,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                    _effect(
                        Dependency.LAYOUT_RENDER,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                    _effect(
                        Dependency.PAGE_OUTPUT,
                        InvalidationAction.RECOMPUTE,
                        InvalidationScope.PAGE,
                        page_id,
                        "effective_reading_order_changed",
                    ),
                )
            )
        assert before_parent is not None and after_parent is not None
        if target_edit.operation == "set_geometry":
            if before_parent.geometry == after_parent.geometry:
                return _result(())
            return _result(
                (
                    _effect(Dependency.HIERARCHY, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, page_id, "effective_geometry_changed"),
                    _effect(Dependency.SOURCE, InvalidationAction.RERUN, InvalidationScope.PARENT, parent_id, "effective_geometry_changed"),
                    _effect(Dependency.TRANSLATION, InvalidationAction.STALE, InvalidationScope.PARENT, parent_id, "effective_geometry_changed"),
                    _effect(Dependency.CLEANUP_BASE, InvalidationAction.REBUILD, InvalidationScope.PARENT, parent_id, "effective_geometry_changed"),
                    _effect(Dependency.STYLE_CACHE, InvalidationAction.RERUN, InvalidationScope.STYLE_CACHE_PREFIX, page_id, "effective_geometry_changed"),
                    _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "effective_geometry_changed"),
                    _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "effective_geometry_changed"),
                )
            )
        if target_edit.operation in {"exclude", "restore"}:
            if before_parent.excluded is after_parent.excluded:
                return _result(())
            cleanup_action = (
                InvalidationAction.REBUILD
                if _cleanup_hierarchy_is_stale(after_effective_page)
                else InvalidationAction.KEEP
            )
            effects = [
                _effect(Dependency.HIERARCHY, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, page_id, "effective_membership_changed"),
                _effect(Dependency.TRANSLATION, _translation_action_for_effective_parent(after_parent), InvalidationScope.PARENT, parent_id, "effective_membership_changed"),
                _effect(Dependency.CLEANUP_BASE, cleanup_action, InvalidationScope.PARENT, parent_id, "effective_cleanup_membership_evaluated"),
                _effect(Dependency.LAYOUT_RENDER, InvalidationAction.RECOMPUTE, InvalidationScope.PARENT, parent_id, "effective_membership_changed"),
                _effect(Dependency.PAGE_OUTPUT, InvalidationAction.RECOMPUTE, InvalidationScope.PAGE, parent_id, "effective_membership_changed"),
            ]
            # EffectivePageSnapshot intentionally does not expose whether this
            # parent contributed qualified automatic style evidence.  Do not
            # infer that semantic fact from resolved render style.  The command
            # service fails closed before persistence while this fact is absent.
            effects.append(
                _effect(
                    Dependency.STYLE_CACHE,
                    InvalidationAction.REQUIRES_FACT,
                    InvalidationScope.STYLE_CACHE_PREFIX,
                    page_id,
                    "qualified_style_donor_membership_is_unavailable",
                    required_fact="style_donor_set_changed",
                )
            )
            return _result(effects)
        raise ValueError(
            "control invalidation is unsupported for this structural operation"
        )

    raise ValueError(f"unsupported control invalidation domain: {domain.value}")


def _control_parent(
    snapshot: EffectivePageSnapshot,
    parent_id: str,
) -> EffectiveParentSnapshot:
    matches = tuple(
        parent for parent in snapshot.parents if parent.parent_id == parent_id
    )
    if len(matches) != 1:
        raise ValueError(f"effective parent identity is not exact: {parent_id}")
    return matches[0]


def _translation_action_for_effective_parent(
    parent: EffectiveParentSnapshot,
) -> InvalidationAction:
    if parent.target_freshness is TargetFreshness.EXCLUDED or parent.excluded:
        return InvalidationAction.EXCLUDED
    if parent.target_freshness is TargetFreshness.STALE:
        return InvalidationAction.STALE
    if parent.target_authority == "user":
        return InvalidationAction.USER_CURRENT
    if parent.target_authority == "translation_revision":
        return InvalidationAction.NEW_REVISION
    if (
        parent.source_authority == "ocr_revision"
        and parent.target_freshness is TargetFreshness.UNAVAILABLE
    ):
        return InvalidationAction.RERUN
    return InvalidationAction.KEEP


def _cleanup_hierarchy_is_stale(snapshot: EffectivePageSnapshot) -> bool:
    return any(
        issue.kind is ProjectionIssueKind.STALE_DEPENDENCY
        and issue.domain == EditDomain.CLEANUP.value
        and issue.reason == "cleaned_page_base_incompatible_with_effective_hierarchy"
        for issue in snapshot.issues
    )


def _user_parent_removal_invalidation(
    *,
    page_id: str,
    reason: str,
) -> InvalidationResult:
    """Invalidate only page-scoped state after a user parent is absent."""

    return _result(
        (
            _effect(
                Dependency.HIERARCHY,
                InvalidationAction.NEW_REVISION,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.CLEANUP_BASE,
                InvalidationAction.REBUILD,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.PAGE_OUTPUT,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
        )
    )


def _user_parent_topology_invalidation(
    *,
    page_id: str,
    parent_id: str,
    reason: str,
) -> InvalidationResult:
    """Return the exact, zero-unknown invalidation for user-parent topology."""

    return _result(
        (
            _effect(
                Dependency.HIERARCHY,
                InvalidationAction.NEW_REVISION,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.SOURCE,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.TRANSLATION,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.CLEANUP_BASE,
                InvalidationAction.REBUILD,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.STYLE_CACHE,
                InvalidationAction.RERUN,
                InvalidationScope.STYLE_CACHE_PREFIX,
                page_id,
                reason,
            ),
            _effect(
                Dependency.RENDER_ELIGIBILITY,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.LAYOUT_RENDER,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.PAGE_OUTPUT,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
        )
    )


def _source_action_for_effective_parent(
    parent: EffectiveParentSnapshot,
) -> InvalidationAction:
    if parent.source_authority == "user":
        return InvalidationAction.USER_CURRENT
    if parent.source_authority == "ocr_revision":
        return InvalidationAction.NEW_REVISION
    if parent.source_authority == "automatic":
        return InvalidationAction.KEEP
    return InvalidationAction.RERUN


def _cleanup_action_for_effective_parent(
    parent: EffectiveParentSnapshot,
) -> InvalidationAction:
    if any(
        requirement.stage.value == "cleanup_base"
        and requirement.state.value == "current"
        for requirement in parent.stage_requirements
    ):
        return InvalidationAction.USER_CURRENT
    return InvalidationAction.REBUILD


def _user_parent_split_invalidation(
    *,
    page_id: str,
    child_parent_ids: tuple[str, str],
    reason: str,
) -> InvalidationResult:
    """Return exact zero-unknown effects for two newly effective children."""

    if len(child_parent_ids) != 2 or len(set(child_parent_ids)) != 2:
        raise ValueError("split invalidation requires two unique child parents")
    effects = [
        _effect(
            Dependency.HIERARCHY,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
        _effect(
            Dependency.CLEANUP_BASE,
            InvalidationAction.REBUILD,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
        _effect(
            Dependency.STYLE_CACHE,
            InvalidationAction.RERUN,
            InvalidationScope.STYLE_CACHE_PREFIX,
            page_id,
            reason,
        ),
        _effect(
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
    ]
    for child_parent_id in child_parent_ids:
        effects.extend(
            (
                _effect(
                    Dependency.SOURCE,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    child_parent_id,
                    reason,
                ),
                _effect(
                    Dependency.TRANSLATION,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    child_parent_id,
                    reason,
                ),
                _effect(
                    Dependency.RENDER_ELIGIBILITY,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    child_parent_id,
                    reason,
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    child_parent_id,
                    reason,
                ),
            )
        )
    return _result(effects)


def _pipeline_parent_merge_invalidation(
    *,
    page_id: str,
    merged_parent_id: str,
    reason: str,
) -> InvalidationResult:
    """Return exact owner effects for one pipeline-backed merged parent."""

    return _result(
        (
            _effect(
                Dependency.HIERARCHY,
                InvalidationAction.NEW_REVISION,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.SOURCE,
                InvalidationAction.KEEP,
                InvalidationScope.PARENT,
                merged_parent_id,
                reason,
            ),
            _effect(
                Dependency.TRANSLATION,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                merged_parent_id,
                reason,
            ),
            _effect(
                Dependency.CLEANUP_BASE,
                InvalidationAction.REBUILD,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.STYLE_CACHE,
                InvalidationAction.RERUN,
                InvalidationScope.STYLE_CACHE_PREFIX,
                page_id,
                reason,
            ),
            _effect(
                Dependency.RENDER_ELIGIBILITY,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                merged_parent_id,
                reason,
            ),
            _effect(
                Dependency.LAYOUT_RENDER,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PARENT,
                merged_parent_id,
                reason,
            ),
            _effect(
                Dependency.PAGE_OUTPUT,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
        )
    )


def _pipeline_parent_merge_restore_invalidation(
    *,
    page_id: str,
    parents: tuple[EffectiveParentSnapshot, EffectiveParentSnapshot],
    reason: str,
) -> InvalidationResult:
    """Restore the two immutable pipeline parents' exact effective owner state."""

    if len(parents) != 2 or len({parent.parent_id for parent in parents}) != 2:
        raise ValueError("merge restore requires two unique source parents")
    effects = [
        _effect(
            Dependency.HIERARCHY,
            InvalidationAction.NEW_REVISION,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
        _effect(
            Dependency.CLEANUP_BASE,
            InvalidationAction.REBUILD,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
        _effect(
            Dependency.STYLE_CACHE,
            InvalidationAction.RERUN,
            InvalidationScope.STYLE_CACHE_PREFIX,
            page_id,
            reason,
        ),
        _effect(
            Dependency.PAGE_OUTPUT,
            InvalidationAction.RECOMPUTE,
            InvalidationScope.PAGE,
            page_id,
            reason,
        ),
    ]
    for parent in parents:
        effects.extend(
            (
                _effect(
                    Dependency.SOURCE,
                    _source_action_for_effective_parent(parent),
                    InvalidationScope.PARENT,
                    parent.parent_id,
                    reason,
                ),
                _effect(
                    Dependency.TRANSLATION,
                    _translation_action_for_effective_parent(parent),
                    InvalidationScope.PARENT,
                    parent.parent_id,
                    reason,
                ),
                _effect(
                    Dependency.RENDER_ELIGIBILITY,
                    InvalidationAction.RERUN,
                    InvalidationScope.PARENT,
                    parent.parent_id,
                    reason,
                ),
                _effect(
                    Dependency.LAYOUT_RENDER,
                    InvalidationAction.RECOMPUTE,
                    InvalidationScope.PARENT,
                    parent.parent_id,
                    reason,
                ),
            )
        )
    return _result(effects)


def _user_parent_split_restore_invalidation(
    *,
    page_id: str,
    parent: EffectiveParentSnapshot,
    reason: str,
) -> InvalidationResult:
    """Restore the original parent's exact effective owner state."""

    parent_id = parent.parent_id
    return _result(
        (
            _effect(
                Dependency.HIERARCHY,
                InvalidationAction.NEW_REVISION,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.SOURCE,
                _source_action_for_effective_parent(parent),
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.TRANSLATION,
                _translation_action_for_effective_parent(parent),
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.CLEANUP_BASE,
                _cleanup_action_for_effective_parent(parent),
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
            _effect(
                Dependency.STYLE_CACHE,
                InvalidationAction.RERUN,
                InvalidationScope.STYLE_CACHE_PREFIX,
                page_id,
                reason,
            ),
            _effect(
                Dependency.RENDER_ELIGIBILITY,
                InvalidationAction.RERUN,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.LAYOUT_RENDER,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PARENT,
                parent_id,
                reason,
            ),
            _effect(
                Dependency.PAGE_OUTPUT,
                InvalidationAction.RECOMPUTE,
                InvalidationScope.PAGE,
                page_id,
                reason,
            ),
        )
    )
