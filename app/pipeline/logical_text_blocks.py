# -*- coding: utf-8 -*-
"""Default logical text block assembly for TextAreaPlan-owned regions."""
from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import math
import re
from typing import Any, Mapping


LOGICAL_TEXT_BLOCK_VERSION = "speech_bubble_logical_text_blocks_v3_physical_source_conservation"

_OWNERSHIP_BLOCK_ANCHOR = "block_anchor"
_OWNERSHIP_TRANSFERRED_CHILD = "transferred_child"
_OWNERSHIP_DUPLICATE_CHILD = "duplicate_child"
_OWNERSHIP_PUNCTUATION_CHILD = "punctuation_child"
_OWNERSHIP_NOISE_REVIEW_ONLY = "noise_review_only"
_OWNERSHIP_STANDALONE_BLOCK = "standalone_block"

_ACTIVE_OWNERSHIP_STATUSES = {_OWNERSHIP_BLOCK_ANCHOR, _OWNERSHIP_STANDALONE_BLOCK}
_NON_RENDERABLE_OWNERSHIP_STATUSES = {
    _OWNERSHIP_TRANSFERRED_CHILD,
    _OWNERSHIP_DUPLICATE_CHILD,
    _OWNERSHIP_PUNCTUATION_CHILD,
    _OWNERSHIP_NOISE_REVIEW_ONLY,
}
_STRONG_PHYSICAL_TIERS = {"strong_model_container", "mask_primary_container"}
_BLOCKING_SOURCE_QUALITY_ACTIONS = {
    "block_auto_translation",
    "source_quality_blocked",
    "split_required",
    "unresolved_review",
}

_V3_STANDALONE_UTTERANCE = "standalone_utterance"
_V3_DEPENDENT_CHILD = "dependent_child"
_V3_BLOCKED_SFX_OR_DECORATIVE = "blocked_sfx_or_decorative"
_V3_REVIEW_ONLY_UNRESOLVED = "review_only_unresolved"


@dataclass
class PhysicalBubbleGroup:
    physical_bubble_id: str
    page_id: str
    member_container_ids: list[str]
    bbox: list[int]
    source: str = "TextAreaPlan"
    reason_codes: list[str] = field(default_factory=list)
    source_model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_text_physical_bubble_id": self.physical_bubble_id,
            "physical_bubble_graph_id": self.physical_bubble_id,
            "page_id": self.page_id,
            "logical_text_physical_bubble_member_container_ids": list(self.member_container_ids),
            "logical_text_physical_bubble_bbox": list(self.bbox),
            "logical_text_physical_bubble_source": self.source,
            "logical_text_physical_bubble_reason_codes": list(self.reason_codes),
            "logical_text_physical_bubble_source_model_ids": list(self.source_model_ids),
        }


@dataclass
class LogicalTextBlock:
    block_id: str
    page_id: str
    container_id: str
    role: str
    member_region_ids: list[str]
    anchor_region_id: str
    transferred_region_ids: list[str] = field(default_factory=list)
    duplicate_region_ids: list[str] = field(default_factory=list)
    punctuation_child_ids: list[str] = field(default_factory=list)
    noise_child_ids: list[str] = field(default_factory=list)
    source_text: str = ""
    reason_codes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    would_change_behavior: bool = False
    member_source_texts: dict[str, str] = field(default_factory=dict)
    anchor_original_text: str = ""
    bbox: list[int] | None = None
    allowed_bbox: list[int] | None = None
    text_conservation_status: str = "complete"
    ownership_status_by_region: dict[str, str] = field(default_factory=dict)
    physical_bubble_id: str | None = None
    physical_bubble_member_container_ids: list[str] = field(default_factory=list)
    physical_bubble_source: str = "TextAreaPlan"
    physical_bubble_reason_codes: list[str] = field(default_factory=list)
    source_quality_status: str = "clean"
    source_quality_reason_codes: list[str] = field(default_factory=list)
    source_quality_action: str = "translate"
    source_reconstruction_status: str = "not_attempted"
    source_reconstruction_applied: bool = False
    source_reconstruction_before_text: str = ""
    source_reconstruction_after_text: str = ""
    source_reconstruction_ocr_confidence: float | None = None
    source_reconstruction_crop_bbox: list[int] = field(default_factory=list)
    source_reconstruction_included_child_region_ids: list[str] = field(default_factory=list)
    source_reconstruction_rejected_child_region_ids: list[str] = field(default_factory=list)
    source_reconstruction_reason_codes: list[str] = field(default_factory=list)
    source_reconstruction_child_fragment_status: list[dict[str, Any]] = field(default_factory=list)
    source_reconstruction_required: bool = False
    source_reconstruction_unresolved_reason: str | None = None
    parent_visual_group_id: str = ""
    parent_visual_group_bbox: list[int] = field(default_factory=list)
    parent_visual_group_child_ids: list[str] = field(default_factory=list)
    reconstruction_rejected_for_visual_overmerge: bool = False

    def to_dict(self) -> dict[str, Any]:
        translation_unit = self.source_quality_action not in _BLOCKING_SOURCE_QUALITY_ACTIONS
        return {
            "logical_text_block_id": self.block_id,
            "page_id": self.page_id,
            "logical_text_block_container_id": self.container_id,
            "logical_text_block_role": self.role,
            "logical_text_block_member_region_ids": list(self.member_region_ids),
            "logical_text_block_anchor_region_id": self.anchor_region_id,
            "logical_text_block_transferred_region_ids": list(self.transferred_region_ids),
            "logical_text_block_duplicate_region_ids": list(self.duplicate_region_ids),
            "logical_text_block_punctuation_child_ids": list(self.punctuation_child_ids),
            "logical_text_block_noise_child_ids": list(self.noise_child_ids),
            "logical_text_block_source_text": self.source_text,
            "logical_text_block_reason_codes": list(self.reason_codes),
            "logical_text_block_confidence": self.confidence,
            "logical_text_block_would_change_behavior": self.would_change_behavior,
            "logical_text_block_member_source_texts": dict(self.member_source_texts),
            "logical_text_block_anchor_original_text": self.anchor_original_text,
            "logical_text_block_bbox": list(self.bbox or []),
            "logical_text_block_allowed_bbox": list(self.allowed_bbox or self.bbox or []),
            "logical_text_block_text_conservation_status": self.text_conservation_status,
            "logical_text_block_ownership_status_by_region": dict(self.ownership_status_by_region),
            "logical_text_physical_bubble_id": self.physical_bubble_id,
            "logical_text_physical_bubble_member_container_ids": list(self.physical_bubble_member_container_ids),
            "logical_text_physical_bubble_source": self.physical_bubble_source,
            "logical_text_physical_bubble_reason_codes": list(self.physical_bubble_reason_codes),
            "logical_text_source_quality_status": self.source_quality_status,
            "logical_text_source_quality_reason_codes": list(self.source_quality_reason_codes),
            "logical_text_source_quality_action": self.source_quality_action,
            "logical_text_source_reconstruction_status": self.source_reconstruction_status,
            "logical_text_source_reconstruction_applied": self.source_reconstruction_applied,
            "logical_text_source_reconstruction_before_text": self.source_reconstruction_before_text,
            "logical_text_source_reconstruction_after_text": self.source_reconstruction_after_text,
            "logical_text_source_reconstruction_ocr_confidence": self.source_reconstruction_ocr_confidence,
            "logical_text_source_reconstruction_crop_bbox": list(self.source_reconstruction_crop_bbox),
            "logical_text_source_reconstruction_included_child_region_ids": list(self.source_reconstruction_included_child_region_ids),
            "logical_text_source_reconstruction_rejected_child_region_ids": list(self.source_reconstruction_rejected_child_region_ids),
            "logical_text_source_reconstruction_reason_codes": list(self.source_reconstruction_reason_codes),
            "logical_text_source_reconstruction_child_fragment_status": list(self.source_reconstruction_child_fragment_status),
            "logical_text_parent_visual_group_id": self.parent_visual_group_id,
            "logical_text_parent_visual_group_bbox": list(self.parent_visual_group_bbox),
            "logical_text_parent_visual_group_child_ids": list(self.parent_visual_group_child_ids),
            "logical_text_reconstruction_rejected_for_visual_overmerge": self.reconstruction_rejected_for_visual_overmerge,
            "physical_bubble_graph_id": self.physical_bubble_id,
            "logical_text_block_v3_status": "translation_unit" if translation_unit else _V3_REVIEW_ONLY_UNRESOLVED,
            "logical_text_block_translation_unit": translation_unit,
            "logical_text_block_source_reconstruction_required": bool(self.source_reconstruction_required),
            "logical_text_block_source_reconstruction_status": self.source_reconstruction_status,
            "logical_text_block_source_reconstruction_crop_bbox": list(self.source_reconstruction_crop_bbox),
            "logical_text_block_included_child_region_ids": list(self.source_reconstruction_included_child_region_ids),
            "logical_text_block_rejected_child_region_ids": list(self.source_reconstruction_rejected_child_region_ids),
            "logical_text_block_unresolved_reason": self.source_reconstruction_unresolved_reason,
            "source_conservation_status": self.text_conservation_status,
            "source_conservation_failure_reason": self.source_reconstruction_unresolved_reason,
        }


@dataclass
class LogicalTextOwnershipRecord:
    page_id: str
    region_id: str
    container_id: str
    ownership_status: str
    block_id: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    source_text: str = ""
    would_change_behavior: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "region_id": self.region_id,
            "logical_text_block_container_id": self.container_id,
            "logical_text_block_id": self.block_id,
            "logical_text_ownership_status": self.ownership_status,
            "logical_text_block_reason_codes": list(self.reason_codes),
            "logical_text_block_source_text": self.source_text,
            "logical_text_block_would_change_behavior": self.would_change_behavior,
        }


@dataclass
class LogicalTextBlockAssemblyResult:
    version: str
    page_id: str
    generated: bool
    applied_count: int
    blocks: list[LogicalTextBlock]
    skipped_container_count: int = 0
    ownership_records: list[LogicalTextOwnershipRecord] = field(default_factory=list)
    speech_container_count: int = 0
    owned_region_count: int = 0
    unowned_meaningful_region_count: int = 0
    unowned_meaningful_region_ids: list[str] = field(default_factory=list)
    conservation_status: str = "complete"
    physical_bubble_groups: list[PhysicalBubbleGroup] = field(default_factory=list)
    render_eligibility_repairs: list[dict[str, Any]] = field(default_factory=list)
    speech_container_meaningful_fragment_count: int = 0
    speech_container_blocked_meaningful_fragment_count: int = 0
    speech_container_blocked_meaningful_region_ids: list[str] = field(default_factory=list)
    speech_container_source_survivor_region_ids: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_text_block_version": self.version,
            "page_id": self.page_id,
            "logical_text_block_generated": self.generated,
            "logical_text_block_applied_count": self.applied_count,
            "logical_text_blocks": [block.to_dict() for block in self.blocks],
            "logical_text_ownership_records": [record.to_dict() for record in self.ownership_records],
            "logical_text_block_skipped_container_count": self.skipped_container_count,
            "logical_text_block_speech_container_count": self.speech_container_count,
            "logical_text_block_owned_region_count": self.owned_region_count,
            "logical_text_block_unowned_meaningful_region_count": self.unowned_meaningful_region_count,
            "logical_text_block_unowned_meaningful_region_ids": list(self.unowned_meaningful_region_ids),
            "logical_text_block_conservation_status": self.conservation_status,
            "logical_text_physical_bubble_groups": [group.to_dict() for group in self.physical_bubble_groups],
            "logical_text_render_eligibility_repairs": list(self.render_eligibility_repairs),
            "logical_text_render_eligibility_repair_count": len(self.render_eligibility_repairs),
            "speech_container_meaningful_fragment_count": self.speech_container_meaningful_fragment_count,
            "speech_container_blocked_meaningful_fragment_count": self.speech_container_blocked_meaningful_fragment_count,
            "speech_container_blocked_meaningful_region_ids": list(self.speech_container_blocked_meaningful_region_ids),
            "speech_container_source_survivor_region_ids": list(self.speech_container_source_survivor_region_ids),
            "logical_text_block_error": self.error,
        }


def apply_same_container_logical_text_blocks(
    regions: list[dict[str, Any]],
    *,
    page_id: str,
    image_size: tuple[int, int] | None = None,
    text_area_plan: Any | None = None,
) -> LogicalTextBlockAssemblyResult:
    """Assign speech-container OCR regions to conservative logical text blocks."""
    try:
        groups = _eligible_speech_groups(regions)
        explicit_parent_nodes_by_container = _explicit_text_area_parent_nodes_by_container(text_area_plan)
        physical_groups, container_to_physical = _build_physical_bubble_groups(
            groups,
            page_id,
            image_size,
            text_area_plan=text_area_plan,
        )
        _augment_groups_with_text_area_owned_fragments(groups, regions, physical_groups)
        for group in physical_groups:
            _stamp_physical_bubble_group(regions, group)
        blocks: list[LogicalTextBlock] = []
        records: list[LogicalTextOwnershipRecord] = []
        skipped = 0
        applied = 0
        owned_region_ids: set[str] = set()
        unowned_meaningful_region_ids: set[str] = set()
        cross_container_components = _cross_container_assembly_components(physical_groups, groups)
        cross_container_ids = {cid for _group, ids in cross_container_components for cid in ids}
        for physical_group, container_ids in cross_container_components:
            members = sorted(
                [region for cid in container_ids for region in groups.get(cid, [])],
                key=_manga_region_sort_key,
            )
            if not members:
                continue
            container_blocks, container_records, unowned_ids = _build_container_blocks(
                page_id,
                physical_group.physical_bubble_id,
                members,
                image_size,
                physical_group,
                explicit_parent_nodes=[],
            )
            member_by_id = {str(region.get("region_id") or ""): region for region in members}
            for block in container_blocks:
                _apply_block_to_regions(block, member_by_id)
                blocks.append(block)
                if block.would_change_behavior:
                    applied += 1
                owned_region_ids.update(block.member_region_ids)
            for record in container_records:
                records.append(record)
                if record.region_id:
                    owned_region_ids.add(record.region_id)
                    _stamp_record_only(member_by_id.get(record.region_id), record)
            unowned_meaningful_region_ids.update(unowned_ids)
        for container_id, members in sorted(groups.items()):
            if container_id in cross_container_ids:
                continue
            members = sorted(members, key=_manga_region_sort_key)
            container_blocks, container_records, unowned_ids = _build_container_blocks(
                page_id,
                container_id,
                members,
                image_size,
                container_to_physical.get(container_id),
                explicit_parent_nodes=explicit_parent_nodes_by_container.get(container_id, []),
            )
            if not container_blocks and len(members) > 1:
                skipped += 1
            member_by_id = {str(region.get("region_id") or ""): region for region in members}
            for block in container_blocks:
                _apply_block_to_regions(block, member_by_id)
                blocks.append(block)
                if block.would_change_behavior:
                    applied += 1
                owned_region_ids.update(block.member_region_ids)
            for record in container_records:
                records.append(record)
                if record.region_id:
                    owned_region_ids.add(record.region_id)
                    _stamp_record_only(member_by_id.get(record.region_id), record)
            unowned_meaningful_region_ids.update(unowned_ids)
        repairs = _repair_render_eligibility(regions)
        final_unowned = _final_unowned_meaningful_speech_regions(regions)
        unowned_meaningful_region_ids.update(final_unowned)
        speech_metrics = _speech_conservation_metrics(regions)
        conservation_status = _page_conservation_status(unowned_meaningful_region_ids, speech_metrics)
        return LogicalTextBlockAssemblyResult(
            version=LOGICAL_TEXT_BLOCK_VERSION,
            page_id=page_id,
            generated=True,
            applied_count=applied,
            blocks=blocks,
            skipped_container_count=skipped,
            ownership_records=records,
            speech_container_count=len(groups),
            owned_region_count=len(owned_region_ids),
            unowned_meaningful_region_count=len(unowned_meaningful_region_ids),
            unowned_meaningful_region_ids=sorted(unowned_meaningful_region_ids),
            conservation_status=conservation_status,
            physical_bubble_groups=physical_groups,
            render_eligibility_repairs=repairs,
            speech_container_meaningful_fragment_count=speech_metrics["meaningful_count"],
            speech_container_blocked_meaningful_fragment_count=len(speech_metrics["blocked_ids"]),
            speech_container_blocked_meaningful_region_ids=speech_metrics["blocked_ids"],
            speech_container_source_survivor_region_ids=speech_metrics["source_survivor_ids"],
        )
    except Exception as exc:
        return LogicalTextBlockAssemblyResult(
            version=LOGICAL_TEXT_BLOCK_VERSION,
            page_id=page_id,
            generated=False,
            applied_count=0,
            blocks=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def enforce_logical_text_render_eligibility(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Clear stale render state after translation planning marks skipped regions."""
    repairs = _repair_render_eligibility(regions)
    unowned = sorted(_final_unowned_meaningful_speech_regions(regions))
    speech_metrics = _speech_conservation_metrics(regions)
    conservation_status = _page_conservation_status(set(unowned), speech_metrics)
    return {
        "logical_text_render_eligibility_repairs": repairs,
        "logical_text_render_eligibility_repair_count": len(repairs),
        "logical_text_block_unowned_meaningful_region_ids": unowned,
        "logical_text_block_unowned_meaningful_region_count": len(unowned),
        "logical_text_block_conservation_status": conservation_status,
        "speech_container_meaningful_fragment_count": speech_metrics["meaningful_count"],
        "speech_container_blocked_meaningful_fragment_count": len(speech_metrics["blocked_ids"]),
        "speech_container_blocked_meaningful_region_ids": speech_metrics["blocked_ids"],
        "speech_container_source_survivor_region_ids": speech_metrics["source_survivor_ids"],
    }


def assess_logical_text_source_quality(
    source_text: str,
    members: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str], str]:
    """Public wrapper for consumers that may repair a logical block source."""
    return _source_quality_assessment(source_text, members or [])


def restore_text_area_owned_speech_fragments(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Restore meaningful short text inside strong TextAreaPlan speech containers.

    This runs before LogicalTextBlock assembly so a prior OCR semantic label such
    as decorative_text cannot hide speech-owned fragments from the block graph.
    """
    restored: list[str] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        flags = region.get("flags") or {}
        suppressed = bool(flags.get("ignore") or str(region.get("skip_reason") or "").strip() or _is_hard_excluded(region))
        if not suppressed:
            continue
        if not _text_area_speech_overrides_late_fragment_preserve(region):
            continue
        flags = region.setdefault("flags", {})
        flags["ignore"] = False
        flags["bg_text"] = False
        flags["needs_review"] = False
        region["type"] = "speech_bubble"
        region["semantic_class"] = "speech_bubble"
        region["skip_reason"] = ""
        region["logical_text_speech_container_override_applied"] = True
        _append_region_reason(region, "text_area_speech_container_override")
        render = region.setdefault("render", {})
        render["semantic_class"] = "speech_bubble"
        render["cleanup_mode"] = "bubble"
        render["classification_reason"] = "text_area_speech_container_override"
        render["logical_text_speech_container_override_applied"] = True
        restored.append(rid)
    return {
        "logical_text_speech_container_override_count": len(restored),
        "logical_text_speech_container_override_region_ids": restored,
    }


def _eligible_speech_groups(regions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    assigned_region_ids: set[str] = set()
    for region in regions:
        if not _is_speech_container_region(region):
            continue
        container_id = str(_text_area_value(region, "text_area_container_id") or "").strip()
        if not container_id:
            continue
        groups.setdefault(container_id, []).append(region)
        rid = str(region.get("region_id") or "")
        if rid:
            assigned_region_ids.add(rid)
    _attach_nearby_dependent_speech_fragments(groups, regions, assigned_region_ids)
    return groups


def _augment_groups_with_text_area_owned_fragments(
    groups: dict[str, list[dict[str, Any]]],
    regions: list[dict[str, Any]],
    physical_groups: list[PhysicalBubbleGroup],
) -> None:
    if not physical_groups:
        return
    group_by_container = {
        container_id: physical_group
        for physical_group in physical_groups
        for container_id in physical_group.member_container_ids
    }
    plan_container_ids = {
        container_id
        for physical_group in physical_groups
        for container_id in physical_group.member_container_ids
    }
    assigned_region_ids = {
        str(region.get("region_id") or "")
        for members in groups.values()
        for region in members
        if str(region.get("region_id") or "")
    }
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or rid in assigned_region_ids:
            continue
        container_id = str(_text_area_value(region, "text_area_container_id") or "").strip()
        if not container_id or container_id not in plan_container_ids:
            continue
        if not _is_text_area_speech_container(region):
            continue
        if any(str(flag).strip() for flag in _text_area_value(region, "text_area_conflict_flags") or []):
            continue
        physical_group = group_by_container.get(container_id)
        if _is_hard_excluded(region) and not (
            _text_area_speech_overrides_late_fragment_preserve(region)
            or _physical_group_speech_fragment_override(region, physical_group)
        ):
            continue
        groups.setdefault(container_id, []).append(region)
        assigned_region_ids.add(rid)


def _build_physical_bubble_groups(
    groups: dict[str, list[dict[str, Any]]],
    page_id: str,
    image_size: tuple[int, int] | None,
    *,
    text_area_plan: Any | None = None,
) -> tuple[list[PhysicalBubbleGroup], dict[str, PhysicalBubbleGroup]]:
    if not groups and not text_area_plan:
        return [], {}
    summaries = _plan_speech_container_summaries(text_area_plan, image_size)
    for container_id, members in groups.items():
        if not members:
            continue
        summary = summaries.setdefault(container_id, {})
        summary.setdefault("bbox", _container_owner_bbox(members, image_size))
        summary.setdefault(
            "tiers",
            {
                str(_text_area_value(region, "text_area_confidence_tier") or "").strip()
                for region in members
            },
        )
        summary.setdefault(
            "conflicts",
            [
                flag
                for region in members
                for flag in (_text_area_value(region, "text_area_conflict_flags") or [])
                if str(flag).strip()
            ],
        )
        summary.setdefault("source_model_ids", _source_model_ids_from_regions(members))

    links: dict[str, set[str]] = {container_id: set() for container_id in summaries}
    container_ids = sorted(summaries)
    for idx, left in enumerate(container_ids):
        for right in container_ids[idx + 1 :]:
            related, _reason = _containers_share_physical_bubble(summaries[left], summaries[right])
            if related:
                links[left].add(right)
                links[right].add(left)

    physical_groups: list[PhysicalBubbleGroup] = []
    container_to_group: dict[str, PhysicalBubbleGroup] = {}
    seen: set[str] = set()
    for container_id in container_ids:
        if container_id in seen:
            continue
        stack = [container_id]
        component: list[str] = []
        seen.add(container_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for linked in sorted(links[current]):
                if linked not in seen:
                    seen.add(linked)
                    stack.append(linked)
        component = sorted(component)
        member_regions = [region for cid in component for region in groups.get(cid, [])]
        component_bbox = _union_bboxes(
            [summaries[cid]["bbox"] for cid in component if cid in summaries],
            image_size,
        )
        reason_codes = ["physical_bubble_owner:speech_text_area_container"]
        if len(component) > 1:
            reason_codes.append("physical_bubble_owner:close_or_overlapping_speech_containers")
            if any(bool(summaries.get(cid, {}).get("speech_context_only")) for cid in component):
                reason_codes.append("physical_bubble_owner:ogkalu_bubble_context_for_text_bubble")
        else:
            reason_codes.append("physical_bubble_owner:single_speech_container")
        source_model_ids = sorted(
            {
                model_id
                for cid in component
                for model_id in (summaries.get(cid, {}).get("source_model_ids") or [])
            }
        )
        physical_group = PhysicalBubbleGroup(
            physical_bubble_id=f"lpb_{page_id}_{len(physical_groups):03d}",
            page_id=page_id,
            member_container_ids=component,
            bbox=component_bbox or _union_region_bboxes(member_regions, image_size),
            source="TextAreaPlan" if text_area_plan else "region_text_area_assignment",
            reason_codes=reason_codes,
            source_model_ids=source_model_ids,
        )
        physical_groups.append(physical_group)
        for cid in component:
            container_to_group[cid] = physical_group
    return physical_groups, container_to_group


def _container_owner_bbox(members: list[dict[str, Any]], image_size: tuple[int, int] | None) -> list[int]:
    for region in members:
        bbox = _text_area_value(region, "text_area_container_bbox")
        if bbox:
            return _clip_bbox(_bbox(bbox), image_size)
    return _union_region_bboxes(members, image_size)


def _plan_speech_container_summaries(
    text_area_plan: Any | None,
    image_size: tuple[int, int] | None,
) -> dict[str, dict[str, Any]]:
    plan = _text_area_plan_to_dict(text_area_plan)
    summaries: dict[str, dict[str, Any]] = {}
    for container in plan.get("containers") or []:
        if not isinstance(container, dict):
            continue
        container_id = str(container.get("container_id") or "").strip()
        if not container_id:
            continue
        container_type = str(container.get("container_type") or "").strip()
        route_intent = str(container.get("route_intent") or "").strip()
        is_speech_authority = (
            container_type == "speech_bubble"
            and route_intent == "translate_speech"
        )
        is_speech_context = _is_ogkalu_bubble_speech_context(container)
        if not is_speech_authority and not is_speech_context:
            continue
        conflicts = [flag for flag in (container.get("conflict_flags") or []) if str(flag).strip()]
        if conflicts:
            continue
        tier = str(container.get("confidence_tier") or "").strip()
        role_evidence = container.get("semantic_role_evidence") if isinstance(container.get("semantic_role_evidence"), dict) else {}
        summaries[container_id] = {
            "bbox": _clip_bbox(_bbox(container.get("bbox")), image_size),
            "tiers": {tier} if tier else set(),
            "conflicts": conflicts,
            "source_model_ids": [str(v) for v in (container.get("source_model_ids") or []) if str(v).strip()],
            "neighboring_speech_context_ids": [
                str(v)
                for v in (role_evidence.get("neighboring_speech_context_ids") or [])
                if str(v).strip()
            ],
            "speech_authority": bool(is_speech_authority),
            "speech_context_only": bool(is_speech_context and not is_speech_authority),
        }
    return summaries


def _explicit_text_area_parent_nodes_by_container(text_area_plan: Any | None) -> dict[str, list[dict[str, Any]]]:
    plan = _text_area_plan_to_dict(text_area_plan)
    graph_plan = plan.get("root_parent_child_plan")
    if hasattr(graph_plan, "to_dict"):
        try:
            graph_plan = graph_plan.to_dict()
        except Exception:
            graph_plan = {}
    if not isinstance(graph_plan, Mapping):
        return {}
    by_container: dict[str, list[dict[str, Any]]] = {}
    for node in graph_plan.get("parent_nodes") or []:
        if hasattr(node, "to_dict"):
            try:
                node = node.to_dict()
            except Exception:
                continue
        if not isinstance(node, Mapping):
            continue
        if not bool(node.get("is_explicit_parent_obligation")):
            continue
        if str(node.get("parent_kind") or "").strip() not in {"", "speech"}:
            continue
        container_id = str(node.get("container_id") or "").strip()
        parent_node_id = str(node.get("parent_node_id") or "").strip()
        if not container_id or not parent_node_id:
            continue
        by_container.setdefault(container_id, []).append(dict(node))
    for nodes in by_container.values():
        nodes.sort(key=lambda node: _manga_bbox_sort_key(_bbox(node.get("bbox"))))
    return by_container


def _is_ogkalu_bubble_speech_context(container: dict[str, Any]) -> bool:
    """Return true for bubble-context boxes that can enlarge a speech root crop.

    This is physical context only. It does not grant translation or cleanup
    authority; an adjacent speech-authorized text_bubble still has to provide
    semantic ownership.
    """
    if str(container.get("route_intent") or "").strip() != "review_or_fallback":
        return False
    if str(container.get("cleanup_authorization") or "").strip() != "review_unknown_not_cleanup":
        return False
    role_evidence = container.get("semantic_role_evidence")
    if not isinstance(role_evidence, dict):
        return False
    classes = {str(v) for v in (role_evidence.get("ogkalu_class_names") or []) if str(v)}
    if "bubble" not in classes:
        return False
    candidate_states = {str(v) for v in (role_evidence.get("cleanup_candidate_states") or []) if str(v)}
    if "cleanup_translate_speech" not in candidate_states:
        return False
    source_ids = [str(v) for v in (container.get("source_model_ids") or []) if str(v).strip()]
    neighbor_ids = [str(v) for v in (role_evidence.get("neighboring_speech_context_ids") or []) if str(v).strip()]
    if not source_ids or not neighbor_ids:
        return False
    try:
        confidence = float(role_evidence.get("model_confidence") or container.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    if confidence < 0.90:
        return False
    return True


def _text_area_plan_to_dict(text_area_plan: Any | None) -> dict[str, Any]:
    if text_area_plan is None:
        return {}
    if isinstance(text_area_plan, dict):
        return text_area_plan
    if hasattr(text_area_plan, "to_dict"):
        try:
            value = text_area_plan.to_dict()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def _source_model_ids_from_regions(members: list[dict[str, Any]]) -> list[str]:
    model_ids: set[str] = set()
    for region in members:
        for key in ("text_area_source_model_ids", "diagnostic_bubble_source_model_ids"):
            value = _text_area_value(region, key)
            if isinstance(value, list):
                model_ids.update(str(v) for v in value if str(v).strip())
        reason_ids = _text_area_value(region, "source_model_ids")
        if isinstance(reason_ids, list):
            model_ids.update(str(v) for v in reason_ids if str(v).strip())
    return sorted(model_ids)


def _containers_share_physical_bubble(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[bool, str]:
    if left.get("conflicts") or right.get("conflicts"):
        return False, ""
    context_related, context_reason = _speech_context_links_authorized_container(left, right)
    if context_related:
        return True, context_reason
    left_tiers = set(left.get("tiers") or [])
    right_tiers = set(right.get("tiers") or [])
    if not (left_tiers & _STRONG_PHYSICAL_TIERS) or not (right_tiers & _STRONG_PHYSICAL_TIERS):
        return False, ""
    left_box = _bbox(left.get("bbox"))
    right_box = _bbox(right.get("bbox"))
    shared_model_ids = set(left.get("source_model_ids") or []) & set(right.get("source_model_ids") or [])
    if _overlap_ratio(left_box, right_box) >= 0.35:
        return True, "physical_bubble_bbox_overlap"
    if shared_model_ids and _overlap_ratio(left_box, right_box) >= 0.04:
        return True, "physical_bubble_shared_model_bbox_overlap"
    if (
        shared_model_ids
        and _vertical_overlap_fraction(left_box, right_box) >= 0.45
        and _horizontal_gap(left_box, right_box) <= 32
    ):
        return True, "physical_bubble_close_horizontal_split"
    if (
        shared_model_ids
        and _horizontal_overlap_fraction(left_box, right_box) >= 0.45
        and _vertical_gap(left_box, right_box) <= 28
    ):
        return True, "physical_bubble_close_vertical_split"
    return False, ""


def _speech_context_links_authorized_container(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[bool, str]:
    pairs = ((left, right), (right, left))
    for authority, context in pairs:
        if not bool(authority.get("speech_authority")):
            continue
        if not bool(context.get("speech_context_only")):
            continue
        authority_ids = {str(v) for v in (authority.get("source_model_ids") or []) if str(v)}
        context_ids = {str(v) for v in (context.get("source_model_ids") or []) if str(v)}
        authority_neighbors = {str(v) for v in (authority.get("neighboring_speech_context_ids") or []) if str(v)}
        context_neighbors = {str(v) for v in (context.get("neighboring_speech_context_ids") or []) if str(v)}
        if not ((authority_ids & context_neighbors) or (context_ids & authority_neighbors)):
            continue
        authority_box = _bbox(authority.get("bbox"))
        context_box = _bbox(context.get("bbox"))
        if _overlap_ratio(authority_box, context_box) >= 0.20:
            return True, "physical_bubble_ogkalu_bubble_context_for_text_bubble"
        if (
            _vertical_overlap_fraction(authority_box, context_box) >= 0.45
            and _horizontal_gap(authority_box, context_box) <= 32
        ):
            return True, "physical_bubble_ogkalu_bubble_context_for_text_bubble"
        if (
            _horizontal_overlap_fraction(authority_box, context_box) >= 0.45
            and _vertical_gap(authority_box, context_box) <= 28
        ):
            return True, "physical_bubble_ogkalu_bubble_context_for_text_bubble"
    return False, ""


def _stamp_physical_bubble_group(
    regions: list[dict[str, Any]],
    physical_group: PhysicalBubbleGroup,
) -> None:
    member_ids = set(physical_group.member_container_ids)
    for region in regions:
        if str(_text_area_value(region, "text_area_container_id") or "").strip() in member_ids:
            region["logical_text_physical_bubble_id"] = physical_group.physical_bubble_id
            region["logical_text_physical_bubble_member_container_ids"] = list(physical_group.member_container_ids)
            region["logical_text_physical_bubble_source"] = physical_group.source
            region["logical_text_physical_bubble_reason_codes"] = list(physical_group.reason_codes)
            render = region.setdefault("render", {})
            render["logical_text_physical_bubble_id"] = physical_group.physical_bubble_id
            render["logical_text_physical_bubble_member_container_ids"] = list(physical_group.member_container_ids)
            render["logical_text_physical_bubble_source"] = physical_group.source
            render["logical_text_physical_bubble_reason_codes"] = list(physical_group.reason_codes)


def _cross_container_assembly_components(
    physical_groups: list[PhysicalBubbleGroup],
    groups: dict[str, list[dict[str, Any]]],
) -> list[tuple[PhysicalBubbleGroup, list[str]]]:
    components: list[tuple[PhysicalBubbleGroup, list[str]]] = []
    for physical_group in physical_groups:
        container_ids = [cid for cid in physical_group.member_container_ids if groups.get(cid)]
        if len(container_ids) != 2:
            continue
        if not _physical_group_allows_cross_container_assembly(physical_group, groups):
            continue
        components.append((physical_group, container_ids))
    return components


def _physical_group_allows_cross_container_assembly(
    physical_group: PhysicalBubbleGroup,
    groups: dict[str, list[dict[str, Any]]],
) -> bool:
    if len(physical_group.member_container_ids) != 2:
        return False
    left_id, right_id = physical_group.member_container_ids
    left_members = groups.get(left_id) or []
    right_members = groups.get(right_id) or []
    if not left_members or not right_members:
        return False
    left_box = _container_owner_bbox(left_members, None)
    right_box = _container_owner_bbox(right_members, None)
    shared_model_ids = set(physical_group.source_model_ids or [])
    if not shared_model_ids:
        return False
    vertical_stack = (
        _horizontal_overlap_fraction(left_box, right_box) >= 0.62
        and _vertical_gap(left_box, right_box) <= 8
    )
    if not vertical_stack:
        return False
    all_members = left_members + right_members
    active_anchor_count = sum(1 for region in all_members if _is_anchor_worthy(region))
    fragmented_count = sum(1 for region in all_members if _is_dependent_fragment(region))
    return active_anchor_count >= 1 and fragmented_count >= 2


def _attach_nearby_dependent_speech_fragments(
    groups: dict[str, list[dict[str, Any]]],
    regions: list[dict[str, Any]],
    assigned_region_ids: set[str],
) -> None:
    if not groups:
        return
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or rid in assigned_region_ids:
            continue
        text = _clean_source_text(str(region.get("ocr_text") or ""))
        if not text or _is_hard_excluded(region) or not _is_dependent_fragment(region):
            continue
        best_container = _nearest_related_speech_container(region, groups)
        if not best_container:
            continue
        groups.setdefault(best_container, []).append(region)
        assigned_region_ids.add(rid)
        region.setdefault("logical_text_block_reason_codes", [])
        reason_codes = region["logical_text_block_reason_codes"]
        if isinstance(reason_codes, list) and "inferred_nearby_speech_container_for_dependent_fragment" not in reason_codes:
            reason_codes.append("inferred_nearby_speech_container_for_dependent_fragment")


def _nearest_related_speech_container(
    region: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
) -> str | None:
    best: tuple[float, str] | None = None
    for container_id, members in groups.items():
        for member in members:
            container_bbox = _union_region_bboxes([region, member], None)
            related, reason = _regions_related_for_block(region, member, container_bbox)
            if not related or not _relation_allows_component_edge(reason, region, member, container_bbox):
                continue
            distance = math.hypot(
                _center_x(_bbox(region.get("bbox"))) - _center_x(_bbox(member.get("bbox"))),
                _center_y(_bbox(region.get("bbox"))) - _center_y(_bbox(member.get("bbox"))),
            )
            if best is None or distance < best[0]:
                best = (distance, container_id)
    return best[1] if best else None


def _is_speech_container_region(region: dict[str, Any]) -> bool:
    if str(_text_area_value(region, "text_area_container_type") or "").strip() != "speech_bubble":
        return False
    if str(_text_area_value(region, "text_area_route_intent") or "").strip() != "translate_speech":
        return False
    if _text_area_value(region, "text_area_ocr_eligible") is False:
        return False
    if any(str(flag).strip() for flag in _text_area_value(region, "text_area_conflict_flags") or []):
        return False
    if not _is_hard_excluded(region):
        return True
    return _text_area_speech_overrides_late_fragment_preserve(region)


def _text_area_speech_overrides_late_fragment_preserve(region: dict[str, Any]) -> bool:
    if bool(region.get("root_transaction_cleared_unsafe_fragment")):
        return False
    tier = str(_text_area_value(region, "text_area_confidence_tier") or "").strip()
    if tier not in _STRONG_PHYSICAL_TIERS:
        return False
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip().lower()
    if semantic in {"caption", "narration_box"}:
        return False
    text = _clean_source_text(str(region.get("ocr_text") or ""))
    body = _source_body(text)
    if not body:
        return False
    if not _is_meaningful_bubble_internal_text(region, text):
        return False
    container_bbox = _bbox(_text_area_value(region, "text_area_container_bbox"))
    if _bbox_inside_ratio(_bbox(region.get("bbox")), container_bbox) < 0.70:
        return False
    reason_text = " ".join(str(v) for v in (_text_area_value(region, "text_area_reason_codes") or []))
    if "sfx" in reason_text.lower() or "decorative" in reason_text.lower():
        return False
    return True


def _physical_group_speech_fragment_override(
    region: dict[str, Any],
    physical_group: PhysicalBubbleGroup | None,
) -> bool:
    if bool(region.get("root_transaction_cleared_unsafe_fragment")):
        return False
    if physical_group is None:
        return False
    tier = str(_text_area_value(region, "text_area_confidence_tier") or "").strip()
    if tier not in _STRONG_PHYSICAL_TIERS and not physical_group.source_model_ids:
        return False
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip().lower()
    if semantic in {"caption", "narration_box"}:
        return False
    text = _clean_source_text(str(region.get("ocr_text") or ""))
    if not _is_meaningful_bubble_internal_text(region, text):
        return False
    reason_text = " ".join(str(v) for v in (_text_area_value(region, "text_area_reason_codes") or []))
    if "sfx" in reason_text.lower() or "decorative" in reason_text.lower():
        return False
    inside_ratio = _bbox_inside_ratio(_bbox(region.get("bbox")), _bbox(_text_area_value(region, "text_area_container_bbox")))
    if inside_ratio <= 0.0:
        inside_ratio = _bbox_inside_ratio(_bbox(region.get("bbox")), physical_group.bbox)
    return inside_ratio >= 0.60


def _is_meaningful_bubble_internal_text(region: dict[str, Any], text: str) -> bool:
    body = _source_body(text)
    if _has_kanji(body):
        return True
    if len(body) >= 2 and any("\u3040" <= ch <= "\u30ff" for ch in body):
        return True
    if _is_particle_dependent_fragment(region):
        return True
    return False


def _text_area_value(region: dict[str, Any], key: str) -> Any:
    if key in region and region.get(key) not in (None, ""):
        return region.get(key)
    render = region.get("render") or {}
    return render.get(key)


def _is_hard_excluded(region: dict[str, Any]) -> bool:
    if bool(region.get("root_transaction_cleared_unsafe_fragment")):
        return True
    flags = region.get("flags") or {}
    if flags.get("bg_text"):
        return True
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip().lower()
    if semantic in {"decorative_text", "sfx"}:
        return True
    render = region.get("render") or {}
    cleanup_mode = str(render.get("cleanup_mode") or "").strip().lower()
    if cleanup_mode == "preserve":
        return True
    reason = str(render.get("classification_reason") or "").strip().lower()
    blocked_tokens = (
        "sfx",
        "decorative",
        "nonbubble",
        "caption",
        "background",
        "without_visual_speech",
        "review_only",
    )
    return any(token in reason for token in blocked_tokens)


def _build_container_blocks(
    page_id: str,
    container_id: str,
    members: list[dict[str, Any]],
    image_size: tuple[int, int] | None,
    physical_group: PhysicalBubbleGroup | None,
    explicit_parent_nodes: list[dict[str, Any]] | None = None,
) -> tuple[list[LogicalTextBlock], list[LogicalTextOwnershipRecord], set[str]]:
    candidates = [region for region in members if _clean_source_text(str(region.get("ocr_text") or ""))]
    if not candidates:
        return [], [], set()
    container_bbox = _union_region_bboxes(candidates, image_size)
    explicit_parent_block = _single_explicit_parent_boundary_block_if_legacy_would_split(
        page_id,
        container_id,
        candidates,
        image_size,
        physical_group,
        explicit_parent_nodes or [],
    )
    if explicit_parent_block is not None:
        return [explicit_parent_block], [], set()
    components = _container_components(candidates, container_bbox)
    multi_component = len(components) > 1
    blocks: list[LogicalTextBlock] = []
    records: list[LogicalTextOwnershipRecord] = []
    unowned_meaningful: set[str] = set()
    for index, component in enumerate(components):
        component = sorted(component, key=lambda r: _logical_block_member_sort_key(r, physical_group))
        component_blocks, component_records, component_unowned = _build_component_block(
            page_id,
            container_id,
            component,
            container_bbox,
            image_size,
            physical_group,
            component_index=index if multi_component else None,
        )
        blocks.extend(component_blocks)
        records.extend(component_records)
        unowned_meaningful.update(component_unowned)
    return blocks, records, unowned_meaningful


def _single_explicit_parent_boundary_block_if_legacy_would_split(
    page_id: str,
    container_id: str,
    candidates: list[dict[str, Any]],
    image_size: tuple[int, int] | None,
    physical_group: PhysicalBubbleGroup | None,
    explicit_parent_nodes: list[dict[str, Any]],
) -> LogicalTextBlock | None:
    if len(explicit_parent_nodes) != 1 or len(candidates) < 2:
        return None
    parent_node = explicit_parent_nodes[0]
    parent_bbox = _clip_bbox(_bbox(parent_node.get("bbox")), image_size)
    if parent_bbox == [0, 0, 1, 1]:
        return None
    parent_members = [
        region
        for region in candidates
        if _region_within_explicit_parent_boundary(region, parent_bbox)
    ]
    if len(parent_members) < 2:
        return None
    source_members = [
        region
        for region in parent_members
        if _source_body(_clean_source_text(str(region.get("ocr_text") or "")))
    ]
    if not _members_form_single_visual_column(source_members, parent_bbox):
        return None
    parent_member_ids = {str(region.get("region_id") or "") for region in parent_members if str(region.get("region_id") or "")}
    candidate_ids = {str(region.get("region_id") or "") for region in candidates if str(region.get("region_id") or "")}
    if parent_member_ids != candidate_ids:
        return None
    if len(_container_components(parent_members, parent_bbox)) < 2:
        return None
    return _build_explicit_parent_boundary_block(
        page_id,
        container_id,
        parent_node,
        parent_members,
        parent_bbox,
        image_size,
        physical_group,
    )


def _region_within_explicit_parent_boundary(region: dict[str, Any], parent_bbox: list[int]) -> bool:
    region_bbox = _bbox(region.get("bbox"))
    if _bbox_inside_ratio(region_bbox, parent_bbox) >= 0.55:
        return True
    rx, ry, rw, rh = region_bbox
    px, py, pw, ph = parent_bbox
    center_x = rx + rw / 2.0
    center_y = ry + rh / 2.0
    return px <= center_x <= px + pw and py <= center_y <= py + ph


def _build_explicit_parent_boundary_block(
    page_id: str,
    container_id: str,
    parent_node: dict[str, Any],
    members: list[dict[str, Any]],
    parent_bbox: list[int],
    image_size: tuple[int, int] | None,
    physical_group: PhysicalBubbleGroup | None,
) -> LogicalTextBlock | None:
    source_texts = {
        str(region.get("region_id") or ""): _clean_source_text(str(region.get("ocr_text") or ""))
        for region in members
    }
    meaningful = [
        region
        for region in members
        if _source_body(source_texts.get(str(region.get("region_id") or ""), ""))
    ]
    if not meaningful:
        return None
    ordered = _explicit_parent_source_order(members, parent_bbox, physical_group)
    anchor = _explicit_parent_source_order(meaningful, parent_bbox, physical_group)[0]
    anchor_id = str(anchor.get("region_id") or "")
    parent_node_id = str(parent_node.get("parent_node_id") or "").strip()
    if not anchor_id or not parent_node_id:
        return None

    reason_codes = [
        "same_text_area_container",
        "logical_text_block_v3",
        "text_area_plan_explicit_parent_boundary",
        "legacy_component_split_overridden_by_explicit_parent",
        "source_evidence_attached_under_explicit_parent",
        "translation_unit:single_block",
    ]
    for reason in _component_relation_reasons(members, parent_bbox):
        reason_codes.append(reason)
    for reason in parent_node.get("reason_codes") or []:
        if str(reason).strip():
            reason_codes.append(str(reason))

    combined = ""
    transferred: list[str] = []
    duplicates: list[str] = []
    punctuation_child_ids: list[str] = []
    noise_child_ids: list[str] = []
    ownership_status: dict[str, str] = {anchor_id: _OWNERSHIP_BLOCK_ANCHOR}
    previous_source_region: dict[str, Any] | None = None

    for region in ordered:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        text = source_texts.get(rid, "")
        if rid != anchor_id and (not _source_body(text) or _is_punctuation_or_ellipsis_only(text)):
            punctuation_child_ids.append(rid)
            ownership_status[rid] = _OWNERSHIP_PUNCTUATION_CHILD
            reason_codes.append("punctuation_child_suppressed")
            continue
        merged, action = _merge_explicit_parent_source_fragment(
            combined,
            text,
            previous_source_region,
            region,
            parent_bbox,
        )
        if rid == anchor_id:
            combined = merged
            previous_source_region = region if _source_body(text) else previous_source_region
            if action == "overlap":
                reason_codes.append("anchor_source_overlap_merged")
            continue
        if action == "duplicate":
            duplicates.append(rid)
            ownership_status[rid] = _OWNERSHIP_DUPLICATE_CHILD
            reason_codes.append("duplicate_fragment_suppressed")
            continue
        if action == "same_column_append":
            reason_codes.append("same_column_source_fragment_joined_under_explicit_parent")
        elif action == "overlap":
            reason_codes.append("overlapping_source_fragment_merged")
        else:
            reason_codes.append("adjacent_source_fragment_joined")
        combined = merged
        previous_source_region = region if _source_body(text) else previous_source_region
        transferred.append(rid)
        ownership_status[rid] = _OWNERSHIP_TRANSFERRED_CHILD

    member_ids = [
        rid
        for rid in [str(region.get("region_id") or "") for region in ordered]
        if rid in ownership_status
    ]
    block_bbox = _clip_bbox(_bbox(parent_bbox), image_size)
    block = LogicalTextBlock(
        block_id=parent_node_id,
        page_id=page_id,
        container_id=container_id,
        role="speech_bubble",
        member_region_ids=member_ids,
        anchor_region_id=anchor_id,
        transferred_region_ids=transferred,
        duplicate_region_ids=duplicates,
        punctuation_child_ids=punctuation_child_ids,
        noise_child_ids=noise_child_ids,
        source_text=combined,
        reason_codes=sorted(set(reason_codes)),
        confidence=_block_confidence(members, transferred, duplicates, punctuation_child_ids, noise_child_ids),
        would_change_behavior=True,
        member_source_texts={rid: source_texts.get(rid, "") for rid in member_ids},
        anchor_original_text=source_texts.get(anchor_id, ""),
        bbox=block_bbox,
        allowed_bbox=block_bbox,
        text_conservation_status=_text_conservation_status(transferred, duplicates, punctuation_child_ids, noise_child_ids),
        ownership_status_by_region=ownership_status,
        physical_bubble_id=physical_group.physical_bubble_id if physical_group else None,
        physical_bubble_member_container_ids=list(physical_group.member_container_ids) if physical_group else [container_id],
        physical_bubble_source=physical_group.source if physical_group else "region_text_area_assignment",
        physical_bubble_reason_codes=list(physical_group.reason_codes) if physical_group else [],
    )
    _apply_source_quality_assessment(block, members)
    return block


def _merge_explicit_parent_source_fragment(
    current: str,
    new: str,
    previous_region: dict[str, Any] | None,
    region: dict[str, Any],
    parent_bbox: list[int],
) -> tuple[str, str]:
    current = _clean_source_text(current)
    new = _clean_source_text(new)
    if not new:
        return current, "duplicate"
    if not current:
        return new, "append"
    merged, action = _merge_source_fragment(current, new)
    if action in {"duplicate", "overlap"}:
        return merged, action
    if previous_region is not None and _same_visual_column_continuation(previous_region, region, parent_bbox):
        return current + new, "same_column_append"
    return merged, action


def _same_visual_column_continuation(
    previous_region: dict[str, Any],
    region: dict[str, Any],
    parent_bbox: list[int],
) -> bool:
    related, reason = _regions_related_for_block(previous_region, region, parent_bbox)
    return bool(related and reason == "same_column_continuation")


def _explicit_parent_source_order(
    members: list[dict[str, Any]],
    parent_bbox: list[int],
    physical_group: PhysicalBubbleGroup | None,
) -> list[dict[str, Any]]:
    source_members = [
        region
        for region in members
        if _source_body(_clean_source_text(str(region.get("ocr_text") or "")))
    ]
    if len(source_members) >= 2 and _members_form_single_visual_column(source_members, parent_bbox):
        return sorted(members, key=lambda region: (_bbox(region.get("bbox"))[1], _bbox(region.get("bbox"))[0]))
    return sorted(members, key=lambda region: _logical_block_member_sort_key(region, physical_group))


def _members_form_single_visual_column(members: list[dict[str, Any]], parent_bbox: list[int]) -> bool:
    if len(members) < 2:
        return False
    for index, left in enumerate(members):
        left_box = _bbox(left.get("bbox"))
        for right in members[index + 1 :]:
            right_box = _bbox(right.get("bbox"))
            if _horizontal_overlap_fraction(left_box, right_box) >= 0.35:
                continue
            if _horizontal_gap(left_box, right_box) <= max(8, int(0.08 * _bbox(parent_bbox)[2])):
                continue
            return False
    return True


def _container_components(members: list[dict[str, Any]], container_bbox: list[int]) -> list[list[dict[str, Any]]]:
    if len(members) < 2:
        return [members]
    links: dict[int, set[int]] = {idx: set() for idx in range(len(members))}
    for idx, region in enumerate(members):
        for other_idx in range(idx + 1, len(members)):
            related, _reason = _regions_related_for_block(region, members[other_idx], container_bbox)
            if related and _relation_allows_component_edge(_reason, region, members[other_idx], container_bbox):
                links[idx].add(other_idx)
                links[other_idx].add(idx)
    components: list[list[dict[str, Any]]] = []
    seen: set[int] = set()
    for idx in range(len(members)):
        if idx in seen:
            continue
        stack = [idx]
        component_indexes: list[int] = []
        seen.add(idx)
        while stack:
            current = stack.pop()
            component_indexes.append(current)
            for linked in sorted(links[current]):
                if linked not in seen:
                    seen.add(linked)
                    stack.append(linked)
        components.append([members[i] for i in sorted(component_indexes)])
    return components


def _build_component_block(
    page_id: str,
    container_id: str,
    members: list[dict[str, Any]],
    container_bbox: list[int],
    image_size: tuple[int, int] | None,
    physical_group: PhysicalBubbleGroup | None,
    component_index: int | None = None,
) -> tuple[list[LogicalTextBlock], list[LogicalTextOwnershipRecord], set[str]]:
    source_texts = {
        str(region.get("region_id") or ""): _clean_source_text(str(region.get("ocr_text") or ""))
        for region in members
    }
    meaningful = [region for region in members if _source_body(source_texts.get(str(region.get("region_id") or ""), ""))]
    punctuation = [region for region in members if region not in meaningful]
    records: list[LogicalTextOwnershipRecord] = []
    unowned: set[str] = set()
    if not meaningful:
        for region in punctuation:
            rid = str(region.get("region_id") or "")
            records.append(
                _ownership_record(
                    page_id,
                    rid,
                    container_id,
                    _OWNERSHIP_PUNCTUATION_CHILD,
                    None,
                    ["punctuation_only_no_speech_block"],
                    source_texts.get(rid, ""),
                    False,
                )
            )
        return [], records, unowned

    if len(meaningful) == 1 and not punctuation:
        anchor = meaningful[0]
        anchor_id = str(anchor.get("region_id") or "")
        anchor_text = source_texts.get(anchor_id, "")
        if _tiny_dependent_speech_fragment_without_anchor(anchor, anchor_text, container_bbox):
            records.append(
                _ownership_record(
                    page_id,
                    anchor_id,
                    container_id,
                    _OWNERSHIP_NOISE_REVIEW_ONLY,
                    None,
                    _tiny_dependent_fragment_reasons(anchor),
                    anchor_text,
                    False,
                )
            )
            return [], records, unowned
        block = _standalone_block(page_id, container_id, anchor, source_texts, physical_group, component_index)
        return [block], records, unowned

    if not _component_should_assemble(members, meaningful, container_bbox):
        standalone_blocks: list[LogicalTextBlock] = []
        for region in members:
            rid = str(region.get("region_id") or "")
            text = source_texts.get(rid, "")
            if region in punctuation:
                status = _OWNERSHIP_PUNCTUATION_CHILD
                reason = ["punctuation_only_not_assembled"]
            elif _tiny_dependent_speech_fragment_without_anchor(region, text, container_bbox):
                status = _OWNERSHIP_NOISE_REVIEW_ONLY
                reason = _tiny_dependent_fragment_reasons(region)
            elif _is_review_noise(region, text):
                status = _OWNERSHIP_NOISE_REVIEW_ONLY
                reason = ["small_uncertain_fragment_review_only"]
            else:
                block = _standalone_block(page_id, container_id, region, source_texts, physical_group, component_index=None)
                standalone_blocks.append(block)
                continue
            records.append(
                _ownership_record(page_id, rid, container_id, status, None, reason, text, False)
            )
        return standalone_blocks, records, unowned

    anchor = _choose_anchor(meaningful)
    anchor_id = str(anchor.get("region_id") or "")
    if not anchor_id:
        return [], records, unowned

    reason_codes = [
        "same_text_area_container",
        "container_local_ocr_graph",
        "logical_text_block_v3",
        "translation_unit:single_block",
    ]
    relation_reasons = _component_relation_reasons(members, container_bbox)
    reason_codes.extend(relation_reasons)

    ordered = sorted(members, key=lambda r: _logical_block_member_sort_key(r, physical_group))
    combined = ""
    transferred: list[str] = []
    duplicates: list[str] = []
    punctuation_child_ids: list[str] = []
    noise_child_ids: list[str] = []
    ownership_status: dict[str, str] = {anchor_id: _OWNERSHIP_BLOCK_ANCHOR}

    for region in ordered:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        text = source_texts.get(rid, "")
        if rid == anchor_id:
            combined, action = _merge_source_fragment(combined, text)
            if action == "overlap":
                reason_codes.append("anchor_source_overlap_merged")
            continue
        if region in punctuation or _is_punctuation_or_ellipsis_only(text):
            punctuation_child_ids.append(rid)
            ownership_status[rid] = _OWNERSHIP_PUNCTUATION_CHILD
            reason_codes.append("punctuation_child_suppressed")
            continue
        if _is_review_noise(region, text):
            noise_child_ids.append(rid)
            ownership_status[rid] = _OWNERSHIP_NOISE_REVIEW_ONLY
            reason_codes.append("noise_fragment_review_only")
            continue
        merged, action = _merge_source_fragment(combined, text)
        if action == "duplicate":
            duplicates.append(rid)
            ownership_status[rid] = _OWNERSHIP_DUPLICATE_CHILD
            reason_codes.append("duplicate_fragment_suppressed")
            continue
        if action == "overlap":
            reason_codes.append("overlapping_source_fragment_merged")
        else:
            reason_codes.append("adjacent_source_fragment_joined")
        combined = merged
        transferred.append(rid)
        ownership_status[rid] = _OWNERSHIP_TRANSFERRED_CHILD

    if not transferred and not duplicates and not punctuation_child_ids and not noise_child_ids:
        block = _standalone_block(page_id, container_id, anchor, source_texts, physical_group, component_index)
        return [block], records, unowned

    block_bbox = _union_region_bboxes(
        [region for region in members if str(region.get("region_id") or "") in ownership_status],
        image_size,
    )
    block_id = f"ltb_{page_id}_{container_id}"
    if component_index is not None:
        block_id = f"{block_id}_c{component_index:02d}"
    member_ids = [rid for rid in [str(region.get("region_id") or "") for region in ordered] if rid in ownership_status]
    block = LogicalTextBlock(
        block_id=block_id,
        page_id=page_id,
        container_id=container_id,
        role="speech_bubble",
        member_region_ids=member_ids,
        anchor_region_id=anchor_id,
        transferred_region_ids=transferred,
        duplicate_region_ids=duplicates,
        punctuation_child_ids=punctuation_child_ids,
        noise_child_ids=noise_child_ids,
        source_text=combined,
        reason_codes=sorted(set(reason_codes)),
        confidence=_block_confidence(members, transferred, duplicates, punctuation_child_ids, noise_child_ids),
        would_change_behavior=True,
        member_source_texts={rid: source_texts.get(rid, "") for rid in member_ids},
        anchor_original_text=source_texts.get(anchor_id, ""),
        bbox=block_bbox,
        allowed_bbox=block_bbox,
        text_conservation_status=_text_conservation_status(transferred, duplicates, punctuation_child_ids, noise_child_ids),
        ownership_status_by_region=ownership_status,
        physical_bubble_id=physical_group.physical_bubble_id if physical_group else None,
        physical_bubble_member_container_ids=list(physical_group.member_container_ids) if physical_group else [container_id],
        physical_bubble_source=physical_group.source if physical_group else "region_text_area_assignment",
        physical_bubble_reason_codes=list(physical_group.reason_codes) if physical_group else [],
    )
    _apply_source_quality_assessment(block, members)
    return [block], records, unowned


def _standalone_block(
    page_id: str,
    container_id: str,
    anchor: dict[str, Any],
    source_texts: dict[str, str],
    physical_group: PhysicalBubbleGroup | None,
    component_index: int | None,
) -> LogicalTextBlock:
    rid = str(anchor.get("region_id") or "")
    block_id = f"ltb_{page_id}_{container_id}_{rid}"
    if component_index is not None:
        block_id = f"ltb_{page_id}_{container_id}_c{component_index:02d}_{rid}"
    text = source_texts.get(rid, "")
    bbox = _bbox(anchor.get("bbox"))
    source_quality_status, source_quality_reasons, source_quality_action = _source_quality_assessment(text, [anchor])
    return LogicalTextBlock(
        block_id=block_id,
        page_id=page_id,
        container_id=container_id,
        role="speech_bubble",
        member_region_ids=[rid],
        anchor_region_id=rid,
        source_text=text,
        reason_codes=["same_text_area_container", "logical_text_block_v3", "standalone_utterance"],
        confidence=_region_confidence(anchor),
        would_change_behavior=False,
        member_source_texts={rid: text},
        anchor_original_text=text,
        bbox=bbox,
        allowed_bbox=bbox,
        text_conservation_status="complete",
        ownership_status_by_region={rid: _OWNERSHIP_STANDALONE_BLOCK},
        physical_bubble_id=physical_group.physical_bubble_id if physical_group else None,
        physical_bubble_member_container_ids=list(physical_group.member_container_ids) if physical_group else [container_id],
        physical_bubble_source=physical_group.source if physical_group else "region_text_area_assignment",
        physical_bubble_reason_codes=list(physical_group.reason_codes) if physical_group else [],
        source_quality_status=source_quality_status,
        source_quality_reason_codes=source_quality_reasons,
        source_quality_action=source_quality_action,
    )


def _component_should_assemble(
    members: list[dict[str, Any]],
    meaningful: list[dict[str, Any]],
    container_bbox: list[int],
) -> bool:
    if len(members) < 2:
        return False
    if any(
        _source_fragments_overlap(
            _source_body(str(a.get("ocr_text") or "")),
            _source_body(str(b.get("ocr_text") or "")),
        )
        for idx, a in enumerate(meaningful)
        for b in meaningful[idx + 1 :]
    ):
        return True
    if not any(_is_anchor_worthy(region) for region in meaningful):
        return False
    relation_reasons = set(_component_relation_reasons(members, container_bbox))
    safe_reasons = {
        "source_overlap_or_duplicate",
        "bbox_overlap_or_containment",
        "dependent_fragment_near_anchor",
        "punctuation_child_near_anchor",
    }
    if relation_reasons & safe_reasons:
        return True
    if len(meaningful) == 1:
        anchor = meaningful[0]
        return any(
            _relation_allows_component_edge(_regions_related_for_block(anchor, region, container_bbox)[1], anchor, region, container_bbox)
            for region in members
            if region is not anchor
        )
    return False


def _apply_source_quality_assessment(block: LogicalTextBlock, members: list[dict[str, Any]]) -> None:
    status, reasons, action = _source_quality_assessment(block.source_text, members)
    block.source_quality_status = status
    block.source_quality_reason_codes = reasons
    block.source_quality_action = action
    if status != "clean":
        block.reason_codes = sorted(set(list(block.reason_codes) + reasons))


def _source_quality_assessment(
    source_text: str,
    members: list[dict[str, Any]],
) -> tuple[str, list[str], str]:
    cleaned = _clean_source_text(source_text)
    compact = _source_body(cleaned)
    reasons: list[str] = []
    speech_members = any(
        str(region.get("type") or region.get("semantic_class") or "").strip() == "speech_bubble"
        or str((region.get("render") or {}).get("text_area_route_intent") or region.get("text_area_route_intent") or "").strip() == "translate_speech"
        for region in members
        if isinstance(region, dict)
    )
    if not compact:
        if speech_members and _is_ellipsis_like_source(cleaned):
            return "fragmented", ["punctuation_only_speech_source"], "translate"
        return "empty", ["empty_logical_text_source"], "unresolved_review"
    known_bad_patterns = (
        "それまで女救出",
        "といいハ風の吹き回しだ果長自ら課長自",
        "折角だー無こも",
        "折角だ無こも",
        "ただキャ涼んで",
        "単に視界が悪いって悪かだけで",
        "嵐最初は遭難したと思",
    )
    if any(pattern in compact for pattern in known_bad_patterns):
        reasons.append("known_malformed_ocr_anchor_pattern")
    if (
        "果長" in cleaned
        or "無、こも" in cleaned
        or "それまで女" in cleaned
        or "悪かだけ" in cleaned
        or "ただキャ、" in cleaned
        or "ただキャ," in cleaned
        or "さるけど" in cleaned
        or "やんし" in cleaned
    ):
        reasons.append("malformed_ocr_anchor_surface")
    if cleaned.count("「") != cleaned.count("」") or cleaned.count("『") != cleaned.count("』"):
        reasons.append("unbalanced_quote_in_logical_source")
    if compact.endswith(("と思", "だと思", "悪かだけで")):
        reasons.append("incomplete_trailing_grammar")
    elif len(compact) <= 4 and compact.endswith(("で", "と", "を", "に", "の", "は", "が")):
        reasons.append("incomplete_trailing_grammar")
    if reasons:
        if speech_members and _speech_source_quality_reasons_allow_translation(cleaned, reasons):
            return "fragmented", sorted(set(reasons + ["speech_short_source_requires_root_proof"])), "translate"
        return "contaminated", sorted(set(reasons)), "source_quality_blocked"

    fragments = [
        part.strip()
        for part in re.split(r"[、，,]+", cleaned)
        if _source_body(part.strip())
    ]
    short_fragments = [part for part in fragments if 0 < len(_source_body(part)) <= 2]
    member_body_count = sum(1 for region in members if _source_body(str(region.get("ocr_text") or "")))
    separator_count = cleaned.count("、") + cleaned.count("，") + cleaned.count(",")
    unusual_kanji = any(token in cleaned for token in ("牢", "返を", "果長", "女、救出", "救出来"))
    orphan_particles = {"と", "で", "に", "を", "が", "は", "の", "も", "し"}
    if (
        (separator_count >= 4 and len(short_fragments) >= 3 and len(compact) >= 8)
        or (unusual_kanji and member_body_count >= 3)
        or (separator_count >= 2 and any(_source_body(fragment) in orphan_particles for fragment in fragments))
    ):
        reasons.append("fragmented_physical_bubble_source")
        if short_fragments:
            reasons.append("many_short_ocr_fragments")
        if unusual_kanji:
            reasons.append("suspect_ocr_substitution_surface")
        if any(_source_body(fragment) in orphan_particles for fragment in fragments):
            reasons.append("orphan_particle_fragment")
        if speech_members and _speech_source_quality_reasons_allow_translation(cleaned, reasons):
            reasons.append("speech_short_source_requires_root_proof")
        return "fragmented", sorted(set(reasons)), "translate"

    return "clean", [], "translate"


def _is_ellipsis_like_source(text: str) -> bool:
    stripped = "".join(ch for ch in str(text or "") if ch.strip())
    if not stripped:
        return False
    ellipsis_chars = ".．…‥・･"
    allowed_chars = ellipsis_chars + "—―－-ー〜～?？!！"
    return any(ch in ellipsis_chars for ch in stripped) and all(ch in allowed_chars for ch in stripped)


def _is_kana_char(ch: str) -> bool:
    return "\u3040" <= ch <= "\u30ff"


def _valid_short_speech_utterance(text: str) -> bool:
    cleaned = _clean_source_text(text)
    body = _source_body(cleaned)
    if _is_ellipsis_like_source(cleaned):
        return True
    if not body or len(body) > 10:
        return False
    if not any(_is_kana_char(ch) or "\u3400" <= ch <= "\u9fff" for ch in body):
        return False
    if any(token in cleaned for token in ("果長", "救出来", "悪かだけ", "無、こも", "それまで女", "返を")):
        return False
    if len(body) <= 2 and all(_is_kana_char(ch) for ch in body):
        return True
    return any(ch in cleaned for ch in ".．…‥・･〜～ー-—―－")


def _speech_source_quality_reasons_allow_translation(text: str, reasons: list[str]) -> bool:
    reason_set = set(reasons or [])
    blocking = {
        "known_malformed_ocr_anchor_pattern",
        "malformed_ocr_anchor_surface",
        "unbalanced_quote_in_logical_source",
        "suspect_ocr_substitution_surface",
    }
    if reason_set & blocking:
        return False
    allowed = {
        "incomplete_trailing_grammar",
        "fragmented_physical_bubble_source",
        "many_short_ocr_fragments",
        "orphan_particle_fragment",
    }
    if not reason_set <= allowed:
        return False
    fragments = [_clean_source_text(part) for part in re.split(r"[、，,]+", text) if _source_body(part)]
    orphan_particles = {"と", "で", "に", "を", "が", "は", "の", "も", "し"}
    useful = [fragment for fragment in fragments if _source_body(fragment) not in orphan_particles]
    return any(_valid_short_speech_utterance(fragment) for fragment in useful) or _valid_short_speech_utterance(text)


def _choose_anchor(meaningful: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_worthy = [region for region in meaningful if _is_anchor_worthy(region)]
    if not anchor_worthy:
        anchor_worthy = meaningful
    # Prefer manga reading order, but do not let tiny dependent fragments own a block.
    return sorted(anchor_worthy, key=_manga_region_sort_key)[0]


def _component_relation_reasons(members: list[dict[str, Any]], container_bbox: list[int]) -> list[str]:
    reasons: set[str] = set()
    for idx, region in enumerate(members):
        for other in members[idx + 1 :]:
            related, reason = _regions_related_for_block(region, other, container_bbox)
            if related and reason:
                reasons.add(reason)
    return sorted(reasons)


def _regions_related_for_block(
    a: dict[str, Any],
    b: dict[str, Any],
    container_bbox: list[int],
) -> tuple[bool, str]:
    a_text = _source_body(str(a.get("ocr_text") or ""))
    b_text = _source_body(str(b.get("ocr_text") or ""))
    if _source_fragments_overlap(a_text, b_text):
        return True, "source_overlap_or_duplicate"

    a_box = _bbox(a.get("bbox"))
    b_box = _bbox(b.get("bbox"))
    if _overlap_ratio(a_box, b_box) >= 0.05:
        return True, "bbox_overlap_or_containment"

    c_x, c_y, c_w, c_h = _bbox(container_bbox)
    _ = c_x, c_y
    vertical_overlap = _vertical_overlap_fraction(a_box, b_box)
    horizontal_overlap = _horizontal_overlap_fraction(a_box, b_box)
    horizontal_gap = _horizontal_gap(a_box, b_box)
    vertical_gap = _vertical_gap(a_box, b_box)
    center_dx = abs(_center_x(a_box) - _center_x(b_box))
    center_dy = abs(_center_y(a_box) - _center_y(b_box))
    container_diag = math.hypot(max(1, c_w), max(1, c_h))

    if (
        _is_punctuation_or_ellipsis_only(str(a.get("ocr_text") or ""))
        or _is_punctuation_or_ellipsis_only(str(b.get("ocr_text") or ""))
    ) and (
        (_is_anchor_worthy(a) or _is_anchor_worthy(b))
        and math.hypot(center_dx, center_dy) <= max(70, 0.34 * container_diag)
    ):
        return True, "punctuation_child_near_anchor"
    if (
        ((_is_dependent_fragment(a) and _is_anchor_worthy(b)) or (_is_dependent_fragment(b) and _is_anchor_worthy(a)))
        and math.hypot(center_dx, center_dy) <= max(96, 0.50 * container_diag)
        and (
            horizontal_gap <= max(60, int(0.35 * c_w))
            or vertical_gap <= max(60, int(0.18 * c_h))
            or vertical_overlap >= 0.20
            or horizontal_overlap >= 0.20
        )
    ):
        return True, "dependent_fragment_near_anchor"
    if vertical_overlap >= 0.35 and horizontal_gap <= max(36, min(112, int(0.45 * c_w))):
        return True, "adjacent_vertical_columns"
    if horizontal_overlap >= 0.35 and vertical_gap <= max(24, min(72, int(0.12 * c_h))):
        return True, "same_column_continuation"
    if (
        horizontal_gap <= max(32, min(128, int(0.45 * c_w)))
        and center_dy <= max(72, int(0.50 * c_h))
        and (_is_anchor_worthy(a) or _is_anchor_worthy(b) or (len(a_text) >= 2 and len(b_text) >= 2))
    ):
        return True, "staggered_bubble_columns"
    return False, ""


def _relation_allows_component_edge(
    reason: str,
    a: dict[str, Any],
    b: dict[str, Any],
    container_bbox: list[int],
) -> bool:
    if reason in {"source_overlap_or_duplicate", "bbox_overlap_or_containment"}:
        return True
    if reason == "punctuation_child_near_anchor":
        return True
    if reason == "dependent_fragment_near_anchor":
        return _dependent_edge_is_safe(a, b, container_bbox)
    return False


def _dependent_edge_is_safe(
    a: dict[str, Any],
    b: dict[str, Any],
    container_bbox: list[int],
) -> bool:
    dependent = (
        a
        if _is_dependent_fragment(a) and (_is_particle_dependent_fragment(a) or not _is_anchor_worthy(a))
        else b
        if _is_dependent_fragment(b) and (_is_particle_dependent_fragment(b) or not _is_anchor_worthy(b))
        else None
    )
    anchor = b if dependent is a else a if dependent is b else None
    if dependent is None or anchor is None or not _is_anchor_worthy(anchor):
        return False
    dep_box = _bbox(dependent.get("bbox"))
    anchor_box = _bbox(anchor.get("bbox"))
    _cx, _cy, cw, ch = _bbox(container_bbox)
    distance = math.hypot(_center_x(dep_box) - _center_x(anchor_box), _center_y(dep_box) - _center_y(anchor_box))
    return distance <= max(96, 0.50 * math.hypot(max(1, cw), max(1, ch)))


def _apply_block_to_regions(block: LogicalTextBlock, member_by_id: dict[str, dict[str, Any]]) -> None:
    anchor = member_by_id.get(block.anchor_region_id)
    if anchor is None:
        return
    if block.source_quality_action in _BLOCKING_SOURCE_QUALITY_ACTIONS:
        _mark_source_quality_block_review_only(block, member_by_id)
        return
    _stamp_region_fields(anchor, block, block.ownership_status_by_region.get(block.anchor_region_id, _OWNERSHIP_BLOCK_ANCHOR))
    _activate_block_anchor_region(anchor, block)
    if not block.would_change_behavior:
        return

    original_anchor_bbox = list(anchor.get("bbox") or [])
    original_anchor_text = str(anchor.get("ocr_text") or "")
    if block.bbox:
        anchor["bbox"] = list(block.bbox)
        anchor["polygon"] = _bbox_to_polygon(block.bbox)
    anchor["ocr_text"] = block.source_text
    anchor["translation"] = ""
    anchor["group_id"] = block.block_id
    render = anchor.setdefault("render", {})
    render["logical_text_block_original_anchor_bbox"] = original_anchor_bbox
    render["logical_text_block_original_anchor_text"] = original_anchor_text
    render["logical_text_block_source_text"] = block.source_text
    render["logical_text_block_reason_codes"] = list(block.reason_codes)
    render["logical_text_block_member_region_ids"] = list(block.member_region_ids)
    render["logical_text_block_transferred_region_ids"] = list(block.transferred_region_ids)
    render["logical_text_block_duplicate_region_ids"] = list(block.duplicate_region_ids)
    render["logical_text_block_punctuation_child_ids"] = list(block.punctuation_child_ids)
    render["logical_text_block_noise_child_ids"] = list(block.noise_child_ids)
    render["logical_text_block_text_conservation_status"] = block.text_conservation_status
    render["cleanup_mode"] = "bubble"

    for rid in (
        list(block.transferred_region_ids)
        + list(block.duplicate_region_ids)
        + list(block.punctuation_child_ids)
        + list(block.noise_child_ids)
    ):
        child = member_by_id.get(rid)
        if child is None:
            continue
        historical_translation = str(child.get("translation") or "")
        child["translation"] = ""
        child["group_id"] = block.block_id
        flags = child.setdefault("flags", {})
        flags["ignore"] = True
        flags["bg_text"] = False
        flags["needs_review"] = False
        child_render = child.setdefault("render", {})
        child_render["cleanup_mode"] = "transferred_to_logical_text_block_anchor"
        child_render["classification_reason"] = _child_classification_reason(block, rid)
        child_render["logical_text_block_transfer_to_region_id"] = block.anchor_region_id
        child_render["logical_text_block_transfer_text"] = str(child.get("ocr_text") or "")
        child_render["logical_text_block_historical_translation"] = historical_translation
        _stamp_region_fields(child, block, block.ownership_status_by_region.get(rid, _OWNERSHIP_TRANSFERRED_CHILD))


def _activate_block_anchor_region(region: dict[str, Any], block: LogicalTextBlock) -> None:
    flags = region.setdefault("flags", {})
    flags["ignore"] = False
    flags["bg_text"] = False
    flags["needs_review"] = False
    region["type"] = "speech_bubble"
    region["semantic_class"] = "speech_bubble"
    region["skip_reason"] = None
    render = region.setdefault("render", {})
    render["cleanup_mode"] = render.get("cleanup_mode") if render.get("cleanup_mode") not in {"preserve"} else "bubble"
    render["classification_reason"] = render.get("classification_reason") or "logical_text_block_active_speech_anchor"
    render["logical_text_source_quality_status"] = block.source_quality_status
    render["logical_text_source_quality_reason_codes"] = list(block.source_quality_reason_codes)
    render["logical_text_source_quality_action"] = block.source_quality_action


def _mark_source_quality_block_review_only(
    block: LogicalTextBlock,
    member_by_id: dict[str, dict[str, Any]],
) -> None:
    for rid in block.member_region_ids:
        region = member_by_id.get(rid)
        if region is None:
            continue
        _stamp_region_fields(region, block, _OWNERSHIP_NOISE_REVIEW_ONLY)
        region["logical_text_source_quality_status"] = block.source_quality_status
        region["logical_text_source_quality_reason_codes"] = list(block.source_quality_reason_codes)
        region["logical_text_source_quality_action"] = block.source_quality_action
        if _is_text_area_speech_container(region) and not any(str(flag).strip() for flag in _text_area_value(region, "text_area_conflict_flags") or []):
            region["speech_source_repair_required"] = True
            region["source_quality_blocked_visual_fail"] = True
            _append_region_reason(region, "speech_source_repair_required")
        flags = region.setdefault("flags", {})
        flags["ignore"] = True
        flags["bg_text"] = False
        flags["needs_review"] = True
        region["translation"] = ""
        region["translated_text"] = ""
        render = region.setdefault("render", {})
        render["cleanup_mode"] = "preserve"
        render["classification_reason"] = "logical_text_source_quality_blocked"
        render["logical_text_source_quality_status"] = block.source_quality_status
        render["logical_text_source_quality_reason_codes"] = list(block.source_quality_reason_codes)
        render["logical_text_source_quality_action"] = block.source_quality_action
        if region.get("speech_source_repair_required"):
            render["speech_source_repair_required"] = True
            render["source_quality_blocked_visual_fail"] = True
        _clear_renderable_state(region, "logical_text_source_quality_blocked")


def _repair_render_eligibility(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        status = str(region.get("logical_text_ownership_status") or "").strip()
        flags = region.setdefault("flags", {})
        ignored = bool(flags.get("ignore")) or str(region.get("skip_reason") or "") == "ignored_by_pipeline"
        if status in _ACTIVE_OWNERSHIP_STATUSES and _tiny_dependent_speech_fragment_without_anchor(
            region,
            _clean_source_text(str(region.get("ocr_text") or "")),
            _bbox(_text_area_value(region, "text_area_container_bbox")),
        ):
            status = _OWNERSHIP_NOISE_REVIEW_ONLY
            region["logical_text_ownership_status"] = status
            region["logical_text_blocked_fragment_resolution"] = "sfx_decorative_conflict"
            region["logical_text_block_would_change_behavior"] = False
            _append_region_reason(region, "tiny_dependent_speech_fragment_without_anchor")
            _append_region_reason(region, "text_area_preserve_conflict_blocks_tiny_fragment")
            ignored = True
            flags["ignore"] = True
            flags["bg_text"] = False
            flags["needs_review"] = True
            region.setdefault("render", {})["classification_reason"] = "logical_text_tiny_dependent_fragment_without_anchor"
        if not status and ignored and _is_text_area_speech_container(region):
            status = _OWNERSHIP_NOISE_REVIEW_ONLY
            region["logical_text_ownership_status"] = status
            region["logical_text_blocked_fragment_resolution"] = "source_quality_blocked"
            region["logical_text_block_would_change_behavior"] = False
            _append_region_reason(region, "ignored_speech_region_render_blocked")
            flags["ignore"] = True
            flags["bg_text"] = False
            flags["needs_review"] = True
            region.setdefault("render", {})["classification_reason"] = "logical_text_ignored_speech_region_render_blocked"
        elif not status and _is_speech_region_for_final_invariant(region) and _is_dependent_fragment(region):
            status = _OWNERSHIP_NOISE_REVIEW_ONLY
            region["logical_text_ownership_status"] = status
            region["logical_text_blocked_fragment_resolution"] = "source_quality_blocked"
            region["logical_text_block_would_change_behavior"] = False
            _append_region_reason(region, "unowned_dependent_speech_fragment_review_only")
            ignored = True
            flags["ignore"] = True
            flags["bg_text"] = False
            flags["needs_review"] = True
            region.setdefault("render", {})["classification_reason"] = "logical_text_unowned_dependent_fragment_review_only"
        elif not status and _tiny_dependent_speech_fragment_without_anchor(
            region,
            _clean_source_text(str(region.get("ocr_text") or "")),
            _bbox(_text_area_value(region, "text_area_container_bbox")),
        ):
            status = _OWNERSHIP_NOISE_REVIEW_ONLY
            region["logical_text_ownership_status"] = status
            region["logical_text_blocked_fragment_resolution"] = "sfx_decorative_conflict"
            region["logical_text_block_would_change_behavior"] = False
            _append_region_reason(region, "tiny_dependent_speech_fragment_without_anchor")
            _append_region_reason(region, "text_area_preserve_conflict_blocks_tiny_fragment")
            ignored = True
            flags["ignore"] = True
            flags["bg_text"] = False
            flags["needs_review"] = True
            region.setdefault("render", {})["classification_reason"] = "logical_text_tiny_dependent_fragment_without_anchor"
        if status in _NON_RENDERABLE_OWNERSHIP_STATUSES or (ignored and status not in _ACTIVE_OWNERSHIP_STATUSES):
            region["ocr_fragment_ownership_status"] = status
            region["ocr_fragment_final_state"] = _v3_ownership_state(status, region)
            region["logical_text_block_v3_status"] = region["ocr_fragment_final_state"]
            region["logical_text_block_translation_unit"] = False
            region["active_translation_unit_id"] = None
            if region["ocr_fragment_final_state"] not in {
                _V3_DEPENDENT_CHILD,
                _OWNERSHIP_DUPLICATE_CHILD,
                _OWNERSHIP_PUNCTUATION_CHILD,
            }:
                region["source_text_represented_by_block_id"] = None
            _copy_v3_fields_to_render(region, region.setdefault("render", {}))
            if not region.get("logical_text_blocked_fragment_resolution"):
                if _text_area_preserve_or_conflict_like(region):
                    region["logical_text_blocked_fragment_resolution"] = "sfx_decorative_conflict"
                elif status == _OWNERSHIP_DUPLICATE_CHILD:
                    region["logical_text_blocked_fragment_resolution"] = "duplicate_child"
                elif status == _OWNERSHIP_PUNCTUATION_CHILD:
                    region["logical_text_blocked_fragment_resolution"] = "punctuation_child"
                elif status == _OWNERSHIP_TRANSFERRED_CHILD:
                    region["logical_text_blocked_fragment_resolution"] = "represented_in_anchor"
                else:
                    region["logical_text_blocked_fragment_resolution"] = "source_quality_blocked"
                region.setdefault("render", {})["logical_text_blocked_fragment_resolution"] = region["logical_text_blocked_fragment_resolution"]
            flags["ignore"] = True
            flags["bg_text"] = False
            if status == _OWNERSHIP_NOISE_REVIEW_ONLY:
                flags["needs_review"] = True
            if _region_has_renderable_state(region):
                reason = (
                    "logical_text_non_renderable_child"
                    if status in _NON_RENDERABLE_OWNERSHIP_STATUSES
                    else "ignored_region_had_renderable_state"
                )
                _clear_renderable_state(region, reason)
                repairs.append(
                    {
                        "region_id": rid,
                        "logical_text_ownership_status": status or None,
                        "logical_text_render_eligibility_reason": reason,
                    }
                )
    return repairs


def _final_unowned_meaningful_speech_regions(regions: list[dict[str, Any]]) -> set[str]:
    unowned: set[str] = set()
    allowed = _ACTIVE_OWNERSHIP_STATUSES | _NON_RENDERABLE_OWNERSHIP_STATUSES
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or not _is_speech_region_for_final_invariant(region):
            continue
        text = _clean_source_text(str(region.get("ocr_text") or ""))
        if not _source_body(text):
            continue
        status = str(region.get("logical_text_ownership_status") or "").strip()
        if status in allowed:
            continue
        unowned.add(rid)
    return unowned


def _speech_conservation_metrics(regions: list[dict[str, Any]]) -> dict[str, Any]:
    meaningful_count = 0
    blocked_ids: list[str] = []
    source_survivor_ids: list[str] = []
    represented = _ACTIVE_OWNERSHIP_STATUSES | {
        _OWNERSHIP_TRANSFERRED_CHILD,
        _OWNERSHIP_DUPLICATE_CHILD,
        _OWNERSHIP_PUNCTUATION_CHILD,
    }
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or not _is_text_area_speech_container(region):
            continue
        if _text_area_preserve_or_conflict_like(region):
            continue
        text = _clean_source_text(str(region.get("ocr_text") or ""))
        if not _source_body(text):
            continue
        meaningful_count += 1
        status = str(region.get("logical_text_ownership_status") or "").strip()
        resolution = str(region.get("logical_text_blocked_fragment_resolution") or "").strip()
        if resolution in {"represented_in_anchor", "duplicate_child", "punctuation_child", "noise_review_only"}:
            continue
        if status in represented:
            continue
        blocked_ids.append(rid)
        if status == _OWNERSHIP_NOISE_REVIEW_ONLY or (region.get("flags") or {}).get("ignore"):
            source_survivor_ids.append(rid)
    return {
        "meaningful_count": meaningful_count,
        "blocked_ids": sorted(set(blocked_ids)),
        "source_survivor_ids": sorted(set(source_survivor_ids)),
    }


def _page_conservation_status(unowned_ids: set[str], speech_metrics: dict[str, Any]) -> str:
    if unowned_ids:
        return "failed_unowned_meaningful_regions"
    if speech_metrics.get("blocked_ids"):
        return "failed_blocked_meaningful_speech"
    return "complete"


def _is_speech_region_for_final_invariant(region: dict[str, Any]) -> bool:
    if not _is_text_area_speech_container(region):
        return False
    if _text_area_preserve_or_conflict_like(region):
        return False
    return True


def _is_text_area_speech_container(region: dict[str, Any]) -> bool:
    if str(_text_area_value(region, "text_area_container_type") or "").strip() != "speech_bubble":
        return False
    return str(_text_area_value(region, "text_area_route_intent") or "").strip() == "translate_speech"


def _append_region_reason(region: dict[str, Any], reason: str) -> None:
    if not reason:
        return
    reasons = region.setdefault("logical_text_block_reason_codes", [])
    if isinstance(reasons, list) and reason not in reasons:
        reasons.append(reason)
    render = region.setdefault("render", {})
    render_reasons = render.setdefault("logical_text_block_reason_codes", [])
    if isinstance(render_reasons, list) and reason not in render_reasons:
        render_reasons.append(reason)


def _region_has_renderable_state(region: dict[str, Any]) -> bool:
    render = region.get("render") or {}
    return bool(
        str(region.get("translation") or region.get("translated_text") or "").strip()
        or region.get("final_render_bbox")
        or render.get("final_render_bbox")
        or render.get("wrapped_lines")
        or region.get("wrapped_lines")
    )


def _clear_renderable_state(region: dict[str, Any], reason: str) -> None:
    historical_translation = str(region.get("translation") or region.get("translated_text") or "")
    region["translation"] = ""
    region["translated_text"] = ""
    region.pop("final_render_bbox", None)
    region.pop("wrapped_lines", None)
    region["logical_text_render_eligibility_repaired"] = True
    region["logical_text_render_eligibility_reason"] = reason
    render = region.setdefault("render", {})
    if historical_translation and not render.get("logical_text_block_historical_translation"):
        render["logical_text_block_historical_translation"] = historical_translation
    render["logical_text_render_eligibility_repaired"] = True
    render["logical_text_render_eligibility_reason"] = reason
    render.pop("final_render_bbox", None)
    render.pop("wrapped_lines", None)
    render["cleanup_mode"] = render.get("cleanup_mode") or "logical_text_non_renderable"


def _stamp_region_fields(region: dict[str, Any], block: LogicalTextBlock, ownership_status: str) -> None:
    region["logical_text_block_id"] = block.block_id
    region["logical_text_block_container_id"] = block.container_id
    region["logical_text_block_role"] = block.role
    region["logical_text_block_member_region_ids"] = list(block.member_region_ids)
    region["logical_text_block_anchor_region_id"] = block.anchor_region_id
    region["logical_text_block_transferred_region_ids"] = list(block.transferred_region_ids)
    region["logical_text_block_duplicate_region_ids"] = list(block.duplicate_region_ids)
    region["logical_text_block_punctuation_child_ids"] = list(block.punctuation_child_ids)
    region["logical_text_block_noise_child_ids"] = list(block.noise_child_ids)
    region["logical_text_block_source_text"] = block.source_text
    region["logical_text_block_reason_codes"] = list(block.reason_codes)
    region["logical_text_block_confidence"] = block.confidence
    region["logical_text_block_would_change_behavior"] = block.would_change_behavior
    region["logical_text_ownership_status"] = ownership_status
    region["logical_text_block_text_conservation_status"] = block.text_conservation_status
    region["logical_text_block_allowed_bbox"] = list(block.allowed_bbox or block.bbox or [])
    region["logical_text_physical_bubble_id"] = block.physical_bubble_id
    region["logical_text_physical_bubble_member_container_ids"] = list(block.physical_bubble_member_container_ids)
    region["logical_text_physical_bubble_source"] = block.physical_bubble_source
    region["logical_text_physical_bubble_reason_codes"] = list(block.physical_bubble_reason_codes)
    region["logical_text_source_quality_status"] = block.source_quality_status
    region["logical_text_source_quality_reason_codes"] = list(block.source_quality_reason_codes)
    region["logical_text_source_quality_action"] = block.source_quality_action
    region["logical_text_blocked_fragment_resolution"] = _blocked_fragment_resolution(block, ownership_status, region)
    region["logical_text_source_reconstruction_status"] = block.source_reconstruction_status
    region["logical_text_source_reconstruction_applied"] = block.source_reconstruction_applied
    region["logical_text_source_reconstruction_before_text"] = block.source_reconstruction_before_text
    region["logical_text_source_reconstruction_after_text"] = block.source_reconstruction_after_text
    region["logical_text_source_reconstruction_ocr_confidence"] = block.source_reconstruction_ocr_confidence
    region["logical_text_source_reconstruction_crop_bbox"] = list(block.source_reconstruction_crop_bbox)
    region["logical_text_source_reconstruction_included_child_region_ids"] = list(block.source_reconstruction_included_child_region_ids)
    region["logical_text_source_reconstruction_rejected_child_region_ids"] = list(block.source_reconstruction_rejected_child_region_ids)
    region["logical_text_source_reconstruction_reason_codes"] = list(block.source_reconstruction_reason_codes)
    region["logical_text_source_reconstruction_child_fragment_status"] = list(block.source_reconstruction_child_fragment_status)
    _stamp_v3_region_fields(region, block, ownership_status)
    render = region.setdefault("render", {})
    render["logical_text_block_id"] = block.block_id
    render["logical_text_ownership_status"] = ownership_status
    render["logical_text_block_text_conservation_status"] = block.text_conservation_status
    render["logical_text_physical_bubble_id"] = block.physical_bubble_id
    render["logical_text_physical_bubble_member_container_ids"] = list(block.physical_bubble_member_container_ids)
    render["logical_text_physical_bubble_source"] = block.physical_bubble_source
    render["logical_text_physical_bubble_reason_codes"] = list(block.physical_bubble_reason_codes)
    render["logical_text_source_quality_status"] = block.source_quality_status
    render["logical_text_source_quality_reason_codes"] = list(block.source_quality_reason_codes)
    render["logical_text_source_quality_action"] = block.source_quality_action
    render["logical_text_blocked_fragment_resolution"] = region["logical_text_blocked_fragment_resolution"]
    render["logical_text_source_reconstruction_status"] = block.source_reconstruction_status
    render["logical_text_source_reconstruction_applied"] = block.source_reconstruction_applied
    render["logical_text_source_reconstruction_before_text"] = block.source_reconstruction_before_text
    render["logical_text_source_reconstruction_after_text"] = block.source_reconstruction_after_text
    render["logical_text_source_reconstruction_ocr_confidence"] = block.source_reconstruction_ocr_confidence
    render["logical_text_source_reconstruction_crop_bbox"] = list(block.source_reconstruction_crop_bbox)
    render["logical_text_source_reconstruction_included_child_region_ids"] = list(block.source_reconstruction_included_child_region_ids)
    render["logical_text_source_reconstruction_rejected_child_region_ids"] = list(block.source_reconstruction_rejected_child_region_ids)
    render["logical_text_source_reconstruction_reason_codes"] = list(block.source_reconstruction_reason_codes)
    render["logical_text_source_reconstruction_child_fragment_status"] = list(block.source_reconstruction_child_fragment_status)
    _copy_v3_fields_to_render(region, render)


def _stamp_record_only(region: dict[str, Any] | None, record: LogicalTextOwnershipRecord) -> None:
    if region is None:
        return
    region["logical_text_block_id"] = record.block_id
    region["logical_text_block_container_id"] = record.container_id
    region["logical_text_ownership_status"] = record.ownership_status
    region["logical_text_block_reason_codes"] = list(record.reason_codes)
    region["logical_text_block_source_text"] = record.source_text
    region["logical_text_block_would_change_behavior"] = record.would_change_behavior
    region["logical_text_blocked_fragment_resolution"] = _record_blocked_fragment_resolution(record, region)
    _stamp_v3_record_fields(region, record)
    region.setdefault("render", {})["logical_text_ownership_status"] = record.ownership_status
    region.setdefault("render", {})["logical_text_blocked_fragment_resolution"] = region["logical_text_blocked_fragment_resolution"]
    _copy_v3_fields_to_render(region, region.setdefault("render", {}))


def _stamp_v3_region_fields(region: dict[str, Any], block: LogicalTextBlock, ownership_status: str) -> None:
    final_state = _v3_ownership_state(ownership_status, region)
    is_translation_unit = ownership_status in _ACTIVE_OWNERSHIP_STATUSES and block.source_quality_action not in _BLOCKING_SOURCE_QUALITY_ACTIONS
    region["physical_bubble_graph_id"] = block.physical_bubble_id
    region["logical_text_block_v3_status"] = final_state
    region["logical_text_block_translation_unit"] = is_translation_unit
    region["ocr_fragment_ownership_status"] = ownership_status
    region["ocr_fragment_final_state"] = final_state
    region["active_translation_unit_id"] = block.block_id if is_translation_unit else None
    region["source_text_represented_by_block_id"] = block.block_id if final_state in {
        _OWNERSHIP_BLOCK_ANCHOR,
        _V3_STANDALONE_UTTERANCE,
        _V3_DEPENDENT_CHILD,
        _OWNERSHIP_DUPLICATE_CHILD,
        _OWNERSHIP_PUNCTUATION_CHILD,
    } else None
    region["logical_text_block_source_reconstruction_required"] = bool(block.source_reconstruction_required)
    region["logical_text_block_source_reconstruction_status"] = block.source_reconstruction_status
    region["logical_text_block_source_reconstruction_crop_bbox"] = list(block.source_reconstruction_crop_bbox)
    region["logical_text_block_included_child_region_ids"] = list(block.source_reconstruction_included_child_region_ids)
    region["logical_text_block_rejected_child_region_ids"] = list(block.source_reconstruction_rejected_child_region_ids)
    region["logical_text_block_unresolved_reason"] = block.source_reconstruction_unresolved_reason
    region["source_conservation_status"] = block.text_conservation_status
    region["source_conservation_failure_reason"] = block.source_reconstruction_unresolved_reason


def _stamp_v3_record_fields(region: dict[str, Any], record: LogicalTextOwnershipRecord) -> None:
    final_state = _v3_ownership_state(record.ownership_status, region)
    is_translation_unit = record.ownership_status in _ACTIVE_OWNERSHIP_STATUSES
    region["physical_bubble_graph_id"] = region.get("logical_text_physical_bubble_id")
    region["logical_text_block_v3_status"] = final_state
    region["logical_text_block_translation_unit"] = is_translation_unit
    region["ocr_fragment_ownership_status"] = record.ownership_status
    region["ocr_fragment_final_state"] = final_state
    region["active_translation_unit_id"] = record.block_id if is_translation_unit else None
    region["source_text_represented_by_block_id"] = record.block_id if final_state in {
        _OWNERSHIP_BLOCK_ANCHOR,
        _V3_STANDALONE_UTTERANCE,
        _V3_DEPENDENT_CHILD,
        _OWNERSHIP_DUPLICATE_CHILD,
        _OWNERSHIP_PUNCTUATION_CHILD,
    } else None
    region["source_conservation_status"] = region.get("logical_text_block_text_conservation_status")
    region["source_conservation_failure_reason"] = None if final_state != _V3_REVIEW_ONLY_UNRESOLVED else "review_only_unresolved"


def _copy_v3_fields_to_render(region: dict[str, Any], render: dict[str, Any]) -> None:
    for key in (
        "physical_bubble_graph_id",
        "logical_text_block_v3_status",
        "logical_text_block_translation_unit",
        "logical_text_block_source_reconstruction_required",
        "logical_text_block_source_reconstruction_status",
        "logical_text_block_source_reconstruction_crop_bbox",
        "logical_text_block_included_child_region_ids",
        "logical_text_block_rejected_child_region_ids",
        "logical_text_block_unresolved_reason",
        "ocr_fragment_ownership_status",
        "ocr_fragment_final_state",
        "active_translation_unit_id",
        "source_text_represented_by_block_id",
        "source_conservation_status",
        "source_conservation_failure_reason",
    ):
        render[key] = region.get(key)


def _v3_ownership_state(ownership_status: str, region: dict[str, Any]) -> str:
    if _text_area_preserve_or_conflict_like(region):
        return _V3_BLOCKED_SFX_OR_DECORATIVE
    if ownership_status == _OWNERSHIP_STANDALONE_BLOCK:
        return _V3_STANDALONE_UTTERANCE
    if ownership_status == _OWNERSHIP_TRANSFERRED_CHILD:
        return _V3_DEPENDENT_CHILD
    if ownership_status in {
        _OWNERSHIP_BLOCK_ANCHOR,
        _OWNERSHIP_DUPLICATE_CHILD,
        _OWNERSHIP_PUNCTUATION_CHILD,
        _OWNERSHIP_NOISE_REVIEW_ONLY,
    }:
        return ownership_status
    return _V3_REVIEW_ONLY_UNRESOLVED


def _blocked_fragment_resolution(block: LogicalTextBlock, ownership_status: str, region: dict[str, Any]) -> str:
    if _text_area_preserve_or_conflict_like(region):
        return "sfx_decorative_conflict"
    if ownership_status in {_OWNERSHIP_BLOCK_ANCHOR, _OWNERSHIP_STANDALONE_BLOCK, _OWNERSHIP_TRANSFERRED_CHILD}:
        return "represented_in_anchor"
    if ownership_status == _OWNERSHIP_DUPLICATE_CHILD:
        return "duplicate_child"
    if ownership_status == _OWNERSHIP_PUNCTUATION_CHILD:
        return "punctuation_child"
    if block.source_quality_action in _BLOCKING_SOURCE_QUALITY_ACTIONS:
        return "speech_source_repair_required" if _is_text_area_speech_container(region) else "source_quality_blocked"
    if ownership_status == _OWNERSHIP_NOISE_REVIEW_ONLY:
        if block.source_reconstruction_applied or "noise_fragment_review_only" in set(block.reason_codes or []):
            return "noise_review_only"
        return "speech_source_repair_required" if _is_text_area_speech_container(region) else "source_quality_blocked"
    return "unresolved_meaningful_speech"


def _record_blocked_fragment_resolution(record: LogicalTextOwnershipRecord, region: dict[str, Any]) -> str:
    if _text_area_preserve_or_conflict_like(region):
        return "sfx_decorative_conflict"
    if record.ownership_status == _OWNERSHIP_PUNCTUATION_CHILD:
        return "punctuation_child"
    if record.ownership_status == _OWNERSHIP_DUPLICATE_CHILD:
        return "duplicate_child"
    if record.ownership_status == _OWNERSHIP_TRANSFERRED_CHILD:
        return "represented_in_anchor"
    if record.ownership_status == _OWNERSHIP_NOISE_REVIEW_ONLY:
        return "noise_review_only"
    if record.ownership_status in _ACTIVE_OWNERSHIP_STATUSES:
        return "represented_in_anchor"
    return "unresolved_meaningful_speech"


def _ownership_record(
    page_id: str,
    region_id: str,
    container_id: str,
    status: str,
    block_id: str | None,
    reasons: list[str],
    source_text: str,
    would_change_behavior: bool,
) -> LogicalTextOwnershipRecord:
    return LogicalTextOwnershipRecord(
        page_id=page_id,
        region_id=region_id,
        container_id=container_id,
        ownership_status=status,
        block_id=block_id,
        reason_codes=reasons,
        source_text=source_text,
        would_change_behavior=would_change_behavior,
    )


def _child_classification_reason(block: LogicalTextBlock, region_id: str) -> str:
    if region_id in block.duplicate_region_ids:
        return "same_container_logical_text_block_duplicate"
    if region_id in block.punctuation_child_ids:
        return "same_container_logical_text_block_punctuation_child"
    if region_id in block.noise_child_ids:
        return "same_container_logical_text_block_noise_review_only"
    return "same_container_logical_text_block_transfer"


def _text_conservation_status(
    transferred: list[str],
    duplicates: list[str],
    punctuation_child_ids: list[str],
    noise_child_ids: list[str],
) -> str:
    if noise_child_ids:
        return "complete_with_noise_review_only"
    if punctuation_child_ids:
        return "complete_with_punctuation_children"
    if duplicates:
        return "complete_with_duplicate_suppression"
    if transferred:
        return "complete_with_transferred_children"
    return "complete"


def _is_anchor_worthy(region: dict[str, Any]) -> bool:
    text = _source_body(str(region.get("ocr_text") or ""))
    if len(text) >= 4:
        return True
    if len(text) >= 2 and _has_kanji(text):
        return True
    _x, _y, w, h = _bbox(region.get("bbox"))
    return len(text) >= 2 and w * h >= 5200


def _tiny_dependent_speech_fragment_without_anchor(
    region: dict[str, Any],
    text: str,
    container_bbox: list[int],
) -> bool:
    body = _source_body(text)
    if not body or _is_punctuation_or_ellipsis_only(text):
        return False
    if not (_is_dependent_fragment(region) or _is_particle_dependent_fragment(region)):
        return False
    if _valid_standalone_short_speech(region, text):
        return False
    if not _text_area_preserve_or_conflict_like(region):
        return False
    _x, _y, w, h = _bbox(region.get("bbox"))
    fragment_area = max(1, w * h)
    _cx, _cy, cw, ch = _bbox(container_bbox or _text_area_value(region, "text_area_container_bbox"))
    container_area = max(1, cw * ch)
    return len(body) <= 2 or _is_particle_dependent_fragment(region) or fragment_area <= max(4200, int(container_area * 0.10))


def _valid_standalone_short_speech(region: dict[str, Any], text: str) -> bool:
    if _text_area_preserve_or_conflict_like(region):
        return False
    if not _is_text_area_speech_container(region):
        return False
    body = _source_body(text)
    if len(body) < 2:
        return False
    tier = str(_text_area_value(region, "text_area_confidence_tier") or "").strip()
    if tier not in _STRONG_PHYSICAL_TIERS:
        return False
    inside_ratio = _bbox_inside_ratio(_bbox(region.get("bbox")), _bbox(_text_area_value(region, "text_area_container_bbox")))
    if inside_ratio < 0.60:
        return False
    if _has_kanji(body):
        return True
    return any("\u3040" <= ch <= "\u30ff" for ch in body)


def _tiny_dependent_fragment_reasons(region: dict[str, Any]) -> list[str]:
    reasons = ["tiny_dependent_speech_fragment_without_anchor"]
    if _text_area_preserve_or_conflict_like(region):
        reasons.append("text_area_preserve_conflict_blocks_tiny_fragment")
    if _is_particle_dependent_fragment(region):
        reasons.append("particle_fragment_requires_anchor")
    return reasons


def _text_area_preserve_or_conflict_like(region: dict[str, Any]) -> bool:
    if any(str(flag).strip() for flag in _text_area_value(region, "text_area_conflict_flags") or []):
        return True
    container_type = str(_text_area_value(region, "text_area_container_type") or "").strip()
    route_intent = str(_text_area_value(region, "text_area_route_intent") or "").strip()
    if container_type == "sfx_decorative_art" or route_intent == "preserve_sfx_decorative":
        return True
    if route_intent in {"translate_speech", "translate_caption_background"} and container_type in {
        "speech_bubble",
        "caption_background",
    }:
        return False
    tier = str(_text_area_value(region, "text_area_confidence_tier") or "").strip()
    if tier == "conflict_preserve_wins":
        return True
    reason_text = " ".join(str(v) for v in (_text_area_value(region, "text_area_reason_codes") or [])).lower()
    conflict_tokens = (
        "sfx",
        "decorative",
        "nonbubble",
        "without_visual_speech",
        "review_only",
        "preserve_sfx",
        "art_text",
        "art_sfx",
    )
    return any(token in reason_text for token in conflict_tokens)


def _is_dependent_fragment(region: dict[str, Any]) -> bool:
    text = _clean_source_text(str(region.get("ocr_text") or ""))
    body = _source_body(text)
    _x, _y, w, h = _bbox(region.get("bbox"))
    if _is_punctuation_or_ellipsis_only(text):
        return True
    if len(body) <= 2 and w * h <= 6000:
        return True
    if _is_particle_dependent_fragment(region):
        return True
    if len(body) <= 4 and w * h <= 4600 and str(_text_area_value(region, "text_area_container_type") or "") == "speech_bubble":
        return True
    return len(body) <= 3 and max(w, h) <= 90


def _is_particle_dependent_fragment(region: dict[str, Any]) -> bool:
    text = _clean_source_text(str(region.get("ocr_text") or ""))
    body = _source_body(text)
    _x, _y, w, h = _bbox(region.get("bbox"))
    return (
        len(body) <= 5
        and body.endswith(("の", "は", "を", "が", "に", "で", "も", "と"))
        and str(_text_area_value(region, "text_area_container_type") or "") == "speech_bubble"
        and max(w, h) <= 180
    )


def _is_review_noise(region: dict[str, Any], text: str) -> bool:
    body = _source_body(text)
    if not body:
        return True
    if _is_particle_dependent_fragment(region):
        return False
    _x, _y, w, h = _bbox(region.get("bbox"))
    if w * h <= 250 and max(w, h) <= 24:
        return True
    if _region_confidence(region) < 0.45 and w * h <= 1500:
        return True
    return len(body) <= 1 and w * h <= 3600 and not _has_kanji(body)


def _has_meaningful_cjk(text: str) -> bool:
    body = _source_body(text)
    if _has_kanji(body):
        return True
    return len(body) >= 2 and any("\u3040" <= ch <= "\u30ff" for ch in body)


def _has_kanji(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _is_punctuation_or_ellipsis_only(text: str) -> bool:
    cleaned = _clean_source_text(text)
    body = _source_body(cleaned)
    if body:
        return False
    return bool(cleaned)


def _merge_source_fragment(current: str, new: str) -> tuple[str, str]:
    current = _clean_source_text(current)
    new = _clean_source_text(new)
    if not new:
        return current, "duplicate"
    if not current:
        return new, "append"
    current_body = _source_body(current)
    new_body = _source_body(new)
    if new_body and new_body in current_body:
        return current, "duplicate"
    if current_body and current_body in new_body:
        return new, "overlap"
    max_overlap = min(len(current), len(new))
    for length in range(max_overlap, 1, -1):
        if current[-length:] == new[:length]:
            return current + new[length:], "overlap"
    best = 0
    for length in range(max_overlap, 1, -1):
        left = current[-length:]
        right = new[:length]
        if difflib.SequenceMatcher(None, _source_body(left), _source_body(right)).ratio() >= 0.86:
            best = length
            break
    if best:
        return current + new[best:], "overlap"
    separator = "" if current.endswith(("、", "。", "…", "...")) or new.startswith(("、", "。", "…", "...")) else "、"
    return current + separator + new, "append"


def _source_fragments_overlap(a_text: str, b_text: str) -> bool:
    if not a_text or not b_text:
        return False
    if a_text in b_text or b_text in a_text:
        return True
    if difflib.SequenceMatcher(None, a_text, b_text).ratio() >= 0.84:
        return True
    max_overlap = min(len(a_text), len(b_text))
    for length in range(max_overlap, 1, -1):
        if a_text[-length:] == b_text[:length] or b_text[-length:] == a_text[:length]:
            return True
        left = a_text[-length:]
        right = b_text[:length]
        if difflib.SequenceMatcher(None, left, right).ratio() >= 0.88:
            return True
        left = b_text[-length:]
        right = a_text[:length]
        if difflib.SequenceMatcher(None, left, right).ratio() >= 0.88:
            return True
    return False


def _block_confidence(
    members: list[dict[str, Any]],
    transferred: list[str],
    duplicates: list[str],
    punctuation_child_ids: list[str],
    noise_child_ids: list[str],
) -> float:
    values = [_region_confidence(region) for region in members]
    base = min(values) if values else 0.75
    if duplicates:
        base = max(base, 0.82)
    if transferred:
        base = max(base, 0.78)
    if punctuation_child_ids:
        base = max(base, 0.76)
    if noise_child_ids:
        base = min(base, 0.70)
    return round(min(1.0, max(0.0, base)), 3)


def _region_confidence(region: dict[str, Any]) -> float:
    values: list[float] = []
    confidence = region.get("confidence") or {}
    if isinstance(confidence, dict):
        if confidence.get("ocr") is not None:
            values.append(float(confidence.get("ocr") or 0.0))
        if confidence.get("det") is not None:
            values.append(float(confidence.get("det") or 0.0))
    return round(min(values) if values else 0.75, 3)


def _manga_region_sort_key(region: dict[str, Any]) -> tuple[int, int, int]:
    x, y, w, _h = _bbox(region.get("bbox"))
    right = x + w
    return (-right, y, x)


def _manga_bbox_sort_key(bbox: list[int]) -> tuple[int, int, int]:
    x, y, w, _h = _bbox(bbox)
    right = x + w
    return (-right, y, x)


def _logical_block_member_sort_key(
    region: dict[str, Any],
    physical_group: PhysicalBubbleGroup | None,
) -> tuple[int, int, int, int, int]:
    if physical_group and len(physical_group.member_container_ids) > 1:
        c_x, c_y, _c_w, _c_h = _bbox(_text_area_value(region, "text_area_container_bbox"))
        x, y, w, _h = _bbox(region.get("bbox"))
        right = x + w
        return (c_y, c_x, y, -right, x)
    manga_key = _manga_region_sort_key(region)
    return (0, 0, manga_key[0], manga_key[1], manga_key[2])


def _union_region_bboxes(regions: list[dict[str, Any]], image_size: tuple[int, int] | None) -> list[int]:
    boxes = [_bbox(region.get("bbox")) for region in regions]
    return _union_bboxes(boxes, image_size) or [0, 0, 1, 1]


def _union_bboxes(boxes: list[list[int]], image_size: tuple[int, int] | None) -> list[int] | None:
    boxes = [_bbox(box) for box in boxes if box]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return _clip_bbox([x0, y0, max(1, x1 - x0), max(1, y1 - y0)], image_size)


def _clip_bbox(bbox: list[int], image_size: tuple[int, int] | None) -> list[int]:
    x0, y0, w, h = _bbox(bbox)
    x1 = x0 + w
    y1 = y0 + h
    if image_size:
        img_w = max(1, int(image_size[0] or 1))
        img_h = max(1, int(image_size[1] or 1))
        x0 = max(0, min(img_w - 1, x0))
        y0 = max(0, min(img_h - 1, y0))
        x1 = max(x0 + 1, min(img_w, x1))
        y1 = max(y0 + 1, min(img_h, y1))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _bbox(value: Any) -> list[int]:
    try:
        x, y, w, h = [int(round(float(v or 0))) for v in list(value or [0, 0, 0, 0])[:4]]
    except Exception:
        return [0, 0, 1, 1]
    return [x, y, max(1, w), max(1, h)]


def _bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x, y, w, h = _bbox(bbox)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _center_x(bbox: list[int]) -> float:
    x, _y, w, _h = _bbox(bbox)
    return x + w / 2.0


def _center_y(bbox: list[int]) -> float:
    _x, y, _w, h = _bbox(bbox)
    return y + h / 2.0


def _overlap_ratio(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = _bbox(a)
    bx, by, bw, bh = _bbox(b)
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / max(1, min(aw * ah, bw * bh))


def _bbox_inside_ratio(inner: list[int], outer: list[int]) -> float:
    ix, iy, iw, ih = _bbox(inner)
    ox, oy, ow, oh = _bbox(outer)
    x0 = max(ix, ox)
    y0 = max(iy, oy)
    x1 = min(ix + iw, ox + ow)
    y1 = min(iy + ih, oy + oh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / max(1, iw * ih)


def _vertical_overlap_fraction(a: list[int], b: list[int]) -> float:
    _ax, ay, _aw, ah = _bbox(a)
    _bx, by, _bw, bh = _bbox(b)
    y0 = max(ay, by)
    y1 = min(ay + ah, by + bh)
    if y1 <= y0:
        return 0.0
    return (y1 - y0) / max(1, min(ah, bh))


def _horizontal_overlap_fraction(a: list[int], b: list[int]) -> float:
    ax, _ay, aw, _ah = _bbox(a)
    bx, _by, bw, _bh = _bbox(b)
    x0 = max(ax, bx)
    x1 = min(ax + aw, bx + bw)
    if x1 <= x0:
        return 0.0
    return (x1 - x0) / max(1, min(aw, bw))


def _horizontal_gap(a: list[int], b: list[int]) -> int:
    ax, _ay, aw, _ah = _bbox(a)
    bx, _by, bw, _bh = _bbox(b)
    if ax <= bx:
        return max(0, bx - (ax + aw))
    return max(0, ax - (bx + bw))


def _vertical_gap(a: list[int], b: list[int]) -> int:
    _ax, ay, _aw, ah = _bbox(a)
    _bx, by, _bw, bh = _bbox(b)
    if ay <= by:
        return max(0, by - (ay + ah))
    return max(0, ay - (by + bh))


def _clean_source_text(text: str) -> str:
    text = str(text or "").replace("\\n", " ").replace("/n", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source_body(text: str) -> str:
    return "".join(ch for ch in _clean_source_text(text) if not _is_punctuation(ch))


def _is_punctuation(ch: str) -> bool:
    if not ch:
        return True
    if ch.isspace():
        return True
    return ch in "。、，,.．・…!！?？ー-〜～~〰︴♡♥♪「」『』（）()[]{}<>《》:：;；/\\|"
