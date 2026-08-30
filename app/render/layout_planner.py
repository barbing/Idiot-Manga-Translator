# -*- coding: utf-8 -*-
"""Page-aware layout planning for parent-bundle render layers.

This module owns renderer-local shape and slot arbitration. It consumes an
immutable cleaned page plus existing RenderLayerPlan contracts and delegates
text measurement to TypesettingEngine. It does not own semantics, cleanup,
parent identity, translation, or style resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.render.typesetting_contracts import (
    FitReport,
    HORIZONTAL_SHAPE_CAPACITY_PROFILE_VERSION,
    HorizontalCapacityRow,
    HorizontalShapeCapacityProfile,
    RenderLayerPlan,
    TypesetLayout,
    copy_jsonish,
    validated_source_text_footprint_ref,
)
from app.render.typesetting_engine import TypesettingEngine
from app.render.parent_layer_effects import resolve_parent_layer_effects

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - required by normal renderer runtime
    Image = None
    ImageDraw = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None


RENDER_LAYOUT_PLANNER_VERSION = "render_layout_planner_v5_typography_aware"
PARENT_RENDER_SLOT_VERSION = "parent_render_slot_v2"

_MASK_OUTSIDE_PENALTY_WEIGHT = 600.0
_SHAPE_BALANCE_PENALTY_WEIGHT = 40.0
_ALIGNMENT_CENTER_PENALTY_WEIGHT = 55.0
_SHAPE_PULL_FRACTION = 0.35
_SHAPE_PULL_MAX_EM = 0.40


@dataclass(frozen=True)
class PlannedLayerResult:
    """Final renderer-local plan and an optional already-measured layout.

    Visual slot scoring must typeset every candidate to compare it.  Carrying
    the selected candidate's immutable result forward prevents the page
    executor from repeating that exact work without moving layout ownership
    out of TypesettingEngine.
    """

    plan: RenderLayerPlan
    layout: TypesetLayout | None = None
    fit_report: FitReport | None = None


class RenderLayoutPlanner:
    """Resolve a final renderer-owned slot before compositor drawing."""

    version = RENDER_LAYOUT_PLANNER_VERSION

    def __init__(self, typesetting_engine: TypesettingEngine) -> None:
        self.typesetting_engine = typesetting_engine

    def plan_page_slots(
        self,
        plans: Sequence[RenderLayerPlan],
    ) -> list[RenderLayerPlan]:
        """Project each upstream parent contract into one renderer-local slot."""

        return _page_slotted_plans(plans)

    def plan_layer(
        self,
        page,
        plan: RenderLayerPlan,
        *,
        occupied_bounds: Sequence[Mapping[str, Any]],
    ) -> RenderLayerPlan:
        return self.plan_layer_with_typeset(
            page,
            plan,
            occupied_bounds=occupied_bounds,
        ).plan

    def plan_layer_with_typeset(
        self,
        page,
        plan: RenderLayerPlan,
        *,
        occupied_bounds: Sequence[Mapping[str, Any]],
    ) -> PlannedLayerResult:
        if (
            _is_latin_shape_band_layer(plan)
            and _is_shape_aware_speech_layer(plan)
        ):
            latin_result = _latin_shape_band_planned_result(
                page,
                plan,
                self.typesetting_engine,
            )
            if latin_result is not None:
                return latin_result
        shape_plan = _shape_aware_plan(page, plan)
        return _visual_slot_scored_result(
            page,
            plan,
            shape_plan,
            self.typesetting_engine,
            occupied_bounds,
        )


def _page_slotted_plans(
    plans: Sequence[RenderLayerPlan],
) -> list[RenderLayerPlan]:
    input_plans = [plan for plan in plans or [] if isinstance(plan, RenderLayerPlan)]
    return [
        _plan_with_parent_slot(plan, _parent_local_slot(plan))
        for plan in input_plans
    ]


def _parent_local_slot(plan: RenderLayerPlan) -> dict[str, Any]:
    """Create one slot without reconstructing root or sibling topology.

    Root and source boxes remain provenance/alignment evidence. Only the
    upstream parent target, hard bounds, and render-allowed area may define
    renderer capacity.
    """

    parent_box = _parent_anchor_box(plan)
    hard_bounds = _parent_hard_bounds(plan) or parent_box
    root_box = _root_evidence_box(plan)
    return _parent_slot_record(
        plan,
        box=parent_box,
        hard_bounds=hard_bounds,
        source="parent_render_slot_parent_contract",
        container_box=parent_box,
        root_box=root_box,
    )


def _plan_with_parent_slot(
    plan: RenderLayerPlan,
    slot: Mapping[str, Any],
) -> RenderLayerPlan:
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    slot_record = copy_jsonish(slot) if isinstance(slot, Mapping) else {}
    metadata["parent_render_slot"] = slot_record
    metadata["target_box_source"] = str(
        slot_record.get("source") or metadata.get("target_box_source") or ""
    )
    box = _bbox_from_value(slot_record.get("box")) or _bbox_from_value(plan.target_box)
    hard = _bbox_from_value(slot_record.get("hard_bounds")) or _bbox_from_value(plan.hard_bounds) or box
    alignment_kind = str(slot_record.get("alignment_anchor_kind") or "")
    requested_center = _point_from_value(
        slot_record.get("alignment_anchor_center")
    )
    metadata["visual_alignment_requested_center"] = copy_jsonish(
        requested_center
    )
    metadata["visual_alignment_anchor_kind"] = alignment_kind
    if alignment_kind in {
        "source_text_footprint_union_bbox",
        "source_contract_bbox",
    } and requested_center and _point_inside_box(requested_center, hard):
        metadata["visual_alignment_center"] = copy_jsonish(requested_center)
        metadata["visual_alignment_policy"] = (
            "source_text_footprint_union_center"
            if alignment_kind == "source_text_footprint_union_bbox"
            else "source_contract_bbox_center"
        )
        metadata["visual_alignment_status"] = "applied_from_parent_slot"
    else:
        metadata.pop("visual_alignment_center", None)
        metadata.pop("visual_alignment_policy", None)
        metadata["visual_alignment_status"] = (
            "anchor_outside_hard_bounds"
            if requested_center
            and alignment_kind
            in {
                "source_text_footprint_union_bbox",
                "source_contract_bbox",
            }
            else "target_box_center_default"
        )
    return replace(
        plan,
        target_box=list(box),
        hard_bounds=list(hard),
        metadata=metadata,
    )


def _parent_slot_record(
    plan: RenderLayerPlan,
    *,
    box: Sequence[int],
    hard_bounds: Sequence[int],
    source: str,
    container_box: Sequence[int],
    root_box: Sequence[int],
) -> dict[str, Any]:
    source_box = _source_contract_box(plan)
    footprint_box = _source_text_footprint_alignment_box(plan)
    anchor_box, anchor_kind = _alignment_anchor_box(plan)
    record = {
        "render_layout_slot_version": PARENT_RENDER_SLOT_VERSION,
        "box": _bbox_from_value(box),
        "hard_bounds": _bbox_from_value(hard_bounds) or _bbox_from_value(box),
        "source": str(source),
        "source_contract_bbox": source_box,
        "source_anchor_center": list(_center_tuple(source_box)) if source_box else [],
        "source_text_footprint_union_bbox": footprint_box,
        "container_bbox": _bbox_from_value(container_box),
        "root_bbox": _bbox_from_value(root_box),
        "sibling_count": 1,
        "parent_id": str(plan.parent_id or ""),
        "root_id": str(plan.root_id or ""),
        "alignment_anchor_bbox": anchor_box,
        "alignment_anchor_center": list(_center_tuple(anchor_box)) if anchor_box else [],
        "alignment_anchor_kind": anchor_kind,
    }
    return record


def _parent_anchor_box(plan: RenderLayerPlan) -> list[int]:
    clipping = plan.clipping_region_ref if isinstance(plan.clipping_region_ref, Mapping) else {}
    for value in (
        plan.target_box,
        plan.hard_bounds,
        clipping.get("render_allowed_area"),
    ):
        box = _bbox_from_value(value)
        if box:
            return box
    return []


def _parent_hard_bounds(plan: RenderLayerPlan) -> list[int]:
    clipping = plan.clipping_region_ref if isinstance(plan.clipping_region_ref, Mapping) else {}
    for value in (
        plan.hard_bounds,
        clipping.get("render_allowed_area"),
        plan.target_box,
    ):
        box = _bbox_from_value(value)
        if box:
            return box
    return []


def _root_evidence_box(plan: RenderLayerPlan) -> list[int]:
    clipping = plan.clipping_region_ref if isinstance(plan.clipping_region_ref, Mapping) else {}
    return _bbox_from_value(clipping.get("root_bbox"))


def _layout_semantic_class(plan: RenderLayerPlan) -> str:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    values = (
        plan.role,
        plan.state,
        style.get("semantic_class"),
        style.get("semantic_kind"),
        style.get("source_role"),
        style.get("route_intent"),
    )
    return " ".join(
        dict.fromkeys(
            text
            for text in (str(value or "").strip().lower() for value in values)
            if text
        )
    )


def _center_tuple(box: Sequence[int]) -> tuple[float, float]:
    bbox = _bbox_from_value(box)
    if not bbox:
        return (0.0, 0.0)
    return (
        float(bbox[0]) + float(bbox[2]) / 2.0,
        float(bbox[1]) + float(bbox[3]) / 2.0,
    )


def _shape_aware_plan(page, plan: RenderLayerPlan) -> RenderLayerPlan:
    """Resolve a speech-bubble-safe rectangular layout box from page geometry.

    This is renderer-owned geometry. It does not decide semantic eligibility,
    create parents, run cleanup, or change text/style authority.
    """

    base_metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    existing = base_metadata.get("shape_aware_composition")
    if isinstance(existing, Mapping) and existing.get("applied"):
        return plan
    if not _is_shape_aware_speech_layer(plan):
        return _plan_with_shape_audit(plan, {"applied": False, "reason": "not_speech_bubble_layer"})
    if np is None:
        return _plan_with_shape_audit(plan, {"applied": False, "reason": "numpy_unavailable"})
    page_box = [0, 0, int(page.size[0]), int(page.size[1])]
    candidate = _shape_candidate_box(plan, page_box)
    if not candidate:
        return _plan_with_shape_audit(plan, {"applied": False, "reason": "missing_candidate_box"})
    safe = _speech_bubble_safe_box_from_page(page, plan, candidate)
    if not safe.get("box"):
        return _plan_with_shape_audit(plan, safe)
    safe_box = safe["box"]
    original_target = list(plan.target_box or [])
    original_hard = list(plan.hard_bounds or [])
    if safe_box == original_target and safe_box == original_hard:
        return _plan_with_shape_audit(plan, safe)
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["shape_aware_composition"] = safe
    clipping = copy_jsonish(plan.clipping_region_ref) if isinstance(plan.clipping_region_ref, Mapping) else {}
    clipping["shape_aware_safe_box"] = list(safe_box)
    return replace(
        plan,
        target_box=list(safe_box),
        hard_bounds=list(original_hard or original_target or safe_box),
        clipping_region_ref=clipping,
        metadata=metadata,
    )


def _plan_with_shape_audit(plan: RenderLayerPlan, audit: Mapping[str, Any]) -> RenderLayerPlan:
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["shape_aware_composition"] = copy_jsonish(audit)
    return replace(plan, metadata=metadata)


def _visual_slot_scored_plan(
    page,
    original_plan: RenderLayerPlan,
    shape_plan: RenderLayerPlan,
    typesetting_engine: TypesettingEngine,
    occupied_bounds: Sequence[Mapping[str, Any]],
) -> RenderLayerPlan:
    return _visual_slot_scored_result(
        page,
        original_plan,
        shape_plan,
        typesetting_engine,
        occupied_bounds,
    ).plan


def _visual_slot_scored_result(
    page,
    original_plan: RenderLayerPlan,
    shape_plan: RenderLayerPlan,
    typesetting_engine: TypesettingEngine,
    occupied_bounds: Sequence[Mapping[str, Any]],
) -> PlannedLayerResult:
    if not _is_shape_aware_speech_layer(original_plan):
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                {"applied": False, "reason": "not_speech_bubble_layer"},
            )
        )
    if np is None:
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                {"applied": False, "reason": "numpy_unavailable"},
            )
        )
    page_box = [0, 0, int(page.size[0]), int(page.size[1])]
    candidate = _shape_candidate_box(original_plan, page_box)
    if not candidate:
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                {"applied": False, "reason": "missing_candidate_box"},
            )
        )
    geometry = _speech_bubble_geometry_from_page(page, original_plan, candidate)
    audit = geometry.get("audit") if isinstance(geometry.get("audit"), Mapping) else {}
    if not audit.get("applied"):
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                audit
                or {"applied": False, "reason": "speech_geometry_unavailable"},
            )
        )

    candidates = _visual_slot_candidates(original_plan, shape_plan, geometry, page_box)
    if not candidates:
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                {"applied": False, "reason": "no_visual_slot_candidates"},
            )
        )

    scored: list[
        tuple[
            tuple[Any, ...],
            float,
            RenderLayerPlan,
            TypesetLayout,
            FitReport,
            dict[str, Any],
        ]
    ] = []
    original_metadata = (
        original_plan.metadata
        if isinstance(original_plan.metadata, Mapping)
        else {}
    )
    slot_record = (
        original_metadata.get("parent_render_slot")
        if isinstance(original_metadata.get("parent_render_slot"), Mapping)
        else {}
    )
    canonical_alignment_center = _point_from_value(
        original_metadata.get("visual_alignment_center")
    )
    canonical_alignment_kind = str(
        original_metadata.get("visual_alignment_anchor_kind")
        or slot_record.get("alignment_anchor_kind")
        or ""
    )
    visual_center_evidence = (
        geometry.get("visual_center_evidence")
        if isinstance(geometry.get("visual_center_evidence"), Mapping)
        else {}
    )
    speech_visual_center = _point_from_value(visual_center_evidence.get("center"))
    alignment_candidates = _visual_alignment_candidates(
        original_plan,
        speech_visual_center=speech_visual_center,
    )
    verified_containers: list[dict[str, Any]] = []
    competitively_pruned: list[dict[str, Any]] = []
    for candidate_record in candidates:
        box = _bbox_from_value(candidate_record.get("box"))
        if not box:
            continue
        containing_sizes = [
            float(item.get("selected_font_size") or 0.0)
            for item in verified_containers
            if _box_inside_tolerant(
                box,
                item.get("box") or [],
                tolerance=0,
            )
        ]
        competitive_floor = max(containing_sizes, default=0.0)
        successful_sizes: list[float] = []
        for alignment in alignment_candidates:
            candidate_plan = _plan_with_visual_slot_box(
                shape_plan,
                box,
                source=str(candidate_record.get("source") or "candidate"),
                alignment=alignment,
            )
            if competitive_floor > 0.0:
                candidate_metadata = copy_jsonish(candidate_plan.metadata)
                candidate_metadata["competitive_fit_probe_font_size"] = int(
                    round(competitive_floor)
                )
                candidate_metadata["competitive_fit_probe_policy"] = (
                    "contained_slot_cannot_beat_verified_font_size"
                )
                candidate_plan = replace(
                    candidate_plan,
                    metadata=candidate_metadata,
                )
            layout, report = typesetting_engine.typeset_layer(candidate_plan)
            if competitive_floor > 0.0 and not (
                bool(report.full_text_placed)
                and str(report.fit_status or "") == "fits"
                and bool(report.hard_bounds_contained)
                and bool(layout.text_placement_complete)
                and bool(layout.hard_bounds_contained)
                and float(layout.selected_font_size or 0.0)
                >= competitive_floor
            ):
                competitively_pruned.append(
                    {
                        "source": str(
                            candidate_record.get("source") or "candidate"
                        ),
                        "box": list(box),
                        "alignment_policy": str(
                            alignment.get("policy") or "target_box_center"
                        ),
                        "verified_containing_font_size": round(
                            float(competitive_floor),
                            6,
                        ),
                        "probe_fit_status": str(report.fit_status or ""),
                        "probe_full_text_placed": bool(
                            report.full_text_placed
                        ),
                        "reason": (
                            "contained_candidate_cannot_match_verified_"
                            "typography_size"
                        ),
                    }
                )
                continue
            typesetting_quality = typesetting_engine.candidate_quality_summary(
                candidate_plan,
                layout,
                report,
            )
            score, score_meta = _score_visual_slot(
                candidate_plan,
                layout,
                report,
                alignment_center=canonical_alignment_center,
                occupied_bounds=occupied_bounds,
                speech_visual_center=speech_visual_center,
                speech_safe_box=_bbox_from_value(audit.get("box")),
                safe_mask=geometry.get("safe_mask"),
                mask_origin_box=_bbox_from_value(geometry.get("candidate_box")),
            )
            scored.append(
                (
                    tuple(typesetting_quality.get("sort_key") or ()),
                    float(score),
                    candidate_plan,
                    layout,
                    report,
                    {
                        "source": str(candidate_record.get("source") or "candidate"),
                        "box": list(box),
                        "alignment_policy": str(alignment.get("policy") or "target_box_center"),
                        "requested_alignment_center": copy_jsonish(alignment.get("center") or []),
                        "applied_alignment_center": copy_jsonish(
                            candidate_plan.metadata.get("visual_alignment_center")
                            if isinstance(candidate_plan.metadata, Mapping)
                            else []
                        ),
                        "score": round(float(score), 4),
                        "typesetting_quality": copy_jsonish(
                            typesetting_quality
                        ),
                        **score_meta,
                    },
                )
            )
            successful_sizes.append(float(layout.selected_font_size or 0.0))
        if successful_sizes:
            verified_containers.append(
                {
                    "box": list(box),
                    "selected_font_size": max(successful_sizes),
                }
            )

    if not scored:
        return PlannedLayerResult(
            plan=_plan_with_visual_slot_audit(
                shape_plan,
                {
                    "applied": False,
                    "reason": "no_scoreable_visual_slot_candidates",
                },
            )
        )
    scored.sort(
        key=lambda item: _visual_slot_sort_key(
            item[0],
            item[1],
            item[2].target_box,
        )
    )
    (
        _typography_key,
        _score,
        selected_plan,
        selected_layout,
        selected_report,
        selected_meta,
    ) = scored[0]
    rejected = [item[5] for item in scored[1:12]]
    anchor_aligned = [
        item[5]
        for item in scored
        if str(item[5].get("alignment_policy") or "")
        in {
            "source_text_footprint_union_center",
            "source_contract_bbox_center",
        }
    ]
    final_audit = {
        "applied": True,
        "source": "shape_aware_typography_first_slot_scoring_v3",
        "selected_source": selected_meta.get("source"),
        "selected_alignment_policy": selected_meta.get("alignment_policy"),
        "selected_box": list(selected_plan.target_box),
        "selected_score": selected_meta.get("score"),
        "candidate_count": len(scored),
        "competitive_pruned_candidate_count": len(competitively_pruned),
        "competitive_pruned_candidates": competitively_pruned,
        "alignment_anchor_kind": canonical_alignment_kind,
        "alignment_anchor_center": copy_jsonish(canonical_alignment_center),
        "speech_component_box": copy_jsonish(audit.get("component_box")),
        "speech_safe_box": copy_jsonish(audit.get("box")),
        "speech_visual_center": copy_jsonish(speech_visual_center),
        "speech_visual_center_evidence": copy_jsonish(visual_center_evidence),
        "alignment_anchor_reference": (
            copy_jsonish(anchor_aligned[0]) if anchor_aligned else {}
        ),
        "selected": selected_meta,
        "rejected_candidates": rejected,
    }
    return PlannedLayerResult(
        plan=_plan_with_visual_slot_audit(selected_plan, final_audit),
        layout=selected_layout,
        fit_report=selected_report,
    )


def _visual_slot_sort_key(
    typesetting_quality_key: Sequence[Any],
    visual_score: float,
    target_box: Sequence[int],
) -> tuple[Any, ...]:
    """Keep typography ahead of mask/centering without redoing break logic."""

    return (
        tuple(typesetting_quality_key or ()),
        float(visual_score),
        _area(target_box),
    )


def _plan_with_visual_slot_audit(plan: RenderLayerPlan, audit: Mapping[str, Any]) -> RenderLayerPlan:
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["visual_slot_scoring"] = copy_jsonish(audit)
    return replace(plan, metadata=metadata)


def _plan_with_visual_slot_box(
    plan: RenderLayerPlan,
    box: Sequence[int],
    *,
    source: str,
    alignment: Mapping[str, Any],
) -> RenderLayerPlan:
    target = _bbox_from_value(box)
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["visual_slot_candidate_source"] = str(source)
    policy = str(alignment.get("policy") or "target_box_center")
    requested_center = _point_from_value(alignment.get("center"))
    applied_center: list[float] = []
    if requested_center:
        if policy in {
            "source_text_footprint_union_center",
            "source_contract_bbox_center",
        }:
            applied_center = requested_center if _point_inside_box(requested_center, target) else []
        else:
            applied_center = _clamp_point_to_box(requested_center, target)
    metadata["visual_alignment_policy"] = policy
    metadata["visual_alignment_requested_center"] = copy_jsonish(requested_center)
    metadata["visual_alignment_shift_from_source"] = copy_jsonish(
        alignment.get("shift_from_source") or []
    )
    metadata["visual_alignment_source_weight"] = float(alignment.get("source_weight") or 0.0)
    metadata["visual_alignment_shape_weight"] = float(alignment.get("shape_weight") or 0.0)
    metadata["visual_alignment_sibling_count"] = int(alignment.get("sibling_count") or 1)
    if applied_center:
        metadata["visual_alignment_center"] = copy_jsonish(applied_center)
        metadata["visual_alignment_status"] = (
            "applied_from_visual_slot_scoring"
        )
    else:
        metadata.pop("visual_alignment_center", None)
        metadata["visual_alignment_status"] = (
            "candidate_alignment_outside_target"
            if requested_center
            else "target_box_center_default"
        )
    clipping = copy_jsonish(plan.clipping_region_ref) if isinstance(plan.clipping_region_ref, Mapping) else {}
    clipping["visual_slot_box"] = list(target)
    return replace(
        plan,
        target_box=list(target),
        hard_bounds=list(plan.hard_bounds or plan.target_box or target),
        clipping_region_ref=clipping,
        metadata=metadata,
    )


def _visual_alignment_candidates(
    plan: RenderLayerPlan,
    *,
    speech_visual_center: Sequence[float],
) -> list[dict[str, Any]]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    alignment_center = _point_from_value(metadata.get("visual_alignment_center"))
    alignment_policy = str(
        metadata.get("visual_alignment_policy")
        or "canonical_alignment_center"
    )
    shape_center = _point_from_value(speech_visual_center)
    sibling_count = 1
    records: list[dict[str, Any]] = []

    if alignment_center:
        records.append(
            {
                "policy": alignment_policy,
                "center": _round_point(alignment_center),
                "shift_from_source": [0.0, 0.0],
                "source_weight": 1.0,
                "shape_weight": 0.0,
                "sibling_count": sibling_count,
            }
        )
        if shape_center:
            dx = float(shape_center[0]) - float(alignment_center[0])
            dy = float(shape_center[1]) - float(alignment_center[1])
            distance = math.hypot(dx, dy)
            style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
            font_size = max(1.0, float(_safe_int(style.get("font_size") or style.get("font_size_hint"), default=24)))
            max_shift = float(font_size) * _SHAPE_PULL_MAX_EM
            fraction = min(_SHAPE_PULL_FRACTION, max_shift / max(1.0, distance))
            balanced = [
                float(alignment_center[0]) + dx * fraction,
                float(alignment_center[1]) + dy * fraction,
            ]
            shift = [
                float(balanced[0]) - float(alignment_center[0]),
                float(balanced[1]) - float(alignment_center[1]),
            ]
            if math.hypot(shift[0], shift[1]) >= 0.5:
                records.append(
                    {
                        "policy": "canonical_shape_balanced_center",
                        "center": _round_point(balanced),
                        "shift_from_source": _round_point(shift),
                        "source_weight": round(float(1.0 - fraction), 4),
                        "shape_weight": round(float(fraction), 4),
                        "sibling_count": sibling_count,
                    }
                )
    elif shape_center:
        records.append(
            {
                "policy": "speech_shape_center",
                "center": _round_point(shape_center),
                "shift_from_source": [],
                "source_weight": 0.0,
                "shape_weight": 1.0,
                "sibling_count": sibling_count,
            }
        )
    else:
        records.append(
            {
                "policy": "target_box_center",
                "center": [],
                "shift_from_source": [],
                "source_weight": 0.0,
                "shape_weight": 0.0,
                "sibling_count": sibling_count,
            }
        )
    return records


def _visual_slot_candidates(
    original_plan: RenderLayerPlan,
    shape_plan: RenderLayerPlan,
    geometry: Mapping[str, Any],
    page_box: Sequence[int],
) -> list[dict[str, Any]]:
    audit = geometry.get("audit") if isinstance(geometry.get("audit"), Mapping) else {}
    candidate_box = _bbox_from_value(geometry.get("candidate_box")) or _bbox_from_value(audit.get("candidate_box"))
    component_box = _bbox_from_value(audit.get("component_box"))
    safe_box = _bbox_from_value(audit.get("box"))
    margin = _safe_int(audit.get("margin"), default=_shape_margin(original_plan, candidate_box or page_box))
    records: list[dict[str, Any]] = []

    def add(source: str, box: Sequence[int]) -> None:
        normalized = _intersect_box(_bbox_from_value(box), page_box)
        if not normalized:
            return
        container = safe_box or component_box or candidate_box or page_box
        if container:
            normalized = _intersect_box(normalized, container) or normalized
        if normalized[2] < 8 or normalized[3] < 8:
            return
        if any(_same_box(normalized, item.get("box", [])) for item in records):
            return
        records.append({"source": source, "box": normalized})

    add("shape_safe_box", safe_box)
    add("speech_component_box", component_box)
    add("current_shape_plan_box", shape_plan.target_box)
    add("original_target_box", original_plan.target_box)

    safe_mask = geometry.get("safe_mask")
    local_component_box = _bbox_from_value(geometry.get("component_box_local"))
    anchor = geometry.get("anchor")
    if (
        np is not None
        and safe_mask is not None
        and local_component_box
        and isinstance(anchor, Sequence)
        and not isinstance(anchor, (str, bytes, bytearray))
    ):
        core = _coverage_core_box(safe_mask, local_component_box, (float(anchor[0]), float(anchor[1])))
        if core and candidate_box:
            add("speech_coverage_core_box", _inset_box(_local_to_page_box(core, candidate_box), margin=max(2, min(margin, 10))))

    return records


def _score_visual_slot(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    *,
    alignment_center: Sequence[float],
    occupied_bounds: Sequence[Mapping[str, Any]],
    speech_visual_center: Sequence[float] = (),
    speech_safe_box: Sequence[int] = (),
    safe_mask=None,
    mask_origin_box: Sequence[int] = (),
) -> tuple[float, dict[str, Any]]:
    # Candidate quality is parent-local. Previously rendered siblings cannot
    # change this parent's slot choice or recreate root topology by draw order.
    del occupied_bounds
    measured = _bbox_from_value(layout.measured_bounds)
    target = _bbox_from_value(plan.target_box)
    score = 0.0
    meta: dict[str, Any] = {
        "fit_status": str(report.fit_status),
        "full_text_placed": bool(report.full_text_placed),
        "measured_bounds": list(measured),
    }
    if not report.full_text_placed:
        score += 1000.0
    if not measured:
        score += 500.0
        return score, meta
    if target and not _box_inside_tolerant(measured, target, tolerance=1):
        score += 250.0
        meta["measured_outside_target"] = True

    measured_center = _point_from_value(layout.visual_center) or _center_box(measured)
    font_size = max(1.0, float(layout.selected_font_size or 1.0))

    if safe_mask is not None and _bbox_from_value(mask_origin_box):
        inside_ratio = _mask_coverage_ratio(measured, safe_mask, mask_origin_box)
        mask_penalty = max(0.0, 1.0 - float(inside_ratio)) * _MASK_OUTSIDE_PENALTY_WEIGHT
        score += mask_penalty
        meta["speech_mask_inside_ratio"] = round(float(inside_ratio), 4)
        meta["speech_mask_outside_penalty"] = round(float(mask_penalty), 4)

    shape_center = _point_from_value(speech_visual_center)
    safe_box = _bbox_from_value(speech_safe_box) or _bbox_from_value(mask_origin_box)
    if measured_center and shape_center and safe_box:
        shape_dx = (float(measured_center[0]) - float(shape_center[0])) / max(
            1.0, float(safe_box[2])
        )
        shape_dy = (float(measured_center[1]) - float(shape_center[1])) / max(
            1.0, float(safe_box[3])
        )
        shape_distance = shape_dx * shape_dx + shape_dy * shape_dy
        shape_penalty = shape_distance * _SHAPE_BALANCE_PENALTY_WEIGHT
        score += shape_penalty
        meta["speech_visual_center_distance"] = round(float(math.sqrt(shape_distance)), 4)
        meta["speech_visual_center_penalty"] = round(float(shape_penalty), 4)

    anchor_center = _point_from_value(alignment_center)
    if measured_center and anchor_center:
        target_box = _bbox_from_value(target)
        normalizer_x = max(
            1.0,
            float(target_box[2]) if target_box else 0.0,
            font_size * 2.0,
        )
        normalizer_y = max(
            1.0,
            float(target_box[3]) if target_box else 0.0,
            font_size * 2.0,
        )
        alignment_dx = (
            float(measured_center[0]) - float(anchor_center[0])
        ) / normalizer_x
        alignment_dy = (
            float(measured_center[1]) - float(anchor_center[1])
        ) / normalizer_y
        alignment_distance = (
            alignment_dx * alignment_dx + alignment_dy * alignment_dy
        )
        alignment_penalty = (
            alignment_distance * _ALIGNMENT_CENTER_PENALTY_WEIGHT
        )
        score += alignment_penalty
        meta["alignment_center_distance"] = round(
            float(math.sqrt(alignment_distance)),
            4,
        )
        meta["alignment_center_penalty"] = round(
            float(alignment_penalty),
            4,
        )

    return score, meta


def _speech_bubble_geometry_from_page(page, plan: RenderLayerPlan, candidate: Sequence[int]) -> dict[str, Any]:
    candidate_box = _bbox_from_value(candidate)
    if not candidate_box:
        return {"audit": {"applied": False, "reason": "missing_candidate_box"}}
    x, y, w, h = candidate_box
    if w <= 4 or h <= 4:
        return {"audit": {"applied": False, "reason": "candidate_box_too_small", "candidate_box": list(candidate_box)}}
    crop = page.crop((x, y, x + w, y + h)).convert("RGB")
    arr = np.asarray(crop)
    if arr.size == 0:
        return {"audit": {"applied": False, "reason": "empty_candidate_crop", "candidate_box": list(candidate_box)}}
    gray = (
        arr[:, :, 0].astype("float32") * 0.299
        + arr[:, :, 1].astype("float32") * 0.587
        + arr[:, :, 2].astype("float32") * 0.114
    )
    threshold = float(np.percentile(gray, 80)) - 8.0
    threshold = max(210.0, min(245.0, threshold))
    white_mask = gray >= threshold
    if int(white_mask.sum()) < max(16, int(w * h * 0.04)):
        return {
            "audit": {
                "applied": False,
                "reason": "no_speech_interior_component",
                "candidate_box": list(candidate_box),
                "white_threshold": round(threshold, 3),
            }
        }

    anchor = _shape_anchor(plan, candidate_box)
    component = _connected_component_near_anchor(white_mask, anchor)
    if component is None or int(component.sum()) < max(16, int(w * h * 0.04)):
        return {
            "audit": {
                "applied": False,
                "reason": "no_anchor_connected_speech_component",
                "candidate_box": list(candidate_box),
                "anchor": [round(float(anchor[0]), 3), round(float(anchor[1]), 3)],
                "white_threshold": round(threshold, 3),
            }
        }

    margin = _shape_margin(plan, candidate_box)
    safe_mask = _erode_component(component, margin)
    if safe_mask is None or int(safe_mask.sum()) < max(16, int(component.sum() * 0.18)):
        safe_mask = component
    component_box = _mask_bbox(component)
    local_box = _mask_bbox(component) or _mask_bbox(safe_mask)
    if not local_box:
        return {
            "audit": {
                "applied": False,
                "reason": "missing_safe_component_box",
                "candidate_box": list(candidate_box),
                "component_pixels": int(component.sum()),
            }
        }

    safe_box = [x + local_box[0], y + local_box[1], local_box[2], local_box[3]]
    safe_box = _inset_box(safe_box, margin=max(2, min(margin, 10)))
    safe_box = _intersect_box(safe_box, candidate_box)
    if not safe_box or safe_box[2] < 8 or safe_box[3] < 8:
        return {
            "audit": {
                "applied": False,
                "reason": "safe_box_too_small_after_margin",
                "candidate_box": list(candidate_box),
                "component_box": _local_to_page_box(component_box, candidate_box),
                "margin": margin,
            }
        }

    visual_center_evidence = _speech_visual_center(safe_mask, candidate_box)
    original = _bbox_from_value(plan.target_box)
    audit = {
        "applied": True,
        "source": "cleaned_page_speech_bubble_interior",
        "candidate_box": list(candidate_box),
        "original_target_box": list(original),
        "box": list(safe_box),
        "component_box": _local_to_page_box(component_box, candidate_box),
        "white_threshold": round(threshold, 3),
        "component_pixels": int(component.sum()),
        "safe_pixels": int(safe_mask.sum()) if safe_mask is not None else int(component.sum()),
        "margin": int(margin),
        "anchor": [round(float(anchor[0]), 3), round(float(anchor[1]), 3)],
        "box_source": "connected_speech_component_bbox",
        "visual_center": copy_jsonish(visual_center_evidence.get("center") or []),
        "visual_center_policy": str(visual_center_evidence.get("policy") or ""),
        "visual_center_max_clearance": visual_center_evidence.get("max_clearance"),
    }
    return {
        "audit": audit,
        "candidate_box": list(candidate_box),
        "component": component,
        "safe_mask": safe_mask,
        "component_box_local": component_box,
        "anchor": anchor,
        "visual_center_evidence": visual_center_evidence,
    }


def _is_shape_aware_speech_layer(plan: RenderLayerPlan) -> bool:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    text = " ".join(
        str(value or "").lower()
        for value in (
            plan.role,
            plan.state,
            style.get("semantic_class"),
            style.get("semantic_kind"),
            style.get("source_role"),
            style.get("route_intent"),
        )
    )
    if "caption" in text or "background" in text:
        return False
    return "speech" in text or "bubble" in text or str(plan.role or "").lower() == "speech"


def _is_latin_shape_band_layer(plan: RenderLayerPlan) -> bool:
    """Return whether one plan explicitly selects the English v2 strategy."""

    style = (
        plan.resolved_render_style
        if isinstance(plan.resolved_render_style, Mapping)
        else {}
    )
    presentation = (
        style.get("target_presentation_policy")
        if isinstance(style.get("target_presentation_policy"), Mapping)
        else {}
    )
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    policy_id = str(
        presentation.get("policy_id")
        or metadata.get("target_presentation_policy_id")
        or ""
    )
    writing_mode = str(
        plan.writing_mode
        or style.get("writing_mode")
        or ""
    ).strip().lower()
    return bool(
        policy_id == "target-presentation:en:v2"
        and str(style.get("target_script") or "") == "Latn"
        and writing_mode == "horizontal"
    )


def _build_horizontal_shape_capacity_profile(
    page,
    plan: RenderLayerPlan,
    candidate_box: Sequence[int],
) -> HorizontalShapeCapacityProfile | None:
    """Build one deterministic pre-effect capacity profile for Latin text."""

    if np is None:
        return None
    bounds = _bbox_from_value(candidate_box)
    if not bounds:
        return None
    page_box = [0, 0, int(page.size[0]), int(page.size[1])]
    bounds = _intersect_box(bounds, page_box)
    if not bounds:
        return None
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    domain = (
        metadata.get("parent_render_domain")
        if isinstance(metadata.get("parent_render_domain"), Mapping)
        else {}
    )
    frame = (
        metadata.get("oriented_layout_frame")
        if isinstance(metadata.get("oriented_layout_frame"), Mapping)
        else {}
    )
    container_id = str(domain.get("container_id") or "")
    exact_domain = bool(
        str(domain.get("policy_id") or "") == "target-presentation:en:v2"
        and str(domain.get("status") or "")
        == "authorized_shape_safe_speech_container"
        and str(domain.get("container_type") or "") == "speech_bubble"
        and str(domain.get("container_authorization_state") or "")
        == "cleanup_translate_speech"
        and not list(domain.get("container_conflict_flags") or [])
    )
    if not exact_domain:
        return None
    polygon = []
    source = ""
    reason_codes: list[str] = []
    if (
        exact_domain
        and str(frame.get("status") or "") == "supported"
        and str(frame.get("coordinate_space") or "") == "page"
        and str(frame.get("container_id") or "") == container_id
        and str(frame.get("container_type") or "") == "speech_bubble"
    ):
        polygon = _capacity_polygon(frame.get("polygon"))
        if polygon:
            source = "exact_text_area_plan_speech_polygon"
            reason_codes.append("exact_oriented_speech_polygon_capacity")
    if exact_domain and not polygon:
        polygon = _capacity_polygon(domain.get("container_polygon"))
        if polygon:
            source = "exact_text_area_plan_speech_polygon"
            reason_codes.append("exact_speech_polygon_capacity")

    rotation_degrees = 0.0
    inverse_rotation_applied = False
    if polygon:
        effects = resolve_parent_layer_effects(plan.resolved_render_style)
        if (
            effects.rotation.availability == "resolved"
            and abs(float(effects.rotation.degrees_clockwise)) >= 1e-9
        ):
            rotation_degrees = float(effects.rotation.degrees_clockwise)
            pivot = _point_from_value(frame.get("center_page"))
            if not pivot:
                pivot = _point_from_value(metadata.get("visual_alignment_center"))
            if not pivot:
                pivot = _center_box(bounds)
            polygon = [
                _rotate_capacity_point(
                    point,
                    pivot,
                    -rotation_degrees,
                )
                for point in polygon
            ]
            inverse_rotation_applied = True
            reason_codes.append("inverse_parent_rotation_applied")
        raw_mask = _polygon_capacity_mask(polygon, bounds)
        if raw_mask is None or int(raw_mask.sum()) <= 0:
            return None
        actual_mask = _erode_component(raw_mask, 2)
        margin = _shape_margin(plan, bounds)
        comfort_mask = _erode_component(actual_mask, margin)
        source_sha256 = _capacity_polygon_sha256(polygon)
    else:
        geometry = _speech_bubble_geometry_from_page(page, plan, bounds)
        actual_mask = geometry.get("component")
        comfort_mask = geometry.get("safe_mask")
        audit = (
            geometry.get("audit")
            if isinstance(geometry.get("audit"), Mapping)
            else {}
        )
        if (
            not bool(audit.get("applied"))
            or actual_mask is None
            or int(actual_mask.sum()) <= 0
        ):
            return None
        if comfort_mask is None or int(comfort_mask.sum()) <= 0:
            comfort_mask = actual_mask
        margin = int(audit.get("margin") or _shape_margin(plan, bounds))
        source = "cleaned_page_speech_interior_fallback"
        source_sha256 = _capacity_mask_sha256(actual_mask)
        reason_codes.append("cleaned_page_speech_interior_capacity_fallback")

    rows = tuple(
        HorizontalCapacityRow(
            y=int(bounds[1] + local_y),
            actual_intervals=_capacity_row_intervals(
                actual_mask,
                local_y,
                origin_x=bounds[0],
            ),
            comfort_intervals=_capacity_row_intervals(
                comfort_mask,
                local_y,
                origin_x=bounds[0],
            ),
        )
        for local_y in range(int(bounds[3]))
    )
    if not any(row.actual_intervals for row in rows):
        return None
    visual_center = _point_from_value(frame.get("center_page"))
    if inverse_rotation_applied and visual_center:
        visual_center = _rotate_capacity_point(
            visual_center,
            visual_center,
            -rotation_degrees,
        )
    if not visual_center:
        visual_center = _point_from_value(
            _speech_visual_center(actual_mask, bounds).get("center")
        )
    if not visual_center:
        visual_center = _center_box(bounds)
    alignment_center = _point_from_value(metadata.get("visual_alignment_center"))
    if not alignment_center or not _point_inside_box(alignment_center, bounds):
        alignment_center = list(visual_center)
    return HorizontalShapeCapacityProfile(
        profile_version=HORIZONTAL_SHAPE_CAPACITY_PROFILE_VERSION,
        strategy_id="latin_horizontal_shape_bands_v1",
        source=source,
        source_sha256=source_sha256,
        actual_mask_sha256=_capacity_mask_sha256(actual_mask),
        comfort_mask_sha256=_capacity_mask_sha256(comfort_mask),
        bounds=tuple(bounds[:4]),
        rows=rows,
        visual_center=(float(visual_center[0]), float(visual_center[1])),
        alignment_center=(
            float(alignment_center[0]),
            float(alignment_center[1]),
        ),
        rotation_degrees_clockwise=rotation_degrees,
        inverse_rotation_applied=inverse_rotation_applied,
        margin_px=margin,
        reason_codes=tuple(reason_codes),
    )


def _latin_shape_band_planned_result(
    page,
    plan: RenderLayerPlan,
    typesetting_engine: TypesettingEngine,
) -> PlannedLayerResult | None:
    page_box = [0, 0, int(page.size[0]), int(page.size[1])]
    candidate = _latin_shape_capacity_candidate_box(plan, page_box)
    if not candidate:
        return None
    profile = _build_horizontal_shape_capacity_profile(page, plan, candidate)
    if not isinstance(profile, HorizontalShapeCapacityProfile):
        return None
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    profile_audit = profile.to_audit_dict()
    metadata["horizontal_shape_capacity_profile"] = profile_audit
    metadata["shape_aware_composition"] = {
        "applied": True,
        "source": "latin_horizontal_shape_capacity_profile",
        "box": list(profile.bounds),
        "margin": int(profile.margin_px),
        "profile_digest": str(profile.profile_digest),
        "reason_codes": list(profile.reason_codes),
    }
    clipping = (
        copy_jsonish(plan.clipping_region_ref)
        if isinstance(plan.clipping_region_ref, Mapping)
        else {}
    )
    clipping["horizontal_shape_capacity_bounds"] = list(profile.bounds)
    profile_plan = replace(
        plan,
        target_box=list(profile.bounds),
        metadata=metadata,
        clipping_region_ref=clipping,
        horizontal_shape_capacity_profile=profile,
    )
    layout, report = typesetting_engine.typeset_layer(profile_plan)
    capacity_layout = (
        layout.metadata.get("horizontal_shape_capacity")
        if isinstance(layout.metadata, Mapping)
        and isinstance(layout.metadata.get("horizontal_shape_capacity"), Mapping)
        else {}
    )
    final_plan = _plan_with_visual_slot_audit(
        profile_plan,
        {
            "applied": True,
            "source": "latin_shape_band_capacity_v1",
            "selected_source": str(profile.source),
            "selected_box": list(profile.bounds),
            "selected_profile_digest": str(profile.profile_digest),
            "selected_line_boxes": copy_jsonish(
                capacity_layout.get("selected_line_boxes") or []
            ),
            "selected_line_widths": copy_jsonish(
                capacity_layout.get("selected_line_widths") or []
            ),
            "fit_status": str(report.fit_status or ""),
            "full_text_placed": bool(report.full_text_placed),
            "candidate_count": int(capacity_layout.get("candidate_count") or 0),
            "fitting_candidate_count": int(
                capacity_layout.get("fitting_candidate_count") or 0
            ),
            "readability_is_render_admission": False,
        },
    )
    return PlannedLayerResult(
        plan=final_plan,
        layout=layout,
        fit_report=report,
    )


def _latin_shape_capacity_candidate_box(
    plan: RenderLayerPlan,
    page_box: Sequence[int],
) -> list[int]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    effective = (
        metadata.get("effective_render_plan")
        if isinstance(metadata.get("effective_render_plan"), Mapping)
        else {}
    )
    authorities = (
        effective.get("field_authority")
        if isinstance(effective.get("field_authority"), Mapping)
        else {}
    )
    domain = (
        metadata.get("parent_render_domain")
        if isinstance(metadata.get("parent_render_domain"), Mapping)
        else {}
    )
    values = (
        (plan.target_box,)
        if str(authorities.get("target_box") or "") == "user"
        else (
            domain.get("automatic_bounds"),
            plan.hard_bounds,
            plan.target_box,
        )
    )
    for value in values:
        box = _intersect_box(_bbox_from_value(value), page_box)
        if box:
            return box
    return []


def _capacity_polygon(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, str):
            parts = item.replace(",", " ").split()
            if len(parts) < 2:
                return []
            raw = parts[:2]
        elif (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) >= 2
        ):
            raw = item[:2]
        else:
            return []
        try:
            point = [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            return []
        if not all(math.isfinite(number) for number in point):
            return []
        points.append(point)
    return points if len(points) >= 3 else []


def _rotate_capacity_point(
    point: Sequence[float],
    pivot: Sequence[float],
    degrees_clockwise: float,
) -> list[float]:
    radians = math.radians(float(degrees_clockwise))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    dx = float(point[0]) - float(pivot[0])
    dy = float(point[1]) - float(pivot[1])
    return [
        float(pivot[0]) + cosine * dx - sine * dy,
        float(pivot[1]) + sine * dx + cosine * dy,
    ]


def _polygon_capacity_mask(
    polygon: Sequence[Sequence[float]],
    bounds: Sequence[int],
):
    box = _bbox_from_value(bounds)
    points = _capacity_polygon(polygon)
    if np is None or not box or not points:
        return None
    local = [
        (
            int(round(float(point[0]) - float(box[0]))),
            int(round(float(point[1]) - float(box[1]))),
        )
        for point in points
    ]
    if cv2 is not None:
        mask = np.zeros((int(box[3]), int(box[2])), dtype="uint8")
        cv2.fillPoly(mask, [np.asarray(local, dtype="int32")], 1)
        return mask.astype(bool)
    if Image is None or ImageDraw is None:
        return None
    image = Image.new("L", (int(box[2]), int(box[3])), 0)
    ImageDraw.Draw(image).polygon(local, fill=255)
    return np.asarray(image, dtype="uint8") > 0


def _capacity_row_intervals(
    mask,
    local_y: int,
    *,
    origin_x: int,
) -> tuple[tuple[int, int], ...]:
    if (
        np is None
        or mask is None
        or local_y < 0
        or local_y >= int(mask.shape[0])
    ):
        return ()
    xs = np.flatnonzero(mask[int(local_y)])
    if len(xs) == 0:
        return ()
    intervals: list[tuple[int, int]] = []
    start = int(xs[0])
    prior = start
    for raw in xs[1:]:
        current = int(raw)
        if current != prior + 1:
            intervals.append((int(origin_x + start), int(origin_x + prior + 1)))
            start = current
        prior = current
    intervals.append((int(origin_x + start), int(origin_x + prior + 1)))
    return tuple(intervals)


def _capacity_polygon_sha256(polygon: Sequence[Sequence[float]]) -> str:
    encoded = json.dumps(
        [
            [round(float(point[0]), 6), round(float(point[1]), 6)]
            for point in polygon
        ],
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _capacity_mask_sha256(mask) -> str:
    if np is None or mask is None:
        return hashlib.sha256(b"").hexdigest()
    value = np.asarray(mask, dtype="uint8")
    header = f"{value.shape[0]}x{value.shape[1]}:".encode("ascii")
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _point_inside_box(point: Sequence[float], box: Sequence[int]) -> bool:
    bbox = _bbox_from_value(box)
    if not bbox or len(point) < 2:
        return False
    return bool(
        float(bbox[0]) <= float(point[0]) <= float(bbox[0] + bbox[2])
        and float(bbox[1]) <= float(point[1]) <= float(bbox[1] + bbox[3])
    )


def _shape_candidate_box(plan: RenderLayerPlan, page_box: Sequence[int]) -> list[int]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    slot = metadata.get("parent_render_slot") if isinstance(metadata.get("parent_render_slot"), Mapping) else {}
    boxes = (
        plan.hard_bounds,
        plan.target_box,
        slot.get("hard_bounds") if isinstance(slot, Mapping) else [],
        slot.get("box") if isinstance(slot, Mapping) else [],
    )
    for value in boxes:
        box = _intersect_box(_bbox_from_value(value), page_box)
        if box:
            return box
    return []


def _speech_bubble_safe_box_from_page(page, plan: RenderLayerPlan, candidate: Sequence[int]) -> dict[str, Any]:
    geometry = _speech_bubble_geometry_from_page(page, plan, candidate)
    audit = geometry.get("audit") if isinstance(geometry.get("audit"), Mapping) else {}
    return copy_jsonish(audit)


def _shape_anchor(plan: RenderLayerPlan, candidate_box: Sequence[int]) -> tuple[float, float]:
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    slot = metadata.get("parent_render_slot") if isinstance(metadata.get("parent_render_slot"), Mapping) else {}
    for value in (
        metadata.get("visual_alignment_center"),
        slot.get("alignment_anchor_center") if isinstance(slot, Mapping) else [],
        _center_box(plan.target_box),
    ):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
            try:
                ax = float(value[0]) - float(candidate_box[0])
                ay = float(value[1]) - float(candidate_box[1])
                return (
                    max(0.0, min(float(candidate_box[2] - 1), ax)),
                    max(0.0, min(float(candidate_box[3] - 1), ay)),
                )
            except Exception:
                continue
    return (float(candidate_box[2]) / 2.0, float(candidate_box[3]) / 2.0)


def _connected_component_near_anchor(mask, anchor: tuple[float, float]):
    if np is None:
        return None
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    ax = int(round(max(0.0, min(float(mask.shape[1] - 1), anchor[0]))))
    ay = int(round(max(0.0, min(float(mask.shape[0] - 1), anchor[1]))))
    if not bool(mask[ay, ax]):
        distances = (xs.astype("float32") - float(ax)) ** 2 + (ys.astype("float32") - float(ay)) ** 2
        nearest = int(np.argmin(distances))
        ax = int(xs[nearest])
        ay = int(ys[nearest])
    if cv2 is not None:
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
        if count <= 1:
            return None
        label = int(labels[ay, ax])
        if label <= 0:
            return None
        return labels == label
    return _flood_fill_component(mask, ax, ay)


def _flood_fill_component(mask, ax: int, ay: int):
    from collections import deque

    h, w = mask.shape[:2]
    if ax < 0 or ay < 0 or ax >= w or ay >= h or not bool(mask[ay, ax]):
        return None
    out = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(ax, ay)])
    out[ay, ax] = True
    while queue:
        x, y = queue.popleft()
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if out[ny, nx] or not bool(mask[ny, nx]):
                    continue
                out[ny, nx] = True
                queue.append((nx, ny))
    return out


def _erode_component(component, margin: int):
    if component is None:
        return None
    amount = max(1, min(10, int(margin)))
    if cv2 is None:
        return component
    kernel = np.ones((amount * 2 + 1, amount * 2 + 1), dtype="uint8")
    eroded = cv2.erode(component.astype("uint8"), kernel, iterations=1).astype(bool)
    return eroded if int(eroded.sum()) > 0 else component


def _speech_visual_center(mask, candidate_box: Sequence[int]) -> dict[str, Any]:
    box = _bbox_from_value(candidate_box)
    if np is None or mask is None or not box or getattr(mask, "size", 0) == 0:
        return {"center": [], "policy": "unavailable", "max_clearance": None}
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"center": [], "policy": "empty_speech_interior", "max_clearance": None}

    policy = "speech_interior_mask_centroid"
    weights = np.ones(len(xs), dtype="float64")
    max_clearance: float | None = None
    if cv2 is not None:
        distance = cv2.distanceTransform(mask.astype("uint8"), cv2.DIST_L2, 5)
        sampled = distance[ys, xs].astype("float64")
        if sampled.size and float(sampled.max()) > 0.0:
            weights = np.square(sampled)
            max_clearance = round(float(sampled.max()), 4)
            policy = "distance_weighted_speech_interior"
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.ones(len(xs), dtype="float64")
    center = [
        float(box[0]) + float(np.average(xs.astype("float64"), weights=weights)),
        float(box[1]) + float(np.average(ys.astype("float64"), weights=weights)),
    ]
    return {
        "center": _round_point(center),
        "policy": policy,
        "max_clearance": max_clearance,
        "mask_pixel_count": int(len(xs)),
    }


def _coverage_core_box(mask, component_box: Sequence[int], anchor: tuple[float, float]) -> list[int]:
    box = _bbox_from_value(component_box)
    if not box:
        return []
    x, y, w, h = box
    sub = mask[y : y + h, x : x + w]
    if sub.size == 0:
        return []
    col_cov = sub.sum(axis=0).astype("float32") / max(1.0, float(h))
    row_cov = sub.sum(axis=1).astype("float32") / max(1.0, float(w))
    col_keep = col_cov >= 0.46
    row_keep = row_cov >= 0.40
    ax = int(round(float(anchor[0]) - float(x)))
    ay = int(round(float(anchor[1]) - float(y)))
    x0, x1 = _contiguous_true_range(col_keep, ax)
    y0, y1 = _contiguous_true_range(row_keep, ay)
    if x1 <= x0 or y1 <= y0:
        return []
    core = [x + x0, y + y0, x1 - x0, y1 - y0]
    if core[2] < max(12, int(w * 0.35)) or core[3] < max(12, int(h * 0.35)):
        return []
    return core


def _contiguous_true_range(values, anchor: int) -> tuple[int, int]:
    length = int(len(values))
    if length <= 0:
        return (0, 0)
    anchor = max(0, min(length - 1, int(anchor)))
    if not bool(values[anchor]):
        true_indices = [idx for idx, value in enumerate(values) if bool(value)]
        if not true_indices:
            return (0, 0)
        anchor = min(true_indices, key=lambda idx: abs(idx - anchor))
    start = anchor
    while start > 0 and bool(values[start - 1]):
        start -= 1
    end = anchor + 1
    while end < length and bool(values[end]):
        end += 1
    return (start, end)


def _mask_bbox(mask) -> list[int]:
    if np is None or mask is None:
        return []
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _shape_margin(plan: RenderLayerPlan, box: Sequence[int]) -> int:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    font_size = _safe_int(style.get("font_size") or style.get("font_size_hint"), default=24)
    bbox = _bbox_from_value(box)
    short_side = min(bbox[2], bbox[3]) if bbox else font_size * 3
    return max(3, min(14, int(round(min(font_size * 0.22, short_side * 0.08)))))


def _center_box(value: Any) -> list[float]:
    box = _bbox_from_value(value)
    if not box:
        return []
    return [float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0]


def _point_from_value(value: Any) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 2
    ):
        return []
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return []


def _round_point(value: Sequence[float]) -> list[float]:
    point = _point_from_value(value)
    return [round(float(point[0]), 4), round(float(point[1]), 4)] if point else []


def _point_inside_box(point: Sequence[float], box: Sequence[int]) -> bool:
    center = _point_from_value(point)
    bounds = _bbox_from_value(box)
    if not center or not bounds:
        return False
    return (
        float(bounds[0]) <= float(center[0]) <= float(bounds[0] + bounds[2])
        and float(bounds[1]) <= float(center[1]) <= float(bounds[1] + bounds[3])
    )


def _clamp_point_to_box(point: Sequence[float], box: Sequence[int]) -> list[float]:
    center = _point_from_value(point)
    bounds = _bbox_from_value(box)
    if not center or not bounds:
        return []
    return _round_point(
        [
            max(float(bounds[0]), min(float(bounds[0] + bounds[2]), float(center[0]))),
            max(float(bounds[1]), min(float(bounds[1] + bounds[3]), float(center[1]))),
        ]
    )


def _local_to_page_box(local: Sequence[int], candidate: Sequence[int]) -> list[int]:
    box = _bbox_from_value(local)
    base = _bbox_from_value(candidate)
    if not box or not base:
        return []
    return [base[0] + box[0], base[1] + box[1], box[2], box[3]]


def _inset_box(box: Sequence[int], *, margin: int) -> list[int]:
    bbox = _bbox_from_value(box)
    if not bbox:
        return []
    x, y, w, h = bbox
    inset = max(0, int(margin))
    if w - inset * 2 < 8 or h - inset * 2 < 8:
        return bbox
    return [x + inset, y + inset, w - inset * 2, h - inset * 2]


def _intersect_box(box: Sequence[int], container: Sequence[int]) -> list[int]:
    b = _bbox_from_value(box)
    c = _bbox_from_value(container)
    if not b or not c:
        return []
    x0 = max(b[0], c[0])
    y0 = max(b[1], c[1])
    x1 = min(b[0] + b[2], c[0] + c[2])
    y1 = min(b[1] + b[3], c[1] + c[3])
    if x1 <= x0 or y1 <= y0:
        return []
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _source_contract_box(plan: RenderLayerPlan) -> list[int]:
    provenance = plan.source_provenance_ref if isinstance(plan.source_provenance_ref, Mapping) else {}
    return _bbox_from_value(
        provenance.get("source_contract_bbox")
        if isinstance(provenance, Mapping)
        else []
    )


def _source_text_footprint_alignment_box(plan: RenderLayerPlan) -> list[int]:
    footprint = validated_source_text_footprint_ref(plan)
    return _bbox_from_value(footprint.get("union_bbox_page_xywh"))


def _alignment_anchor_box(
    plan: RenderLayerPlan,
    *,
    preferred_kind: str = "",
) -> tuple[list[int], str]:
    candidates = {
        "source_text_footprint_union_bbox": (
            _source_text_footprint_alignment_box(plan)
        ),
        "source_contract_bbox": _source_contract_box(plan),
        "parent_contract_bbox": _parent_anchor_box(plan),
    }
    if preferred_kind in candidates and candidates[preferred_kind]:
        return list(candidates[preferred_kind]), preferred_kind
    for kind in (
        "source_text_footprint_union_bbox",
        "source_contract_bbox",
        "parent_contract_bbox",
    ):
        if candidates[kind]:
            return list(candidates[kind]), kind
    return [], "unavailable"


def _same_box(first: Sequence[int], second: Sequence[int]) -> bool:
    a = _bbox_from_value(first)
    b = _bbox_from_value(second)
    return bool(a and b and a == b)


def _box_inside_tolerant(box: Sequence[int], container: Sequence[int], *, tolerance: int = 0) -> bool:
    b = _bbox_from_value(box)
    c = _bbox_from_value(container)
    if not b or not c:
        return False
    tol = max(0, int(tolerance))
    return (
        b[0] >= c[0] - tol
        and b[1] >= c[1] - tol
        and b[0] + b[2] <= c[0] + c[2] + tol
        and b[1] + b[3] <= c[1] + c[3] + tol
    )


def _mask_coverage_ratio(
    box: Sequence[int],
    mask,
    mask_origin_box: Sequence[int],
) -> float:
    measured = _bbox_from_value(box)
    origin = _bbox_from_value(mask_origin_box)
    if np is None or mask is None or not measured or not origin or getattr(mask, "size", 0) == 0:
        return 0.0
    mx, my, mw, mh = measured
    ox, oy, ow, oh = origin
    x0 = max(0, int(mx - ox))
    y0 = max(0, int(my - oy))
    x1 = min(int(ow), int(mx + mw - ox), int(mask.shape[1]))
    y1 = min(int(oh), int(my + mh - oy), int(mask.shape[0]))
    inside = int(mask[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0
    return max(0.0, min(1.0, float(inside) / max(1.0, float(mw * mh))))


def _area(box: Sequence[int]) -> int:
    bbox = _bbox_from_value(box)
    if not bbox:
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_from_value(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        for key in ("bbox", "box", "target_box", "hard_bounds", "render_allowed_area"):
            box = _bbox_from_value(value.get(key))
            if box:
                return box
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[int] = []
        for item in list(value)[:4]:
            try:
                items.append(int(round(float(item))))
            except Exception:
                return []
        if len(items) == 4 and items[2] > 0 and items[3] > 0:
            return items
    return []


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default
