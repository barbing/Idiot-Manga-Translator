# -*- coding: utf-8 -*-
"""Bounded post-typeset fitting from exact shaped-ink evidence.

This Stage 4.5 planner may translate an intact TypesetLayout inside its existing
RenderLayerPlan hard bounds. It never changes font size, writing mode, breaks,
relative glyph geometry, or parent identity.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.render.typesetting_contracts import FitReport, GlyphPlacement, RenderLayerPlan, TypesetLayout


INK_BOUND_LAYOUT_FITTER_VERSION = "ink_bound_layout_fitter_v1"
INK_BOUND_FIT_POLICY = "complete_shaped_ink_block_translation_v1"


@dataclass(frozen=True)
class InkBoundFitResult:
    layout: TypesetLayout
    report: FitReport
    audit: dict[str, Any]

    @property
    def applied(self) -> bool:
        return bool(self.audit.get("status") == "shifted")


class InkBoundLayoutFitter:
    """Translate a complete raster-evidenced block by the smallest valid delta."""

    version = INK_BOUND_LAYOUT_FITTER_VERSION

    def fit(
        self,
        plan: RenderLayerPlan,
        layout: TypesetLayout,
        report: FitReport,
        raster_placements: Sequence[Mapping[str, Any]] | None,
    ) -> InkBoundFitResult:
        hard_bounds = _xywh_to_xyxy(plan.hard_bounds or plan.target_box)
        records = [item for item in list(raster_placements or []) if isinstance(item, Mapping)]
        triggered = [item for item in records if _is_hard_bound_failure(item)]
        ink_boxes = [box for item in records if (box := _ink_box(item))]
        ink_union = _union_xyxy(ink_boxes)
        base_audit: dict[str, Any] = {
            "ink_bound_layout_fitter_version": self.version,
            "policy": INK_BOUND_FIT_POLICY,
            "policy_owner": "ink_bound_layout_fitter",
            "status": "not_required",
            "triggered_failure_count": len(triggered),
            "evidence_placement_count": len(ink_boxes),
            "parent_hard_bounds": list(hard_bounds),
            "input_ink_bounds": list(ink_union),
            "allowed_shift_x": [],
            "allowed_shift_y": [],
            "selected_shift": [0, 0],
            "output_ink_bounds": list(ink_union),
            "relative_geometry_preserved": True,
            "font_size_changed": False,
            "breaks_changed": False,
            "writing_mode_changed": False,
            "issues": [],
        }
        if not triggered:
            return InkBoundFitResult(layout=layout, report=report, audit=base_audit)
        if not hard_bounds:
            return InkBoundFitResult(
                layout=layout,
                report=report,
                audit=_failed_audit(base_audit, "parent_hard_bounds_missing"),
            )
        if not ink_union:
            return InkBoundFitResult(
                layout=layout,
                report=report,
                audit=_failed_audit(base_audit, "complete_ink_union_missing"),
            )

        min_dx = int(hard_bounds[0] - ink_union[0])
        max_dx = int(hard_bounds[2] - ink_union[2])
        min_dy = int(hard_bounds[1] - ink_union[1])
        max_dy = int(hard_bounds[3] - ink_union[3])
        base_audit["allowed_shift_x"] = [min_dx, max_dx]
        base_audit["allowed_shift_y"] = [min_dy, max_dy]
        if min_dx > max_dx or min_dy > max_dy:
            return InkBoundFitResult(
                layout=layout,
                report=report,
                audit=_failed_audit(base_audit, "complete_ink_union_exceeds_parent_hard_bounds"),
            )

        dx = _closest_to_zero(min_dx, max_dx)
        dy = _closest_to_zero(min_dy, max_dy)
        output_ink = _shift_xyxy(ink_union, dx, dy)
        base_audit["selected_shift"] = [dx, dy]
        base_audit["output_ink_bounds"] = output_ink
        if not _contains(hard_bounds, output_ink):
            return InkBoundFitResult(
                layout=layout,
                report=report,
                audit=_failed_audit(base_audit, "selected_shift_does_not_contain_complete_ink"),
            )
        if dx == 0 and dy == 0:
            return InkBoundFitResult(
                layout=layout,
                report=report,
                audit=_failed_audit(base_audit, "hard_bound_failure_without_resolving_translation"),
            )

        base_audit["status"] = "shifted"
        shifted_layout = _shift_layout(layout, dx, dy, base_audit)
        shifted_report = _annotate_report(report, base_audit)
        return InkBoundFitResult(
            layout=shifted_layout,
            report=shifted_report,
            audit=base_audit,
        )


def _is_hard_bound_failure(item: Mapping[str, Any]) -> bool:
    containment = item.get("hard_bound_containment")
    if isinstance(containment, Mapping) and not bool(containment.get("accepted")):
        return True
    return "raster_ink_exceeds_parent_hard_bounds" in {
        str(issue) for issue in list(item.get("issues") or [])
    }


def _ink_box(item: Mapping[str, Any]) -> list[int]:
    if str(item.get("status") or "") == "no_ink":
        return []
    composite = _xyxy(item.get("composite_bounds"))
    if composite:
        return composite
    if str(item.get("status") or "") == "primitive":
        return _xyxy(item.get("placement_bbox"))
    return []


def _shift_layout(
    layout: TypesetLayout,
    dx: int,
    dy: int,
    fit_audit: Mapping[str, Any],
) -> TypesetLayout:
    glyphs = [_shift_glyph(item, dx, dy) for item in list(layout.glyphs or [])]
    columns = [_shift_coordinate_record(item, dx, dy) for item in list(layout.columns or [])]
    lines = [_shift_coordinate_record(item, dx, dy) for item in list(layout.lines or [])]
    measured_bounds = _shift_xywh(layout.measured_bounds, dx, dy)
    visual_center = list(layout.visual_center or [])
    if len(visual_center) >= 2:
        visual_center = [float(visual_center[0]) + dx, float(visual_center[1]) + dy]
    metadata = deepcopy(dict(layout.metadata or {}))
    metadata["ink_bound_fit"] = deepcopy(dict(fit_audit))
    return replace(
        layout,
        glyphs=glyphs,
        columns=columns,
        lines=lines,
        measured_bounds=measured_bounds,
        visual_center=visual_center,
        metadata=metadata,
    )


def _shift_glyph(value: GlyphPlacement | Mapping[str, Any], dx: int, dy: int):
    if isinstance(value, GlyphPlacement):
        metadata = deepcopy(dict(value.metadata or {}))
        metadata["ink_bound_fit_shift"] = [dx, dy]
        return replace(
            value,
            bbox=_shift_xywh(value.bbox, dx, dy),
            position=_shift_point(value.position, dx, dy),
            metadata=metadata,
        )
    if isinstance(value, Mapping):
        item = deepcopy(dict(value))
        item["bbox"] = _shift_xywh(item.get("bbox"), dx, dy)
        item["position"] = _shift_point(item.get("position"), dx, dy)
        metadata = deepcopy(dict(item.get("metadata") or {}))
        metadata["ink_bound_fit_shift"] = [dx, dy]
        item["metadata"] = metadata
        return item
    return value


def _shift_coordinate_record(value: Any, dx: int, dy: int) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    item = deepcopy(dict(value))
    for key in ("x", "raw_x"):
        if item.get(key) is not None:
            item[key] = _shift_scalar(item.get(key), dx)
    for key in ("y", "raw_y"):
        if item.get(key) is not None:
            item[key] = _shift_scalar(item.get(key), dy)
    for key in ("bbox", "box", "display_box", "raw_box", "centered_block_box", "measured_bounds"):
        if key in item:
            item[key] = _shift_xywh(item.get(key), dx, dy)
    alignment = item.get("layout_visual_alignment")
    if isinstance(alignment, Mapping):
        adjusted = deepcopy(dict(alignment))
        if "measured_bounds" in adjusted:
            adjusted["measured_bounds"] = _shift_xywh(adjusted.get("measured_bounds"), dx, dy)
        prior_shift = list(adjusted.get("shift") or [0, 0])
        if len(prior_shift) >= 2:
            adjusted["shift"] = [
                _shift_scalar(prior_shift[0], dx),
                _shift_scalar(prior_shift[1], dy),
            ]
        adjusted["ink_bound_fit_shift"] = [dx, dy]
        item["layout_visual_alignment"] = adjusted
    item["ink_bound_fit_shift"] = [dx, dy]
    return item


def _annotate_report(report: FitReport, fit_audit: Mapping[str, Any]) -> FitReport:
    metadata = deepcopy(dict(report.metadata or {}))
    metadata["ink_bound_fit"] = deepcopy(dict(fit_audit))
    return replace(report, metadata=metadata)


def _failed_audit(audit: Mapping[str, Any], issue: str) -> dict[str, Any]:
    out = deepcopy(dict(audit))
    out["status"] = "cannot_fit"
    out["issues"] = list(dict.fromkeys([*(out.get("issues") or []), issue]))
    return out


def _closest_to_zero(lower: int, upper: int) -> int:
    if lower <= 0 <= upper:
        return 0
    if lower > 0:
        return int(lower)
    return int(upper)


def _shift_scalar(value: Any, delta: int) -> Any:
    try:
        return value + delta
    except Exception:
        try:
            return float(value) + delta
        except Exception:
            return value


def _shift_point(value: Any, dx: int, dy: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    items = list(value)
    if len(items) < 2:
        return []
    try:
        return [float(items[0]) + dx, float(items[1]) + dy]
    except Exception:
        return []


def _shift_xywh(value: Any, dx: int, dy: int) -> list[int]:
    box = _four_ints(value)
    if not box:
        return []
    return [box[0] + dx, box[1] + dy, box[2], box[3]]


def _shift_xyxy(value: Sequence[int], dx: int, dy: int) -> list[int]:
    return [int(value[0]) + dx, int(value[1]) + dy, int(value[2]) + dx, int(value[3]) + dy]


def _xywh_to_xyxy(value: Any) -> list[int]:
    box = _four_ints(value)
    if not box or box[2] <= 0 or box[3] <= 0:
        return []
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _xyxy(value: Any) -> list[int]:
    box = _four_ints(value)
    if not box or box[2] <= box[0] or box[3] <= box[1]:
        return []
    return box


def _four_ints(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        items = [int(round(float(item))) for item in list(value)[:4]]
    except Exception:
        return []
    return items if len(items) == 4 else []


def _union_xyxy(boxes: Sequence[Sequence[int]]) -> list[int]:
    valid = [list(box) for box in boxes if len(box) == 4]
    if not valid:
        return []
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return bool(
        len(outer) == 4
        and len(inner) == 4
        and inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )
