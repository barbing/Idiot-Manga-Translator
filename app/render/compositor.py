# -*- coding: utf-8 -*-
"""Stage 5 renderer compositor.

The compositor is intentionally narrow: it draws completed TypesetLayout glyph
placements onto a CleanedPageBase image. It does not run cleanup, create render
regions, reinterpret parent identity, or make style decisions.
"""
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.render.font_manager import FontManager
from app.render.glyph_rasterizer import (
    GLYPH_RASTER_AUTHORITY,
    FreeTypeGlyphRasterizer,
)
from app.render.ink_bound_layout_fitter import InkBoundLayoutFitter
from app.render.layout_planner import RenderLayoutPlanner
from app.render.typesetting_contracts import (
    FitReport,
    GlyphPlacement,
    RenderLayerPlan,
    TypesetLayout,
    copy_jsonish,
    fit_reports_to_audit_dict,
    render_layer_plans_to_audit_dict,
    typeset_layouts_to_audit_dict,
)
from app.render.typesetting_engine import TypesettingEngine

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None

RENDERER_COMPOSITOR_VERSION = "renderer_compositor_stage5_v2"


@dataclass
class CompositorResult:
    """Audit payload returned by the Stage 5 compositor."""

    cleaned_page_base_path: str
    output_path: str
    plans: list[RenderLayerPlan]
    layouts: list[TypesetLayout]
    fit_reports: list[FitReport]
    layer_audits: list[dict[str, Any]]
    elapsed_ms: float = 0.0
    status: str = "not_started"
    issues: list[str] | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        issues = list(self.issues or [])
        drawn_layers = [item for item in self.layer_audits if item.get("drawn")]
        return {
            "renderer_compositor_version": RENDERER_COMPOSITOR_VERSION,
            "status": self.status,
            "cleaned_page_base_path": self.cleaned_page_base_path,
            "output_path": self.output_path,
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "drawing_authority": "typeset_glyph_placements",
            "raster_authority": GLYPH_RASTER_AUTHORITY,
            "pillow_string_raster_used": False,
            "input_authority": "parent_execution_bundle_render_layer_plan",
            "cleanup_mutation_allowed": False,
            "renderer_cleanup_mutation_applied": False,
            "legacy_region_rendering_used": False,
            "layer_count": len(self.plans),
            "layout_count": len(self.layouts),
            "fit_report_count": len(self.fit_reports),
            "drawn_layer_count": len(drawn_layers),
            "issues": issues,
            "layers": copy_jsonish(self.layer_audits),
            "render_layer_plans": render_layer_plans_to_audit_dict(self.plans),
            "typeset_layouts": typeset_layouts_to_audit_dict(self.layouts),
            "fit_reports": fit_reports_to_audit_dict(self.fit_reports),
        }


class RendererCompositor:
    """Draw RenderLayerPlan records after Stage 4 typesetting."""

    def __init__(
        self,
        *,
        font_manager: FontManager | None = None,
        typesetting_engine: TypesettingEngine | None = None,
        layout_planner: RenderLayoutPlanner | None = None,
        glyph_rasterizer: FreeTypeGlyphRasterizer | None = None,
        ink_bound_fitter: InkBoundLayoutFitter | None = None,
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.typesetting_engine = typesetting_engine or TypesettingEngine(self.font_manager)
        self.layout_planner = layout_planner or RenderLayoutPlanner(self.typesetting_engine)
        self.glyph_rasterizer = glyph_rasterizer or FreeTypeGlyphRasterizer()
        self.ink_bound_fitter = ink_bound_fitter or InkBoundLayoutFitter()

    def compose(
        self,
        cleaned_page_base_path: str,
        output_path: str,
        plans: Sequence[RenderLayerPlan],
    ) -> CompositorResult:
        if Image is None or ImageDraw is None:
            raise RuntimeError("Pillow is not installed.")
        if not cleaned_page_base_path:
            raise ValueError("cleaned_page_base_path is required")
        if not output_path:
            raise ValueError("output_path is required")

        start = time.perf_counter()
        ordered_plans = _ordered_plans(plans)
        layouts: list[TypesetLayout] = []
        reports: list[FitReport] = []
        layer_audits: list[dict[str, Any]] = []
        issues: list[str] = []

        with Image.open(cleaned_page_base_path) as source:
            page = source.convert("RGBA")
        adjusted_plans: list[RenderLayerPlan] = []
        occupied_bounds: list[dict[str, Any]] = []
        for plan in ordered_plans:
            adjusted_plan = self.layout_planner.plan_layer(
                page,
                plan,
                occupied_bounds=occupied_bounds,
            )
            adjusted_plans.append(adjusted_plan)
            layout, report = self.typesetting_engine.typeset_layer(adjusted_plan)
            candidate_page = page.copy()
            audit = self._draw_layout(candidate_page, adjusted_plan, layout, report)
            fit_result = self.ink_bound_fitter.fit(
                adjusted_plan,
                layout,
                report,
                audit.get("raster_placements") or [],
            )
            if fit_result.applied:
                layout = fit_result.layout
                report = fit_result.report
                candidate_page = page.copy()
                audit = self._draw_layout(candidate_page, adjusted_plan, layout, report)
            ink_fit_audit = copy_jsonish(fit_result.audit)
            ink_fit_audit["post_fit_failed_raster_placement_count"] = int(
                audit.get("failed_raster_placement_count") or 0
            )
            ink_fit_audit["post_fit_hard_bound_containment_failure_count"] = int(
                audit.get("hard_bound_containment_failure_count") or 0
            )
            audit["ink_bound_fit"] = ink_fit_audit
            audit["issues"] = _unique_strings(
                [
                    *(audit.get("issues") or []),
                    *(fit_result.audit.get("issues") or []),
                ]
            )
            page = candidate_page
            layouts.append(layout)
            reports.append(report)
            layer_audits.append(audit)
            issues.extend(str(item) for item in audit.get("issues", []) or [])
            if layout.measured_bounds:
                occupied_bounds.append(
                    {
                        "root_id": str(adjusted_plan.root_id or ""),
                        "parent_id": str(adjusted_plan.parent_id or ""),
                        "box": list(layout.measured_bounds),
                    }
                )

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        _save_image(page, output_path)
        elapsed = (time.perf_counter() - start) * 1000.0
        return CompositorResult(
            cleaned_page_base_path=cleaned_page_base_path,
            output_path=output_path,
            plans=adjusted_plans,
            layouts=layouts,
            fit_reports=reports,
            layer_audits=layer_audits,
            elapsed_ms=elapsed,
            status="completed",
            issues=_unique_strings(issues),
        )

    def _draw_layout(
        self,
        page,
        plan: RenderLayerPlan,
        layout: TypesetLayout,
        report: FitReport,
    ) -> dict[str, Any]:
        glyphs = [_glyph_to_dict(item) for item in layout.glyphs]
        issues = list(report.issues or [])
        shaped_runs = (
            layout.metadata.get("shaped_runs", [])
            if isinstance(layout.metadata, Mapping)
            else []
        )
        shaped_by_run = {
            str((item.get("metadata") or {}).get("run_id") or ""): item
            for item in shaped_runs
            if isinstance(item, Mapping)
            and isinstance(item.get("metadata"), Mapping)
            and str((item.get("metadata") or {}).get("run_id") or "")
        }

        drawn_glyph_count = 0
        raster_placements: list[dict[str, Any]] = []
        for glyph in glyphs:
            raster_audit = _draw_glyph(
                page,
                font_manager=self.font_manager,
                glyph_rasterizer=self.glyph_rasterizer,
                shaped_by_run=shaped_by_run,
                plan=plan,
                layout=layout,
                glyph=glyph,
            )
            raster_placements.append(raster_audit)
            issues.extend(str(item) for item in raster_audit.get("issues", []) or [])
            if raster_audit.get("status") in {"drawn", "primitive"}:
                drawn_glyph_count += 1
        return _layer_audit(
            plan,
            layout,
            report,
            drawn=drawn_glyph_count > 0,
            drawn_glyph_count=drawn_glyph_count,
            raster_placements=raster_placements,
            issues=issues,
        )


def _draw_glyph(
    page,
    *,
    font_manager: FontManager,
    glyph_rasterizer: FreeTypeGlyphRasterizer,
    shaped_by_run: Mapping[str, Mapping[str, Any]],
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    glyph: Mapping[str, Any],
) -> dict[str, Any]:
    text = str(glyph.get("text") or "")
    bbox = _glyph_bounds(glyph.get("bbox"))
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    run_id = str(metadata.get("run_id") or "")
    mode = str(metadata.get("placement_mode") or "")
    base_audit = {
        "raster_authority": GLYPH_RASTER_AUTHORITY,
        "status": "failed",
        "placement_text": text,
        "run_id": run_id,
        "placement_mode": mode,
        "placement_bbox": list(bbox),
        "writing_mode": str(glyph.get("writing_mode") or layout.writing_mode or ""),
        "font_face_id": str(metadata.get("font_face_id") or glyph.get("font_family") or ""),
        "font_path": str(metadata.get("font_path") or ""),
        "font_fallback_used": bool(metadata.get("font_fallback_used")),
        "punctuation_occurrences": copy_jsonish(metadata.get("punctuation_occurrences") or []),
        "symbol_occurrences": copy_jsonish(metadata.get("symbol_occurrences") or []),
        "requested_glyph_ids": [],
        "drawn_glyph_ids": [],
        "advances_offsets_consumed": False,
        "issues": [],
    }
    if not text:
        return _failed_raster_audit(base_audit, "raster_empty_placement_text")
    if not bbox:
        return _failed_raster_audit(base_audit, "raster_invalid_placement_bbox")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return _failed_raster_audit(base_audit, "raster_invalid_placement_bbox")
    font_size = int(round(float(glyph.get("font_size") or layout.selected_font_size or 1)))
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    fill = _parse_color(style.get("fill_color") or style.get("color") or "#000000", default=(0, 0, 0, 255))
    stroke_fill = _parse_color(style.get("stroke_color") or style.get("stroke") or "#FFFFFF", default=(255, 255, 255, 255))
    stroke_width = _safe_int(style.get("stroke_width"), default=0)
    if stroke_width < 0:
        stroke_width = 0

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if str(layout.writing_mode or "").lower() == "vertical" and mode in {
        "vertical_ellipsis_sequence",
        "vertical_dash_sequence",
        "vertical_wave_sequence",
    }:
        face_id = str(metadata.get("font_face_id") or glyph.get("font_family") or layout.selected_font_face or "")
        face = font_manager.face(face_id)
        if face is None:
            return _failed_raster_audit(base_audit, f"raster_primitive_font_face_missing:{face_id}")
        font = font_manager.load_font(face, max(1, font_size))
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        primitive_evidence = _draw_compact_vertical_sequence(
            layer,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            glyph=glyph,
            width=width,
            height=height,
            layout=layout,
        )
        if not primitive_evidence:
            return _failed_raster_audit(base_audit, f"raster_primitive_draw_failed:{mode}")
        containment = _hard_bound_containment(
            page,
            layer,
            (x0, y0),
            plan.hard_bounds or plan.target_box,
        )
        alpha_sum = int(sum(layer.getchannel("A").getdata()))
        raw_ids = metadata.get("shaped_glyph_ids")
        shaped_ids = (
            [int(item) for item in raw_ids]
            if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray))
            else [int(metadata.get("shaped_glyph_id"))]
            if metadata.get("shaped_glyph_id") is not None
            else []
        )
        primitive_audit = {
            **base_audit,
            **primitive_evidence,
            "raster_authority": "renderer_punctuation_primitive",
            "status": "primitive" if bool(containment.get("accepted")) else "failed",
            "primitive_type": mode,
            "policy_owner": "punctuation_policy_v1",
            "position_policy": "semantic_vertical_punctuation_primitive",
            "font_face_id": face.face_id,
            "font_path": face.path,
            "requested_glyph_ids": shaped_ids,
            "shaped_glyph_ids": shaped_ids,
            "drawn_glyph_ids": [],
            "advances_offsets_consumed": False,
            "alpha_composition_policy": "single_target_copy",
            "source_alpha_sum": alpha_sum,
            "target_alpha_sum": alpha_sum,
            "raster_clipped_to_target": False,
            "hard_bound_containment": containment,
            "composite_dest": [x0, y0],
            "composite_bounds": list(containment.get("raster_alpha_bounds") or []),
            "issues": [] if bool(containment.get("accepted")) else ["raster_ink_exceeds_parent_hard_bounds"],
        }
        if bool(containment.get("accepted")):
            page.alpha_composite(layer, dest=(x0, y0))
        return primitive_audit

    if bool(metadata.get("space_run")) or text.isspace():
        return {
            **base_audit,
            "raster_authority": "layout_no_ink_space",
            "status": "no_ink",
            "issues": [],
        }

    shaped_run = shaped_by_run.get(run_id)
    if not isinstance(shaped_run, Mapping):
        return _failed_raster_audit(base_audit, f"raster_missing_shaped_run:{run_id}")
    shaped_face_id = str(shaped_run.get("font_face_id") or "")
    shaped_font_path = str(shaped_run.get("font_path") or "")
    placement_face_id = str(metadata.get("font_face_id") or glyph.get("font_family") or "")
    placement_font_path = str(metadata.get("font_path") or "")
    if placement_face_id and shaped_face_id != placement_face_id:
        return _failed_raster_audit(
            base_audit,
            f"raster_face_mismatch:{placement_face_id}:{shaped_face_id}",
        )
    if placement_font_path and not _same_path(placement_font_path, shaped_font_path):
        return _failed_raster_audit(base_audit, "raster_font_path_mismatch")

    requested_ids: Sequence[int] | None = None
    raw_ids = metadata.get("shaped_glyph_ids")
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
        requested_ids = [int(item) for item in raw_ids]
    elif metadata.get("shaped_glyph_id") is not None:
        requested_ids = [int(metadata.get("shaped_glyph_id"))]
    result = glyph_rasterizer.rasterize(
        shaped_run=shaped_run,
        requested_glyph_ids=requested_ids,
        target_size=(width, height),
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=stroke_width,
        position_policy=(
            "compact_vertical_sequence_preserved"
            if mode == "vertical_ellipsis_sequence"
            else "harfbuzz"
        ),
    )
    raster_audit = {
        **base_audit,
        **result.audit,
        "placement_text": text,
        "run_id": run_id,
        "placement_mode": mode,
        "placement_bbox": list(bbox),
        "writing_mode": str(glyph.get("writing_mode") or layout.writing_mode or ""),
        "font_fallback_used": bool(metadata.get("font_fallback_used")),
    }
    if result.drawn:
        offset = list(result.audit.get("composite_offset") or [0, 0])
        offset_x = int(offset[0]) if len(offset) > 0 else 0
        offset_y = int(offset[1]) if len(offset) > 1 else 0
        composite_dest = (x0 + offset_x, y0 + offset_y)
        containment = _hard_bound_containment(
            page,
            result.image,
            composite_dest,
            plan.hard_bounds or plan.target_box,
        )
        raster_audit["hard_bound_containment"] = containment
        raster_audit["composite_dest"] = list(composite_dest)
        raster_audit["composite_bounds"] = list(containment.get("raster_alpha_bounds") or [])
        if not bool(containment.get("accepted")):
            rasterized_ids = list(raster_audit.get("drawn_glyph_ids") or [])
            raster_audit.update(
                {
                    "status": "failed",
                    "rasterized_glyph_ids": rasterized_ids,
                    "drawn_glyph_ids": [],
                    "issues": _unique_strings(
                        [
                            *(raster_audit.get("issues") or []),
                            "raster_ink_exceeds_parent_hard_bounds",
                        ]
                    ),
                }
            )
        else:
            page.alpha_composite(result.image, dest=composite_dest)
    return raster_audit


def _draw_compact_vertical_sequence(
    layer,
    *,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    glyph: Mapping[str, Any],
    width: int,
    height: int,
    layout: TypesetLayout,
) -> dict[str, Any]:
    if str(layout.writing_mode or "").lower() != "vertical":
        return {}
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    mode = str(metadata.get("placement_mode") or "")
    if mode not in {"vertical_ellipsis_sequence", "vertical_dash_sequence", "vertical_wave_sequence"}:
        return {}
    if mode == "vertical_ellipsis_sequence":
        return _draw_vertical_ellipsis_dots(
            layer,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            glyph=glyph,
            width=width,
            height=height,
            font_size=int(getattr(font, "size", max(width, height)) or max(width, height)),
        )
    if mode == "vertical_dash_sequence":
        return _draw_vertical_dash_line(
            layer,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            glyph=glyph,
            width=width,
            height=height,
        )
    return _draw_vertical_wave_line(
        layer,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        glyph=glyph,
        width=width,
        height=height,
    )


def _draw_vertical_dash_line(
    layer,
    *,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    glyph: Mapping[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    if ImageDraw is None:
        return {}
    font_size = int(getattr(font, "size", max(width, height)) or max(width, height))
    line_width = max(2, int(round(float(font_size) * 0.09)))
    pad_y = max(1, int(round(float(font_size) * 0.04)))
    x = int(round((float(width) - 1.0) / 2.0))
    y0 = max(0, pad_y)
    y1 = min(height - 1, height - pad_y - 1)
    if y1 <= y0:
        return {}
    draw = ImageDraw.Draw(layer)
    if stroke_width > 0:
        draw.line(
            (x, y0, x, y1),
            fill=stroke_fill,
            width=max(line_width + stroke_width * 2, line_width),
        )
    draw.line((x, y0, x, y1), fill=fill, width=line_width)
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    dash_units = int(metadata.get("dash_unit_count") or max(1, len(str(glyph.get("text") or ""))))
    return {
        "dash_unit_count": dash_units,
        "continuous_segment_count": 1,
        "continuous_multi_cell_dash": True,
    }


def _draw_vertical_wave_line(
    layer,
    *,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    glyph: Mapping[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    if ImageDraw is None:
        return {}
    font_size = int(getattr(font, "size", max(width, height)) or max(width, height))
    line_width = max(2, int(round(float(font_size) * 0.08)))
    amplitude = max(2.0, min(float(width) * 0.24, float(font_size) * 0.14))
    pad_y = max(1, int(round(float(font_size) * 0.04)))
    x_center = (float(width) - 1.0) / 2.0
    y0 = max(0, pad_y)
    y1 = min(height - 1, height - pad_y - 1)
    if y1 <= y0:
        return {}
    height_span = max(1.0, float(y1 - y0))
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    wave_units = int(metadata.get("wave_unit_count") or max(1, len(str(glyph.get("text") or ""))))
    cycles = float(max(1, wave_units))
    points: list[tuple[float, float]] = []
    for y in range(int(y0), int(y1) + 1):
        t = float(y - y0) / height_span
        phase = t * math.tau * cycles
        x = x_center + math.sin(phase) * amplitude
        points.append((x, float(y)))
    if len(points) < 2:
        return {}
    draw = ImageDraw.Draw(layer)
    if stroke_width > 0:
        draw.line(points, fill=stroke_fill, width=max(line_width + stroke_width * 2, line_width), joint="curve")
    draw.line(points, fill=fill, width=line_width, joint="curve")
    return {
        "wave_unit_count": wave_units,
        "wave_cycle_count": round(cycles, 3),
        "continuous_multi_cell_wave": True,
        "wave_source_classes": [
            str(item.get("source_class") or "")
            for item in list(metadata.get("punctuation_occurrences") or [])
            if isinstance(item, Mapping)
        ],
    }


def _draw_vertical_ellipsis_dots(
    layer,
    *,
    fill,
    stroke_width: int,
    stroke_fill,
    glyph: Mapping[str, Any],
    width: int,
    height: int,
    font_size: int,
) -> dict[str, Any]:
    if ImageDraw is None:
        return {}
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    unit_count = int(
        metadata.get("ellipsis_unit_count")
        or max(1, len(str(glyph.get("text") or "")))
    )
    unit_count = max(1, unit_count)
    dot_count = max(1, int(metadata.get("ellipsis_dot_count") or unit_count * 3))
    sequence_group_count = max(1, int(metadata.get("ellipsis_sequence_group_count") or 1))
    diameter = max(2, int(round(float(font_size) * 0.12)))
    radius = float(diameter) / 2.0
    safe_stroke = max(0, int(stroke_width))
    center_x = (float(width) - 1.0) / 2.0
    edge_inset = float(height) * (0.25 / float(unit_count))
    first_center_y = edge_inset
    last_center_y = float(height) - edge_inset
    dot_pitch = (
        (last_center_y - first_center_y) / float(dot_count - 1)
        if dot_count > 1
        else 0.0
    )
    centers: list[list[float]] = []
    draw = ImageDraw.Draw(layer)
    for dot_index in range(dot_count):
        center_y = first_center_y + dot_pitch * float(dot_index) if dot_count > 1 else float(height) / 2.0
        centers.append([round(center_x, 3), round(center_y, 3)])
        if safe_stroke > 0:
            outer = radius + float(safe_stroke)
            draw.ellipse(
                (
                    center_x - outer,
                    center_y - outer,
                    center_x + outer,
                    center_y + outer,
                ),
                fill=stroke_fill,
            )
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=fill,
        )
    pitch_deltas = [
        round(float(centers[index + 1][1]) - float(centers[index][1]), 3)
        for index in range(len(centers) - 1)
    ]
    max_pitch_delta = (
        max(pitch_deltas) - min(pitch_deltas)
        if pitch_deltas
        else 0.0
    )
    return {
        "ellipsis_unit_count": unit_count,
        "dot_count": dot_count,
        "dot_column_count": 1,
        "sequence_group_count": sequence_group_count,
        "dot_diameter_px": diameter,
        "dot_centers": centers,
        "dot_pitch_px": round(dot_pitch, 3),
        "dot_pitch_deltas": pitch_deltas,
        "max_dot_pitch_delta_px": round(max_pitch_delta, 3),
        "ellipsis_policy": "one_continuous_uniform_dot_sequence",
    }


def _layer_audit(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    *,
    drawn: bool,
    drawn_glyph_count: int = 0,
    raster_placements: Sequence[Mapping[str, Any]] | None = None,
    issues: Sequence[str] | None = None,
) -> dict[str, Any]:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    glyph_text = "".join(str(item.get("text") or "") for item in (_glyph_to_dict(g) for g in layout.glyphs))
    normalized = str(layout.normalized_text or "")
    raster_items = [dict(item) for item in list(raster_placements or [])]
    return {
        "renderer_compositor_version": RENDERER_COMPOSITOR_VERSION,
        "drawing_authority": "typeset_glyph_placements",
        "raster_authority": GLYPH_RASTER_AUTHORITY,
        "pillow_string_raster_used": False,
        "cleanup_mutation_allowed": False,
        "renderer_cleanup_mutation_applied": False,
        "legacy_region_rendering_used": False,
        "page_id": plan.page_id,
        "layer_id": plan.layer_id,
        "bundle_id": plan.bundle_id,
        "parent_id": plan.parent_id,
        "root_id": plan.root_id,
        "draw_order": int(plan.draw_order),
        "drawn": bool(drawn),
        "glyph_count": len(list(layout.glyphs or [])),
        "drawn_glyph_count": int(drawn_glyph_count),
        "shaped_raster_placement_count": sum(item.get("status") == "drawn" for item in raster_items),
        "primitive_raster_placement_count": sum(item.get("status") == "primitive" for item in raster_items),
        "no_ink_placement_count": sum(item.get("status") == "no_ink" for item in raster_items),
        "failed_raster_placement_count": sum(item.get("status") == "failed" for item in raster_items),
        "natural_ink_overhang_placement_count": sum(
            any(float(value) > 0 for value in item.get("logical_cell_overhang_px", []) or [])
            for item in raster_items
            if item.get("status") in {"drawn", "primitive"}
        ),
        "hard_bound_containment_failure_count": sum(
            isinstance(item.get("hard_bound_containment"), Mapping)
            and not bool(item.get("hard_bound_containment", {}).get("accepted"))
            for item in raster_items
        ),
        "raster_placements": copy_jsonish(raster_items),
        "glyph_text_matches_layout": glyph_text == normalized,
        "full_text_placed": bool(report.full_text_placed),
        "fit_status": report.fit_status,
        "overflow_risk": bool(report.overflow_risk),
        "clipping_risk": bool(report.clipping_risk),
        "selected_font_face": layout.selected_font_face,
        "selected_font_size": float(layout.selected_font_size),
        "writing_mode": layout.writing_mode,
        "target_box": list(plan.target_box or []),
        "hard_bounds": list(plan.hard_bounds or []),
        "measured_bounds": list(layout.measured_bounds or []),
        "shape_aware_composition": copy_jsonish(
            plan.metadata.get("shape_aware_composition", {})
            if isinstance(plan.metadata, Mapping)
            else {}
        ),
        "visual_slot_scoring": copy_jsonish(
            plan.metadata.get("visual_slot_scoring", {})
            if isinstance(plan.metadata, Mapping)
            else {}
        ),
        "fill_color": style.get("fill_color") or style.get("color"),
        "stroke_color": style.get("stroke_color") or style.get("stroke"),
        "stroke_width": style.get("stroke_width"),
        "issues": _unique_strings(issues or []),
    }


def _failed_raster_audit(audit: Mapping[str, Any], issue: str) -> dict[str, Any]:
    payload = dict(audit)
    payload["status"] = "failed"
    payload["issues"] = _unique_strings([*(payload.get("issues") or []), issue])
    return payload


def _hard_bound_containment(
    page,
    raster,
    dest: tuple[int, int],
    hard_bounds: Sequence[Any] | None,
) -> dict[str, Any]:
    alpha_box = raster.getchannel("A").getbbox() if raster is not None else None
    hard_xyxy = _glyph_bounds(hard_bounds)
    page_xyxy = [0, 0, int(page.size[0]), int(page.size[1])]
    if not alpha_box:
        return {
            "accepted": False,
            "reason": "raster_alpha_empty",
            "parent_hard_bounds": list(hard_xyxy),
            "page_bounds": page_xyxy,
            "raster_alpha_bounds": [],
        }
    actual = [
        int(dest[0] + alpha_box[0]),
        int(dest[1] + alpha_box[1]),
        int(dest[0] + alpha_box[2]),
        int(dest[1] + alpha_box[3]),
    ]
    inside_page = _xyxy_contains(page_xyxy, actual)
    inside_parent = bool(hard_xyxy) and _xyxy_contains(hard_xyxy, actual)
    return {
        "accepted": bool(inside_page and inside_parent),
        "reason": (
            "complete_natural_ink_inside_parent_hard_bounds"
            if inside_page and inside_parent
            else "natural_ink_crosses_parent_or_page_hard_bounds"
        ),
        "parent_hard_bounds": list(hard_xyxy),
        "page_bounds": page_xyxy,
        "raster_alpha_bounds": actual,
        "inside_page_bounds": bool(inside_page),
        "inside_parent_hard_bounds": bool(inside_parent),
    }


def _xyxy_contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    if len(outer) != 4 or len(inner) != 4:
        return False
    return (
        int(inner[0]) >= int(outer[0])
        and int(inner[1]) >= int(outer[1])
        and int(inner[2]) <= int(outer[2])
        and int(inner[3]) <= int(outer[3])
    )


def _same_path(left: str, right: str) -> bool:
    if not left or not right:
        return left == right
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except Exception:
        return left == right


def _ordered_plans(plans: Sequence[RenderLayerPlan]) -> list[RenderLayerPlan]:
    return sorted(
        [plan for plan in plans or [] if isinstance(plan, RenderLayerPlan)],
        key=lambda plan: (int(plan.draw_order), str(plan.layer_id)),
    )


def _glyph_to_dict(value: GlyphPlacement | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, GlyphPlacement):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"text": str(value)}


def _glyph_bounds(value: Any) -> list[int]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: list[int] = []
    for item in list(value)[:4]:
        try:
            out.append(int(round(float(item))))
        except Exception:
            return []
    if len(out) != 4:
        return []
    x, y, w, h = out
    if w <= 0 or h <= 0:
        return []
    return [x, y, x + w, y + h]


def _parse_color(value: Any, *, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#"):
            text = text[1:]
        if len(text) in {6, 8}:
            try:
                r = int(text[0:2], 16)
                g = int(text[2:4], 16)
                b = int(text[4:6], 16)
                a = int(text[6:8], 16) if len(text) == 8 else 255
                return r, g, b, a
            except ValueError:
                return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            items = [int(round(float(item))) for item in list(value)[:4]]
            if len(items) == 3:
                items.append(255)
            if len(items) == 4:
                return tuple(max(0, min(255, item)) for item in items)  # type: ignore[return-value]
        except Exception:
            return default
    return default


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _save_image(image, output_path: str) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output_path, quality=95)
    else:
        image.save(output_path)


def _unique_strings(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return output
