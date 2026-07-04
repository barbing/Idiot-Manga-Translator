# -*- coding: utf-8 -*-
"""Stage 5 renderer compositor.

The compositor is intentionally narrow: it draws completed TypesetLayout glyph
placements onto a CleanedPageBase image. It does not run cleanup, create render
regions, reinterpret parent identity, or make style decisions.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from app.render.font_manager import FontManager
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
from app.render.typesetting_text import grapheme_clusters

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime dependency
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


RENDERER_COMPOSITOR_VERSION = "renderer_compositor_stage5_v1"


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
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.typesetting_engine = typesetting_engine or TypesettingEngine(self.font_manager)

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
            adjusted_plan = _shape_aware_plan(page, plan)
            adjusted_plan = _visual_slot_scored_plan(
                page,
                plan,
                adjusted_plan,
                self.typesetting_engine,
                occupied_bounds,
            )
            adjusted_plans.append(adjusted_plan)
            layout, report = self.typesetting_engine.typeset_layer(adjusted_plan)
            layouts.append(layout)
            reports.append(report)
            audit = self._draw_layout(page, adjusted_plan, layout, report)
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
        face = self.font_manager.face(layout.selected_font_face)
        issues = list(report.issues or [])
        if not face:
            issues.append("compositor_missing_font_face")
            return _layer_audit(plan, layout, report, drawn=False, issues=issues)

        drawn_glyph_count = 0
        for glyph in glyphs:
            if _draw_glyph(
                page,
                font_manager=self.font_manager,
                face=face,
                plan=plan,
                layout=layout,
                glyph=glyph,
            ):
                drawn_glyph_count += 1
        return _layer_audit(
            plan,
            layout,
            report,
            drawn=drawn_glyph_count > 0,
            drawn_glyph_count=drawn_glyph_count,
            issues=issues,
        )


def _draw_glyph(
    page,
    *,
    font_manager: FontManager,
    face,
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    glyph: Mapping[str, Any],
) -> bool:
    text = str(glyph.get("text") or "")
    bbox = _glyph_bounds(glyph.get("bbox"))
    if not text or not bbox:
        return False
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return False
    font_size = int(round(float(glyph.get("font_size") or layout.selected_font_size or 1)))
    font = font_manager.load_font(face, max(1, font_size))
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    fill = _parse_color(style.get("fill_color") or style.get("color") or "#000000", default=(0, 0, 0, 255))
    stroke_fill = _parse_color(style.get("stroke_color") or style.get("stroke") or "#FFFFFF", default=(255, 255, 255, 255))
    stroke_width = _safe_int(style.get("stroke_width"), default=0)
    if stroke_width < 0:
        stroke_width = 0

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if _draw_compact_vertical_sequence(
        layer,
        text=text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        glyph=glyph,
        width=width,
        height=height,
        layout=layout,
    ):
        page.alpha_composite(layer, dest=(x0, y0))
        return True
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    if str(layout.writing_mode or "").lower() == "vertical" and str(metadata.get("placement_mode") or "") == "vertical_punctuation":
        if _draw_text_ink_centered(
            layer,
            text=text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            target_box=(0, 0, width, height),
        ):
            page.alpha_composite(layer, dest=(x0, y0))
            return True
    draw = ImageDraw.Draw(layer)
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    except Exception:
        text_bbox = font.getbbox(text)
    tw = max(1, int(text_bbox[2] - text_bbox[0]))
    th = max(1, int(text_bbox[3] - text_bbox[1]))
    tx = (width - tw) / 2.0 - float(text_bbox[0])
    ty = (height - th) / 2.0 - float(text_bbox[1])
    draw.text((tx, ty), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    page.alpha_composite(layer, dest=(x0, y0))
    return True


def _draw_compact_vertical_sequence(
    layer,
    *,
    text: str,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    glyph: Mapping[str, Any],
    width: int,
    height: int,
    layout: TypesetLayout,
) -> bool:
    if str(layout.writing_mode or "").lower() != "vertical":
        return False
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    mode = str(metadata.get("placement_mode") or "")
    if mode not in {"vertical_ellipsis_sequence", "vertical_dash_sequence", "vertical_punctuation_sequence"}:
        return False
    clusters = [cluster for cluster in grapheme_clusters(text) if cluster]
    if len(clusters) <= 1:
        return False
    if mode == "vertical_dash_sequence":
        return _draw_vertical_dash_line(
            layer,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            width=width,
            height=height,
        )
    draw = ImageDraw.Draw(layer)
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    for cluster in clusters:
        try:
            box = draw.textbbox((0, 0), cluster, font=font, stroke_width=stroke_width)
        except Exception:
            box = font.getbbox(cluster)
        boxes.append((cluster, box))
    max_w = max(max(1, int(box[2] - box[0])) for _cluster, box in boxes)
    max_h = max(max(1, int(box[3] - box[1])) for _cluster, box in boxes)
    if len(clusters) == 1:
        step = 0.0
    else:
        available_step = max(1.0, float(height - max_h)) / float(len(clusters) - 1)
        if mode == "vertical_dash_sequence":
            desired_step = max_h * 0.42
        elif mode == "vertical_ellipsis_sequence":
            desired_step = max_h * 0.50
        else:
            desired_step = max_h * 0.58
        step = max(1.0, min(desired_step, available_step))
    total_h = float(max_h) + step * float(len(clusters) - 1)
    start_y = max(0.0, (float(height) - total_h) / 2.0)
    for index, (cluster, box) in enumerate(boxes):
        tw = max(1, int(box[2] - box[0]))
        th = max(1, int(box[3] - box[1]))
        tx = (float(width) - float(tw)) / 2.0 - float(box[0])
        ty = start_y + float(index) * step + (float(max_h) - float(th)) / 2.0 - float(box[1])
        draw.text((tx, ty), cluster, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    return max_w > 0


def _draw_text_ink_centered(
    layer,
    *,
    text: str,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    target_box: tuple[int, int, int, int],
) -> bool:
    if Image is None or ImageDraw is None:
        return False
    x0, y0, x1, y1 = target_box
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    font_size = int(getattr(font, "size", max(width, height)) or max(width, height))
    margin = max(8, int(font_size * 2 + stroke_width * 4))
    scratch = Image.new("RGBA", (width + margin * 2, height + margin * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    try:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    except Exception:
        bbox = font.getbbox(text)
    tx = float(margin) - float(bbox[0])
    ty = float(margin) - float(bbox[1])
    draw.text((tx, ty), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    alpha = scratch.getchannel("A")
    ink_box = alpha.getbbox()
    if not ink_box:
        return False
    ink = scratch.crop(ink_box)
    ink_w, ink_h = ink.size
    paste_x = int(round((float(width) - float(ink_w)) / 2.0))
    paste_y = int(round((float(height) - float(ink_h)) / 2.0))
    src_x0 = max(0, -paste_x)
    src_y0 = max(0, -paste_y)
    src_x1 = min(ink_w, width - paste_x)
    src_y1 = min(ink_h, height - paste_y)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return False
    clipped = ink.crop((src_x0, src_y0, src_x1, src_y1))
    dx = int(x0 + max(0, paste_x))
    dy = int(y0 + max(0, paste_y))
    layer.alpha_composite(clipped, dest=(dx, dy))
    return True


def _draw_vertical_dash_line(
    layer,
    *,
    font,
    fill,
    stroke_width: int,
    stroke_fill,
    width: int,
    height: int,
) -> bool:
    if ImageDraw is None:
        return False
    font_size = int(getattr(font, "size", max(width, height)) or max(width, height))
    line_width = max(2, int(round(float(font_size) * 0.09)))
    pad_y = max(1, int(round(float(font_size) * 0.04)))
    x = int(round((float(width) - 1.0) / 2.0))
    y0 = max(0, pad_y)
    y1 = min(height - 1, height - pad_y - 1)
    if y1 <= y0:
        return False
    draw = ImageDraw.Draw(layer)
    if stroke_width > 0:
        draw.line(
            (x, y0, x, y1),
            fill=stroke_fill,
            width=max(line_width + stroke_width * 2, line_width),
        )
    draw.line((x, y0, x, y1), fill=fill, width=line_width)
    return True


def _layer_audit(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    *,
    drawn: bool,
    drawn_glyph_count: int = 0,
    issues: Sequence[str] | None = None,
) -> dict[str, Any]:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    glyph_text = "".join(str(item.get("text") or "") for item in (_glyph_to_dict(g) for g in layout.glyphs))
    normalized = str(layout.normalized_text or "")
    return {
        "renderer_compositor_version": RENDERER_COMPOSITOR_VERSION,
        "drawing_authority": "typeset_glyph_placements",
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


def _ordered_plans(plans: Sequence[RenderLayerPlan]) -> list[RenderLayerPlan]:
    return sorted(
        [plan for plan in plans or [] if isinstance(plan, RenderLayerPlan)],
        key=lambda plan: (int(plan.draw_order), str(plan.layer_id)),
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
        hard_bounds=list(safe_box),
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
    if not _is_shape_aware_speech_layer(original_plan):
        return _plan_with_visual_slot_audit(shape_plan, {"applied": False, "reason": "not_speech_bubble_layer"})
    if np is None:
        return _plan_with_visual_slot_audit(shape_plan, {"applied": False, "reason": "numpy_unavailable"})
    page_box = [0, 0, int(page.size[0]), int(page.size[1])]
    candidate = _shape_candidate_box(original_plan, page_box)
    if not candidate:
        return _plan_with_visual_slot_audit(shape_plan, {"applied": False, "reason": "missing_candidate_box"})
    geometry = _speech_bubble_geometry_from_page(page, original_plan, candidate)
    audit = geometry.get("audit") if isinstance(geometry.get("audit"), Mapping) else {}
    if not audit.get("applied"):
        return _plan_with_visual_slot_audit(shape_plan, audit or {"applied": False, "reason": "speech_geometry_unavailable"})

    candidates = _visual_slot_candidates(original_plan, shape_plan, geometry, page_box)
    if not candidates:
        return _plan_with_visual_slot_audit(shape_plan, {"applied": False, "reason": "no_visual_slot_candidates"})

    scored: list[tuple[float, RenderLayerPlan, dict[str, Any]]] = []
    source_box = _source_contract_box(original_plan)
    for candidate_record in candidates:
        box = _bbox_from_value(candidate_record.get("box"))
        if not box:
            continue
        candidate_plan = _plan_with_visual_slot_box(
            shape_plan,
            box,
            source=str(candidate_record.get("source") or "candidate"),
        )
        layout, report = typesetting_engine.typeset_layer(candidate_plan)
        score, score_meta = _score_visual_slot(
            candidate_plan,
            layout,
            report,
            geometry=geometry,
            source_box=source_box,
            occupied_bounds=occupied_bounds,
        )
        scored.append(
            (
                float(score),
                candidate_plan,
                {
                    "source": str(candidate_record.get("source") or "candidate"),
                    "box": list(box),
                    "score": round(float(score), 4),
                    **score_meta,
                },
            )
        )

    if not scored:
        return _plan_with_visual_slot_audit(shape_plan, {"applied": False, "reason": "no_scoreable_visual_slot_candidates"})
    scored.sort(key=lambda item: (item[0], _area(item[1].target_box)))
    _score, selected_plan, selected_meta = scored[0]
    rejected = [item[2] for item in scored[1:8]]
    final_audit = {
        "applied": True,
        "source": "stage5_visual_slot_scoring",
        "selected_source": selected_meta.get("source"),
        "selected_box": list(selected_plan.target_box),
        "selected_score": selected_meta.get("score"),
        "candidate_count": len(scored),
        "source_contract_bbox": list(source_box),
        "speech_component_box": copy_jsonish(audit.get("component_box")),
        "speech_safe_box": copy_jsonish(audit.get("box")),
        "selected": selected_meta,
        "rejected_candidates": rejected,
    }
    return _plan_with_visual_slot_audit(selected_plan, final_audit)


def _plan_with_visual_slot_audit(plan: RenderLayerPlan, audit: Mapping[str, Any]) -> RenderLayerPlan:
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["visual_slot_scoring"] = copy_jsonish(audit)
    return replace(plan, metadata=metadata)


def _plan_with_visual_slot_box(plan: RenderLayerPlan, box: Sequence[int], *, source: str) -> RenderLayerPlan:
    target = _bbox_from_value(box)
    metadata = copy_jsonish(plan.metadata) if isinstance(plan.metadata, Mapping) else {}
    metadata["visual_slot_candidate_source"] = str(source)
    clipping = copy_jsonish(plan.clipping_region_ref) if isinstance(plan.clipping_region_ref, Mapping) else {}
    clipping["visual_slot_box"] = list(target)
    return replace(plan, target_box=list(target), hard_bounds=list(target), clipping_region_ref=clipping, metadata=metadata)


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
    source_box = _source_contract_box(original_plan)
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

    component = geometry.get("component")
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

    if source_box:
        container = safe_box or component_box or candidate_box or page_box
        add("source_footprint_padded_box", _source_padded_box(source_box, container, original_plan))
        add("source_footprint_wide_box", _source_padded_box(source_box, container, original_plan, scale=1.8))
    if component is not None:
        _ = component  # evidence exists; mask scoring is applied later.
    return records


def _score_visual_slot(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    *,
    geometry: Mapping[str, Any],
    source_box: Sequence[int],
    occupied_bounds: Sequence[Mapping[str, Any]],
) -> tuple[float, dict[str, Any]]:
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

    inside_ratio = _layout_mask_inside_ratio(layout, geometry)
    if inside_ratio is not None:
        mask_penalty = (1.0 - inside_ratio) * 220.0
        score += mask_penalty
        meta["speech_mask_inside_ratio"] = round(float(inside_ratio), 4)
        meta["speech_mask_penalty"] = round(float(mask_penalty), 4)

    if source_box:
        measured_center = _center_box(measured)
        source_center = _center_box(source_box)
        if measured_center and source_center:
            normalized_distance = (
                abs(float(measured_center[0]) - float(source_center[0])) / max(1.0, float(source_box[2]))
                + abs(float(measured_center[1]) - float(source_center[1])) / max(1.0, float(source_box[3]))
            )
            source_penalty = normalized_distance * 90.0
            score += source_penalty
            meta["source_center_distance"] = round(float(normalized_distance), 4)
            meta["source_center_penalty"] = round(float(source_penalty), 4)
        if target:
            area_ratio = float(_area(target)) / max(1.0, float(_area(source_box)))
            area_penalty = max(0.0, area_ratio - 2.75) * 2.0
            score += area_penalty
            meta["target_to_source_area_ratio"] = round(float(area_ratio), 4)
            meta["target_area_penalty"] = round(float(area_penalty), 4)

    overlap_penalty = 0.0
    for occupied in occupied_bounds or []:
        if str(occupied.get("root_id") or "") != str(plan.root_id or ""):
            continue
        other = _bbox_from_value(occupied.get("box"))
        if not other:
            continue
        overlap_penalty += _box_iou(measured, other) * 420.0
    if overlap_penalty:
        score += overlap_penalty
        meta["same_root_overlap_penalty"] = round(float(overlap_penalty), 4)
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
    }
    return {
        "audit": audit,
        "candidate_box": list(candidate_box),
        "component": component,
        "safe_mask": safe_mask,
        "component_box_local": component_box,
        "anchor": anchor,
    }


def _is_shape_aware_speech_layer(plan: RenderLayerPlan) -> bool:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    slot = metadata.get("parent_render_slot") if isinstance(metadata.get("parent_render_slot"), Mapping) else {}
    text = " ".join(
        str(value or "").lower()
        for value in (
            plan.role,
            style.get("semantic_class"),
            style.get("semantic_kind"),
            style.get("source_role"),
            style.get("route_intent"),
            slot.get("source"),
        )
    )
    if "caption" in text or "background" in text:
        return False
    return "speech" in text or "bubble" in text or str(plan.role or "").lower() == "speech"


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
        slot.get("source_anchor_center") if isinstance(slot, Mapping) else [],
        _center_box(plan.source_provenance_ref.get("source_contract_bbox") if isinstance(plan.source_provenance_ref, Mapping) else []),
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
    metadata = plan.metadata if isinstance(plan.metadata, Mapping) else {}
    slot = metadata.get("parent_render_slot") if isinstance(metadata.get("parent_render_slot"), Mapping) else {}
    for value in (
        provenance.get("source_contract_bbox") if isinstance(provenance, Mapping) else [],
        slot.get("source_contract_bbox") if isinstance(slot, Mapping) else [],
        slot.get("source_box") if isinstance(slot, Mapping) else [],
    ):
        box = _bbox_from_value(value)
        if box:
            return box
    return []


def _source_padded_box(
    source_box: Sequence[int],
    container: Sequence[int],
    plan: RenderLayerPlan,
    *,
    scale: float = 1.0,
) -> list[int]:
    source = _bbox_from_value(source_box)
    bounds = _bbox_from_value(container)
    if not source:
        return []
    if not bounds:
        bounds = source
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    font_size = _safe_int(style.get("font_size") or style.get("font_size_hint"), default=24)
    sx, sy, sw, sh = source
    pad_x = max(6, int(round(float(font_size) * 0.65 * float(scale))), int(round(float(sw) * 0.08 * float(scale))))
    pad_y = max(6, int(round(float(font_size) * 0.55 * float(scale))), int(round(float(sh) * 0.08 * float(scale))))
    padded = [sx - pad_x, sy - pad_y, sw + pad_x * 2, sh + pad_y * 2]
    return _intersect_box(padded, bounds) or _intersect_box(padded, source) or list(source)


def _layout_mask_inside_ratio(layout: TypesetLayout, geometry: Mapping[str, Any]) -> float | None:
    if np is None:
        return None
    mask = geometry.get("safe_mask")
    if mask is None or not hasattr(mask, "shape") or int(mask.sum()) <= 0:
        mask = geometry.get("component")
    if mask is None or not hasattr(mask, "shape") or int(mask.sum()) <= 0:
        return None
    candidate = _bbox_from_value(geometry.get("candidate_box"))
    if not candidate:
        return None
    samples = 0
    inside = 0
    height, width = mask.shape[:2]
    for glyph in layout.glyphs or []:
        box = _bbox_from_value(_glyph_to_dict(glyph).get("bbox"))
        if not box:
            continue
        gx, gy, gw, gh = box
        points = (
            (gx + gw / 2.0, gy + gh / 2.0),
            (gx + gw * 0.25, gy + gh * 0.25),
            (gx + gw * 0.75, gy + gh * 0.25),
            (gx + gw * 0.25, gy + gh * 0.75),
            (gx + gw * 0.75, gy + gh * 0.75),
        )
        for px, py in points:
            lx = int(round(float(px) - float(candidate[0])))
            ly = int(round(float(py) - float(candidate[1])))
            samples += 1
            if 0 <= lx < width and 0 <= ly < height and bool(mask[ly, lx]):
                inside += 1
    if samples <= 0:
        return None
    return float(inside) / float(samples)


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


def _box_iou(first: Sequence[int], second: Sequence[int]) -> float:
    a = _bbox_from_value(first)
    b = _bbox_from_value(second)
    if not a or not b:
        return 0.0
    inter = _intersect_box(a, b)
    if not inter:
        return 0.0
    inter_area = float(_area(inter))
    union = float(_area(a) + _area(b)) - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


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
