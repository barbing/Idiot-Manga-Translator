# -*- coding: utf-8 -*-
"""Stage 5 renderer compositor.

The compositor is intentionally narrow: it draws completed TypesetLayout glyph
placements onto a CleanedPageBase image. It does not run cleanup, create render
regions, reinterpret parent identity, or make style decisions.
"""
from __future__ import annotations

import os
import math
from typing import Any, Iterable, Mapping, Sequence

from app.render.font_manager import FontManager
from app.render.glyph_rasterizer import (
    GLYPH_RASTER_AUTHORITY,
    FreeTypeGlyphRasterizer,
)
from app.render.parent_layer_effects import (
    ParentLayerEffectsResolution,
    resolve_parent_layer_effects,
    shadow_color_rgba,
)
from app.render.typesetting_contracts import (
    DrawingPrimitive,
    FitReport,
    GlyphPlacement,
    RenderLayerPlan,
    TypesetLayout,
    copy_jsonish,
)
from app.render.typesetting_text import source_text_requires_visible_glyph

try:
    from PIL import Image, ImageDraw, ImageFilter
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFilter = None

RENDERER_COMPOSITOR_VERSION = "renderer_compositor_stage5_v6"
PARENT_LAYER_COMPOSITION_VERSION = "isolated_parent_layer_atomic_effects_v4"


AlphaSurfaceStats = tuple[tuple[int, int, int, int] | None, int]


def _alpha_channel_stats(alpha) -> AlphaSurfaceStats:
    """Return exact nonzero bounds and sum without scanning zero canvas margins."""

    bounds = alpha.getbbox()
    if not bounds:
        return None, 0
    normalized_bounds = tuple(int(value) for value in bounds)
    return normalized_bounds, int(sum(alpha.crop(normalized_bounds).getdata()))


def _surface_alpha_stats(surface) -> AlphaSurfaceStats:
    return _alpha_channel_stats(surface.getchannel("A"))


class RendererCompositor:
    """Draw and atomically composite one completed typeset parent layer."""

    def __init__(
        self,
        *,
        font_manager: FontManager | None = None,
        glyph_rasterizer: FreeTypeGlyphRasterizer | None = None,
    ) -> None:
        self.font_manager = font_manager or FontManager()
        self.glyph_rasterizer = glyph_rasterizer or FreeTypeGlyphRasterizer()

    def compose_layer(
        self,
        page,
        plan: RenderLayerPlan,
        layout: TypesetLayout,
        report: FitReport,
    ) -> dict[str, Any]:
        """Rasterize one completed layout and atomically composite it onto ``page``.

        Page ordering, layout planning, typesetting, and ink-fit retry ownership
        intentionally live outside this class.
        """

        if Image is None or ImageDraw is None:
            raise RuntimeError("Pillow is not installed.")
        glyphs = [_glyph_to_dict(item) for item in layout.glyphs]
        issues = list(report.issues or [])
        parent_effects = resolve_parent_layer_effects(plan.resolved_render_style)
        effect_envelope = (
            layout.metadata.get("parent_layer_effect_envelope")
            if isinstance(layout.metadata, Mapping)
            and isinstance(layout.metadata.get("parent_layer_effect_envelope"), Mapping)
            else {}
        )
        base_preflight_issue = _base_layout_preflight_issue(
            plan,
            layout,
            report,
            glyphs,
        )
        if (
            not bool(report.hard_bounds_contained)
            or not bool(layout.hard_bounds_contained)
        ):
            issues.append("parent_layer_base_layout_not_contained")
        optional_effect_degradation = _optional_effect_degradation_reason(
            parent_effects,
            layout,
            report,
            effect_envelope,
        )
        if base_preflight_issue:
            raster_placements = [
                _preflight_failed_raster_audit(glyph, base_preflight_issue)
                for glyph in glyphs
            ]
            issues.append(base_preflight_issue)
            return _layer_audit(
                plan,
                layout,
                report,
                drawn=False,
                drawn_glyph_count=0,
                raster_placements=raster_placements,
                parent_layer_composition=_parent_layer_composition_audit(
                    page,
                    glyphs,
                    raster_placements,
                    status="rejected",
                    rejection_reason=base_preflight_issue,
                    effect_resolution=parent_effects,
                ),
                issues=issues,
            )
        primitive_by_id, primitive_index_issues = _build_drawing_primitive_index(
            layout.drawing_primitives,
            glyphs,
        )
        if primitive_index_issues:
            raster_placements = [
                _preflight_failed_raster_audit(
                    glyph,
                    primitive_index_issues[0],
                )
                for glyph in glyphs
            ]
            issues.extend(primitive_index_issues)
            return _layer_audit(
                plan,
                layout,
                report,
                drawn=False,
                drawn_glyph_count=0,
                raster_placements=raster_placements,
                parent_layer_composition=_parent_layer_composition_audit(
                    page,
                    glyphs,
                    raster_placements,
                    status="rejected",
                    rejection_reason=primitive_index_issues[0],
                    effect_resolution=parent_effects,
                ),
                issues=issues,
            )
        shaped_runs = (
            layout.metadata.get("shaped_runs", [])
            if isinstance(layout.metadata, Mapping)
            else []
        )
        shaped_by_run, shaped_index_issues = _build_shaped_run_index(shaped_runs)
        if shaped_index_issues:
            raster_placements = [
                _preflight_failed_raster_audit(
                    glyph,
                    shaped_index_issues[0],
                )
                for glyph in glyphs
            ]
            issues.extend(shaped_index_issues)
            return _layer_audit(
                plan,
                layout,
                report,
                drawn=False,
                drawn_glyph_count=0,
                raster_placements=raster_placements,
                parent_layer_composition=_parent_layer_composition_audit(
                    page,
                    glyphs,
                    raster_placements,
                    status="rejected",
                    rejection_reason=shaped_index_issues[0],
                    effect_resolution=parent_effects,
                ),
                issues=issues,
            )

        notdef_span_ids = {
            span_id
            for span_id, item in shaped_by_run.items()
            if _shaped_audit_contains_notdef(item)
        }
        if notdef_span_ids:
            raster_placements = [
                _notdef_preflight_raster_audit(glyph, notdef_span_ids)
                for glyph in glyphs
            ]
            issues.append("raster_notdef_glyph_forbidden")
            issues.extend(
                issue
                for item in raster_placements
                for issue in item.get("issues", [])
            )
            return _layer_audit(
                plan,
                layout,
                report,
                drawn=False,
                drawn_glyph_count=0,
                raster_placements=raster_placements,
                parent_layer_composition=_parent_layer_composition_audit(
                    page,
                    glyphs,
                    raster_placements,
                    status="rejected",
                    rejection_reason="raster_notdef_glyph_forbidden",
                    effect_resolution=parent_effects,
                ),
                issues=issues,
            )

        provenance_results = [
            _resolve_shaped_placement_provenance(glyph, shaped_by_run)
            for glyph in glyphs
        ]
        if any(issue for _shaped, issue in provenance_results):
            raster_placements = [
                _preflight_failed_raster_audit(
                    glyph,
                    issue or "raster_parent_rejected_provenance",
                )
                for glyph, (_shaped, issue) in zip(glyphs, provenance_results)
            ]
            issues.append("raster_provenance_preflight_failed")
            issues.extend(
                issue
                for item in raster_placements
                for issue in item.get("issues", [])
            )
            return _layer_audit(
                plan,
                layout,
                report,
                drawn=False,
                drawn_glyph_count=0,
                raster_placements=raster_placements,
                parent_layer_composition=_parent_layer_composition_audit(
                    page,
                    glyphs,
                    raster_placements,
                    status="rejected",
                    rejection_reason="raster_provenance_preflight_failed",
                    effect_resolution=parent_effects,
                ),
                issues=issues,
            )

        parent_surface = Image.new("RGBA", page.size, (0, 0, 0, 0))
        staged_glyph_count = 0
        raster_placements: list[dict[str, Any]] = []
        for glyph in glyphs:
            raster_audit = _draw_glyph(
                parent_surface,
                font_manager=self.font_manager,
                glyph_rasterizer=self.glyph_rasterizer,
                shaped_by_run=shaped_by_run,
                primitive_by_id=primitive_by_id,
                plan=plan,
                layout=layout,
                glyph=glyph,
            )
            raster_placements.append(raster_audit)
            issues.extend(str(item) for item in raster_audit.get("issues", []) or [])
            if raster_audit.get("status") in {"drawn", "primitive"}:
                staged_glyph_count += 1

        expected_visible_count = sum(_glyph_requires_visible_ink(item) for item in glyphs)
        failed_count = sum(
            str(item.get("status") or "") not in {"drawn", "primitive", "no_ink"}
            for item in raster_placements
        )
        parent_alpha_stats = _surface_alpha_stats(parent_surface)
        parent_alpha_bounds = parent_alpha_stats[0]
        combined_containment = (
            _hard_bound_containment(
                page,
                parent_surface,
                (0, 0),
                plan.hard_bounds or plan.target_box,
                alpha_bounds=parent_alpha_bounds or (),
            )
            if parent_alpha_bounds
            else {
                "accepted": expected_visible_count == 0,
                "reason": (
                    "no_visible_placements"
                    if expected_visible_count == 0
                    else "visible_parent_layer_alpha_empty"
                ),
                "parent_hard_bounds": list(_glyph_bounds(plan.hard_bounds or plan.target_box)),
                "page_bounds": [0, 0, int(page.size[0]), int(page.size[1])],
                "raster_alpha_bounds": [],
            }
        )
        rejection_reasons: list[str] = []
        if failed_count:
            rejection_reasons.append("parent_layer_contains_failed_placement")
        if staged_glyph_count != expected_visible_count:
            rejection_reasons.append("parent_layer_visible_placement_count_mismatch")
        if expected_visible_count and not parent_alpha_bounds:
            rejection_reasons.append("parent_layer_visible_alpha_empty")
        if parent_alpha_bounds and not _page_safe_containment(combined_containment):
            rejection_reasons.append("parent_layer_combined_ink_exceeds_page_bounds")
        elif parent_alpha_bounds and not bool(combined_containment.get("accepted")):
            issues.append("parent_layer_combined_ink_exceeds_hard_bounds")

        commit_surface = parent_surface
        final_containment = combined_containment
        if optional_effect_degradation:
            effect_application = _degraded_parent_effect_audit(
                parent_effects,
                parent_surface,
                combined_containment,
                reason=optional_effect_degradation,
                alpha_stats=parent_alpha_stats,
            )
            issues.extend(effect_application.get("issues") or [])
        else:
            effect_application = _inactive_parent_effect_audit(
                parent_effects,
                parent_surface,
                combined_containment,
                alpha_stats=parent_alpha_stats,
            )
        commit_alpha_stats = parent_alpha_stats
        if (
            not rejection_reasons
            and expected_visible_count > 0
            and parent_effects.active
            and not optional_effect_degradation
        ):
            commit_surface, effect_application = _apply_parent_layer_effects(
                page,
                parent_surface,
                plan=plan,
                layout=layout,
                effects=parent_effects,
                base_alpha_stats=parent_alpha_stats,
            )
            final_alpha_bounds = list(
                effect_application.get("final_alpha_bounds") or []
            )
            commit_alpha_stats = (
                (
                    tuple(int(value) for value in final_alpha_bounds)
                    if len(final_alpha_bounds) == 4
                    else None
                ),
                int(effect_application.get("final_alpha_sum") or 0),
            )
            final_containment = dict(
                effect_application.get("final_alpha_containment") or {}
            )
            if not bool(effect_application.get("accepted")):
                effect_issue = str(
                    effect_application.get("rejection_reason")
                    or "effect_raster_envelope_mismatch"
                )
                effect_application = _degraded_parent_effect_audit(
                    parent_effects,
                    parent_surface,
                    combined_containment,
                    reason=effect_issue,
                    attempted=effect_application,
                    alpha_stats=parent_alpha_stats,
                )
                commit_surface = parent_surface
                commit_alpha_stats = parent_alpha_stats
                final_containment = combined_containment
                issues.extend(effect_application.get("issues") or [effect_issue])

        committed = not rejection_reasons and expected_visible_count > 0
        no_ink_only = not rejection_reasons and expected_visible_count == 0
        if committed:
            page.alpha_composite(commit_surface)
        elif rejection_reasons:
            issues.extend(rejection_reasons)
        parent_layer_composition = _parent_layer_composition_audit(
            page,
            glyphs,
            raster_placements,
            surface=commit_surface if committed else parent_surface,
            status=("committed" if committed else "no_ink" if no_ink_only else "rejected"),
            rejection_reason=rejection_reasons[0] if rejection_reasons else "",
            combined_containment=final_containment,
            base_containment=combined_containment,
            page_composite_count=1 if committed else 0,
            effect_resolution=parent_effects,
            effect_application=effect_application,
            surface_alpha_stats=commit_alpha_stats,
        )
        return _layer_audit(
            plan,
            layout,
            report,
            drawn=committed,
            drawn_glyph_count=staged_glyph_count if committed else 0,
            raster_placements=raster_placements,
            parent_layer_composition=parent_layer_composition,
            issues=issues,
        )


def _draw_glyph(
    page,
    *,
    font_manager: FontManager,
    glyph_rasterizer: FreeTypeGlyphRasterizer,
    shaped_by_run: Mapping[str, Mapping[str, Any]],
    primitive_by_id: Mapping[str, Mapping[str, Any]],
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    glyph: Mapping[str, Any],
) -> dict[str, Any]:
    text = str(glyph.get("text") or "")
    bbox = _glyph_bounds(glyph.get("bbox"))
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    run_id = str(metadata.get("run_id") or "")
    font_span_id = str(metadata.get("font_span_id") or run_id)
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
    if metadata.get("font_span_id"):
        base_audit["logical_run_id"] = str(metadata.get("logical_run_id") or run_id)
        base_audit["font_span_id"] = font_span_id
    if not text:
        return _failed_raster_audit(base_audit, "raster_empty_placement_text")
    if bool(metadata.get("space_run")) or text.isspace():
        return {
            **base_audit,
            "raster_authority": "layout_no_ink_space",
            "status": "no_ink",
            "issues": [],
        }
    if not source_text_requires_visible_glyph(text):
        return {
            **base_audit,
            "raster_authority": "unicode_default_ignorable_no_ink",
            "status": "no_ink",
            "issues": [],
        }
    if not bbox:
        return _failed_raster_audit(base_audit, "raster_invalid_placement_bbox")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return _failed_raster_audit(base_audit, "raster_invalid_placement_bbox")
    font_size = int(round(float(glyph.get("font_size") or layout.selected_font_size or 1)))
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    fill, stroke_fill, stroke_width = _resolved_v3_paint(style)

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    shaped_run, provenance_issue = _resolve_shaped_placement_provenance(
        glyph,
        shaped_by_run,
    )
    if provenance_issue or not isinstance(shaped_run, Mapping):
        return _failed_raster_audit(
            base_audit,
            provenance_issue or f"raster_missing_shaped_run:{font_span_id}",
        )

    if str(layout.writing_mode or "").lower() == "vertical" and mode in {
        "vertical_ellipsis_sequence",
        "vertical_dash_sequence",
        "vertical_wave_sequence",
    }:
        primitive_id = str(metadata.get("drawing_primitive_id") or "")
        primitive = primitive_by_id.get(primitive_id)
        if not primitive_id or not isinstance(primitive, Mapping):
            return _failed_raster_audit(
                base_audit,
                f"drawing_primitive_missing:{primitive_id or 'unbound'}",
            )
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        primitive_evidence = _draw_compact_vertical_sequence(
            layer,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            primitive=primitive,
            origin=(x0, y0),
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
            "raster_authority": "typesetting_drawing_primitive_v1",
            "status": "primitive" if _page_safe_containment(containment) else "failed",
            "primitive_id": primitive_id,
            "primitive_type": str(primitive.get("kind") or mode),
            "policy_owner": "TypesettingEngine",
            "geometry_owner": "TypesettingEngine",
            "position_policy": "finalized_typeset_layout_geometry",
            "font_face_id": str(metadata.get("font_face_id") or glyph.get("font_family") or layout.selected_font_face or ""),
            "font_path": str(metadata.get("font_path") or ""),
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
            "issues": (
                []
                if bool(containment.get("accepted"))
                else ["raster_ink_exceeds_parent_hard_bounds"]
                if _page_safe_containment(containment)
                else ["raster_ink_exceeds_page_bounds"]
            ),
        }
        if _page_safe_containment(containment):
            page.alpha_composite(layer, dest=(x0, y0))
        return primitive_audit

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
            else "compact_horizontal_sequence_preserved"
            if mode == "vertical_emphasis_sequence"
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
        if not _page_safe_containment(containment):
            rasterized_ids = list(raster_audit.get("drawn_glyph_ids") or [])
            raster_audit.update(
                {
                    "status": "failed",
                    "rasterized_glyph_ids": rasterized_ids,
                    "drawn_glyph_ids": [],
                    "issues": _unique_strings(
                        [
                            *(raster_audit.get("issues") or []),
                            "raster_ink_exceeds_page_bounds",
                        ]
                    ),
                }
            )
        else:
            if not bool(containment.get("accepted")):
                raster_audit["issues"] = _unique_strings(
                    [
                        *(raster_audit.get("issues") or []),
                        "raster_ink_exceeds_parent_hard_bounds",
                    ]
                )
            page.alpha_composite(result.image, dest=composite_dest)
    return raster_audit


_VERTICAL_PRIMITIVE_KINDS = frozenset(
    {
        "vertical_ellipsis_sequence",
        "vertical_dash_sequence",
        "vertical_wave_sequence",
    }
)


def _build_drawing_primitive_index(
    values: Sequence[DrawingPrimitive | Mapping[str, Any]],
    glyphs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate one-to-one finalized primitive consumption before drawing."""

    primitive_by_id: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for value in list(values or []):
        primitive = _drawing_primitive_to_dict(value)
        primitive_id = str(primitive.get("primitive_id") or "")
        if not primitive_id:
            issues.append("drawing_primitive_id_missing")
            continue
        if primitive_id in primitive_by_id:
            issues.append(f"drawing_primitive_duplicate_id:{primitive_id}")
            continue
        primitive_by_id[primitive_id] = primitive
        geometry_issue = _drawing_primitive_geometry_issue(primitive)
        if geometry_issue:
            issues.append(
                f"drawing_primitive_geometry_invalid:{primitive_id}:{geometry_issue}"
            )

    consumed: list[str] = []
    for index, glyph in enumerate(glyphs):
        metadata = (
            glyph.get("metadata")
            if isinstance(glyph.get("metadata"), Mapping)
            else {}
        )
        mode = str(metadata.get("placement_mode") or "")
        if mode not in _VERTICAL_PRIMITIVE_KINDS:
            continue
        primitive_id = str(metadata.get("drawing_primitive_id") or "")
        if not primitive_id:
            issues.append(f"drawing_primitive_placement_id_missing:{index}")
            continue
        if primitive_id in consumed:
            issues.append(f"drawing_primitive_multiple_consumers:{primitive_id}")
            continue
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            issues.append(f"drawing_primitive_missing:{primitive_id}")
            continue
        consumed.append(primitive_id)
        contract_issue = _drawing_primitive_contract_issue(primitive, glyph)
        if contract_issue:
            issues.append(
                f"drawing_primitive_contract_mismatch:{primitive_id}:{contract_issue}"
            )

    for primitive_id in primitive_by_id:
        if primitive_id not in consumed:
            issues.append(f"drawing_primitive_unconsumed:{primitive_id}")
    return primitive_by_id, _unique_strings(issues)


def _drawing_primitive_contract_issue(
    primitive: Mapping[str, Any],
    glyph: Mapping[str, Any],
) -> str:
    metadata = (
        glyph.get("metadata")
        if isinstance(glyph.get("metadata"), Mapping)
        else {}
    )
    primitive_metadata = (
        primitive.get("metadata")
        if isinstance(primitive.get("metadata"), Mapping)
        else {}
    )
    mode = str(metadata.get("placement_mode") or "")
    if str(primitive.get("kind") or "") != mode:
        return "kind"
    if str(primitive.get("source_text") or "") != str(glyph.get("text") or ""):
        return "source_text"
    if _xywh_int(primitive.get("bounds")) != _xywh_int(glyph.get("bbox")):
        return "bounds"
    primitive_tokens = [str(item) for item in list(primitive.get("token_ids") or [])]
    placement_tokens = [str(item) for item in list(metadata.get("token_ids") or [])]
    if primitive_tokens != placement_tokens:
        return "token_ids"
    if str(primitive.get("orientation") or "") != "vertical":
        return "orientation"
    if str(glyph.get("writing_mode") or "") != "vertical":
        return "placement_writing_mode"
    if not bool(metadata.get("primitive_geometry_final")):
        return "placement_geometry_not_final"
    if str(primitive_metadata.get("geometry_owner") or "") != "TypesettingEngine":
        return "geometry_owner"
    if str(primitive_metadata.get("geometry_status") or "") != "final":
        return "geometry_status"
    if primitive_metadata.get("relative_geometry_recomputation_allowed") is not False:
        return "geometry_recomputation_policy"

    expected_units_key = {
        "vertical_ellipsis_sequence": "ellipsis_unit_count",
        "vertical_dash_sequence": "dash_unit_count",
        "vertical_wave_sequence": "wave_unit_count",
    }.get(mode, "")
    expected_units = _strict_int(metadata.get(expected_units_key), default=0)
    if expected_units > 0 and _strict_int(primitive.get("unit_count"), default=0) != expected_units:
        return "unit_count"
    if mode == "vertical_ellipsis_sequence":
        expected_visible = _strict_int(metadata.get("ellipsis_dot_count"), default=0)
        expected_groups = _strict_int(
            metadata.get("ellipsis_sequence_group_count"),
            default=0,
        )
        if expected_visible > 0 and _strict_int(primitive.get("visible_count"), default=0) != expected_visible:
            return "visible_count"
        if expected_groups > 0 and _strict_int(primitive.get("sequence_group_count"), default=0) != expected_groups:
            return "sequence_group_count"
    return ""


def _drawing_primitive_geometry_issue(primitive: Mapping[str, Any]) -> str:
    kind = str(primitive.get("kind") or "")
    if kind not in _VERTICAL_PRIMITIVE_KINDS:
        return "unsupported_kind"
    bounds = _xywh_float(primitive.get("bounds"))
    if not bounds:
        return "bounds"
    x, y, width, height = bounds
    unit_count = _strict_int(primitive.get("unit_count"), default=0)
    visible_count = _strict_int(primitive.get("visible_count"), default=0)
    sequence_group_count = _strict_int(
        primitive.get("sequence_group_count"),
        default=0,
    )
    if unit_count < 1 or visible_count < 1 or sequence_group_count < 1:
        return "counts"
    primitive_metadata = (
        primitive.get("metadata")
        if isinstance(primitive.get("metadata"), Mapping)
        else {}
    )
    outline_width = _strict_float(
        primitive_metadata.get("outline_width_px"),
        default=0.0,
    )
    if outline_width < 0.0:
        return "outline_width"

    if kind == "vertical_ellipsis_sequence":
        diameter = _strict_float(primitive.get("diameter_px"), default=0.0)
        pitch = _strict_float(primitive.get("pitch_px"), default=-1.0)
        centers = _point_sequence(primitive.get("centers"))
        if diameter <= 0.0 or pitch < 0.0:
            return "ellipsis_metrics"
        if len(centers) != visible_count:
            return "ellipsis_center_count"
        radius = diameter / 2.0 + outline_width
        if any(
            center_x - radius < x - 1e-3
            or center_x + radius > x + width + 1e-3
            or center_y - radius < y - 1e-3
            or center_y + radius > y + height + 1e-3
            for center_x, center_y in centers
        ):
            return "ellipsis_centers_outside_bounds"
        if len(centers) > 1:
            if any(abs(center_x - centers[0][0]) > 1e-3 for center_x, _ in centers):
                return "ellipsis_multiple_columns"
            deltas = [
                centers[index + 1][1] - centers[index][1]
                for index in range(len(centers) - 1)
            ]
            if any(delta <= 0.0 for delta in deltas):
                return "ellipsis_nonascending_centers"
            if any(abs(delta - pitch) > 0.02 for delta in deltas):
                return "ellipsis_pitch"
        return ""

    line_width = _strict_float(primitive.get("line_width_px"), default=0.0)
    points = _point_sequence(primitive.get("points"))
    minimum_points = 2
    if line_width <= 0.0 or len(points) < minimum_points:
        return "line_geometry"
    margin = line_width / 2.0 + outline_width
    if any(
        point_x - margin < x - 1e-3
        or point_x + margin > x + width + 1e-3
        or point_y - margin < y - 1e-3
        or point_y + margin > y + height + 1e-3
        for point_x, point_y in points
    ):
        return "line_points_outside_bounds"
    return ""


def _draw_compact_vertical_sequence(
    layer,
    *,
    fill,
    stroke_width: int,
    stroke_fill,
    primitive: Mapping[str, Any],
    origin: Sequence[int],
) -> dict[str, Any]:
    if ImageDraw is None:
        return {}
    kind = str(primitive.get("kind") or "")
    if kind not in _VERTICAL_PRIMITIVE_KINDS:
        return {}
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    if kind == "vertical_ellipsis_sequence":
        return _draw_finalized_vertical_ellipsis(
            layer,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            primitive=primitive,
            origin=(origin_x, origin_y),
        )
    return _draw_finalized_vertical_line(
        layer,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        primitive=primitive,
        origin=(origin_x, origin_y),
    )


def _draw_finalized_vertical_ellipsis(
    layer,
    *,
    fill,
    stroke_width: int,
    stroke_fill,
    primitive: Mapping[str, Any],
    origin: Sequence[float],
) -> dict[str, Any]:
    centers = _point_sequence(primitive.get("centers"))
    diameter = float(primitive.get("diameter_px") or 0.0)
    radius = diameter / 2.0
    safe_stroke = max(0, int(stroke_width))
    origin_x, origin_y = float(origin[0]), float(origin[1])
    draw = ImageDraw.Draw(layer)
    for center_x, center_y in centers:
        local_x = center_x - origin_x
        local_y = center_y - origin_y
        if safe_stroke > 0:
            outer = radius + float(safe_stroke)
            draw.ellipse(
                (
                    local_x - outer,
                    local_y - outer,
                    local_x + outer,
                    local_y + outer,
                ),
                fill=stroke_fill,
            )
        draw.ellipse(
            (
                local_x - radius,
                local_y - radius,
                local_x + radius,
                local_y + radius,
            ),
            fill=fill,
        )
    deltas = [
        round(centers[index + 1][1] - centers[index][1], 4)
        for index in range(len(centers) - 1)
    ]
    maximum_delta = max(deltas) - min(deltas) if deltas else 0.0
    return {
        "primitive_geometry_source": "TypesetLayout.drawing_primitives",
        "primitive_bounds": list(primitive.get("bounds") or []),
        "ellipsis_unit_count": int(primitive.get("unit_count") or 0),
        "dot_count": int(primitive.get("visible_count") or 0),
        "dot_column_count": 1,
        "sequence_group_count": int(primitive.get("sequence_group_count") or 0),
        "dot_diameter_px": float(diameter),
        "dot_centers": [list(item) for item in centers],
        "dot_pitch_px": float(primitive.get("pitch_px") or 0.0),
        "dot_pitch_deltas": deltas,
        "max_dot_pitch_delta_px": round(float(maximum_delta), 4),
        "ellipsis_policy": "one_continuous_uniform_dot_sequence",
    }


def _draw_finalized_vertical_line(
    layer,
    *,
    fill,
    stroke_width: int,
    stroke_fill,
    primitive: Mapping[str, Any],
    origin: Sequence[float],
) -> dict[str, Any]:
    points = _point_sequence(primitive.get("points"))
    origin_x, origin_y = float(origin[0]), float(origin[1])
    local_points = [
        (point_x - origin_x, point_y - origin_y)
        for point_x, point_y in points
    ]
    requested_width = float(primitive.get("line_width_px") or 0.0)
    raster_width = max(1, int(round(requested_width)))
    safe_stroke = max(0, int(stroke_width))
    draw = ImageDraw.Draw(layer)
    if safe_stroke > 0:
        draw.line(
            local_points,
            fill=stroke_fill,
            width=raster_width + safe_stroke * 2,
            joint="curve",
        )
    draw.line(
        local_points,
        fill=fill,
        width=raster_width,
        joint="curve",
    )
    kind = str(primitive.get("kind") or "")
    metadata = (
        primitive.get("metadata")
        if isinstance(primitive.get("metadata"), Mapping)
        else {}
    )
    evidence = {
        "primitive_geometry_source": "TypesetLayout.drawing_primitives",
        "primitive_bounds": list(primitive.get("bounds") or []),
        "primitive_points": [list(item) for item in points],
        "line_width_px": float(requested_width),
    }
    if kind == "vertical_dash_sequence":
        evidence.update(
            {
                "dash_unit_count": int(primitive.get("unit_count") or 0),
                "continuous_segment_count": 1,
                "continuous_multi_cell_dash": True,
            }
        )
    else:
        evidence.update(
            {
                "wave_unit_count": int(primitive.get("unit_count") or 0),
                "wave_cycle_count": float(
                    metadata.get("wave_cycle_count")
                    or primitive.get("unit_count")
                    or 0
                ),
                "continuous_multi_cell_wave": True,
                "wave_source_classes": [
                    str(item.get("source_class") or "")
                    for item in list(metadata.get("punctuation_occurrences") or [])
                    if isinstance(item, Mapping)
                ],
            }
        )
    return evidence


def _glyph_requires_visible_ink(glyph: Mapping[str, Any]) -> bool:
    text = str(glyph.get("text") or "")
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    if not text or bool(metadata.get("space_run")) or text.isspace():
        return False
    return bool(source_text_requires_visible_glyph(text))


def _base_layout_preflight_issue(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    glyphs: Sequence[Mapping[str, Any]],
) -> str:
    """Return the first hard base-text failure, independent of optional effects."""

    if (
        not bool(report.text_placement_complete)
        or not bool(layout.text_placement_complete)
    ):
        return "parent_layer_base_layout_not_renderable"
    if str(layout.original_text or "") != str(plan.translated_text or ""):
        return "parent_layer_layout_source_text_mismatch"
    glyph_text = "".join(str(item.get("text") or "") for item in glyphs)
    if glyph_text != str(layout.normalized_text or ""):
        return "parent_layer_layout_glyph_text_mismatch"
    if (
        bool(plan.render_required)
        and source_text_requires_visible_glyph(str(plan.translated_text or ""))
        and not any(_glyph_requires_visible_ink(item) for item in glyphs)
    ):
        return "parent_layer_visible_glyphs_missing"
    return ""


def _optional_effect_degradation_reason(
    effects: ParentLayerEffectsResolution,
    layout: TypesetLayout,
    report: FitReport,
    effect_envelope: Mapping[str, Any],
) -> str:
    """Classify an effect-only failure without changing base-text eligibility."""

    metadata = layout.metadata if isinstance(layout.metadata, Mapping) else {}
    degradation = (
        metadata.get("optional_effect_degradation")
        if isinstance(metadata.get("optional_effect_degradation"), Mapping)
        else {}
    )
    if str(degradation.get("status") or "") == "degraded_to_base":
        return str(degradation.get("reason") or "optional_parent_effect_unavailable")
    if effects.status == "invalid":
        return "parent_layer_effect_contract_invalid"
    if effects.active and (
        "parent_layer_effect_envelope_exceeds_hard_bounds"
        in {str(item) for item in list(report.issues or [])}
    ):
        return "parent_layer_effect_envelope_exceeds_hard_bounds"
    if (
        effects.active
        and effect_envelope
        and effect_envelope.get("contained") is False
    ):
        return "parent_layer_effect_envelope_exceeds_hard_bounds"
    return ""


def _inactive_parent_effect_audit(
    effects: ParentLayerEffectsResolution,
    surface,
    containment: Mapping[str, Any],
    *,
    alpha_stats: AlphaSurfaceStats | None = None,
) -> dict[str, Any]:
    resolved_stats = alpha_stats or _surface_alpha_stats(surface)
    bounds = list(resolved_stats[0] or ())
    alpha_sum = int(resolved_stats[1])
    status = "unavailable" if effects.status == "unavailable" else "no_visible_effect"
    shadow = effects.shadow.to_audit_dict()
    if shadow.get("availability") == "resolved":
        shadow["source"] = "rotated_parent_alpha"
    return {
        "parent_layer_effects_version": "parent_layer_effect_application_v1",
        "status": status,
        "accepted": True,
        "requested": bool(effects.requested),
        "active": False,
        "whole_parent_transform_count": 0,
        "rotation": effects.rotation.to_audit_dict(),
        "shadow": shadow,
        "effect_application_order": "none_stage3a_exact_path",
        "base_alpha_bounds": bounds,
        "base_alpha_sum": alpha_sum,
        "rotated_alpha_bounds": bounds,
        "rotated_alpha_sum": alpha_sum,
        "shadow_alpha_bounds": [],
        "shadow_alpha_sum": 0,
        "final_alpha_bounds": bounds,
        "final_alpha_sum": alpha_sum,
        "final_alpha_containment": copy_jsonish(containment),
        "predicted_envelope": [],
        "predicted_envelope_contains_actual": True,
        "rotation_sampling": "none",
        "shadow_offset_sampling": "none",
        "untransformed_fallback_used": False,
        "rejection_reason": "",
        "issues": [],
    }


def _degraded_parent_effect_audit(
    effects: ParentLayerEffectsResolution,
    surface,
    containment: Mapping[str, Any],
    *,
    reason: str,
    attempted: Mapping[str, Any] | None = None,
    alpha_stats: AlphaSurfaceStats | None = None,
) -> dict[str, Any]:
    """Describe a base-text commit after an optional effect could not be used."""

    resolved_stats = alpha_stats or _surface_alpha_stats(surface)
    bounds = list(resolved_stats[0] or ())
    alpha_sum = int(resolved_stats[1])
    attempted_audit = dict(attempted or {})
    shadow = effects.shadow.to_audit_dict()
    if shadow.get("availability") == "resolved":
        shadow["source"] = "rotated_parent_alpha"
    accepted = bool(bounds and containment.get("accepted"))
    return {
        "parent_layer_effects_version": "parent_layer_effect_application_v2",
        "status": "degraded_to_base",
        "accepted": accepted,
        "requested": bool(effects.requested),
        "active": bool(effects.active),
        "whole_parent_transform_count": 0,
        "attempted_status": str(attempted_audit.get("status") or "not_attempted"),
        "attempted_whole_parent_transform_count": int(
            attempted_audit.get("whole_parent_transform_count") or 0
        ),
        "rotation": effects.rotation.to_audit_dict(),
        "shadow": shadow,
        "effect_application_order": "base_text_only_after_optional_effect_degradation",
        "base_alpha_bounds": bounds,
        "base_alpha_sum": alpha_sum,
        "rotated_alpha_bounds": bounds,
        "rotated_alpha_sum": alpha_sum,
        "shadow_alpha_bounds": [],
        "shadow_alpha_sum": 0,
        "final_alpha_bounds": bounds,
        "final_alpha_sum": alpha_sum,
        "final_alpha_containment": copy_jsonish(containment),
        "predicted_envelope": list(attempted_audit.get("predicted_envelope") or []),
        "predicted_envelope_contains_actual": True,
        "rotation_sampling": "none_committed_base",
        "shadow_offset_sampling": "none_committed_base",
        "untransformed_fallback_used": True,
        "degradation_reason": str(reason or "optional_parent_effect_unavailable"),
        "rejection_reason": "",
        "issues": _unique_strings(
            [
                str(reason or "optional_parent_effect_unavailable"),
                "optional_parent_effect_degraded_to_base",
                *effects.issues,
                *(attempted_audit.get("issues") or []),
            ]
        ),
    }


def _apply_parent_layer_effects(
    page,
    parent_surface,
    *,
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    effects: ParentLayerEffectsResolution,
    base_alpha_stats: AlphaSurfaceStats | None = None,
) -> tuple[Any, dict[str, Any]]:
    resolved_base_stats = base_alpha_stats or _surface_alpha_stats(parent_surface)
    base_bounds = list(resolved_base_stats[0] or ())
    base_sum = int(resolved_base_stats[1])
    pivot = _effect_pivot(layout, plan)
    angle = (
        float(effects.rotation.degrees_clockwise)
        if effects.rotation.availability == "resolved"
        else 0.0
    )
    rotation_active = abs(angle) >= 1e-9
    rotated = parent_surface
    rotation_sampling = "none"
    if rotation_active:
        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        rotated = (
            parent_surface.convert("RGBa")
            .rotate(
                -angle,
                resample=resampling,
                center=(float(pivot[0]), float(pivot[1])),
                expand=False,
                fillcolor=(0, 0, 0, 0),
            )
            .convert("RGBA")
        )
        rotation_sampling = "premultiplied_rgba_bicubic_expand_false"
    rotated_alpha = rotated.getchannel("A")
    rotated_stats = (
        resolved_base_stats
        if rotated is parent_surface
        else _alpha_channel_stats(rotated_alpha)
    )
    rotated_bounds = list(rotated_stats[0] or ())
    rotated_sum = int(rotated_stats[1])

    shadow_surface = Image.new("RGBA", parent_surface.size, (0, 0, 0, 0))
    shadow_bounds: list[int] = []
    shadow_sum = 0
    shadow_sampling = "none"
    shadow = effects.shadow.to_audit_dict()
    if effects.shadow.availability == "resolved":
        shadow["source"] = "rotated_parent_alpha"
    if effects.shadow.visible:
        shadow_alpha = rotated_alpha
        blur = float(effects.shadow.blur_radius_px)
        if blur > 0.0:
            if ImageFilter is None:
                return parent_surface, {
                    "status": "rejected",
                    "accepted": False,
                    "requested": True,
                    "active": True,
                    "whole_parent_transform_count": 0,
                    "rotation": effects.rotation.to_audit_dict(),
                    "shadow": shadow,
                    "rejection_reason": "pillow_gaussian_blur_unavailable",
                    "untransformed_fallback_used": False,
                    "issues": ["pillow_gaussian_blur_unavailable"],
                }
            shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=blur))
        offset_x, offset_y = [float(value) for value in effects.shadow.offset_px]
        shadow_alpha, shadow_sampling = _translate_alpha(
            shadow_alpha,
            offset_x,
            offset_y,
        )
        red, green, blue, color_alpha = shadow_color_rgba(effects.shadow.color)
        if color_alpha < 255:
            shadow_alpha = shadow_alpha.point(
                lambda value: int(round(float(value) * float(color_alpha) / 255.0))
            )
        shadow_surface = Image.new("RGBA", parent_surface.size, (red, green, blue, 0))
        shadow_surface.putalpha(shadow_alpha)
        shadow_stats = _alpha_channel_stats(shadow_alpha)
        shadow_bounds = list(shadow_stats[0] or ())
        shadow_sum = int(shadow_stats[1])

    final_surface = (
        Image.alpha_composite(shadow_surface, rotated)
        if effects.shadow.visible
        else rotated
    )
    final_stats = (
        rotated_stats
        if final_surface is rotated
        else _surface_alpha_stats(final_surface)
    )
    final_bounds = list(final_stats[0] or ())
    final_sum = int(final_stats[1])
    containment = _hard_bound_containment(
        page,
        final_surface,
        (0, 0),
        plan.hard_bounds or plan.target_box,
        alpha_bounds=final_bounds,
    )
    predicted = _glyph_bounds(layout.measured_bounds)
    predicted_contains_actual = bool(
        predicted and final_bounds and _xyxy_contains(predicted, final_bounds)
    )
    accepted = bool(
        final_bounds
        and containment.get("accepted")
        and predicted_contains_actual
    )
    rejection_reason = ""
    if not final_bounds:
        rejection_reason = "transformed_parent_layer_alpha_empty"
    elif not predicted_contains_actual:
        rejection_reason = "effect_raster_envelope_mismatch"
    elif not bool(containment.get("accepted")):
        rejection_reason = "transformed_parent_layer_exceeds_hard_bounds"
    issues = [rejection_reason] if rejection_reason else []
    return final_surface, {
        "parent_layer_effects_version": "parent_layer_effect_application_v1",
        "status": "applied" if accepted else "rejected",
        "accepted": accepted,
        "requested": bool(effects.requested),
        "active": True,
        "whole_parent_transform_count": 1,
        "rotation": effects.rotation.to_audit_dict(),
        "shadow": shadow,
        "effect_application_order": "rotate_complete_parent_then_shadow_from_rotated_alpha_then_text",
        "rotation_pivot": [round(float(pivot[0]), 6), round(float(pivot[1]), 6)],
        "base_alpha_bounds": base_bounds,
        "base_alpha_sum": base_sum,
        "rotated_alpha_bounds": rotated_bounds,
        "rotated_alpha_sum": rotated_sum,
        "shadow_alpha_bounds": shadow_bounds,
        "shadow_alpha_sum": shadow_sum,
        "final_alpha_bounds": final_bounds,
        "final_alpha_sum": final_sum,
        "final_alpha_containment": copy_jsonish(containment),
        "predicted_envelope": list(predicted),
        "predicted_envelope_contains_actual": predicted_contains_actual,
        "rotation_sampling": rotation_sampling,
        "shadow_offset_sampling": shadow_sampling,
        "untransformed_fallback_used": False,
        "rejection_reason": rejection_reason,
        "issues": issues,
    }


def _translate_alpha(alpha, offset_x: float, offset_y: float):
    if abs(offset_x - round(offset_x)) < 1e-9 and abs(offset_y - round(offset_y)) < 1e-9:
        shifted = Image.new("L", alpha.size, 0)
        shifted.paste(alpha, (int(round(offset_x)), int(round(offset_y))))
        return shifted, "integer_exact_paste"
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    transform = getattr(getattr(Image, "Transform", Image), "AFFINE")
    shifted = alpha.transform(
        alpha.size,
        transform,
        (1.0, 0.0, -float(offset_x), 0.0, 1.0, -float(offset_y)),
        resample=resampling,
        fillcolor=0,
    )
    return shifted, "fractional_affine_bicubic"


def _effect_pivot(layout: TypesetLayout, plan: RenderLayerPlan) -> tuple[float, float]:
    if isinstance(layout.visual_center, Sequence) and len(layout.visual_center) >= 2:
        try:
            return float(layout.visual_center[0]), float(layout.visual_center[1])
        except (TypeError, ValueError):
            pass
    metadata = layout.metadata if isinstance(layout.metadata, Mapping) else {}
    base = _glyph_bounds(metadata.get("base_measured_bounds"))
    if not base:
        base = _glyph_bounds(layout.measured_bounds) or _glyph_bounds(plan.target_box)
    return (
        float(base[0] + base[2]) / 2.0,
        float(base[1] + base[3]) / 2.0,
    )


def _parent_layer_composition_audit(
    page,
    glyphs: Sequence[Mapping[str, Any]],
    raster_placements: Sequence[Mapping[str, Any]],
    *,
    status: str,
    surface=None,
    rejection_reason: str = "",
    combined_containment: Mapping[str, Any] | None = None,
    base_containment: Mapping[str, Any] | None = None,
    page_composite_count: int = 0,
    effect_resolution: ParentLayerEffectsResolution | None = None,
    effect_application: Mapping[str, Any] | None = None,
    surface_alpha_stats: AlphaSurfaceStats | None = None,
) -> dict[str, Any]:
    raster_items = [dict(item) for item in raster_placements]
    expected_visible_count = sum(_glyph_requires_visible_ink(item) for item in glyphs)
    staged_visible_count = sum(
        str(item.get("status") or "") in {"drawn", "primitive"}
        for item in raster_items
    )
    no_ink_count = sum(str(item.get("status") or "") == "no_ink" for item in raster_items)
    failed_count = sum(
        str(item.get("status") or "") not in {"drawn", "primitive", "no_ink"}
        for item in raster_items
    )
    hard_bound_failure_count = sum(
        isinstance(item.get("hard_bound_containment"), Mapping)
        and not bool(item.get("hard_bound_containment", {}).get("accepted"))
        for item in raster_items
    )
    alpha_bounds: list[int] = []
    alpha_sum = 0
    if surface is not None:
        resolved_stats = surface_alpha_stats or _surface_alpha_stats(surface)
        if resolved_stats[0]:
            alpha_bounds = [int(item) for item in resolved_stats[0]]
        alpha_sum = int(resolved_stats[1])
    committed = str(status or "") == "committed"
    effects = effect_resolution or ParentLayerEffectsResolution()
    effects_audit = effects.to_audit_dict()
    application = dict(effect_application or {})
    composition_issues = _unique_strings(
        [
            rejection_reason,
            *effects.issues,
            *(application.get("issues") or []),
        ]
    )
    if application:
        effects_status = str(application.get("status") or "rejected")
    elif effects.status == "invalid":
        effects_status = "invalid"
    elif effects.requested and str(status or "") == "rejected":
        effects_status = "rejected"
    else:
        effects_status = "unavailable" if effects.status == "unavailable" else "no_visible_effect"
    rotation = dict(application.get("rotation") or effects_audit.get("rotation") or {})
    shadow = dict(application.get("shadow") or effects_audit.get("shadow") or {})
    if shadow.get("availability") == "resolved" and "source" not in shadow:
        shadow["source"] = "rotated_parent_alpha"
    containment = dict(combined_containment or {})
    if not containment:
        containment = {
            "accepted": False,
            "reason": "surface_not_created_preflight_rejection",
            "parent_hard_bounds": [],
            "page_bounds": [0, 0, int(page.size[0]), int(page.size[1])],
            "raster_alpha_bounds": [],
        }
    return {
        "parent_layer_composition_version": PARENT_LAYER_COMPOSITION_VERSION,
        "policy": PARENT_LAYER_COMPOSITION_VERSION,
        "coordinate_space": "page",
        "surface_mode": "RGBA",
        "surface_created": surface is not None,
        "surface_size": [int(page.size[0]), int(page.size[1])],
        "initial_surface_alpha_sum": 0,
        "effects_status": effects_status,
        "effect_requested": bool(effects.requested),
        "parent_layer_effects": effects_audit,
        "rotation_status": str(rotation.get("availability") or "unavailable"),
        "shadow_status": str(shadow.get("availability") or "unavailable"),
        "rotation": rotation,
        "shadow": shadow,
        "whole_parent_transform_count": int(
            application.get("whole_parent_transform_count") or 0
        ),
        "effect_application_order": str(
            application.get("effect_application_order") or "none_stage3a_exact_path"
        ),
        "expected_placement_count": len(glyphs),
        "audited_placement_count": len(raster_items),
        "expected_visible_placement_count": int(expected_visible_count),
        "staged_visible_placement_count": int(staged_visible_count),
        "committed_placement_count": int(staged_visible_count if committed else 0),
        "no_ink_placement_count": int(no_ink_count),
        "failed_placement_count": int(failed_count),
        "hard_bound_containment_failure_count": int(hard_bound_failure_count),
        "surface_alpha_bounds": alpha_bounds,
        "surface_alpha_sum": int(alpha_sum),
        "base_alpha_bounds": list(application.get("base_alpha_bounds") or alpha_bounds),
        "base_alpha_sum": int(application.get("base_alpha_sum") or alpha_sum),
        "rotated_alpha_bounds": list(application.get("rotated_alpha_bounds") or alpha_bounds),
        "rotated_alpha_sum": int(application.get("rotated_alpha_sum") or alpha_sum),
        "shadow_alpha_bounds": list(application.get("shadow_alpha_bounds") or []),
        "shadow_alpha_sum": int(application.get("shadow_alpha_sum") or 0),
        "final_alpha_bounds": list(application.get("final_alpha_bounds") or alpha_bounds),
        "final_alpha_sum": int(application.get("final_alpha_sum") or alpha_sum),
        "hard_bound_containment": copy_jsonish(containment),
        "base_alpha_containment": copy_jsonish(base_containment or containment),
        "final_alpha_containment": copy_jsonish(
            application.get("final_alpha_containment") or containment
        ),
        "predicted_envelope": list(application.get("predicted_envelope") or []),
        "predicted_envelope_contains_actual": bool(
            application.get("predicted_envelope_contains_actual", True)
        ),
        "rotation_sampling": str(application.get("rotation_sampling") or "none"),
        "shadow_offset_sampling": str(
            application.get("shadow_offset_sampling") or "none"
        ),
        "accepted": bool(committed),
        "status": str(status or "rejected"),
        "page_composite_count": int(page_composite_count),
        "atomic_commit": True,
        "partial_parent_pixels_committed": False,
        "untransformed_fallback_used": bool(
            application.get("untransformed_fallback_used", False)
        ),
        "optional_effect_degraded": bool(
            str(application.get("status") or "") == "degraded_to_base"
        ),
        "effect_degradation_reason": str(
            application.get("degradation_reason") or ""
        ),
        "effect_attempted_status": str(
            application.get("attempted_status") or "not_attempted"
        ),
        "effect_attempted_whole_parent_transform_count": int(
            application.get("attempted_whole_parent_transform_count") or 0
        ),
        "rejection_reason": str(rejection_reason or ""),
        "issues": composition_issues,
    }


def _layer_audit(
    plan: RenderLayerPlan,
    layout: TypesetLayout,
    report: FitReport,
    *,
    drawn: bool,
    drawn_glyph_count: int = 0,
    raster_placements: Sequence[Mapping[str, Any]] | None = None,
    parent_layer_composition: Mapping[str, Any] | None = None,
    issues: Sequence[str] | None = None,
) -> dict[str, Any]:
    style = plan.resolved_render_style if isinstance(plan.resolved_render_style, Mapping) else {}
    glyph_text = "".join(str(item.get("text") or "") for item in (_glyph_to_dict(g) for g in layout.glyphs))
    normalized = str(layout.normalized_text or "")
    raster_items = [dict(item) for item in list(raster_placements or [])]
    composition = dict(parent_layer_composition or {})
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
        "render_status": str(composition.get("status") or "rejected"),
        "render_rejection_reason": str(composition.get("rejection_reason") or ""),
        "parent_layer_composition": copy_jsonish(composition),
        "raster_placements": copy_jsonish(raster_items),
        "glyph_text_matches_layout": glyph_text == normalized,
        "full_text_placed": bool(report.text_placement_complete),
        "text_placement_complete": bool(report.text_placement_complete),
        "hard_bounds_contained": bool(report.hard_bounds_contained),
        "fit_quality": str(report.fit_quality or ""),
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
        "fill_color": (
            style.get("fill", {}).get("color")
            if isinstance(style.get("fill"), Mapping)
            else None
        ),
        "stroke_color": (
            style.get("outline", {}).get("color")
            if isinstance(style.get("outline"), Mapping)
            else None
        ),
        "stroke_width": (
            style.get("outline", {}).get("target_width_px")
            if isinstance(style.get("outline"), Mapping)
            else None
        ),
        "issues": _unique_strings(issues or []),
    }


def _failed_raster_audit(audit: Mapping[str, Any], issue: str) -> dict[str, Any]:
    payload = dict(audit)
    payload["status"] = "failed"
    payload["issues"] = _unique_strings([*(payload.get("issues") or []), issue])
    return payload


def _shaped_audit_contains_notdef(shaped_run: Mapping[str, Any]) -> bool:
    for glyph in list(shaped_run.get("glyphs") or []):
        if not isinstance(glyph, Mapping):
            continue
        try:
            glyph_id = int(glyph.get("glyph_id"))
        except (TypeError, ValueError):
            continue
        if glyph_id == 0 and source_text_requires_visible_glyph(
            str(glyph.get("text") or "")
        ):
            return True
    return False


def _build_shaped_run_index(
    shaped_runs: Sequence[Any],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    output: dict[str, Mapping[str, Any]] = {}
    issues: list[str] = []
    for item in shaped_runs:
        if not isinstance(item, Mapping):
            issues.append("raster_invalid_shaped_run_record")
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        span_id = str(metadata.get("font_span_id") or metadata.get("run_id") or "")
        if not span_id:
            issues.append("raster_missing_shaped_span_id")
            continue
        if span_id in output:
            issues.append("raster_duplicate_shaped_span_id")
            continue
        output[span_id] = item
    return output, _unique_strings(issues)


def _resolve_shaped_placement_provenance(
    glyph: Mapping[str, Any],
    shaped_by_run: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str]:
    text = str(glyph.get("text") or "")
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    if bool(metadata.get("space_run")) or text.isspace():
        return None, ""
    if not source_text_requires_visible_glyph(text):
        return None, ""
    run_id = str(metadata.get("run_id") or "")
    span_id = str(metadata.get("font_span_id") or run_id)
    shaped_run = shaped_by_run.get(span_id)
    if not isinstance(shaped_run, Mapping):
        return None, f"raster_missing_shaped_run:{span_id}"
    if _shaped_audit_contains_notdef(shaped_run):
        return shaped_run, "raster_notdef_glyph_forbidden"
    shaped_metadata = (
        shaped_run.get("metadata")
        if isinstance(shaped_run.get("metadata"), Mapping)
        else {}
    )
    shaped_run_id = str(shaped_metadata.get("run_id") or "")
    shaped_span_id = str(shaped_metadata.get("font_span_id") or shaped_run_id)
    if run_id and shaped_run_id != run_id:
        return shaped_run, f"raster_run_identity_mismatch:{run_id}:{shaped_run_id}"
    if span_id and shaped_span_id != span_id:
        return shaped_run, f"raster_span_identity_mismatch:{span_id}:{shaped_span_id}"
    placement_logical_id = str(metadata.get("logical_run_id") or "")
    shaped_logical_id = str(shaped_metadata.get("logical_run_id") or "")
    if (
        placement_logical_id
        and shaped_logical_id
        and placement_logical_id != shaped_logical_id
    ):
        return shaped_run, (
            f"raster_logical_run_identity_mismatch:"
            f"{placement_logical_id}:{shaped_logical_id}"
        )
    shaped_face_id = str(shaped_run.get("font_face_id") or "")
    shaped_font_path = str(shaped_run.get("font_path") or "")
    placement_face_id = str(metadata.get("font_face_id") or glyph.get("font_family") or "")
    placement_font_path = str(metadata.get("font_path") or "")
    if placement_face_id and shaped_face_id != placement_face_id:
        return shaped_run, f"raster_face_mismatch:{placement_face_id}:{shaped_face_id}"
    if placement_font_path and not _same_path(placement_font_path, shaped_font_path):
        return shaped_run, "raster_font_path_mismatch"
    shaped_text = str(shaped_run.get("normalized_text") or shaped_run.get("text") or "")
    shaped_glyphs = list(shaped_run.get("glyphs") or [])
    if source_text_requires_visible_glyph(shaped_text) and not shaped_glyphs:
        return shaped_run, "raster_visible_shaped_run_empty"
    if text != shaped_text:
        return shaped_run, f"raster_text_provenance_mismatch:{text}:{shaped_text}"
    shaped_glyph_ids: list[int] = []
    for shaped_glyph in shaped_glyphs:
        if not isinstance(shaped_glyph, Mapping):
            return shaped_run, "raster_invalid_shaped_glyph_record"
        try:
            shaped_glyph_id = int(shaped_glyph.get("glyph_id"))
        except (TypeError, ValueError):
            return shaped_run, "raster_invalid_shaped_glyph_id"
        if shaped_glyph_id <= 0 and source_text_requires_visible_glyph(
            str(shaped_glyph.get("text") or shaped_text)
        ):
            return shaped_run, "raster_notdef_glyph_forbidden"
        shaped_glyph_ids.append(shaped_glyph_id)
    requested_raw = metadata.get("shaped_glyph_ids")
    if isinstance(requested_raw, Sequence) and not isinstance(
        requested_raw,
        (str, bytes, bytearray),
    ):
        try:
            requested_glyph_ids = [int(item) for item in requested_raw]
        except (TypeError, ValueError):
            return shaped_run, "raster_invalid_requested_glyph_id"
        if requested_glyph_ids and not _is_contiguous_subsequence(
            requested_glyph_ids,
            shaped_glyph_ids,
        ):
            return shaped_run, "raster_requested_glyph_sequence_mismatch"
    return shaped_run, ""


def _is_contiguous_subsequence(values: Sequence[int], source: Sequence[int]) -> bool:
    requested = list(values)
    available = list(source)
    if not requested:
        return True
    if len(requested) > len(available):
        return False
    return any(
        available[index : index + len(requested)] == requested
        for index in range(len(available) - len(requested) + 1)
    )


def _notdef_preflight_raster_audit(
    glyph: Mapping[str, Any],
    notdef_span_ids: set[str],
) -> dict[str, Any]:
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    run_id = str(metadata.get("run_id") or "")
    font_span_id = str(metadata.get("font_span_id") or run_id)
    issue = (
        "raster_notdef_glyph_forbidden"
        if font_span_id in notdef_span_ids
        else "raster_parent_rejected_notdef"
    )
    return _preflight_failed_raster_audit(glyph, issue)


def _preflight_failed_raster_audit(
    glyph: Mapping[str, Any],
    issue: str,
) -> dict[str, Any]:
    metadata = glyph.get("metadata") if isinstance(glyph.get("metadata"), Mapping) else {}
    run_id = str(metadata.get("run_id") or "")
    font_span_id = str(metadata.get("font_span_id") or run_id)
    audit: dict[str, Any] = {
        "raster_authority": GLYPH_RASTER_AUTHORITY,
        "status": "failed",
        "placement_text": str(glyph.get("text") or ""),
        "run_id": run_id,
        "placement_mode": str(metadata.get("placement_mode") or ""),
        "placement_bbox": list(_glyph_bounds(glyph.get("bbox"))),
        "font_face_id": str(metadata.get("font_face_id") or glyph.get("font_family") or ""),
        "font_path": str(metadata.get("font_path") or ""),
        "requested_glyph_ids": [],
        "drawn_glyph_ids": [],
        "advances_offsets_consumed": False,
        "issues": [str(issue or "raster_preflight_failed")],
    }
    if metadata.get("font_span_id"):
        audit["logical_run_id"] = str(metadata.get("logical_run_id") or run_id)
        audit["font_span_id"] = font_span_id
    return audit


def _hard_bound_containment(
    page,
    raster,
    dest: tuple[int, int],
    hard_bounds: Sequence[Any] | None,
    *,
    alpha_bounds: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if alpha_bounds is None:
        alpha_box = raster.getchannel("A").getbbox() if raster is not None else None
    else:
        values = [int(value) for value in alpha_bounds]
        alpha_box = tuple(values) if len(values) == 4 else None
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


def _page_safe_containment(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("raster_alpha_bounds")
        and value.get("inside_page_bounds")
    )


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


def _drawing_primitive_to_dict(
    value: DrawingPrimitive | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, DrawingPrimitive):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {"primitive_id": "", "kind": str(value)}


def _glyph_to_dict(value: GlyphPlacement | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, GlyphPlacement):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"text": str(value)}


def _xywh_int(value: Any) -> list[int]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return []
    output: list[int] = []
    for item in list(value)[:4]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        output.append(int(round(number)))
    if len(output) != 4 or output[2] <= 0 or output[3] <= 0:
        return []
    return output


def _xywh_float(value: Any) -> list[float]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return []
    output: list[float] = []
    for item in list(value)[:4]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        output.append(number)
    if len(output) != 4 or output[2] <= 0.0 or output[3] <= 0.0:
        return []
    return output


def _strict_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _strict_int(value: Any, *, default: int) -> int:
    number = _strict_float(value, default=float(default))
    return int(round(number))


def _point_sequence(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    output: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
            return []
        values = list(item)
        if len(values) < 2:
            return []
        try:
            point = (float(values[0]), float(values[1]))
        except (TypeError, ValueError):
            return []
        if not all(math.isfinite(number) for number in point):
            return []
        output.append(point)
    return output


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


def _resolved_v3_paint(
    style: Mapping[str, Any] | None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    """Resolve the frozen nested v3 fill/outline contract for raster only."""

    values = dict(style or {})
    fill_record = values.get("fill")
    outline_record = values.get("outline")
    fill_value = (
        fill_record.get("color")
        if isinstance(fill_record, Mapping)
        else "#000000"
    )
    outline_value = (
        outline_record.get("color")
        if isinstance(outline_record, Mapping)
        else "#FFFFFF"
    )
    outline_width = 0
    if isinstance(outline_record, Mapping) and outline_record.get("present") is True:
        outline_width = max(
            0,
            _safe_int(outline_record.get("target_width_px"), default=0),
        )
    return (
        _parse_color(fill_value, default=(0, 0, 0, 255)),
        _parse_color(outline_value, default=(255, 255, 255, 255)),
        outline_width,
    )


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


def _unique_strings(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return output
