# -*- coding: utf-8 -*-
"""Canonical parent-bundle renderer facade."""
from __future__ import annotations

import os
import re
from typing import Any, Mapping

from app.pipeline.debug_artifacts import mark_render_region
from app.render.render_execution import (
    PageRenderExecutor,
    PageRenderResult,
    PageRenderTransactionError,
)
from app.render.render_layer_adapter import (
    build_render_layer_plans_from_parent_bundles,
    render_layer_plan_audit,
)


def _renderer_perf_mark_region(
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
    region_id: str,
    **fields,
) -> None:
    mark_render_region(debug_context, region_id, **fields)
    if perf_telemetry_context is not None and perf_telemetry_context is not debug_context:
        mark_render_region(perf_telemetry_context, region_id, **fields)


def render_parent_execution_bundles(
    image_path: str,
    output_path: str,
    parent_execution_bundles: list[Any],
    font_name: str,
    inpaint_mode: str = "fast",
    use_gpu: bool = True,
    model_id: str = "iopaint/anime-manga-big-lama",
    debug_context: dict | None = None,
    source_glyph_masks: object | None = None,
    render_eligibility: object | None = None,
    perf_telemetry_context: dict | None = None,
    cleaned_page_base: Mapping[str, Any] | None = None,
) -> PageRenderResult:
    """Render finalized parent bundles through the sole canonical path.

    The legacy compatibility parameters remain in the public signature for the
    existing controller and GUI callers. Typography comes exclusively from
    each bundle's resolved style, and cleanup has already produced
    ``image_path``/``cleaned_page_base`` before this boundary.
    """

    _stamp_parent_bundle_renderer_audit_ids(
        parent_execution_bundles,
        debug_context=debug_context,
        perf_telemetry_context=perf_telemetry_context,
    )
    render_eligibility_by_region = _render_eligibility_by_region(render_eligibility)
    compositor_bundles = _renderer_stage5_renderable_parent_bundles(
        parent_execution_bundles,
        render_eligibility_by_region=render_eligibility_by_region,
        debug_context=debug_context,
        perf_telemetry_context=perf_telemetry_context,
    )
    page_id = _renderer_stage5_page_id(
        compositor_bundles,
        parent_execution_bundles,
        debug_context,
        image_path,
    )
    cleaned_page_base_ref = _renderer_stage5_cleaned_page_base_ref(
        image_path,
        debug_context,
        cleaned_page_base=cleaned_page_base,
    )
    plans = build_render_layer_plans_from_parent_bundles(
        page_id=page_id,
        parent_execution_bundles=compositor_bundles,
        cleaned_page_base=cleaned_page_base_ref,
    )
    compositor_result = PageRenderExecutor().compose(
        image_path,
        output_path,
        plans,
    )
    audit = compositor_result.to_audit_dict()
    _stamp_parent_bundle_render_layout_summaries(compositor_bundles, audit)
    if debug_context is not None:
        debug_context["render_layer_audit"] = render_layer_plan_audit(plans)
        debug_context["renderer_compositor"] = audit
        debug_context["stage5_renderer_compositor_active"] = True
        debug_context["legacy_render_translations_bypassed_for_parent_bundles"] = True
    for layer in audit.get("layers", []) or []:
        if not isinstance(layer, dict):
            continue
        _renderer_perf_mark_region(
            debug_context,
            perf_telemetry_context,
            str(layer.get("bundle_id") or layer.get("parent_id") or ""),
            renderer_compositor_version=layer.get("renderer_compositor_version"),
            renderer_input_authority="parent_execution_bundle",
            render_layer_id=layer.get("layer_id"),
            text_block_root_id=layer.get("root_id"),
            parent_logical_text_unit_id=layer.get("parent_id"),
            parent_execution_bundle_id=layer.get("bundle_id"),
            drawing_authority=layer.get("drawing_authority"),
            legacy_region_rendering_used=False,
            renderer_cleanup_mutation_applied=False,
            cleanup_applied=False,
            typeset_fit_status=layer.get("fit_status"),
            typeset_full_text_placed=layer.get("full_text_placed"),
            selected_font_face=layer.get("selected_font_face"),
            selected_font_size=layer.get("selected_font_size"),
            rendered_glyph_count=layer.get("drawn_glyph_count"),
            glyph_text_matches_layout=layer.get("glyph_text_matches_layout"),
        )
    if compositor_result.status != "completed" or not compositor_result.output_committed:
        raise PageRenderTransactionError(compositor_result)
    return compositor_result


def _renderer_stage5_renderable_parent_bundles(
    parent_execution_bundles: list[Any],
    *,
    render_eligibility_by_region: dict[str, object],
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
) -> list[Any]:
    output: list[Any] = []
    for bundle in parent_execution_bundles or []:
        bundle_id = str(
            getattr(bundle, "bundle_id", "") or getattr(bundle, "parent_id", "") or ""
        )
        parent_id = str(getattr(bundle, "parent_id", "") or "")
        decision = render_eligibility_by_region.get(
            bundle_id
        ) or render_eligibility_by_region.get(parent_id)
        status = _render_eligibility_status(decision)
        if status.startswith("suppressed_"):
            _renderer_perf_mark_region(
                debug_context,
                perf_telemetry_context,
                bundle_id or parent_id,
                renderer_input_authority="parent_execution_bundle",
                render_suppressed_by_upstream_eligibility=True,
                render_eligibility_status=status,
                render_eligibility_reason=_render_eligibility_value(decision, "reason"),
                legacy_region_rendering_used=False,
                renderer_cleanup_mutation_applied=False,
                cleanup_applied=False,
                render_eligibility_diagnostic_only=True,
            )
        output.append(bundle)
    return output


def _renderer_stage5_page_id(
    compositor_bundles: list[Any],
    all_bundles: list[Any],
    debug_context: dict | None,
    image_path: str,
) -> str:
    for source in (compositor_bundles, all_bundles):
        for bundle in source or []:
            page_id = str(getattr(bundle, "page_id", "") or "")
            if page_id:
                return page_id
    if isinstance(debug_context, dict):
        page_id = str(debug_context.get("page_id") or debug_context.get("page") or "")
        if page_id:
            return page_id
    stem = os.path.splitext(os.path.basename(str(image_path or "")))[0]
    return stem or "unknown_page"


def _renderer_stage5_cleaned_page_base_ref(
    image_path: str,
    debug_context: dict | None,
    *,
    cleaned_page_base: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    existing: Mapping[str, Any] | None = None
    if isinstance(cleaned_page_base, Mapping) and cleaned_page_base:
        existing = cleaned_page_base
    source_image_path = (
        str(existing.get("source_image_path") or "") if existing is not None else ""
    )
    if isinstance(debug_context, dict):
        if existing is None:
            debug_existing = debug_context.get("cleaned_page_base") or debug_context.get(
                "cleaned_page_base_ref"
            )
            if isinstance(debug_existing, Mapping):
                existing = debug_existing
        if not source_image_path:
            source_image_path = str(
                debug_context.get("source_image_path")
                or debug_context.get("source_path")
                or debug_context.get("original_image_path")
                or ""
            )
    if isinstance(existing, Mapping):
        ref = dict(existing)
        ref.setdefault("image_path", image_path)
        ref.setdefault("valid", os.path.isfile(str(ref.get("image_path") or "")))
        ref.setdefault("stage", "cleaned_page_base")
        if source_image_path:
            ref.setdefault("source_image_path", source_image_path)
        return ref
    ref = {
        "cleaned_page_base_version": "cleaned_page_base_runtime_ref_v1",
        "image_path": image_path,
        "valid": os.path.isfile(str(image_path or "")),
        "stage": "cleaned_page_base",
        "source": "render_parent_execution_bundles_input",
    }
    if source_image_path:
        ref["source_image_path"] = source_image_path
    return ref


def _stamp_parent_bundle_renderer_audit_ids(
    parent_execution_bundles: list[Any],
    *,
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
) -> None:
    for index, bundle in enumerate(parent_execution_bundles or []):
        bundle_id = str(
            getattr(bundle, "bundle_id", "") or getattr(bundle, "parent_id", "") or ""
        )
        if not bundle_id:
            continue
        page_id = str(getattr(bundle, "page_id", "") or "")
        renderer_audit_id = str(getattr(bundle, "renderer_audit_id", "") or "")
        if not renderer_audit_id:
            renderer_audit_id = (
                f"raudit_{_renderer_safe_id(page_id)}_{_renderer_safe_id(bundle_id)}"
            )
            try:
                setattr(bundle, "renderer_audit_id", renderer_audit_id)
            except Exception:
                pass
        parent_id = str(getattr(bundle, "parent_id", "") or bundle_id)
        root_id = str(getattr(bundle, "root_id", "") or "")
        fields = {
            "renderer_audit_id": renderer_audit_id,
            "renderer_input_authority": "parent_execution_bundle",
            "parent_execution_bundle_id": bundle_id,
            "parent_logical_text_unit_id": parent_id,
            "text_block_root_id": root_id,
            "parent_execution_bundle_render_index": index,
        }
        _renderer_perf_mark_region(
            debug_context,
            perf_telemetry_context,
            bundle_id,
            **fields,
        )


def _stamp_parent_bundle_render_layout_summaries(
    parent_execution_bundles: list[Any],
    compositor_audit: Mapping[str, Any],
) -> None:
    """Persist compact layout and text-conservation evidence on rendered parents."""

    def by_bundle(key: str) -> dict[str, Mapping[str, Any]]:
        records = compositor_audit.get(key) or []
        return {
            str(item.get("bundle_id") or item.get("parent_id") or ""): item
            for item in records
            if isinstance(item, Mapping)
            and str(item.get("bundle_id") or item.get("parent_id") or "")
        }

    plans = by_bundle("render_layer_plans")
    layouts = by_bundle("typeset_layouts")
    layers = by_bundle("layers")
    reports = by_bundle("fit_reports")
    for bundle in parent_execution_bundles or []:
        bundle_id = str(
            getattr(bundle, "bundle_id", "") or getattr(bundle, "parent_id", "") or ""
        )
        layout = layouts.get(bundle_id)
        if not bundle_id or layout is None:
            continue
        plan = plans.get(bundle_id) or {}
        layer = layers.get(bundle_id) or {}
        report = reports.get(bundle_id) or {}
        translated_text = str(
            plan.get("translated_text") or getattr(bundle, "translated_text", "") or ""
        )
        original_text = str(layout.get("original_text") or "")
        normalized_text = str(layout.get("normalized_text") or "")
        glyph_text = "".join(
            str(item.get("text") or "")
            for item in layout.get("glyphs") or []
            if isinstance(item, Mapping)
        )
        line_records = [
            _compact_layout_record(item, "line_index")
            for item in layout.get("lines") or []
        ]
        column_records = [
            _compact_layout_record(item, "column_index")
            for item in layout.get("columns") or []
        ]
        translated_matches_original = translated_text == original_text
        normalized_matches_glyphs = normalized_text == glyph_text
        text_placement_complete = bool(
            layer.get("text_placement_complete")
        ) and bool(report.get("text_placement_complete"))
        hard_bounds_contained = bool(
            layer.get("hard_bounds_contained")
        ) and bool(report.get("hard_bounds_contained"))
        full_text_placed = text_placement_complete
        glyph_text_matches_layout = bool(layer.get("glyph_text_matches_layout"))
        layer_drawn = bool(layer.get("drawn"))
        failed_raster_placement_count = int(
            layer.get("failed_raster_placement_count") or 0
        )
        hard_bound_containment_failure_count = int(
            layer.get("hard_bound_containment_failure_count") or 0
        )
        parent_layer_composition = (
            layer.get("parent_layer_composition")
            if isinstance(layer.get("parent_layer_composition"), Mapping)
            else {}
        )
        parent_layer_composition_status = str(
            parent_layer_composition.get("status") or "missing"
        )
        parent_layer_page_composite_count = int(
            parent_layer_composition.get("page_composite_count") or 0
        )
        parent_layer_effect_requested = bool(
            parent_layer_composition.get("effect_requested")
        )
        parent_layer_effects_status = str(
            parent_layer_composition.get("effects_status") or "unavailable"
        )
        parent_layer_effect_contract = (
            parent_layer_composition.get("parent_layer_effects")
            if isinstance(parent_layer_composition.get("parent_layer_effects"), Mapping)
            else {}
        )
        parent_layer_effect_active = bool(parent_layer_effect_contract.get("active"))
        parent_layer_effect_contract_status = str(
            parent_layer_effect_contract.get("status") or "unavailable"
        )
        parent_layer_final_alpha_containment = (
            parent_layer_composition.get("final_alpha_containment")
            if isinstance(parent_layer_composition.get("final_alpha_containment"), Mapping)
            else {}
        )
        parent_layer_final_alpha_contained = bool(
            parent_layer_final_alpha_containment.get("accepted")
        )
        parent_layer_final_alpha_page_safe = bool(
            parent_layer_final_alpha_containment.get(
                "inside_page_bounds",
                parent_layer_final_alpha_contained,
            )
        )
        parent_layer_untransformed_fallback_used = bool(
            parent_layer_composition.get("untransformed_fallback_used")
        )
        parent_layer_effect_degraded = bool(
            parent_layer_effects_status == "degraded_to_base"
            and parent_layer_untransformed_fallback_used
            and parent_layer_final_alpha_page_safe
        )
        if parent_layer_effect_requested:
            parent_layer_effect_commit_complete = bool(
                parent_layer_final_alpha_page_safe
                and (
                    parent_layer_effect_degraded
                    or (
                        parent_layer_effect_contract_status == "resolved"
                        and not parent_layer_untransformed_fallback_used
                        and (
                            (
                                parent_layer_effect_active
                                and parent_layer_effects_status == "applied"
                            )
                            or (
                                not parent_layer_effect_active
                                and parent_layer_effects_status == "no_visible_effect"
                            )
                        )
                    )
                )
            )
        else:
            parent_layer_effect_commit_complete = bool(
                parent_layer_final_alpha_page_safe
                and parent_layer_effect_contract_status == "unavailable"
                and parent_layer_effects_status == "unavailable"
                and not parent_layer_effect_active
                and not parent_layer_untransformed_fallback_used
            )
        render_commit_complete = (
            layer_drawn
            and failed_raster_placement_count == 0
            and parent_layer_composition_status == "committed"
            and parent_layer_page_composite_count == 1
            and parent_layer_final_alpha_page_safe
            and parent_layer_effect_commit_complete
        )
        conservation_complete = (
            translated_matches_original
            and normalized_matches_glyphs
            and full_text_placed
            and glyph_text_matches_layout
            and render_commit_complete
        )
        summary = {
            "parent_render_layout_summary_version": "parent_render_layout_summary_v4",
            "renderer_audit_id": str(getattr(bundle, "renderer_audit_id", "") or ""),
            "page_id": str(
                layout.get("page_id") or getattr(bundle, "page_id", "") or ""
            ),
            "layer_id": str(layout.get("layer_id") or ""),
            "bundle_id": bundle_id,
            "parent_id": str(
                layout.get("parent_id") or getattr(bundle, "parent_id", "") or ""
            ),
            "root_id": str(
                layout.get("root_id") or getattr(bundle, "root_id", "") or ""
            ),
            "translated_text": translated_text,
            "layout_original_text": original_text,
            "layout_normalized_text": normalized_text,
            "layout_glyph_text": glyph_text,
            "wrapped_lines": [str(item.get("text") or "") for item in line_records],
            "wrapped_columns": [
                str(item.get("text") or "") for item in column_records
            ],
            "line_records": line_records,
            "column_records": column_records,
            "translated_text_matches_layout_original": translated_matches_original,
            "normalized_text_matches_layout_glyphs": normalized_matches_glyphs,
            "full_text_placed": full_text_placed,
            "text_placement_complete": text_placement_complete,
            "hard_bounds_contained": hard_bounds_contained,
            "fit_quality": str(report.get("fit_quality") or ""),
            "glyph_text_matches_layout": glyph_text_matches_layout,
            "layer_drawn": layer_drawn,
            "failed_raster_placement_count": failed_raster_placement_count,
            "hard_bound_containment_failure_count": (
                hard_bound_containment_failure_count
            ),
            "parent_layer_composition_status": parent_layer_composition_status,
            "parent_layer_page_composite_count": parent_layer_page_composite_count,
            "parent_layer_effect_requested": parent_layer_effect_requested,
            "parent_layer_effect_active": parent_layer_effect_active,
            "parent_layer_effects_status": parent_layer_effects_status,
            "parent_layer_final_alpha_contained": parent_layer_final_alpha_contained,
            "parent_layer_final_alpha_page_safe": (
                parent_layer_final_alpha_page_safe
            ),
            "parent_layer_untransformed_fallback_used": (
                parent_layer_untransformed_fallback_used
            ),
            "parent_layer_effect_degraded": parent_layer_effect_degraded,
            "parent_layer_effect_commit_complete": parent_layer_effect_commit_complete,
            "render_commit_complete": render_commit_complete,
            "conservation_status": "complete" if conservation_complete else "failed",
            "selected_font_face": str(layout.get("selected_font_face") or ""),
            "selected_font_size": float(layout.get("selected_font_size") or 0.0),
            "writing_mode": str(layout.get("writing_mode") or ""),
            "fit_status": str(
                layout.get("fit_status") or report.get("fit_status") or ""
            ),
            "fallback_used": bool(report.get("fallback_used")),
            "fallback_reason": str(report.get("fallback_reason") or ""),
            "overflow_risk": bool(report.get("overflow_risk")),
            "clipping_risk": bool(report.get("clipping_risk")),
            "punctuation_normalization_applied": list(
                report.get("punctuation_normalization_applied") or []
            ),
        }
        try:
            setattr(bundle, "render_layout_summary", summary)
        except Exception:
            continue


def _compact_layout_record(item: Any, index_key: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    record = {
        index_key: int(item.get(index_key) or 0),
        "text": str(item.get("text") or ""),
        "writing_mode": str(item.get("writing_mode") or ""),
    }
    for key in (
        "x",
        "y",
        "width",
        "height",
        "measured_advance",
        "row_units",
        "item_count",
    ):
        if item.get(key) is not None:
            record[key] = item.get(key)
    if item.get("run_ids") is not None:
        record["run_ids"] = list(item.get("run_ids") or [])
    return record


def _renderer_safe_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def _render_eligibility_by_region(
    render_eligibility: object | None,
) -> dict[str, object]:
    if render_eligibility is None:
        return {}
    decisions = getattr(render_eligibility, "decisions_by_region_id", None)
    if isinstance(decisions, dict):
        return {str(key): value for key, value in decisions.items()}
    if isinstance(render_eligibility, dict):
        raw = render_eligibility.get("decisions_by_region_id")
        if isinstance(raw, dict):
            return {str(key): value for key, value in raw.items()}
        raw_list = render_eligibility.get("decisions")
        if isinstance(raw_list, list):
            output: dict[str, object] = {}
            for item in raw_list:
                rid = str(_render_eligibility_value(item, "region_id") or "")
                if rid:
                    output[rid] = item
            return output
    return {}


def _render_eligibility_status(decision: object | None) -> str:
    return str(_render_eligibility_value(decision, "status") or "")


def _render_eligibility_value(decision: object | None, key: str):
    if decision is None:
        return None
    if isinstance(decision, dict):
        return decision.get(key)
    value = getattr(decision, key, None)
    return getattr(value, "value", value)
