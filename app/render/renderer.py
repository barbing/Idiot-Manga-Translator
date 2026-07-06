# -*- coding: utf-8 -*-
"""Simple renderer for translated text."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
import re
import time
from typing import Any, Dict, List, Tuple

from app.pipeline.debug_artifacts import (
    add_count,
    add_timing,
    debug_artifact_level,
    debug_stage_artifact_dir,
    mark_render_region,
    mask_stats,
    write_debug_stage_image_path,
    write_pre_render_source_erasure_debug_image as _debug_write_pre_render_source_erasure_debug_image,
)
from app.pipeline import cleanup_execution
from app.pipeline import source_glyph_masks as source_glyph_mask_stage
from app.render.compositor import RendererCompositor
from app.render.render_layer_adapter import build_render_layer_plans_from_parent_bundles, render_layer_plan_audit

_TOP_ROW_CAPTION_REASONS = {
    "top_row_background_caption_candidate",
    "top_row_caption_fragment_candidate",
}
_SOURCE_ERASURE_CONTRACT_COVERAGE_THRESHOLD = 0.90
_SOURCE_ERASURE_CONTRACT_RESIDUAL_THRESHOLD = 0.12
_PHASE2E_TARGET_ROOT_IDS: dict[str, str] = {}
_PHASE2E_TARGET_REGION_IDS: dict[tuple[str, str], str] = {}
_RENDERER_REGION_LOOP_TELEMETRY_FLAG = "MT_RENDERER_REGION_LOOP_TELEMETRY"
_RENDERER_LAYOUT_TELEMETRY_FLAG = "MT_RENDERER_LAYOUT_TELEMETRY"
_RENDERER_CANDIDATE_SHADOW_FLAG = "MT_RENDERER_CANDIDATE_SHADOW"
_RENDERER_FAST_LAYOUT_FLAG = "MT_RENDERER_FAST_LAYOUT"
_RENDERER_FAST_LAYOUT_SHADOW_FLAG = "MT_RENDERER_FAST_LAYOUT_SHADOW"
_RENDERER_CLEANUP_MUTATION_ENABLED = False
_RENDERER_FAST_LAYOUT_FINALIST_CAP = 48
_RENDERER_LAYOUT_HOT_HELPER_TIMING_KEYS = {
    "renderer_text_wrap_measure_time",
}
_RENDERER_LAYOUT_HOT_HELPER_COUNT_CATEGORIES = {
    "renderer_text_wrap_measure",
}
_RENDERER_MICRO_CONTEXT: ContextVar[dict | None] = ContextVar(
    "_RENDERER_MICRO_CONTEXT",
    default=None,
)
_RENDERER_MICRO_SEEN_KEYS: ContextVar[dict[str, set] | None] = ContextVar(
    "_RENDERER_MICRO_SEEN_KEYS",
    default=None,
)
_RENDERER_FONT_LOAD_CACHE: ContextVar[dict[tuple[str, int, str], object] | None] = ContextVar(
    "_RENDERER_FONT_LOAD_CACHE",
    default=None,
)
_RENDERER_VERTICAL_FONT_FIT_CACHE: ContextVar[
    dict[tuple[object, ...], tuple[object, object]] | None
] = ContextVar(
    "_RENDERER_VERTICAL_FONT_FIT_CACHE",
    default=None,
)
_RENDERER_CURRENT_REGION_ID: ContextVar[str] = ContextVar(
    "_RENDERER_CURRENT_REGION_ID",
    default="",
)


def _renderer_env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _renderer_region_loop_micro_enabled() -> bool:
    return _renderer_env_flag_enabled(_RENDERER_REGION_LOOP_TELEMETRY_FLAG)


def _renderer_layout_micro_enabled() -> bool:
    return _renderer_env_flag_enabled(_RENDERER_LAYOUT_TELEMETRY_FLAG)


def _renderer_candidate_shadow_enabled() -> bool:
    return _renderer_env_flag_enabled(_RENDERER_CANDIDATE_SHADOW_FLAG)


def _renderer_fast_layout_shadow_enabled() -> bool:
    return _renderer_env_flag_enabled(_RENDERER_FAST_LAYOUT_SHADOW_FLAG)


def _renderer_fast_layout_enabled() -> bool:
    raw = os.environ.get(_RENDERER_FAST_LAYOUT_FLAG)
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return False
    # Shadow diagnostics intentionally keep exhaustive selection as the actual
    # output so the oracle remains meaningful.
    if _renderer_candidate_shadow_enabled() or _renderer_fast_layout_shadow_enabled():
        return False
    return True


def _renderer_cleanup_mutation_enabled() -> bool:
    return _RENDERER_CLEANUP_MUTATION_ENABLED


def _renderer_cleanup_disabled_fields() -> dict[str, object]:
    return {
        "renderer_cleanup_applied": False,
        "renderer_cleanup_mutation_disabled": True,
        "renderer_cleanup_extraction_status": "disabled_phase5_phase6_boundary",
        "renderer_cleanup_disabled_reason": "phase5_cleanup_owns_pixel_mutation",
    }


def _renderer_micro_env_enabled() -> bool:
    return (
        _renderer_region_loop_micro_enabled()
        or _renderer_layout_micro_enabled()
        or _renderer_candidate_shadow_enabled()
        or _renderer_fast_layout_shadow_enabled()
    )


def _renderer_micro_begin(debug_context: dict | None, perf_telemetry_context: dict | None = None):
    target_context = debug_context if isinstance(debug_context, dict) else None
    if target_context is None and isinstance(perf_telemetry_context, dict):
        target_context = perf_telemetry_context
    enabled = isinstance(target_context, dict) and _renderer_micro_env_enabled()
    token = _RENDERER_MICRO_CONTEXT.set(target_context if enabled else None)
    _RENDERER_MICRO_SEEN_KEYS.set({} if enabled else None)
    if enabled and target_context is not None:
        if _renderer_region_loop_micro_enabled():
            target_context["renderer_region_loop_micro_telemetry_enabled"] = True
            target_context["renderer_region_loop_micro_telemetry_flag"] = _RENDERER_REGION_LOOP_TELEMETRY_FLAG
        if _renderer_layout_micro_enabled():
            target_context["renderer_layout_micro_telemetry_enabled"] = True
            target_context["renderer_layout_micro_telemetry_flag"] = _RENDERER_LAYOUT_TELEMETRY_FLAG
            target_context["renderer_layout_micro_telemetry_hot_helper_policy"] = (
                "skip_wrap_measure_inner_timing_unless_region_loop_telemetry_enabled"
            )
        if _renderer_candidate_shadow_enabled():
            target_context["renderer_candidate_shadow_enabled"] = True
            target_context["renderer_candidate_shadow_flag"] = _RENDERER_CANDIDATE_SHADOW_FLAG
            target_context["renderer_candidate_shadow_policy"] = (
                "shadow_only_no_candidate_selection_or_render_behavior_change"
            )
        if _renderer_fast_layout_shadow_enabled():
            target_context["renderer_fast_layout_shadow_enabled"] = True
            target_context["renderer_fast_layout_shadow_flag"] = _RENDERER_FAST_LAYOUT_SHADOW_FLAG
            target_context["renderer_fast_layout_shadow_policy"] = (
                "shadow_only_exhaustive_v5_remains_actual_output"
            )
    return token


def _renderer_micro_reset(token) -> None:
    try:
        _RENDERER_MICRO_CONTEXT.reset(token)
    except Exception:
        _RENDERER_MICRO_CONTEXT.set(None)
    _RENDERER_MICRO_SEEN_KEYS.set(None)


def _renderer_font_cache_begin():
    return (
        _RENDERER_FONT_LOAD_CACHE.set({}),
        _RENDERER_VERTICAL_FONT_FIT_CACHE.set({}),
    )


def _renderer_font_cache_reset(token) -> None:
    font_token = token
    fit_token = None
    if isinstance(token, tuple) and len(token) == 2:
        font_token, fit_token = token
    try:
        _RENDERER_FONT_LOAD_CACHE.reset(font_token)
    except Exception:
        _RENDERER_FONT_LOAD_CACHE.set(None)
    try:
        if fit_token is not None:
            _RENDERER_VERTICAL_FONT_FIT_CACHE.reset(fit_token)
        else:
            _RENDERER_VERTICAL_FONT_FIT_CACHE.set(None)
    except Exception:
        _RENDERER_VERTICAL_FONT_FIT_CACHE.set(None)


@contextmanager
def _renderer_region_scope(region_id: str):
    token = _RENDERER_CURRENT_REGION_ID.set(str(region_id or ""))
    try:
        yield
    finally:
        try:
            _RENDERER_CURRENT_REGION_ID.reset(token)
        except Exception:
            _RENDERER_CURRENT_REGION_ID.set("")


@contextmanager
def _renderer_context_cleanup(micro_token, font_cache_token):
    try:
        yield
    finally:
        _renderer_font_cache_reset(font_cache_token)
        _renderer_micro_reset(micro_token)


def _renderer_perf_add_timing(
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
    key: str,
    seconds: float,
) -> None:
    add_timing(debug_context, key, seconds)
    if perf_telemetry_context is not None and perf_telemetry_context is not debug_context:
        add_timing(perf_telemetry_context, key, seconds)


def _renderer_perf_mark_region(
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
    region_id: str,
    **fields,
) -> None:
    mark_render_region(debug_context, region_id, **fields)
    if perf_telemetry_context is not None and perf_telemetry_context is not debug_context:
        mark_render_region(perf_telemetry_context, region_id, **fields)


def _renderer_micro_start() -> float:
    return time.perf_counter() if _RENDERER_MICRO_CONTEXT.get() is not None else 0.0


def _renderer_micro_add(key: str, start: float) -> None:
    context = _RENDERER_MICRO_CONTEXT.get()
    if context is not None and start > 0:
        if (
            _renderer_layout_micro_enabled()
            and not _renderer_region_loop_micro_enabled()
            and key in _RENDERER_LAYOUT_HOT_HELPER_TIMING_KEYS
        ):
            return
        add_timing(context, key, time.perf_counter() - start)


def _renderer_micro_count_key(category: str, key: object) -> None:
    context = _RENDERER_MICRO_CONTEXT.get()
    seen_by_category = _RENDERER_MICRO_SEEN_KEYS.get()
    if context is None or seen_by_category is None:
        return
    safe_category = re.sub(r"[^a-zA-Z0-9_]+", "_", str(category or "")).strip("_")
    if not safe_category:
        return
    if (
        _renderer_layout_micro_enabled()
        and not _renderer_region_loop_micro_enabled()
        and safe_category in _RENDERER_LAYOUT_HOT_HELPER_COUNT_CATEGORIES
    ):
        return
    seen = seen_by_category.setdefault(safe_category, set())
    add_count(context, f"{safe_category}_calls")
    try:
        stable_key = repr(key)
    except Exception:
        stable_key = str(id(key))
    if stable_key in seen:
        add_count(context, f"{safe_category}_duplicate_input_calls")
    else:
        seen.add(stable_key)
        add_count(context, f"{safe_category}_unique_input_calls")


def _renderer_shadow_context() -> dict | None:
    if not _renderer_candidate_shadow_enabled():
        return None
    return _RENDERER_MICRO_CONTEXT.get()


def _renderer_fast_layout_shadow_context() -> dict | None:
    if not _renderer_fast_layout_shadow_enabled():
        return None
    return _RENDERER_MICRO_CONTEXT.get()


def _renderer_shadow_box(box) -> tuple[int, int, int, int] | None:
    try:
        if not box or len(box) < 4:
            return None
        return tuple(int(round(float(v))) for v in box[:4])
    except Exception:
        return None


def _renderer_shadow_box_fp(box) -> str:
    parsed = _renderer_shadow_box(box)
    if not parsed:
        return ""
    return ",".join(str(int(v)) for v in parsed)


def _renderer_shadow_rounded_box_fp(box, step: int = 4) -> str:
    parsed = _renderer_shadow_box(box)
    if not parsed:
        return ""
    safe_step = max(1, int(step or 1))
    rounded = [int(round(v / safe_step) * safe_step) for v in parsed]
    return ",".join(str(v) for v in rounded)


def _renderer_shadow_source_columns_fp(source_columns: dict[str, object] | None) -> str:
    if not isinstance(source_columns, dict):
        return "none"
    return "|".join(
        [
            f"count={int(source_columns.get('count') or 0)}",
            f"vertical={1 if bool(source_columns.get('vertical')) else 0}",
            f"union={_renderer_shadow_box_fp(source_columns.get('union'))}",
            f"shape={_renderer_shadow_box_fp(source_columns.get('shape_box'))}",
            f"source={str(source_columns.get('source') or '')}",
        ]
    )


def _renderer_shadow_candidate_fp(candidate: dict[str, object]) -> str:
    return "|".join(
        [
            str(candidate.get("bbox_fp") or ""),
            f"compact={1 if bool(candidate.get('compact')) else 0}",
            f"lh={float(candidate.get('line_height') or 0.0):.3f}",
            f"orient={candidate.get('orientation') or ''}",
            f"cols={candidate.get('source_columns_fp') or ''}",
        ]
    )


def _renderer_shadow_box_area(box) -> int:
    parsed = _renderer_shadow_box(box)
    if not parsed:
        return 0
    return max(0, parsed[2] - parsed[0]) * max(0, parsed[3] - parsed[1])


def _renderer_shadow_contains(outer, inner) -> bool:
    a = _renderer_shadow_box(outer)
    b = _renderer_shadow_box(inner)
    if not a or not b:
        return False
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _renderer_shadow_build_candidates(
    boxes,
    line_heights,
    *,
    source_orientation: str,
    source_columns: dict[str, object] | None,
) -> list[dict[str, object]]:
    source_fp = _renderer_shadow_source_columns_fp(source_columns)
    candidates: list[dict[str, object]] = []
    index = 0
    for box in boxes or []:
        box_tuple = _renderer_shadow_box(box)
        if not box_tuple:
            continue
        for compact in (False, True):
            for lh_index, lh in enumerate(line_heights or []):
                candidate = {
                    "index": index,
                    "bbox": box_tuple,
                    "bbox_fp": _renderer_shadow_box_fp(box_tuple),
                    "rounded_bbox_fp": _renderer_shadow_rounded_box_fp(box_tuple),
                    "compact": bool(compact),
                    "line_height": round(float(lh or 0.0), 3),
                    "line_height_index": int(lh_index),
                    "orientation": str(source_orientation or ""),
                    "source_columns_fp": source_fp,
                    "area": _renderer_shadow_box_area(box_tuple),
                }
                candidate["fingerprint"] = _renderer_shadow_candidate_fp(candidate)
                candidates.append(candidate)
                index += 1
    return candidates


def _renderer_shadow_keep_exact(candidates: list[dict[str, object]]) -> set[int]:
    seen: set[str] = set()
    kept: set[int] = set()
    for candidate in candidates:
        key = str(candidate.get("fingerprint") or "")
        if key in seen:
            continue
        seen.add(key)
        kept.add(int(candidate.get("index") or 0))
    return kept


def _renderer_shadow_keep_rounded(candidates: list[dict[str, object]]) -> set[int]:
    seen: set[tuple[object, ...]] = set()
    kept: set[int] = set()
    for candidate in candidates:
        key = (
            candidate.get("rounded_bbox_fp"),
            bool(candidate.get("compact")),
            float(candidate.get("line_height") or 0.0),
            candidate.get("orientation"),
            candidate.get("source_columns_fp"),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.add(int(candidate.get("index") or 0))
    return kept


def _renderer_shadow_keep_dominated(candidates: list[dict[str, object]]) -> set[int]:
    kept_candidates: list[dict[str, object]] = []
    kept: set[int] = set()
    for candidate in candidates:
        dominated = False
        for existing in kept_candidates:
            same_mode = (
                bool(existing.get("compact")) == bool(candidate.get("compact"))
                and float(existing.get("line_height") or 0.0) == float(candidate.get("line_height") or 0.0)
                and str(existing.get("orientation") or "") == str(candidate.get("orientation") or "")
                and str(existing.get("source_columns_fp") or "") == str(candidate.get("source_columns_fp") or "")
            )
            if same_mode and _renderer_shadow_contains(existing.get("bbox"), candidate.get("bbox")):
                dominated = True
                break
        if dominated:
            continue
        kept_candidates.append(candidate)
        kept.add(int(candidate.get("index") or 0))
    return kept


def _renderer_shadow_keep_top_k(candidates: list[dict[str, object]], k: int = 64) -> set[int]:
    if len(candidates) <= k:
        return {int(candidate.get("index") or 0) for candidate in candidates}
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            1 if bool(candidate.get("compact")) else 0,
            abs(int(candidate.get("line_height_index") or 0)),
            -int(candidate.get("area") or 0),
            int(candidate.get("index") or 0),
        ),
    )
    return {int(candidate.get("index") or 0) for candidate in ranked[:k]}


def _renderer_shadow_keep_line_height_compact(candidates: list[dict[str, object]]) -> set[int]:
    compact_slots = {0, 3, 6, 9}
    kept: set[int] = set()
    for candidate in candidates:
        if not bool(candidate.get("compact")) or int(candidate.get("line_height_index") or 0) in compact_slots:
            kept.add(int(candidate.get("index") or 0))
    return kept


def _renderer_shadow_summarize_rules(
    candidates: list[dict[str, object]],
    selected_fingerprint: str,
) -> list[dict[str, object]]:
    before = len(candidates)
    selected_indices = {
        int(candidate.get("index") or 0)
        for candidate in candidates
        if str(candidate.get("fingerprint") or "") == selected_fingerprint
    }
    rules = [
        ("exact_candidate_fingerprint_dedupe", _renderer_shadow_keep_exact(candidates)),
        ("rounded_bbox_dedupe_same_mode", _renderer_shadow_keep_rounded(candidates)),
        ("dominated_bbox_removal_same_mode", _renderer_shadow_keep_dominated(candidates)),
        ("cheap_prefilter_top_k_before_v5", _renderer_shadow_keep_top_k(candidates)),
        ("line_height_compact_mode_prune", _renderer_shadow_keep_line_height_compact(candidates)),
    ]
    summaries: list[dict[str, object]] = []
    for name, kept in rules:
        after = len(kept)
        retained = bool(selected_indices and selected_indices.intersection(kept))
        if not selected_indices:
            reason = "selected_candidate_not_found_in_shadow_candidates"
        elif retained:
            reason = ""
        else:
            reason = "selected_candidate_pruned_by_shadow_rule"
        summaries.append(
            {
                "rule": name,
                "candidate_count_before": before,
                "candidate_count_after": after,
                "estimated_v5_scoring_calls_avoided": max(0, before - after),
                "estimated_vertical_font_fit_calls_avoided": max(0, before - after),
                "selected_candidate_retained": retained,
                "selected_candidate_fingerprint": selected_fingerprint,
                "failure_reason": reason,
            }
        )
    return summaries


def _renderer_candidate_shadow_record(
    *,
    text: str,
    source_orientation: str,
    candidates: list[dict[str, object]],
    selected: dict[str, object],
    selected_rank: int,
    candidate_loop_time: float,
) -> None:
    context = _renderer_shadow_context()
    if context is None:
        return
    selected_fingerprint = str(selected.get("fingerprint") or "")
    rules = _renderer_shadow_summarize_rules(candidates, selected_fingerprint)
    page_id = str(context.get("page_id") or "")
    source_path = str(context.get("source_path") or "")
    record = {
        "page_id": page_id,
        "image_basename": os.path.basename(source_path) if source_path else "",
        "region_id": _RENDERER_CURRENT_REGION_ID.get(),
        "translated_text_length": len(str(text or "")),
        "orientation": str(source_orientation or ""),
        "candidate_family": "render_readability_v5",
        "candidate_source_label": "v5_boxes_x_compact_x_line_height",
        "total_candidate_count_before_v5_scoring": len(candidates),
        "v5_scored_candidate_count": len(candidates),
        "vertical_font_fit_call_count_estimate": len(candidates),
        "selected_candidate_fingerprint": selected_fingerprint,
        "selected_candidate_bbox_fingerprint": selected.get("bbox_fp") or "",
        "selected_candidate_rank": int(selected_rank or 0),
        "selected_candidate_compact_mode": bool(selected.get("compact")),
        "selected_candidate_line_height": selected.get("line_height"),
        "candidate_loop_time": round(float(candidate_loop_time or 0.0), 6),
        "selected_score_summary": selected.get("score_summary") or {},
        "shadow_rules": rules,
    }
    counts = context.setdefault("counts", {})
    records = counts.setdefault("renderer_candidate_shadow_records", [])
    if isinstance(records, list):
        records.append(record)
    add_count(context, "renderer_candidate_shadow_region_records")
    add_count(context, "renderer_candidate_shadow_v5_scored_candidates", len(candidates))


def _renderer_fast_layout_iou(a, b) -> float:
    box_a = _renderer_shadow_box(a)
    box_b = _renderer_shadow_box(b)
    if not box_a or not box_b:
        return 0.0
    inter = _xyxy_intersection_area(box_a, box_b)
    union = max(1, _xyxy_area(box_a) + _xyxy_area(box_b) - inter)
    return max(0.0, min(1.0, inter / union))


def _renderer_fast_layout_class_rank(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 1
    if "geometry_limit" in text:
        return 4
    if "text_too_dense" in text:
        return 3
    if "unresolved" in text:
        return 2
    return 0


def _renderer_fast_layout_candidate_features(
    candidate: dict[str, object],
    *,
    text: str,
    allowed_area_box,
    shape_box,
    source_columns: dict[str, object],
) -> dict[str, object]:
    box = _renderer_shadow_box(candidate.get("bbox"))
    if not box:
        return {
            "outside_ratio": 1.0,
            "density": 999.0,
            "density_penalty": 999.0,
            "source_alignment": 0.0,
            "edge_contact": 1.0,
            "aspect_penalty": 999.0,
            "cheap_score": 99999.0,
        }
    area = max(1, _xyxy_area(box))
    chars = max(1, len(_meaningful_render_chars(text)))
    density = chars / max(1.0, area / 1000.0)
    outside = _outside_ratio(box, allowed_area_box) if allowed_area_box else 0.0
    source_union = _renderer_shadow_box(source_columns.get("union")) if isinstance(source_columns, dict) else None
    source_alignment = 0.0
    if source_union:
        source_alignment = _xyxy_intersection_area(box, source_union) / max(1, _xyxy_area(source_union))
    limit = _renderer_shadow_box(shape_box) or _renderer_shadow_box(allowed_area_box)
    edge_contact = 0.0
    if limit:
        tolerance = 3
        contacts = 0
        contacts += 1 if abs(box[0] - limit[0]) <= tolerance else 0
        contacts += 1 if abs(box[1] - limit[1]) <= tolerance else 0
        contacts += 1 if abs(box[2] - limit[2]) <= tolerance else 0
        contacts += 1 if abs(box[3] - limit[3]) <= tolerance else 0
        edge_contact = contacts / 4.0
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    vertical = bool(source_columns.get("vertical")) if isinstance(source_columns, dict) else height >= width
    aspect = height / width
    aspect_penalty = abs(aspect - 2.0) if vertical else abs((width / height) - 2.0)
    density_penalty = abs(density - 0.26)
    cheap_score = (
        outside * 400.0
        + density_penalty * 18.0
        + edge_contact * 4.0
        + aspect_penalty * 1.5
        - min(1.0, source_alignment) * 3.0
        + (0.25 if bool(candidate.get("compact")) else 0.0)
    )
    return {
        "outside_ratio": round(float(outside), 6),
        "density": round(float(density), 6),
        "density_penalty": round(float(density_penalty), 6),
        "source_alignment": round(float(source_alignment), 6),
        "edge_contact": round(float(edge_contact), 6),
        "aspect_penalty": round(float(aspect_penalty), 6),
        "cheap_score": round(float(cheap_score), 6),
    }


def _renderer_fast_layout_finalists(
    candidates: list[dict[str, object]],
    *,
    current,
    text: str,
    allowed_area_box,
    shape_box,
    source_columns: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not candidates:
        return [], {
            "retention_reason_counts": {},
            "cap": _RENDERER_FAST_LAYOUT_FINALIST_CAP,
            "candidate_count_before": 0,
            "candidate_count_after": 0,
        }
    current_box = _renderer_shadow_box(current)
    source_union = _renderer_shadow_box(source_columns.get("union")) if isinstance(source_columns, dict) else None
    enriched: list[dict[str, object]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["fast_features"] = _renderer_fast_layout_candidate_features(
            item,
            text=text,
            allowed_area_box=allowed_area_box,
            shape_box=shape_box,
            source_columns=source_columns,
        )
        enriched.append(item)

    selected_indices: set[int] = set()
    reasons: dict[int, set[str]] = {}
    candidates_by_bbox: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    for candidate in enriched:
        box = _renderer_shadow_box(candidate.get("bbox"))
        if box:
            candidates_by_bbox.setdefault(box, []).append(candidate)

    def keep(candidate: dict[str, object], reason: str, *, all_variants_for_box: bool = False) -> None:
        if all_variants_for_box:
            box = _renderer_shadow_box(candidate.get("bbox"))
            if box:
                for variant in candidates_by_bbox.get(box, []):
                    index = int(variant.get("index") or 0)
                    selected_indices.add(index)
                    reasons.setdefault(index, set()).add(reason)
                return
        index = int(candidate.get("index") or 0)
        selected_indices.add(index)
        reasons.setdefault(index, set()).add(reason)

    for candidate in enriched:
        if current_box and _renderer_shadow_box(candidate.get("bbox")) == current_box:
            keep(candidate, "current_box")

    source_guided = [
        candidate
        for candidate in enriched
        if source_union and _xyxy_intersection_area(candidate.get("bbox"), source_union) > 0
    ]
    for candidate in sorted(
        source_guided,
        key=lambda item: (
            -float((item.get("fast_features") or {}).get("source_alignment") or 0.0),
            float((item.get("fast_features") or {}).get("cheap_score") or 99999.0),
            int(item.get("index") or 0),
        ),
    )[:8]:
        keep(candidate, "source_column_guided", all_variants_for_box=True)

    shape_contained = []
    for candidate in enriched:
        if shape_box is None or _outside_ratio(candidate.get("bbox"), shape_box) <= 0.001:
            shape_contained.append(candidate)
    if shape_contained:
        keep(
            max(shape_contained, key=lambda item: (int(item.get("area") or 0), -int(item.get("index") or 0))),
            "largest_shape_contained",
            all_variants_for_box=True,
        )

    cheap_density = sorted(
        enriched,
        key=lambda item: (
            float((item.get("fast_features") or {}).get("density_penalty") or 999.0),
            float((item.get("fast_features") or {}).get("cheap_score") or 99999.0),
            int(item.get("index") or 0),
        ),
    )
    for candidate in cheap_density[:8]:
        keep(candidate, "best_cheap_density", all_variants_for_box=True)

    for compact_value in (False, True):
        compact_candidates = [candidate for candidate in enriched if bool(candidate.get("compact")) is compact_value]
        if compact_candidates:
            keep(
                min(
                    compact_candidates,
                    key=lambda item: (
                        float((item.get("fast_features") or {}).get("cheap_score") or 99999.0),
                        int(item.get("index") or 0),
                    ),
                ),
                "compact_mode_representative" if compact_value else "noncompact_mode_representative",
                all_variants_for_box=True,
            )

    for candidate in sorted(
        enriched,
        key=lambda item: (
            float((item.get("fast_features") or {}).get("cheap_score") or 99999.0),
            int(item.get("index") or 0),
        ),
    ):
        if len(selected_indices) >= _RENDERER_FAST_LAYOUT_FINALIST_CAP:
            break
        keep(candidate, "cheap_score_fill", all_variants_for_box=True)

    finalists = [candidate for candidate in enriched if int(candidate.get("index") or 0) in selected_indices]
    finalists.sort(key=lambda item: int(item.get("index") or 0))
    if len(finalists) > _RENDERER_FAST_LAYOUT_FINALIST_CAP:
        priority_reasons = {
            "current_box": 0,
            "source_column_guided": 1,
            "largest_shape_contained": 2,
            "best_cheap_density": 3,
            "noncompact_mode_representative": 4,
            "compact_mode_representative": 4,
            "cheap_score_fill": 5,
        }

        def priority(candidate: dict[str, object]) -> tuple[int, float, int]:
            index = int(candidate.get("index") or 0)
            reason_set = reasons.get(index) or set()
            reason_rank = min((priority_reasons.get(reason, 99) for reason in reason_set), default=99)
            return (
                reason_rank,
                float((candidate.get("fast_features") or {}).get("cheap_score") or 99999.0),
                index,
            )

        kept = sorted(finalists, key=priority)[:_RENDERER_FAST_LAYOUT_FINALIST_CAP]
        kept_indices = {int(candidate.get("index") or 0) for candidate in kept}
        finalists = [candidate for candidate in finalists if int(candidate.get("index") or 0) in kept_indices]

    reason_counts: dict[str, int] = {}
    for index in {int(candidate.get("index") or 0) for candidate in finalists}:
        for reason in sorted(reasons.get(index) or []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return finalists, {
        "retention_reason_counts": reason_counts,
        "cap": _RENDERER_FAST_LAYOUT_FINALIST_CAP,
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(finalists),
    }


def _renderer_fast_layout_selected_shadow(score: dict[str, object], fallback_text: str) -> dict[str, object]:
    line_height = round(float(score.get("line_height_scale") or 0.0), 3)
    selected = {
        "bbox": _renderer_shadow_box(score.get("bbox")),
        "bbox_fp": _renderer_shadow_box_fp(score.get("bbox")),
        "compact": bool(score.get("compact_layout")),
        "line_height": line_height,
        "orientation": str(score.get("source_orientation") or ""),
        "source_columns_fp": "",
        "area": _renderer_shadow_box_area(score.get("bbox")),
    }
    selected["fingerprint"] = _renderer_shadow_candidate_fp(selected)
    selected["score_summary"] = {
        "score": score.get("score"),
        "fit_ratio": score.get("fit_ratio"),
        "density": score.get("density"),
        "edge_contact": score.get("edge_contact"),
        "outside_ratio": score.get("outside_ratio"),
        "wrapped_count": score.get("wrapped_count"),
        "selected_font_size": score.get("selected_font_size"),
        "source_column_count": score.get("source_column_count"),
        "final_class": score.get("final_class"),
        "text_length": len(str(fallback_text or "")),
    }
    return selected


def _renderer_fast_layout_score_summary(score: dict[str, object]) -> dict[str, object]:
    return {
        "score": score.get("score"),
        "fit_ratio": score.get("fit_ratio"),
        "density": score.get("density"),
        "edge_contact": score.get("edge_contact"),
        "outside_ratio": score.get("outside_ratio"),
        "wrapped_count": score.get("wrapped_count"),
        "selected_font_size": score.get("selected_font_size"),
        "final_class": score.get("final_class"),
    }


def _renderer_fast_layout_risk_class(exhaustive: dict[str, object], fast: dict[str, object]) -> str:
    exhaustive_fp = str(exhaustive.get("fingerprint") or "")
    fast_fp = str(fast.get("fingerprint") or "")
    if exhaustive_fp and exhaustive_fp == fast_fp:
        return "identical_selection"
    ex_summary = exhaustive.get("score_summary") if isinstance(exhaustive.get("score_summary"), dict) else {}
    fast_summary = fast.get("score_summary") if isinstance(fast.get("score_summary"), dict) else {}
    ex_score = float(ex_summary.get("score") or 0.0)
    fast_score = float(fast_summary.get("score") or 0.0)
    ex_fit = float(ex_summary.get("fit_ratio") or 0.0)
    fast_fit = float(fast_summary.get("fit_ratio") or 0.0)
    ex_edge = float(ex_summary.get("edge_contact") or 0.0)
    fast_edge = float(fast_summary.get("edge_contact") or 0.0)
    ex_outside = float(ex_summary.get("outside_ratio") or 0.0)
    fast_outside = float(fast_summary.get("outside_ratio") or 0.0)
    ex_font = int(ex_summary.get("selected_font_size") or 0)
    fast_font = int(fast_summary.get("selected_font_size") or 0)
    ex_wrapped = int(ex_summary.get("wrapped_count") or 0)
    fast_wrapped = int(fast_summary.get("wrapped_count") or 0)
    ex_class = str(ex_summary.get("final_class") or "")
    fast_class = str(fast_summary.get("final_class") or "")
    iou = _renderer_fast_layout_iou(exhaustive.get("bbox"), fast.get("bbox"))
    font_delta = fast_font - ex_font if ex_font and fast_font else 0
    wrapped_delta = fast_wrapped - ex_wrapped
    if (
        fast_fit - ex_fit > 0.08
        or ("text_too_dense" in fast_class and "text_too_dense" not in ex_class)
        or (font_delta < -2 and fast_fit >= ex_fit - 0.02)
        or wrapped_delta > 1
    ):
        return "text_fit_regression"
    if (
        (fast_outside > 0.02 and fast_outside - ex_outside > 0.02)
        or (fast_edge >= 0.75 and fast_edge - ex_edge > 0.15)
        or ("geometry_limit" in fast_class and "geometry_limit" not in ex_class)
    ):
        return "edge_or_overflow_regression"
    near_score_limit = ex_score + max(2.0, 0.15 * abs(ex_score))
    if (
        iou >= 0.85
        and fast_score <= near_score_limit
        and abs(font_delta) <= 2
        and abs(wrapped_delta) <= 1
        and _renderer_fast_layout_class_rank(fast_class) <= _renderer_fast_layout_class_rank(ex_class)
    ):
        return "near_equivalent_selection"
    return "layout_shift_review_needed"


def _choose_render_readability_fast_candidate(
    current,
    expanded,
    allowed,
    shape_box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    before_layout: dict[str, object],
    shape_source: str,
    source_columns: dict[str, object],
    *,
    boxes=None,
    line_heights=None,
    candidates=None,
) -> tuple[dict[str, object], dict[str, object]]:
    boxes = list(boxes) if boxes is not None else _render_readability_v5_boxes(current, expanded, allowed, shape_box, source_columns, source_size_hint)
    if line_heights is None:
        line_heights = [float(line_height_scale)]
        for scale in (1.0, 0.98, 0.96, 0.94, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72):
            candidate_lh = max(0.72, min(1.2, float(line_height_scale) * scale))
            if all(abs(candidate_lh - existing) > 0.01 for existing in line_heights):
                line_heights.append(candidate_lh)
    candidates = list(candidates) if candidates is not None else _renderer_shadow_build_candidates(
        boxes,
        line_heights,
        source_orientation=source_orientation,
        source_columns=source_columns,
    )
    finalists, finalist_meta = _renderer_fast_layout_finalists(
        candidates,
        current=current,
        text=text,
        allowed_area_box=allowed,
        shape_box=shape_box,
        source_columns=source_columns,
    )
    best = dict(before_layout)
    best["bbox"] = tuple(current)
    best["compact_layout"] = False
    best["line_height_scale"] = round(float(line_height_scale), 3)
    best["shape_source"] = shape_source
    start = time.perf_counter()
    token = _RENDERER_MICRO_CONTEXT.set(None)
    try:
        for candidate in finalists:
            box = candidate.get("bbox")
            score = _score_render_readability_v5_candidate(
                box,
                text,
                region_font,
                float(candidate.get("line_height") or line_height_scale),
                wrap_mode,
                source_orientation,
                source_size_hint,
                source_size_min,
                source_size_max,
                compact_layout=bool(candidate.get("compact")),
                allowed_area_box=allowed,
                shape_box=shape_box,
                shape_source=shape_source,
                source_columns=source_columns,
            )
            score["bbox"] = tuple(box)
            score["compact_layout"] = bool(candidate.get("compact"))
            score["line_height_scale"] = round(float(candidate.get("line_height") or line_height_scale), 3)
            score["source_orientation"] = source_orientation
            if float(score.get("score") or 9999.0) < float(best.get("score") or 9999.0):
                best = score
    finally:
        try:
            _RENDERER_MICRO_CONTEXT.reset(token)
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    best["candidate_count"] = len(finalists)
    best["fast_layout_candidate_count_before"] = len(candidates)
    best["fast_layout_finalist_count"] = len(finalists)
    best["fast_layout_candidates_avoided"] = max(0, len(candidates) - len(finalists))
    meta = dict(finalist_meta)
    meta.update(
        {
            "fast_loop_time": round(float(elapsed), 6),
            "estimated_v5_calls_avoided": max(0, len(candidates) - len(finalists)),
            "estimated_vertical_font_fit_calls_avoided": max(0, len(candidates) - len(finalists)),
        }
    )
    return best, meta


def _renderer_fast_layout_oracle_record(
    *,
    text: str,
    source_orientation: str,
    exhaustive: dict[str, object],
    fast: dict[str, object],
    exhaustive_fingerprint: str,
    fast_fingerprint: str,
    exhaustive_loop_time: float,
    fast_meta: dict[str, object],
) -> None:
    context = _renderer_fast_layout_shadow_context()
    if context is None:
        return
    ex_summary = _renderer_fast_layout_score_summary(exhaustive)
    fast_summary = _renderer_fast_layout_score_summary(fast)
    exhaustive_shadow = {
        "bbox": _renderer_shadow_box(exhaustive.get("bbox")),
        "fingerprint": exhaustive_fingerprint,
        "score_summary": ex_summary,
    }
    fast_shadow = {
        "bbox": _renderer_shadow_box(fast.get("bbox")),
        "fingerprint": fast_fingerprint,
        "score_summary": fast_summary,
    }
    risk = _renderer_fast_layout_risk_class(exhaustive_shadow, fast_shadow)
    ex_font = int(ex_summary.get("selected_font_size") or 0)
    fast_font = int(fast_summary.get("selected_font_size") or 0)
    ex_wrapped = int(ex_summary.get("wrapped_count") or 0)
    fast_wrapped = int(fast_summary.get("wrapped_count") or 0)
    ex_score = float(ex_summary.get("score") or 0.0)
    fast_score = float(fast_summary.get("score") or 0.0)
    source_path = str(context.get("source_path") or "")
    record = {
        "page_id": str(context.get("page_id") or ""),
        "image_basename": os.path.basename(source_path) if source_path else "",
        "region_id": _RENDERER_CURRENT_REGION_ID.get(),
        "translated_text_length": len(str(text or "")),
        "orientation": str(source_orientation or ""),
        "exhaustive_candidate_count": int(exhaustive.get("candidate_count") or 0),
        "fast_finalist_count": int(fast_meta.get("candidate_count_after") or 0),
        "exhaustive_selected_candidate_fingerprint": exhaustive_fingerprint,
        "fast_selected_candidate_fingerprint": fast_fingerprint,
        "exhaustive_score": ex_summary.get("score"),
        "fast_score": fast_summary.get("score"),
        "score_delta": round(float(fast_score - ex_score), 6),
        "bbox_iou": round(float(_renderer_fast_layout_iou(exhaustive.get("bbox"), fast.get("bbox"))), 6),
        "selected_font_size_delta": int(fast_font - ex_font) if ex_font and fast_font else 0,
        "wrapped_count_delta": int(fast_wrapped - ex_wrapped),
        "exhaustive_selected_bbox": list(_renderer_shadow_box(exhaustive.get("bbox")) or ()),
        "fast_selected_bbox": list(_renderer_shadow_box(fast.get("bbox")) or ()),
        "estimated_v5_calls_avoided": int(fast_meta.get("estimated_v5_calls_avoided") or 0),
        "estimated_vertical_font_fit_calls_avoided": int(
            fast_meta.get("estimated_vertical_font_fit_calls_avoided") or 0
        ),
        "exhaustive_loop_time": round(float(exhaustive_loop_time or 0.0), 6),
        "fast_loop_time": round(float(fast_meta.get("fast_loop_time") or 0.0), 6),
        "risk_class": risk,
        "exhaustive_score_summary": ex_summary,
        "fast_score_summary": fast_summary,
        "fast_finalist_retention_reasons": fast_meta.get("retention_reason_counts") or {},
    }
    counts = context.setdefault("counts", {})
    records = counts.setdefault("renderer_fast_layout_oracle_records", [])
    if isinstance(records, list):
        records.append(record)
    add_count(context, "renderer_fast_layout_oracle_region_records")
    add_count(context, "renderer_fast_layout_oracle_candidates_avoided", record["estimated_v5_calls_avoided"])

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageChops = None
    ImageDraw = None
    ImageFont = None
    ImageStat = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None


def render_parent_execution_bundles(
    image_path: str,
    output_path: str,
    parent_execution_bundles: list[Any],
    font_name: str,
    inpaint_mode: str = "fast",
    use_gpu: bool = True,
    model_id: str = cleanup_execution.FIXED_CLEANUP_INPAINT_MODEL_ID,
    debug_context: dict | None = None,
    source_glyph_masks: object | None = None,
    render_eligibility: object | None = None,
    perf_telemetry_context: dict | None = None,
) -> None:
    """Render translated text from finalized parent execution bundles."""

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
    page_id = _renderer_stage5_page_id(compositor_bundles, parent_execution_bundles, debug_context, image_path)
    cleaned_page_base = _renderer_stage5_cleaned_page_base_ref(image_path, debug_context)
    plans = build_render_layer_plans_from_parent_bundles(
        page_id=page_id,
        parent_execution_bundles=compositor_bundles,
        cleaned_page_base=cleaned_page_base,
    )
    compositor_result = RendererCompositor().compose(image_path, output_path, plans)
    audit = compositor_result.to_audit_dict()
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


def _renderer_stage5_renderable_parent_bundles(
    parent_execution_bundles: list[Any],
    *,
    render_eligibility_by_region: dict[str, object],
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
) -> list[Any]:
    output: list[Any] = []
    for bundle in parent_execution_bundles or []:
        bundle_id = str(getattr(bundle, "bundle_id", "") or getattr(bundle, "parent_id", "") or "")
        parent_id = str(getattr(bundle, "parent_id", "") or "")
        decision = render_eligibility_by_region.get(bundle_id) or render_eligibility_by_region.get(parent_id)
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
            )
            continue
        output.append(bundle)
    return output


def _renderer_stage5_page_id(
    compositor_bundles: list[Any],
    all_bundles: list[Any],
    debug_context: dict | None,
    image_path: str,
) -> str:
    for source in (
        compositor_bundles,
        all_bundles,
    ):
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


def _renderer_stage5_cleaned_page_base_ref(image_path: str, debug_context: dict | None) -> dict[str, object]:
    existing = None
    source_image_path = ""
    if isinstance(debug_context, dict):
        existing = debug_context.get("cleaned_page_base") or debug_context.get("cleaned_page_base_ref")
        source_image_path = str(
            debug_context.get("source_image_path")
            or debug_context.get("source_path")
            or debug_context.get("original_image_path")
            or ""
        )
    if isinstance(existing, dict):
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
        bundle_id = str(getattr(bundle, "bundle_id", "") or getattr(bundle, "parent_id", "") or "")
        if not bundle_id:
            continue
        page_id = str(getattr(bundle, "page_id", "") or "")
        renderer_audit_id = str(getattr(bundle, "renderer_audit_id", "") or "")
        if not renderer_audit_id:
            renderer_audit_id = f"raudit_{_renderer_safe_id(page_id)}_{_renderer_safe_id(bundle_id)}"
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
        execution_region = getattr(bundle, "execution_region", None)
        if isinstance(execution_region, dict):
            execution_region.update(fields)
            render = execution_region.setdefault("render", {})
            if isinstance(render, dict):
                render.update(fields)
        _renderer_perf_mark_region(debug_context, perf_telemetry_context, bundle_id, **fields)


def _renderer_safe_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def render_translations(
    image_path: str,
    output_path: str,
    regions: List[Dict[str, object]],
    font_name: str,
    inpaint_mode: str = "fast",
    use_gpu: bool = True,
    model_id: str = cleanup_execution.FIXED_CLEANUP_INPAINT_MODEL_ID,
    debug_context: dict | None = None,
    source_glyph_masks: object | None = None,
    render_eligibility: object | None = None,
    perf_telemetry_context: dict | None = None,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    micro_token = _renderer_micro_begin(debug_context, perf_telemetry_context)
    font_cache_token = _renderer_font_cache_begin()
    with _renderer_context_cleanup(micro_token, font_cache_token), Image.open(image_path) as img:
        img = img.convert("RGB")
        working = img
        img_w, img_h = img.size
        renderer_cleanup_mutation_enabled = _renderer_cleanup_mutation_enabled()
        render_regions: List[Tuple[object, ...]] = []
        background_boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]] = []
        caption_boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]] = []
        local_cleanup_masks: List[Tuple[object, str, str]] = []
        global_text_mask_region_ids: List[str] = []
        global_cleanup_candidates: List[dict[str, object]] = []
        text_mask = None
        bubble_text_mask = None
        bubble_area_mask = None
        other_text_mask = None
        if cv2 is not None and np is not None:
            text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bubble_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bubble_area_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            other_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if debug_context is not None:
            debug_context["renderer_cleanup_extraction"] = {
                "renderer_cleanup_mutation_enabled": renderer_cleanup_mutation_enabled,
                "renderer_cleanup_mutation_disabled": not renderer_cleanup_mutation_enabled,
                "renderer_cleanup_extraction_status": (
                    "enabled"
                    if renderer_cleanup_mutation_enabled
                    else "disabled_phase5_phase6_boundary"
                ),
                "renderer_cleanup_disabled_reason": (
                    None
                    if renderer_cleanup_mutation_enabled
                    else "phase5_cleanup_owns_pixel_mutation"
                ),
            }
        img_np = np.array(img) if cv2 is not None and np is not None else None
        source_np_for_erasure_proof = img_np.copy() if img_np is not None else None
        preserved_text_obstacles = _preserved_text_obstacle_boxes(regions)
        regions_by_id = {
            str(item.get("region_id", "") or ""): item
            for item in regions
            if isinstance(item, dict) and str(item.get("region_id", "") or "")
        }
        model_fusion_proof_candidates = _build_model_fusion_mutation_proof_candidates(
            image_path,
            regions,
            debug_context,
            img_w,
            img_h,
        )
        source_glyph_masks_by_region = _source_glyph_masks_by_region(source_glyph_masks)
        render_eligibility_by_region = _render_eligibility_by_region(render_eligibility)

        mask_generation_start = time.time()
        renderer_region_loop_start = time.time()
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_id = str(region.get("region_id", "") or "")
            source_grounding_decision = render_eligibility_by_region.get(region_id)
            if _render_eligibility_status(source_grounding_decision) == "suppressed_source_ungrounded":
                _renderer_perf_mark_region(
                    debug_context,
                    perf_telemetry_context,
                    region_id,
                    render_source_grounded=False,
                    source_grounding_status="suppressed_source_ungrounded",
                    source_grounding_failure_reason=_render_eligibility_value(source_grounding_decision, "reason"),
                    render_suppressed_by_source_grounding=True,
                    translated_text_suppressed=True,
                    translated_text_suppressed_reason="source_ungrounded_render_eligibility",
                    cleanup_applied=False,
                )
                continue
            pre_render_cleanup_already_committed = False
            raw_text = str(region.get("translation", "")).strip()
            text = _normalize_text(raw_text)
            flags = region.get("flags", {}) or {}
            if not text or flags.get("ignore"):
                continue
            region_type = str(region.get("type", "") or "")
            is_background = region_type == "background_text" or bool(flags.get("bg_text"))
            bbox = region.get("bbox", [0, 0, 0, 0])
            polygon = region.get("polygon")
            poly_bounds = _polygon_bounds(polygon)
            if poly_bounds and _box_area(poly_bounds) < _box_area(bbox) * 0.85:
                x, y, w, h = [int(v) for v in poly_bounds]
            else:
                x, y, w, h = [int(v) for v in bbox]
            mask_pad = _fill_padding(w, h)
            text_pad = _text_padding(w, h)
            mx0 = max(0, x - mask_pad)
            my0 = max(0, y - mask_pad)
            mx1 = min(img_w, x + w + mask_pad)
            my1 = min(img_h, y + h + mask_pad)
            tx0 = max(0, x - text_pad)
            ty0 = max(0, y - text_pad)
            tx1 = min(img_w, x + w + text_pad)
            ty1 = min(img_h, y + h + text_pad)
            base_box = (x, y, x + w, y + h)
            render_box = (tx0, ty0, tx1, ty1)
            render = region.get("render") or {}
            if not isinstance(render, dict):
                render = {}
            render = _render_with_style_contract(region, render)
            source_orientation = str(render.get("source_orientation", "") or "").strip().lower()
            source_size_hint = max(0, int(render.get("source_size_hint", render.get("font_size", 0)) or 0))
            source_size_min = max(0, int(render.get("source_size_min", 0) or 0))
            source_size_max = max(0, int(render.get("source_size_max", 0) or 0))
            parent_style_font_size_locked = bool(render.get("font_size_locked"))
            parent_execution_layout_box, parent_execution_layout_source = _parent_execution_hard_layout_box(
                region,
                render,
                img_w,
                img_h,
            )
            parent_execution_layout_authoritative = parent_execution_layout_box is not None
            cleanup_mode = str(render.get("cleanup_mode", "bubble" if not is_background else "background_box") or "").strip().lower()
            top_row_caption = _is_top_row_nonbubble_caption(
                region,
                render,
                region_type,
                flags,
                bbox,
                img_h,
            )
            side_caption = _is_vertical_side_caption(region, render)
            forced_color = None
            stats = None
            estimated_box = None
            if img_np is not None:
                stats = _box_luma_stats(img_np, base_box)
                if stats:
                    mean, p20, p80 = stats
                    if p80 < 95:
                        forced_color = (255, 255, 255)
                    elif p20 > 175:
                        forced_color = (0, 0, 0)
            if region_type in {"background_text", "decorative_text", "narration_box"}:
                render_box = base_box
            region_mask = None
            bubble_box = None
            bubble_mask = None
            region_cleanup_box = None
            region_cleanup_pixels = 0
            source_glyph_erasure_fields: dict[str, object] = {}
            recovered_speech_anchor = _is_recovered_logical_speech_anchor(region, render)
            logical_text_cleanup_anchor = _is_logical_text_cleanup_anchor(region, render)
            if img_np is not None and text_mask is not None and not is_background:
                region_masks_start = time.time()
                region_mask, bubble_box, bubble_mask = _region_masks(
                    img_np,
                    (mx0, my0, mx1, my1),
                    bbox,
                    polygon,
                )
                add_timing(debug_context, "renderer_region_masks_time", time.time() - region_masks_start)
                if region_mask is not None:
                    use_recovered_local_cleanup = (
                        renderer_cleanup_mutation_enabled
                        and logical_text_cleanup_anchor
                        and cleanup_mode in {"bubble", "speech_bubble", "speech_strong"}
                    )
                    if use_recovered_local_cleanup:
                        recovered_dilate = max(3, min(8, int(max(min(w, h) * 0.10, h * 0.025))))
                        source_glyph_lookup_start = time.time()
                        source_glyph_record = source_glyph_masks_by_region.get(region_id)
                        add_timing(
                            debug_context,
                            "renderer_source_glyph_record_lookup_time",
                            time.time() - source_glyph_lookup_start,
                        )
                        local_mask = _source_glyph_record_mask(source_glyph_record)
                        source_glyph_fallback_used = False
                        source_glyph_fallback_reason = None
                        refinement_fields: dict[str, object] = {}
                        recovered_fallback_start = time.time()
                        fallback_mask = _recovered_speech_source_cleanup_mask(
                            img_np,
                            region,
                            render,
                            regions_by_id,
                            bbox,
                            polygon,
                            dilate_px=max(2, min(4, recovered_dilate)),
                        )
                        add_timing(
                            debug_context,
                            "renderer_recovered_speech_fallback_mask_time",
                            time.time() - recovered_fallback_start,
                        )
                        if local_mask is None:
                            local_mask = fallback_mask
                            if local_mask is not None:
                                source_glyph_fallback_used = True
                                source_glyph_fallback_reason = (
                                    "missing_precomputed_recovered_speech_mask"
                                    if recovered_speech_anchor
                                    else "missing_precomputed_logical_text_block_member_mask"
                                )
                        elif fallback_mask is not None:
                            recovered_refine_start = time.time()
                            merged_mask, merge_fields = _merge_recovered_speech_cleanup_masks(local_mask, fallback_mask)
                            add_timing(
                                debug_context,
                                "renderer_recovered_mask_refine_time",
                                time.time() - recovered_refine_start,
                            )
                            if merged_mask is not None:
                                local_mask = merged_mask
                                source_glyph_fallback_used = True
                                source_glyph_fallback_reason = merge_fields.get("source_glyph_mask_fallback_reason")
                                refinement_fields.update(merge_fields)
                        if local_mask is not None:
                            recovered_refine_start = time.time()
                            refined_mask, close_fields = _refine_recovered_speech_cleanup_mask(local_mask)
                            add_timing(
                                debug_context,
                                "renderer_recovered_mask_refine_time",
                                time.time() - recovered_refine_start,
                            )
                            if refined_mask is not None:
                                local_mask = refined_mask
                            if close_fields:
                                refinement_fields.update(close_fields)
                        # Recovered logical anchors may represent several OCR
                        # fragments across a physical speech container. The
                        # per-region bubble mask is often derived from the
                        # anchor bbox and can clip away represented child/source
                        # glyphs, which leaves Japanese visible. The recovered
                        # mask is already clipped to the TextAreaPlan container
                        # when that evidence is available, so do not intersect
                        # it with the narrower region mask here.
                        if local_mask is not None:
                            local_cleanup_stats = mask_stats(local_mask)
                            region_cleanup_box = _mask_stats_box(local_cleanup_stats)
                            region_cleanup_pixels = int((local_cleanup_stats or {}).get("pixels") or 0)
                            source_glyph_audit_start = time.time()
                            source_glyph_erasure_fields = _source_glyph_record_audit_fields(source_glyph_record)
                            add_timing(
                                debug_context,
                                "renderer_source_glyph_audit_fields_time",
                                time.time() - source_glyph_audit_start,
                            )
                            if not source_glyph_erasure_fields:
                                source_glyph_audit_start = time.time()
                                source_glyph_erasure_fields = _source_glyph_erasure_audit_fields(
                                    img_np,
                                    local_mask,
                                    _source_erasure_expected_box(
                                        region,
                                        render,
                                        regions_by_id,
                                        bbox,
                                        img_w,
                                        img_h,
                                    ),
                                )
                                add_timing(
                                    debug_context,
                                    "renderer_source_glyph_audit_fields_time",
                                    time.time() - source_glyph_audit_start,
                                )
                            if source_glyph_fallback_used:
                                source_glyph_erasure_fields.update(
                                    {
                                        "cleanup_source_tracking_required": True,
                                        "source_glyph_mask_generation_required": True,
                                        "source_glyph_mask_generation_status": "generated_consumed",
                                        "source_glyph_mask_not_generated_reason": None,
                                        "source_glyph_mask_review_only": False,
                                        "source_glyph_mask_required": True,
                                        "source_glyph_mask_generated": True,
                                        "source_glyph_mask_consumed_by_renderer": True,
                                        "source_glyph_mask_generation_method": (
                                            "renderer_shared_fallback_speech_recovered_anchor_glyph_local"
                                            if recovered_speech_anchor
                                            else "renderer_shared_fallback_logical_text_block_member_glyph_local"
                                        ),
                                        "source_glyph_mask_fallback_used": True,
                                        "source_glyph_mask_fallback_reason": source_glyph_fallback_reason,
                                    }
                                )
                            if refinement_fields:
                                source_glyph_erasure_fields.update(refinement_fields)
                            cleanup_tag = (
                                "speech_recovered_anchor_glyph_local"
                                if recovered_speech_anchor
                                else "logical_text_block_member_glyph_local"
                            )
                            local_cleanup_masks.append((local_mask, cleanup_tag, region_id))
                            mark_render_region(
                                debug_context,
                                region_id,
                                cleanup_applied=True,
                                cleanup_mode=cleanup_tag,
                                cleanup_mask=local_cleanup_stats,
                                **source_glyph_erasure_fields,
                            )
                        else:
                            use_recovered_local_cleanup = False
                    if not use_recovered_local_cleanup:
                        if renderer_cleanup_mutation_enabled and bubble_mask is not None:
                            region_mask = cv2.bitwise_and(region_mask, bubble_mask)
                        if renderer_cleanup_mutation_enabled:
                            cleanup_stats = mask_stats(region_mask)
                            region_cleanup_box = _mask_stats_box(cleanup_stats)
                            region_cleanup_pixels = int((cleanup_stats or {}).get("pixels") or 0)
                            mark_render_region(
                                debug_context,
                                region_id,
                                cleanup_applied=True,
                                cleanup_mode=cleanup_mode,
                                cleanup_mask=cleanup_stats,
                            )
                            text_mask = cv2.bitwise_or(text_mask, region_mask)
                            if region_id and region_id not in global_text_mask_region_ids:
                                global_text_mask_region_ids.append(region_id)
                            if region_id:
                                global_cleanup_candidates.append(
                                    _cleanup_partition_candidate(
                                        region,
                                        render,
                                        region_id,
                                        region_mask.copy(),
                                    )
                                )
                            if bubble_mask is not None and bubble_text_mask is not None:
                                bubble_text_mask = cv2.bitwise_or(bubble_text_mask, region_mask)
                                if bubble_area_mask is not None:
                                    bubble_area_mask = cv2.bitwise_or(bubble_area_mask, bubble_mask)
                            elif other_text_mask is not None:
                                other_text_mask = cv2.bitwise_or(other_text_mask, region_mask)
                if bubble_box is not None:
                    bubble_area = _box_area([bubble_box[0], bubble_box[1], bubble_box[2] - bubble_box[0], bubble_box[3] - bubble_box[1]])
                    base_area = _box_area([base_box[0], base_box[1], base_box[2] - base_box[0], base_box[3] - base_box[1]])
                    if base_area and bubble_area >= base_area * 0.6:
                        if region_type == "speech_bubble":
                            chosen_limit_box = bubble_box
                            if source_orientation == "vertical":
                                micro_start = _renderer_micro_start()
                                edge_box = _estimate_vertical_speech_box_from_edges(img_np, base_box)
                                white_box = _estimate_speech_bubble_box(img_np, base_box)
                                _renderer_micro_add("renderer_bubble_box_estimation_time", micro_start)
                                best_area = bubble_area
                                estimated_box = None
                                for candidate_box in (edge_box, white_box):
                                    if candidate_box is None:
                                        continue
                                    cand_area = _box_area([candidate_box[0], candidate_box[1], candidate_box[2] - candidate_box[0], candidate_box[3] - candidate_box[1]])
                                    if cand_area > best_area:
                                        chosen_limit_box = candidate_box
                                        estimated_box = candidate_box
                                        best_area = cand_area
                            micro_start = _renderer_micro_start()
                            render_box = _bubble_inner_layout_box(
                                bubble_mask,
                                chosen_limit_box,
                                base_box,
                                text,
                                img_w,
                                img_h,
                                preferred_size=source_size_hint or None,
                                min_size=source_size_min or None,
                            ) or chosen_limit_box
                            _renderer_micro_add("renderer_bubble_inner_layout_box_time", micro_start)
                        else:
                            limited = _intersect_box(render_box, bubble_box)
                            render_box = limited or bubble_box
                    else:
                        render_box = base_box
                else:
                    if region_type == "speech_bubble":
                        micro_start = _renderer_micro_start()
                        edge_box = _estimate_vertical_speech_box_from_edges(img_np, base_box)
                        white_box = _estimate_speech_bubble_box(img_np, base_box)
                        _renderer_micro_add("renderer_bubble_box_estimation_time", micro_start)
                        estimated_box = edge_box
                        if white_box is not None:
                            if estimated_box is None:
                                estimated_box = white_box
                            else:
                                edge_area = _box_area([estimated_box[0], estimated_box[1], estimated_box[2] - estimated_box[0], estimated_box[3] - estimated_box[1]])
                                white_area = _box_area([white_box[0], white_box[1], white_box[2] - white_box[0], white_box[3] - white_box[1]])
                                if white_area > edge_area:
                                    estimated_box = white_box
                        if estimated_box is not None:
                            micro_start = _renderer_micro_start()
                            render_box = _bubble_inner_layout_box(
                                None,
                                estimated_box,
                                base_box,
                                text,
                                img_w,
                                img_h,
                                preferred_size=source_size_hint or None,
                                min_size=source_size_min or None,
                            ) or estimated_box
                            _renderer_micro_add("renderer_bubble_inner_layout_box_time", micro_start)
                        else:
                            render_box = _fallback_speech_render_box(base_box, text, img_w, img_h)
                    else:
                        render_box = base_box
                if region_type == "speech_bubble":
                    render_limit = bubble_box if bubble_box is not None else estimated_box
                    if source_orientation == "vertical" and estimated_box is not None:
                        est_area = _box_area([estimated_box[0], estimated_box[1], estimated_box[2] - estimated_box[0], estimated_box[3] - estimated_box[1]])
                        if render_limit is None:
                            render_limit = estimated_box
                        else:
                            limit_area = _box_area([render_limit[0], render_limit[1], render_limit[2] - render_limit[0], render_limit[3] - render_limit[1]])
                            if est_area > limit_area:
                                render_limit = estimated_box
                    micro_start = _renderer_micro_start()
                    render_box = _stabilize_tiny_speech_render_box(
                        render_box,
                        base_box,
                        text,
                        img_w,
                        img_h,
                        limit_box=render_limit,
                        preferred_size=source_size_hint or None,
                        min_size=source_size_min or None,
                    )
                    _renderer_micro_add("renderer_tiny_speech_stabilize_time", micro_start)
                    if (
                        renderer_cleanup_mutation_enabled
                        and bubble_mask is None
                        and region_mask is not None
                        and not use_recovered_local_cleanup
                    ):
                        # Missing bubble masks on true speech bubbles were creating synthetic
                        # rectangular fills and the gray-box artifacts visible on small bubbles.
                        # Fall back to strong local glyph cleanup instead of inventing a fake
                        # bubble interior.
                        if other_text_mask is not None:
                            other_text_mask = cv2.bitwise_and(other_text_mask, cv2.bitwise_not(region_mask))
                        speech_dilate = max(4, int(max(min(w, h) * 0.08, h * 0.03)))
                        local_region_mask_start = time.time()
                        local_mask = _local_region_text_mask(
                            img_np,
                            bbox,
                            polygon,
                            dilate_px=speech_dilate,
                            strong_vertical=(source_orientation == "vertical" and h > w * 0.9),
                            glyph_only=True,
                        )
                        add_timing(
                            debug_context,
                            "renderer_local_region_text_mask_time",
                            time.time() - local_region_mask_start,
                        )
                        if local_mask is not None:
                            if (
                                cleanup_mode in {"bubble", "speech_bubble", "speech_strong"}
                                and source_orientation == "vertical"
                                and h > w * 0.9
                            ):
                                local_mask = _expand_vertical_speech_cleanup_neighbors(
                                    img_np,
                                    local_mask,
                                    bbox,
                                )
                            local_cleanup_stats = mask_stats(local_mask)
                            region_cleanup_box = _mask_stats_box(local_cleanup_stats) or region_cleanup_box
                            region_cleanup_pixels = int((local_cleanup_stats or {}).get("pixels") or region_cleanup_pixels or 0)
                            local_cleanup_masks.append((local_mask, "speech_strong", region_id))
                            mark_render_region(
                                debug_context,
                                region_id,
                                cleanup_applied=True,
                                cleanup_mode="speech_strong",
                                cleanup_mask=local_cleanup_stats,
                            )
                            if text_mask is not None:
                                text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(region_mask))
                                if region_id in global_text_mask_region_ids:
                                    global_text_mask_region_ids.remove(region_id)
                            if bubble_text_mask is not None:
                                bubble_text_mask = cv2.bitwise_and(bubble_text_mask, cv2.bitwise_not(region_mask))
                elif region_type in {"background_text", "narration_box"}:
                    render_box = _stabilize_tiny_nonbubble_render_box(
                        render_box,
                        base_box,
                        text,
                        img_w,
                        img_h,
                        preferred_size=source_size_hint or None,
                        min_size=source_size_min or None,
                        wrap_mode=str(render.get("wrap_mode", "auto") or "auto"),
                    )
            meaningful_background = False
            if is_background:
                background_text = str(region.get("ocr_text", "") or "").strip()
                background_body = "".join(ch for ch in background_text if ch.strip() and ch not in "。，、！？：；…‥・･ー-—―－〜～「」『』（）()[]【】<>〈〉《》“”‘’\"' ")
                contains_kanji = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in background_body)
                contains_kana = any(0x3040 <= ord(ch) <= 0x30FF for ch in background_body)
                contains_digits = any(ch.isdigit() for ch in background_text)
                meaningful_background = (
                    region_type == "background_text"
                    and (
                        (len(background_body) >= 4 and (contains_kanji or contains_kana))
                        or (contains_kanji and contains_digits)
                        or any(marker in background_text for marker in ("日目", "回目", "生活", "最終日", "無人島"))
                    )
                )
            if is_background and cleanup_mode not in {"preserve", "local_text_mask"}:
                bg_area_ratio = (max(1, w) * max(1, h)) / max(1, img_w * img_h)
                contrast_span = None
                if stats:
                    _mean, p20, p80 = stats
                    contrast_span = p80 - p20
                if region_type == "background_text" and (
                    bg_area_ratio >= 0.015
                    or (w <= 60 and h >= 150 and contrast_span is not None and contrast_span >= 28)
                    or (contrast_span is not None and contrast_span >= 52 and bg_area_ratio >= 0.002)
                ):
                    cleanup_mode = "local_text_mask" if meaningful_background else "preserve"
            if is_background and meaningful_background and cleanup_mode == "preserve":
                cleanup_mode = "local_text_mask"
            if is_background and cleanup_mode == "local_text_mask":
                render_box = _shrink_box(render_box, max(1, int(min(w, h) * 0.02)))
                if (
                    renderer_cleanup_mutation_enabled
                    and not pre_render_cleanup_already_committed
                    and text_mask is not None
                    and img_np is not None
                ):
                    if top_row_caption:
                        local_dilate = max(1, min(2, int(min(w, h) * 0.04)))
                    elif meaningful_background and side_caption and h > w * 1.5:
                        local_dilate = 1
                    elif meaningful_background and h > w * 1.5:
                        local_dilate = max(8, int(max(w * 0.30, h * 0.05)))
                    elif meaningful_background:
                        local_dilate = max(6, int(max(min(w, h) * 0.12, max(w, h) * 0.03)))
                    else:
                        local_dilate = max(2, int(min(w, h) * 0.04))
                    source_glyph_lookup_start = time.time()
                    source_glyph_record = source_glyph_masks_by_region.get(region_id)
                    add_timing(
                        debug_context,
                        "renderer_source_glyph_record_lookup_time",
                        time.time() - source_glyph_lookup_start,
                    )
                    local_mask = _source_glyph_record_mask(source_glyph_record)
                    used_precomputed_caption_mask = local_mask is not None
                    source_glyph_fallback_used = False
                    source_glyph_fallback_reason = None
                    if used_precomputed_caption_mask:
                        pass
                    elif top_row_caption:
                        local_mask = _top_row_caption_glyph_mask(
                            img_np,
                            bbox,
                            polygon,
                            dilate_px=local_dilate,
                        )
                        local_mask = _expand_bright_top_row_caption_neighbor_mask(
                            img_np,
                            local_mask,
                            bbox,
                        )
                    elif side_caption:
                        local_mask = _source_glyph_local_mask(
                            img_np,
                            bbox,
                            polygon,
                            dilate_px=local_dilate,
                            limit_box=_xywh_value_to_xyxy(
                                _canonical_region_render_value(region, render, "text_area_container_bbox"),
                                img_w,
                                img_h,
                            ),
                        )
                        if local_mask is not None:
                            source_glyph_fallback_used = True
                            source_glyph_fallback_reason = "missing_precomputed_side_caption_mask"
                    else:
                        local_region_mask_start = time.time()
                        local_mask = _local_region_text_mask(
                            img_np,
                            bbox,
                            polygon,
                            dilate_px=local_dilate,
                            strong_vertical=(meaningful_background and h > w * 1.2 and not side_caption),
                            glyph_only=bool(side_caption),
                        )
                        add_timing(
                            debug_context,
                            "renderer_local_region_text_mask_time",
                            time.time() - local_region_mask_start,
                        )
                    if local_mask is not None:
                        local_cleanup_stats = mask_stats(local_mask)
                        caption_erasure_fields = {}
                        if used_precomputed_caption_mask or side_caption:
                            source_glyph_audit_start = time.time()
                            caption_erasure_fields = _source_glyph_record_audit_fields(source_glyph_record)
                            add_timing(
                                debug_context,
                                "renderer_source_glyph_audit_fields_time",
                                time.time() - source_glyph_audit_start,
                            )
                            if not caption_erasure_fields:
                                expected_box = tuple(int(v) for v in base_box[:4])
                                source_glyph_audit_start = time.time()
                                caption_erasure_fields = _source_glyph_erasure_audit_fields(
                                    img_np,
                                    local_mask,
                                    expected_box,
                                )
                                add_timing(
                                    debug_context,
                                    "renderer_source_glyph_audit_fields_time",
                                    time.time() - source_glyph_audit_start,
                                )
                            if source_glyph_fallback_used:
                                caption_erasure_fields.update(
                                    {
                                        "cleanup_source_tracking_required": True,
                                        "source_glyph_mask_generation_required": True,
                                        "source_glyph_mask_generation_status": "generated_consumed",
                                        "source_glyph_mask_not_generated_reason": None,
                                        "source_glyph_mask_review_only": False,
                                        "source_glyph_mask_required": True,
                                        "source_glyph_mask_generated": True,
                                        "source_glyph_mask_consumed_by_renderer": True,
                                        "source_glyph_mask_generation_method": "renderer_shared_fallback_side_caption_glyph_local",
                                        "source_glyph_mask_fallback_used": True,
                                        "source_glyph_mask_fallback_reason": source_glyph_fallback_reason,
                                    }
                                )
                        if used_precomputed_caption_mask:
                            cleanup_tag = str(caption_erasure_fields.get("source_glyph_mask_generation_method") or "caption_background_logical_block_glyph_local")
                        elif side_caption:
                            cleanup_tag = "side_caption_glyph_local"
                        else:
                            cleanup_tag = "caption_strong" if meaningful_background else cleanup_mode
                        local_cleanup_masks.append((local_mask, cleanup_tag, region_id))
                        mark_render_region(
                            debug_context,
                            region_id,
                            cleanup_applied=True,
                            cleanup_mode=cleanup_tag,
                            cleanup_mask=local_cleanup_stats,
                            **caption_erasure_fields,
                        )
            elif (
                renderer_cleanup_mutation_enabled
                and is_background
                and cleanup_mode != "preserve"
                and not pre_render_cleanup_already_committed
            ):
                pad = max(1, int(min(w, h) * 0.04)) if region_type == "background_text" else max(1, int(min(w, h) * 0.02))
                render_box = _shrink_box(render_box, pad)
                bg_pad = min(12, max(2, int(min(w, h) * 0.10))) if region_type == "background_text" else min(28, max(4, int(min(w, h) * 0.18)))
                fill_color = _estimate_box_fill(img_np, base_box) if img_np is not None else None
                expanded_box = None
                if img_np is not None and region_type == "narration_box":
                    stats = _box_luma_stats(img_np, base_box)
                    if stats:
                        _mean, _p20, _p80 = stats
                        if _p80 < 95:
                            expanded_box = _expand_dark_box(img_np, base_box)
                if expanded_box is not None:
                    ex0, ey0, ex1, ey1 = expanded_box
                    render_box = _intersect_box(render_box, expanded_box) or expanded_box
                    background_boxes.append(((ex0, ey0, ex1, ey1), fill_color))
                else:
                    background_boxes.append(
                        (
                            (
                                max(0, x - bg_pad),
                                max(0, y - bg_pad),
                                min(img_w, x + w + bg_pad),
                                min(img_h, y + h + bg_pad),
                            ),
                            fill_color,
                        )
                    )
                mark_render_region(
                    debug_context,
                    region_id,
                    cleanup_applied=True,
                    cleanup_mode=cleanup_mode,
                    cleanup_mask={"pixels": None, "bbox": list(background_boxes[-1][0]) if background_boxes else None},
                )
            elif not is_background:
                if region_type != "speech_bubble":
                    shrink = max(2, int(min(w, h) * 0.03))
                    render_box = _shrink_box(render_box, shrink)
            if parent_execution_layout_authoritative:
                render_box = parent_execution_layout_box
            region_font = render.get("font") or font_name
            font_style = str(render.get("font_style", "") or "dialogue")
            if _has_cjk(text) and _is_cjk_unsupported_font(region_font):
                region_font = font_name
            configured_color = _parse_color(render.get("color"))
            stroke_width = max(0, int(render.get("stroke_width", 0) or 0))
            stroke_color = _parse_color(render.get("stroke"))
            if top_row_caption:
                forced_color = (255, 255, 255)
                stroke_color = (20, 20, 20)
                stroke_width = max(2, stroke_width)
            line_height = _coerce_render_line_height(render, source_orientation, region_type)
            font_size_override = max(0, int(render.get("font_size", 0) or 0))
            if (
                font_size_override <= 0
                and source_size_hint <= 0
                and not parent_execution_layout_authoritative
                and region_type == "speech_bubble"
                and h > w * 1.6
                and w <= 90
                and str(render.get("wrap_mode", "auto") or "auto").strip().lower() == "vertical"
            ):
                font_size_override = _estimate_vertical_speech_preferred_size(base_box, text)
            if min(w, h) <= 40:
                stroke_width = min(stroke_width, 1)
            effective_color = forced_color or configured_color
            if region_type == "narration_box":
                if img_np is not None:
                    fill_color = _estimate_box_fill(img_np, base_box)
                else:
                    fill_color = None
                expanded_box = _expand_dark_box(img_np, base_box) if img_np is not None else None
                if expanded_box is not None:
                    render_box = _shrink_box(expanded_box, max(2, int(min(w, h) * 0.08)))
                    caption_boxes.append((expanded_box, fill_color))
                else:
                    caption_boxes.append((base_box, fill_color))
            obstacle_adjustment = None
            if region_type == "speech_bubble":
                render_box, text_area_constraint_adjustment = _apply_text_area_speech_container_constraint(
                    render_box,
                    base_box,
                    region,
                    render,
                    text,
                    img_w,
                    img_h,
                    preferred_size=source_size_hint or None,
                    min_size=source_size_min or None,
                )
            else:
                text_area_constraint_adjustment = None
            allowed_area_box, allowed_area_source = _render_allowed_area_box_for_audit(
                region,
                render,
                region_type,
                img_w,
                img_h,
            )
            if region_type == "speech_bubble" and not parent_execution_layout_authoritative:
                render_box, obstacle_ids = _avoid_preserved_text_obstacles(
                    render_box,
                    base_box,
                    region_cleanup_box,
                    preserved_text_obstacles,
                    region_id,
                )
                if obstacle_ids:
                    obstacle_adjustment = {
                        "reason": "preserved_text_obstacle_render_box_fallback",
                        "obstacle_region_ids": obstacle_ids,
                    }
            render_constraint_adjustment = None
            if region_type == "speech_bubble" and not parent_execution_layout_authoritative:
                micro_start = _renderer_micro_start()
                render_box, render_constraint_adjustment = _apply_render_constraint_assist(
                    render_box,
                    base_box,
                    region,
                    region_id,
                    debug_context,
                    image_path,
                    img_w,
                    img_h,
                    preserved_text_obstacles,
                    model_fusion_proof_candidates,
                )
                _renderer_micro_add("renderer_render_constraint_assist_time", micro_start)
                if render_constraint_adjustment and render_constraint_adjustment.get("model_fusion_mutation_proof_applied"):
                    render_constraint_adjustment["translated_text"] = text
                    render_constraint_adjustment["wrapped_lines_before"] = _preview_wrapped_lines_for_box(
                        text,
                        render_constraint_adjustment.get("render_constraint_previous_final_render_bbox") or render_box,
                        region_font,
                        str(render.get("wrap_mode", "auto") or "auto"),
                        source_orientation,
                        line_height,
                        source_size_hint if source_size_hint > 0 else None,
                        source_size_min if source_size_min > 0 else None,
                        source_size_max if source_size_max > 0 else None,
                    )
            render_fit_adjustment = None
            micro_start = _renderer_micro_start()
            with _renderer_region_scope(region_id):
                render_box, render_fit_adjustment = _apply_root_local_render_fit_pass(
                    render_box,
                    allowed_area_box,
                    region,
                    render,
                    source_glyph_erasure_fields,
                    region_cleanup_box,
                    text,
                    region_type,
                    region_font,
                    line_height,
                    str(render.get("wrap_mode", "auto") or "auto"),
                    source_orientation,
                    source_size_hint,
                    source_size_min,
                    source_size_max,
                )
            _renderer_micro_add("renderer_root_local_render_fit_pass_time", micro_start)
            if render_fit_adjustment and not parent_style_font_size_locked:
                try:
                    adjusted_line_height = float(render_fit_adjustment.get("render_layout_v2_line_height_scale") or line_height)
                    line_height = _coerce_render_line_height(
                        {
                            "line_height": adjusted_line_height,
                            "wrap_mode": render.get("wrap_mode", "auto"),
                        },
                        source_orientation,
                        region_type,
                    )
                except Exception:
                    pass
            elif render_fit_adjustment:
                render_fit_adjustment["render_layout_line_height_override_suppressed"] = True
                render_fit_adjustment["render_layout_line_height_override_suppressed_reason"] = (
                    "parent_style_authoritative"
                )
            if parent_execution_layout_authoritative and allowed_area_box is not None:
                hard_box = tuple(int(v) for v in allowed_area_box[:4])
                render_box = _intersect_box(render_box, hard_box) or hard_box
            micro_start = _renderer_micro_start()
            render_quality_debug = _render_cleanup_constraint_debug(
                render_box,
                region_cleanup_box,
                allowed_area_box,
                allowed_area_source,
                region_cleanup_pixels,
            )
            render_quality_debug.update(source_glyph_erasure_fields)
            visual_contract_block_reason = _render_visual_contract_block_reason(
                render_fit_adjustment,
                render_quality_debug,
                text,
            )
            render_regions.append(
                (
                    region_id,
                    text,
                    render_box,
                    region_type,
                    region_font,
                    font_style,
                    effective_color,
                    stroke_width,
                    stroke_color,
                    line_height,
                    font_size_override,
                    str(render.get("wrap_mode", "auto") or "auto"),
                    source_orientation,
                    source_size_hint,
                    source_size_min,
                    source_size_max,
                    parent_style_font_size_locked,
                    render_constraint_adjustment,
                    render_fit_adjustment,
                    parent_execution_layout_authoritative,
                    parent_execution_layout_box,
                    parent_execution_layout_source,
                )
            )
            render_debug = {
                "final_render_bbox": list(render_box),
                "selected_font_family": region_font,
                "translated_text": text,
            }
            render_debug.update(_render_style_debug_fields(render))
            for key in (
                "text_block_root_id",
                "parent_logical_text_unit_id",
                "active_translation_unit_id",
                "parent_logical_text_unit_anchor_child_id",
                "parent_logical_text_unit_represented_child_ids",
                "parent_logical_text_unit_child_segment_ids",
                "represented_by_parent_id",
                "source_text_represented_by_block_id",
            ):
                value = _canonical_region_render_value(region, render, key)
                if value not in (None, ""):
                    render_debug[key] = value
            render_debug.update(render_quality_debug)
            if parent_execution_layout_authoritative:
                render_debug.update(
                    {
                        "parent_execution_renderer_layout_authority": "hard_execution_region",
                        "parent_execution_renderer_hard_bbox": list(parent_execution_layout_box),
                        "parent_execution_renderer_hard_bbox_source": parent_execution_layout_source,
                        "parent_execution_renderer_legacy_expansion_suppressed": True,
                    }
                )
            if obstacle_adjustment:
                render_debug["render_box_adjustment"] = obstacle_adjustment
            if text_area_constraint_adjustment:
                render_debug["text_area_render_constraint"] = text_area_constraint_adjustment
            if render_constraint_adjustment:
                render_debug.update(render_constraint_adjustment)
            if render_fit_adjustment:
                render_debug.update(render_fit_adjustment)
            if visual_contract_block_reason:
                render_debug.update(
                    {
                        "render_blocked_by_visual_contract": False,
                        "render_visual_contract_blocker_reason": visual_contract_block_reason,
                        "render_visual_contract_audit_status": "audit_only_render_allowed",
                        "render_readability_warning_reason": visual_contract_block_reason,
                        "render_readability_hard_blocker": True,
                        "render_allowed_despite_audit_blocker": True,
                        "render_suppressed_by_legacy_reason": False,
                    }
                )
            if cleanup_mode == "preserve":
                render_debug["renderer_cleanup_applied"] = False
            if not renderer_cleanup_mutation_enabled:
                render_debug.update(_renderer_cleanup_disabled_fields())
            _renderer_micro_add("renderer_render_debug_payload_time", micro_start)
            micro_start = _renderer_micro_start()
            mark_render_region(debug_context, region_id, **render_debug)
            _renderer_micro_add("renderer_mark_render_region_time", micro_start)
        _renderer_perf_add_timing(
            debug_context,
            perf_telemetry_context,
            "renderer_region_loop_total_time",
            time.time() - renderer_region_loop_start,
        )
        if renderer_cleanup_mutation_enabled and img_np is not None and text_mask is not None:
            represented_child_start = time.time()
            _add_represented_child_cleanup_candidates(
                regions,
                img_np,
                text_mask,
                global_cleanup_candidates,
                global_text_mask_region_ids,
                source_glyph_masks_by_region,
                debug_context,
                img_w,
                img_h,
            )
            add_timing(
                debug_context,
                "renderer_represented_child_cleanup_candidate_time",
                time.time() - represented_child_start,
            )
        _renderer_perf_add_timing(
            debug_context,
            perf_telemetry_context,
            "mask_generation_time",
            time.time() - mask_generation_start,
        )

        if render_regions and renderer_cleanup_mutation_enabled:
            if caption_boxes:
                working = _apply_caption_fill(working, caption_boxes)
                if cv2 is not None and np is not None:
                    img_np = np.array(working)

            if background_boxes:
                working = _apply_background_fill(working, background_boxes)
                if cv2 is not None and np is not None:
                    img_np = np.array(working)

            if local_cleanup_masks:
                for local_mask, _cleanup_mode, _region_id in local_cleanup_masks:
                    cleanup_start = time.time()
                    cleanup_debug: dict[str, object] = {}
                    partition_meta = _cleanup_partition_metadata_for_region(
                        regions_by_id.get(str(_region_id or "")),
                        cleanup_scope="source_glyph" if _cleanup_mode in {
                            "speech_recovered_anchor_glyph_local",
                            "caption_background_logical_block_glyph_local",
                            "side_caption_glyph_local",
                        } else "local",
                        cleanup_partition_id=f"local_{_region_id or 'unknown'}",
                    )
                    pre_cleanup_image = working
                    working = _apply_local_text_removal(
                        working,
                        local_mask,
                        inpaint_mode,
                        use_gpu,
                        model_id=model_id,
                        cleanup_tag=_cleanup_mode,
                        debug_info=cleanup_debug,
                    )
                    cleanup_elapsed = time.time() - cleanup_start
                    add_timing(debug_context, "inpainting_time", cleanup_elapsed)
                    add_count(debug_context, "inpaint_calls")
                    local_stats = mask_stats(local_mask)
                    local_bbox = _mask_stats_box(local_stats)
                    operation = {
                        "operation_kind": "local_text_removal",
                        "region_id": _region_id,
                        **partition_meta,
                        "cleanup_mode": _cleanup_mode,
                        "elapsed_sec": round(cleanup_elapsed, 6),
                        "backend": cleanup_debug.get("backend"),
                        "backend_detail": cleanup_debug.get("backend_detail"),
                        "requested_inpaint_mode": inpaint_mode,
                        "effective_inpaint_mode": cleanup_debug.get("effective_inpaint_mode"),
                        "mask_pixels": int((local_stats or {}).get("pixels") or 0),
                        "mask_bbox": list(local_bbox) if local_bbox else None,
                        "mask_bbox_area": _xyxy_area(local_bbox) if local_bbox else 0,
                        "crop_bbox": cleanup_debug.get("crop_bbox"),
                        "crop_area": cleanup_debug.get("crop_area"),
                        "mask_ratio": cleanup_debug.get("mask_ratio"),
                        "root_cleanup_repair_v2_attempted": cleanup_debug.get("root_cleanup_repair_v2_attempted"),
                        "root_cleanup_repair_v2_status": cleanup_debug.get("root_cleanup_repair_v2_status"),
                        "root_cleanup_repair_v2_backend": cleanup_debug.get("root_cleanup_repair_v2_backend"),
                        "root_cleanup_repair_v2_attempt_count": cleanup_debug.get("root_cleanup_repair_v2_attempt_count"),
                        "root_cleanup_repair_v2_reason": cleanup_debug.get("root_cleanup_repair_v2_reason"),
                    }
                    expected_mask = _cleanup_expected_mask_for_operation(debug_context, operation, local_mask)
                    operation = _record_cleanup_operation_trace(
                        debug_context,
                        operation,
                        pre_cleanup_image,
                        working,
                        local_mask,
                        expected_mask=expected_mask,
                    )
                    working, retry_operation = _maybe_apply_cleanup_effectiveness_retry(
                        working=working,
                        cleanup_mask=local_mask,
                        expected_mask=expected_mask,
                        operation=operation,
                        debug_context=debug_context,
                        inpaint_mode=inpaint_mode,
                        use_gpu=use_gpu,
                        model_id=model_id,
                        region_id=str(_region_id or ""),
                    )
                    _record_render_cleanup_operation(debug_context, operation)
                    mark_render_region(
                        debug_context,
                        _region_id,
                        cleanup_inpaint_time_sec=round(cleanup_elapsed, 6),
                        cleanup_backend=cleanup_debug.get("backend"),
                        cleanup_backend_detail=cleanup_debug.get("backend_detail"),
                        cleanup_operation_kind="local_text_removal",
                        cleanup_requested_inpaint_mode=inpaint_mode,
                        cleanup_effective_inpaint_mode=cleanup_debug.get("effective_inpaint_mode"),
                        cleanup_mask_pixels=int((local_stats or {}).get("pixels") or 0),
                        cleanup_mask_bbox_area=_xyxy_area(local_bbox) if local_bbox else 0,
                        cleanup_crop_bbox=cleanup_debug.get("crop_bbox"),
                        cleanup_crop_area=cleanup_debug.get("crop_area"),
                        cleanup_mask_ratio=cleanup_debug.get("mask_ratio"),
                        root_cleanup_repair_v2_attempted=cleanup_debug.get("root_cleanup_repair_v2_attempted"),
                        root_cleanup_repair_v2_status=cleanup_debug.get("root_cleanup_repair_v2_status"),
                        root_cleanup_repair_v2_backend=cleanup_debug.get("root_cleanup_repair_v2_backend"),
                        root_cleanup_repair_v2_attempt_count=cleanup_debug.get("root_cleanup_repair_v2_attempt_count"),
                        root_cleanup_repair_v2_reason=cleanup_debug.get("root_cleanup_repair_v2_reason"),
                        cleanup_effectiveness_retry_attempted=operation.get("cleanup_effectiveness_retry_attempted"),
                        cleanup_effectiveness_retry_status=operation.get("cleanup_effectiveness_retry_status"),
                        cleanup_effectiveness_retry_operation_id=operation.get("cleanup_effectiveness_retry_operation_id"),
                        cleanup_effectiveness_retry_residual_ratio=operation.get("cleanup_effectiveness_retry_residual_ratio"),
                        **partition_meta,
                    )
                    _mark_source_children_covered_by_mask(
                        debug_context,
                        local_mask,
                        str(partition_meta.get("cleanup_partition_id") or f"local_{_region_id or 'unknown'}"),
                        str(partition_meta.get("cleanup_partition_scope") or "local"),
                    )
                    if text_mask is not None:
                        text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(local_mask))
                    if other_text_mask is not None:
                        other_text_mask = cv2.bitwise_and(other_text_mask, cv2.bitwise_not(local_mask))
                if cv2 is not None and np is not None:
                    img_np = np.array(working)

            if bubble_area_mask is not None and bubble_text_mask is not None and bubble_area_mask.any():
                if cv2 is not None and np is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    bubble_area_mask = cv2.morphologyEx(bubble_area_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                working = _apply_bubble_fill(working, bubble_area_mask, bubble_text_mask, img_np)
                if text_mask is not None:
                    text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(bubble_text_mask))

            # Now apply text removal for everything left after local/source-glyph cleanup.
            # Prefer parent/root partitions so unrelated page areas do not force one
            # page-spanning AI crop. The previous global mask remains the fail-closed
            # fallback when partitioning is unavailable or rejected.
            if text_mask is not None and text_mask.any():
                partitioned = False
                if cv2 is not None and np is not None:
                    partition_source = working
                    try:
                        cleanup_partition_start = time.time()
                        partitions = _build_cleanup_partitions(
                            text_mask,
                            global_cleanup_candidates,
                            global_text_mask_region_ids,
                            img_w,
                            img_h,
                        )
                        add_timing(
                            debug_context,
                            "renderer_cleanup_partition_build_time",
                            time.time() - cleanup_partition_start,
                        )
                        if _should_use_cleanup_partitions(text_mask, partitions, inpaint_mode, img_w, img_h):
                            working = _apply_cleanup_partitions(
                                working,
                                partitions,
                                inpaint_mode,
                                use_gpu,
                                model_id,
                                debug_context,
                            )
                            partitioned = True
                    except Exception as exc:
                        working = partition_source
                        _record_render_cleanup_operation(
                            debug_context,
                            {
                                "operation_kind": "cleanup_partitioning_failed",
                                "cleanup_partition_scope": "global_fallback",
                                "cleanup_partition_id": "global_fallback",
                                "region_ids": list(global_text_mask_region_ids),
                                "fallback_reason": f"{type(exc).__name__}:{exc}",
                            },
                        )
                if not partitioned:
                    cleanup_start = time.time()
                    cleanup_debug: dict[str, object] = {}
                    global_stats = mask_stats(text_mask)
                    global_bbox = _mask_stats_box(global_stats)
                    pre_cleanup_image = working
                    working = _apply_text_removal(working, text_mask, inpaint_mode, use_gpu, model_id=model_id, debug_info=cleanup_debug)
                    cleanup_elapsed = time.time() - cleanup_start
                    add_timing(debug_context, "inpainting_time", cleanup_elapsed)
                    add_count(debug_context, "inpaint_calls")
                    operation = {
                            "operation_kind": "global_text_mask_removal",
                            "region_ids": list(global_text_mask_region_ids),
                            "cleanup_mode": "global_text_mask",
                            "cleanup_partition_id": "global_fallback",
                            "cleanup_partition_scope": "global_fallback",
                            "fallback_reason": "partitioning_rejected_or_unavailable",
                            "elapsed_sec": round(cleanup_elapsed, 6),
                            "backend": cleanup_debug.get("backend"),
                            "backend_detail": cleanup_debug.get("backend_detail"),
                            "requested_inpaint_mode": inpaint_mode,
                            "effective_inpaint_mode": cleanup_debug.get("effective_inpaint_mode"),
                            "mask_pixels": int((global_stats or {}).get("pixels") or 0),
                            "mask_bbox": list(global_bbox) if global_bbox else None,
                            "mask_bbox_area": _xyxy_area(global_bbox) if global_bbox else 0,
                            "crop_bbox": cleanup_debug.get("crop_bbox"),
                            "crop_area": cleanup_debug.get("crop_area"),
                            "mask_ratio": cleanup_debug.get("mask_ratio"),
                        }
                    operation = _record_cleanup_operation_trace(
                        debug_context,
                        operation,
                        pre_cleanup_image,
                        working,
                        text_mask,
                    )
                    _record_render_cleanup_operation(debug_context, operation)
                    _mark_source_children_covered_by_mask(
                        debug_context,
                        text_mask,
                        "global_fallback",
                        "global_fallback",
                    )
                    required_source_children = set(
                        debug_context.get("source_child_cleanup_required_region_ids") or []
                    ) if debug_context else set()
                    for rid in global_text_mask_region_ids:
                        source_child_fields = {}
                        if rid in required_source_children:
                            covered = _source_child_cleanup_is_covered(debug_context, rid)
                            source_child_fields = {
                                "source_child_cleanup_covered": covered,
                                "source_child_cleanup_partition_id": "global_fallback",
                                "source_child_cleanup_partition_scope": "global_fallback",
                                "source_child_cleanup_pending_parent_cleanup_proof": not covered,
                                "source_child_cleanup_missing_reason": (
                                    None if covered else "final_cleanup_mask_overlap_below_threshold"
                                ),
                            }
                        mark_render_region(
                            debug_context,
                            rid,
                            cleanup_partition_id="global_fallback",
                            cleanup_partition_scope="global_fallback",
                            cleanup_partition_fallback_reason="partitioning_rejected_or_unavailable",
                            **source_child_fields,
                        )
                    _refresh_source_child_cleanup_counts(debug_context)

        if (
            render_regions
            and renderer_cleanup_mutation_enabled
            and cv2 is not None
            and np is not None
            and source_np_for_erasure_proof is not None
        ):
            legacy_proof_render_regions = list(render_regions)
        else:
            legacy_proof_render_regions = []
        if legacy_proof_render_regions and cv2 is not None and np is not None and source_np_for_erasure_proof is not None:
            pre_render_contract_start = time.time()
            working, source_audit_failed_region_ids = _apply_pre_render_source_erasure_contract(
                source_np_for_erasure_proof,
                working,
                legacy_proof_render_regions,
                regions_by_id,
                debug_context,
                perf_telemetry_context,
                inpaint_mode,
                use_gpu,
                model_id,
            )
            _renderer_perf_add_timing(
                debug_context,
                perf_telemetry_context,
                "renderer_pre_render_contract_total_time",
                time.time() - pre_render_contract_start,
            )
            if source_audit_failed_region_ids:
                unsafe_cleanup_region_ids: set[str] = set()
                unsafe_cleanup_decisions = {}
                try:
                    from app.pipeline.render_eligibility import build_unsafe_cleanup_render_decisions

                    region_meta = (
                        debug_context.get("regions", {})
                        if isinstance(debug_context, dict) and isinstance(debug_context.get("regions"), dict)
                        else {}
                    )
                    failed_audit_fields = {
                        str(region_id): dict(region_meta.get(str(region_id), {}) or {})
                        for region_id in source_audit_failed_region_ids
                    }
                    unsafe_cleanup_result = build_unsafe_cleanup_render_decisions(
                        page_id=str((debug_context or {}).get("page_id") or os.path.splitext(os.path.basename(image_path))[0]),
                        regions=regions,
                        region_audit_fields=failed_audit_fields,
                    )
                    unsafe_cleanup_audit = unsafe_cleanup_result.to_audit_dict()
                    if isinstance(debug_context, dict):
                        contracts = debug_context.get("render_eligibility_contracts")
                        if isinstance(contracts, dict):
                            contracts["runtime_unsafe_cleanup_records"] = unsafe_cleanup_audit.get("suppressed_records", [])
                            contracts["runtime_unsafe_cleanup_summary"] = unsafe_cleanup_audit.get("summary", {})
                            if unsafe_cleanup_audit.get("errors"):
                                contracts["runtime_unsafe_cleanup_errors"] = unsafe_cleanup_audit.get("errors")
                    unsafe_cleanup_decisions = _render_eligibility_by_region(unsafe_cleanup_result)
                    unsafe_cleanup_region_ids = {
                        str(region_id)
                        for region_id, decision in unsafe_cleanup_decisions.items()
                        if _render_eligibility_status(decision) == "suppressed_unsafe_cleanup_render"
                    }
                except Exception as exc:
                    if isinstance(debug_context, dict):
                        contracts = debug_context.get("render_eligibility_contracts")
                        if isinstance(contracts, dict):
                            contracts.setdefault("runtime_unsafe_cleanup_errors", []).append(
                                f"{type(exc).__name__}: {exc}"
                            )
                for region_id in source_audit_failed_region_ids:
                    region_id = str(region_id)
                    decision = unsafe_cleanup_decisions.get(region_id)
                    if region_id in unsafe_cleanup_region_ids:
                        reason = _render_eligibility_value(decision, "reason") or "unsafe_cleanup_render_failed_audit_only"
                        _renderer_perf_mark_region(
                            debug_context,
                            perf_telemetry_context,
                            region_id,
                            unsafe_cleanup_render_status="suppressed_unsafe_cleanup_render",
                            unsafe_cleanup_render_failure_reason=reason,
                            render_source_grounded=True,
                            render_suppressed_by_unsafe_cleanup=True,
                            render_blocked_by_visual_contract=True,
                            render_visual_contract_blocker_reason=reason,
                            render_visual_contract_audit_status="suppressed_unsafe_cleanup_render",
                            render_allowed_despite_audit_blocker=False,
                            translated_text_suppressed=True,
                            translated_text_suppressed_reason="unsafe_cleanup_render_failed_audit_only",
                            cleanup_applied=False,
                        )
                    else:
                        mark_render_region(
                            debug_context,
                            region_id,
                            render_allowed_despite_audit_blocker=True,
                            render_suppressed_by_legacy_reason=False,
                        )
                if unsafe_cleanup_region_ids:
                    render_regions = [
                        entry
                        for entry in render_regions
                        if str(entry[0] or "") not in unsafe_cleanup_region_ids
                    ]
            _write_pre_render_source_erasure_debug_image(working, debug_context)

        draw = ImageDraw.Draw(working)
        median_height = 0
        preferred_size = None
        if render_regions:
            heights = sorted(max(1, box[3] - box[1]) for _rid, _text, box, *_ in render_regions)
            median_height = heights[len(heights) // 2]
            preferred_size = max(12, int(median_height * 0.33))
        for (
            region_id,
            text,
            box,
            region_type,
            region_font,
            font_style,
            forced_color,
            stroke_width,
            stroke_color,
            line_height_scale,
            font_size_override,
            wrap_mode,
            source_orientation,
            source_size_hint,
            source_size_min,
            source_size_max,
            parent_style_font_size_locked,
            render_constraint_adjustment,
            render_fit_adjustment,
            parent_execution_layout_authoritative,
            parent_execution_layout_box,
            parent_execution_layout_source,
        ) in render_regions:
            x0, y0, x1, y1 = box
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            vertical_layout = _should_use_vertical_layout(text, w, h, wrap_mode, source_orientation=source_orientation)
            narrow_vertical = vertical_layout and w <= 68 and h >= int(w * 1.25)
            local_preferred = None
            local_min = None
            local_max = None
            if font_size_override > 0:
                local_preferred = font_size_override
                if parent_style_font_size_locked:
                    local_min = local_preferred
                    local_max = local_preferred
                elif source_size_min > 0:
                    local_preferred = max(local_preferred, source_size_min)
                    local_min = source_size_min
                if not parent_style_font_size_locked and source_size_max > 0:
                    local_max = max(local_preferred, source_size_max)
            elif source_size_hint > 0:
                local_preferred = source_size_hint
                if parent_style_font_size_locked:
                    local_min = local_preferred
                    local_max = local_preferred
                else:
                    local_min = source_size_min if source_size_min > 0 else None
                    local_max = source_size_max if source_size_max > 0 else None
            elif (
                preferred_size
                and not parent_execution_layout_authoritative
                and h >= preferred_size * 2
                and not narrow_vertical
            ):
                local_preferred = min(preferred_size, max(12, int(h * 0.7)))
            fill_color = forced_color or _resolve_text_color(working, box)
            if vertical_layout:
                vertical_diag = _draw_vertical_text(
                    draw,
                    text,
                    box,
                    region_font,
                    fill_color,
                    stroke_width,
                    stroke_color,
                    line_height_scale=line_height_scale,
                    preferred_size=local_preferred,
                    min_size=local_min,
                    max_size=local_max,
                    compact_layout=bool((render_fit_adjustment or {}).get("render_fit_compact_layout")),
                )
                if isinstance(vertical_diag, dict):
                    if parent_style_font_size_locked and local_preferred:
                        _mark_parent_style_font_size_result(vertical_diag, local_preferred)
                    if parent_execution_layout_authoritative:
                        vertical_diag.update(
                            {
                                "parent_execution_renderer_layout_authority": "hard_execution_region",
                                "parent_execution_renderer_hard_bbox": list(parent_execution_layout_box or box),
                                "parent_execution_renderer_hard_bbox_source": parent_execution_layout_source,
                                "parent_execution_renderer_font_policy": "fit_complete_text_to_parent_box",
                            }
                        )
                    vertical_diag["text_completeness_passed"] = _wrapped_text_contains_translated_text(
                        text,
                        [str(line) for line in (vertical_diag.get("wrapped_lines") or [])],
                    )
                    if render_constraint_adjustment and render_constraint_adjustment.get("model_fusion_mutation_proof_applied"):
                        vertical_diag.update(_model_fusion_mutation_proof_after_fields(text, vertical_diag.get("wrapped_lines")))
                    mark_render_region(debug_context, region_id, **vertical_diag)
                continue
            base_font = _fit_font(
                draw,
                text,
                w,
                h,
                region_font,
                preferred_size=local_preferred,
                min_size=local_min,
                max_size=local_max,
                line_height_scale=line_height_scale,
            )
            best_lines = _wrap_text(draw, text, base_font, w)
            best_font = base_font
            best_height = _measure_lines_height(base_font, best_lines, line_height_scale)
            max_lines = max(1, min(8, int(h / max(1, int(_text_height(base_font, "A") * line_height_scale))) + 1))
            for lines_count in range(2, max_lines + 1):
                test_lines = _wrap_text(draw, text, base_font, w, max_lines=lines_count)
                test_height = _measure_lines_height(base_font, test_lines, line_height_scale)
                if test_height <= h and len(test_lines) > len(best_lines):
                    best_lines = test_lines
                    best_height = test_height
            if best_height > h and local_preferred:
                for size in range(local_preferred - 1, 9, -1):
                    test_font = _load_font(_find_font_path(region_font), size, _sample_char(text))
                    test_lines = _wrap_text(draw, text, test_font, w, max_lines=max_lines)
                    test_height = _measure_lines_height(test_font, test_lines, line_height_scale)
                    if test_height <= h:
                        best_font = test_font
                        best_lines = test_lines
                        best_height = test_height
                        break
            offset_y = y0 + max(0, (h - best_height) // 2)
            for line in best_lines:
                bbox = best_font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                offset_x = x0 + max(0, (w - line_width) // 2) - bbox[0]
                draw.text(
                    (offset_x, offset_y - bbox[1]),
                    line,
                    fill=fill_color,
                    font=best_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                )
                offset_y += max(1, int(line_height * line_height_scale))
            mark_render_region(
                debug_context,
                region_id,
                selected_font_size=getattr(best_font, "size", None),
                wrapped_lines=best_lines,
                text_completeness_passed=_wrapped_text_contains_translated_text(text, best_lines),
                measured_rendered_width=max((_text_width(best_font, line) for line in best_lines), default=0),
                measured_rendered_height=best_height,
                fit_ratio=max(
                    max((_text_width(best_font, line) for line in best_lines), default=0) / max(1, w),
                    best_height / max(1, h),
                ),
                **(
                    _parent_style_font_size_result_fields(
                        getattr(best_font, "size", None),
                        local_preferred,
                    )
                    if parent_style_font_size_locked and local_preferred
                    else {}
                ),
                **(
                    {
                        "parent_execution_renderer_layout_authority": "hard_execution_region",
                        "parent_execution_renderer_hard_bbox": list(parent_execution_layout_box or box),
                        "parent_execution_renderer_hard_bbox_source": parent_execution_layout_source,
                        "parent_execution_renderer_font_policy": "fit_complete_text_to_parent_box",
                    }
                    if parent_execution_layout_authoritative
                    else {}
                ),
                **(
                    _model_fusion_mutation_proof_after_fields(text, best_lines)
                    if render_constraint_adjustment and render_constraint_adjustment.get("model_fusion_mutation_proof_applied")
                    else {}
                ),
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save with high quality to prevent JPEG compression artifacts
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            working.save(output_path, quality=95, optimize=True)
        else:
            working.save(output_path)

_RENDER_CONSTRAINT_ASSIST_VERSION = "text_area_render_constraint_assist_legacy_quarantined_v1"
_RENDER_CONSTRAINT_ASSIST_STATUS = "experimental_opt_in_allowlisted"
_MODEL_FUSION_MUTATION_PROOF_VERSION = "model_fusion_mutation_proof_phase4b7_v1"
_MODEL_FUSION_MUTATION_PROOF_FLAG = "MT_MODEL_FUSION_MUTATION_PROOF"

_PHASE3C_RENDER_CONSTRAINT_ALLOWLIST: tuple[dict[str, object], ...] = ()
_MODEL_FUSION_MUTATION_PROOF_ALLOWLIST: tuple[dict[str, object], ...] = ()


def _build_model_fusion_mutation_proof_candidates(
    image_path: str,
    regions: List[Dict[str, object]],
    debug_context: dict | None,
    img_w: int,
    img_h: int,
) -> dict[str, dict[str, object]]:
    if not _model_fusion_mutation_proof_enabled():
        return {}
    page_id = str((debug_context or {}).get("page_id") or "").strip()
    if not page_id:
        page_id = os.path.splitext(os.path.basename(str(image_path or "")))[0]
    page_digits = "".join(ch for ch in page_id if ch.isdigit()).zfill(3)
    allowed_entries = [
        entry for entry in _MODEL_FUSION_MUTATION_PROOF_ALLOWLIST
        if page_digits == str(entry.get("page_id") or "").zfill(3)
    ]
    if not allowed_entries:
        return {}
    if cv2 is None or np is None:
        return {}

    try:
        from app.pipeline.bubble_detection import BubbleDetectionInput, run_bubble_detection

        service_result = run_bubble_detection(
            BubbleDetectionInput(
                page_id=page_digits,
                image_path=image_path,
                image_size=(img_w, img_h),
                regions=regions,
                mode="model_fusion_render_constraint_proof",
            )
        )
        if not service_result.generated:
            if service_result.error:
                print(f"[ModelFusionProof] bubble detection service failed closed: {service_result.error}")
            return {}
    except Exception as exc:  # pragma: no cover - proof path must fail closed
        print(f"[ModelFusionProof] failed to build model-fusion proof candidates: {exc}")
        return {}

    candidates: dict[str, dict[str, object]] = {}
    for region in regions:
        region_id = str(region.get("region_id", "") or "")
        source_box = _region_source_box_xyxy(region)
        if not region_id or not source_box:
            continue
        for entry in allowed_entries:
            if not _region_matches_render_constraint_allowlist_entry(region, region_id, debug_context, image_path, source_box, entry):
                continue
            model_candidate = _model_fusion_candidate_for_region(
                page_id=page_digits,
                region_id=region_id,
                source_box=source_box,
                entry=entry,
                service_result=service_result,
                img_w=img_w,
                img_h=img_h,
            )
            if model_candidate:
                candidates[region_id] = model_candidate
            break
    return candidates


def _model_fusion_candidate_for_region(
    *,
    page_id: str,
    region_id: str,
    source_box,
    entry: dict[str, object],
    service_result,
    img_w: int,
    img_h: int,
) -> dict[str, object] | None:
    region_link = None
    for link in getattr(service_result, "region_model_links", []) or []:
        if str(link.get("region_id") or "") == str(region_id or ""):
            region_link = link
            break
    if not region_link:
        return None

    mask_evidence_by_id = {
        str(evidence.get("model_evidence_id") or evidence.get("evidence_id") or ""): evidence
        for evidence in getattr(service_result, "bubble_model_evidence", []) or []
    }
    text_evidence_by_id = {
        str(evidence.get("model_evidence_id") or evidence.get("evidence_id") or ""): evidence
        for evidence in getattr(service_result, "text_area_model_evidence", []) or []
    }

    best_link = None
    best_score = -1.0
    for link in region_link.get("kitsumed_links", []) or []:
        mask_id = str(link.get("evidence_id") or "")
        evidence = mask_evidence_by_id.get(mask_id)
        if not evidence:
            continue
        ratio = float(link.get("mask_overlap_ratio") or 0.0)
        center_hit = bool(link.get("center_inside_mask"))
        confidence = float(link.get("confidence") or evidence.get("confidence") or 0.0)
        if confidence < 0.70 or (ratio < 0.05 and not center_hit):
            continue
        score = confidence + ratio + (0.15 if center_hit else 0.0)
        if score > best_score:
            best_score = score
            best_link = {
                "evidence_id": mask_id,
                "evidence": evidence,
                "confidence": confidence,
                "mask_overlap_ratio": ratio,
                "center_inside_mask": center_hit,
            }
    if not best_link:
        return None

    support_ids = []
    best_mask_id = str(best_link.get("evidence_id") or "")
    for detection in text_evidence_by_id.values():
        class_name = str(detection.get("class_name") or "")
        if class_name not in {"bubble", "text_bubble"}:
            continue
        linked_masks = [str(item) for item in detection.get("linked_bubble_mask_ids", []) or []]
        if best_mask_id in linked_masks:
            support_ids.append(str(detection.get("model_evidence_id") or detection.get("evidence_id") or ""))

    best_evidence = best_link["evidence"]
    allowed_area = _clamped_xyxy(
        best_evidence.get("mask_bbox_xyxy") or best_evidence.get("mask_bbox"),
        img_w,
        img_h,
    )
    if not allowed_area:
        return None
    inferred_id = f"model_fusion:{page_id}:{best_mask_id}"
    reason_codes = [
        "model_fusion_candidate_class:safe_future_render_constraint_hint",
        "kitsumed_mask_primary_geometry",
        "bubble_detection_service_output",
        f"kitsumed_confidence:{float(best_link.get('confidence') or 0.0):.3f}",
        f"kitsumed_source_overlap_ratio:{float(best_link.get('mask_overlap_ratio') or 0.0):.3f}",
    ]
    if support_ids:
        reason_codes.append("ogkalu_bubble_or_text_bubble_support")
    return {
        "candidate_id": f"phase4b7:{page_id}:{region_id}:{best_mask_id}",
        "model_fusion_candidate_id": f"{page_id}:model_fusion_render_constraint:{region_id}",
        "allowlist_id": entry["allowlist_id"],
        "region_id": region_id,
        "candidate_class": entry["candidate_class"],
        "suggestion_type": entry["suggestion_type"],
        "confidence": entry["confidence"],
        "severity": entry["severity"],
        "proposed_action": entry["proposed_action"],
        "container_type": entry["container_type"],
        "proposed_allowed_area": list(allowed_area),
        "inferred_container_id": inferred_id,
        "inferred_container_bbox": list(allowed_area),
        "linked_kitsumed_mask_ids": [best_mask_id],
        "linked_ogkalu_detection_ids": support_ids,
        "reason_codes": reason_codes,
        "planner_suggestion_id": "model_fusion_phase4b7_render_constraint_hint",
    }


def _apply_render_constraint_assist(
    render_box,
    source_box,
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    img_w: int,
    img_h: int,
    preserved_text_obstacles: list[dict[str, object]],
    model_fusion_proof_candidates: dict[str, dict[str, object]] | None = None,
):
    if not _render_constraint_assist_enabled() and not _model_fusion_mutation_proof_enabled():
        return render_box, None
    candidates = build_render_constraint_assist_candidates(
        render_box,
        source_box,
        region,
        region_id,
        debug_context,
        image_path,
        img_w,
        img_h,
        model_fusion_proof_candidates=model_fusion_proof_candidates,
    )
    proof_candidate_seen = None
    for candidate in candidates:
        if candidate.get("evidence_source") == "model_fusion_assist_proof":
            proof_candidate_seen = candidate
        if not is_render_constraint_assist_eligible(
            candidate,
            region,
            source_box,
            region_id,
            preserved_text_obstacles,
        ):
            continue
        return apply_render_constraint_candidate(candidate)
    if proof_candidate_seen is not None:
        return render_box, _model_fusion_mutation_proof_fallback_adjustment(
            proof_candidate_seen,
            render_box,
            "candidate_failed_phase3_render_constraint_eligibility",
        )
    return render_box, None


def build_render_constraint_assist_candidates(
    render_box,
    source_box,
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    img_w: int,
    img_h: int,
    model_fusion_proof_candidates: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if _render_constraint_assist_enabled():
        candidates.extend(
            _render_constraint_candidates_from_planner_metadata(
                render_box,
                source_box,
                region,
                region_id,
                debug_context,
                image_path,
                img_w,
                img_h,
            )
        )
        candidates.extend(
            _render_constraint_candidates_from_allowlist(
                render_box,
                source_box,
                region,
                region_id,
                debug_context,
                image_path,
                img_w,
                img_h,
            )
        )
    candidates.extend(
        _render_constraint_candidates_from_model_fusion_proof(
            render_box,
            source_box,
            region_id,
            img_w,
            img_h,
            model_fusion_proof_candidates or {},
        )
    )
    return candidates


def _render_constraint_candidates_from_planner_metadata(
    render_box,
    source_box,
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    img_w: int,
    img_h: int,
) -> list[dict[str, object]]:
    render = region.get("render") or {}
    if not isinstance(render, dict):
        render = {}
    raw_suggestions = []
    for value in (
        region.get("diagnostic_render_plan_suggestions"),
        render.get("diagnostic_render_plan_suggestions"),
        region.get("render_plan_suggestions"),
        render.get("render_plan_suggestions"),
    ):
        if isinstance(value, list):
            raw_suggestions.extend(item for item in value if isinstance(item, dict))
    candidates = []
    for suggestion in raw_suggestions:
        allowed = _clamped_xyxy(suggestion.get("proposed_allowed_area"), img_w, img_h)
        if not allowed:
            continue
        candidate = _normalize_render_constraint_candidate(
            suggestion,
            render_box,
            source_box,
            region,
            region_id,
            evidence_source="planner",
            allowed_area=allowed,
        )
        allowlist_entry = _matching_render_constraint_allowlist_entry(
            region,
            region_id,
            debug_context,
            image_path,
            source_box,
        )
        if allowlist_entry:
            candidate["allowlist_id"] = allowlist_entry["allowlist_id"]
            candidate["candidate_id"] = allowlist_entry["candidate_id"]
        candidates.append(candidate)
    return candidates


def _render_constraint_candidates_from_allowlist(
    render_box,
    source_box,
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    img_w: int,
    img_h: int,
) -> list[dict[str, object]]:
    entry = _matching_render_constraint_allowlist_entry(
        region,
        region_id,
        debug_context,
        image_path,
        source_box,
    )
    if not entry:
        return []
    allowed_area = _clamped_xyxy(entry["proposed_allowed_area"], img_w, img_h)
    if not allowed_area:
        return []
    previous = tuple(int(v) for v in render_box[:4])
    source = tuple(int(v) for v in source_box[:4])
    outside_ratio = _outside_ratio(previous, allowed_area)
    source_inside_ratio = _inside_ratio(source, allowed_area)
    reason_codes = [
        "legacy_render_constraint_allowlist",
        f"allowlist:{entry['allowlist_id']}",
        f"page:{entry['page_id']}",
        "planner_reconstructed_from_phase3a_reference",
        "final_render_bbox_outside_inferred_container",
        "source_bbox_inside_inferred_container",
        f"outside_allowed_ratio:{outside_ratio:.3f}",
        f"source_inside_ratio:{source_inside_ratio:.3f}",
    ]
    return [
        {
            "candidate_id": entry["candidate_id"],
            "allowlist_id": entry["allowlist_id"],
            "region_id": str(region_id or ""),
            "suggestion_type": entry["suggestion_type"],
            "confidence": entry["confidence"],
            "severity": entry["severity"],
            "proposed_action": entry["proposed_action"],
            "current_render_bbox": list(previous),
            "source_bbox": list(source),
            "proposed_allowed_area": list(allowed_area),
            "inferred_container_id": entry["inferred_container_id"],
            "inferred_container_bbox": list(allowed_area),
            "container_type": entry["container_type"],
            "reason_codes": reason_codes,
            "evidence_source": "target_allowlist",
            "would_change_behavior": True,
            "phase3_status": _RENDER_CONSTRAINT_ASSIST_STATUS,
            "planner_suggestion_id": entry["planner_suggestion_id"],
            "outside_allowed_ratio": outside_ratio,
            "source_inside_ratio": source_inside_ratio,
        }
    ]


def _render_constraint_candidates_from_model_fusion_proof(
    render_box,
    source_box,
    region_id: str,
    img_w: int,
    img_h: int,
    model_fusion_proof_candidates: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if not _model_fusion_mutation_proof_enabled():
        return []
    proof = model_fusion_proof_candidates.get(str(region_id or ""))
    if not proof:
        return []
    allowed_area = _clamped_xyxy(proof.get("proposed_allowed_area"), img_w, img_h)
    if not allowed_area:
        return []
    previous = tuple(int(v) for v in render_box[:4])
    source = tuple(int(v) for v in source_box[:4])
    outside_ratio = _outside_ratio(previous, allowed_area)
    source_inside_ratio = _inside_ratio(source, allowed_area)
    reason_codes = [str(item) for item in (proof.get("reason_codes") or []) if str(item)]
    reason_codes.extend(
        [
            "model_fusion_mutation_proof",
            f"allowlist:{proof.get('allowlist_id')}",
            "final_render_bbox_outside_inferred_container",
            "source_bbox_inside_model_fusion_container",
            f"outside_allowed_ratio:{outside_ratio:.3f}",
            f"source_inside_ratio:{source_inside_ratio:.3f}",
        ]
    )
    return [
        {
            "candidate_id": proof.get("candidate_id"),
            "allowlist_id": proof.get("allowlist_id"),
            "region_id": str(region_id or ""),
            "suggestion_type": proof.get("suggestion_type"),
            "confidence": proof.get("confidence"),
            "severity": proof.get("severity"),
            "proposed_action": proof.get("proposed_action"),
            "current_render_bbox": list(previous),
            "source_bbox": list(source),
            "proposed_allowed_area": list(allowed_area),
            "inferred_container_id": proof.get("inferred_container_id"),
            "inferred_container_bbox": proof.get("inferred_container_bbox") or list(allowed_area),
            "container_type": proof.get("container_type"),
            "reason_codes": sorted(set(reason_codes)),
            "evidence_source": "model_fusion_assist_proof",
            "would_change_behavior": True,
            "phase3_status": "model_fusion_mutation_proof_allowlisted",
            "planner_suggestion_id": proof.get("planner_suggestion_id"),
            "outside_allowed_ratio": outside_ratio,
            "source_inside_ratio": source_inside_ratio,
            "model_fusion_candidate_id": proof.get("model_fusion_candidate_id"),
            "model_fusion_candidate_class": proof.get("candidate_class"),
        }
    ]


def is_render_constraint_assist_eligible(
    candidate: dict[str, object],
    region: dict,
    source_box,
    region_id: str,
    preserved_text_obstacles: list[dict[str, object]],
) -> bool:
    if str(region.get("type", "") or "") != "speech_bubble":
        return False
    if str(candidate.get("suggestion_type") or "") != "speech_render_outside_container":
        return False
    if str(candidate.get("confidence") or "") != "high":
        return False
    if str(candidate.get("severity") or "") not in {"serious", "blocker"}:
        return False
    if str(candidate.get("proposed_action") or "") != "clamp_to_container":
        return False
    if str(candidate.get("container_type") or "") != "speech_bubble":
        return False
    if not candidate.get("allowlist_id"):
        return False
    allowed_area = _clamped_xyxy(candidate.get("proposed_allowed_area"), 10**9, 10**9)
    current_box = _clamped_xyxy(candidate.get("current_render_bbox"), 10**9, 10**9)
    source = tuple(int(v) for v in source_box[:4])
    if not allowed_area or not current_box:
        return False
    outside_ratio = float(candidate.get("outside_allowed_ratio") or _outside_ratio(current_box, allowed_area))
    source_inside_ratio = float(candidate.get("source_inside_ratio") or _inside_ratio(source, allowed_area))
    if outside_ratio < 0.20:
        return False
    if source_inside_ratio < 0.80:
        return False
    if _render_constraint_candidate_introduces_preserved_obstacle_conflict(
        allowed_area,
        source,
        preserved_text_obstacles,
        region_id,
    ):
        return False
    return True


def apply_render_constraint_candidate(candidate: dict[str, object]):
    previous = tuple(int(v) for v in candidate.get("current_render_bbox") or ())
    allowed_area = tuple(int(v) for v in candidate.get("proposed_allowed_area") or ())
    if len(previous) != 4 or len(allowed_area) != 4:
        return previous, None
    adjustment = {
        "render_constraint_applied": True,
        "render_constraint_version": _RENDER_CONSTRAINT_ASSIST_VERSION,
        "render_constraint_source": candidate.get("evidence_source"),
        "render_constraint_candidate_id": candidate.get("candidate_id"),
        "render_constraint_scope": f"allowlist:{candidate.get('allowlist_id')}",
        "render_constraint_previous_final_render_bbox": list(previous),
        "render_constraint_new_final_render_bbox": list(allowed_area),
        "render_constraint_planner_suggestion_id": candidate.get("planner_suggestion_id"),
        "render_constraint_planner_suggestion_type": candidate.get("suggestion_type"),
        "render_constraint_reason_codes": candidate.get("reason_codes") or [],
        "render_constraint_inferred_container_id": candidate.get("inferred_container_id"),
        "render_constraint_inferred_container_bbox": candidate.get("inferred_container_bbox"),
        "render_constraint_proposed_action": candidate.get("proposed_action"),
        "render_constraint_font_wrap_recomputed": True,
        "render_constraint_text_completeness_policy": "no_silent_truncation",
        "render_constraint_previous_outside_container_ratio": round(float(candidate.get("outside_allowed_ratio") or 0.0), 3),
    }
    if candidate.get("evidence_source") == "model_fusion_assist_proof":
        adjustment.update(
            {
                "model_fusion_mutation_proof_enabled": True,
                "model_fusion_mutation_proof_applied": True,
                "model_fusion_mutation_proof_version": _MODEL_FUSION_MUTATION_PROOF_VERSION,
                "model_fusion_mutation_proof_candidate_id": candidate.get("model_fusion_candidate_id") or candidate.get("candidate_id"),
                "model_fusion_source_container_id": candidate.get("inferred_container_id"),
                "previous_final_render_bbox": list(previous),
                "new_final_render_bbox": list(allowed_area),
                "fallback_reason": None,
                "would_change_behavior": True,
            }
        )
    return allowed_area, adjustment


def _render_constraint_assist_enabled() -> bool:
    value = os.getenv("MT_TEXT_AREA_RENDER_CONSTRAINTS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _model_fusion_mutation_proof_enabled() -> bool:
    legacy = os.getenv("MT_LEGACY_PAGE_SPECIFIC_ASSIST", "").strip().lower() in {"1", "true", "yes", "on"}
    assist = os.getenv("MT_MODEL_FUSION_ASSIST", "").strip().lower() in {"1", "true", "yes", "on"}
    proof = os.getenv(_MODEL_FUSION_MUTATION_PROOF_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return legacy and assist and proof


def _model_fusion_mutation_proof_fallback_adjustment(
    candidate: dict[str, object],
    render_box,
    reason: str,
) -> dict[str, object]:
    previous = list(tuple(int(v) for v in render_box[:4]))
    return {
        "model_fusion_mutation_proof_enabled": True,
        "model_fusion_mutation_proof_applied": False,
        "model_fusion_mutation_proof_version": _MODEL_FUSION_MUTATION_PROOF_VERSION,
        "model_fusion_mutation_proof_candidate_id": candidate.get("model_fusion_candidate_id") or candidate.get("candidate_id"),
        "model_fusion_source_container_id": candidate.get("inferred_container_id"),
        "previous_final_render_bbox": previous,
        "new_final_render_bbox": previous,
        "translated_text": None,
        "wrapped_lines_before": None,
        "wrapped_lines_after": None,
        "text_completeness_passed": False,
        "fallback_reason": reason,
        "would_change_behavior": False,
    }


def _preview_wrapped_lines_for_box(
    text: str,
    box,
    font_name: str,
    wrap_mode: str,
    source_orientation: str,
    line_height_scale: float,
    preferred_size: int | None,
    min_size: int | None,
    max_size: int | None,
) -> list[str] | None:
    if Image is None or ImageDraw is None:
        return None
    clamped = _clamped_xyxy(box, 10**9, 10**9)
    if not clamped:
        return None
    x0, y0, x1, y1 = clamped
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    try:
        if _should_use_vertical_layout(text, width, height, wrap_mode, source_orientation=source_orientation):
            tokens = _vertical_tokens(text)
            if not tokens:
                return []
            font, layout = _fit_vertical_font(
                text,
                width,
                height,
                font_name,
                preferred_size=preferred_size,
                min_size=min_size,
                max_size=max_size,
                line_height_scale=line_height_scale,
            )
            rows, cols, _cell_height, _col_width, _col_gap = layout
            return ["".join(tokens[col * rows : (col + 1) * rows]) for col in range(cols)]
        scratch = Image.new("RGB", (8, 8), "white")
        draw = ImageDraw.Draw(scratch)
        font = _fit_font(
            draw,
            text,
            width,
            height,
            font_name,
            preferred_size=preferred_size,
            line_height_scale=line_height_scale,
        )
        lines = _wrap_text(draw, text, font, width)
        best_height = _measure_lines_height(font, lines, line_height_scale)
        max_lines = max(1, min(8, int(height / max(1, int(_text_height(font, "A") * line_height_scale))) + 1))
        for lines_count in range(2, max_lines + 1):
            test_lines = _wrap_text(draw, text, font, width, max_lines=lines_count)
            test_height = _measure_lines_height(font, test_lines, line_height_scale)
            if test_height <= height and len(test_lines) > len(lines):
                lines = test_lines
                best_height = test_height
        return [str(line) for line in lines]
    except Exception:
        return None


def _model_fusion_mutation_proof_after_fields(text: str, wrapped_lines) -> dict[str, object]:
    if isinstance(wrapped_lines, str):
        wrapped = [wrapped_lines]
    elif isinstance(wrapped_lines, list):
        wrapped = [str(item) for item in wrapped_lines]
    elif isinstance(wrapped_lines, tuple):
        wrapped = [str(item) for item in wrapped_lines]
    else:
        wrapped = []
    return {
        "wrapped_lines_after": wrapped,
        "text_completeness_passed": _wrapped_text_contains_translated_text(text, wrapped),
    }


def _wrapped_text_contains_translated_text(text: str, wrapped_lines: list[str]) -> bool:
    translated = _meaningful_render_chars(text)
    rendered = _meaningful_render_chars("".join(wrapped_lines))
    if not translated:
        return True
    if not rendered:
        return False
    from collections import Counter

    translated_counts = Counter(translated)
    rendered_counts = Counter(rendered)
    for char, count in translated_counts.items():
        if rendered_counts[char] < count:
            return False
    return True


def _meaningful_render_chars(text: str) -> list[str]:
    chars = []
    for char in str(text or ""):
        if char.isspace() or _is_render_punctuation(char):
            continue
        chars.append(char)
    return chars


def _is_render_punctuation(char: str) -> bool:
    code = ord(char)
    return (
        0x2000 <= code <= 0x206F
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFF65
        or char in set(".,!?;:'\"()[]{}<>/\\|-_~`")
    )


def _normalize_render_constraint_candidate(
    suggestion: dict[str, object],
    render_box,
    source_box,
    region: dict,
    region_id: str,
    *,
    evidence_source: str,
    allowed_area,
) -> dict[str, object]:
    previous = tuple(int(v) for v in render_box[:4])
    source = tuple(int(v) for v in source_box[:4])
    outside_ratio = _outside_ratio(previous, allowed_area)
    source_inside_ratio = _inside_ratio(source, allowed_area)
    reason_codes = [str(item) for item in (suggestion.get("reason_codes") or []) if str(item)]
    if not any(str(item).startswith("outside_allowed_ratio:") for item in reason_codes):
        reason_codes.append(f"outside_allowed_ratio:{outside_ratio:.3f}")
    if not any(str(item).startswith("source_inside_ratio:") for item in reason_codes):
        reason_codes.append(f"source_inside_ratio:{source_inside_ratio:.3f}")
    return {
        "candidate_id": str(suggestion.get("suggestion_id") or f"planner:{region_id}"),
        "region_id": str(region_id or suggestion.get("region_id") or ""),
        "suggestion_type": str(suggestion.get("suggestion_type") or ""),
        "confidence": str(suggestion.get("confidence") or ""),
        "severity": str(suggestion.get("severity") or ""),
        "proposed_action": str(suggestion.get("proposed_action") or ""),
        "current_render_bbox": list(previous),
        "source_bbox": list(source),
        "proposed_allowed_area": list(allowed_area),
        "inferred_container_id": suggestion.get("inferred_container_id"),
        "inferred_container_bbox": suggestion.get("inferred_container_bbox") or list(allowed_area),
        "container_type": suggestion.get("container_type"),
        "reason_codes": reason_codes,
        "evidence_source": evidence_source,
        "would_change_behavior": True,
        "phase3_status": _RENDER_CONSTRAINT_ASSIST_STATUS,
        "planner_suggestion_id": suggestion.get("suggestion_id"),
        "outside_allowed_ratio": outside_ratio,
        "source_inside_ratio": source_inside_ratio,
    }


def _matching_render_constraint_allowlist_entry(
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    source_box,
) -> dict[str, object] | None:
    for entry in _PHASE3C_RENDER_CONSTRAINT_ALLOWLIST:
        if _region_matches_render_constraint_allowlist_entry(region, region_id, debug_context, image_path, source_box, entry):
            return dict(entry)
    return None


def _region_matches_render_constraint_allowlist_entry(
    region: dict,
    region_id: str,
    debug_context: dict | None,
    image_path: str,
    source_box,
    entry: dict[str, object],
) -> bool:
    page_id = str((debug_context or {}).get("page_id") or "").strip()
    if not page_id:
        page_id = os.path.splitext(os.path.basename(str(image_path or "")))[0]
    digits = "".join(ch for ch in page_id if ch.isdigit())
    if digits.zfill(3) != str(entry.get("page_id") or "").zfill(3):
        return False
    region_ids = entry.get("region_ids") or set()
    if region_ids and region_id not in region_ids:
        ocr_text = str(region.get("ocr_text", "") or "")
        if not all(token in ocr_text for token in (entry.get("ocr_contains") or ())):
            return False
    ocr_text = str(region.get("ocr_text", "") or "")
    if not all(token in ocr_text for token in (entry.get("ocr_contains") or ())):
        return False
    expected_source = entry.get("source_bbox_xyxy")
    tolerance = int(entry.get("source_bbox_tolerance", 0) or 0)
    if expected_source and tolerance >= 0:
        try:
            sx0, sy0, sx1, sy1 = [int(v) for v in source_box[:4]]
            ex0, ey0, ex1, ey1 = [int(v) for v in expected_source[:4]]
        except Exception:
            return False
        if max(abs(sx0 - ex0), abs(sy0 - ey0), abs(sx1 - ex1), abs(sy1 - ey1)) > tolerance:
            return False
    return True


def _render_constraint_candidate_introduces_preserved_obstacle_conflict(
    allowed_area,
    source_box,
    obstacles: list[dict[str, object]],
    current_region_id: str,
) -> bool:
    for obstacle in obstacles or []:
        obstacle_id = str(obstacle.get("region_id", "") or "")
        if obstacle_id and obstacle_id == current_region_id:
            continue
        obstacle_box = obstacle.get("box")
        if not obstacle_box:
            continue
        if not _substantial_xyxy_overlap(allowed_area, obstacle_box):
            continue
        if _substantial_xyxy_overlap(source_box, obstacle_box):
            continue
        return True
    return False


def _clamped_xyxy(value, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x0 = max(0, min(int(img_w), x0))
    y0 = max(0, min(int(img_h), y0))
    x1 = max(0, min(int(img_w), x1))
    y1 = max(0, min(int(img_h), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _outside_ratio(box, allowed_area) -> float:
    area = max(1, _xyxy_area(box))
    inside = _xyxy_intersection_area(box, allowed_area)
    return max(0.0, min(1.0, 1.0 - (inside / area)))


def _inside_ratio(box, allowed_area) -> float:
    area = max(1, _xyxy_area(box))
    inside = _xyxy_intersection_area(box, allowed_area)
    return max(0.0, min(1.0, inside / area))


def _cleanup_partition_candidate(region: dict, render: dict, region_id: str, mask) -> dict[str, object]:
    meta = _cleanup_partition_metadata_for_region(region, render=render)
    key = str(meta.get("cleanup_partition_key") or "")
    if not key:
        key = f"global_fallback:{region_id or 'unknown'}"
        meta["cleanup_partition_scope"] = "global_fallback"
        meta["fallback_reason"] = "missing_parent_root_container_metadata"
    return {
        "region_id": region_id,
        "region_ids": [region_id] if region_id else [],
        "mask": mask,
        **meta,
        "cleanup_partition_key": key,
    }


def _add_represented_child_cleanup_candidates(
    regions: list[dict],
    img_np,
    text_mask,
    cleanup_candidates: list[dict[str, object]],
    cleanup_region_ids: list[str],
    source_glyph_masks_by_region: dict[str, object],
    debug_context: dict | None,
    img_w: int,
    img_h: int,
) -> None:
    if cv2 is None or np is None or img_np is None or text_mask is None:
        return
    active_parent_ids = _active_translation_parent_ids(regions)
    parent_child_ids: dict[str, list[str]] = {}
    records: list[dict[str, object]] = []
    required = 0
    generated = 0
    missing = 0
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("region_id", "") or "")
        if not region_id:
            continue
        should_cleanup, parent_id, reason = _source_child_cleanup_requirement(region, active_parent_ids)
        if not should_cleanup:
            continue
        required += 1
        source_glyph_lookup_start = time.time()
        source_glyph_record = source_glyph_masks_by_region.get(region_id)
        add_timing(
            debug_context,
            "renderer_source_glyph_record_lookup_time",
            time.time() - source_glyph_lookup_start,
        )
        mask, method, missing_reason = _represented_child_cleanup_mask(
            region,
            source_glyph_record,
            img_np,
            img_w,
            img_h,
            debug_context,
        )
        if mask is None or not np.any(mask):
            missing += 1
            reason_text = missing_reason or "source_child_mask_unavailable"
            mark_render_region(
                debug_context,
                region_id,
                source_child_cleanup_required=True,
                source_child_cleanup_covered=False,
                source_child_cleanup_missing_reason=reason_text,
                represented_child_cleanup_proof_status="mask_missing",
            )
            records.append(
                {
                    "region_id": region_id,
                    "parent_logical_text_unit_id": parent_id,
                    "required": True,
                    "covered": False,
                    "generation_method": method,
                    "missing_reason": reason_text,
                    "represented_child_cleanup_proof_status": "mask_missing",
                }
            )
            continue
        text_mask[:] = cv2.bitwise_or(text_mask, mask.astype(np.uint8))
        if region_id not in cleanup_region_ids:
            cleanup_region_ids.append(region_id)
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        candidate = _cleanup_partition_candidate(region, render, region_id, mask.copy())
        candidate["source_child_region_ids"] = [region_id]
        candidate["source_child_cleanup_reason"] = reason
        cleanup_candidates.append(candidate)
        generated += 1
        parent_child_ids.setdefault(parent_id or "", []).append(region_id)
        stats = mask_stats(mask)
        mark_render_region(
            debug_context,
            region_id,
            cleanup_applied=True,
            cleanup_mode="represented_child_source_cleanup",
            cleanup_mask=stats,
            source_child_cleanup_required=True,
            source_child_cleanup_covered=False,
            source_child_cleanup_pending_parent_cleanup_proof=True,
            source_child_cleanup_partition_id="pending_parent_root_partition",
            source_child_cleanup_missing_reason=None,
            source_child_cleanup_generation_method=method,
            source_child_cleanup_reason=reason,
            represented_child_cleanup_proof_status="pending_parent_cleanup_proof",
        )
        if debug_context is not None:
            debug_context.setdefault("_source_child_cleanup_masks", {})[region_id] = mask.copy()
        records.append(
            {
                "region_id": region_id,
                "parent_logical_text_unit_id": parent_id,
                "required": True,
                "covered": False,
                "pending_parent_cleanup_proof": True,
                "generation_method": method,
                "mask_pixels": int((stats or {}).get("pixels") or 0),
                "mask_bbox": (stats or {}).get("bbox"),
                "reason": reason,
                "represented_child_cleanup_proof_status": "pending_parent_cleanup_proof",
            }
        )
    for parent_id, child_ids in parent_child_ids.items():
        if not parent_id or not child_ids:
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            if not _is_active_translation_region(region):
                continue
            if parent_id not in _region_parent_ids(region):
                continue
            parent_region_id = str(region.get("region_id", "") or "")
            if parent_region_id:
                mark_render_region(
                    debug_context,
                    parent_region_id,
                    parent_cleanup_child_region_ids=child_ids,
                )
            break
    if debug_context is not None:
        debug_context["source_child_cleanup_records"] = records
        debug_context["uncleaned_source_child_count"] = missing
        debug_context["source_child_cleanup_required_region_ids"] = [
            str(record.get("region_id") or "")
            for record in records
            if record.get("required") and record.get("region_id")
        ]
        debug_context.setdefault("counts", {})["source_child_cleanup_required"] = required
        debug_context.setdefault("counts", {})["source_child_cleanup_candidate_generated"] = generated
        debug_context.setdefault("counts", {})["source_child_cleanup_covered"] = 0
        debug_context.setdefault("counts", {})["uncleaned_source_child_count"] = missing


def _mark_source_children_covered_by_mask(
    debug_context: dict | None,
    cleanup_mask,
    partition_id: str,
    partition_scope: str,
) -> None:
    if not debug_context or cleanup_mask is None or cv2 is None or np is None:
        return
    child_masks = debug_context.get("_source_child_cleanup_masks") or {}
    if not isinstance(child_masks, dict):
        return
    for region_id, child_mask in child_masks.items():
        if child_mask is None or getattr(child_mask, "shape", None) != getattr(cleanup_mask, "shape", None):
            continue
        child_pixels = int((child_mask > 0).sum())
        if child_pixels <= 0:
            continue
        child_mask_bool = child_mask > 0
        operation_overlap = int((child_mask_bool & (cleanup_mask > 0)).sum())
        accumulated_masks = debug_context.setdefault("_source_child_cleanup_accumulated_masks", {})
        prior = accumulated_masks.get(str(region_id))
        current_overlap_mask = (child_mask_bool & (cleanup_mask > 0)).astype(np.uint8) * 255
        if prior is not None and getattr(prior, "shape", None) == getattr(cleanup_mask, "shape", None):
            accumulated = cv2.bitwise_or(prior.astype(np.uint8), current_overlap_mask)
        else:
            accumulated = current_overlap_mask
        accumulated_masks[str(region_id)] = accumulated
        overlap = int((accumulated > 0).sum())
        overlap_ratio = overlap / max(1, child_pixels)
        entry = debug_context.setdefault("regions", {}).setdefault(str(region_id), {})
        already_covered = entry.get("source_child_cleanup_covered") is True
        existing_ratio = _float_or_zero(entry.get("source_child_cleanup_coverage_ratio"))
        if overlap_ratio < 0.90:
            if not already_covered and overlap_ratio >= existing_ratio:
                mark_render_region(
                    debug_context,
                    str(region_id),
                    source_child_cleanup_covered=False,
                    source_child_cleanup_pending_parent_cleanup_proof=True,
                    source_child_cleanup_coverage_ratio=round(float(overlap_ratio), 3),
                    source_child_cleanup_operation_overlap_ratio=round(float(operation_overlap / max(1, child_pixels)), 3),
                    source_child_cleanup_partition_id=partition_id,
                    source_child_cleanup_partition_scope=partition_scope,
                    source_child_cleanup_missing_reason="final_cleanup_mask_overlap_below_threshold",
                    represented_child_cleanup_proof_status="failed",
                    represented_child_cleanup_required=True,
                    represented_child_cleanup_mask_consumed=True,
                    represented_child_cleanup_coverage_ratio=round(float(overlap_ratio), 3),
                    represented_child_cleanup_failure_reason="final_cleanup_mask_overlap_below_threshold",
                )
                for record in debug_context.get("source_child_cleanup_records") or []:
                    if isinstance(record, dict) and str(record.get("region_id") or "") == str(region_id):
                        record["covered"] = False
                        record["pending_parent_cleanup_proof"] = True
                        record["coverage_ratio"] = round(float(overlap_ratio), 3)
                        record["operation_overlap_ratio"] = round(float(operation_overlap / max(1, child_pixels)), 3)
                        record["cleanup_partition_id"] = partition_id
                        record["cleanup_partition_scope"] = partition_scope
                        record["missing_reason"] = "final_cleanup_mask_overlap_below_threshold"
                        record["represented_child_cleanup_proof_status"] = "failed"
            continue
        mark_render_region(
            debug_context,
            str(region_id),
            source_child_cleanup_covered=True,
            source_child_cleanup_pending_parent_cleanup_proof=False,
            source_child_cleanup_coverage_ratio=round(float(overlap_ratio), 3),
            source_child_cleanup_operation_overlap_ratio=round(float(operation_overlap / max(1, child_pixels)), 3),
            source_child_cleanup_partition_id=partition_id,
            source_child_cleanup_partition_scope=partition_scope,
            source_child_cleanup_missing_reason=None,
            represented_child_cleanup_proof_status="passed",
            represented_child_cleanup_required=True,
            represented_child_cleanup_mask_consumed=True,
            represented_child_cleanup_coverage_ratio=round(float(overlap_ratio), 3),
            represented_child_cleanup_failure_reason=None,
        )
        for record in debug_context.get("source_child_cleanup_records") or []:
            if isinstance(record, dict) and str(record.get("region_id") or "") == str(region_id):
                record["covered"] = True
                record["pending_parent_cleanup_proof"] = False
                record["coverage_ratio"] = round(float(overlap_ratio), 3)
                record["operation_overlap_ratio"] = round(float(operation_overlap / max(1, child_pixels)), 3)
                record["cleanup_partition_id"] = partition_id
                record["cleanup_partition_scope"] = partition_scope
                record["missing_reason"] = None
                record["represented_child_cleanup_proof_status"] = "passed"
    _refresh_source_child_cleanup_counts(debug_context)


def _refresh_source_child_cleanup_counts(debug_context: dict | None) -> None:
    if not debug_context:
        return
    records = [record for record in debug_context.get("source_child_cleanup_records") or [] if isinstance(record, dict)]
    required = sum(1 for record in records if record.get("required"))
    covered = sum(1 for record in records if record.get("required") and record.get("covered"))
    missing = max(0, required - covered)
    debug_context["uncleaned_source_child_count"] = missing
    counts = debug_context.setdefault("counts", {})
    counts["source_child_cleanup_required"] = required
    counts["source_child_cleanup_covered"] = covered
    counts["uncleaned_source_child_count"] = missing


def _source_child_cleanup_is_covered(debug_context: dict | None, region_id: str) -> bool:
    if not debug_context:
        return False
    entry = (debug_context.get("regions") or {}).get(str(region_id), {})
    return isinstance(entry, dict) and entry.get("source_child_cleanup_covered") is True


def _render_visual_contract_block_reason(
    render_fit_adjustment: dict | None,
    render_quality_debug: dict | None,
    text: str,
) -> str:
    chars = len(_meaningful_render_chars(text))
    if chars < 8:
        return ""
    quality = render_quality_debug or {}
    outside = _float_or_zero(quality.get("final_render_outside_allowed_area_ratio"))
    if outside > 0.12:
        return "render_outside_root_or_allowed_area"
    if quality.get("render_outside_text_area_container") is True:
        return "render_outside_text_area_container"
    adjustment = render_fit_adjustment or {}
    v5_class = str(adjustment.get("render_readability_v5_final_class") or "")
    v5_density = _float_or_zero(adjustment.get("render_readability_v5_density_after"))
    v5_edge = _float_or_zero(adjustment.get("render_readability_v5_edge_contact_after"))
    if v5_class.startswith("render_readability_v5_unresolved"):
        return v5_class.replace("render_readability_v5_", "render_")
    if v5_class == "render_readability_v5_accepted_complete_dense_watch":
        if v5_density >= 0.42 or v5_edge >= 0.75 or chars >= 18:
            return "render_readability_dense_watch_requires_visual_review"
    after_density = _float_or_zero(adjustment.get("render_fit_after_density"))
    if chars >= 12 and after_density >= 0.55:
        return "render_density_high"
    return ""


def _phase2e_target_row_id(
    debug_context: dict | None,
    *,
    region_id: str | None = None,
    root_id: str | None = None,
    operation: dict[str, object] | None = None,
    region: dict | None = None,
    meta: dict | None = None,
) -> str:
    candidate_root = str(root_id or "").strip()
    if not candidate_root and operation:
        candidate_root = str(
            operation.get("text_block_root_id")
            or operation.get("root_id")
            or operation.get("visible_area_id")
            or ""
        ).strip()
    if not candidate_root and meta:
        candidate_root = str(meta.get("text_block_root_id") or meta.get("root_id") or "").strip()
    if not candidate_root and region:
        candidate_root = str(region.get("text_block_root_id") or region.get("root_id") or "").strip()
    if candidate_root in _PHASE2E_TARGET_ROOT_IDS:
        return _PHASE2E_TARGET_ROOT_IDS[candidate_root]
    page_id = str((debug_context or {}).get("page_id") or "").strip()
    candidate_region = str(region_id or "").strip()
    if not candidate_region and operation:
        candidate_region = str(operation.get("region_id") or "").strip()
    if not candidate_region and meta:
        candidate_region = str(meta.get("region_id") or "").strip()
    if not candidate_region and region:
        candidate_region = str(region.get("region_id") or "").strip()
    return _PHASE2E_TARGET_REGION_IDS.get((page_id, candidate_region), "")


def _phase2e_region_snapshot(meta: dict | None) -> dict[str, object]:
    if not isinstance(meta, dict):
        return {}
    keys = [
        "region_id",
        "text_block_root_id",
        "cleanup_partition_id",
        "cleanup_mask_bbox",
        "source_glyph_mask_id",
        "source_glyph_erasure_bbox",
        "source_glyph_erasure_expected_area_bbox",
        "source_glyph_erasure_coverage_ratio",
        "cleanup_covers_source_glyphs",
        "cleanup_operation_failure_class",
        "cleanup_effectiveness_retry_status",
        "cleanup_effectiveness_retry_operation_id",
        "cleanup_effectiveness_retry_residual_ratio",
        "pre_render_source_erasure_status",
        "pre_render_source_erasure_retry_status",
        "pre_render_source_erasure_retry_mask_bbox",
        "pre_render_source_erasure_retry_mask_coverage",
        "pre_render_source_erasure_failure_reason",
        "pre_render_source_erasure_residual_ratio",
        "source_erasure_warning_class",
        "source_erasure_warning_reason",
        "source_erasure_failure_reason",
        "cleanup_visual_validation_failed",
        "cleanup_source_erasure_failure_reason",
    ]
    return {key: meta.get(key) for key in keys if key in meta}


def _phase2e_mask_summary(mask) -> dict[str, object]:
    if mask is None:
        return {"pixels": 0, "bbox": None}
    stats = mask_stats(mask) or {}
    bbox = _mask_stats_box(stats)
    return {
        "pixels": int((stats or {}).get("pixels") or 0),
        "bbox": list(bbox) if bbox else None,
    }


def _phase2e_append_trace(
    debug_context: dict | None,
    stage: str,
    *,
    region_id: str | None = None,
    root_id: str | None = None,
    operation: dict[str, object] | None = None,
    region: dict | None = None,
    meta: dict | None = None,
    fields: dict[str, object] | None = None,
) -> None:
    if not debug_context:
        return
    target_row_id = _phase2e_target_row_id(
        debug_context,
        region_id=region_id,
        root_id=root_id,
        operation=operation,
        region=region,
        meta=meta,
    )
    if not target_row_id:
        return
    record: dict[str, object] = {
        "phase2e_target_row_id": target_row_id,
        "stage": stage,
        "page_id": debug_context.get("page_id"),
        "region_id": region_id or (operation or {}).get("region_id") or (region or {}).get("region_id"),
        "root_id": root_id
        or (operation or {}).get("text_block_root_id")
        or (operation or {}).get("root_id")
        or (region or {}).get("text_block_root_id")
        or (meta or {}).get("text_block_root_id"),
        "region_state": _phase2e_region_snapshot(meta),
    }
    if operation:
        for key in [
            "cleanup_operation_id",
            "operation_kind",
            "cleanup_partition_id",
            "cleanup_partition_scope",
            "cleanup_mode",
            "backend",
            "cleanup_operation_failure_class",
            "cleanup_effectiveness_retry_status",
            "cleanup_effectiveness_retry_reason",
            "cleanup_retry_parent_operation_id",
            "cleanup_mask_bbox",
            "cleanup_mask_pixels",
            "mask_source_overlap_ratio",
            "source_residual_ratio_after_cleanup",
            "pre_clean_crop_path",
            "cleanup_mask_crop_path",
            "post_clean_crop_path",
            "residual_original_vs_post_clean_path",
        ]:
            if key in operation:
                record[key] = operation.get(key)
    if fields:
        record.update(fields)
    debug_context.setdefault("phase2e_cleanup_propagation_trace", []).append(record)


def _apply_pre_render_source_erasure_contract(
    source_np,
    working,
    render_regions: list[tuple[object, ...]],
    regions_by_id: dict[str, dict],
    debug_context: dict | None,
    perf_telemetry_context: dict | None,
    inpaint_mode: str,
    use_gpu: bool,
    model_id: str,
):
    if source_np is None or working is None or cv2 is None or np is None:
        return working, set()
    micro_start = _renderer_micro_start()
    region_meta = debug_context.setdefault("regions", {}) if debug_context is not None else {}
    clean_np = np.array(working)
    failed: dict[str, dict[str, object]] = {}
    _renderer_micro_add("renderer_pre_render_candidate_collect_time", micro_start)
    for render_entry in render_regions:
        region_id = str(render_entry[0] or "")
        if not region_id:
            continue
        region = regions_by_id.get(region_id) or {}
        meta = region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}
        _phase2e_append_trace(
            debug_context,
            "before_pre_render_proof",
            region_id=region_id,
            region=region,
            meta=meta,
            fields={
                "proof_image_state": "working_snapshot_before_pre_render_retries",
                "region_state_updated": bool(isinstance(meta, dict)),
            },
        )
        pre_render_proof_start = time.time()
        micro_start = _renderer_micro_start()
        proof = _pre_render_source_erasure_proof(
            source_np,
            clean_np,
            region,
            meta,
            debug_context=debug_context,
            perf_telemetry_context=perf_telemetry_context,
        )
        _renderer_perf_add_timing(
            debug_context,
            perf_telemetry_context,
            "renderer_pre_render_proof_time",
            time.time() - pre_render_proof_start,
        )
        _renderer_micro_add("renderer_pre_render_proof_time", micro_start)
        micro_start = _renderer_micro_start()
        _phase2e_append_trace(
            debug_context,
            "after_pre_render_proof",
            region_id=region_id,
            region=region,
            meta=meta,
            fields={
                "proof_required": proof.get("required"),
                "proof_passed": proof.get("passed"),
                "proof_failure_reason": proof.get("failure_reason"),
                "proof_expected_box": list(proof.get("expected_box") or []),
                "proof_expected_pixels": proof.get("expected_pixels"),
                "proof_residual_pixels": proof.get("residual_pixels"),
                "proof_residual_ratio": proof.get("residual_ratio"),
                "proof_effective_coverage": proof.get("coverage"),
                "proof_expected_mask": _phase2e_mask_summary(proof.get("_expected_mask")),
            },
        )
        _renderer_micro_add("renderer_pre_render_audit_record_time", micro_start)
        if not proof.get("required"):
            continue
        if proof.get("passed"):
            _mark_pre_render_source_erasure_pass(debug_context, region_id, proof)
        else:
            failed[region_id] = proof

    retry_region_ids: set[str] = set()
    for region_id, proof in failed.items():
        micro_start = _renderer_micro_start()
        repair_mask = _pre_render_source_erasure_repair_mask(source_np, proof)
        _renderer_micro_add("renderer_pre_render_mask_build_time", micro_start)
        if repair_mask is None:
            continue
        retry_cleanup_tag = "pre_render_source_erasure_contract_retry"
        if not _pre_render_repair_mask_is_safe(repair_mask, proof, source_np.shape):
            mark_render_region(
                debug_context,
                region_id,
                pre_render_source_erasure_retry_attempted=True,
                pre_render_source_erasure_retry_status="rejected",
                pre_render_source_erasure_retry_reason="repair_mask_not_safe_root_local",
            )
            continue
        micro_start = _renderer_micro_start()
        cleanup_start = time.time()
        cleanup_debug: dict[str, object] = {}
        pre_cleanup_image = working
        working = _apply_local_text_removal(
            working,
            repair_mask,
            inpaint_mode,
            use_gpu,
            model_id=model_id,
            cleanup_tag=retry_cleanup_tag,
            debug_info=cleanup_debug,
        )
        cleanup_elapsed = time.time() - cleanup_start
        retry_region_ids.add(region_id)
        _renderer_perf_add_timing(debug_context, perf_telemetry_context, "inpainting_time", cleanup_elapsed)
        add_count(debug_context, "inpaint_calls")
        _renderer_micro_add("renderer_pre_render_cleanup_execution_time", micro_start)
        stats = mask_stats(repair_mask) or {}
        bbox = _mask_stats_box(stats)
        operation = {
                "operation_kind": "pre_render_source_erasure_contract_retry",
                "region_id": region_id,
                "cleanup_mode": retry_cleanup_tag,
                "cleanup_partition_id": f"pre_render_source_erasure_{region_id}",
                "cleanup_partition_scope": "root_local",
                "elapsed_sec": round(cleanup_elapsed, 6),
                "backend": cleanup_debug.get("backend"),
                "backend_detail": cleanup_debug.get("backend_detail"),
                "requested_inpaint_mode": inpaint_mode,
                "effective_inpaint_mode": cleanup_debug.get("effective_inpaint_mode"),
                "mask_pixels": int((stats or {}).get("pixels") or 0),
                "mask_bbox": list(bbox) if bbox else None,
                "mask_bbox_area": _xyxy_area(bbox) if bbox else 0,
                "crop_bbox": cleanup_debug.get("crop_bbox"),
                "crop_area": cleanup_debug.get("crop_area"),
                "mask_ratio": cleanup_debug.get("mask_ratio"),
            }
        expected_mask = proof.get("_expected_mask")
        micro_start = _renderer_micro_start()
        operation = _record_cleanup_operation_trace(
            debug_context,
            operation,
            pre_cleanup_image,
            working,
            repair_mask,
            expected_mask=expected_mask,
        )
        _renderer_micro_add("renderer_pre_render_audit_record_time", micro_start)
        micro_start = _renderer_micro_start()
        working, retry_operation = _maybe_apply_cleanup_effectiveness_retry(
            working=working,
            cleanup_mask=repair_mask,
            expected_mask=expected_mask,
            operation=operation,
            debug_context=debug_context,
            inpaint_mode=inpaint_mode,
            use_gpu=use_gpu,
            model_id=model_id,
            region_id=region_id,
        )
        _renderer_micro_add("renderer_pre_render_cleanup_execution_time", micro_start)
        _record_render_cleanup_operation(debug_context, operation)
        retry_coverage = _mask_overlap_ratio(expected_mask, repair_mask)
        micro_start = _renderer_micro_start()
        mark_render_region(
            debug_context,
            region_id,
            pre_render_source_erasure_retry_attempted=True,
            pre_render_source_erasure_retry_status="applied",
            pre_render_source_erasure_retry_mask=stats,
            pre_render_source_erasure_retry_mask_bbox=list(bbox) if bbox else None,
            pre_render_source_erasure_retry_mask_coverage=round(float(retry_coverage), 3),
            cleanup_effectiveness_retry_attempted=operation.get("cleanup_effectiveness_retry_attempted"),
            cleanup_effectiveness_retry_status=operation.get("cleanup_effectiveness_retry_status"),
            cleanup_effectiveness_retry_operation_id=operation.get("cleanup_effectiveness_retry_operation_id"),
            cleanup_effectiveness_retry_residual_ratio=operation.get("cleanup_effectiveness_retry_residual_ratio"),
            source_glyph_erasure_coverage_ratio=round(max(
                retry_coverage,
                _float_or_zero((region_meta.get(region_id, {}) or {}).get("source_glyph_erasure_coverage_ratio")),
            ), 3),
            cleanup_covers_source_glyphs=retry_coverage >= _SOURCE_ERASURE_CONTRACT_COVERAGE_THRESHOLD,
            cleanup_source_erasure_failure_reason=None if retry_coverage >= _SOURCE_ERASURE_CONTRACT_COVERAGE_THRESHOLD else "cleanup_mask_misses_source_glyphs",
        )
        _renderer_micro_add("renderer_pre_render_audit_record_time", micro_start)
        _phase2e_append_trace(
            debug_context,
            "after_pre_render_retry_mark_region",
            region_id=region_id,
            operation=operation,
            region=regions_by_id.get(region_id) or {},
            meta=(region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}),
            fields={
                "retry_operation_id": operation.get("cleanup_operation_id"),
                "retry_operation_class": operation.get("cleanup_operation_failure_class"),
                "retry_mask": _phase2e_mask_summary(repair_mask),
                "retry_expected_mask": _phase2e_mask_summary(expected_mask),
                "retry_mask_coverage": round(float(retry_coverage), 3),
                "region_state_updated": bool(isinstance(region_meta, dict) and region_id in region_meta),
            },
        )
        _mark_source_children_covered_by_mask(
            debug_context,
            repair_mask,
            f"pre_render_source_erasure_{region_id}",
            "root_local",
        )

    if retry_region_ids:
        clean_np = np.array(working)
        for region_id in sorted(retry_region_ids):
            region = regions_by_id.get(region_id) or {}
            meta = region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}
            pre_render_proof_start = time.time()
            micro_start = _renderer_micro_start()
            proof = _pre_render_source_erasure_proof(
                source_np,
                clean_np,
                region,
                meta,
                debug_context=debug_context,
                perf_telemetry_context=perf_telemetry_context,
            )
            _renderer_perf_add_timing(
                debug_context,
                perf_telemetry_context,
                "renderer_pre_render_proof_time",
                time.time() - pre_render_proof_start,
            )
            _renderer_micro_add("renderer_pre_render_proof_time", micro_start)
            micro_start = _renderer_micro_start()
            _phase2e_append_trace(
                debug_context,
                "after_pre_render_retry_proof",
                region_id=region_id,
                region=region,
                meta=meta,
                fields={
                    "proof_image_state": "working_snapshot_after_pre_render_retries",
                    "proof_required": proof.get("required"),
                    "proof_passed": proof.get("passed"),
                    "proof_failure_reason": proof.get("failure_reason"),
                    "proof_expected_box": list(proof.get("expected_box") or []),
                    "proof_expected_pixels": proof.get("expected_pixels"),
                    "proof_residual_pixels": proof.get("residual_pixels"),
                    "proof_residual_ratio": proof.get("residual_ratio"),
                    "proof_effective_coverage": proof.get("coverage"),
                    "proof_expected_mask": _phase2e_mask_summary(proof.get("_expected_mask")),
                },
            )
            _renderer_micro_add("renderer_pre_render_audit_record_time", micro_start)
            if proof.get("passed"):
                failed.pop(region_id, None)
                _mark_pre_render_source_erasure_pass(debug_context, region_id, proof, retry=True)
            else:
                failed[region_id] = proof

    blocked: set[str] = set()
    for region_id, proof in failed.items():
        micro_start = _renderer_micro_start()
        reason = str(proof.get("failure_reason") or "pre_render_source_erasure_failed")
        warning_class = "mask_generation_failure" if "mask" in reason or "coverage" in reason else "true_source_residual_blocker"
        meta = region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}
        retry_status = str(meta.get("pre_render_source_erasure_retry_status") or "")
        retry_fields: dict[str, object] = {}
        if retry_status == "applied":
            retry_fields["pre_render_source_erasure_retry_status"] = "failed"
        mark_render_region(
            debug_context,
            region_id,
            pre_render_source_erasure_required=True,
            pre_render_source_erasure_status="failed_audit_only",
            pre_render_source_erasure_failure_reason=reason,
            pre_render_source_erasure_residual_ratio=proof.get("residual_ratio"),
            pre_render_residual_score=proof.get("residual_ratio"),
            pre_render_source_erasure_residual_pixels=proof.get("residual_pixels"),
            pre_render_source_erasure_expected_pixels=proof.get("expected_pixels"),
            source_erasure_warning_class=warning_class,
            source_erasure_warning_reason=reason,
            source_erasure_failure_reason=reason,
            cleanup_visual_validation_failed=True,
            cleanup_source_erasure_failure_reason=reason,
            source_erasure_proof_action="audit_only_render_allowed",
            render_blocked_by_visual_contract=False,
            render_visual_contract_blocker_reason=reason,
            render_allowed_despite_audit_blocker=True,
            render_suppressed_by_legacy_reason=False,
            **retry_fields,
        )
        _renderer_micro_add("renderer_pre_render_decision_time", micro_start)
        micro_start = _renderer_micro_start()
        _phase2e_append_trace(
            debug_context,
            "after_pre_render_failure_mark_region",
            region_id=region_id,
            region=regions_by_id.get(region_id) or {},
            meta=(region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}),
            fields={
                "proof_failure_reason": reason,
                "proof_warning_class": warning_class,
                "proof_residual_ratio": proof.get("residual_ratio"),
                "proof_residual_pixels": proof.get("residual_pixels"),
                "proof_expected_pixels": proof.get("expected_pixels"),
                "region_state_updated": bool(isinstance(region_meta, dict) and region_id in region_meta),
            },
        )
        _renderer_micro_add("renderer_pre_render_audit_record_time", micro_start)
        if isinstance(meta, dict) and meta.get("source_child_cleanup_required") is True:
            mark_render_region(
                debug_context,
                region_id,
                source_child_cleanup_covered=False,
                source_child_cleanup_pending_parent_cleanup_proof=True,
                source_child_cleanup_missing_reason="pre_render_source_residual_remaining",
                represented_child_cleanup_proof_status="failed",
            )
        blocked.add(region_id)
    if blocked:
        _refresh_source_child_cleanup_counts(debug_context)
    return working, blocked


def _mark_pre_render_source_erasure_pass(
    debug_context: dict | None,
    region_id: str,
    proof: dict[str, object],
    *,
    retry: bool = False,
) -> None:
    mark_render_region(
        debug_context,
        region_id,
        pre_render_source_erasure_required=True,
        pre_render_source_erasure_status="passed_after_retry" if retry else "passed",
        pre_render_source_erasure_failure_reason=None,
        pre_render_source_erasure_residual_ratio=proof.get("residual_ratio"),
        pre_render_residual_score=proof.get("residual_ratio"),
        pre_render_source_erasure_residual_pixels=proof.get("residual_pixels"),
        pre_render_source_erasure_expected_pixels=proof.get("expected_pixels"),
        pre_render_source_erasure_retry_status="succeeded" if retry else None,
        cleanup_visual_validation_failed=False,
        source_erasure_warning_class="repaired_source_residual",
        source_erasure_warning_reason="pre_render_source_erasure_proof_passed",
        cleanup_source_erasure_failure_reason=None,
        source_erasure_proof_action="audit_passed",
    )
    region_meta = (debug_context or {}).get("regions") or {}
    meta = region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}
    _phase2e_append_trace(
        debug_context,
        "after_pre_render_pass_mark_region",
        region_id=region_id,
        meta=meta,
        fields={
            "proof_passed": True,
            "proof_pass_retry": retry,
            "proof_residual_ratio": proof.get("residual_ratio"),
            "proof_expected_box": list(proof.get("expected_box") or []),
            "proof_expected_pixels": proof.get("expected_pixels"),
            "proof_expected_mask": _phase2e_mask_summary(proof.get("_expected_mask")),
            "region_state_updated": bool(isinstance(region_meta, dict) and region_id in region_meta),
        },
    )


def _pre_render_source_erasure_proof(
    source_np,
    clean_np,
    region: dict,
    meta: dict,
    *,
    debug_context: dict | None = None,
    perf_telemetry_context: dict | None = None,
) -> dict[str, object]:
    required = _region_needs_pre_render_source_erasure_check(region, meta)
    result: dict[str, object] = {"required": required, "passed": True}
    if not required:
        return result
    box = _pre_render_source_erasure_box(region, meta, source_np.shape)
    if box is None:
        result.update({"passed": False, "failure_reason": "pre_render_source_erasure_missing_expected_box"})
        return result
    expected_mask_start = time.time()
    micro_start = _renderer_micro_start()
    expected_mask = _pre_render_expected_source_mask(source_np, region, meta, box)
    _renderer_perf_add_timing(
        debug_context,
        perf_telemetry_context,
        "renderer_pre_render_expected_mask_time",
        time.time() - expected_mask_start,
    )
    _renderer_micro_add("renderer_pre_render_expected_mask_time", micro_start)
    if expected_mask is None or not np.any(expected_mask):
        result.update(
            {
                "passed": False,
                "failure_reason": "pre_render_expected_source_mask_missing",
                "expected_box": box,
                "expected_pixels": 0,
            }
        )
        return result
    expected_pixels = int((expected_mask > 0).sum())
    source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
    clean_gray = cv2.cvtColor(clean_np, cv2.COLOR_RGB2GRAY)
    expected = expected_mask > 0
    source_values = source_gray[expected]
    clean_values = clean_gray[expected]
    if source_values.size <= 0:
        result.update({"passed": False, "failure_reason": "pre_render_expected_source_mask_empty"})
        return result
    source_dark = source_values <= 150
    source_light = source_values >= 155
    unchanged = np.abs(clean_values.astype(np.int16) - source_values.astype(np.int16)) < 45
    still_dark = source_dark & (clean_values <= 160)
    still_light = source_light & (clean_values >= 145)
    residual_pixels = int((unchanged & (still_dark | still_light)).sum())
    residual_ratio = residual_pixels / max(1, expected_pixels)
    coverage = _float_or_none_contract(meta.get("source_glyph_erasure_coverage_ratio"))
    retry_coverage = _float_or_none_contract(meta.get("pre_render_source_erasure_retry_mask_coverage"))
    effective_coverage = max(
        value for value in [coverage, retry_coverage] if value is not None
    ) if coverage is not None or retry_coverage is not None else None
    result.update(
        {
            "passed": True,
            "expected_box": box,
            "expected_pixels": expected_pixels,
            "residual_pixels": residual_pixels,
            "residual_ratio": round(float(residual_ratio), 3),
            "coverage": round(float(effective_coverage), 3) if effective_coverage is not None else None,
            "_expected_mask": expected_mask,
        }
    )
    if meta.get("cleanup_visual_artifact_risk") is True or meta.get("cleanup_artifact_risk") is True:
        result.update({"passed": False, "failure_reason": "cleanup_mask_too_large_for_source_glyphs"})
        return result
    if effective_coverage is not None and effective_coverage < _SOURCE_ERASURE_CONTRACT_COVERAGE_THRESHOLD:
        result.update({"passed": False, "failure_reason": "pre_render_cleanup_mask_misses_source_glyphs"})
        return result
    if residual_ratio >= _SOURCE_ERASURE_CONTRACT_RESIDUAL_THRESHOLD and residual_pixels >= max(24, int(expected_pixels * 0.08)):
        result.update({"passed": False, "failure_reason": "pre_render_source_residual_pixels_remaining"})
        return result
    return result


def _region_needs_pre_render_source_erasure_check(region: dict, meta: dict) -> bool:
    if not isinstance(region, dict):
        return False
    flags = region.get("flags", {}) or {}
    if isinstance(flags, dict) and (flags.get("ignore") or flags.get("sfx")):
        return False
    if str(region.get("type") or "").strip() == "decorative_text":
        return False
    if not str(region.get("translation") or region.get("translated_text") or "").strip():
        return bool(meta.get("source_child_cleanup_required"))
    cleanup_mode = str(meta.get("cleanup_mode") or (region.get("render") or {}).get("cleanup_mode") or "").strip().lower()
    if cleanup_mode in {"preserve", "none", "skip"}:
        return False
    return bool(
        meta.get("cleanup_applied")
        or meta.get("source_glyph_mask_consumed_by_renderer")
        or meta.get("source_child_cleanup_required")
        or meta.get("cleanup_source_tracking_required")
    )


def _pre_render_source_erasure_box(region: dict, meta: dict, shape) -> tuple[int, int, int, int] | None:
    img_h, img_w = int(shape[0]), int(shape[1])
    for value in (
        meta.get("source_glyph_erasure_bbox"),
        meta.get("source_glyph_erasure_expected_area_bbox"),
        meta.get("cleanup_mask_bbox"),
        region.get("bbox"),
    ):
        box = _xyxy_or_xywh_to_xyxy(value, img_w, img_h)
        if box is not None:
            return box
    return None


def _pre_render_expected_source_mask(source_np, region: dict, meta: dict, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    bbox = [x0, y0, x1 - x0, y1 - y0]
    mask = _source_glyph_local_mask(
        source_np,
        bbox,
        None,
        dilate_px=1,
        limit_box=box,
        bright_context_only=False,
    )
    light_mask = None
    if _pre_render_should_try_light_glyphs(source_np, box, region, meta):
        light_mask = _top_row_caption_glyph_mask(source_np, bbox, None, dilate_px=1)
        if light_mask is not None:
            limiter = np.zeros(source_np.shape[:2], dtype=np.uint8)
            cv2.rectangle(limiter, (x0, y0), (x1, y1), 255, thickness=-1)
            light_mask = cv2.bitwise_and(light_mask, limiter)
    if mask is not None and light_mask is not None:
        combined = cv2.bitwise_or(mask, light_mask)
    else:
        combined = mask if mask is not None else light_mask
    return _pre_render_refine_expected_source_mask(source_np, combined, box)


def _pre_render_refine_expected_source_mask(source_np, mask, box):
    if mask is None or cv2 is None or np is None:
        return mask
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    crop_mask = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    if crop_mask.size == 0 or not np.any(crop_mask):
        return None
    crop_area = max(1, crop_mask.shape[0] * crop_mask.shape[1])
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(crop_mask, 8)
    kept = np.zeros_like(crop_mask)
    max_component_area = max(1600, int(crop_area * 0.16))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cx = int(stats[idx, cv2.CC_STAT_LEFT])
        cy = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 2 or area > max_component_area or cw <= 0 or ch <= 0:
            continue
        density = area / max(1, cw * ch)
        if density > 0.88:
            continue
        spans_width = cw >= int(crop_mask.shape[1] * 0.88)
        spans_height = ch >= int(crop_mask.shape[0] * 0.88)
        if spans_width and ch <= max(5, int(crop_mask.shape[0] * 0.08)):
            continue
        if spans_height and cw <= max(5, int(crop_mask.shape[1] * 0.08)):
            continue
        touches_outer = (
            cx <= 1
            or cy <= 1
            or cx + cw >= crop_mask.shape[1] - 1
            or cy + ch >= crop_mask.shape[0] - 1
        )
        if touches_outer and area > max(120, int(crop_area * 0.025)) and (spans_width or spans_height):
            continue
        kept[labels == idx] = 255
    if not np.any(kept):
        return None
    refined = np.zeros_like(mask)
    refined[y0:y1, x0:x1] = kept
    return refined


def _pre_render_should_try_light_glyphs(source_np, box, region: dict, meta: dict) -> bool:
    route = str(
        meta.get("text_area_route_intent")
        or region.get("text_area_route_intent")
        or ((region.get("render") or {}).get("text_area_route_intent") if isinstance(region.get("render"), dict) else "")
        or ""
    )
    region_type = str(region.get("type") or "")
    x0, y0, x1, y1 = box
    crop = source_np[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    p20 = float(np.percentile(gray, 20))
    p80 = float(np.percentile(gray, 80))
    median = float(np.median(gray))
    dark_or_contrast = median < 185.0 or p20 < 130.0 or (p80 - p20) >= 42.0
    caption_like = region_type in {"background_text", "narration_box"} or route in {"translate_caption", "translate_caption_background"}
    return caption_like and dark_or_contrast


def _pre_render_source_erasure_repair_mask(source_np, proof: dict[str, object]):
    expected_mask = proof.get("_expected_mask")
    if expected_mask is None or cv2 is None or np is None:
        return None
    repair = expected_mask.copy().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, kernel, iterations=1)
    repair = cv2.dilate(repair, kernel, iterations=1)
    box = proof.get("expected_box")
    if isinstance(box, tuple) and len(box) >= 4:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
        limiter = np.zeros(source_np.shape[:2], dtype=np.uint8)
        cv2.rectangle(limiter, (x0, y0), (x1, y1), 255, thickness=-1)
        repair = cv2.bitwise_and(repair, limiter)
    return repair if np.any(repair) else None


def _pre_render_repair_mask_is_safe(repair_mask, proof: dict[str, object], shape) -> bool:
    if repair_mask is None or cv2 is None or np is None:
        return False
    box = proof.get("expected_box")
    if not isinstance(box, tuple) or len(box) < 4:
        return False
    expected_area = max(1, _xyxy_area(box))
    image_area = max(1, int(shape[0]) * int(shape[1]))
    stats = mask_stats(repair_mask) or {}
    bbox = _mask_stats_box(stats)
    pixels = int((stats or {}).get("pixels") or 0)
    if pixels <= 0 or bbox is None:
        return False
    if expected_area > int(image_area * 0.16):
        return False
    if pixels > int(expected_area * 0.46):
        return False
    if _xyxy_area(bbox) > int(expected_area * 1.10):
        return False
    return True


def _mask_overlap_ratio(expected_mask, cleanup_mask) -> float:
    if expected_mask is None or cleanup_mask is None or cv2 is None or np is None:
        return 0.0
    if getattr(expected_mask, "shape", None) != getattr(cleanup_mask, "shape", None):
        return 0.0
    expected_pixels = int((expected_mask > 0).sum())
    if expected_pixels <= 0:
        return 0.0
    overlap = int(((expected_mask > 0) & (cleanup_mask > 0)).sum())
    return overlap / max(1, expected_pixels)


def _write_pre_render_source_erasure_debug_image(working, debug_context: dict | None) -> None:
    _debug_write_pre_render_source_erasure_debug_image(working, debug_context)


def _xyxy_or_xywh_to_xyxy(value, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        a, b, c, d = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if c > a and d > b:
        x0, y0, x1, y1 = a, b, c, d
    elif c > 0 and d > 0:
        x0, y0, x1, y1 = a, b, a + c, b + d
    else:
        return None
    x0 = max(0, min(img_w, x0))
    y0 = max(0, min(img_h, y0))
    x1 = max(x0 + 1, min(img_w, x1))
    y1 = max(y0 + 1, min(img_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _float_or_none_contract(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _float_or_zero(value) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _source_child_cleanup_requirement(region: dict, active_parent_ids: set[str]) -> tuple[bool, str | None, str | None]:
    if _is_active_translation_region(region):
        return False, None, None
    if not _has_japanese_source_text(str(region.get("ocr_text", "") or "")):
        return False, None, None
    if _is_preserved_source_child(region):
        return False, None, None
    parent_id = _represented_child_parent_id(region)
    if not parent_id or parent_id not in active_parent_ids:
        return False, parent_id, "parent_translation_unit_not_active"
    final_state = str(
        region.get("child_final_state")
        or region.get("ocr_fragment_final_state")
        or region.get("logical_text_ownership_status")
        or ""
    ).strip().lower()
    represented = bool(
        region.get("represented_by_parent_id")
        or region.get("source_text_represented_by_block_id")
        or region.get("source_conservation_status") == "represented"
    )
    child_states = {
        "parent_child",
        "dependent_child",
        "transferred_child",
        "duplicate_child",
        "punctuation_child",
        "noise_review_only",
    }
    if final_state in child_states or represented:
        return True, parent_id, final_state or "represented_by_parent"
    return False, parent_id, f"unsupported_child_state:{final_state or 'missing'}"


def _active_translation_parent_ids(regions: list[dict]) -> set[str]:
    ids: set[str] = set()
    for region in regions:
        if not isinstance(region, dict) or not _is_active_translation_region(region):
            continue
        ids.update(_region_parent_ids(region))
    return ids


def _is_active_translation_region(region: dict) -> bool:
    flags = region.get("flags", {}) or {}
    if isinstance(flags, dict) and flags.get("ignore"):
        return False
    text = str(region.get("translation", "") or "").strip()
    if not text:
        return False
    if region.get("render_independently") is False and str(region.get("child_final_state") or "").lower() not in {
        "parent_anchor",
        "standalone_parent",
    }:
        return False
    return True


def _region_parent_ids(region: dict) -> set[str]:
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    ids: set[str] = set()
    for key in (
        "source_glyph_mask_parent_logical_text_unit_id",
        "text_block_parent_logical_text_unit_id",
        "parent_logical_text_unit_id",
        "represented_by_parent_id",
        "source_text_represented_by_block_id",
        "active_translation_unit_id",
        "logical_text_block_id",
    ):
        value = render.get(key)
        if value in (None, "", [], {}):
            value = region.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item or "").strip()
                if text:
                    ids.add(text)
        else:
            text = str(value or "").strip()
            if text:
                ids.add(text)
    return ids


def _represented_child_parent_id(region: dict) -> str | None:
    preferred = (
        region.get("represented_by_parent_id")
        or region.get("source_text_represented_by_block_id")
        or region.get("parent_logical_text_unit_id")
        or region.get("active_translation_unit_id")
        or region.get("logical_text_block_id")
    )
    text = str(preferred or "").strip()
    return text or None


def _is_preserved_source_child(region: dict) -> bool:
    flags = region.get("flags", {}) or {}
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    semantic = str(region.get("type", "") or "").strip().lower()
    text_area_type = str(region.get("text_area_container_type", "") or "").strip().lower()
    route_intent = str(region.get("text_area_route_intent", "") or "").strip().lower()
    cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
    conflict_flags = {str(item).strip().lower() for item in (region.get("text_area_conflict_flags") or [])}
    if isinstance(flags, dict) and (flags.get("sfx") or flags.get("sign")):
        return True
    if semantic in {"decorative_text", "sfx", "sign"}:
        return True
    if text_area_type == "sfx_decorative_art":
        return True
    if route_intent == "preserve_sfx_decorative":
        return True
    if cleanup_mode == "preserve":
        return True
    return bool(conflict_flags & {"sfx_conflict", "decorative_conflict", "preserve_conflict"})


def _has_japanese_source_text(text: str) -> bool:
    return any(
        "\u3040" <= ch <= "\u30ff"
        or "\u3400" <= ch <= "\u9fff"
        or ch in {"ー", "々", "〆", "ヶ"}
        for ch in str(text or "")
    )


def _represented_child_cleanup_mask(
    region: dict,
    source_glyph_record: object | None,
    img_np,
    img_w: int,
    img_h: int,
    debug_context: dict | None = None,
):
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    record_mask = _source_glyph_record_mask(source_glyph_record)
    if record_mask is not None and getattr(record_mask, "shape", None) == img_np.shape[:2] and np.any(record_mask):
        method = "source_glyph_mask_represented_child"
        mask = record_mask.astype(np.uint8)
    else:
        bbox = region.get("bbox") or [0, 0, 0, 0]
        polygon = region.get("polygon")
        orientation = str(render.get("source_orientation", "") or "").lower()
        try:
            _x, _y, w, h = [int(v) for v in bbox[:4]]
        except Exception:
            return None, "invalid_child_bbox", "invalid_child_bbox"
        dilate_px = max(2, min(5, int(max(min(w, h) * 0.08, 2))))
        local_region_mask_start = time.time()
        mask = _local_region_text_mask(
            img_np,
            bbox,
            polygon,
            dilate_px=dilate_px,
            strong_vertical=(orientation == "vertical" or h > w * 1.35),
            glyph_only=True,
        )
        add_timing(
            debug_context,
            "renderer_local_region_text_mask_time",
            time.time() - local_region_mask_start,
        )
        method = "represented_child_glyph_local"
        if mask is None or not np.any(mask):
            box = _xywh_value_to_xyxy(bbox, img_w, img_h)
            if box is None:
                return None, method, "child_glyph_mask_unavailable"
            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            pad = max(2, min(5, int(max((box[2] - box[0]), (box[3] - box[1])) * 0.04)))
            x0, y0, x1, y1 = _expand_box(box, pad, pad, img_w, img_h)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
            method = "represented_child_bbox_fallback"
    limit_box = _represented_child_cleanup_limit_box(region, img_w, img_h)
    if limit_box is not None:
        limit_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        x0, y0, x1, y1 = limit_box
        cv2.rectangle(limit_mask, (x0, y0), (x1, y1), 255, thickness=-1)
        mask = cv2.bitwise_and(mask.astype(np.uint8), limit_mask)
    stats = mask_stats(mask)
    box = _mask_stats_box(stats)
    image_area = max(1, int(img_w) * int(img_h))
    bbox_area = _xyxy_area(box) if box else 0
    if not box or int((stats or {}).get("pixels") or 0) <= 0:
        return None, method, "empty_child_cleanup_mask"
    if bbox_area > max(240000, int(image_area * 0.12)):
        return None, method, "child_cleanup_mask_too_large"
    return mask.astype(np.uint8), method, None


def _represented_child_cleanup_limit_box(region: dict, img_w: int, img_h: int):
    for key in (
        "text_area_container_bbox",
        "logical_text_block_allowed_bbox",
        "render_allowed_area_bbox",
    ):
        box = _xywh_value_to_xyxy(region.get(key), img_w, img_h)
        if box is not None:
            return box
        value = region.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
            except Exception:
                continue
            if x1 > x0 and y1 > y0:
                return (
                    max(0, min(img_w, x0)),
                    max(0, min(img_h, y0)),
                    max(0, min(img_w, x1)),
                    max(0, min(img_h, y1)),
                )
    return None


def _cleanup_partition_metadata_for_region(
    region: dict | None,
    *,
    render: dict | None = None,
    cleanup_scope: str | None = None,
    cleanup_partition_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(region, dict):
        region = {}
    if render is None:
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
    if not isinstance(render, dict):
        render = {}
    parent_id = _first_cleanup_id(
        region,
        render,
        (
            "source_glyph_mask_parent_logical_text_unit_id",
            "text_block_parent_logical_text_unit_id",
            "parent_logical_text_unit_id",
            "active_translation_unit_id",
            "logical_text_block_id",
        ),
    )
    root_id = _first_cleanup_id(
        region,
        render,
        (
            "source_glyph_mask_text_block_root_id",
            "text_block_root_id",
            "physical_bubble_graph_id",
            "logical_text_physical_bubble_id",
        ),
    )
    container_id = _first_cleanup_id(
        region,
        render,
        (
            "text_area_container_id",
            "diagnostic_text_container_id",
            "logical_text_block_container_id",
            "model_fusion_source_container_id",
        ),
    )
    if cleanup_scope:
        scope = cleanup_scope
    elif parent_id:
        scope = "parent_unit"
    elif root_id or container_id:
        scope = "root_container"
    else:
        scope = "global_fallback"
    if parent_id:
        key = f"parent:{parent_id}"
    elif root_id:
        key = f"root:{root_id}"
    elif container_id:
        key = f"container:{container_id}"
    else:
        key = ""
    meta: dict[str, object] = {
        "cleanup_partition_scope": scope,
        "parent_logical_text_unit_id": parent_id,
        "text_block_root_id": root_id,
        "text_area_container_id": container_id,
        "cleanup_partition_key": key,
    }
    if cleanup_partition_id:
        meta["cleanup_partition_id"] = cleanup_partition_id
    return meta


def _first_cleanup_id(region: dict, render: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = render.get(key)
        if value in (None, "", [], {}):
            value = region.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        text = str(value).strip()
        if text:
            return text
    return None


def _build_cleanup_partitions(
    text_mask,
    candidates: list[dict[str, object]],
    fallback_region_ids: list[str],
    img_w: int,
    img_h: int,
) -> list[dict[str, object]]:
    if cv2 is None or np is None or text_mask is None or not np.any(text_mask):
        return []
    groups: dict[str, dict[str, object]] = {}
    covered = np.zeros_like(text_mask, dtype=np.uint8)
    for candidate in candidates:
        mask = candidate.get("mask")
        if mask is None or getattr(mask, "shape", None) != text_mask.shape:
            continue
        remaining = cv2.bitwise_and(mask.astype(np.uint8), text_mask.astype(np.uint8))
        if not np.any(remaining):
            continue
        covered = cv2.bitwise_or(covered, remaining)
        key = str(candidate.get("cleanup_partition_key") or "")
        if not key:
            key = f"global_fallback:{candidate.get('region_id') or len(groups)}"
        group = groups.setdefault(
            key,
            {
                "mask": np.zeros_like(text_mask, dtype=np.uint8),
                "candidates": [],
                "region_ids": [],
                "source_child_region_ids": [],
                "cleanup_partition_scope": candidate.get("cleanup_partition_scope") or "global_fallback",
                "parent_logical_text_unit_id": candidate.get("parent_logical_text_unit_id"),
                "text_block_root_id": candidate.get("text_block_root_id"),
                "text_area_container_id": candidate.get("text_area_container_id"),
                "fallback_reason": candidate.get("fallback_reason"),
            },
        )
        group["mask"] = cv2.bitwise_or(group["mask"], remaining)
        group["candidates"].append({**candidate, "mask": remaining})
        for rid in candidate.get("region_ids") or []:
            if rid and rid not in group["region_ids"]:
                group["region_ids"].append(rid)
        for rid in candidate.get("source_child_region_ids") or []:
            if rid and rid not in group["source_child_region_ids"]:
                group["source_child_region_ids"].append(rid)

    partitions: list[dict[str, object]] = []
    for group in groups.values():
        _append_cleanup_group_partitions(partitions, group, img_w, img_h)

    residual = cv2.bitwise_and(text_mask.astype(np.uint8), cv2.bitwise_not(covered))
    if np.any(residual):
        partitions.append(
            {
                "cleanup_partition_id": f"cp_{len(partitions):03d}",
                "cleanup_partition_scope": "global_fallback",
                "mask": residual,
                "region_ids": list(fallback_region_ids),
                "parent_logical_text_unit_id": None,
                "text_block_root_id": None,
                "text_area_container_id": None,
                "fallback_reason": "residual_mask_pixels_without_region_candidate",
            }
        )
    partitions.sort(key=_cleanup_partition_sort_key)
    for idx, partition in enumerate(partitions):
        partition["cleanup_partition_id"] = partition.get("cleanup_partition_id") or f"cp_{idx:03d}"
    return partitions


def _append_cleanup_group_partitions(
    partitions: list[dict[str, object]],
    group: dict[str, object],
    img_w: int,
    img_h: int,
) -> None:
    mask = group.get("mask")
    if mask is None or not np.any(mask):
        return
    stats = mask_stats(mask)
    box = _mask_stats_box(stats)
    bbox_area = _xyxy_area(box) if box else 0
    image_area = max(1, int(img_w) * int(img_h))
    candidates = group.get("candidates") or []
    split_large = (
        len(candidates) > 1
        and bbox_area > max(420000, int(image_area * 0.18))
    )
    if split_large:
        for candidate in candidates:
            candidate_mask = candidate.get("mask")
            if candidate_mask is None or not np.any(candidate_mask):
                continue
            partitions.append(
                {
                    "cleanup_partition_scope": str(group.get("cleanup_partition_scope") or "root_container") + "_split",
                    "mask": candidate_mask,
                    "region_ids": list(candidate.get("region_ids") or []),
                    "source_child_region_ids": list(candidate.get("source_child_region_ids") or []),
                    "parent_logical_text_unit_id": group.get("parent_logical_text_unit_id"),
                    "text_block_root_id": group.get("text_block_root_id"),
                    "text_area_container_id": group.get("text_area_container_id"),
                    "fallback_reason": "split_large_parent_root_partition",
                }
            )
        return
    partitions.append(
        {
            "cleanup_partition_scope": group.get("cleanup_partition_scope") or "root_container",
            "mask": mask,
            "region_ids": list(group.get("region_ids") or []),
            "source_child_region_ids": list(group.get("source_child_region_ids") or []),
            "parent_logical_text_unit_id": group.get("parent_logical_text_unit_id"),
            "text_block_root_id": group.get("text_block_root_id"),
            "text_area_container_id": group.get("text_area_container_id"),
            "fallback_reason": group.get("fallback_reason"),
        }
    )


def _cleanup_partition_sort_key(partition: dict[str, object]) -> tuple[int, int, str]:
    stats = mask_stats(partition.get("mask"))
    box = _mask_stats_box(stats)
    if not box:
        return (10**9, 10**9, str(partition.get("cleanup_partition_id") or ""))
    return (int(box[1]), int(box[0]), str(partition.get("cleanup_partition_id") or ""))


def _should_use_cleanup_partitions(text_mask, partitions: list[dict[str, object]], mode: str, img_w: int, img_h: int) -> bool:
    if not partitions or text_mask is None or not np.any(text_mask):
        return False
    if len(partitions) <= 1:
        return False
    stats = mask_stats(text_mask)
    box = _mask_stats_box(stats)
    bbox_area = _xyxy_area(box) if box else 0
    image_area = max(1, int(img_w) * int(img_h))
    mode_norm = str(mode or "fast").strip().lower()
    return mode_norm == "ai" and (
        bbox_area > max(360000, int(image_area * 0.13))
        or any(part.get("cleanup_partition_scope") != "global_fallback" for part in partitions)
    )


def _apply_cleanup_partitions(
    image,
    partitions: list[dict[str, object]],
    mode: str,
    use_gpu: bool,
    model_id: str,
    debug_context: dict | None,
):
    working = image
    for index, partition in enumerate(partitions):
        mask = partition.get("mask")
        if mask is None or not np.any(mask):
            continue
        partition_id = str(partition.get("cleanup_partition_id") or f"cp_{index:03d}")
        partition["cleanup_partition_id"] = partition_id
        cleanup_start = time.time()
        cleanup_debug: dict[str, object] = {}
        pre_cleanup_image = working
        working = _apply_text_removal(
            working,
            mask,
            mode,
            use_gpu,
            model_id=model_id,
            debug_info=cleanup_debug,
        )
        cleanup_elapsed = time.time() - cleanup_start
        add_timing(debug_context, "inpainting_time", cleanup_elapsed)
        add_count(debug_context, "inpaint_calls")
        stats = mask_stats(mask)
        box = _mask_stats_box(stats)
        region_ids = list(partition.get("region_ids") or [])
        source_child_region_ids = list(partition.get("source_child_region_ids") or [])
        operation = {
            "operation_kind": "partitioned_text_mask_removal",
            "cleanup_partition_id": partition_id,
            "cleanup_partition_scope": partition.get("cleanup_partition_scope"),
            "region_ids": region_ids,
            "source_child_region_ids": source_child_region_ids,
            "parent_logical_text_unit_id": partition.get("parent_logical_text_unit_id"),
            "text_block_root_id": partition.get("text_block_root_id"),
            "text_area_container_id": partition.get("text_area_container_id"),
            "fallback_reason": partition.get("fallback_reason"),
            "cleanup_mode": "partitioned_text_mask",
            "elapsed_sec": round(cleanup_elapsed, 6),
            "backend": cleanup_debug.get("backend"),
            "backend_detail": cleanup_debug.get("backend_detail"),
            "requested_inpaint_mode": mode,
            "effective_inpaint_mode": cleanup_debug.get("effective_inpaint_mode"),
            "mask_pixels": int((stats or {}).get("pixels") or 0),
            "mask_bbox": list(box) if box else None,
            "mask_bbox_area": _xyxy_area(box) if box else 0,
            "crop_bbox": cleanup_debug.get("crop_bbox"),
            "crop_area": cleanup_debug.get("crop_area"),
            "mask_ratio": cleanup_debug.get("mask_ratio"),
        }
        operation = _record_cleanup_operation_trace(
            debug_context,
            operation,
            pre_cleanup_image,
            working,
            mask,
        )
        _record_render_cleanup_operation(debug_context, operation)
        _mark_source_children_covered_by_mask(
            debug_context,
            mask,
            partition_id,
            str(partition.get("cleanup_partition_scope") or "partitioned"),
        )
        for rid in region_ids:
            source_child_fields = {}
            if rid in source_child_region_ids:
                covered = _source_child_cleanup_is_covered(debug_context, rid)
                source_child_fields = {
                    "source_child_cleanup_partition_id": partition_id,
                    "source_child_cleanup_partition_scope": partition.get("cleanup_partition_scope"),
                    "source_child_cleanup_pending_parent_cleanup_proof": not covered,
                }
                if not covered:
                    source_child_fields["source_child_cleanup_covered"] = False
                    source_child_fields["source_child_cleanup_missing_reason"] = "final_cleanup_mask_overlap_below_threshold"
            mark_render_region(
                debug_context,
                rid,
                cleanup_inpaint_time_sec=round(cleanup_elapsed, 6),
                cleanup_backend=cleanup_debug.get("backend"),
                cleanup_backend_detail=cleanup_debug.get("backend_detail"),
                cleanup_operation_kind="partitioned_text_mask_removal",
                cleanup_requested_inpaint_mode=mode,
                cleanup_effective_inpaint_mode=cleanup_debug.get("effective_inpaint_mode"),
                cleanup_mask_pixels=int((stats or {}).get("pixels") or 0),
                cleanup_mask_bbox_area=_xyxy_area(box) if box else 0,
                cleanup_crop_bbox=cleanup_debug.get("crop_bbox"),
                cleanup_crop_area=cleanup_debug.get("crop_area"),
                cleanup_mask_ratio=cleanup_debug.get("mask_ratio"),
                cleanup_partition_id=partition_id,
                cleanup_partition_scope=partition.get("cleanup_partition_scope"),
                cleanup_partition_fallback_reason=partition.get("fallback_reason"),
                **source_child_fields,
            )
        _refresh_source_child_cleanup_counts(debug_context)
    return working


def _record_render_cleanup_operation(debug_context: dict | None, operation: dict[str, object]) -> None:
    if not debug_context:
        return
    debug_context.setdefault("render_cleanup_operations", []).append(operation)


def _render_eligibility_by_region(render_eligibility: object | None) -> dict[str, object]:
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


def _cleanup_trace_page_dir(debug_context: dict | None) -> str:
    return debug_stage_artifact_dir(debug_context, "cleanup_trace")


def _cleanup_trace_crop_box(operation: dict[str, object], mask) -> tuple[int, int, int, int] | None:
    raw = operation.get("crop_bbox") or operation.get("mask_bbox")
    if raw and len(raw) >= 4:
        try:
            x0, y0, x1, y1 = [int(float(v)) for v in list(raw)[:4]]
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
        except Exception:
            pass
    stats = mask_stats(mask)
    box = _mask_stats_box(stats)
    if box:
        return tuple(int(v) for v in box)
    return None


def _clip_cleanup_trace_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _save_cleanup_trace_image(debug_context: dict | None, image, box: tuple[int, int, int, int] | None, path: str) -> str:
    if Image is None or image is None or box is None:
        return ""
    try:
        written_path, ok, _error = write_debug_stage_image_path(
            debug_context,
            "cleanup_trace",
            path,
            image.crop(box),
        )
        return written_path if ok else ""
    except Exception:
        return ""


def _save_cleanup_trace_mask(debug_context: dict | None, mask, box: tuple[int, int, int, int] | None, path: str) -> str:
    if Image is None or np is None or mask is None or box is None:
        return ""
    try:
        x0, y0, x1, y1 = box
        crop = np.asarray(mask)[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        img = Image.fromarray(((crop > 0).astype(np.uint8) * 255), mode="L")
        written_path, ok, _error = write_debug_stage_image_path(debug_context, "cleanup_trace", path, img)
        return written_path if ok else ""
    except Exception:
        return ""


def _save_cleanup_residual_map(
    debug_context: dict | None,
    before_image,
    after_image,
    box: tuple[int, int, int, int] | None,
    path: str,
) -> tuple[str, int, float]:
    if Image is None or ImageChops is None or before_image is None or after_image is None or box is None:
        return "", 0, 0.0
    try:
        before_crop = before_image.crop(box).convert("RGB")
        after_crop = after_image.crop(box).convert("RGB")
        diff = ImageChops.difference(before_crop, after_crop).convert("L")
        threshold = diff.point(lambda px: 255 if px > 18 else 0)
        pixels = int(threshold.histogram()[255])
        total = max(1, threshold.size[0] * threshold.size[1])
        heat = Image.new("RGB", threshold.size, (0, 0, 0))
        heat.paste((255, 32, 32), mask=threshold)
        blended = Image.blend(after_crop, heat, 0.45)
        written_path, ok, _error = write_debug_stage_image_path(debug_context, "cleanup_trace", path, blended)
        return (written_path if ok else ""), pixels, round(pixels / total, 4)
    except Exception:
        return "", 0, 0.0


def _cleanup_trace_source_image(debug_context: dict | None):
    if not debug_context or Image is None:
        return None
    source_path = str(debug_context.get("source_path") or "").strip()
    if not source_path or not os.path.isfile(source_path):
        return None
    try:
        with Image.open(source_path) as original:
            return original.convert("RGB")
    except Exception:
        return None


def _cleanup_expected_mask_for_operation(debug_context: dict | None, operation: dict[str, object], cleanup_mask):
    if debug_context is None or np is None:
        return None
    expected = None
    child_masks = debug_context.get("_source_child_cleanup_masks") or {}
    ids: list[str] = []
    for key in ("region_id", "region_ids", "source_child_region_ids"):
        value = operation.get(key)
        if isinstance(value, (list, tuple, set)):
            ids.extend(str(item) for item in value if item)
        elif value:
            ids.append(str(value))
    for rid in ids:
        child_mask = child_masks.get(rid)
        if child_mask is None or getattr(child_mask, "shape", None) != getattr(cleanup_mask, "shape", None):
            continue
        expected = child_mask.copy() if expected is None else cv2.bitwise_or(expected, child_mask)
    scope = str(operation.get("cleanup_partition_scope") or "")
    mode = str(operation.get("cleanup_mode") or "")
    if expected is None and ("source_glyph" in scope or "glyph" in mode or str(operation.get("operation_kind") or "") == "pre_render_source_erasure_contract_retry"):
        expected = cleanup_mask
    return expected


def _cleanup_source_like_mask(source_image, cleanup_mask, expected_mask):
    if Image is None or cv2 is None or np is None or source_image is None or cleanup_mask is None:
        return None
    try:
        source_np = np.asarray(source_image.convert("RGB"))
        if source_np.shape[:2] != cleanup_mask.shape[:2]:
            return None
        if expected_mask is not None and getattr(expected_mask, "shape", None) == getattr(cleanup_mask, "shape", None):
            expected_bool = expected_mask > 0
        else:
            expected_bool = cleanup_mask > 0
        if not np.any(expected_bool):
            return None
        source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
        stats = mask_stats(expected_mask if expected_mask is not None else cleanup_mask)
        box = _mask_stats_box(stats)
        local_median = 255.0
        if box:
            x0, y0, x1, y1 = [int(v) for v in box]
            crop = source_gray[y0:y1, x0:x1]
            if crop.size:
                local_median = float(np.median(crop))
        dark_source = source_gray <= 185
        light_source = (source_gray >= 188) & (local_median < 172.0)
        source_like = expected_bool & (dark_source | light_source)
        if int(source_like.sum()) < 8:
            return None
        return (source_like.astype(np.uint8) * 255)
    except Exception:
        return None


def _cleanup_trace_mask_overlap(cleanup_mask, expected_mask) -> tuple[int, float, int]:
    if np is None or cleanup_mask is None or expected_mask is None:
        return 0, 0.0, 0
    if getattr(cleanup_mask, "shape", None) != getattr(expected_mask, "shape", None):
        return 0, 0.0, int((expected_mask > 0).sum())
    expected_pixels = int((expected_mask > 0).sum())
    if expected_pixels <= 0:
        return 0, 0.0, 0
    overlap = int(((cleanup_mask > 0) & (expected_mask > 0)).sum())
    return overlap, round(overlap / max(1, expected_pixels), 4), expected_pixels


def _cleanup_trace_change_metrics(before_image, after_image, cleanup_mask, expected_mask, source_image=None) -> dict[str, object]:
    result: dict[str, object] = {
        "changed_pixels_inside_mask": 0,
        "changed_pixel_ratio_inside_mask": 0.0,
        "source_residual_pixels_after_cleanup": "",
        "source_residual_ratio_after_cleanup": "",
        "source_expected_pixels": "",
        "mask_source_overlap_pixels": "",
        "mask_source_overlap_ratio": "",
    }
    if np is None or before_image is None or after_image is None or cleanup_mask is None:
        return result
    try:
        before_np = np.asarray(before_image.convert("RGB"))
        after_np = np.asarray(after_image.convert("RGB"))
        diff = np.max(np.abs(before_np.astype(np.int16) - after_np.astype(np.int16)), axis=2)
        mask_bool = cleanup_mask > 0
        mask_pixels = int(mask_bool.sum())
        changed = int(((diff > 8) & mask_bool).sum())
        result["changed_pixels_inside_mask"] = changed
        result["changed_pixel_ratio_inside_mask"] = round(changed / max(1, mask_pixels), 4)
        source_like_mask = _cleanup_source_like_mask(source_image, cleanup_mask, expected_mask)
        if source_like_mask is not None and getattr(source_like_mask, "shape", None) == getattr(cleanup_mask, "shape", None):
            expected_bool = source_like_mask > 0
            expected_pixels = int(expected_bool.sum())
            if expected_pixels > 0:
                source_np = np.asarray(source_image.convert("RGB"))
                source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
                after_gray = cv2.cvtColor(after_np, cv2.COLOR_RGB2GRAY)
                dark_source = source_gray <= 185
                light_source = source_gray >= 188
                similar_to_source = np.abs(after_gray.astype(np.int16) - source_gray.astype(np.int16)) <= 38
                still_source_colored = (
                    (dark_source & (after_gray <= 172))
                    | (light_source & (after_gray >= 178))
                )
                similar = similar_to_source & still_source_colored
                residual = int((similar & expected_bool).sum())
                overlap = int(((cleanup_mask > 0) & expected_bool).sum())
                result["source_expected_pixels"] = expected_pixels
                result["mask_source_overlap_pixels"] = overlap
                result["mask_source_overlap_ratio"] = round(overlap / max(1, expected_pixels), 4)
                result["source_residual_pixels_after_cleanup"] = residual
                result["source_residual_ratio_after_cleanup"] = round(residual / max(1, expected_pixels), 4)
        elif expected_mask is not None and getattr(expected_mask, "shape", None) == getattr(cleanup_mask, "shape", None):
            expected_bool = expected_mask > 0
            expected_pixels = int(expected_bool.sum())
            if expected_pixels > 0:
                similar = diff <= 24
                residual = int((similar & expected_bool).sum())
                result["source_residual_pixels_after_cleanup"] = residual
                result["source_residual_ratio_after_cleanup"] = round(residual / max(1, expected_pixels), 4)
    except Exception:
        pass
    return result


def _cleanup_trace_classification(
    *,
    mask_pixels: int,
    overlap_ratio: float | None,
    changed_ratio: float,
    residual_ratio: float | str | None,
    expected_source_pixels: int,
) -> str:
    if mask_pixels <= 0:
        return "cleanup_operation_noop"
    if expected_source_pixels <= 0:
        return "operation_mapping_mismatch"
    try:
        residual = float(residual_ratio)
    except Exception:
        residual = None
    if residual is not None and residual < 0.16:
        return "passed"
    if overlap_ratio is not None and overlap_ratio < 0.55:
        return "mask_misses_source_pixels"
    if changed_ratio <= 0.01:
        return "cleanup_operation_noop"
    if residual is None:
        return "operation_mapping_mismatch"
    if residual >= 0.30 and changed_ratio < 0.08:
        return "cleanup_operation_ineffective"
    if residual >= 0.16:
        return "cleanup_applied_but_residual_remains"
    return "passed"


def _record_cleanup_operation_trace(
    debug_context: dict | None,
    operation: dict[str, object],
    before_image,
    after_image,
    cleanup_mask,
    expected_mask=None,
) -> dict[str, object]:
    if not debug_context or cleanup_mask is None or before_image is None or after_image is None:
        return operation
    page_dir = _cleanup_trace_page_dir(debug_context)
    if not page_dir:
        return operation
    counter = int(debug_context.get("_cleanup_operation_trace_counter") or 0)
    debug_context["_cleanup_operation_trace_counter"] = counter + 1
    page_id = str(debug_context.get("page_id") or "page")
    operation_id = f"{page_id}_cleanup_{counter:04d}"
    operation["cleanup_operation_id"] = operation_id
    expected_mask = expected_mask if expected_mask is not None else _cleanup_expected_mask_for_operation(debug_context, operation, cleanup_mask)
    crop_box = _cleanup_trace_crop_box(operation, cleanup_mask)
    if crop_box is not None and hasattr(before_image, "size"):
        crop_box = _clip_cleanup_trace_box(crop_box, int(before_image.size[0]), int(before_image.size[1]))
    trace_dir = os.path.join(page_dir, "cleanup_operation_traces", operation_id)
    source_image = _cleanup_trace_source_image(debug_context)
    cleanup_trace_metric_start = time.time()
    overlap_pixels, overlap_ratio, expected_pixels = _cleanup_trace_mask_overlap(cleanup_mask, expected_mask)
    change_metrics = _cleanup_trace_change_metrics(before_image, after_image, cleanup_mask, expected_mask, source_image=source_image)
    source_expected_pixels = change_metrics.get("source_expected_pixels")
    try:
        if source_expected_pixels not in ("", None):
            expected_pixels = int(source_expected_pixels)
            overlap_pixels = int(change_metrics.get("mask_source_overlap_pixels") or overlap_pixels)
            overlap_ratio = float(change_metrics.get("mask_source_overlap_ratio") or overlap_ratio)
    except Exception:
        pass
    changed_ratio = float(change_metrics.get("changed_pixel_ratio_inside_mask") or 0.0)
    residual_ratio = change_metrics.get("source_residual_ratio_after_cleanup")
    mask_pixels = int(operation.get("mask_pixels") or 0)
    failure_class = _cleanup_trace_classification(
        mask_pixels=mask_pixels,
        overlap_ratio=overlap_ratio if expected_pixels > 0 else None,
        changed_ratio=changed_ratio,
        residual_ratio=residual_ratio,
        expected_source_pixels=int(expected_pixels or 0),
    )
    add_timing(
        debug_context,
        "renderer_cleanup_trace_metric_time",
        time.time() - cleanup_trace_metric_start,
    )
    artifact_level = str(debug_context.get("debug_artifact_level") or debug_artifact_level())
    save_trace_images = artifact_level == "full" or (artifact_level == "lite" and failure_class != "passed")
    pre_path = ""
    post_path = ""
    mask_path = ""
    pre_post_map = ""
    original_post_map = ""
    changed_proxy_pixels = 0
    changed_proxy_ratio = 0.0
    cleanup_trace_image_write_start = time.time()
    if save_trace_images:
        pre_path = _save_cleanup_trace_image(debug_context, before_image, crop_box, os.path.join(trace_dir, "pre_clean_crop.png"))
        post_path = _save_cleanup_trace_image(debug_context, after_image, crop_box, os.path.join(trace_dir, "post_clean_crop.png"))
        mask_path = _save_cleanup_trace_mask(debug_context, cleanup_mask, crop_box, os.path.join(trace_dir, "cleanup_mask_crop.png"))
        pre_post_map, changed_proxy_pixels, changed_proxy_ratio = _save_cleanup_residual_map(
            debug_context,
            before_image,
            after_image,
            crop_box,
            os.path.join(trace_dir, "residual_pre_clean_vs_post_clean.png"),
        )
        if source_image is not None:
            original_post_map, _orig_pixels, _orig_ratio = _save_cleanup_residual_map(
                debug_context,
                source_image,
                after_image,
                crop_box,
                os.path.join(trace_dir, "residual_original_vs_post_clean.png"),
            )
    add_timing(
        debug_context,
        "renderer_cleanup_trace_image_write_time",
        time.time() - cleanup_trace_image_write_start,
    )
    trace = {
        "page_id": page_id,
        "cleanup_operation_id": operation_id,
        "operation_kind": operation.get("operation_kind"),
        "region_id": operation.get("region_id"),
        "region_ids": operation.get("region_ids") or [],
        "root_id": operation.get("text_block_root_id"),
        "cleanup_partition_id": operation.get("cleanup_partition_id"),
        "cleanup_partition_scope": operation.get("cleanup_partition_scope"),
        "backend": operation.get("backend"),
        "effective_inpaint_mode": operation.get("effective_inpaint_mode"),
        "cleanup_mask_bbox": operation.get("mask_bbox"),
        "cleanup_mask_pixels": mask_pixels,
        "crop_bbox": operation.get("crop_bbox"),
        "pre_clean_crop_path": pre_path,
        "cleanup_mask_crop_path": mask_path,
        "post_clean_crop_path": post_path,
        "residual_pre_clean_vs_post_clean_path": pre_post_map,
        "residual_original_vs_post_clean_path": original_post_map,
        "changed_pixels_inside_mask": change_metrics.get("changed_pixels_inside_mask") or changed_proxy_pixels,
        "changed_pixel_ratio_inside_mask": changed_ratio or changed_proxy_ratio,
        "expected_source_pixels": expected_pixels,
        "mask_source_overlap_pixels": overlap_pixels,
        "mask_source_overlap_ratio": overlap_ratio if expected_pixels > 0 else "",
        "mask_overlaps_expected_source_glyphs": bool(expected_pixels > 0 and overlap_ratio >= _SOURCE_ERASURE_CONTRACT_COVERAGE_THRESHOLD),
        "source_residual_pixels_after_cleanup": change_metrics.get("source_residual_pixels_after_cleanup"),
        "source_residual_ratio_after_cleanup": residual_ratio,
        "cleanup_operation_failure_class": failure_class,
        "debug_artifact_level": artifact_level,
        "cleanup_trace_images_saved": bool(save_trace_images),
        "cleanup_trace_images_skipped_reason": "" if save_trace_images else f"{artifact_level}_metadata_only_passed_cleanup",
    }
    target_row_id = _phase2e_target_row_id(debug_context, operation=operation)
    if target_row_id:
        trace["phase2e_target_row_id"] = target_row_id
        operation["phase2e_target_row_id"] = target_row_id
    operation.update(trace)
    debug_context.setdefault("cleanup_operation_traces", []).append(trace)
    _phase2e_append_trace(
        debug_context,
        "cleanup_operation_recorded",
        operation=operation,
        meta=((debug_context.get("regions") or {}).get(str(operation.get("region_id") or ""), {}) if isinstance(debug_context.get("regions"), dict) else {}),
        fields={
            "cleanup_expected_mask": _phase2e_mask_summary(expected_mask),
            "cleanup_mask": _phase2e_mask_summary(cleanup_mask),
        },
    )
    return operation


_CLEANUP_EFFECTIVENESS_RETRY_FAILURE_CLASSES = {
    "cleanup_applied_but_residual_remains",
    "cleanup_operation_ineffective",
    "cleanup_operation_noop",
}


def _update_cleanup_operation_trace(debug_context: dict | None, operation_id: str, fields: dict[str, object]) -> None:
    if not debug_context or not operation_id:
        return
    for record in debug_context.get("cleanup_operation_traces") or []:
        if isinstance(record, dict) and str(record.get("cleanup_operation_id") or "") == operation_id:
            record.update(fields)
            return


def _cleanup_trace_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _cleanup_effectiveness_retry_allowed(operation: dict[str, object], cleanup_mask, expected_mask) -> tuple[bool, str]:
    failure_class = str(operation.get("cleanup_operation_failure_class") or "")
    if failure_class not in _CLEANUP_EFFECTIVENESS_RETRY_FAILURE_CLASSES:
        return False, "failure_class_not_targeted"
    operation_kind = str(operation.get("operation_kind") or "")
    if operation_kind not in {"local_text_removal", "pre_render_source_erasure_contract_retry"}:
        return False, "operation_kind_not_retryable"
    if cleanup_mask is None or not np.any(cleanup_mask):
        return False, "empty_cleanup_mask"
    expected_pixels = int(_cleanup_trace_float(operation.get("expected_source_pixels")) or 0)
    if expected_pixels < 18:
        return False, "too_few_expected_source_pixels"
    overlap_ratio = _cleanup_trace_float(operation.get("mask_source_overlap_ratio"))
    if overlap_ratio is None or overlap_ratio < 0.86:
        return False, "mask_source_overlap_too_low"
    residual_ratio = _cleanup_trace_float(operation.get("source_residual_ratio_after_cleanup"))
    if residual_ratio is None or residual_ratio < 0.16:
        return False, "source_residual_below_retry_threshold"
    stats = mask_stats(cleanup_mask) or {}
    bbox = _mask_stats_box(stats)
    if not bbox:
        return False, "cleanup_mask_bbox_missing"
    bbox_area = _xyxy_area(bbox)
    if bbox_area > 180000:
        return False, "cleanup_mask_bbox_too_large_for_retry"
    mask_pixels = int((stats or {}).get("pixels") or 0)
    if mask_pixels > 52000:
        return False, "cleanup_mask_too_large_for_retry"
    if expected_mask is not None and getattr(expected_mask, "shape", None) != getattr(cleanup_mask, "shape", None):
        return False, "expected_mask_shape_mismatch"
    return True, ""


def _cleanup_effectiveness_retry_mask(source_image, current_image, cleanup_mask, expected_mask):
    if Image is None or cv2 is None or np is None or source_image is None or current_image is None or cleanup_mask is None:
        return None, "retry_inputs_missing"
    try:
        source_like = _cleanup_source_like_mask(source_image, cleanup_mask, expected_mask)
        if source_like is None or not np.any(source_like):
            return None, "source_like_mask_missing"
        source_np = np.asarray(source_image.convert("RGB"))
        current_np = np.asarray(current_image.convert("RGB"))
        if source_np.shape != current_np.shape or source_np.shape[:2] != cleanup_mask.shape[:2]:
            return None, "image_shape_mismatch"
        source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_np, cv2.COLOR_RGB2GRAY)
        source_like_bool = source_like > 0
        dark_source = source_gray <= 185
        light_source = source_gray >= 188
        similar_to_source = np.abs(current_gray.astype(np.int16) - source_gray.astype(np.int16)) <= 38
        still_source_colored = (
            (dark_source & (current_gray <= 172))
            | (light_source & (current_gray >= 178))
        )
        residual = (source_like_bool & similar_to_source & still_source_colored).astype(np.uint8) * 255
        residual_pixels = int((residual > 0).sum())
        if residual_pixels < 10:
            return None, "residual_pixels_below_retry_threshold"
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        retry = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, kernel, iterations=1)
        retry = cv2.dilate(retry, kernel, iterations=1)
        stats = mask_stats(cleanup_mask) or {}
        bbox = _mask_stats_box(stats)
        if not bbox:
            return None, "cleanup_mask_bbox_missing"
        x0, y0, x1, y1 = [int(v) for v in bbox]
        pad = max(2, min(6, int(max(x1 - x0, y1 - y0) * 0.025)))
        limiter = np.zeros_like(retry, dtype=np.uint8)
        lx0 = max(0, x0 - pad)
        ly0 = max(0, y0 - pad)
        lx1 = min(retry.shape[1], x1 + pad)
        ly1 = min(retry.shape[0], y1 + pad)
        limiter[ly0:ly1, lx0:lx1] = 255
        retry = cv2.bitwise_and(retry, limiter)
        retry_pixels = int((retry > 0).sum())
        if retry_pixels <= 0:
            return None, "retry_mask_empty_after_limiter"
        if retry_pixels > max(residual_pixels + 2400, int(residual_pixels * 4.5)):
            return None, "retry_mask_growth_too_large"
        if retry_pixels > max(int((cleanup_mask > 0).sum() * 1.15), residual_pixels + 9000):
            return None, "retry_mask_exceeds_cleanup_mask_bound"
        return retry, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _maybe_apply_cleanup_effectiveness_retry(
    *,
    working,
    cleanup_mask,
    expected_mask,
    operation: dict[str, object],
    debug_context: dict | None,
    inpaint_mode: str,
    use_gpu: bool,
    model_id: str,
    region_id: str | None = None,
) -> tuple[object, dict[str, object] | None]:
    if debug_context is None or cv2 is None or np is None:
        return working, None
    allowed, reason = _cleanup_effectiveness_retry_allowed(operation, cleanup_mask, expected_mask)
    operation_id = str(operation.get("cleanup_operation_id") or "")
    region_meta = debug_context.get("regions") or {}
    _phase2e_append_trace(
        debug_context,
        "cleanup_effectiveness_retry_gate",
        region_id=region_id,
        operation=operation,
        meta=(region_meta.get(str(region_id or operation.get("region_id") or ""), {}) if isinstance(region_meta, dict) else {}),
        fields={
            "retry_allowed": allowed,
            "retry_gate_reason": reason,
            "cleanup_mask": _phase2e_mask_summary(cleanup_mask),
            "expected_mask": _phase2e_mask_summary(expected_mask),
        },
    )
    if not allowed:
        if reason not in {"failure_class_not_targeted", "source_residual_below_retry_threshold"}:
            _update_cleanup_operation_trace(
                debug_context,
                operation_id,
                {
                    "cleanup_effectiveness_retry_attempted": False,
                    "cleanup_effectiveness_retry_status": "skipped",
                    "cleanup_effectiveness_retry_reason": reason,
                },
            )
        return working, None
    source_image = _cleanup_trace_source_image(debug_context)
    retry_mask, retry_reason = _cleanup_effectiveness_retry_mask(source_image, working, cleanup_mask, expected_mask)
    if retry_mask is None:
        fields = {
            "cleanup_effectiveness_retry_attempted": True,
            "cleanup_effectiveness_retry_status": "rejected",
            "cleanup_effectiveness_retry_reason": retry_reason,
        }
        operation.update(fields)
        _update_cleanup_operation_trace(debug_context, operation_id, fields)
        _phase2e_append_trace(
            debug_context,
            "cleanup_effectiveness_retry_rejected",
            region_id=region_id,
            operation=operation,
            meta=(region_meta.get(str(region_id or operation.get("region_id") or ""), {}) if isinstance(region_meta, dict) else {}),
            fields={"retry_reason": retry_reason},
        )
        return working, None
    cleanup_start = time.time()
    retry_debug: dict[str, object] = {}
    pre_retry_image = working
    retry_working = _apply_local_text_removal(
        working,
        retry_mask,
        inpaint_mode,
        use_gpu,
        model_id=model_id,
        cleanup_tag="cleanup_effectiveness_retry",
        debug_info=retry_debug,
    )
    cleanup_elapsed = time.time() - cleanup_start
    add_timing(debug_context, "inpainting_time", cleanup_elapsed)
    add_count(debug_context, "inpaint_calls")
    add_count(debug_context, "cleanup_effectiveness_retry_attempted")
    retry_stats = mask_stats(retry_mask) or {}
    retry_bbox = _mask_stats_box(retry_stats)
    retry_operation = {
        "operation_kind": "cleanup_effectiveness_retry",
        "region_id": region_id or operation.get("region_id"),
        "cleanup_mode": "cleanup_effectiveness_retry",
        "cleanup_partition_id": f"{operation.get('cleanup_partition_id') or operation_id}_effectiveness_retry",
        "cleanup_partition_scope": operation.get("cleanup_partition_scope") or "root_local",
        "parent_logical_text_unit_id": operation.get("parent_logical_text_unit_id"),
        "text_block_root_id": operation.get("text_block_root_id") or operation.get("root_id"),
        "text_area_container_id": operation.get("text_area_container_id"),
        "elapsed_sec": round(cleanup_elapsed, 6),
        "backend": retry_debug.get("backend"),
        "backend_detail": retry_debug.get("backend_detail"),
        "requested_inpaint_mode": inpaint_mode,
        "effective_inpaint_mode": retry_debug.get("effective_inpaint_mode"),
        "mask_pixels": int((retry_stats or {}).get("pixels") or 0),
        "mask_bbox": list(retry_bbox) if retry_bbox else None,
        "mask_bbox_area": _xyxy_area(retry_bbox) if retry_bbox else 0,
        "crop_bbox": retry_debug.get("crop_bbox"),
        "crop_area": retry_debug.get("crop_area"),
        "mask_ratio": retry_debug.get("mask_ratio"),
        "cleanup_retry_parent_operation_id": operation_id,
    }
    retry_operation = _record_cleanup_operation_trace(
        debug_context,
        retry_operation,
        pre_retry_image,
        retry_working,
        retry_mask,
        expected_mask=expected_mask,
    )
    retry_class = str(retry_operation.get("cleanup_operation_failure_class") or "")
    retry_succeeded = retry_class == "passed"
    retry_status = "succeeded" if retry_succeeded else "failed_residual_remaining"
    add_count(debug_context, "cleanup_effectiveness_retry_succeeded" if retry_succeeded else "cleanup_effectiveness_retry_failed")
    parent_fields = {
        "cleanup_effectiveness_retry_attempted": True,
        "cleanup_effectiveness_retry_status": retry_status,
        "cleanup_effectiveness_retry_operation_id": retry_operation.get("cleanup_operation_id"),
        "cleanup_effectiveness_retry_mask_pixels": retry_operation.get("cleanup_mask_pixels"),
        "cleanup_effectiveness_retry_residual_ratio": retry_operation.get("source_residual_ratio_after_cleanup"),
        "cleanup_initial_failure_class": operation.get("cleanup_operation_failure_class"),
        "cleanup_operation_failure_class": "cleanup_retry_succeeded" if retry_succeeded else "cleanup_retry_failed_residual_remaining",
    }
    operation.update(parent_fields)
    _update_cleanup_operation_trace(debug_context, operation_id, parent_fields)
    _record_render_cleanup_operation(debug_context, retry_operation)
    if region_id:
        mark_render_region(
            debug_context,
            str(region_id),
            cleanup_effectiveness_retry_attempted=True,
            cleanup_effectiveness_retry_status=retry_status,
            cleanup_effectiveness_retry_operation_id=retry_operation.get("cleanup_operation_id"),
            cleanup_effectiveness_retry_mask_pixels=retry_operation.get("cleanup_mask_pixels"),
            cleanup_effectiveness_retry_residual_ratio=retry_operation.get("source_residual_ratio_after_cleanup"),
        )
    region_meta_after = debug_context.get("regions") or {}
    _phase2e_append_trace(
        debug_context,
        "cleanup_effectiveness_retry_recorded",
        region_id=region_id,
        operation=retry_operation,
        meta=(region_meta_after.get(str(region_id or operation.get("region_id") or ""), {}) if isinstance(region_meta_after, dict) else {}),
        fields={
            "parent_cleanup_operation_id": operation_id,
            "retry_status": retry_status,
            "retry_succeeded": retry_succeeded,
            "region_state_updated": bool(isinstance(region_meta_after, dict) and str(region_id or operation.get("region_id") or "") in region_meta_after),
        },
    )
    return retry_working, retry_operation


def _apply_text_removal(
    image,
    text_mask,
    mode: str,
    use_gpu: bool,
    model_id: str = cleanup_execution.FIXED_CLEANUP_INPAINT_MODEL_ID,
    debug_info: dict | None = None,
):
    raise RuntimeError(
        "renderer cleanup mutation is disabled; cleanup must be completed by "
        "app.pipeline cleanup runtime before render_translations()"
    )


def _apply_local_text_removal(
    image,
    local_mask,
    mode: str,
    use_gpu: bool,
    model_id: str = cleanup_execution.FIXED_CLEANUP_INPAINT_MODEL_ID,
    cleanup_tag: str | None = None,
    debug_info: dict | None = None,
):
    raise RuntimeError(
        "renderer cleanup mutation is disabled; cleanup must be completed by "
        "app.pipeline cleanup runtime before render_translations()"
    )


def _refine_recovered_speech_cleanup_mask(mask):
    if cv2 is None or np is None or mask is None:
        return mask, {}
    try:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    except Exception:
        return mask, {}
    before_pixels = int((mask_u8 > 0).sum())
    if before_pixels <= 0:
        return mask, {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    refined = cv2.dilate(closed, kernel, iterations=1)
    closed_pixels = int((closed > 0).sum())
    after_pixels = int((refined > 0).sum())
    if after_pixels <= before_pixels:
        return mask_u8, {
            "cleanup_mask_refinement": "recovered_speech_connected_component_close",
            "cleanup_mask_refinement_added_pixels": 0,
        }
    if after_pixels > max(before_pixels + 12000, int(before_pixels * 3.2)):
        return closed, {
            "cleanup_mask_refinement": "recovered_speech_connected_component_close_only",
            "cleanup_mask_refinement_added_pixels": max(0, closed_pixels - before_pixels),
            "cleanup_mask_refinement_rejected_reason": "dilation_growth_too_large",
        }
    return refined, {
        "cleanup_mask_refinement": "recovered_speech_connected_component_close_dilate",
        "cleanup_mask_refinement_added_pixels": after_pixels - before_pixels,
    }


def _merge_recovered_speech_cleanup_masks(primary_mask, fallback_mask):
    if cv2 is None or np is None or primary_mask is None or fallback_mask is None:
        return None, {}
    try:
        primary = (primary_mask > 0).astype(np.uint8) * 255
        fallback = (fallback_mask > 0).astype(np.uint8) * 255
    except Exception:
        return None, {}
    primary_pixels = int((primary > 0).sum())
    fallback_pixels = int((fallback > 0).sum())
    if primary_pixels <= 0 or fallback_pixels <= 0:
        return None, {}
    merged = cv2.bitwise_or(primary, fallback)
    merged_pixels = int((merged > 0).sum())
    added = max(0, merged_pixels - primary_pixels)
    if added <= 0:
        return None, {}
    ys, xs = np.where(merged > 0)
    bbox_area = 0
    if ys.size and xs.size:
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if merged_pixels > max(primary_pixels + 18000, int(primary_pixels * 3.8)):
        return None, {
            "source_glyph_mask_fallback_used": False,
            "source_glyph_mask_fallback_reason": "recovered_speech_fallback_mask_growth_too_large",
            "cleanup_mask_fallback_merge_added_pixels": int(added),
        }
    if bbox_area > 180000:
        return None, {
            "source_glyph_mask_fallback_used": False,
            "source_glyph_mask_fallback_reason": "recovered_speech_fallback_mask_bbox_too_large",
            "cleanup_mask_fallback_merge_added_pixels": int(added),
            "cleanup_mask_fallback_merge_bbox_area": int(bbox_area),
        }
    return merged, {
        "source_glyph_mask_fallback_used": True,
        "source_glyph_mask_fallback_reason": "merged_precomputed_and_root_local_recovered_speech_masks",
        "cleanup_mask_fallback_merge_added_pixels": int(added),
        "cleanup_mask_fallback_merge_bbox_area": int(bbox_area),
    }


def _process_fast_inpaint(image, mask):
    if cv2 is None or np is None:
        return image
    img_np = np.array(image)
    kernel_size = max(3, int(max(mask.shape) * 0.0015))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, dilated, 3, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _is_top_row_nonbubble_caption(region, render, region_type: str, flags, bbox, image_h: int) -> bool:
    """Narrow guard for outlined top-row captions over art, not speech bubbles."""
    if region_type != "background_text":
        return False
    if not isinstance(render, dict):
        return False
    if not isinstance(flags, dict) or not flags.get("bg_text"):
        return False
    cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
    if cleanup_mode != "local_text_mask":
        return False
    reason = str(render.get("classification_reason", "") or "").strip().lower()
    if reason not in _TOP_ROW_CAPTION_REASONS:
        return False
    if region.get("bubble_id") or render.get("bubble_id"):
        return False
    page_class = str(region.get("page_class", render.get("page_class", "")) or "").strip().lower()
    if page_class in {"cover", "contents", "title"}:
        return False
    try:
        _x, y, w, h = [int(v) for v in bbox]
    except Exception:
        return False
    if w <= 0 or h <= 0:
        return False
    if h < max(80, int(w * 1.2)):
        return False
    top_limit = max(120, int(max(1, image_h) * 0.22))
    bottom_limit = max(260, int(max(1, image_h) * 0.35))
    return y <= top_limit and y + h <= bottom_limit


def _is_vertical_side_caption(region, render) -> bool:
    return source_glyph_mask_stage.is_vertical_side_caption(region, render)


def _top_row_caption_glyph_mask(img_np, bbox, polygon, dilate_px: int = 1):
    """Build a bbox-bounded mask from source caption glyph pixels instead of a filled box."""
    if cv2 is None or np is None:
        return None
    if img_np is None:
        return None
    try:
        x, y, w, h = [int(v) for v in bbox]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = img_np.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    delta = gray.astype(np.int16) - blur.astype(np.int16)
    contrast = (np.abs(delta) > 18).astype(np.uint8) * 255
    bright = (gray >= 168).astype(np.uint8) * 255
    near_contrast = cv2.dilate(
        contrast,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    near_bright = cv2.dilate(
        bright,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    dark_outline = (((gray <= 95) & (near_bright > 0) & (contrast > 0)).astype(np.uint8)) * 255
    glyph = cv2.bitwise_or(cv2.bitwise_and(bright, near_contrast), dark_outline)
    glyph = cv2.bitwise_or(glyph, (((gray >= 135) & (contrast > 0)).astype(np.uint8)) * 255)

    allowed = np.zeros_like(glyph)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            shifted = []
            for poly in polys:
                shifted.append(np.array([(int(px - x0), int(py - y0)) for px, py in poly], dtype=np.int32))
            cv2.fillPoly(allowed, shifted, 255)
        except Exception:
            allowed[:, :] = 255
    else:
        allowed[:, :] = 255
    glyph = cv2.bitwise_and(glyph, allowed)
    glyph = _filter_top_row_caption_glyph_components(glyph, allowed)
    glyph = cv2.morphologyEx(
        glyph,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        glyph = cv2.dilate(glyph, kernel, iterations=1)
        glyph = cv2.bitwise_and(glyph, allowed)
    min_pixels = max(12, int((x1 - x0) * (y1 - y0) * 0.012))
    if int((glyph > 0).sum()) < min_pixels:
        fallback = cv2.bitwise_and(bright, allowed)
        if dilate_px > 0:
            fallback = cv2.dilate(fallback, kernel, iterations=1)
            fallback = cv2.bitwise_and(fallback, allowed)
        glyph = fallback
    if not np.any(glyph):
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = glyph
    return mask


def _filter_top_row_caption_glyph_components(glyph, allowed):
    if cv2 is None or np is None or glyph is None:
        return glyph
    glyph = cv2.bitwise_and(glyph.astype(np.uint8), allowed.astype(np.uint8))
    if not np.any(glyph):
        return glyph
    crop_area = max(1, int(glyph.shape[0]) * int(glyph.shape[1]))
    raw_pixels = int((glyph > 0).sum())
    if raw_pixels < max(80, int(crop_area * 0.18)):
        return glyph
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(glyph, 8)
    kept = np.zeros_like(glyph)
    crop_h, crop_w = glyph.shape[:2]
    max_component_area = max(1500, int(crop_area * 0.075))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cx = int(stats[idx, cv2.CC_STAT_LEFT])
        cy = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 4 or cw <= 0 or ch <= 0:
            continue
        density = area / max(1, cw * ch)
        spans_width = cw >= int(crop_w * 0.82)
        spans_height = ch >= int(crop_h * 0.82)
        if area > max_component_area and (spans_width or spans_height or density > 0.58):
            continue
        if cw > max(70, int(crop_w * 0.80)) and ch > max(18, int(crop_h * 0.12)):
            continue
        if ch > max(110, int(crop_h * 0.76)) and cw > max(24, int(crop_w * 0.18)):
            continue
        touches_outer = (
            cx <= 1
            or cy <= 1
            or cx + cw >= crop_w - 1
            or cy + ch >= crop_h - 1
        )
        if touches_outer and area > max(180, int(crop_area * 0.020)) and (cw > crop_w * 0.35 or ch > crop_h * 0.35):
            continue
        if density > 0.92 and area > 120:
            continue
        kept[labels == idx] = 255
    kept_pixels = int((kept > 0).sum())
    if kept_pixels < max(12, int(raw_pixels * 0.015)):
        return glyph
    return kept


def _expand_bright_top_row_caption_neighbor_mask(img_np, base_mask, bbox):
    """Add only nearby glyph pixels from bright uniform caption areas missed by detection."""
    if cv2 is None or np is None or img_np is None or base_mask is None:
        return base_mask
    try:
        x, y, w, h = [int(v) for v in bbox]
    except Exception:
        return base_mask
    if w <= 0 or h <= 0:
        return base_mask
    # This is intentionally limited to very narrow caption detections on bright
    # text areas; dark artwork captions keep their bbox-bounded glyph cleanup.
    if w > 45 or h < 110 or h < w * 3.0:
        return base_mask
    img_h, img_w = img_np.shape[:2]
    if y > max(160, int(img_h * 0.28)) or y + h > max(320, int(img_h * 0.38)):
        return base_mask

    combined = base_mask.copy()
    side_pad = max(18, min(48, int(w * 1.6)))
    y_pad = max(4, min(12, int(h * 0.06)))
    for side in ("right", "left"):
        if side == "right":
            sx0 = min(img_w, x + w)
            sx1 = min(img_w, x + w + side_pad)
        else:
            sx0 = max(0, x - side_pad)
            sx1 = max(0, x)
        sy0 = max(0, y - y_pad)
        sy1 = min(img_h, y + h + y_pad)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        crop = img_np[sy0:sy1, sx0:sx1]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if gray.size == 0:
            continue
        mean = float(gray.mean())
        p20 = float(np.percentile(gray, 20))
        p50 = float(np.percentile(gray, 50))
        if mean < 225.0 or p20 < 230.0 or p50 < 248.0:
            continue
        glyph = _bright_caption_neighbor_glyph_mask(gray)
        if glyph is None or not np.any(glyph):
            continue
        combined[sy0:sy1, sx0:sx1] = cv2.bitwise_or(combined[sy0:sy1, sx0:sx1], glyph)
    return combined


def _bright_caption_neighbor_glyph_mask(gray):
    if cv2 is None or np is None:
        return None
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    delta = np.abs(gray.astype(np.int16) - blur.astype(np.int16))
    raw = (((gray <= 108) | ((gray <= 180) & (delta > 18))) & (gray < 245)).astype(np.uint8) * 255
    if not np.any(raw):
        return None
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    crop_area = max(1, raw.shape[0] * raw.shape[1])
    max_component_area = max(120, int(crop_area * 0.035))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 6 or area > max_component_area:
            continue
        if ch > max(55, int(raw.shape[0] * 0.35)):
            continue
        if cw > max(32, int(raw.shape[1] * 0.90)):
            continue
        kept[labels == idx] = 255
    if int((kept > 0).sum()) < 28:
        return None
    kept = cv2.morphologyEx(
        kept,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    kept = cv2.dilate(
        kept,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return kept


def _apply_caption_fill(
    image,
    boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]],
):
    if cv2 is None or np is None or not boxes:
        return image
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    result = img_np.copy()
    for box, fill_color in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        inset = max(2, int(min(bw, bh) * 0.08))
        ix0 = min(x1 - 1, x0 + inset)
        iy0 = min(y1 - 1, y0 + inset)
        ix1 = max(ix0 + 1, x1 - inset)
        iy1 = max(iy0 + 1, y1 - inset)
        if fill_color is None:
            border_samples = _sample_box_border(img_np, x0, y0, x1, y1, pad=max(1, inset // 2))
            if border_samples.size >= 3:
                fill_color = tuple(int(v) for v in np.median(border_samples.reshape(-1, 3), axis=0).tolist())
            else:
                fill_color = _estimate_box_fill(img_np, (x0, y0, x1, y1)) or (32, 32, 32)
        result[iy0:iy1, ix0:ix1] = np.array(fill_color, dtype=np.uint8)
    return Image.fromarray(result)


def _apply_background_fill(
    image,
    boxes: List[Tuple[Tuple[int, int, int, int], Tuple[int, int, int] | None]],
):
    if cv2 is None or np is None or not boxes:
        return image
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    for box, fill_color in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        if fill_color is None:
            samples = _sample_box_border(img_np, x0, y0, x1, y1, pad=3)
            if samples.size < 3:
                samples = img_np[y0:y1, x0:x1].reshape(-1, 3)
            if samples.size < 3:
                continue
            median = np.median(samples, axis=0)
            fill_color = tuple(int(v) for v in median.tolist())
        img_np[y0:y1, x0:x1] = np.array(fill_color, dtype=np.uint8)
    return Image.fromarray(img_np)


def _box_luma_stats(img_np, box: Tuple[int, int, int, int]):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return None
    mean = float(gray.mean())
    p20 = float(np.percentile(gray, 20))
    p80 = float(np.percentile(gray, 80))
    return mean, p20, p80


def _estimate_box_fill(img_np, box: Tuple[int, int, int, int]):
    stats = _box_luma_stats(img_np, box)
    if stats is None:
        return None
    mean, p20, p80 = stats
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    pixels = crop.reshape(-1, 3)
    gray_flat = gray.reshape(-1)
    if p80 < 95:
        mask = gray_flat <= np.percentile(gray_flat, 50)
    elif p20 > 175:
        mask = gray_flat >= np.percentile(gray_flat, 50)
    else:
        mask = slice(None)
    sample = pixels[mask] if not isinstance(mask, slice) else pixels
    if sample.size < 3:
        sample = pixels
    median = np.median(sample.reshape(-1, 3), axis=0)
    return tuple(int(v) for v in median.tolist())

def _sample_box_border(img_np, x0: int, y0: int, x1: int, y1: int, pad: int = 3):
    h, w = img_np.shape[:2]
    pad = max(1, pad)
    ix0 = min(max(x0 + pad, 0), w)
    iy0 = min(max(y0 + pad, 0), h)
    ix1 = max(min(x1 - pad, w), ix0 + 1)
    iy1 = max(min(y1 - pad, h), iy0 + 1)
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(outer, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1)
    cv2.rectangle(inner, (ix0, iy0), (ix1 - 1, iy1 - 1), 255, thickness=-1)
    border = cv2.subtract(outer, inner)
    return img_np[border > 0].reshape(-1, 3)


def _expand_dark_box(img_np, box: Tuple[int, int, int, int]):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img_np.shape[:2]
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    pad = int(max(x1 - x0, y1 - y0) * 1.5)
    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(w, x1 + pad)
    ry1 = min(h, y1 + pad)
    roi = img_np[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    thresh = 120
    _, dark = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels = cv2.connectedComponents(dark)
    if num_labels <= 1:
        return None
    cy = int((y0 + y1) / 2) - ry0
    cx = int((x0 + x1) / 2) - rx0
    if cy < 0 or cx < 0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
        return None
    label = labels[cy, cx]
    if label == 0:
        return None
    ys, xs = np.where(labels == label)
    if ys.size == 0 or xs.size == 0:
        return None
    bx0 = int(xs.min()) + rx0
    by0 = int(ys.min()) + ry0
    bx1 = int(xs.max()) + rx0
    by1 = int(ys.max()) + ry0
    base_area = max(1, (x1 - x0) * (y1 - y0))
    expanded_area = max(1, (bx1 - bx0) * (by1 - by0))
    if expanded_area > base_area * 8:
        return None
    return (bx0, by0, bx1, by1)


def _apply_bubble_fill(image, bubble_mask, text_mask, reference_np):
    raise RuntimeError(
        "renderer cleanup mutation is disabled; cleanup must be completed by "
        "app.pipeline cleanup runtime before render_translations()"
    )


def _resolve_text_color(image, box, default=(0, 0, 0)):
    if ImageStat is None or image is None:
        return default
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    try:
        # Sample a larger area to include the bubble rim/background
        pad = max(10, int(min(x1 - x0, y1 - y0) * 0.15))
        bx0 = max(0, x0 - pad)
        by0 = max(0, y0 - pad)
        bx1 = min(image.width, x1 + pad)
        by1 = min(image.height, y1 + pad)

        crop = image.crop((bx0, by0, bx1, by1))

        # Mask out the center (text area) to only checking the rim
        mask = Image.new("L", crop.size, 255)
        draw = ImageDraw.Draw(mask)
        # Inner box relative to crop
        ix0 = x0 - bx0
        iy0 = y0 - by0
        ix1 = x1 - bx0
        iy1 = y1 - by0
        draw.rectangle((ix0, iy0, ix1, iy1), fill=0)

        stat = ImageStat.Stat(crop, mask)
        if not stat.mean or len(stat.mean) < 3:
            return default

        mean = stat.mean
        lum = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]

        # If background is bright, use black text. If dark, use white.
        if lum > 140:
             return (0, 0, 0)
        elif lum < 100:
             return (255, 255, 255)

    except Exception:
        return default
    return default


_RENDER_STYLE_FLAT_FIELDS = {
    "font_family": "font",
    "font_size": "font_size",
    "font_size_hint": "source_size_hint",
    "font_size_min": "source_size_min",
    "font_size_max": "source_size_max",
    "font_size_locked": "font_size_locked",
    "font_size_policy": "font_size_policy",
    "font_size_fallback_policy": "font_size_fallback_policy",
    "source_orientation": "source_orientation",
    "wrap_mode": "wrap_mode",
    "line_height": "line_height",
    "align": "align",
    "fill_color": "color",
    "stroke_color": "stroke",
    "stroke_width": "stroke_width",
    "style_class": "font_style",
    "font_weight": "font_weight",
    "spacing_profile": "spacing_profile",
}


def _render_with_style_contract(region: dict, render: dict) -> dict:
    style = _render_style_contract(region, render)
    if not style:
        return render
    merged = dict(render)
    merged["render_style"] = style
    for nested_key, flat_key in _RENDER_STYLE_FLAT_FIELDS.items():
        value = style.get(nested_key)
        if _style_value_present(value):
            merged[flat_key] = value
    return merged


def _render_style_contract(region: dict, render: dict) -> dict:
    for source in (render.get("render_style"), region.get("render_style")):
        if isinstance(source, dict):
            style = dict(source)
            style.setdefault("render_style_owner", "parent_execution_bundle")
            return style
    return {}


def _render_style_debug_fields(render: dict) -> dict:
    style = render.get("render_style")
    if not isinstance(style, dict) or not style:
        return {}
    fields = {
        "render_style": dict(style),
        "render_style_owner": style.get("render_style_owner"),
        "render_style_version": style.get("render_style_version"),
        "render_style_source": style.get("render_style_source"),
        "render_style_provider": style.get("render_style_provider"),
        "render_style_provider_model": style.get("render_style_provider_model"),
        "render_style_confidence": style.get("render_style_confidence"),
    }
    for nested_key in (
        "style_class",
        "font_family",
        "font_weight",
        "fill_color",
        "stroke_color",
        "stroke_width",
        "source_orientation",
        "wrap_mode",
        "line_height",
        "font_size",
        "font_size_hint",
        "font_size_min",
        "font_size_max",
        "font_size_locked",
        "font_size_policy",
        "font_size_fallback_policy",
        "spacing_profile",
    ):
        value = style.get(nested_key)
        if _style_value_present(value):
            fields[f"render_style_{nested_key}"] = value
    return {key: value for key, value in fields.items() if value not in (None, "", [])}


def _parent_style_font_size_result_fields(selected_size, resolved_size) -> dict[str, object]:
    try:
        resolved = int(resolved_size or 0)
    except Exception:
        resolved = 0
    try:
        selected = int(selected_size or 0)
    except Exception:
        selected = 0
    if resolved <= 0:
        return {}
    fields: dict[str, object] = {
        "parent_style_font_size_locked": True,
        "parent_style_resolved_font_size": resolved,
        "parent_style_font_size_preserved": selected == resolved,
    }
    if selected > 0 and selected != resolved:
        fields.update(
            {
                "parent_style_font_size_fallback_applied": True,
                "parent_style_font_size_fallback_reason": "resolved_parent_font_size_did_not_fit_execution_region",
                "parent_style_font_size_fallback_selected_size": selected,
            }
        )
    return fields


def _mark_parent_style_font_size_result(diag: dict, resolved_size) -> None:
    if str(diag.get("selected_font_size_kind") or "") == "ellipsis_dot_diameter":
        diag.update(
            {
                "parent_style_font_size_locked": True,
                "parent_style_resolved_font_size": int(resolved_size or 0),
                "parent_style_font_size_preserved": True,
                "parent_style_font_size_audit_note": "ellipsis_renderer_reports_dot_diameter",
            }
        )
        return
    diag.update(_parent_style_font_size_result_fields(diag.get("selected_font_size"), resolved_size))


def _style_value_present(value) -> bool:
    return value is not None and value != "" and value != []


def _parse_color(value) -> Tuple[int, int, int] | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text.startswith("#") or len(text) != 7:
        return None
    try:
        return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))
    except Exception:
        return None


def _region_masks(
    img_np,
    pad_box: Tuple[int, int, int, int],
    bbox: List[int],
    polygon,
):
    if cv2 is None or np is None:
        return None, None
    x0, y0, x1, y1 = pad_box
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
    if white.mean() < 5:
        _, white = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)

    text_mask = np.zeros((white.shape[0], white.shape[1]), dtype=np.uint8)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            for poly in polys:
                poly = np.array(poly, dtype=np.int32)
                poly[:, 0] = poly[:, 0] - x0
                poly[:, 1] = poly[:, 1] - y0
                cv2.fillPoly(text_mask, [poly], 255)
        except Exception:
            text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    else:
        text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    use_mask = white
    text_pixels = int((text_mask > 0).sum())
    if text_pixels:
        _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
        white_overlap = int(((white > 0) & (text_mask > 0)).sum())
        dark_overlap = int(((dark > 0) & (text_mask > 0)).sum())
        if dark_overlap > max(4, white_overlap * 1.2):
            use_mask = dark
    bubble_box, bubble_mask = _find_bubble_box(use_mask, text_mask)
    cleanup_bubble_mask = bubble_mask
    if bubble_mask is not None:
        bubble_pixels = gray[bubble_mask > 0]
        if bubble_pixels.size > 0:
            bubble_mean = float(bubble_pixels.mean())
            if bubble_mean > 165:
                threshold = min(140, bubble_mean - 35)
                stroke = (gray < threshold).astype(np.uint8) * 255
            elif bubble_mean < 90:
                threshold = max(170, bubble_mean + 60)
                stroke = (gray > threshold).astype(np.uint8) * 255
            else:
                stroke = None
            if stroke is not None:
                stroke = cv2.bitwise_and(stroke, bubble_mask)
                stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, kernel, iterations=1)
                stroke = cv2.dilate(stroke, kernel, iterations=1)
                text_mask = cv2.bitwise_or(text_mask, stroke)
        bubble_area = int((bubble_mask > 0).sum())
        if bubble_area > 0:
            erode_px = max(1, int(round(math.sqrt(bubble_area) * 0.012)))
            erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erode_px * 2 + 1, erode_px * 2 + 1),
            )
            candidate_cleanup = cv2.erode(bubble_mask, erode_kernel, iterations=1)
            if candidate_cleanup is not None and candidate_cleanup.any():
                cleanup_bubble_mask = candidate_cleanup
    adaptive_block = max(15, int(min(gray.shape[:2]) / 8) * 2 + 1)
    base_limit = cv2.dilate(text_mask, kernel, iterations=1)
    if cleanup_bubble_mask is not None:
        base_limit = cv2.bitwise_and(base_limit, cleanup_bubble_mask)
    try:
        ink_dark = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            adaptive_block,
            12,
        )
        ink_light = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            12,
        )
        dark_overlap = int(((ink_dark > 0) & (base_limit > 0)).sum())
        light_overlap = int(((ink_light > 0) & (base_limit > 0)).sum())
        if max(dark_overlap, light_overlap) > 5:
            ink = ink_dark if dark_overlap >= light_overlap else ink_light
            ink = cv2.bitwise_and(ink, base_limit)
            ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
            ink = cv2.dilate(ink, kernel, iterations=1)
            text_mask = cv2.bitwise_or(text_mask, ink)
    except Exception:
        pass
    if cleanup_bubble_mask is not None:
        # Expand text removal slightly but keep it strictly inside the bubble area.
        expanded = cv2.dilate(text_mask, kernel, iterations=1)
        text_mask = cv2.bitwise_and(expanded, cleanup_bubble_mask)
    full_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = text_mask
    if bubble_box:
        bx0, by0, bx1, by1 = bubble_box
        bubble_box = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
        bubble_box = _ensure_box_contains(bubble_box, pad_box)
    if bubble_mask is not None:
        full_bubble = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        full_bubble[y0:y1, x0:x1] = cleanup_bubble_mask if cleanup_bubble_mask is not None else bubble_mask
        bubble_mask = full_bubble
    return full_mask, bubble_box, bubble_mask


def _ensure_box_contains(outer, inner):
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        min(ox0, ix0),
        min(oy0, iy0),
        max(ox1, ix1),
        max(oy1, iy1),
    )


def _rect_mask(shape, bbox, x0, y0):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    rx0 = max(0, x - x0)
    ry0 = max(0, y - y0)
    rx1 = min(shape[1], rx0 + w)
    ry1 = min(shape[0], ry0 + h)
    cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), 255, thickness=-1)
    return mask


def _find_bubble_box(white_mask, text_mask):
    num_labels, labels = cv2.connectedComponents(white_mask)
    best_label = 0
    best_overlap = 0
    if num_labels <= 1:
        return None, None
    for label in range(1, num_labels):
        overlap = np.sum((labels == label) & (text_mask > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_label == 0:
        return None, None
    text_pixels = int((text_mask > 0).sum())
    ys, xs = np.where(labels == best_label)
    if ys.size == 0 or xs.size == 0:
        return None, None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    area = (x1 - x0) * (y1 - y0)
    total_area = white_mask.shape[0] * white_mask.shape[1]
    if total_area > 0 and area / total_area > 0.6:
        return None, None
    if text_pixels and area > max(text_pixels * 25, 16000):
        return None, None
    bubble_mask = (labels == best_label).astype(np.uint8) * 255
    return (x0, y0, x1, y1), bubble_mask


def _intersect_box(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _limit_box(box, limit):
    bx0, by0, bx1, by1 = box
    lx0, ly0, lx1, ly1 = limit
    limited = _intersect_box(box, limit)
    if not limited:
        return box
    # If the limit is significantly smaller, use it; otherwise keep the original.
    bw = bx1 - bx0
    bh = by1 - by0
    lw = limited[2] - limited[0]
    lh = limited[3] - limited[1]
    if lw * lh < bw * bh * 0.85:
        return limited
    return box


def _normalize_polygons(polygon) -> List[List[List[float]]]:
    if polygon is None:
        return []
    if hasattr(polygon, "tolist"):
        try:
            polygon = polygon.tolist()
        except Exception:
            return []
    if isinstance(polygon, (list, tuple)) and len(polygon) > 0:
        first = polygon[0]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
            return [polygon]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            return [p for p in polygon if p]
    return []


def _polygon_bounds(polygon) -> Tuple[int, int, int, int] | None:
    polys = _normalize_polygons(polygon)
    if not polys:
        return None
    xs = []
    ys = []
    for poly in polys:
        for point in poly:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _box_area(box) -> int:
    if not box:
        return 0
    if len(box) == 4:
        return int(max(1, box[2]) * max(1, box[3]))
    return 0


def _shrink_box(box, padding):
    x0, y0, x1, y1 = box
    x0 = x0 + padding
    y0 = y0 + padding
    x1 = x1 - padding
    y1 = y1 - padding
    if x1 <= x0 or y1 <= y0:
        return box
    return (x0, y0, x1, y1)


def _region_source_box_xyxy(region: dict) -> tuple[int, int, int, int] | None:
    try:
        bbox = region.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = [int(v) for v in bbox[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def _mask_stats_box(stats) -> tuple[int, int, int, int] | None:
    if not isinstance(stats, dict):
        return None
    box = stats.get("bbox")
    if not box:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _preserved_text_obstacle_boxes(regions: List[Dict[str, object]]) -> list[dict[str, object]]:
    obstacles: list[dict[str, object]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_type = str(region.get("type", "") or "").strip().lower()
        flags = region.get("flags", {}) or {}
        render = region.get("render") or {}
        if not isinstance(render, dict):
            render = {}
        cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
        if region_type not in {"decorative_text", "sfx"} and not flags.get("sfx"):
            continue
        if cleanup_mode != "preserve":
            continue
        if not flags.get("ignore") and str(region.get("translation", "") or "").strip():
            continue
        box = _region_source_box_xyxy(region)
        if box is None:
            continue
        obstacles.append(
            {
                "region_id": str(region.get("region_id", "") or ""),
                "box": box,
            }
        )
    return obstacles


def _xyxy_area(box) -> int:
    if not box:
        return 0
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return 0
    return max(0, x1 - x0) * max(0, y1 - y0)


def _xyxy_intersection_area(a, b) -> int:
    if not a or not b:
        return 0
    try:
        ax0, ay0, ax1, ay1 = [int(v) for v in a[:4]]
        bx0, by0, bx1, by1 = [int(v) for v in b[:4]]
    except Exception:
        return 0
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _substantial_xyxy_overlap(candidate, obstacle) -> bool:
    overlap = _xyxy_intersection_area(candidate, obstacle)
    if overlap <= 0:
        return False
    candidate_area = max(1, _xyxy_area(candidate))
    obstacle_area = max(1, _xyxy_area(obstacle))
    smaller_area = max(1, min(candidate_area, obstacle_area))
    return (
        overlap / smaller_area >= 0.12
        or (overlap / candidate_area >= 0.08 and overlap / obstacle_area >= 0.08)
    )


def _clamp_box_away_from_obstacle(render_box, source_box, obstacle_box):
    rx0, ry0, rx1, ry1 = [int(v) for v in render_box[:4]]
    sx0, sy0, sx1, sy1 = [int(v) for v in source_box[:4]]
    ox0, oy0, ox1, oy1 = [int(v) for v in obstacle_box[:4]]
    margin = 2
    candidates = []
    if ox1 <= sx0:
        candidates.append((max(rx0, ox1 + margin), ry0, rx1, ry1))
    if ox0 >= sx1:
        candidates.append((rx0, ry0, min(rx1, ox0 - margin), ry1))
    if oy1 <= sy0:
        candidates.append((rx0, max(ry0, oy1 + margin), rx1, ry1))
    if oy0 >= sy1:
        candidates.append((rx0, ry0, rx1, min(ry1, oy0 - margin)))
    valid = []
    source_area = max(1, _xyxy_area(source_box))
    source_w = max(1, sx1 - sx0)
    source_h = max(1, sy1 - sy0)
    for candidate in candidates:
        cx0, cy0, cx1, cy1 = candidate
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) < int(source_w * 0.65) or (cy1 - cy0) < int(source_h * 0.65):
            continue
        if _xyxy_area(candidate) < int(source_area * 0.55):
            continue
        valid.append(candidate)
    if not valid:
        return render_box
    return min(valid, key=lambda candidate: _xyxy_intersection_area(candidate, obstacle_box))


def _avoid_preserved_text_obstacles(
    render_box,
    source_box,
    cleanup_box,
    obstacles: list[dict[str, object]],
    current_region_id: str,
):
    if not obstacles:
        return render_box, []
    adjusted = tuple(int(v) for v in render_box[:4])
    source_box = tuple(int(v) for v in source_box[:4])
    cleanup_box = tuple(int(v) for v in cleanup_box[:4]) if cleanup_box else None
    adjusted_ids: list[str] = []
    for obstacle in obstacles:
        obstacle_id = str(obstacle.get("region_id", "") or "")
        if obstacle_id and obstacle_id == current_region_id:
            continue
        obstacle_box = obstacle.get("box")
        if not obstacle_box:
            continue
        obstacle_box = tuple(int(v) for v in obstacle_box[:4])
        if not _substantial_xyxy_overlap(adjusted, obstacle_box):
            continue
        if _substantial_xyxy_overlap(source_box, obstacle_box):
            continue
        if cleanup_box is not None and _substantial_xyxy_overlap(cleanup_box, obstacle_box):
            continue
        clamped = _clamp_box_away_from_obstacle(adjusted, source_box, obstacle_box)
        if clamped == adjusted:
            clamped = source_box
        if _xyxy_intersection_area(clamped, obstacle_box) < _xyxy_intersection_area(adjusted, obstacle_box):
            adjusted = clamped
            if obstacle_id:
                adjusted_ids.append(obstacle_id)
    return adjusted, adjusted_ids


def _apply_text_area_speech_container_constraint(
    render_box,
    source_box,
    region: dict,
    render: dict,
    text: str,
    img_w: int,
    img_h: int,
    *,
    preferred_size: int | None = None,
    min_size: int | None = None,
):
    limit_box, limit_source = _text_area_speech_container_limit_box(region, render, img_w, img_h)
    if limit_box is None:
        return render_box, None
    current = tuple(int(v) for v in render_box[:4])
    outside = _outside_ratio(current, limit_box)
    limit_area = max(1, _xyxy_area(limit_box))
    current_area = max(1, _xyxy_area(current))
    source_inside = _inside_ratio(tuple(int(v) for v in source_box[:4]), limit_box)
    recovered_anchor = _is_recovered_logical_speech_anchor(region, render)
    outside_threshold = 0.08 if recovered_anchor else 0.25
    area_factor = 1.08 if recovered_anchor else 1.25
    source_inside_threshold = 0.45 if recovered_anchor else 0.55
    if outside < outside_threshold and current_area <= int(limit_area * area_factor):
        return render_box, None
    if source_inside < source_inside_threshold:
        return render_box, None
    constrained = _bubble_inner_layout_box(
        None,
        limit_box,
        source_box,
        text,
        img_w,
        img_h,
        preferred_size=preferred_size,
        min_size=min_size,
    )
    if constrained is None:
        target_w, target_h = _minimum_vertical_box_size(
            source_box,
            text,
            preferred_size=preferred_size,
            min_size=min_size,
        )
        constrained = _expand_box_to_size_within_limit(
            source_box,
            target_w,
            target_h,
            limit_box,
            center=((source_box[0] + source_box[2]) / 2.0, (source_box[1] + source_box[3]) / 2.0),
        )
    if constrained is None or tuple(int(v) for v in constrained[:4]) == current:
        return render_box, None
    constrained = tuple(int(v) for v in constrained[:4])
    return constrained, {
        "applied": True,
        "source": "TextAreaPlan",
        "limit_source": limit_source,
        "previous_final_render_bbox": list(current),
        "new_final_render_bbox": list(constrained),
        "text_area_allowed_bbox": list(limit_box),
        "outside_text_area_before": round(outside, 3),
        "source_inside_text_area": round(source_inside, 3),
        "reason": "speech_render_box_constrained_to_text_area_container",
    }


def _text_area_speech_container_limit_box(region: dict, render: dict, img_w: int, img_h: int):
    if not isinstance(region, dict):
        return None, ""
    if not isinstance(render, dict):
        render = {}
    parent_box, parent_source = _parent_execution_hard_layout_box(region, render, img_w, img_h)
    if parent_box is not None:
        return parent_box, parent_source
    source_recovered = _is_recovered_logical_speech_anchor(region, render)
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "").strip()
    route_intent = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "").strip()
    if container_type != "speech_bubble" or route_intent not in {"", "translate_speech"}:
        return None, ""
    conflict_flags = _canonical_region_render_list(region, render, "text_area_conflict_flags")
    if any(str(flag).strip() for flag in conflict_flags):
        return None, ""
    tier = str(_canonical_region_render_value(region, render, "text_area_confidence_tier") or "").strip()
    reason_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_reason_codes"))
    strong_container = (
        tier in {"strong_model_container", "mask_primary_container"}
        or "speech_mask_container" in reason_text
    )
    if not strong_container:
        return None, ""

    candidates = []
    if source_recovered:
        candidates.append(
            (
                "logical_text_source_reconstruction_crop_bbox",
                _canonical_region_render_value(region, render, "logical_text_source_reconstruction_crop_bbox"),
            )
        )
    ownership_status = str(_canonical_region_render_value(region, render, "logical_text_ownership_status") or "").strip()
    if source_recovered or ownership_status in {"block_anchor", "standalone_block"}:
        candidates.append(
            (
                "logical_text_block_allowed_bbox",
                _canonical_region_render_value(region, render, "logical_text_block_allowed_bbox"),
            )
        )
    if source_recovered or ownership_status in {"block_anchor", "standalone_block"}:
        candidates.append(
            (
                "text_area_container_bbox",
                _canonical_region_render_value(region, render, "text_area_container_bbox"),
            )
        )
    for source, bbox in candidates:
        xyxy = _xywh_value_to_xyxy(bbox, img_w, img_h)
        if xyxy is None:
            continue
        area = _xyxy_area(xyxy)
        if area <= 0 or area > int(max(1, img_w * img_h) * 0.22):
            continue
        return xyxy, source
    return None, ""


def _is_parent_execution_bundle_region(region: dict, render: dict) -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    markers = (
        _canonical_region_render_value(region, render, "parent_execution_bundle_id"),
        _canonical_region_render_value(region, render, "parent_execution_bundle_version"),
        _canonical_region_render_value(region, render, "parent_execution_authoritative"),
        _canonical_region_render_value(region, render, "execution_region_authority"),
        _canonical_region_render_value(region, render, "execution_region_role"),
        _canonical_region_render_value(region, render, "logical_text_ownership_status"),
    )
    for marker in markers:
        text = str(marker or "").strip()
        if text in {
            "parent_execution_bundle",
            "parent_execution_bundle_v1",
            "parent_execution",
        }:
            return True
        if text.lower() == "true":
            return True
    return False


def _parent_execution_hard_layout_box(region: dict, render: dict, img_w: int, img_h: int):
    if not _is_parent_execution_bundle_region(region, render):
        return None, ""
    for source in (
        "render_allowed_area",
        "parent_logical_text_unit_render_allowed_area",
        "logical_text_block_allowed_bbox",
        "bbox",
    ):
        xyxy = _xywh_value_to_xyxy(
            _canonical_region_render_value(region, render, source),
            img_w,
            img_h,
        )
        if xyxy is None:
            continue
        if _xyxy_area(xyxy) <= 0:
            continue
        return xyxy, source
    return None, ""


def _is_recovered_logical_speech_anchor(region: dict, render: dict) -> bool:
    return source_glyph_mask_stage.is_recovered_logical_speech_anchor(region, render)


def _is_logical_text_cleanup_anchor(region: dict, render: dict) -> bool:
    return source_glyph_mask_stage.is_logical_text_cleanup_anchor(region, render)


def _canonical_region_render_value(region: dict, render: dict, key: str):
    if isinstance(region, dict) and key in region:
        value = region.get(key)
        if value not in (None, ""):
            return value
    if isinstance(render, dict) and key in render:
        value = render.get(key)
        if value not in (None, ""):
            return value
    return None


def _canonical_region_render_list(region: dict, render: dict, key: str) -> list:
    if isinstance(region, dict) and key in region:
        value = region.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value not in (None, ""):
            return [value]
    if isinstance(render, dict) and key in render:
        value = render.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value not in (None, ""):
            return [value]
    return []


def _render_allowed_area_box_for_audit(
    region: dict,
    render: dict,
    region_type: str,
    img_w: int,
    img_h: int,
):
    parent_box, parent_source = _parent_execution_hard_layout_box(region, render, img_w, img_h)
    if parent_box is not None:
        return parent_box, parent_source
    if region_type == "speech_bubble":
        return _text_area_speech_container_limit_box(region, render, img_w, img_h)
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "").strip()
    if container_type != "caption_background":
        return None, ""
    xyxy = _xywh_value_to_xyxy(_canonical_region_render_value(region, render, "text_area_container_bbox"), img_w, img_h)
    if xyxy is None:
        return None, ""
    return xyxy, "text_area_container_bbox"


def _apply_root_local_render_fit_pass(
    render_box,
    allowed_area_box,
    region: dict,
    render: dict,
    source_glyph_erasure_fields: dict | None,
    cleanup_box,
    text: str,
    region_type: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
):
    current = tuple(int(v) for v in render_box[:4])
    chars = len(_meaningful_render_chars(text))
    before_density = chars / max(1.0, _xyxy_area(current) / 1000.0)
    micro_start = _renderer_micro_start()
    before_layout = _score_render_layout_v2_candidate(
        current,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=False,
        allowed_area_box=allowed_area_box,
    )
    _renderer_micro_add("renderer_before_v2_score_time", micro_start)
    fields: dict[str, object] = {
        "render_fit_action_attempted": False,
        "render_fit_action_status": "not_required",
        "render_fit_before_density": round(float(before_density), 3),
        "render_fit_after_density": round(float(before_density), 3),
        "render_fit_before_bbox": list(current),
        "render_fit_after_bbox": list(current),
        "render_fit_rejection_reason": "",
        "render_fit_compact_layout": False,
        "render_layout_v2_attempted": False,
        "render_layout_v2_status": "not_required",
        "render_layout_v2_before_score": before_layout.get("score"),
        "render_layout_v2_after_score": before_layout.get("score"),
        "render_layout_v2_before_fit_ratio": before_layout.get("fit_ratio"),
        "render_layout_v2_after_fit_ratio": before_layout.get("fit_ratio"),
        "render_layout_v2_line_height_scale": round(float(line_height_scale), 3),
        "render_layout_v2_compact_layout": False,
        "render_layout_v2_selected_font_size": before_layout.get("selected_font_size"),
        "render_layout_v2_rejection_reason": "",
        "render_layout_v3_attempted": False,
        "render_layout_v3_status": "not_required",
        "render_layout_v3_before_score": None,
        "render_layout_v3_after_score": None,
        "render_layout_v3_candidate_count": 0,
        "render_layout_v3_selected_candidate": {},
        "render_layout_v3_rejection_reason": "",
        "render_layout_v3_shape_source": "",
        "render_layout_v3_edge_contact_before": None,
        "render_layout_v3_edge_contact_after": None,
        "render_layout_v3_density_before": round(float(before_density), 3),
        "render_layout_v3_density_after": round(float(before_density), 3),
        "render_readability_v4_attempted": False,
        "render_readability_v4_status": "not_required",
        "render_readability_v4_before_score": None,
        "render_readability_v4_after_score": None,
        "render_readability_v4_candidate_count": 0,
        "render_readability_v4_selected_candidate": {},
        "render_readability_v4_unresolved_reason": "",
        "render_readability_v4_edge_contact_before": None,
        "render_readability_v4_edge_contact_after": None,
        "render_readability_v4_density_before": round(float(before_density), 3),
        "render_readability_v4_density_after": round(float(before_density), 3),
        "render_readability_v4_shape_source": "",
        "render_readability_v4_final_class": "render_readability_not_evaluated",
        "render_readability_v5_attempted": False,
        "render_readability_v5_status": "not_required",
        "render_readability_v5_before_score": None,
        "render_readability_v5_after_score": None,
        "render_readability_v5_candidate_count": 0,
        "render_readability_v5_selected_candidate": {},
        "render_readability_v5_source_column_count": 0,
        "render_readability_v5_shape_source": "",
        "render_readability_v5_density_before": round(float(before_density), 3),
        "render_readability_v5_density_after": round(float(before_density), 3),
        "render_readability_v5_edge_contact_before": None,
        "render_readability_v5_edge_contact_after": None,
        "render_readability_v5_final_class": "render_readability_v5_resolved",
        "render_readability_v5_unresolved_reason": "",
    }
    route = str(
        _canonical_region_render_value(region, render, "text_area_route_intent")
        or _canonical_region_render_value(region, render, "route_intent")
        or ""
    ).strip()
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "").strip()
    eligible = (
        region_type == "speech_bubble"
        or container_type in {"speech_bubble", "caption_background"}
        or route in {"translate_speech", "translate_caption", "translate_caption_background"}
    )
    if not eligible:
        fields["render_fit_action_status"] = "not_applicable_preserve_or_nontext_root"
        fields["render_layout_v2_status"] = "not_applicable_preserve_or_nontext_root"
        fields["render_layout_v3_status"] = "not_applicable_preserve_or_nontext_root"
        fields["render_readability_v4_status"] = "not_applicable_preserve_or_nontext_root"
        fields["render_readability_v4_final_class"] = "render_readability_not_applicable"
        fields["render_readability_v5_status"] = "not_applicable_preserve_or_nontext_root"
        fields["render_readability_v5_final_class"] = "render_readability_v5_resolved"
        return current, fields
    if allowed_area_box is None:
        fields["render_fit_action_status"] = "rejected"
        fields["render_fit_rejection_reason"] = "missing_root_or_container_allowed_area"
        fields["render_layout_v2_status"] = "unresolved_warning"
        fields["render_layout_v2_rejection_reason"] = "missing_root_or_container_allowed_area"
        fields["render_layout_v3_status"] = "unresolved_warning"
        fields["render_layout_v3_rejection_reason"] = "missing_root_or_container_allowed_area"
        fields["render_readability_v4_status"] = "unresolved_warning"
        fields["render_readability_v4_unresolved_reason"] = "missing_root_or_container_allowed_area"
        fields["render_readability_v4_final_class"] = "render_readability_unresolved_geometry_limit"
        fields["render_readability_v5_status"] = "unresolved_warning"
        fields["render_readability_v5_unresolved_reason"] = "missing_root_or_container_allowed_area"
        fields["render_readability_v5_final_class"] = "render_readability_v5_unresolved_geometry_limit"
        return current, fields
    allowed = tuple(int(v) for v in allowed_area_box[:4])
    if _xyxy_area(allowed) <= 0:
        fields["render_fit_action_status"] = "rejected"
        fields["render_fit_rejection_reason"] = "invalid_root_or_container_allowed_area"
        fields["render_layout_v2_status"] = "unresolved_warning"
        fields["render_layout_v2_rejection_reason"] = "invalid_root_or_container_allowed_area"
        fields["render_layout_v3_status"] = "unresolved_warning"
        fields["render_layout_v3_rejection_reason"] = "invalid_root_or_container_allowed_area"
        fields["render_readability_v4_status"] = "unresolved_warning"
        fields["render_readability_v4_unresolved_reason"] = "invalid_root_or_container_allowed_area"
        fields["render_readability_v4_final_class"] = "render_readability_unresolved_geometry_limit"
        fields["render_readability_v5_status"] = "unresolved_warning"
        fields["render_readability_v5_unresolved_reason"] = "invalid_root_or_container_allowed_area"
        fields["render_readability_v5_final_class"] = "render_readability_v5_unresolved_geometry_limit"
        return current, fields
    ax0, ay0, ax1, ay1 = allowed
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    inset_x = max(1, min(8, int(aw * 0.025)))
    inset_y = max(1, min(8, int(ah * 0.025)))
    target = (ax0 + inset_x, ay0 + inset_y, ax1 - inset_x, ay1 - inset_y)
    if target[2] <= target[0] or target[3] <= target[1]:
        target = allowed
    shape_box, shape_source = _shape_aware_render_layout_v3_box(current, target, allowed, region, render, region_type)
    micro_start = _renderer_micro_start()
    before_v3 = _score_render_layout_v3_candidate(
        current,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=False,
        allowed_area_box=allowed,
        shape_source=shape_source,
    )
    _renderer_micro_add("renderer_before_v3_score_time", micro_start)
    micro_start = _renderer_micro_start()
    before_v4 = _score_render_readability_v4_candidate(
        current,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=False,
        allowed_area_box=allowed,
        shape_box=shape_box,
        shape_source=shape_source,
    )
    _renderer_micro_add("renderer_before_v4_score_time", micro_start)
    micro_start = _renderer_micro_start()
    source_columns = _render_readability_v5_source_columns(
        current,
        allowed,
        shape_box,
        source_glyph_erasure_fields or {},
        cleanup_box,
        text,
        source_orientation,
        source_size_hint,
    )
    _renderer_micro_add("renderer_layout_source_column_generation_time", micro_start)
    _renderer_micro_add("renderer_fit_pass_candidate_generation_time", micro_start)
    micro_start = _renderer_micro_start()
    before_v5 = _score_render_readability_v5_candidate(
        current,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=False,
        allowed_area_box=allowed,
        shape_box=shape_box,
        shape_source=shape_source,
        source_columns=source_columns,
    )
    _renderer_micro_add("renderer_before_v5_score_time", micro_start)
    fields.update(
        {
            "render_layout_v3_before_score": before_v3.get("score"),
            "render_layout_v3_after_score": before_v3.get("score"),
            "render_layout_v3_shape_source": shape_source,
            "render_layout_v3_edge_contact_before": before_v3.get("edge_contact"),
            "render_layout_v3_edge_contact_after": before_v3.get("edge_contact"),
            "render_layout_v3_density_before": before_v3.get("density"),
            "render_layout_v3_density_after": before_v3.get("density"),
            "render_readability_v4_before_score": before_v4.get("score"),
            "render_readability_v4_after_score": before_v4.get("score"),
            "render_readability_v4_shape_source": shape_source,
            "render_readability_v4_edge_contact_before": before_v4.get("edge_contact"),
            "render_readability_v4_edge_contact_after": before_v4.get("edge_contact"),
            "render_readability_v4_density_before": before_v4.get("density"),
            "render_readability_v4_density_after": before_v4.get("density"),
            "render_readability_v4_final_class": before_v4.get("final_class") or "render_readability_not_required",
            "render_readability_v5_before_score": before_v5.get("score"),
            "render_readability_v5_after_score": before_v5.get("score"),
            "render_readability_v5_source_column_count": int(source_columns.get("count") or 0),
            "render_readability_v5_shape_source": before_v5.get("shape_source") or shape_source,
            "render_readability_v5_density_before": before_v5.get("density"),
            "render_readability_v5_density_after": before_v5.get("density"),
            "render_readability_v5_edge_contact_before": before_v5.get("edge_contact"),
            "render_readability_v5_edge_contact_after": before_v5.get("edge_contact"),
            "render_readability_v5_final_class": before_v5.get("final_class") or "render_readability_v5_resolved",
        }
    )
    current_area = max(1, _xyxy_area(current))
    target_area = max(1, _xyxy_area(target))
    outside = _outside_ratio(current, allowed)
    area_gain = target_area / current_area
    density_high = before_density >= 0.18
    before_fit_ratio = float(before_layout.get("fit_ratio") or 0.0)
    before_edge_contact = float(before_v3.get("edge_contact") or 0.0)
    before_v4_score = float(before_v4.get("score") or 0.0)
    short_text_low_risk = (
        chars < 10
        and before_edge_contact < 0.75
        and before_density < 0.25
    )
    if short_text_low_risk:
        fields["render_fit_action_status"] = "not_required_short_text_low_risk"
        fields["render_layout_v2_status"] = "not_required_short_text_low_risk"
        fields["render_layout_v3_status"] = "not_required_short_text_low_risk"
        fields["render_readability_v4_status"] = "not_required_short_text_low_risk"
        fields["render_readability_v4_final_class"] = "render_readability_not_required"
        fields["render_readability_v5_status"] = "not_required_short_text_low_risk"
        fields["render_readability_v5_final_class"] = "render_readability_v5_resolved"
        return current, fields
    before_v5_score = float(before_v5.get("score") or 0.0)
    tight_box = area_gain >= 1.06 or outside > 0.01 or before_fit_ratio >= 0.92
    if not density_high and not tight_box and before_edge_contact < 0.5 and before_v4_score < 15.0 and before_v5_score < 12.0:
        fields["render_fit_action_status"] = "not_required_density_within_budget"
        fields["render_layout_v2_status"] = "not_required_density_within_budget"
        fields["render_layout_v3_status"] = "not_required_density_within_budget"
        fields["render_readability_v4_status"] = "not_required_density_within_budget"
        fields["render_readability_v4_final_class"] = "render_readability_not_required"
        fields["render_readability_v5_status"] = "not_required_density_within_budget"
        fields["render_readability_v5_final_class"] = "render_readability_v5_resolved"
        return current, fields
    fields["render_fit_action_attempted"] = True
    fields["render_layout_v2_attempted"] = True
    fields["render_layout_v3_attempted"] = True
    fields["render_readability_v4_attempted"] = True
    fields["render_readability_v5_attempted"] = True
    if target_area < current_area * 0.92 and before_edge_contact < 0.75 and before_fit_ratio < 0.96:
        fields["render_fit_action_status"] = "rejected"
        fields["render_fit_rejection_reason"] = "allowed_area_would_shrink_render_box"
        fields["render_fit_compact_layout"] = density_high
        fields["render_layout_v2_status"] = "unresolved_warning"
        fields["render_layout_v2_rejection_reason"] = "allowed_area_would_shrink_render_box"
        fields["render_layout_v3_status"] = "unresolved_warning"
        fields["render_layout_v3_rejection_reason"] = "allowed_area_would_shrink_render_box"
        fields["render_readability_v4_status"] = "unresolved_warning"
        fields["render_readability_v4_unresolved_reason"] = "allowed_area_would_shrink_render_box"
        fields["render_readability_v4_final_class"] = "render_readability_unresolved_geometry_limit"
        fields["render_readability_v5_status"] = "unresolved_warning"
        fields["render_readability_v5_unresolved_reason"] = "allowed_area_would_shrink_render_box"
        fields["render_readability_v5_final_class"] = "render_readability_v5_unresolved_geometry_limit"
        return current, fields
    if area_gain > 1.85 and outside <= 0.01:
        # Keep the change root-local but avoid swallowing a whole bubble when a
        # normal render box only needs a modest fit improvement.
        cx0, cy0, cx1, cy1 = current
        pad_x = min(max(4, int((cx1 - cx0) * 0.10)), max(0, min(cx0 - ax0, ax1 - cx1)))
        pad_y = min(max(4, int((cy1 - cy0) * 0.10)), max(0, min(cy0 - ay0, ay1 - cy1)))
        target = (
            max(ax0 + inset_x, cx0 - pad_x),
            max(ay0 + inset_y, cy0 - pad_y),
            min(ax1 - inset_x, cx1 + pad_x),
            min(ay1 - inset_y, cy1 + pad_y),
        )
        target_area = max(1, _xyxy_area(target))
    if target_area <= current_area * 1.02 and outside <= 0.01 and before_edge_contact < 0.75 and before_fit_ratio < 0.96:
        target = current
        target_area = current_area
    micro_start = _renderer_micro_start()
    layout_choice = _choose_render_readability_v5_candidate(
        current,
        target,
        allowed,
        shape_box,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        before_v5,
        shape_source,
        source_columns,
    )
    _renderer_micro_add("renderer_fit_pass_candidate_scoring_time", micro_start)
    chosen = tuple(layout_choice.get("bbox") or target)
    after_density = chars / max(1.0, max(1, _xyxy_area(chosen)) / 1000.0)
    score_gain = float(before_v5.get("score") or 0.0) - float(layout_choice.get("score") or 0.0)
    changed_box = tuple(chosen) != tuple(current)
    changed_layout = bool(layout_choice.get("compact_layout")) or abs(float(layout_choice.get("line_height_scale") or line_height_scale) - line_height_scale) >= 0.01
    unresolved = bool(layout_choice.get("warning_unresolved"))
    v5_final_class = _render_readability_v5_final_class(
        "unresolved_warning" if unresolved else "applied",
        "layout_unfit_review_only" if unresolved else "",
        layout_choice,
    )
    v5_is_dense_watch = v5_final_class == "render_readability_v5_accepted_complete_dense_watch"
    if not changed_box and not changed_layout and score_gain < 0.5:
        final_class = _render_readability_v4_final_class(
            "unresolved_warning",
            "no_effective_root_local_layout_candidate",
            before_v4,
        )
        v5_final_class = _render_readability_v5_final_class(
            "unresolved_warning",
            "no_effective_root_local_layout_candidate",
            before_v5,
        )
        v5_status = "accepted_complete_dense_watch" if v5_final_class == "render_readability_v5_accepted_complete_dense_watch" else "unresolved_warning"
        v5_reason = "" if v5_status == "accepted_complete_dense_watch" else "no_effective_root_local_layout_candidate"
        fields["render_fit_action_status"] = "rejected"
        fields["render_fit_rejection_reason"] = "no_effective_root_local_layout_candidate"
        fields["render_layout_v2_status"] = "unresolved_warning"
        fields["render_layout_v2_rejection_reason"] = "no_effective_root_local_layout_candidate"
        fields["render_layout_v3_status"] = "unresolved_warning"
        fields["render_layout_v3_rejection_reason"] = "no_effective_root_local_layout_candidate"
        fields["render_fit_compact_layout"] = density_high
        fields["render_readability_v4_status"] = "unresolved_warning"
        fields["render_readability_v4_unresolved_reason"] = "no_effective_root_local_layout_candidate"
        fields["render_readability_v4_after_score"] = before_v4.get("score")
        fields["render_readability_v4_candidate_count"] = layout_choice.get("candidate_count")
        fields["render_readability_v4_selected_candidate"] = layout_choice.get("selected_candidate") or {}
        fields["render_readability_v4_final_class"] = final_class
        fields["render_readability_v5_status"] = v5_status
        fields["render_readability_v5_unresolved_reason"] = v5_reason
        fields["render_readability_v5_after_score"] = before_v5.get("score")
        fields["render_readability_v5_candidate_count"] = layout_choice.get("candidate_count")
        fields["render_readability_v5_selected_candidate"] = layout_choice.get("selected_candidate") or {}
        fields["render_readability_v5_edge_contact_after"] = before_v5.get("edge_contact")
        fields["render_readability_v5_density_after"] = before_v5.get("density")
        fields["render_readability_v5_shape_source"] = before_v5.get("shape_source") or shape_source
        fields["render_readability_v5_final_class"] = v5_final_class
        return current, fields
    layout_status = "applied"
    if unresolved:
        layout_status = "unresolved_warning"
    unresolved_reason = "layout_unfit_review_only" if unresolved else ""
    final_class = _render_readability_v4_final_class(layout_status, unresolved_reason, layout_choice)
    v5_status = layout_status
    v5_reason = unresolved_reason
    if v5_is_dense_watch:
        v5_status = "accepted_complete_dense_watch"
        v5_reason = ""
    fields.update(
        {
            "render_fit_action_status": (
                "layout_v5_candidate_applied_dense_watch"
                if v5_is_dense_watch and (changed_box or changed_layout)
                else ("layout_v2_candidate_applied" if layout_status == "applied" else "layout_v2_unresolved_warning")
            ),
            "render_fit_after_density": round(float(after_density), 3),
            "render_fit_after_bbox": list(chosen),
            "render_fit_density_improvement": round(float(before_density - after_density), 3),
            "render_fit_area_gain": round(float(max(1, _xyxy_area(chosen)) / current_area), 3),
            "render_fit_compact_layout": bool(layout_choice.get("compact_layout")),
            "render_layout_v2_status": layout_status,
            "render_layout_v2_after_score": layout_choice.get("v2_score", layout_choice.get("score")),
            "render_layout_v2_after_fit_ratio": layout_choice.get("fit_ratio"),
            "render_layout_v2_line_height_scale": layout_choice.get("line_height_scale"),
            "render_layout_v2_compact_layout": bool(layout_choice.get("compact_layout")),
            "render_layout_v2_selected_font_size": layout_choice.get("selected_font_size"),
            "render_layout_v2_rejection_reason": (
                unresolved_reason
                if unresolved
                else ""
            ),
            "render_layout_v3_status": layout_status,
            "render_layout_v3_after_score": layout_choice.get("v3_score", layout_choice.get("score")),
            "render_layout_v3_candidate_count": layout_choice.get("candidate_count"),
            "render_layout_v3_selected_candidate": layout_choice.get("selected_candidate") or {},
            "render_layout_v3_rejection_reason": (
                unresolved_reason
                if unresolved
                else ""
            ),
            "render_layout_v3_shape_source": layout_choice.get("shape_source") or shape_source,
            "render_layout_v3_edge_contact_after": layout_choice.get("edge_contact"),
            "render_layout_v3_density_after": layout_choice.get("density"),
            "render_readability_v4_status": layout_status,
            "render_readability_v4_after_score": layout_choice.get("v4_score", layout_choice.get("score")),
            "render_readability_v4_candidate_count": layout_choice.get("candidate_count"),
            "render_readability_v4_selected_candidate": layout_choice.get("selected_candidate") or {},
            "render_readability_v4_unresolved_reason": unresolved_reason,
            "render_readability_v4_edge_contact_after": layout_choice.get("edge_contact"),
            "render_readability_v4_density_after": layout_choice.get("density"),
            "render_readability_v4_shape_source": layout_choice.get("shape_source") or shape_source,
            "render_readability_v4_final_class": final_class,
            "render_readability_v5_status": v5_status,
            "render_readability_v5_after_score": layout_choice.get("score"),
            "render_readability_v5_candidate_count": layout_choice.get("candidate_count"),
            "render_readability_v5_selected_candidate": layout_choice.get("selected_candidate") or {},
            "render_readability_v5_unresolved_reason": v5_reason,
            "render_readability_v5_edge_contact_after": layout_choice.get("edge_contact"),
            "render_readability_v5_density_after": layout_choice.get("density"),
            "render_readability_v5_shape_source": layout_choice.get("shape_source") or shape_source,
            "render_readability_v5_final_class": v5_final_class,
        }
    )
    return chosen, fields


def _choose_render_layout_v2_candidate(
    current,
    expanded,
    allowed,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    before_layout: dict[str, object],
) -> dict[str, object]:
    boxes = [tuple(current)]
    for box in (tuple(expanded), tuple(allowed)):
        if _xyxy_area(box) > 0 and box not in boxes and _outside_ratio(box, allowed) <= 0.001:
            boxes.append(box)
    line_heights = [float(line_height_scale)]
    for scale in (0.94, 0.88):
        candidate = max(0.78, min(1.2, float(line_height_scale) * scale))
        if all(abs(candidate - existing) > 0.01 for existing in line_heights):
            line_heights.append(candidate)
    best = dict(before_layout)
    best["bbox"] = tuple(current)
    best["compact_layout"] = False
    best["line_height_scale"] = round(float(line_height_scale), 3)
    for box in boxes:
        for compact in (False, True):
            for lh in line_heights:
                score = _score_render_layout_v2_candidate(
                    box,
                    text,
                    region_font,
                    lh,
                    wrap_mode,
                    source_orientation,
                    source_size_hint,
                    source_size_min,
                    source_size_max,
                    compact_layout=compact,
                    allowed_area_box=allowed,
                )
                score["bbox"] = tuple(box)
                score["compact_layout"] = compact
                score["line_height_scale"] = round(float(lh), 3)
                if float(score.get("score") or 9999.0) < float(best.get("score") or 9999.0):
                    best = score
    return best


def _shape_aware_render_layout_v3_box(current, target, allowed, region: dict, render: dict, region_type: str):
    ax0, ay0, ax1, ay1 = [int(v) for v in allowed[:4]]
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "").strip()
    route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "").strip()
    if region_type == "speech_bubble" or container_type == "speech_bubble" or route == "translate_speech":
        inset_x = max(2, min(16, int(aw * 0.045)))
        inset_y = max(2, min(14, int(ah * 0.040)))
        source = "speech_container_ellipse_inner_rect"
    elif container_type == "caption_background" or route in {"translate_caption", "translate_caption_background"}:
        inset_x = max(1, min(8, int(aw * 0.025)))
        inset_y = max(1, min(8, int(ah * 0.025)))
        source = "caption_background_root_inner_rect"
    else:
        inset_x = max(1, min(8, int(aw * 0.025)))
        inset_y = max(1, min(8, int(ah * 0.025)))
        source = "text_area_allowed_rect"
    shape_box = (ax0 + inset_x, ay0 + inset_y, ax1 - inset_x, ay1 - inset_y)
    if shape_box[2] <= shape_box[0] or shape_box[3] <= shape_box[1]:
        return tuple(target), "text_area_allowed_rect"
    if _xyxy_area(shape_box) < _xyxy_area(current) * 0.82:
        return tuple(target), source + "_fallback_target"
    return shape_box, source


def _edge_contact_ratio_xyxy(box, allowed) -> float:
    if not box or not allowed:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    ax0, ay0, ax1, ay1 = [int(v) for v in allowed[:4]]
    threshold = max(3, min(14, int(min(max(1, ax1 - ax0), max(1, ay1 - ay0)) * 0.035)))
    touches = 0
    if abs(x0 - ax0) <= threshold:
        touches += 1
    if abs(y0 - ay0) <= threshold:
        touches += 1
    if abs(x1 - ax1) <= threshold:
        touches += 1
    if abs(y1 - ay1) <= threshold:
        touches += 1
    return min(1.0, touches / 4.0)


def _choose_render_layout_v3_candidate(
    current,
    expanded,
    allowed,
    shape_box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    before_layout: dict[str, object],
    shape_source: str,
) -> dict[str, object]:
    boxes = [tuple(current)]
    for box in (tuple(expanded), tuple(shape_box), tuple(allowed)):
        if _xyxy_area(box) > 0 and box not in boxes and _outside_ratio(box, allowed) <= 0.001:
            boxes.append(box)
    # Add an intermediate box between the current render and the shape-aware
    # safe area. This often reduces edge contact without jumping to the whole
    # root/container rectangle.
    sx0, sy0, sx1, sy1 = [int(v) for v in tuple(shape_box)[:4]]
    cx0, cy0, cx1, cy1 = [int(v) for v in tuple(current)[:4]]
    mid = (
        max(sx0, int(round(cx0 * 0.45 + sx0 * 0.55))),
        max(sy0, int(round(cy0 * 0.45 + sy0 * 0.55))),
        min(sx1, int(round(cx1 * 0.45 + sx1 * 0.55))),
        min(sy1, int(round(cy1 * 0.45 + sy1 * 0.55))),
    )
    if _xyxy_area(mid) > 0 and mid not in boxes and _outside_ratio(mid, allowed) <= 0.001:
        boxes.append(mid)
    line_heights = [float(line_height_scale)]
    for scale in (1.0, 0.94, 0.88, 0.82):
        candidate = max(0.76, min(1.2, float(line_height_scale) * scale))
        if all(abs(candidate - existing) > 0.01 for existing in line_heights):
            line_heights.append(candidate)
    best = dict(before_layout)
    best["bbox"] = tuple(current)
    best["compact_layout"] = False
    best["line_height_scale"] = round(float(line_height_scale), 3)
    best["shape_source"] = shape_source
    candidate_count = 0
    micro_start = _renderer_micro_start()
    for box in boxes:
        for compact in (False, True):
            for lh in line_heights:
                candidate_count += 1
                score = _score_render_layout_v3_candidate(
                    box,
                    text,
                    region_font,
                    lh,
                    wrap_mode,
                    source_orientation,
                    source_size_hint,
                    source_size_min,
                    source_size_max,
                    compact_layout=compact,
                    allowed_area_box=allowed,
                    shape_source=shape_source,
                )
                score["bbox"] = tuple(box)
                score["compact_layout"] = compact
                score["line_height_scale"] = round(float(lh), 3)
                if float(score.get("score") or 9999.0) < float(best.get("score") or 9999.0):
                    best = score
    _renderer_micro_add("renderer_v3_candidate_scoring_loop_time", micro_start)
    best["candidate_count"] = candidate_count
    best["selected_candidate"] = {
        "bbox": list(best.get("bbox") or []),
        "compact_layout": bool(best.get("compact_layout")),
        "line_height_scale": best.get("line_height_scale"),
        "fit_ratio": best.get("fit_ratio"),
        "density": best.get("density"),
        "edge_contact": best.get("edge_contact"),
        "wrapped_count": best.get("wrapped_count"),
        "shape_source": best.get("shape_source") or shape_source,
    }
    return best


def _choose_render_readability_v4_candidate(
    current,
    expanded,
    allowed,
    shape_box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    before_layout: dict[str, object],
    shape_source: str,
) -> dict[str, object]:
    boxes = _render_readability_v4_boxes(current, expanded, allowed, shape_box)
    line_heights = [float(line_height_scale)]
    for scale in (1.0, 0.96, 0.92, 0.88, 0.82, 0.76, 0.72):
        candidate = max(0.72, min(1.2, float(line_height_scale) * scale))
        if all(abs(candidate - existing) > 0.01 for existing in line_heights):
            line_heights.append(candidate)
    best = dict(before_layout)
    best["bbox"] = tuple(current)
    best["compact_layout"] = False
    best["line_height_scale"] = round(float(line_height_scale), 3)
    best["shape_source"] = shape_source
    candidate_count = 0
    micro_start = _renderer_micro_start()
    for box in boxes:
        for compact in (False, True):
            for lh in line_heights:
                candidate_count += 1
                score = _score_render_readability_v4_candidate(
                    box,
                    text,
                    region_font,
                    lh,
                    wrap_mode,
                    source_orientation,
                    source_size_hint,
                    source_size_min,
                    source_size_max,
                    compact_layout=compact,
                    allowed_area_box=allowed,
                    shape_box=shape_box,
                    shape_source=shape_source,
                )
                score["bbox"] = tuple(box)
                score["compact_layout"] = compact
                score["line_height_scale"] = round(float(lh), 3)
                if float(score.get("score") or 9999.0) < float(best.get("score") or 9999.0):
                    best = score
    _renderer_micro_add("renderer_v4_candidate_scoring_loop_time", micro_start)
    best["candidate_count"] = candidate_count
    best["selected_candidate"] = {
        "bbox": list(best.get("bbox") or []),
        "compact_layout": bool(best.get("compact_layout")),
        "line_height_scale": best.get("line_height_scale"),
        "fit_ratio": best.get("fit_ratio"),
        "density": best.get("density"),
        "edge_contact": best.get("edge_contact"),
        "wrapped_count": best.get("wrapped_count"),
        "selected_font_size": best.get("selected_font_size"),
        "v3_score": best.get("v3_score"),
        "v4_score": best.get("score"),
        "shape_source": best.get("shape_source") or shape_source,
        "final_class": best.get("final_class"),
    }
    return best


def _render_readability_v4_boxes(current, expanded, allowed, shape_box) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []

    def add_box(box) -> None:
        if not box:
            return
        candidate = tuple(int(v) for v in box[:4])
        if _xyxy_area(candidate) <= 0:
            return
        if _outside_ratio(candidate, allowed) > 0.001:
            return
        if candidate not in boxes:
            boxes.append(candidate)

    current = tuple(int(v) for v in current[:4])
    expanded = tuple(int(v) for v in expanded[:4])
    allowed = tuple(int(v) for v in allowed[:4])
    shape_box = tuple(int(v) for v in shape_box[:4])
    for box in (current, expanded, shape_box, allowed):
        add_box(box)

    ax0, ay0, ax1, ay1 = allowed
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    for frac in (0.035, 0.055, 0.075):
        ix = max(1, min(22, int(aw * frac)))
        iy = max(1, min(20, int(ah * frac)))
        add_box((ax0 + ix, ay0 + iy, ax1 - ix, ay1 - iy))

    cx0, cy0, cx1, cy1 = current
    for frac in (0.06, 0.12, 0.20, 0.30):
        pad_x = max(1, int((cx1 - cx0) * frac))
        pad_y = max(1, int((cy1 - cy0) * frac))
        add_box((max(ax0, cx0 - pad_x), max(ay0, cy0 - pad_y), min(ax1, cx1 + pad_x), min(ay1, cy1 + pad_y)))

    sx0, sy0, sx1, sy1 = shape_box
    for weight in (0.35, 0.55, 0.75):
        add_box(
            (
                int(round(cx0 * (1.0 - weight) + sx0 * weight)),
                int(round(cy0 * (1.0 - weight) + sy0 * weight)),
                int(round(cx1 * (1.0 - weight) + sx1 * weight)),
                int(round(cy1 * (1.0 - weight) + sy1 * weight)),
            )
        )
    return boxes


def _score_render_readability_v4_candidate(
    box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    *,
    compact_layout: bool,
    allowed_area_box,
    shape_box,
    shape_source: str,
) -> dict[str, object]:
    micro_start = _renderer_micro_start()
    _renderer_micro_count_key(
        "renderer_score_readability_v4",
        (
            tuple(int(v) for v in box[:4]),
            str(text or ""),
            str(region_font or ""),
            round(float(line_height_scale or 0.0), 4),
            str(wrap_mode or ""),
            str(source_orientation or ""),
            int(source_size_hint or 0),
            int(source_size_min or 0),
            int(source_size_max or 0),
            bool(compact_layout),
            tuple(int(v) for v in allowed_area_box[:4]) if allowed_area_box else None,
            tuple(int(v) for v in shape_box[:4]) if shape_box else None,
            str(shape_source or ""),
        ),
    )
    base = _score_render_layout_v3_candidate(
        box,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=compact_layout,
        allowed_area_box=allowed_area_box,
        shape_source=shape_source,
    )
    chars = len(_meaningful_render_chars(text))
    edge_contact = float(base.get("edge_contact") or 0.0)
    density = float(base.get("density") or 0.0)
    fit_ratio = float(base.get("fit_ratio") or 0.0)
    selected_font_size = int(base.get("selected_font_size") or 0)
    wrapped_count = int(base.get("wrapped_count") or 0)
    allowed_area = max(1, _xyxy_area(allowed_area_box))
    area_ratio = max(0.0, min(1.5, _xyxy_area(box) / allowed_area))
    shape_edge = _edge_contact_ratio_xyxy(box, shape_box) if shape_box else edge_contact
    score = float(base.get("score") or 0.0)
    score += max(0.0, edge_contact - 0.25) * 9.0
    score += max(0.0, shape_edge - 0.35) * 4.0
    score += max(0.0, density - 0.24) * 85.0
    score += max(0.0, fit_ratio - 0.94) * 55.0
    if wrapped_count >= 5 and chars >= 10:
        score += (wrapped_count - 4) * 2.4
    if selected_font_size and selected_font_size < 30 and chars >= 10:
        score += (30 - selected_font_size) * 0.9
    if chars < 10 and edge_contact >= 0.75:
        score += 4.0
    if area_ratio >= 0.92 and edge_contact >= 0.75:
        score += 2.0
    warning_unresolved = bool(
        score >= 15.0
        or density >= 0.36
        or fit_ratio >= 1.0
        or (edge_contact >= 0.75 and chars >= 8)
        or (selected_font_size and selected_font_size < 28 and chars >= 12)
    )
    enriched = dict(base)
    enriched.update(
        {
            "v3_score": base.get("score"),
            "score": round(float(score), 3),
            "edge_contact": round(float(edge_contact), 3),
            "shape_edge_contact": round(float(shape_edge), 3),
            "area_ratio": round(float(area_ratio), 3),
            "shape_source": shape_source,
            "warning_unresolved": warning_unresolved,
        }
    )
    enriched["final_class"] = _render_readability_v4_final_class(
        "unresolved_warning" if warning_unresolved else "applied",
        "layout_unfit_review_only" if warning_unresolved else "",
        enriched,
    )
    _renderer_micro_add("renderer_score_readability_v4_time", micro_start)
    return enriched


def _render_readability_v4_final_class(status: str, reason: str, score_data: dict[str, object]) -> str:
    status = str(status or "")
    reason = str(reason or "")
    if status.startswith("not_applicable"):
        return "render_readability_not_applicable"
    if status.startswith("not_required"):
        return "render_readability_not_required"
    score = float(score_data.get("score") or 0.0)
    density = float(score_data.get("density") or 0.0)
    fit_ratio = float(score_data.get("fit_ratio") or 0.0)
    edge_contact = float(score_data.get("edge_contact") or 0.0)
    selected_font_size = int(score_data.get("selected_font_size") or 0)
    wrapped_count = int(score_data.get("wrapped_count") or 0)
    if status == "applied" and not bool(score_data.get("warning_unresolved")):
        return "render_readability_resolved"
    if any(token in reason for token in ("missing_root", "invalid_root", "allowed_area", "shrink_render_box")):
        return "render_readability_unresolved_geometry_limit"
    if density >= 0.34 or wrapped_count >= 5 or (selected_font_size and selected_font_size < 28):
        return "render_readability_unresolved_text_too_dense"
    if edge_contact >= 0.75 or fit_ratio >= 1.0:
        return "render_readability_unresolved_geometry_limit"
    if score >= 15.0:
        return "render_readability_unresolved_text_too_dense"
    return "render_readability_unresolved_renderer_policy"


def _render_readability_v5_source_columns(
    current,
    allowed,
    shape_box,
    source_glyph_erasure_fields: dict[str, object],
    cleanup_box,
    text: str,
    source_orientation: str,
    source_size_hint: int,
) -> dict[str, object]:
    boxes: list[tuple[int, int, int, int]] = []
    allowed = tuple(int(v) for v in allowed[:4])
    shape_box = tuple(int(v) for v in shape_box[:4]) if shape_box else allowed

    def parse_box(value) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            box = tuple(int(round(float(v))) for v in value[:4])
        except Exception:
            return None
        if _xyxy_area(box) <= 0:
            return None
        clipped = (
            max(allowed[0], min(allowed[2], box[0])),
            max(allowed[1], min(allowed[3], box[1])),
            max(allowed[0], min(allowed[2], box[2])),
            max(allowed[1], min(allowed[3], box[3])),
        )
        if _xyxy_area(clipped) <= 0:
            return None
        return clipped

    for key in (
        "source_glyph_erasure_bbox",
        "cleanup_mask_bbox",
        "source_child_cleanup_bbox",
        "source_child_cleanup_bboxes",
    ):
        value = source_glyph_erasure_fields.get(key) if isinstance(source_glyph_erasure_fields, dict) else None
        if key.endswith("bboxes") and isinstance(value, list):
            for item in value:
                parsed = parse_box(item)
                if parsed and parsed not in boxes:
                    boxes.append(parsed)
            continue
        parsed = parse_box(value)
        if parsed and parsed not in boxes:
            boxes.append(parsed)
    parsed_cleanup = parse_box(cleanup_box)
    if parsed_cleanup and parsed_cleanup not in boxes:
        boxes.append(parsed_cleanup)
    if not boxes:
        parsed_current = parse_box(current)
        if parsed_current:
            boxes.append(parsed_current)

    if boxes:
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        source_union = (x0, y0, x1, y1)
    else:
        source_union = tuple(int(v) for v in current[:4])
    sw = max(1, source_union[2] - source_union[0])
    sh = max(1, source_union[3] - source_union[1])
    meaningful_chars = max(1, len(_meaningful_render_chars(text)))
    size_basis = int(source_size_hint or 0)
    if size_basis <= 0:
        size_basis = max(18, min(40, int(round((sw + sh) / 14.0))))
    vertical = str(source_orientation or "").strip().lower() == "vertical" or sh >= sw * 1.15
    if vertical:
        row_capacity = max(1, int(sh / max(12, size_basis * 1.05)))
        count_by_text = int(math.ceil(meaningful_chars / max(1, row_capacity)))
        count_by_width = int(math.ceil(sw / max(18, size_basis * 1.28)))
        column_count = max(1, min(5, max(count_by_text, min(count_by_width, count_by_text + 1))))
    else:
        column_count = 1

    return {
        "count": int(column_count),
        "boxes": [list(box) for box in boxes],
        "union": list(source_union),
        "source": "source_glyph_or_cleanup_bbox" if boxes else "current_render_box",
        "vertical": bool(vertical),
        "shape_box": list(shape_box),
    }


def _choose_render_readability_v5_candidate(
    current,
    expanded,
    allowed,
    shape_box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    before_layout: dict[str, object],
    shape_source: str,
    source_columns: dict[str, object],
) -> dict[str, object]:
    boxes = _render_readability_v5_boxes(current, expanded, allowed, shape_box, source_columns, source_size_hint)
    fast_layout_enabled = _renderer_fast_layout_enabled()
    line_heights = [float(line_height_scale)]
    line_height_scales = (
        (1.0, 0.96, 0.92, 0.84, 0.76)
        if fast_layout_enabled
        else (1.0, 0.98, 0.96, 0.94, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72)
    )
    for scale in line_height_scales:
        candidate = max(0.72, min(1.2, float(line_height_scale) * scale))
        if all(abs(candidate - existing) > 0.01 for existing in line_heights):
            line_heights.append(candidate)
    shadow_candidates = None
    if _renderer_shadow_context() is not None or _renderer_fast_layout_shadow_context() is not None:
        shadow_candidates = _renderer_shadow_build_candidates(
            boxes,
            line_heights,
            source_orientation=source_orientation,
            source_columns=source_columns,
        )
    best = dict(before_layout)
    best["bbox"] = tuple(current)
    best["compact_layout"] = False
    best["line_height_scale"] = round(float(line_height_scale), 3)
    best["shape_source"] = shape_source
    candidate_count = 0
    micro_start = _renderer_micro_start()
    if fast_layout_enabled:
        if shadow_candidates is None:
            shadow_candidates = _renderer_shadow_build_candidates(
                boxes,
                line_heights,
                source_orientation=source_orientation,
                source_columns=source_columns,
            )
        best, fast_meta = _choose_render_readability_fast_candidate(
            current,
            expanded,
            allowed,
            shape_box,
            text,
            region_font,
            line_height_scale,
            wrap_mode,
            source_orientation,
            source_size_hint,
            source_size_min,
            source_size_max,
            before_layout,
            shape_source,
            source_columns,
            boxes=boxes,
            line_heights=line_heights,
            candidates=shadow_candidates,
        )
        candidate_count = int(best.get("candidate_count") or 0)
        best["render_readability_v5_fast_layout_enabled"] = True
        best["render_readability_v5_fast_layout_meta"] = fast_meta
        shadow_loop_time = 0.0
    else:
        shadow_loop_start = time.perf_counter() if shadow_candidates is not None else 0.0
        for box in boxes:
            for compact in (False, True):
                for lh in line_heights:
                    candidate_count += 1
                    score = _score_render_readability_v5_candidate(
                        box,
                        text,
                        region_font,
                        lh,
                        wrap_mode,
                        source_orientation,
                        source_size_hint,
                        source_size_min,
                        source_size_max,
                        compact_layout=compact,
                        allowed_area_box=allowed,
                        shape_box=shape_box,
                        shape_source=shape_source,
                        source_columns=source_columns,
                    )
                    score["bbox"] = tuple(box)
                    score["compact_layout"] = compact
                    score["line_height_scale"] = round(float(lh), 3)
                    if float(score.get("score") or 9999.0) < float(best.get("score") or 9999.0):
                        best = score
        shadow_loop_time = time.perf_counter() - shadow_loop_start if shadow_loop_start > 0 else 0.0
    _renderer_micro_add("renderer_v5_candidate_scoring_loop_time", micro_start)
    best["candidate_count"] = candidate_count
    shadow_recording_enabled = (
        _renderer_shadow_context() is not None
        or _renderer_fast_layout_shadow_context() is not None
    )
    if shadow_candidates is not None and shadow_recording_enabled:
        selected_box = _renderer_shadow_box(best.get("bbox") or current) or _renderer_shadow_box(current)
        source_fp = _renderer_shadow_source_columns_fp(source_columns)
        selected_shadow = {
            "bbox": selected_box,
            "bbox_fp": _renderer_shadow_box_fp(selected_box),
            "rounded_bbox_fp": _renderer_shadow_rounded_box_fp(selected_box),
            "compact": bool(best.get("compact_layout")),
            "line_height": round(float(best.get("line_height_scale") or line_height_scale), 3),
            "line_height_index": -1,
            "orientation": str(source_orientation or ""),
            "source_columns_fp": source_fp,
            "area": _renderer_shadow_box_area(selected_box),
        }
        selected_shadow["fingerprint"] = _renderer_shadow_candidate_fp(selected_shadow)
        selected_rank = 0
        for candidate in shadow_candidates:
            if str(candidate.get("fingerprint") or "") == str(selected_shadow.get("fingerprint") or ""):
                selected_rank = int(candidate.get("index") or 0) + 1
                selected_shadow["line_height_index"] = int(candidate.get("line_height_index") or 0)
                break
        selected_shadow["score_summary"] = {
            "score": best.get("score"),
            "fit_ratio": best.get("fit_ratio"),
            "density": best.get("density"),
            "edge_contact": best.get("edge_contact"),
            "outside_ratio": best.get("outside_ratio"),
            "wrapped_count": best.get("wrapped_count"),
            "selected_font_size": best.get("selected_font_size"),
            "source_column_count": best.get("source_column_count"),
            "final_class": best.get("final_class"),
        }
        if _renderer_shadow_context() is not None:
            _renderer_candidate_shadow_record(
                text=text,
                source_orientation=source_orientation,
                candidates=shadow_candidates,
                selected=selected_shadow,
                selected_rank=selected_rank,
                candidate_loop_time=shadow_loop_time,
            )
        if _renderer_fast_layout_shadow_context() is not None:
            fast_choice, fast_meta = _choose_render_readability_fast_candidate(
                current,
                expanded,
                allowed,
                shape_box,
                text,
                region_font,
                line_height_scale,
                wrap_mode,
                source_orientation,
                source_size_hint,
                source_size_min,
                source_size_max,
                before_layout,
                shape_source,
                source_columns,
                boxes=boxes,
                line_heights=line_heights,
                candidates=shadow_candidates,
            )
            fast_box = _renderer_shadow_box(fast_choice.get("bbox") or current) or _renderer_shadow_box(current)
            fast_shadow = {
                "bbox": fast_box,
                "bbox_fp": _renderer_shadow_box_fp(fast_box),
                "compact": bool(fast_choice.get("compact_layout")),
                "line_height": round(float(fast_choice.get("line_height_scale") or line_height_scale), 3),
                "orientation": str(source_orientation or ""),
                "source_columns_fp": source_fp,
            }
            fast_fingerprint = _renderer_shadow_candidate_fp(fast_shadow)
            _renderer_fast_layout_oracle_record(
                text=text,
                source_orientation=source_orientation,
                exhaustive=best,
                fast=fast_choice,
                exhaustive_fingerprint=str(selected_shadow.get("fingerprint") or ""),
                fast_fingerprint=fast_fingerprint,
                exhaustive_loop_time=shadow_loop_time,
                fast_meta=fast_meta,
            )
    best["selected_candidate"] = {
        "bbox": list(best.get("bbox") or []),
        "compact_layout": bool(best.get("compact_layout")),
        "line_height_scale": best.get("line_height_scale"),
        "fit_ratio": best.get("fit_ratio"),
        "density": best.get("density"),
        "edge_contact": best.get("edge_contact"),
        "wrapped_count": best.get("wrapped_count"),
        "selected_font_size": best.get("selected_font_size"),
        "source_column_count": best.get("source_column_count"),
        "v4_score": best.get("v4_score"),
        "v5_score": best.get("score"),
        "shape_source": best.get("shape_source") or shape_source,
        "final_class": best.get("final_class"),
    }
    return best


def _render_readability_v5_boxes(
    current,
    expanded,
    allowed,
    shape_box,
    source_columns: dict[str, object],
    source_size_hint: int,
) -> list[tuple[int, int, int, int]]:
    boxes = _render_readability_v4_boxes(current, expanded, allowed, shape_box)
    allowed = tuple(int(v) for v in allowed[:4])
    shape_box = tuple(int(v) for v in shape_box[:4]) if shape_box else allowed

    def add_box(box) -> None:
        if not box:
            return
        candidate = tuple(int(v) for v in box[:4])
        if _xyxy_area(candidate) <= 0:
            return
        if _outside_ratio(candidate, allowed) > 0.001:
            return
        if candidate not in boxes:
            boxes.append(candidate)

    union_value = source_columns.get("union") if isinstance(source_columns, dict) else None
    if isinstance(union_value, (list, tuple)) and len(union_value) >= 4:
        source_box = tuple(int(v) for v in union_value[:4])
        if _xyxy_area(source_box) > 0:
            source_w = max(1, source_box[2] - source_box[0])
            source_h = max(1, source_box[3] - source_box[1])
            current_w = max(1, int(current[2]) - int(current[0]))
            current_h = max(1, int(current[3]) - int(current[1]))
            count = max(1, int(source_columns.get("count") or 1))
            size_basis = max(18, int(source_size_hint or min(42, max(18, source_h / 7))))
            if bool(source_columns.get("vertical")):
                target_w = max(current_w, int(count * size_basis * 1.22), int(source_w * 1.04))
                target_h = max(current_h, int(source_h * 1.04))
                add_box(_expand_box_to_size_within_limit(source_box, target_w, target_h, shape_box))
                add_box(_expand_box_to_size_within_limit(source_box, target_w, max(target_h, int((shape_box[3] - shape_box[1]) * 0.96)), shape_box))
            else:
                add_box(_expand_box_to_size_within_limit(source_box, max(current_w, int(source_w * 1.06)), max(current_h, int(source_h * 1.08)), shape_box))
    return boxes


def _score_render_readability_v5_candidate(
    box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    *,
    compact_layout: bool,
    allowed_area_box,
    shape_box,
    shape_source: str,
    source_columns: dict[str, object],
) -> dict[str, object]:
    micro_start = _renderer_micro_start()
    _renderer_micro_count_key(
        "renderer_score_readability_v5",
        (
            tuple(int(v) for v in box[:4]),
            str(text or ""),
            str(region_font or ""),
            round(float(line_height_scale or 0.0), 4),
            str(wrap_mode or ""),
            str(source_orientation or ""),
            int(source_size_hint or 0),
            int(source_size_min or 0),
            int(source_size_max or 0),
            bool(compact_layout),
            tuple(int(v) for v in allowed_area_box[:4]) if allowed_area_box else None,
            tuple(int(v) for v in shape_box[:4]) if shape_box else None,
            str(shape_source or ""),
            repr(source_columns or {}),
        ),
    )
    base = _score_render_readability_v4_candidate(
        box,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=compact_layout,
        allowed_area_box=allowed_area_box,
        shape_box=shape_box,
        shape_source=shape_source,
    )
    chars = len(_meaningful_render_chars(text))
    density = float(base.get("density") or 0.0)
    fit_ratio = float(base.get("fit_ratio") or 0.0)
    edge_contact = float(base.get("edge_contact") or 0.0)
    selected_font_size = int(base.get("selected_font_size") or 0)
    wrapped_count = int(base.get("wrapped_count") or 0)
    source_count = max(0, int((source_columns or {}).get("count") or 0))
    outside = float(base.get("outside_ratio") or 0.0)
    score = float(base.get("score") or 0.0)
    score += max(0.0, density - 0.30) * 38.0
    score += max(0.0, fit_ratio - 0.96) * 42.0
    score += max(0.0, edge_contact - 0.35) * 10.0
    score += max(0.0, outside) * 400.0
    if wrapped_count >= 5 and chars >= 10:
        score += (wrapped_count - 4) * 2.0
    if source_count > 0 and wrapped_count > 0:
        score += max(0, wrapped_count - source_count - 1) * 1.5
    if selected_font_size and selected_font_size < 30 and chars >= 10:
        score += (30 - selected_font_size) * 0.9
    dense_watch = _render_readability_v5_dense_watch_ok(
        {
            **base,
            "score": score,
            "density": density,
            "fit_ratio": fit_ratio,
            "edge_contact": edge_contact,
            "selected_font_size": selected_font_size,
            "wrapped_count": wrapped_count,
            "source_column_count": source_count,
            "chars": chars,
        }
    )
    warning_unresolved = bool(
        not dense_watch
        and (
            score >= 18.0
            or density >= 0.40
            or fit_ratio >= 1.08
            or outside > 0.02
            or (edge_contact >= 0.75 and chars >= 8)
            or (selected_font_size and selected_font_size < 28 and chars >= 12)
            or (wrapped_count >= 6 and chars >= 12)
        )
    )
    enriched = dict(base)
    enriched.update(
        {
            "v4_score": base.get("score"),
            "score": round(float(score), 3),
            "edge_contact": round(float(edge_contact), 3),
            "density": round(float(density), 3),
            "fit_ratio": round(float(fit_ratio), 3),
            "source_column_count": int(source_count),
            "shape_source": f"{shape_source}+source_column_guided" if source_count else shape_source,
            "dense_watch": dense_watch,
            "warning_unresolved": warning_unresolved,
        }
    )
    enriched["final_class"] = _render_readability_v5_final_class(
        "unresolved_warning" if warning_unresolved else "applied",
        "layout_unfit_review_only" if warning_unresolved else "",
        enriched,
    )
    _renderer_micro_add("renderer_score_readability_v5_time", micro_start)
    return enriched


def _render_readability_v5_dense_watch_ok(score_data: dict[str, object]) -> bool:
    score = float(score_data.get("score") or 0.0)
    density = float(score_data.get("density") or 0.0)
    fit_ratio = float(score_data.get("fit_ratio") or 0.0)
    edge_contact = float(score_data.get("edge_contact") or 0.0)
    outside = float(score_data.get("outside_ratio") or 0.0)
    selected_font_size = int(score_data.get("selected_font_size") or 0)
    wrapped_count = int(score_data.get("wrapped_count") or 0)
    chars = int(score_data.get("chars") or 0)
    if chars <= 0:
        chars = 1
    if score >= 12.0:
        return False
    if outside > 0.02 or edge_contact >= 0.75:
        return False
    if fit_ratio >= 0.96:
        return False
    if selected_font_size and selected_font_size < 32 and chars >= 10:
        return False
    if wrapped_count >= 4:
        return False
    return 0.24 <= density < 0.28


def _render_readability_v5_final_class(status: str, reason: str, score_data: dict[str, object]) -> str:
    status = str(status or "")
    reason = str(reason or "")
    if status.startswith("not_applicable") or status.startswith("not_required"):
        return "render_readability_v5_resolved"
    if bool(score_data.get("dense_watch")) or _render_readability_v5_dense_watch_ok(score_data):
        return "render_readability_v5_accepted_complete_dense_watch"
    score = float(score_data.get("score") or 0.0)
    density = float(score_data.get("density") or 0.0)
    fit_ratio = float(score_data.get("fit_ratio") or 0.0)
    edge_contact = float(score_data.get("edge_contact") or 0.0)
    outside = float(score_data.get("outside_ratio") or 0.0)
    selected_font_size = int(score_data.get("selected_font_size") or 0)
    wrapped_count = int(score_data.get("wrapped_count") or 0)
    if status == "applied" and not bool(score_data.get("warning_unresolved")):
        return "render_readability_v5_resolved"
    if any(token in reason for token in ("missing_root", "invalid_root", "allowed_area", "shrink_render_box")):
        return "render_readability_v5_unresolved_geometry_limit"
    if outside > 0.02 or edge_contact >= 0.75 or fit_ratio >= 1.08:
        return "render_readability_v5_unresolved_geometry_limit"
    if density >= 0.40 or wrapped_count >= 6 or (selected_font_size and selected_font_size < 28):
        return "render_readability_v5_unresolved_text_too_dense"
    if score >= 18.0:
        return "render_readability_v5_unresolved_text_too_dense"
    return "render_readability_v5_unresolved_renderer_policy"


def _score_render_layout_v3_candidate(
    box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    *,
    compact_layout: bool,
    allowed_area_box,
    shape_source: str,
) -> dict[str, object]:
    micro_start = _renderer_micro_start()
    _renderer_micro_count_key(
        "renderer_score_layout_v3",
        (
            tuple(int(v) for v in box[:4]),
            str(text or ""),
            str(region_font or ""),
            round(float(line_height_scale or 0.0), 4),
            str(wrap_mode or ""),
            str(source_orientation or ""),
            int(source_size_hint or 0),
            int(source_size_min or 0),
            int(source_size_max or 0),
            bool(compact_layout),
            tuple(int(v) for v in allowed_area_box[:4]) if allowed_area_box else None,
            str(shape_source or ""),
        ),
    )
    base = _score_render_layout_v2_candidate(
        box,
        text,
        region_font,
        line_height_scale,
        wrap_mode,
        source_orientation,
        source_size_hint,
        source_size_min,
        source_size_max,
        compact_layout=compact_layout,
        allowed_area_box=allowed_area_box,
    )
    edge_contact = _edge_contact_ratio_xyxy(box, allowed_area_box)
    chars = len(_meaningful_render_chars(text))
    density = float(base.get("density") or 0.0)
    fit_ratio = float(base.get("fit_ratio") or 0.0)
    selected_font_size = int(base.get("selected_font_size") or 0)
    wrapped_count = int(base.get("wrapped_count") or 0)
    score = float(base.get("score") or 0.0)
    score += max(0.0, edge_contact - 0.25) * 11.0
    if wrapped_count >= 4 and chars >= 14:
        score += (wrapped_count - 3) * 1.5
    if selected_font_size and selected_font_size < 30 and chars >= 10:
        score += (30 - selected_font_size) * 0.7
    if "ellipse" in str(shape_source or "") and edge_contact >= 0.75:
        score += 4.0
    unresolved = bool(score >= 14.0 or density >= 0.34 or fit_ratio >= 0.985 or (edge_contact >= 0.75 and chars >= 12))
    enriched = dict(base)
    enriched.update(
        {
            "v2_score": base.get("score"),
            "score": round(float(score), 3),
            "edge_contact": round(float(edge_contact), 3),
            "shape_source": shape_source,
            "warning_unresolved": unresolved,
        }
    )
    _renderer_micro_add("renderer_score_layout_v3_time", micro_start)
    return enriched


def _score_render_layout_v2_candidate(
    box,
    text: str,
    region_font: str,
    line_height_scale: float,
    wrap_mode: str,
    source_orientation: str,
    source_size_hint: int,
    source_size_min: int,
    source_size_max: int,
    *,
    compact_layout: bool,
    allowed_area_box,
) -> dict[str, object]:
    micro_start = _renderer_micro_start()
    _renderer_micro_count_key(
        "renderer_score_layout_v2",
        (
            tuple(int(v) for v in box[:4]),
            str(text or ""),
            str(region_font or ""),
            round(float(line_height_scale or 0.0), 4),
            str(wrap_mode or ""),
            str(source_orientation or ""),
            int(source_size_hint or 0),
            int(source_size_min or 0),
            int(source_size_max or 0),
            bool(compact_layout),
            tuple(int(v) for v in allowed_area_box[:4]) if allowed_area_box else None,
        ),
    )
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    chars = len(_meaningful_render_chars(text))
    density = chars / max(1.0, _xyxy_area((x0, y0, x1, y1)) / 1000.0)
    outside = _outside_ratio((x0, y0, x1, y1), allowed_area_box) if allowed_area_box else 0.0
    fit_ratio = 0.0
    selected_font_size = 0
    wrapped_count = 0
    if _should_use_vertical_layout(text, width, height, wrap_mode, source_orientation=source_orientation):
        if compact_layout:
            side_pad = max(1, int(width * 0.02))
            top_pad = max(1, int(height * 0.025))
            bottom_pad = max(1, int(height * 0.015))
        else:
            side_pad = max(1, int(width * 0.04))
            top_pad = max(2, int(height * 0.05))
            bottom_pad = max(1, int(height * 0.03))
        inner_w = max(1, width - side_pad * 2)
        inner_h = max(1, height - top_pad - bottom_pad)
        try:
            font, layout = _fit_vertical_font(
                text,
                inner_w,
                inner_h,
                region_font,
                preferred_size=source_size_hint if source_size_hint > 0 else None,
                min_size=source_size_min if source_size_min > 0 else None,
                max_size=source_size_max if source_size_max > 0 else None,
                line_height_scale=line_height_scale,
            )
            rows, cols, cell_height, col_width, col_gap = layout
            max_rows_used = max(
                len((_vertical_tokens(text) or [])[col * rows : (col + 1) * rows])
                for col in range(max(1, cols))
            )
            measured_w = cols * col_width + max(0, cols - 1) * col_gap
            measured_h = max_rows_used * cell_height
            fit_ratio = max(measured_w / max(1, inner_w), measured_h / max(1, inner_h))
            selected_font_size = int(getattr(font, "size", 0) or 0)
            wrapped_count = int(cols)
        except Exception:
            fit_ratio = 1.5
    score = 0.0
    score += max(0.0, density - 0.18) * 95.0
    score += max(0.0, fit_ratio - 0.90) * 70.0
    score += max(0.0, outside) * 400.0
    if selected_font_size and selected_font_size < 28 and chars >= 8:
        score += (28 - selected_font_size) * 0.9
    if wrapped_count >= 5 and chars >= 10:
        score += (wrapped_count - 4) * 2.0
    result = {
        "score": round(float(score), 3),
        "density": round(float(density), 3),
        "fit_ratio": round(float(fit_ratio), 3),
        "selected_font_size": selected_font_size or None,
        "wrapped_count": wrapped_count,
        "outside_ratio": round(float(outside), 3),
        "warning_unresolved": bool(score >= 12.0 or density >= 0.34 or fit_ratio >= 0.97),
    }
    _renderer_micro_add("renderer_score_layout_v2_time", micro_start)
    return result


def _render_cleanup_constraint_debug(
    render_box,
    cleanup_box,
    allowed_area_box,
    allowed_area_source: str,
    cleanup_pixels: int,
) -> dict[str, object]:
    debug: dict[str, object] = {}
    final_box = tuple(int(v) for v in render_box[:4]) if render_box else None
    if allowed_area_box is not None:
        allowed = tuple(int(v) for v in allowed_area_box[:4])
        outside = _outside_ratio(final_box, allowed) if final_box else 0.0
        debug["render_allowed_area_bbox"] = list(allowed)
        debug["render_allowed_area_source"] = allowed_area_source
        debug["final_render_outside_allowed_area_ratio"] = round(outside, 3)
        debug["render_outside_text_area_container"] = outside > 0.12
        if cleanup_pixels:
            allowed_area = max(1, _xyxy_area(allowed))
            cleanup_ratio = cleanup_pixels / allowed_area
            if cleanup_ratio > 0.60:
                debug["cleanup_artifact_risk"] = True
                debug["cleanup_artifact_risk_reason"] = "cleanup_mask_large_relative_to_text_area_container"
            else:
                debug["cleanup_artifact_risk"] = False
    if final_box and cleanup_box:
        cleanup = tuple(int(v) for v in cleanup_box[:4])
        render_area = max(1, _xyxy_area(final_box))
        cleanup_intersection = _xyxy_intersection_area(final_box, cleanup)
        mismatch = max(0.0, min(1.0, 1.0 - (cleanup_intersection / render_area)))
        debug["cleanup_render_mismatch_ratio"] = round(mismatch, 3)
        debug["cleanup_render_area_mismatch"] = mismatch > 0.70
        if allowed_area_box is not None:
            cleanup_outside = _outside_ratio(cleanup, tuple(int(v) for v in allowed_area_box[:4]))
            debug["cleanup_outside_allowed_area_ratio"] = round(cleanup_outside, 3)
            debug["cleanup_does_not_cover_source_glyphs"] = cleanup_intersection <= 0
    return debug


def _xywh_value_to_xyxy(value, img_w: int, img_h: int):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    x0 = max(0, min(img_w, x))
    y0 = max(0, min(img_h, y))
    x1 = max(x0 + 1, min(img_w, x + w))
    y1 = max(y0 + 1, min(img_h, y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _expand_box(box, pad_x: int, pad_y: int, max_w: int, max_h: int):
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - max(0, pad_x)),
        max(0, y0 - max(0, pad_y)),
        min(max_w, x1 + max(0, pad_x)),
        min(max_h, y1 + max(0, pad_y)),
    )


def _expand_box_to_size_within_limit(
    box,
    target_w: int,
    target_h: int,
    limit_box,
    center: tuple[float, float] | None = None,
):
    if limit_box is None:
        return box
    lx0, ly0, lx1, ly1 = [int(v) for v in limit_box]
    limit_w = max(1, lx1 - lx0)
    limit_h = max(1, ly1 - ly0)
    target_w = max(1, min(int(target_w), limit_w))
    target_h = max(1, min(int(target_h), limit_h))
    if center is None:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
    else:
        cx, cy = center
    x0 = int(round(cx - target_w / 2.0))
    y0 = int(round(cy - target_h / 2.0))
    x1 = x0 + target_w
    y1 = y0 + target_h
    if x0 < lx0:
        x1 += lx0 - x0
        x0 = lx0
    if x1 > lx1:
        x0 -= x1 - lx1
        x1 = lx1
    if y0 < ly0:
        y1 += ly0 - y0
        y0 = ly0
    if y1 > ly1:
        y0 -= y1 - ly1
        y1 = ly1
    x0 = max(lx0, x0)
    y0 = max(ly0, y0)
    x1 = min(lx1, x1)
    y1 = min(ly1, y1)
    if x1 <= x0 or y1 <= y0:
        return box
    return (x0, y0, x1, y1)


def _vertical_semantic_token_count(text: str) -> int:
    tokens = _vertical_tokens(_normalize_text(text))
    if not tokens:
        return 0
    punctuation = _vertical_punct_chars()
    vertical_lines = _vertical_line_tokens()
    semantic_tokens = [tok for tok in tokens if tok not in punctuation and tok not in vertical_lines]
    return len(semantic_tokens) or len(tokens)


def _minimum_vertical_box_size(
    source_box,
    text: str,
    preferred_size: int | None = None,
    min_size: int | None = None,
) -> tuple[int, int]:
    sx0, sy0, sx1, sy1 = [int(v) for v in source_box]
    source_w = max(1, sx1 - sx0)
    source_h = max(1, sy1 - sy0)
    source_w = max(source_w + 4, int(math.ceil(source_w / 0.92)))
    source_h = max(source_h + 6, int(math.ceil(source_h / 0.92)))
    semantic_count = _vertical_semantic_token_count(text)
    size_floor = max(
        int(preferred_size or 0) if preferred_size and preferred_size > 0 else 0,
        int(min_size or 0) if min_size and min_size > 0 else 0,
    )
    if size_floor > 0:
        if _is_vertical_ellipsis_text(text):
            width_floor = int(math.ceil(size_floor * 1.20))
        elif semantic_count <= 2:
            width_floor = int(math.ceil(size_floor * 1.62))
        elif semantic_count <= 4:
            width_floor = int(math.ceil(size_floor * 1.54))
        elif semantic_count <= 6:
            width_floor = int(math.ceil(size_floor * 1.46))
        else:
            width_floor = int(math.ceil(size_floor * 1.34))
        source_w = max(source_w, width_floor)
    target_w = source_w
    target_h = source_h
    if _is_vertical_ellipsis_text(text):
        target_w = max(target_w, 52)
        target_h = max(target_h, 136)
    elif semantic_count <= 2:
        target_w = max(target_w, 54)
        target_h = max(target_h, 132)
    elif semantic_count <= 4:
        target_w = max(target_w, 62)
        target_h = max(target_h, 148)
    elif semantic_count <= 6:
        target_w = max(target_w, 68)
        target_h = max(target_h, 160)
    return target_w, target_h


def _vertical_inner_dimensions(box) -> tuple[int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    outer_width = max(1, x1 - x0)
    outer_height = max(1, y1 - y0)
    side_pad = max(1, int(outer_width * 0.04))
    top_pad = max(2, int(outer_height * 0.05))
    bottom_pad = max(1, int(outer_height * 0.03))
    return max(1, outer_width - side_pad * 2), max(1, outer_height - top_pad - bottom_pad)


def _bubble_inner_layout_box(
    bubble_mask,
    bubble_box,
    source_box,
    text: str,
    max_w: int,
    max_h: int,
    preferred_size: int | None = None,
    min_size: int | None = None,
):
    source_cx = (source_box[0] + source_box[2]) / 2.0
    source_cy = (source_box[1] + source_box[3]) / 2.0
    target_w, target_h = _minimum_vertical_box_size(
        source_box,
        text,
        preferred_size=preferred_size,
        min_size=min_size,
    )
    limit_box = None
    candidate = None
    if cv2 is not None and np is not None and bubble_mask is not None:
        ys, xs = np.where(bubble_mask > 0)
        if ys.size and xs.size:
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            roi = bubble_mask[y0:y1, x0:x1]
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            margin = max(2, int(min(bw, bh) * 0.04))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
            eroded = cv2.erode(roi, kernel, iterations=1)
            if eroded is not None and eroded.any():
                ey, ex = np.where(eroded > 0)
                if ey.size and ex.size:
                    limit_box = (
                        max(0, x0 + int(ex.min())),
                        max(0, y0 + int(ey.min())),
                        min(max_w, x0 + int(ex.max()) + 1),
                        min(max_h, y0 + int(ey.max()) + 1),
                    )
                cy = int((source_box[1] + source_box[3]) / 2.0)
                cx = int((source_box[0] + source_box[2]) / 2.0)
                local_cx = min(max(cx - x0, 0), eroded.shape[1] - 1)
                local_cy = min(max(cy - y0, 0), eroded.shape[0] - 1)
                row = eroded[local_cy]
                col = eroded[:, local_cx]
                if row[local_cx] > 0:
                    lx = local_cx
                    while lx > 0 and row[lx - 1] > 0:
                        lx -= 1
                    rx = local_cx
                    while rx + 1 < row.shape[0] and row[rx + 1] > 0:
                        rx += 1
                else:
                    xhits = np.where(row > 0)[0]
                    lx = int(xhits.min()) if xhits.size else 0
                    rx = int(xhits.max()) if xhits.size else eroded.shape[1] - 1
                if col[local_cy] > 0:
                    ty = local_cy
                    while ty > 0 and col[ty - 1] > 0:
                        ty -= 1
                    by = local_cy
                    while by + 1 < col.shape[0] and col[by + 1] > 0:
                        by += 1
                else:
                    yhits = np.where(col > 0)[0]
                    ty = int(yhits.min()) if yhits.size else 0
                    by = int(yhits.max()) if yhits.size else eroded.shape[0] - 1
                candidate = (
                    max(0, x0 + lx),
                    max(0, y0 + ty),
                    min(max_w, x0 + rx + 1),
                    min(max_h, y0 + by + 1),
                )
                candidate = _shrink_box(candidate, max(1, int(min(candidate[2] - candidate[0], candidate[3] - candidate[1]) * 0.02)))
    if bubble_box is not None:
        bubble_limit = _shrink_box(
            bubble_box,
            max(2, int(min(max(1, bubble_box[2] - bubble_box[0]), max(1, bubble_box[3] - bubble_box[1])) * 0.05)),
        )
        if limit_box is None:
            limit_box = bubble_limit
        else:
            current_area = max(1, (limit_box[2] - limit_box[0]) * (limit_box[3] - limit_box[1]))
            bubble_area = max(1, (bubble_limit[2] - bubble_limit[0]) * (bubble_limit[3] - bubble_limit[1]))
            if bubble_area > current_area:
                limit_box = bubble_limit
        if candidate is None:
            candidate = limit_box
    if candidate is None:
        return None
    if limit_box is None:
        limit_box = candidate
    candidate_w = max(1, candidate[2] - candidate[0])
    candidate_h = max(1, candidate[3] - candidate[1])
    limit_w = max(1, limit_box[2] - limit_box[0])
    limit_h = max(1, limit_box[3] - limit_box[1])
    if (
        candidate_w < int(target_w * 0.82)
        or candidate_h < int(target_h * 0.82)
    ) and (limit_w >= candidate_w or limit_h >= candidate_h):
        candidate = limit_box
        candidate_w = limit_w
        candidate_h = limit_h
    target_w = max(target_w, candidate_w)
    target_h = max(target_h, candidate_h)
    return _expand_box_to_size_within_limit(
        candidate,
        target_w,
        target_h,
        limit_box,
        center=(source_cx, source_cy),
    )


def _local_region_text_mask(
    img_np,
    bbox,
    polygon,
    dilate_px: int = 2,
    strong_vertical: bool = False,
    glyph_only: bool = False,
):
    if cv2 is None or np is None:
        return None
    if img_np is None:
        return None
    seed_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            for poly in polys:
                cv2.fillPoly(seed_mask, [np.array(poly, dtype=np.int32)], 255)
        except Exception:
            polys = None
    if not polys:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(seed_mask, (x, y), (x + max(1, w), y + max(1, h)), 255, thickness=-1)
    mask = np.zeros_like(seed_mask) if glyph_only else seed_mask.copy()
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        seed_mask = cv2.dilate(seed_mask, kernel, iterations=1)
        if not glyph_only:
            mask = seed_mask.copy()
        x, y, w, h = [int(v) for v in bbox]
        if w > 0 and h > 0:
            pad = max(4, dilate_px * 2)
            rx0 = max(0, x - pad)
            ry0 = max(0, y - pad)
            rx1 = min(img_np.shape[1], x + w + pad)
            ry1 = min(img_np.shape[0], y + h + pad)
            if rx1 > rx0 and ry1 > ry0:
                gray = cv2.cvtColor(img_np[ry0:ry1, rx0:rx1], cv2.COLOR_RGB2GRAY)
                base_local = seed_mask[ry0:ry1, rx0:rx1]
                verticalish = h > w * 1.5
                if verticalish:
                    limit_x = max(6, min(max(pad, dilate_px * 3), int(max(8, w * 0.30))))
                    limit_y = max(4, min(max(pad, dilate_px * 2), int(max(6, h * 0.12))))
                else:
                    limit_x = max(4, min(max(pad, dilate_px * 2), int(max(5, w * 0.20))))
                    limit_y = max(4, min(max(pad, dilate_px * 2), int(max(5, h * 0.12))))
                if strong_vertical and verticalish:
                    limit_x = max(limit_x, max(10, int(max(w * 0.90, h * 0.12))))
                    limit_y = max(limit_y, max(8, int(h * 0.10)))
                limit_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (max(3, limit_x * 2 + 1), max(3, limit_y * 2 + 1)),
                )
                limit_mask = cv2.dilate(base_local, limit_kernel, iterations=1)
                block = max(15, int(min(gray.shape[:2]) / 6) * 2 + 1)
                try:
                    dark = cv2.adaptiveThreshold(
                        gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        block,
                        8,
                    )
                    light = cv2.adaptiveThreshold(
                        gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block,
                        8,
                    )
                    dark_overlap = int(((dark > 0) & (base_local > 0)).sum())
                    light_overlap = int(((light > 0) & (base_local > 0)).sum())
                    if max(dark_overlap, light_overlap) > 4:
                        if glyph_only:
                            base_pixels = max(1, int((base_local > 0).sum()))
                            ink_candidates = []
                            if dark_overlap > 4:
                                dark_ratio = dark_overlap / base_pixels
                                if dark_ratio <= 0.55:
                                    ink_candidates.append((dark_ratio, dark))
                            if light_overlap > 4:
                                light_ratio = light_overlap / base_pixels
                                if light_ratio <= 0.55:
                                    ink_candidates.append((light_ratio, light))
                            if ink_candidates:
                                ink = min(ink_candidates, key=lambda item: item[0])[1]
                            else:
                                ink = dark if dark_overlap <= light_overlap else light
                        else:
                            ink = dark if dark_overlap >= light_overlap else light
                        ink = cv2.bitwise_and(ink, limit_mask)
                        if strong_vertical and verticalish and not glyph_only:
                            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=2)
                            ink = cv2.dilate(ink, kernel, iterations=3)
                        else:
                            ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
                            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)
                            ink = cv2.dilate(ink, kernel, iterations=1)
                        mask[ry0:ry1, rx0:rx1] = cv2.bitwise_or(mask[ry0:ry1, rx0:rx1], ink)
                except Exception:
                    pass
    if glyph_only and not np.any(mask):
        return None
    return mask


def _source_glyph_local_mask(
    img_np,
    bbox,
    polygon=None,
    dilate_px: int = 2,
    limit_box=None,
    bright_context_only: bool = False,
):
    return source_glyph_mask_stage.build_source_glyph_local_mask(
        img_np,
        bbox,
        polygon=polygon,
        dilate_px=dilate_px,
        limit_box=limit_box,
        bright_context_only=bright_context_only,
    )


def _source_erasure_expected_box(
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    img_w: int,
    img_h: int,
):
    return source_glyph_mask_stage.source_erasure_expected_box(
        region,
        render,
        regions_by_id,
        fallback_bbox,
        img_w,
        img_h,
    )


def _source_glyph_erasure_audit_fields(img_np, cleanup_mask, expected_box) -> dict[str, object]:
    return source_glyph_mask_stage.source_glyph_erasure_audit_fields(
        img_np,
        cleanup_mask,
        expected_box,
    )


def _recovered_speech_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    dilate_px: int = 3,
):
    return source_glyph_mask_stage.build_recovered_speech_source_cleanup_mask(
        img_np,
        region,
        render,
        regions_by_id,
        fallback_bbox,
        fallback_polygon,
        dilate_px=dilate_px,
    )


def _same_speech_container_member(anchor: dict, member: dict) -> bool:
    anchor_container = str(anchor.get("text_area_container_id", "") or "")
    member_container = str(member.get("text_area_container_id", "") or "")
    if anchor_container and member_container and anchor_container == member_container:
        return True
    anchor_physical = str(anchor.get("logical_text_physical_bubble_id", "") or "")
    member_physical = str(member.get("logical_text_physical_bubble_id", "") or "")
    return bool(anchor_physical and member_physical and anchor_physical == member_physical)


def _expand_vertical_speech_cleanup_neighbors(img_np, local_mask, bbox):
    """Recover adjacent vertical source columns when bubble detection failed."""
    if cv2 is None or np is None:
        return local_mask
    if img_np is None or local_mask is None:
        return local_mask
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return local_mask
    if h < max(60, int(w * 1.35)):
        return local_mask

    img_h, img_w = img_np.shape[:2]
    x_pad = max(70, int(max(w * 4.0, h * 0.85)))
    y_pad = max(16, int(h * 0.25))
    sx0 = max(0, x - x_pad)
    sx1 = min(img_w, x + w + x_pad)
    sy0 = max(0, y - y_pad)
    sy1 = min(img_h, y + h + y_pad)
    if sx1 <= sx0 or sy1 <= sy0:
        return local_mask
    crop_area = (sx1 - sx0) * (sy1 - sy0)
    if crop_area > max(120000, int(img_w * img_h * 0.045)):
        return local_mask

    crop = img_np[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return local_mask
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    p20 = float(np.percentile(gray, 20))
    bright_ratio = float((gray > 185).sum()) / max(1, gray.size)
    dark_ratio = float((gray < 90).sum()) / max(1, gray.size)
    if median < 205 or p20 < 145 or bright_ratio < 0.68 or dark_ratio > 0.28:
        return local_mask

    base_local = local_mask[sy0:sy1, sx0:sx1]
    if base_local.size == 0 or not np.any(base_local):
        return local_mask

    ink = (gray < 145).astype(np.uint8) * 255
    # Connect glyphs in a vertical column without bridging across columns.
    col_kernel_h = max(17, int(h * 0.28) | 1)
    col_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, col_kernel_h))
    columns = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, col_kernel, iterations=1)
    columns = cv2.dilate(columns, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), iterations=1)

    num_labels, labels = cv2.connectedComponents(columns)
    if num_labels <= 1:
        return local_mask

    accepted = np.zeros_like(base_local)
    base_pixels = int((base_local > 0).sum())
    source_cx = x + w / 2.0
    source_cy = y + h / 2.0
    min_h = max(34, int(h * 0.42))
    max_h = int(h * 1.45)
    max_w = max(48, int(w * 2.8))

    for label in range(1, num_labels):
        comp = labels == label
        ys, xs = np.where(comp)
        if ys.size == 0 or xs.size == 0:
            continue
        cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        comp_w = cx1 - cx0
        comp_h = cy1 - cy0
        if comp_h < min_h or comp_h > max_h:
            continue
        if comp_w < 6 or comp_w > max_w:
            continue
        if comp_h / max(1, comp_w) < 1.25:
            continue
        if cx0 <= 2 or cx1 >= gray.shape[1] - 2:
            continue

        gx0, gy0, gx1, gy1 = sx0 + cx0, sy0 + cy0, sx0 + cx1, sy0 + cy1
        cand_cx = (gx0 + gx1) / 2.0
        cand_cy = (gy0 + gy1) / 2.0
        if abs(cand_cy - source_cy) > max(22, h * 0.30):
            continue
        if abs(cand_cx - source_cx) > max(85, h * 0.95):
            continue

        dark_pixels = int((ink[cy0:cy1, cx0:cx1] > 0).sum())
        density = dark_pixels / max(1, comp_w * comp_h)
        if density < 0.025 or density > 0.48:
            continue

        nb = max(4, int(min(comp_w, comp_h) * 0.16))
        nx0 = max(0, cx0 - nb)
        ny0 = max(0, cy0 - nb)
        nx1 = min(gray.shape[1], cx1 + nb)
        ny1 = min(gray.shape[0], cy1 + nb)
        neighborhood = gray[ny0:ny1, nx0:nx1]
        if neighborhood.size == 0:
            continue
        if float(np.median(neighborhood)) < 195:
            continue
        if float((neighborhood > 170).sum()) / max(1, neighborhood.size) < 0.58:
            continue

        pad_x = max(3, int(comp_w * 0.18))
        pad_y = max(8, int(comp_h * 0.10))
        rx0 = max(0, cx0 - pad_x)
        ry0 = max(0, cy0 - pad_y)
        rx1 = min(accepted.shape[1], cx1 + pad_x)
        ry1 = min(accepted.shape[0], cy1 + pad_y)
        band = ink[:, rx0:rx1]
        band_ys, _band_xs = np.where(band > 0)
        if band_ys.size:
            ry0 = min(ry0, max(0, int(band_ys.min()) - pad_y))
            ry1 = max(ry1, min(accepted.shape[0], int(band_ys.max()) + 1 + pad_y))
        cv2.rectangle(accepted, (rx0, ry0), (rx1, ry1), 255, thickness=-1)

    if not np.any(accepted):
        return local_mask

    expanded_local = cv2.bitwise_or(base_local, accepted)
    added_pixels = int(((accepted > 0) & (base_local == 0)).sum())
    if added_pixels > max(18000, base_pixels * 3):
        return local_mask

    expanded = local_mask.copy()
    expanded[sy0:sy1, sx0:sx1] = expanded_local
    return expanded


def _fallback_speech_render_box(box, text: str, max_w: int, max_h: int):
    x0, y0, x1, y1 = box
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    if box_h < box_w * 1.8:
        return box
    target_w, target_h = _minimum_vertical_box_size(box, text)
    page_limit = (0, 0, max_w, max_h)
    return _expand_box_to_size_within_limit(
        box,
        max(target_w, box_w),
        max(target_h, box_h),
        page_limit,
    )


def _stabilize_tiny_speech_render_box(
    render_box,
    base_box,
    text: str,
    max_w: int,
    max_h: int,
    limit_box=None,
    preferred_size: int | None = None,
    min_size: int | None = None,
):
    rx0, ry0, rx1, ry1 = render_box
    rw = max(1, rx1 - rx0)
    rh = max(1, ry1 - ry0)
    if rh <= rw * 1.12:
        return render_box
    tokens = _vertical_tokens(_normalize_text(text))
    if not tokens:
        return render_box
    semantic_tokens = [
        tok for tok in tokens if tok not in _vertical_punct_chars() and tok not in _vertical_line_tokens()
    ]
    token_count = len(tokens)
    content_count = len(semantic_tokens) or token_count
    base_w = max(1, base_box[2] - base_box[0])
    base_h = max(1, base_box[3] - base_box[1])
    target_w, target_h = _minimum_vertical_box_size(
        base_box,
        text,
        preferred_size=preferred_size,
        min_size=min_size,
    )
    if min_size is not None and min_size > 0:
        if content_count <= 2:
            target_w = max(target_w, int(math.ceil(min_size * 1.62)))
        elif content_count <= 4:
            target_w = max(target_w, int(math.ceil(min_size * 1.54)))
        else:
            target_w = max(target_w, int(math.ceil(min_size * 1.42)))
    target_w = max(target_w, base_w, rw)
    target_h = max(target_h, base_h, rh)
    if content_count >= 8:
        target_w = max(target_w, min(240, max(int(base_w * 1.18), int(base_h * 0.92))))
        target_h = max(target_h, min(max_h, max(int(base_h * 1.10), base_h + 14)))
    if limit_box is not None:
        return _expand_box_to_size_within_limit(
            render_box,
            target_w,
            target_h,
            limit_box,
            center=((base_box[0] + base_box[2]) / 2.0, (base_box[1] + base_box[3]) / 2.0),
        )
    expand_x = max(0, target_w - rw) // 2
    expand_y = max(0, target_h - rh) // 2
    if expand_x <= 0 and expand_y <= 0:
        return render_box
    return _expand_box(render_box, expand_x, expand_y, max_w, max_h)


def _stabilize_tiny_nonbubble_render_box(
    render_box,
    base_box,
    text: str,
    max_w: int,
    max_h: int,
    preferred_size: int | None = None,
    min_size: int | None = None,
    wrap_mode: str = "vertical",
):
    rx0, ry0, rx1, ry1 = render_box
    rw = max(1, rx1 - rx0)
    rh = max(1, ry1 - ry0)
    mode = str(wrap_mode or "vertical").strip().lower()
    if mode == "horizontal":
        semantic_count = len(_non_punct_chars(text))
        if semantic_count < 4:
            return render_box
        size_floor = max(
            int(preferred_size or 0) if preferred_size and preferred_size > 0 else 0,
            int(min_size or 0) if min_size and min_size > 0 else 0,
        )
        if size_floor <= 0:
            return render_box
        target_h = max(rh, int(math.ceil(size_floor * 1.44)))
        target_w = max(rw, int(math.ceil(size_floor * max(4.8, min(8.0, semantic_count * 0.92)))))
        expand_x = max(0, target_w - rw) // 2
        expand_y = max(0, target_h - rh) // 2
        if expand_x <= 0 and expand_y <= 0:
            return render_box
        return _expand_box(render_box, expand_x, expand_y, max_w, max_h)
    if rh <= rw * 1.35:
        return render_box
    tokens = _vertical_tokens(_normalize_text(text))
    if not tokens:
        return render_box
    punctuation = _vertical_punct_chars()
    vertical_lines = _vertical_line_tokens()
    semantic_tokens = [tok for tok in tokens if tok not in punctuation and tok not in vertical_lines]
    if len(semantic_tokens) < 2 and not _is_vertical_ellipsis_text(text):
        return render_box
    base_w = max(1, base_box[2] - base_box[0])
    base_h = max(1, base_box[3] - base_box[1])
    target_w = max(rw, base_w)
    target_h = max(rh, base_h)
    if min_size is not None and min_size > 0:
        if len(semantic_tokens) <= 3:
            target_w = max(target_w, int(math.ceil(min_size * 1.52)))
        else:
            target_w = max(target_w, int(math.ceil(min_size * 1.34)))
    if _is_vertical_ellipsis_text(text):
        target_w = max(rw, 54)
        target_h = max(rh, 128)
    elif len(semantic_tokens) <= 3:
        target_w = max(rw, 54)
        target_h = max(rh, 126)
    elif len(semantic_tokens) <= 6:
        target_w = max(rw, 60)
        target_h = max(rh, 144)
    elif len(semantic_tokens) <= 10 and rw <= 44:
        target_w = max(rw, 64)
        target_h = max(rh, 162)
    else:
        return render_box
    expand_x = max(0, target_w - rw) // 2
    expand_y = max(0, target_h - rh) // 2
    if expand_x <= 0 and expand_y <= 0:
        return render_box
    return _expand_box(render_box, expand_x, expand_y, max_w, max_h)


def _estimate_speech_bubble_box(img_np, base_box):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in base_box]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    small_vertical = bh > bw * 1.6 and bw <= 72
    if bh <= max(40, int(bw * 1.25)):
        return None
    img_h, img_w = img_np.shape[:2]
    pad_x = max(28, int(max(bw * 2.6, bh * 0.48)))
    pad_y = max(12, int(bh * 0.16))
    sx0, sy0, sx1, sy1 = _expand_box((x0, y0, x1, y1), pad_x, pad_y, img_w, img_h)
    roi = img_np[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_mask = _rect_mask((gray.shape[0], gray.shape[1]), [x0, y0, bw, bh], sx0, sy0)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    best_box = None
    best_score = None
    for thresh in (205, 195, 185, 170, 155):
        _, white = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)
        bubble_box, _bubble_mask = _find_bubble_box(white, text_mask)
        if bubble_box is None:
            continue
        bx0, by0, bx1, by1 = bubble_box
        est_w = max(1, bx1 - bx0)
        est_h = max(1, by1 - by0)
        est_area = est_w * est_h
        base_area = max(1, bw * bh)
        area_ratio = est_area / base_area
        min_w_expand = 1.08 if small_vertical else 1.30
        min_h_ratio = 0.76 if small_vertical else 0.88
        if est_w <= int(bw * min_w_expand) or est_h <= int(bh * min_h_ratio):
            continue
        min_area_ratio = 1.10 if small_vertical else 1.35
        if area_ratio < min_area_ratio or area_ratio > 12.0:
            continue
        if est_area >= (gray.shape[0] * gray.shape[1]) * 0.72:
            continue
        abs_box = (sx0 + bx0, sy0 + by0, sx0 + bx1, sy0 + by1)
        score = area_ratio + abs((sx0 + (bx0 + bx1) // 2) - (x0 + bw // 2)) / max(1, bw) * 0.25
        if best_score is None or score < best_score:
            best_box = abs_box
            best_score = score
    return best_box


def _estimate_vertical_speech_box_from_edges(img_np, base_box):
    if cv2 is None or np is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in base_box]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    small_vertical = bh > bw * 1.6 and bw <= 72
    if bh <= max(40, int(bw * 1.25)):
        return None
    img_h, img_w = img_np.shape[:2]
    pad_x = max(40, int(max(bw * 3.0, bh * 0.55)))
    pad_y = max(16, int(bh * 0.18))
    sx0, sy0, sx1, sy1 = _expand_box((x0, y0, x1, y1), pad_x, pad_y, img_w, img_h)
    roi = img_np[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    rx0 = x0 - sx0
    ry0 = y0 - sy0
    rx1 = x1 - sx0
    ry1 = y1 - sy0
    if rx0 < 0 or ry0 < 0 or rx1 > edges.shape[1] or ry1 > edges.shape[0]:
        return None
    left_hits = []
    right_hits = []
    for rel in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
        row = int(ry0 + bh * rel)
        row = max(0, min(edges.shape[0] - 1, row))
        left = np.where(edges[row, : max(0, rx0 - 2)] > 0)[0]
        right = np.where(edges[row, min(edges.shape[1], rx1 + 2) :] > 0)[0]
        if left.size:
            left_hits.append(int(left.max()))
        if right.size:
            right_hits.append(int(right.min() + rx1 + 2))
    if len(left_hits) < 2 or len(right_hits) < 2:
        return None
    left_edge = int(np.median(left_hits))
    right_edge = int(np.median(right_hits))
    est_w = right_edge - left_edge
    min_expand = 1.05 if small_vertical else 1.15
    if est_w <= int(bw * min_expand):
        return None
    max_est_w = min(edges.shape[1] - 1, max((110 if small_vertical else 140), int(bw * (5.2 if small_vertical else 4.5))))
    if est_w > max_est_w:
        return None
    top_pad = max(6, int(bh * 0.08))
    bottom_pad = max(6, int(bh * 0.08))
    return (
        max(0, sx0 + left_edge),
        max(0, y0 - top_pad),
        min(img_w, sx0 + right_edge),
        min(img_h, y1 + bottom_pad),
    )


def _estimate_vertical_speech_preferred_size(base_box, text: str) -> int:
    x0, y0, x1, y1 = [int(v) for v in base_box]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    tokens = _vertical_tokens(_normalize_text(text))
    if not tokens:
        return 0
    punctuation = _vertical_punct_chars()
    vertical_lines = _vertical_line_tokens()
    semantic_tokens = [tok for tok in tokens if tok not in punctuation and tok not in vertical_lines]
    if not semantic_tokens:
        hint = min(bh * 0.28, max(bw * 1.20, 18))
    elif len(semantic_tokens) <= 2:
        hint = min(bh * 0.24, max(bw * 1.05, 16))
    elif len(tokens) <= 4:
        hint = min(bh * 0.22, max(bw * 0.96, 15))
    else:
        hint = min(bh * 0.20, max(bw * 0.92, 14))
    return max(14, min(32, int(hint)))


def _wrap_text(draw, text: str, font, max_width: int, max_lines: int | None = None) -> List[str]:
    micro_start = _renderer_micro_start()
    _renderer_micro_count_key(
        "renderer_text_wrap_measure",
        (
            str(text or ""),
            getattr(font, "path", None),
            getattr(font, "size", None),
            int(max_width or 0),
            int(max_lines) if max_lines is not None else None,
        ),
    )
    words = _tokenize_text(text)
    lines: List[str] = []
    current = ""
    has_space = " " in text
    for idx, ch in enumerate(words):
        candidate = (current + " " + ch).strip() if has_space else current + ch
        if _is_punct_only(ch) and current:
            current = f"{current}{ch}"
            continue
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
            if max_lines is not None and len(lines) >= max_lines - 1:
                remaining = words[idx + 1 :]
                if remaining:
                    current = _join_tokens([current] + remaining, has_space)
                break
    if current:
        lines.append(current)
    lines = _fix_leading_punct(lines)
    _renderer_micro_add("renderer_text_wrap_measure_time", micro_start)
    return lines


def _should_use_vertical_layout(
    text: str,
    width: int,
    height: int,
    wrap_mode: str,
    source_orientation: str = "",
) -> bool:
    mode = str(wrap_mode or "auto").strip().lower()
    source_orientation = str(source_orientation or "").strip().lower()
    if mode == "vertical":
        return True
    if mode == "horizontal":
        return False
    if source_orientation == "vertical" and _has_cjk(text):
        return True
    if source_orientation == "horizontal":
        return False
    if not _has_cjk(text):
        return False
    return height > width * 1.12


def _vertical_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for ch in str(text or ""):
        if ch in {"\r", "\n", "\t", " "}:
            continue
        tokens.append(ch)
    return tokens


def _vertical_punct_chars() -> set[str]:
    return {
        "。",
        "．",
        "，",
        "、",
        "！",
        "？",
        "：",
        "；",
        "…",
        "·",
        "—",
        "―",
        "－",
        "-",
        "～",
        "〜",
        "「",
        "」",
        "『",
        "』",
        "“",
        "”",
        "‘",
        "’",
    }


def _vertical_line_tokens() -> set[str]:
    return {
        "\u2014",
        "\u2015",
        "\uff0d",
        "-",
        "\u30fc",
    }


def _is_vertical_ellipsis_text(text: str) -> bool:
    tokens = _vertical_tokens(text)
    if not tokens:
        return False
    ellipsis_tokens = {".", "．", "…", "‥", "・", "･"}
    allowed_tokens = ellipsis_tokens | {"?", "？", "!", "！"}
    return any(token in ellipsis_tokens for token in tokens) and all(token in allowed_tokens for token in tokens)


def _draw_vertical_ellipsis(
    draw,
    box,
    fill_color,
    stroke_width: int,
    stroke_color,
    preferred_size: int | None = None,
    min_size: int | None = None,
):
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    cx = x0 + width / 2.0
    size_basis = 0.0
    if preferred_size is not None and preferred_size > 0:
        size_basis = max(size_basis, float(preferred_size))
    if min_size is not None and min_size > 0:
        size_basis = max(size_basis, float(min_size))
    if size_basis > 0:
        radius = max(5.0, min(size_basis * 0.34, width * 0.32, height * 0.11))
    else:
        radius = max(5.0, min(width * 0.30, height * 0.10))
    top_pad = max(radius * 1.5, height * 0.12)
    bottom_pad = max(radius * 1.5, height * 0.12)
    span = max(radius * 4.0, height - top_pad - bottom_pad)
    centers = [y0 + top_pad + span * ratio for ratio in (0.15, 0.50, 0.85)]
    for cy in centers:
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        if stroke_color and stroke_width > 0:
            outline_box = (
                cx - radius - stroke_width,
                cy - radius - stroke_width,
                cx + radius + stroke_width,
                cy + radius + stroke_width,
            )
            draw.ellipse(outline_box, fill=stroke_color)
        draw.ellipse(bbox, fill=fill_color)
    measured_height = int((max(centers) - min(centers)) + radius * 2) if centers else int(radius * 2)
    measured_width = int(radius * 2)
    return {
        "selected_font_size": int(radius * 2),
        "selected_font_size_kind": "ellipsis_dot_diameter",
        "wrapped_lines": ["ellipsis"],
        "measured_rendered_width": measured_width,
        "measured_rendered_height": measured_height,
        "fit_ratio": max(measured_width / max(1, width), measured_height / max(1, height)),
    }


def _fit_vertical_font(
    text: str,
    box_width: int,
    box_height: int,
    font_name: str,
    preferred_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    line_height_scale: float = 1.0,
):
    micro_start = _renderer_micro_start()
    font_path = _find_font_path(font_name)
    cache_key = (
        str(text or ""),
        int(box_width or 0),
        int(box_height or 0),
        str(font_path or font_name or ""),
        int(preferred_size) if preferred_size is not None else None,
        int(min_size) if min_size is not None else None,
        int(max_size) if max_size is not None else None,
        round(float(line_height_scale or 0.0), 4),
    )
    fit_cache = _RENDERER_VERTICAL_FONT_FIT_CACHE.get()
    if fit_cache is not None and cache_key in fit_cache:
        _renderer_micro_add("renderer_fit_vertical_font_time", micro_start)
        return fit_cache[cache_key]
    _renderer_micro_count_key(
        "renderer_fit_vertical_font",
        cache_key,
    )
    sample = _sample_char(text)
    tokens = _vertical_tokens(text) or ["字"]
    short_vertical = len(tokens) <= 5
    punctuation = _vertical_punct_chars()
    vertical_lines = _vertical_line_tokens()
    punctuation_only = all(token in punctuation or token in vertical_lines for token in tokens)
    semantic_tokens = [token for token in tokens if token not in punctuation and token not in vertical_lines]
    content_count = len(semantic_tokens) or len(tokens)
    short_vertical = content_count <= 6
    if punctuation_only:
        width_factor = 0.96
        height_factor = 0.34
    elif content_count <= 2:
        width_factor = 0.95
        height_factor = 0.29
    elif short_vertical:
        width_factor = 0.94
        height_factor = 0.24
    else:
        width_factor = 0.92
        height_factor = 0.19
    start_size = min(72, max(10, int(min(box_width * width_factor, box_height * height_factor))))
    computed_min_size = max(8, int(min(box_height * 0.10, box_width * 0.45)))
    if punctuation_only:
        computed_min_size = max(computed_min_size, 20)
    elif content_count <= 2:
        computed_min_size = max(computed_min_size, 18)
    elif short_vertical and box_width <= 40:
        computed_min_size = max(computed_min_size, 14)
    elif short_vertical and box_width <= 52:
        computed_min_size = max(computed_min_size, 12)
    if min_size is not None and min_size > 0:
        computed_min_size = max(computed_min_size, int(min_size))
    source_max_size = int(max_size) if max_size is not None and max_size > 0 else 0
    if source_max_size > 0 and content_count >= 8:
        computed_min_size = min(computed_min_size, source_max_size)
    if preferred_size is not None and preferred_size > 0:
        start_size = max(start_size, int(preferred_size))
    if source_max_size > 0:
        start_size = min(start_size, source_max_size)
    if start_size < computed_min_size:
        start_size = computed_min_size
    if preferred_size is not None and preferred_size > 0:
        preferred_target = max(computed_min_size, int(preferred_size))
        if source_max_size > 0:
            preferred_target = min(preferred_target, source_max_size)
        preferred_fit = _largest_fitting_vertical_font_layout(
            tokens=tokens,
            box_width=box_width,
            box_height=box_height,
            font_path=font_path,
            sample=sample,
            high_size=preferred_target,
            low_size=preferred_target,
            line_height_scale=line_height_scale,
            content_count=content_count,
        )
        if preferred_fit is not None and _vertical_layout_has_readable_margin(
            tokens,
            preferred_fit[1],
            box_width,
            box_height,
            content_count,
        ):
            font, layout = preferred_fit
            return _renderer_vertical_font_fit_result(cache_key, font, layout, micro_start)
        if preferred_target > computed_min_size:
            comfortable_fit = _largest_comfortable_vertical_font_layout(
                tokens=tokens,
                box_width=box_width,
                box_height=box_height,
                font_path=font_path,
                sample=sample,
                high_size=preferred_target - 1,
                low_size=computed_min_size,
                line_height_scale=line_height_scale,
                content_count=content_count,
            )
            if comfortable_fit is not None:
                font, layout = comfortable_fit
                return _renderer_vertical_font_fit_result(cache_key, font, layout, micro_start)
        start_size = min(start_size, max(computed_min_size, preferred_target - 1))
    fitted = _largest_fitting_vertical_font_layout(
        tokens=tokens,
        box_width=box_width,
        box_height=box_height,
        font_path=font_path,
        sample=sample,
        high_size=start_size,
        low_size=computed_min_size,
        line_height_scale=line_height_scale,
        content_count=content_count,
    )
    if fitted is None and computed_min_size > 8:
        # When source-derived minimums are too large for a narrow vertical box,
        # prefer a smaller complete layout over silent truncation.
        fitted = _largest_fitting_vertical_font_layout(
            tokens=tokens,
            box_width=box_width,
            box_height=box_height,
            font_path=font_path,
            sample=sample,
            high_size=computed_min_size - 1,
            low_size=8,
            line_height_scale=line_height_scale,
            content_count=content_count,
        )
    if fitted is not None:
        font, layout = fitted
        return _renderer_vertical_font_fit_result(cache_key, font, layout, micro_start)
    font = _load_font(font_path, computed_min_size, sample)
    layout = _measure_vertical_layout(font, tokens, box_width, box_height, line_height_scale)
    if layout is None:
        cell_height = max(1, int(_text_height(font, "国") * line_height_scale * 1.04))
        col_width = max(1, int(max(_text_width(font, tok) for tok in tokens) * 1.08))
        col_gap = max(0, int(getattr(font, "size", 12) * 0.04))
        rows = max(1, box_height // max(1, cell_height))
        cols = 1
        while cols < len(tokens):
            total_width = cols * col_width + max(0, cols - 1) * col_gap
            if total_width > box_width:
                cols = max(1, cols - 1)
                break
            cols += 1
        required_cols = max(1, int(math.ceil(len(tokens) / max(1, rows))))
        cols = max(required_cols, min(len(tokens), cols))
        layout = (rows, cols, cell_height, col_width, col_gap)
    return _renderer_vertical_font_fit_result(cache_key, font, layout, micro_start)


def _renderer_vertical_font_fit_result(cache_key, font, layout, micro_start: float):
    fit_cache = _RENDERER_VERTICAL_FONT_FIT_CACHE.get()
    if fit_cache is not None:
        fit_cache[cache_key] = (font, layout)
    _renderer_micro_add("renderer_fit_vertical_font_time", micro_start)
    return font, layout


def _largest_fitting_vertical_font_layout(
    *,
    tokens: List[str],
    box_width: int,
    box_height: int,
    font_path: str,
    sample: str,
    high_size: int,
    low_size: int,
    line_height_scale: float,
    content_count: int,
):
    """Find the largest fitting vertical font size without a linear scan."""

    low = max(8, int(low_size or 8))
    high = max(low, int(high_size or low))
    best = None
    while low <= high:
        size = (low + high) // 2
        font = _load_font(font_path, size, sample)
        layout = _measure_vertical_layout(font, tokens, box_width, box_height, line_height_scale)
        if layout is None:
            high = size - 1
            continue
        best = (font, layout)
        low = size + 1
    if best is None:
        return None
    font, layout = best
    return font, _rebalance_long_narrow_vertical_layout(tokens, layout, box_width, box_height, content_count)


def _largest_comfortable_vertical_font_layout(
    *,
    tokens: List[str],
    box_width: int,
    box_height: int,
    font_path: str,
    sample: str,
    high_size: int,
    low_size: int,
    line_height_scale: float,
    content_count: int,
):
    low = max(8, int(low_size or 8))
    high = max(low, int(high_size or low))
    best = None
    while low <= high:
        size = (low + high) // 2
        font = _load_font(font_path, size, sample)
        layout = _measure_vertical_layout(font, tokens, box_width, box_height, line_height_scale)
        if layout is None:
            high = size - 1
            continue
        layout = _rebalance_long_narrow_vertical_layout(tokens, layout, box_width, box_height, content_count)
        if _vertical_layout_has_readable_margin(tokens, layout, box_width, box_height, content_count):
            best = (font, layout)
            low = size + 1
        else:
            high = size - 1
    return best


def _vertical_layout_has_readable_margin(
    tokens: List[str],
    layout,
    box_width: int,
    box_height: int,
    content_count: int,
) -> bool:
    if layout is None:
        return False
    rows, cols, cell_height, col_width, col_gap = layout
    cols = max(1, int(cols or 1))
    max_rows_used = 0
    for col in range(cols):
        max_rows_used = max(max_rows_used, len(tokens[col * rows : (col + 1) * rows]))
    max_rows_used = max(1, max_rows_used)
    height_usage = (max_rows_used * max(1, cell_height)) / max(1, box_height)
    width_usage = (
        cols * max(1, col_width) + max(0, cols - 1) * max(0, col_gap)
    ) / max(1, box_width)
    if content_count <= 3:
        height_limit, width_limit = 0.96, 0.96
    elif content_count <= 8:
        height_limit, width_limit = 0.92, 0.94
    elif content_count <= 14:
        height_limit, width_limit = 0.90, 0.92
    else:
        height_limit, width_limit = 0.88, 0.90
    return height_usage <= height_limit and width_usage <= width_limit


def _rebalance_long_narrow_vertical_layout(
    tokens: List[str],
    layout,
    box_width: int,
    box_height: int,
    content_count: int,
):
    rows, cols, cell_height, col_width, col_gap = layout
    if content_count < 18 or cols < 2:
        return layout
    if box_width > 130 or box_height < box_width * 3.0:
        return layout
    max_rows_used = 0
    for col in range(cols):
        max_rows_used = max(max_rows_used, len(tokens[col * rows : (col + 1) * rows]))
    if max_rows_used <= 0:
        return layout
    current_height_ratio = (max_rows_used * cell_height) / max(1, box_height)
    if current_height_ratio < 0.94:
        return layout
    balanced_rows = max(1, int(math.ceil(len(tokens) / max(1, cols))))
    punctuation = _vertical_punct_chars() | _vertical_line_tokens()
    while balanced_rows < rows and balanced_rows < len(tokens) and tokens[balanced_rows] in punctuation:
        balanced_rows += 1
    if balanced_rows >= rows:
        return layout
    balanced_height_ratio = (balanced_rows * cell_height) / max(1, box_height)
    if balanced_height_ratio > 0.90:
        return layout
    if balanced_rows < max(6, int(math.ceil(rows * 0.55))):
        return layout
    return balanced_rows, cols, cell_height, col_width, col_gap


def _measure_vertical_layout(font, tokens: List[str], box_width: int, box_height: int, line_height_scale: float):
    if not tokens:
        return None
    cell_height = max(1, int(_text_height(font, "国") * line_height_scale * 1.04))
    col_width = max(1, int(max(_text_width(font, tok) for tok in tokens) * 1.08))
    col_gap = max(0, int(getattr(font, "size", 12) * 0.04))
    rows = max(1, box_height // max(1, cell_height))
    if rows <= 0:
        return None
    cols_needed = (len(tokens) + rows - 1) // rows
    if cols_needed <= 0:
        cols_needed = 1
    total_width = cols_needed * col_width + max(0, cols_needed - 1) * col_gap
    if total_width > box_width:
        return None
    return rows, cols_needed, cell_height, col_width, col_gap


def _text_width(font, text: str) -> int:
    bbox = font.getbbox(text)
    return max(1, int(bbox[2] - bbox[0]))


def _draw_vertical_text(
    draw,
    text: str,
    box: Tuple[int, int, int, int],
    font_name: str,
    fill_color,
    stroke_width: int,
    stroke_color,
    line_height_scale: float = 1.0,
    preferred_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    compact_layout: bool = False,
):
    text = _trim_unmatched_quotes(text)
    x0, y0, x1, y1 = box
    outer_width = max(1, x1 - x0)
    outer_height = max(1, y1 - y0)
    if compact_layout:
        side_pad = max(1, int(outer_width * 0.02))
        top_pad = max(1, int(outer_height * 0.025))
        bottom_pad = max(1, int(outer_height * 0.015))
    else:
        side_pad = max(1, int(outer_width * 0.04))
        top_pad = max(2, int(outer_height * 0.05))
        bottom_pad = max(1, int(outer_height * 0.03))
    inner_x0 = min(x1 - 1, x0 + side_pad)
    inner_x1 = max(inner_x0 + 1, x1 - side_pad)
    inner_y0 = min(y1 - 1, y0 + top_pad)
    inner_y1 = max(inner_y0 + 1, y1 - bottom_pad)
    width = max(1, inner_x1 - inner_x0)
    height = max(1, inner_y1 - inner_y0)
    tokens = _vertical_tokens(text)
    if not tokens:
        return None
    if _is_vertical_ellipsis_text(text):
        return _draw_vertical_ellipsis(
            draw,
            (inner_x0, inner_y0, inner_x1, inner_y1),
            fill_color,
            stroke_width,
            stroke_color,
            preferred_size=preferred_size,
            min_size=min_size,
        )
        return
    base_font, layout = _fit_vertical_font(
        text,
        width,
        height,
        font_name,
        preferred_size=preferred_size,
        min_size=min_size,
        max_size=max_size,
        line_height_scale=line_height_scale,
    )
    rows, cols, cell_height, col_width, col_gap = layout
    total_width = cols * col_width + max(0, cols - 1) * col_gap
    start_x = inner_x0 + max(0, (width - total_width) // 2)
    punctuation = _vertical_punct_chars()
    vertical_lines = _vertical_line_tokens()
    wrapped_columns = []
    max_rows_used = 0
    for col in range(cols):
        col_tokens = tokens[col * rows : (col + 1) * rows]
        wrapped_columns.append("".join(col_tokens))
        max_rows_used = max(max_rows_used, len(col_tokens))
        col_x = start_x + (cols - 1 - col) * (col_width + col_gap)
        offset_y = inner_y0
        for row, token in enumerate(col_tokens):
            font = base_font
            local_stroke = stroke_width
            if token in punctuation:
                local_stroke = min(local_stroke, 1)
            if token in vertical_lines:
                cell_top = offset_y + row * cell_height
                cell_bottom = cell_top + cell_height
                center_x = col_x + max(0, col_width // 2)
                margin_y = max(1, int(cell_height * 0.12))
                line_width = max(1, int(max(col_width, getattr(base_font, "size", 12)) * 0.06))
                line_y0 = cell_top + margin_y
                line_y1 = cell_bottom - margin_y
                if stroke_color and local_stroke > 0:
                    draw.line(
                        (center_x, line_y0, center_x, line_y1),
                        fill=stroke_color,
                        width=line_width + local_stroke * 2,
                    )
                draw.line(
                    (center_x, line_y0, center_x, line_y1),
                    fill=fill_color,
                    width=line_width,
                )
                continue
            bbox = font.getbbox(token)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]
            draw_x = col_x + max(0, (col_width - char_w) // 2) - bbox[0]
            draw_y = offset_y + row * cell_height + max(0, (cell_height - char_h) // 2) - bbox[1]
            if token in {"，", "、", "。", "．", "：", "；"}:
                draw_x += max(0, int(col_width * 0.12))
                draw_y -= max(0, int(cell_height * 0.06))
            draw.text(
                (draw_x, draw_y),
                token,
                fill=fill_color,
                font=font,
                stroke_width=local_stroke,
                stroke_fill=stroke_color,
            )
    measured_height = max_rows_used * cell_height
    return {
        "selected_font_size": getattr(base_font, "size", None),
        "wrapped_lines": wrapped_columns,
        "measured_rendered_width": total_width,
        "measured_rendered_height": measured_height,
        "fit_ratio": max(total_width / max(1, width), measured_height / max(1, height)),
        "render_fit_compact_layout": bool(compact_layout),
    }


def _join_tokens(tokens: List[str], has_space: bool) -> str:
    if not tokens:
        return ""
    if not has_space:
        return "".join(tokens)
    combined = tokens[0]
    for token in tokens[1:]:
        if _is_punct_only(token):
            combined = f"{combined}{token}"
        else:
            combined = f"{combined} {token}"
    return combined


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [part.strip() for part in cleaned.split("\n") if part.strip()]
    if not parts:
        return ""
    if _has_cjk("".join(parts)):
        if len(parts) >= 2:
            first = parts[0]
            rest = "".join(parts[1:])
            first_body = re.sub(r"[，。！？：；、…\s]", "", first)
            if first and rest and len(first_body) <= 6 and first[-1] not in "，。！？：；、…,.!?;:":
                cleaned = f"{first}，{rest}"
            elif any(" " in part for part in parts):
                cleaned = " ".join(parts)
            else:
                cleaned = "".join(parts)
        else:
            cleaned = parts[0]
    else:
        cleaned = " ".join(parts)
    if _has_cjk(cleaned):
        cleaned = re.sub(r"[.．]{2,}", "…", cleaned)
        cleaned = re.sub(r"…{2,}", "…", cleaned)
        cleaned = cleaned.replace("．", "。")
        cleaned = cleaned.replace(":", "：").replace(";", "；")
        cleaned = cleaned.replace("!", "！").replace("?", "？")
        cleaned = cleaned.replace(".", "。")
        cleaned = re.sub(r"\s*([。，、！？：；…])\s*", r"\1", cleaned)
        cleaned = _trim_unmatched_quotes(cleaned)
    return cleaned.strip()


def _trim_unmatched_quotes(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""
    quote_pairs = (
        ("「", "」"),
        ("『", "』"),
        ("“", "”"),
        ("‘", "’"),
    )
    changed = True
    while changed and cleaned:
        changed = False
        for opener, closer in quote_pairs:
            if cleaned.startswith(opener) and cleaned.count(opener) > cleaned.count(closer):
                cleaned = cleaned[1:].lstrip()
                changed = True
            if cleaned.endswith(closer) and cleaned.count(closer) > cleaned.count(opener):
                cleaned = cleaned[:-1].rstrip()
                changed = True
    return cleaned


def _tokenize_text(text: str) -> List[str]:
    if " " in text:
        parts = [p for p in text.split(" ") if p]
        tokens: List[str] = []
        for part in parts:
            if _is_punct_only(part) and tokens:
                tokens[-1] = f"{tokens[-1]}{part}"
            else:
                tokens.append(part)
        return tokens
    tokens = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch in {".", "．"}:
            end = index
            while end < len(text) and text[end] in {".", "．"}:
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if ch == "…":
            end = index
            while end < len(text) and text[end] == "…":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_punct_char(ch) and tokens:
            tokens[-1] = f"{tokens[-1]}{ch}"
        else:
            tokens.append(ch)
        index += 1
    return tokens


def _is_punct_char(ch: str) -> bool:
    return ch in {
        "。",
        "．",
        "，",
        "、",
        "！",
        "？",
        "：",
        "；",
        "…",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "·",
        "—",
        "～",
        "…",
    }


def _is_punct_only(text: str) -> bool:
    if not text:
        return True
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    return all(_is_punct_char(ch) for ch in stripped)


def _fix_leading_punct(lines: List[str]) -> List[str]:
    if len(lines) <= 1:
        return lines
    fixed: List[str] = [lines[0]]
    for line in lines[1:]:
        line = line.lstrip()
        if line and _is_punct_char(line[0]):
            fixed[-1] = f"{fixed[-1]}{line[0]}"
            remainder = line[1:].lstrip()
            if remainder:
                fixed.append(remainder)
        else:
            fixed.append(line)
    return fixed


def _text_height(font, text: str) -> int:
    bbox = font.getbbox(text)
    return max(1, int(bbox[3] - bbox[1]))


def _measure_lines_height(font, lines: List[str], line_height_scale: float = 1.0) -> int:
    if not lines:
        return 0
    total = 0
    for line in lines:
        height = _text_height(font, line)
        total += max(1, int(height * line_height_scale))
    return total


def _fill_padding(w: int, h: int) -> int:
    base = max(12, int(min(w, h) * 0.35), int(max(w, h) * 0.1))
    return min(base, 80)


def _text_padding(w: int, h: int) -> int:
    base = max(6, int(min(w, h) * 0.12), int(max(w, h) * 0.03))
    return min(base, 28)


def _coerce_render_line_height(render: dict, source_orientation: str, region_type: str) -> float:
    try:
        raw = float(render.get("line_height", 1.0) or 1.0)
    except Exception:
        raw = 1.0
    wrap_mode = str(render.get("wrap_mode", "auto") or "auto").strip().lower()
    horizontal = wrap_mode == "horizontal" or str(source_orientation or "").strip().lower() == "horizontal"
    caption_like = str(region_type or "").strip().lower() in {"background_text", "decorative_text", "narration_box"}
    if horizontal:
        floor = 1.18 if caption_like else 1.14
        ceiling = 1.32
    else:
        floor = 1.10 if caption_like else 1.06
        ceiling = 1.22
    return max(floor, min(ceiling, raw))


def _expand_box(box, pad_x: int, pad_y: int, max_w: int, max_h: int):
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - max(0, int(pad_x))),
        max(0, y0 - max(0, int(pad_y))),
        min(max_w, x1 + max(0, int(pad_x))),
        min(max_h, y1 + max(0, int(pad_y))),
    )


def _fit_font(
    draw,
    text: str,
    max_width: int,
    max_height: int,
    font_name: str,
    preferred_size: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    line_height_scale: float = 1.0,
):
    micro_start = _renderer_micro_start()
    font_path = _find_font_path(font_name)
    _renderer_micro_count_key(
        "renderer_fit_font",
        (
            str(text or ""),
            int(max_width or 0),
            int(max_height or 0),
            str(font_path or font_name or ""),
            int(preferred_size) if preferred_size is not None else None,
            int(min_size) if min_size is not None else None,
            int(max_size) if max_size is not None else None,
            round(float(line_height_scale or 0.0), 4),
        ),
    )
    sample = _sample_char(text)
    start_size = min(72, max(12, int(min(max_height * 0.75, max_width * 1.15))))
    computed_min_size = max(8, int(min(max_height * 0.18, max_width * 0.55)))
    if min_size is not None and min_size > 0:
        computed_min_size = max(computed_min_size, int(min_size))
    if max_size is not None and max_size > 0:
        start_size = min(start_size, max(computed_min_size, int(max_size)))
    if preferred_size is not None:
        target = max(computed_min_size, min(int(preferred_size), start_size))
        font = _load_font(font_path, target, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = _measure_lines_height(font, lines, line_height_scale)
        if total_height <= max_height:
            _renderer_micro_add("renderer_fit_font_time", micro_start)
            return font
        for size in range(target - 1, computed_min_size - 1, -1):
            font = _load_font(font_path, size, sample)
            lines = _wrap_text(draw, text, font, max_width)
            total_height = _measure_lines_height(font, lines, line_height_scale)
            if total_height <= max_height:
                _renderer_micro_add("renderer_fit_font_time", micro_start)
                return font
    for size in range(start_size, computed_min_size - 1, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = _measure_lines_height(font, lines, line_height_scale)
        if total_height <= max_height:
            _renderer_micro_add("renderer_fit_font_time", micro_start)
            return font
    for size in range(computed_min_size - 1, 9, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = _measure_lines_height(font, lines, line_height_scale)
        if total_height <= max_height:
            _renderer_micro_add("renderer_fit_font_time", micro_start)
            return font
    font = _load_font(font_path, 8, sample)
    _renderer_micro_add("renderer_fit_font_time", micro_start)
    return font


def _load_font(font_path: str | None, size: int, sample: str):
    micro_start = _renderer_micro_start()
    cache_key = (str(font_path or ""), int(size or 0), str(sample or ""))
    _renderer_micro_count_key("renderer_font_load", cache_key)
    if ImageFont is None:
        raise RuntimeError("Pillow is not installed.")
    cache = _RENDERER_FONT_LOAD_CACHE.get()
    if cache is not None and cache_key in cache:
        add_count(_RENDERER_MICRO_CONTEXT.get(), "renderer_font_load_cache_hits")
        _renderer_micro_add("renderer_font_load_time", micro_start)
        return cache[cache_key]
    add_count(_RENDERER_MICRO_CONTEXT.get(), "renderer_font_load_cache_misses")
    candidates = []
    if font_path and os.path.exists(font_path):
        candidates.append(font_path)
    candidates.extend(_fallback_font_paths())
    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            font = ImageFont.truetype(path, size=size)
        except Exception:
            continue
        if _font_supports_text(font, sample):
            if cache is not None:
                cache[cache_key] = font
            _renderer_micro_add("renderer_font_load_time", micro_start)
            return font
    font = ImageFont.load_default()
    if cache is not None:
        cache[cache_key] = font
    _renderer_micro_add("renderer_font_load_time", micro_start)
    return font


def _font_supports_text(font, text: str) -> bool:
    if not text:
        return True
    try:
        mask = font.getmask(text)
        return mask.getbbox() is not None
    except Exception:
        return False


def _sample_char(text: str) -> str:
    for ch in text:
        if _is_cjk(ch):
            return ch
    return "A"


def _resolve_styled_font_name(base_font: str, font_style: str) -> str:
    return base_font or "Microsoft YaHei"


def _find_font_path(font_name: str) -> str | None:
    if not font_name:
        return None
    if os.path.isfile(font_name):
        return font_name
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return None
    lowered = font_name.lower().replace(" ", "")
    fallback_fonts = [
        "Noto Sans CJK",
        "Microsoft YaHei",
        "SimSun",
        "MS Gothic",
        "Yu Gothic",
        "Meiryo",
    ]
    for entry in os.listdir(fonts_dir):
        name, ext = os.path.splitext(entry)
        if ext.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        if lowered in name.lower().replace(" ", ""):
            return os.path.join(fonts_dir, entry)
    for font in fallback_fonts:
        lowered = font.lower().replace(" ", "")
        for entry in os.listdir(fonts_dir):
            name, ext = os.path.splitext(entry)
            if ext.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            if lowered in name.lower().replace(" ", ""):
                return os.path.join(fonts_dir, entry)
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            return candidate
    return None


def _fallback_font_paths() -> List[str]:
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return []
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    paths = []
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            paths.append(candidate)
    return paths


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(ch) for ch in text)


def _source_glyph_masks_by_region(source_glyph_masks: object | None) -> dict[str, object]:
    if source_glyph_masks is None:
        return {}
    masks = getattr(source_glyph_masks, "masks_by_region", None)
    if isinstance(masks, dict):
        return {str(key): value for key, value in masks.items()}
    if isinstance(source_glyph_masks, dict):
        if isinstance(source_glyph_masks.get("masks_by_region"), dict):
            return {str(key): value for key, value in source_glyph_masks["masks_by_region"].items()}
        return {str(key): value for key, value in source_glyph_masks.items()}
    return {}


def _source_glyph_record_mask(record: object | None):
    if record is None:
        return None
    if isinstance(record, dict):
        return record.get("mask")
    return getattr(record, "mask", None)


def _source_glyph_record_audit_fields(record: object | None) -> dict[str, object]:
    if record is None:
        return {}
    if hasattr(record, "render_audit_fields"):
        try:
            return dict(record.render_audit_fields())
        except Exception:
            return {}
    if isinstance(record, dict):
        return {
            "source_glyph_mask_id": record.get("source_glyph_mask_id") or record.get("mask_id"),
            "source_glyph_mask_generation_method": record.get("generation_method"),
            "source_glyph_mask_failure_reason": record.get("failure_reason"),
            "source_glyph_mask_parent_logical_text_unit_id": record.get("source_glyph_mask_parent_logical_text_unit_id") or record.get("parent_logical_text_unit_id"),
            "source_glyph_mask_text_block_root_id": record.get("source_glyph_mask_text_block_root_id") or record.get("text_block_root_id"),
            "source_glyph_mask_anchor_child_id": record.get("source_glyph_mask_anchor_child_id") or record.get("anchor_child_id"),
            "source_glyph_mask_child_segment_ids": record.get("source_glyph_mask_child_segment_ids") or record.get("child_segment_ids"),
            "cleanup_source_tracking_required": record.get("cleanup_source_tracking_required"),
            "source_glyph_mask_generation_required": record.get("source_glyph_mask_generation_required"),
            "source_glyph_mask_generation_status": record.get("source_glyph_mask_generation_status"),
            "source_glyph_mask_not_generated_reason": record.get("source_glyph_mask_not_generated_reason"),
            "source_glyph_mask_review_only": record.get("source_glyph_mask_review_only"),
            "source_glyph_mask_required": record.get("source_glyph_mask_required"),
            "source_glyph_mask_generated": record.get("source_glyph_mask_generated"),
            "source_glyph_mask_consumed_by_renderer": record.get("source_glyph_mask_consumed_by_renderer"),
            "source_glyph_mask_missing_reason": record.get("source_glyph_mask_missing_reason"),
            "source_glyph_mask_fallback_used": record.get("source_glyph_mask_fallback_used"),
            "source_glyph_mask_fallback_reason": record.get("source_glyph_mask_fallback_reason"),
            "source_glyph_erasure_bbox": record.get("source_glyph_erasure_bbox"),
            "source_glyph_erasure_expected_area_bbox": record.get("source_glyph_erasure_expected_area_bbox"),
            "source_glyph_erasure_expected_pixels": record.get("source_glyph_erasure_expected_pixels"),
            "source_glyph_erasure_coverage_ratio": record.get("source_glyph_erasure_coverage_ratio"),
            "cleanup_covers_source_glyphs": record.get("cleanup_covers_source_glyphs"),
            "cleanup_source_erasure_failure_reason": record.get("cleanup_source_erasure_failure_reason"),
            "cleanup_visual_artifact_risk": record.get("cleanup_visual_artifact_risk"),
            "phase2c_mask_adjustment_reason": record.get("phase2c_mask_adjustment_reason"),
        }
    return {}


def _is_cjk_unsupported_font(font_name: str) -> bool:
    if not font_name:
        return False
    name = font_name.lower()
    return "gothic" in name or "meiryo" in name
