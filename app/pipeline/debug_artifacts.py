# -*- coding: utf-8 -*-
"""Opt-in debug artifact writer for manga translation page diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from app.pipeline.debug_runtime import (
    debug_artifact_level,
    debug_disabled_stages,
    debug_enabled,
    debug_pages,
    debug_root,
    debug_stage_enabled,
    debug_stages,
    page_debug_dir,
    perf_telemetry_enabled,
    perf_telemetry_root,
    safe_trace_token,
    stage_artifact_dir,
    stage_artifact_path,
    write_image_path,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None

_PHASE2E_TARGET_ROOT_IDS: dict[str, str] = {}
_PHASE2E_TARGET_REGION_IDS: dict[tuple[str, str], str] = {}


def page_matches(page_name: str, pages: set[str]) -> bool:
    if not pages:
        return True
    stem = os.path.splitext(os.path.basename(page_name))[0]
    candidates = {stem}
    for match in re.findall(r"\d+", stem):
        candidates.add(match)
        candidates.add(match.zfill(3))
    return bool(candidates & pages)


def _phase2e_page_id(context: dict[str, Any] | None, audit: dict[str, Any] | None = None) -> str:
    value = ""
    if isinstance(context, dict):
        value = str(context.get("page_id") or "")
    if not value and isinstance(audit, dict):
        value = str(audit.get("page_id") or "")
    return value.strip()


def _phase2e_target_row_id(region: dict[str, Any], page_id: str) -> str:
    root_id = str(region.get("text_block_root_id") or region.get("root_id") or "").strip()
    if root_id in _PHASE2E_TARGET_ROOT_IDS:
        return _PHASE2E_TARGET_ROOT_IDS[root_id]
    region_id = str(region.get("region_id") or "").strip()
    return _PHASE2E_TARGET_REGION_IDS.get((page_id, region_id), "")


def _phase2e_append_debug_validation_trace(
    enriched: dict[str, Any],
    *,
    page_id: str,
    region: dict[str, Any],
    stage: str,
    fields: dict[str, Any],
) -> None:
    target_row_id = _phase2e_target_row_id(region, page_id)
    if not target_row_id:
        return
    record = {
        "phase2e_target_row_id": target_row_id,
        "stage": stage,
        "page_id": page_id,
        "region_id": region.get("region_id"),
        "root_id": region.get("text_block_root_id") or region.get("root_id"),
    }
    record.update(fields)
    enriched.setdefault("phase2e_debug_validation_trace", []).append(record)


def new_page_context(
    page_name: str,
    source_path: str,
    output_path: str,
    root_dir: str,
    settings: Any = None,
) -> dict[str, Any]:
    page_id = os.path.splitext(os.path.basename(page_name))[0]
    return {
        "enabled": True,
        "debug_artifact_level": debug_artifact_level(settings),
        "debug_stages": sorted(debug_stages(settings)),
        "debug_disabled_stages": sorted(debug_disabled_stages(settings)),
        "page_id": page_id,
        "source_path": source_path,
        "output_path": output_path,
        "debug_dir": root_dir,
        "page_class": None,
        "regions": {},
        "timing": {
            "total_page_time": None,
            "page_functional_time": 0.0,
            "page_postprocess_time": 0.0,
            "page_cycle_time": 0.0,
            "page_process_cpu_time": 0.0,
            "python_gc_time": 0.0,
            "cuda_cache_clear_time": 0.0,
            "memory_maintenance_time": 0.0,
            "image_loading_time": 0.0,
            "bubble_detection_time": 0.0,
            "text_area_plan_build_time": 0.0,
            "detection_time": 0.0,
            "detection_primary_time": 0.0,
            "detection_background_time": 0.0,
            "ocr_time": 0.0,
            "grouping_time": 0.0,
            "glossary_time": 0.0,
            "translation_time": 0.0,
            "logical_text_block_time": 0.0,
            "text_block_hierarchy_time": 0.0,
            "hierarchy_initial_build_time": 0.0,
            "root_reconstruction_time": 0.0,
            "parent_ocr_time": 0.0,
            "hierarchy_final_build_time": 0.0,
            "source_glyph_mask_time": 0.0,
            "mask_generation_time": 0.0,
            "inpainting_time": 0.0,
            "rendering_time": 0.0,
            "ui_controller_overhead_time": 0.0,
        },
        "counts": {
            "detected_regions": 0,
            "ocr_results": 0,
            "translated_regions": 0,
            "skipped_regions": 0,
            "inpaint_calls": 0,
            "ocr_request_count": 0,
            "translation_request_count": 0,
            "translation_request_payload_items": 0,
        },
        "worker_initialization": {},
        "ocr_request_summary": {},
        "translation_request_summary": {},
    }


def debug_stage_artifact_dir(context: dict[str, Any] | None, stage: str, *parts: str) -> str:
    return stage_artifact_dir(context, stage, *parts)


def write_debug_stage_image_path(
    context: dict[str, Any] | None,
    stage: str,
    path: str,
    image: Any,
    *,
    quality: int | None = None,
) -> tuple[str, bool, str]:
    return write_image_path(context, stage, path, image, quality=quality)


def write_debug_image_file(path: str, image: Any, *, quality: int | None = None) -> str:
    """Write a debug image to an already authorized artifact path."""

    if not path or image is None:
        return ""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not hasattr(image, "save"):
            return ""
        if quality is not None:
            image.save(path, quality=int(quality))
        else:
            image.save(path)
        return path
    except Exception:
        return ""


def write_cleanup_process_debug_artifacts(
    context: dict[str, Any] | None,
    *,
    source_image_path: str = "",
    image_size: Any = None,
    text_foreground_segmentation: Any = None,
    component_authorization_map: Any = None,
    source_glyph_masks: Any = None,
    cleanup_job_contracts: Any = None,
    cleanup_mask_contracts: Any = None,
    cleanup_plan_contracts: Any = None,
    cleanup_runtime_contracts: Any = None,
    cleanup_upstream_commit_result: Any = None,
    cleaned_page_base: Any = None,
) -> dict[str, Any]:
    """Write a compact cleanup-process ladder for debug diagnosis.

    This is intentionally a debug-only view. It records page-level masks and
    overlays at each cleanup boundary while keeping per-parent details in JSON.
    It does not change cleanup authorization, backend execution, proof, commit,
    or rendering behavior.
    """

    if not context or context.get("perf_telemetry_only") or Image is None:
        return {}
    artifact_dir = debug_stage_artifact_dir(context, "cleanup_process", "cleanup_process")
    if not artifact_dir:
        return {}
    size = (
        _normalize_image_size(image_size)
        or _normalize_image_size(getattr(text_foreground_segmentation, "image_size", None))
        or _image_size_from_path(source_image_path)
    )
    if not size:
        return {}
    page_id = str(context.get("page_id") or "page").strip() or "page"
    page_token = safe_trace_token(page_id, "page")
    source_image_ref = _copy_or_write_debug_image(
        os.path.join(artifact_dir, f"{page_token}_original.jpg"),
        source_image_path,
        None,
    )

    segmentation = {
        "audit": _json_safe(
            text_foreground_segmentation.to_audit_dict()
            if hasattr(text_foreground_segmentation, "to_audit_dict")
            else text_foreground_segmentation
        ),
        "raw_mask": _save_binary_mask_array(
            os.path.join(artifact_dir, f"{page_token}_ctd_raw_mask.png"),
            _binary_mask_array(getattr(text_foreground_segmentation, "raw_mask", None), size),
        ),
        "refined_mask": _save_binary_mask_array(
            os.path.join(artifact_dir, f"{page_token}_ctd_refined_mask.png"),
            _binary_mask_array(getattr(text_foreground_segmentation, "refined_mask", None), size),
        ),
    }
    segmentation["raw_overlay"] = _save_mask_overlay_array(
        os.path.join(artifact_dir, f"{page_token}_ctd_raw_overlay.jpg"),
        source_image_path,
        size,
        _binary_mask_array(getattr(text_foreground_segmentation, "raw_mask", None), size),
        (255, 150, 0),
    )
    segmentation["refined_overlay"] = _save_mask_overlay_array(
        os.path.join(artifact_dir, f"{page_token}_ctd_refined_overlay.jpg"),
        source_image_path,
        size,
        _binary_mask_array(getattr(text_foreground_segmentation, "refined_mask", None), size),
        (30, 144, 255),
    )
    segmentation["raw_mask_stats"] = _mask_stats(
        _binary_mask_array(getattr(text_foreground_segmentation, "raw_mask", None), size)
    )
    segmentation["refined_mask_stats"] = _mask_stats(
        _binary_mask_array(getattr(text_foreground_segmentation, "refined_mask", None), size)
    )

    component_records = _component_authorization_records(component_authorization_map)
    component_authorization = {
        "audit_summary": _json_safe(_component_authorization_summary(component_authorization_map)),
        "component_count": len(component_records),
        "state_counts": _component_state_counts(component_records),
        "authorization_overlay": _write_component_authorization_overlay(
            os.path.join(artifact_dir, f"{page_token}_component_authorization_overlay.jpg"),
            source_image_path,
            size,
            component_records,
            mode="auth",
        ),
        "projection_overlay": _write_component_authorization_overlay(
            os.path.join(artifact_dir, f"{page_token}_projection_quality_overlay.jpg"),
            source_image_path,
            size,
            component_records,
            mode="projection",
        ),
        "protected_overlay": _write_component_authorization_overlay(
            os.path.join(artifact_dir, f"{page_token}_protected_component_overlay.jpg"),
            source_image_path,
            size,
            component_records,
            mode="protected",
        ),
        "components": _json_safe(component_records),
    }

    source_glyph_union = _union_source_glyph_masks(source_glyph_masks, size)
    source_glyph_overlay_ref = _save_mask_overlay_array(
        os.path.join(artifact_dir, f"{page_token}_source_glyph_overlay.jpg"),
        source_image_path,
        size,
        source_glyph_union,
        (30, 144, 255),
    )
    source_glyph = {
        "summary": _json_safe(
            source_glyph_masks.to_audit_dict()
            if hasattr(source_glyph_masks, "to_audit_dict")
            else source_glyph_masks
        ),
        "union_mask": _save_binary_mask_array(
            os.path.join(artifact_dir, f"{page_token}_source_glyph_union.png"),
            source_glyph_union,
        ),
        "union_overlay": source_glyph_overlay_ref,
        "union_stats": _mask_stats(source_glyph_union),
    }

    masks = list(getattr(cleanup_mask_contracts, "masks", []) or [])
    foreground_union = _empty_binary_mask(size)
    erase_union = _empty_binary_mask(size)
    records: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        mask_id = str(getattr(mask, "cleanup_mask_id", "") or f"mask_{index:03d}")
        foreground_arr = _binary_mask_array(getattr(mask, "foreground_mask", None), size)
        erase_arr = _binary_mask_array(getattr(mask, "erase_mask", None), size)
        foreground_union = _mask_union(foreground_union, foreground_arr)
        erase_union = _mask_union(erase_union, erase_arr)
        records.append(
            {
                "cleanup_mask_id": mask_id,
                "cleanup_job_id": str(getattr(mask, "cleanup_job_id", "") or ""),
                "parent_execution_bundle_id": str(getattr(mask, "parent_execution_bundle_id", "") or ""),
                "parent_logical_text_unit_id": str(getattr(mask, "parent_logical_text_unit_id", "") or ""),
                "text_block_root_id": str(getattr(mask, "text_block_root_id", "") or ""),
                "target_region_ids": _json_safe(list(getattr(mask, "target_region_ids", []) or [])),
                "mask_source": str(getattr(mask, "mask_source", "") or ""),
                "mask_method": str(getattr(mask, "mask_method", "") or ""),
                "clean_mask_state": str(getattr(mask, "clean_mask_state", "") or ""),
                "clean_mask_failure_reason": str(getattr(mask, "clean_mask_failure_reason", "") or ""),
                "foreground_metadata_pixels": getattr(mask, "foreground_mask_pixels", None),
                "erase_metadata_pixels": getattr(mask, "erase_mask_pixels", None),
                "foreground_metadata_bbox": _json_safe(getattr(mask, "foreground_mask_bbox", None)),
                "erase_metadata_bbox": _json_safe(getattr(mask, "erase_mask_bbox", None)),
                "foreground_mask": _mask_stats(foreground_arr),
                "erase_mask": _mask_stats(erase_arr),
                "cleanup_authorization": str(getattr(mask, "cleanup_authorization", "") or ""),
                "semantic_authorization_state": str(getattr(mask, "semantic_authorization_state", "") or ""),
                "projection_quality_state": str(getattr(mask, "projection_quality_state", "") or ""),
                "projection_quality_reasons": _json_safe(list(getattr(mask, "projection_quality_reasons", []) or [])),
                "owned_component_ids": _json_safe(list(getattr(mask, "owned_component_ids", []) or [])),
                "protected_component_ids": _json_safe(list(getattr(mask, "protected_component_ids", []) or [])),
                "ambiguous_component_ids": _json_safe(list(getattr(mask, "ambiguous_component_ids", []) or [])),
                "unowned_component_ids": _json_safe(list(getattr(mask, "unowned_component_ids", []) or [])),
                "component_ownership_status": str(getattr(mask, "component_ownership_status", "") or ""),
                "clean_mask_foreground_ref": str(getattr(mask, "clean_mask_foreground_ref", "") or ""),
                "clean_mask_erase_ref": str(getattr(mask, "clean_mask_erase_ref", "") or ""),
                "component_ownership_overlay_ref": str(getattr(mask, "component_ownership_overlay_ref", "") or ""),
                "rejected_component_overlay_ref": str(getattr(mask, "rejected_component_overlay_ref", "") or ""),
            }
        )
    foreground_union_ref = os.path.join(artifact_dir, f"{page_token}_accepted_foreground_union.png")
    erase_union_ref = os.path.join(artifact_dir, f"{page_token}_accepted_erase_union.png")
    _save_binary_mask_array(foreground_union_ref, foreground_union)
    _save_binary_mask_array(erase_union_ref, erase_union)

    backend_input_ref_count = _runtime_mask_ref_count(cleanup_runtime_contracts, "backend_input_mask_ref")
    runtime_backend_input_union = (
        _union_runtime_mask_refs(
            cleanup_runtime_contracts,
            size,
            ref_attr="backend_input_mask_ref",
        )
        if backend_input_ref_count
        else _empty_binary_mask(size)
    )
    runtime_commit_union = _union_runtime_result_masks(cleanup_runtime_contracts, size)
    runtime = {
        "summary": _json_safe(
            cleanup_runtime_contracts.to_audit_dict()
            if hasattr(cleanup_runtime_contracts, "to_audit_dict")
            else cleanup_runtime_contracts
        ),
        "backend_input_union": (
            _save_binary_mask_array(
                os.path.join(artifact_dir, f"{page_token}_runtime_backend_input_union.png"),
                runtime_backend_input_union,
            )
            if backend_input_ref_count
            else ""
        ),
        "backend_input_union_stats": _mask_stats(runtime_backend_input_union) if backend_input_ref_count else {"pixels": 0, "bbox": None},
        "backend_input_union_status": (
            "captured_from_runtime_refs"
            if backend_input_ref_count
            else "not_captured_without_cleanup_runtime_stage_per_job_refs"
        ),
        "result_commit_union": _save_binary_mask_array(
            os.path.join(artifact_dir, f"{page_token}_runtime_result_commit_union.png"),
            runtime_commit_union,
        ),
        "result_commit_union_stats": _mask_stats(runtime_commit_union),
    }

    commit_mask_ref = str(getattr(cleanup_upstream_commit_result, "commit_mask_ref", "") or "")
    commit_cleaned_ref = str(getattr(cleanup_upstream_commit_result, "committed_image_ref", "") or "")
    commit_diff_ref = str(getattr(cleanup_upstream_commit_result, "commit_diff_ref", "") or "")
    commit_mask_copy = _copy_or_write_debug_image(
        os.path.join(artifact_dir, f"{page_token}_upstream_commit_mask.png"),
        commit_mask_ref,
        None,
    )
    if not commit_mask_copy and _mask_stats(runtime_commit_union).get("pixels", 0):
        commit_mask_copy = _save_binary_mask_array(
            os.path.join(artifact_dir, f"{page_token}_upstream_commit_mask.png"),
            runtime_commit_union,
        )
    cleaned_page_ref = _copy_or_write_debug_image(
        os.path.join(artifact_dir, f"{page_token}_cleaned_page.png"),
        commit_cleaned_ref,
        getattr(cleanup_upstream_commit_result, "cleaned_image", None),
    )
    commit_diff_copy = _copy_or_write_debug_image(
        os.path.join(artifact_dir, f"{page_token}_cleaned_diff.png"),
        commit_diff_ref,
        None,
    )
    if not commit_diff_copy and getattr(cleanup_upstream_commit_result, "cleaned_image", None) is not None:
        commit_diff_copy = _write_image_difference(
            os.path.join(artifact_dir, f"{page_token}_cleaned_diff.png"),
            source_image_path,
            getattr(cleanup_upstream_commit_result, "cleaned_image", None),
        )

    commit = {
        "summary": _json_safe(
            cleanup_upstream_commit_result.to_audit_dict()
            if hasattr(cleanup_upstream_commit_result, "to_audit_dict")
            else cleanup_upstream_commit_result
        ),
        "commit_mask": commit_mask_copy or commit_mask_ref,
        "cleaned_page": cleaned_page_ref or commit_cleaned_ref,
        "cleaned_diff": commit_diff_copy or commit_diff_ref,
        "cleaned_page_base": _json_safe(cleaned_page_base or {}),
    }

    summary = {
        "version": "cleanup_process_debug_v2",
        "page_id": page_id,
        "image_size": [int(size[0]), int(size[1])],
        "source_image_ref": source_image_ref,
        "mask_count": len(records),
        "stage_contract": (
            "This debug-only cleanup ladder records stage evidence. It does not authorize cleanup, "
            "execute cleanup, weaken proof, commit pixels, or affect rendering."
        ),
        "segmentation": segmentation,
        "component_authorization": component_authorization,
        "source_glyph": source_glyph,
        "cleanup_jobs": _compact_contract_summary(cleanup_job_contracts),
        "cleanup_masks": _compact_contract_summary(cleanup_mask_contracts),
        "cleanup_plans": _compact_contract_summary(cleanup_plan_contracts),
        "accepted_foreground_union": _mask_stats(foreground_union),
        "accepted_erase_union": _mask_stats(erase_union),
        "accepted_foreground_union_ref": foreground_union_ref,
        "accepted_erase_union_ref": erase_union_ref,
        "runtime": runtime,
        "commit": commit,
        "masks": records,
    }
    summary_ref = os.path.join(artifact_dir, f"{page_token}_cleanup_process_summary.json")
    _write_json_file(summary_ref, summary)
    context["cleanup_process_artifacts"] = summary
    return summary


def write_pre_render_source_erasure_debug_image(working: Any, debug_context: dict[str, Any] | None) -> None:
    if not debug_context or working is None:
        return
    page_id = str(debug_context.get("page_id") or "page")
    path = stage_artifact_path(debug_context, "cleanup_trace", f"{page_id}_pre_render_cleaned.jpg")
    if not path:
        return
    written_path, ok, error = write_image_path(debug_context, "cleanup_trace", path, working, quality=92)
    if ok:
        debug_context["pre_render_source_erasure_image_path"] = written_path
    elif error != "debug_stage_disabled":
        debug_context["pre_render_source_erasure_image_error"] = error


def new_perf_page_context(page_name: str, source_path: str, output_path: str, root_dir: str) -> dict[str, Any]:
    page_id = os.path.splitext(os.path.basename(page_name))[0]
    return {
        "enabled": False,
        "perf_telemetry_only": True,
        "debug_artifact_level": "off",
        "page_id": page_id,
        "source_path": source_path,
        "output_path": output_path,
        "perf_telemetry_dir": root_dir,
        "page_class": None,
        "regions": {},
        "timing": {
            "total_page_time": None,
            "page_functional_time": 0.0,
            "page_postprocess_time": 0.0,
            "page_cycle_time": 0.0,
            "page_process_cpu_time": 0.0,
            "python_gc_time": 0.0,
            "cuda_cache_clear_time": 0.0,
            "memory_maintenance_time": 0.0,
            "image_loading_time": 0.0,
            "bubble_detection_time": 0.0,
            "text_area_plan_build_time": 0.0,
            "detection_time": 0.0,
            "detection_primary_time": 0.0,
            "detection_background_time": 0.0,
            "ocr_time": 0.0,
            "grouping_time": 0.0,
            "glossary_time": 0.0,
            "translation_time": 0.0,
            "logical_text_block_time": 0.0,
            "text_block_hierarchy_time": 0.0,
            "hierarchy_initial_build_time": 0.0,
            "root_reconstruction_time": 0.0,
            "parent_ocr_time": 0.0,
            "hierarchy_final_build_time": 0.0,
            "source_glyph_mask_time": 0.0,
            "cleanup_mask_contract_time": 0.0,
            "text_area_component_authorization_time": 0.0,
            "cleanup_mask_build_time": 0.0,
            "cleanup_plan_build_time": 0.0,
            "cleanup_runtime_contract_time": 0.0,
            "cleanup_backend_time": 0.0,
            "cleanup_model_call_time": 0.0,
            "cleanup_proof_time": 0.0,
            "cleanup_runtime_overhead_time": 0.0,
            "cleanup_model_pre_page_warmup_time": 0.0,
            "cleanup_upstream_commit_time": 0.0,
            "render_eligibility_contract_time": 0.0,
            "mask_generation_time": 0.0,
            "renderer_region_loop_total_time": 0.0,
            "renderer_pre_render_contract_total_time": 0.0,
            "renderer_pre_render_proof_time": 0.0,
            "renderer_pre_render_expected_mask_time": 0.0,
            "renderer_cleanup_trace_metric_time": 0.0,
            "inpainting_time": 0.0,
            "rendering_time": 0.0,
            "output_save_time": 0.0,
            "ui_controller_overhead_time": 0.0,
        },
        "counts": {
            "detected_regions": 0,
            "ocr_results": 0,
            "translated_regions": 0,
            "skipped_regions": 0,
            "inpaint_calls": 0,
            "cleanup_ai_backend_calls": 0,
            "cleanup_result_records": 0,
            "cleanup_proof_records": 0,
            "ocr_request_count": 0,
            "translation_request_count": 0,
            "translation_request_payload_items": 0,
        },
        "worker_initialization": {},
        "ocr_request_summary": {},
        "translation_request_summary": {},
    }


def add_timing(context: dict[str, Any] | None, key: str, seconds: float) -> None:
    if not context:
        return
    timing = context.setdefault("timing", {})
    timing[key] = float(timing.get(key) or 0.0) + max(0.0, float(seconds or 0.0))


def set_timing(context: dict[str, Any] | None, key: str, seconds: float) -> None:
    if not context:
        return
    context.setdefault("timing", {})[key] = max(0.0, float(seconds or 0.0))


def add_count(context: dict[str, Any] | None, key: str, count: int = 1) -> None:
    if not context:
        return
    counts = context.setdefault("counts", {})
    counts[key] = int(counts.get(key) or 0) + int(count or 0)


def set_count(context: dict[str, Any] | None, key: str, count: int) -> None:
    if not context:
        return
    context.setdefault("counts", {})[key] = int(count or 0)


def mark_translation_plan(context: dict[str, Any] | None, regions: list[dict], pending_texts: dict[str, list[str]]) -> None:
    if not context:
        return
    pending_ids = {rid for ids in pending_texts.values() for rid in ids}
    region_meta = context.setdefault("regions", {})
    route_owned_translatable_count = 0
    route_owned_translation_queued_count = 0
    route_owned_ocr_warning_count = 0
    route_owned_ocr_blocker_count = 0
    for region in regions:
        rid = str(region.get("region_id", "") or "")
        if not rid:
            continue
        entry = region_meta.setdefault(rid, {})
        flags = region.get("flags", {}) or {}
        render = region.get("render", {}) or {}
        route_intent = str(region.get("text_area_route_intent") or render.get("text_area_route_intent") or "").strip()
        ocr_state = str(region.get("text_area_ocr_transaction_state") or render.get("text_area_ocr_transaction_state") or "").strip()
        if route_intent in {"translate_speech", "translate_caption_background"}:
            route_owned_translatable_count += 1
            represented_fragment = bool(
                region.get("route_owned_ocr_fragment_represented_by_parent")
                or render.get("route_owned_ocr_fragment_represented_by_parent")
            )
            ocr_blocked = (
                ocr_state in {"ocr_empty_blocker", "ocr_punctuation_only_blocker", "ocr_malformed_blocker"}
                or bool(region.get("translation_blocked_by_ocr_transaction"))
            ) and not represented_fragment
            ocr_warning = ocr_state == "recognized_low_confidence_warning"
            if ocr_blocked:
                route_owned_ocr_blocker_count += 1
            if ocr_warning:
                route_owned_ocr_warning_count += 1
            entry["route_intent"] = route_intent
            entry["text_area_ocr_transaction_state"] = ocr_state
            entry["text_area_ocr_blocked"] = ocr_blocked
            entry["route_owned_ocr_fragment_represented_by_parent"] = represented_fragment
            entry["text_area_ocr_warning"] = ocr_warning
            entry["text_area_ocr_blocker_reason"] = region.get("text_area_ocr_blocker_reason") or render.get("text_area_ocr_blocker_reason") or ""
            entry["text_area_ocr_warning_reason"] = region.get("text_area_ocr_warning_reason") or render.get("text_area_ocr_warning_reason") or ""
            region["route_owned_ocr_blocked"] = ocr_blocked
            region["route_owned_ocr_fragment_represented_by_parent"] = represented_fragment
            region["route_owned_ocr_warning"] = ocr_warning
        if rid in pending_ids:
            entry["sent_to_translation"] = True
            entry["skip_reason"] = None
            region["sent_to_translation"] = True
            region["skip_reason"] = None
            if route_intent in {"translate_speech", "translate_caption_background"}:
                entry["route_owned_translation_queued"] = True
                region["route_owned_translation_queued"] = True
                route_owned_translation_queued_count += 1
        elif flags.get("ignore"):
            entry["sent_to_translation"] = False
            entry["skip_reason"] = "ignored_by_pipeline"
            region["sent_to_translation"] = False
            region["skip_reason"] = "ignored_by_pipeline"
            if route_intent in {"translate_speech", "translate_caption_background"}:
                entry["route_owned_translation_queued"] = False
                region["route_owned_translation_queued"] = False
        elif str(region.get("translation", "") or "").strip():
            entry["sent_to_translation"] = False
            entry["skip_reason"] = "translation_cache_or_pretranslated"
            region["sent_to_translation"] = False
            region["skip_reason"] = "translation_cache_or_pretranslated"
            if route_intent in {"translate_speech", "translate_caption_background"}:
                entry["route_owned_translation_queued"] = True
                region["route_owned_translation_queued"] = True
                route_owned_translation_queued_count += 1
        else:
            entry["sent_to_translation"] = False
            entry["skip_reason"] = "not_queued_for_translation"
            region["sent_to_translation"] = False
            region["skip_reason"] = "not_queued_for_translation"
            if route_intent in {"translate_speech", "translate_caption_background"}:
                entry["route_owned_translation_queued"] = False
                region["route_owned_translation_queued"] = False
    set_count(context, "route_owned_translatable_count", route_owned_translatable_count)
    set_count(context, "route_owned_translation_queued_count", route_owned_translation_queued_count)
    set_count(context, "route_owned_ocr_warning_count", route_owned_ocr_warning_count)
    set_count(context, "route_owned_ocr_blocker_count", route_owned_ocr_blocker_count)


def mark_render_region(context: dict[str, Any] | None, region_id: str, **fields: Any) -> None:
    if not context or not region_id:
        return
    entry = context.setdefault("regions", {}).setdefault(str(region_id), {})
    for key, value in fields.items():
        entry[key] = _json_safe(value)


def mask_stats(mask) -> dict[str, Any] | None:
    if mask is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(mask)
        ys, xs = np.where(arr > 0)
        pixels = int(ys.size)
        if pixels <= 0:
            return {"pixels": 0, "bbox": None}
        return {
            "pixels": pixels,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
        }
    except Exception:
        return None


def write_page_artifacts(context: dict[str, Any] | None, regions: list[dict]) -> None:
    if not context:
        return
    page_id = str(context.get("page_id") or "page")
    page_dir = page_debug_dir(context)
    if not page_dir:
        return

    audit = _build_audit(context, regions)
    audit = _add_rollback_forbidden_marker_scan(audit, regions)
    audit = _maybe_add_text_area_diagnostics(audit, regions, audit_path=os.path.join(page_dir, f"{page_id}_region_audit.json"))
    audit = _maybe_add_route_advisor(audit)
    audit = _maybe_add_render_planner(audit)
    audit = _maybe_write_text_area_overlays(context, audit, page_dir)
    audit = _maybe_write_source_glyph_mask_overlay(context, audit, page_dir)
    audit = _maybe_add_source_erasure_visual_validation(context, audit)
    audit = _maybe_add_caption_background_visual_evidence(context, audit)
    audit = _maybe_add_root_final_state_closeout(context, audit)
    audit = _maybe_write_text_block_hierarchy_artifacts(context, audit, page_dir)
    audit = _maybe_write_text_area_plan_artifacts(context, audit, page_dir)
    if debug_stage_enabled(context, "audit"):
        audit_path = os.path.join(page_dir, f"{page_id}_region_audit.json")
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2)

    if debug_stage_enabled(context, "timing"):
        timing_path = os.path.join(page_dir, f"{page_id}_timing.json")
        with open(timing_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "page_id": page_id,
                    "page_class": context.get("page_class"),
                    "debug_artifact_level": context.get("debug_artifact_level") or debug_artifact_level(),
                    "debug_stages": context.get("debug_stages", []),
                    "debug_disabled_stages": context.get("debug_disabled_stages", []),
                    "timing": context.get("timing", {}),
                    "counts": context.get("counts", {}),
                    "scoped_ocr_trace": context.get("scoped_ocr_trace", []),
                    "route_owned_ocr_retry_attempts": context.get("route_owned_ocr_retry_attempts", []),
                    "render_cleanup_operations": context.get("render_cleanup_operations", []),
                    "cleanup_operation_traces": context.get("cleanup_operation_traces", []),
                    "phase2e_cleanup_propagation_trace": context.get("phase2e_cleanup_propagation_trace", []),
                    "translation_unit_timings": context.get("translation_unit_timings", []),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    write_translation_review_artifacts(context, page_dir)
    if debug_stage_enabled(context, "overlay"):
        overlay_path = os.path.join(page_dir, f"{page_id}_overlay.jpg")
        _write_overlay(context, audit["regions"], overlay_path)


def write_perf_timing_artifact(context: dict[str, Any] | None, regions: list[dict]) -> str:
    if not context:
        return ""
    page_id = str(context.get("page_id") or "page")
    root_dir = str(context.get("perf_telemetry_dir") or "").strip()
    if not root_dir:
        output_path = str(context.get("output_path") or "")
        export_dir = os.path.dirname(output_path) if output_path else ""
        root_dir = os.path.join(export_dir, "performance_timing") if export_dir else "performance_timing"
    os.makedirs(root_dir, exist_ok=True)
    payload = {
        "page_id": page_id,
        "perf_telemetry_only": True,
        "source_path": context.get("source_path") or "",
        "output_path": context.get("output_path") or "",
        "output_sha256": _sha256_file(context.get("output_path") or ""),
        "timing": _json_safe(context.get("timing") or {}),
        "counts": _json_safe(context.get("counts") or {}),
        "cleanup_runtime_summary": _json_safe(context.get("cleanup_runtime_summary") or {}),
        "cleanup_job_timings": _json_safe(context.get("cleanup_job_timings") or []),
        "cleanup_model_warmup": _json_safe(context.get("cleanup_model_warmup") or {}),
        "worker_initialization": _json_safe(context.get("worker_initialization") or {}),
        "ocr_request_summary": _json_safe(context.get("ocr_request_summary") or {}),
        "translation_request_summary": _json_safe(context.get("translation_request_summary") or {}),
        "watch_regions": _perf_watch_regions(context, regions),
    }
    timing_path = os.path.join(root_dir, f"{page_id}_timing.json")
    with open(timing_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return timing_path


def append_perf_timing_overhead_artifact(
    context: dict[str, Any] | None,
    *,
    telemetry_artifact_write_time: float,
    observed_page_cycle_with_telemetry: float,
) -> str:
    """Record telemetry's own write cost outside the page timing payload."""

    if not context:
        return ""
    root_dir = str(context.get("perf_telemetry_dir") or "").strip()
    if not root_dir:
        output_path = str(context.get("output_path") or "")
        export_dir = os.path.dirname(output_path) if output_path else ""
        root_dir = os.path.join(export_dir, "performance_timing") if export_dir else "performance_timing"
    os.makedirs(root_dir, exist_ok=True)
    path = os.path.join(root_dir, "_telemetry_overhead.jsonl")
    record = {
        "page_id": str(context.get("page_id") or "page"),
        "telemetry_artifact_write_time": round(max(0.0, float(telemetry_artifact_write_time or 0.0)), 6),
        "observed_page_cycle_with_telemetry": round(
            max(0.0, float(observed_page_cycle_with_telemetry or 0.0)),
            6,
        ),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perf_watch_regions(context: dict[str, Any], regions: list[dict]) -> dict[str, Any]:
    page_id = str(context.get("page_id") or "")
    watch_by_page: dict[str, set[str]] = {}
    watch_ids = watch_by_page.get(page_id, set())
    if not watch_ids:
        return {}
    region_meta = context.get("regions") if isinstance(context.get("regions"), dict) else {}
    regions_by_id = {
        str(region.get("region_id") or ""): region
        for region in regions or []
        if isinstance(region, dict) and str(region.get("region_id") or "")
    }
    result: dict[str, Any] = {}
    for region_id in sorted(watch_ids):
        region = regions_by_id.get(region_id, {})
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        meta = region_meta.get(region_id, {}) if isinstance(region_meta, dict) else {}
        source_status = (
            meta.get("source_grounding_status")
            or region.get("source_grounding_status")
            or render.get("source_grounding_status")
        )
        unsafe_status = (
            meta.get("unsafe_cleanup_render_status")
            or region.get("unsafe_cleanup_render_status")
            or render.get("unsafe_cleanup_render_status")
            or meta.get("render_visual_contract_audit_status")
        )
        suppressed_reason = (
            meta.get("translated_text_suppressed_reason")
            or region.get("translated_text_suppressed_reason")
            or render.get("translated_text_suppressed_reason")
        )
        result[region_id] = {
            "region_present": bool(region),
            "source_grounding_status": source_status,
            "unsafe_cleanup_render_status": unsafe_status,
            "translated_text_suppressed": (
                meta.get("translated_text_suppressed")
                or region.get("translated_text_suppressed")
                or render.get("translated_text_suppressed")
            ),
            "translated_text_suppressed_reason": suppressed_reason,
            "visible_unsuppressed": not bool(source_status or unsafe_status or suppressed_reason),
        }
    return result


def _add_rollback_forbidden_marker_scan(audit: dict[str, Any], regions: list[dict]) -> dict[str, Any]:
    """Record whether inactive rollback-era ownership markers leaked into active regions."""
    inv_prefix = "root_" + "text_" + "inventory"
    graph_prefix = "root_" + "child_" + "graph"
    inv_synthetic = inv_prefix + "_synthetic_region"
    graph_synthetic = graph_prefix + "_v3_" + "synthetic_" + "parent_region"
    forbidden_ids: list[str] = []
    for region in regions or []:
        if not isinstance(region, dict):
            continue
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        active_source = str(region.get("text_area_detection_source") or render.get("text_area_detection_source") or "")
        has_marker = bool(
            region.get(inv_synthetic)
            or render.get(inv_synthetic)
            or region.get(graph_synthetic)
            or render.get(graph_synthetic)
            or active_source.startswith(inv_prefix + "_")
            or active_source.startswith(graph_prefix + "_")
        )
        if has_marker:
            forbidden_ids.append(str(region.get("region_id") or "unknown"))
    enriched = dict(audit)
    enriched["rollback_forbidden_marker_count"] = len(forbidden_ids)
    enriched["rollback_forbidden_marker_region_ids"] = forbidden_ids
    enriched["rollback_guard_status"] = "clear" if not forbidden_ids else "blocked"
    return enriched


def write_translation_review_artifacts(context: dict[str, Any], page_dir: str) -> None:
    """Write OCR-to-translation review artifacts owned by debug mode."""

    if not debug_stage_enabled(context, "translation_review"):
        return
    records = context.get("translation_unit_timings") or []
    if not isinstance(records, list) or not records:
        return
    page_id = str(context.get("page_id") or "page")
    json_path = os.path.join(page_dir, f"{page_id}_translation_unit_timings.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    headers = [
        "source_text",
        "root_id",
        "parent_logical_text_unit_id",
        "translation_path",
        "llm_call_count",
        "total_unit_latency_sec",
        "max_prompt_char_count",
        "source_text_length",
        "japanese_char_count",
        "punctuation_ellipsis_ratio",
        "translated_text_length",
        "cache_status",
        "json_repair_fallback_status",
        "failure_retry_reason",
        "pre_ensure_translation",
        "post_ensure_translation",
        "pre_ensure_language_ok",
        "pre_ensure_bad_shape",
        "pre_ensure_bad_shape_reasons",
        "pre_ensure_kana_ratio",
        "pre_ensure_chinese_ratio",
        "pre_ensure_prompt_leak",
        "pre_ensure_repetition_loop",
        "pre_ensure_merged_batch_output",
        "pre_ensure_source_reuse",
        "pre_ensure_format_artifact_reasons",
        "translation_format_normalized",
        "translation_before_format_normalization",
        "translation_after_format_normalization",
        "translation_format_normalization_reasons",
        "translation_format_normalization_stages",
        "ensure_retry_required",
        "ensure_retry_required_reason",
        "ensure_retry_skipped_reason",
        "ensure_retry_soft_warning_reasons",
        "ensure_retry_hard_failure_reasons",
        "ensure_retry_unresolved_format_warning_reasons",
        "batch_output_acceptance_status",
        "batch_initial_size",
        "batch_effective_size",
        "batch_chunk_index",
        "batch_latency_sec",
        "batch_empty_translation_ratio",
        "adaptive_batch_split_trigger",
        "adaptive_batch_next_effective_size",
        "compact_repair_group_size",
        "compact_repair_group_count",
        "compact_repair_group_index",
        "compact_repair_accepted_count",
        "compact_repair_failed_ids",
        "compact_repair_single_fallback_count",
        "unit_origin",
        "evidence_scopes",
    ]
    md_path = os.path.join(page_dir, f"{page_id}_translation_unit_timings.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for record in records:
            if not isinstance(record, dict):
                continue
            row = []
            for header in headers:
                value = record.get(header)
                if isinstance(value, (list, tuple)):
                    value = ",".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                row.append(str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>"))
            handle.write("| " + " | ".join(row) + " |\n")


def _maybe_write_text_area_plan_artifacts(context: dict[str, Any], audit: dict[str, Any], page_dir: str) -> dict[str, Any]:
    text_area_enabled = debug_stage_enabled(context, "text_area_plan")
    graph_enabled = debug_stage_enabled(context, "root_parent_child")
    if not text_area_enabled and not graph_enabled:
        return audit
    plan = context.get("text_area_plan")
    if not plan:
        return audit
    try:
        from app.pipeline.text_area_plan import (
            write_root_parent_child_overlay_artifacts,
            write_text_area_plan_artifacts,
        )

        if text_area_enabled:
            paths = write_text_area_plan_artifacts(
                page_dir=page_dir,
                image_path=context.get("source_path") or "",
                plan=plan,
                pre_ocr_plan=context.get("text_area_plan_pre_ocr") or plan,
                scoped_detection_candidates=context.get("scoped_detection_candidates") or [],
                scoped_ocr_candidates=context.get("scoped_ocr_candidates") or [],
                fallback_decisions=context.get("fallback_decisions") or [],
                blocked_text_area_candidates=context.get("blocked_text_area_candidates") or [],
            )
        else:
            if isinstance(plan, dict):
                plan_payload = plan
            elif hasattr(plan, "to_dict"):
                plan_payload = plan.to_dict()
            else:
                plan_payload = {}
            graph_plan = plan_payload.get("root_parent_child_plan") if isinstance(plan_payload, dict) else None
            paths = write_root_parent_child_overlay_artifacts(
                page_dir=page_dir,
                image_path=context.get("source_path") or "",
                graph_plan=graph_plan if isinstance(graph_plan, dict) else {},
            )
        enriched = dict(audit)
        enriched["text_area_plan_version"] = plan.get("version") if isinstance(plan, dict) else None
        enriched["text_area_plan_generated"] = bool(plan.get("generated")) if isinstance(plan, dict) else False
        enriched["text_area_plan"] = plan
        enriched["text_area_plan_artifact_paths"] = paths
        enriched["scoped_detection_candidates"] = context.get("scoped_detection_candidates") or []
        enriched["scoped_ocr_candidates"] = context.get("scoped_ocr_candidates") or []
        enriched["fallback_decisions"] = context.get("fallback_decisions") or []
        enriched["blocked_text_area_candidates"] = context.get("blocked_text_area_candidates") or []
        enriched["caption_container_recovery_candidates"] = context.get("caption_container_recovery_candidates") or []
        enriched["caption_component_recovery_candidates"] = context.get("caption_component_recovery_candidates") or []
        enriched["caption_localization_candidates"] = _caption_localization_candidates(plan)
        enriched["text_area_plan_pre_ocr"] = context.get("text_area_plan_pre_ocr") or plan
        return enriched
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched = dict(audit)
        enriched["text_area_plan_generated"] = False
        enriched["text_area_plan_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def _caption_localization_candidates(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for container in plan.get("containers", []) or []:
        if not isinstance(container, dict) or container.get("container_type") != "caption_background":
            continue
        candidates.append(
            {
                "candidate_id": f"caption_{container.get('container_id')}",
                "container_id": container.get("container_id"),
                "bbox": container.get("bbox"),
                "route_intent": container.get("route_intent"),
                "ocr_eligible": bool(container.get("ocr_eligible")),
                "comic_text_detector_scope_eligible": bool(container.get("comic_text_detector_scope_eligible")),
                "confidence_tier": container.get("confidence_tier"),
                "fallback_reason": container.get("fallback_reason"),
                "source_model_ids": container.get("source_model_ids") or [],
                "reason_codes": container.get("evidence_reason_codes") or [],
                "conflict_flags": container.get("conflict_flags") or [],
                "would_change_behavior": False,
            }
        )
    return candidates


def _maybe_write_text_area_overlays(context: dict[str, Any], audit: dict[str, Any], page_dir: str) -> dict[str, Any]:
    if not debug_stage_enabled(context, "text_area_diagnostics"):
        return audit
    if not audit.get("diagnostic_generated"):
        return audit
    try:
        from app.pipeline.text_area_diagnostics import write_text_area_diagnostic_overlays

        enriched = dict(audit)
        overlay_result = write_text_area_diagnostic_overlays(enriched, page_dir)
        enriched["diagnostic_overlay_paths"] = overlay_result
        return enriched
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["diagnostic_overlay_paths"] = {"generated": False, "error": str(exc)}
        return enriched


def _maybe_write_source_glyph_mask_overlay(
    context: dict[str, Any],
    audit: dict[str, Any],
    page_dir: str,
) -> dict[str, Any]:
    if not debug_stage_enabled(context, "source_glyph"):
        return audit
    if Image is None or ImageDraw is None:
        return audit
    source_info = context.get("source_glyph_masks") or {}
    if not isinstance(source_info, dict):
        return audit
    records = source_info.get("source_glyph_mask_coverage_records") or []
    if not isinstance(records, list):
        records = []
    if not records:
        return audit
    base_path = str(context.get("source_path") or "")
    if not os.path.isfile(base_path):
        return audit
    try:
        with Image.open(base_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = _overlay_font()
        legend = (
            "SourceGlyphMask: magenta=mask bbox, blue=allowed area, "
            "orange=erasure bbox, gray=missing required"
        )
        draw.rectangle((8, 8, 820, 34), fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        draw.text((14, 12), legend, fill=(0, 0, 0), font=font)
        regions_by_id = {
            str(region.get("region_id") or ""): region
            for region in audit.get("regions", []) or []
            if isinstance(region, dict) and str(region.get("region_id") or "")
        }
        for record in records:
            if not isinstance(record, dict):
                continue
            rid = str(record.get("region_id") or "")
            region = regions_by_id.get(rid) or {}
            if isinstance(region, dict):
                merged = dict(record)
                for key in (
                    "source_glyph_mask_required",
                    "cleanup_source_tracking_required",
                    "source_glyph_mask_generation_required",
                    "source_glyph_mask_generated",
                    "source_glyph_mask_consumed_by_renderer",
                    "source_glyph_mask_generation_status",
                    "source_glyph_mask_requirement_status",
                    "source_glyph_mask_generation_method",
                    "source_glyph_mask_missing_reason",
                    "source_glyph_mask_not_generated_reason",
                    "source_glyph_mask_review_only",
                    "source_glyph_mask_fallback_used",
                    "source_glyph_mask_fallback_reason",
                    "source_glyph_erasure_bbox",
                    "source_glyph_erasure_expected_area_bbox",
                    "source_glyph_erasure_expected_pixels",
                    "source_glyph_erasure_coverage_ratio",
                    "cleanup_covers_source_glyphs",
                    "source_glyph_mask_expected_bbox",
                    "source_glyph_mask_actual_bbox",
                    "source_glyph_mask_expected_overlap_ratio",
                    "source_glyph_mask_container_clip_reason",
                    "caption_background_mask_generation_method",
                    "caption_background_mask_coverage_ratio",
                    "phase2c_mask_adjustment_reason",
                    "side_caption_mask_generation_method",
                    "side_caption_mask_coverage_ratio",
                    "represented_child_cleanup_required",
                    "represented_child_cleanup_mask_generated",
                    "represented_child_cleanup_mask_consumed",
                    "represented_child_cleanup_coverage_ratio",
                    "represented_child_cleanup_failure_reason",
                    "cleanup_visual_artifact_risk",
                    "cleanup_source_erasure_failure_reason",
                ):
                    value = region.get(key)
                    if value is not None:
                        merged[key] = value
                record = merged
            generated = bool(record.get("source_glyph_mask_generated"))
            consumed = bool(record.get("source_glyph_mask_consumed_by_renderer"))
            generation_required = bool(
                record.get("source_glyph_mask_generation_required")
                or record.get("source_glyph_mask_required")
            )
            review_only = bool(record.get("source_glyph_mask_review_only"))
            mask_box = _xyxy(record.get("mask_bbox"))
            allowed_box = _xyxy(record.get("cleanup_allowed_area"))
            erasure_box = _xyxy(record.get("source_glyph_erasure_bbox"))
            if allowed_box:
                draw.rectangle(allowed_box, outline=(0, 90, 255), width=2)
            if erasure_box:
                draw.rectangle(erasure_box, outline=(255, 150, 0), width=2)
            if mask_box:
                draw.rectangle(mask_box, outline=(220, 0, 220), width=3 if consumed else 2)
                label_box = mask_box
            else:
                label_box = _bbox_xyxy(region.get("bbox"))
                if generation_required and label_box:
                    draw.rectangle(label_box, outline=(140, 140, 140), width=2)
            if label_box:
                status = str(record.get("source_glyph_mask_generation_status") or "")
                state = (
                    "consumed"
                    if consumed
                    else "review-only"
                    if review_only and generated
                    else "generated"
                    if generated
                    else "missing"
                    if generation_required
                    else "not-generated"
                )
                reason = (
                    record.get("source_glyph_mask_missing_reason")
                    or record.get("source_glyph_mask_not_generated_reason")
                    or status
                    or record.get("source_glyph_mask_generation_method")
                    or ""
                )
                label = f"{rid} {state} {reason}"
                x0, y0, _x1, _y1 = label_box
                ty = max(0, int(y0) - 16)
                draw.rectangle((int(x0), ty, int(x0) + min(760, len(label) * 7 + 8), ty + 16), fill=(255, 255, 230))
                draw.text((int(x0) + 4, ty + 1), label, fill=(0, 0, 0), font=font)
        overlay_path = os.path.join(page_dir, f"{context.get('page_id')}_source_glyph_mask_overlay.jpg")
        canvas.save(overlay_path, quality=92)
        enriched = dict(audit)
        enriched["source_glyph_mask_overlay_path"] = overlay_path
        return enriched
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched = dict(audit)
        enriched["source_glyph_mask_overlay_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def _maybe_add_source_erasure_visual_validation(
    context: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    if Image is None:
        return audit
    source_path = str(context.get("source_path") or "")
    output_path = str(context.get("output_path") or "")
    pre_render_path = str(context.get("pre_render_source_erasure_image_path") or "")
    validation_path = pre_render_path if os.path.isfile(pre_render_path) else output_path
    validation_stage = "pre_render_cleaned_image" if validation_path == pre_render_path and pre_render_path else "post_render_output_fallback"
    if not os.path.isfile(source_path) or not os.path.isfile(validation_path):
        return audit
    try:
        import numpy as np

        with Image.open(source_path) as src_img:
            src = np.asarray(src_img.convert("L"), dtype=np.int16)
        with Image.open(validation_path) as out_img:
            out = np.asarray(out_img.convert("L"), dtype=np.int16)
        if src.shape != out.shape:
            return audit
        enriched = dict(audit)
        page_id = _phase2e_page_id(context, audit)
        enriched["source_erasure_visual_validation_stage"] = validation_stage
        enriched["source_erasure_visual_validation_image_path"] = validation_path
        regions = [dict(region) for region in audit.get("regions", []) or []]
        render_boxes_by_parent: dict[str, list[tuple[int, int, int, int]]] = {}
        render_boxes_by_root: dict[str, list[tuple[int, int, int, int]]] = {}
        for region in regions:
            if not str(region.get("translated_text") or "").strip():
                continue
            render_box = _xyxy(region.get("final_render_bbox"))
            if not render_box:
                continue
            parent_id = str(region.get("parent_logical_text_unit_id") or "")
            root_id = str(region.get("text_block_root_id") or "")
            if parent_id:
                render_boxes_by_parent.setdefault(parent_id, []).append(render_box)
            if root_id:
                render_boxes_by_root.setdefault(root_id, []).append(render_box)
        for region in regions:
            if not _region_needs_source_erasure_visual_check(region):
                continue
            box = (
                _xyxy(region.get("source_glyph_erasure_bbox"))
                or _xyxy(region.get("cleanup_mask_bbox"))
            )
            if not box and region.get("source_child_cleanup_required") is True:
                box = _bbox_xyxy(region.get("bbox"))
            if not box:
                continue
            x0, y0, x1, y1 = _clip_box(box, src.shape[1], src.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            src_crop = src[y0:y1, x0:x1]
            out_crop = out[y0:y1, x0:x1]
            src_dark = src_crop < 135
            src_dark_count = int(src_dark.sum())
            if src_dark_count < 20:
                previous_class = region.get("source_erasure_warning_class")
                previous_status = region.get("pre_render_source_erasure_status")
                region["source_erasure_visual_residual_score"] = 0.0
                region["source_erasure_visual_residual_score_raw"] = 0.0
                region["source_erasure_visual_residual_dark_pixels"] = src_dark_count
                region["source_erasure_warning_class"] = "benign_texture_nonblocking"
                region["source_erasure_warning_reason"] = "too_few_source_dark_pixels_for_glyph_residual"
                _phase2e_append_debug_validation_trace(
                    enriched,
                    page_id=page_id,
                    region=region,
                    stage="debug_validation_too_few_source_dark_pixels",
                    fields={
                        "validation_stage": validation_stage,
                        "validation_image_path": validation_path,
                        "validation_bbox": [x0, y0, x1, y1],
                        "source_dark_pixels": src_dark_count,
                        "previous_source_erasure_warning_class": previous_class,
                        "previous_pre_render_source_erasure_status": previous_status,
                        "selected_source_erasure_warning_class": region.get("source_erasure_warning_class"),
                        "warning_overwrote_previous": previous_class != region.get("source_erasure_warning_class"),
                    },
                )
                continue
            unchanged_dark = src_dark & (out_crop < 150) & (np.abs(out_crop - src_crop) < 35)
            raw_residual = float(unchanged_dark.sum()) / float(src_dark_count)
            if validation_stage == "pre_render_cleaned_image":
                outside_eval = np.ones_like(src_dark, dtype=bool)
                outside_dark_count = src_dark_count
                outside_unchanged = int(unchanged_dark.sum())
                render_overlap_dark = 0
            else:
                render_exclusion = _render_overlap_mask_for_region(
                    region,
                    (x0, y0, x1, y1),
                    render_boxes_by_parent,
                    render_boxes_by_root,
                    np,
                )
                if render_exclusion is not None:
                    outside_eval = ~render_exclusion
                    outside_dark_count = int((src_dark & outside_eval).sum())
                    outside_unchanged = int((unchanged_dark & outside_eval).sum())
                    render_overlap_dark = max(0, src_dark_count - outside_dark_count)
                else:
                    outside_eval = np.ones_like(src_dark, dtype=bool)
                    outside_dark_count = src_dark_count
                    outside_unchanged = int(unchanged_dark.sum())
                    render_overlap_dark = 0
            residual_mask = unchanged_dark & outside_eval
            residual = float(outside_unchanged) / float(outside_dark_count) if outside_dark_count >= 20 else 0.0
            render_overlap_ratio = float(render_overlap_dark) / float(src_dark_count) if src_dark_count else 0.0
            shape_summary = _source_residual_shape_summary(residual_mask, np)
            region["source_erasure_visual_residual_score_raw"] = round(raw_residual, 3)
            region["source_erasure_visual_residual_score"] = round(residual, 3)
            region["source_erasure_residual_outside_render_dark_pixels"] = outside_dark_count
            region["source_erasure_residual_overlap_with_render_ratio"] = round(render_overlap_ratio, 3)
            region["source_erasure_visual_residual_dark_pixels"] = src_dark_count
            region["source_erasure_visual_validation_bbox"] = [x0, y0, x1, y1]
            region["source_erasure_visual_validation_stage"] = validation_stage
            region.update(shape_summary)
            previous_class = region.get("source_erasure_warning_class")
            previous_status = region.get("pre_render_source_erasure_status")
            previous_failure = region.get("cleanup_source_erasure_failure_reason")
            warning_class = _source_erasure_warning_class(
                region,
                residual,
                raw_residual,
                render_overlap_ratio,
                shape_summary,
            )
            region["source_erasure_warning_class"] = warning_class
            region["source_erasure_warning_reason"] = _source_erasure_warning_reason(
                region,
                warning_class,
                residual,
                raw_residual,
                render_overlap_ratio,
            )
            if warning_class in {
                "true_source_residual_blocker",
                "mask_generation_failure",
                "represented_child_cleanup_missing",
                "cleanup_backend_failure",
            }:
                region["cleanup_visual_validation_failed"] = True
                reason = str(region.get("cleanup_source_erasure_failure_reason") or "").strip()
                if reason:
                    region["cleanup_source_erasure_failure_reason"] = reason + "," + region["source_erasure_warning_reason"]
                else:
                    region["cleanup_source_erasure_failure_reason"] = region["source_erasure_warning_reason"]
            else:
                region["cleanup_visual_validation_failed"] = False
                if warning_class in {
                    "benign_texture_nonblocking",
                    "repaired_source_residual",
                    "translated_text_overlap",
                    "bubble_border_or_art_line",
                }:
                    region["cleanup_source_erasure_failure_reason"] = None
                    if _source_erasure_visual_validation_repairs_pre_render_failure(region, warning_class):
                        region["previous_pre_render_source_erasure_status"] = region.get("pre_render_source_erasure_status")
                        region["previous_pre_render_source_erasure_failure_reason"] = region.get(
                            "pre_render_source_erasure_failure_reason"
                        )
                        region["source_erasure_status_attribution_repaired"] = True
                        region["source_erasure_status_attribution_repair_reason"] = (
                            "repaired_source_residual_below_threshold"
                        )
                        region["pre_render_source_erasure_status"] = "passed_visual_validation"
                        region["pre_render_source_erasure_failure_reason"] = None
                        region["source_erasure_failure_reason"] = None
                        region["source_erasure_proof_action"] = "audit_passed_visual_validation"
            _phase2e_append_debug_validation_trace(
                enriched,
                page_id=page_id,
                region=region,
                stage="debug_validation_warning_class_selected",
                fields={
                    "validation_stage": validation_stage,
                    "validation_image_path": validation_path,
                    "validation_bbox": [x0, y0, x1, y1],
                    "raw_residual": round(raw_residual, 3),
                    "residual": round(residual, 3),
                    "render_overlap_ratio": round(render_overlap_ratio, 3),
                    "outside_dark_count": outside_dark_count,
                    "outside_unchanged_dark_count": outside_unchanged,
                    "source_dark_pixels": src_dark_count,
                    "shape_summary": shape_summary,
                    "previous_source_erasure_warning_class": previous_class,
                    "previous_pre_render_source_erasure_status": previous_status,
                    "previous_cleanup_source_erasure_failure_reason": previous_failure,
                    "selected_source_erasure_warning_class": warning_class,
                    "selected_source_erasure_warning_reason": region.get("source_erasure_warning_reason"),
                    "cleanup_visual_validation_failed": region.get("cleanup_visual_validation_failed"),
                    "cleanup_source_erasure_failure_reason": region.get("cleanup_source_erasure_failure_reason"),
                    "warning_overwrote_previous": previous_class != warning_class,
                    "pre_render_status_preserved": previous_status == region.get("pre_render_source_erasure_status"),
                },
            )
        enriched["regions"] = regions
        return enriched
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["source_erasure_visual_validation_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def _maybe_add_caption_background_visual_evidence(
    context: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    if Image is None:
        return audit
    hierarchy = context.get("text_block_hierarchy") or audit.get("text_block_hierarchy")
    if not isinstance(hierarchy, dict) or not hierarchy.get("text_block_hierarchy_generated"):
        return audit
    source_path = str(context.get("source_path") or "")
    if not os.path.isfile(source_path):
        return audit
    try:
        import numpy as np

        with Image.open(source_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        enriched_hierarchy = dict(hierarchy)
        roots = [dict(root) for root in hierarchy.get("text_area_root_blocks", []) or [] if isinstance(root, dict)]
        caption_records: list[dict[str, Any]] = []
        caption_recovery_candidates = audit.get("caption_container_recovery_candidates") or context.get(
            "caption_container_recovery_candidates"
        ) or []
        caption_component_candidates = audit.get("caption_component_recovery_candidates") or context.get(
            "caption_component_recovery_candidates"
        ) or []
        recovery_by_container: dict[str, list[dict[str, Any]]] = {}
        for candidate in caption_recovery_candidates:
            if not isinstance(candidate, dict):
                continue
            cid = str(candidate.get("text_area_container_id") or "").strip()
            if cid:
                recovery_by_container.setdefault(cid, []).append(candidate)
        components_by_container: dict[str, list[dict[str, Any]]] = {}
        for candidate in caption_component_candidates:
            if not isinstance(candidate, dict):
                continue
            cid = str(candidate.get("text_area_container_id") or "").strip()
            if cid:
                components_by_container.setdefault(cid, []).append(candidate)
        for root in roots:
            if root.get("root_type") != "caption_background" and root.get("route_policy") != "translate_caption":
                continue
            bbox = _bbox_xyxy(root.get("bbox"))
            score_record = _caption_textlike_score(gray, bbox, np) if bbox else {
                "caption_background_visual_textlike_score": 0.0,
                "caption_background_visual_textlike_component_count": 0,
                "caption_background_visibility_status": "invalid_caption_root_bbox",
            }
            has_parent = bool(root.get("root_accepted_parent_ids") or root.get("parent_unit_ids"))
            recovery_records: list[dict[str, Any]] = []
            component_records: list[dict[str, Any]] = []
            for cid in [str(item) for item in (root.get("text_area_container_ids") or []) if str(item)]:
                recovery_records.extend(recovery_by_container.get(cid, []))
                component_records.extend(components_by_container.get(cid, []))
            sfx_record = _caption_background_sfx_rejection_assessment(root, score_record, recovery_records)
            score_record.update(sfx_record)
            component_summary = _caption_component_split_summary(component_records, recovery_records)
            score_record.update(component_summary)
            if (
                not has_parent
                and score_record.get("caption_background_sfx_rejection_status")
                == "rejected_sfx_decorative_art_candidate"
            ):
                status = "sfx_or_decorative_art_evidence_not_caption_source"
                recovery_v2_status = "nonblocking_sfx_or_decorative_caption_candidate"
                recovery_v2_reason = score_record.get("caption_background_sfx_rejection_reason") or "sfx_decorative_caption_candidate"
                recovery_v3_status = "rejected_sfx_decorative_art_candidate"
                recovery_v3_reason = recovery_v2_reason
                recovery_v4_status = "rejected_sfx_decorative_art_candidate"
                recovery_v4_reason = recovery_v2_reason
            elif (
                not has_parent
                and bool(component_summary.get("caption_component_v4_nonblocking_mixed_art_review"))
            ):
                status = "sfx_or_decorative_art_evidence_not_caption_source"
                recovery_v2_status = "root_local_caption_recovery_failed_explicit_blocker"
                recovery_v2_reason = component_summary.get("caption_component_nonrecoverable_reason") or "caption_component_v4_mixed_art_review"
                recovery_v3_status = "root_local_caption_recovery_failed_explicit_blocker"
                recovery_v3_reason = recovery_v2_reason
                recovery_v4_status = "nonblocking_mixed_art_review"
                recovery_v4_reason = (
                    component_summary.get("caption_component_v4_rejection_reasons")
                    or component_summary.get("caption_component_nonrecoverable_reason")
                    or "caption_component_v4_art_dominated_no_safe_caption_component"
                )
            elif not has_parent and score_record["caption_background_visual_textlike_score"] >= 0.45:
                status = "textlike_glyph_evidence_without_source"
                recovery_v2_status = "root_local_caption_recovery_failed_explicit_blocker"
                recovery_v2_reason = (
                    component_summary.get("caption_component_nonrecoverable_reason")
                    or "textlike_glyph_evidence_without_accepted_caption_source"
                )
                recovery_v3_status = "root_local_caption_recovery_failed_explicit_blocker"
                recovery_v3_reason = recovery_v2_reason
                recovery_v4_status = component_summary.get("caption_component_v4_status") or "root_local_caption_recovery_failed_explicit_blocker"
                recovery_v4_reason = (
                    component_summary.get("caption_component_v4_rejection_reasons")
                    or recovery_v2_reason
                )
            elif not has_parent:
                status = "no_visible_textlike_glyph_evidence"
                recovery_v2_status = "nonblocking_no_visible_caption_source"
                recovery_v2_reason = "caption_root_has_no_meaningful_textlike_glyph_evidence"
                recovery_v3_status = "nonblocking_no_visible_caption_source"
                recovery_v3_reason = recovery_v2_reason
                recovery_v4_status = "nonblocking_no_visible_caption_source"
                recovery_v4_reason = recovery_v2_reason
            else:
                status = "accepted_caption_source_present"
                recovery_v2_status = "accepted_root_owned_caption_source"
                recovery_v2_reason = "caption_background_parent_unit_accepted"
                recovery_v3_status = "accepted_root_owned_caption_source"
                recovery_v3_reason = recovery_v2_reason
                recovery_v4_status = "accepted_root_owned_caption_source"
                recovery_v4_reason = recovery_v2_reason
            score_record["caption_background_visibility_status"] = status
            score_record["caption_background_recovery_v2_status"] = recovery_v2_status
            score_record["caption_background_recovery_v2_reason"] = recovery_v2_reason
            score_record["caption_background_recovery_v3_status"] = recovery_v3_status
            score_record["caption_background_recovery_v3_reason"] = recovery_v3_reason
            score_record["caption_background_recovery_v4_status"] = recovery_v4_status
            score_record["caption_background_recovery_v4_reason"] = recovery_v4_reason
            root.update(score_record)
            root["caption_component_recovery_records"] = component_records
            caption_records.append({"root_id": root.get("root_id"), **score_record})
        enriched_hierarchy["text_area_root_blocks"] = roots
        enriched_hierarchy["caption_background_visual_evidence_records"] = caption_records
        enriched_hierarchy["caption_component_recovery_records"] = caption_component_candidates
        enriched = dict(audit)
        enriched["text_block_hierarchy"] = enriched_hierarchy
        enriched["text_area_root_blocks"] = roots
        context["text_block_hierarchy"] = enriched_hierarchy
        return enriched
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["caption_background_visual_evidence_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def _caption_textlike_score(gray, bbox: tuple[int, int, int, int] | None, np_module) -> dict[str, Any]:
    if not bbox:
        return {
            "caption_background_visual_textlike_score": 0.0,
            "caption_background_visual_textlike_component_count": 0,
            "caption_background_dark_pixel_ratio": 0.0,
            "caption_background_component_count": 0,
            "caption_background_large_component_count": 0,
            "caption_background_line_like_component_count": 0,
            "caption_background_largest_component_area": 0,
            "caption_background_large_component_area_ratio": 0.0,
            "caption_background_line_like_area_ratio": 0.0,
        }
    x0, y0, x1, y1 = _clip_box(bbox, gray.shape[1], gray.shape[0])
    if x1 <= x0 or y1 <= y0:
        return {
            "caption_background_visual_textlike_score": 0.0,
            "caption_background_visual_textlike_component_count": 0,
            "caption_background_dark_pixel_ratio": 0.0,
            "caption_background_component_count": 0,
            "caption_background_large_component_count": 0,
            "caption_background_line_like_component_count": 0,
            "caption_background_largest_component_area": 0,
            "caption_background_large_component_area_ratio": 0.0,
            "caption_background_line_like_area_ratio": 0.0,
        }
    crop = gray[y0:y1, x0:x1]
    dark = crop < 110
    dark_ratio = float(dark.sum()) / float(max(1, dark.size))
    # Count compact-ish connected dark components without adding a hard cv2 dependency.
    try:
        import cv2

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark.astype("uint8"), 8)
        textlike = 0
        component_count = 0
        large_component_count = 0
        line_like_component_count = 0
        total_component_area = 0
        large_component_area = 0
        line_like_area = 0
        largest_component_area = 0
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < 8:
                continue
            component_count += 1
            total_component_area += area
            largest_component_area = max(largest_component_area, area)
            aspect = max(w / max(1, h), h / max(1, w))
            if area > max(1200, int(dark.size * 0.045)) or w > crop.shape[1] * 0.55 or h > crop.shape[0] * 0.55:
                large_component_count += 1
                large_component_area += area
            if aspect > 8.0:
                line_like_component_count += 1
                line_like_area += area
            if area > max(1200, int(dark.size * 0.18)):
                continue
            if w <= 1 or h <= 1:
                continue
            if w > crop.shape[1] * 0.8 or h > crop.shape[0] * 0.8:
                continue
            if aspect > 14:
                continue
            textlike += 1
        score = min(1.0, (textlike / 4.0) + min(0.25, dark_ratio * 2.0))
    except Exception:
        textlike = 0
        component_count = 0
        large_component_count = 0
        line_like_component_count = 0
        total_component_area = 0
        large_component_area = 0
        line_like_area = 0
        largest_component_area = 0
        score = min(1.0, dark_ratio * 2.0)
    large_ratio = float(large_component_area) / float(max(1, total_component_area))
    line_like_ratio = float(line_like_area) / float(max(1, total_component_area))
    return {
        "caption_background_visual_textlike_score": round(float(score), 3),
        "caption_background_visual_textlike_component_count": int(textlike),
        "caption_background_dark_pixel_ratio": round(float(dark_ratio), 4),
        "caption_background_component_count": int(component_count),
        "caption_background_large_component_count": int(large_component_count),
        "caption_background_line_like_component_count": int(line_like_component_count),
        "caption_background_largest_component_area": int(largest_component_area),
        "caption_background_large_component_area_ratio": round(float(large_ratio), 3),
        "caption_background_line_like_area_ratio": round(float(line_like_ratio), 3),
    }


def _caption_background_sfx_rejection_assessment(
    root: dict[str, Any],
    score_record: dict[str, Any],
    recovery_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    sfx_score = 0.0
    rejected_texts = [
        str(record.get("ocr_text") or "").strip()
        for record in recovery_records
        if str(record.get("status") or "").startswith("rejected") and str(record.get("ocr_text") or "").strip()
    ]
    meaningful_rejected = [
        text for text in rejected_texts if _caption_rejected_ocr_text_is_meaningful_caption(text)
    ]
    sfx_rejected = [
        text for text in rejected_texts if _caption_rejected_ocr_text_looks_sfx_or_punctuation(text)
    ]
    if rejected_texts and len(sfx_rejected) == len(rejected_texts):
        sfx_score += 0.45
        reasons.append("all_rejected_ocr_candidates_are_sfx_or_punctuation")
    if meaningful_rejected:
        reasons.append("meaningful_rejected_caption_ocr_candidate_present")
        sfx_score -= 0.35
    dark_ratio = _float(score_record.get("caption_background_dark_pixel_ratio")) or 0.0
    large_ratio = _float(score_record.get("caption_background_large_component_area_ratio")) or 0.0
    line_ratio = _float(score_record.get("caption_background_line_like_area_ratio")) or 0.0
    component_count = int(_float(score_record.get("caption_background_component_count")) or 0)
    textlike_count = int(_float(score_record.get("caption_background_visual_textlike_component_count")) or 0)
    large_count = int(_float(score_record.get("caption_background_large_component_count")) or 0)
    line_count = int(_float(score_record.get("caption_background_line_like_component_count")) or 0)
    reason_text = " ".join(str(item) for item in (root.get("reason_codes") or []))
    if "dark_or_art_context" in reason_text and large_ratio >= 0.28:
        sfx_score += 0.25
        reasons.append("dark_or_art_context_with_large_impact_components")
    if large_ratio >= 0.42 and large_count >= 2:
        sfx_score += 0.25
        reasons.append("large_noncaption_stroke_components")
    if large_ratio >= 0.30 and large_count >= 3 and textlike_count >= 4 and not meaningful_rejected:
        sfx_score += 0.25
        reasons.append("caption_root_dominated_by_impact_stroke_cluster")
    if line_ratio >= 0.22 or line_count >= max(3, textlike_count // 3):
        sfx_score += 0.20
        reasons.append("motion_line_or_impact_stroke_geometry")
    if component_count and textlike_count / max(1, component_count) < 0.35 and large_ratio >= 0.25:
        sfx_score += 0.15
        reasons.append("low_compact_text_fraction")
    if dark_ratio >= 0.50 and large_ratio >= 0.35:
        sfx_score += 0.20
        reasons.append("dense_art_fill_not_caption_column")
    status = "not_rejected"
    if (
        sfx_score >= 0.50
        or (
            sfx_score >= 0.45
            and rejected_texts
            and len(sfx_rejected) == len(rejected_texts)
        )
    ) and not meaningful_rejected:
        status = "rejected_sfx_decorative_art_candidate"
    return {
        "caption_background_sfx_artlike_score": round(max(0.0, min(1.0, sfx_score)), 3),
        "caption_background_sfx_rejection_status": status,
        "caption_background_sfx_rejection_reason": ",".join(reasons),
        "caption_background_rejected_ocr_texts": rejected_texts,
        "caption_background_meaningful_rejected_ocr_texts": meaningful_rejected,
    }


def _caption_component_split_summary(records: list[dict[str, Any]], recovery_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "caption_component_split_attempted": False,
            "caption_component_count": 0,
            "caption_component_caption_like_count": 0,
            "caption_component_sfx_artlike_count": 0,
            "caption_component_accepted_count": 0,
            "caption_component_rejected_count": 0,
            "caption_component_nonrecoverable_reason": "",
            "caption_component_v4_attempted": False,
            "caption_component_v4_status": "not_attempted",
            "caption_component_v4_candidate_count": 0,
            "caption_component_v4_accepted_count": 0,
            "caption_component_v4_rejected_count": 0,
            "caption_component_v4_nonblocking_mixed_art_review": False,
            "caption_component_v4_rejection_reasons": "",
            "caption_component_v4_accepted_text": "",
            "caption_component_v4_candidate_bboxes": [],
            "caption_component_v4_reading_order": [],
        }
    caption_like = 0
    sfx_like = 0
    accepted = 0
    rejected = 0
    scheduled = 0
    reasons: list[str] = []
    v4_attempted = False
    v4_candidate_count = 0
    v4_accepted = 0
    v4_rejected = 0
    v4_reasons: list[str] = []
    v4_accepted_texts: list[str] = []
    v4_candidate_bboxes: list[list[int]] = []
    v4_reading_orders: list[Any] = []
    for record in records:
        role = str(record.get("component_role") or record.get("caption_component_role") or "")
        status = str(record.get("status") or "")
        if record.get("caption_component_v4_candidate") or record.get("caption_component_v4_candidate_id"):
            v4_attempted = True
            if record.get("caption_component_v4_candidate_id"):
                v4_candidate_count += 1
            bbox = record.get("bbox") or record.get("component_bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                v4_candidate_bboxes.append([int(float(v or 0)) for v in bbox[:4]])
            if record.get("caption_component_v4_reading_order"):
                v4_reading_orders.append(record.get("caption_component_v4_reading_order"))
        if role == "caption_like":
            caption_like += 1
        if role in {"sfx_decorative_art_like", "punctuation_or_noise"}:
            sfx_like += 1
        if status.startswith("accepted"):
            accepted += 1
        if status.startswith("rejected"):
            rejected += 1
            reason = str(record.get("rejection_reason") or record.get("component_reason") or "")
            if reason:
                reasons.append(reason)
            if record.get("caption_component_v4_candidate") or record.get("caption_component_v4_candidate_id"):
                v4_rejected += 1
                if reason:
                    v4_reasons.append(reason)
        if "scheduled" in status:
            scheduled += 1
    for record in recovery_records:
        component_id = str(record.get("caption_component_id") or "").strip()
        if not component_id:
            continue
        status = str(record.get("status") or "")
        if record.get("caption_component_v4_candidate_id"):
            v4_attempted = True
        if status.startswith("accepted"):
            accepted += 1
            if record.get("caption_component_v4_candidate_id"):
                v4_accepted += 1
                text = str(record.get("ocr_text") or "").strip()
                if text:
                    v4_accepted_texts.append(text)
        if status.startswith("rejected"):
            rejected += 1
            reason = str(record.get("reason") or record.get("status") or "")
            if reason:
                reasons.append(reason)
            if record.get("caption_component_v4_candidate_id"):
                v4_rejected += 1
                if reason:
                    v4_reasons.append(reason)
            if record.get("caption_component_v4_reading_order"):
                v4_reading_orders.append(record.get("caption_component_v4_reading_order"))
    nonrecoverable = ""
    if accepted <= 0:
        if caption_like <= 0:
            nonrecoverable = "caption_component_split_no_caption_like_components"
        elif scheduled > 0 and rejected > 0:
            nonrecoverable = "caption_component_ocr_rejected_or_empty"
        elif reasons:
            nonrecoverable = ",".join(sorted(set(reasons)))
        else:
            nonrecoverable = "caption_component_recovery_unaccepted"
    art_dominated = (
        sfx_like >= max(20, caption_like * 4)
        or any("component_cluster_covers_most_of_root" in reason for reason in reasons)
    )
    v4_nonblocking_mixed = bool(v4_attempted and v4_accepted <= 0 and art_dominated)
    if v4_accepted > 0:
        v4_status = "accepted_root_owned_caption_source"
    elif v4_nonblocking_mixed:
        v4_status = "nonblocking_mixed_art_review"
    elif v4_attempted:
        v4_status = "recovery_failed_explicit_review"
    else:
        v4_status = "not_attempted"
    return {
        "caption_component_split_attempted": True,
        "caption_component_count": len(records),
        "caption_component_caption_like_count": caption_like,
        "caption_component_sfx_artlike_count": sfx_like,
        "caption_component_accepted_count": accepted,
        "caption_component_rejected_count": rejected,
        "caption_component_nonrecoverable_reason": nonrecoverable,
        "caption_component_v4_attempted": v4_attempted,
        "caption_component_v4_status": v4_status,
        "caption_component_v4_candidate_count": v4_candidate_count,
        "caption_component_v4_accepted_count": v4_accepted,
        "caption_component_v4_rejected_count": v4_rejected,
        "caption_component_v4_nonblocking_mixed_art_review": v4_nonblocking_mixed,
        "caption_component_v4_rejection_reasons": ",".join(sorted(set(reason for reason in v4_reasons if reason))),
        "caption_component_v4_accepted_text": " / ".join(v4_accepted_texts),
        "caption_component_v4_candidate_bboxes": v4_candidate_bboxes[:12],
        "caption_component_v4_reading_order": v4_reading_orders[:12],
    }


def _caption_rejected_ocr_text_is_meaningful_caption(text: str) -> bool:
    body = "".join(ch for ch in str(text or "") if ch.strip() and not _caption_ocr_punct_char(ch))
    if len(body) < 5:
        return False
    if _caption_rejected_ocr_text_looks_sfx_or_punctuation(text):
        return False
    has_kanji = any("\u4e00" <= ch <= "\u9fff" for ch in body)
    has_kana = any("\u3040" <= ch <= "\u30ff" for ch in body)
    has_digit = any(ch.isdigit() for ch in body)
    return bool(has_digit or has_kanji or (has_kana and len(body) >= 6))


def _caption_rejected_ocr_text_looks_sfx_or_punctuation(text: str) -> bool:
    body = "".join(ch for ch in str(text or "") if ch.strip() and not _caption_ocr_punct_char(ch))
    if not body:
        return True
    kana_count = sum(1 for ch in body if "\u3040" <= ch <= "\u30ff")
    katakana_count = sum(1 for ch in body if "\u30a0" <= ch <= "\u30ff")
    if len(body) <= 4 and kana_count == len(body):
        return True
    if len(body) <= 5 and katakana_count == len(body):
        return True
    if len(body) >= 3 and len(set(body)) <= 2 and kana_count >= max(2, len(body) - 1):
        return True
    return False


def _source_erasure_visual_validation_repairs_pre_render_failure(
    region: dict[str, Any],
    warning_class: str,
) -> bool:
    """Promote exact repaired residual proof/status contradictions to pass status."""

    return (
        warning_class == "repaired_source_residual"
        and str(region.get("source_erasure_visual_validation_stage") or "") == "pre_render_cleaned_image"
        and str(region.get("source_erasure_warning_reason") or "") == "pre_render_source_erasure_residual_below_threshold"
        and str(region.get("translated_text") or "").strip()
        and region.get("translated_independently") is True
        and region.get("source_child_cleanup_required") is not True
        and str(region.get("pre_render_source_erasure_status") or "") == "failed_audit_only"
        and str(region.get("pre_render_source_erasure_failure_reason") or "")
        == "pre_render_source_residual_pixels_remaining"
        and str(region.get("cleanup_visual_validation_failed") or "").strip().lower()
        not in {"1", "true", "yes", "on"}
    )


def _caption_ocr_punct_char(ch: str) -> bool:
    return ch in "。．，、！？：；…・･·.!,?;:「」『』“”\"'‘’（）()[]【】ー～~"


def _region_needs_source_erasure_visual_check(region: dict[str, Any]) -> bool:
    if str(region.get("translated_text") or "").strip():
        return not bool(region.get("is_sfx") or region.get("is_decorative"))
    if region.get("source_child_cleanup_required") is True:
        return True
    if region.get("source_glyph_mask_consumed_by_renderer") is True:
        return True
    return False


def _render_overlap_mask_for_region(
    region: dict[str, Any],
    crop_box: tuple[int, int, int, int],
    render_boxes_by_parent: dict[str, list[tuple[int, int, int, int]]],
    render_boxes_by_root: dict[str, list[tuple[int, int, int, int]]],
    np_module,
):
    x0, y0, x1, y1 = crop_box
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if width <= 0 or height <= 0:
        return None
    parent_id = str(region.get("parent_logical_text_unit_id") or region.get("represented_by_parent_id") or "")
    root_id = str(region.get("text_block_root_id") or "")
    boxes = list(render_boxes_by_parent.get(parent_id) or [])
    if not boxes and root_id:
        boxes = list(render_boxes_by_root.get(root_id) or [])
    if not boxes:
        box = _xyxy(region.get("final_render_bbox"))
        boxes = [box] if box else []
    if not boxes:
        return None
    mask = np_module.zeros((height, width), dtype=bool)
    for box in boxes:
        if not box:
            continue
        bx0, by0, bx1, by1 = _expand_box(box, 6)
        ix0 = max(x0, bx0)
        iy0 = max(y0, by0)
        ix1 = min(x1, bx1)
        iy1 = min(y1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = True
    return mask


def _source_residual_shape_summary(residual_mask, np_module) -> dict[str, Any]:
    summary = {
        "source_erasure_residual_component_count": 0,
        "source_erasure_residual_line_like_ratio": 0.0,
        "source_erasure_residual_largest_component_area": 0,
    }
    try:
        total = int((residual_mask > 0).sum())
    except Exception:
        return summary
    if total <= 0:
        return summary
    try:
        import cv2

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (residual_mask > 0).astype("uint8"),
            8,
        )
        components = 0
        line_like_area = 0
        largest = 0
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < 3:
                continue
            components += 1
            largest = max(largest, area)
            aspect = max(w / max(1, h), h / max(1, w))
            thin = min(w, h) <= 3 and max(w, h) >= 12
            if aspect >= 7.0 or thin:
                line_like_area += area
        summary["source_erasure_residual_component_count"] = int(components)
        summary["source_erasure_residual_line_like_ratio"] = round(line_like_area / max(1, total), 3)
        summary["source_erasure_residual_largest_component_area"] = int(largest)
    except Exception:
        ys, xs = np_module.where(residual_mask > 0)
        if ys.size and xs.size:
            w = int(xs.max() - xs.min() + 1)
            h = int(ys.max() - ys.min() + 1)
            aspect = max(w / max(1, h), h / max(1, w))
            summary["source_erasure_residual_component_count"] = 1
            summary["source_erasure_residual_line_like_ratio"] = 1.0 if aspect >= 7.0 else 0.0
            summary["source_erasure_residual_largest_component_area"] = int(total)
    return summary


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (int(x0) - pad, int(y0) - pad, int(x1) + pad, int(y1) + pad)


def _source_erasure_warning_class(
    region: dict[str, Any],
    residual: float,
    raw_residual: float,
    render_overlap_ratio: float,
    shape_summary: dict[str, Any] | None = None,
) -> str:
    if region.get("source_child_cleanup_required") is True and region.get("source_child_cleanup_covered") is not True:
        return "represented_child_cleanup_missing"
    coverage = _float(region.get("source_glyph_erasure_coverage_ratio"))
    expected = bool(region.get("translated_text") or region.get("source_child_cleanup_required") or region.get("source_glyph_mask_consumed_by_renderer"))
    pre_render_validation = str(region.get("source_erasure_visual_validation_stage") or "") == "pre_render_cleaned_image"
    represented_child_cleanup_passed = (
        region.get("source_child_cleanup_required") is True
        and region.get("source_child_cleanup_covered") is True
    )
    if expected and coverage is not None and coverage < 0.90 and not represented_child_cleanup_passed:
        return "mask_generation_failure"
    shape_summary = shape_summary or {}
    line_like_ratio = _float(shape_summary.get("source_erasure_residual_line_like_ratio")) or 0.0
    largest_component = int(_float(shape_summary.get("source_erasure_residual_largest_component_area")) or 0)
    outside_pixels = int(_float(region.get("source_erasure_residual_outside_render_dark_pixels")) or 0)
    total_dark = int(_float(region.get("source_erasure_visual_residual_dark_pixels")) or 0)
    outside_ratio = float(outside_pixels) / float(max(1, total_dark))
    glyph_coverage_high = (
        coverage is not None
        and coverage >= 0.93
        and region.get("cleanup_covers_source_glyphs") is True
    )
    if expected and residual >= 0.22:
        if not pre_render_validation and _source_residual_likely_translated_text_overlap(
            region,
            residual,
            raw_residual,
            render_overlap_ratio,
            shape_summary,
        ):
            return "translated_text_overlap"
        if not pre_render_validation and raw_residual >= 0.22 and render_overlap_ratio >= 0.88 and outside_ratio <= 0.10 and residual < 0.10:
            return "translated_text_overlap"
        return "true_source_residual_blocker"
    if not pre_render_validation and expected and raw_residual >= 0.22 and render_overlap_ratio >= 0.88 and outside_ratio <= 0.10:
        return "translated_text_overlap"
    if not expected and glyph_coverage_high and residual >= 0.22 and outside_pixels <= 7000:
        if line_like_ratio >= 0.25 or largest_component >= 1800:
            return "bubble_border_or_art_line"
        return "benign_texture_nonblocking"
    if not expected and residual >= 0.22 and line_like_ratio >= 0.45 and (largest_component >= 18 or outside_pixels < 1800):
        return "bubble_border_or_art_line"
    if residual >= 0.22 and region.get("root_cleanup_repair_v2_attempted") is True:
        return "cleanup_backend_failure"
    if residual >= 0.22:
        return "true_source_residual_blocker"
    if not pre_render_validation and raw_residual >= 0.22 and render_overlap_ratio >= 0.88 and outside_ratio <= 0.10:
        return "translated_text_overlap"
    if residual < 0.08 and coverage is not None and coverage >= 0.90:
        return "repaired_source_residual"
    return "benign_texture_nonblocking"


def _source_residual_likely_translated_text_overlap(
    region: dict[str, Any],
    residual: float,
    raw_residual: float,
    render_overlap_ratio: float,
    shape_summary: dict[str, Any],
) -> bool:
    """Separate rendered target glyphs from source residuals after local proof passes."""
    if residual < 0.22 or raw_residual < 0.22:
        return False
    coverage = _float(region.get("source_glyph_erasure_coverage_ratio"))
    if coverage is None or coverage < 0.93:
        return False
    if region.get("cleanup_covers_source_glyphs") is not True:
        return False
    if region.get("cleanup_visual_artifact_risk") is True:
        return False
    if render_overlap_ratio < 0.78:
        return False
    outside_pixels = int(_float(region.get("source_erasure_residual_outside_render_dark_pixels")) or 0)
    largest_component = int(_float(shape_summary.get("source_erasure_residual_largest_component_area")) or 0)
    line_like_ratio = _float(shape_summary.get("source_erasure_residual_line_like_ratio")) or 0.0
    if outside_pixels <= 1000 and largest_component <= 900:
        return True
    if outside_pixels <= 1300 and largest_component <= 650 and line_like_ratio >= 0.45:
        return True
    return False


def _source_erasure_warning_reason(
    region: dict[str, Any],
    warning_class: str,
    residual: float,
    raw_residual: float,
    render_overlap_ratio: float,
) -> str:
    if warning_class == "represented_child_cleanup_missing":
        return "represented_child_source_not_cleaned"
    if warning_class == "mask_generation_failure":
        return "source_glyph_mask_missing_or_low_coverage"
    if warning_class == "cleanup_backend_failure":
        return "source_dark_pixels_remain_after_root_local_repair"
    if warning_class == "translated_text_overlap":
        return "residual_dark_pixels_overlap_rendered_translation"
    if warning_class == "bubble_border_or_art_line":
        return "residual_matches_bubble_border_or_art_line"
    if warning_class == "true_source_residual_blocker":
        if str(region.get("source_erasure_visual_validation_stage") or "") == "pre_render_cleaned_image":
            return "source_pixels_remain_in_pre_render_cleaned_image"
        return "source_dark_pixels_remain_outside_rendered_text"
    if warning_class == "repaired_source_residual":
        if str(region.get("source_erasure_visual_validation_stage") or "") == "pre_render_cleaned_image":
            return "pre_render_source_erasure_residual_below_threshold"
        return "source_erasure_residual_below_threshold_after_render_exclusion"
    if raw_residual >= 0.22 and render_overlap_ratio >= 0.35:
        return "residual_overlaps_rendered_translation_or_texture"
    return "probable_benign_texture_residual"


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(x0))),
        max(0, min(height, int(y0))),
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
    )


def _maybe_add_root_final_state_closeout(
    context: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    hierarchy = context.get("text_block_hierarchy") or audit.get("text_block_hierarchy")
    if not isinstance(hierarchy, dict) or not hierarchy.get("text_block_hierarchy_generated"):
        return audit
    try:
        from app.pipeline.text_block_hierarchy import enrich_hierarchy_with_root_closeout

        regions = [dict(region) for region in audit.get("regions", []) or []]
        enriched_hierarchy = enrich_hierarchy_with_root_closeout(
            hierarchy,
            regions,
            source_image_path=str(context.get("source_path") or ""),
        )
        root_by_id = {
            str(root.get("root_id") or ""): root
            for root in enriched_hierarchy.get("text_area_root_blocks", []) or []
            if isinstance(root, dict)
        }
        for region in regions:
            root = root_by_id.get(str(region.get("text_block_root_id") or ""))
            if not root:
                continue
            for key in (
                "root_final_state",
                "root_final_state_reason",
                "root_has_meaningful_visible_source",
                "root_is_punctuation_only",
                "root_is_sfx_decorative",
                "root_is_caption_background",
                "root_closeout_blocker",
                "root_closeout_warning_reasons",
                "source_erasure_expected",
                "source_erasure_mask_coverage",
                "source_erasure_visual_residual_score",
                "source_erasure_warning_class_counts",
                "source_erasure_failure_reason",
                "source_glyph_mask_id",
                "cleanup_partition_id",
                "render_text_completeness_pass",
                "render_missing_characters",
                "render_outside_root_ratio",
                "render_density_score",
                "render_readability_warning_reason",
                "render_warning_action_status",
                "render_warning_action_reason",
                "render_warning_action_reasons",
                "render_fit_action_attempted",
                "render_fit_action_status",
                "render_fit_before_density",
                "render_fit_after_density",
                "render_fit_before_bbox",
                "render_fit_after_bbox",
                "render_fit_rejection_reason",
                "render_layout_v2_attempted",
                "render_layout_v2_status",
                "render_layout_v2_before_score",
                "render_layout_v2_after_score",
                "render_layout_v2_before_fit_ratio",
                "render_layout_v2_after_fit_ratio",
                "render_layout_v2_line_height_scale",
                "render_layout_v2_compact_layout",
                "render_layout_v2_selected_font_size",
                "render_layout_v2_rejection_reason",
                "render_layout_v3_attempted",
                "render_layout_v3_status",
                "render_layout_v3_before_score",
                "render_layout_v3_after_score",
                "render_layout_v3_candidate_count",
                "render_layout_v3_selected_candidate",
                "render_layout_v3_rejection_reason",
                "render_layout_v3_shape_source",
                "render_layout_v3_edge_contact_before",
                "render_layout_v3_edge_contact_after",
                "render_layout_v3_density_before",
                "render_layout_v3_density_after",
                "render_readability_v4_attempted",
                "render_readability_v4_status",
                "render_readability_v4_before_score",
                "render_readability_v4_after_score",
                "render_readability_v4_candidate_count",
                "render_readability_v4_selected_candidate",
                "render_readability_v4_unresolved_reason",
                "render_readability_v4_edge_contact_before",
                "render_readability_v4_edge_contact_after",
                "render_readability_v4_density_before",
                "render_readability_v4_density_after",
                "render_readability_v4_shape_source",
                "render_readability_v4_final_class",
                "render_readability_v5_attempted",
                "render_readability_v5_status",
                "render_readability_v5_before_score",
                "render_readability_v5_after_score",
                "render_readability_v5_candidate_count",
                "render_readability_v5_selected_candidate",
                "render_readability_v5_source_column_count",
                "render_readability_v5_shape_source",
                "render_readability_v5_density_before",
                "render_readability_v5_density_after",
                "render_readability_v5_edge_contact_before",
                "render_readability_v5_edge_contact_after",
                "render_readability_v5_final_class",
                "render_readability_v5_unresolved_reason",
                "caption_background_visual_textlike_score",
                "caption_background_visual_textlike_component_count",
                "caption_background_dark_pixel_ratio",
                "caption_background_visibility_status",
                "caption_background_recovery_v2_status",
                "caption_background_recovery_v2_reason",
                "caption_background_recovery_v3_status",
                "caption_background_recovery_v3_reason",
                "caption_background_recovery_v4_status",
                "caption_background_recovery_v4_reason",
                "caption_background_sfx_artlike_score",
                "caption_background_sfx_rejection_status",
                "caption_background_sfx_rejection_reason",
                "caption_background_component_count",
                "caption_background_large_component_count",
                "caption_background_line_like_component_count",
                "caption_background_large_component_area_ratio",
                "caption_background_line_like_area_ratio",
                "caption_background_rejected_ocr_texts",
                "caption_background_meaningful_rejected_ocr_texts",
                "caption_component_split_attempted",
                "caption_component_count",
                "caption_component_caption_like_count",
                "caption_component_sfx_artlike_count",
                "caption_component_accepted_count",
                "caption_component_rejected_count",
                "caption_component_nonrecoverable_reason",
                "caption_component_v4_attempted",
                "caption_component_v4_status",
                "caption_component_v4_candidate_count",
                "caption_component_v4_accepted_count",
                "caption_component_v4_rejected_count",
                "caption_component_v4_nonblocking_mixed_art_review",
                "caption_component_v4_rejection_reasons",
                "caption_component_v4_accepted_text",
                "caption_component_v4_candidate_bboxes",
                "caption_component_v4_reading_order",
            ):
                region[key] = _json_safe(root.get(key))
            region["root_render_wrapped_lines"] = _json_safe(root.get("render_wrapped_lines"))
        enriched = dict(audit)
        enriched["regions"] = regions
        enriched["text_block_hierarchy"] = enriched_hierarchy
        enriched["text_area_root_blocks"] = enriched_hierarchy.get("text_area_root_blocks") or []
        enriched["text_block_hierarchy_summary"] = enriched_hierarchy.get("text_block_hierarchy_summary") or {}
        enriched["root_final_state_counts"] = enriched_hierarchy.get("root_final_state_counts") or {}
        enriched["unresolved_meaningful_blocker_count"] = enriched_hierarchy.get("unresolved_meaningful_blocker_count")
        enriched["cleanup_warning_root_count"] = enriched_hierarchy.get("cleanup_warning_root_count")
        enriched["render_warning_root_count"] = enriched_hierarchy.get("render_warning_root_count")
        enriched["punctuation_only_nonblocking_count"] = enriched_hierarchy.get("punctuation_only_nonblocking_count")
        enriched["page_metadata_closeout_candidate"] = enriched_hierarchy.get("page_metadata_closeout_candidate")
        enriched["page_visual_evaluation_required"] = enriched_hierarchy.get("page_visual_evaluation_required")
        enriched["page_visual_evaluation_status"] = enriched_hierarchy.get("page_visual_evaluation_status")
        enriched["page_visual_evaluation_pass_source"] = enriched_hierarchy.get("page_visual_evaluation_pass_source")
        enriched["page_visual_closeout_blocked_reason"] = enriched_hierarchy.get("page_visual_closeout_blocked_reason")
        enriched["page_visual_closeout_pass"] = enriched_hierarchy.get("page_visual_closeout_pass")
        context["text_block_hierarchy"] = enriched_hierarchy
        return enriched
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched = dict(audit)
        enriched["root_final_state_closeout_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def write_root_parent_child_debug_artifacts(
    *,
    page_dir: str,
    hierarchy: dict[str, Any],
    source_glyph_masks: dict[str, Any] | None = None,
    root_reconstruction_executor: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write root/parent/child debug overlays and tables from the debug module."""

    from app.pipeline.text_block_hierarchy import write_text_block_hierarchy_artifacts
    from app.pipeline.text_block_root_graph import write_root_graph_debug_artifacts

    paths = write_text_block_hierarchy_artifacts(
        page_dir=page_dir,
        hierarchy=hierarchy,
        source_glyph_masks=source_glyph_masks or {},
    )
    root_graph_paths = write_root_graph_debug_artifacts(
        page_dir=page_dir,
        hierarchy=hierarchy,
        root_reconstruction_executor=root_reconstruction_executor or {},
    )
    paths.update(root_graph_paths)
    return paths


def _maybe_write_text_block_hierarchy_artifacts(
    context: dict[str, Any],
    audit: dict[str, Any],
    page_dir: str,
) -> dict[str, Any]:
    if not debug_stage_enabled(context, "root_parent_child"):
        return audit
    hierarchy = context.get("text_block_hierarchy") or audit.get("text_block_hierarchy")
    if not isinstance(hierarchy, dict) or not hierarchy.get("text_block_hierarchy_generated"):
        return audit
    try:
        paths = write_root_parent_child_debug_artifacts(
            page_dir=page_dir,
            hierarchy=hierarchy,
            source_glyph_masks=context.get("source_glyph_masks") or audit.get("source_glyph_masks") or {},
            root_reconstruction_executor=(
                context.get("root_reconstruction_executor")
                or audit.get("root_reconstruction_executor")
                or {}
            ),
        )
        enriched = dict(audit)
        enriched["text_block_hierarchy_artifact_paths"] = paths
        return enriched
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched = dict(audit)
        enriched["text_block_hierarchy_artifact_error"] = f"{type(exc).__name__}: {exc}"
        return enriched


def _maybe_add_text_area_diagnostics(
    audit: dict[str, Any],
    regions: list[dict],
    *,
    audit_path: str,
) -> dict[str, Any]:
    try:
        from app.pipeline.text_area_diagnostics import enrich_audit_with_text_area_diagnostics

        project_regions = {
            str(region.get("region_id") or ""): region
            for region in regions
            if str(region.get("region_id") or "")
        }
        return enrich_audit_with_text_area_diagnostics(
            audit,
            audit_path=audit_path,
            project_regions=project_regions,
        )
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
        enriched["diagnostic_version"] = "text_area_diagnostics_phase1_v1"
        enriched["diagnostic_generated"] = False
        enriched["diagnostic_error"] = str(exc)
        enriched["diagnostic_runtime_sec"] = 0.0
        return enriched


def _maybe_add_route_advisor(audit: dict[str, Any]) -> dict[str, Any]:
    try:
        from app.pipeline.text_area_route_advisor import enrich_audit_with_route_advisor

        return enrich_audit_with_route_advisor(audit)
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
        enriched["route_advisor_version"] = "text_area_route_advisor_phase2a_v1"
        enriched["route_advisor_generated"] = False
        enriched["route_advisor_error"] = str(exc)
        enriched["route_advisor_runtime_sec"] = 0.0
        enriched["route_suggestions"] = []
        for region in enriched.get("regions", []) or []:
            region["diagnostic_route_suggestions"] = []
        return enriched


def _maybe_add_render_planner(audit: dict[str, Any]) -> dict[str, Any]:
    try:
        from app.pipeline.text_area_render_planner import enrich_audit_with_render_planner

        return enrich_audit_with_render_planner(audit)
    except Exception as exc:  # pragma: no cover - diagnostic isolation
        enriched = dict(audit)
        enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
        enriched["render_planner_version"] = "text_area_render_planner_phase3a_v1"
        enriched["render_planner_generated"] = False
        enriched["render_planner_error"] = str(exc)
        enriched["render_planner_runtime_sec"] = 0.0
        enriched["render_plan_suggestions"] = []
        for region in enriched.get("regions", []) or []:
            region["diagnostic_render_plan_suggestions"] = []
            region["diagnostic_render_plan_severity"] = None
            region["diagnostic_render_plan_reason_codes"] = []
        return enriched


def _audit_value(meta: dict[str, Any], region: dict[str, Any], render: dict[str, Any], key: str, default: Any = None) -> Any:
    for source in (meta, region, render):
        if isinstance(source, dict) and key in source and source.get(key) is not None:
            return source.get(key)
    return default


def _build_audit(context: dict[str, Any], regions: list[dict]) -> dict[str, Any]:
    region_meta = context.get("regions", {}) or {}
    audit_regions = []
    translated = 0
    skipped = 0
    for order_idx, region in enumerate(regions):
        rid = str(region.get("region_id", "") or "")
        meta = region_meta.get(rid, {}) if isinstance(region_meta, dict) else {}
        flags = region.get("flags", {}) or {}
        confidence = region.get("confidence", {}) or {}
        render = region.get("render", {}) or {}
        translation = str(region.get("translation", "") or "")
        if translation.strip():
            translated += 1
        if flags.get("ignore") or not translation.strip():
            skipped += 1
        semantic_class = str(region.get("type", "") or "unknown")
        audit_regions.append(
            {
                "page_id": context.get("page_id"),
                "region_id": rid,
                "parent_execution_bundle_id": _audit_value(meta, region, render, "parent_execution_bundle_id"),
                "parent_execution_bundle_version": _audit_value(meta, region, render, "parent_execution_bundle_version"),
                "parent_execution_state": _audit_value(meta, region, render, "parent_execution_state"),
                "parent_execution_authoritative": _audit_value(meta, region, render, "parent_execution_authoritative"),
                "execution_region_authority": _audit_value(meta, region, render, "execution_region_authority"),
                "execution_region_role": _audit_value(meta, region, render, "execution_region_role"),
                "legacy_region_execution_authority": _audit_value(meta, region, render, "legacy_region_execution_authority"),
                "source_region_evidence_only": _audit_value(meta, region, render, "source_region_evidence_only"),
                "renderer_audit_id": _audit_value(meta, region, render, "renderer_audit_id"),
                "renderer_input_authority": _audit_value(meta, region, render, "renderer_input_authority"),
                "render_style": _json_safe(_audit_value(meta, region, render, "render_style", {})),
                "render_style_owner": _audit_value(meta, region, render, "render_style_owner"),
                "render_style_version": _audit_value(meta, region, render, "render_style_version"),
                "render_style_source": _audit_value(meta, region, render, "render_style_source"),
                "render_style_provider": _audit_value(meta, region, render, "render_style_provider"),
                "render_style_provider_model": _audit_value(meta, region, render, "render_style_provider_model"),
                "render_style_confidence": _audit_value(meta, region, render, "render_style_confidence"),
                "render_style_style_class": _audit_value(meta, region, render, "render_style_style_class"),
                "render_style_font_family": _audit_value(meta, region, render, "render_style_font_family"),
                "render_style_font_weight": _audit_value(meta, region, render, "render_style_font_weight"),
                "render_style_fill_color": _audit_value(meta, region, render, "render_style_fill_color"),
                "render_style_stroke_color": _audit_value(meta, region, render, "render_style_stroke_color"),
                "render_style_stroke_width": _audit_value(meta, region, render, "render_style_stroke_width"),
                "render_style_source_orientation": _audit_value(meta, region, render, "render_style_source_orientation"),
                "render_style_wrap_mode": _audit_value(meta, region, render, "render_style_wrap_mode"),
                "render_style_font_size": _audit_value(meta, region, render, "render_style_font_size"),
                "render_style_font_size_hint": _audit_value(meta, region, render, "render_style_font_size_hint"),
                "render_style_font_size_locked": _audit_value(meta, region, render, "render_style_font_size_locked"),
                "render_style_font_size_policy": _audit_value(meta, region, render, "render_style_font_size_policy"),
                "cleanup_job_ids": _json_safe(_audit_value(meta, region, render, "cleanup_job_ids", [])),
                "cleanup_mask_ids": _json_safe(_audit_value(meta, region, render, "cleanup_mask_ids", [])),
                "render_decision_id": _audit_value(meta, region, render, "render_decision_id"),
                "source_glyph_mask_ids": _json_safe(_audit_value(meta, region, render, "source_glyph_mask_ids", [])),
                "source_region_ids": _json_safe(_audit_value(meta, region, render, "source_region_ids", [])),
                "represented_child_ids": _json_safe(_audit_value(meta, region, render, "represented_child_ids", [])),
                "text_block_root_id": _audit_value(meta, region, render, "text_block_root_id"),
                "parent_logical_text_unit_id": _audit_value(meta, region, render, "parent_logical_text_unit_id"),
                "bbox": _json_safe(region.get("bbox")),
                "polygon": _json_safe(region.get("polygon")),
                "detection_confidence": _json_safe(confidence.get("det")),
                "ocr_confidence": _json_safe(confidence.get("ocr")),
                "ocr_text": region.get("ocr_text", ""),
                "normalized_ocr_text": _normalize_text_for_audit(region.get("ocr_text", "")),
                "reading_order_index": meta.get("reading_order_index", order_idx),
                "group_id": meta.get("group_id") or region.get("group_id"),
                "bubble_id": meta.get("bubble_id") or region.get("bubble_id"),
                "semantic_class": semantic_class,
                "is_speech_bubble": semantic_class == "speech_bubble",
                "is_background": semantic_class == "background_text" or bool(flags.get("bg_text")),
                "is_decorative": semantic_class == "decorative_text",
                "is_sfx": semantic_class == "decorative_text" or bool(flags.get("sfx")),
                "is_sign": semantic_class == "sign" or bool(flags.get("sign")),
                "is_unknown": semantic_class not in {"speech_bubble", "background_text", "decorative_text", "narration_box"},
                "sent_to_translation": meta.get("sent_to_translation"),
                "translated_text": translation,
                "skip_reason": meta.get("skip_reason") or _default_skip_reason(region),
                "classification_reason": meta.get("classification_reason") or render.get("classification_reason"),
                "route_owned_ocr_retry_attempted": meta.get("route_owned_ocr_retry_attempted") or render.get("route_owned_ocr_retry_attempted") or region.get("route_owned_ocr_retry_attempted"),
                "route_owned_ocr_retry_status": meta.get("route_owned_ocr_retry_status") or render.get("route_owned_ocr_retry_status") or region.get("route_owned_ocr_retry_status"),
                "route_owned_ocr_retry_original_bbox": meta.get("route_owned_ocr_retry_original_bbox") or render.get("route_owned_ocr_retry_original_bbox") or region.get("route_owned_ocr_retry_original_bbox") or [],
                "route_owned_ocr_retry_bbox": meta.get("route_owned_ocr_retry_bbox") or render.get("route_owned_ocr_retry_bbox") or region.get("route_owned_ocr_retry_bbox") or [],
                "route_owned_ocr_retry_original_text": meta.get("route_owned_ocr_retry_original_text") or render.get("route_owned_ocr_retry_original_text") or region.get("route_owned_ocr_retry_original_text"),
                "route_owned_ocr_retry_original_confidence": meta.get("route_owned_ocr_retry_original_confidence") or render.get("route_owned_ocr_retry_original_confidence") or region.get("route_owned_ocr_retry_original_confidence"),
                "route_owned_ocr_retry_original_state": meta.get("route_owned_ocr_retry_original_state") or render.get("route_owned_ocr_retry_original_state") or region.get("route_owned_ocr_retry_original_state"),
                "route_owned_ocr_retry_text": meta.get("route_owned_ocr_retry_text") or render.get("route_owned_ocr_retry_text") or region.get("route_owned_ocr_retry_text"),
                "route_owned_ocr_retry_confidence": meta.get("route_owned_ocr_retry_confidence") or render.get("route_owned_ocr_retry_confidence") or region.get("route_owned_ocr_retry_confidence"),
                "route_owned_ocr_retry_state": meta.get("route_owned_ocr_retry_state") or render.get("route_owned_ocr_retry_state") or region.get("route_owned_ocr_retry_state"),
                "route_owned_ocr_retry_failure_reason": meta.get("route_owned_ocr_retry_failure_reason") or render.get("route_owned_ocr_retry_failure_reason") or region.get("route_owned_ocr_retry_failure_reason"),
                "glossary_terms_available": meta.get("glossary_terms_available"),
                "glossary_terms_applied": meta.get("glossary_terms_applied"),
                "glossary_terms_ignored": meta.get("glossary_terms_ignored"),
                "prompt_glossary_section_included": meta.get("prompt_glossary_section_included"),
                "terminology_consistency_warnings": meta.get("terminology_consistency_warnings"),
                "cleanup_applied": meta.get("cleanup_applied"),
                "cleanup_mode": meta.get("cleanup_mode") or render.get("cleanup_mode"),
                "cleanup_runtime_class": meta.get("cleanup_runtime_class"),
                "cleanup_runtime_status": meta.get("cleanup_runtime_status"),
                "cleanup_runtime_failure_reason": meta.get("cleanup_runtime_failure_reason"),
                "cleanup_runtime_plan_id": meta.get("cleanup_runtime_plan_id"),
                "cleanup_runtime_result_id": meta.get("cleanup_runtime_result_id"),
                "cleanup_runtime_proof_id": meta.get("cleanup_runtime_proof_id"),
                "cleanup_runtime_renderer_consumed": meta.get("cleanup_runtime_renderer_consumed"),
                "cleanup_runtime_render_consumption_decision_if_consumed": meta.get(
                    "cleanup_runtime_render_consumption_decision_if_consumed"
                ),
                "cleanup_runtime_render_blocked": meta.get("cleanup_runtime_render_blocked"),
                "cleanup_runtime_render_block_reason": meta.get("cleanup_runtime_render_block_reason"),
                "cleanup_applied_upstream": meta.get("cleanup_applied_upstream"),
                "cleanup_committed_to_working_image": meta.get("cleanup_committed_to_working_image"),
                "cleanup_upstream_commit_status": meta.get("cleanup_upstream_commit_status"),
                "cleanup_upstream_commit_failure_reason": meta.get("cleanup_upstream_commit_failure_reason"),
                "cleanup_upstream_committed_pixel_count": meta.get("cleanup_upstream_committed_pixel_count"),
                "cleanup_upstream_cleaned_image_ref": meta.get("cleanup_upstream_cleaned_image_ref"),
                "cleanup_upstream_diff_ref": meta.get("cleanup_upstream_diff_ref"),
                "cleanup_upstream_mask_ref": meta.get("cleanup_upstream_mask_ref"),
                "render_suppressed_by_cleanup_proof": meta.get("render_suppressed_by_cleanup_proof"),
                "render_source_grounded": meta.get("render_source_grounded"),
                "source_grounding_status": meta.get("source_grounding_status"),
                "source_grounding_failure_reason": meta.get("source_grounding_failure_reason"),
                "render_suppressed_by_source_grounding": meta.get("render_suppressed_by_source_grounding"),
                "translated_text_suppressed": meta.get("translated_text_suppressed"),
                "translated_text_suppressed_reason": meta.get("translated_text_suppressed_reason"),
                "cleanup_mask": meta.get("cleanup_mask"),
                "cleanup_mask_bbox": _mask_bbox(meta.get("cleanup_mask")),
                "cleanup_inpaint_time_sec": meta.get("cleanup_inpaint_time_sec"),
                "cleanup_backend": meta.get("cleanup_backend"),
                "cleanup_backend_detail": meta.get("cleanup_backend_detail"),
                "cleanup_operation_kind": meta.get("cleanup_operation_kind"),
                "cleanup_requested_inpaint_mode": meta.get("cleanup_requested_inpaint_mode"),
                "cleanup_effective_inpaint_mode": meta.get("cleanup_effective_inpaint_mode"),
                "cleanup_mask_pixels": meta.get("cleanup_mask_pixels"),
                "cleanup_mask_bbox_area": meta.get("cleanup_mask_bbox_area"),
                "cleanup_crop_bbox": meta.get("cleanup_crop_bbox"),
                "cleanup_crop_area": meta.get("cleanup_crop_area"),
                "cleanup_mask_ratio": meta.get("cleanup_mask_ratio"),
                "root_cleanup_repair_v2_attempted": meta.get("root_cleanup_repair_v2_attempted"),
                "root_cleanup_repair_v2_status": meta.get("root_cleanup_repair_v2_status"),
                "root_cleanup_repair_v2_backend": meta.get("root_cleanup_repair_v2_backend"),
                "root_cleanup_repair_v2_attempt_count": meta.get("root_cleanup_repair_v2_attempt_count"),
                "root_cleanup_repair_v2_reason": meta.get("root_cleanup_repair_v2_reason"),
                "cleanup_partition_id": meta.get("cleanup_partition_id"),
                "cleanup_partition_scope": meta.get("cleanup_partition_scope"),
                "cleanup_partition_fallback_reason": meta.get("cleanup_partition_fallback_reason"),
                "source_child_cleanup_required": meta.get("source_child_cleanup_required"),
                "source_child_cleanup_covered": meta.get("source_child_cleanup_covered"),
                "source_child_cleanup_partition_id": meta.get("source_child_cleanup_partition_id"),
                "source_child_cleanup_partition_scope": meta.get("source_child_cleanup_partition_scope"),
                "source_child_cleanup_missing_reason": meta.get("source_child_cleanup_missing_reason"),
                "source_child_cleanup_coverage_ratio": meta.get("source_child_cleanup_coverage_ratio"),
                "source_child_cleanup_generation_method": meta.get("source_child_cleanup_generation_method"),
                "source_child_cleanup_reason": meta.get("source_child_cleanup_reason"),
                "parent_cleanup_child_region_ids": meta.get("parent_cleanup_child_region_ids") or [],
                "final_render_bbox": meta.get("final_render_bbox"),
                "selected_font_family": meta.get("selected_font_family") or render.get("font"),
                "selected_font_size": meta.get("selected_font_size"),
                "selected_font_size_kind": meta.get("selected_font_size_kind"),
                "parent_style_font_size_locked": meta.get("parent_style_font_size_locked") or render.get("font_size_locked"),
                "parent_style_resolved_font_size": meta.get("parent_style_resolved_font_size") or render.get("font_size"),
                "parent_style_font_size_preserved": meta.get("parent_style_font_size_preserved"),
                "parent_style_font_size_fallback_applied": meta.get("parent_style_font_size_fallback_applied"),
                "parent_style_font_size_fallback_reason": meta.get("parent_style_font_size_fallback_reason"),
                "parent_style_font_size_fallback_selected_size": meta.get("parent_style_font_size_fallback_selected_size"),
                "parent_style_font_size_audit_note": meta.get("parent_style_font_size_audit_note"),
                "wrapped_lines": meta.get("wrapped_lines"),
                "measured_rendered_width": meta.get("measured_rendered_width"),
                "measured_rendered_height": meta.get("measured_rendered_height"),
                "fit_ratio": meta.get("fit_ratio"),
                "render_fit_action_attempted": meta.get("render_fit_action_attempted"),
                "render_fit_action_status": meta.get("render_fit_action_status"),
                "render_fit_before_density": meta.get("render_fit_before_density"),
                "render_fit_after_density": meta.get("render_fit_after_density"),
                "render_fit_before_bbox": meta.get("render_fit_before_bbox"),
                "render_fit_after_bbox": meta.get("render_fit_after_bbox"),
                "render_fit_rejection_reason": meta.get("render_fit_rejection_reason"),
                "render_fit_compact_layout": meta.get("render_fit_compact_layout"),
                "render_fit_density_improvement": meta.get("render_fit_density_improvement"),
                "render_fit_area_gain": meta.get("render_fit_area_gain"),
                "render_layout_v2_attempted": meta.get("render_layout_v2_attempted"),
                "render_layout_v2_status": meta.get("render_layout_v2_status"),
                "render_layout_v2_before_score": meta.get("render_layout_v2_before_score"),
                "render_layout_v2_after_score": meta.get("render_layout_v2_after_score"),
                "render_layout_v2_before_fit_ratio": meta.get("render_layout_v2_before_fit_ratio"),
                "render_layout_v2_after_fit_ratio": meta.get("render_layout_v2_after_fit_ratio"),
                "render_layout_v2_line_height_scale": meta.get("render_layout_v2_line_height_scale"),
                "render_layout_v2_compact_layout": meta.get("render_layout_v2_compact_layout"),
                "render_layout_v2_selected_font_size": meta.get("render_layout_v2_selected_font_size"),
                "render_layout_v2_rejection_reason": meta.get("render_layout_v2_rejection_reason"),
                "render_layout_v3_attempted": meta.get("render_layout_v3_attempted"),
                "render_layout_v3_status": meta.get("render_layout_v3_status"),
                "render_layout_v3_before_score": meta.get("render_layout_v3_before_score"),
                "render_layout_v3_after_score": meta.get("render_layout_v3_after_score"),
                "render_layout_v3_candidate_count": meta.get("render_layout_v3_candidate_count"),
                "render_layout_v3_selected_candidate": meta.get("render_layout_v3_selected_candidate"),
                "render_layout_v3_rejection_reason": meta.get("render_layout_v3_rejection_reason"),
                "render_layout_v3_shape_source": meta.get("render_layout_v3_shape_source"),
                "render_layout_v3_edge_contact_before": meta.get("render_layout_v3_edge_contact_before"),
                "render_layout_v3_edge_contact_after": meta.get("render_layout_v3_edge_contact_after"),
                "render_layout_v3_density_before": meta.get("render_layout_v3_density_before"),
                "render_layout_v3_density_after": meta.get("render_layout_v3_density_after"),
                "render_readability_v4_attempted": meta.get("render_readability_v4_attempted"),
                "render_readability_v4_status": meta.get("render_readability_v4_status"),
                "render_readability_v4_before_score": meta.get("render_readability_v4_before_score"),
                "render_readability_v4_after_score": meta.get("render_readability_v4_after_score"),
                "render_readability_v4_candidate_count": meta.get("render_readability_v4_candidate_count"),
                "render_readability_v4_selected_candidate": meta.get("render_readability_v4_selected_candidate"),
                "render_readability_v4_unresolved_reason": meta.get("render_readability_v4_unresolved_reason"),
                "render_readability_v4_edge_contact_before": meta.get("render_readability_v4_edge_contact_before"),
                "render_readability_v4_edge_contact_after": meta.get("render_readability_v4_edge_contact_after"),
                "render_readability_v4_density_before": meta.get("render_readability_v4_density_before"),
                "render_readability_v4_density_after": meta.get("render_readability_v4_density_after"),
                "render_readability_v4_shape_source": meta.get("render_readability_v4_shape_source"),
                "render_readability_v4_final_class": meta.get("render_readability_v4_final_class"),
                "render_readability_v5_attempted": meta.get("render_readability_v5_attempted"),
                "render_readability_v5_status": meta.get("render_readability_v5_status"),
                "render_readability_v5_before_score": meta.get("render_readability_v5_before_score"),
                "render_readability_v5_after_score": meta.get("render_readability_v5_after_score"),
                "render_readability_v5_candidate_count": meta.get("render_readability_v5_candidate_count"),
                "render_readability_v5_selected_candidate": meta.get("render_readability_v5_selected_candidate"),
                "render_readability_v5_source_column_count": meta.get("render_readability_v5_source_column_count"),
                "render_readability_v5_shape_source": meta.get("render_readability_v5_shape_source"),
                "render_readability_v5_density_before": meta.get("render_readability_v5_density_before"),
                "render_readability_v5_density_after": meta.get("render_readability_v5_density_after"),
                "render_readability_v5_edge_contact_before": meta.get("render_readability_v5_edge_contact_before"),
                "render_readability_v5_edge_contact_after": meta.get("render_readability_v5_edge_contact_after"),
                "render_readability_v5_final_class": meta.get("render_readability_v5_final_class"),
                "render_readability_v5_unresolved_reason": meta.get("render_readability_v5_unresolved_reason"),
                "render_blocked_by_visual_contract": meta.get("render_blocked_by_visual_contract"),
                "render_visual_contract_blocker_reason": meta.get("render_visual_contract_blocker_reason"),
                "render_visual_contract_audit_status": meta.get("render_visual_contract_audit_status"),
                "render_allowed_despite_audit_blocker": meta.get("render_allowed_despite_audit_blocker"),
                "render_suppressed_by_legacy_reason": meta.get("render_suppressed_by_legacy_reason"),
                "render_readability_hard_blocker": meta.get("render_readability_hard_blocker"),
                "render_readability_warning_reason": meta.get("render_readability_warning_reason"),
                "render_allowed_area_bbox": meta.get("render_allowed_area_bbox") or render.get("render_allowed_area_bbox") or [],
                "render_allowed_area_source": meta.get("render_allowed_area_source") or render.get("render_allowed_area_source"),
                "final_render_outside_allowed_area_ratio": meta.get("final_render_outside_allowed_area_ratio"),
                "render_outside_text_area_container": meta.get("render_outside_text_area_container"),
                "cleanup_render_mismatch_ratio": meta.get("cleanup_render_mismatch_ratio"),
                "cleanup_render_area_mismatch": meta.get("cleanup_render_area_mismatch"),
                "cleanup_outside_allowed_area_ratio": meta.get("cleanup_outside_allowed_area_ratio"),
                "cleanup_does_not_cover_source_glyphs": meta.get("cleanup_does_not_cover_source_glyphs"),
                "source_glyph_mask_id": meta.get("source_glyph_mask_id"),
                "cleanup_source_tracking_required": meta.get("cleanup_source_tracking_required"),
                "source_glyph_mask_generation_required": meta.get("source_glyph_mask_generation_required"),
                "source_glyph_mask_generation_status": meta.get("source_glyph_mask_generation_status"),
                "source_glyph_mask_requirement_status": meta.get("source_glyph_mask_requirement_status"),
                "source_glyph_mask_not_generated_reason": meta.get("source_glyph_mask_not_generated_reason"),
                "source_glyph_mask_review_only": meta.get("source_glyph_mask_review_only"),
                "source_glyph_mask_required": meta.get("source_glyph_mask_required"),
                "source_glyph_mask_generated": meta.get("source_glyph_mask_generated"),
                "source_glyph_mask_consumed_by_renderer": meta.get("source_glyph_mask_consumed_by_renderer"),
                "source_glyph_mask_generation_method": meta.get("source_glyph_mask_generation_method"),
                "source_glyph_mask_missing_reason": meta.get("source_glyph_mask_missing_reason"),
                "source_glyph_mask_fallback_used": meta.get("source_glyph_mask_fallback_used"),
                "source_glyph_mask_fallback_reason": meta.get("source_glyph_mask_fallback_reason"),
                "source_glyph_mask_failure_reason": meta.get("source_glyph_mask_failure_reason"),
                "source_glyph_erasure_bbox": meta.get("source_glyph_erasure_bbox"),
                "source_glyph_erasure_expected_area_bbox": meta.get("source_glyph_erasure_expected_area_bbox"),
                "source_glyph_erasure_expected_pixels": meta.get("source_glyph_erasure_expected_pixels"),
                "source_glyph_erasure_coverage_ratio": meta.get("source_glyph_erasure_coverage_ratio"),
                "cleanup_covers_source_glyphs": meta.get("cleanup_covers_source_glyphs"),
                "source_glyph_mask_expected_bbox": meta.get("source_glyph_mask_expected_bbox"),
                "source_glyph_mask_actual_bbox": meta.get("source_glyph_mask_actual_bbox"),
                "source_glyph_mask_expected_overlap_ratio": meta.get("source_glyph_mask_expected_overlap_ratio"),
                "source_glyph_mask_container_clip_reason": meta.get("source_glyph_mask_container_clip_reason"),
                "caption_background_mask_generation_method": meta.get("caption_background_mask_generation_method"),
                "caption_background_mask_coverage_ratio": meta.get("caption_background_mask_coverage_ratio"),
                "phase2c_mask_adjustment_reason": meta.get("phase2c_mask_adjustment_reason"),
                "side_caption_mask_generation_method": meta.get("side_caption_mask_generation_method"),
                "side_caption_mask_coverage_ratio": meta.get("side_caption_mask_coverage_ratio"),
                "cleanup_source_erasure_failure_reason": meta.get("cleanup_source_erasure_failure_reason"),
                "cleanup_visual_artifact_risk": meta.get("cleanup_visual_artifact_risk"),
                "pre_render_source_erasure_required": meta.get("pre_render_source_erasure_required"),
                "pre_render_source_erasure_status": meta.get("pre_render_source_erasure_status"),
                "pre_render_source_erasure_failure_reason": meta.get("pre_render_source_erasure_failure_reason"),
                "pre_render_source_erasure_residual_ratio": meta.get("pre_render_source_erasure_residual_ratio"),
                "pre_render_residual_score": meta.get("pre_render_residual_score"),
                "pre_render_source_erasure_residual_pixels": meta.get("pre_render_source_erasure_residual_pixels"),
                "pre_render_source_erasure_expected_pixels": meta.get("pre_render_source_erasure_expected_pixels"),
                "pre_render_source_erasure_retry_attempted": meta.get("pre_render_source_erasure_retry_attempted"),
                "pre_render_source_erasure_retry_status": meta.get("pre_render_source_erasure_retry_status"),
                "pre_render_source_erasure_retry_reason": meta.get("pre_render_source_erasure_retry_reason"),
                "pre_render_source_erasure_retry_mask": meta.get("pre_render_source_erasure_retry_mask"),
                "pre_render_source_erasure_retry_mask_bbox": meta.get("pre_render_source_erasure_retry_mask_bbox"),
                "pre_render_source_erasure_retry_mask_coverage": meta.get("pre_render_source_erasure_retry_mask_coverage"),
                "source_erasure_failure_reason": meta.get("source_erasure_failure_reason"),
                "source_erasure_proof_action": meta.get("source_erasure_proof_action"),
                "represented_child_cleanup_proof_status": meta.get("represented_child_cleanup_proof_status"),
                "represented_child_cleanup_required": meta.get("represented_child_cleanup_required"),
                "represented_child_cleanup_mask_generated": meta.get("represented_child_cleanup_mask_generated"),
                "represented_child_cleanup_mask_consumed": meta.get("represented_child_cleanup_mask_consumed"),
                "represented_child_cleanup_coverage_ratio": meta.get("represented_child_cleanup_coverage_ratio"),
                "represented_child_cleanup_failure_reason": meta.get("represented_child_cleanup_failure_reason"),
                "cleanup_artifact_risk": meta.get("cleanup_artifact_risk"),
                "cleanup_artifact_risk_reason": meta.get("cleanup_artifact_risk_reason"),
                "render_constraint_fallback_reason": meta.get("render_constraint_fallback_reason") or render.get("render_constraint_fallback_reason"),
                "source_orientation": render.get("source_orientation"),
                "source_size_hint": render.get("source_size_hint"),
                "source_size_min": render.get("source_size_min"),
                "source_size_max": render.get("source_size_max"),
                "font_size": render.get("font_size"),
                "font_size_locked": render.get("font_size_locked"),
                "font_size_policy": render.get("font_size_policy"),
                "font_size_fallback_policy": render.get("font_size_fallback_policy"),
                "route_assist_applied": meta.get("route_assist_applied") or render.get("route_assist_applied"),
                "route_assist_version": meta.get("route_assist_version") or render.get("route_assist_version"),
                "route_assist_suggestion_type": meta.get("route_assist_suggestion_type") or render.get("route_assist_suggestion_type"),
                "route_assist_reason_codes": meta.get("route_assist_reason_codes") or render.get("route_assist_reason_codes"),
                "route_assist_previous_semantic_class": meta.get("route_assist_previous_semantic_class") or render.get("route_assist_previous_semantic_class"),
                "route_assist_previous_cleanup_mode": meta.get("route_assist_previous_cleanup_mode") or render.get("route_assist_previous_cleanup_mode"),
                "route_assist_previous_bg_text": meta.get("route_assist_previous_bg_text") or render.get("route_assist_previous_bg_text"),
                "route_assist_new_semantic_class": meta.get("route_assist_new_semantic_class") or render.get("route_assist_new_semantic_class"),
                "route_assist_new_cleanup_mode": meta.get("route_assist_new_cleanup_mode") or render.get("route_assist_new_cleanup_mode"),
                "route_assist_new_bg_text": meta.get("route_assist_new_bg_text") or render.get("route_assist_new_bg_text"),
                "route_assist_confidence": meta.get("route_assist_confidence") or render.get("route_assist_confidence"),
                "route_assist_linked_container_id": meta.get("route_assist_linked_container_id") or render.get("route_assist_linked_container_id"),
                "route_assist_linked_ownership_id": meta.get("route_assist_linked_ownership_id") or render.get("route_assist_linked_ownership_id"),
                "render_constraint_applied": meta.get("render_constraint_applied") or render.get("render_constraint_applied"),
                "render_constraint_version": meta.get("render_constraint_version") or render.get("render_constraint_version"),
                "render_constraint_source": meta.get("render_constraint_source") or render.get("render_constraint_source"),
                "render_constraint_candidate_id": meta.get("render_constraint_candidate_id") or render.get("render_constraint_candidate_id"),
                "render_constraint_scope": meta.get("render_constraint_scope") or render.get("render_constraint_scope"),
                "render_constraint_previous_final_render_bbox": meta.get("render_constraint_previous_final_render_bbox") or render.get("render_constraint_previous_final_render_bbox"),
                "render_constraint_new_final_render_bbox": meta.get("render_constraint_new_final_render_bbox") or render.get("render_constraint_new_final_render_bbox"),
                "render_constraint_planner_suggestion_id": meta.get("render_constraint_planner_suggestion_id") or render.get("render_constraint_planner_suggestion_id"),
                "render_constraint_planner_suggestion_type": meta.get("render_constraint_planner_suggestion_type") or render.get("render_constraint_planner_suggestion_type"),
                "render_constraint_reason_codes": meta.get("render_constraint_reason_codes") or render.get("render_constraint_reason_codes"),
                "render_constraint_inferred_container_id": meta.get("render_constraint_inferred_container_id") or render.get("render_constraint_inferred_container_id"),
                "render_constraint_inferred_container_bbox": meta.get("render_constraint_inferred_container_bbox") or render.get("render_constraint_inferred_container_bbox"),
                "render_constraint_proposed_action": meta.get("render_constraint_proposed_action") or render.get("render_constraint_proposed_action"),
                "render_constraint_font_wrap_recomputed": meta.get("render_constraint_font_wrap_recomputed") or render.get("render_constraint_font_wrap_recomputed"),
                "render_constraint_text_completeness_policy": meta.get("render_constraint_text_completeness_policy") or render.get("render_constraint_text_completeness_policy"),
                "render_constraint_previous_outside_container_ratio": meta.get("render_constraint_previous_outside_container_ratio") or render.get("render_constraint_previous_outside_container_ratio"),
                "previous_final_render_bbox": _meta_or_render(meta, render, "previous_final_render_bbox"),
                "new_final_render_bbox": _meta_or_render(meta, render, "new_final_render_bbox"),
                "wrapped_lines_before": _meta_or_render(meta, render, "wrapped_lines_before"),
                "wrapped_lines_after": _meta_or_render(meta, render, "wrapped_lines_after"),
                "text_completeness_passed": _meta_or_render(meta, render, "text_completeness_passed"),
                "fallback_reason": _meta_or_render(meta, render, "fallback_reason"),
                "would_change_behavior": _meta_or_render(meta, render, "would_change_behavior"),
                "timing": meta.get("timing"),
                "flags": _json_safe(flags),
                "text_area_container_id": meta.get("text_area_container_id") or region.get("text_area_container_id"),
                "text_area_container_type": meta.get("text_area_container_type") or region.get("text_area_container_type"),
                "text_area_route_intent": meta.get("text_area_route_intent") or region.get("text_area_route_intent"),
                "text_area_ocr_eligible": meta.get("text_area_ocr_eligible") if "text_area_ocr_eligible" in meta else region.get("text_area_ocr_eligible"),
                "text_area_detection_source": meta.get("text_area_detection_source") or region.get("text_area_detection_source"),
                "text_area_fallback_reason": meta.get("text_area_fallback_reason") or region.get("text_area_fallback_reason"),
                "text_area_confidence_tier": meta.get("text_area_confidence_tier") or region.get("text_area_confidence_tier"),
                "text_area_container_bbox": meta.get("text_area_container_bbox") or region.get("text_area_container_bbox") or [],
                "text_area_reason_codes": meta.get("text_area_reason_codes") or region.get("text_area_reason_codes") or [],
                "text_area_conflict_flags": meta.get("text_area_conflict_flags") or region.get("text_area_conflict_flags") or [],
                "text_area_pre_ocr_authority": meta.get("text_area_pre_ocr_authority") if "text_area_pre_ocr_authority" in meta else region.get("text_area_pre_ocr_authority"),
                "text_area_enriched_from_region": meta.get("text_area_enriched_from_region") if "text_area_enriched_from_region" in meta else region.get("text_area_enriched_from_region"),
                "text_area_ocr_eligibility_reason": meta.get("text_area_ocr_eligibility_reason") or region.get("text_area_ocr_eligibility_reason"),
                "logical_text_block_id": meta.get("logical_text_block_id") or region.get("logical_text_block_id"),
                "physical_bubble_graph_id": meta.get("physical_bubble_graph_id") or region.get("physical_bubble_graph_id") or render.get("physical_bubble_graph_id"),
                "logical_text_block_v3_status": meta.get("logical_text_block_v3_status") or region.get("logical_text_block_v3_status") or render.get("logical_text_block_v3_status"),
                "logical_text_block_translation_unit": (
                    meta.get("logical_text_block_translation_unit")
                    if "logical_text_block_translation_unit" in meta
                    else region.get("logical_text_block_translation_unit") or render.get("logical_text_block_translation_unit")
                ),
                "logical_text_block_container_id": meta.get("logical_text_block_container_id") or region.get("logical_text_block_container_id"),
                "logical_text_block_role": meta.get("logical_text_block_role") or region.get("logical_text_block_role"),
                "logical_text_block_member_region_ids": meta.get("logical_text_block_member_region_ids") or region.get("logical_text_block_member_region_ids") or [],
                "logical_text_block_anchor_region_id": meta.get("logical_text_block_anchor_region_id") or region.get("logical_text_block_anchor_region_id"),
                "logical_text_block_transferred_region_ids": meta.get("logical_text_block_transferred_region_ids") or region.get("logical_text_block_transferred_region_ids") or [],
                "logical_text_block_duplicate_region_ids": meta.get("logical_text_block_duplicate_region_ids") or region.get("logical_text_block_duplicate_region_ids") or [],
                "logical_text_block_punctuation_child_ids": meta.get("logical_text_block_punctuation_child_ids") or region.get("logical_text_block_punctuation_child_ids") or [],
                "logical_text_block_noise_child_ids": meta.get("logical_text_block_noise_child_ids") or region.get("logical_text_block_noise_child_ids") or [],
                "logical_text_block_source_text": meta.get("logical_text_block_source_text") or region.get("logical_text_block_source_text"),
                "logical_text_block_reason_codes": meta.get("logical_text_block_reason_codes") or region.get("logical_text_block_reason_codes") or [],
                "logical_text_block_confidence": meta.get("logical_text_block_confidence") if "logical_text_block_confidence" in meta else region.get("logical_text_block_confidence"),
                "logical_text_block_would_change_behavior": meta.get("logical_text_block_would_change_behavior") if "logical_text_block_would_change_behavior" in meta else region.get("logical_text_block_would_change_behavior"),
                "logical_text_ownership_status": meta.get("logical_text_ownership_status") or region.get("logical_text_ownership_status"),
                "logical_text_block_text_conservation_status": meta.get("logical_text_block_text_conservation_status") or region.get("logical_text_block_text_conservation_status"),
                "logical_text_block_allowed_bbox": meta.get("logical_text_block_allowed_bbox") or region.get("logical_text_block_allowed_bbox") or [],
                "logical_text_block_historical_translation": meta.get("logical_text_block_historical_translation") or render.get("logical_text_block_historical_translation"),
                "logical_text_physical_bubble_id": meta.get("logical_text_physical_bubble_id") or region.get("logical_text_physical_bubble_id") or render.get("logical_text_physical_bubble_id"),
                "logical_text_physical_bubble_member_container_ids": meta.get("logical_text_physical_bubble_member_container_ids") or region.get("logical_text_physical_bubble_member_container_ids") or render.get("logical_text_physical_bubble_member_container_ids") or [],
                "logical_text_physical_bubble_source": meta.get("logical_text_physical_bubble_source") or region.get("logical_text_physical_bubble_source") or render.get("logical_text_physical_bubble_source"),
                "logical_text_physical_bubble_reason_codes": meta.get("logical_text_physical_bubble_reason_codes") or region.get("logical_text_physical_bubble_reason_codes") or render.get("logical_text_physical_bubble_reason_codes") or [],
                "logical_text_source_quality_status": meta.get("logical_text_source_quality_status") or region.get("logical_text_source_quality_status") or render.get("logical_text_source_quality_status"),
                "logical_text_source_quality_reason_codes": meta.get("logical_text_source_quality_reason_codes") or region.get("logical_text_source_quality_reason_codes") or render.get("logical_text_source_quality_reason_codes") or [],
                "logical_text_source_quality_action": meta.get("logical_text_source_quality_action") or region.get("logical_text_source_quality_action") or render.get("logical_text_source_quality_action"),
                "logical_text_blocked_fragment_resolution": meta.get("logical_text_blocked_fragment_resolution") or region.get("logical_text_blocked_fragment_resolution") or render.get("logical_text_blocked_fragment_resolution"),
                "logical_text_source_reconstruction_status": meta.get("logical_text_source_reconstruction_status") or region.get("logical_text_source_reconstruction_status") or render.get("logical_text_source_reconstruction_status"),
                "logical_text_source_reconstruction_applied": bool(
                    meta.get("logical_text_source_reconstruction_applied")
                    or region.get("logical_text_source_reconstruction_applied")
                    or render.get("logical_text_source_reconstruction_applied")
                ),
                "logical_text_source_reconstruction_before_text": meta.get("logical_text_source_reconstruction_before_text") or region.get("logical_text_source_reconstruction_before_text") or render.get("logical_text_source_reconstruction_before_text"),
                "logical_text_source_reconstruction_after_text": meta.get("logical_text_source_reconstruction_after_text") or region.get("logical_text_source_reconstruction_after_text") or render.get("logical_text_source_reconstruction_after_text"),
                "logical_text_source_reconstruction_ocr_confidence": (
                    meta.get("logical_text_source_reconstruction_ocr_confidence")
                    if "logical_text_source_reconstruction_ocr_confidence" in meta
                    else region.get("logical_text_source_reconstruction_ocr_confidence") or render.get("logical_text_source_reconstruction_ocr_confidence")
                ),
                "logical_text_source_reconstruction_crop_bbox": meta.get("logical_text_source_reconstruction_crop_bbox") or region.get("logical_text_source_reconstruction_crop_bbox") or render.get("logical_text_source_reconstruction_crop_bbox") or [],
                "logical_text_block_source_reconstruction_required": (
                    meta.get("logical_text_block_source_reconstruction_required")
                    if "logical_text_block_source_reconstruction_required" in meta
                    else region.get("logical_text_block_source_reconstruction_required") or render.get("logical_text_block_source_reconstruction_required")
                ),
                "logical_text_block_source_reconstruction_status": meta.get("logical_text_block_source_reconstruction_status") or region.get("logical_text_block_source_reconstruction_status") or render.get("logical_text_block_source_reconstruction_status"),
                "logical_text_block_source_reconstruction_crop_bbox": meta.get("logical_text_block_source_reconstruction_crop_bbox") or region.get("logical_text_block_source_reconstruction_crop_bbox") or render.get("logical_text_block_source_reconstruction_crop_bbox") or [],
                "logical_text_block_included_child_region_ids": meta.get("logical_text_block_included_child_region_ids") or region.get("logical_text_block_included_child_region_ids") or render.get("logical_text_block_included_child_region_ids") or [],
                "logical_text_block_rejected_child_region_ids": meta.get("logical_text_block_rejected_child_region_ids") or region.get("logical_text_block_rejected_child_region_ids") or render.get("logical_text_block_rejected_child_region_ids") or [],
                "logical_text_block_unresolved_reason": meta.get("logical_text_block_unresolved_reason") or region.get("logical_text_block_unresolved_reason") or render.get("logical_text_block_unresolved_reason"),
                "logical_text_source_reconstruction_included_child_region_ids": meta.get("logical_text_source_reconstruction_included_child_region_ids") or region.get("logical_text_source_reconstruction_included_child_region_ids") or render.get("logical_text_source_reconstruction_included_child_region_ids") or [],
                "logical_text_source_reconstruction_rejected_child_region_ids": meta.get("logical_text_source_reconstruction_rejected_child_region_ids") or region.get("logical_text_source_reconstruction_rejected_child_region_ids") or render.get("logical_text_source_reconstruction_rejected_child_region_ids") or [],
                "logical_text_source_reconstruction_reason_codes": meta.get("logical_text_source_reconstruction_reason_codes") or region.get("logical_text_source_reconstruction_reason_codes") or render.get("logical_text_source_reconstruction_reason_codes") or [],
                "logical_text_source_reconstruction_child_fragment_status": meta.get("logical_text_source_reconstruction_child_fragment_status") or region.get("logical_text_source_reconstruction_child_fragment_status") or render.get("logical_text_source_reconstruction_child_fragment_status") or [],
                "speech_source_repair_required": bool(
                    meta.get("speech_source_repair_required")
                    or region.get("speech_source_repair_required")
                    or render.get("speech_source_repair_required")
                ),
                "punctuation_only_speech_recovery_required": bool(
                    meta.get("punctuation_only_speech_recovery_required")
                    or region.get("punctuation_only_speech_recovery_required")
                    or render.get("punctuation_only_speech_recovery_required")
                ),
                "source_quality_blocked_visual_fail": bool(
                    meta.get("source_quality_blocked_visual_fail")
                    or region.get("source_quality_blocked_visual_fail")
                    or render.get("source_quality_blocked_visual_fail")
                ),
                "logical_text_speech_container_override_applied": bool(
                    meta.get("logical_text_speech_container_override_applied")
                    or region.get("logical_text_speech_container_override_applied")
                    or render.get("logical_text_speech_container_override_applied")
                ),
                "logical_text_render_eligibility_repaired": (
                    meta.get("logical_text_render_eligibility_repaired")
                    if "logical_text_render_eligibility_repaired" in meta
                    else region.get("logical_text_render_eligibility_repaired") or render.get("logical_text_render_eligibility_repaired")
                ),
                "logical_text_render_eligibility_reason": meta.get("logical_text_render_eligibility_reason") or region.get("logical_text_render_eligibility_reason") or render.get("logical_text_render_eligibility_reason"),
                "ocr_fragment_ownership_status": meta.get("ocr_fragment_ownership_status") or region.get("ocr_fragment_ownership_status") or render.get("ocr_fragment_ownership_status"),
                "ocr_fragment_final_state": meta.get("ocr_fragment_final_state") or region.get("ocr_fragment_final_state") or render.get("ocr_fragment_final_state"),
                "active_translation_unit_id": meta.get("active_translation_unit_id") or region.get("active_translation_unit_id") or render.get("active_translation_unit_id"),
                "source_text_represented_by_block_id": meta.get("source_text_represented_by_block_id") or region.get("source_text_represented_by_block_id") or render.get("source_text_represented_by_block_id"),
                "source_conservation_status": meta.get("source_conservation_status") or region.get("source_conservation_status") or render.get("source_conservation_status"),
                "source_conservation_failure_reason": meta.get("source_conservation_failure_reason") or region.get("source_conservation_failure_reason") or render.get("source_conservation_failure_reason"),
                "source_glyph_mask_anchor_block_id": meta.get("source_glyph_mask_anchor_block_id") or render.get("source_glyph_mask_anchor_block_id"),
                "source_glyph_mask_parent_logical_text_unit_id": meta.get("source_glyph_mask_parent_logical_text_unit_id") or render.get("source_glyph_mask_parent_logical_text_unit_id"),
                "source_glyph_mask_text_block_root_id": meta.get("source_glyph_mask_text_block_root_id") or render.get("source_glyph_mask_text_block_root_id"),
                "source_glyph_mask_anchor_child_id": meta.get("source_glyph_mask_anchor_child_id") or render.get("source_glyph_mask_anchor_child_id"),
                "source_glyph_mask_child_segment_ids": meta.get("source_glyph_mask_child_segment_ids") or render.get("source_glyph_mask_child_segment_ids") or [],
                "route_owned_foreground_contract_status": meta.get("route_owned_foreground_contract_status") or render.get("route_owned_foreground_contract_status"),
                "foreground_mask_method": meta.get("foreground_mask_method") or render.get("foreground_mask_method"),
                "foreground_mask_pixels": meta.get("foreground_mask_pixels") or render.get("foreground_mask_pixels"),
                "foreground_mask_bbox": meta.get("foreground_mask_bbox") or render.get("foreground_mask_bbox"),
                "erase_mask_method": meta.get("erase_mask_method") or render.get("erase_mask_method"),
                "erase_mask_pixels": meta.get("erase_mask_pixels") or render.get("erase_mask_pixels"),
                "erase_mask_bbox": meta.get("erase_mask_bbox") or render.get("erase_mask_bbox"),
                "erase_mask_growth_ratio": meta.get("erase_mask_growth_ratio") or render.get("erase_mask_growth_ratio"),
                "erase_mask_allowed_area": meta.get("erase_mask_allowed_area") or render.get("erase_mask_allowed_area"),
                "erase_mask_rejected_reason": meta.get("erase_mask_rejected_reason") or render.get("erase_mask_rejected_reason"),
                "erase_mask_artifact_risk": (
                    meta.get("erase_mask_artifact_risk")
                    if "erase_mask_artifact_risk" in meta
                    else render.get("erase_mask_artifact_risk")
                ),
                "erase_mask_visual_scope": meta.get("erase_mask_visual_scope") or render.get("erase_mask_visual_scope"),
                "text_block_root_id": meta.get("text_block_root_id") or region.get("text_block_root_id") or render.get("text_block_root_id"),
                "parent_logical_text_unit_id": meta.get("parent_logical_text_unit_id") or region.get("parent_logical_text_unit_id") or render.get("parent_logical_text_unit_id"),
                "child_recognized_text_segment_id": meta.get("child_recognized_text_segment_id") or region.get("child_recognized_text_segment_id") or render.get("child_recognized_text_segment_id"),
                "child_final_state": meta.get("child_final_state") or region.get("child_final_state") or render.get("child_final_state"),
                "represented_by_parent_id": meta.get("represented_by_parent_id") or region.get("represented_by_parent_id") or render.get("represented_by_parent_id"),
                "translated_independently": (
                    meta.get("translated_independently")
                    if "translated_independently" in meta
                    else region.get("translated_independently") or render.get("translated_independently")
                ),
                "cleanup_independently": (
                    meta.get("cleanup_independently")
                    if "cleanup_independently" in meta
                    else region.get("cleanup_independently") or render.get("cleanup_independently")
                ),
                "render_independently": (
                    meta.get("render_independently")
                    if "render_independently" in meta
                    else region.get("render_independently") or render.get("render_independently")
                ),
                "hierarchy_unresolved_reason": meta.get("hierarchy_unresolved_reason") or region.get("hierarchy_unresolved_reason") or render.get("hierarchy_unresolved_reason"),
                "hierarchy_reason_codes": meta.get("hierarchy_reason_codes") or region.get("hierarchy_reason_codes") or render.get("hierarchy_reason_codes") or [],
                "parent_logical_text_unit_child_segment_ids": meta.get("parent_logical_text_unit_child_segment_ids") or region.get("parent_logical_text_unit_child_segment_ids") or render.get("parent_logical_text_unit_child_segment_ids") or [],
                "parent_logical_text_unit_anchor_child_id": meta.get("parent_logical_text_unit_anchor_child_id") or region.get("parent_logical_text_unit_anchor_child_id") or render.get("parent_logical_text_unit_anchor_child_id"),
            }
        )
    set_count(context, "translated_regions", translated)
    set_count(context, "skipped_regions", skipped)
    source_glyph_mask_summary = dict(context.get("source_glyph_masks") or {})
    if audit_regions:
        tracking_required = sum(1 for region in audit_regions if region.get("cleanup_source_tracking_required") is True)
        generation_required = sum(1 for region in audit_regions if region.get("source_glyph_mask_generation_required") is True)
        region_required = sum(1 for region in audit_regions if region.get("source_glyph_mask_required") is True)
        region_generated = sum(1 for region in audit_regions if region.get("source_glyph_mask_generated") is True)
        region_consumed = sum(1 for region in audit_regions if region.get("source_glyph_mask_consumed_by_renderer") is True)
        region_review_only = sum(1 for region in audit_regions if region.get("source_glyph_mask_review_only") is True)
        fallback_counts: dict[str, int] = {}
        missing_counts: dict[str, int] = {}
        not_generated_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for region in audit_regions:
            status = str(region.get("source_glyph_mask_generation_status") or "")
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
            fallback_reason = str(region.get("source_glyph_mask_fallback_reason") or "")
            if fallback_reason:
                fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
            missing_reason = str(region.get("source_glyph_mask_missing_reason") or "")
            if missing_reason:
                missing_counts[missing_reason] = missing_counts.get(missing_reason, 0) + 1
            not_generated_reason = str(region.get("source_glyph_mask_not_generated_reason") or "")
            if not_generated_reason:
                not_generated_counts[not_generated_reason] = not_generated_counts.get(not_generated_reason, 0) + 1
        source_glyph_mask_summary["cleanup_source_tracking_region_required_count"] = tracking_required
        source_glyph_mask_summary["source_glyph_mask_region_generation_required_count"] = generation_required
        source_glyph_mask_summary["source_glyph_mask_region_required_count"] = region_required
        source_glyph_mask_summary["source_glyph_mask_region_generated_count"] = region_generated
        source_glyph_mask_summary["source_glyph_mask_region_consumed_count"] = region_consumed
        source_glyph_mask_summary["source_glyph_mask_region_review_only_count"] = region_review_only
        source_glyph_mask_summary["source_glyph_mask_region_generation_status_counts"] = status_counts
        source_glyph_mask_summary["source_glyph_mask_region_missing_reason_counts"] = missing_counts
        source_glyph_mask_summary["source_glyph_mask_region_not_generated_reason_counts"] = not_generated_counts
        source_glyph_mask_summary["source_glyph_mask_region_fallback_reason_counts"] = fallback_counts
    hierarchy = dict(context.get("text_block_hierarchy") or {})
    source_child_cleanup_records = list(context.get("source_child_cleanup_records") or [])
    uncleaned_source_child_count = sum(
        1
        for region in audit_regions
        if region.get("source_child_cleanup_required") is True
        and region.get("source_child_cleanup_covered") is not True
    )
    set_count(context, "uncleaned_source_child_count", uncleaned_source_child_count)
    return {
        "page_id": context.get("page_id"),
        "page_class": context.get("page_class"),
        "debug_artifact_level": context.get("debug_artifact_level") or debug_artifact_level(),
        "source_path": context.get("source_path"),
        "output_path": context.get("output_path"),
        "timing": context.get("timing", {}),
        "counts": context.get("counts", {}),
        "scoped_ocr_trace": context.get("scoped_ocr_trace", []),
        "route_owned_ocr_retry_attempts": context.get("route_owned_ocr_retry_attempts", []),
        "render_cleanup_operations": context.get("render_cleanup_operations", []),
        "cleanup_operation_traces": context.get("cleanup_operation_traces", []),
        "phase2e_cleanup_propagation_trace": context.get("phase2e_cleanup_propagation_trace", []),
        "source_child_cleanup_records": source_child_cleanup_records,
        "uncleaned_source_child_count": uncleaned_source_child_count,
        "route_assist": context.get("route_assist"),
        "text_block_hierarchy": hierarchy,
        "text_block_hierarchy_version": hierarchy.get("text_block_hierarchy_version"),
        "text_block_hierarchy_generated": bool(hierarchy.get("text_block_hierarchy_generated")),
        "text_area_root_blocks": hierarchy.get("text_area_root_blocks") or [],
        "parent_logical_text_units": hierarchy.get("parent_logical_text_units") or [],
        "child_recognized_text_segments": hierarchy.get("child_recognized_text_segments") or [],
        "text_block_hierarchy_summary": hierarchy.get("text_block_hierarchy_summary") or {},
        "unresolved_child_segments": hierarchy.get("unresolved_child_segments") or [],
        "translation_unit_ids": hierarchy.get("translation_unit_ids") or [],
        "cleanup_unit_ids": hierarchy.get("cleanup_unit_ids") or [],
        "render_unit_ids": hierarchy.get("render_unit_ids") or [],
        "source_glyph_masks": source_glyph_mask_summary,
        "cleanup_job_contracts": context.get("cleanup_job_contracts") or {},
        "cleanup_mask_contracts": context.get("cleanup_mask_contracts") or {},
        "render_eligibility_contracts": context.get("render_eligibility_contracts") or {},
        "cleanup_plan_contracts": context.get("cleanup_plan_contracts") or {},
        "cleanup_backend_inventory": context.get("cleanup_backend_inventory") or {},
        "cleanup_result_contracts": context.get("cleanup_result_contracts") or {},
        "cleanup_proof_contracts": context.get("cleanup_proof_contracts") or {},
        "cleanup_runtime_result_contracts": context.get("cleanup_runtime_result_contracts") or {},
        "cleanup_runtime_proof_contracts": context.get("cleanup_runtime_proof_contracts") or {},
        "cleanup_runtime_status": context.get("cleanup_runtime_status") or {},
        "cleanup_upstream_commit_contracts": context.get("cleanup_upstream_commit_contracts") or {},
        "cleanup_upstream_renderer_input_path": context.get("cleanup_upstream_renderer_input_path") or "",
        "authorized_source_style_views": context.get("authorized_source_style_views") or {},
        "authorized_source_style_view_artifacts": context.get(
            "authorized_source_style_view_artifacts"
        ) or {},
        "parent_font_detection": context.get("parent_font_detection") or {},
        "logical_text_source_reconstruction": context.get("logical_text_source_reconstruction") or {},
        "logical_text_punctuation_only_speech_recovery": context.get("logical_text_punctuation_only_speech_recovery") or {},
        "root_reconstruction_executor": context.get("root_reconstruction_executor") or {},
        "caption_component_recovery_candidates": context.get("caption_component_recovery_candidates") or [],
        "caption_container_recovery_candidates": context.get("caption_container_recovery_candidates") or [],
        "translation_unit_timings": context.get("translation_unit_timings") or [],
        "regions": audit_regions,
    }


def _write_overlay(context: dict[str, Any], audit_regions: list[dict[str, Any]], overlay_path: str) -> None:
    if Image is None or ImageDraw is None:
        return
    base_path = str(context.get("output_path") or context.get("source_path") or "")
    if not os.path.isfile(base_path):
        base_path = str(context.get("source_path") or "")
    if not os.path.isfile(base_path):
        return
    with Image.open(base_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _overlay_font()
    draw.rectangle((8, 8, 520, 90), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((16, 14), "Debug overlay: red=detected, green=render, cyan=cleanup", fill=(0, 0, 0), font=font)
    draw.text((16, 38), "Labels: order:id | class | status | OCR", fill=(0, 0, 0), font=font)
    draw.text((16, 62), f"page={context.get('page_id')} class={context.get('page_class')}", fill=(0, 0, 0), font=font)
    for region in audit_regions:
        bbox = _bbox_xyxy(region.get("bbox"))
        render_box = _xyxy(region.get("final_render_bbox"))
        cleanup_box = _xyxy(region.get("cleanup_mask_bbox"))
        if bbox:
            draw.rectangle(bbox, outline=(255, 0, 0), width=3)
        if cleanup_box:
            draw.rectangle(cleanup_box, outline=(0, 180, 220), width=2)
        if render_box:
            draw.rectangle(render_box, outline=(0, 180, 0), width=3)
        label_box = render_box or bbox
        if label_box:
            status = "skip" if region.get("skip_reason") else ("tx" if region.get("translated_text") else "empty")
            ocr = _short_label(region.get("ocr_text", ""))
            label = (
                f"{region.get('reading_order_index')}:{region.get('region_id')} | "
                f"{region.get('semantic_class')} | {status} | {ocr}"
            )
            x0, y0, _x1, _y1 = label_box
            ty = max(0, int(y0) - 18)
            draw.rectangle((int(x0), ty, int(x0) + min(720, len(label) * 8 + 8), ty + 18), fill=(255, 255, 220))
            draw.text((int(x0) + 4, ty + 2), label, fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
    canvas.save(overlay_path, quality=92)


def _overlay_font():
    if ImageFont is None:
        return None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        try:
            if os.path.isfile(path):
                return ImageFont.truetype(path, 14)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _normalize_image_size(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        width = int(round(float(value[0])))
        height = int(round(float(value[1])))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _empty_binary_mask(size: tuple[int, int]) -> Any:
    try:
        import numpy as np

        return np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    except Exception:
        return None


def _binary_mask_array(mask: Any, size: tuple[int, int]) -> Any:
    try:
        import numpy as np

        if mask is None:
            return np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = np.any(arr > 0, axis=2)
        if arr.ndim != 2:
            return np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
        arr = (arr > 0).astype("uint8")
        if arr.shape != (int(size[1]), int(size[0])):
            if Image is None:
                return np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
            pil_mask = Image.fromarray(arr * 255, mode="L")
            resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", 0)
            pil_mask = pil_mask.resize((int(size[0]), int(size[1])), resample)
            arr = (np.asarray(pil_mask) > 0).astype("uint8")
        return arr
    except Exception:
        return _empty_binary_mask(size)


def _mask_union(left: Any, right: Any) -> Any:
    try:
        import numpy as np

        if left is None:
            return right
        if right is None:
            return left
        return np.maximum(left, right)
    except Exception:
        return left if left is not None else right


def _save_binary_mask_array(path: str, arr: Any) -> str:
    if not path or arr is None or Image is None:
        return ""
    try:
        import numpy as np

        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray((np.asarray(arr) > 0).astype("uint8") * 255, mode="L").save(path)
        return path
    except Exception:
        return ""


def _mask_stats(arr: Any) -> dict[str, Any]:
    try:
        import numpy as np

        mask = np.asarray(arr) > 0
        pixels = int(np.count_nonzero(mask))
        if pixels <= 0:
            return {"pixels": 0, "bbox": None}
        ys, xs = np.nonzero(mask)
        return {
            "pixels": pixels,
            "bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ],
        }
    except Exception:
        return {"pixels": 0, "bbox": None}


def _write_json_file(path: str, value: Any) -> str:
    if not path:
        return ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return ""


def _image_size_from_path(path: str) -> tuple[int, int] | None:
    if not path or Image is None or not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _copy_or_write_debug_image(path: str, source_path: str = "", image: Any = None) -> str:
    if not path or Image is None:
        return ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if image is not None and hasattr(image, "save"):
            image.save(path)
            return path
        if source_path and os.path.isfile(source_path):
            with Image.open(source_path) as src:
                src.save(path)
            return path
    except Exception:
        return ""
    return ""


def _write_image_difference(path: str, source_path: str, cleaned_image: Any) -> str:
    if not path or not source_path or Image is None or cleaned_image is None:
        return ""
    try:
        from PIL import ImageChops

        with Image.open(source_path) as source:
            before = source.convert("RGB")
        after = cleaned_image.convert("RGB") if hasattr(cleaned_image, "convert") else Image.fromarray(cleaned_image).convert("RGB")
        if before.size != after.size:
            after = after.resize(before.size)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ImageChops.difference(before, after).save(path)
        return path
    except Exception:
        return ""


def _component_authorization_records(component_authorization_map: Any) -> list[dict[str, Any]]:
    if component_authorization_map is None:
        return []
    if hasattr(component_authorization_map, "components"):
        raw_records = list(getattr(component_authorization_map, "components", []) or [])
    elif isinstance(component_authorization_map, dict):
        raw_records = list(component_authorization_map.get("components") or [])
    else:
        raw_records = []
    records: list[dict[str, Any]] = []
    for record in raw_records:
        if hasattr(record, "to_dict"):
            payload = record.to_dict()
        elif isinstance(record, dict):
            payload = dict(record)
        else:
            payload = dict(getattr(record, "__dict__", {}) or {})
        records.append(payload)
    return records


def _component_authorization_summary(component_authorization_map: Any) -> dict[str, Any]:
    if component_authorization_map is None:
        return {}
    if hasattr(component_authorization_map, "to_audit_dict"):
        try:
            payload = component_authorization_map.to_audit_dict()
            if isinstance(payload, dict):
                return dict(payload.get("summary") or {})
        except Exception:
            return {}
    if isinstance(component_authorization_map, dict):
        return dict(component_authorization_map.get("summary") or {})
    return {}


def _component_state_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        state = str(
            record.get("final_mask_authorization_state")
            or record.get("authorization_state")
            or record.get("group_authorization_state")
            or "unknown"
        )
        counts[state] = counts.get(state, 0) + 1
    return counts


def _write_component_authorization_overlay(
    path: str,
    source_path: str,
    size: tuple[int, int],
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> str:
    if not path or Image is None or ImageDraw is None:
        return ""
    try:
        if source_path and os.path.isfile(source_path):
            with Image.open(source_path) as image:
                base = image.convert("RGBA")
        else:
            base = Image.new("RGBA", size, (255, 255, 255, 255))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        colors = {
            "green": (0, 210, 0, 120),
            "red": (240, 0, 0, 130),
            "orange": (255, 150, 0, 130),
            "gray": (120, 120, 120, 110),
            "grey": (120, 120, 120, 110),
            "blue": (30, 144, 255, 120),
        }
        for record in records:
            bbox = record.get("component_bbox") or record.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            state = str(record.get("final_mask_authorization_state") or record.get("authorization_state") or "")
            final_color = str(record.get("visual_debug_color") or "").lower()
            semantic_color = str(record.get("semantic_visual_color") or "").lower()
            if mode == "auth":
                color_name = final_color or semantic_color
            else:
                color_name = semantic_color or final_color
            if mode == "protected":
                if not (
                    state.startswith("protect_")
                    or color_name in {"red", "orange"}
                    or bool(record.get("must_not_mutate"))
                ):
                    continue
            elif mode == "projection":
                if state.startswith("cleanup_"):
                    color_name = "green"
                elif state.startswith("protect_"):
                    color_name = "red"
                elif "ambiguous" in state or bool(record.get("requires_visual_review")):
                    color_name = "orange"
                elif not color_name:
                    color_name = "gray"
            elif not color_name:
                color_name = "gray"
            x0, y0, x1, y1 = [int(round(float(item))) for item in bbox[:4]]
            rgba = colors.get(color_name, colors["gray"])
            draw.rectangle((x0, y0, x1, y1), fill=rgba, outline=rgba[:3] + (220,), width=2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.alpha_composite(base, overlay).convert("RGB").save(path, quality=92)
        return path
    except Exception:
        return ""


def _save_mask_overlay_array(
    path: str,
    source_path: str,
    size: tuple[int, int],
    arr: Any,
    color: tuple[int, int, int],
) -> str:
    if not path or arr is None or Image is None:
        return ""
    try:
        import numpy as np

        if source_path and os.path.isfile(source_path):
            with Image.open(source_path) as image:
                base = image.convert("RGBA")
        else:
            base = Image.new("RGBA", size, (255, 255, 255, 255))
        mask = _binary_mask_array(arr, size)
        overlay_arr = np.zeros((int(size[1]), int(size[0]), 4), dtype=np.uint8)
        overlay_arr[mask > 0] = [int(color[0]), int(color[1]), int(color[2]), 125]
        overlay = Image.fromarray(overlay_arr, mode="RGBA")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.alpha_composite(base, overlay).convert("RGB").save(path, quality=92)
        return path
    except Exception:
        return ""


def _union_source_glyph_masks(source_glyph_masks: Any, size: tuple[int, int]) -> Any:
    union = _empty_binary_mask(size)
    records = []
    masks_by_region = getattr(source_glyph_masks, "masks_by_region", None)
    if isinstance(masks_by_region, dict):
        records = list(masks_by_region.values())
    elif isinstance(source_glyph_masks, dict):
        raw = source_glyph_masks.get("masks_by_region") or source_glyph_masks.get("source_glyph_masks") or []
        records = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
    for record in records:
        candidate = (
            getattr(record, "erase_mask", None)
            if not isinstance(record, dict)
            else record.get("erase_mask")
        )
        if candidate is None:
            candidate = (
                getattr(record, "foreground_mask", None)
                if not isinstance(record, dict)
                else record.get("foreground_mask")
            )
        if candidate is None:
            candidate = getattr(record, "mask", None) if not isinstance(record, dict) else record.get("mask")
        union = _mask_union(union, _binary_mask_array(candidate, size))
    return union


def _union_runtime_mask_refs(cleanup_runtime_contracts: Any, size: tuple[int, int], *, ref_attr: str) -> Any:
    union = _empty_binary_mask(size)
    for result in list(getattr(cleanup_runtime_contracts, "result_records", []) or []):
        ref = str(getattr(result, ref_attr, "") or "")
        if not ref and isinstance(result, dict):
            ref = str(result.get(ref_attr) or "")
        union = _mask_union(union, _binary_mask_from_ref(ref, size))
    return union


def _runtime_mask_ref_count(cleanup_runtime_contracts: Any, ref_attr: str) -> int:
    count = 0
    for result in list(getattr(cleanup_runtime_contracts, "result_records", []) or []):
        ref = str(getattr(result, ref_attr, "") or "")
        if not ref and isinstance(result, dict):
            ref = str(result.get(ref_attr) or "")
        if ref:
            count += 1
    return count


def _union_runtime_result_masks(cleanup_runtime_contracts: Any, size: tuple[int, int]) -> Any:
    union = _empty_binary_mask(size)
    for result in list(getattr(cleanup_runtime_contracts, "result_records", []) or []):
        mask = getattr(result, "commit_mask", None)
        if mask is None:
            mask = getattr(result, "commit_foreground_mask", None)
        union = _mask_union(union, _binary_mask_array(mask, size))
    return union


def _binary_mask_from_ref(path: str, size: tuple[int, int]) -> Any:
    if not path or Image is None or not os.path.isfile(path):
        return _empty_binary_mask(size)
    try:
        with Image.open(path) as image:
            return _binary_mask_array(image.convert("L"), size)
    except Exception:
        return _empty_binary_mask(size)


def _compact_contract_summary(contract: Any) -> dict[str, Any]:
    if contract is None:
        return {}
    if hasattr(contract, "to_audit_dict"):
        try:
            payload = contract.to_audit_dict()
        except Exception:
            payload = {}
    elif isinstance(contract, dict):
        payload = contract
    else:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return {
        "version": payload.get("version"),
        "page_id": payload.get("page_id"),
        "summary": _json_safe(payload.get("summary") or {}),
        "error_count": len(payload.get("errors") or []),
        "errors": _json_safe(payload.get("errors") or []),
    }


def _bbox_xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    return (x, y, x + max(0, w), y + max(0, h))


def _xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _mask_bbox(mask_info: Any) -> Any:
    if isinstance(mask_info, dict):
        return mask_info.get("bbox")
    return None


def _meta_or_render(meta: dict[str, Any], render: dict[str, Any], key: str) -> Any:
    if key in meta:
        return meta.get(key)
    if key in render:
        return render.get(key)
    return None


def _default_skip_reason(region: dict) -> str | None:
    flags = region.get("flags", {}) or {}
    if flags.get("ignore"):
        return "ignored_by_pipeline"
    if not str(region.get("translation", "") or "").strip():
        return "empty_translation"
    return None


def _normalize_text_for_audit(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _short_label(text: Any, limit: int = 24) -> str:
    value = re.sub(r"\s+", "", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
