# -*- coding: utf-8 -*-
"""Container-local source glyph mask generation for cleanup/rendering.

This stage owns the source-glyph reasoning that was prototyped in the
renderer. It keeps raw masks in memory and exposes JSON-safe metadata for
debug/audit artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any

from app.pipeline.parent_execution_bundle import parent_execution_region_records

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None
    np = None


SOURCE_GLYPH_MASK_VERSION = "source_glyph_masks_v1"
SOURCE_GLYPH_CLEANUP_COVERAGE_THRESHOLD = 0.90
SOURCE_GLYPH_PHASE4_CONTRACT_VERSION = "source_glyph_contract_phase4_v1"
SOURCE_GLYPH_CLASS_VOCABULARY = (
    "speech_flat_bubble",
    "speech_complex_bubble",
    "caption_flat_background",
    "caption_dark_or_screentone",
    "title_or_sign",
    "side_caption_glyph_local",
    "small_reaction",
    "art_entangled_ambiguous",
    "preserve_sfx_decorative",
)
SOURCE_GLYPH_COMPATIBILITY_FIELDS = (
    "source_glyph_mask_generation_status",
    "source_glyph_mask_consumed_by_renderer",
    "cleanup_allowed_area",
    "erase_mask_method",
    "erase_mask_pixels",
    "erase_mask_bbox",
    "erase_mask_growth_ratio",
    "erase_mask_allowed_area",
    "erase_mask_rejected_reason",
    "erase_mask_artifact_risk",
    "erase_mask_visual_scope",
    "source_glyph_erasure_bbox",
    "source_glyph_erasure_expected_area_bbox",
    "source_glyph_erasure_expected_pixels",
    "source_glyph_erasure_coverage_ratio",
    "cleanup_covers_source_glyphs",
)


@dataclass
class SourceGlyphMask:
    page_id: str
    region_id: str
    logical_block_id: str | None
    text_area_container_id: str | None
    container_type: str | None
    route_intent: str | None
    mask_bbox: list[int] | None
    cleanup_allowed_area: list[int] | None
    source_glyph_erasure_bbox: list[int] | None
    source_glyph_erasure_coverage_ratio: float | None
    cleanup_covers_source_glyphs: bool | None
    cleanup_visual_artifact_risk: bool | None
    generation_method: str
    source_glyph_erasure_expected_area_bbox: list[int] | None = None
    source_glyph_erasure_expected_pixels: int | None = None
    text_block_root_id: str | None = None
    parent_logical_text_unit_id: str | None = None
    parent_execution_bundle_id: str | None = None
    anchor_child_id: str | None = None
    child_segment_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    mask_id: str | None = None
    cleanup_mode: str | None = None
    cleanup_source_erasure_failure_reason: str | None = None
    cleanup_source_tracking_required: bool = False
    source_glyph_mask_generation_required: bool = False
    source_glyph_mask_generation_status: str | None = None
    source_glyph_mask_not_generated_reason: str | None = None
    source_glyph_mask_review_only: bool = False
    source_glyph_mask_required: bool = False
    source_glyph_mask_generated: bool = False
    source_glyph_mask_consumed_by_renderer: bool = False
    source_glyph_mask_missing_reason: str | None = None
    source_glyph_mask_fallback_used: bool = False
    source_glyph_mask_fallback_reason: str | None = None
    represented_child_cleanup_required: bool | None = None
    represented_child_cleanup_mask_generated: bool | None = None
    represented_child_cleanup_mask_consumed: bool | None = None
    represented_child_cleanup_coverage_ratio: float | None = None
    represented_child_cleanup_failure_reason: str | None = None
    source_glyph_mask_expected_bbox: list[int] | None = None
    source_glyph_mask_actual_bbox: list[int] | None = None
    source_glyph_mask_expected_overlap_ratio: float | None = None
    source_glyph_mask_container_clip_reason: str | None = None
    caption_background_mask_generation_method: str | None = None
    caption_background_mask_coverage_ratio: float | None = None
    side_caption_mask_generation_method: str | None = None
    side_caption_mask_coverage_ratio: float | None = None
    phase2c_mask_adjustment_reason: str | None = None
    route_owned_foreground_contract_status: str | None = None
    foreground_mask_method: str | None = None
    foreground_mask_pixels: int | None = None
    foreground_mask_bbox: list[int] | None = None
    erase_mask_method: str | None = None
    erase_mask_pixels: int | None = None
    erase_mask_bbox: list[int] | None = None
    erase_mask_growth_ratio: float | None = None
    erase_mask_allowed_area: list[int] | None = None
    erase_mask_rejected_reason: str | None = None
    erase_mask_artifact_risk: bool | None = None
    erase_mask_visual_scope: str | None = None
    text_instance_ids: list[str] = field(default_factory=list)
    mask_ref: str | None = None
    bbox: list[int] | None = None
    coverage_ratio: float | None = None
    allowed_area_ref: list[int] | None = None
    allowed_area_validity: str | None = None
    class_specific_contract: str | None = None
    quality_status: str | None = None
    cleanup_suitability_status: str | None = None
    cleanup_suitability_reason: str | None = None
    compatibility_source_fields: dict[str, Any] = field(default_factory=dict)
    mask: Any = field(default=None, repr=False, compare=False)
    foreground_mask: Any = field(default=None, repr=False, compare=False)
    erase_mask: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "source_glyph_contract_version": SOURCE_GLYPH_PHASE4_CONTRACT_VERSION,
            "source_glyph_mask_id": self.mask_id,
            "text_instance_ids": list(self.text_instance_ids),
            "mask_ref": self.mask_ref or self.mask_id,
            "bbox": self.bbox,
            "coverage_ratio": self.coverage_ratio,
            "allowed_area_ref": self.allowed_area_ref,
            "allowed_area_validity": self.allowed_area_validity,
            "class_specific_contract": self.class_specific_contract,
            "quality_status": self.quality_status,
            "cleanup_suitability_status": self.cleanup_suitability_status,
            "cleanup_suitability_reason": self.cleanup_suitability_reason,
            "compatibility_source_fields": dict(self.compatibility_source_fields),
            "source_glyph_mask_anchor_block_id": self.logical_block_id,
            "parent_execution_bundle_id": self.parent_execution_bundle_id,
            "parent_logical_text_unit_id": self.parent_logical_text_unit_id,
            "text_block_root_id": self.text_block_root_id,
            "source_glyph_mask_parent_logical_text_unit_id": self.parent_logical_text_unit_id,
            "source_glyph_mask_text_block_root_id": self.text_block_root_id,
            "source_glyph_mask_anchor_child_id": self.anchor_child_id,
            "source_glyph_mask_child_segment_ids": list(self.child_segment_ids),
            "page_id": self.page_id,
            "region_id": self.region_id,
            "logical_block_id": self.logical_block_id,
            "text_area_container_id": self.text_area_container_id,
            "container_type": self.container_type,
            "route_intent": self.route_intent,
            "mask_bbox": self.mask_bbox,
            "cleanup_allowed_area": self.cleanup_allowed_area,
            "source_glyph_erasure_bbox": self.source_glyph_erasure_bbox,
            "source_glyph_erasure_expected_area_bbox": self.source_glyph_erasure_expected_area_bbox,
            "source_glyph_erasure_expected_pixels": self.source_glyph_erasure_expected_pixels,
            "source_glyph_erasure_coverage_ratio": self.source_glyph_erasure_coverage_ratio,
            "cleanup_covers_source_glyphs": self.cleanup_covers_source_glyphs,
            "cleanup_visual_artifact_risk": self.cleanup_visual_artifact_risk,
            "generation_method": self.generation_method,
            "failure_reason": self.failure_reason,
            "cleanup_mode": self.cleanup_mode,
            "cleanup_source_erasure_failure_reason": self.cleanup_source_erasure_failure_reason,
            "cleanup_source_tracking_required": self.cleanup_source_tracking_required,
            "source_glyph_mask_generation_required": self.source_glyph_mask_generation_required,
            "source_glyph_mask_generation_status": self.source_glyph_mask_generation_status,
            "source_glyph_mask_not_generated_reason": self.source_glyph_mask_not_generated_reason,
            "source_glyph_mask_review_only": self.source_glyph_mask_review_only,
            "source_glyph_mask_required": self.source_glyph_mask_required,
            "source_glyph_mask_generated": self.source_glyph_mask_generated,
            "source_glyph_mask_consumed_by_renderer": self.source_glyph_mask_consumed_by_renderer,
            "source_glyph_mask_missing_reason": self.source_glyph_mask_missing_reason,
            "source_glyph_mask_fallback_used": self.source_glyph_mask_fallback_used,
            "source_glyph_mask_fallback_reason": self.source_glyph_mask_fallback_reason,
            "represented_child_cleanup_required": self.represented_child_cleanup_required,
            "represented_child_cleanup_mask_generated": self.represented_child_cleanup_mask_generated,
            "represented_child_cleanup_mask_consumed": self.represented_child_cleanup_mask_consumed,
            "represented_child_cleanup_coverage_ratio": self.represented_child_cleanup_coverage_ratio,
            "represented_child_cleanup_failure_reason": self.represented_child_cleanup_failure_reason,
            "source_glyph_mask_expected_bbox": self.source_glyph_mask_expected_bbox,
            "source_glyph_mask_actual_bbox": self.source_glyph_mask_actual_bbox,
            "source_glyph_mask_expected_overlap_ratio": self.source_glyph_mask_expected_overlap_ratio,
            "source_glyph_mask_container_clip_reason": self.source_glyph_mask_container_clip_reason,
            "caption_background_mask_generation_method": self.caption_background_mask_generation_method,
            "caption_background_mask_coverage_ratio": self.caption_background_mask_coverage_ratio,
            "side_caption_mask_generation_method": self.side_caption_mask_generation_method,
            "side_caption_mask_coverage_ratio": self.side_caption_mask_coverage_ratio,
            "phase2c_mask_adjustment_reason": self.phase2c_mask_adjustment_reason,
            "route_owned_foreground_contract_status": self.route_owned_foreground_contract_status,
            "foreground_mask_method": self.foreground_mask_method,
            "foreground_mask_pixels": self.foreground_mask_pixels,
            "foreground_mask_bbox": self.foreground_mask_bbox,
            "erase_mask_method": self.erase_mask_method,
            "erase_mask_pixels": self.erase_mask_pixels,
            "erase_mask_bbox": self.erase_mask_bbox,
            "erase_mask_growth_ratio": self.erase_mask_growth_ratio,
            "erase_mask_allowed_area": self.erase_mask_allowed_area,
            "erase_mask_rejected_reason": self.erase_mask_rejected_reason,
            "erase_mask_artifact_risk": self.erase_mask_artifact_risk,
            "erase_mask_visual_scope": self.erase_mask_visual_scope,
        }

    def render_audit_fields(self) -> dict[str, Any]:
        return {
            "source_glyph_contract_version": SOURCE_GLYPH_PHASE4_CONTRACT_VERSION,
            "source_glyph_mask_id": self.mask_id,
            "text_instance_ids": list(self.text_instance_ids),
            "mask_ref": self.mask_ref or self.mask_id,
            "bbox": self.bbox,
            "coverage_ratio": self.coverage_ratio,
            "allowed_area_ref": self.allowed_area_ref,
            "allowed_area_validity": self.allowed_area_validity,
            "class_specific_contract": self.class_specific_contract,
            "quality_status": self.quality_status,
            "cleanup_suitability_status": self.cleanup_suitability_status,
            "cleanup_suitability_reason": self.cleanup_suitability_reason,
            "compatibility_source_fields": dict(self.compatibility_source_fields),
            "source_glyph_mask_anchor_block_id": self.logical_block_id,
            "parent_execution_bundle_id": self.parent_execution_bundle_id,
            "source_glyph_mask_parent_logical_text_unit_id": self.parent_logical_text_unit_id,
            "source_glyph_mask_text_block_root_id": self.text_block_root_id,
            "source_glyph_mask_anchor_child_id": self.anchor_child_id,
            "source_glyph_mask_child_segment_ids": list(self.child_segment_ids),
            "source_glyph_mask_generation_method": self.generation_method,
            "source_glyph_mask_failure_reason": self.failure_reason,
            "cleanup_source_tracking_required": self.cleanup_source_tracking_required,
            "source_glyph_mask_generation_required": self.source_glyph_mask_generation_required,
            "source_glyph_mask_generation_status": self.source_glyph_mask_generation_status,
            "source_glyph_mask_not_generated_reason": self.source_glyph_mask_not_generated_reason,
            "source_glyph_mask_review_only": self.source_glyph_mask_review_only,
            "source_glyph_mask_required": self.source_glyph_mask_required,
            "source_glyph_mask_generated": self.source_glyph_mask_generated,
            "source_glyph_mask_consumed_by_renderer": self.source_glyph_mask_consumed_by_renderer,
            "source_glyph_mask_missing_reason": self.source_glyph_mask_missing_reason,
            "source_glyph_mask_fallback_used": self.source_glyph_mask_fallback_used,
            "source_glyph_mask_fallback_reason": self.source_glyph_mask_fallback_reason,
            "represented_child_cleanup_required": self.represented_child_cleanup_required,
            "represented_child_cleanup_mask_generated": self.represented_child_cleanup_mask_generated,
            "represented_child_cleanup_mask_consumed": self.represented_child_cleanup_mask_consumed,
            "represented_child_cleanup_coverage_ratio": self.represented_child_cleanup_coverage_ratio,
            "represented_child_cleanup_failure_reason": self.represented_child_cleanup_failure_reason,
            "source_glyph_mask_expected_bbox": self.source_glyph_mask_expected_bbox,
            "source_glyph_mask_actual_bbox": self.source_glyph_mask_actual_bbox,
            "source_glyph_mask_expected_overlap_ratio": self.source_glyph_mask_expected_overlap_ratio,
            "source_glyph_mask_container_clip_reason": self.source_glyph_mask_container_clip_reason,
            "caption_background_mask_generation_method": self.caption_background_mask_generation_method,
            "caption_background_mask_coverage_ratio": self.caption_background_mask_coverage_ratio,
            "side_caption_mask_generation_method": self.side_caption_mask_generation_method,
            "side_caption_mask_coverage_ratio": self.side_caption_mask_coverage_ratio,
            "phase2c_mask_adjustment_reason": self.phase2c_mask_adjustment_reason,
            "route_owned_foreground_contract_status": self.route_owned_foreground_contract_status,
            "foreground_mask_method": self.foreground_mask_method,
            "foreground_mask_pixels": self.foreground_mask_pixels,
            "foreground_mask_bbox": self.foreground_mask_bbox,
            "erase_mask_method": self.erase_mask_method,
            "erase_mask_pixels": self.erase_mask_pixels,
            "erase_mask_bbox": self.erase_mask_bbox,
            "erase_mask_growth_ratio": self.erase_mask_growth_ratio,
            "erase_mask_allowed_area": self.erase_mask_allowed_area,
            "erase_mask_rejected_reason": self.erase_mask_rejected_reason,
            "erase_mask_artifact_risk": self.erase_mask_artifact_risk,
            "erase_mask_visual_scope": self.erase_mask_visual_scope,
            "source_glyph_erasure_bbox": self.source_glyph_erasure_bbox,
            "source_glyph_erasure_expected_area_bbox": self.source_glyph_erasure_expected_area_bbox,
            "source_glyph_erasure_expected_pixels": self.source_glyph_erasure_expected_pixels,
            "source_glyph_erasure_coverage_ratio": self.source_glyph_erasure_coverage_ratio,
            "cleanup_covers_source_glyphs": self.cleanup_covers_source_glyphs,
            "cleanup_source_erasure_failure_reason": self.cleanup_source_erasure_failure_reason,
            "cleanup_visual_artifact_risk": self.cleanup_visual_artifact_risk,
        }


@dataclass
class SourceGlyphMaskResult:
    page_id: str
    version: str
    generated: bool
    runtime_sec: float
    masks_by_region: dict[str, SourceGlyphMask] = field(default_factory=dict)
    coverage_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def get(self, region_id: str) -> SourceGlyphMask | None:
        return self.masks_by_region.get(str(region_id or ""))

    def to_audit_dict(self) -> dict[str, Any]:
        masks = [mask.to_audit_dict() for mask in self.masks_by_region.values()]
        by_method: dict[str, int] = {}
        for mask in self.masks_by_region.values():
            by_method[mask.generation_method] = by_method.get(mask.generation_method, 0) + 1
        tracking_required = [record for record in self.coverage_records if record.get("cleanup_source_tracking_required")]
        generation_required = [record for record in self.coverage_records if record.get("source_glyph_mask_generation_required")]
        required = [record for record in self.coverage_records if record.get("source_glyph_mask_required")]
        generated = [record for record in self.coverage_records if record.get("source_glyph_mask_generated")]
        consumed = [record for record in self.coverage_records if record.get("source_glyph_mask_consumed_by_renderer")]
        review_only = [record for record in self.coverage_records if record.get("source_glyph_mask_review_only")]
        missing_reasons: dict[str, int] = {}
        not_generated_reasons: dict[str, int] = {}
        fallback_reasons: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {name: 0 for name in SOURCE_GLYPH_CLASS_VOCABULARY}
        allowed_area_counts: dict[str, int] = {}
        quality_counts: dict[str, int] = {}
        suitability_counts: dict[str, int] = {}
        for record in self.coverage_records:
            status = str(record.get("source_glyph_mask_generation_status") or "")
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1
            class_name = str(record.get("class_specific_contract") or "")
            if class_name:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            allowed_area_status = str(record.get("allowed_area_validity") or "")
            if allowed_area_status:
                allowed_area_counts[allowed_area_status] = allowed_area_counts.get(allowed_area_status, 0) + 1
            quality_status = str(record.get("quality_status") or "")
            if quality_status:
                quality_counts[quality_status] = quality_counts.get(quality_status, 0) + 1
            suitability_status = str(record.get("cleanup_suitability_status") or "")
            if suitability_status:
                suitability_counts[suitability_status] = suitability_counts.get(suitability_status, 0) + 1
            reason = str(record.get("source_glyph_mask_missing_reason") or "")
            if reason:
                missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
            not_generated_reason = str(record.get("source_glyph_mask_not_generated_reason") or "")
            if not_generated_reason:
                not_generated_reasons[not_generated_reason] = not_generated_reasons.get(not_generated_reason, 0) + 1
            fallback = str(record.get("source_glyph_mask_fallback_reason") or "")
            if fallback:
                fallback_reasons[fallback] = fallback_reasons.get(fallback, 0) + 1
        return {
            "source_glyph_mask_version": self.version,
            "source_glyph_contract_version": SOURCE_GLYPH_PHASE4_CONTRACT_VERSION,
            "source_glyph_required_class_vocabulary": list(SOURCE_GLYPH_CLASS_VOCABULARY),
            "source_glyph_mask_generated": self.generated,
            "source_glyph_mask_runtime_sec": round(float(self.runtime_sec or 0.0), 4),
            "source_glyph_mask_count": len(masks),
            "source_glyph_mask_counts_by_method": by_method,
            "source_glyph_mask_class_counts": class_counts,
            "source_glyph_mask_allowed_area_validity_counts": allowed_area_counts,
            "source_glyph_mask_quality_status_counts": quality_counts,
            "source_glyph_mask_cleanup_suitability_status_counts": suitability_counts,
            "cleanup_source_tracking_required_count": len(tracking_required),
            "source_glyph_mask_generation_required_count": len(generation_required),
            "source_glyph_mask_required_count": len(required),
            "source_glyph_mask_generated_count": len(generated),
            "source_glyph_mask_consumed_count": len(consumed),
            "source_glyph_mask_review_only_count": len(review_only),
            "source_glyph_mask_generation_status_counts": status_counts,
            "source_glyph_mask_missing_reason_counts": missing_reasons,
            "source_glyph_mask_not_generated_reason_counts": not_generated_reasons,
            "source_glyph_mask_fallback_reason_counts": fallback_reasons,
            "source_glyph_mask_errors": list(self.errors),
            "source_glyph_masks": masks,
            "source_glyph_mask_coverage_records": list(self.coverage_records),
        }

    def region_audit_fields(self) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        for record in self.coverage_records:
            rid = str(record.get("region_id") or "")
            if rid:
                fields[rid] = dict(record)
        for rid, mask in self.masks_by_region.items():
            fields.setdefault(rid, {}).update(mask.render_audit_fields())
        return fields


def generate_source_glyph_masks(
    *,
    page_id: str,
    image_path: str,
    regions: list[dict[str, Any]],
) -> SourceGlyphMaskResult:
    start = time.time()
    errors: list[str] = []
    masks_by_region: dict[str, SourceGlyphMask] = {}
    if cv2 is None or np is None or Image is None:
        return SourceGlyphMaskResult(
            page_id=page_id,
            version=SOURCE_GLYPH_MASK_VERSION,
            generated=False,
            runtime_sec=time.time() - start,
            errors=["missing_cv2_numpy_or_pillow"],
        )
    if not image_path or not os.path.isfile(image_path):
        return SourceGlyphMaskResult(
            page_id=page_id,
            version=SOURCE_GLYPH_MASK_VERSION,
            generated=False,
            runtime_sec=time.time() - start,
            errors=["missing_image_path"],
        )
    try:
        with Image.open(image_path) as image:
            img_np = np.array(image.convert("RGB"))
    except Exception as exc:
        return SourceGlyphMaskResult(
            page_id=page_id,
            version=SOURCE_GLYPH_MASK_VERSION,
            generated=False,
            runtime_sec=time.time() - start,
            errors=[f"image_load_failed:{type(exc).__name__}:{exc}"],
        )

    img_h, img_w = img_np.shape[:2]
    regions_by_id = {
        str(region.get("region_id", "") or ""): region
        for region in regions
        if isinstance(region, dict) and str(region.get("region_id", "") or "")
    }
    active_parent_ids = _active_translation_parent_ids(regions)
    coverage_records: list[dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("region_id", "") or "")
        if not region_id:
            continue
        render = region.get("render") or {}
        if not isinstance(render, dict):
            render = {}
        bbox = region.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        polygon = region.get("polygon")
        cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
        if not cleanup_mode:
            cleanup_mode = "background_box" if _is_background_region(region) else "bubble"
        coverage = _base_coverage_record(page_id, region, render, cleanup_mode)
        coverage["source_glyph_mask_requirement_status"] = "not_required"

        try:
            _x, _y, w, h = [int(round(float(v))) for v in bbox[:4]]
        except Exception:
            coverage["source_glyph_mask_generation_status"] = "invalid_bbox"
            coverage["source_glyph_mask_missing_reason"] = None
            coverage["source_glyph_mask_not_generated_reason"] = "invalid_bbox"
            coverage.update(
                _source_glyph_phase4_contract_fields(
                    page_id=page_id,
                    region=region,
                    render=render,
                    cleanup_mode=cleanup_mode,
                    method=None,
                    generated=False,
                    required=False,
                    review_only=False,
                    consumed_by_renderer=False,
                    failure_reason="invalid_bbox",
                )
            )
            coverage_records.append(coverage)
            continue

        try:
            source_mask = None
            method = None
            expected_box = None
            allowed_area = None
            consumed_by_renderer = False
            cleanup_source_tracking_required = bool(coverage.get("cleanup_source_tracking_required"))
            generation_required = False
            review_only = False
            missing_reason = ""
            not_generated_reason = str(coverage.get("source_glyph_mask_not_generated_reason") or "")
            is_represented_child_mask = False
            is_caption_mask = False
            is_side_caption_mask = False
            container_clip_reason = None
            if (
                _is_recovered_logical_speech_anchor(region, render)
                and cleanup_mode in {"bubble", "speech_bubble", "speech_strong"}
            ):
                recovered_dilate = max(3, min(8, int(max(min(w, h) * 0.10, h * 0.025))))
                source_mask = _recovered_speech_source_cleanup_mask(
                    img_np,
                    region,
                    render,
                    regions_by_id,
                    bbox,
                    polygon,
                    dilate_px=max(2, min(4, recovered_dilate)),
                )
                method = "speech_recovered_anchor_glyph_local"
                expected_box = _source_erasure_expected_box(
                    region,
                    render,
                    regions_by_id,
                    bbox,
                    img_w,
                    img_h,
                )
                allowed_area = _recovered_speech_allowed_area(region, render, img_w, img_h)
                consumed_by_renderer = True
                generation_required = True
                not_generated_reason = ""
            elif _is_route_owned_cleanup_only_anchor(region, render):
                route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
                is_caption = route in {"translate_caption", "translate_caption_background"} or _is_background_region(region)
                limit_box = _xywh_value_to_xyxy(
                    _canonical_region_render_value(region, render, "text_area_container_bbox"),
                    img_w,
                    img_h,
                )
                if is_caption:
                    source_mask = _caption_background_source_cleanup_mask(
                        img_np,
                        bbox,
                        None,
                        dilate_px=2,
                        limit_box=limit_box,
                    )
                    method = "caption_background_cleanup_only_glyph_local"
                else:
                    cleanup_only_dilate = max(2, min(4, int(max(min(w, h) * 0.08, h * 0.02))))
                    source_mask = _source_glyph_local_mask(
                        img_np,
                        bbox,
                        polygon,
                        dilate_px=cleanup_only_dilate,
                        limit_box=limit_box,
                        bright_context_only=True,
                    )
                    method = "speech_cleanup_only_glyph_local"
                expected_box = _source_erasure_expected_box(
                    region,
                    render,
                    regions_by_id,
                    bbox,
                    img_w,
                    img_h,
                )
                allowed_area = _recovered_speech_allowed_area(region, render, img_w, img_h)
                consumed_by_renderer = True
                generation_required = True
                not_generated_reason = ""
                is_caption_mask = is_caption
                is_side_caption_mask = bool(is_caption and _is_vertical_side_caption(region, render))
                container_clip_reason = _container_clip_reason(expected_box, allowed_area)
            elif _is_logical_text_cleanup_anchor(region, render):
                route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
                is_caption = route in {"translate_caption", "translate_caption_background"} or _is_background_region(region)
                is_side_caption_region = bool(is_caption and _is_vertical_side_caption(region, render))
                logical_dilate = 1 if is_caption else max(2, min(4, int(max(min(w, h) * 0.08, h * 0.02))))
                if is_side_caption_region:
                    source_mask = _caption_background_source_cleanup_mask(
                        img_np,
                        bbox,
                        None,
                        dilate_px=2,
                        limit_box=_xywh_value_to_xyxy(
                            _canonical_region_render_value(region, render, "text_area_container_bbox"),
                            img_w,
                            img_h,
                        ),
                    )
                elif is_caption:
                    source_mask = _caption_background_logical_source_cleanup_mask(
                        img_np,
                        region,
                        render,
                        regions_by_id,
                        bbox,
                        polygon,
                        dilate_px=max(1, logical_dilate),
                    )
                else:
                    source_mask = _logical_text_source_cleanup_mask(
                        img_np,
                        region,
                        render,
                        regions_by_id,
                        bbox,
                        polygon,
                        dilate_px=logical_dilate,
                        bright_context_only=True,
                    )
                method = (
                    "side_caption_glyph_local"
                    if is_side_caption_region
                    else "caption_background_logical_block_glyph_local"
                    if is_caption
                    else "logical_text_block_member_glyph_local"
                )
                expected_box = _source_erasure_expected_box(
                    region,
                    render,
                    regions_by_id,
                    bbox,
                    img_w,
                    img_h,
                )
                allowed_area = _recovered_speech_allowed_area(region, render, img_w, img_h)
                consumed_by_renderer = True
                generation_required = True
                not_generated_reason = ""
                is_caption_mask = is_caption
                is_side_caption_mask = is_side_caption_region
                container_clip_reason = _container_clip_reason(expected_box, allowed_area)
            elif _should_generate_side_caption_mask(region, render, bbox, img_np):
                limit_box = _xywh_value_to_xyxy(
                    _canonical_region_render_value(region, render, "text_area_container_bbox"),
                    img_w,
                    img_h,
                )
                source_mask = _caption_background_source_cleanup_mask(
                    img_np,
                    bbox,
                    None,
                    dilate_px=2,
                    limit_box=limit_box,
                )
                method = "side_caption_glyph_local"
                expected_box = _xywh_value_to_xyxy(bbox, img_w, img_h)
                allowed_area = limit_box
                consumed_by_renderer = True
                generation_required = True
                not_generated_reason = ""
                is_caption_mask = True
                is_side_caption_mask = True
                container_clip_reason = _container_clip_reason(expected_box, allowed_area)
            elif _should_generate_represented_child_mask(region, render, active_parent_ids):
                limit_box = _represented_child_cleanup_limit_box(region, img_w, img_h)
                source_mask, method, missing_reason = _represented_child_source_cleanup_mask(
                    img_np,
                    region,
                    render,
                    bbox,
                    polygon,
                    img_w,
                    img_h,
                    limit_box=limit_box,
                )
                expected_box = _xywh_value_to_xyxy(bbox, img_w, img_h)
                allowed_area = limit_box
                consumed_by_renderer = True
                generation_required = True
                not_generated_reason = missing_reason or ""
                is_represented_child_mask = True
                container_clip_reason = _container_clip_reason(expected_box, allowed_area)
            elif _should_generate_speech_review_mask(region, render, bbox, cleanup_mode, img_np):
                limit_box = _xywh_value_to_xyxy(
                    _canonical_region_render_value(region, render, "text_area_container_bbox"),
                    img_w,
                    img_h,
                )
                source_mask = _source_glyph_local_mask(
                    img_np,
                    bbox,
                    polygon,
                    dilate_px=1,
                    limit_box=limit_box,
                    bright_context_only=True,
                )
                method = "speech_cleanup_glyph_local_review"
                expected_box = _xywh_value_to_xyxy(bbox, img_w, img_h)
                allowed_area = limit_box
                consumed_by_renderer = False
                review_only = True
                not_generated_reason = ""
            if not method:
                reason = not_generated_reason or _default_missing_reason(region, render, cleanup_mode)
                coverage["source_glyph_mask_required"] = False
                coverage["source_glyph_mask_generation_required"] = False
                coverage["source_glyph_mask_review_only"] = False
                coverage["source_glyph_mask_requirement_status"] = "not_required"
                coverage["source_glyph_mask_generation_status"] = (
                    "not_applicable_tracking_only" if cleanup_source_tracking_required else "not_applicable"
                )
                coverage["source_glyph_mask_missing_reason"] = None
                coverage["source_glyph_mask_not_generated_reason"] = reason
                coverage.update(
                    _source_glyph_phase4_contract_fields(
                        page_id=page_id,
                        region=region,
                        render=render,
                        cleanup_mode=cleanup_mode,
                        method=None,
                        generated=False,
                        required=False,
                        review_only=False,
                        consumed_by_renderer=False,
                        failure_reason=reason,
                    )
                )
                coverage_records.append(coverage)
                continue
            if source_mask is None:
                reason = f"{method}_mask_not_generated"
                if generation_required:
                    errors.append(f"{region_id}:{method}:mask_not_generated")
                coverage["source_glyph_mask_required"] = generation_required
                coverage["source_glyph_mask_generation_required"] = generation_required
                coverage["source_glyph_mask_review_only"] = review_only
                coverage["source_glyph_mask_requirement_status"] = (
                    "required_failed" if generation_required else "review_only_non_rendering"
                )
                coverage["source_glyph_mask_generated"] = False
                coverage["source_glyph_mask_generation_method"] = method
                coverage["source_glyph_mask_generation_status"] = "failed" if generation_required else "review_not_generated"
                coverage["source_glyph_mask_missing_reason"] = reason if generation_required else None
                coverage["source_glyph_mask_not_generated_reason"] = reason
                coverage.update(
                    _source_glyph_phase4_contract_fields(
                        page_id=page_id,
                        region=region,
                        render=render,
                        cleanup_mode=cleanup_mode,
                        method=method,
                        generated=False,
                        required=generation_required,
                        review_only=review_only,
                        consumed_by_renderer=False,
                        failure_reason=reason,
                        allowed_area=allowed_area,
                        container_clip_reason=container_clip_reason,
                    )
                )
                coverage_records.append(coverage)
                continue
            stats = mask_stats(source_mask)
            mask_bbox = _mask_stats_box(stats)
            expected_mask = None
            if is_caption_mask and expected_box is not None:
                expected_mask = _caption_background_source_cleanup_mask(
                    img_np,
                    [expected_box[0], expected_box[1], expected_box[2] - expected_box[0], expected_box[3] - expected_box[1]],
                    None,
                    dilate_px=1,
                    limit_box=allowed_area or expected_box,
                )
            audit = _source_glyph_erasure_audit_fields(
                img_np,
                source_mask,
                expected_box,
                expected_source_mask=expected_mask,
            )
            if method == "speech_cleanup_glyph_local_review" and not _review_mask_quality_passes(
                source_mask,
                audit,
                expected_box,
                img_w,
                img_h,
            ):
                reason = (
                    audit.get("cleanup_source_erasure_failure_reason")
                    or "review_mask_quality_failed"
                )
                required_failure = _rendered_region_requires_source_mask(region, render, cleanup_mode)
                coverage["source_glyph_mask_required"] = bool(required_failure)
                coverage["source_glyph_mask_generation_required"] = bool(required_failure)
                coverage["source_glyph_mask_review_only"] = not required_failure
                coverage["source_glyph_mask_requirement_status"] = (
                    "required_failed" if required_failure else "review_only_non_rendering"
                )
                coverage["source_glyph_mask_generated"] = False
                coverage["source_glyph_mask_generation_method"] = method
                coverage["source_glyph_mask_generation_status"] = (
                    "required_quality_failed" if required_failure else "review_quality_failed"
                )
                coverage["source_glyph_mask_missing_reason"] = reason if required_failure else None
                coverage["source_glyph_mask_not_generated_reason"] = reason
                coverage["cleanup_source_erasure_failure_reason"] = reason if required_failure else None
                coverage["source_glyph_erasure_bbox"] = audit.get("source_glyph_erasure_bbox")
                coverage["source_glyph_erasure_coverage_ratio"] = audit.get("source_glyph_erasure_coverage_ratio")
                coverage["cleanup_covers_source_glyphs"] = audit.get("cleanup_covers_source_glyphs")
                coverage["cleanup_visual_artifact_risk"] = audit.get("cleanup_visual_artifact_risk")
                coverage.update(
                    _source_glyph_phase4_contract_fields(
                        page_id=page_id,
                        region=region,
                        render=render,
                        cleanup_mode=cleanup_mode,
                        method=method,
                        mask_bbox=mask_bbox,
                        allowed_area=allowed_area,
                        audit=audit,
                        generated=False,
                        required=bool(required_failure),
                        review_only=not required_failure,
                        consumed_by_renderer=False,
                        failure_reason=reason,
                        container_clip_reason=container_clip_reason,
                    )
                )
                coverage_records.append(coverage)
                continue
            foreground_contract = _route_owned_foreground_contract(
                img_np,
                region,
                render,
                regions_by_id,
                bbox,
                polygon,
                allowed_area,
                expected_box,
                is_caption_mask=is_caption_mask,
                is_side_caption_mask=is_side_caption_mask,
            )
            foreground_contract_fields = foreground_contract.get("fields", {})
            mask_id = f"sgm_{page_id}_{region_id}_{method}"
            phase4_audit = dict(audit)
            if isinstance(foreground_contract_fields, dict):
                phase4_audit.update(foreground_contract_fields)
            phase4_fields = _source_glyph_phase4_contract_fields(
                page_id=page_id,
                region=region,
                render=render,
                cleanup_mode=cleanup_mode,
                method=method,
                mask_id=mask_id,
                mask_bbox=mask_bbox,
                allowed_area=allowed_area,
                audit=phase4_audit,
                generated=True,
                required=generation_required,
                review_only=review_only and not consumed_by_renderer,
                consumed_by_renderer=consumed_by_renderer,
                failure_reason=None,
                container_clip_reason=container_clip_reason,
            )
            mask_record = SourceGlyphMask(
                page_id=page_id,
                region_id=region_id,
                logical_block_id=str(_canonical_region_render_value(region, render, "logical_text_block_id") or "") or None,
                text_area_container_id=str(_canonical_region_render_value(region, render, "text_area_container_id") or "") or None,
                container_type=str(_canonical_region_render_value(region, render, "text_area_container_type") or region.get("type") or "") or None,
                route_intent=str(_canonical_region_render_value(region, render, "text_area_route_intent") or "") or None,
                mask_bbox=list(mask_bbox) if mask_bbox else None,
                cleanup_allowed_area=list(allowed_area) if allowed_area else None,
                source_glyph_erasure_bbox=audit.get("source_glyph_erasure_bbox"),
                source_glyph_erasure_expected_area_bbox=audit.get("source_glyph_erasure_expected_area_bbox"),
                source_glyph_erasure_expected_pixels=audit.get("source_glyph_erasure_expected_pixels"),
                source_glyph_erasure_coverage_ratio=audit.get("source_glyph_erasure_coverage_ratio"),
                cleanup_covers_source_glyphs=audit.get("cleanup_covers_source_glyphs"),
                cleanup_visual_artifact_risk=audit.get("cleanup_visual_artifact_risk"),
                generation_method=method,
                text_block_root_id=str(_canonical_region_render_value(region, render, "text_block_root_id") or "") or None,
                parent_logical_text_unit_id=str(_canonical_region_render_value(region, render, "parent_logical_text_unit_id") or "") or None,
                parent_execution_bundle_id=str(
                    region.get("parent_execution_bundle_id")
                    or render.get("parent_execution_bundle_id")
                    or region_id
                    or ""
                ) or None,
                anchor_child_id=str(
                    _canonical_region_render_value(region, render, "parent_logical_text_unit_anchor_child_id")
                    or _canonical_region_render_value(region, render, "child_recognized_text_segment_id")
                    or ""
                ) or None,
                child_segment_ids=[
                    str(value)
                    for value in (_canonical_region_render_list(region, render, "parent_logical_text_unit_child_segment_ids") or [])
                    if str(value).strip()
                ],
                failure_reason=phase4_fields["failure_reason"],
                mask_id=mask_id,
                cleanup_mode=method,
                cleanup_source_erasure_failure_reason=audit.get("cleanup_source_erasure_failure_reason"),
                cleanup_source_tracking_required=cleanup_source_tracking_required,
                source_glyph_mask_generation_required=generation_required,
                source_glyph_mask_generation_status=(
                    "generated_consumed" if consumed_by_renderer else "generated_review_only"
                ),
                source_glyph_mask_not_generated_reason=None,
                source_glyph_mask_review_only=review_only and not consumed_by_renderer,
                source_glyph_mask_required=generation_required,
                source_glyph_mask_generated=True,
                source_glyph_mask_consumed_by_renderer=consumed_by_renderer,
                source_glyph_mask_missing_reason=None,
                source_glyph_mask_fallback_used=False,
                source_glyph_mask_fallback_reason=None,
                represented_child_cleanup_required=is_represented_child_mask,
                represented_child_cleanup_mask_generated=True if is_represented_child_mask else None,
                represented_child_cleanup_mask_consumed=consumed_by_renderer if is_represented_child_mask else None,
                represented_child_cleanup_coverage_ratio=(
                    audit.get("source_glyph_erasure_coverage_ratio") if is_represented_child_mask else None
                ),
                represented_child_cleanup_failure_reason=(
                    audit.get("cleanup_source_erasure_failure_reason") if is_represented_child_mask else None
                ),
                source_glyph_mask_expected_bbox=audit.get("source_glyph_erasure_expected_area_bbox"),
                source_glyph_mask_actual_bbox=list(mask_bbox) if mask_bbox else None,
                source_glyph_mask_expected_overlap_ratio=audit.get("source_glyph_erasure_coverage_ratio"),
                source_glyph_mask_container_clip_reason=container_clip_reason,
                caption_background_mask_generation_method=method if is_caption_mask and not is_side_caption_mask else None,
                caption_background_mask_coverage_ratio=(
                    audit.get("source_glyph_erasure_coverage_ratio") if is_caption_mask and not is_side_caption_mask else None
                ),
                side_caption_mask_generation_method=method if is_side_caption_mask else None,
                side_caption_mask_coverage_ratio=(
                    audit.get("source_glyph_erasure_coverage_ratio") if is_side_caption_mask else None
                ),
                phase2c_mask_adjustment_reason=(
                    "caption_background_textlike_component_filter"
                    if is_caption_mask
                    else None
                ),
                route_owned_foreground_contract_status=foreground_contract_fields.get("route_owned_foreground_contract_status"),
                foreground_mask_method=foreground_contract_fields.get("foreground_mask_method"),
                foreground_mask_pixels=foreground_contract_fields.get("foreground_mask_pixels"),
                foreground_mask_bbox=foreground_contract_fields.get("foreground_mask_bbox"),
                erase_mask_method=foreground_contract_fields.get("erase_mask_method"),
                erase_mask_pixels=foreground_contract_fields.get("erase_mask_pixels"),
                erase_mask_bbox=foreground_contract_fields.get("erase_mask_bbox"),
                erase_mask_growth_ratio=foreground_contract_fields.get("erase_mask_growth_ratio"),
                erase_mask_allowed_area=foreground_contract_fields.get("erase_mask_allowed_area"),
                erase_mask_rejected_reason=foreground_contract_fields.get("erase_mask_rejected_reason"),
                erase_mask_artifact_risk=foreground_contract_fields.get("erase_mask_artifact_risk"),
                erase_mask_visual_scope=foreground_contract_fields.get("erase_mask_visual_scope"),
                text_instance_ids=phase4_fields["text_instance_ids"],
                mask_ref=phase4_fields["mask_ref"],
                bbox=phase4_fields["bbox"],
                coverage_ratio=phase4_fields["coverage_ratio"],
                allowed_area_ref=phase4_fields["allowed_area_ref"],
                allowed_area_validity=phase4_fields["allowed_area_validity"],
                class_specific_contract=phase4_fields["class_specific_contract"],
                quality_status=phase4_fields["quality_status"],
                cleanup_suitability_status=phase4_fields["cleanup_suitability_status"],
                cleanup_suitability_reason=phase4_fields["cleanup_suitability_reason"],
                compatibility_source_fields=phase4_fields["compatibility_source_fields"],
                mask=source_mask,
                foreground_mask=foreground_contract.get("foreground_mask"),
                erase_mask=foreground_contract.get("erase_mask"),
            )
            masks_by_region[region_id] = mask_record
            coverage.update(mask_record.render_audit_fields())
            coverage["source_glyph_mask_required"] = generation_required
            coverage["source_glyph_mask_generation_required"] = generation_required
            coverage["source_glyph_mask_generation_status"] = (
                "generated_consumed" if consumed_by_renderer else "generated_review_only"
            )
            coverage["source_glyph_mask_requirement_status"] = (
                "required_generated"
                if generation_required and consumed_by_renderer
                else "review_only_generated"
                if review_only
                else "generated_tracking"
            )
            coverage["source_glyph_mask_not_generated_reason"] = None
            coverage["source_glyph_mask_review_only"] = review_only and not consumed_by_renderer
            coverage["source_glyph_mask_generated"] = True
            coverage["source_glyph_mask_consumed_by_renderer"] = consumed_by_renderer
            coverage["source_glyph_mask_missing_reason"] = None
            coverage["source_glyph_mask_fallback_used"] = False
            coverage["source_glyph_mask_fallback_reason"] = None
            coverage_records.append(coverage)
        except Exception as exc:  # pragma: no cover - fail closed per region
            errors.append(f"{region_id}:{type(exc).__name__}:{exc}")
            reason = f"exception:{type(exc).__name__}"
            coverage["source_glyph_mask_generation_status"] = "failed_exception"
            coverage["source_glyph_mask_requirement_status"] = "required_failed"
            coverage["source_glyph_mask_missing_reason"] = reason
            coverage["source_glyph_mask_not_generated_reason"] = reason
            coverage.update(
                _source_glyph_phase4_contract_fields(
                    page_id=page_id,
                    region=region,
                    render=render,
                    cleanup_mode=cleanup_mode,
                    method=coverage.get("source_glyph_mask_generation_method"),
                    generated=False,
                    required=True,
                    review_only=False,
                    consumed_by_renderer=False,
                    failure_reason=reason,
                )
            )
            coverage_records.append(coverage)
            continue

    return SourceGlyphMaskResult(
        page_id=page_id,
        version=SOURCE_GLYPH_MASK_VERSION,
        generated=True,
        runtime_sec=time.time() - start,
        masks_by_region=masks_by_region,
        coverage_records=coverage_records,
        errors=errors,
    )


def generate_source_glyph_masks_for_parent_bundles(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: list[Any],
) -> SourceGlyphMaskResult:
    """Generate SourceGlyph masks from finalized parent execution bundles."""

    return generate_source_glyph_masks(
        page_id=page_id,
        image_path=image_path,
        regions=parent_execution_region_records(parent_execution_bundles),
    )


def mask_stats(mask) -> dict[str, Any] | None:
    if mask is None or np is None:
        return None
    try:
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


def build_source_glyph_local_mask(
    img_np,
    bbox,
    polygon=None,
    dilate_px: int = 2,
    limit_box=None,
    bright_context_only: bool = False,
):
    return _source_glyph_local_mask(
        img_np,
        bbox,
        polygon=polygon,
        dilate_px=dilate_px,
        limit_box=limit_box,
        bright_context_only=bright_context_only,
    )


def build_recovered_speech_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    dilate_px: int = 3,
):
    return _recovered_speech_source_cleanup_mask(
        img_np,
        region,
        render,
        regions_by_id,
        fallback_bbox,
        fallback_polygon,
        dilate_px=dilate_px,
    )


def source_erasure_expected_box(
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    img_w: int,
    img_h: int,
):
    return _source_erasure_expected_box(region, render, regions_by_id, fallback_bbox, img_w, img_h)


def source_glyph_erasure_audit_fields(img_np, cleanup_mask, expected_box) -> dict[str, Any]:
    return _source_glyph_erasure_audit_fields(img_np, cleanup_mask, expected_box)


def is_recovered_logical_speech_anchor(region: dict, render: dict) -> bool:
    return _is_recovered_logical_speech_anchor(region, render)


def is_logical_text_cleanup_anchor(region: dict, render: dict) -> bool:
    return _is_logical_text_cleanup_anchor(region, render)


def is_vertical_side_caption(region: dict, render: dict) -> bool:
    return _is_vertical_side_caption(region, render)


def xywh_value_to_xyxy(value, img_w: int, img_h: int):
    return _xywh_value_to_xyxy(value, img_w, img_h)


def _source_glyph_phase4_contract_fields(
    *,
    page_id: str | None = None,
    region: dict,
    render: dict,
    cleanup_mode: str,
    method: str | None,
    mask_id: str | None = None,
    mask_bbox=None,
    allowed_area=None,
    audit: dict[str, Any] | None = None,
    generated: bool = False,
    required: bool = False,
    review_only: bool = False,
    consumed_by_renderer: bool = False,
    failure_reason: str | None = None,
    container_clip_reason: str | None = None,
) -> dict[str, Any]:
    audit = audit if isinstance(audit, dict) else {}
    source_class = _source_glyph_class_specific_contract(
        region,
        render,
        cleanup_mode=cleanup_mode,
        method=method,
        audit=audit,
        required=required,
        generated=generated,
        review_only=review_only,
    )
    allowed_ref = _phase4_allowed_area_ref(allowed_area, audit, region, render)
    allowed_validity = _phase4_allowed_area_validity(
        allowed_ref,
        required=required,
        review_only=review_only,
        source_class=source_class,
        container_clip_reason=container_clip_reason,
    )
    quality_status = _phase4_quality_status(
        source_class=source_class,
        generated=generated,
        required=required,
        review_only=review_only,
        allowed_area_validity=allowed_validity,
        audit=audit,
        failure_reason=failure_reason,
    )
    suitability_status, suitability_reason = _phase4_cleanup_suitability(
        source_class=source_class,
        quality_status=quality_status,
        allowed_area_validity=allowed_validity,
        audit=audit,
        generated=generated,
        required=required,
        review_only=review_only,
        consumed_by_renderer=consumed_by_renderer,
        failure_reason=failure_reason,
    )
    protected_reason = _phase4_generic_protection_reason(region, render, audit)
    if protected_reason is not None:
        suitability_status = "blocked"
        suitability_reason = protected_reason
        if failure_reason is None:
            failure_reason = protected_reason
    bbox = _phase4_bbox(mask_bbox, audit, region, render)
    coverage = _float_or_none_contract(audit.get("source_glyph_erasure_coverage_ratio"))
    if failure_reason is None and quality_status in {"blocked", "missing"}:
        failure_reason = (
            str(audit.get("cleanup_source_erasure_failure_reason") or "")
            or _default_missing_reason(region, render, cleanup_mode)
            or "source_glyph_contract_blocked"
        )
    if failure_reason is None and suitability_status == "blocked":
        failure_reason = suitability_reason or "source_glyph_cleanup_suitability_blocked"
    return {
        "source_glyph_contract_version": SOURCE_GLYPH_PHASE4_CONTRACT_VERSION,
        "source_glyph_required_class_vocabulary": list(SOURCE_GLYPH_CLASS_VOCABULARY),
        "text_instance_ids": _source_glyph_text_instance_ids(region, render),
        "mask_ref": str(mask_id or "") or None,
        "bbox": bbox,
        "coverage_ratio": coverage,
        "allowed_area_ref": allowed_ref,
        "allowed_area_validity": allowed_validity,
        "class_specific_contract": source_class,
        "quality_status": quality_status,
        "cleanup_suitability_status": suitability_status,
        "cleanup_suitability_reason": suitability_reason,
        "failure_reason": failure_reason,
        "compatibility_source_fields": _phase4_compatibility_source_fields(
            region=region,
            render=render,
            method=method,
            mask_id=mask_id,
            mask_bbox=mask_bbox,
            allowed_area=allowed_area,
            audit=audit,
            generated=generated,
            required=required,
            review_only=review_only,
            consumed_by_renderer=consumed_by_renderer,
        ),
    }


def _phase4_generic_protection_reason(region: dict, render: dict, audit: dict[str, Any]) -> str | None:
    source_grounding_values = [
        _canonical_region_render_value(region, render, "source_grounding_status"),
        _canonical_region_render_value(region, render, "source_grounding_failure_reason"),
        _canonical_region_render_value(region, render, "translated_text_suppressed_reason"),
    ]
    if _canonical_region_render_value(region, render, "render_suppressed_by_source_grounding") is True:
        return "source_grounding_protected"
    source_grounding_text = " ".join(str(value or "") for value in source_grounding_values).lower()
    if (
        "source_ungrounded" in source_grounding_text
        or "ungrounded" in source_grounding_text
        or "mislocalized_source_erasure" in source_grounding_text
    ):
        return "source_grounding_protected"

    unsafe_values = [
        _canonical_region_render_value(region, render, "render_visual_contract_audit_status"),
        _canonical_region_render_value(region, render, "render_visual_contract_blocker_reason"),
        _canonical_region_render_value(region, render, "translated_text_suppressed_reason"),
        _canonical_region_render_value(region, render, "cleanup_artifact_risk_reason"),
    ]
    unsafe_text = " ".join(str(value or "") for value in unsafe_values).lower()
    if "unsafe_cleanup" in unsafe_text or "unsafe cleanup" in unsafe_text:
        return "unsafe_cleanup_protected"
    return None


def _source_glyph_class_specific_contract(
    region: dict,
    render: dict,
    *,
    cleanup_mode: str,
    method: str | None,
    audit: dict[str, Any],
    required: bool,
    generated: bool,
    review_only: bool,
) -> str:
    flags = region.get("flags", {}) or {}
    method_text = str(method or "")
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "")
    route_intent = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
    region_type = str(region.get("type") or "")
    reason_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_reason_codes")).lower()
    conflict_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_conflict_flags")).lower()
    route_and_type = " ".join([container_type, route_intent, region_type, cleanup_mode, method_text, reason_text, conflict_text]).lower()
    if flags.get("sfx") or region_type == "decorative_text" or container_type == "sfx_decorative_art" or route_intent == "preserve_sfx_decorative":
        return "preserve_sfx_decorative"
    if "art" in conflict_text or "art_entangled" in route_and_type or "art-entangled" in route_and_type:
        return "art_entangled_ambiguous"
    if "side_caption" in method_text or _is_vertical_side_caption(region, render):
        return "side_caption_glyph_local"
    if "title" in route_and_type or "sign" in route_and_type:
        return "title_or_sign"
    if "small_reaction" in route_and_type or _phase4_is_small_reaction(region):
        return "small_reaction"
    is_caption = (
        container_type == "caption_background"
        or route_intent in {"translate_caption", "translate_caption_background"}
        or region_type in {"background_text", "narration_box"}
        or "caption" in method_text
    )
    if is_caption:
        if (
            "dark" in route_and_type
            or "screentone" in route_and_type
            or "texture" in route_and_type
            or audit.get("cleanup_visual_artifact_risk") is True
            or _float_or_none_contract(audit.get("source_glyph_erasure_coverage_ratio")) not in (None, 1.0)
        ):
            return "caption_dark_or_screentone"
        return "caption_flat_background"
    if (
        "represented_child" in method_text
        or "recovered" in method_text
        or "fallback" in method_text
        or not generated
        or review_only
    ):
        return "speech_complex_bubble"
    return "speech_flat_bubble" if required else "speech_complex_bubble"


def _phase4_cleanup_suitability(
    *,
    source_class: str,
    quality_status: str,
    allowed_area_validity: str,
    audit: dict[str, Any],
    generated: bool,
    required: bool,
    review_only: bool,
    consumed_by_renderer: bool,
    failure_reason: str | None,
) -> tuple[str, str]:
    if source_class in {"preserve_sfx_decorative", "art_entangled_ambiguous"}:
        return "blocked", f"{source_class}_not_cleanup_candidate"
    if not required and not generated:
        return "not_cleanup_candidate", "source_glyph_mask_not_required"
    if quality_status in {"blocked", "missing"}:
        return "blocked", str(failure_reason or audit.get("cleanup_source_erasure_failure_reason") or "source_glyph_quality_blocked")
    if allowed_area_validity != "valid":
        return "blocked", f"allowed_area_{allowed_area_validity}"
    if review_only or source_class in {"speech_complex_bubble", "caption_dark_or_screentone", "small_reaction"}:
        return "review_required", f"{source_class}_requires_later_stage_review"
    if audit.get("cleanup_covers_source_glyphs") is not True:
        return "blocked", str(audit.get("cleanup_source_erasure_failure_reason") or "cleanup_mask_misses_source_glyphs")
    if audit.get("cleanup_visual_artifact_risk") is True:
        return "review_required", "cleanup_visual_artifact_risk"
    return "suitable_for_later_cleanup", "class_specific_source_glyph_contract_passed"


def _phase4_quality_status(
    *,
    source_class: str,
    generated: bool,
    required: bool,
    review_only: bool,
    allowed_area_validity: str,
    audit: dict[str, Any],
    failure_reason: str | None,
) -> str:
    if source_class in {"preserve_sfx_decorative", "art_entangled_ambiguous"}:
        return "blocked"
    if not required and not generated:
        return "not_required"
    if required and not generated:
        return "missing"
    if failure_reason:
        return "blocked"
    if allowed_area_validity not in {"valid", "not_applicable"}:
        return "blocked"
    if audit.get("cleanup_covers_source_glyphs") is False:
        return "blocked"
    if audit.get("cleanup_visual_artifact_risk") is True or review_only:
        return "review"
    coverage = _float_or_none_contract(audit.get("source_glyph_erasure_coverage_ratio"))
    if coverage is not None and coverage < SOURCE_GLYPH_CLEANUP_COVERAGE_THRESHOLD:
        return "blocked"
    return "usable"


def _phase4_allowed_area_ref(allowed_area, audit: dict[str, Any], region: dict, render: dict):
    value = allowed_area
    if value is None:
        value = audit.get("erase_mask_allowed_area")
    if value is None:
        value = _phase4_xywh_to_xyxy_list(_canonical_region_render_value(region, render, "text_area_container_bbox"))
    xyxy = _coerce_box_list(value)
    return xyxy


def _phase4_allowed_area_validity(
    allowed_area,
    *,
    required: bool,
    review_only: bool,
    source_class: str,
    container_clip_reason: str | None,
) -> str:
    if source_class in {"preserve_sfx_decorative", "art_entangled_ambiguous"}:
        return "not_applicable"
    if not required and not review_only:
        return "not_applicable"
    if not allowed_area:
        return "missing"
    try:
        x0, y0, x1, y1 = [int(v) for v in allowed_area[:4]]
    except Exception:
        return "invalid"
    if x1 <= x0 or y1 <= y0:
        return "invalid"
    if container_clip_reason:
        return "conflicting"
    return "valid"


def _phase4_bbox(mask_bbox, audit: dict[str, Any], region: dict, render: dict):
    for value in (
        mask_bbox,
        audit.get("source_glyph_erasure_bbox"),
        audit.get("source_glyph_erasure_expected_area_bbox"),
        _phase4_xywh_to_xyxy_list(_canonical_region_render_value(region, render, "text_area_container_bbox")),
    ):
        box = _coerce_box_list(value)
        if box:
            return box
    return None


def _phase4_compatibility_source_fields(
    *,
    region: dict,
    render: dict,
    method: str | None,
    mask_id: str | None,
    mask_bbox,
    allowed_area,
    audit: dict[str, Any],
    generated: bool,
    required: bool,
    review_only: bool,
    consumed_by_renderer: bool,
) -> dict[str, Any]:
    return {
        "source_glyph_mask_id": mask_id,
        "source_glyph_mask_generation_method": method,
        "source_glyph_mask_generation_required": required,
        "source_glyph_mask_generated": generated,
        "source_glyph_mask_review_only": review_only,
        "source_glyph_mask_consumed_by_renderer": consumed_by_renderer,
        "cleanup_allowed_area": _coerce_box_list(allowed_area),
        "mask_bbox": _coerce_box_list(mask_bbox),
        "source_glyph_erasure_bbox": _coerce_box_list(audit.get("source_glyph_erasure_bbox")),
        "source_glyph_erasure_expected_area_bbox": _coerce_box_list(audit.get("source_glyph_erasure_expected_area_bbox")),
        "source_glyph_erasure_expected_pixels": audit.get("source_glyph_erasure_expected_pixels"),
        "source_glyph_erasure_coverage_ratio": audit.get("source_glyph_erasure_coverage_ratio"),
        "cleanup_covers_source_glyphs": audit.get("cleanup_covers_source_glyphs"),
        "cleanup_source_erasure_failure_reason": audit.get("cleanup_source_erasure_failure_reason"),
        "cleanup_visual_artifact_risk": audit.get("cleanup_visual_artifact_risk"),
        "erase_mask_method": audit.get("erase_mask_method"),
        "erase_mask_pixels": audit.get("erase_mask_pixels"),
        "erase_mask_bbox": _coerce_box_list(audit.get("erase_mask_bbox")),
        "erase_mask_growth_ratio": audit.get("erase_mask_growth_ratio"),
        "erase_mask_allowed_area": _coerce_box_list(audit.get("erase_mask_allowed_area")),
        "erase_mask_rejected_reason": audit.get("erase_mask_rejected_reason"),
        "erase_mask_artifact_risk": audit.get("erase_mask_artifact_risk"),
        "erase_mask_visual_scope": audit.get("erase_mask_visual_scope"),
        "text_area_container_type": _canonical_region_render_value(region, render, "text_area_container_type"),
        "text_area_route_intent": _canonical_region_render_value(region, render, "text_area_route_intent"),
    }


def _source_glyph_text_instance_ids(region: dict, render: dict) -> list[str]:
    ids: list[str] = []
    for key in (
        "text_instance_id",
        "visual_text_instance_id",
        "child_recognized_text_segment_id",
        "parent_logical_text_unit_anchor_child_id",
    ):
        value = _canonical_region_render_value(region, render, key)
        if value:
            ids.append(str(value))
    for key in (
        "parent_logical_text_unit_child_segment_ids",
        "logical_text_block_member_region_ids",
        "logical_text_source_reconstruction_included_child_region_ids",
    ):
        for value in _canonical_region_render_list(region, render, key):
            if value:
                ids.append(str(value))
    region_id = str(region.get("region_id") or "")
    if region_id:
        ids.append(region_id)
    seen: set[str] = set()
    result: list[str] = []
    for value in ids:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _phase4_is_small_reaction(region: dict) -> bool:
    bbox = region.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        _x, _y, w, h = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return False
    text = str(region.get("ocr_text") or region.get("text") or "").strip()
    return max(w, h) <= 90 and len(text) <= 4


def _coerce_box_list(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None


def _phase4_xywh_to_xyxy_list(value) -> list[int] | None:
    box = _coerce_box_list(value)
    if not box:
        return None
    x, y, w, h = box
    return [x, y, x + max(0, w), y + max(0, h)]


def _float_or_none_contract(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _base_coverage_record(page_id: str, region: dict, render: dict, cleanup_mode: str) -> dict[str, Any]:
    flags = region.get("flags", {}) or {}
    region_id = str(region.get("region_id", "") or "")
    renderable = bool(str(region.get("translation", "") or "").strip()) and not bool(flags.get("ignore"))
    cleanup_only = _is_route_owned_cleanup_only_anchor(region, render)
    ocr_backed = bool(str(region.get("ocr_text", "") or "").strip())
    region_type = str(region.get("type", "") or "")
    cleanup_enabled = (renderable or cleanup_only) and cleanup_mode not in {"", "preserve", "none", "skip"}
    tracking_required = bool(cleanup_enabled and (ocr_backed or cleanup_only) and region_type not in {"decorative_text"})
    not_generated_reason = None if tracking_required else _default_missing_reason(region, render, cleanup_mode)
    record = {
        "page_id": page_id,
        "region_id": region_id,
        "parent_execution_bundle_id": str(
            region.get("parent_execution_bundle_id")
            or render.get("parent_execution_bundle_id")
            or region_id
            or ""
        ),
        "parent_logical_text_unit_id": str(
            _canonical_region_render_value(region, render, "parent_logical_text_unit_id")
            or ""
        ),
        "text_block_root_id": str(
            _canonical_region_render_value(region, render, "text_block_root_id")
            or ""
        ),
        "semantic_class": region_type,
        "cleanup_mode": cleanup_mode,
        "cleanup_source_tracking_required": tracking_required,
        "source_glyph_mask_generation_required": False,
        "source_glyph_mask_generation_status": "not_attempted" if tracking_required else "not_applicable",
        "source_glyph_mask_not_generated_reason": not_generated_reason,
        "source_glyph_mask_review_only": False,
        "source_glyph_mask_required": False,
        "source_glyph_mask_generated": False,
        "source_glyph_mask_consumed_by_renderer": False,
        "source_glyph_mask_generation_method": None,
        "source_glyph_mask_missing_reason": None,
        "source_glyph_mask_fallback_used": False,
        "source_glyph_mask_fallback_reason": None,
        "source_glyph_erasure_bbox": None,
        "source_glyph_erasure_expected_area_bbox": None,
        "source_glyph_erasure_expected_pixels": None,
        "source_glyph_erasure_coverage_ratio": None,
        "cleanup_covers_source_glyphs": None,
        "cleanup_visual_artifact_risk": None,
        "route_owned_foreground_contract_status": None,
        "foreground_mask_method": None,
        "foreground_mask_pixels": None,
        "foreground_mask_bbox": None,
        "erase_mask_method": None,
        "erase_mask_pixels": None,
        "erase_mask_bbox": None,
        "erase_mask_growth_ratio": None,
        "erase_mask_allowed_area": None,
        "erase_mask_rejected_reason": None,
        "erase_mask_artifact_risk": None,
        "erase_mask_visual_scope": None,
        "text_area_container_id": _canonical_region_render_value(region, render, "text_area_container_id"),
        "text_area_container_type": _canonical_region_render_value(region, render, "text_area_container_type"),
        "text_area_route_intent": _canonical_region_render_value(region, render, "text_area_route_intent"),
    }
    record.update(
        _source_glyph_phase4_contract_fields(
            page_id=page_id,
            region=region,
            render=render,
            cleanup_mode=cleanup_mode,
            method=None,
            generated=False,
            required=False,
            review_only=False,
            consumed_by_renderer=False,
            failure_reason=not_generated_reason,
        )
    )
    return record



def _default_missing_reason(region: dict, render: dict, cleanup_mode: str) -> str | None:
    flags = region.get("flags", {}) or {}
    if flags.get("ignore"):
        return "ignored_by_pipeline"
    cleanup_only = _is_route_owned_cleanup_only_anchor(region, render)
    if not str(region.get("translation", "") or "").strip():
        if cleanup_only:
            return "cleanup_only_source_glyph_mask_required"
        return "not_renderable_or_empty_translation"
    if not str(region.get("ocr_text", "") or "").strip() and not cleanup_only:
        return "no_ocr_text"
    if cleanup_mode in {"", "preserve", "none", "skip"}:
        return "cleanup_not_enabled"
    if str(region.get("type", "") or "") == "decorative_text" or flags.get("sfx"):
        return "sfx_decorative_preserve"
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "")
    route_intent = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
    if container_type == "sfx_decorative_art" or route_intent == "preserve_sfx_decorative":
        return "text_area_preserve_route_not_cleanup_mask_candidate"
    if _canonical_region_render_list(region, render, "text_area_conflict_flags"):
        return "text_area_conflict_flags"
    return "source_glyph_mask_not_required_for_cleanup_mode"


def _should_generate_speech_review_mask(region: dict, render: dict, bbox, cleanup_mode: str, img_np) -> bool:
    if str(region.get("type", "") or "") != "speech_bubble":
        return False
    if cleanup_mode not in {"bubble", "speech_bubble", "speech_strong"}:
        return False
    if not str(region.get("translation", "") or "").strip() or not str(region.get("ocr_text", "") or "").strip():
        return False
    flags = region.get("flags", {}) or {}
    if flags.get("ignore") or flags.get("sfx"):
        return False
    if _canonical_region_render_list(region, render, "text_area_conflict_flags"):
        return False
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "")
    route_intent = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
    if container_type and container_type != "speech_bubble":
        return False
    if route_intent and route_intent != "translate_speech":
        return False
    try:
        _x, _y, w, h = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return False
    page_area = max(1, int(img_np.shape[0]) * int(img_np.shape[1]))
    box_area = max(1, w) * max(1, h)
    if box_area > int(page_area * 0.04):
        return False
    tier = str(_canonical_region_render_value(region, render, "text_area_confidence_tier") or "")
    reason_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_reason_codes"))
    return tier in {"strong_model_container", "mask_primary_container"} or "speech" in reason_text


def _review_mask_quality_passes(source_mask, audit: dict[str, Any], expected_box, img_w: int, img_h: int) -> bool:
    if source_mask is None or expected_box is None:
        return False
    if audit.get("cleanup_covers_source_glyphs") is not True:
        return False
    if audit.get("cleanup_visual_artifact_risk") is True:
        return False
    stats = mask_stats(source_mask) or {}
    bbox = _mask_stats_box(stats)
    if bbox is None:
        return False
    mask_area = _xyxy_area(bbox)
    expected_area = max(1, _xyxy_area(expected_box))
    page_area = max(1, int(img_w) * int(img_h))
    if mask_area > int(expected_area * 0.72):
        return False
    if mask_area > int(page_area * 0.018):
        return False
    return True


def _should_generate_side_caption_mask(region: dict, render: dict, bbox, img_np) -> bool:
    if not _is_vertical_side_caption(region, render):
        return False
    if not str(region.get("translation", "") or "").strip():
        return False
    cleanup_mode = str(render.get("cleanup_mode", "") or "").strip().lower()
    if not cleanup_mode:
        cleanup_mode = "background_box"
    if cleanup_mode not in {"local_text_mask", "background_box", "caption_strong"}:
        return False
    if cleanup_mode == "background_box":
        try:
            _x, _y, w, h = [int(round(float(v))) for v in bbox[:4]]
        except Exception:
            return False
        stats = _box_luma_stats(img_np, _xywh_to_xyxy_tuple(bbox))
        contrast_span = None
        if stats:
            _mean, p20, p80 = stats
            contrast_span = p80 - p20
        page_area = max(1, img_np.shape[0] * img_np.shape[1])
        bg_area_ratio = (max(1, w) * max(1, h)) / page_area
        if not (
            bg_area_ratio >= 0.015
            or (w <= 60 and h >= 150 and contrast_span is not None and contrast_span >= 28)
            or (contrast_span is not None and contrast_span >= 52 and bg_area_ratio >= 0.002)
        ):
            return False
    return True


def _should_generate_represented_child_mask(
    region: dict,
    render: dict,
    active_parent_ids: set[str],
) -> bool:
    if _is_active_translation_region(region):
        return False
    if not _has_japanese_source_text(str(region.get("ocr_text", "") or "")):
        return False
    if _is_preserved_source_child(region, render):
        return False
    parent_id = _represented_child_parent_id(region, render)
    if not parent_id or parent_id not in active_parent_ids:
        return False
    final_state = str(
        region.get("child_final_state")
        or region.get("ocr_fragment_final_state")
        or _canonical_region_render_value(region, render, "logical_text_ownership_status")
        or ""
    ).strip().lower()
    represented = bool(
        region.get("represented_by_parent_id")
        or region.get("source_text_represented_by_block_id")
        or region.get("source_conservation_status") == "represented"
    )
    return final_state in {
        "parent_child",
        "dependent_child",
        "transferred_child",
        "duplicate_child",
        "punctuation_child",
        "noise_review_only",
    } or represented


def _represented_child_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    bbox,
    polygon,
    img_w: int,
    img_h: int,
    *,
    limit_box,
):
    try:
        _x, _y, w, h = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None, "represented_child_invalid_bbox", "invalid_child_bbox"
    orientation = str(_canonical_region_render_value(region, render, "source_orientation") or "").lower()
    vertical = orientation == "vertical" or h > w * 1.35
    dilate_px = max(2, min(5, int(max(min(w, h) * 0.08, 2))))
    mask = _source_glyph_local_mask(
        img_np,
        bbox,
        polygon,
        dilate_px=dilate_px,
        limit_box=limit_box,
        bright_context_only=False,
    )
    method = "represented_child_glyph_local"
    if mask is None or not np.any(mask):
        mask = _source_glyph_local_mask(
            img_np,
            bbox,
            polygon,
            dilate_px=max(dilate_px, 3 if vertical else 2),
            limit_box=limit_box,
            bright_context_only=True,
        )
        method = "represented_child_glyph_local_bright_context"
    if mask is None or not np.any(mask):
        box = _xywh_value_to_xyxy(bbox, img_w, img_h)
        if box is None:
            return None, method, "child_glyph_mask_unavailable"
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        pad = max(2, min(5, int(max((box[2] - box[0]), (box[3] - box[1])) * 0.04)))
        x0, y0, x1, y1 = _expand_box(box, pad, pad, img_w, img_h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        if limit_box is not None:
            limit = np.zeros(img_np.shape[:2], dtype=np.uint8)
            lx0, ly0, lx1, ly1 = [int(v) for v in limit_box[:4]]
            cv2.rectangle(limit, (max(0, lx0), max(0, ly0)), (min(img_w, lx1), min(img_h, ly1)), 255, thickness=-1)
            mask = cv2.bitwise_and(mask, limit)
        method = "represented_child_bbox_fallback"
    stats = mask_stats(mask)
    box = _mask_stats_box(stats)
    image_area = max(1, int(img_w) * int(img_h))
    bbox_area = _xyxy_area(box) if box else 0
    if not box or int((stats or {}).get("pixels") or 0) <= 0:
        return None, method, "empty_child_cleanup_mask"
    if bbox_area > max(240000, int(image_area * 0.12)):
        return None, method, "child_cleanup_mask_too_large"
    return mask.astype(np.uint8), method, None


def _active_translation_parent_ids(regions: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for region in regions:
        if not isinstance(region, dict) or not _is_active_translation_region(region):
            continue
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        ids.update(_region_parent_ids(region, render))
    return ids


def _is_active_translation_region(region: dict) -> bool:
    flags = region.get("flags", {}) or {}
    if isinstance(flags, dict) and flags.get("ignore"):
        return False
    if not str(region.get("translation") or region.get("translated_text") or "").strip():
        return False
    if region.get("render_independently") is False and str(region.get("child_final_state") or "").lower() not in {
        "parent_anchor",
        "standalone_parent",
    }:
        return False
    return True


def _region_parent_ids(region: dict, render: dict) -> set[str]:
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
            values = value
        else:
            values = [value]
        for item in values:
            text = str(item or "").strip()
            if text:
                ids.add(text)
    return ids


def _represented_child_parent_id(region: dict, render: dict) -> str | None:
    for key in (
        "represented_by_parent_id",
        "source_text_represented_by_block_id",
        "parent_logical_text_unit_id",
        "active_translation_unit_id",
        "logical_text_block_id",
    ):
        value = region.get(key)
        if value in (None, "", [], {}):
            value = render.get(key)
        if value not in (None, "", [], {}):
            return str(value).strip() or None
    return None


def _is_preserved_source_child(region: dict, render: dict) -> bool:
    flags = region.get("flags", {}) or {}
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


def _is_background_region(region: dict) -> bool:
    flags = region.get("flags", {}) or {}
    return str(region.get("type", "") or "") == "background_text" or bool(flags.get("bg_text"))


def _is_vertical_side_caption(region, render) -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    if str(region.get("type", "") or "") != "background_text":
        return False
    reason_codes = []
    for source in (render, region):
        values = source.get("text_area_reason_codes") if isinstance(source, dict) else None
        if isinstance(values, (list, tuple)):
            reason_codes.extend(str(item) for item in values)
        elif values:
            reason_codes.append(str(values))
    return "deterministic_vertical_side_caption_search" in " ".join(reason_codes)


def _is_recovered_logical_speech_anchor(region: dict, render: dict) -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    applied = bool(_canonical_region_render_value(region, render, "logical_text_source_reconstruction_applied"))
    action = str(_canonical_region_render_value(region, render, "logical_text_source_quality_action") or "").strip()
    ownership = str(_canonical_region_render_value(region, render, "logical_text_ownership_status") or "").strip()
    return applied and action == "reocr_recovered" and ownership == "block_anchor"


def _is_accepted_standalone_speech_cleanup_anchor(region: dict, render: dict) -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    flags = region.get("flags", {}) or {}
    if bool(region.get("skip_translation")) or bool(flags.get("ignore")) or bool(flags.get("sfx")):
        return False
    if bool(flags.get("is_sfx")) or bool(flags.get("is_decorative")) or bool(flags.get("is_art")):
        return False
    if not str(region.get("ocr_text") or region.get("source_text") or "").strip():
        return False
    if not str(region.get("translation") or region.get("translated_text") or "").strip():
        return False

    route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "").strip()
    region_type = str(region.get("type", "") or "").strip()
    container_type = str(_canonical_region_render_value(region, render, "text_area_container_type") or "").strip()
    cleanup_mode = str(render.get("cleanup_mode") or region.get("cleanup_mode") or "").strip().lower()
    if cleanup_mode and cleanup_mode not in {"bubble", "speech_bubble", "speech_strong"}:
        return False
    if route and route != "translate_speech":
        return False
    if container_type and container_type != "speech_bubble":
        return False
    if region_type and region_type != "speech_bubble":
        return False

    conflict_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_conflict_flags")).lower()
    reason_text = " ".join(str(item) for item in _canonical_region_render_list(region, render, "text_area_reason_codes")).lower()
    blocked_tokens = ("preserve", "sfx", "decorative", "art", "non_translation", "ignore")
    if any(token in conflict_text for token in blocked_tokens):
        return False
    if any(token in reason_text for token in ("preserve_sfx", "sfx_decorative", "art_entangled", "non_translation_art")):
        return False

    ownership = str(_canonical_region_render_value(region, render, "logical_text_ownership_status") or "").strip()
    block_status = str(_canonical_region_render_value(region, render, "logical_text_block_v3_status") or "").strip()
    if ownership not in {"block_anchor", "standalone_block", "standalone_utterance", ""}:
        return False
    if block_status and block_status not in {"standalone_utterance", "standalone_parent", "block_anchor"}:
        return False
    container_bbox = _canonical_region_render_value(region, render, "text_area_container_bbox")
    return _coerce_box_list(container_bbox) is not None


def _is_route_owned_cleanup_only_anchor(region: dict, render: dict) -> bool:
    # SourceGlyphMask may support translated logical-text cleanup anchors, but
    # route-owned cleanup-only anchors are no longer production authority.
    return False


def _is_logical_text_cleanup_anchor(region: dict, render: dict) -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    ownership = str(_canonical_region_render_value(region, render, "logical_text_ownership_status") or "").strip()
    child_state = str(
        _canonical_region_render_value(region, render, "child_final_state")
        or _canonical_region_render_value(region, render, "ocr_fragment_final_state")
        or ""
    ).strip()
    flags = region.get("flags", {}) or {}
    if bool(region.get("skip_translation")) or bool(flags.get("sfx")):
        return False
    region_type = str(region.get("type", "") or "")
    route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "")
    if route not in {"translate_speech", "translate_caption", "translate_caption_background"} and region_type not in {"speech_bubble", "background_text", "narration_box"}:
        return False
    if not str(region.get("translation") or region.get("translated_text") or "").strip():
        return False
    member_ids = set(
        str(item)
        for key in (
            "logical_text_source_reconstruction_included_child_region_ids",
            "logical_text_block_transferred_region_ids",
            "logical_text_block_member_region_ids",
        )
        for item in _canonical_region_render_list(region, render, key)
        if str(item).strip()
    )
    has_dependent_children = len(member_ids) > 1 or bool(
        _canonical_region_render_list(region, render, "logical_text_source_reconstruction_root_internal_child_bboxes")
    )
    is_caption = route in {"translate_caption", "translate_caption_background"} or region_type in {"background_text", "narration_box"}
    if ownership not in {"block_anchor", "standalone_block", "standalone_utterance"}:
        if not (is_caption and child_state in {"standalone_parent", "parent_anchor"}):
            return False
    return bool(has_dependent_children or is_caption or _is_accepted_standalone_speech_cleanup_anchor(region, render))


def _rendered_region_requires_source_mask(region: dict, render: dict, cleanup_mode: str = "") -> bool:
    if not isinstance(region, dict):
        return False
    if not isinstance(render, dict):
        render = {}
    flags = region.get("flags", {}) or {}
    if bool(flags.get("ignore")) or bool(region.get("skip_translation")):
        return False
    if _is_route_owned_cleanup_only_anchor(region, render):
        return True
    if not str(region.get("translation") or region.get("translated_text") or "").strip():
        return False
    cleanup = str(cleanup_mode or render.get("cleanup_mode") or "").strip().lower()
    if cleanup == "preserve":
        return False
    route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "").strip()
    region_type = str(region.get("type") or "").strip()
    return (
        route in {"translate_speech", "translate_caption", "translate_caption_background"}
        or region_type in {"speech_bubble", "background_text", "narration_box"}
        or _is_logical_text_cleanup_anchor(region, render)
    )


def _recovered_speech_allowed_area(region: dict, render: dict, img_w: int, img_h: int):
    limit_box = _xywh_value_to_xyxy(
        _canonical_region_render_value(region, render, "text_area_container_bbox"),
        img_w,
        img_h,
    )
    crop_limit_box = _xywh_value_to_xyxy(
        _canonical_region_render_value(region, render, "logical_text_source_reconstruction_crop_bbox"),
        img_w,
        img_h,
    )
    if crop_limit_box is not None and (
        limit_box is None or _xyxy_area(crop_limit_box) > int(_xyxy_area(limit_box) * 1.20)
    ):
        return crop_limit_box
    return limit_box


def _logical_text_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    dilate_px: int = 2,
    bright_context_only: bool = False,
):
    if cv2 is None or np is None or img_np is None:
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    region_id = str(region.get("region_id", "") or "")
    img_h, img_w = img_np.shape[:2]
    limit_box = _recovered_speech_allowed_area(region, render, img_w, img_h)
    ordered_ids: list[str] = []
    if region_id:
        ordered_ids.append(region_id)
    for key in (
        "logical_text_source_reconstruction_included_child_region_ids",
        "logical_text_block_transferred_region_ids",
        "logical_text_block_member_region_ids",
    ):
        for rid in _canonical_region_render_list(region, render, key):
            rid = str(rid or "")
            if rid and rid not in ordered_ids:
                ordered_ids.append(rid)
    for child_bbox in _canonical_region_render_list(region, render, "logical_text_source_reconstruction_root_internal_child_bboxes"):
        if not isinstance(child_bbox, (list, tuple)) or len(child_bbox) < 4:
            continue
        local = _source_glyph_local_mask(
            img_np,
            child_bbox,
            None,
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=bright_context_only,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    for rid in ordered_ids:
        member = region if rid == region_id else regions_by_id.get(rid)
        if not isinstance(member, dict):
            continue
        member_bbox = member.get("bbox")
        if not isinstance(member_bbox, (list, tuple)) or len(member_bbox) < 4:
            continue
        member_flags = member.get("flags", {}) or {}
        if member_flags.get("sfx") and rid != region_id:
            continue
        local = _source_glyph_local_mask(
            img_np,
            member_bbox,
            member.get("polygon"),
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=bright_context_only,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    if not np.any(mask):
        fallback = _source_glyph_local_mask(
            img_np,
            fallback_bbox,
            fallback_polygon,
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=bright_context_only,
        )
        return fallback
    return mask


def _caption_background_logical_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    dilate_px: int = 1,
):
    if cv2 is None or np is None or img_np is None:
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    region_id = str(region.get("region_id", "") or "")
    img_h, img_w = img_np.shape[:2]
    limit_box = _recovered_speech_allowed_area(region, render, img_w, img_h)
    ordered_ids: list[str] = []
    if region_id:
        ordered_ids.append(region_id)
    for key in (
        "logical_text_source_reconstruction_included_child_region_ids",
        "logical_text_block_transferred_region_ids",
        "logical_text_block_member_region_ids",
    ):
        for rid in _canonical_region_render_list(region, render, key):
            rid = str(rid or "")
            if rid and rid not in ordered_ids:
                ordered_ids.append(rid)
    for child_bbox in _canonical_region_render_list(region, render, "logical_text_source_reconstruction_root_internal_child_bboxes"):
        if not isinstance(child_bbox, (list, tuple)) or len(child_bbox) < 4:
            continue
        local = _caption_background_source_cleanup_mask(
            img_np,
            child_bbox,
            None,
            dilate_px=dilate_px,
            limit_box=limit_box,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    for rid in ordered_ids:
        member = region if rid == region_id else regions_by_id.get(rid)
        if not isinstance(member, dict):
            continue
        member_bbox = member.get("bbox")
        if not isinstance(member_bbox, (list, tuple)) or len(member_bbox) < 4:
            continue
        if _is_preserved_source_child(member, member.get("render") if isinstance(member.get("render"), dict) else {}):
            continue
        local = _caption_background_source_cleanup_mask(
            img_np,
            member_bbox,
            member.get("polygon"),
            dilate_px=dilate_px,
            limit_box=limit_box,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    if not np.any(mask):
        return _caption_background_source_cleanup_mask(
            img_np,
            fallback_bbox,
            fallback_polygon,
            dilate_px=dilate_px,
            limit_box=limit_box,
        )
    return mask


def _caption_background_source_cleanup_mask(
    img_np,
    bbox,
    polygon=None,
    dilate_px: int = 1,
    limit_box=None,
):
    if cv2 is None or np is None or img_np is None:
        return None
    dark_mask = _source_glyph_local_mask(
        img_np,
        bbox,
        polygon,
        dilate_px=max(1, int(dilate_px)),
        limit_box=limit_box,
        bright_context_only=False,
    )
    bright_mask = _bright_caption_glyph_mask(
        img_np,
        bbox,
        polygon,
        dilate_px=max(1, int(dilate_px)),
        limit_box=limit_box,
    )
    if dark_mask is None:
        return bright_mask
    if bright_mask is None:
        return dark_mask
    return cv2.bitwise_or(dark_mask.astype(np.uint8), bright_mask.astype(np.uint8))


def _bright_caption_glyph_mask(
    img_np,
    bbox,
    polygon=None,
    dilate_px: int = 1,
    limit_box=None,
):
    """Build a glyph-local mask for light caption text on dark/background art."""
    if cv2 is None or np is None or img_np is None:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = img_np.shape[:2]
    x0 = max(0, min(img_w, x))
    y0 = max(0, min(img_h, y))
    x1 = max(x0 + 1, min(img_w, x + w))
    y1 = max(y0 + 1, min(img_h, y + h))
    if limit_box is not None:
        try:
            lx0, ly0, lx1, ly1 = [int(v) for v in limit_box[:4]]
            x0 = max(x0, lx0)
            y0 = max(y0, ly0)
            x1 = min(x1, lx1)
            y1 = min(y1, ly1)
        except Exception:
            pass
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
            shifted = [
                np.array([(int(px - x0), int(py - y0)) for px, py in poly], dtype=np.int32)
                for poly in polys
            ]
            cv2.fillPoly(allowed, shifted, 255)
        except Exception:
            allowed[:, :] = 255
    else:
        allowed[:, :] = 255
    if limit_box is not None:
        limit_local = np.zeros_like(glyph)
        try:
            lx0, ly0, lx1, ly1 = [int(v) for v in limit_box[:4]]
            cv2.rectangle(
                limit_local,
                (max(0, lx0 - x0), max(0, ly0 - y0)),
                (min(x1 - x0, lx1 - x0), min(y1 - y0, ly1 - y0)),
                255,
                thickness=-1,
            )
            allowed = cv2.bitwise_and(allowed, limit_local)
        except Exception:
            pass
    glyph = cv2.bitwise_and(glyph, allowed)
    glyph = _filter_caption_background_glyph_components(glyph, allowed)
    glyph = cv2.morphologyEx(
        glyph,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(dilate_px) * 2 + 1, int(dilate_px) * 2 + 1),
        )
        glyph = cv2.dilate(glyph, kernel, iterations=1)
        glyph = cv2.bitwise_and(glyph, allowed)
    min_pixels = max(12, int((x1 - x0) * (y1 - y0) * 0.006))
    if int((glyph > 0).sum()) < min_pixels:
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = glyph
    return mask


def _filter_caption_background_glyph_components(glyph, allowed):
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


def _recovered_speech_source_cleanup_mask(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    dilate_px: int = 3,
):
    if cv2 is None or np is None or img_np is None:
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    region_id = str(region.get("region_id", "") or "")
    img_h, img_w = img_np.shape[:2]
    limit_box = _recovered_speech_allowed_area(region, render, img_w, img_h)
    ordered_ids: list[str] = []
    if region_id:
        ordered_ids.append(region_id)
    crop_bbox = _canonical_region_render_value(region, render, "logical_text_source_reconstruction_crop_bbox")
    if isinstance(crop_bbox, (list, tuple)) and len(crop_bbox) >= 4:
        crop_local = _source_glyph_local_mask(
            img_np,
            crop_bbox,
            None,
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=True,
        )
        if crop_local is not None:
            mask = cv2.bitwise_or(mask, crop_local)
    for child_bbox in _canonical_region_render_list(region, render, "logical_text_source_reconstruction_root_internal_child_bboxes"):
        if not isinstance(child_bbox, (list, tuple)) or len(child_bbox) < 4:
            continue
        local = _source_glyph_local_mask(
            img_np,
            child_bbox,
            None,
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=True,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    for key in (
        "logical_text_source_reconstruction_included_child_region_ids",
        "logical_text_block_transferred_region_ids",
        "logical_text_block_member_region_ids",
    ):
        for rid in _canonical_region_render_list(region, render, key):
            rid = str(rid or "")
            if rid and rid not in ordered_ids:
                ordered_ids.append(rid)
    for rid in ordered_ids:
        member = region if rid == region_id else regions_by_id.get(rid)
        if not isinstance(member, dict):
            continue
        member_bbox = member.get("bbox")
        if not isinstance(member_bbox, (list, tuple)) or len(member_bbox) < 4:
            continue
        member_type = str(member.get("type", "") or "")
        member_flags = member.get("flags", {}) or {}
        if member_type in {"decorative_text", "background_text", "narration_box"} and not _same_speech_container_member(region, member):
            continue
        if member_flags.get("sfx") and not _same_speech_container_member(region, member):
            continue
        local = _source_glyph_local_mask(
            img_np,
            member_bbox,
            member.get("polygon"),
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=True,
        )
        if local is not None:
            mask = cv2.bitwise_or(mask, local)
    if not np.any(mask):
        return _source_glyph_local_mask(
            img_np,
            fallback_bbox,
            fallback_polygon,
            dilate_px=dilate_px,
            limit_box=limit_box,
            bright_context_only=True,
        )
    return mask


def _source_glyph_local_mask(
    img_np,
    bbox,
    polygon=None,
    dilate_px: int = 2,
    limit_box=None,
    bright_context_only: bool = False,
):
    """Build a tight dark-glyph mask for source text inside an allowed area."""
    if cv2 is None or np is None or img_np is None:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = img_np.shape[:2]
    x0 = max(0, min(img_w, x))
    y0 = max(0, min(img_h, y))
    x1 = max(x0 + 1, min(img_w, x + w))
    y1 = max(y0 + 1, min(img_h, y + h))
    if limit_box is not None:
        try:
            lx0, ly0, lx1, ly1 = [int(v) for v in limit_box[:4]]
            x0 = max(x0, lx0)
            y0 = max(y0, ly0)
            x1 = min(x1, lx1)
            y1 = min(y1, ly1)
        except Exception:
            pass
    if x1 <= x0 or y1 <= y0:
        return None

    allowed = np.zeros(img_np.shape[:2], dtype=np.uint8)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            for poly in polys:
                cv2.fillPoly(allowed, [np.array(poly, dtype=np.int32)], 255)
        except Exception:
            polys = None
    if not polys:
        cv2.rectangle(allowed, (x0, y0), (x1, y1), 255, thickness=-1)
    else:
        rect = np.zeros_like(allowed)
        cv2.rectangle(rect, (x0, y0), (x1, y1), 255, thickness=-1)
        allowed = cv2.bitwise_and(allowed, rect)
    if limit_box is not None:
        limit = np.zeros_like(allowed)
        try:
            lx0, ly0, lx1, ly1 = [int(v) for v in limit_box[:4]]
            cv2.rectangle(
                limit,
                (max(0, lx0), max(0, ly0)),
                (min(img_w, lx1), min(img_h, ly1)),
                255,
                thickness=-1,
            )
            allowed = cv2.bitwise_and(allowed, limit)
        except Exception:
            pass
    ys, xs = np.where(allowed > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    rx0, ry0, rx1, ry1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = img_np[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    allowed_local = allowed[ry0:ry1, rx0:rx1]
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    delta = np.abs(gray.astype(np.int16) - blur.astype(np.int16))
    raw = (((gray <= 165) & ((delta >= 7) | (gray <= 120))).astype(np.uint8)) * 255
    raw = cv2.bitwise_and(raw, allowed_local)
    if not np.any(raw):
        return None

    paper_glyph_zone = None
    if bright_context_only:
        paper = (((gray >= 190) & (allowed_local > 0)).astype(np.uint8)) * 255
        if np.any(paper):
            paper = cv2.morphologyEx(
                paper,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            paper_glyph_zone = cv2.dilate(
                paper,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    crop_area = max(1, raw.shape[0] * raw.shape[1])
    max_component_area = max(2600, int(crop_area * 0.18))
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cx = int(stats[idx, cv2.CC_STAT_LEFT])
        cy = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max_component_area:
            continue
        if cw <= 0 or ch <= 0:
            continue
        density = area / max(1, cw * ch)
        if density > 0.82:
            continue
        if cw > max(42, int(raw.shape[1] * 0.82)) and ch > max(22, int(raw.shape[0] * 0.18)):
            continue
        if ch > max(90, int(raw.shape[0] * 0.88)) and cw > max(12, int(raw.shape[1] * 0.18)):
            continue
        if (
            (cx <= 1 or cy <= 1 or cx + cw >= raw.shape[1] - 1 or cy + ch >= raw.shape[0] - 1)
            and area > max(120, int(crop_area * 0.018))
            and (cw > raw.shape[1] * 0.45 or ch > raw.shape[0] * 0.45)
        ):
            continue
        if bright_context_only:
            comp_mask = (labels == idx).astype(np.uint8)
            if paper_glyph_zone is not None and not np.any((comp_mask > 0) & (paper_glyph_zone > 0)):
                continue
            halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            halo = cv2.dilate(comp_mask, halo_kernel, iterations=1)
            ring = (halo > 0) & (comp_mask == 0) & (allowed_local > 0)
            values = gray[ring]
            if values.size < 8:
                continue
            median = float(np.median(values))
            p20 = float(np.percentile(values, 20))
            bright_ratio = float((values >= 185).sum()) / max(1, values.size)
            if median < 198 or p20 < 150 or bright_ratio < 0.62:
                continue
        kept[labels == idx] = 255
    if not np.any(kept):
        return None
    kernel_size = max(1, int(dilate_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size * 2 + 1, kernel_size * 2 + 1),
    )
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel, iterations=1)
    kept = cv2.dilate(kept, kernel, iterations=1)
    kept = cv2.bitwise_and(kept, allowed_local)
    if not np.any(kept):
        return None
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    mask[ry0:ry1, rx0:rx1] = kept
    return mask


def _source_erasure_expected_box(
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    img_w: int,
    img_h: int,
):
    boxes: list[tuple[int, int, int, int]] = []
    for value in (
        _canonical_region_render_value(region, render, "logical_text_source_reconstruction_crop_bbox"),
        fallback_bbox,
    ):
        box = _xywh_value_to_xyxy(value, img_w, img_h)
        if box is not None:
            boxes.append(box)
    for key in (
        "logical_text_source_reconstruction_included_child_region_ids",
        "logical_text_block_transferred_region_ids",
        "logical_text_block_member_region_ids",
    ):
        for rid in _canonical_region_render_list(region, render, key):
            member = regions_by_id.get(str(rid or ""))
            if not isinstance(member, dict):
                continue
            box = _xywh_value_to_xyxy(member.get("bbox"), img_w, img_h)
            if box is not None:
                boxes.append(box)
    for child_bbox in (
        _canonical_region_render_list(region, render, "logical_text_source_reconstruction_root_internal_child_bboxes")
    ):
        box = _xywh_value_to_xyxy(child_bbox, img_w, img_h)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    x0 = max(0, min(box[0] for box in boxes))
    y0 = max(0, min(box[1] for box in boxes))
    x1 = min(img_w, max(box[2] for box in boxes))
    y1 = min(img_h, max(box[3] for box in boxes))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _source_glyph_erasure_audit_fields(img_np, cleanup_mask, expected_box, *, expected_source_mask=None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "cleanup_covers_source_glyphs": None,
        "cleanup_source_erasure_failure_reason": None,
        "cleanup_visual_artifact_risk": None,
        "source_glyph_erasure_expected_area_bbox": None,
        "source_glyph_erasure_expected_pixels": None,
    }
    if cv2 is None or np is None or img_np is None or cleanup_mask is None or expected_box is None:
        return fields
    try:
        x0, y0, x1, y1 = [int(v) for v in expected_box[:4]]
    except Exception:
        return fields
    if x1 <= x0 or y1 <= y0:
        return fields
    bbox = [x0, y0, x1 - x0, y1 - y0]
    expected_glyphs = expected_source_mask
    if expected_glyphs is not None and getattr(expected_glyphs, "shape", None) != getattr(cleanup_mask, "shape", None):
        expected_glyphs = None
    if expected_glyphs is None:
        expected_glyphs = _source_glyph_local_mask(img_np, bbox, None, dilate_px=1, limit_box=(x0, y0, x1, y1))
    fields["source_glyph_erasure_expected_area_bbox"] = [x0, y0, x1, y1]
    if expected_glyphs is None or not np.any(expected_glyphs):
        fields["cleanup_covers_source_glyphs"] = False
        fields["cleanup_source_erasure_failure_reason"] = "no_expected_source_glyph_mask"
        fields["source_glyph_erasure_bbox"] = [x0, y0, x1, y1]
        fields["source_glyph_erasure_coverage_ratio"] = 0.0
        fields["source_glyph_erasure_expected_pixels"] = 0
        return fields
    glyph_stats = mask_stats(expected_glyphs) or {}
    glyph_bbox = _mask_stats_box(glyph_stats)
    expected_pixels = int((expected_glyphs > 0).sum())
    covered_pixels = int(((expected_glyphs > 0) & (cleanup_mask > 0)).sum())
    coverage = covered_pixels / max(1, expected_pixels)
    cleanup_pixels = int((cleanup_mask > 0).sum())
    expected_area = max(1, _xyxy_area((x0, y0, x1, y1)))
    artifact_ratio = cleanup_pixels / expected_area
    fields["source_glyph_erasure_bbox"] = list(glyph_bbox) if glyph_bbox else [x0, y0, x1, y1]
    fields["source_glyph_erasure_expected_pixels"] = expected_pixels
    fields["source_glyph_erasure_coverage_ratio"] = round(coverage, 3)
    fields["cleanup_covers_source_glyphs"] = coverage >= SOURCE_GLYPH_CLEANUP_COVERAGE_THRESHOLD
    fields["cleanup_visual_artifact_risk"] = artifact_ratio > 0.42
    if coverage < SOURCE_GLYPH_CLEANUP_COVERAGE_THRESHOLD:
        fields["cleanup_source_erasure_failure_reason"] = "cleanup_mask_misses_source_glyphs"
    elif artifact_ratio > 0.42:
        fields["cleanup_source_erasure_failure_reason"] = "cleanup_mask_too_large_for_source_glyphs"
    return fields


def build_route_owned_caption_foreground_contract(
    img_np,
    bbox,
    polygon=None,
    *,
    allowed_area=None,
    halo_px: int = 2,
    method_prefix: str = "route_owned_caption_background",
) -> dict[str, Any]:
    """Build review-only foreground/erase masks for route-owned captions.

    The returned masks are for audit and offline benchmark use only.
    Production cleanup consumes CleanupMask foreground/erase records before
    rendering; SourceGlyphMask output is provenance, not renderer cleanup input.
    """
    fields = _empty_foreground_contract_fields("not_generated")
    result: dict[str, Any] = {"fields": fields, "foreground_mask": None, "erase_mask": None}
    if cv2 is None or np is None or img_np is None:
        fields["route_owned_foreground_contract_status"] = "unavailable"
        fields["erase_mask_rejected_reason"] = "cv2_or_numpy_unavailable"
        return result
    img_h, img_w = img_np.shape[:2]
    source_box = _xywh_value_to_xyxy(bbox, img_w, img_h)
    if source_box is None:
        fields["route_owned_foreground_contract_status"] = "rejected"
        fields["erase_mask_rejected_reason"] = "invalid_source_bbox"
        return result
    if allowed_area is None:
        allowed_area = source_box
        fields["route_owned_foreground_contract_status"] = "review_only_fallback_allowed_area"
        fields["erase_mask_rejected_reason"] = "missing_route_owned_allowed_area"
    else:
        try:
            ax0, ay0, ax1, ay1 = [int(v) for v in allowed_area[:4]]
            allowed_area = (
                max(0, min(img_w, ax0)),
                max(0, min(img_h, ay0)),
                max(0, min(img_w, ax1)),
                max(0, min(img_h, ay1)),
            )
        except Exception:
            allowed_area = source_box
            fields["route_owned_foreground_contract_status"] = "review_only_fallback_allowed_area"
            fields["erase_mask_rejected_reason"] = "invalid_route_owned_allowed_area"
    if _xyxy_area(allowed_area) <= 0:
        fields["route_owned_foreground_contract_status"] = "rejected"
        fields["erase_mask_rejected_reason"] = "empty_allowed_area"
        return result

    foreground = _caption_background_foreground_text_mask(
        img_np,
        [source_box[0], source_box[1], source_box[2] - source_box[0], source_box[3] - source_box[1]],
        polygon,
        limit_box=allowed_area,
    )
    if foreground is None or not np.any(foreground):
        fields["route_owned_foreground_contract_status"] = "rejected"
        fields["foreground_mask_method"] = f"{method_prefix}_foreground_text_mask"
        fields["erase_mask_rejected_reason"] = "empty_foreground_text_mask"
        fields["erase_mask_allowed_area"] = list(allowed_area)
        return result

    foreground = _clip_mask_to_box(foreground, allowed_area)
    foreground_stats = mask_stats(foreground) or {}
    foreground_bbox = _mask_stats_box(foreground_stats)
    foreground_pixels = int((foreground > 0).sum())
    if foreground_pixels <= 0 or foreground_bbox is None:
        fields["route_owned_foreground_contract_status"] = "rejected"
        fields["foreground_mask_method"] = f"{method_prefix}_foreground_text_mask"
        fields["erase_mask_rejected_reason"] = "empty_foreground_after_allowed_clip"
        fields["erase_mask_allowed_area"] = list(allowed_area)
        return result

    radius = max(1, min(3, int(halo_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    erase = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    erase = cv2.dilate(erase, kernel, iterations=1)
    erase = _clip_mask_to_box(erase, allowed_area)
    erase = _keep_components_intersecting_seed(erase, foreground)
    erase_stats = mask_stats(erase) or {}
    erase_bbox = _mask_stats_box(erase_stats)
    erase_pixels = int((erase > 0).sum())
    growth_ratio = round(float(erase_pixels) / max(1, foreground_pixels), 3)
    rejection = _route_owned_erase_mask_rejection_reason(
        erase,
        foreground,
        allowed_area,
        growth_ratio,
    )
    artifact_risk = bool(rejection)
    status = fields.get("route_owned_foreground_contract_status") or "generated"
    if rejection:
        status = "review_only_rejected"
    elif status == "not_generated":
        status = "generated"

    fields.update(
        {
            "route_owned_foreground_contract_status": status,
            "foreground_mask_method": f"{method_prefix}_foreground_text_mask",
            "foreground_mask_pixels": foreground_pixels,
            "foreground_mask_bbox": list(foreground_bbox) if foreground_bbox else None,
            "erase_mask_method": f"{method_prefix}_erase_mask_halo_r{radius}",
            "erase_mask_pixels": erase_pixels,
            "erase_mask_bbox": list(erase_bbox) if erase_bbox else None,
            "erase_mask_growth_ratio": growth_ratio,
            "erase_mask_allowed_area": list(allowed_area),
            "erase_mask_rejected_reason": rejection or fields.get("erase_mask_rejected_reason"),
            "erase_mask_artifact_risk": artifact_risk,
            "erase_mask_visual_scope": "route_owned_caption_background_allowed_area",
        }
    )
    result["foreground_mask"] = foreground.astype(np.uint8)
    result["erase_mask"] = erase.astype(np.uint8)
    return result


def _route_owned_foreground_contract(
    img_np,
    region: dict,
    render: dict,
    regions_by_id: dict[str, dict],
    fallback_bbox,
    fallback_polygon,
    allowed_area,
    expected_box,
    *,
    is_caption_mask: bool,
    is_side_caption_mask: bool,
) -> dict[str, Any]:
    fields = _empty_foreground_contract_fields("not_applicable")
    result: dict[str, Any] = {"fields": fields, "foreground_mask": None, "erase_mask": None}
    if not is_caption_mask:
        return result
    route = str(_canonical_region_render_value(region, render, "text_area_route_intent") or "").strip()
    region_type = str(region.get("type", "") or "").strip()
    if is_side_caption_mask or _is_vertical_side_caption(region, render):
        fields["route_owned_foreground_contract_status"] = "protected_not_applicable"
        fields["erase_mask_rejected_reason"] = "side_caption_protected"
        fields["erase_mask_visual_scope"] = "side_caption_preserved"
        return result
    if _is_preserved_source_child(region, render):
        fields["route_owned_foreground_contract_status"] = "protected_not_applicable"
        fields["erase_mask_rejected_reason"] = "sfx_decorative_or_preserve_route"
        fields["erase_mask_visual_scope"] = "protected_route"
        return result
    active_caption_anchor = _is_logical_text_cleanup_anchor(region, render) and (
        route in {"translate_caption", "translate_caption_background"}
        or region_type in {"background_text", "narration_box"}
    )
    if not active_caption_anchor:
        fields["route_owned_foreground_contract_status"] = "not_applicable"
        fields["erase_mask_rejected_reason"] = "not_active_translated_caption_cleanup_anchor"
        return result
    contract_allowed_area = allowed_area
    if contract_allowed_area is None:
        contract_allowed_area = expected_box
    contract = build_route_owned_caption_foreground_contract(
        img_np,
        fallback_bbox,
        fallback_polygon,
        allowed_area=contract_allowed_area,
        halo_px=2,
        method_prefix="route_owned_caption_background",
    )
    foreground = contract.get("foreground_mask")
    erase = contract.get("erase_mask")
    if foreground is None:
        return contract
    for rid in (
        str(item or "")
        for key in (
            "logical_text_source_reconstruction_included_child_region_ids",
            "logical_text_block_transferred_region_ids",
            "logical_text_block_member_region_ids",
        )
        for item in _canonical_region_render_list(region, render, key)
        if str(item or "").strip()
    ):
        member = regions_by_id.get(rid)
        if not isinstance(member, dict) or member is region:
            continue
        member_render = member.get("render") if isinstance(member.get("render"), dict) else {}
        if _is_preserved_source_child(member, member_render):
            continue
        member_bbox = member.get("bbox")
        if not isinstance(member_bbox, (list, tuple)) or len(member_bbox) < 4:
            continue
        member_contract = build_route_owned_caption_foreground_contract(
            img_np,
            member_bbox,
            member.get("polygon"),
            allowed_area=contract_allowed_area,
            halo_px=2,
            method_prefix="route_owned_caption_background_member",
        )
        member_foreground = member_contract.get("foreground_mask")
        member_erase = member_contract.get("erase_mask")
        if member_foreground is not None:
            foreground = cv2.bitwise_or(foreground.astype(np.uint8), member_foreground.astype(np.uint8))
        if erase is not None and member_erase is not None:
            erase = cv2.bitwise_or(erase.astype(np.uint8), member_erase.astype(np.uint8))
    if foreground is not contract.get("foreground_mask") or erase is not contract.get("erase_mask"):
        merged = _foreground_contract_fields_from_masks(
            foreground,
            erase,
            contract_allowed_area,
            status_prefix=str((contract.get("fields") or {}).get("route_owned_foreground_contract_status") or "generated"),
            method_prefix="route_owned_caption_background_merged",
        )
        contract["fields"].update(merged)
        contract["foreground_mask"] = foreground
        contract["erase_mask"] = erase
    return contract


def _empty_foreground_contract_fields(status: str | None = None) -> dict[str, Any]:
    return {
        "route_owned_foreground_contract_status": status,
        "foreground_mask_method": None,
        "foreground_mask_pixels": None,
        "foreground_mask_bbox": None,
        "erase_mask_method": None,
        "erase_mask_pixels": None,
        "erase_mask_bbox": None,
        "erase_mask_growth_ratio": None,
        "erase_mask_allowed_area": None,
        "erase_mask_rejected_reason": None,
        "erase_mask_artifact_risk": None,
        "erase_mask_visual_scope": None,
    }


def _foreground_contract_fields_from_masks(
    foreground,
    erase,
    allowed_area,
    *,
    status_prefix: str,
    method_prefix: str,
) -> dict[str, Any]:
    fields = _empty_foreground_contract_fields(status_prefix)
    foreground_stats = mask_stats(foreground) or {}
    erase_stats = mask_stats(erase) or {}
    foreground_bbox = _mask_stats_box(foreground_stats)
    erase_bbox = _mask_stats_box(erase_stats)
    foreground_pixels = int((foreground > 0).sum()) if foreground is not None else 0
    erase_pixels = int((erase > 0).sum()) if erase is not None else 0
    growth_ratio = round(float(erase_pixels) / max(1, foreground_pixels), 3)
    rejection = _route_owned_erase_mask_rejection_reason(
        erase,
        foreground,
        allowed_area,
        growth_ratio,
    )
    fields.update(
        {
            "route_owned_foreground_contract_status": "review_only_rejected" if rejection else status_prefix,
            "foreground_mask_method": f"{method_prefix}_foreground_text_mask",
            "foreground_mask_pixels": foreground_pixels,
            "foreground_mask_bbox": list(foreground_bbox) if foreground_bbox else None,
            "erase_mask_method": f"{method_prefix}_erase_mask_halo",
            "erase_mask_pixels": erase_pixels,
            "erase_mask_bbox": list(erase_bbox) if erase_bbox else None,
            "erase_mask_growth_ratio": growth_ratio,
            "erase_mask_allowed_area": list(allowed_area) if allowed_area else None,
            "erase_mask_rejected_reason": rejection,
            "erase_mask_artifact_risk": bool(rejection),
            "erase_mask_visual_scope": "route_owned_caption_background_allowed_area",
        }
    )
    return fields


def _caption_background_foreground_text_mask(img_np, bbox, polygon=None, *, limit_box=None):
    if cv2 is None or np is None or img_np is None:
        return None
    dark_mask = _source_glyph_local_mask(
        img_np,
        bbox,
        polygon,
        dilate_px=0,
        limit_box=limit_box,
        bright_context_only=False,
    )
    bright_mask = _bright_caption_glyph_mask(
        img_np,
        bbox,
        polygon,
        dilate_px=0,
        limit_box=limit_box,
    )
    if dark_mask is None:
        return bright_mask
    if bright_mask is None:
        return dark_mask
    foreground = cv2.bitwise_or(dark_mask.astype(np.uint8), bright_mask.astype(np.uint8))
    if not np.any(foreground):
        return None
    return foreground.astype(np.uint8)


def _clip_mask_to_box(mask, box):
    if mask is None or box is None or np is None or cv2 is None:
        return mask
    clipped = np.zeros_like(mask, dtype=np.uint8)
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return clipped
    h, w = mask.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(x0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(y0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return clipped
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def _keep_components_intersecting_seed(mask, seed):
    if cv2 is None or np is None or mask is None or seed is None:
        return mask
    mask = mask.astype(np.uint8)
    seed_bool = seed > 0
    if not np.any(mask) or not np.any(seed_bool):
        return np.zeros_like(mask, dtype=np.uint8)
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, 8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num_labels):
        component = labels == idx
        if np.any(component & seed_bool):
            kept[component] = 255
    return kept


def _route_owned_erase_mask_rejection_reason(erase, foreground, allowed_area, growth_ratio: float) -> str | None:
    if erase is None or foreground is None or allowed_area is None or not np.any(erase):
        return "empty_erase_mask"
    erase_stats = mask_stats(erase) or {}
    erase_bbox = _mask_stats_box(erase_stats)
    if erase_bbox is None:
        return "empty_erase_bbox"
    allowed_area_size = max(1, _xyxy_area(allowed_area))
    erase_bbox_area = max(1, _xyxy_area(erase_bbox))
    erase_pixels = int((erase > 0).sum())
    ax0, ay0, ax1, ay1 = [int(v) for v in allowed_area[:4]]
    ex0, ey0, ex1, ey1 = [int(v) for v in erase_bbox[:4]]
    allowed_w = max(1, ax1 - ax0)
    allowed_h = max(1, ay1 - ay0)
    erase_w = max(1, ex1 - ex0)
    erase_h = max(1, ey1 - ey0)
    if growth_ratio > 4.0:
        return "erase_mask_growth_too_large"
    if erase_bbox_area / allowed_area_size > 0.55:
        return "erase_mask_bbox_too_large_for_allowed_area"
    if erase_w >= int(allowed_w * 0.90) and erase_h >= int(allowed_h * 0.45):
        return "erase_mask_broad_top_band_risk"
    if erase_pixels / allowed_area_size > 0.35:
        return "erase_mask_pixels_too_large_for_allowed_area"
    if _mask_has_long_boundary_component(erase, allowed_area):
        return "erase_mask_panel_or_art_boundary_risk"
    return None


def _mask_has_long_boundary_component(mask, allowed_area) -> bool:
    if cv2 is None or np is None or mask is None or allowed_area is None or not np.any(mask):
        return False
    ax0, ay0, ax1, ay1 = [int(v) for v in allowed_area[:4]]
    allowed_w = max(1, ax1 - ax0)
    allowed_h = max(1, ay1 - ay0)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, 8)
    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        touches = x <= ax0 + 1 or y <= ay0 + 1 or x + w >= ax1 - 1 or y + h >= ay1 - 1
        if touches and area > 80 and (w > allowed_w * 0.60 or h > allowed_h * 0.60):
            return True
    return False


def _same_speech_container_member(anchor: dict, member: dict) -> bool:
    anchor_container = str(anchor.get("text_area_container_id", "") or "")
    member_container = str(member.get("text_area_container_id", "") or "")
    if anchor_container and member_container and anchor_container == member_container:
        return True
    anchor_physical = str(anchor.get("logical_text_physical_bubble_id", "") or "")
    member_physical = str(member.get("logical_text_physical_bubble_id", "") or "")
    return bool(anchor_physical and member_physical and anchor_physical == member_physical)


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


def _normalize_polygons(polygon) -> list[list[list[float]]]:
    if not polygon:
        return []
    if hasattr(polygon, "tolist"):
        try:
            polygon = polygon.tolist()
        except Exception:
            return []
    if isinstance(polygon, dict):
        points = polygon.get("points")
        return _normalize_polygons(points)
    if not isinstance(polygon, (list, tuple)):
        return []
    if polygon and isinstance(polygon[0], (int, float)) and len(polygon) >= 6:
        pts = []
        values = list(polygon)
        for idx in range(0, len(values) - 1, 2):
            pts.append([float(values[idx]), float(values[idx + 1])])
        return [pts] if len(pts) >= 3 else []
    if polygon and isinstance(polygon[0], (list, tuple)):
        first = polygon[0]
        if first and isinstance(first[0], (int, float)):
            pts = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                pts.append([float(point[0]), float(point[1])])
            return [pts] if len(pts) >= 3 else []
        polys = []
        for candidate in polygon:
            polys.extend(_normalize_polygons(candidate))
        return polys
    return []


def _xywh_to_xyxy_tuple(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return (0, 0, 1, 1)
    x, y, w, h = [int(round(float(v))) for v in value[:4]]
    return (x, y, x + max(1, w), y + max(1, h))


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


def _xyxy_area(box) -> int:
    if not box:
        return 0
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    return max(0, x1 - x0) * max(0, y1 - y0)


def _xyxy_intersection_area(a, b) -> int:
    if not a or not b:
        return 0
    ax0, ay0, ax1, ay1 = [int(v) for v in a[:4]]
    bx0, by0, bx1, by1 = [int(v) for v in b[:4]]
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _container_clip_reason(expected_box, allowed_area) -> str | None:
    if expected_box is None or allowed_area is None:
        return None
    area = max(1, _xyxy_area(expected_box))
    inside = _xyxy_intersection_area(expected_box, allowed_area)
    ratio = max(0.0, min(1.0, inside / area))
    if ratio >= 0.995:
        return None
    return f"expected_bbox_clipped_to_container:{round(float(ratio), 3)}"


def _expand_box(box, pad_x: int, pad_y: int, max_w: int, max_h: int):
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    return (
        max(0, x0 - max(0, int(pad_x))),
        max(0, y0 - max(0, int(pad_y))),
        min(int(max_w), x1 + max(0, int(pad_x))),
        min(int(max_h), y1 + max(0, int(pad_y))),
    )


def _mask_stats_box(stats) -> tuple[int, int, int, int] | None:
    if not isinstance(stats, dict):
        return None
    bbox = stats.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _box_luma_stats(img_np, box):
    if cv2 is None or np is None or img_np is None or box is None:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return None
    img_h, img_w = img_np.shape[:2]
    x0 = max(0, min(img_w, x0))
    x1 = max(x0 + 1, min(img_w, x1))
    y0 = max(0, min(img_h, y0))
    y1 = max(y0 + 1, min(img_h, y1))
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(gray.mean()), float(np.percentile(gray, 20)), float(np.percentile(gray, 80))
