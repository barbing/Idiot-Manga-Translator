"""Output-neutral CleanupMask contract builder.

This module turns CleanupJob candidates, CTD foreground pixels, and canonical
TextAreaPlan component authorization into CleanupMask contract records.
SourceGlyphMask evidence is provenance only. This module does not execute
cleanup, choose a backend, run proof, or feed masks into the renderer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised in target conda env
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - exercised in target conda env
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from app.pipeline.cleanup_contracts import (
    CleanupClass,
    CleanupJob,
    CleanupMask,
    TextForegroundSegmentationMask,
)
from app.pipeline.debug_runtime import diagnostic_enabled, write_diagnostic_checkpoint


CLEANUP_MASK_CONTRACT_VERSION = "cleanup_masks_phase2"
MAX_ERASE_GROWTH_RATIO = 3.5
MAX_ERASE_PAGE_PIXEL_RATIO = 0.12
MAX_ERASE_BBOX_PAGE_RATIO = 0.25
MAX_ALLOWED_PAGE_RATIO = 0.85
MAX_CAPTION_FLAT_SMALL_EXCEPTION_GROWTH_RATIO = 4.25
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_PIXELS = 1_200
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_BBOX_AREA = 18_000
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_AREA = 30_000
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_PAGE_RATIO = 0.025
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_ALLOWED_RATIO = 0.08
MIN_SEGMENTATION_READY_PIXELS = 32
MIN_SEGMENTATION_READY_COVERAGE_RATIO = 0.05
MIN_SEGMENTATION_READY_SMALL_COVERAGE_RATIO = 0.035
MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_RATIO = 0.75
MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_SMALL_RATIO = 0.55
FRAGMENT_ONLY_MAX_PIXELS = 320
FRAGMENT_ONLY_MAX_COVERAGE_RATIO = 0.025
PROTECTED_DOMINANT_OVERLAP_RATIO = 0.35
COMPONENT_OWNER_STRONG_PIXELS = 64
AUTH_CLEANUP_COMPONENT_STATES = {
    "cleanup_translate_speech",
    "cleanup_translate_background",
    "cleanup_translate_caption",
}
AUTH_PROTECTED_COMPONENT_STATES = {
    "protect_sfx_decorative",
    "protect_art_or_non_text",
}
AUTH_REVIEW_COMPONENT_STATES = {
    "review_unknown_not_cleanup",
    "outside_cleanup_scope",
}
AUTH_AMBIGUOUS_COMPONENT_STATE = "ambiguous_component_owner"
PROJECTION_READY_STATE = "projection_ready"
MASK_READY_STATE = "mask_ready"


def _cleanup_perf_contract_diag_enabled() -> bool:
    return diagnostic_enabled("MT_CLEANUP_PERF_CONTRACT_DIAGNOSTIC")


def _cleanup_perf_contract_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _cleanup_perf_contract_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_cleanup_perf_contract_json_safe(item) for item in list(value)[:80]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(item) for item in tuple(shape)]}
    return str(value)


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields: Any) -> None:
    if not _cleanup_perf_contract_diag_enabled():
        return
    try:
        write_diagnostic_checkpoint(
            "cleanup_perf_contract_checkpoints.jsonl",
            module="app.pipeline.cleanup_masks",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
        )
    except Exception:
        return


@dataclass(frozen=True)
class CleanupMaskBuildResult:
    """CleanupMask build output without renderer consumption."""

    page_id: str
    version: str
    masks: list[CleanupMask] = field(default_factory=list)
    masks_by_job_id: dict[str, list[CleanupMask]] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    text_foreground_segmentation: dict[str, Any] | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "text_foreground_segmentation": _json_safe(self.text_foreground_segmentation or {}),
            "masks": [_json_safe(mask) for mask in self.masks],
            "masks_by_job_id": {
                str(job_id): [_json_safe(mask) for mask in masks]
                for job_id, masks in self.masks_by_job_id.items()
            },
            "rejected_records": _json_safe(self.rejected_records),
            "protected_records": _json_safe(self.protected_records),
            "skipped_records": _json_safe(self.skipped_records),
            "errors": list(self.errors),
            "summary": {
                "mask_count": len(self.masks),
                "rejected_record_count": len(self.rejected_records),
                "protected_record_count": len(self.protected_records),
                "skipped_record_count": len(self.skipped_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
            },
        }


@dataclass(frozen=True)
class _SourceEvidence:
    region_id: str
    mask_id: str
    audit: dict[str, Any]
    raw: Any


@dataclass(frozen=True)
class _EffectiveMaskBuild:
    foreground: np.ndarray | None
    erase: np.ndarray | None
    status: str
    failure_reason: str
    audit: dict[str, Any]
    rejected: bool = False


@dataclass(frozen=True)
class _OwnershipBinding:
    owner_mask: np.ndarray | None
    protected_mask: np.ndarray | None
    owned_bbox: list[int] | None
    protected_bbox: list[int] | None
    method: str
    status: str
    protected_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _ComponentOwnershipProjection:
    labels: np.ndarray | None
    components: list[dict[str, Any]]
    component_label_by_id: dict[str, int]
    component_mask_slices_by_id: dict[str, tuple[list[int], np.ndarray]] = field(default_factory=dict)


def _segmentation_foreground_mask(segmentation: TextForegroundSegmentationMask | Any | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    if segmentation is None:
        return None, {
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": "text_foreground_segmentation_missing",
        }
    refined = _get_value(segmentation, "refined_mask")
    if refined is None:
        refined = _get_value(segmentation, "mask_refined")
    if refined is None:
        refined = _get_value(segmentation, "mask")
    mask = _to_binary_mask(refined)
    audit = _segmentation_audit(segmentation)
    if mask is None or int(np.count_nonzero(mask)) <= 0:
        audit.update(
            {
                "segmentation_mask_status": "segmentation_mask_missing",
                "segmentation_mask_failure_reason": "refined_segmentation_mask_empty_or_missing",
            }
        )
        return None, audit
    audit.setdefault("segmentation_mask_status", "segmentation_mask_ready")
    audit.setdefault("segmentation_mask_failure_reason", "")
    audit.setdefault("text_pixel_count", int(np.count_nonzero(mask)))
    return mask.astype(np.uint8), audit


def _segmentation_audit(segmentation: Any) -> dict[str, Any]:
    if hasattr(segmentation, "to_audit_dict"):
        raw = segmentation.to_audit_dict()
        if isinstance(raw, Mapping):
            audit = dict(raw)
        else:
            audit = {}
    elif isinstance(segmentation, Mapping):
        audit = dict(segmentation)
    else:
        audit = {}
        for key in (
            "page_id",
            "image_size",
            "raw_mask_ref",
            "refined_mask_ref",
            "threshold_used",
            "provider",
            "backend",
            "runtime_ms",
            "text_pixel_count",
            "connected_component_stats",
            "block_associations",
            "keep_undetected_mask",
            "confidence",
            "provenance",
        ):
            value = _get_value(segmentation, key)
            if value is not None:
                audit[key] = value
    stats = audit.get("connected_component_stats")
    component_count = None
    if isinstance(stats, Mapping):
        component_count = stats.get("component_count")
    return {
        "page_id": audit.get("page_id", ""),
        "image_size": audit.get("image_size"),
        "raw_mask_ref": audit.get("raw_mask_ref", ""),
        "refined_mask_ref": audit.get("refined_mask_ref", ""),
        "threshold_used": audit.get("threshold_used"),
        "provider": audit.get("provider", ""),
        "backend": audit.get("backend", ""),
        "runtime_ms": audit.get("runtime_ms"),
        "text_pixel_count": audit.get("text_pixel_count", 0),
        "connected_component_stats": stats or {},
        "block_associations": list(audit.get("block_associations") or []),
        "keep_undetected_mask": bool(audit.get("keep_undetected_mask", False)),
        "confidence": audit.get("confidence", {}),
        "provenance": audit.get("provenance", {}),
        "segmentation_provider": audit.get("provider", ""),
        "segmentation_mask_ref": audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": audit.get("text_pixel_count", 0),
        "segmentation_component_count": component_count,
    }


def build_cleanup_masks(
    *,
    page_id: str,
    job_candidates: Sequence[CleanupJob],
    source_glyph_masks: Any,
    image_size: tuple[int, int] | None = None,
    source_image_path: str | Path | None = None,
    source_image: Any | None = None,
    text_foreground_segmentation: TextForegroundSegmentationMask | Any | None = None,
    page_region_records: Sequence[Mapping[str, Any]] | None = None,
    component_authorization_map: Any | None = None,
) -> CleanupMaskBuildResult:
    """Build CleanupMask contracts from segmentation foreground and source-glyph provenance."""

    started = time.time()
    evidence_records = _index_source_evidence(source_glyph_masks)
    segmentation_mask, segmentation_audit = _segmentation_foreground_mask(text_foreground_segmentation)
    source_np, source_error = _source_image_array(source_image=source_image, source_image_path=source_image_path)
    _cleanup_perf_contract_checkpoint(
        "cleanup_mask_build",
        "module_start",
        page_id=page_id,
        job_count=len(job_candidates or []),
        source_evidence_count=len(evidence_records),
        segmentation_available=segmentation_mask is not None,
        segmentation_text_pixels=segmentation_audit.get("text_pixel_count", 0),
        source_image_available=source_np is not None,
        source_image_error=source_error,
    )
    evidence_by_id: dict[str, _SourceEvidence] = {}
    for evidence in evidence_records:
        if not evidence.mask_id:
            continue
        existing = evidence_by_id.get(evidence.mask_id)
        if existing is None or (_evidence_has_raw_mask(evidence) and not _evidence_has_raw_mask(existing)):
            evidence_by_id[evidence.mask_id] = evidence
    evidence_by_region: dict[str, list[_SourceEvidence]] = {}
    for evidence in evidence_records:
        if evidence.region_id:
            evidence_by_region.setdefault(evidence.region_id, []).append(evidence)
    region_records = _index_region_records(page_region_records)
    if component_authorization_map is None:
        segmentation_audit = {
            **segmentation_audit,
            "component_authorization_map_status": "missing",
            "component_projection_method": "text_area_component_authorization_map_required",
            "component_ownership_inventory": [],
            "component_ownership_state_counts": {},
            "segmentation_component_count": 0,
        }
        return CleanupMaskBuildResult(
            page_id=page_id,
            version=CLEANUP_MASK_CONTRACT_VERSION,
            masks=[],
            masks_by_job_id={},
            rejected_records=[
                {
                    **_base_job_record(page_id, job),
                    "reason": "component_authorization_map_missing",
                }
                for job in (job_candidates or [])
            ],
            protected_records=[],
            skipped_records=[],
            errors=["component_authorization_map_missing"],
            text_foreground_segmentation=segmentation_audit,
        )

    component_projection = _component_projection_from_authorization_map(
        page_id=page_id,
        segmentation_mask=segmentation_mask,
        component_authorization_map=component_authorization_map,
    )
    segmentation_audit = {
        **segmentation_audit,
        "component_ownership_inventory": component_projection.components,
        "component_ownership_state_counts": _component_ownership_state_counts(component_projection.components),
        "segmentation_component_count": len(component_projection.components),
        "component_projection_method": "text_area_component_authorization_map",
        "component_authorization_map_status": "ready",
        "component_authorization_map": _component_authorization_map_audit(component_authorization_map),
    }

    masks: list[CleanupMask] = []
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    rejected_records: list[dict[str, Any]] = []
    protected_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for job in job_candidates or []:
        job_started = time.time()
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "start",
            page_id=page_id,
            cleanup_job_id=str(getattr(job, "cleanup_job_id", "") or ""),
            target_region_count=len(getattr(job, "target_region_ids", []) or []),
            required_source_glyph_count=len(_required_source_glyph_ids(job)),
        )
        try:
            base = _base_job_record(page_id, job)
            protection_reason = _job_protection_reason(job)
            if protection_reason:
                protected_records.append({**base, "reason": protection_reason})
                continue

            matched = _matching_evidence(job, evidence_by_id, evidence_by_region)
            required_source_ids = _required_source_glyph_ids(job)
            missing_required_ids: list[str] = []
            if required_source_ids:
                matched = [source for source in matched if source.mask_id in required_source_ids]
                matched_by_id = {source.mask_id for source in matched if source.mask_id}
                missing_required_ids = [mask_id for mask_id in required_source_ids if mask_id not in matched_by_id]

            source = _primary_source_evidence(job, matched) if matched else None
            allowed = (
                _allowed_area_from_regions(job, region_records)
                or _job_allowed_cleanup_area(job)
                or _allowed_area_from_evidence(matched)
            )
            if allowed is None:
                rejected_records.append({**base, "reason": "allowed_cleanup_area_missing_or_invalid"})
                continue

            allowed_rejection = _allowed_area_rejection(allowed, image_size)
            if allowed_rejection:
                rejected_records.append({**base, "reason": allowed_rejection, "allowed_area": allowed})
                continue

            source_seed_foreground, _source_seed_erase, foreground_source_keys, erase_source_keys, consumed_source_ids, missing_foreground_source_ids, used_transitional_erase = _union_masks_from_evidence(
                matched,
                allowed=allowed,
            )
            mask_shape = (
                segmentation_mask.shape
                if segmentation_mask is not None
                else (source_seed_foreground.shape if source_seed_foreground is not None else None)
            )
            ownership_binding = _build_ownership_binding(
                job=job,
                allowed=allowed,
                shape=mask_shape,
                region_records=region_records,
            )
            effective = _build_component_projected_text_mask(
                job=job,
                segmentation_mask=segmentation_mask,
                segmentation_audit=segmentation_audit,
                seed_foreground=source_seed_foreground,
                allowed=allowed,
                matched=matched,
                ownership_binding=ownership_binding,
                component_projection=component_projection,
                missing_source_glyph_mask_ids=missing_required_ids,
            )
            if effective.foreground is None:
                if str(effective.audit.get("component_projection_method") or "") == "text_area_component_authorization_map":
                    rejected_records.append(
                        {
                            **base,
                            "reason": effective.failure_reason or "cleanup_mask_missing_no_owned_segmentation_components",
                            "required_source_glyph_mask_ids": required_source_ids,
                            "missing_source_glyph_mask_ids": missing_required_ids,
                            **effective.audit,
                        }
                    )
                    continue
                if source_seed_foreground is None:
                    rejected_records.append(
                        {
                            **base,
                            "reason": effective.failure_reason
                            or "segmentation_mask_missing",
                            "required_source_glyph_mask_ids": required_source_ids,
                            "missing_source_glyph_mask_ids": missing_foreground_source_ids,
                            **effective.audit,
                        }
                    )
                    continue
                fallback = _build_effective_text_mask(
                    job=job,
                    source_np=source_np,
                    source_error=source_error,
                    seed_foreground=source_seed_foreground,
                    allowed=allowed,
                    matched=matched,
                )
                fallback_audit = {
                    **fallback.audit,
                    "ownership_binding_status": effective.audit.get("ownership_binding_status", ""),
                    "ownership_binding_method": effective.audit.get("ownership_binding_method", ""),
                    "cleanup_owned_unit_bbox": effective.audit.get("cleanup_owned_unit_bbox"),
                    "cleanup_owned_unit_mask_ref": effective.audit.get("cleanup_owned_unit_mask_ref", ""),
                    "protected_mask_ref": effective.audit.get("protected_mask_ref", ""),
                    "protected_overlap_pixels": effective.audit.get("protected_overlap_pixels"),
                    "segmentation_pixels_before_binding": effective.audit.get("segmentation_pixels_before_binding"),
                    "segmentation_pixels_after_owner_clip": effective.audit.get("segmentation_pixels_after_owner_clip"),
                    "segmentation_pixels_after_protection_subtract": effective.audit.get("segmentation_pixels_after_protection_subtract"),
                    "sourceglyph_overlap_pixels": effective.audit.get("sourceglyph_overlap_pixels"),
                    "sourceglyph_overlap_ratio": effective.audit.get("sourceglyph_overlap_ratio"),
                    "segmentation_outside_sourceglyph_pixels": effective.audit.get("segmentation_outside_sourceglyph_pixels"),
                    "effective_coverage_ratio": fallback.audit.get("text_block_coverage_estimate", 0.0),
                    "effective_coverage_status": effective.audit.get("effective_coverage_status", ""),
                    "segmentation_mask_status": effective.audit.get("segmentation_mask_status", ""),
                    "segmentation_mask_failure_reason": effective.failure_reason
                    or effective.audit.get("segmentation_mask_failure_reason", ""),
                    "mask_completion_method": (
                        f"local_contrast_fallback_after_{effective.audit.get('segmentation_mask_status') or 'segmentation_unavailable'}"
                    ),
                    "clean_mask_authority": "diagnostic_non_segmentation_fallback",
                    "non_segmentation_or_local_fallback_used": True,
                    "projection_quality_state": "diagnostic_non_segmentation_fallback",
                    "projection_quality_reasons": [
                        "diagnostic_non_segmentation_fallback_not_executable"
                    ],
                    "mask_readiness_state": "mask_not_ready",
                    "mask_readiness_failure_reason": "diagnostic_non_segmentation_fallback_not_executable",
                    "bbox_executable_foreground_detected": False,
                    "page_level_executable_foreground_detected": False,
                    "clean_mask_state": "cleanup_mask_unresolved_after_segmentation",
                    "clean_mask_failure_reason": effective.failure_reason
                    or fallback.failure_reason
                    or "segmentation_unavailable_local_contrast_fallback",
                }
                effective = _EffectiveMaskBuild(
                    foreground=fallback.foreground,
                    erase=fallback.erase,
                    status=(
                        "cleanup_mask_unresolved_after_segmentation"
                        if fallback.foreground is not None and fallback.erase is not None
                        else fallback.status
                    ),
                    failure_reason=effective.failure_reason
                    or fallback.failure_reason
                    or "segmentation_unavailable_local_contrast_fallback",
                    audit=fallback_audit,
                    rejected=fallback.rejected,
                )
            if effective.foreground is None or effective.erase is None:
                rejected_records.append(
                    {
                        **base,
                        "reason": effective.failure_reason or "effective_mask_failed_insufficient_evidence",
                        **effective.audit,
                    }
                )
                continue
            foreground = effective.foreground
            erase = effective.erase
            foreground_pixels = int(np.count_nonzero(foreground))
            erase_pixels = int(np.count_nonzero(erase))
            if erase_pixels <= 0:
                rejected_records.append({**base, "reason": "erase_mask_empty_after_allowed_area_clip"})
                continue

            foreground_bbox = _mask_bbox(foreground)
            erase_bbox = _mask_bbox(erase)
            if erase_bbox is None:
                rejected_records.append({**base, "reason": "erase_mask_bbox_empty"})
                continue
            mask_allowed_area = _valid_bbox(effective.audit.get("component_projected_execution_scope_bbox")) or allowed

            growth_ratio = _growth_ratio(erase_pixels, foreground_pixels)
            artifact_risk = (
                "source_glyph_reported_artifact_risk"
                if any(_truthy(_first_present(item.audit, "cleanup_visual_artifact_risk", default=False)) for item in matched)
                else ""
            )
            exception_reason = _caption_flat_small_mask_growth_exception_reason(
                job=job,
                source=source,
                allowed=allowed,
                erase_bbox=erase_bbox,
                erase_pixels=erase_pixels,
                foreground_pixels=foreground_pixels,
                growth_ratio=growth_ratio,
                image_size=image_size,
                artifact_risk=artifact_risk,
            )
            if (
                not exception_reason
                and growth_ratio > MAX_ERASE_GROWTH_RATIO
                and _component_authorized_growth_exception_applies(effective)
            ):
                exception_reason = "text_area_component_authorization_map_growth_exception"
            broad_rejection = _broad_mask_rejection(
                allowed=mask_allowed_area,
                erase_bbox=erase_bbox,
                erase_pixels=erase_pixels,
                growth_ratio=growth_ratio,
                image_size=image_size,
                allow_growth_exception=bool(exception_reason),
            )
            component_projected = _effective_uses_component_projection(effective)
            if broad_rejection and not (component_projected and effective.rejected):
                rejected_records.append(
                    {
                        **base,
                        "reason": broad_rejection,
                        "allowed_area": allowed,
                        "erase_mask_bbox": erase_bbox,
                        "erase_mask_pixels": erase_pixels,
                        "foreground_mask_pixels": foreground_pixels,
                        "growth_ratio": growth_ratio,
                        "mask_contract_exception_reason": exception_reason,
                        "effective_mask_status": effective.status,
                        "effective_mask_failure_reason": broad_rejection,
                        **effective.audit,
                    }
                )
                continue

            completion_method = str(effective.audit.get("mask_completion_method") or "")
            clean_authority = str(effective.audit.get("clean_mask_authority") or "")
            if completion_method.startswith("text_foreground_segmentation"):
                mask_source = "cleanup_mask_from_text_foreground_segmentation"
            elif (
                completion_method == "text_area_component_authorization_map_projection"
                or clean_authority == "text_area_component_authorization_map"
            ):
                mask_source = "cleanup_mask_from_text_area_component_authorization_map"
            else:
                mask_source = "cleanup_mask_diagnostic_non_segmentation_fallback"
            seed_method = _mask_method_union(matched, erase_source_keys or foreground_source_keys)
            mask_method = (
                f"{completion_method}|seed:{seed_method}"
                if completion_method
                else seed_method
            )
            visual_scope = _cleanup_visual_scope_for_unit(
                required_source_ids=required_source_ids,
                consumed_source_ids=consumed_source_ids,
            )
            cleanup_mask = CleanupMask(
                cleanup_mask_id=f"cmask_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}",
                cleanup_job_id=str(job.cleanup_job_id),
                parent_execution_bundle_id=str(
                    getattr(job, "parent_execution_bundle_id", "")
                    or getattr(job, "parent_logical_text_unit_id", "")
                    or ""
                ),
                parent_logical_text_unit_id=str(getattr(job, "parent_logical_text_unit_id", "") or ""),
                text_block_root_id=str(getattr(job, "text_block_root_id", "") or ""),
                text_area_container_id=str(getattr(job, "text_area_container_id", "") or ""),
                target_region_ids=list(getattr(job, "target_region_ids", []) or []),
                represented_region_ids=list(
                    getattr(job, "cleanup_unit_child_region_ids", None)
                    or getattr(job, "target_region_ids", [])
                    or []
                ),
                foreground_mask_source_id=consumed_source_ids[0] if consumed_source_ids else None,
                foreground_mask_source_ids=consumed_source_ids,
                consumed_source_glyph_mask_ids=consumed_source_ids,
                missing_source_glyph_mask_ids=[],
                foreground_mask_bbox=foreground_bbox,
                foreground_mask_pixels=foreground_pixels,
                erase_mask_bbox=erase_bbox,
                erase_mask_pixels=erase_pixels,
                allowed_area=mask_allowed_area,
                growth_ratio=growth_ratio,
                mask_source=mask_source,
                mask_method=mask_method,
                rejection_reason=effective.failure_reason if effective.rejected else "",
                mask_contract_exception_reason=exception_reason,
                artifact_risk=artifact_risk,
                visual_scope=visual_scope,
                protected=False,
                protection_reason="",
                effective_mask_status=effective.status,
                effective_mask_failure_reason=effective.failure_reason,
                seed_foreground_pixels=effective.audit.get("seed_foreground_pixels"),
                completed_foreground_pixels=effective.audit.get("completed_foreground_pixels"),
                component_count_before=effective.audit.get("component_count_before"),
                component_count_after=effective.audit.get("component_count_after"),
                largest_component_pixels_before=effective.audit.get("largest_component_pixels_before"),
                largest_component_pixels_after=effective.audit.get("largest_component_pixels_after"),
                text_block_coverage_estimate=effective.audit.get("text_block_coverage_estimate"),
                bbox_fill_ratio_before=effective.audit.get("bbox_fill_ratio_before"),
                bbox_fill_ratio_after=effective.audit.get("bbox_fill_ratio_after"),
                analysis_scope_bbox=effective.audit.get("analysis_scope_bbox"),
                executable_erase_bbox=effective.audit.get("executable_erase_bbox"),
                mask_completion_method=effective.audit.get("mask_completion_method", ""),
                polarity_mode=effective.audit.get("polarity_mode", ""),
                source_seed_mask_ids=consumed_source_ids,
                recovered_component_count=effective.audit.get("recovered_component_count"),
                rejected_component_count=effective.audit.get("rejected_component_count"),
                rejected_component_reasons=effective.audit.get("rejected_component_reasons", []),
                segmentation_mask_status=effective.audit.get("segmentation_mask_status", ""),
                segmentation_mask_failure_reason=effective.audit.get("segmentation_mask_failure_reason", ""),
                segmentation_provider=effective.audit.get("segmentation_provider", ""),
                segmentation_mask_ref=effective.audit.get("segmentation_mask_ref", ""),
                segmentation_text_pixels=effective.audit.get("segmentation_text_pixels"),
                segmentation_component_count=effective.audit.get("segmentation_component_count"),
                segmentation_binding_method=effective.audit.get("segmentation_binding_method", ""),
                segmentation_block_associations=effective.audit.get("segmentation_block_associations", []),
                ownership_binding_status=effective.audit.get("ownership_binding_status", ""),
                ownership_binding_method=effective.audit.get("ownership_binding_method", ""),
                cleanup_owned_unit_bbox=effective.audit.get("cleanup_owned_unit_bbox"),
                cleanup_owned_unit_mask_ref=effective.audit.get("cleanup_owned_unit_mask_ref", ""),
                protected_mask_ref=effective.audit.get("protected_mask_ref", ""),
                protected_overlap_pixels=effective.audit.get("protected_overlap_pixels"),
                segmentation_pixels_before_binding=effective.audit.get("segmentation_pixels_before_binding"),
                segmentation_pixels_after_owner_clip=effective.audit.get("segmentation_pixels_after_owner_clip"),
                segmentation_pixels_after_protection_subtract=effective.audit.get("segmentation_pixels_after_protection_subtract"),
                sourceglyph_overlap_pixels=effective.audit.get("sourceglyph_overlap_pixels"),
                sourceglyph_overlap_ratio=effective.audit.get("sourceglyph_overlap_ratio"),
                segmentation_outside_sourceglyph_pixels=effective.audit.get("segmentation_outside_sourceglyph_pixels"),
                effective_coverage_ratio=effective.audit.get("effective_coverage_ratio"),
                effective_coverage_status=effective.audit.get("effective_coverage_status", ""),
                component_ownership_status=effective.audit.get("component_ownership_status", ""),
                owned_component_ids=effective.audit.get("owned_component_ids", []),
                protected_component_ids=effective.audit.get("protected_component_ids", []),
                ambiguous_component_ids=effective.audit.get("ambiguous_component_ids", []),
                unowned_component_ids=effective.audit.get("unowned_component_ids", []),
                component_projection_method=effective.audit.get("component_projection_method", ""),
                owned_component_pixel_count=effective.audit.get("owned_component_pixel_count"),
                protected_component_pixel_count=effective.audit.get("protected_component_pixel_count"),
                ambiguous_component_pixel_count=effective.audit.get("ambiguous_component_pixel_count"),
                sourceglyph_overlap_component_ids=effective.audit.get("sourceglyph_overlap_component_ids", []),
                sourceglyph_missing_component_ids=effective.audit.get("sourceglyph_missing_component_ids", []),
                ownership_projection_failure_reason=effective.audit.get("ownership_projection_failure_reason", ""),
                effective_component_coverage_ratio=effective.audit.get("effective_component_coverage_ratio"),
                owned_segmentation_pixels=effective.audit.get("owned_segmentation_pixels"),
                executable_foreground_pixels=effective.audit.get("executable_foreground_pixels"),
                committed_cleanup_mask_pixels=effective.audit.get("committed_cleanup_mask_pixels"),
                owned_segmentation_to_executable_ratio=effective.audit.get("owned_segmentation_to_executable_ratio"),
                owned_segmentation_to_commit_ratio=effective.audit.get("owned_segmentation_to_commit_ratio"),
                ready_but_sparse_violation=bool(effective.audit.get("ready_but_sparse_violation", False)),
                sourceglyph_executable_influence_detected=bool(
                    effective.audit.get("sourceglyph_executable_influence_detected", False)
                ),
                segmentation_contract_override_detected=bool(
                    effective.audit.get("segmentation_contract_override_detected", False)
                ),
                clean_mask_authority=effective.audit.get("clean_mask_authority", ""),
                non_segmentation_or_local_fallback_used=bool(
                    effective.audit.get("non_segmentation_or_local_fallback_used", False)
                ),
                bbox_executable_foreground_detected=bool(effective.audit.get("bbox_executable_foreground_detected", False)),
                page_level_executable_foreground_detected=bool(effective.audit.get("page_level_executable_foreground_detected", False)),
                clean_mask_foreground_pixels=effective.audit.get("clean_mask_foreground_pixels"),
                clean_mask_erase_pixels=effective.audit.get("clean_mask_erase_pixels"),
                foreground_to_owned_segmentation_ratio=effective.audit.get("foreground_to_owned_segmentation_ratio"),
                erase_to_foreground_ratio=effective.audit.get("erase_to_foreground_ratio"),
                protected_component_pixels_removed=effective.audit.get("protected_component_pixels_removed"),
                ambiguous_component_pixels=effective.audit.get("ambiguous_component_pixels"),
                unowned_component_pixels=effective.audit.get("unowned_component_pixels"),
                clean_mask_state=effective.audit.get("clean_mask_state", ""),
                clean_mask_failure_reason=effective.audit.get("clean_mask_failure_reason", ""),
                cleanup_authorization=effective.audit.get("cleanup_authorization", ""),
                authorization_source_stage=effective.audit.get("authorization_source_stage", ""),
                semantic_authorization_state=effective.audit.get("semantic_authorization_state", ""),
                projection_quality_state=effective.audit.get("projection_quality_state", ""),
                projection_quality_reasons=effective.audit.get("projection_quality_reasons", []),
                mask_readiness_state=effective.audit.get("mask_readiness_state", ""),
                mask_readiness_failure_reason=effective.audit.get("mask_readiness_failure_reason", ""),
                semantic_authority_owner=effective.audit.get("semantic_authority_owner", ""),
                projection_owner=effective.audit.get("projection_owner", ""),
                foreground_outside_allowed_pixels=effective.audit.get("foreground_outside_allowed_pixels"),
                upstream_container_mismatch_pixels=effective.audit.get("upstream_container_mismatch_pixels"),
                upstream_container_mismatch_ratio=effective.audit.get("upstream_container_mismatch_ratio"),
                green_to_foreground_component_coverage_ratio=effective.audit.get("green_to_foreground_component_coverage_ratio"),
                green_to_erase_component_coverage_ratio=effective.audit.get("green_to_erase_component_coverage_ratio"),
                ctd_refined_segmentation_mask_ref=effective.audit.get("ctd_refined_segmentation_mask_ref", ""),
                cleanup_owned_segmentation_foreground_mask_ref=effective.audit.get("cleanup_owned_segmentation_foreground_mask_ref", ""),
                protected_component_mask_ref=effective.audit.get("protected_component_mask_ref", ""),
                clean_mask_foreground_ref=effective.audit.get("clean_mask_foreground_ref", ""),
                clean_mask_erase_ref=effective.audit.get("clean_mask_erase_ref", ""),
                component_ownership_overlay_ref=effective.audit.get("component_ownership_overlay_ref", ""),
                rejected_component_overlay_ref=effective.audit.get("rejected_component_overlay_ref", ""),
                sourceglyph_overlap_overlay_ref=effective.audit.get("sourceglyph_overlap_overlay_ref", ""),
                clean_mask_annotation_ref=effective.audit.get("clean_mask_annotation_ref", ""),
                foreground_mask=foreground.copy(),
                erase_mask=erase.copy(),
            )
            masks.append(cleanup_mask)
            masks_by_job_id.setdefault(str(job.cleanup_job_id), []).append(cleanup_mask)
            _cleanup_perf_contract_checkpoint(
                "cleanup_mask_job",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                matched_evidence_count=len(matched),
                consumed_source_glyph_count=len(consumed_source_ids),
                consumed_source_glyph_mask_ids=consumed_source_ids,
                missing_source_glyph_count=0,
                foreground_pixels=foreground_pixels,
                erase_pixels=erase_pixels,
                allowed_area=allowed,
                visual_scope=visual_scope,
                growth_ratio=growth_ratio,
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    unbound_contract_record = _build_unbound_semantic_cleanup_contract_error_record(
        page_id=page_id,
        component_projection=component_projection,
        existing_masks=masks,
    )
    if unbound_contract_record is not None:
        rejected_records.append(unbound_contract_record)
        errors.append("cleanup_job_binding_contract_error")

    for record in rejected_records:
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "rejected",
            page_id=page_id,
            cleanup_job_id=str(record.get("cleanup_job_id") or ""),
            region_id=str(record.get("region_id") or ""),
            reason=str(record.get("reason") or ""),
            required_source_glyph_mask_ids=record.get("required_source_glyph_mask_ids", []),
            missing_source_glyph_mask_ids=record.get("missing_source_glyph_mask_ids", []),
            foreground_pixels=record.get("foreground_mask_pixels", ""),
            erase_pixels=record.get("erase_mask_pixels", ""),
            allowed_area=record.get("allowed_area", ""),
            erase_mask_bbox=record.get("erase_mask_bbox", ""),
            growth_ratio=record.get("growth_ratio", ""),
        )
    for record in protected_records:
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "protected",
            page_id=page_id,
            cleanup_job_id=str(record.get("cleanup_job_id") or ""),
            region_id=str(record.get("region_id") or ""),
            reason=str(record.get("reason") or ""),
        )
    _cleanup_perf_contract_checkpoint(
        "cleanup_mask_build",
        "module_end",
        page_id=page_id,
        mask_count=len(masks),
        rejected_count=len(rejected_records),
        protected_count=len(protected_records),
        skipped_count=len(skipped_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return CleanupMaskBuildResult(
        page_id=page_id,
        version=CLEANUP_MASK_CONTRACT_VERSION,
        masks=masks,
        masks_by_job_id=masks_by_job_id,
        rejected_records=rejected_records,
        protected_records=protected_records,
        skipped_records=skipped_records,
        errors=errors,
        text_foreground_segmentation=segmentation_audit,
    )


def _index_source_evidence(source_glyph_masks: Any) -> list[_SourceEvidence]:
    records: list[_SourceEvidence] = []

    masks_by_region = _get_mapping(source_glyph_masks, "masks_by_region")
    if masks_by_region:
        for region_id, value in masks_by_region.items():
            for item in _sequence_or_single(value):
                records.append(_source_evidence(str(region_id or ""), item))

    for key in ("source_glyph_masks", "source_glyph_mask_coverage_records", "coverage_records"):
        value = _get_value(source_glyph_masks, key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                evidence = _source_evidence("", item)
                if evidence.region_id or evidence.mask_id:
                    records.append(evidence)

    deduped: list[_SourceEvidence] = []
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        key = (record.region_id, record.mask_id, id(record.raw))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _source_evidence(region_id: str, raw: Any) -> _SourceEvidence:
    audit = _audit_mapping(raw)
    if region_id and "region_id" not in audit:
        audit = {"region_id": region_id, **audit}
    source_region_id = str(_first_present(audit, "region_id", "target_region_id", default=region_id) or "")
    mask_id = str(_first_present(audit, "source_glyph_mask_id", "mask_id", "id", default="") or "")
    return _SourceEvidence(region_id=source_region_id, mask_id=mask_id, audit=audit, raw=raw)


def _audit_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_audit_dict"):
        audit = value.to_audit_dict()
        if isinstance(audit, Mapping):
            return dict(audit)
    if is_dataclass(value):
        output: dict[str, Any] = {}
        for name in (
            "region_id",
            "mask_id",
            "source_glyph_mask_id",
            "generation_method",
            "foreground_mask_method",
            "foreground_mask_pixels",
            "foreground_mask_bbox",
            "erase_mask_method",
            "erase_mask_pixels",
            "erase_mask_bbox",
            "erase_mask_growth_ratio",
            "erase_mask_allowed_area",
            "cleanup_allowed_area",
            "cleanup_visual_artifact_risk",
            "erase_mask_artifact_risk",
        ):
            if hasattr(value, name):
                output[name] = getattr(value, name)
        return output
    return {}


def _matching_evidence(
    job: CleanupJob,
    evidence_by_id: Mapping[str, _SourceEvidence],
    evidence_by_region: Mapping[str, list[_SourceEvidence]],
) -> list[_SourceEvidence]:
    matches: list[_SourceEvidence] = []
    seen: set[tuple[str, str]] = set()
    for mask_id in getattr(job, "source_glyph_mask_ids", []) or []:
        evidence = evidence_by_id.get(str(mask_id))
        if evidence is None:
            continue
        key = (evidence.region_id, evidence.mask_id)
        if key not in seen:
            matches.append(evidence)
            seen.add(key)
    for region_id in getattr(job, "target_region_ids", []) or []:
        for evidence in evidence_by_region.get(str(region_id), []):
            key = (evidence.region_id, evidence.mask_id)
            if key not in seen:
                matches.append(evidence)
                seen.add(key)
    return matches


def _required_source_glyph_ids(job: CleanupJob) -> list[str]:
    ids: list[str] = []
    for attr in ("cleanup_unit_required_source_glyph_mask_ids", "source_glyph_mask_ids"):
        for item in getattr(job, attr, []) or []:
            text = str(item or "")
            if text and text not in ids:
                ids.append(text)
    return ids


def _cleanup_visual_scope_for_unit(
    *,
    required_source_ids: Sequence[str],
    consumed_source_ids: Sequence[str],
) -> str:
    ids: list[str] = []
    for item in list(required_source_ids or []) + list(consumed_source_ids or []):
        text = str(item or "")
        if text and text not in ids:
            ids.append(text)
    if len(ids) <= 1:
        return "source_glyph_local"
    return "source_glyph_union"


def _primary_source_evidence(job: CleanupJob, matched: Sequence[_SourceEvidence]) -> _SourceEvidence:
    anchor_region_id = str(getattr(job, "cleanup_unit_anchor_region_id", "") or "")
    for evidence in matched:
        if anchor_region_id and evidence.region_id == anchor_region_id:
            return evidence
    for evidence in matched:
        if evidence.mask_id in getattr(job, "source_glyph_mask_ids", []) or []:
            return evidence
    return matched[0]


def _allowed_area_from_evidence(matched: Sequence[_SourceEvidence]) -> list[int] | None:
    candidates: list[list[int]] = []
    for evidence in matched:
        bbox = _valid_bbox(
            _first_present(
                evidence.audit,
                "cleanup_allowed_area",
                "erase_mask_allowed_area",
                "allowed_cleanup_area",
                "text_area_container_bbox",
                "source_glyph_erasure_bbox",
                "source_glyph_bbox",
                "bbox",
                default=None,
            )
        )
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return _union_bboxes(candidates)
    return None


def _allowed_area_from_regions(job: CleanupJob, region_records: Mapping[str, Mapping[str, Any]]) -> list[int] | None:
    candidates: list[list[int]] = []
    for region_id in getattr(job, "target_region_ids", []) or []:
        region = region_records.get(str(region_id))
        bbox = _region_bbox(region) if isinstance(region, Mapping) else None
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return _union_bboxes(candidates)
    return None


def _job_allowed_cleanup_area(job: CleanupJob) -> list[int] | None:
    value = getattr(job, "allowed_cleanup_area", None)
    bbox = _xywh_bbox(value)
    if bbox is not None:
        return bbox
    return _valid_bbox(value)


def _union_masks_from_evidence(
    matched: Sequence[_SourceEvidence],
    *,
    allowed: list[int],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    list[str],
    list[str],
    list[str],
    list[str],
    bool,
]:
    foreground_union: np.ndarray | None = None
    erase_union: np.ndarray | None = None
    foreground_source_keys: list[str] = []
    erase_source_keys: list[str] = []
    consumed_source_ids: list[str] = []
    missing_foreground_source_ids: list[str] = []
    used_transitional_erase = False

    for evidence in matched:
        foreground, foreground_source = _binary_mask_from_evidence(
            evidence,
            keys=("foreground_mask", "mask"),
        )
        if foreground is None:
            missing_foreground_source_ids.append(evidence.mask_id or evidence.region_id or "unknown_source_glyph")
            continue
        if foreground_union is None:
            foreground_union = np.zeros_like(foreground, dtype=np.uint8)
            erase_union = np.zeros_like(foreground, dtype=np.uint8)
        else:
            foreground = _coerce_mask_shape(foreground, foreground_union.shape)
        foreground = _clip_mask_to_bbox(foreground, allowed)
        foreground_union = np.maximum(foreground_union, foreground)
        if foreground_source and foreground_source not in foreground_source_keys:
            foreground_source_keys.append(foreground_source)

        erase, erase_source = _binary_mask_from_evidence(evidence, keys=("erase_mask",))
        if erase is None:
            erase = foreground.copy()
            erase_source = foreground_source
        else:
            erase = _coerce_mask_shape(erase, foreground_union.shape)
            used_transitional_erase = True
        erase = _clip_mask_to_bbox(erase, allowed)
        erase_union = np.maximum(erase_union, erase)
        if erase_source and erase_source not in erase_source_keys:
            erase_source_keys.append(erase_source)
        if evidence.mask_id and evidence.mask_id not in consumed_source_ids:
            consumed_source_ids.append(evidence.mask_id)

    return (
        foreground_union,
        erase_union,
        foreground_source_keys,
        erase_source_keys,
        consumed_source_ids,
        missing_foreground_source_ids,
        used_transitional_erase,
    )


def _build_component_projected_text_mask(
    *,
    job: CleanupJob,
    segmentation_mask: np.ndarray | None,
    segmentation_audit: Mapping[str, Any],
    seed_foreground: np.ndarray | None,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
    ownership_binding: _OwnershipBinding,
    component_projection: _ComponentOwnershipProjection,
    missing_source_glyph_mask_ids: Sequence[str],
) -> _EffectiveMaskBuild:
    base_audit = {
        "segmentation_provider": segmentation_audit.get("segmentation_provider") or segmentation_audit.get("provider", ""),
        "segmentation_mask_ref": segmentation_audit.get("segmentation_mask_ref") or segmentation_audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": segmentation_audit.get("segmentation_text_pixels") or segmentation_audit.get("text_pixel_count", 0),
        "segmentation_component_count": len(component_projection.components),
        "segmentation_block_associations": [],
        "component_projection_method": "text_area_component_authorization_map",
    }
    base_audit.update(_ownership_base_audit(ownership_binding))
    if segmentation_mask is None or component_projection.labels is None:
        audit = {
            **base_audit,
            "seed_foreground_pixels": int(np.count_nonzero(seed_foreground > 0)) if seed_foreground is not None else 0,
            "completed_foreground_pixels": 0,
            "component_count_before": 0,
            "component_count_after": 0,
            "largest_component_pixels_before": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_before": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "segmentation_component_projection_missing",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["segmentation_mask_missing"],
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": str(
                segmentation_audit.get("segmentation_mask_failure_reason") or "text_foreground_segmentation_missing"
            ),
            "segmentation_binding_method": "not_attempted",
            "component_ownership_status": "cleanup_mask_structural_invalid",
            "ownership_projection_failure_reason": "segmentation_mask_missing",
            "owned_component_ids": [],
            "protected_component_ids": [],
            "ambiguous_component_ids": [],
            "unowned_component_ids": [],
            "owned_component_pixel_count": 0,
            "protected_component_pixel_count": 0,
            "ambiguous_component_pixel_count": 0,
            "sourceglyph_overlap_component_ids": [],
            "sourceglyph_missing_component_ids": [],
            "effective_component_coverage_ratio": 0.0,
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="cleanup_mask_structural_invalid",
            failure_reason="segmentation_mask_missing",
            audit=audit,
            rejected=True,
        )

    projection = _component_projection_for_job(
        job=job,
        allowed=allowed,
        component_projection=component_projection,
    )
    owned_ids = projection["owned_component_ids"]
    protected_ids = projection["protected_component_ids"]
    ambiguous_ids = projection["ambiguous_component_ids"]
    unowned_ids = projection["unowned_component_ids"]
    foreground = _component_mask_from_ids(component_projection, owned_ids)
    foreground_pixels = int(np.count_nonzero(foreground > 0)) if foreground is not None else 0
    owned_segmentation_pixels = int(projection["owned_component_pixel_count"])
    owned_to_executable_ratio = round(
        float(foreground_pixels) / float(max(1, owned_segmentation_pixels)),
        4,
    ) if owned_segmentation_pixels > 0 else 0.0
    allowed_mask = _bbox_mask(foreground.shape, allowed) if foreground is not None else None
    foreground_outside_allowed_pixels = (
        int(np.count_nonzero((foreground > 0) & (allowed_mask <= 0)))
        if foreground is not None and allowed_mask is not None
        else 0
    )
    upstream_container_mismatch_ratio = round(
        float(foreground_outside_allowed_pixels) / float(max(1, foreground_pixels)),
        4,
    ) if foreground_pixels > 0 else 0.0
    before_binding = int(np.count_nonzero(_clip_mask_to_bbox((segmentation_mask > 0).astype(np.uint8), allowed)))
    bbox = _mask_bbox(foreground) if foreground is not None else None
    sourceglyph_overlap, sourceglyph_ratio, segmentation_outside_sourceglyph = _sourceglyph_overlap_metrics(
        foreground=foreground if foreground is not None else np.zeros_like(segmentation_mask, dtype=np.uint8),
        seed=seed_foreground,
    )
    sourceglyph_overlap_ids = [
        component_id
        for component_id in owned_ids
        if _component_by_id(component_projection, component_id).get("sourceglyph_overlap_pixels", 0)
    ]
    sourceglyph_missing_ids = [
        component_id
        for component_id in owned_ids
        if not _component_by_id(component_projection, component_id).get("sourceglyph_overlap_pixels", 0)
    ]
    execution_allowed = _component_projected_execution_allowed_area(
        source_allowed=allowed,
        foreground_bbox=bbox,
        mask_shape=foreground.shape if foreground is not None else segmentation_mask.shape,
    )
    component_analysis_scope = bbox or execution_allowed
    coverage_ratio = _text_block_coverage_estimate(bbox, component_analysis_scope) if bbox is not None else 0.0
    stats = _component_stats(foreground) if foreground is not None else {"component_count": 0, "largest_component_pixels": 0}
    status = "cleanup_mask_ready_from_owned_segmentation_components"
    failure_reason = ""
    rejected = False
    if not owned_ids or foreground is None or foreground_pixels <= 0:
        if ambiguous_ids:
            status = "cleanup_mask_unresolved_ambiguous_components"
            failure_reason = "cleanup_mask_unresolved_ambiguous_components"
        elif protected_ids:
            status = "cleanup_mask_unresolved_protected_overlap"
            failure_reason = "cleanup_mask_unresolved_protected_overlap"
        elif unowned_ids:
            status = "cleanup_mask_unresolved_unowned_visible_text"
            failure_reason = "cleanup_mask_unresolved_unowned_visible_text"
        else:
            status = "cleanup_mask_missing_no_owned_segmentation_components"
            failure_reason = "cleanup_mask_missing_no_owned_segmentation_components"
        rejected = True
    else:
        coverage_reason = _segmentation_effective_coverage_reason(
            foreground=foreground,
            analysis_scope=component_analysis_scope,
            job=job,
            protected_overlap_pixels=int(projection["protected_component_pixel_count"]),
            owner_pixels=max(1, int(projection["owned_component_pixel_count"]) + int(projection["protected_component_pixel_count"])),
        )
        owned_ratio_reason = _owned_segmentation_executable_coverage_reason(
            ratio=owned_to_executable_ratio,
            owned_pixels=owned_segmentation_pixels,
            foreground_pixels=foreground_pixels,
            job=job,
        )
        unsafe_reason = _segmentation_foreground_unsafe_reason(foreground, execution_allowed, job)
        if coverage_reason or owned_ratio_reason or unsafe_reason or ambiguous_ids or unowned_ids:
            status = "cleanup_mask_partial_owned_components"
            failure_reason = (
                coverage_reason
                or owned_ratio_reason
                or unsafe_reason
                or ("effective_mask_not_ready" if ambiguous_ids else "")
                or "cleanup_mask_partial_owned_components"
            )
            rejected = bool(coverage_reason or owned_ratio_reason or unsafe_reason or ambiguous_ids)

    seed_pixels = int(np.count_nonzero(seed_foreground > 0)) if seed_foreground is not None else 0
    erase = None
    if foreground is not None and foreground_pixels > 0:
        erase = _effective_erase_from_foreground(foreground, execution_allowed, job)
    erase_pixels = int(np.count_nonzero(erase > 0)) if erase is not None else 0
    erase_to_foreground_ratio = round(float(erase_pixels) / float(max(1, foreground_pixels)), 4) if foreground_pixels > 0 else 0.0
    green_to_erase_ratio = 0.0
    if erase is not None and foreground is not None and owned_segmentation_pixels > 0:
        green_to_erase_ratio = round(
            float(np.count_nonzero((erase > 0) & (foreground > 0))) / float(max(1, owned_segmentation_pixels)),
            4,
        )
    sourceglyph_executable_influence_detected = False
    ready_but_sparse_violation = bool(
        status == "cleanup_mask_ready_from_owned_segmentation_components"
        and _owned_segmentation_executable_coverage_reason(
            ratio=owned_to_executable_ratio,
            owned_pixels=owned_segmentation_pixels,
            foreground_pixels=foreground_pixels,
            job=job,
        )
    )
    audit = {
        **base_audit,
        "seed_foreground_pixels": seed_pixels,
        "completed_foreground_pixels": foreground_pixels,
        "component_count_before": len(component_projection.components),
        "component_count_after": len(owned_ids),
        "largest_component_pixels_before": _largest_component_pixels(component_projection.components),
        "largest_component_pixels_after": int(stats.get("largest_component_pixels") or 0),
        "text_block_coverage_estimate": coverage_ratio,
        "bbox_fill_ratio_before": 0.0,
        "bbox_fill_ratio_after": _mask_fill_ratio(foreground) if foreground is not None else 0.0,
        "analysis_scope_bbox": allowed,
        "component_projected_analysis_scope_bbox": component_analysis_scope,
        "component_projected_execution_scope_bbox": execution_allowed,
        "component_projected_execution_scope_source": "text_area_component_authorization_map_owned_component_union",
        "executable_erase_bbox": _mask_bbox(erase) if erase is not None else None,
        "mask_completion_method": "text_area_component_authorization_map_projection",
        "polarity_mode": "segmentation",
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
        "recovered_component_count": len(owned_ids),
        "rejected_component_count": len(protected_ids) + len(ambiguous_ids) + len(unowned_ids),
        "rejected_component_reasons": projection["unresolved_reasons"],
        "segmentation_mask_status": "segmentation_mask_ready" if segmentation_mask is not None else "segmentation_mask_missing",
        "segmentation_mask_failure_reason": failure_reason if status != "cleanup_mask_ready_from_owned_segmentation_components" else "",
        "segmentation_binding_method": "text_area_component_authorization_map_projection",
        "segmentation_pixels_before_binding": before_binding,
        "segmentation_pixels_after_owner_clip": owned_segmentation_pixels,
        "segmentation_pixels_after_protection_subtract": foreground_pixels,
        "protected_overlap_pixels": int(projection["protected_component_pixel_count"]),
        "sourceglyph_overlap_pixels": sourceglyph_overlap,
        "sourceglyph_overlap_ratio": sourceglyph_ratio,
        "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
        "effective_coverage_ratio": coverage_ratio,
        "effective_coverage_status": "effective_coverage_ready" if not rejected else failure_reason,
        "component_ownership_status": status,
        "owned_component_ids": owned_ids,
        "protected_component_ids": protected_ids,
        "ambiguous_component_ids": ambiguous_ids,
        "unowned_component_ids": unowned_ids,
        "owned_component_pixel_count": owned_segmentation_pixels,
        "protected_component_pixel_count": int(projection["protected_component_pixel_count"]),
        "ambiguous_component_pixel_count": int(projection["ambiguous_component_pixel_count"]),
        "sourceglyph_overlap_component_ids": sourceglyph_overlap_ids,
        "sourceglyph_missing_component_ids": sourceglyph_missing_ids,
        "ownership_projection_failure_reason": failure_reason,
        "effective_component_coverage_ratio": coverage_ratio,
        "missing_source_glyph_mask_ids": list(missing_source_glyph_mask_ids or []),
        "owned_segmentation_pixels": owned_segmentation_pixels,
        "executable_foreground_pixels": foreground_pixels,
        "committed_cleanup_mask_pixels": None,
        "owned_segmentation_to_executable_ratio": owned_to_executable_ratio,
        "owned_segmentation_to_commit_ratio": None,
        "ready_but_sparse_violation": ready_but_sparse_violation,
        "sourceglyph_executable_influence_detected": sourceglyph_executable_influence_detected,
        "segmentation_contract_override_detected": False,
        "cleanup_authorization": projection.get("cleanup_authorization", ""),
        "authorization_source_stage": projection.get("authorization_source_stage", ""),
        "semantic_authorization_state": projection.get("semantic_authorization_state", ""),
        "projection_quality_state": projection.get("projection_quality_state", ""),
        "projection_quality_reasons": projection.get("projection_quality_reasons", []),
        "mask_readiness_state": projection.get("mask_readiness_state", ""),
        "mask_readiness_failure_reason": projection.get("mask_readiness_failure_reason", ""),
        "semantic_authority_owner": projection.get("semantic_authority_owner", ""),
        "projection_owner": projection.get("projection_owner", ""),
        "foreground_outside_allowed_pixels": foreground_outside_allowed_pixels,
        "upstream_container_mismatch_pixels": foreground_outside_allowed_pixels,
        "upstream_container_mismatch_ratio": upstream_container_mismatch_ratio,
        "green_to_foreground_component_coverage_ratio": owned_to_executable_ratio,
        "green_to_erase_component_coverage_ratio": green_to_erase_ratio,
        "clean_mask_authority": "text_area_component_authorization_map",
        "non_segmentation_or_local_fallback_used": False,
        "bbox_executable_foreground_detected": False,
        "page_level_executable_foreground_detected": False,
        "clean_mask_foreground_pixels": foreground_pixels,
        "clean_mask_erase_pixels": erase_pixels,
        "foreground_to_owned_segmentation_ratio": owned_to_executable_ratio,
        "erase_to_foreground_ratio": erase_to_foreground_ratio,
        "protected_component_pixels_removed": int(projection["protected_component_pixel_count"]),
        "ambiguous_component_pixels": int(projection["ambiguous_component_pixel_count"]),
        "unowned_component_pixels": sum(
            int(_component_by_id(component_projection, component_id).get("pixel_count") or 0)
            for component_id in unowned_ids
        ),
        "clean_mask_state": status,
        "clean_mask_failure_reason": failure_reason,
        "ctd_refined_segmentation_mask_ref": str(base_audit.get("segmentation_mask_ref") or ""),
    }
    return _EffectiveMaskBuild(
        foreground=foreground,
        erase=erase,
        status=status,
        failure_reason=failure_reason,
        audit=audit,
        rejected=rejected,
    )


def _build_segmentation_text_mask(
    *,
    job: CleanupJob,
    segmentation_mask: np.ndarray | None,
    segmentation_audit: Mapping[str, Any],
    seed_foreground: np.ndarray | None,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
    ownership_binding: _OwnershipBinding | None = None,
) -> _EffectiveMaskBuild:
    seed = (
        _clip_mask_to_bbox((seed_foreground > 0).astype(np.uint8), allowed)
        if seed_foreground is not None
        else None
    )
    seed_pixels = int(np.count_nonzero(seed)) if seed is not None else 0
    seed_stats = _component_stats(seed) if seed is not None else {"component_count": 0, "largest_component_pixels": 0}
    base_audit = {
        "seed_foreground_pixels": seed_pixels,
        "component_count_before": seed_stats["component_count"],
        "largest_component_pixels_before": seed_stats["largest_component_pixels"],
        "bbox_fill_ratio_before": _mask_fill_ratio(seed) if seed is not None else 0.0,
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
        "segmentation_provider": segmentation_audit.get("segmentation_provider") or segmentation_audit.get("provider", ""),
        "segmentation_mask_ref": segmentation_audit.get("segmentation_mask_ref") or segmentation_audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": segmentation_audit.get("segmentation_text_pixels") or segmentation_audit.get("text_pixel_count", 0),
        "segmentation_component_count": segmentation_audit.get("segmentation_component_count"),
        "segmentation_block_associations": [],
    }
    if ownership_binding is None:
        fallback_shape = segmentation_mask.shape if segmentation_mask is not None else (seed.shape if seed is not None else None)
        ownership_binding = _build_ownership_binding(
            job=job,
            allowed=allowed,
            shape=fallback_shape,
            region_records={},
        )
    base_audit.update(_ownership_base_audit(ownership_binding))
    if segmentation_mask is None:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "segmentation_mask_missing",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 1,
            "rejected_component_reasons": ["segmentation_mask_missing"],
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": str(
                segmentation_audit.get("segmentation_mask_failure_reason") or "text_foreground_segmentation_missing"
            ),
            "segmentation_binding_method": "not_attempted",
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="segmentation_mask_missing",
            failure_reason="segmentation_mask_missing",
            audit=audit,
            rejected=True,
        )
    segmentation_binary = (segmentation_mask > 0).astype(np.uint8)
    analysis_scope = ownership_binding.owned_bbox or allowed
    before_binding = int(np.count_nonzero(_clip_mask_to_bbox(segmentation_binary, allowed)))
    owner_mask = ownership_binding.owner_mask
    if owner_mask is None:
        owner_mask = _bbox_mask(segmentation_binary.shape, allowed)
    foreground_owner = np.where(owner_mask > 0, segmentation_binary, 0).astype(np.uint8)
    after_owner = int(np.count_nonzero(foreground_owner))
    protected_mask = ownership_binding.protected_mask
    protected_overlap = int(
        np.count_nonzero((foreground_owner > 0) & (protected_mask > 0))
    ) if protected_mask is not None else 0
    if protected_mask is not None:
        foreground = np.where(protected_mask > 0, 0, foreground_owner).astype(np.uint8)
    else:
        foreground = foreground_owner
    after_protection = int(np.count_nonzero(foreground))
    sourceglyph_overlap, sourceglyph_ratio, segmentation_outside_sourceglyph = _sourceglyph_overlap_metrics(
        foreground=foreground,
        seed=seed,
    )
    binding_method = (
        "segmentation_cleanup_ownership_mask_clip_protected_subtract"
        if protected_mask is not None and protected_overlap > 0
        else "segmentation_cleanup_ownership_mask_clip"
    )
    # Ownership binding must not synthesize or crop executable foreground pixels;
    # grouping remains a downstream accounting concern, while erase dilation is
    # the only step bounded by safety scopes.
    pixels = int(np.count_nonzero(foreground))
    if pixels <= 0:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": analysis_scope,
            "executable_erase_bbox": None,
            "mask_completion_method": "text_foreground_segmentation_clipped_empty",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 1,
            "rejected_component_reasons": ["effective_mask_no_segmentation_in_owned_area"],
            "segmentation_mask_status": "cleanup_mask_unresolved_after_segmentation",
            "segmentation_mask_failure_reason": "effective_mask_no_segmentation_in_owned_area",
            "segmentation_binding_method": binding_method,
            "segmentation_pixels_before_binding": before_binding,
            "segmentation_pixels_after_owner_clip": after_owner,
            "segmentation_pixels_after_protection_subtract": after_protection,
            "protected_overlap_pixels": protected_overlap,
            "sourceglyph_overlap_pixels": sourceglyph_overlap,
            "sourceglyph_overlap_ratio": sourceglyph_ratio,
            "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
            "effective_coverage_ratio": 0.0,
            "effective_coverage_status": "effective_mask_no_segmentation_in_owned_area",
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="effective_mask_no_segmentation_in_owned_area",
            failure_reason="effective_mask_no_segmentation_in_owned_area",
            audit=audit,
            rejected=True,
        )
    coverage_reason = _segmentation_effective_coverage_reason(
        foreground=foreground,
        analysis_scope=analysis_scope,
        job=job,
        protected_overlap_pixels=protected_overlap,
        owner_pixels=after_owner,
    )
    unsafe_reason = _segmentation_foreground_unsafe_reason(foreground, allowed, job)
    if coverage_reason or unsafe_reason:
        rejected_reason = coverage_reason or unsafe_reason
        audit = _effective_audit(
            seed=seed if seed is not None else np.zeros_like(foreground, dtype=np.uint8),
            completed=foreground,
            erase=foreground,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="text_foreground_segmentation_rejected",
            polarity="segmentation",
            rejected_reasons=[rejected_reason],
            recovered_count=0,
        )
        return _EffectiveMaskBuild(
            foreground=foreground,
            erase=foreground.copy(),
            status=rejected_reason,
            failure_reason=rejected_reason,
            audit={
                **base_audit,
                **audit,
                "segmentation_mask_status": "cleanup_mask_unresolved_after_segmentation",
                "segmentation_mask_failure_reason": rejected_reason,
                "segmentation_binding_method": f"{binding_method}_coverage_rejected",
                "segmentation_block_associations": _segmentation_blocks_for_scope(segmentation_audit, analysis_scope),
                "segmentation_pixels_before_binding": before_binding,
                "segmentation_pixels_after_owner_clip": after_owner,
                "segmentation_pixels_after_protection_subtract": after_protection,
                "protected_overlap_pixels": protected_overlap,
                "sourceglyph_overlap_pixels": sourceglyph_overlap,
                "sourceglyph_overlap_ratio": sourceglyph_ratio,
                "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
                "effective_coverage_ratio": audit.get("text_block_coverage_estimate"),
                "effective_coverage_status": rejected_reason,
            },
            rejected=True,
        )
    erase = _effective_erase_from_foreground(foreground, allowed, job)
    if protected_mask is not None:
        erase = np.where(protected_mask > 0, 0, erase).astype(np.uint8)
    audit = _effective_audit(
        seed=seed if seed is not None else np.zeros_like(foreground, dtype=np.uint8),
        completed=foreground,
        erase=erase,
        allowed=allowed,
        analysis_scope=analysis_scope,
        method="text_foreground_segmentation_owned_foreground_projection",
        polarity="segmentation",
        rejected_reasons=[],
        recovered_count=0,
    )
    return _EffectiveMaskBuild(
        foreground=foreground,
        erase=erase,
        status="cleanup_mask_ready_from_segmentation",
        failure_reason="",
        audit={
            **base_audit,
            **audit,
            "segmentation_mask_status": "segmentation_mask_ready",
            "segmentation_mask_failure_reason": "",
            "segmentation_binding_method": f"{binding_method}_sourceglyph_provenance_only",
            "segmentation_block_associations": _segmentation_blocks_for_scope(segmentation_audit, analysis_scope),
            "segmentation_pixels_before_binding": before_binding,
            "segmentation_pixels_after_owner_clip": after_owner,
            "segmentation_pixels_after_protection_subtract": after_protection,
            "protected_overlap_pixels": protected_overlap,
            "sourceglyph_overlap_pixels": sourceglyph_overlap,
            "sourceglyph_overlap_ratio": sourceglyph_ratio,
            "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
            "effective_coverage_ratio": audit.get("text_block_coverage_estimate"),
            "effective_coverage_status": "effective_coverage_ready",
            "owned_segmentation_pixels": after_owner,
            "executable_foreground_pixels": after_protection,
            "committed_cleanup_mask_pixels": None,
            "owned_segmentation_to_executable_ratio": round(float(after_protection) / float(max(1, after_owner)), 4),
            "owned_segmentation_to_commit_ratio": None,
            "ready_but_sparse_violation": False,
            "sourceglyph_executable_influence_detected": False,
            "segmentation_contract_override_detected": False,
        },
        rejected=False,
    )


def _build_ownership_binding(
    *,
    job: CleanupJob,
    allowed: list[int],
    shape: tuple[int, int] | None,
    region_records: Mapping[str, Mapping[str, Any]] | None,
) -> _OwnershipBinding:
    if shape is None:
        return _OwnershipBinding(
            owner_mask=None,
            protected_mask=None,
            owned_bbox=allowed,
            protected_bbox=None,
            method="ownership_binding_shape_unavailable",
            status="ownership_binding_unresolved_invalid_mask_shape",
            protected_records=[],
        )
    owner_mask = _bbox_mask(shape, allowed)
    protected_mask = np.zeros(shape, dtype=np.uint8)
    protected_records: list[dict[str, Any]] = []
    target_ids = {str(item or "") for item in getattr(job, "target_region_ids", []) or [] if str(item or "")}
    for region_id, region in (region_records or {}).items():
        reason = _region_protection_reason(region)
        if not reason:
            continue
        bbox = _region_bbox(region)
        if bbox is None:
            continue
        region_mask = _bbox_mask(shape, bbox)
        if not np.any((region_mask > 0) & (owner_mask > 0)):
            continue
        protected_mask = np.maximum(protected_mask, region_mask)
        protected_records.append(
            {
                "region_id": region_id,
                "is_target_region": region_id in target_ids,
                "bbox": bbox,
                "reason": reason,
            }
        )
    if not np.any(protected_mask > 0):
        protected_mask_out: np.ndarray | None = None
        protected_bbox = None
    else:
        protected_mask_out = protected_mask
        protected_bbox = _mask_bbox(protected_mask)
    return _OwnershipBinding(
        owner_mask=owner_mask,
        protected_mask=protected_mask_out,
        owned_bbox=allowed,
        protected_bbox=protected_bbox,
        method="cleanup_job_allowed_area_validation_scope_with_region_protection_subtract",
        status="ownership_binding_ready",
        protected_records=protected_records,
    )


def _ownership_base_audit(binding: _OwnershipBinding) -> dict[str, Any]:
    return {
        "ownership_binding_status": binding.status,
        "ownership_binding_method": binding.method,
        "cleanup_owned_unit_bbox": binding.owned_bbox,
        "cleanup_owned_unit_mask_ref": "",
        "protected_mask_ref": "",
        "protected_overlap_pixels": 0,
        "protected_region_records": list(binding.protected_records or [])[:40],
        "segmentation_pixels_before_binding": 0,
        "segmentation_pixels_after_owner_clip": 0,
        "segmentation_pixels_after_protection_subtract": 0,
        "sourceglyph_overlap_pixels": 0,
        "sourceglyph_overlap_ratio": 0.0,
        "segmentation_outside_sourceglyph_pixels": 0,
        "effective_coverage_ratio": 0.0,
        "effective_coverage_status": "",
    }


def _component_projection_from_authorization_map(
    *,
    page_id: str,
    segmentation_mask: np.ndarray | None,
    component_authorization_map: Any,
) -> _ComponentOwnershipProjection:
    if segmentation_mask is None:
        return _ComponentOwnershipProjection(labels=None, components=[], component_label_by_id={})
    binary = (segmentation_mask > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) <= 0:
        return _ComponentOwnershipProjection(labels=np.zeros_like(binary, dtype=np.int32), components=[], component_label_by_id={})
    labels, max_label = _connected_component_labels(binary)
    payload = _component_authorization_map_audit(component_authorization_map)
    records = list(payload.get("components") or payload.get("component_authorizations") or [])
    components: list[dict[str, Any]] = []
    label_by_id: dict[str, int] = {}
    mask_slices_by_id: dict[str, tuple[list[int], np.ndarray]] = {}
    used_labels: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        label = _component_label_from_authorization_record(
            record=record,
            labels=labels,
            max_label=max_label,
            used_labels=used_labels,
        )
        if label <= 0:
            continue
        used_labels.add(label)
        component_id = str(record.get("component_id") or f"tauthcomp_{_safe_id(page_id)}_{index:04d}")
        bbox = _valid_bbox(record.get("component_bbox")) or _component_bbox_from_label(labels, label)
        pixel_count = int(record.get("component_pixel_count") or int(np.count_nonzero(labels == label)))
        clipped_bbox = None
        if bbox is not None:
            height, width = labels.shape[:2]
            sx0, sy0, sx1, sy1 = _clip_bbox_to_size(bbox, width, height)
            if sx1 > sx0 and sy1 > sy0:
                clipped_bbox = [sx0, sy0, sx1, sy1]
                mask_slices_by_id[component_id] = (
                    clipped_bbox,
                    (labels[sy0:sy1, sx0:sx1] == label).astype(np.uint8),
                )
        authorization_state = str(
            record.get("final_mask_authorization_state")
            or record.get("authorization_state")
            or "review_unknown_not_cleanup"
        )
        projection_quality_state = str(record.get("projection_quality_state") or "")
        mask_readiness_state = str(record.get("mask_readiness_state") or "")
        ownership_state = _authorization_state_to_ownership_state(authorization_state)
        if authorization_state in AUTH_CLEANUP_COMPONENT_STATES and (
            projection_quality_state != PROJECTION_READY_STATE
            or mask_readiness_state != MASK_READY_STATE
            or str(record.get("job_binding_state") or "") != "bound_unique"
        ):
            ownership_state = "projection_not_ready"
        cleanup_authorizations = []
        if authorization_state in AUTH_CLEANUP_COMPONENT_STATES:
            cleanup_authorizations.append(authorization_state)
        components.append(
            {
                "component_id": component_id,
                "label": int(label),
                "ownership_state": ownership_state,
                "authorization_state": authorization_state,
                "semantic_authorization_state": str(record.get("semantic_authorization_state") or authorization_state),
                "semantic_visual_color": str(record.get("semantic_visual_color") or record.get("visual_debug_color") or ""),
                "job_binding_state": str(record.get("job_binding_state") or ""),
                "job_binding_failure_reason": str(record.get("job_binding_failure_reason") or ""),
                "final_mask_authorization_state": authorization_state,
                "projection_quality_state": projection_quality_state,
                "projection_quality_reasons": [str(item) for item in record.get("projection_quality_reasons") or [] if str(item)],
                "mask_readiness_state": mask_readiness_state,
                "mask_readiness_failure_reason": str(record.get("mask_readiness_failure_reason") or ""),
                "semantic_unit_ids": [str(item) for item in record.get("semantic_unit_ids") or [] if str(item)],
                "semantic_unit_states": [str(item) for item in record.get("semantic_unit_states") or [] if str(item)],
                "semantic_kind": str(record.get("semantic_kind") or ""),
                "semantic_kinds": [str(item) for item in record.get("semantic_kinds") or [] if str(item)],
                "source_evidence_ids": [str(item) for item in record.get("source_evidence_ids") or [] if str(item)],
                "semantic_authority_owner": str(record.get("semantic_authority_owner") or ""),
                "projection_owner": str(record.get("projection_owner") or ""),
                "owner_cleanup_job_id": str(record.get("owner_cleanup_job_id") or ""),
                "owner_region_ids": [str(item) for item in record.get("owning_region_ids") or [] if str(item)],
                "scope_cleanup_job_ids": [str(item) for item in record.get("scope_cleanup_job_ids") or [] if str(item)],
                "candidate_cleanup_job_ids": [str(item) for item in record.get("candidate_cleanup_job_ids") or [] if str(item)],
                "cleanup_authorizations": cleanup_authorizations,
                "authorization_source_stages": [str(record.get("source_stage") or "")] if record.get("source_stage") else [],
                "explicit_cleanup_authority": bool(record.get("explicit_cleanup_authority")),
                "explicit_protected_authority": bool(record.get("explicit_protected_authority")),
                "explicit_authority_source": str(record.get("explicit_authority_source") or ""),
                "protected_region_ids": [str(item) for item in record.get("protection_region_ids") or [] if str(item)],
                "protected_reason": ",".join(str(item) for item in record.get("reason_codes") or [] if str(item)),
                "bbox": clipped_bbox or bbox,
                "pixel_count": pixel_count,
                "centroid": list(record.get("centroid") or []),
                "owner_overlap_pixels": int(record.get("overlap_pixels") or 0),
                "owner_overlap_ratio": float(record.get("overlap_ratio") or 0.0),
                "protected_overlap_pixels": int(record.get("protected_overlap_pixels") or 0),
                "protected_overlap_ratio": float(record.get("protected_overlap_ratio") or 0.0),
                "sourceglyph_overlap_pixels": int(record.get("sourceglyph_overlap_pixels") or 0),
                "sourceglyph_overlap_ratio": 0.0,
                "sourceglyph_overlap_ids": [],
                "sourceglyph_missing": bool(record.get("sourceglyph_missing", True)),
                "ambiguity_reasons": [str(item) for item in record.get("ambiguity_reasons") or record.get("reason_codes") or [] if str(item)],
                "owning_container_ids": [str(item) for item in record.get("owning_container_ids") or [] if str(item)],
                "protection_container_ids": [str(item) for item in record.get("protection_container_ids") or [] if str(item)],
                "candidate_container_ids": [str(item) for item in record.get("candidate_container_ids") or [] if str(item)],
                "visual_debug_color": str(record.get("visual_debug_color") or ""),
                "group_id": str(record.get("group_id") or ""),
                "group_authorization_state": str(record.get("group_authorization_state") or ""),
                "review_required": bool(record.get("review_required")),
            }
        )
        label_by_id[component_id] = int(label)
    return _ComponentOwnershipProjection(
        labels=labels.astype(np.int32),
        components=components,
        component_label_by_id=label_by_id,
        component_mask_slices_by_id=mask_slices_by_id,
    )


def _connected_component_labels(binary: np.ndarray) -> tuple[np.ndarray, int]:
    if cv2 is not None:
        label_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return labels.astype(np.int32), int(label_count) - 1
    labels = np.zeros_like(binary, dtype=np.int32)
    height, width = binary.shape[:2]
    label = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x] <= 0 or labels[y, x] != 0:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            while stack:
                sx, sy = stack.pop()
                for ny in range(max(0, sy - 1), min(height, sy + 2)):
                    for nx in range(max(0, sx - 1), min(width, sx + 2)):
                        if binary[ny, nx] > 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((nx, ny))
    return labels, label


def _component_authorization_map_audit(component_authorization_map: Any) -> dict[str, Any]:
    if component_authorization_map is None:
        return {}
    if hasattr(component_authorization_map, "to_audit_dict"):
        try:
            payload = component_authorization_map.to_audit_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            return {}
    if hasattr(component_authorization_map, "to_dict"):
        try:
            payload = component_authorization_map.to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            return {}
    if isinstance(component_authorization_map, Mapping):
        return dict(component_authorization_map)
    return {}


def _component_label_from_authorization_record(
    *,
    record: Mapping[str, Any],
    labels: np.ndarray,
    max_label: int,
    used_labels: set[int],
) -> int:
    try:
        label = int(record.get("label") or 0)
    except Exception:
        label = 0
    if label > 0 and label <= max_label and label not in used_labels:
        return label
    bbox = _valid_bbox(record.get("component_bbox"))
    if bbox is None:
        return 0
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, labels.shape[1], labels.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    crop = labels[y0:y1, x0:x1]
    values, counts = np.unique(crop[crop > 0], return_counts=True)
    pairs = sorted(
        ((int(label_value), int(count)) for label_value, count in zip(values, counts) if int(label_value) not in used_labels),
        key=lambda item: item[1],
        reverse=True,
    )
    return pairs[0][0] if pairs else 0


def _component_bbox_from_label(labels: np.ndarray, label: int) -> list[int]:
    ys, xs = np.where(labels == int(label))
    if xs.size <= 0 or ys.size <= 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _authorization_state_to_ownership_state(authorization_state: str) -> str:
    if authorization_state in AUTH_CLEANUP_COMPONENT_STATES:
        return "cleanup_owned"
    if authorization_state == "protect_art_or_non_text":
        return "protected_art_or_non_text"
    if authorization_state == "protect_sfx_decorative":
        return "protected_sfx_decorative"
    if authorization_state == AUTH_AMBIGUOUS_COMPONENT_STATE:
        return AUTH_AMBIGUOUS_COMPONENT_STATE
    if authorization_state == "outside_cleanup_scope":
        return "outside_cleanup_scope"
    return "review_unknown_not_cleanup"


def _region_cleanup_authorization(region: Mapping[str, Any] | None) -> str:
    if not isinstance(region, Mapping):
        return ""
    auth = str(
        region.get("cleanup_authorization")
        or region.get("text_area_cleanup_authorization")
        or region.get("semantic_authorization_state")
        or region.get("text_area_semantic_authorization_state")
        or ""
    )
    if auth in AUTH_CLEANUP_COMPONENT_STATES | AUTH_PROTECTED_COMPONENT_STATES | AUTH_REVIEW_COMPONENT_STATES | {AUTH_AMBIGUOUS_COMPONENT_STATE}:
        return auth
    return ""


def _component_projection_for_job(
    *,
    job: CleanupJob,
    allowed: list[int],
    component_projection: _ComponentOwnershipProjection,
) -> dict[str, Any]:
    job_id = str(getattr(job, "cleanup_job_id", "") or "")
    def intersects_allowed(item: Mapping[str, Any]) -> bool:
        return _component_pixels_in_allowed(item, allowed) > 0

    owned = [
        item
        for item in component_projection.components
        if item.get("ownership_state") == "cleanup_owned"
        and str(item.get("projection_quality_state") or "") == PROJECTION_READY_STATE
        and str(item.get("mask_readiness_state") or "") == MASK_READY_STATE
        and _component_authorized_for_job(item, job_id)
        and intersects_allowed(item)
    ]
    protected = [
        item
        for item in component_projection.components
        if str(item.get("ownership_state") or "").startswith("protected")
        and intersects_allowed(item)
    ]
    ambiguous = [
        item
        for item in component_projection.components
        if intersects_allowed(item)
        and (
            item.get("ownership_state") in {"ambiguous_multi_owner", AUTH_AMBIGUOUS_COMPONENT_STATE}
            or (item.get("ownership_state") == "projection_not_ready" and _component_authorized_for_job(item, job_id))
        )
    ]
    unowned = [
        item
        for item in component_projection.components
        if item.get("ownership_state") in {"unowned_visible_text", "review_unknown_not_cleanup", "outside_cleanup_scope"}
        and intersects_allowed(item)
    ]
    cleanup_authorizations = sorted(
        {
            str(auth)
            for item in owned
            for auth in (item.get("cleanup_authorizations") or [])
            if str(auth)
        }
    )
    source_stages = sorted(
        {
            str(stage)
            for item in owned
            for stage in (item.get("authorization_source_stages") or [])
            if str(stage)
        }
    )
    relevant = owned + [
        item
        for item in ambiguous
        if item.get("ownership_state") == "projection_not_ready" and _component_authorized_for_job(item, job_id)
    ]
    protected_pixel_count = sum(_component_pixels_in_allowed(item, allowed) for item in protected)
    ambiguous_pixel_count = sum(_component_pixels_in_allowed(item, allowed) for item in ambiguous)
    return {
        "owned_component_ids": [str(item["component_id"]) for item in owned],
        "protected_component_ids": [str(item["component_id"]) for item in protected],
        "ambiguous_component_ids": [str(item["component_id"]) for item in ambiguous],
        "unowned_component_ids": [str(item["component_id"]) for item in unowned],
        "owned_component_pixel_count": sum(int(item.get("pixel_count") or 0) for item in owned),
        "protected_component_pixel_count": protected_pixel_count,
        "ambiguous_component_pixel_count": ambiguous_pixel_count,
        "unresolved_reasons": _component_projection_unresolved_reasons(protected, ambiguous, unowned),
        "cleanup_authorization": ",".join(cleanup_authorizations),
        "authorization_source_stage": ",".join(source_stages),
        "semantic_authorization_state": ",".join(
            sorted({str(item.get("semantic_authorization_state") or "") for item in relevant if str(item.get("semantic_authorization_state") or "")})
        ),
        "projection_quality_state": ",".join(
            sorted({str(item.get("projection_quality_state") or "") for item in relevant if str(item.get("projection_quality_state") or "")})
        ),
        "projection_quality_reasons": sorted(
            {
                str(reason)
                for item in relevant
                for reason in (item.get("projection_quality_reasons") or [])
                if str(reason)
            }
        ),
        "mask_readiness_state": ",".join(
            sorted({str(item.get("mask_readiness_state") or "") for item in relevant if str(item.get("mask_readiness_state") or "")})
        ),
        "mask_readiness_failure_reason": ",".join(
            sorted({str(item.get("mask_readiness_failure_reason") or "") for item in relevant if str(item.get("mask_readiness_failure_reason") or "")})
        ),
        "semantic_authority_owner": ",".join(
            sorted({str(item.get("semantic_authority_owner") or "") for item in relevant if str(item.get("semantic_authority_owner") or "")})
        ),
        "projection_owner": ",".join(
            sorted({str(item.get("projection_owner") or "") for item in relevant if str(item.get("projection_owner") or "")})
        ),
    }


def _build_unbound_semantic_cleanup_contract_error_record(
    *,
    page_id: str,
    component_projection: _ComponentOwnershipProjection,
    existing_masks: Sequence[CleanupMask],
) -> dict[str, Any] | None:
    if component_projection.labels is None:
        return None
    covered: set[str] = set()
    for mask in existing_masks or []:
        covered.update(str(item) for item in getattr(mask, "owned_component_ids", []) or [] if str(item))
    unbound = [
        item
        for item in component_projection.components
        if str(item.get("semantic_authorization_state") or item.get("authorization_state") or "") in AUTH_CLEANUP_COMPONENT_STATES
        and str(item.get("component_id") or "") not in covered
        and str(item.get("job_binding_state") or "") in {"missing_cleanup_job"}
    ]
    if not unbound:
        return None
    component_ids = [str(item.get("component_id") or "") for item in unbound if str(item.get("component_id") or "")]
    if not component_ids:
        return None
    component_mask = _component_mask_from_ids(component_projection, component_ids)
    if component_mask is None or int(np.count_nonzero(component_mask > 0)) <= 0:
        return None
    component_pixels = int(np.count_nonzero(component_mask > 0))
    bbox = _mask_bbox(component_mask)
    cleanup_job_id = f"component_authorization_binding_contract_error_{_safe_id(page_id)}"
    source_stages = sorted(
        {
            str(stage)
            for item in unbound
            for stage in (item.get("authorization_source_stages") or [])
            if str(stage)
        }
    )
    return {
        "page_id": str(page_id),
        "cleanup_job_id": cleanup_job_id,
        "region_id": "",
        "reason": "cleanup_job_binding_contract_error",
        "component_ids": component_ids,
        "component_bbox": bbox,
        "component_pixels": component_pixels,
        "component_count": len(component_ids),
        "required_cleanup_job_binding": "bound_unique",
        "actual_job_binding_states": sorted(
            {str(item.get("job_binding_state") or "") for item in unbound if str(item.get("job_binding_state") or "")}
        ),
        "job_binding_failure_reasons": sorted(
            {
                str(item.get("job_binding_failure_reason") or "")
                for item in unbound
                if str(item.get("job_binding_failure_reason") or "")
            }
        ),
        "owned_component_ids": component_ids,
        "foreground_mask_pixels": 0,
        "erase_mask_pixels": 0,
        "executable_foreground_pixels": 0,
        "clean_mask_foreground_pixels": 0,
        "clean_mask_erase_pixels": 0,
        "effective_mask_status": "cleanup_mask_rejected_binding_contract_error",
        "clean_mask_state": "cleanup_mask_rejected_binding_contract_error",
        "clean_mask_failure_reason": "cleanup_job_binding_contract_error",
        "ownership_binding_status": "cleanup_job_binding_contract_error",
        "ownership_binding_method": "text_area_component_authorization_map",
        "component_ownership_status": "cleanup_mask_rejected_binding_contract_error",
        "component_projection_method": "text_area_component_authorization_map",
        "owned_component_pixel_count": sum(int(item.get("pixel_count") or 0) for item in unbound),
        "owned_segmentation_pixels": sum(int(item.get("pixel_count") or 0) for item in unbound),
        "authorization_source_stage": ",".join(source_stages),
        "mask_contract_exception_reason": "cleanup_job_binding_contract_error",
    }


def _component_authorized_for_job(component: Mapping[str, Any], job_id: str) -> bool:
    if not job_id:
        return False
    return (
        str(component.get("owner_cleanup_job_id") or "") == job_id
        and str(component.get("job_binding_state") or "") == "bound_unique"
    )


def _component_pixels_in_allowed(component: Mapping[str, Any], allowed: list[int]) -> int:
    bbox = _valid_bbox(component.get("bbox"))
    allowed_box = _valid_bbox(allowed)
    if bbox is None or allowed_box is None:
        return 0
    intersection = _intersect_bboxes(bbox, allowed_box)
    if intersection is None:
        return 0
    bbox_area = max(1, _bbox_area(bbox))
    intersection_area = max(0, _bbox_area(intersection))
    pixels = int(component.get("pixel_count") or 0)
    if intersection_area >= bbox_area:
        return pixels
    return int(round(pixels * (intersection_area / float(bbox_area))))


def _component_projection_unresolved_reasons(
    protected: Sequence[Mapping[str, Any]],
    ambiguous: Sequence[Mapping[str, Any]],
    unowned: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if protected:
        reasons.append("protected_component_overlap")
    if ambiguous:
        if any(str(item.get("ownership_state") or "") == "projection_not_ready" for item in ambiguous):
            reasons.append("effective_mask_not_ready")
        reasons.append("ambiguous_component_owner_components")
    if unowned:
        reasons.append("unowned_visible_text_components")
    return reasons


def _component_mask_from_ids(
    component_projection: _ComponentOwnershipProjection,
    component_ids: Sequence[str],
) -> np.ndarray | None:
    if component_projection.labels is None or not component_ids:
        return None
    output = np.zeros_like(component_projection.labels, dtype=np.uint8)
    missing_labels: list[int] = []
    used = False
    for component_id in component_ids:
        component_id = str(component_id)
        cached = component_projection.component_mask_slices_by_id.get(component_id)
        if cached is not None:
            bbox, mask_slice = cached
            valid = _valid_bbox(bbox)
            if valid is None:
                continue
            x0, y0, x1, y1 = valid
            if x1 <= x0 or y1 <= y0:
                continue
            output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], mask_slice.astype(np.uint8))
            used = True
            continue
        if component_id in component_projection.component_label_by_id:
            missing_labels.append(component_projection.component_label_by_id[component_id])
    if missing_labels:
        output[np.isin(component_projection.labels, missing_labels)] = 1
        used = True
    if not used:
        return None
    return output


def _component_projected_execution_allowed_area(
    *,
    source_allowed: Sequence[int],
    foreground_bbox: Sequence[int] | None,
    mask_shape: Sequence[int],
) -> list[int]:
    base = _valid_bbox(source_allowed)
    foreground_box = _valid_bbox(foreground_bbox)
    if foreground_box is None:
        return base or [0, 0, int(mask_shape[1]), int(mask_shape[0])]
    # Component authorization is the cleanup-owned page-space surface. The
    # source/job box remains provenance, but it must not clip explicitly bound
    # components that TextAreaPlan has already authorized.
    execution_box = _union_bboxes([box for box in (base, _expand_bbox(foreground_box, 8)) if box is not None])
    if execution_box is None:
        execution_box = foreground_box
    height = int(mask_shape[0]) if len(mask_shape) >= 1 else 0
    width = int(mask_shape[1]) if len(mask_shape) >= 2 else 0
    if width <= 0 or height <= 0:
        return execution_box
    x0, y0, x1, y1 = _clip_bbox_to_size(execution_box, width, height)
    return [x0, y0, x1, y1]


def _component_by_id(component_projection: _ComponentOwnershipProjection, component_id: str) -> dict[str, Any]:
    for item in component_projection.components:
        if item.get("component_id") == component_id:
            return item
    return {}


def _largest_component_pixels(components: Sequence[Mapping[str, Any]]) -> int:
    return max([int(item.get("pixel_count") or 0) for item in components] or [0])


def _component_ownership_state_counts(components: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in components:
        state = str(item.get("ownership_state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def _bbox_mask(shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, width, height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def _index_region_records(records: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records or []):
        if not isinstance(record, Mapping):
            continue
        region_id = str(record.get("region_id") or record.get("id") or f"region_{index}")
        if region_id:
            output[region_id] = record
    return output


def _region_bbox(region: Mapping[str, Any]) -> list[int] | None:
    for key in ("xyxy", "bounds"):
        bbox = _valid_bbox(region.get(key))
        if bbox is not None:
            return bbox
    bbox = _xywh_bbox(region.get("bbox"))
    if bbox is not None:
        return bbox
    render = region.get("render")
    if isinstance(render, Mapping):
        for key in ("bbox", "source_bbox", "text_area_container_bbox"):
            bbox = _xywh_bbox(render.get(key)) if key in {"bbox", "text_area_container_bbox"} else _valid_or_xywh_bbox(render.get(key))
            if bbox is not None:
                return bbox
    return None


def _xywh_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _valid_or_xywh_bbox(value: Any) -> list[int] | None:
    bbox = _valid_bbox(value)
    if bbox is not None:
        return bbox
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _region_protection_reason(region: Mapping[str, Any]) -> str:
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    flags = region.get("flags") if isinstance(region.get("flags"), Mapping) else {}
    if _truthy(region.get("must_not_mutate")) or _truthy(region.get("text_area_must_not_mutate")):
        auth = _region_cleanup_authorization(region)
        if auth == "protect_art_or_non_text":
            return "explicit_art_or_non_text_authorization"
        if auth == "protect_sfx_decorative":
            return "explicit_sfx_decorative_authorization"
        if auth in {"review_unknown_not_cleanup", "outside_cleanup_scope"}:
            return auth
    del render, flags
    return ""


def _sourceglyph_overlap_metrics(
    *,
    foreground: np.ndarray,
    seed: np.ndarray | None,
) -> tuple[int, float, int]:
    foreground_pixels = int(np.count_nonzero(foreground > 0))
    if seed is None or int(np.count_nonzero(seed > 0)) <= 0:
        return 0, 0.0, foreground_pixels
    seed_mask = _coerce_mask_shape((seed > 0).astype(np.uint8), foreground.shape)
    overlap = int(np.count_nonzero((foreground > 0) & (seed_mask > 0)))
    outside = max(0, foreground_pixels - overlap)
    ratio = round(float(overlap) / float(max(1, foreground_pixels)), 4)
    return overlap, ratio, outside


def _segmentation_effective_coverage_reason(
    *,
    foreground: np.ndarray,
    analysis_scope: list[int],
    job: CleanupJob,
    protected_overlap_pixels: int,
    owner_pixels: int,
) -> str:
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "effective_mask_no_segmentation_in_owned_area"
    pixels = int(np.count_nonzero(foreground > 0))
    coverage = _text_block_coverage_estimate(bbox, analysis_scope)
    stats = _component_stats(foreground)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    small_or_glyph_local = cleanup_value in {
        CleanupClass.SMALL_REACTION.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        CleanupClass.TITLE_OR_SIGN.value,
    }
    min_coverage = MIN_SEGMENTATION_READY_SMALL_COVERAGE_RATIO if small_or_glyph_local else MIN_SEGMENTATION_READY_COVERAGE_RATIO
    if owner_pixels > 0 and protected_overlap_pixels / float(max(1, owner_pixels)) > PROTECTED_DOMINANT_OVERLAP_RATIO:
        return "effective_mask_protected_overlap_removed"
    if pixels < MIN_SEGMENTATION_READY_PIXELS:
        return "effective_mask_fragment_only"
    if pixels <= FRAGMENT_ONLY_MAX_PIXELS and coverage < FRAGMENT_ONLY_MAX_COVERAGE_RATIO:
        return "effective_mask_fragment_only"
    if coverage < min_coverage:
        return "effective_mask_incomplete_under_coverage"
    largest_component = int(stats.get("largest_component_pixels") or 0)
    if largest_component <= 0 or (pixels <= FRAGMENT_ONLY_MAX_PIXELS and largest_component < pixels * 0.45):
        return "effective_mask_fragment_only"
    return ""


def _owned_segmentation_executable_coverage_reason(
    *,
    ratio: float,
    owned_pixels: int,
    foreground_pixels: int,
    job: CleanupJob,
) -> str:
    if owned_pixels <= 0:
        return ""
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    small_or_glyph_local = cleanup_value in {
        CleanupClass.SMALL_REACTION.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        CleanupClass.TITLE_OR_SIGN.value,
    }
    min_ratio = (
        MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_SMALL_RATIO
        if small_or_glyph_local
        else MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_RATIO
    )
    if foreground_pixels < MIN_SEGMENTATION_READY_PIXELS:
        return "effective_mask_fragment_only"
    if ratio < min_ratio:
        return "effective_mask_incomplete_under_coverage"
    return ""


def _expand_bbox(bbox: list[int], margin: int) -> list[int]:
    return [
        int(bbox[0]) - margin,
        int(bbox[1]) - margin,
        int(bbox[2]) + margin,
        int(bbox[3]) + margin,
    ]


def _intersect_bboxes(a: list[int], b: list[int]) -> list[int] | None:
    box_a = _valid_bbox(a)
    box_b = _valid_bbox(b)
    if box_a is None or box_b is None:
        return None
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _segmentation_blocks_for_scope(segmentation_audit: Mapping[str, Any], scope: list[int]) -> list[dict[str, Any]]:
    blocks = list(segmentation_audit.get("block_associations") or [])
    output: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        bbox = _valid_bbox(block.get("line_bbox")) or _valid_bbox(block.get("xyxy"))
        if bbox is None:
            continue
        if _intersect_bboxes(bbox, scope) is not None:
            output.append(dict(block))
    return output[:20]


def _segmentation_foreground_unsafe_reason(
    foreground: np.ndarray,
    allowed: list[int],
    job: CleanupJob,
) -> str:
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "segmentation_mask_under_coverage"
    pixels = int(np.count_nonzero(foreground > 0))
    allowed_area = max(1, _bbox_area(allowed))
    bbox_area = max(1, _bbox_area(bbox))
    pixel_ratio = pixels / float(allowed_area)
    fill_ratio = pixels / float(bbox_area)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    max_pixel_ratio = 0.48 if speech_like else 0.34
    if pixel_ratio > max_pixel_ratio:
        return "segmentation_mask_wrong_owner_broad_background_capture"
    if fill_ratio > 0.72 and bbox_area > 1800:
        return "segmentation_mask_wrong_owner_rectangular_background_capture"
    return ""


def _group_segmentation_text_components(
    foreground: np.ndarray,
    scope: list[int],
    job: CleanupJob,
) -> np.ndarray:
    if cv2 is None or not np.any(foreground > 0):
        return foreground
    stats = _component_stats(foreground)
    if int(stats.get("component_count") or 0) <= 4:
        return foreground
    axis = _text_axis(foreground, scope, job)
    if axis == "vertical":
        kernel = np.ones((5, 13), dtype=np.uint8)
    elif axis == "horizontal":
        kernel = np.ones((13, 5), dtype=np.uint8)
    else:
        kernel = np.ones((5, 5), dtype=np.uint8)
    grouped = cv2.morphologyEx((foreground > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    grouped = _clip_mask_to_bbox(grouped.astype(np.uint8), scope)
    unsafe = _grouped_segmentation_too_broad(grouped, foreground, scope)
    if unsafe:
        return foreground
    return grouped


def _grouped_segmentation_too_broad(
    grouped: np.ndarray,
    original: np.ndarray,
    scope: list[int],
) -> bool:
    original_pixels = max(1, int(np.count_nonzero(original > 0)))
    grouped_pixels = int(np.count_nonzero(grouped > 0))
    if grouped_pixels <= original_pixels:
        return False
    if grouped_pixels / float(original_pixels) > 2.35:
        return True
    scope_area = max(1, _bbox_area(scope))
    if grouped_pixels / float(scope_area) > 0.42:
        return True
    return False


def _source_image_array(
    *,
    source_image: Any | None,
    source_image_path: str | Path | None,
) -> tuple[np.ndarray | None, str]:
    if source_image is not None:
        try:
            if Image is not None and isinstance(source_image, Image.Image):
                return np.asarray(source_image.convert("RGB")), ""
            arr = np.asarray(source_image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=2)
            if arr.ndim == 3:
                return arr[:, :, :3].astype(np.uint8), ""
        except Exception as exc:
            return None, f"source_image_array_error:{type(exc).__name__}"
    if source_image_path:
        if Image is None:
            return None, "pillow_unavailable"
        try:
            with Image.open(source_image_path) as img:
                return np.asarray(img.convert("RGB")), ""
        except Exception as exc:
            return None, f"source_image_load_error:{type(exc).__name__}"
    return None, "source_image_unavailable"


def _build_effective_text_mask(
    *,
    job: CleanupJob,
    source_np: np.ndarray | None,
    source_error: str,
    seed_foreground: np.ndarray,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
) -> _EffectiveMaskBuild:
    seed = _clip_mask_to_bbox((seed_foreground > 0).astype(np.uint8), allowed)
    seed_pixels = int(np.count_nonzero(seed))
    seed_stats = _component_stats(seed)
    seed_bbox = _mask_bbox(seed)
    seed_fill = _mask_fill_ratio(seed)
    base_audit = {
        "seed_foreground_pixels": seed_pixels,
        "component_count_before": seed_stats["component_count"],
        "largest_component_pixels_before": seed_stats["largest_component_pixels"],
        "bbox_fill_ratio_before": seed_fill,
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
    }
    if seed_pixels <= 0 or seed_bbox is None:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "sourceglyph_seed_empty",
            "polarity_mode": "unknown",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["empty_sourceglyph_seed"],
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="effective_mask_failed_insufficient_evidence",
            failure_reason="effective_mask_failed_insufficient_evidence",
            audit=audit,
            rejected=True,
        )

    suspect_reason = _seed_effective_mask_suspect_reason(seed, allowed, job)
    if source_np is None:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=seed,
            erase=erase,
            allowed=allowed,
            analysis_scope=seed_bbox,
            method="sourceglyph_seed_used_without_source_image",
            polarity="unknown",
            rejected_reasons=[source_error or "source_image_unavailable"],
            recovered_count=0,
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed,
            erase=erase,
            status=(
                "effective_mask_failed_insufficient_evidence"
            ),
            failure_reason=(
                "effective_mask_failed_insufficient_evidence"
            ),
            audit={**base_audit, **audit},
            rejected=True,
        )

    analysis_scope = _analysis_scope_bbox(seed_bbox, allowed, job, suspect=bool(suspect_reason))
    reconstructed, reconstruction_audit = _reconstruct_text_strokes(
        source_np=source_np,
        seed=seed,
        allowed=allowed,
        analysis_scope=analysis_scope,
        job=job,
    )
    if reconstructed is None or int(np.count_nonzero(reconstructed)) <= 0:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=seed,
            erase=erase,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="sourceglyph_seed_reconstruction_failed",
            polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
            rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or [])
            + ["no_text_strokes_found"],
            recovered_count=0,
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed,
            erase=erase,
            status=(
                "effective_mask_failed_no_text_strokes_found"
            ),
            failure_reason=(
                "effective_mask_failed_no_text_strokes_found"
            ),
            audit={**base_audit, **audit, **reconstruction_audit},
            rejected=True,
        )

    completed = np.maximum(seed, reconstructed.astype(np.uint8))
    completed = _clip_mask_to_bbox(completed, allowed)
    completed = _bridge_text_stroke_gaps(completed, allowed, job)
    completed = _clip_mask_to_bbox(completed, allowed)
    unsafe_reason = _completed_mask_unsafe_reason(completed, allowed, job)
    if unsafe_reason:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=completed,
            erase=erase,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="local_text_stroke_reconstruction_rejected",
            polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
            rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or [])
            + [unsafe_reason],
            recovered_count=int(reconstruction_audit.get("recovered_component_count") or 0),
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed if rejected else completed,
            erase=erase,
            status=(
                "effective_mask_failed_unsafe_background_capture"
            ),
            failure_reason=unsafe_reason,
            audit={**base_audit, **audit},
            rejected=True,
        )

    erase = _effective_erase_from_foreground(completed, allowed, job)
    recovered_pixels = int(np.count_nonzero(completed)) - seed_pixels
    method = (
        "local_text_stroke_reconstruction"
        if recovered_pixels > max(24, int(seed_pixels * 0.08)) or suspect_reason
        else "sourceglyph_seed_effective"
    )
    audit = _effective_audit(
        seed=seed,
        completed=completed,
        erase=erase,
        allowed=allowed,
        analysis_scope=analysis_scope,
        method=method,
        polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
        rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or []),
        recovered_count=int(reconstruction_audit.get("recovered_component_count") or 0),
    )
    return _EffectiveMaskBuild(
        foreground=completed,
        erase=erase,
        status="cleanup_mask_unresolved_after_segmentation",
        failure_reason=suspect_reason or "local_contrast_fallback_diagnostic_only",
        audit={**base_audit, **audit},
        rejected=True,
    )


def _reconstruct_text_strokes(
    *,
    source_np: np.ndarray,
    seed: np.ndarray,
    allowed: list[int],
    analysis_scope: list[int],
    job: CleanupJob,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = source_np.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(analysis_scope, width, height)
    if x1 <= x0 or y1 <= y0:
        return None, {
            "polarity_mode": "unknown",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["invalid_analysis_scope"],
        }
    crop = source_np[y0:y1, x0:x1]
    seed_crop = seed[y0:y1, x0:x1]
    gray = _gray(crop)
    polarity = _polarity_mode(gray, seed_crop)
    candidate = _local_stroke_candidates(gray, polarity)
    if candidate is None or not np.any(candidate > 0):
        return None, {
            "polarity_mode": polarity,
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["local_contrast_found_no_strokes"],
        }
    selected, recovered_count, rejected_reasons = _select_textlike_components(
        candidate=candidate,
        seed_crop=seed_crop,
        crop_origin=(x0, y0),
        allowed=allowed,
        job=job,
    )
    if not np.any(selected > 0):
        return None, {
            "polarity_mode": polarity,
            "recovered_component_count": recovered_count,
            "rejected_component_count": len(rejected_reasons),
            "rejected_component_reasons": rejected_reasons,
        }
    output = np.zeros(seed.shape, dtype=np.uint8)
    output[y0:y1, x0:x1] = selected.astype(np.uint8)
    output = _clip_mask_to_bbox(output, allowed)
    return output, {
        "polarity_mode": polarity,
        "recovered_component_count": recovered_count,
        "rejected_component_count": len(rejected_reasons),
        "rejected_component_reasons": rejected_reasons[:40],
    }


def _local_stroke_candidates(gray: np.ndarray, polarity: str) -> np.ndarray | None:
    if gray.size <= 0:
        return None
    if cv2 is not None:
        block = max(15, min(51, (min(gray.shape[:2]) // 2) * 2 + 1))
        dark_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            7,
        )
        mean = cv2.blur(gray, (max(9, block // 2), max(9, block // 2)))
        dark_local = ((gray.astype(np.int16) + 18) < mean.astype(np.int16)).astype(np.uint8) * 255
        light_local = ((gray.astype(np.int16) - 18) > mean.astype(np.int16)).astype(np.uint8) * 255
        if polarity == "dark_on_light":
            candidate = (dark_adaptive > 0) | (dark_local > 0)
        elif polarity == "light_on_dark":
            candidate = light_local > 0
        else:
            candidate = (dark_local > 0) | (light_local > 0)
        candidate = candidate.astype(np.uint8)
        kernel = np.ones((2, 2), dtype=np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        return candidate.astype(np.uint8)
    mean_value = float(np.mean(gray))
    if polarity == "light_on_dark":
        return (gray > mean_value + 18).astype(np.uint8)
    if polarity == "dark_on_light":
        return (gray < mean_value - 18).astype(np.uint8)
    return ((gray < mean_value - 18) | (gray > mean_value + 18)).astype(np.uint8)


def _select_textlike_components(
    *,
    candidate: np.ndarray,
    seed_crop: np.ndarray,
    crop_origin: tuple[int, int],
    allowed: list[int],
    job: CleanupJob,
) -> tuple[np.ndarray, int, list[str]]:
    selected = np.zeros_like(candidate, dtype=np.uint8)
    rejected_reasons: list[str] = []
    if cv2 is None:
        selected = candidate.astype(np.uint8)
        return selected, 1 if np.any(selected > 0) else 0, rejected_reasons
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )
    seed_dilated = seed_crop.astype(np.uint8)
    if np.any(seed_dilated > 0):
        kernel = np.ones((19, 19), dtype=np.uint8)
        seed_dilated = cv2.dilate(seed_dilated, kernel, iterations=1)
    seed_box = _mask_bbox(seed_crop)
    axis = _text_axis(seed_crop, allowed, job)
    recovered_count = 0
    for label in range(1, labels_count):
        area_pixels = int(stats[label, cv2.CC_STAT_AREA])
        if area_pixels < 4:
            rejected_reasons.append("component_too_small")
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = area_pixels / max(1, w * h)
        abs_box = [x + crop_origin[0], y + crop_origin[1], x + w + crop_origin[0], y + h + crop_origin[1]]
        if _component_reject_reason(abs_box, area_pixels, fill_ratio, allowed, job):
            rejected_reasons.append(_component_reject_reason(abs_box, area_pixels, fill_ratio, allowed, job))
            continue
        component = labels == label
        seed_overlap = int(np.count_nonzero(component & (seed_dilated > 0)))
        aligned = _component_aligned_with_seed(
            local_box=[x, y, x + w, y + h],
            seed_box=seed_box,
            axis=axis,
            crop_shape=candidate.shape,
        )
        if seed_overlap <= 0 and not aligned:
            rejected_reasons.append("component_not_seed_connected_or_aligned")
            continue
        selected[component] = 1
        recovered_count += 1
    return selected, recovered_count, rejected_reasons


def _component_reject_reason(
    box: list[int],
    area_pixels: int,
    fill_ratio: float,
    allowed: list[int],
    job: CleanupJob,
) -> str:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    allowed_area = max(1, _bbox_area(allowed))
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if area_pixels > allowed_area * (0.24 if speech_like else 0.16):
        return "component_area_too_large_for_allowed_text_area"
    if fill_ratio > 0.72 and area_pixels > 180:
        return "component_mostly_solid_background_or_patch"
    if width >= height * 12 and height <= 6:
        return "component_panel_or_bubble_border_horizontal"
    if height >= width * 14 and width <= 6:
        return "component_panel_or_bubble_border_vertical"
    border_touch = (
        box[0] <= allowed[0] + 2
        or box[1] <= allowed[1] + 2
        or box[2] >= allowed[2] - 2
        or box[3] >= allowed[3] - 2
    )
    if border_touch and area_pixels > allowed_area * 0.06 and (width <= 8 or height <= 8):
        return "component_touches_allowed_border_like_line"
    return ""


def _component_aligned_with_seed(
    *,
    local_box: list[int],
    seed_box: list[int] | None,
    axis: str,
    crop_shape: tuple[int, int],
) -> bool:
    if seed_box is None:
        return True
    crop_h, crop_w = crop_shape[:2]
    if axis == "vertical":
        pad = max(18, int(crop_w * 0.22))
        return local_box[0] <= seed_box[2] + pad and local_box[2] >= seed_box[0] - pad
    if axis == "horizontal":
        pad = max(18, int(crop_h * 0.22))
        return local_box[1] <= seed_box[3] + pad and local_box[3] >= seed_box[1] - pad
    pad_x = max(20, int(crop_w * 0.20))
    pad_y = max(20, int(crop_h * 0.20))
    return (
        local_box[0] <= seed_box[2] + pad_x
        and local_box[2] >= seed_box[0] - pad_x
        and local_box[1] <= seed_box[3] + pad_y
        and local_box[3] >= seed_box[1] - pad_y
    )


def _analysis_scope_bbox(
    seed_bbox: list[int],
    allowed: list[int],
    job: CleanupJob,
    *,
    suspect: bool,
) -> list[int]:
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    background_like = cleanup_value in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }
    if suspect or background_like:
        return list(allowed)
    pad = max(12, int(max(seed_bbox[2] - seed_bbox[0], seed_bbox[3] - seed_bbox[1]) * 0.20))
    return [
        max(allowed[0], seed_bbox[0] - pad),
        max(allowed[1], seed_bbox[1] - pad),
        min(allowed[2], seed_bbox[2] + pad),
        min(allowed[3], seed_bbox[3] + pad),
    ]


def _polarity_mode(gray: np.ndarray, seed_crop: np.ndarray) -> str:
    if gray.size <= 0:
        return "unknown"
    if np.any(seed_crop > 0):
        seed_values = gray[seed_crop > 0]
        seed_mean = float(np.mean(seed_values))
        if cv2 is not None:
            kernel = np.ones((9, 9), dtype=np.uint8)
            halo = cv2.dilate(seed_crop.astype(np.uint8), kernel, iterations=1)
            background_values = gray[halo <= 0]
        else:
            background_values = gray[seed_crop <= 0]
        background_mean = float(np.mean(background_values)) if background_values.size else float(np.mean(gray))
        if seed_mean < background_mean - 10:
            return "dark_on_light"
        if seed_mean > background_mean + 10:
            return "light_on_dark"
    mean = float(np.mean(gray))
    if mean < 105:
        return "light_on_dark"
    if mean > 175:
        return "dark_on_light"
    return "mixed_or_screentone"


def _gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    if cv2 is not None:
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    ).astype(np.uint8)


def _seed_effective_mask_suspect_reason(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    pixels = int(np.count_nonzero(mask > 0))
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "formal_mask_exists_but_effective_mask_missing"
    bbox_area = _bbox_area(bbox)
    allowed_area = max(1, _bbox_area(allowed))
    width_ratio = (bbox[2] - bbox[0]) / max(1, allowed[2] - allowed[0])
    height_ratio = (bbox[3] - bbox[1]) / max(1, allowed[3] - allowed[1])
    bbox_coverage = bbox_area / allowed_area
    stats = _component_stats(mask)
    largest = int(stats["largest_component_pixels"])
    if pixels < 80:
        return "fragmented_character_only_mask"
    if pixels < 1200 and bbox_coverage < 0.16:
        return "fragmented_character_only_mask"
    if largest < max(24, int(pixels * 0.12)) and int(stats["component_count"]) > 18:
        return "fragmented_character_only_mask"
    if bbox_coverage < 0.10 and (width_ratio < 0.30 or height_ratio < 0.30):
        return "fragmented_character_only_mask"
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    if cleanup_value in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    } and bbox_coverage < 0.22:
        return "effective_mask_under_50_percent"
    return ""


def _completed_mask_unsafe_reason(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    pixels = int(np.count_nonzero(mask > 0))
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "effective_mask_failed_no_text_strokes_found"
    allowed_area = max(1, _bbox_area(allowed))
    bbox_area = max(1, _bbox_area(bbox))
    fill_ratio = pixels / bbox_area
    pixel_ratio = pixels / allowed_area
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if pixel_ratio > (0.42 if speech_like else 0.26):
        return "effective_mask_failed_unsafe_background_capture"
    if fill_ratio > 0.66 and bbox_area > 1200:
        return "effective_mask_failed_unsafe_background_capture"
    return ""


def _effective_erase_from_foreground(
    foreground: np.ndarray,
    allowed: list[int],
    job: CleanupJob,
) -> np.ndarray:
    foreground = (foreground > 0).astype(np.uint8)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    outline_sensitive_text = cleanup_value in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        iterations = (
            4
            if outline_sensitive_text
            else 2
            if speech_like and int(np.count_nonzero(foreground)) < 1400
            else 1
        )
        erase = cv2.dilate(foreground, kernel, iterations=iterations).astype(np.uint8)
    else:
        erase = _binary_dilate(foreground, radius=2 if outline_sensitive_text else 1)
    bounded_dilation = _clip_mask_to_bbox(erase.astype(np.uint8), allowed)
    return np.maximum(foreground, bounded_dilation).astype(np.uint8)


def _bridge_text_stroke_gaps(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> np.ndarray:
    if cv2 is None or not np.any(mask > 0):
        return mask
    axis = _text_axis(mask, allowed, job)
    if axis == "vertical":
        kernel = np.ones((3, 7), dtype=np.uint8)
    elif axis == "horizontal":
        kernel = np.ones((7, 3), dtype=np.uint8)
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
    bridged = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    return _clip_mask_to_bbox(bridged.astype(np.uint8), allowed)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    padded = np.pad((mask > 0).astype(np.uint8), radius, mode="constant")
    output = np.zeros_like(mask, dtype=np.uint8)
    for dy in range(0, radius * 2 + 1):
        for dx in range(0, radius * 2 + 1):
            output = np.maximum(output, padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]])
    return output.astype(np.uint8)


def _text_axis(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "mixed"
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    if height >= width * 1.35:
        return "vertical"
    if width >= height * 1.35:
        return "horizontal"
    mode = " ".join(
        str(value or "").lower()
        for value in (
            getattr(job, "cleanup_mode", ""),
            getattr(job, "classification_reason", ""),
            getattr(job, "route_intent", ""),
            getattr(job, "semantic_class", ""),
        )
    )
    if "vertical" in mode or "side_caption" in mode:
        return "vertical"
    return "mixed"


def _effective_audit(
    *,
    seed: np.ndarray,
    completed: np.ndarray,
    erase: np.ndarray,
    allowed: list[int],
    analysis_scope: list[int],
    method: str,
    polarity: str,
    rejected_reasons: Sequence[str],
    recovered_count: int,
) -> dict[str, Any]:
    completed_stats = _component_stats(completed)
    completed_bbox = _mask_bbox(completed)
    coverage = _text_block_coverage_estimate(completed_bbox, allowed)
    return {
        "completed_foreground_pixels": int(np.count_nonzero(completed > 0)),
        "component_count_after": completed_stats["component_count"],
        "largest_component_pixels_after": completed_stats["largest_component_pixels"],
        "text_block_coverage_estimate": coverage,
        "bbox_fill_ratio_after": _mask_fill_ratio(completed),
        "analysis_scope_bbox": list(analysis_scope),
        "executable_erase_bbox": _mask_bbox(erase),
        "mask_completion_method": method,
        "polarity_mode": polarity,
        "recovered_component_count": recovered_count,
        "rejected_component_count": len([reason for reason in rejected_reasons if reason]),
        "rejected_component_reasons": list(dict.fromkeys(str(reason) for reason in rejected_reasons if reason))[:40],
    }


def _component_stats(mask: np.ndarray) -> dict[str, int]:
    if not np.any(mask > 0):
        return {"component_count": 0, "largest_component_pixels": 0}
    if cv2 is None:
        return {
            "component_count": 1,
            "largest_component_pixels": int(np.count_nonzero(mask > 0)),
        }
    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]
    return {
        "component_count": len(areas),
        "largest_component_pixels": max(areas) if areas else 0,
    }


def _mask_fill_ratio(mask: np.ndarray) -> float:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return 0.0
    return round(float(np.count_nonzero(mask > 0)) / float(max(1, _bbox_area(bbox))), 4)


def _text_block_coverage_estimate(bbox: list[int] | None, allowed: list[int]) -> float:
    if bbox is None:
        return 0.0
    allowed_area = max(1, _bbox_area(allowed))
    return round(float(_bbox_area(bbox)) / float(allowed_area), 4)


def _binary_mask_from_evidence(
    evidence: _SourceEvidence,
    *,
    keys: Sequence[str],
) -> tuple[np.ndarray | None, str]:
    for key in keys:
        value = _get_value(evidence.raw, key)
        if value is None and isinstance(evidence.audit, Mapping):
            value = evidence.audit.get(key)
        mask = _to_binary_mask(value)
        if mask is not None:
            return mask, key
    return None, ""


def _evidence_has_raw_mask(evidence: _SourceEvidence) -> bool:
    for key in ("foreground_mask", "erase_mask", "mask"):
        if _to_binary_mask(_get_value(evidence.raw, key)) is not None:
            return True
    return False


def _to_binary_mask(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    return (arr > 0).astype(np.uint8)


def _coerce_mask_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.copy()
    output = np.zeros(shape, dtype=np.uint8)
    height = min(int(shape[0]), int(mask.shape[0]))
    width = min(int(shape[1]), int(mask.shape[1]))
    if height > 0 and width > 0:
        output[:height, :width] = mask[:height, :width]
    return output


def _clip_mask_to_bbox(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    clipped = np.zeros_like(mask, dtype=np.uint8)
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, width, height)
    if x1 <= x0 or y1 <= y0:
        return clipped
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def _clip_bbox_to_size(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width))
    y0 = max(0, min(int(y0), height))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    return x0, y0, x1, y1


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _growth_ratio(erase_pixels: int, foreground_pixels: int) -> float:
    if foreground_pixels <= 0:
        return float("inf")
    return round(float(erase_pixels) / float(foreground_pixels), 4)


def _broad_mask_rejection(
    *,
    allowed: list[int],
    erase_bbox: list[int],
    erase_pixels: int,
    growth_ratio: float,
    image_size: tuple[int, int] | None,
    allow_growth_exception: bool = False,
) -> str:
    if growth_ratio > MAX_ERASE_GROWTH_RATIO and not allow_growth_exception:
        return "erase_mask_growth_too_large"
    allowed_area = _bbox_area(allowed)
    if erase_pixels > allowed_area:
        return "erase_mask_pixels_exceed_allowed_area"
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if erase_pixels / page_area > MAX_ERASE_PAGE_PIXEL_RATIO:
            return "erase_mask_pixels_too_broad_for_page"
        if _bbox_area(erase_bbox) / page_area > MAX_ERASE_BBOX_PAGE_RATIO:
            return "erase_mask_bbox_too_broad_for_page"
    return ""


def _caption_flat_small_mask_growth_exception_reason(
    *,
    job: CleanupJob,
    source: _SourceEvidence | None,
    allowed: list[int],
    erase_bbox: list[int],
    erase_pixels: int,
    foreground_pixels: int,
    growth_ratio: float,
    image_size: tuple[int, int] | None,
    artifact_risk: str,
) -> str:
    if growth_ratio <= MAX_ERASE_GROWTH_RATIO:
        return ""
    if growth_ratio > MAX_CAPTION_FLAT_SMALL_EXCEPTION_GROWTH_RATIO:
        return ""
    cleanup_class = getattr(job, "cleanup_class", "")
    if _enum_value(cleanup_class) != CleanupClass.CAPTION_FLAT_BACKGROUND.value:
        return ""
    if source is not None and _is_side_caption_glyph_local(job, source):
        return ""
    if artifact_risk:
        return ""
    method_text = " ".join(
        str(value or "")
        for value in (
            getattr(job, "classification_reason", ""),
            source.mask_id if source is not None else "",
            _first_present(source.audit, "generation_method", "source_glyph_mask_generation_method", default="")
            if source is not None
            else "",
        )
    ).lower()
    if "glyph_local" not in method_text:
        return ""
    allowed_area = _bbox_area(allowed)
    erase_bbox_area = _bbox_area(erase_bbox)
    if foreground_pixels <= 0 or erase_pixels <= 0:
        return ""
    if erase_pixels > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_PIXELS:
        return ""
    if erase_bbox_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_BBOX_AREA:
        return ""
    if allowed_area <= 0 or allowed_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_AREA:
        return ""
    if erase_pixels / max(1, allowed_area) > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_ALLOWED_RATIO:
        return ""
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if allowed_area / page_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_PAGE_RATIO:
            return ""
    return "caption_flat_small_mask_growth_exception"


def _effective_uses_component_projection(effective: _EffectiveMaskBuild) -> bool:
    audit = effective.audit if isinstance(effective.audit, Mapping) else {}
    joined = " ".join(
        str(audit.get(key) or "")
        for key in (
            "component_projection_method",
            "mask_completion_method",
            "segmentation_binding_method",
            "clean_mask_authority",
        )
    )
    return "text_area_component_authorization_map" in joined


def _component_authorized_growth_exception_applies(effective: _EffectiveMaskBuild) -> bool:
    """Treat growth as audit/proof risk for upstream-authorized component masks.

    Thin vertical manga text can legitimately need a larger erase halo than the
    foreground-pixel count suggests. Once TextAreaPlan has explicitly authorized
    a projection-ready component and CleanupMask has the exact foreground/erase
    arrays, dropping that component here silently breaks the root cleanup
    obligation. Broad page/bbox limits still run in _broad_mask_rejection.
    """

    audit = effective.audit if isinstance(effective.audit, Mapping) else {}
    if not _effective_uses_component_projection(effective):
        return False
    if int(audit.get("protected_component_pixels_removed") or 0) > 0:
        return False
    if int(audit.get("ambiguous_component_pixels") or 0) > 0:
        return False
    projection_state = str(audit.get("projection_quality_state") or "")
    if projection_state and projection_state != PROJECTION_READY_STATE:
        return False
    readiness_state = str(audit.get("mask_readiness_state") or "")
    if readiness_state and readiness_state != MASK_READY_STATE:
        return False
    authorization = " ".join(
        str(audit.get(key) or "")
        for key in (
            "cleanup_authorization",
            "semantic_authorization_state",
        )
    )
    return any(state in authorization for state in AUTH_CLEANUP_COMPONENT_STATES)


def _allowed_area_rejection(allowed: list[int], image_size: tuple[int, int] | None) -> str:
    if _bbox_area(allowed) <= 0:
        return "allowed_cleanup_area_empty"
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if _bbox_area(allowed) / page_area > MAX_ALLOWED_PAGE_RATIO:
            return "allowed_cleanup_area_near_full_page"
    return ""


def _job_protection_reason(job: CleanupJob) -> str:
    if bool(getattr(job, "protected", False)):
        return str(getattr(job, "protection_reason", "") or "job_protected")
    cleanup_class = getattr(job, "cleanup_class", "")
    cleanup_value = cleanup_class.value if isinstance(cleanup_class, Enum) else str(cleanup_class)
    combined = " ".join(
        str(value or "")
        for value in (
            cleanup_value,
            getattr(job, "route_intent", ""),
            getattr(job, "semantic_class", ""),
            getattr(job, "cleanup_mode", ""),
        )
    ).lower()
    if cleanup_value == CleanupClass.PRESERVE_SFX_DECORATIVE.value:
        return "cleanup_class_preserve_sfx_decorative"
    for marker in ("preserve", "sfx", "decorative", "art_only", "non_translation_art"):
        if marker in combined:
            return f"route_or_semantic_{marker}"
    return ""


def _is_side_caption_glyph_local(job: CleanupJob, evidence: _SourceEvidence) -> bool:
    cleanup_class = getattr(job, "cleanup_class", "")
    cleanup_value = cleanup_class.value if isinstance(cleanup_class, Enum) else str(cleanup_class)
    if cleanup_value == CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value:
        return True
    metadata = " ".join(
        str(value or "")
        for value in (
            cleanup_value,
            getattr(job, "classification_reason", ""),
            evidence.mask_id,
            _first_present(evidence.audit, "generation_method", "source_glyph_mask_generation_method", default=""),
        )
    ).lower()
    return "side_caption_glyph_local" in metadata


def _mask_method(evidence: _SourceEvidence, source_key: str) -> str:
    method = str(
        _first_present(
            evidence.audit,
            "foreground_mask_method",
            "erase_mask_method",
            "generation_method",
            "source_glyph_mask_generation_method",
            default="",
        )
        or ""
    )
    parts = [part for part in (method, source_key, "contract_only_not_renderer_consumed") if part]
    return "|".join(parts)


def _mask_method_union(evidence_records: Sequence[_SourceEvidence], source_keys: Sequence[str]) -> str:
    methods: list[str] = []
    for evidence in evidence_records:
        method = str(
            _first_present(
                evidence.audit,
                "foreground_mask_method",
                "erase_mask_method",
                "generation_method",
                "source_glyph_mask_generation_method",
                default="",
            )
            or ""
        )
        if method and method not in methods:
            methods.append(method)
    for source_key in source_keys or []:
        if source_key and source_key not in methods:
            methods.append(source_key)
    if evidence_records:
        methods.append("source_glyph_union_contract_only_not_renderer_consumed")
    else:
        methods.append("segmentation_component_projection_contract_only_not_renderer_consumed")
    return "|".join(methods)


def _base_job_record(page_id: str, job: CleanupJob) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "cleanup_job_id": str(getattr(job, "cleanup_job_id", "") or ""),
        "parent_execution_bundle_id": str(
            getattr(job, "parent_execution_bundle_id", "")
            or getattr(job, "parent_logical_text_unit_id", "")
            or ""
        ),
        "parent_logical_text_unit_id": str(getattr(job, "parent_logical_text_unit_id", "") or ""),
        "text_block_root_id": str(getattr(job, "text_block_root_id", "") or ""),
        "target_region_ids": list(getattr(job, "target_region_ids", []) or []),
        "cleanup_class": _enum_value(getattr(job, "cleanup_class", "")),
        "route_intent": str(getattr(job, "route_intent", "") or ""),
        "semantic_class": str(getattr(job, "semantic_class", "") or ""),
        "cleanup_mode": str(getattr(job, "cleanup_mode", "") or ""),
    }


def _get_mapping(source: Any, key: str) -> Mapping[Any, Any] | None:
    value = _get_value(source, key)
    return value if isinstance(value, Mapping) else None


def _get_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    if hasattr(source, key):
        return getattr(source, key)
    return None


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _sequence_or_single(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _union_bboxes(bboxes: Sequence[Sequence[int]]) -> list[int] | None:
    valid: list[list[int]] = []
    for bbox in bboxes:
        box = _valid_bbox(bbox)
        if box is not None:
            valid.append(box)
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _safe_id(value: Any) -> str:
    text = str(value or "none")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "none"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "risk", "risky"}
    return bool(value)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_audit_dict"):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items() if not _looks_like_raw_array(key)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _looks_like_raw_array(key: Any) -> bool:
    return str(key).lower() in {"mask", "foreground_mask", "erase_mask", "image", "array", "pixels_raw", "bitmap", "crop"}
