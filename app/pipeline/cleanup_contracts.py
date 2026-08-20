"""Cleanup contract types and conservative Phase 0 candidate building.

This module defines ownership boundaries only. It does not execute cleanup,
choose a backend, inspect pixels, or wire itself into the runtime pipeline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from app.pipeline.debug_runtime import diagnostic_enabled, write_diagnostic_checkpoint
from app.pipeline.parent_execution_bundle import parent_execution_region_records


CLEANUP_CONTRACT_VERSION = "cleanup_contracts_phase0"


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
            module="app.pipeline.cleanup_contracts",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
        )
    except Exception:
        return


class CleanupClass(str, Enum):
    """Stable cleanup classes owned by cleanup contracts."""

    SPEECH_FLAT_BUBBLE = "speech_flat_bubble"
    SPEECH_COMPLEX_BUBBLE = "speech_complex_bubble"
    CAPTION_FLAT_BACKGROUND = "caption_flat_background"
    CAPTION_DARK_OR_SCREENTONE = "caption_dark_or_screentone"
    TITLE_OR_SIGN = "title_or_sign"
    SMALL_REACTION = "small_reaction"
    BACKGROUND_ART_TEXT = "background_art_text"
    ART_ENTANGLED_AMBIGUOUS = "art_entangled_ambiguous"
    SIDE_CAPTION_GLYPH_LOCAL = "side_caption_glyph_local"
    PRESERVE_SFX_DECORATIVE = "preserve_sfx_decorative"


class ProofStatus(str, Enum):
    """Cleanup proof lifecycle status."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CleanupJob:
    """A request to remove source glyphs for translated text."""

    page_id: str
    cleanup_job_id: str
    target_region_ids: list[str]
    logical_text_unit_ids: list[str]
    cleanup_class: CleanupClass
    route_intent: str
    semantic_class: str
    source_text_present: bool
    translated_text_present: bool
    source_glyph_mask_ids: list[str]
    source_glyph_evidence: list[dict[str, Any]]
    allowed_cleanup_area: list[int] | None
    protected: bool = False
    protection_reason: str = ""
    proof_required: bool = True
    proof_level: str = "source_glyph_removal"
    classification_reason: str = ""
    text_block_root_id: str | None = None
    parent_logical_text_unit_id: str | None = None
    parent_execution_bundle_id: str | None = None
    text_area_container_id: str | None = None
    container_type: str = ""
    cleanup_mode: str = ""
    source_glyph_erasure_expected_area_bbox: list[int] | None = None
    source_glyph_erasure_bbox: list[int] | None = None
    cleanup_unit_id: str = ""
    cleanup_unit_level: str = ""
    cleanup_unit_anchor_region_id: str = ""
    cleanup_unit_child_region_ids: list[str] = field(default_factory=list)
    cleanup_unit_required_source_glyph_mask_ids: list[str] = field(default_factory=list)
    cleanup_unit_missing_source_glyph_mask_ids: list[str] = field(default_factory=list)
    cleanup_only_obligation: bool = False

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self)


@dataclass(frozen=True)
class TextForegroundSegmentationMask:
    """Dense visible-text foreground evidence for cleanup mask construction."""

    page_id: str
    image_size: tuple[int, int] | list[int] | None
    raw_mask_ref: str = ""
    refined_mask_ref: str = ""
    threshold_used: int | float | None = None
    provider: str = ""
    backend: str = ""
    runtime_ms: float | None = None
    text_pixel_count: int = 0
    connected_component_stats: dict[str, Any] = field(default_factory=dict)
    block_associations: list[dict[str, Any]] = field(default_factory=list)
    keep_undetected_mask: bool = False
    confidence: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_mask: Any = field(default=None, repr=False, compare=False)
    refined_mask: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self, omit={"raw_mask", "refined_mask"})


@dataclass(frozen=True)
class CleanupMask:
    """Foreground and erase-mask evidence for one cleanup job."""

    cleanup_mask_id: str
    cleanup_job_id: str
    parent_execution_bundle_id: str = ""
    parent_logical_text_unit_id: str = ""
    text_block_root_id: str = ""
    text_area_container_id: str = ""
    target_region_ids: list[str] = field(default_factory=list)
    represented_region_ids: list[str] = field(default_factory=list)
    foreground_mask_source_id: str | None = None
    foreground_mask_source_ids: list[str] = field(default_factory=list)
    consumed_source_glyph_mask_ids: list[str] = field(default_factory=list)
    missing_source_glyph_mask_ids: list[str] = field(default_factory=list)
    foreground_mask_bbox: list[int] | None = None
    foreground_mask_pixels: int | None = None
    erase_mask_bbox: list[int] | None = None
    erase_mask_pixels: int | None = None
    allowed_area: list[int] | None = None
    growth_ratio: float | None = None
    mask_source: str = ""
    mask_method: str = ""
    rejection_reason: str = ""
    mask_contract_exception_reason: str = ""
    artifact_risk: str = ""
    visual_scope: str = ""
    protected: bool = False
    protection_reason: str = ""
    effective_mask_status: str = ""
    effective_mask_failure_reason: str = ""
    seed_foreground_pixels: int | None = None
    completed_foreground_pixels: int | None = None
    component_count_before: int | None = None
    component_count_after: int | None = None
    largest_component_pixels_before: int | None = None
    largest_component_pixels_after: int | None = None
    text_block_coverage_estimate: float | None = None
    bbox_fill_ratio_before: float | None = None
    bbox_fill_ratio_after: float | None = None
    analysis_scope_bbox: list[int] | None = None
    executable_erase_bbox: list[int] | None = None
    mask_completion_method: str = ""
    polarity_mode: str = ""
    source_seed_mask_ids: list[str] = field(default_factory=list)
    recovered_component_count: int | None = None
    rejected_component_count: int | None = None
    rejected_component_reasons: list[str] = field(default_factory=list)
    segmentation_mask_status: str = ""
    segmentation_mask_failure_reason: str = ""
    segmentation_provider: str = ""
    segmentation_mask_ref: str = ""
    segmentation_text_pixels: int | None = None
    segmentation_component_count: int | None = None
    segmentation_binding_method: str = ""
    segmentation_block_associations: list[dict[str, Any]] = field(default_factory=list)
    ownership_binding_status: str = ""
    ownership_binding_method: str = ""
    cleanup_owned_unit_bbox: list[int] | None = None
    cleanup_owned_unit_mask_ref: str = ""
    protected_mask_ref: str = ""
    protected_overlap_pixels: int | None = None
    segmentation_pixels_before_binding: int | None = None
    segmentation_pixels_after_owner_clip: int | None = None
    segmentation_pixels_after_protection_subtract: int | None = None
    sourceglyph_overlap_pixels: int | None = None
    sourceglyph_overlap_ratio: float | None = None
    segmentation_outside_sourceglyph_pixels: int | None = None
    effective_coverage_ratio: float | None = None
    effective_coverage_status: str = ""
    component_ownership_status: str = ""
    owned_component_ids: list[str] = field(default_factory=list)
    protected_component_ids: list[str] = field(default_factory=list)
    ambiguous_component_ids: list[str] = field(default_factory=list)
    unowned_component_ids: list[str] = field(default_factory=list)
    component_projection_method: str = ""
    owned_component_pixel_count: int | None = None
    protected_component_pixel_count: int | None = None
    ambiguous_component_pixel_count: int | None = None
    sourceglyph_overlap_component_ids: list[str] = field(default_factory=list)
    sourceglyph_missing_component_ids: list[str] = field(default_factory=list)
    ownership_projection_failure_reason: str = ""
    effective_component_coverage_ratio: float | None = None
    owned_segmentation_pixels: int | None = None
    executable_foreground_pixels: int | None = None
    committed_cleanup_mask_pixels: int | None = None
    owned_segmentation_to_executable_ratio: float | None = None
    owned_segmentation_to_commit_ratio: float | None = None
    ready_but_sparse_violation: bool = False
    sourceglyph_executable_influence_detected: bool = False
    segmentation_contract_override_detected: bool = False
    clean_mask_authority: str = ""
    non_segmentation_or_local_fallback_used: bool = False
    bbox_executable_foreground_detected: bool = False
    page_level_executable_foreground_detected: bool = False
    clean_mask_foreground_pixels: int | None = None
    clean_mask_erase_pixels: int | None = None
    foreground_to_owned_segmentation_ratio: float | None = None
    erase_to_foreground_ratio: float | None = None
    protected_component_pixels_removed: int | None = None
    ambiguous_component_pixels: int | None = None
    unowned_component_pixels: int | None = None
    clean_mask_state: str = ""
    clean_mask_failure_reason: str = ""
    cleanup_authorization: str = ""
    authorization_source_stage: str = ""
    semantic_authorization_state: str = ""
    projection_quality_state: str = ""
    projection_quality_reasons: list[str] = field(default_factory=list)
    mask_readiness_state: str = ""
    mask_readiness_failure_reason: str = ""
    semantic_authority_owner: str = ""
    projection_owner: str = ""
    foreground_outside_allowed_pixels: int | None = None
    upstream_container_mismatch_pixels: int | None = None
    upstream_container_mismatch_ratio: float | None = None
    green_to_foreground_component_coverage_ratio: float | None = None
    green_to_erase_component_coverage_ratio: float | None = None
    ctd_refined_segmentation_mask_ref: str = ""
    cleanup_owned_segmentation_foreground_mask_ref: str = ""
    protected_component_mask_ref: str = ""
    clean_mask_foreground_ref: str = ""
    clean_mask_erase_ref: str = ""
    component_ownership_overlay_ref: str = ""
    rejected_component_overlay_ref: str = ""
    sourceglyph_overlap_overlay_ref: str = ""
    clean_mask_annotation_ref: str = ""
    foreground_mask: Any = field(default=None, repr=False, compare=False)
    erase_mask: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self, omit={"foreground_mask", "erase_mask"})


@dataclass(frozen=True)
class CleanupPlan:
    """Selected cleanup method, backend, parameters, and proof policy."""

    cleanup_plan_id: str
    cleanup_job_id: str
    cleanup_mask_id: str
    cleanup_class: CleanupClass
    selected_backend: str
    cleanup_method: str
    parent_execution_bundle_id: str = ""
    parent_logical_text_unit_id: str = ""
    text_block_root_id: str = ""
    target_region_ids: list[str] = field(default_factory=list)
    backend_parameters: dict[str, Any] = field(default_factory=dict)
    inpaint_mode: str = ""
    crop_context_bbox: list[int] | None = None
    fallback_policy: list[str] = field(default_factory=list)
    expected_runtime_class: str = ""
    proof_thresholds: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self)


@dataclass(frozen=True)
class CleanupResult:
    """Backend execution result metadata and optional output references."""

    cleanup_result_id: str
    cleanup_plan_id: str
    cleanup_job_id: str
    cleanup_mask_id: str
    operation_bbox: list[int] | None
    page: str = ""
    region_id: str = ""
    parent_execution_bundle_id: str = ""
    parent_logical_text_unit_id: str = ""
    text_block_root_id: str = ""
    target_region_ids: list[str] = field(default_factory=list)
    cleanup_class: CleanupClass | str = ""
    pixel_changed: bool = False
    changed_pixel_count: int = 0
    diff_ref: str | None = None
    mask_ref: str | None = None
    renderer_consumed: bool = False
    execution_backend: str = ""
    execution_status: str = ""
    failure_reason: str = ""
    mask_stats: dict[str, Any] = field(default_factory=dict)
    backend_name: str = ""
    backend_parameters: dict[str, Any] = field(default_factory=dict)
    backend_kind: str = ""
    backend_method: str = ""
    requested_model_id: str = ""
    actual_model_name: str = ""
    actual_model_path: str = ""
    model_invocation_attempted: bool = False
    model_invocation_succeeded: bool = False
    input_image_hash: str = ""
    input_mask_hash: str = ""
    raw_backend_output_hash: str = ""
    final_clipped_output_hash: str = ""
    fallback_reason: str = ""
    backend_input_mask_ref: str | None = None
    model_input_image_ref: str | None = None
    model_input_mask_ref: str | None = None
    raw_backend_output_ref: str | None = None
    final_clipped_output_ref: str | None = None
    runtime_ms: float | None = None
    result_space: str = "page"
    crop_bbox: list[int] | None = None
    crop_width: int | None = None
    crop_height: int | None = None
    fallback_status: str = ""
    errors: list[str] = field(default_factory=list)
    cleaned_image_ref: str | None = None
    cleaned_crop_ref: str | None = None
    commit_mask: Any = field(default=None, repr=False, compare=False)
    commit_foreground_mask: Any = field(default=None, repr=False, compare=False)
    cleaned_image: Any = field(default=None, repr=False, compare=False)
    cleaned_crop: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(
            self,
            omit={"commit_mask", "commit_foreground_mask", "cleaned_image", "cleaned_crop"},
        )


@dataclass(frozen=True)
class CleanupProof:
    """Pre-render cleanup proof evidence."""

    cleanup_proof_id: str
    cleanup_result_id: str
    cleanup_job_id: str
    cleanup_plan_id: str
    proof_status: ProofStatus
    parent_execution_bundle_id: str = ""
    parent_logical_text_unit_id: str = ""
    text_block_root_id: str = ""
    target_region_ids: list[str] = field(default_factory=list)
    source_glyph_removal_passed: bool | None = None
    source_residual_ratio: float | None = None
    changed_outside_allowed_pixels: int | None = None
    collateral_damage_score: float | None = None
    mask_containment_ratio: float | None = None
    render_consumption_decision_if_consumed: str = ""
    residual_source_text_risk: str = ""
    mask_containment_passed: bool | None = None
    collateral_damage_risk: str = ""
    backend_failure_visible: bool | None = None
    failure_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self)


@dataclass(frozen=True)
class CleanupObligationRecord:
    """Audit-only accounting for accepted text cleanup responsibility.

    This record is not an ownership layer and must not choose cleanup masks,
    plans, backends, proof policy, or render behavior.
    """

    page_id: str
    region_id: str
    predicate_class: str
    obligation_status: str
    cleanup_outcome_state: str
    first_failing_stage: str
    failure_reason: str = ""
    cleanup_job_id: str = ""
    representing_cleanup_job_id: str = ""
    represented_region_ids: list[str] = field(default_factory=list)
    source_glyph_mask_ids: list[str] = field(default_factory=list)
    explicit_preserve_evidence: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return _dataclass_audit_dict(self)


@dataclass(frozen=True)
class CleanupJobBuildResult:
    """Candidate build output without cleanup execution."""

    page_id: str
    version: str
    jobs: list[CleanupJob] = field(default_factory=list)
    obligation_records: list[CleanupObligationRecord] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "jobs": [_json_safe(job) for job in self.jobs],
            "obligation_records": [_json_safe(record) for record in self.obligation_records],
            "protected_records": _json_safe(self.protected_records),
            "skipped_records": _json_safe(self.skipped_records),
            "errors": list(self.errors),
            "summary": {
                "job_count": len(self.jobs),
                "obligation_record_count": len(self.obligation_records),
                "protected_record_count": len(self.protected_records),
                "skipped_record_count": len(self.skipped_records),
                "error_count": len(self.errors),
            },
        }


@dataclass(frozen=True)
class _CleanupUnit:
    """Derived cleanup-unit membership from existing source/region metadata."""

    unit_id: str
    level: str
    anchor_region_id: str
    child_region_ids: list[str]
    source_records: list[dict[str, Any]]
    required_source_glyph_mask_ids: list[str]
    missing_source_glyph_mask_ids: list[str]
    text_block_root_id: str | None = None
    parent_logical_text_unit_id: str | None = None
    text_area_container_id: str | None = None
    ambiguous_reason: str = ""


def build_cleanup_job_candidates(
    *,
    page_id: str,
    regions: Sequence[Mapping[str, Any]],
    source_glyph_masks: Any,
) -> CleanupJobBuildResult:
    """Build conservative cleanup job candidates from existing metadata only.

    The builder is intentionally limited to pre-existing region/render and
    SourceGlyphMask audit metadata. It does not inspect pixels, create masks,
    choose backends, or execute cleanup.
    """

    started = time.time()
    source_records_by_region = _source_records_by_region(source_glyph_masks)
    finalized_parent_ids_by_region = _finalized_parent_ids_by_region(regions)
    _cleanup_perf_contract_checkpoint(
        "cleanup_job_build",
        "module_start",
        page_id=page_id,
        region_count=len(regions or []),
        source_region_count=len(source_records_by_region),
    )
    jobs: list[CleanupJob] = []
    obligation_records: list[CleanupObligationRecord] = []
    protected_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, region in enumerate(regions or []):
        if not isinstance(region, Mapping):
            errors.append(f"region index {index} is not mapping-like")
            continue

        region_id = str(_first_present(region, "region_id", "id", default=f"region_{index}"))
        render = _mapping_or_empty(region.get("render"))
        expected_parent_id = _optional_str(
            _first_present(
                region,
                "parent_execution_bundle_id",
                "parent_logical_text_unit_id",
                "parent_execution_parent_id",
                default=_first_present(
                    render,
                    "parent_execution_bundle_id",
                    "parent_logical_text_unit_id",
                    "parent_execution_parent_id",
                    default=None,
                ),
            )
        )
        direct_source_records = _source_records_for_finalized_parent(
            source_records_by_region.get(region_id, []),
            region_id=region_id,
            expected_parent_id=expected_parent_id,
        )
        expanded_source_records = _expand_cleanup_unit_source_records(
            region_id=region_id,
            source_records=direct_source_records,
            source_records_by_region=source_records_by_region,
            expected_parent_id=expected_parent_id,
        )
        cleanup_unit = _cleanup_unit_for_region(
            page_id=page_id,
            region_id=region_id,
            source_records=expanded_source_records,
            expected_parent_id=expected_parent_id,
            finalized_parent_ids_by_region=finalized_parent_ids_by_region,
        )
        parent_execution_authoritative = bool(
            region.get("parent_execution_authoritative")
            or render.get("parent_execution_authoritative")
            or str(region.get("execution_region_authority") or "")
            == "parent_execution_bundle"
            or str(render.get("execution_region_authority") or "")
            == "parent_execution_bundle"
        )
        parent_cleanup_required = bool(
            region.get("cleanup_required")
            or render.get("cleanup_required")
        )
        parent_execution_id = str(
            region.get("parent_execution_bundle_id")
            or region.get("parent_logical_text_unit_id")
            or region_id
        )
        if parent_execution_authoritative and parent_execution_id:
            cleanup_unit = _CleanupUnit(
                unit_id=f"cu_{_safe_id(page_id)}_{_safe_id(parent_execution_id)}",
                level="parent",
                anchor_region_id=parent_execution_id,
                child_region_ids=[parent_execution_id],
                source_records=cleanup_unit.source_records,
                required_source_glyph_mask_ids=cleanup_unit.required_source_glyph_mask_ids,
                missing_source_glyph_mask_ids=cleanup_unit.missing_source_glyph_mask_ids,
                text_block_root_id=str(region.get("text_block_root_id") or "") or None,
                parent_logical_text_unit_id=parent_execution_id,
                text_area_container_id=str(region.get("text_area_container_id") or "") or None,
                ambiguous_reason="",
            )
        source_records = cleanup_unit.source_records
        route_intent = _string_from_first(
            render,
            region,
            source_records,
            keys=(
                "text_area_route_intent",
                "route_intent",
                "route",
                "intent",
            ),
        )
        semantic_class = _string_from_first(
            region,
            render,
            source_records,
            keys=("semantic_class", "type", "region_type", "kind"),
        )
        cleanup_mode = _string_from_first(
            render,
            region,
            source_records,
            keys=("cleanup_mode", "cleanup", "erasure_mode"),
        )
        translated_text = _first_present(
            region,
            "translation",
            "translated_text",
            default=_first_present(render, "translation", "translated_text", default=""),
        )
        source_text = _first_present(
            region,
            "ocr_text",
            "source_text",
            "text",
            default=_first_present(render, "source_text", "ocr_text", "text", default=""),
        )
        predicate_class = _cleanup_obligation_predicate(
            region=region,
            render=render,
            translated_text=translated_text,
            route_intent=route_intent,
            semantic_class=semantic_class,
            cleanup_mode=cleanup_mode,
            source_records=source_records,
        )
        cleanup_only_obligation = predicate_class == "accepted_cleanup_only_obligation"
        accepted_cleanup_obligation = _predicate_is_accepted_cleanup_obligation(predicate_class)
        protected_reason = _protected_reason(region, render, route_intent, semantic_class, cleanup_mode)
        base_record = _base_accounting_record(
            page_id=page_id,
            region_id=region_id,
            route_intent=route_intent,
            semantic_class=semantic_class,
            cleanup_mode=cleanup_mode,
        )

        if cleanup_unit.ambiguous_reason:
            if accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="cleanup_contract_error_invalid_input",
                        cleanup_outcome_state="cleanup_contract_error_invalid_input",
                        first_failing_stage="cleanup_contract_error_invalid_input",
                        failure_reason=cleanup_unit.ambiguous_reason,
                        source_records=source_records,
                        represented_region_ids=cleanup_unit.child_region_ids,
                    )
                )
            skipped_records.append({**base_record, "reason": "cleanup_unit_membership_ambiguous", "detail": cleanup_unit.ambiguous_reason})
            continue

        if _explicitly_inactive(region, render):
            if accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="cleanup_contract_error_invalid_input",
                        cleanup_outcome_state="cleanup_contract_error_invalid_input",
                        first_failing_stage="cleanup_contract_error_invalid_input",
                        failure_reason="region_inactive",
                        source_records=source_records,
                    )
                )
            skipped_records.append({**base_record, "reason": "region_inactive"})
            continue

        if protected_reason:
            if accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="valid_preserve_or_non_cleanup_owned",
                        cleanup_outcome_state="preserve_or_non_cleanup_owned",
                        first_failing_stage="intentional_preserve_valid",
                        failure_reason=protected_reason,
                        source_records=source_records,
                        explicit_preserve_evidence=protected_reason,
                    )
                )
            elif predicate_class == "preserve_or_non_cleanup_owned":
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="valid_preserve_or_non_cleanup_owned",
                        cleanup_outcome_state="preserve_or_non_cleanup_owned",
                        first_failing_stage="intentional_preserve_valid",
                        failure_reason=protected_reason,
                        source_records=source_records,
                        explicit_preserve_evidence=protected_reason,
                    )
                )
            protected_records.append({**base_record, "reason": protected_reason})
            continue

        if not _truthy_text(translated_text) and not cleanup_only_obligation:
            if accepted_cleanup_obligation:
                represented_job_id, represented_regions = _represented_cleanup_job_hint(
                    page_id=page_id,
                    region_id=region_id,
                    regions=regions,
                    source_records=source_records,
                    source_records_by_region=source_records_by_region,
                )
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status=(
                            "represented_by_parent_cleanup_obligation"
                            if represented_job_id
                            else "cleanup_contract_error_missing_sourceglyph"
                        ),
                        cleanup_outcome_state=(
                            "cleanup_pending_downstream"
                            if represented_job_id
                            else "cleanup_contract_error_missing_sourceglyph"
                        ),
                        first_failing_stage=(
                            ""
                            if represented_job_id
                            else "cleanup_contract_error_missing_sourceglyph"
                        ),
                        failure_reason=(
                            "represented_child_source_text_covered_by_parent_cleanup_obligation"
                            if represented_job_id
                            else "transferred_child_has_no_translated_text_and_no_representing_cleanup_obligation"
                        ),
                        source_records=source_records,
                        representing_cleanup_job_id=represented_job_id,
                        represented_region_ids=represented_regions,
                    )
                )
                skipped_records.append(
                    {
                        **base_record,
                        "reason": (
                            "represented_child_cleanup_obligation"
                            if represented_job_id
                            else "accepted_transferred_child_missing_cleanup_obligation"
                        ),
                    }
                )
                continue
            if predicate_class == "translation_intent_candidate":
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="audit_only_translation_intent",
                        cleanup_outcome_state="",
                        first_failing_stage="",
                        failure_reason="translation_intent_not_accepted_or_render_obligated",
                        source_records=source_records,
                    )
                )
            skipped_records.append({**base_record, "reason": "no_translated_text"})
            continue

        if _region_flag(region, render, "ignore", "skip_translation", "no_translation"):
            if accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="cleanup_contract_error_invalid_input",
                        cleanup_outcome_state="cleanup_contract_error_invalid_input",
                        first_failing_stage="cleanup_contract_error_invalid_input",
                        failure_reason="region_marked_not_translated",
                        source_records=source_records,
                    )
                )
            skipped_records.append({**base_record, "reason": "region_marked_not_translated"})
            continue

        if not source_records and not (
            parent_execution_authoritative and parent_cleanup_required
        ):
            if accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="cleanup_contract_error_missing_sourceglyph",
                        cleanup_outcome_state="cleanup_contract_error_missing_sourceglyph",
                        first_failing_stage="cleanup_contract_error_missing_sourceglyph",
                        failure_reason="missing_source_glyph_evidence",
                        source_records=source_records,
                    )
                )
            skipped_records.append(
                {**base_record, "reason": "missing_source_glyph_evidence_phase0_no_job"}
            )
            continue

        cleanup_class, classification_reason, skip_reason = _classify_cleanup_class(
            region=region,
            render=render,
            source_records=source_records,
            route_intent=route_intent,
            semantic_class=semantic_class,
            cleanup_mode=cleanup_mode,
        )
        job_blocking_reason = ""
        if skip_reason:
            skip_record = {
                **base_record,
                "reason": skip_reason,
                "cleanup_class": cleanup_class.value,
                "classification_reason": classification_reason,
            }
            if cleanup_class == CleanupClass.ART_ENTANGLED_AMBIGUOUS:
                skip_record.update(
                    {
                        "unsupported_future_cleanup_class": CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
                        "unsupported_cleanup_reason": "artifact_risk_or_non_flat_background_not_caption_flat_background",
                    }
                )
            if accepted_cleanup_obligation and cleanup_class != CleanupClass.PRESERVE_SFX_DECORATIVE:
                job_blocking_reason = skip_reason
                if not classification_reason:
                    classification_reason = skip_reason
                skip_record["reason"] = "job_created_for_accepted_unit_with_blocking_cleanup_class"
                skip_record["blocking_reason"] = skip_reason
            elif accepted_cleanup_obligation:
                obligation_records.append(
                    _obligation_record(
                        page_id=page_id,
                        region_id=region_id,
                        predicate_class=predicate_class,
                        obligation_status="blocked_non_executable_cleanup_class",
                        cleanup_outcome_state=_accounting_outcome_for_blocker(skip_reason, cleanup_class),
                        first_failing_stage="cleanup_job_accounted_non_executable",
                        failure_reason=skip_reason,
                        source_records=source_records,
                    )
                )
                skipped_records.append(skip_record)
                continue
            else:
                skipped_records.append(skip_record)
                continue

        evidence = [_json_safe(record) for record in source_records]
        source_mask_ids = cleanup_unit.required_source_glyph_mask_ids or _source_mask_ids(source_records)
        first_record = _anchor_source_record(source_records, region_id) or (source_records[0] if source_records else {})
        cleanup_job_id = f"cjob_{_safe_id(page_id)}_{_safe_id(cleanup_unit.anchor_region_id or region_id)}"
        parent_logical_text_unit_id = _optional_str(
            _first_present(
                region,
                "parent_logical_text_unit_id",
                default=cleanup_unit.parent_logical_text_unit_id or _first_present(first_record, "parent_logical_text_unit_id", default=None),
            )
        )
        parent_execution_bundle_id = _optional_str(
            _first_present(
                region,
                "parent_execution_bundle_id",
                "parent_id",
                default=(
                    parent_logical_text_unit_id
                    or cleanup_unit.parent_logical_text_unit_id
                    or _first_present(first_record, "parent_execution_bundle_id", "parent_id", "parent_logical_text_unit_id", default=None)
                ),
            )
        ) or parent_logical_text_unit_id or region_id
        target_region_ids = _parent_first_target_region_ids(
            parent_execution_bundle_id or parent_logical_text_unit_id or region_id,
            cleanup_unit.child_region_ids or [region_id],
        )
        if accepted_cleanup_obligation:
            obligation_records.append(
                _obligation_record(
                    page_id=page_id,
                    region_id=region_id,
                    predicate_class=predicate_class,
                    obligation_status="cleanup_job_created",
                    cleanup_outcome_state="cleanup_pending_downstream",
                    first_failing_stage="",
                    failure_reason="",
                    source_records=source_records,
                    cleanup_job_id=cleanup_job_id,
                    represented_region_ids=cleanup_unit.child_region_ids,
                )
            )
        jobs.append(
            CleanupJob(
                page_id=page_id,
                cleanup_job_id=cleanup_job_id,
                target_region_ids=target_region_ids,
                logical_text_unit_ids=_list_strings(
                    _first_present(
                        region,
                        "logical_text_unit_id",
                        "logical_block_id",
                        "logical_text_unit_ids",
                        default=(
                            [parent_logical_text_unit_id]
                            if parent_logical_text_unit_id
                            else []
                        ),
                    )
                ),
                cleanup_class=cleanup_class,
                route_intent=route_intent,
                semantic_class=semantic_class,
                source_text_present=_truthy_text(source_text),
                translated_text_present=_truthy_text(translated_text),
                source_glyph_mask_ids=source_mask_ids,
                source_glyph_evidence=evidence,
                allowed_cleanup_area=_bbox_from_first(
                    first_record,
                    render,
                    region,
                    keys=("cleanup_allowed_area", "allowed_cleanup_area", "text_area_container_bbox"),
                ),
                classification_reason=classification_reason or job_blocking_reason,
                text_block_root_id=_optional_str(
                    _first_present(region, "text_block_root_id", default=cleanup_unit.text_block_root_id or _first_present(first_record, "text_block_root_id", default=None))
                ),
                parent_logical_text_unit_id=parent_logical_text_unit_id,
                parent_execution_bundle_id=parent_execution_bundle_id,
                text_area_container_id=_optional_str(
                    _first_present(
                        region,
                        "text_area_container_id",
                        default=cleanup_unit.text_area_container_id or _first_present(first_record, "text_area_container_id", default=None),
                    )
                ),
                container_type=_string_from_first(
                    render,
                    region,
                    first_record,
                    keys=("container_type", "text_area_container_type"),
                ),
                cleanup_mode=cleanup_mode,
                source_glyph_erasure_expected_area_bbox=_bbox_from_first(
                    first_record,
                    render,
                    region,
                    keys=("source_glyph_erasure_expected_area_bbox", "expected_area_bbox"),
                ),
                source_glyph_erasure_bbox=_bbox_from_first(
                    first_record,
                    render,
                    region,
                    keys=("source_glyph_erasure_bbox", "source_glyph_bbox", "bbox"),
                ),
                cleanup_unit_id=cleanup_unit.unit_id,
                cleanup_unit_level=cleanup_unit.level,
                cleanup_unit_anchor_region_id=cleanup_unit.anchor_region_id,
                cleanup_unit_child_region_ids=cleanup_unit.child_region_ids,
                cleanup_unit_required_source_glyph_mask_ids=cleanup_unit.required_source_glyph_mask_ids,
                cleanup_unit_missing_source_glyph_mask_ids=cleanup_unit.missing_source_glyph_mask_ids,
                cleanup_only_obligation=cleanup_only_obligation,
            )
        )
        _cleanup_perf_contract_checkpoint(
            "cleanup_job_membership",
            "created",
            page_id=page_id,
            region_id=region_id,
            cleanup_job_id=cleanup_job_id,
            cleanup_class=cleanup_class.value,
            cleanup_unit_id=cleanup_unit.unit_id,
            cleanup_unit_level=cleanup_unit.level,
            target_region_count=len(cleanup_unit.child_region_ids or [region_id]),
            target_region_ids=cleanup_unit.child_region_ids or [region_id],
            required_source_glyph_count=len(source_mask_ids),
            required_source_glyph_mask_ids=source_mask_ids,
            source_evidence_count=len(evidence),
            classification_reason=classification_reason or job_blocking_reason,
        )

    result = CleanupJobBuildResult(
        page_id=page_id,
        version=CLEANUP_CONTRACT_VERSION,
        jobs=jobs,
        obligation_records=obligation_records,
        protected_records=protected_records,
        skipped_records=skipped_records,
        errors=errors,
    )
    _cleanup_perf_contract_checkpoint(
        "cleanup_job_build",
        "module_end",
        page_id=page_id,
        job_count=len(jobs),
        obligation_count=len(obligation_records),
        protected_count=len(protected_records),
        skipped_count=len(skipped_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return result


def build_cleanup_job_candidates_for_parent_bundles(
    *,
    page_id: str,
    parent_execution_bundles: Sequence[Any],
    source_glyph_masks: Any,
) -> CleanupJobBuildResult:
    """Build cleanup jobs from finalized parent execution bundles."""

    return build_cleanup_job_candidates(
        page_id=page_id,
        regions=parent_execution_region_records(parent_execution_bundles),
        source_glyph_masks=source_glyph_masks,
    )


def _dataclass_audit_dict(obj: Any, *, omit: set[str] | None = None) -> dict[str, Any]:
    omitted = omit or set()
    return {
        item.name: _json_safe(getattr(obj, item.name))
        for item in fields(obj)
        if item.name not in omitted
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and hasattr(value, "to_audit_dict"):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items() if not _looks_like_raw_array(key)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _looks_like_raw_array(key: Any) -> bool:
    name = str(key).lower()
    return name in {"mask", "image", "array", "pixels_raw", "bitmap", "crop"}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _string_from_first(*sources: Any, keys: Sequence[str]) -> str:
    for source in sources:
        if isinstance(source, Mapping):
            value = _first_present(source, *keys, default=None)
            if value is not None:
                return str(value)
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            for item in source:
                if isinstance(item, Mapping):
                    value = _first_present(item, *keys, default=None)
                    if value is not None:
                        return str(value)
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _truthy_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _region_flag(*items: Any) -> bool:
    sources: list[Mapping[str, Any]] = []
    names: list[str] = []
    for item in items:
        if isinstance(item, Mapping) and not names:
            sources.append(item)
        else:
            names.append(str(item))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in names:
            value = source.get(name)
            if isinstance(value, str):
                if value.strip().lower() in {"1", "true", "yes", "required"}:
                    return True
            elif bool(value):
                return True
    return False


def _predicate_is_accepted_cleanup_obligation(predicate_class: str) -> bool:
    return predicate_class in {
        "accepted_translated_render_obligation",
        "accepted_cleanup_only_obligation",
    }


def _cleanup_only_authorized(
    *,
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
) -> bool:
    # Route-owned text must remain translation-owned. Cleanup-only is not a
    # valid authority source for speech/caption/background text.
    return False


def _cleanup_obligation_predicate(
    *,
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    translated_text: Any,
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
    source_records: list[dict[str, Any]],
) -> str:
    if _truthy_text(translated_text):
        return "accepted_translated_render_obligation"
    if _cleanup_only_authorized(
        region=region,
        render=render,
        route_intent=route_intent,
        semantic_class=semantic_class,
        cleanup_mode=cleanup_mode,
    ):
        return "accepted_cleanup_only_obligation"
    if _region_flag(
        region,
        render,
        "sent_to_translation",
        "render_attempted",
        "renderer_draw_attempted",
        "render_eligible",
        "eligible_for_rendering",
        "accepted_translation",
        "accepted_translated",
    ):
        return "accepted_translated_render_obligation"
    combined = " ".join(
        str(value or "")
        for value in (
            route_intent,
            semantic_class,
            cleanup_mode,
            _string_from_first(region, render, keys=("classification_reason", "skip_reason")),
        )
    ).lower()
    if "transfer" in combined and source_records:
        return "accepted_translated_render_obligation"
    if _protected_reason(region, render, route_intent, semantic_class, cleanup_mode):
        return "preserve_or_non_cleanup_owned"
    if any(marker in combined for marker in ("translate", "speech", "caption", "narration", "background", "sign")):
        return "translation_intent_candidate"
    return ""


def _protected_reason(
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
) -> str:
    combined = " ".join((route_intent, semantic_class, cleanup_mode)).lower()
    if cleanup_mode.lower() in {"preserve", "preserved"}:
        return f"cleanup_mode_{cleanup_mode.lower()}"
    if semantic_class.lower() in {
        "art",
        "art_embedded_text",
        "sfx",
        "decorative",
        "decorative_text",
        "preserve",
        "non_text",
        "non-text",
    }:
        return f"semantic_class_{semantic_class.lower()}"
    if _region_flag(region, render, "preserve", "is_sfx", "is_decorative", "is_art", "decorative"):
        return "region_metadata_preserve_sfx_decorative_or_art"
    for marker in ("preserve", "sfx", "decorative", "art_only", "non_translation_art", "non_text", "non-text"):
        if marker in combined:
            return f"route_or_semantic_{marker}"
    return ""


def _obligation_record(
    *,
    page_id: str,
    region_id: str,
    predicate_class: str,
    obligation_status: str,
    cleanup_outcome_state: str,
    first_failing_stage: str,
    failure_reason: str,
    source_records: list[dict[str, Any]],
    cleanup_job_id: str = "",
    representing_cleanup_job_id: str = "",
    represented_region_ids: list[str] | None = None,
    explicit_preserve_evidence: str = "",
) -> CleanupObligationRecord:
    return CleanupObligationRecord(
        page_id=page_id,
        region_id=region_id,
        predicate_class=predicate_class,
        obligation_status=obligation_status,
        cleanup_outcome_state=cleanup_outcome_state,
        first_failing_stage=first_failing_stage,
        failure_reason=failure_reason,
        cleanup_job_id=cleanup_job_id,
        representing_cleanup_job_id=representing_cleanup_job_id,
        represented_region_ids=represented_region_ids or [],
        source_glyph_mask_ids=_source_mask_ids(source_records),
        explicit_preserve_evidence=explicit_preserve_evidence,
    )


def _cleanup_unit_for_region(
    *,
    page_id: str,
    region_id: str,
    source_records: list[dict[str, Any]],
    expected_parent_id: str | None = None,
    finalized_parent_ids_by_region: Mapping[str, str] | None = None,
) -> _CleanupUnit:
    records = [record for record in source_records or [] if isinstance(record, Mapping)]
    child_region_ids = _region_ids_from_source_records(records) or [region_id]
    if expected_parent_id:
        owner_by_region = finalized_parent_ids_by_region or {}
        child_region_ids = [
            child_region_id
            for child_region_id in child_region_ids
            if not owner_by_region.get(child_region_id)
            or owner_by_region.get(child_region_id) == expected_parent_id
        ]
    if region_id not in child_region_ids:
        child_region_ids.append(region_id)
        child_region_ids = sorted(set(child_region_ids))
    parent_ids = _source_record_unit_ids(
        records,
        keys=(
            "parent_logical_text_unit_id",
            "parent_id",
            "logical_text_unit_id",
        ),
    )
    root_ids = _source_record_unit_ids(
        records,
        keys=(
            "text_block_root_id",
            "root_id",
            "source_glyph_mask_text_block_root_id",
        ),
    )
    container_ids = _source_record_unit_ids(
        records,
        keys=("text_area_container_id", "container_id"),
    )
    ambiguous_reason = ""
    if len(parent_ids) > 1:
        ambiguous_reason = f"multiple_parent_logical_text_unit_ids:{','.join(parent_ids)}"
    elif len(root_ids) > 1 and not parent_ids:
        ambiguous_reason = f"multiple_text_block_root_ids:{','.join(root_ids)}"

    parent_id = parent_ids[0] if len(parent_ids) == 1 else None
    root_id = root_ids[0] if len(root_ids) == 1 else None
    container_id = container_ids[0] if len(container_ids) == 1 else None
    if parent_id:
        level = "parent"
        unit_key = parent_id
    elif root_id and len(child_region_ids) > 1:
        level = "root"
        unit_key = root_id
    else:
        level = "standalone"
        unit_key = region_id

    required_ids = _source_mask_ids(records)
    missing_ids = [
        f"missing_source_glyph_id_for_{record_id}"
        for record_id in child_region_ids
        if not any(str(_first_present(record, "source_glyph_mask_id", "mask_id", "id", default="") or "") for record in records if record_id in _region_ids_from_source_record(record))
    ]
    return _CleanupUnit(
        unit_id=f"cu_{_safe_id(page_id)}_{_safe_id(unit_key)}",
        level=level,
        anchor_region_id=region_id,
        child_region_ids=sorted(set(child_region_ids)),
        source_records=records,
        required_source_glyph_mask_ids=required_ids,
        missing_source_glyph_mask_ids=missing_ids,
        text_block_root_id=root_id,
        parent_logical_text_unit_id=parent_id,
        text_area_container_id=container_id,
        ambiguous_reason=ambiguous_reason,
    )


def _finalized_parent_ids_by_region(
    regions: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for region in regions or []:
        if not isinstance(region, Mapping):
            continue
        render = _mapping_or_empty(region.get("render"))
        region_id = str(_first_present(region, "region_id", "id", default="") or "")
        parent_id = _optional_str(
            _first_present(
                region,
                "parent_execution_bundle_id",
                "parent_logical_text_unit_id",
                "parent_execution_parent_id",
                default=_first_present(
                    render,
                    "parent_execution_bundle_id",
                    "parent_logical_text_unit_id",
                    "parent_execution_parent_id",
                    default=None,
                ),
            )
        )
        if region_id and parent_id:
            output[region_id] = parent_id
    return output


def _parent_first_target_region_ids(parent_id: str | None, region_ids: Sequence[Any]) -> list[str]:
    output: list[str] = []
    parent = str(parent_id or "").strip()
    if parent:
        output.append(parent)
    for item in region_ids or []:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _expand_cleanup_unit_source_records(
    *,
    region_id: str,
    source_records: list[dict[str, Any]],
    source_records_by_region: Mapping[str, list[dict[str, Any]]],
    expected_parent_id: str | None = None,
) -> list[dict[str, Any]]:
    """Expand anchor evidence to its represented child SourceGlyph records.

    Expansion follows only existing region/source links already present in
    SourceGlyphMask audit metadata. It does not infer a new parent/root owner.
    """

    records: list[dict[str, Any]] = []
    seen_masks: set[str] = set()

    def add_record(record: Mapping[str, Any]) -> bool:
        mask_id = str(_first_present(record, "source_glyph_mask_id", "mask_id", "id", default="") or "")
        key = mask_id or f"raw_{id(record)}"
        if key in seen_masks:
            return False
        seen_masks.add(key)
        records.append(dict(record))
        return True

    for record in source_records or []:
        if isinstance(record, Mapping):
            add_record(record)

    pending = set(_region_ids_from_source_records(records) or [region_id])
    pending.add(region_id)
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for record in source_records_by_region.get(current, []) or []:
            if not isinstance(record, Mapping):
                continue
            if expected_parent_id and not _source_record_matches_finalized_parent(
                record,
                region_id=current,
                expected_parent_id=expected_parent_id,
            ):
                continue
            added = add_record(record)
            if added:
                for related_id in _region_ids_from_source_record(record):
                    if related_id not in visited:
                        pending.add(related_id)
    return records


def _source_records_for_finalized_parent(
    records: Sequence[Mapping[str, Any]],
    *,
    region_id: str,
    expected_parent_id: str | None,
) -> list[dict[str, Any]]:
    if not expected_parent_id:
        return [dict(record) for record in records if isinstance(record, Mapping)]
    return [
        dict(record)
        for record in records
        if isinstance(record, Mapping)
        and _source_record_matches_finalized_parent(
            record,
            region_id=region_id,
            expected_parent_id=expected_parent_id,
        )
    ]


def _source_record_matches_finalized_parent(
    record: Mapping[str, Any],
    *,
    region_id: str,
    expected_parent_id: str,
) -> bool:
    parent_ids = _source_record_unit_ids(
        [record],
        keys=(
            "parent_execution_bundle_id",
            "parent_logical_text_unit_id",
            "parent_id",
            "logical_text_unit_id",
        ),
    )
    if parent_ids:
        return expected_parent_id in parent_ids
    direct_region_id = _optional_str(
        _first_present(record, "region_id", "target_region_id", default=None)
    )
    return direct_region_id == region_id


def _source_record_unit_ids(records: Sequence[Mapping[str, Any]], *, keys: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        for key in keys:
            value = _first_present(record, key, default=None)
            for item in _sequence_or_single(value):
                text = str(item or "").strip()
                if text and text not in ids:
                    ids.append(text)
        compatibility = record.get("compatibility_source_fields")
        if isinstance(compatibility, Mapping):
            for item in _source_record_unit_ids([compatibility], keys=keys):
                if item not in ids:
                    ids.append(item)
    return sorted(ids)


def _anchor_source_record(records: Sequence[Mapping[str, Any]], region_id: str) -> dict[str, Any] | None:
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        direct = _optional_str(_first_present(record, "region_id", "target_region_id", default=None))
        if direct == region_id:
            return dict(record)
    for record in records or []:
        if isinstance(record, Mapping):
            return dict(record)
    return None


def _accounting_outcome_for_blocker(reason: str, cleanup_class: CleanupClass) -> str:
    text = f"{reason} {cleanup_class.value}".lower()
    if "sourceglyph" in text or "source_glyph" in text:
        return "cleanup_contract_error_missing_sourceglyph"
    if "erase" in text or "too_large" in text:
        return "cleanup_degraded_minimal_erasure_applied"
    if "artifact" in text or "art_entangled" in text or "background_art" in text or "ambiguous" in text:
        return "cleanup_partially_completed_with_warnings"
    if "preserve" in text or "sfx" in text or "decorative" in text:
        return "preserve_or_non_cleanup_owned"
    return "cleanup_contract_error_invalid_input"


def _represented_cleanup_job_hint(
    *,
    page_id: str,
    region_id: str,
    regions: Sequence[Mapping[str, Any]],
    source_records: list[dict[str, Any]],
    source_records_by_region: Mapping[str, list[dict[str, Any]]],
) -> tuple[str, list[str]]:
    related_ids = set(_region_ids_from_source_records(source_records))
    related_ids.discard(region_id)
    if not related_ids:
        return "", []
    region_by_id = {
        str(_first_present(region, "region_id", "id", default="")): region
        for region in regions or []
        if isinstance(region, Mapping)
    }
    for candidate_id in sorted(related_ids):
        candidate = region_by_id.get(candidate_id)
        if not candidate:
            continue
        candidate_render = _mapping_or_empty(candidate.get("render"))
        translated_text = _first_present(
            candidate,
            "translation",
            "translated_text",
            default=_first_present(candidate_render, "translation", "translated_text", default=""),
        )
        if not _truthy_text(translated_text):
            continue
        candidate_source = source_records_by_region.get(candidate_id, [])
        if candidate_source:
            return (
                f"cjob_{_safe_id(page_id)}_{_safe_id(candidate_id)}",
                sorted(related_ids | {region_id, candidate_id}),
            )
    return "", sorted(related_ids | {region_id})


def _base_accounting_record(
    *,
    page_id: str,
    region_id: str,
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "region_id": region_id,
        "route_intent": route_intent,
        "semantic_class": semantic_class,
        "cleanup_mode": cleanup_mode,
    }


def _explicitly_inactive(*sources: Mapping[str, Any]) -> bool:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("active", "is_active", "translation_active", "active_translated"):
            if key not in source:
                continue
            value = source.get(key)
            if isinstance(value, str):
                if value.strip().lower() in {"0", "false", "no", "inactive"}:
                    return True
            elif value is False:
                return True
    return False


def _classify_cleanup_class(
    *,
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    source_records: list[dict[str, Any]],
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
) -> tuple[CleanupClass, str, str]:
    metadata = " ".join(
        str(item)
        for item in (
            route_intent,
            semantic_class,
            cleanup_mode,
            _string_from_first(region, render, keys=("container_type", "text_area_container_type")),
            *(
                str(_first_present(record, "generation_method", "method", "source_glyph_mask_id", default=""))
                for record in source_records
            ),
        )
    ).lower()

    speech_bubble_authorized = _speech_bubble_cleanup_authorized(
        metadata=metadata,
        route_intent=route_intent,
        semantic_class=semantic_class,
        cleanup_mode=cleanup_mode,
    )
    contract_class = _cleanup_class_from_source_contracts(source_records)
    if contract_class == CleanupClass.PRESERVE_SFX_DECORATIVE:
        return (
            contract_class,
            "",
            f"source_glyph_class_contract_excludes_{contract_class.value}",
        )
    if contract_class == CleanupClass.SMALL_REACTION and speech_bubble_authorized:
        return (
            CleanupClass.SPEECH_FLAT_BUBBLE,
            "translated_speech_bubble_authority_overrides_small_reaction_source_glyph_strategy",
            "",
        )
    if contract_class in {
        CleanupClass.ART_ENTANGLED_AMBIGUOUS,
        CleanupClass.BACKGROUND_ART_TEXT,
    }:
        return (
            CleanupClass.CAPTION_DARK_OR_SCREENTONE,
            f"source_glyph_class_contract_soft_risk_{contract_class.value}_strategy",
            "",
        )
    if contract_class is not None:
        return (
            contract_class,
            f"source_glyph_class_contract_{contract_class.value}",
            "",
        )
    if len(source_records or []) <= 1 and any(_truthy(record.get("cleanup_visual_artifact_risk")) for record in source_records):
        return (
            CleanupClass.CAPTION_DARK_OR_SCREENTONE,
            "cleanup_visual_artifact_risk_soft_strategy_metadata",
            "",
        )

    if "side_caption_glyph_local" in metadata:
        return (
            CleanupClass.SIDE_CAPTION_GLYPH_LOCAL,
            "source_glyph_metadata_side_caption_glyph_local",
            "",
        )
    if any(marker in metadata for marker in ("caption", "background", "narration")):
        return (
            CleanupClass.CAPTION_FLAT_BACKGROUND,
            "metadata_caption_background_no_dark_or_art_inference",
            "",
        )
    if any(marker in metadata for marker in ("speech", "bubble", "dialogue", "translate_speech")):
        return (
            CleanupClass.SPEECH_FLAT_BUBBLE,
            "metadata_translate_speech_no_pixel_inference",
            "",
        )
    return (
        CleanupClass.CAPTION_DARK_OR_SCREENTONE,
        "metadata_insufficient_cleanup_strategy_default",
        "",
    )


def _speech_bubble_cleanup_authorized(
    *,
    metadata: str,
    route_intent: str,
    semantic_class: str,
    cleanup_mode: str,
) -> bool:
    route = str(route_intent or "").lower()
    semantic = str(semantic_class or "").lower()
    cleanup = str(cleanup_mode or "").lower()
    if "translate_speech" not in route:
        return False
    if "speech" in semantic or "bubble" in semantic:
        return True
    if cleanup in {"bubble", "speech", "speech_bubble"}:
        return True
    return "speech" in metadata and "bubble" in metadata


def _cleanup_class_from_source_contracts(source_records: list[dict[str, Any]]) -> CleanupClass | None:
    for record in source_records:
        value = _first_present(
            record,
            "class_specific_contract",
            "cleanup_class",
            default=None,
        )
        if value is None and isinstance(record.get("compatibility_source_fields"), Mapping):
            value = _first_present(
                record.get("compatibility_source_fields", {}),
                "class_specific_contract",
                "cleanup_class",
                default=None,
            )
        normalized = str(value or "").strip().lower()
        if not normalized:
            continue
        for cleanup_class in CleanupClass:
            if normalized == cleanup_class.value:
                return cleanup_class
    return None


def _source_records_by_region(source_glyph_masks: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record in _iter_source_records(source_glyph_masks):
        normalized = _normalize_source_record(record)
        region_ids = _region_ids_from_source_record(normalized)
        if not region_ids:
            continue
        for region_id in region_ids:
            grouped.setdefault(region_id, []).append(normalized)

    return grouped


def _region_ids_from_source_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    region_ids: list[str] = []
    for record in records or []:
        if isinstance(record, Mapping):
            region_ids.extend(_region_ids_from_source_record(record))
    return sorted(set(region_ids))


def _region_ids_from_source_record(record: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    direct = _optional_str(_first_present(record, "region_id", "target_region_id", default=None))
    if direct:
        ids.append(direct)
    for key in ("text_instance_ids", "target_region_ids", "represented_region_ids"):
        value = record.get(key)
        for item in _sequence_or_single(value):
            text = str(item or "")
            if text.startswith("r") and text[1:].isdigit():
                ids.append(text)
    compatibility = record.get("compatibility_source_fields")
    if isinstance(compatibility, Mapping):
        ids.extend(_region_ids_from_source_record(compatibility))
    return sorted(set(ids))


def _iter_source_records(source_glyph_masks: Any) -> list[Any]:
    if source_glyph_masks is None:
        return []
    if isinstance(source_glyph_masks, Mapping):
        records: list[Any] = []
        for key in ("source_glyph_mask_coverage_records", "source_glyph_masks", "coverage_records"):
            value = source_glyph_masks.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                records.extend(value)
        masks_by_region = source_glyph_masks.get("masks_by_region")
        if isinstance(masks_by_region, Mapping):
            for region_id, value in masks_by_region.items():
                for item in _sequence_or_single(value):
                    records.append(_source_record_for_region(region_id, item))
        return records

    records = []
    for attr in ("source_glyph_mask_coverage_records", "source_glyph_masks", "coverage_records"):
        value = getattr(source_glyph_masks, attr, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            records.extend(value)
    masks_by_region = getattr(source_glyph_masks, "masks_by_region", None)
    if isinstance(masks_by_region, Mapping):
        for region_id, value in masks_by_region.items():
            for item in _sequence_or_single(value):
                records.append(_source_record_for_region(region_id, item))
    return records


def _source_record_for_region(region_id: Any, record: Any) -> dict[str, Any]:
    if hasattr(record, "to_audit_dict"):
        audit_record = record.to_audit_dict()
        if isinstance(audit_record, Mapping):
            return {"region_id": region_id, **dict(audit_record)}
    if isinstance(record, Mapping):
        return {"region_id": region_id, **dict(record)}
    return {"region_id": region_id, "mask": record}


def _normalize_source_record(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_audit_dict"):
        value = record.to_audit_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(record, Mapping):
        return dict(record)
    output: dict[str, Any] = {}
    for name in (
        "region_id",
        "target_region_id",
        "source_glyph_mask_id",
        "mask_id",
        "generation_method",
        "method",
        "source_glyph_erasure_bbox",
        "source_glyph_erasure_expected_area_bbox",
        "cleanup_allowed_area",
        "text_area_container_bbox",
        "text_area_container_id",
        "container_id",
        "text_block_root_id",
        "root_id",
        "source_glyph_mask_text_block_root_id",
        "parent_execution_bundle_id",
        "parent_logical_text_unit_id",
        "parent_id",
        "logical_text_unit_id",
        "text_instance_ids",
        "target_region_ids",
        "represented_region_ids",
        "child_segment_ids",
        "cleanup_class",
        "class_specific_contract",
        "cleanup_visual_artifact_risk",
    ):
        if hasattr(record, name):
            output[name] = getattr(record, name)
    return output


def _sequence_or_single(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _source_mask_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        value = _first_present(record, "source_glyph_mask_id", "mask_id", "id", default=None)
        if value is not None and str(value) not in ids:
            ids.append(str(value))
    return ids


def _bbox_from_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> list[int] | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        value = _first_present(source, *keys, default=None)
        bbox = _coerce_bbox(value)
        if bbox is not None:
            return bbox
    return None


def _coerce_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        return [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _safe_id(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
