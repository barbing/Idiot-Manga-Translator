"""Cleanup planning, execution, and proof contracts.

This module consumes accepted CleanupMask records as the pre-render production
obligations for source-text removal. SourceGlyph/root/bbox metadata remains
provenance and audit scope; executable cleanup pixels come from accepted
page-space foreground/erase masks, and renderer code remains text/layout only.
"""

from __future__ import annotations

import json
import os
import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from app.pipeline.cleanup_contracts import (
    CleanupClass,
    CleanupJob,
    CleanupMask,
    CleanupPlan,
    CleanupProof,
    CleanupResult,
    ProofStatus,
)
from app.pipeline.cleanup_masks import CleanupMaskBuildResult
from app.pipeline import cleanup_execution
from app.pipeline import cleanup_backend_runner
from app.pipeline.debug_artifacts import mask_stats, write_debug_image_file
from app.pipeline.debug_runtime import diagnostic_enabled, write_diagnostic_checkpoint

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


CLEANUP_PLAN_CONTRACT_VERSION = "cleanup_plans_phase5_cleanup_mask_obligations"
CLEANUP_RUNTIME_CONTRACT_VERSION = "cleanup_runtime_phase5_cleanup_mask_obligations"
CLEANUP_UPSTREAM_COMMIT_VERSION = "cleanup_upstream_commit_phase5_pre_render_working_image"

MAX_PLAN_GROWTH_RATIO = 2.25
MAX_ERASE_MASK_PIXELS = 20_000
MAX_ERASE_BBOX_AREA = 120_000
MAX_ERASE_ALLOWED_RATIO = 0.35
MAX_ALLOWED_PAGE_RATIO = 0.20

PROOF_MASK_SOURCE_OVERLAP_RATIO = 0.90
PROOF_RESIDUAL_RATIO = 0.12
PROOF_RESIDUAL_PIXEL_FRACTION = 0.08
PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS = 16
PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO = 0.001
PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO = 0.10

RUNTIME_SUPPORTED_CLEANUP_CLASSES = {
    CleanupClass.SPEECH_FLAT_BUBBLE.value,
    CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
    CleanupClass.CAPTION_FLAT_BACKGROUND.value,
    CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
    CleanupClass.TITLE_OR_SIGN.value,
    CleanupClass.SMALL_REACTION.value,
    CleanupClass.BACKGROUND_ART_TEXT.value,
    CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
    CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
}
RUNTIME_UNSAFE_CLEANUP_CLASSES = {
    CleanupClass.PRESERVE_SFX_DECORATIVE.value,
}

VALID_CLEANUP_VISUAL_SCOPES = {
    "source_glyph_local",
    "source_glyph_union",
    "source_glyph_union_partition",
    "segmentation_component",
}
LEGACY_CLEANUP_VISUAL_SCOPE_MAP = {
    "glyph_local": "source_glyph_local",
}
ARTIFACT_RISK_PARTITION_REQUIRED_REASON = "artifact_risk_partition_required"
MAX_PARTITION_COMPONENTS = 32
CANONICAL_COMPONENT_PROJECTION_METHOD = "text_area_component_authorization_map"
COMPONENT_PROJECTION_METHODS = {
    CANONICAL_COMPONENT_PROJECTION_METHOD,
}
PROJECTION_READY_STATE = "projection_ready"
MASK_READY_STATE = "mask_ready"
PROJECTION_OUTSIDE_AUTHORIZED_AREA_STATE = "projection_outside_authorized_area"
MASK_NOT_READY_STATE = "mask_not_ready"
COMPONENT_PROJECTION_AUDIT_ONLY_REJECTION_REASONS = {
    "effective_mask_not_ready",
}
COMPONENT_PROJECTION_AUDIT_ONLY_REJECTED_COMPONENT_REASONS = {
    "effective_mask_not_ready",
    "ambiguous_component_owner_components",
}

_CLEANUP_RUNTIME_PERF_COLLECTOR: ContextVar[dict[str, Any] | None] = ContextVar(
    "cleanup_runtime_perf_collector",
    default=None,
)


@contextmanager
def collect_cleanup_runtime_perf(enabled: bool):
    """Collect passive cleanup timings without placing them in contract objects."""

    if not enabled:
        yield None
        return
    collector: dict[str, Any] = {"jobs": [], "contract": {}}
    token = _CLEANUP_RUNTIME_PERF_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _CLEANUP_RUNTIME_PERF_COLLECTOR.reset(token)


def _perf_elapsed_ms(started: float) -> float:
    return round(max(0.0, (time.perf_counter() - started) * 1000.0), 3)


def _perf_add(bucket: dict[str, Any] | None, key: str, value: float) -> None:
    if bucket is None:
        return
    bucket[key] = round(float(bucket.get(key) or 0.0) + max(0.0, float(value or 0.0)), 3)


def _page014_timeout_diag_enabled() -> bool:
    return diagnostic_enabled("MT_PAGE014_TIMEOUT_DIAGNOSTIC")


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
            module="app.pipeline.cleanup_planning",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
        )
    except Exception:
        return


def _page014_timeout_checkpoint(stage: str, event: str, **fields: Any) -> None:
    _cleanup_perf_contract_checkpoint(stage, event, **fields)
    if not _page014_timeout_diag_enabled():
        return
    try:
        write_diagnostic_checkpoint(
            "page014_timeout_checkpoints.jsonl",
            module="app.pipeline.cleanup_planning",
            stage=stage,
            event=event,
            fields=fields,
            include_monotonic=False,
        )
    except Exception:
        return


@dataclass(frozen=True)
class CleanupPlanBuildResult:
    """Strict caption-flat CleanupPlan contracts."""

    page_id: str
    version: str
    plans: list[CleanupPlan] = field(default_factory=list)
    plans_by_job_id: dict[str, list[CleanupPlan]] = field(default_factory=dict)
    plans_by_cleanup_mask_id: dict[str, list[CleanupPlan]] = field(default_factory=dict)
    backend_inventory: dict[str, Any] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "plans": [_json_safe(plan) for plan in self.plans],
            "plans_by_job_id": {
                str(job_id): [_json_safe(plan) for plan in plans]
                for job_id, plans in self.plans_by_job_id.items()
            },
            "plans_by_cleanup_mask_id": {
                str(mask_id): [_json_safe(plan) for plan in plans]
                for mask_id, plans in self.plans_by_cleanup_mask_id.items()
            },
            "backend_inventory": _json_safe(self.backend_inventory),
            "rejected_records": _json_safe(self.rejected_records),
            "protected_records": _json_safe(self.protected_records),
            "skipped_records": _json_safe(self.skipped_records),
            "errors": list(self.errors),
            "summary": {
                "cleanup_authority": "accepted_cleanup_mask_obligation",
                "plan_count": len(self.plans),
                "planned_cleanup_mask_count": len(self.plans_by_cleanup_mask_id),
                "rejected_record_count": len(self.rejected_records),
                "protected_record_count": len(self.protected_records),
                "skipped_record_count": len(self.skipped_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
                "backend_candidate_count": int(self.backend_inventory.get("candidate_count") or 0),
                "backend_available_count": int(self.backend_inventory.get("available_count") or 0),
            },
        }


@dataclass(frozen=True)
class CleanupRuntimeContractResult:
    """Cleanup-owned Phase 5 runtime result/proof records.

    These records are intentionally pre-render and never consumed by the
    renderer in Phase 5.
    """

    page_id: str
    version: str
    runtime_class: CleanupClass
    status_records: list[dict[str, Any]] = field(default_factory=list)
    result_records: list[CleanupResult] = field(default_factory=list)
    proof_records: list[CleanupProof] = field(default_factory=list)
    parent_cleanup_unit_aggregate_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        passed = [
            proof
            for proof in self.proof_records
            if proof.proof_status == ProofStatus.PASSED
        ]
        pixel_changed_passed = [
            proof
            for proof in passed
            for result in self.result_records
            if result.cleanup_result_id == proof.cleanup_result_id and bool(result.pixel_changed)
        ]
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "runtime_class": _enum_value(self.runtime_class),
            "status_records": _json_safe(self.status_records),
            "results": [_json_safe(record) for record in self.result_records],
            "proofs": [_json_safe(record) for record in self.proof_records],
            "parent_cleanup_unit_aggregate_records": _json_safe(
                self.parent_cleanup_unit_aggregate_records
            ),
            "errors": list(self.errors),
            "summary": {
                "status_count": len(self.status_records),
                "result_count": len(self.result_records),
                "proof_count": len(self.proof_records),
                "parent_cleanup_unit_aggregate_count": len(
                    self.parent_cleanup_unit_aggregate_records
                ),
                "proof_passed_count": len(passed),
                "proof_passed_pixel_changed_count": len(pixel_changed_passed),
                "renderer_consumed": False,
            },
        }


@dataclass(frozen=True)
class CleanupUpstreamCommitResult:
    """Cleanup-owned page image committed before renderer invocation."""

    page_id: str
    version: str
    cleaned_image: Any = field(repr=False, compare=False)
    committed_image_ref: str = ""
    commit_diff_ref: str = ""
    commit_mask_ref: str = ""
    commit_records: list[dict[str, Any]] = field(default_factory=list)
    blocked_records: list[dict[str, Any]] = field(default_factory=list)
    root_transaction_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def committed_region_ids(self) -> list[str]:
        return sorted(
            {
                str(record.get("region_id") or "")
                for record in self.commit_records
                if record.get("cleanup_committed_to_working_image") and str(record.get("region_id") or "")
            }
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "cleanup_applied_upstream": bool(self.commit_records),
            "cleanup_committed_to_working_image": bool(self.commit_records),
            "cleanup_upstream_committed_region_ids": self.committed_region_ids,
            "cleanup_upstream_cleaned_image_ref": self.committed_image_ref,
            "cleanup_upstream_diff_ref": self.commit_diff_ref,
            "cleanup_upstream_mask_ref": self.commit_mask_ref,
            "commit_records": _json_safe(self.commit_records),
            "blocked_records": _json_safe(self.blocked_records),
            "root_transaction_records": _json_safe(self.root_transaction_records),
            "errors": list(self.errors),
            "summary": {
                "committed_count": len(self.commit_records),
                "blocked_count": len(self.blocked_records),
                "root_transaction_count": len(self.root_transaction_records),
                "root_committed_count": len(
                    [
                        record
                        for record in self.root_transaction_records
                        if record.get("root_transaction_status") == "root_committed"
                    ]
                ),
                "root_blocked_count": len(
                    [
                        record
                        for record in self.root_transaction_records
                        if record.get("root_transaction_status") == "root_blocked"
                    ]
                ),
                "error_count": len(self.errors),
                "renderer_consumed": False,
            },
        }


def build_cleanup_plans(
    *,
    page_id: str,
    job_candidates: Sequence[CleanupJob] | Any,
    mask_contracts: CleanupMaskBuildResult | Sequence[CleanupMask] | Any,
    image_size: tuple[int, int] | None = None,
    source_image_path: str | None = None,
    source_image: Any | None = None,
    render_eligibility: Any | None = None,
    inpaint_mode: str = "fast",
) -> CleanupPlanBuildResult:
    """Build cleanup-unit plans for accepted cleanup-owned masks."""

    plan_started = time.time()
    jobs = _extract_jobs(job_candidates)
    jobs_by_id = {str(job.cleanup_job_id): job for job in jobs}
    render_eligibility_by_region = _render_eligibility_by_region_id(render_eligibility)
    mask_result = _extract_masks(mask_contracts)
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    for cleanup_mask in mask_result:
        masks_by_job_id.setdefault(str(cleanup_mask.cleanup_job_id), []).append(cleanup_mask)
    _cleanup_perf_contract_checkpoint(
        "cleanup_plan_build",
        "module_start",
        page_id=page_id,
        job_count=len(jobs),
        mask_count=len(mask_result),
        source_image_path=source_image_path or "",
    )

    plans: list[CleanupPlan] = []
    plans_by_job_id: dict[str, list[CleanupPlan]] = {}
    plans_by_cleanup_mask_id: dict[str, list[CleanupPlan]] = {}
    rejected_records: list[dict[str, Any]] = []
    protected_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    errors: list[str] = []
    source_np, source_error = _source_image_array(source_image=source_image, source_image_path=source_image_path)
    backend_candidates = cleanup_backend_runner.inventory_local_cleanup_backends()
    backend_inventory = cleanup_backend_runner.inventory_to_audit_dict(backend_candidates)

    for cleanup_mask in mask_result:
        try:
            cleanup_mask_id = str(cleanup_mask.cleanup_mask_id)
            job = jobs_by_id.get(str(cleanup_mask.cleanup_job_id))
            if job is None:
                rejected_records.append(
                    {
                        "page_id": page_id,
                        "cleanup_job_id": str(cleanup_mask.cleanup_job_id),
                        "cleanup_mask_id": cleanup_mask_id,
                        "stage": "CleanupPlan",
                        "owner": "CleanupPlan accounting",
                        "reason": "cleanup_plan_rejected_missing_cleanup_job",
                    }
                )
                continue
            base = _job_base_record(page_id, job)
            base_with_mask = {
                **base,
                "cleanup_mask_id": cleanup_mask_id,
                "stage": "CleanupPlan",
                "owner": "CleanupPlan accounting",
            }
            suppressed_decision = _source_grounding_suppressed_decision(job, render_eligibility_by_region)
            blocked_reason = _job_exclusion_reason(job)
            if blocked_reason:
                protected_records.append({**base_with_mask, "reason": blocked_reason})
                continue
            mask_rejection = _mask_exclusion_reason(cleanup_mask, image_size)
            if not _accepted_executable_cleanup_mask(cleanup_mask):
                reason = mask_rejection or _cleanup_mask_non_executable_defect(cleanup_mask)
                _cleanup_perf_contract_checkpoint(
                    "cleanup_plan_mask_filter",
                    "rejected",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    cleanup_mask_id=cleanup_mask_id,
                    reason=reason,
                    visual_scope=str(cleanup_mask.visual_scope or ""),
                    foreground_pixels=cleanup_mask.foreground_mask_pixels,
                    erase_pixels=cleanup_mask.erase_mask_pixels,
                    allowed_area=cleanup_mask.allowed_area,
                    consumed_source_glyph_count=len(cleanup_mask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_count=len(cleanup_mask.missing_source_glyph_mask_ids or []),
                )
                rejected_records.append(
                    {
                        **base_with_mask,
                        "reason": reason,
                        "typed_reason": reason,
                    }
                )
                continue

            flat_rejection, flat_metrics = _flat_background_pre_execution_rejection(
                cleanup_mask=cleanup_mask,
                source_np=source_np,
                source_error=source_error,
            )
            cleanup_class = job.cleanup_class
            identity = _cleanup_identity_fields(job, cleanup_mask, fallback_region_id=cleanup_mask_id)
            plan = _runtime_speech_flat_plan(
                page_id=page_id,
                job=job,
                cleanup_mask=cleanup_mask,
                inpaint_mode=inpaint_mode,
                cleanup_class=cleanup_class,
            )
            params = dict(plan.backend_parameters or {})
            params.update(
                {
                    "text_block_root_id": str(
                        getattr(job, "text_block_root_id", "")
                        or identity.get("text_block_root_id")
                        or getattr(job, "parent_logical_text_unit_id", "")
                        or getattr(job, "cleanup_unit_id", "")
                        or getattr(job, "cleanup_job_id", "")
                        or ""
                    ),
                    "parent_execution_bundle_id": str(identity.get("parent_execution_bundle_id") or ""),
                    "parent_logical_text_unit_id": str(identity.get("parent_logical_text_unit_id") or ""),
                    "cleanup_obligation_id": str(cleanup_mask.cleanup_mask_id),
                    "mask_space": "page",
                    "bbox_format": "xyxy_exclusive",
                    "cleanup_plan_authority": "formal_cleanup_plan_build_result",
                    "formal_cleanup_plan": True,
                    "renderer_consumed_by_phase": "phase5_phase6_cleanup_contract_status_only",
                    "eligibility_reason": "cleanup_unit_segmentation_foreground_mask_contract",
                    "mask_contract_exception_reason": cleanup_mask.mask_contract_exception_reason,
                    "owned_segmentation_pixels": getattr(cleanup_mask, "owned_segmentation_pixels", None),
                    "executable_foreground_pixels": getattr(cleanup_mask, "executable_foreground_pixels", None),
                    "owned_segmentation_to_executable_ratio": getattr(
                        cleanup_mask,
                        "owned_segmentation_to_executable_ratio",
                        None,
                    ),
                    "ready_but_sparse_violation": bool(
                        getattr(cleanup_mask, "ready_but_sparse_violation", False)
                    ),
                    "sourceglyph_executable_influence_detected": bool(
                        getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)
                    ),
                    "segmentation_contract_override_detected": bool(
                        getattr(cleanup_mask, "segmentation_contract_override_detected", False)
                    ),
                    "component_projection_method": str(
                        getattr(cleanup_mask, "component_projection_method", "") or ""
                    ),
                    "pre_execution_risk_reason": mask_rejection or flat_rejection,
                    "partition_strategy_required": bool(mask_rejection),
                    "flatness_metrics": flat_metrics,
                    "backend_inventory": backend_inventory,
                    "render_eligibility_suppressed_source_ungrounded": bool(suppressed_decision is not None),
                    "render_eligibility_source_grounding_failure_reason": (
                        _decision_value(suppressed_decision, "reason")
                        if suppressed_decision is not None
                        else ""
                    ),
                }
            )
            plan = CleanupPlan(
                cleanup_plan_id=(
                    f"cplan_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}_{_safe_id(cleanup_mask.cleanup_mask_id)}_formal"
                ),
                cleanup_job_id=str(job.cleanup_job_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                cleanup_class=cleanup_class,
                selected_backend=plan.selected_backend,
                cleanup_method=plan.cleanup_method,
                parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
                parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
                text_block_root_id=str(
                    getattr(job, "text_block_root_id", "")
                    or identity.get("text_block_root_id")
                    or getattr(job, "parent_logical_text_unit_id", "")
                    or ""
                ),
                target_region_ids=list(identity.get("target_region_ids") or []),
                backend_parameters=params,
                inpaint_mode=plan.inpaint_mode,
                crop_context_bbox=list(cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area or []),
                fallback_policy=["runtime_adaptive_cleanup_contract"],
                expected_runtime_class="formal_cleanup_plan",
                proof_thresholds=dict(plan.proof_thresholds or {}),
            )
            plans.append(plan)
            plans_by_job_id.setdefault(str(job.cleanup_job_id), []).append(plan)
            plans_by_cleanup_mask_id.setdefault(str(cleanup_mask.cleanup_mask_id), []).append(plan)
            _cleanup_perf_contract_checkpoint(
                "cleanup_plan_build",
                "plan_created",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                cleanup_plan_id=str(plan.cleanup_plan_id),
                selected_backend=str(plan.selected_backend),
                visual_scope=str(cleanup_mask.visual_scope or ""),
                expected_runtime_class=plan.expected_runtime_class,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    _cleanup_perf_contract_checkpoint(
        "cleanup_plan_build",
        "module_end",
        page_id=page_id,
        plan_count=len(plans),
        rejected_count=len(rejected_records),
        protected_count=len(protected_records),
        skipped_count=len(skipped_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - plan_started) * 1000.0, 3),
    )
    return CleanupPlanBuildResult(
        page_id=page_id,
        version=CLEANUP_PLAN_CONTRACT_VERSION,
        plans=plans,
        plans_by_job_id=plans_by_job_id,
        plans_by_cleanup_mask_id=plans_by_cleanup_mask_id,
        rejected_records=rejected_records,
        protected_records=protected_records,
        skipped_records=skipped_records,
        errors=errors,
        backend_inventory=backend_inventory,
    )


def prove_cleanup_result(
    *,
    source_image: Any,
    before_image: Any,
    cleanup_result: CleanupResult,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    proof_context: Mapping[str, Any] | None = None,
) -> CleanupProof:
    """Proof one cleanup result before translated text drawing."""

    metrics: dict[str, Any] = {}
    failure_reason = "passed"
    proof_status = ProofStatus.PASSED
    if (
        str(cleanup_result.execution_status or "") == "contract_error"
        or str(cleanup_result.failure_reason or "")
        in {
            "model_required_backend_not_invoked",
            "model_required_backend_failed",
        }
    ):
        return _proof(
            cleanup_result,
            cleanup_plan,
            status=ProofStatus.FAILED,
            failure_reason=cleanup_result.failure_reason or "cleanup_backend_contract_error",
            metrics={
                "backend_contract_error": True,
                "model_required": bool(
                    (cleanup_result.backend_parameters or {}).get("model_required")
                ),
                "backend_kind": str(cleanup_result.backend_kind or ""),
                "backend_method": str(cleanup_result.backend_method or ""),
                "model_invocation_attempted": bool(cleanup_result.model_invocation_attempted),
                "model_invocation_succeeded": bool(cleanup_result.model_invocation_succeeded),
                "proof_scope": "cleanup_mask_foreground",
                "sourceglyph_bbox_residual_failure_impossible": True,
            },
        )
    try:
        if np is None:
            raise RuntimeError("numpy_unavailable")
        source_np = (
            proof_context.get("source_np")
            if isinstance(proof_context, Mapping)
            else None
        )
        source_gray = (
            proof_context.get("source_gray")
            if isinstance(proof_context, Mapping)
            else None
        )
        source_hash = (
            str(proof_context.get("source_image_hash") or "")
            if isinstance(proof_context, Mapping)
            else ""
        )
        if source_np is None:
            source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        if source_gray is None:
            source_gray = _gray(source_np)
        if source_hash and str(cleanup_result.input_image_hash or "") == source_hash:
            before_np = source_np
            before_gray = source_gray
        else:
            before_np = np.asarray(before_image.convert("RGB") if hasattr(before_image, "convert") else before_image)
            before_gray = _gray(before_np)
        page_shape = source_np.shape[:2]
        proof_denominator_area = int(page_shape[0] * page_shape[1])
        crop_bbox = _cleanup_result_crop_bbox(cleanup_result)
        cleaned_crop = _cleaned_crop_payload_for_result(cleanup_result)
        crop_local = (
            str(getattr(cleanup_result, "result_space", "") or "") == "crop"
            and cleaned_crop is not None
            and _crop_shape_matches(cleaned_crop, crop_bbox)
        )
        if crop_local:
            x0, y0, x1, y1 = crop_bbox or [0, 0, 0, 0]
            cleaned_np = np.asarray(
                cleaned_crop.convert("RGB") if hasattr(cleaned_crop, "convert") else cleaned_crop
            )
            source_np = source_np[y0:y1, x0:x1]
            source_gray = source_gray[y0:y1, x0:x1]
            before_np = before_np[y0:y1, x0:x1]
            before_gray = before_gray[y0:y1, x0:x1]
            mask_shape = source_np.shape[:2]
        else:
            cleaned_image = cleanup_result.cleaned_image or _load_cleanup_image(cleanup_result.cleaned_image_ref)
            cleaned_np = np.asarray(cleaned_image.convert("RGB") if hasattr(cleaned_image, "convert") else cleaned_image)
            mask_shape = source_np.shape[:2]
        foreground = _binary_mask(getattr(cleanup_result, "commit_foreground_mask", None), page_shape)
        if foreground is None or not np.any(foreground):
            foreground = _binary_mask(cleanup_mask.foreground_mask, page_shape)
        erase = _binary_mask(getattr(cleanup_result, "commit_mask", None), page_shape)
        if erase is None or not np.any(erase):
            erase = _binary_mask(cleanup_mask.erase_mask, page_shape)
        allowed = _bbox_to_mask(cleanup_mask.allowed_area, page_shape)
        if crop_local and crop_bbox is not None:
            x0, y0, x1, y1 = crop_bbox
            if foreground is not None:
                foreground = foreground[y0:y1, x0:x1]
            if erase is not None:
                erase = erase[y0:y1, x0:x1]
            if allowed is not None:
                allowed = allowed[y0:y1, x0:x1]
        if foreground is None or not np.any(foreground):
            return _proof(
                cleanup_result,
                cleanup_plan,
                status=ProofStatus.INCONCLUSIVE,
                failure_reason="inconclusive_missing_evidence",
                metrics={"missing": "source_foreground_evidence"},
            )
        if erase is None or not np.any(erase):
            return _proof(
                cleanup_result,
                cleanup_plan,
                status=ProofStatus.INCONCLUSIVE,
                failure_reason="inconclusive_missing_evidence",
                metrics={"missing": "erase_mask"},
            )
        allowed_present = allowed is not None and np.any(allowed > 0)
        if allowed is None:
            allowed = np.zeros(mask_shape, dtype=np.uint8)

        foreground_pixels = int(np.count_nonzero(foreground))
        erase_foreground_overlap = int(np.count_nonzero((erase > 0) & (foreground > 0)))
        mask_source_overlap_ratio = erase_foreground_overlap / max(1, foreground_pixels)

        cleaned_gray = _gray(cleaned_np)
        diff_gray = np.abs(cleaned_gray.astype(np.int16) - before_gray.astype(np.int16))
        changed = diff_gray > 8
        legacy_residual = _source_residual_mask(source_gray, cleaned_gray, foreground)
        legacy_residual_pixels = int(np.count_nonzero(legacy_residual))
        legacy_residual_ratio = legacy_residual_pixels / max(1, foreground_pixels)
        source_dark_foreground = (foreground > 0) & (source_gray < 210)
        source_light_foreground = (foreground > 0) & ~(source_gray < 210)
        source_dark_foreground_pixels = int(np.count_nonzero(source_dark_foreground))
        source_light_foreground_pixels = int(np.count_nonzero(source_light_foreground))
        cleanup_class_value = _enum_value(cleanup_plan.cleanup_class)
        baseline_residual_dark = _dark_source_residual_mask(source_gray, before_gray, foreground)
        baseline_residual_dark_pixels = int(np.count_nonzero(baseline_residual_dark))
        baseline_residual_dark_ratio = baseline_residual_dark_pixels / max(1, source_dark_foreground_pixels)
        residual_dark = _dark_source_residual_mask(source_gray, cleaned_gray, foreground)
        residual_dark_pixels = int(np.count_nonzero(residual_dark))
        residual_dark_ratio = residual_dark_pixels / max(1, source_dark_foreground_pixels)
        light_context = _source_light_foreground_context(
            source_gray,
            foreground,
        )
        light_residual_scope_active = bool(light_context.get("dark_context", False))
        residual_light_unchanged_pixels = int(
            np.count_nonzero(legacy_residual & source_light_foreground)
        )
        residual_light_unchanged_ratio = (
            residual_light_unchanged_pixels / max(1, source_light_foreground_pixels)
        )
        params = cleanup_plan.backend_parameters or {}
        proof_scope = _proof_scope_metadata(cleanup_plan, cleanup_mask)
        erasure_box = proof_scope.get("proof_residual_measurement_bbox")
        if crop_local:
            erasure_box = _page_bbox_to_crop_bbox(erasure_box, crop_bbox)
        visual_residual: Any | None = None
        visual_residual_dark_pixels = 0
        visual_residual_pixels = 0
        visual_residual_ratio = 0.0
        visual_residual_baseline_pixels = 0
        visual_residual_baseline_ratio = 0.0
        if erasure_box is not None:
            visual_residual_baseline = _source_erasure_box_residual_mask(
                source_gray,
                before_gray,
                erasure_box,
            )
            visual_residual = _source_erasure_box_residual_mask(
                source_gray,
                cleaned_gray,
                erasure_box,
            )
            visual_erasure_region = _bbox_to_mask(erasure_box, source_np.shape[:2])
            visual_source_dark = (visual_erasure_region > 0) & (source_gray < 135)
            visual_residual_dark_pixels = int(np.count_nonzero(visual_source_dark))
            visual_residual_baseline_pixels = int(np.count_nonzero(visual_residual_baseline))
            visual_residual_pixels = int(np.count_nonzero(visual_residual))
            if visual_residual_dark_pixels >= 20:
                visual_residual_baseline_ratio = (
                    visual_residual_baseline_pixels / max(1, visual_residual_dark_pixels)
                )
                visual_residual_ratio = visual_residual_pixels / max(1, visual_residual_dark_pixels)

        outside_allowed = changed & ~(allowed > 0)
        changed_outside_allowed_pixels = int(np.count_nonzero(outside_allowed))
        changed_outside_allowed_ratio = changed_outside_allowed_pixels / max(1, proof_denominator_area)
        outside_accepted_erase = changed & ~(erase > 0)
        changed_outside_accepted_mask_pixels = int(np.count_nonzero(outside_accepted_erase))
        changed_outside_accepted_mask_ratio = changed_outside_accepted_mask_pixels / max(1, proof_denominator_area)
        accepted_changed = changed & (erase > 0)
        outside_erase_inside_allowed = changed & (allowed > 0) & ~(erase > 0)
        outside_erase_inside_allowed_pixels = int(np.count_nonzero(outside_erase_inside_allowed))
        allowed_changed = changed & (allowed > 0)
        outside_erase_inside_allowed_ratio = outside_erase_inside_allowed_pixels / max(1, int(np.count_nonzero(allowed_changed)))
        collateral_ratio = changed_outside_accepted_mask_pixels / max(1, int(np.count_nonzero(changed)))
        changed_inside_erase = accepted_changed
        changed_inside_erase_pixels = int(np.count_nonzero(changed_inside_erase))
        white_patch_candidate = changed_inside_erase & (cleaned_gray > 235) & (source_gray < 180)
        white_patch_pixels = int(np.count_nonzero(white_patch_candidate))
        white_patch_ratio = white_patch_pixels / max(1, changed_inside_erase_pixels)
        white_patch_context = _white_patch_context_metrics(
            source_gray=source_gray,
            white_patch_mask=white_patch_candidate,
            erase_mask=erase,
        )
        white_patch_light_context = bool(white_patch_context.get("light_context", False))
        white_patch_cleaned_median_luma: float | None = None
        white_patch_matches_light_context = False
        if white_patch_pixels > 0 and white_patch_light_context:
            try:
                white_patch_cleaned_median_luma = float(np.median(cleaned_gray[white_patch_candidate]))
                context_median = _float_or_none(white_patch_context.get("median_luma"))
                white_patch_matches_light_context = (
                    context_median is not None
                    and abs(white_patch_cleaned_median_luma - context_median) <= 6.0
                )
            except Exception:
                white_patch_cleaned_median_luma = None
                white_patch_matches_light_context = False
        white_patch_light_context_exempt = (
            white_patch_light_context
            and (
                cleanup_class_value
                in {
                    CleanupClass.CAPTION_FLAT_BACKGROUND.value,
                }
                or white_patch_matches_light_context
            )
        )
        dark_caption_white_patch_risk = (
            cleanup_class_value
            in {
                CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
                CleanupClass.BACKGROUND_ART_TEXT.value,
                CleanupClass.TITLE_OR_SIGN.value,
                CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
            }
            and white_patch_pixels > max(96, int(foreground_pixels * 0.45))
            and not white_patch_light_context_exempt
        )
        broad_white_patch_risk = (
            dark_caption_white_patch_risk
            or (
                cleanup_class_value
                not in {
                    CleanupClass.SPEECH_FLAT_BUBBLE.value,
                    CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
                }
                and white_patch_pixels > max(96, int(foreground_pixels * 0.45))
                and white_patch_ratio > 0.70
                and not white_patch_light_context_exempt
            )
        )
        residual_pixel_limit = max(24, int(source_dark_foreground_pixels * PROOF_RESIDUAL_PIXEL_FRACTION))
        residual_light_pixel_limit = max(
            4,
            int(source_light_foreground_pixels * PROOF_RESIDUAL_PIXEL_FRACTION),
        )
        source_light_foreground_ratio = source_light_foreground_pixels / max(1, foreground_pixels)
        light_context_median_luma = _float_or_none(light_context.get("median_luma"))
        light_context_dark_ratio = _float_or_none(light_context.get("dark_ratio")) or 0.0
        light_context_is_light = (
            light_context_median_luma is not None
            and light_context_median_luma >= 225.0
            and light_context_dark_ratio <= 0.15
        )
        source_light_text_on_dark_context = (
            light_residual_scope_active
            and source_light_foreground_ratio > 0.25
        )
        light_residual_strict_cleanup_class = cleanup_class_value in {
            CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
            CleanupClass.BACKGROUND_ART_TEXT.value,
            CleanupClass.TITLE_OR_SIGN.value,
            CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        }
        light_residual_scope_active = bool(
            light_residual_scope_active
            or (
                light_residual_strict_cleanup_class
                and not light_context_is_light
                and source_dark_foreground_pixels > 0
                and source_light_foreground_pixels
                >= max(16, int(foreground_pixels * 0.08))
                and residual_light_unchanged_pixels
                >= max(8, int(source_light_foreground_pixels * PROOF_RESIDUAL_PIXEL_FRACTION))
            )
        )
        source_light_text_on_dark_context = (
            light_residual_scope_active
            and source_light_foreground_ratio > 0.25
        )
        outside_boundary_residual = _outside_accepted_mask_boundary_residual_mask(
            source_gray=source_gray,
            cleaned_gray=cleaned_gray,
            foreground=foreground,
            erase=erase,
            allowed=allowed,
            source_light_foreground_ratio=source_light_foreground_ratio,
        )
        outside_boundary_residual_pixels = int(np.count_nonzero(outside_boundary_residual))
        outside_boundary_residual_ratio = outside_boundary_residual_pixels / max(1, foreground_pixels)
        outside_boundary_residual_pixel_limit = max(16, int(foreground_pixels * 0.015))
        outside_boundary_residual_remaining = (
            not source_light_text_on_dark_context
            and outside_boundary_residual_pixels >= outside_boundary_residual_pixel_limit
            and outside_boundary_residual_ratio >= 0.015
        )
        dark_source_residual_remaining = (
            False
            if source_light_text_on_dark_context
            else not (
                residual_dark_ratio < PROOF_RESIDUAL_RATIO
                or residual_dark_pixels < residual_pixel_limit
            )
        )
        light_context_dark_shadow_residual = np.zeros_like(foreground, dtype=bool)
        light_context_dark_shadow_pixels = 0
        light_context_dark_shadow_ratio = 0.0
        light_context_dark_shadow_remaining = False
        if (
            light_residual_strict_cleanup_class
            and light_context_is_light
            and light_context_median_luma is not None
            and source_dark_foreground_pixels > 0
        ):
            # A model may restore screentone or panel lines inside the original
            # glyph footprint. Treat only source-like dark pixels as proof
            # residual; absolute darkness in a light neighborhood is audit
            # evidence, not source text by itself.
            light_context_dark_shadow_residual = (
                residual_dark
                & (cleaned_gray < max(0.0, light_context_median_luma - 10.0))
            )
            light_context_dark_shadow_pixels = int(
                np.count_nonzero(light_context_dark_shadow_residual)
            )
            light_context_dark_shadow_ratio = (
                light_context_dark_shadow_pixels / max(1, source_dark_foreground_pixels)
            )
            light_context_dark_shadow_remaining = not (
                light_context_dark_shadow_ratio < PROOF_RESIDUAL_RATIO
                or light_context_dark_shadow_pixels < residual_pixel_limit
            )
        if light_context_dark_shadow_remaining and not broad_white_patch_risk:
            dark_source_residual_remaining = True
        light_source_residual_remaining = (
            light_residual_scope_active
            and source_light_foreground_pixels > 0
            and not (
                residual_light_unchanged_ratio < PROOF_RESIDUAL_RATIO
                or residual_light_unchanged_pixels < residual_light_pixel_limit
            )
        )
        cleaned_light_foreground = (
            (foreground > 0)
            & (source_light_foreground | light_residual_scope_active)
            & (cleaned_gray > 190)
        )
        cleaned_light_foreground_pixels = int(np.count_nonzero(cleaned_light_foreground))
        cleaned_light_foreground_ratio = (
            cleaned_light_foreground_pixels / max(1, source_light_foreground_pixels)
        )
        cleaned_light_source_residual_remaining = (
            light_residual_scope_active
            and source_light_foreground_pixels > 0
            and not (
                cleaned_light_foreground_ratio < PROOF_RESIDUAL_RATIO
                or cleaned_light_foreground_pixels < residual_light_pixel_limit
            )
        )
        preliminary_executable_residual_mask = residual_dark | light_context_dark_shadow_residual
        if light_source_residual_remaining or cleaned_light_source_residual_remaining:
            preliminary_executable_residual_mask = residual_dark | (legacy_residual & source_light_foreground)
            if cleaned_light_source_residual_remaining:
                preliminary_executable_residual_mask = (
                    preliminary_executable_residual_mask | cleaned_light_foreground
                )
        residual_component_metrics = _mask_connected_component_metrics(
            preliminary_executable_residual_mask
        )
        residual_largest_component_ratio = _float_or_none(
            residual_component_metrics.get("largest_component_ratio")
        ) or 0.0
        residual_component_count = int(
            _int_or_none(residual_component_metrics.get("component_count")) or 0
        )
        residual_fragmented_texture_like = (
            residual_component_count >= 3
            and residual_largest_component_ratio <= 0.25
        )
        model_backend_executed = (
            str(cleanup_result.backend_kind or "") == "model_inpaint"
            and bool(cleanup_result.model_invocation_attempted)
            and bool(cleanup_result.model_invocation_succeeded)
        )
        model_clearance_cleanup_class = cleanup_class_value in {
            CleanupClass.CAPTION_FLAT_BACKGROUND.value,
            CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
            CleanupClass.BACKGROUND_ART_TEXT.value,
            CleanupClass.TITLE_OR_SIGN.value,
            CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        }
        changed_foreground_pixels = int(np.count_nonzero(changed & (foreground > 0)))
        changed_foreground_ratio = changed_foreground_pixels / max(1, foreground_pixels)
        residual_dark_reduction_ratio = (
            (baseline_residual_dark_pixels - residual_dark_pixels)
            / max(1, baseline_residual_dark_pixels)
        )
        actual_model_name = str(cleanup_result.actual_model_name or "")
        actual_model_path = str(cleanup_result.actual_model_path or "").replace("\\", "/")
        backend_method = str(cleanup_result.backend_method or "")
        fixed_cleanup_model_backend = (
            backend_method == "cleanup_owned_simple_lama_inpaint"
            and (
                "iopaint/anime-manga-big-lama" in actual_model_name
                or "iopaint/anime-manga-big-lama" in actual_model_path
            )
        )
        model_containment_safe = (
            changed_outside_accepted_mask_pixels <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            and changed_outside_accepted_mask_ratio <= PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
            and collateral_ratio <= PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
        )
        model_inpaint_source_text_clearance_base = (
            model_backend_executed
            and model_clearance_cleanup_class
            and bool(cleanup_result.pixel_changed)
            and changed_foreground_pixels >= max(24, int(foreground_pixels * 0.30))
            and changed_foreground_ratio >= 0.30
            and residual_dark_reduction_ratio >= 0.55
            and residual_dark_ratio <= 0.42
            and model_containment_safe
        )
        model_white_patch_non_destructive = (
            not broad_white_patch_risk
            or (
                fixed_cleanup_model_backend
                and model_backend_executed
                and model_clearance_cleanup_class
                and model_containment_safe
                and white_patch_ratio <= 0.25
                and white_patch_pixels <= max(1024, int(foreground_pixels * 0.35))
            )
        )
        model_low_absolute_residual_acceptance = (
            model_inpaint_source_text_clearance_base
            and fixed_cleanup_model_backend
            and model_white_patch_non_destructive
            and changed_foreground_ratio >= 0.70
            and residual_dark_reduction_ratio >= 0.75
            and residual_dark_ratio <= 0.20
            and residual_dark_pixels <= max(128, int(source_dark_foreground_pixels * 0.20))
        )
        model_inpaint_source_text_clearance_candidate = (
            model_inpaint_source_text_clearance_base
            and fixed_cleanup_model_backend
            and (
                residual_fragmented_texture_like
                or model_low_absolute_residual_acceptance
            )
            and model_white_patch_non_destructive
        )
        model_restored_background_texture_acceptance = (
            # Policy note: this is not semantic authorization and does not
            # broaden cleanup pixels. It allows fixed-model cleanup to pass
            # when remaining foreground residual has been reduced to bounded
            # fragmented texture inside the already accepted mask.
            model_inpaint_source_text_clearance_base
            and fixed_cleanup_model_backend
            and model_white_patch_non_destructive
            and not light_source_residual_remaining
            and not cleaned_light_source_residual_remaining
            and residual_dark_reduction_ratio >= 0.60
            and changed_foreground_ratio >= 0.70
            and (
                (
                    residual_component_count >= 3
                    and residual_largest_component_ratio <= 0.35
                )
                or (
                    residual_dark_pixels <= max(
                        160,
                        int(source_dark_foreground_pixels * 0.25),
                    )
                    and residual_dark_reduction_ratio >= 0.75
                )
            )
        )
        model_light_text_dark_context_acceptance = (
            fixed_cleanup_model_backend
            and source_light_text_on_dark_context
            and model_backend_executed
            and model_clearance_cleanup_class
            and bool(cleanup_result.pixel_changed)
            and changed_foreground_ratio >= 0.70
            and residual_dark_reduction_ratio >= 0.55
            and residual_dark_ratio <= 0.42
            and model_containment_safe
            and not light_source_residual_remaining
            and not cleaned_light_source_residual_remaining
        )
        model_fragmented_texture_acceptance = bool(
            model_inpaint_source_text_clearance_candidate
            and not light_source_residual_remaining
            and not cleaned_light_source_residual_remaining
        )
        model_light_text_white_patch_acceptance = (
            (
                model_light_text_dark_context_acceptance
                or model_fragmented_texture_acceptance
            )
            and broad_white_patch_risk
            and residual_dark_reduction_ratio >= 0.80
            and white_patch_ratio <= 0.25
            and model_white_patch_non_destructive
        )
        executable_residual_mask = preliminary_executable_residual_mask
        executable_residual_pixels = int(np.count_nonzero(executable_residual_mask))
        executable_residual_ratio = executable_residual_pixels / max(1, foreground_pixels)
        residual_mask_ref, residual_mask_missing_reason = _write_residual_source_mask(
            cleanup_result=cleanup_result,
            residual_mask=executable_residual_mask,
        )

        backend_name = str(cleanup_result.backend_name or "")
        candidate_status = str((cleanup_result.backend_parameters or {}).get("candidate_status") or "")
        backend_noop = (
            bool(cleanup_result.errors)
            or backend_name in {"", "none", "error"}
            or candidate_status not in {"completed"}
        )
        model_inpaint_acceptable_residual_commit = False
        metrics = {
            "source_foreground_pixels": foreground_pixels,
            "erase_foreground_overlap_pixels": erase_foreground_overlap,
            "mask_source_overlap_ratio": round(mask_source_overlap_ratio, 4),
            "residual_pass_metric": "dark_source_foreground",
            "source_dark_foreground_pixels": source_dark_foreground_pixels,
            "source_light_foreground_pixels": source_light_foreground_pixels,
            "source_light_foreground_ratio": round(source_light_foreground_ratio, 4),
            "source_light_text_on_dark_context": source_light_text_on_dark_context,
            "source_residual_baseline_dark_pixels": baseline_residual_dark_pixels,
            "source_residual_baseline_dark_ratio": round(baseline_residual_dark_ratio, 4),
            "residual_dark_source_pixels": residual_dark_pixels,
            "residual_dark_source_ratio": round(residual_dark_ratio, 4),
            "executable_text_residual_pixels": executable_residual_pixels,
            "executable_text_residual_ratio": round(executable_residual_ratio, 4),
            "executable_text_residual_pixel_limit": residual_pixel_limit,
            "executable_text_residual_component_count": residual_component_count,
            "executable_text_residual_largest_component_pixels": residual_component_metrics.get(
                "largest_component_pixels",
                0,
            ),
            "executable_text_residual_largest_component_ratio": round(
                residual_largest_component_ratio,
                4,
            ),
            "model_inpaint_residual_fragmented_texture_like": bool(
                residual_fragmented_texture_like
            ),
            "model_inpaint_low_absolute_residual_acceptance": bool(
                model_low_absolute_residual_acceptance
            ),
            "proof_residual_scope": "cleanup_mask_foreground",
            "source_residual_delta_dark_pixels": int(baseline_residual_dark_pixels - residual_dark_pixels),
            "source_residual_delta_dark_ratio": round(
                baseline_residual_dark_ratio - residual_dark_ratio,
                4,
            ),
            "model_backend_executed": bool(model_backend_executed),
            "model_clearance_cleanup_class": bool(model_clearance_cleanup_class),
            "model_inpaint_source_text_clearance_candidate": bool(
                model_inpaint_source_text_clearance_candidate
            ),
            "model_inpaint_source_text_clearance_base": bool(
                model_inpaint_source_text_clearance_base
            ),
            "model_inpaint_white_patch_non_destructive": bool(
                model_white_patch_non_destructive
            ),
            "model_inpaint_acceptable_residual_commit": bool(
                model_inpaint_acceptable_residual_commit
            ),
            "model_inpaint_source_text_cleared": bool(
                not dark_source_residual_remaining
                and not light_context_dark_shadow_remaining
                and not light_source_residual_remaining
                and not cleaned_light_source_residual_remaining
            ),
            "model_inpaint_fixed_cleanup_backend": bool(fixed_cleanup_model_backend),
            "model_inpaint_clearance_override_disabled": not bool(
                model_inpaint_acceptable_residual_commit
            ),
            "model_inpaint_clearance_acceptance_policy": "",
            "model_inpaint_light_text_dark_context_acceptance": bool(
                model_light_text_dark_context_acceptance
            ),
            "model_inpaint_light_text_white_patch_acceptance": bool(
                model_light_text_white_patch_acceptance
            ),
            "model_inpaint_fragmented_texture_acceptance": bool(
                model_fragmented_texture_acceptance
            ),
            "model_inpaint_restored_background_texture_acceptance": bool(
                model_restored_background_texture_acceptance
            ),
            "model_changed_foreground_pixels": changed_foreground_pixels,
            "model_changed_foreground_ratio": round(changed_foreground_ratio, 4),
            "model_residual_dark_reduction_ratio": round(residual_dark_reduction_ratio, 4),
            "legacy_residual_source_pixels": legacy_residual_pixels,
            "legacy_residual_source_ratio": round(legacy_residual_ratio, 4),
            "residual_source_pixels": legacy_residual_pixels,
            "residual_source_ratio": round(legacy_residual_ratio, 4),
            "residual_light_unchanged_foreground_pixels": residual_light_unchanged_pixels,
            "residual_light_unchanged_foreground_ratio": round(residual_light_unchanged_ratio, 4),
            "residual_light_unchanged_pixel_limit": residual_light_pixel_limit,
            "residual_light_context_scope_active": light_residual_scope_active,
            "residual_light_context_median_luma": light_context.get("median_luma"),
            "residual_light_context_dark_ratio": light_context.get("dark_ratio"),
            "residual_light_context_light_like": light_context_is_light,
            "dark_source_residual_remaining": dark_source_residual_remaining,
            "light_context_dark_shadow_residual_pixels": light_context_dark_shadow_pixels,
            "light_context_dark_shadow_residual_ratio": round(
                light_context_dark_shadow_ratio,
                4,
            ),
            "light_context_dark_shadow_residual_remaining": light_context_dark_shadow_remaining,
            "light_source_residual_remaining": light_source_residual_remaining,
            "cleaned_light_foreground_pixels": cleaned_light_foreground_pixels,
            "cleaned_light_foreground_ratio": round(cleaned_light_foreground_ratio, 4),
            "cleaned_light_source_residual_remaining": cleaned_light_source_residual_remaining,
            "outside_accepted_mask_boundary_residual_pixels": outside_boundary_residual_pixels,
            "outside_accepted_mask_boundary_residual_ratio": round(outside_boundary_residual_ratio, 4),
            "outside_accepted_mask_boundary_residual_pixel_limit": outside_boundary_residual_pixel_limit,
            "outside_accepted_mask_boundary_residual_remaining": outside_boundary_residual_remaining,
            "cleanup_result_space": "crop" if crop_local else "page",
            "cleanup_result_crop_bbox": list(crop_bbox or []),
            "proof_scope": proof_scope.get("proof_scope"),
            "proof_scope_bbox": proof_scope.get("proof_scope_bbox"),
            "proof_scope_uses_parent_expected_bbox": proof_scope.get(
                "proof_scope_uses_parent_expected_bbox"
            ),
            "proof_scope_excludes_parent_expected_bbox": proof_scope.get(
                "proof_scope_excludes_parent_expected_bbox"
            ),
            "parent_cleanup_unit_id": proof_scope.get("parent_cleanup_unit_id"),
            "parent_expected_erasure_bbox": proof_scope.get("parent_expected_erasure_bbox"),
            "source_glyph_erasure_bbox_input": proof_scope.get("source_glyph_erasure_bbox_input"),
            "source_glyph_erasure_expected_area_bbox": proof_scope.get(
                "source_glyph_erasure_expected_area_bbox"
            ),
            "source_glyph_erasure_bbox": erasure_box,
            "source_glyph_erasure_visual_dark_pixels": visual_residual_dark_pixels,
            "source_glyph_erasure_visual_residual_baseline_pixels": visual_residual_baseline_pixels,
            "source_glyph_erasure_visual_residual_baseline_ratio": round(
                visual_residual_baseline_ratio,
                4,
            ),
            "source_glyph_erasure_visual_residual_pixels": visual_residual_pixels,
            "source_glyph_erasure_visual_residual_ratio": round(visual_residual_ratio, 4),
            "audit_bbox_residual_scope": "source_glyph_erasure_bbox",
            "audit_bbox_residual_pixels": visual_residual_pixels,
            "audit_bbox_residual_ratio": round(visual_residual_ratio, 4),
            "source_glyph_erasure_visual_residual_delta_pixels": int(
                visual_residual_baseline_pixels - visual_residual_pixels
            ),
            "source_glyph_erasure_visual_residual_delta_ratio": round(
                visual_residual_baseline_ratio - visual_residual_ratio,
                4,
            ),
            "residual_source_pixel_limit": residual_pixel_limit,
            "allowed_area_present": bool(allowed_present),
            "allowed_area_is_audit_only": True,
            "changed_outside_allowed_pixels": changed_outside_allowed_pixels,
            "changed_outside_allowed_ratio": round(changed_outside_allowed_ratio, 6),
            "changed_outside_accepted_mask_pixels": changed_outside_accepted_mask_pixels,
            "changed_outside_accepted_mask_ratio": round(changed_outside_accepted_mask_ratio, 6),
            "changed_outside_erase_inside_allowed_pixels": outside_erase_inside_allowed_pixels,
            "changed_outside_erase_inside_allowed_ratio": round(outside_erase_inside_allowed_ratio, 4),
            "accepted_mask_collateral_ratio": round(collateral_ratio, 4),
            "changed_inside_erase_pixels": changed_inside_erase_pixels,
            "broad_white_patch_pixels": white_patch_pixels,
            "broad_white_patch_ratio": round(white_patch_ratio, 4),
            "broad_white_patch_risk": bool(broad_white_patch_risk),
            "broad_white_patch_light_context": white_patch_light_context,
            "broad_white_patch_light_context_exempt": white_patch_light_context_exempt,
            "broad_white_patch_context_median_luma": white_patch_context.get("median_luma", 0.0),
            "broad_white_patch_context_light_ratio": white_patch_context.get("light_ratio", 0.0),
            "broad_white_patch_context_dark_ratio": white_patch_context.get("dark_ratio", 0.0),
            "broad_white_patch_cleaned_median_luma": white_patch_cleaned_median_luma,
            "broad_white_patch_matches_light_context": white_patch_matches_light_context,
            "collateral_art_damage_delta": round(collateral_ratio, 4),
            "backend_name": backend_name,
            "backend_candidate_id": (cleanup_result.backend_parameters or {}).get("backend_candidate_id"),
            "candidate_status": candidate_status,
            "pixel_changed": bool(cleanup_result.pixel_changed),
            "changed_pixel_count": int(cleanup_result.changed_pixel_count or 0),
            "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
            "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
            "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
            "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
            "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
            "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        }
        if residual_mask_ref:
            metrics["residual_source_mask_ref"] = residual_mask_ref
        else:
            metrics["residual_source_mask_ref_missing_reason"] = residual_mask_missing_reason
        if backend_noop:
            failure_reason = "backend_noop_or_error"
        elif mask_source_overlap_ratio < PROOF_MASK_SOURCE_OVERLAP_RATIO:
            failure_reason = "mask_missing_source_pixels"
        elif source_dark_foreground_pixels <= 0 and not (
            light_residual_scope_active and source_light_foreground_pixels > 0
        ):
            failure_reason = "inconclusive_missing_dark_source_foreground"
        elif (
            dark_source_residual_remaining
            or light_source_residual_remaining
            or cleaned_light_source_residual_remaining
        ) and not (
            model_inpaint_acceptable_residual_commit
            or model_restored_background_texture_acceptance
        ):
            failure_reason = "source_residual_remaining"
        elif (
            changed_outside_accepted_mask_pixels > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            or changed_outside_accepted_mask_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        ):
            failure_reason = "changed_outside_accepted_mask"
        elif collateral_ratio > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO:
            failure_reason = "collateral_change_too_broad"
        elif broad_white_patch_risk:
            failure_reason = "broad_white_patch_risk"
        else:
            failure_reason = "passed"

        if failure_reason == "inconclusive_missing_dark_source_foreground":
            proof_status = ProofStatus.INCONCLUSIVE
        elif failure_reason != "passed":
            proof_status = ProofStatus.FAILED

        source_residual_improved = (
            residual_dark_pixels < baseline_residual_dark_pixels
        )
        source_visible_area_increased = (
            residual_dark_pixels > baseline_residual_dark_pixels + 4
        )
        allowed_area_safe = (
            changed_outside_accepted_mask_pixels <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            and changed_outside_accepted_mask_ratio <= PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        )
        collateral_safe = collateral_ratio <= PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
        degraded_cleanup_diagnostic = (
            proof_status == ProofStatus.FAILED
            and bool(cleanup_result.pixel_changed)
            and str(failure_reason or "").startswith("source_residual")
            and source_residual_improved
            and not source_visible_area_increased
            and allowed_area_safe
            and collateral_safe
            and not broad_white_patch_risk
            and mask_source_overlap_ratio >= PROOF_MASK_SOURCE_OVERLAP_RATIO
        )
        if failure_reason == "passed" and model_restored_background_texture_acceptance:
            proof_quality_state = "proof_passed_model_inpaint_restored_background_texture"
        elif failure_reason == "passed" and model_inpaint_acceptable_residual_commit:
            proof_quality_state = "proof_passed_model_inpaint_residual_accepted"
        elif failure_reason == "passed":
            proof_quality_state = "proof_passed_clean"
        elif str(failure_reason or "").startswith("source_residual"):
            proof_quality_state = "proof_failed_residual_warning"
        elif failure_reason in {"changed_outside_accepted_mask", "changed_outside_allowed_area", "collateral_change_too_broad"}:
            proof_quality_state = "proof_failed_collateral_warning"
        else:
            proof_quality_state = "proof_failed_warning"
        metrics.update(
            {
                "source_residual_improved": bool(source_residual_improved),
                "source_visible_area_increased": bool(source_visible_area_increased),
                "allowed_area_safe": bool(allowed_area_safe),
                "collateral_safe": bool(collateral_safe),
                "degraded_commit_allowed": False,
                "degraded_cleanup_diagnostic": bool(degraded_cleanup_diagnostic),
                "proof_quality_state": proof_quality_state,
                "proof_thresholds_unchanged": True,
                "erase_limits_unchanged": True,
            }
        )
    except Exception as exc:
        proof_status = ProofStatus.INCONCLUSIVE
        failure_reason = "inconclusive_missing_evidence"
        metrics = {"exception": f"{type(exc).__name__}: {exc}"}

    return _proof(
        cleanup_result,
        cleanup_plan,
        status=proof_status,
        failure_reason=failure_reason,
        metrics=metrics,
    )


def _cleanup_proof_page_context(source_image: Any) -> dict[str, Any]:
    if np is None or source_image is None:
        return {}
    try:
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        source_np = np.ascontiguousarray(source_np)
    except Exception:
        return {}
    return {
        "source_np": source_np,
        "source_gray": _gray(source_np),
        "source_image_hash": _hash_array(source_np),
    }


def _proof_scope_metadata(cleanup_plan: CleanupPlan, cleanup_mask: CleanupMask) -> dict[str, Any]:
    params = cleanup_plan.backend_parameters or {}
    visual_scope = _normalise_cleanup_visual_scope(getattr(cleanup_mask, "visual_scope", ""))
    partitioned = (
        _truthy(params.get("partitioned_cleanup"))
        or str(params.get("proof_scope") or "") == "component_partition"
        or visual_scope == "source_glyph_union_partition"
    )
    source_bbox = _valid_bbox(params.get("source_glyph_erasure_bbox")) or _valid_bbox(
        getattr(cleanup_mask, "erase_mask_bbox", None)
    )
    expected_bbox = _valid_bbox(params.get("source_glyph_erasure_expected_area_bbox"))
    parent_expected = _valid_bbox(params.get("parent_expected_erasure_bbox"))
    if partitioned:
        proof_bbox = _valid_bbox(params.get("proof_scope_bbox")) or source_bbox
        return {
            "proof_scope": "component_partition",
            "proof_scope_bbox": proof_bbox,
            "proof_residual_measurement_bbox": proof_bbox,
            "proof_scope_uses_parent_expected_bbox": False,
            "proof_scope_excludes_parent_expected_bbox": bool(parent_expected or expected_bbox),
            "parent_cleanup_unit_id": str(
                params.get("parent_cleanup_unit_id")
                or params.get("cleanup_unit_id")
                or cleanup_plan.cleanup_job_id
                or ""
            ),
            "parent_expected_erasure_bbox": parent_expected or expected_bbox,
            "source_glyph_erasure_bbox_input": source_bbox,
            "source_glyph_erasure_expected_area_bbox": expected_bbox,
        }
    proof_bbox = _valid_bbox(params.get("proof_scope_bbox")) or _union_valid_bboxes(
        source_bbox,
        expected_bbox,
    )
    return {
        "proof_scope": "cleanup_unit_full",
        "proof_scope_bbox": proof_bbox,
        "proof_residual_measurement_bbox": proof_bbox,
        "proof_scope_uses_parent_expected_bbox": bool(expected_bbox),
        "proof_scope_excludes_parent_expected_bbox": False,
        "parent_cleanup_unit_id": str(
            params.get("parent_cleanup_unit_id")
            or params.get("cleanup_unit_id")
            or cleanup_plan.cleanup_job_id
            or ""
        ),
        "parent_expected_erasure_bbox": parent_expected or expected_bbox,
        "source_glyph_erasure_bbox_input": source_bbox,
        "source_glyph_erasure_expected_area_bbox": expected_bbox,
    }


def run_cleanup_runtime_contract(
    *,
    page_id: str,
    image: Any,
    source_image: Any,
    job_candidates: Sequence[CleanupJob] | Any,
    mask_contracts: CleanupMaskBuildResult | Sequence[CleanupMask] | Any,
    plan_contracts: CleanupPlanBuildResult | Sequence[CleanupPlan] | Any | None = None,
    render_eligibility: Any | None,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None = None,
    inpaint_mode: str = "fast",
    runtime_class: CleanupClass | None = None,
    prewarmed_cleanup_model: bool = False,
) -> CleanupRuntimeContractResult:
    """Run Phase 5 cleanup-owned runtime proof without renderer consumption."""

    runtime_started = time.time()
    perf_collector = _CLEANUP_RUNTIME_PERF_COLLECTOR.get()
    perf_contract = perf_collector.get("contract") if isinstance(perf_collector, dict) else None
    perf_contract_started = time.perf_counter() if perf_contract is not None else 0.0
    status_records: list[dict[str, Any]] = []
    result_records: list[CleanupResult] = []
    proof_records: list[CleanupProof] = []
    parent_cleanup_unit_aggregate_records: list[dict[str, Any]] = []
    errors: list[str] = []
    jobs = _extract_jobs(job_candidates)
    jobs_by_id = {str(job.cleanup_job_id): job for job in jobs}
    masks = _extract_masks(mask_contracts)
    formal_plans = _extract_plans(plan_contracts)
    masks_by_cleanup_mask_id = {str(cleanup_mask.cleanup_mask_id): cleanup_mask for cleanup_mask in masks}
    formal_plans_by_cleanup_mask_id = _plans_by_cleanup_mask_id(formal_plans)
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    for cleanup_mask in masks:
        masks_by_job_id.setdefault(str(cleanup_mask.cleanup_job_id), []).append(cleanup_mask)
    render_eligibility_by_region = _render_eligibility_by_region_id(render_eligibility)
    proof_context = _cleanup_proof_page_context(source_image)
    _page014_timeout_checkpoint(
        "cleanup_runtime_contract",
        "start",
        page_id=page_id,
        job_count=len(jobs),
        mask_count=len(masks),
        plan_count=len(formal_plans),
        artifact_dir=artifact_dir or "",
    )
    if artifact_dir:
        try:
            os.makedirs(artifact_dir, exist_ok=True)
        except Exception as exc:
            errors.append(f"artifact_dir:{type(exc).__name__}: {exc}")

    runtime_obligations: list[tuple[CleanupJob, CleanupMask, CleanupPlan]] = []
    planned_mask_ids: set[str] = set()
    for formal_plan in formal_plans:
        cleanup_mask_id = str(formal_plan.cleanup_mask_id)
        cleanup_mask = masks_by_cleanup_mask_id.get(cleanup_mask_id)
        if cleanup_mask is None:
            status_records.append(
                {
                    "page_id": page_id,
                    "region_id": "",
                    "target_region_ids": [],
                    "cleanup_job_id": str(formal_plan.cleanup_job_id),
                    "cleanup_mask_id": cleanup_mask_id,
                    "cleanup_plan_id": str(formal_plan.cleanup_plan_id),
                    "runtime_status": "contract_error",
                    "failure_reason": "cleanup_contract_error_missing_cleanup_mask_for_formal_plan",
                    "cleanup_outcome_state": "cleanup_contract_error_missing_cleanup_mask_for_formal_plan",
                    "contract_error_original_reason": "formal_cleanup_plan_bound_mask_missing",
                    "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                    "renderer_consumed": False,
                }
            )
            continue
        job = jobs_by_id.get(str(formal_plan.cleanup_job_id)) or jobs_by_id.get(str(cleanup_mask.cleanup_job_id))
        if job is None:
            status_records.append(
                {
                    "page_id": page_id,
                    "region_id": "",
                    "target_region_ids": [],
                    "cleanup_job_id": str(formal_plan.cleanup_job_id or cleanup_mask.cleanup_job_id),
                    "cleanup_mask_id": cleanup_mask_id,
                    "cleanup_plan_id": str(formal_plan.cleanup_plan_id),
                    "runtime_status": "contract_error",
                    "failure_reason": "cleanup_contract_error_missing_cleanup_job_for_formal_plan",
                    "cleanup_outcome_state": "cleanup_contract_error_missing_cleanup_job_for_formal_plan",
                    "contract_error_original_reason": "formal_cleanup_plan_bound_job_missing",
                    "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                    "renderer_consumed": False,
                }
            )
            continue
        planned_mask_ids.add(cleanup_mask_id)
        runtime_obligations.append((job, cleanup_mask, formal_plan))

    for cleanup_mask in masks:
        cleanup_mask_id = str(cleanup_mask.cleanup_mask_id)
        if cleanup_mask_id in planned_mask_ids:
            continue
        if not _accepted_executable_cleanup_mask(cleanup_mask):
            continue
        job = jobs_by_id.get(str(cleanup_mask.cleanup_job_id))
        if job is None:
            status_records.append(
                {
                    "page_id": page_id,
                    "region_id": "",
                    "target_region_ids": [],
                    "cleanup_job_id": str(cleanup_mask.cleanup_job_id),
                    "cleanup_mask_id": cleanup_mask_id,
                    "runtime_status": "contract_error",
                    "failure_reason": "cleanup_contract_error_missing_cleanup_job_for_accepted_mask",
                    "cleanup_outcome_state": "cleanup_contract_error_missing_cleanup_job_for_accepted_mask",
                    "contract_error_original_reason": "accepted_cleanup_mask_bound_job_missing",
                    "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                    "renderer_consumed": False,
                }
            )
            continue
        base_record = _runtime_base_record(
            page_id,
            job,
            runtime_class=job.cleanup_class,
        )
        status_records.append(
            {
                **base_record,
                "runtime_status": "contract_error",
                "cleanup_mask_id": cleanup_mask_id,
                "failure_reason": "cleanup_contract_error_missing_formal_cleanup_plan",
                "cleanup_outcome_state": "cleanup_contract_error_missing_formal_cleanup_plan",
                "contract_error_original_reason": "formal_cleanup_plan_missing_for_accepted_cleanup_mask",
                "matching_formal_plan_count": len(formal_plans_by_cleanup_mask_id.get(cleanup_mask_id, [])),
                "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                "renderer_consumed": False,
            }
        )

    backend_contexts_by_mask_id = _backend_contexts_by_cleanup_mask_id(
        runtime_obligations,
        image,
    )
    model_required_obligation_count = sum(
        1
        for _job, obligation_mask, obligation_plan in runtime_obligations
        if _model_required_for_plan(obligation_plan, obligation_mask)
    )
    warmup_min_plans = 4
    try:
        warmup_min_plans = max(
            1,
            int(os.environ.get("MT_CLEANUP_INPAINT_WARMUP_MIN_PLANS", warmup_min_plans)),
        )
    except Exception:
        warmup_min_plans = 4
    if prewarmed_cleanup_model:
        _page014_timeout_checkpoint(
            "cleanup_runtime_model_warmup",
            "skipped",
            page_id=page_id,
            reason="prewarmed_before_page_loop",
            model_required_obligation_count=model_required_obligation_count,
            warmup_min_plans=warmup_min_plans,
        )
    elif use_gpu and model_required_obligation_count >= warmup_min_plans:
        warmup_started = time.time()
        _page014_timeout_checkpoint(
            "cleanup_runtime_model_warmup",
            "start",
            page_id=page_id,
            model_required_obligation_count=model_required_obligation_count,
            warmup_min_plans=warmup_min_plans,
        )
        try:
            from app.inpaint.simple_lama_engine import warm_cleanup_inpaint_model

            warmup_record = warm_cleanup_inpaint_model(use_gpu=use_gpu, model_id=model_id)
        except Exception as exc:
            warmup_record = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _page014_timeout_checkpoint(
            "cleanup_runtime_model_warmup",
            "end",
            page_id=page_id,
            model_required_obligation_count=model_required_obligation_count,
            warmup_status=str(warmup_record.get("status") if isinstance(warmup_record, Mapping) else ""),
            warmup_elapsed_ms=(
                warmup_record.get("elapsed_ms")
                if isinstance(warmup_record, Mapping)
                else None
            ),
            elapsed_ms=round((time.time() - warmup_started) * 1000.0, 3),
        )
    if perf_contract is not None:
        perf_contract["pre_job_setup_ms"] = _perf_elapsed_ms(perf_contract_started)
    perf_job_loop_started = time.perf_counter() if perf_contract is not None else 0.0

    for job, cleanup_mask, formal_plan in runtime_obligations:
        job_started = time.time()
        planned_runtime_class = _runtime_cleanup_class_for_job(job, runtime_class)
        base_record = _runtime_base_record(
            page_id,
            job,
            runtime_class=planned_runtime_class or _enum_value(job.cleanup_class),
        )
        target_region_ids = [str(rid) for rid in job.target_region_ids or [] if str(rid)]
        _page014_timeout_checkpoint(
            "cleanup_runtime_job",
            "start",
            page_id=page_id,
            cleanup_job_id=str(job.cleanup_job_id),
            region_id=target_region_ids[0] if target_region_ids else "",
            cleanup_class=_enum_value(job.cleanup_class),
        )
        try:
            suppressed_decision = _source_grounding_suppressed_decision(job, render_eligibility_by_region)
            if suppressed_decision is not None:
                base_record = {
                    **base_record,
                    "render_eligibility_suppressed_source_ungrounded": True,
                    "render_eligibility_source_grounding_failure_reason": _decision_value(
                        suppressed_decision,
                        "reason",
                    ),
                }
            if planned_runtime_class is None:
                contract_error = _contract_error_for_reason(
                    "runtime_cleanup_class_not_executable"
                )
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason=contract_error,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": "runtime_cleanup_class_not_executable",
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue
            exclusion = _runtime_job_exclusion_reason(job, cleanup_mask)
            if exclusion:
                contract_error = _contract_error_for_reason(exclusion)
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason=contract_error,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": exclusion,
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue
            runtime_mask = _runtime_glyph_precision_mask(
                cleanup_mask,
                source_image,
                planned_runtime_class,
            )
            mask_rejection = _runtime_structural_mask_error(runtime_mask)
            if mask_rejection:
                _cleanup_perf_contract_checkpoint(
                    "cleanup_runtime_mask_filter",
                    "rejected",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    reason=mask_rejection,
                    visual_scope=str(runtime_mask.visual_scope or ""),
                    foreground_pixels=runtime_mask.foreground_mask_pixels,
                    erase_pixels=runtime_mask.erase_mask_pixels,
                    allowed_area=runtime_mask.allowed_area,
                    consumed_source_glyph_count=len(runtime_mask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_count=len(runtime_mask.missing_source_glyph_mask_ids or []),
                )
                partition_started = time.time()
                _cleanup_perf_contract_checkpoint(
                    "cleanup_partitioning",
                    "start",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    parent_rejection_reason=mask_rejection,
                )
                partitioned = _partition_runtime_cleanup_mask(
                    runtime_mask,
                    rejection_reason=mask_rejection,
                    max_components=MAX_PARTITION_COMPONENTS,
                )
                _cleanup_perf_contract_checkpoint(
                    "cleanup_partitioning",
                    "end",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    parent_rejection_reason=mask_rejection,
                    partition_count=len(partitioned),
                    elapsed_ms=round((time.time() - partition_started) * 1000.0, 3),
                )
                if partitioned:
                    (
                        partition_results,
                        partition_proofs,
                        partition_statuses,
                        partition_aggregate_records,
                    ) = _run_partitioned_cleanup_attempts(
                        image=image.copy() if hasattr(image, "copy") else image,
                        source_image=source_image,
                        formal_plan=formal_plan,
                        job=job,
                        parent_cleanup_mask=runtime_mask,
                        inpaint_mode=inpaint_mode,
                        cleanup_class=planned_runtime_class,
                        partition_masks=partitioned,
                        base_record=base_record,
                        target_region_ids=target_region_ids,
                        parent_rejection_reason=mask_rejection,
                        page_id=page_id,
                        region_id=target_region_ids[0] if target_region_ids else "",
                        use_gpu=use_gpu,
                        model_id=model_id,
                        artifact_dir=artifact_dir,
                    )
                    result_records.extend(partition_results)
                    proof_records.extend(partition_proofs)
                    status_records.extend(partition_statuses)
                    parent_cleanup_unit_aggregate_records.extend(partition_aggregate_records)
                    _page014_timeout_checkpoint(
                        "cleanup_runtime_job",
                        "end",
                        page_id=page_id,
                        cleanup_job_id=str(job.cleanup_job_id),
                        region_id=target_region_ids[0] if target_region_ids else "",
                        runtime_status="partitioned",
                        failure_reason=mask_rejection,
                        partition_count=len(partitioned),
                        elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                    )
                    continue
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    reason=mask_rejection,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                contract_error = (
                    "cleanup_contract_error_component_limit"
                    if _is_partitionable_mask_rejection(mask_rejection)
                    else _contract_error_for_reason(mask_rejection)
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "cleanup_mask_id": runtime_mask.cleanup_mask_id,
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": mask_rejection,
                        "partitioned_cleanup_attempted": _is_partitionable_mask_rejection(mask_rejection),
                        "partitioned_cleanup_status": "no_valid_component_subplans",
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue

            pre_execution_risk_reason = str(
                (formal_plan.backend_parameters or {}).get("pre_execution_risk_reason")
                or ""
            )
            if pre_execution_risk_reason:
                _cleanup_perf_contract_checkpoint(
                    "cleanup_runtime_mask_filter",
                    "risk_recorded_execute_full_mask",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    risk_reason=pre_execution_risk_reason,
                    visual_scope=str(runtime_mask.visual_scope or ""),
                    foreground_pixels=runtime_mask.foreground_mask_pixels,
                    erase_pixels=runtime_mask.erase_mask_pixels,
                    allowed_area=runtime_mask.allowed_area,
                )

            plan = _runtime_plan_from_formal_plan(
                formal_plan=formal_plan,
                page_id=page_id,
                job=job,
                cleanup_mask=runtime_mask,
                inpaint_mode=inpaint_mode,
                cleanup_class=planned_runtime_class,
            )
            before = image.copy() if hasattr(image, "copy") else image
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "start",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                cleanup_plan_id=str(plan.cleanup_plan_id),
                cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
            )
            job_results, job_proofs, attempt_records = _run_adaptive_cleanup_attempts(
                image=before,
                source_image=source_image,
                cleanup_plan=plan,
                cleanup_mask=runtime_mask,
                backend_context=backend_contexts_by_mask_id.get(str(runtime_mask.cleanup_mask_id)),
                page_id=page_id,
                region_id=target_region_ids[0] if target_region_ids else "",
                use_gpu=use_gpu,
                model_id=model_id,
                artifact_dir=artifact_dir,
                proof_context=proof_context,
            )
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                attempt_count=len(attempt_records),
                result_count=len(job_results),
                proof_count=len(job_proofs),
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
            result_records.extend(job_results)
            proof_records.extend(job_proofs)
            cleanup_result, cleanup_proof = _select_best_commit_candidate_result(job_results, job_proofs)
            if cleanup_result is None:
                cleanup_result, cleanup_proof = _best_failed_attempt(job_results, job_proofs)
            proof_passed = bool(cleanup_proof and cleanup_proof.proof_status == ProofStatus.PASSED)
            degraded_cleanup_diagnostic = bool(
                cleanup_proof
                and isinstance(cleanup_proof.metrics, Mapping)
                and cleanup_proof.metrics.get("degraded_cleanup_diagnostic", False)
            )
            runtime_status = (
                "passed"
                if proof_passed and cleanup_result and cleanup_result.pixel_changed
                else "failed"
            )
            failure_reason = (
                "passed"
                if runtime_status == "passed"
                else (cleanup_proof.failure_reason if cleanup_proof else "")
                or (cleanup_result.failure_reason if cleanup_result else "")
                or "cleanup_runtime_proof_failed"
            )
            selected_attempt = _selected_attempt_record(attempt_records, cleanup_result)
            selected_proof_metrics = _json_safe(cleanup_proof.metrics or {}) if cleanup_proof else {}
            status_records.append(
                {
                    **base_record,
                    "runtime_status": runtime_status,
                    "cleanup_plan_id": cleanup_result.cleanup_plan_id if cleanup_result else plan.cleanup_plan_id,
                    "cleanup_mask_id": runtime_mask.cleanup_mask_id,
                    "cleanup_obligation_id": runtime_mask.cleanup_mask_id,
                    "text_block_root_id": (plan.backend_parameters or {}).get(
                        "text_block_root_id",
                        base_record.get("text_block_root_id", ""),
                    ),
                    "cleanup_result_id": cleanup_result.cleanup_result_id if cleanup_result else "",
                    "cleanup_proof_id": cleanup_proof.cleanup_proof_id if cleanup_proof else "",
                    "proof_status": _enum_value(cleanup_proof.proof_status) if cleanup_proof else "",
                    "failure_reason": failure_reason,
                    "cleanup_outcome_state": _cleanup_outcome_state_for_failure(failure_reason, runtime_status=runtime_status),
                    "pixel_changed": bool(cleanup_result and cleanup_result.pixel_changed),
                    "changed_pixel_count": int((cleanup_result.changed_pixel_count if cleanup_result else 0) or 0),
                    "cleaned_image_ref": cleanup_result.cleaned_image_ref if cleanup_result else None,
                    "diff_ref": cleanup_result.diff_ref if cleanup_result else None,
                    "mask_ref": cleanup_result.mask_ref if cleanup_result else None,
                    "proof_metrics": selected_proof_metrics,
                    "proof_quality_state": selected_proof_metrics.get("proof_quality_state", ""),
                    "degraded_commit_allowed": False,
                    "degraded_cleanup_diagnostic": bool(degraded_cleanup_diagnostic),
                    "source_residual_delta_dark_pixels": selected_proof_metrics.get(
                        "source_residual_delta_dark_pixels",
                        0,
                    ),
                    "source_residual_delta_dark_ratio": selected_proof_metrics.get(
                        "source_residual_delta_dark_ratio",
                        0,
                    ),
                    "collateral_art_damage_delta": selected_proof_metrics.get(
                        "collateral_art_damage_delta",
                        0,
                    ),
                    "proof_scope": selected_proof_metrics.get("proof_scope", "cleanup_unit_full"),
                    "proof_scope_bbox": selected_proof_metrics.get("proof_scope_bbox", []),
                    "parent_cleanup_unit_id": selected_proof_metrics.get(
                        "parent_cleanup_unit_id",
                        base_record.get("cleanup_unit_id", ""),
                    ),
                    "parent_expected_erasure_bbox": selected_proof_metrics.get(
                        "parent_expected_erasure_bbox",
                        [],
                    ),
                    "adaptive_attempt_count": len(attempt_records),
                    "adaptive_attempts": _json_safe(attempt_records),
                    "selected_attempt_index": selected_attempt.get("attempt_index") if selected_attempt else None,
                    "selected_attempt_strategy": selected_attempt.get("strategy") if selected_attempt else "",
                    "render_consumption_decision_if_consumed": (
                        cleanup_proof.render_consumption_decision_if_consumed
                        if cleanup_proof
                        else "block_future_renderer_consumption"
                    ),
                    "renderer_consumed": False,
                }
            )
            _page014_timeout_checkpoint(
                "cleanup_runtime_job",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                runtime_status=runtime_status,
                failure_reason=failure_reason,
                attempt_count=len(attempt_records),
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _page014_timeout_checkpoint(
                "cleanup_runtime_job",
                "error",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                error=message,
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
            errors.append(message)
            status_records.append(
                {
                    **base_record,
                    "runtime_status": "inconclusive",
                    "failure_reason": message,
                    "cleanup_outcome_state": "cleanup_partially_completed_with_warnings",
                    "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                    "renderer_consumed": False,
                }
            )

    if perf_contract is not None:
        perf_contract["job_loop_and_selection_ms"] = _perf_elapsed_ms(perf_job_loop_started)
        perf_contract["contract_total_ms"] = _perf_elapsed_ms(perf_contract_started)
        perf_contract["runtime_obligation_count"] = len(runtime_obligations)
        perf_contract["result_count"] = len(result_records)
        perf_contract["proof_count"] = len(proof_records)
    _page014_timeout_checkpoint(
        "cleanup_runtime_contract",
        "end",
        page_id=page_id,
        status_count=len(status_records),
        result_count=len(result_records),
        proof_count=len(proof_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - runtime_started) * 1000.0, 3),
    )
    return CleanupRuntimeContractResult(
        page_id=page_id,
        version=CLEANUP_RUNTIME_CONTRACT_VERSION,
        runtime_class=runtime_class or "phase5_cleanup_owned_general",
        status_records=status_records,
        result_records=result_records,
        proof_records=proof_records,
        parent_cleanup_unit_aggregate_records=parent_cleanup_unit_aggregate_records,
        errors=errors,
    )


def commit_cleanup_runtime_results_to_working_image(
    *,
    page_id: str,
    source_image: Any,
    runtime_contract: CleanupRuntimeContractResult | Any,
    artifact_dir: str | None = None,
    excluded_region_ids: set[str] | None = None,
) -> CleanupUpstreamCommitResult:
    """Commit proof-passed runtime cleanup results to a pre-render working image."""

    commit_started = time.time()
    errors: list[str] = []
    commit_records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []
    excluded = {str(region_id) for region_id in (excluded_region_ids or set()) if str(region_id)}
    working = source_image.copy() if hasattr(source_image, "copy") else source_image
    if Image is None or np is None:
        return CleanupUpstreamCommitResult(
            page_id=page_id,
            version=CLEANUP_UPSTREAM_COMMIT_VERSION,
            cleaned_image=working,
            blocked_records=[
                {
                    "page_id": page_id,
                    "region_id": "",
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": "pillow_or_numpy_unavailable",
                    "renderer_consumed": False,
                }
            ],
        )

    result_records = list(getattr(runtime_contract, "result_records", []) or [])
    proof_records = list(getattr(runtime_contract, "proof_records", []) or [])
    status_records = list(getattr(runtime_contract, "status_records", []) or [])
    _page014_timeout_checkpoint(
        "cleanup_upstream_commit",
        "start",
        page_id=page_id,
        result_count=len(result_records),
        proof_count=len(proof_records),
        status_count=len(status_records),
    )
    proof_by_result_id = {str(proof.cleanup_result_id): proof for proof in proof_records}
    status_by_result_id = {
        str(status.get("cleanup_result_id") or ""): status
        for status in status_records
        if isinstance(status, Mapping) and str(status.get("cleanup_result_id") or "")
    }

    try:
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        working_np = np.asarray(working.convert("RGB") if hasattr(working, "convert") else working).copy()
    except Exception as exc:
        return CleanupUpstreamCommitResult(
            page_id=page_id,
            version=CLEANUP_UPSTREAM_COMMIT_VERSION,
            cleaned_image=working,
            blocked_records=[
                {
                    "page_id": page_id,
                    "region_id": "",
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": f"source_image_array_error:{type(exc).__name__}: {exc}",
                    "renderer_consumed": False,
                }
            ],
        )

    commit_entries: list[dict[str, Any]] = []
    root_transaction_records: list[dict[str, Any]] = []
    result_ids = {str(result.cleanup_result_id) for result in result_records}
    selected_result_ids = {
        str(status.get("cleanup_result_id") or "")
        for status in status_records
        if isinstance(status, Mapping) and str(status.get("cleanup_result_id") or "")
    }
    for cleanup_result in result_records:
        result_id = str(cleanup_result.cleanup_result_id)
        if selected_result_ids and result_id not in selected_result_ids:
            continue
        proof = proof_by_result_id.get(result_id)
        status = status_by_result_id.get(result_id, {})
        entry = _upstream_commit_entry_for_result(
            page_id=page_id,
            cleanup_result=cleanup_result,
            proof=proof,
            status=status,
            excluded_region_ids=excluded,
        )
        commit_entries.append(entry)

    for status in status_records:
        if not isinstance(status, Mapping):
            continue
        result_id = str(status.get("cleanup_result_id") or "")
        if result_id and result_id in result_ids:
            continue
        if not _status_addresses_cleanup_obligation(status):
            continue
        commit_entries.append(_upstream_commit_entry_for_status(page_id=page_id, status=status))

    entries_by_root: dict[str, list[dict[str, Any]]] = {}
    for entry in commit_entries:
        entries_by_root.setdefault(str(entry.get("text_block_root_id") or "root_unknown"), []).append(entry)

    for root_id in sorted(entries_by_root):
        root_started = time.time()
        root_entries = entries_by_root[root_id]
        root_obligation_ids = sorted(
            {
                str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
                for entry in root_entries
                if str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
            }
        )
        failed_entries = [entry for entry in root_entries if str(entry.get("failure_reason") or "")]
        candidate_entries = [
            entry
            for entry in root_entries
            if not str(entry.get("failure_reason") or "") and entry.get("cleanup_result") is not None
        ]
        proof_passed_obligations = sorted(
            {
                str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
                for entry in candidate_entries
                if str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
            }
        )
        root_failure = ""
        if failed_entries:
            root_failure = "root_transaction_incomplete"
        elif not root_obligation_ids:
            root_failure = "root_transaction_missing_cleanup_obligation"
        elif set(proof_passed_obligations) != set(root_obligation_ids):
            root_failure = "root_transaction_incomplete"

        root_commit_records: list[dict[str, Any]] = []
        root_committed_obligations: set[str] = set()
        if not root_failure:
            root_working_np = working_np.copy()
            for entry in candidate_entries:
                result_started = time.time()
                cleanup_result = entry["cleanup_result"]
                proof = entry.get("proof")
                status = entry.get("status") if isinstance(entry.get("status"), Mapping) else {}
                base = dict(entry.get("base") or {})
                region_id = str(base.get("region_id") or "")
                result_id = str(base.get("cleanup_result_id") or "")
                cleaned_payload = _cleaned_crop_payload_for_result(cleanup_result)
                if cleaned_payload is None:
                    cleaned_payload = cleanup_result.cleaned_image
                if cleaned_payload is None:
                    cleaned_payload = _load_cleanup_image(cleanup_result.cleaned_image_ref)
                if cleaned_payload is None:
                    root_failure = "runtime_cleaned_image_missing"
                    entry["failure_reason"] = root_failure
                    break
                try:
                    accepted_commit_mask = _cleanup_result_commit_mask(cleanup_result, source_np.shape[:2])
                    committed_pixels, commit_mask = _commit_runtime_result_pixels(
                        working_np=root_working_np,
                        source_np=source_np,
                        cleaned_image=cleaned_payload,
                        operation_bbox=cleanup_result.operation_bbox,
                        accepted_mask=accepted_commit_mask,
                    )
                except Exception as exc:
                    errors.append(f"{region_id}:{type(exc).__name__}: {exc}")
                    committed_pixels = 0
                    commit_mask = None
                if committed_pixels <= 0:
                    root_failure = "no_pixels_committed_to_working_image"
                    entry["failure_reason"] = root_failure
                    break
                proof_metrics = proof.metrics if proof is not None and isinstance(proof.metrics, Mapping) else {}
                owned_segmentation_pixels = _int_or_none(base.get("owned_segmentation_pixels"))
                committed_cleanup_mask_pixels = int(np.count_nonzero(commit_mask)) if commit_mask is not None else 0
                owned_segmentation_to_commit_ratio = (
                    round(float(committed_cleanup_mask_pixels) / float(max(1, owned_segmentation_pixels)), 4)
                    if owned_segmentation_pixels is not None
                    else None
                )
                degraded_warning_commit = _proof_allows_degraded_commit(proof, cleanup_result, status)
                root_commit_records.append(
                    {
                        **base,
                        "cleanup_applied_upstream": True,
                        "cleanup_committed_to_working_image": True,
                        "committed_pixel_count": committed_pixels,
                        "commit_mask_pixels": committed_cleanup_mask_pixels,
                        "committed_cleanup_mask_pixels": committed_cleanup_mask_pixels,
                        "accepted_erase_mask_hash": str(
                            (cleanup_result.backend_parameters or {}).get("input_mask_hash")
                            or cleanup_result.input_mask_hash
                            or ""
                        ),
                        "committed_mask_hash": _hash_mask(commit_mask),
                        "owned_segmentation_to_commit_ratio": owned_segmentation_to_commit_ratio,
                        "failure_reason": "passed",
                        "cleanup_commit_quality": (
                            "residual_warning_commit"
                            if degraded_warning_commit
                            else "proof_passed_clean_commit"
                        ),
                        "residual_warning": bool(degraded_warning_commit),
                        "proof_quality_state": proof_metrics.get("proof_quality_state", ""),
                        "source_residual_delta_dark_pixels": proof_metrics.get(
                            "source_residual_delta_dark_pixels",
                            0,
                        ),
                        "source_residual_delta_dark_ratio": proof_metrics.get(
                            "source_residual_delta_dark_ratio",
                            0,
                        ),
                        "collateral_art_damage_delta": proof_metrics.get("collateral_art_damage_delta", 0),
                        "broad_white_patch_risk": bool(proof_metrics.get("broad_white_patch_risk", False)),
                        "root_transaction_status": "root_committed",
                    }
                )
                root_committed_obligations.add(
                    str(base.get("cleanup_obligation_id") or base.get("cleanup_mask_id") or "")
                )
                _page014_timeout_checkpoint(
                    "cleanup_upstream_commit_result",
                    "root_candidate_committed",
                    page_id=page_id,
                    region_id=region_id,
                    cleanup_result_id=result_id,
                    text_block_root_id=root_id,
                    committed_pixel_count=committed_pixels,
                    elapsed_ms=round((time.time() - result_started) * 1000.0, 3),
                )
            if not root_failure:
                working_np[:] = root_working_np
                commit_records.extend(root_commit_records)

        if root_failure:
            for entry in root_entries:
                base = dict(entry.get("base") or {})
                entry_failure = str(entry.get("failure_reason") or "")
                blocked_reason = entry_failure or root_failure
                if not entry_failure and root_failure == "root_transaction_incomplete":
                    blocked_reason = "root_transaction_incomplete"
                blocked_records.append(
                    {
                        **base,
                        "cleanup_applied_upstream": False,
                        "cleanup_committed_to_working_image": False,
                        "failure_reason": blocked_reason,
                        "root_transaction_status": "root_blocked",
                        "root_block_reason": root_failure,
                    }
                )
                _page014_timeout_checkpoint(
                    "cleanup_upstream_commit_result",
                    "root_blocked",
                    page_id=page_id,
                    region_id=str(base.get("region_id") or ""),
                    cleanup_result_id=str(base.get("cleanup_result_id") or ""),
                    text_block_root_id=root_id,
                    reason=blocked_reason,
                    elapsed_ms=round((time.time() - root_started) * 1000.0, 3),
                )

        root_status = "root_blocked" if root_failure else "root_committed"
        root_transaction_records.append(
            {
                "page_id": page_id,
                "text_block_root_id": root_id,
                "root_id": root_id,
                "accepted_cleanup_mask_ids": root_obligation_ids,
                "planned_cleanup_mask_ids": root_obligation_ids,
                "executed_cleanup_mask_ids": sorted(
                    {
                        str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
                        for entry in root_entries
                        if entry.get("cleanup_result") is not None
                        and str(entry.get("cleanup_obligation_id") or entry.get("cleanup_mask_id") or "")
                    }
                ),
                "proof_passed_cleanup_mask_ids": proof_passed_obligations,
                "committed_cleanup_mask_ids": sorted(root_committed_obligations),
                "blocked_cleanup_mask_ids": (
                    sorted(set(root_obligation_ids) - set(root_committed_obligations))
                    if root_failure
                    else []
                ),
                "root_obligation_count": len(root_obligation_ids),
                "root_completed_obligation_count": len(root_committed_obligations),
                "root_transaction_status": root_status,
                "block_reason": root_failure,
                "renderer_consumed": False,
            }
        )

    committed_image = Image.fromarray(working_np.astype("uint8"), mode="RGB")
    refs = _write_upstream_commit_artifacts(
        artifact_dir=artifact_dir,
        page_id=page_id,
        before=source_image,
        cleaned=committed_image,
        committed_records=commit_records,
    )
    for record in commit_records:
        record["cleanup_upstream_cleaned_image_ref"] = refs.get("committed_image_ref", "")
        record["cleanup_upstream_diff_ref"] = refs.get("commit_diff_ref", "")
        record["cleanup_upstream_mask_ref"] = refs.get("commit_mask_ref", "")

    _page014_timeout_checkpoint(
        "cleanup_upstream_commit",
        "end",
        page_id=page_id,
        committed_count=len(commit_records),
        blocked_count=len(blocked_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - commit_started) * 1000.0, 3),
    )
    return CleanupUpstreamCommitResult(
        page_id=page_id,
        version=CLEANUP_UPSTREAM_COMMIT_VERSION,
        cleaned_image=committed_image,
        committed_image_ref=refs.get("committed_image_ref", ""),
        commit_diff_ref=refs.get("commit_diff_ref", ""),
        commit_mask_ref=refs.get("commit_mask_ref", ""),
        commit_records=commit_records,
        blocked_records=blocked_records,
        root_transaction_records=root_transaction_records,
        errors=errors,
    )


def _upstream_commit_entry_for_result(
    *,
    page_id: str,
    cleanup_result: CleanupResult,
    proof: CleanupProof | None,
    status: Mapping[str, Any],
    excluded_region_ids: set[str],
) -> dict[str, Any]:
    result_id = str(cleanup_result.cleanup_result_id)
    result_backend_parameters = (
        cleanup_result.backend_parameters
        if isinstance(getattr(cleanup_result, "backend_parameters", None), Mapping)
        else {}
    )
    identity = _cleanup_identity_fields(
        cleanup_result,
        proof,
        status,
        result_backend_parameters,
        fallback_region_id=str(cleanup_result.region_id or status.get("region_id") or ""),
    )
    region_id = str(
        identity.get("parent_execution_bundle_id")
        or identity.get("parent_logical_text_unit_id")
        or cleanup_result.region_id
        or status.get("region_id")
        or ""
    )
    owned_segmentation_pixels = _int_or_none(result_backend_parameters.get("owned_segmentation_pixels"))
    executable_foreground_pixels = _int_or_none(result_backend_parameters.get("executable_foreground_pixels"))
    owned_to_executable_ratio = _float_or_none(
        result_backend_parameters.get("owned_segmentation_to_executable_ratio")
    )
    cleanup_obligation_id = _commit_cleanup_obligation_id(
        cleanup_result=cleanup_result,
        status=status,
        backend_parameters=result_backend_parameters,
    )
    text_block_root_id = _commit_text_block_root_id(
        cleanup_result=cleanup_result,
        status=status,
        backend_parameters=result_backend_parameters,
        cleanup_obligation_id=cleanup_obligation_id,
    )
    base = {
        "page_id": page_id,
        "region_id": region_id,
        "parent_execution_bundle_id": str(identity.get("parent_execution_bundle_id") or ""),
        "parent_logical_text_unit_id": str(identity.get("parent_logical_text_unit_id") or ""),
        "target_region_ids": list(identity.get("target_region_ids") or []),
        "text_block_root_id": text_block_root_id,
        "cleanup_obligation_id": cleanup_obligation_id,
        "mask_space": str(result_backend_parameters.get("mask_space") or status.get("mask_space") or "page"),
        "bbox_format": str(
            result_backend_parameters.get("bbox_format")
            or status.get("bbox_format")
            or "xyxy_exclusive"
        ),
        "cleanup_result_id": result_id,
        "cleanup_proof_id": str(getattr(proof, "cleanup_proof_id", "") or ""),
        "cleanup_job_id": str(cleanup_result.cleanup_job_id),
        "cleanup_mask_id": str(cleanup_result.cleanup_mask_id),
        "cleanup_plan_id": str(cleanup_result.cleanup_plan_id),
        "cleanup_unit_id": str(status.get("cleanup_unit_id") or cleanup_result.cleanup_job_id),
        "cleanup_class": _enum_value(cleanup_result.cleanup_class),
        "pixel_changed": bool(cleanup_result.pixel_changed),
        "changed_pixel_count": int(cleanup_result.changed_pixel_count or 0),
        "renderer_consumed": False,
        "runtime_cleaned_image_ref": cleanup_result.cleaned_image_ref or "",
        "runtime_diff_ref": cleanup_result.diff_ref or "",
        "runtime_mask_ref": cleanup_result.mask_ref or "",
        "owned_segmentation_pixels": owned_segmentation_pixels,
        "executable_foreground_pixels": executable_foreground_pixels,
        "foreground_mask_pixels": result_backend_parameters.get("foreground_mask_pixels"),
        "erase_mask_pixels": result_backend_parameters.get("erase_mask_pixels"),
        "owned_segmentation_to_executable_ratio": owned_to_executable_ratio,
        "committed_cleanup_mask_pixels": 0,
        "owned_segmentation_to_commit_ratio": 0.0 if owned_segmentation_pixels is not None else None,
        "ready_but_sparse_violation": bool(result_backend_parameters.get("ready_but_sparse_violation", False)),
        "sourceglyph_executable_influence_detected": bool(
            result_backend_parameters.get("sourceglyph_executable_influence_detected", False)
        ),
        "segmentation_contract_override_detected": bool(
            result_backend_parameters.get("segmentation_contract_override_detected", False)
        ),
        "partitioned_cleanup": bool(status.get("partitioned_cleanup", False)),
        "partition_index": status.get("partition_index", ""),
        "partition_count": status.get("partition_count", ""),
        "parent_cleanup_mask_id": status.get("parent_cleanup_mask_id", ""),
        "parent_cleanup_unit_id": status.get("parent_cleanup_unit_id", ""),
        "parent_cleanup_unit_aggregate_status": status.get("parent_cleanup_unit_aggregate_status", ""),
        "parent_cleanup_unit_aggregate_complete": status.get("parent_cleanup_unit_aggregate_complete", ""),
        "partition_parent_full_mask_proof_complete": bool(
            status.get("partition_parent_full_mask_proof_complete", False)
        ),
        "parent_cleanup_unit_component_attempt_count": status.get(
            "parent_cleanup_unit_component_attempt_count",
            0,
        ),
    }
    failure_reason = _upstream_commit_block_reason(cleanup_result, proof, status, excluded_region_ids)
    return {
        "text_block_root_id": text_block_root_id,
        "cleanup_obligation_id": cleanup_obligation_id,
        "cleanup_mask_id": str(cleanup_result.cleanup_mask_id),
        "cleanup_result": cleanup_result,
        "proof": proof,
        "status": status,
        "base": base,
        "failure_reason": failure_reason,
    }


def _upstream_commit_entry_for_status(
    *,
    page_id: str,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup_mask_id = str(status.get("cleanup_mask_id") or "")
    identity = _cleanup_identity_fields(
        status,
        fallback_region_id=str(status.get("region_id") or ""),
    )
    region_id = str(
        identity.get("parent_execution_bundle_id")
        or identity.get("parent_logical_text_unit_id")
        or status.get("region_id")
        or ""
    )
    cleanup_obligation_id = str(
        status.get("cleanup_obligation_id")
        or status.get("parent_cleanup_mask_id")
        or cleanup_mask_id
        or ""
    )
    text_block_root_id = str(
        status.get("text_block_root_id")
        or status.get("parent_cleanup_unit_id")
        or status.get("cleanup_unit_id")
        or cleanup_obligation_id
        or status.get("cleanup_job_id")
        or "root_unknown"
    )
    failure_reason = str(
        status.get("failure_reason")
        or status.get("cleanup_outcome_state")
        or "runtime_result_missing_for_obligation"
    )
    base = {
        "page_id": page_id,
        "region_id": region_id,
        "parent_execution_bundle_id": str(identity.get("parent_execution_bundle_id") or ""),
        "parent_logical_text_unit_id": str(identity.get("parent_logical_text_unit_id") or ""),
        "target_region_ids": list(identity.get("target_region_ids") or []),
        "text_block_root_id": text_block_root_id,
        "cleanup_obligation_id": cleanup_obligation_id,
        "mask_space": str(status.get("mask_space") or "page"),
        "bbox_format": str(status.get("bbox_format") or "xyxy_exclusive"),
        "cleanup_result_id": str(status.get("cleanup_result_id") or ""),
        "cleanup_proof_id": str(status.get("cleanup_proof_id") or ""),
        "cleanup_job_id": str(status.get("cleanup_job_id") or ""),
        "cleanup_mask_id": cleanup_mask_id,
        "cleanup_plan_id": str(status.get("cleanup_plan_id") or ""),
        "cleanup_unit_id": str(status.get("cleanup_unit_id") or ""),
        "cleanup_class": str(status.get("cleanup_class") or ""),
        "pixel_changed": bool(status.get("pixel_changed")),
        "changed_pixel_count": int(status.get("changed_pixel_count") or 0),
        "renderer_consumed": False,
        "runtime_cleaned_image_ref": str(status.get("cleaned_image_ref") or ""),
        "runtime_diff_ref": str(status.get("diff_ref") or ""),
        "runtime_mask_ref": str(status.get("mask_ref") or ""),
        "owned_segmentation_pixels": _int_or_none(status.get("owned_segmentation_pixels")),
        "executable_foreground_pixels": _int_or_none(status.get("executable_foreground_pixels")),
        "committed_cleanup_mask_pixels": 0,
        "partitioned_cleanup": bool(status.get("partitioned_cleanup", False)),
        "partition_index": status.get("partition_index", ""),
        "partition_count": status.get("partition_count", ""),
        "parent_cleanup_mask_id": status.get("parent_cleanup_mask_id", ""),
        "parent_cleanup_unit_id": status.get("parent_cleanup_unit_id", ""),
        "parent_cleanup_unit_aggregate_status": status.get("parent_cleanup_unit_aggregate_status", ""),
        "parent_cleanup_unit_aggregate_complete": status.get("parent_cleanup_unit_aggregate_complete", ""),
        "partition_parent_full_mask_proof_complete": bool(
            status.get("partition_parent_full_mask_proof_complete", False)
        ),
        "parent_cleanup_unit_component_attempt_count": status.get(
            "parent_cleanup_unit_component_attempt_count",
            0,
        ),
    }
    return {
        "text_block_root_id": text_block_root_id,
        "cleanup_obligation_id": cleanup_obligation_id,
        "cleanup_mask_id": cleanup_mask_id,
        "cleanup_result": None,
        "proof": None,
        "status": status,
        "base": base,
        "failure_reason": failure_reason,
    }


def _commit_cleanup_obligation_id(
    *,
    cleanup_result: CleanupResult,
    status: Mapping[str, Any],
    backend_parameters: Mapping[str, Any],
) -> str:
    return str(
        backend_parameters.get("cleanup_obligation_id")
        or status.get("cleanup_obligation_id")
        or status.get("parent_cleanup_mask_id")
        or backend_parameters.get("parent_cleanup_mask_id")
        or cleanup_result.cleanup_mask_id
        or ""
    )


def _commit_text_block_root_id(
    *,
    cleanup_result: CleanupResult,
    status: Mapping[str, Any],
    backend_parameters: Mapping[str, Any],
    cleanup_obligation_id: str,
) -> str:
    return str(
        backend_parameters.get("text_block_root_id")
        or status.get("text_block_root_id")
        or status.get("parent_cleanup_unit_id")
        or backend_parameters.get("parent_cleanup_unit_id")
        or status.get("cleanup_unit_id")
        or cleanup_obligation_id
        or cleanup_result.cleanup_job_id
        or "root_unknown"
    )


def _status_addresses_cleanup_obligation(status: Mapping[str, Any]) -> bool:
    return bool(
        str(status.get("cleanup_mask_id") or "")
        or str(status.get("cleanup_obligation_id") or "")
        or str(status.get("parent_cleanup_mask_id") or "")
    )


def _upstream_commit_block_reason(
    cleanup_result: CleanupResult,
    proof: CleanupProof | None,
    status: Mapping[str, Any],
    excluded_region_ids: set[str],
) -> str:
    region_id = str(cleanup_result.region_id or status.get("region_id") or "")
    target_region_ids = [str(rid) for rid in status.get("target_region_ids", []) if str(rid)]
    if region_id in excluded_region_ids or any(rid in excluded_region_ids for rid in target_region_ids):
        return "protected_region_excluded_from_upstream_commit"
    if bool(cleanup_result.renderer_consumed):
        return "runtime_result_already_renderer_consumed"
    if proof is None:
        return "missing_cleanup_proof"
    if proof.proof_status != ProofStatus.PASSED:
        return proof.failure_reason or "cleanup_proof_not_passed"
    if isinstance(proof.metrics, Mapping) and bool(proof.metrics.get("broad_white_patch_risk", False)):
        return "broad_white_patch_risk"
    if (
        bool(status.get("partitioned_cleanup", False))
        and status.get("parent_cleanup_unit_aggregate_complete") is False
        and not bool(status.get("partition_parent_full_mask_proof_complete", False))
    ):
        return "parent_cleanup_unit_aggregate_incomplete"
    if str(status.get("runtime_status") or "") not in {"", "passed", "warning", "failed"}:
        return str(status.get("failure_reason") or "cleanup_runtime_status_not_passed")
    if not cleanup_result.pixel_changed or int(cleanup_result.changed_pixel_count or 0) <= 0:
        return "cleanup_result_not_pixel_changing"
    if proof.source_glyph_removal_passed is False:
        return "source_glyph_removal_proof_failed"
    if proof.mask_containment_passed is False:
        return "mask_containment_proof_failed"
    changed_outside = None
    if isinstance(proof.metrics, Mapping):
        changed_outside = _int_or_none(proof.metrics.get("changed_outside_accepted_mask_pixels"))
    if changed_outside is None:
        changed_outside = _int_or_none(proof.changed_outside_allowed_pixels)
    if changed_outside is None and isinstance(proof.metrics, Mapping):
        changed_outside = _int_or_none(proof.metrics.get("changed_outside_allowed_pixels"))
    if changed_outside is not None and changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS:
        return "changed_outside_accepted_mask"
    changed_outside_ratio = None
    if isinstance(proof.metrics, Mapping):
        changed_outside_ratio = _float_or_none(proof.metrics.get("changed_outside_accepted_mask_ratio"))
        if changed_outside_ratio is None:
            changed_outside_ratio = _float_or_none(proof.metrics.get("changed_outside_allowed_ratio"))
    if changed_outside_ratio is not None and changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO:
        return "changed_outside_accepted_mask_ratio"
    decision = str(proof.render_consumption_decision_if_consumed or "")
    if decision and decision not in {
        "allow_stage6_consumption_candidate",
    }:
        return decision
    return ""


def _load_cleanup_image(path: str | None) -> Any | None:
    if not path or Image is None:
        return None
    try:
        if not os.path.isfile(path):
            return None
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except Exception:
        return None


def _cleanup_result_commit_mask(cleanup_result: CleanupResult, shape: tuple[int, int] | Sequence[int]) -> Any | None:
    in_memory = _binary_mask(getattr(cleanup_result, "commit_mask", None), shape)
    if in_memory is not None and np is not None and np.any(in_memory > 0):
        return in_memory
    return _load_binary_mask_ref(cleanup_result.mask_ref, shape)


def _commit_runtime_result_pixels(
    *,
    working_np: Any,
    source_np: Any,
    cleaned_image: Any,
    operation_bbox: Sequence[int] | None,
    accepted_mask: Any | None = None,
) -> tuple[int, Any | None]:
    if np is None:
        return 0, None
    cleaned_np = np.asarray(cleaned_image.convert("RGB") if hasattr(cleaned_image, "convert") else cleaned_image)
    if working_np.shape != source_np.shape:
        return 0, None
    accepted = None
    if accepted_mask is not None:
        accepted = np.asarray(accepted_mask) > 0
        if accepted.shape[:2] != source_np.shape[:2]:
            return 0, None
    operation_box = _valid_bbox(operation_bbox)
    box = None
    if cleaned_np.shape != source_np.shape and operation_box is not None:
        ox0, oy0, ox1, oy1 = operation_box
        if tuple(cleaned_np.shape[:2]) == (oy1 - oy0, ox1 - ox0):
            box = operation_box
    if box is None:
        box = _mask_page_bbox(accepted) if accepted is not None else None
    if box is None:
        box = operation_box
    if box is None:
        return 0, None
    height, width = source_np.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return 0, None
    before_crop = source_np[y0:y1, x0:x1]
    if cleaned_np.shape == source_np.shape:
        cleaned_crop = cleaned_np[y0:y1, x0:x1]
    elif tuple(cleaned_np.shape[:2]) == (y1 - y0, x1 - x0):
        cleaned_crop = cleaned_np
    else:
        return 0, None
    diff = np.abs(cleaned_crop.astype(np.int16) - before_crop.astype(np.int16))
    changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
    if accepted is not None:
        changed = changed & accepted[y0:y1, x0:x1]
    changed_pixels = int(np.count_nonzero(changed))
    if changed_pixels <= 0:
        return 0, None
    target_crop = working_np[y0:y1, x0:x1]
    target_crop[changed] = cleaned_crop[changed]
    working_np[y0:y1, x0:x1] = target_crop
    mask = np.zeros(source_np.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = changed.astype(np.uint8)
    return changed_pixels, mask


def _mask_page_bbox(mask: Any | None) -> list[int] | None:
    if np is None or mask is None:
        return None
    try:
        arr = np.asarray(mask) > 0
        ys, xs = np.where(arr)
    except Exception:
        return None
    if xs.size <= 0 or ys.size <= 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_connected_component_metrics(mask: Any | None) -> dict[str, Any]:
    if np is None or mask is None:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
        }
    try:
        arr = np.asarray(mask) > 0
    except Exception:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
        }
    pixels = int(np.count_nonzero(arr))
    if pixels <= 0:
        return {
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
        }
    try:
        if cv2 is not None:
            component_count, labels = cv2.connectedComponents(
                arr.astype(np.uint8),
                None,
                8,
            )
            if component_count <= 1:
                largest = pixels
            else:
                sizes = np.bincount(labels.ravel())[1:]
                largest = int(sizes.max()) if sizes.size else 0
            return {
                "component_count": max(0, int(component_count) - 1),
                "largest_component_pixels": largest,
                "largest_component_ratio": round(float(largest) / float(max(1, pixels)), 4),
            }
    except Exception:
        pass
    return {
        "component_count": 1,
        "largest_component_pixels": pixels,
        "largest_component_ratio": 1.0,
    }


def _load_binary_mask_ref(path: str | None, shape: tuple[int, int] | Sequence[int]) -> Any | None:
    if np is None or Image is None:
        return None
    if not path:
        return None
    try:
        if not os.path.isfile(str(path)):
            return None
        with Image.open(str(path)) as img:
            arr = np.asarray(img.convert("L")) > 0
        expected = tuple(int(v) for v in tuple(shape)[:2])
        if arr.shape[:2] != expected:
            return None
        return arr
    except Exception:
        return None


def _write_upstream_commit_artifacts(
    *,
    artifact_dir: str | None,
    page_id: str,
    before: Any,
    cleaned: Any,
    committed_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not artifact_dir or Image is None or not committed_records:
        return {}
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        stem = f"{_safe_id(page_id)}_phase5_upstream_commit"
        committed_path = os.path.join(artifact_dir, f"{stem}_working_image.png")
        diff_path = os.path.join(artifact_dir, f"{stem}_diff.png")
        mask_path = os.path.join(artifact_dir, f"{stem}_mask.png")
        _save_image(cleaned, committed_path)
        _save_diff(before, cleaned, diff_path)
        _save_commit_diff_mask(before, cleaned, mask_path)
        return {
            "committed_image_ref": committed_path,
            "commit_diff_ref": diff_path,
            "commit_mask_ref": mask_path,
        }
    except Exception:
        return {}


def _save_commit_diff_mask(before: Any, after: Any, path: str) -> None:
    if np is None or Image is None:
        return
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
        if before_np.shape != after_np.shape:
            return
        diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
        changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
        write_debug_image_file(path, Image.fromarray(changed.astype(np.uint8) * 255))
    except Exception:
        return


def execute_cleanup_runtime_plan(
    *,
    image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    backend_context: Mapping[str, Any] | None = None,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None = None,
) -> CleanupResult:
    """Execute a Phase 5 runtime plan on a scratch image copy."""

    started = time.time()
    perf_collector = _CLEANUP_RUNTIME_PERF_COLLECTOR.get()
    perf_record: dict[str, Any] | None = None
    perf_stages: dict[str, Any] | None = None
    runtime_perf: dict[str, Any] | None = None
    perf_total_started = 0.0
    perf_setup_started = 0.0
    if isinstance(perf_collector, dict):
        perf_stages = {"runtime": {}, "local": {}, "backend": {}, "runner": {}}
        runtime_perf = perf_stages["runtime"]
        perf_record = {"stages": perf_stages}
        perf_total_started = time.perf_counter()
        perf_setup_started = perf_total_started
    params = cleanup_plan.backend_parameters or {}
    identity = _cleanup_identity_fields(cleanup_plan, cleanup_mask, params, fallback_region_id=region_id)
    canonical_region_id = _canonical_parent_region_id(cleanup_plan, cleanup_mask, params, fallback_region_id=region_id)
    debug_info: dict[str, Any] = {}
    mode = str(params.get("inpaint_mode") or cleanup_plan.inpaint_mode or "fast")
    cleanup_tag = str(params.get("cleanup_tag") or "speech_strong")
    model_required = _model_required_for_plan(cleanup_plan, cleanup_mask)
    if model_required:
        mode = "ai"
        cleanup_tag = "authorized_ai_inpaint_full_mask"
    backend_erase_mask = cleanup_mask.erase_mask
    backend_foreground_mask = cleanup_mask.foreground_mask
    backend_context_mask_ids: list[str] = []
    backend_context_used = False
    backend_context_executable_input_used = False
    backend_context_failure_reason = ""
    if isinstance(backend_context, Mapping):
        backend_context_mask_ids = [
            str(item)
            for item in (backend_context.get("cleanup_mask_ids") or [])
            if str(item)
        ]
        backend_context_used = bool(backend_context_mask_ids)
        debug_info["backend_context_mask_available"] = backend_context_used
        debug_info["backend_context_cleanup_mask_ids"] = backend_context_mask_ids
        if model_required:
            context_masks = _root_backend_context_execution_masks(
                cleanup_mask=cleanup_mask,
                backend_context=backend_context,
            )
            if context_masks is not None:
                backend_erase_mask, backend_foreground_mask = context_masks
                backend_context_executable_input_used = True
            else:
                backend_context_failure_reason = "invalid_or_unbound_root_backend_context"
        debug_info["backend_context_executable_input_used"] = bool(
            backend_context_executable_input_used
        )
        debug_info["backend_context_audit_only"] = not bool(
            backend_context_executable_input_used
        )
        debug_info["backend_context_failure_reason"] = backend_context_failure_reason
    commit_mask_source = backend_erase_mask if backend_context_executable_input_used else cleanup_mask.erase_mask
    commit_foreground_mask = (
        backend_foreground_mask if backend_context_executable_input_used else cleanup_mask.foreground_mask
    )
    commit_mask = commit_mask_source
    if runtime_perf is not None:
        _perf_add(runtime_perf, "pre_mask_setup_ms", _perf_elapsed_ms(perf_setup_started))
    perf_mask_started = time.perf_counter() if runtime_perf is not None else 0.0
    if model_required:
        model_backend_mask = _standardized_model_inpaint_support_mask(backend_erase_mask)
        if model_backend_mask is not None and np is not None and np.any(model_backend_mask > 0):
            backend_erase_mask = model_backend_mask
            debug_info["model_inpaint_mask_prepared"] = True
            debug_info["model_inpaint_mask_standardization"] = "cleanup_runtime_support_mask"
        debug_info["model_inpaint_commit_surface"] = "candidate_safe_core_plus_support"
    if runtime_perf is not None:
        _perf_add(runtime_perf, "mask_standardization_ms", _perf_elapsed_ms(perf_mask_started))
    perf_post_mask_started = time.perf_counter() if runtime_perf is not None else 0.0
    if backend_context_executable_input_used:
        debug_info["backend_context_commit_scope"] = "same_root_accepted_mask_union"
    else:
        debug_info["backend_context_commit_scope"] = "cleanup_mask"
    debug_info["model_required"] = bool(model_required)
    debug_info["requested_model_id"] = str(model_id or "")
    _page014_timeout_checkpoint(
        "cleanup_execute_runtime_plan",
        "start",
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_tag=cleanup_tag,
        inpaint_mode=mode,
        model_required=bool(model_required),
    )
    if runtime_perf is not None:
        _perf_add(runtime_perf, "post_mask_setup_ms", _perf_elapsed_ms(perf_post_mask_started))
    perf_hash_started = time.perf_counter() if runtime_perf is not None else 0.0
    input_image_hash = _hash_image(image)
    if runtime_perf is not None:
        _perf_add(runtime_perf, "input_image_hash_ms", _perf_elapsed_ms(perf_hash_started))
    perf_hash_started = time.perf_counter() if runtime_perf is not None else 0.0
    input_mask_hash = _hash_mask(backend_erase_mask)
    if runtime_perf is not None:
        _perf_add(runtime_perf, "input_mask_hash_ms", _perf_elapsed_ms(perf_hash_started))
    perf_execution_started = time.perf_counter() if runtime_perf is not None else 0.0
    execution_kwargs = {
        "model_id": model_id,
        "cleanup_tag": cleanup_tag,
        "foreground_mask": backend_foreground_mask,
        "debug_info": debug_info,
        "return_full_image": False,
    }
    if perf_stages is not None:
        execution_kwargs["perf_timings"] = perf_stages
    execution = cleanup_execution.apply_local_text_removal(
        image,
        backend_erase_mask,
        mode,
        use_gpu,
        **execution_kwargs,
    )
    if runtime_perf is not None:
        _perf_add(runtime_perf, "local_execution_ms", _perf_elapsed_ms(perf_execution_started))
    perf_result_prepare_started = time.perf_counter() if runtime_perf is not None else 0.0
    operation_bbox = _valid_bbox(execution.crop_bbox) or _valid_bbox(cleanup_mask.erase_mask_bbox)
    raw_cleaned_crop = execution.cleaned_crop
    if raw_cleaned_crop is None and execution.cleaned_image is not None and operation_bbox is not None:
        raw_cleaned_crop = _crop_payload_from_page_image(execution.cleaned_image, operation_bbox)
    result_space = "crop" if raw_cleaned_crop is not None and operation_bbox is not None else "page"
    cleaned_crop = None
    raw_cleaned_image = None
    cleaned_image = None
    if runtime_perf is not None:
        _perf_add(runtime_perf, "result_prepare_ms", _perf_elapsed_ms(perf_result_prepare_started))
    if result_space == "crop":
        if model_required:
            perf_safe_commit_started = time.perf_counter() if runtime_perf is not None else 0.0
            safe_commit_mask, safe_commit_metrics = _safe_model_inpaint_commit_mask(
                before=image,
                candidate=raw_cleaned_crop,
                crop_bbox=operation_bbox,
                core_mask=commit_mask_source,
                support_mask=backend_erase_mask,
                cleanup_class=cleanup_plan.cleanup_class,
            )
            if safe_commit_mask is not None:
                commit_mask = safe_commit_mask
                debug_info.update(safe_commit_metrics)
            if runtime_perf is not None:
                _perf_add(runtime_perf, "safe_commit_ms", _perf_elapsed_ms(perf_safe_commit_started))
        perf_clip_started = time.perf_counter() if runtime_perf is not None else 0.0
        cleaned_crop = _clip_cleaned_crop_to_authorized_mask(
            before=image,
            candidate_crop=raw_cleaned_crop,
            crop_bbox=operation_bbox,
            accepted_mask=commit_mask,
        )
        if runtime_perf is not None:
            _perf_add(runtime_perf, "authorized_clip_ms", _perf_elapsed_ms(perf_clip_started))
        perf_output_hash_started = time.perf_counter() if runtime_perf is not None else 0.0
        raw_backend_output_hash = _hash_image(raw_cleaned_crop)
        final_clipped_output_hash = _hash_image(cleaned_crop)
        if runtime_perf is not None:
            _perf_add(runtime_perf, "output_hash_ms", _perf_elapsed_ms(perf_output_hash_started))
    else:
        raw_cleaned_image = execution.cleaned_image or image
        perf_output_hash_started = time.perf_counter() if runtime_perf is not None else 0.0
        raw_backend_output_hash = _hash_image(raw_cleaned_image)
        if runtime_perf is not None:
            _perf_add(runtime_perf, "output_hash_ms", _perf_elapsed_ms(perf_output_hash_started))
        if model_required:
            perf_safe_commit_started = time.perf_counter() if runtime_perf is not None else 0.0
            safe_commit_mask, safe_commit_metrics = _safe_model_inpaint_commit_mask(
                before=image,
                candidate=raw_cleaned_image,
                crop_bbox=None,
                core_mask=commit_mask_source,
                support_mask=backend_erase_mask,
                cleanup_class=cleanup_plan.cleanup_class,
            )
            if safe_commit_mask is not None:
                commit_mask = safe_commit_mask
                debug_info.update(safe_commit_metrics)
            if runtime_perf is not None:
                _perf_add(runtime_perf, "safe_commit_ms", _perf_elapsed_ms(perf_safe_commit_started))
        perf_clip_started = time.perf_counter() if runtime_perf is not None else 0.0
        cleaned_image = _clip_cleaned_candidate_to_authorized_mask(
            before=image,
            candidate=raw_cleaned_image,
            accepted_mask=commit_mask,
        )
        if runtime_perf is not None:
            _perf_add(runtime_perf, "authorized_clip_ms", _perf_elapsed_ms(perf_clip_started))
        perf_output_hash_started = time.perf_counter() if runtime_perf is not None else 0.0
        final_clipped_output_hash = _hash_image(cleaned_image)
        if runtime_perf is not None:
            _perf_add(runtime_perf, "output_hash_ms", _perf_elapsed_ms(perf_output_hash_started))
    runtime_ms = (time.time() - started) * 1000.0
    perf_changed_started = time.perf_counter() if runtime_perf is not None else 0.0
    if result_space == "crop":
        changed_pixels = _changed_pixel_count_crop(
            before=image,
            cleaned_crop=cleaned_crop,
            crop_bbox=operation_bbox,
            accepted_mask=commit_mask,
        )
    else:
        changed_pixels = _changed_pixel_count(image, cleaned_image)
    if runtime_perf is not None:
        _perf_add(runtime_perf, "changed_pixel_count_ms", _perf_elapsed_ms(perf_changed_started))
    perf_metadata_started = time.perf_counter() if runtime_perf is not None else 0.0
    pixel_changed = changed_pixels > 0
    backend = str(execution.backend or debug_info.get("backend") or "unknown")
    backend_kind = str(execution.backend_kind or debug_info.get("backend_kind") or _backend_kind_for_name(backend))
    model_attempted = bool(execution.model_invocation_attempted or debug_info.get("model_invocation_attempted", False))
    model_succeeded = bool(execution.model_invocation_succeeded or debug_info.get("model_invocation_succeeded", False))
    if backend_kind == "model_inpaint":
        model_attempted = True
        model_succeeded = bool(model_succeeded or execution.cleaned_image is not None)
    fallback_reason = str(
        execution.backend_detail
        if backend_kind in {"backend_error", "noop"}
        else debug_info.get("fallback_reason", "")
        or ""
    )
    model_contract_failure = ""
    if model_required and backend_kind != "model_inpaint":
        model_contract_failure = "model_required_backend_not_invoked"
    elif model_required and not model_succeeded:
        model_contract_failure = "model_required_backend_failed"
    if model_contract_failure:
        execution_status = "contract_error"
        failure_reason = model_contract_failure
        pixel_changed = False
        changed_pixels = 0
    else:
        execution_status = "completed" if pixel_changed else "completed_no_pixel_change"
        failure_reason = "" if pixel_changed else "no_pixels_changed"
    raw_artifact_image = raw_cleaned_image
    cleaned_artifact_image = cleaned_image
    if runtime_perf is not None:
        _perf_add(runtime_perf, "metadata_finalize_ms", _perf_elapsed_ms(perf_metadata_started))
    perf_artifact_started = time.perf_counter() if runtime_perf is not None else 0.0
    if artifact_dir and result_space == "crop":
        raw_artifact_image = _materialize_cleaned_crop_on_page(
            before=image,
            cleaned_crop=raw_cleaned_crop,
            crop_bbox=operation_bbox,
        )
        cleaned_artifact_image = _materialize_cleaned_crop_on_page(
            before=image,
            cleaned_crop=cleaned_crop,
            crop_bbox=operation_bbox,
            accepted_mask=commit_mask,
        )
    refs = _write_runtime_artifacts(
        artifact_dir=artifact_dir,
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
        before=image,
        backend_input_mask=backend_erase_mask,
        raw_cleaned=raw_artifact_image if raw_artifact_image is not None else image,
        cleaned=cleaned_artifact_image if cleaned_artifact_image is not None else image,
        commit_mask=commit_mask,
    )
    if runtime_perf is not None:
        _perf_add(runtime_perf, "artifact_write_ms", _perf_elapsed_ms(perf_artifact_started))
    _page014_timeout_checkpoint(
        "cleanup_execute_runtime_plan",
        "end",
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        backend=backend,
        backend_kind=backend_kind,
        model_required=bool(model_required),
        model_invocation_attempted=bool(model_attempted),
        model_invocation_succeeded=bool(model_succeeded),
        pixel_changed=pixel_changed,
        changed_pixel_count=changed_pixels,
        artifact_ref_count=len([value for value in refs.values() if value]),
        elapsed_ms=round(runtime_ms, 3),
    )
    perf_result_build_started = time.perf_counter() if runtime_perf is not None else 0.0
    result = CleanupResult(
        cleanup_result_id=f"cres_{_safe_id(cleanup_plan.cleanup_plan_id)}",
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_job_id=str(cleanup_plan.cleanup_job_id),
        cleanup_mask_id=str(cleanup_plan.cleanup_mask_id),
        operation_bbox=operation_bbox,
        page=page_id,
        region_id=canonical_region_id,
        parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
        parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
        text_block_root_id=str(identity.get("text_block_root_id") or ""),
        target_region_ids=list(identity.get("target_region_ids") or []),
        cleanup_class=cleanup_plan.cleanup_class,
        pixel_changed=pixel_changed,
        changed_pixel_count=changed_pixels,
        cleaned_image_ref=refs.get("cleaned_image_ref"),
        cleaned_crop_ref=refs.get("cleaned_crop_ref"),
        diff_ref=refs.get("diff_ref"),
        mask_ref=refs.get("mask_ref"),
        backend_input_mask_ref=refs.get("backend_input_mask_ref"),
        model_input_image_ref=refs.get("model_input_image_ref"),
        model_input_mask_ref=refs.get("model_input_mask_ref"),
        raw_backend_output_ref=refs.get("raw_backend_output_ref"),
        final_clipped_output_ref=refs.get("final_clipped_output_ref"),
        renderer_consumed=False,
        execution_backend=backend,
        execution_status=execution_status,
        failure_reason=failure_reason,
        mask_stats=mask_stats(cleanup_mask.erase_mask) or {},
        backend_name=backend,
        backend_kind=backend_kind,
        backend_method=str(execution.backend_method or backend),
        requested_model_id=str(execution.requested_model_id or model_id or ""),
        actual_model_name=str(execution.actual_model_name or ""),
        actual_model_path=str(execution.actual_model_path or ""),
        model_invocation_attempted=model_attempted,
        model_invocation_succeeded=model_succeeded,
        input_image_hash=input_image_hash,
        input_mask_hash=input_mask_hash,
        raw_backend_output_hash=raw_backend_output_hash,
        final_clipped_output_hash=final_clipped_output_hash,
        result_space=result_space,
        crop_bbox=operation_bbox if result_space == "crop" else None,
        crop_width=(operation_bbox[2] - operation_bbox[0]) if result_space == "crop" and operation_bbox else None,
        crop_height=(operation_bbox[3] - operation_bbox[1]) if result_space == "crop" and operation_bbox else None,
        fallback_reason=fallback_reason,
        backend_parameters={
            "backend_detail": execution.backend_detail,
            "candidate_status": "completed" if pixel_changed else "no_pixel_change",
            "backend_kind": backend_kind,
            "backend_method": str(execution.backend_method or backend),
            "model_required": bool(model_required),
            "model_required_reason": (
                "inpaint_mode_ai_requires_model_for_accepted_normal_text"
                if model_required
                else ""
            ),
            "model_invocation_attempted": bool(model_attempted),
            "model_invocation_succeeded": bool(model_succeeded),
            "requested_model_id": str(execution.requested_model_id or model_id or ""),
            "actual_model_name": str(execution.actual_model_name or ""),
            "actual_model_path": str(execution.actual_model_path or ""),
            "input_image_hash": input_image_hash,
            "input_mask_hash": input_mask_hash,
            "raw_backend_output_hash": raw_backend_output_hash,
            "final_clipped_output_hash": final_clipped_output_hash,
            "result_space": result_space,
            "fallback_reason": fallback_reason,
            "model_backend_contract_failure": model_contract_failure,
            "text_block_root_id": str(params.get("text_block_root_id") or ""),
            "cleanup_obligation_id": str(params.get("cleanup_obligation_id") or cleanup_plan.cleanup_mask_id),
            "parent_cleanup_mask_id": str(params.get("parent_cleanup_mask_id") or ""),
            "mask_space": str(params.get("mask_space") or "page"),
            "bbox_format": str(params.get("bbox_format") or "xyxy_exclusive"),
            "cleanup_tag": cleanup_tag,
            "effective_inpaint_mode": execution.effective_inpaint_mode,
            "crop_bbox": execution.crop_bbox,
            "crop_area": execution.crop_area,
            "crop_result_contract": result_space == "crop",
            "mask_ratio": execution.mask_ratio,
            "foreground_mask_pixels": getattr(cleanup_mask, "foreground_mask_pixels", None),
            "erase_mask_pixels": getattr(cleanup_mask, "erase_mask_pixels", None),
            "owned_segmentation_pixels": getattr(cleanup_mask, "owned_segmentation_pixels", None),
            "executable_foreground_pixels": getattr(cleanup_mask, "executable_foreground_pixels", None),
            "owned_segmentation_to_executable_ratio": getattr(
                cleanup_mask,
                "owned_segmentation_to_executable_ratio",
                None,
            ),
            "ready_but_sparse_violation": bool(getattr(cleanup_mask, "ready_but_sparse_violation", False)),
            "sourceglyph_executable_influence_detected": bool(
                getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)
            ),
            "segmentation_contract_override_detected": bool(
                getattr(cleanup_mask, "segmentation_contract_override_detected", False)
            ),
            "backend_context_mask_used": bool(backend_context_used),
            "backend_context_cleanup_mask_ids": backend_context_mask_ids,
            "backend_context_executable_input_used": bool(backend_context_executable_input_used),
            "backend_context_audit_only": bool(
                backend_context_used and not backend_context_executable_input_used
            ),
            "backend_context_failure_reason": backend_context_failure_reason,
            "backend_context_clipped_to_cleanup_mask": bool(
                backend_context_executable_input_used or model_required
            ),
            "backend_context_commit_scope": str(
                debug_info.get("backend_context_commit_scope") or "cleanup_mask"
            ),
            "backend_candidate_clipped_to_cleanup_mask": True,
            "model_inpaint_mask_prepared": bool(debug_info.get("model_inpaint_mask_prepared", False)),
            "model_inpaint_mask_standardization": str(
                debug_info.get("model_inpaint_mask_standardization") or ""
            ),
            "model_inpaint_commit_surface": str(
                debug_info.get("model_inpaint_commit_surface") or ""
            ),
            "model_inpaint_commit_guard": str(
                debug_info.get("model_inpaint_commit_guard") or ""
            ),
            "model_inpaint_core_commit_pixels": debug_info.get("model_inpaint_core_commit_pixels"),
            "model_inpaint_support_pixels": debug_info.get("model_inpaint_support_pixels"),
            "model_inpaint_support_ring_pixels": debug_info.get("model_inpaint_support_ring_pixels"),
            "model_inpaint_safe_support_ring_pixels": debug_info.get("model_inpaint_safe_support_ring_pixels"),
            "model_inpaint_suppressed_support_ring_pixels": debug_info.get(
                "model_inpaint_suppressed_support_ring_pixels"
            ),
            "model_inpaint_suppressed_core_pixels": debug_info.get(
                "model_inpaint_suppressed_core_pixels"
            ),
            "model_inpaint_commit_pixels": debug_info.get("model_inpaint_commit_pixels"),
            "backend_elapsed_ms": debug_info.get("backend_elapsed_ms"),
            "model_call_elapsed_ms": debug_info.get("model_call_elapsed_ms"),
            "model_load_elapsed_ms": debug_info.get("model_load_elapsed_ms"),
            "backend_crop_width": debug_info.get("backend_crop_width"),
            "backend_crop_height": debug_info.get("backend_crop_height"),
            "backend_crop_area": debug_info.get("backend_crop_area"),
            "backend_mask_bbox": debug_info.get("backend_mask_bbox"),
        },
        runtime_ms=runtime_ms,
        fallback_status=execution_status,
        errors=list(execution.errors or []),
        commit_mask=commit_mask,
        commit_foreground_mask=commit_foreground_mask,
        cleaned_image=cleaned_image,
        cleaned_crop=cleaned_crop,
    )
    if runtime_perf is not None and perf_record is not None and isinstance(perf_collector, dict):
        _perf_add(runtime_perf, "result_build_ms", _perf_elapsed_ms(perf_result_build_started))
        runtime_perf["total_observed_ms"] = _perf_elapsed_ms(perf_total_started)
        runtime_leaf_keys = (
            "pre_mask_setup_ms",
            "mask_standardization_ms",
            "post_mask_setup_ms",
            "input_image_hash_ms",
            "input_mask_hash_ms",
            "local_execution_ms",
            "result_prepare_ms",
            "safe_commit_ms",
            "authorized_clip_ms",
            "output_hash_ms",
            "changed_pixel_count_ms",
            "metadata_finalize_ms",
            "artifact_write_ms",
            "result_build_ms",
        )
        leaf_sum_ms = sum(float(runtime_perf.get(key) or 0.0) for key in runtime_leaf_keys)
        runtime_perf["leaf_sum_ms"] = round(leaf_sum_ms, 3)
        runtime_perf["unattributed_ms"] = round(
            max(0.0, float(runtime_perf["total_observed_ms"]) - leaf_sum_ms),
            3,
        )
        perf_record.update(
            {
                "cleanup_result_id": str(result.cleanup_result_id),
                "cleanup_plan_id": str(result.cleanup_plan_id),
                "cleanup_job_id": str(result.cleanup_job_id),
                "cleanup_mask_id": str(result.cleanup_mask_id),
                "region_id": str(result.region_id or ""),
                "parent_execution_bundle_id": str(result.parent_execution_bundle_id or ""),
                "backend_name": str(result.backend_name or ""),
                "backend_kind": str(result.backend_kind or ""),
                "backend_method": str(result.backend_method or ""),
                "runtime_ms": round(float(result.runtime_ms or 0.0), 3),
            }
        )
        perf_collector.setdefault("jobs", []).append(perf_record)
    return result


def _proof(
    cleanup_result: CleanupResult,
    cleanup_plan: CleanupPlan,
    *,
    status: ProofStatus,
    failure_reason: str,
    metrics: dict[str, Any],
) -> CleanupProof:
    pass_residual_ratio = _float_or_none(metrics.get("residual_dark_source_ratio"))
    if pass_residual_ratio is None:
        pass_residual_ratio = _float_or_none(metrics.get("residual_source_ratio"))
    degraded_cleanup_diagnostic = bool(metrics.get("degraded_cleanup_diagnostic"))
    proof_quality_state = str(metrics.get("proof_quality_state") or "")
    containment_pixels = _int_or_none(metrics.get("changed_outside_accepted_mask_pixels"))
    if containment_pixels is None:
        containment_pixels = _int_or_none(metrics.get("changed_outside_allowed_pixels"))
    collateral_score = _float_or_none(metrics.get("accepted_mask_collateral_ratio"))
    if collateral_score is None:
        collateral_score = _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio"))
    render_decision = (
        "allow_stage6_consumption_candidate"
        if status == ProofStatus.PASSED
        else "block_future_renderer_consumption"
    )
    return CleanupProof(
        cleanup_proof_id=f"cproof_{_safe_id(cleanup_result.cleanup_result_id)}",
        cleanup_result_id=cleanup_result.cleanup_result_id,
        cleanup_job_id=cleanup_result.cleanup_job_id,
        cleanup_plan_id=cleanup_plan.cleanup_plan_id,
        proof_status=status,
        parent_execution_bundle_id=str(
            cleanup_result.parent_execution_bundle_id
            or cleanup_plan.parent_execution_bundle_id
            or ""
        ),
        parent_logical_text_unit_id=str(
            cleanup_result.parent_logical_text_unit_id
            or cleanup_plan.parent_logical_text_unit_id
            or ""
        ),
        text_block_root_id=str(
            cleanup_result.text_block_root_id
            or cleanup_plan.text_block_root_id
            or ""
        ),
        target_region_ids=list(cleanup_result.target_region_ids or cleanup_plan.target_region_ids or []),
        source_glyph_removal_passed=status == ProofStatus.PASSED,
        source_residual_ratio=pass_residual_ratio,
        changed_outside_allowed_pixels=_int_or_none(metrics.get("changed_outside_allowed_pixels")),
        collateral_damage_score=collateral_score,
        mask_containment_ratio=_float_or_none(metrics.get("mask_source_overlap_ratio")),
        render_consumption_decision_if_consumed=render_decision,
        residual_source_text_risk=(
            "low"
            if status == ProofStatus.PASSED
            else ("diagnostic_residual_not_committable" if degraded_cleanup_diagnostic else "visible_or_unknown")
        ),
        mask_containment_passed=(
            containment_pixels <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            if containment_pixels is not None
            else None
        ),
        collateral_damage_risk=(
            "low"
            if status == ProofStatus.PASSED
            else "possible"
        ),
        backend_failure_visible=failure_reason == "backend_noop_or_error",
        failure_reason=failure_reason,
        metrics={**metrics, "proof_quality_state": proof_quality_state or metrics.get("proof_quality_state", "")},
    )


def _select_best_passed_result(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    passed = [
        proof
        for proof in proof_records
        if proof.proof_status == ProofStatus.PASSED and proof.cleanup_result_id in by_result_id
    ]
    if not passed:
        return None, None
    passed.sort(key=lambda proof: _proof_selection_key(proof, by_result_id[proof.cleanup_result_id]))
    proof = passed[0]
    return by_result_id[proof.cleanup_result_id], proof


def _select_best_commit_candidate_result(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    candidates = [
        proof
        for proof in proof_records
        if proof.cleanup_result_id in by_result_id
        and _proof_commit_candidate(by_result_id[proof.cleanup_result_id], proof)
    ]
    if not candidates:
        return None, None
    candidates.sort(
        key=lambda proof: (
            0 if proof.proof_status == ProofStatus.PASSED else 1,
            *_proof_selection_key(proof, by_result_id[proof.cleanup_result_id]),
        )
    )
    proof = candidates[0]
    return by_result_id[proof.cleanup_result_id], proof


def _best_failed_attempt(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    available = [proof for proof in proof_records if proof.cleanup_result_id in by_result_id]
    if not available:
        return (result_records[0], None) if result_records else (None, None)
    available.sort(key=lambda proof: _proof_selection_key(proof, by_result_id[proof.cleanup_result_id]))
    proof = available[0]
    return by_result_id[proof.cleanup_result_id], proof


def _proof_selection_key(proof: CleanupProof, result: CleanupResult) -> tuple[float, ...]:
    metrics = proof.metrics or {}
    candidate_status = str((result.backend_parameters or {}).get("candidate_status") or "")
    failure_reason = str(proof.failure_reason or result.failure_reason or "")
    pixel_changed = bool(result.pixel_changed and int(result.changed_pixel_count or 0) > 0)
    noop_rank = 1 if (
        not pixel_changed
        or candidate_status in {"no_pixel_change", "completed_no_pixel_change"}
        or failure_reason == "backend_noop_or_error"
    ) else 0
    status_rank = {
        ProofStatus.PASSED: 0,
        ProofStatus.FAILED: 1,
        ProofStatus.INCONCLUSIVE: 2,
    }.get(proof.proof_status, 3)
    residual = _first_metric_float(
        metrics,
        "executable_text_residual_ratio",
        "residual_dark_source_ratio",
        "residual_source_ratio",
        default=999.0,
    )
    residual_pixels = _first_metric_float(
        metrics,
        "executable_text_residual_pixels",
        "residual_dark_source_pixels",
        "residual_source_pixels",
        default=999_999.0,
    )
    collateral = _first_metric_float(
        metrics,
        "changed_outside_erase_inside_allowed_ratio",
        default=999.0,
    )
    changed_rank = -float(max(0, int(result.changed_pixel_count or 0)))
    runtime = float(result.runtime_ms if result.runtime_ms is not None else 999_999.0)
    return (
        float(noop_rank),
        float(status_rank),
        residual,
        residual_pixels,
        collateral,
        changed_rank,
        runtime,
    )


def _first_metric_float(metrics: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return float(value)
    return float(default)


def _plan_for_result(
    plans: Sequence[CleanupPlan],
    cleanup_result: CleanupResult | None,
) -> CleanupPlan | None:
    if cleanup_result is None:
        return None
    for plan in plans:
        if plan.cleanup_plan_id == cleanup_result.cleanup_plan_id:
            return plan
    return None


def _runtime_base_record(
    page_id: str,
    job: CleanupJob,
    *,
    runtime_class: CleanupClass,
) -> dict[str, Any]:
    target_region_ids = [str(rid) for rid in job.target_region_ids or [] if str(rid)]
    cleanup_unit_id = str(getattr(job, "cleanup_unit_id", "") or job.cleanup_job_id)
    identity = _cleanup_identity_fields(job, fallback_region_id=target_region_ids[0] if target_region_ids else "")
    parent_execution_bundle_id = str(identity.get("parent_execution_bundle_id") or "")
    parent_logical_text_unit_id = str(identity.get("parent_logical_text_unit_id") or "")
    text_block_root_id = str(
        getattr(job, "text_block_root_id", "")
        or identity.get("text_block_root_id")
        or parent_logical_text_unit_id
        or cleanup_unit_id
        or job.cleanup_job_id
        or ""
    )
    return {
        "page_id": page_id,
        "region_id": parent_execution_bundle_id or parent_logical_text_unit_id or (target_region_ids[0] if target_region_ids else ""),
        "parent_execution_bundle_id": parent_execution_bundle_id,
        "parent_logical_text_unit_id": parent_logical_text_unit_id,
        "target_region_ids": list(identity.get("target_region_ids") or target_region_ids),
        "cleanup_job_id": str(job.cleanup_job_id),
        "text_block_root_id": text_block_root_id,
        "mask_space": "page",
        "bbox_format": "xyxy_exclusive",
        "cleanup_unit_id": cleanup_unit_id,
        "cleanup_unit_level": str(getattr(job, "cleanup_unit_level", "") or ""),
        "cleanup_unit_anchor_region_id": str(
            getattr(job, "cleanup_unit_anchor_region_id", "") or ""
        ),
        "cleanup_unit_child_region_ids": list(
            getattr(job, "cleanup_unit_child_region_ids", []) or []
        ),
        "cleanup_unit_required_source_glyph_mask_ids": list(
            getattr(job, "cleanup_unit_required_source_glyph_mask_ids", []) or []
        ),
        "cleanup_unit_missing_source_glyph_mask_ids": list(
            getattr(job, "cleanup_unit_missing_source_glyph_mask_ids", []) or []
        ),
        "cleanup_class": _enum_value(job.cleanup_class),
        "runtime_class": _enum_value(runtime_class),
        "renderer_consumed": False,
    }


def _runtime_cleanup_class_for_job(job: CleanupJob, requested_class: CleanupClass | None) -> CleanupClass | None:
    job_class_value = _enum_value(job.cleanup_class)
    if requested_class is not None and job_class_value != _enum_value(requested_class):
        return None
    if job_class_value in RUNTIME_UNSAFE_CLEANUP_CLASSES:
        return None
    if job_class_value not in RUNTIME_SUPPORTED_CLEANUP_CLASSES:
        return None
    return job.cleanup_class


def _runtime_job_exclusion_reason(job: CleanupJob, cleanup_mask: CleanupMask | None = None) -> str:
    if bool(job.protected):
        return job.protection_reason or "job_protected"
    cleanup_only = bool(getattr(job, "cleanup_only_obligation", False))
    if cleanup_only:
        return "cleanup_only_obligation_not_runtime_authority"
    source_evidence_reason = _runtime_source_evidence_exclusion_reason(job, cleanup_mask)
    if source_evidence_reason:
        return source_evidence_reason
    combined = " ".join(
        str(value or "")
        for value in (
            _enum_value(job.cleanup_class),
            job.route_intent,
            job.semantic_class,
            job.cleanup_mode,
            job.container_type,
            " ".join(job.source_glyph_mask_ids or []),
        )
    ).lower()
    for blocked in (
        "preserve",
        "sfx",
        "decorative",
        "non_text",
        "non_translation_art",
        "art_only",
        "source_grounding_protected",
        "source_ungrounded",
    ):
        if blocked in combined:
            return f"phase5_runtime_excludes_{blocked}"
    if not job.source_text_present and not cleanup_only:
        return "source_text_missing"
    if not job.translated_text_present and not cleanup_only:
        return "translated_text_missing"
    return ""


def _runtime_source_evidence_exclusion_reason(
    job: CleanupJob,
    cleanup_mask: CleanupMask | None = None,
) -> str:
    if cleanup_mask is not None:
        consumed_ids = _unique_texts(getattr(cleanup_mask, "consumed_source_glyph_mask_ids", []) or [])
        missing_ids = _unique_texts(getattr(cleanup_mask, "missing_source_glyph_mask_ids", []) or [])
        foreground_pixels = int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0)
        has_foreground_array = False
        if np is not None and getattr(cleanup_mask, "foreground_mask", None) is not None:
            try:
                has_foreground_array = bool(np.any(np.asarray(cleanup_mask.foreground_mask) > 0))
            except Exception:
                has_foreground_array = False
        has_source_ids = bool(consumed_ids or getattr(cleanup_mask, "foreground_mask_source_id", None))
        if missing_ids:
            return "phase5_runtime_requires_consumable_sourceglyph_evidence"
        if _is_component_projected_cleanup_mask(cleanup_mask) and (foreground_pixels > 0 or has_foreground_array):
            return ""
        if has_source_ids and (foreground_pixels > 0 or has_foreground_array):
            return ""
    if not job.source_glyph_evidence or not job.source_glyph_mask_ids:
        return "phase5_runtime_requires_sourceglyph_evidence"
    eligible_records = 0
    for item in job.source_glyph_evidence or []:
        if not isinstance(item, Mapping):
            continue
        if _truthy(_first_source_value(item, "source_glyph_mask_review_only")):
            continue
        if _truthy(_first_source_value(item, "preserve", "is_sfx", "is_decorative", "is_art")):
            continue
        if _source_value_false(item, "cleanup_covers_source_glyphs"):
            continue
        coverage = _float_or_none(_first_source_value(item, "source_glyph_erasure_coverage_ratio"))
        if coverage is not None and coverage < PROOF_MASK_SOURCE_OVERLAP_RATIO:
            continue
        if not _valid_bbox(
            _first_source_value(
                item,
                "source_glyph_erasure_bbox",
                "source_glyph_bbox",
                "bbox",
            )
        ):
            continue
        eligible_records += 1
    if eligible_records <= 0:
        return "phase5_runtime_requires_consumable_sourceglyph_evidence"
    return ""


def _job_source_contract_values(job: CleanupJob, key: str) -> set[str]:
    values: set[str] = set()
    for item in job.source_glyph_evidence or []:
        if not isinstance(item, Mapping):
            continue
        value = item.get(key)
        if value is None and isinstance(item.get("compatibility_source_fields"), Mapping):
            value = item.get("compatibility_source_fields", {}).get(key)
        text = str(value or "").strip()
        if text:
            values.add(text)
    return values


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _effective_mask_kwargs(cleanup_mask: CleanupMask) -> dict[str, Any]:
    return {
        "parent_execution_bundle_id": str(
            getattr(cleanup_mask, "parent_execution_bundle_id", "")
            or getattr(cleanup_mask, "parent_logical_text_unit_id", "")
            or ""
        ),
        "parent_logical_text_unit_id": str(getattr(cleanup_mask, "parent_logical_text_unit_id", "") or ""),
        "text_block_root_id": str(getattr(cleanup_mask, "text_block_root_id", "") or ""),
        "text_area_container_id": str(getattr(cleanup_mask, "text_area_container_id", "") or ""),
        "target_region_ids": list(getattr(cleanup_mask, "target_region_ids", []) or []),
        "represented_region_ids": list(getattr(cleanup_mask, "represented_region_ids", []) or []),
        "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
        "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
        "seed_foreground_pixels": getattr(cleanup_mask, "seed_foreground_pixels", None),
        "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
        "component_count_before": getattr(cleanup_mask, "component_count_before", None),
        "component_count_after": getattr(cleanup_mask, "component_count_after", None),
        "largest_component_pixels_before": getattr(cleanup_mask, "largest_component_pixels_before", None),
        "largest_component_pixels_after": getattr(cleanup_mask, "largest_component_pixels_after", None),
        "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        "bbox_fill_ratio_before": getattr(cleanup_mask, "bbox_fill_ratio_before", None),
        "bbox_fill_ratio_after": getattr(cleanup_mask, "bbox_fill_ratio_after", None),
        "analysis_scope_bbox": list(getattr(cleanup_mask, "analysis_scope_bbox", None) or []),
        "executable_erase_bbox": list(getattr(cleanup_mask, "executable_erase_bbox", None) or []),
        "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
        "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
        "source_seed_mask_ids": list(getattr(cleanup_mask, "source_seed_mask_ids", []) or []),
        "recovered_component_count": getattr(cleanup_mask, "recovered_component_count", None),
        "rejected_component_count": getattr(cleanup_mask, "rejected_component_count", None),
        "rejected_component_reasons": list(getattr(cleanup_mask, "rejected_component_reasons", []) or []),
        "segmentation_mask_status": str(getattr(cleanup_mask, "segmentation_mask_status", "") or ""),
        "segmentation_mask_failure_reason": str(getattr(cleanup_mask, "segmentation_mask_failure_reason", "") or ""),
        "segmentation_provider": str(getattr(cleanup_mask, "segmentation_provider", "") or ""),
        "segmentation_mask_ref": str(getattr(cleanup_mask, "segmentation_mask_ref", "") or ""),
        "segmentation_text_pixels": getattr(cleanup_mask, "segmentation_text_pixels", None),
        "segmentation_component_count": getattr(cleanup_mask, "segmentation_component_count", None),
        "segmentation_binding_method": str(getattr(cleanup_mask, "segmentation_binding_method", "") or ""),
        "segmentation_block_associations": list(getattr(cleanup_mask, "segmentation_block_associations", []) or []),
        "ownership_binding_status": str(getattr(cleanup_mask, "ownership_binding_status", "") or ""),
        "ownership_binding_method": str(getattr(cleanup_mask, "ownership_binding_method", "") or ""),
        "cleanup_owned_unit_bbox": list(getattr(cleanup_mask, "cleanup_owned_unit_bbox", None) or []),
        "cleanup_owned_unit_mask_ref": str(getattr(cleanup_mask, "cleanup_owned_unit_mask_ref", "") or ""),
        "protected_mask_ref": str(getattr(cleanup_mask, "protected_mask_ref", "") or ""),
        "protected_overlap_pixels": getattr(cleanup_mask, "protected_overlap_pixels", None),
        "segmentation_pixels_before_binding": getattr(cleanup_mask, "segmentation_pixels_before_binding", None),
        "segmentation_pixels_after_owner_clip": getattr(cleanup_mask, "segmentation_pixels_after_owner_clip", None),
        "segmentation_pixels_after_protection_subtract": getattr(
            cleanup_mask,
            "segmentation_pixels_after_protection_subtract",
            None,
        ),
        "sourceglyph_overlap_pixels": getattr(cleanup_mask, "sourceglyph_overlap_pixels", None),
        "sourceglyph_overlap_ratio": getattr(cleanup_mask, "sourceglyph_overlap_ratio", None),
        "segmentation_outside_sourceglyph_pixels": getattr(cleanup_mask, "segmentation_outside_sourceglyph_pixels", None),
        "effective_coverage_ratio": getattr(cleanup_mask, "effective_coverage_ratio", None),
        "effective_coverage_status": str(getattr(cleanup_mask, "effective_coverage_status", "") or ""),
        "component_ownership_status": str(getattr(cleanup_mask, "component_ownership_status", "") or ""),
        "owned_component_ids": list(getattr(cleanup_mask, "owned_component_ids", []) or []),
        "protected_component_ids": list(getattr(cleanup_mask, "protected_component_ids", []) or []),
        "ambiguous_component_ids": list(getattr(cleanup_mask, "ambiguous_component_ids", []) or []),
        "unowned_component_ids": list(getattr(cleanup_mask, "unowned_component_ids", []) or []),
        "component_projection_method": str(getattr(cleanup_mask, "component_projection_method", "") or ""),
        "owned_component_pixel_count": getattr(cleanup_mask, "owned_component_pixel_count", None),
        "protected_component_pixel_count": getattr(cleanup_mask, "protected_component_pixel_count", None),
        "ambiguous_component_pixel_count": getattr(cleanup_mask, "ambiguous_component_pixel_count", None),
        "sourceglyph_overlap_component_ids": list(getattr(cleanup_mask, "sourceglyph_overlap_component_ids", []) or []),
        "sourceglyph_missing_component_ids": list(getattr(cleanup_mask, "sourceglyph_missing_component_ids", []) or []),
        "ownership_projection_failure_reason": str(getattr(cleanup_mask, "ownership_projection_failure_reason", "") or ""),
        "effective_component_coverage_ratio": getattr(cleanup_mask, "effective_component_coverage_ratio", None),
        "owned_segmentation_pixels": getattr(cleanup_mask, "owned_segmentation_pixels", None),
        "executable_foreground_pixels": getattr(cleanup_mask, "executable_foreground_pixels", None),
        "committed_cleanup_mask_pixels": getattr(cleanup_mask, "committed_cleanup_mask_pixels", None),
        "owned_segmentation_to_executable_ratio": getattr(cleanup_mask, "owned_segmentation_to_executable_ratio", None),
        "owned_segmentation_to_commit_ratio": getattr(cleanup_mask, "owned_segmentation_to_commit_ratio", None),
        "ready_but_sparse_violation": bool(getattr(cleanup_mask, "ready_but_sparse_violation", False)),
        "sourceglyph_executable_influence_detected": bool(
            getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)
        ),
        "segmentation_contract_override_detected": bool(
            getattr(cleanup_mask, "segmentation_contract_override_detected", False)
        ),
    }


def _first_source_value(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) not in (None, ""):
            return item.get(key)
        compat = item.get("compatibility_source_fields")
        if isinstance(compat, Mapping) and key in compat and compat.get(key) not in (None, ""):
            return compat.get(key)
    return None


def _source_value_false(item: Mapping[str, Any], key: str) -> bool:
    value = _first_source_value(item, key)
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no"}
    return value is False


def _runtime_speech_flat_mask(cleanup_mask: CleanupMask, source_image: Any) -> CleanupMask:
    """Compatibility wrapper for older callers."""

    return _runtime_glyph_precision_mask(cleanup_mask, source_image, None)


def _backend_contexts_by_cleanup_mask_id(
    runtime_obligations: Sequence[tuple[CleanupJob, CleanupMask, CleanupPlan]],
    image: Any,
) -> dict[str, dict[str, Any]]:
    if np is None:
        return {}
    try:
        shape = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image).shape[:2]
    except Exception:
        return {}
    groups: dict[str, list[tuple[CleanupJob, CleanupMask, CleanupPlan]]] = {}
    for job, cleanup_mask, cleanup_plan in runtime_obligations or []:
        root_id = _explicit_runtime_root_id(job, cleanup_mask, cleanup_plan)
        if not root_id:
            continue
        if not _cleanup_plan_can_use_root_backend_context(cleanup_plan, cleanup_mask):
            continue
        groups.setdefault(root_id, []).append((job, cleanup_mask, cleanup_plan))

    contexts: dict[str, dict[str, Any]] = {}
    for root_id, entries in groups.items():
        accepted_entries = [
            entry
            for entry in entries
            if _accepted_executable_cleanup_mask(entry[1])
        ]
        if len(accepted_entries) <= 1:
            continue
        union_erase = np.zeros(shape, dtype=np.uint8)
        union_foreground = np.zeros(shape, dtype=np.uint8)
        cleanup_mask_ids: list[str] = []
        for _, cleanup_mask, _ in accepted_entries:
            erase = _binary_mask(cleanup_mask.erase_mask, shape)
            foreground = _binary_mask(cleanup_mask.foreground_mask, shape)
            if erase is None or not np.any(erase):
                continue
            union_erase = np.maximum(union_erase, erase.astype(np.uint8))
            if foreground is not None and np.any(foreground):
                union_foreground = np.maximum(union_foreground, foreground.astype(np.uint8))
            cleanup_mask_ids.append(str(cleanup_mask.cleanup_mask_id))
        if len(cleanup_mask_ids) <= 1 or not np.any(union_erase):
            continue
        if not np.any(union_foreground):
            union_foreground = union_erase.copy()
        context = {
            "text_block_root_id": root_id,
            "erase_mask": union_erase,
            "foreground_mask": union_foreground,
            "cleanup_mask_ids": cleanup_mask_ids,
            "context_source": "same_text_block_root_accepted_cleanup_masks",
        }
        for cleanup_mask_id in cleanup_mask_ids:
            contexts[cleanup_mask_id] = context
    return contexts


def _explicit_runtime_root_id(
    job: CleanupJob,
    cleanup_mask: CleanupMask,
    cleanup_plan: CleanupPlan,
) -> str:
    params = cleanup_plan.backend_parameters or {}
    root_id = str(
        params.get("text_block_root_id")
        or getattr(job, "text_block_root_id", "")
        or ""
    ).strip()
    if not root_id:
        return ""
    fallback_ids = {
        str(getattr(job, "cleanup_job_id", "") or ""),
        str(getattr(job, "cleanup_unit_id", "") or ""),
        str(getattr(job, "parent_logical_text_unit_id", "") or ""),
        str(getattr(cleanup_mask, "cleanup_mask_id", "") or ""),
    }
    if root_id in fallback_ids:
        return ""
    return root_id


def _cleanup_plan_can_use_root_backend_context(
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
) -> bool:
    cleanup_class = _enum_value(cleanup_plan.cleanup_class)
    semantic_state = str(getattr(cleanup_mask, "semantic_authorization_state", "") or "").strip()
    cleanup_auth = str(getattr(cleanup_mask, "cleanup_authorization", "") or "").strip()
    cleanup_tag = str((cleanup_plan.backend_parameters or {}).get("cleanup_tag") or "").strip().lower()
    if semantic_state in {"cleanup_translate_background", "cleanup_translate_caption"}:
        return True
    if cleanup_auth in {"cleanup_translate_background", "cleanup_translate_caption"}:
        return True
    if cleanup_class in {
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }:
        return True
    return cleanup_tag in {"background_narration_inpaint_first", "side_caption_glyph_local"}


def _valid_backend_context_mask(candidate: Any, accepted_mask: Any) -> bool:
    if np is None or candidate is None or accepted_mask is None:
        return False
    try:
        candidate_arr = np.asarray(candidate)
        accepted_arr = np.asarray(accepted_mask)
    except Exception:
        return False
    return (
        candidate_arr.ndim == 2
        and accepted_arr.ndim == 2
        and candidate_arr.shape[:2] == accepted_arr.shape[:2]
        and bool(np.any(candidate_arr > 0))
    )


def _root_backend_context_execution_masks(
    *,
    cleanup_mask: CleanupMask,
    backend_context: Mapping[str, Any],
) -> tuple[Any, Any] | None:
    """Return exact same-root accepted-mask union for model input.

    The union is executable only when it is assembled from accepted cleanup
    masks that belong to the same text-block root. Model proof and commit use
    the same union surface so the backend cannot clear a complete root and then
    lose sibling text during child-mask clipping.
    """

    if np is None or not isinstance(backend_context, Mapping):
        return None
    cleanup_mask_id = str(getattr(cleanup_mask, "cleanup_mask_id", "") or "")
    context_mask_ids = {
        str(item)
        for item in (backend_context.get("cleanup_mask_ids") or [])
        if str(item)
    }
    if not cleanup_mask_id or cleanup_mask_id not in context_mask_ids:
        return None
    accepted = np.asarray(cleanup_mask.erase_mask) if cleanup_mask.erase_mask is not None else None
    if accepted is None or accepted.ndim != 2:
        return None
    shape = tuple(int(v) for v in accepted.shape[:2])
    context_erase = _binary_mask(backend_context.get("erase_mask"), shape)
    if context_erase is None or not np.any(context_erase > 0):
        return None
    accepted_binary = (accepted > 0).astype(np.uint8)
    if np.any((accepted_binary > 0) & ~(context_erase > 0)):
        return None
    context_foreground = _binary_mask(backend_context.get("foreground_mask"), shape)
    if context_foreground is None or not np.any(context_foreground > 0):
        context_foreground = context_erase.copy()
    return (
        (context_erase > 0).astype(np.uint8) * 255,
        (context_foreground > 0).astype(np.uint8) * 255,
    )


def _standardized_model_inpaint_support_mask(mask: Any) -> Any | None:
    """Return the explicit model inpaint support mask used for proof/commit.

    This mirrors the historical LaMa support dilation, but exposes it before
    backend execution so clipping and proof use the same executable surface.
    """

    if np is None or mask is None:
        return None
    try:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = np.any(arr > 0, axis=2)
        elif arr.ndim != 2:
            return None
        binary = (arr > 0).astype(np.uint8) * 255
    except Exception:
        return None
    if not np.any(binary > 0):
        return binary
    kernel_size = max(5, int(max(binary.shape) * 0.005))
    if cv2 is None:
        pad = kernel_size
        ys, xs = np.where(binary > 0)
        if xs.size <= 0 or ys.size <= 0:
            return binary
        output = binary.copy()
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(binary.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(binary.shape[0], int(ys.max()) + pad + 1)
        output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], binary[y0:y1, x0:x1])
        return output
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(binary, kernel, iterations=2)


def _safe_model_inpaint_commit_mask(
    *,
    before: Any,
    candidate: Any,
    crop_bbox: Sequence[int] | None,
    core_mask: Any,
    support_mask: Any,
    cleanup_class: CleanupClass | str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Build the page-space model commit surface from candidate pixels.

    The model may receive a wider support mask for context, but runtime commit
    must not blindly write that whole support region back. Core accepted erase
    pixels normally commit. For speech-bubble cleanup, bright core padding and
    support-ring pixels are withheld when the model output visibly darkens them.
    For non-speech cleanup, support-ring pixels are still committed on textured
    caption/background surfaces so bright text outlines can be erased; the guard
    only applies when the source support area is itself a flat bright surface.
    """

    metrics: dict[str, Any] = {
        "model_inpaint_commit_surface": "candidate_safe_core_plus_support",
        "model_inpaint_commit_guard": "suppress_darkened_bright_support_ring",
    }
    if np is None:
        return None, metrics
    before_np = _image_to_rgb_array(before)
    candidate_np = _image_to_rgb_array(candidate)
    if before_np is None or candidate_np is None:
        return None, metrics
    page_shape = tuple(int(v) for v in before_np.shape[:2])
    core = _binary_mask(core_mask, page_shape)
    support = _binary_mask(support_mask, page_shape)
    if core is None or not np.any(core > 0):
        return None, metrics
    if support is None or not np.any(support > 0):
        support = core.copy()

    candidate_page = candidate_np
    box = _valid_bbox(crop_bbox)
    if candidate_np.shape[:2] != before_np.shape[:2]:
        if box is None:
            return None, metrics
        x0, y0, x1, y1 = box
        if tuple(candidate_np.shape[:2]) != (y1 - y0, x1 - x0):
            return None, metrics
        candidate_page = before_np.copy()
        candidate_page[y0:y1, x0:x1] = candidate_np
    elif candidate_np.shape != before_np.shape:
        return None, metrics

    core_bool = core > 0
    support_bool = support > 0
    support_ring = support_bool & ~core_bool
    before_gray = _gray(before_np)
    candidate_gray = _gray(candidate_page)
    cleanup_class_value = _enum_value(cleanup_class)
    speech_bubble_core_guard = cleanup_class_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
    }
    bright_source = before_gray >= 220
    darkened_candidate = (candidate_gray + 8 < before_gray) & (candidate_gray < 230)
    support_values = before_gray[support_bool]
    if support_values.size:
        support_bright_ratio = float(np.count_nonzero(support_values >= 220)) / float(support_values.size)
        support_dark_ratio = float(np.count_nonzero(support_values <= 96)) / float(support_values.size)
    else:
        support_bright_ratio = 0.0
        support_dark_ratio = 0.0
    flat_bright_support_context = bool(
        support_bright_ratio >= 0.82 and support_dark_ratio <= 0.04
    )
    support_ring_guard = bool(speech_bubble_core_guard or flat_bright_support_context)
    unsafe_ring = (
        support_ring & bright_source & darkened_candidate
        if support_ring_guard
        else np.zeros(page_shape, dtype=bool)
    )
    safe_ring = support_ring & ~unsafe_ring
    unsafe_core = (
        core_bool & bright_source & darkened_candidate
        if speech_bubble_core_guard
        else np.zeros(page_shape, dtype=bool)
    )

    commit = np.zeros(page_shape, dtype=np.uint8)
    commit[core_bool & ~unsafe_core] = 255
    commit[safe_ring] = 255
    metrics.update(
        {
            "model_inpaint_core_commit_pixels": int(np.count_nonzero(core_bool)),
            "model_inpaint_support_pixels": int(np.count_nonzero(support_bool)),
            "model_inpaint_support_ring_pixels": int(np.count_nonzero(support_ring)),
            "model_inpaint_safe_support_ring_pixels": int(np.count_nonzero(safe_ring)),
            "model_inpaint_suppressed_support_ring_pixels": int(np.count_nonzero(unsafe_ring)),
            "model_inpaint_suppressed_core_pixels": int(np.count_nonzero(unsafe_core)),
            "model_inpaint_commit_pixels": int(np.count_nonzero(commit)),
            "model_inpaint_support_bright_ratio": round(support_bright_ratio, 6),
            "model_inpaint_support_dark_ratio": round(support_dark_ratio, 6),
            "model_inpaint_support_ring_guard": bool(support_ring_guard),
        }
    )
    return commit, metrics


def _model_required_for_plan(cleanup_plan: CleanupPlan, cleanup_mask: CleanupMask) -> bool:
    mode = str(cleanup_plan.inpaint_mode or (cleanup_plan.backend_parameters or {}).get("inpaint_mode") or "").lower()
    if mode != "ai":
        return False
    if _formal_plan_allows_non_model_method(cleanup_plan):
        return False
    if not _normal_text_cleanup_obligation(cleanup_plan, cleanup_mask):
        return False
    return True


def _formal_plan_allows_non_model_method(cleanup_plan: CleanupPlan) -> bool:
    params = cleanup_plan.backend_parameters or {}
    allowed = bool(
        params.get("allow_non_model_method")
        or params.get("non_model_method_allowed")
    )
    reason = str(
        params.get("non_model_method_allowed_reason")
        or params.get("allow_non_model_method_reason")
        or ""
    ).strip()
    return bool(allowed and reason)


def _normal_text_cleanup_obligation(cleanup_plan: CleanupPlan, cleanup_mask: CleanupMask) -> bool:
    if bool(getattr(cleanup_mask, "protected", False)):
        return False
    semantic_state = str(getattr(cleanup_mask, "semantic_authorization_state", "") or "")
    cleanup_authorization = str(getattr(cleanup_mask, "cleanup_authorization", "") or "")
    if semantic_state in {
        "cleanup_translate_speech",
        "cleanup_translate_background",
        "cleanup_translate_caption",
    }:
        return True
    if cleanup_authorization in {
        "cleanup_translate_speech",
        "cleanup_translate_background",
        "cleanup_translate_caption",
    }:
        return True
    return _enum_value(cleanup_plan.cleanup_class) in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }


def _backend_kind_for_name(backend: object) -> str:
    name = str(backend or "").strip().lower()
    if not name or name == "none":
        return "noop"
    if "backend_error" in name or "error" in name:
        return "backend_error"
    if "cleanup_ai_inpaint" in name or "simple_lama" in name or "model_inpaint" in name:
        return "model_inpaint"
    if "cv2" in name or "opencv" in name or "conservative_local_inpaint" in name:
        return "opencv_inpaint"
    if "residual" in name or "repair" in name:
        return "residual_repair"
    if any(token in name for token in ("fill", "white", "median", "local", "directional", "bubble")):
        return "heuristic_fill"
    return "heuristic_fill"


def _hash_image(image: Any) -> str:
    if np is None or image is None:
        return ""
    try:
        arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    except Exception:
        return ""
    return _hash_array(arr)


def _hash_mask(mask: Any) -> str:
    if np is None or mask is None:
        return ""
    try:
        arr = (np.asarray(mask) > 0).astype(np.uint8)
    except Exception:
        return ""
    return _hash_array(arr)


def _cleanup_result_crop_bbox(cleanup_result: CleanupResult) -> list[int] | None:
    return (
        _valid_bbox(getattr(cleanup_result, "crop_bbox", None))
        or _valid_bbox(getattr(cleanup_result, "operation_bbox", None))
    )


def _image_to_rgb_array(image: Any) -> Any | None:
    if np is None or image is None:
        return None
    try:
        return np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    except Exception:
        return None


def _crop_payload_from_page_image(page_image: Any, crop_bbox: Sequence[int] | None) -> Any | None:
    if Image is None:
        return None
    box = _valid_bbox(crop_bbox)
    if box is None or page_image is None:
        return None
    try:
        if hasattr(page_image, "crop"):
            return page_image.crop(tuple(box))
        arr = _image_to_rgb_array(page_image)
        if arr is None:
            return None
        x0, y0, x1, y1 = box
        return Image.fromarray(np.asarray(arr[y0:y1, x0:x1]).astype("uint8"), mode="RGB")
    except Exception:
        return None


def _cleaned_crop_payload_for_result(cleanup_result: CleanupResult) -> Any | None:
    crop = getattr(cleanup_result, "cleaned_crop", None)
    if crop is not None:
        return crop
    return _load_cleanup_image(getattr(cleanup_result, "cleaned_crop_ref", None))


def _crop_shape_matches(cleaned_crop: Any, crop_bbox: Sequence[int] | None) -> bool:
    box = _valid_bbox(crop_bbox)
    arr = _image_to_rgb_array(cleaned_crop)
    if box is None or arr is None:
        return False
    x0, y0, x1, y1 = box
    return tuple(arr.shape[:2]) == (int(y1 - y0), int(x1 - x0))


def _clip_cleaned_crop_to_authorized_mask(
    *,
    before: Any,
    candidate_crop: Any,
    crop_bbox: Sequence[int],
    accepted_mask: Any,
) -> Any:
    if np is None or Image is None or accepted_mask is None:
        return candidate_crop
    box = _valid_bbox(crop_bbox)
    if box is None:
        return candidate_crop
    try:
        x0, y0, x1, y1 = box
        before_np = _image_to_rgb_array(before)
        candidate_np = _image_to_rgb_array(candidate_crop)
        accepted = np.asarray(accepted_mask) > 0
        if (
            before_np is None
            or candidate_np is None
            or accepted.shape[:2] != before_np.shape[:2]
            or tuple(candidate_np.shape[:2]) != (y1 - y0, x1 - x0)
        ):
            return candidate_crop
        clipped = before_np[y0:y1, x0:x1].copy()
        accepted_crop = accepted[y0:y1, x0:x1]
        clipped[accepted_crop] = candidate_np[accepted_crop]
        return Image.fromarray(np.clip(clipped, 0, 255).astype(np.uint8), mode="RGB")
    except Exception:
        return candidate_crop


def _materialize_cleaned_crop_on_page(
    *,
    before: Any,
    cleaned_crop: Any,
    crop_bbox: Sequence[int] | None,
    accepted_mask: Any | None = None,
) -> Any | None:
    if np is None or Image is None:
        return None
    box = _valid_bbox(crop_bbox)
    before_np = _image_to_rgb_array(before)
    crop_np = _image_to_rgb_array(cleaned_crop)
    if box is None or before_np is None or crop_np is None:
        return None
    x0, y0, x1, y1 = box
    if tuple(crop_np.shape[:2]) != (y1 - y0, x1 - x0):
        return None
    page_np = before_np.copy()
    if accepted_mask is None:
        page_np[y0:y1, x0:x1] = crop_np
    else:
        try:
            accepted = np.asarray(accepted_mask) > 0
            if accepted.shape[:2] != before_np.shape[:2]:
                return None
            accepted_crop = accepted[y0:y1, x0:x1]
            target = page_np[y0:y1, x0:x1]
            target[accepted_crop] = crop_np[accepted_crop]
            page_np[y0:y1, x0:x1] = target
        except Exception:
            return None
    return Image.fromarray(np.clip(page_np, 0, 255).astype(np.uint8), mode="RGB")


def _cleanup_result_page_payload(cleanup_result: CleanupResult, before_image: Any) -> Any | None:
    cleaned_image = cleanup_result.cleaned_image
    if cleaned_image is None:
        cleaned_image = _load_cleanup_image(cleanup_result.cleaned_image_ref)
    if cleaned_image is not None:
        return cleaned_image
    cleaned_crop = _cleaned_crop_payload_for_result(cleanup_result)
    if cleaned_crop is None:
        return None
    return _materialize_cleaned_crop_on_page(
        before=before_image,
        cleaned_crop=cleaned_crop,
        crop_bbox=_cleanup_result_crop_bbox(cleanup_result),
        accepted_mask=getattr(cleanup_result, "commit_mask", None),
    )


def _changed_pixel_count_crop(
    *,
    before: Any,
    cleaned_crop: Any,
    crop_bbox: Sequence[int] | None,
    accepted_mask: Any | None = None,
) -> int:
    if np is None:
        return 0
    box = _valid_bbox(crop_bbox)
    before_np = _image_to_rgb_array(before)
    crop_np = _image_to_rgb_array(cleaned_crop)
    if box is None or before_np is None or crop_np is None:
        return 0
    x0, y0, x1, y1 = box
    if tuple(crop_np.shape[:2]) != (y1 - y0, x1 - x0):
        return 0
    before_crop = before_np[y0:y1, x0:x1]
    diff = np.abs(crop_np.astype(np.int16) - before_crop.astype(np.int16))
    changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
    if accepted_mask is not None:
        try:
            accepted = np.asarray(accepted_mask) > 0
            if accepted.shape[:2] == before_np.shape[:2]:
                changed = changed & accepted[y0:y1, x0:x1]
        except Exception:
            pass
    return int(np.count_nonzero(changed))


def _page_bbox_to_crop_bbox(
    bbox: Sequence[int] | None,
    crop_bbox: Sequence[int] | None,
) -> list[int] | None:
    box = _valid_bbox(bbox)
    crop = _valid_bbox(crop_bbox)
    if box is None or crop is None:
        return None
    x0 = max(box[0], crop[0])
    y0 = max(box[1], crop[1])
    x1 = min(box[2], crop[2])
    y1 = min(box[3], crop[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 - crop[0], y0 - crop[1], x1 - crop[0], y1 - crop[1]]


def _hash_array(arr: Any) -> str:
    if np is None:
        return ""
    try:
        array = np.ascontiguousarray(arr)
        digest = hashlib.sha256()
        digest.update(str(tuple(array.shape)).encode("ascii", errors="ignore"))
        digest.update(str(array.dtype).encode("ascii", errors="ignore"))
        digest.update(array.tobytes())
        return digest.hexdigest()
    except Exception:
        return ""


def _clip_cleaned_candidate_to_authorized_mask(
    *,
    before: Any,
    candidate: Any,
    accepted_mask: Any,
) -> Any:
    if np is None or Image is None or accepted_mask is None:
        return candidate
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        candidate_np = np.asarray(candidate.convert("RGB") if hasattr(candidate, "convert") else candidate)
        accepted = np.asarray(accepted_mask) > 0
    except Exception:
        return candidate
    if before_np.shape != candidate_np.shape or accepted.shape[:2] != before_np.shape[:2]:
        return candidate
    clipped = before_np.copy()
    clipped[accepted] = candidate_np[accepted]
    return Image.fromarray(np.clip(clipped, 0, 255).astype(np.uint8))


def _runtime_glyph_precision_mask(
    cleanup_mask: CleanupMask,
    source_image: Any,
    cleanup_class: CleanupClass | None,
) -> CleanupMask:
    """Build runtime geometry from the completed CleanupMask foreground.

    Accepted cleanup masks are already page-space obligations. Runtime may use
    allowed/container boxes for audit and dilation safety, but it must not crop
    accepted foreground or erase pixels back to those boxes.
    """

    if np is None:
        return cleanup_mask
    runtime_mask_source = (
        "text_foreground_segmentation_runtime_precision"
        if str(getattr(cleanup_mask, "mask_source", "") or "") == "cleanup_mask_from_text_foreground_segmentation"
        else "non_segmentation_diagnostic_runtime_precision"
    )
    effective_status = str(getattr(cleanup_mask, "effective_mask_status", "") or "")
    if effective_status.startswith("effective_mask_failed_"):
        return _runtime_rejected_mask(
            cleanup_mask,
            reason=effective_status,
            method_suffix=effective_status,
            foreground=cleanup_mask.foreground_mask,
            erase=cleanup_mask.erase_mask,
        )
    accepted = _accepted_executable_cleanup_mask(cleanup_mask)
    allowed = _valid_bbox(cleanup_mask.allowed_area)
    foreground = _binary_cleanup_mask_array(cleanup_mask.foreground_mask)
    if foreground is None:
        foreground = _binary_cleanup_mask_array_from_shape(cleanup_mask.foreground_mask_bbox, source_image)
        if foreground is not None:
            foreground[:] = 0
    if allowed is None or foreground is None:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="segmentation_foreground_contract_defect",
            method_suffix="missing_foreground_or_allowed_area",
        )
    if not accepted:
        foreground = _clip_mask_to_valid_bbox(foreground, allowed)
    defect = ""
    if not accepted:
        # Geometry guards are diagnostic for non-accepted masks; accepted
        # component-projected masks are the cleanup obligation.
        defect = _glyph_foreground_geometry_defect_reason(
            foreground=foreground,
            allowed=allowed,
            cleanup_class=cleanup_class,
        )
    if defect:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="segmentation_foreground_contract_defect",
            method_suffix=defect,
            foreground=foreground,
        )
    erase = _binary_cleanup_mask_array(cleanup_mask.erase_mask)
    if erase is None or not np.any(erase > 0):
        erase = _glyph_local_erase_from_foreground(foreground, allowed, cleanup_class)
    elif not accepted:
        erase = _clip_mask_to_valid_bbox(erase, allowed)
    else:
        erase = np.maximum((erase > 0).astype(np.uint8), (foreground > 0).astype(np.uint8))
    foreground_pixels = int(np.count_nonzero(foreground))
    erase_pixels = int(np.count_nonzero(erase))
    foreground_bbox = _mask_bbox(foreground)
    erase_bbox = _mask_bbox(erase)
    if foreground_pixels <= 0 or erase_pixels <= 0 or foreground_bbox is None or erase_bbox is None:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="segmentation_foreground_contract_defect",
            method_suffix="empty_glyph_component_after_clip",
            foreground=foreground,
            erase=erase,
        )
    runtime = CleanupMask(
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_job_id=str(cleanup_mask.cleanup_job_id),
        foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
        foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
        consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
        foreground_mask_bbox=foreground_bbox,
        foreground_mask_pixels=foreground_pixels,
        erase_mask_bbox=erase_bbox,
        erase_mask_pixels=erase_pixels,
        allowed_area=allowed,
        growth_ratio=_growth_ratio(erase_pixels, foreground_pixels),
        mask_source=runtime_mask_source,
        mask_method="phase5_phase6_runtime_segmentation_foreground_glyph_precision",
        rejection_reason="",
        mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
        artifact_risk=cleanup_mask.artifact_risk,
        visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
        protected=cleanup_mask.protected,
        protection_reason=cleanup_mask.protection_reason,
        **_effective_mask_kwargs(cleanup_mask),
        foreground_mask=foreground,
        erase_mask=erase,
    )
    if _mask_exclusion_reason(runtime, None) in {
        "growth_ratio_too_high",
        "erase_mask_too_large",
        "erase_bbox_too_broad",
        "erase_allowed_ratio_too_high",
    }:
        foreground_only = CleanupMask(
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            cleanup_job_id=str(cleanup_mask.cleanup_job_id),
            foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
            foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
            consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
            missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
            foreground_mask_bbox=foreground_bbox,
            foreground_mask_pixels=foreground_pixels,
            erase_mask_bbox=foreground_bbox,
            erase_mask_pixels=foreground_pixels,
            allowed_area=allowed,
            growth_ratio=1.0,
            mask_source=runtime_mask_source,
            mask_method="phase5_phase6_runtime_segmentation_foreground_minimal_erasure",
            rejection_reason="",
            mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
            artifact_risk=cleanup_mask.artifact_risk,
            visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
            protected=cleanup_mask.protected,
            protection_reason=cleanup_mask.protection_reason,
            **_effective_mask_kwargs(cleanup_mask),
            foreground_mask=foreground,
            erase_mask=foreground.copy(),
        )
        if not _mask_exclusion_reason(foreground_only, None):
            return foreground_only
    return runtime


def _binary_cleanup_mask_array(value: Any) -> Any | None:
    if np is None or value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    if arr.size <= 0:
        return None
    return (arr > 0).astype(np.uint8)


def _binary_cleanup_mask_array_from_shape(bbox: Sequence[int] | None, source_image: Any) -> Any | None:
    if np is None:
        return None
    try:
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        if source_np.ndim < 2:
            return None
        return np.zeros(source_np.shape[:2], dtype=np.uint8)
    except Exception:
        box = _valid_bbox(bbox)
        if box is None:
            return None
        return np.zeros((max(1, box[3]), max(1, box[2])), dtype=np.uint8)


def _runtime_rejected_mask(
    cleanup_mask: CleanupMask,
    *,
    reason: str,
    method_suffix: str,
    foreground: Any | None = None,
    erase: Any | None = None,
) -> CleanupMask:
    foreground_arr = foreground if foreground is not None else cleanup_mask.foreground_mask
    erase_arr = erase if erase is not None else cleanup_mask.erase_mask
    runtime_mask_source = (
        "text_foreground_segmentation_runtime_precision"
        if str(getattr(cleanup_mask, "mask_source", "") or "") == "cleanup_mask_from_text_foreground_segmentation"
        else "non_segmentation_diagnostic_runtime_precision"
    )
    return CleanupMask(
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_job_id=str(cleanup_mask.cleanup_job_id),
        foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
        foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
        consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
        foreground_mask_bbox=_mask_bbox(foreground_arr) or cleanup_mask.foreground_mask_bbox,
        foreground_mask_pixels=int(np.count_nonzero(foreground_arr)) if np is not None and foreground_arr is not None else cleanup_mask.foreground_mask_pixels,
        erase_mask_bbox=_mask_bbox(erase_arr) or cleanup_mask.erase_mask_bbox,
        erase_mask_pixels=int(np.count_nonzero(erase_arr)) if np is not None and erase_arr is not None else cleanup_mask.erase_mask_pixels,
        allowed_area=_valid_bbox(cleanup_mask.allowed_area),
        growth_ratio=cleanup_mask.growth_ratio,
        mask_source=runtime_mask_source,
        mask_method=f"phase5_phase6_runtime_segmentation_foreground_rejected:{method_suffix}",
        rejection_reason=reason,
        mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
        artifact_risk=cleanup_mask.artifact_risk,
        visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
        protected=cleanup_mask.protected,
        protection_reason=cleanup_mask.protection_reason,
        **_effective_mask_kwargs(cleanup_mask),
        foreground_mask=foreground_arr,
        erase_mask=erase_arr,
    )


def _glyph_foreground_geometry_defect_reason(
    *,
    foreground: Any,
    allowed: Sequence[int],
    cleanup_class: CleanupClass | None,
) -> str:
    pixels = int(np.count_nonzero(foreground > 0))
    if pixels <= 0:
        return "empty_segmentation_foreground"
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "empty_segmentation_foreground_bbox"
    bbox_area = _bbox_area(bbox)
    allowed_area = _bbox_area(allowed)
    density = float(pixels) / max(1, bbox_area)
    allowed_ratio = float(pixels) / max(1, allowed_area)
    cleanup_value = _enum_value(cleanup_class)
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if pixels < 4:
        return "segmentation_foreground_too_small"
    if density > 0.72 and bbox_area > 1600:
        return "segmentation_foreground_rectangular_fill"
    if bbox_area > (80_000 if speech_like else 45_000) and density > (0.22 if speech_like else 0.14):
        return "segmentation_foreground_broad_background_chunk"
    if allowed_area > 0 and allowed_ratio > (0.55 if speech_like else 0.35) and pixels > 2400:
        return "segmentation_foreground_mostly_allowed_background"
    return ""


def _glyph_local_erase_from_foreground(
    foreground: Any,
    allowed: Sequence[int],
    cleanup_class: CleanupClass | None,
) -> Any:
    foreground_u8 = (foreground > 0).astype(np.uint8)
    if cv2 is None:
        return np.maximum(foreground_u8, _clip_mask_to_valid_bbox(foreground_u8, allowed))
    cleanup_value = _enum_value(cleanup_class)
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    kernel = np.ones((3, 3), dtype=np.uint8)
    iterations = 2 if speech_like and int(np.count_nonzero(foreground > 0)) < 1200 else 1
    erase = cv2.dilate(foreground_u8, kernel, iterations=iterations)
    # Allowed-area constrains only the dilation halo. The accepted text
    # foreground itself remains executable even when upstream boxes are stale.
    return np.maximum(foreground_u8, _clip_mask_to_valid_bbox(erase, allowed))


def _runtime_speech_flat_plan(
    *,
    page_id: str,
    job: CleanupJob,
    cleanup_mask: CleanupMask,
    inpaint_mode: str,
    cleanup_class: CleanupClass,
) -> CleanupPlan:
    cleanup_plan_id = f"cplan_phase5_runtime_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}"
    cleanup_class_value = _enum_value(cleanup_class)
    cleanup_tag = _runtime_cleanup_tag_for_class(cleanup_class)
    source_bbox = _valid_bbox(job.source_glyph_erasure_bbox) or _valid_bbox(cleanup_mask.erase_mask_bbox)
    expected_bbox = _valid_bbox(job.source_glyph_erasure_expected_area_bbox)
    proof_scope_bbox = _union_valid_bboxes(source_bbox, expected_bbox)
    parent_cleanup_unit_id = str(getattr(job, "cleanup_unit_id", "") or job.cleanup_job_id or "")
    identity = _cleanup_identity_fields(job, cleanup_mask, fallback_region_id=parent_cleanup_unit_id)
    text_block_root_id = str(
        getattr(job, "text_block_root_id", "")
        or identity.get("text_block_root_id")
        or getattr(job, "parent_logical_text_unit_id", "")
        or parent_cleanup_unit_id
        or job.cleanup_job_id
        or ""
    )
    cleanup_obligation_id = str(cleanup_mask.cleanup_mask_id)
    backend_parameters = {
        "text_block_root_id": text_block_root_id,
        "cleanup_obligation_id": cleanup_obligation_id,
        "mask_space": "page",
        "bbox_format": "xyxy_exclusive",
        "foreground_mask_pixels": getattr(cleanup_mask, "foreground_mask_pixels", None),
        "erase_mask_pixels": getattr(cleanup_mask, "erase_mask_pixels", None),
        "cleanup_tag": cleanup_tag,
        "parent_execution_bundle_id": identity.get("parent_execution_bundle_id", ""),
        "parent_logical_text_unit_id": identity.get("parent_logical_text_unit_id", ""),
        "target_region_ids": list(identity.get("target_region_ids") or job.target_region_ids or []),
        "allowed_area": list(cleanup_mask.allowed_area or []),
        "source_glyph_erasure_bbox": list(source_bbox or []),
        "source_glyph_erasure_expected_area_bbox": list(expected_bbox or []),
        "proof_scope": "cleanup_unit_full",
        "proof_scope_bbox": list(proof_scope_bbox or []),
        "parent_cleanup_unit_id": parent_cleanup_unit_id,
        "parent_expected_erasure_bbox": list(expected_bbox or []),
        "represented_source_glyph_ids": list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        "consumed_source_glyph_mask_ids": list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        "missing_source_glyph_mask_ids": list(cleanup_mask.missing_source_glyph_mask_ids or []),
        "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
        "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
        "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
        "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
        "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
        "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        "renderer_consumed_by_phase": "phase5_runtime_contract_false",
        "eligibility_reason": "phase5_general_sourceglyph_allowed_area_contract",
        "effectiveness_closeout_reason": "phase5_runtime_source_residual_backend_repair",
    }
    return CleanupPlan(
        cleanup_plan_id=cleanup_plan_id,
        cleanup_job_id=str(job.cleanup_job_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_class=cleanup_class,
        selected_backend="app.pipeline.cleanup_execution.apply_local_text_removal",
        cleanup_method=f"{cleanup_class_value}_local_text_removal",
        parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
        parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
        text_block_root_id=text_block_root_id,
        target_region_ids=list(identity.get("target_region_ids") or []),
        backend_parameters=backend_parameters,
        inpaint_mode=str(inpaint_mode or "fast"),
        crop_context_bbox=list(cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area or []),
        fallback_policy=["record_failure_no_renderer_consumption"],
        expected_runtime_class="cleanup_owned_phase5_runtime",
        proof_thresholds={
            "mask_source_overlap_ratio_min": PROOF_MASK_SOURCE_OVERLAP_RATIO,
            "source_residual_ratio_max": PROOF_RESIDUAL_RATIO,
            "source_residual_pixel_fraction_max": PROOF_RESIDUAL_PIXEL_FRACTION,
            "changed_outside_allowed_pixels_max": PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS,
            "changed_outside_allowed_ratio_max": PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO,
            "changed_outside_erase_inside_allowed_ratio_max": PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO,
        },
    )


def _runtime_plan_from_formal_plan(
    *,
    formal_plan: CleanupPlan,
    page_id: str,
    job: CleanupJob,
    cleanup_mask: CleanupMask,
    inpaint_mode: str,
    cleanup_class: CleanupClass,
) -> CleanupPlan:
    """Derive runtime geometry from a formal CleanupPlan without adding eligibility."""

    runtime_plan = _runtime_speech_flat_plan(
        page_id=page_id,
        job=job,
        cleanup_mask=cleanup_mask,
        inpaint_mode=inpaint_mode,
        cleanup_class=cleanup_class,
    )
    params = dict(runtime_plan.backend_parameters or {})
    formal_params = dict(formal_plan.backend_parameters or {})
    params.update(
        {
            key: value
            for key, value in formal_params.items()
            if key
            not in {
                "cleanup_tag",
                "target_region_ids",
                "allowed_area",
                "source_glyph_erasure_bbox",
                "source_glyph_erasure_expected_area_bbox",
                "proof_scope",
                "proof_scope_bbox",
                "parent_cleanup_unit_id",
                "parent_expected_erasure_bbox",
                "represented_source_glyph_ids",
                "consumed_source_glyph_mask_ids",
                "missing_source_glyph_mask_ids",
            }
        }
    )
    params.update(
        {
            "formal_cleanup_plan_id": str(formal_plan.cleanup_plan_id),
            "formal_cleanup_mask_id": str(formal_plan.cleanup_mask_id),
            "cleanup_plan_authority": "formal_cleanup_plan_build_result",
            "runtime_plan_derived_from_formal_plan": True,
        }
    )
    return CleanupPlan(
        cleanup_plan_id=f"{formal_plan.cleanup_plan_id}_runtime",
        cleanup_job_id=str(formal_plan.cleanup_job_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_class=cleanup_class,
        selected_backend=runtime_plan.selected_backend,
        cleanup_method=f"{runtime_plan.cleanup_method}_formal_plan_authorized",
        parent_execution_bundle_id=str(
            formal_plan.parent_execution_bundle_id
            or runtime_plan.parent_execution_bundle_id
            or getattr(cleanup_mask, "parent_execution_bundle_id", "")
            or ""
        ),
        parent_logical_text_unit_id=str(
            formal_plan.parent_logical_text_unit_id
            or runtime_plan.parent_logical_text_unit_id
            or getattr(cleanup_mask, "parent_logical_text_unit_id", "")
            or ""
        ),
        text_block_root_id=str(
            formal_plan.text_block_root_id
            or runtime_plan.text_block_root_id
            or getattr(cleanup_mask, "text_block_root_id", "")
            or ""
        ),
        target_region_ids=list(
            formal_plan.target_region_ids
            or runtime_plan.target_region_ids
            or getattr(cleanup_mask, "target_region_ids", [])
            or []
        ),
        backend_parameters=params,
        inpaint_mode=str(inpaint_mode or formal_plan.inpaint_mode or runtime_plan.inpaint_mode or "fast"),
        crop_context_bbox=list(cleanup_mask.erase_mask_bbox or formal_plan.crop_context_bbox or []),
        fallback_policy=list(formal_plan.fallback_policy or runtime_plan.fallback_policy or []),
        expected_runtime_class="formal_cleanup_plan_runtime",
        proof_thresholds=dict(formal_plan.proof_thresholds or runtime_plan.proof_thresholds or {}),
    )


def _runtime_cleanup_tag_for_class(cleanup_class: CleanupClass) -> str:
    cleanup_class_value = _enum_value(cleanup_class)
    if cleanup_class_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }:
        return "speech_glyph_local"
    if cleanup_class_value in {
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }:
        return "side_caption_glyph_local"
    if cleanup_class_value in {
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
    }:
        return "background_narration_inpaint_first"
    return "minimal_erasure"


def _is_partitionable_mask_rejection(reason: str) -> bool:
    reason_text = str(reason or "").lower()
    if reason_text.startswith("mask_artifact_risk_"):
        return True
    return reason_text in {
        ARTIFACT_RISK_PARTITION_REQUIRED_REASON,
        "erase_mask_too_large",
        "erase_bbox_too_broad",
        "erase_allowed_ratio_too_high",
    }


def _partition_runtime_cleanup_mask(
    cleanup_mask: CleanupMask,
    *,
    rejection_reason: str,
    max_components: int,
) -> list[CleanupMask]:
    if np is None or cv2 is None or not _is_partitionable_mask_rejection(rejection_reason):
        return []
    erase = cleanup_mask.erase_mask
    foreground_mask = cleanup_mask.foreground_mask
    if foreground_mask is None:
        return []
    foreground_arr = np.asarray(foreground_mask)
    if foreground_arr.ndim == 3:
        foreground_arr = np.any(foreground_arr > 0, axis=2).astype(np.uint8)
    if foreground_arr.ndim != 2 or not np.any(foreground_arr > 0):
        return []
    allowed = _valid_bbox(cleanup_mask.allowed_area)
    if allowed is None:
        return []
    foreground_arr = _clip_mask_to_valid_bbox((foreground_arr > 0).astype(np.uint8), allowed)
    if not np.any(foreground_arr > 0):
        return []
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (foreground_arr > 0).astype(np.uint8),
        connectivity=8,
    )
    component_masks: list[Any] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_masks.append((area, (labels == label).astype(np.uint8)))
    component_masks.sort(key=lambda item: item[0], reverse=True)

    candidates: list[CleanupMask] = []
    for _area, component in component_masks:
        for local in _bounded_component_slices(component):
            foreground = _clip_mask_to_valid_bbox(local, allowed)
            if not np.any(foreground > 0):
                continue
            geometry_defect = _glyph_foreground_geometry_defect_reason(
                foreground=foreground,
                allowed=allowed,
                cleanup_class=None,
            )
            if geometry_defect:
                continue
            local_erase = _glyph_local_erase_from_foreground(foreground, allowed, None)
            erase_pixels = int(np.count_nonzero(local_erase))
            foreground_pixels = int(np.count_nonzero(foreground))
            erase_bbox = _mask_bbox(local_erase)
            foreground_bbox = _mask_bbox(foreground)
            if erase_bbox is None or foreground_bbox is None:
                continue
            partition_index = len(candidates)
            child_artifact_risk = ""
            child_mask_method = f"{cleanup_mask.mask_method}|bounded_connected_component_partition"
            if _is_artifact_risk_partition_reason(rejection_reason):
                child_mask_method = (
                    f"{child_mask_method}|artifact_risk_bounded_component:"
                    f"{_safe_reason_token(cleanup_mask.artifact_risk)}"
                )
            submask = CleanupMask(
                cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_part{partition_index:02d}",
                cleanup_job_id=str(cleanup_mask.cleanup_job_id),
                foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
                foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
                consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
                missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
                foreground_mask_bbox=foreground_bbox,
                foreground_mask_pixels=foreground_pixels,
                erase_mask_bbox=erase_bbox,
                erase_mask_pixels=erase_pixels,
                allowed_area=allowed,
                growth_ratio=_growth_ratio(erase_pixels, foreground_pixels),
                mask_source=f"{cleanup_mask.mask_source}|partitioned_cleanup_component",
                mask_method=child_mask_method,
                rejection_reason="",
                mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
                artifact_risk=child_artifact_risk,
                visual_scope=_partition_visual_scope(cleanup_mask.visual_scope),
                protected=cleanup_mask.protected,
                protection_reason=cleanup_mask.protection_reason,
                **_effective_mask_kwargs(cleanup_mask),
                foreground_mask=foreground,
                erase_mask=local_erase.astype(np.uint8),
            )
            if _mask_exclusion_reason(submask, None):
                minimal = CleanupMask(
                    cleanup_mask_id=submask.cleanup_mask_id,
                    cleanup_job_id=submask.cleanup_job_id,
                    foreground_mask_source_id=submask.foreground_mask_source_id,
                    foreground_mask_source_ids=list(submask.foreground_mask_source_ids or []),
                    consumed_source_glyph_mask_ids=list(submask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_mask_ids=list(submask.missing_source_glyph_mask_ids or []),
                    foreground_mask_bbox=foreground_bbox,
                    foreground_mask_pixels=foreground_pixels,
                    erase_mask_bbox=foreground_bbox,
                    erase_mask_pixels=foreground_pixels,
                    allowed_area=allowed,
                    growth_ratio=1.0,
                    mask_source=submask.mask_source,
                    mask_method=f"{child_mask_method}|minimal_foreground_erasure",
                    rejection_reason="",
                    mask_contract_exception_reason=submask.mask_contract_exception_reason,
                    artifact_risk=child_artifact_risk,
                    visual_scope=submask.visual_scope,
                    protected=submask.protected,
                    protection_reason=submask.protection_reason,
                    **_effective_mask_kwargs(submask),
                    foreground_mask=foreground,
                    erase_mask=foreground.copy(),
                )
                if _mask_exclusion_reason(minimal, None):
                    continue
                submask = minimal
            if _mask_exclusion_reason(submask, None):
                continue
            candidates.append(submask)
            if len(candidates) >= max(1, int(max_components)):
                return candidates
    return candidates


def _bounded_component_slices(component: Any) -> list[Any]:
    if np is None:
        return []
    bbox = _mask_bbox(component)
    if bbox is None:
        return []
    if (
        int(np.count_nonzero(component > 0)) <= MAX_ERASE_MASK_PIXELS
        and _bbox_area(bbox) <= MAX_ERASE_BBOX_AREA
    ):
        return [component]
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    slices: list[Any] = []
    # Keep partitioning deterministic and bounded.  The long axis is split
    # first, but every slice remains clipped to the original component mask.
    split_count = 3
    if height >= width:
        step = max(1, int((height + split_count - 1) / split_count))
        for index in range(split_count):
            sy0 = y0 + index * step
            sy1 = min(y1, sy0 + step)
            if sy1 <= sy0:
                continue
            local = np.zeros_like(component, dtype=np.uint8)
            local[sy0:sy1, x0:x1] = component[sy0:sy1, x0:x1]
            if np.any(local > 0):
                slices.append(local)
    else:
        step = max(1, int((width + split_count - 1) / split_count))
        for index in range(split_count):
            sx0 = x0 + index * step
            sx1 = min(x1, sx0 + step)
            if sx1 <= sx0:
                continue
            local = np.zeros_like(component, dtype=np.uint8)
            local[y0:y1, sx0:sx1] = component[y0:y1, sx0:sx1]
            if np.any(local > 0):
                slices.append(local)
    return slices


def _run_partitioned_cleanup_attempts(
    *,
    image: Any,
    source_image: Any,
    formal_plan: CleanupPlan,
    job: CleanupJob,
    parent_cleanup_mask: CleanupMask,
    inpaint_mode: str,
    cleanup_class: CleanupClass,
    partition_masks: Sequence[CleanupMask],
    base_record: Mapping[str, Any],
    target_region_ids: Sequence[str],
    parent_rejection_reason: str,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
) -> tuple[list[CleanupResult], list[CleanupProof], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[CleanupResult] = []
    proofs: list[CleanupProof] = []
    statuses: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    selected_child_records: list[tuple[CleanupResult, CleanupProof, Mapping[str, Any]]] = []
    child_result_count = 0
    child_proof_count = 0
    partition_count = len(partition_masks)
    base_params = dict(formal_plan.backend_parameters or {})
    region_id = _canonical_parent_region_id(formal_plan, parent_cleanup_mask, base_record, fallback_region_id=region_id) or region_id
    parent_cleanup_unit_id = str(
        base_params.get("parent_cleanup_unit_id")
        or getattr(job, "cleanup_unit_id", "")
        or getattr(job, "cleanup_job_id", "")
        or ""
    )
    parent_cleanup_mask_id = str(
        base_params.get("cleanup_obligation_id")
        or base_params.get("formal_cleanup_mask_id")
        or formal_plan.cleanup_mask_id
        or parent_cleanup_mask.cleanup_mask_id
        or ""
    )
    text_block_root_id = str(
        base_params.get("text_block_root_id")
        or getattr(job, "text_block_root_id", "")
        or parent_cleanup_unit_id
        or ""
    )
    parent_full_plan = _runtime_plan_from_formal_plan(
        formal_plan=formal_plan,
        page_id=page_id,
        job=job,
        inpaint_mode=inpaint_mode,
        cleanup_class=cleanup_class,
        cleanup_mask=parent_cleanup_mask,
    )
    parent_full_plan = _adaptive_plan_variant(
        parent_full_plan,
        attempt_index=0,
        strategy="partition_parent_full_mask_first",
        cleanup_tag=str(
            (parent_full_plan.backend_parameters or {}).get("cleanup_tag")
            or "cleanup_effectiveness_retry"
        ),
        retry_reason=parent_rejection_reason,
        suffix="_partition_parent_full_mask_first",
    )
    parent_full_params = dict(parent_full_plan.backend_parameters or {})
    parent_full_params.update(
        {
            "partition_parent_full_mask_first_attempt": True,
            "partition_strategy_required": True,
            "partition_count": partition_count,
            "partition_parent_rejection_reason": parent_rejection_reason,
            "cleanup_obligation_id": parent_cleanup_mask_id,
            "parent_cleanup_mask_id": parent_cleanup_mask_id,
            "text_block_root_id": text_block_root_id,
            "mask_space": "page",
            "bbox_format": "xyxy_exclusive",
        }
    )
    parent_full_plan = CleanupPlan(
        cleanup_plan_id=parent_full_plan.cleanup_plan_id,
        cleanup_job_id=parent_full_plan.cleanup_job_id,
        cleanup_mask_id=parent_full_plan.cleanup_mask_id,
        cleanup_class=parent_full_plan.cleanup_class,
        selected_backend=parent_full_plan.selected_backend,
        cleanup_method=f"{parent_full_plan.cleanup_method}_partition_parent_full_mask_first",
        parent_execution_bundle_id=parent_full_plan.parent_execution_bundle_id,
        parent_logical_text_unit_id=parent_full_plan.parent_logical_text_unit_id,
        text_block_root_id=parent_full_plan.text_block_root_id,
        target_region_ids=list(parent_full_plan.target_region_ids or []),
        backend_parameters=parent_full_params,
        inpaint_mode=parent_full_plan.inpaint_mode,
        crop_context_bbox=list(parent_cleanup_mask.erase_mask_bbox or parent_full_plan.crop_context_bbox or []),
        fallback_policy=list(parent_full_plan.fallback_policy or []),
        expected_runtime_class=parent_full_plan.expected_runtime_class,
        proof_thresholds=dict(parent_full_plan.proof_thresholds or {}),
    )
    parent_full_results, parent_full_proofs, parent_full_attempts = _run_adaptive_cleanup_attempts(
        image=image,
        source_image=source_image,
        cleanup_plan=parent_full_plan,
        cleanup_mask=parent_cleanup_mask,
        page_id=page_id,
        region_id=region_id,
        use_gpu=use_gpu,
        model_id=model_id,
        artifact_dir=artifact_dir,
    )
    parent_full_result, parent_full_proof = _select_best_commit_candidate_result(
        parent_full_results,
        parent_full_proofs,
    )
    if parent_full_result is not None and parent_full_proof is not None:
        proof_metrics = _json_safe(parent_full_proof.metrics or {})
        parent_status = {
            **dict(base_record),
            "runtime_status": "passed",
            "cleanup_plan_id": parent_full_result.cleanup_plan_id,
            "cleanup_mask_id": str(parent_cleanup_mask.cleanup_mask_id),
            "cleanup_obligation_id": parent_cleanup_mask_id,
            "parent_cleanup_mask_id": parent_cleanup_mask_id,
            "text_block_root_id": text_block_root_id,
            "cleanup_result_id": parent_full_result.cleanup_result_id,
            "cleanup_proof_id": parent_full_proof.cleanup_proof_id,
            "proof_status": _enum_value(parent_full_proof.proof_status),
            "failure_reason": "passed",
            "cleanup_outcome_state": "cleanup_completed",
            "pixel_changed": bool(parent_full_result.pixel_changed),
            "changed_pixel_count": int(parent_full_result.changed_pixel_count or 0),
            "cleaned_image_ref": parent_full_result.cleaned_image_ref,
            "diff_ref": parent_full_result.diff_ref,
            "mask_ref": parent_full_result.mask_ref,
            "proof_metrics": proof_metrics,
            "proof_quality_state": proof_metrics.get("proof_quality_state", ""),
            "degraded_commit_allowed": False,
            "partitioned_cleanup": False,
            "partition_parent_full_mask_first_attempt": True,
            "partition_parent_full_mask_proof_complete": True,
            "partition_count": partition_count,
            "partition_backend_result_count": 0,
            "partition_backend_proof_count": 0,
            "partition_parent_rejection_reason": parent_rejection_reason,
            "parent_cleanup_unit_id": parent_cleanup_unit_id,
            "parent_cleanup_unit_aggregate_status": "parent_full_mask_proof_complete",
            "parent_cleanup_unit_aggregate_complete": True,
            "parent_cleanup_unit_component_attempt_count": partition_count,
            "adaptive_attempt_count": len(parent_full_attempts),
            "adaptive_attempts": _json_safe(parent_full_attempts),
            "target_region_ids": list(target_region_ids),
            "render_consumption_decision_if_consumed": (
                parent_full_proof.render_consumption_decision_if_consumed
            ),
            "renderer_consumed": False,
        }
        return [parent_full_result], [parent_full_proof], [parent_status], aggregate_records
    parent_full_failed_result, parent_full_failed_proof = _best_failed_attempt(
        parent_full_results,
        parent_full_proofs,
    )
    for index, partition_mask in enumerate(partition_masks):
        plan = _runtime_plan_from_formal_plan(
            formal_plan=formal_plan,
            page_id=page_id,
            job=job,
            inpaint_mode=inpaint_mode,
            cleanup_class=cleanup_class,
            cleanup_mask=partition_mask,
        )
        plan = _adaptive_plan_variant(
            plan,
            attempt_index=index,
            strategy="partition_component",
            cleanup_tag=str((plan.backend_parameters or {}).get("cleanup_tag") or "cleanup_effectiveness_retry"),
            retry_reason=parent_rejection_reason,
            suffix=f"_part{index:02d}",
        )
        params = dict(plan.backend_parameters or {})
        params["partitioned_cleanup"] = True
        params["partition_index"] = index
        params["partition_count"] = partition_count
        params["partition_parent_rejection_reason"] = parent_rejection_reason
        params["partition_parent_artifact_risk"] = (
            parent_rejection_reason if _is_artifact_risk_partition_reason(parent_rejection_reason) else ""
        )
        parent_expected = (
            _valid_bbox(params.get("parent_expected_erasure_bbox"))
            or _valid_bbox(params.get("source_glyph_erasure_expected_area_bbox"))
            or _valid_bbox(getattr(job, "source_glyph_erasure_expected_area_bbox", None))
        )
        partition_bbox = _valid_bbox(partition_mask.erase_mask_bbox)
        parent_cleanup_unit_id = str(
            params.get("parent_cleanup_unit_id")
            or getattr(job, "cleanup_unit_id", "")
            or getattr(job, "cleanup_job_id", "")
            or ""
        )
        parent_cleanup_mask_id = str(
            params.get("cleanup_obligation_id")
            or params.get("formal_cleanup_mask_id")
            or formal_plan.cleanup_mask_id
            or ""
        )
        text_block_root_id = str(
            params.get("text_block_root_id")
            or getattr(job, "text_block_root_id", "")
            or parent_cleanup_unit_id
            or ""
        )
        params["cleanup_obligation_id"] = parent_cleanup_mask_id
        params["parent_cleanup_mask_id"] = parent_cleanup_mask_id
        params["text_block_root_id"] = text_block_root_id
        params["mask_space"] = "page"
        params["bbox_format"] = "xyxy_exclusive"
        params["source_glyph_erasure_bbox"] = list(partition_bbox or [])
        params["source_glyph_erasure_expected_area_bbox"] = []
        params["parent_expected_erasure_bbox"] = list(parent_expected or [])
        params["proof_scope"] = "component_partition"
        params["proof_scope_bbox"] = list(partition_bbox or [])
        params["proof_scope_excludes_parent_expected_bbox"] = True
        params["parent_cleanup_unit_id"] = parent_cleanup_unit_id
        params["partition_parent_cleanup_job_id"] = str(getattr(job, "cleanup_job_id", "") or plan.cleanup_job_id)
        params["represented_source_glyph_ids"] = list(partition_mask.consumed_source_glyph_mask_ids or [])
        params["consumed_source_glyph_mask_ids"] = list(partition_mask.consumed_source_glyph_mask_ids or [])
        params["missing_source_glyph_mask_ids"] = list(partition_mask.missing_source_glyph_mask_ids or [])
        plan = CleanupPlan(
            cleanup_plan_id=plan.cleanup_plan_id,
            cleanup_job_id=plan.cleanup_job_id,
            cleanup_mask_id=plan.cleanup_mask_id,
            cleanup_class=plan.cleanup_class,
            selected_backend=plan.selected_backend,
            cleanup_method=f"{plan.cleanup_method}_partition_component",
            parent_execution_bundle_id=plan.parent_execution_bundle_id,
            parent_logical_text_unit_id=plan.parent_logical_text_unit_id,
            text_block_root_id=plan.text_block_root_id,
            target_region_ids=list(plan.target_region_ids or []),
            backend_parameters=params,
            inpaint_mode=plan.inpaint_mode,
            crop_context_bbox=list(partition_mask.erase_mask_bbox or plan.crop_context_bbox or []),
            fallback_policy=list(plan.fallback_policy or []),
            expected_runtime_class=plan.expected_runtime_class,
            proof_thresholds=dict(plan.proof_thresholds or {}),
        )
        partition_results, partition_proofs, attempt_records = _run_adaptive_cleanup_attempts(
            image=image,
            source_image=source_image,
            cleanup_plan=plan,
            cleanup_mask=partition_mask,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
        )
        child_result_count += len(partition_results)
        child_proof_count += len(partition_proofs)
        result, proof = _select_best_commit_candidate_result(partition_results, partition_proofs)
        if result is None:
            result, proof = _best_failed_attempt(partition_results, partition_proofs)
        if result is None or proof is None:
            statuses.append(
                {
                    **dict(base_record),
                    "runtime_status": "contract_error",
                    "cleanup_mask_id": partition_mask.cleanup_mask_id,
                    "cleanup_obligation_id": parent_cleanup_mask_id,
                    "parent_cleanup_mask_id": parent_cleanup_mask_id,
                    "text_block_root_id": text_block_root_id,
                    "failure_reason": "cleanup_contract_error_invalid_input",
                    "cleanup_outcome_state": "cleanup_contract_error_invalid_input",
                    "contract_error_original_reason": "partition_attempt_missing_result_or_proof",
                    "partitioned_cleanup": True,
                    "partition_index": index,
                    "partition_count": partition_count,
                    "partition_parent_rejection_reason": parent_rejection_reason,
                    "target_region_ids": list(target_region_ids),
                    "renderer_consumed": False,
                }
            )
            continue
        proof_passed = proof.proof_status == ProofStatus.PASSED
        degraded_cleanup_diagnostic = bool((proof.metrics or {}).get("degraded_cleanup_diagnostic", False))
        runtime_status = (
            "passed"
            if proof_passed and result.pixel_changed
            else "failed"
        )
        failure_reason = "passed" if runtime_status == "passed" else proof.failure_reason or result.failure_reason
        proof_metrics = _json_safe(proof.metrics or {})
        selected_attempt = _selected_attempt_record(attempt_records, result)
        child_status = {
                **dict(base_record),
                "runtime_status": runtime_status,
                "cleanup_plan_id": result.cleanup_plan_id,
                "cleanup_mask_id": partition_mask.cleanup_mask_id,
                "cleanup_obligation_id": parent_cleanup_mask_id,
                "parent_cleanup_mask_id": parent_cleanup_mask_id,
                "text_block_root_id": text_block_root_id,
                "cleanup_result_id": result.cleanup_result_id,
                "cleanup_proof_id": proof.cleanup_proof_id,
                "proof_status": _enum_value(proof.proof_status),
                "failure_reason": failure_reason,
                "cleanup_outcome_state": _cleanup_outcome_state_for_failure(
                    failure_reason,
                    runtime_status=runtime_status,
                ),
                "pixel_changed": bool(result.pixel_changed),
                "changed_pixel_count": int(result.changed_pixel_count or 0),
                "cleaned_image_ref": result.cleaned_image_ref,
                "diff_ref": result.diff_ref,
                "mask_ref": result.mask_ref,
                "proof_metrics": proof_metrics,
                "proof_quality_state": proof_metrics.get("proof_quality_state", ""),
                "degraded_commit_allowed": False,
                "degraded_cleanup_diagnostic": bool(degraded_cleanup_diagnostic),
                "source_residual_delta_dark_pixels": proof_metrics.get("source_residual_delta_dark_pixels", 0),
                "source_residual_delta_dark_ratio": proof_metrics.get("source_residual_delta_dark_ratio", 0),
                "collateral_art_damage_delta": proof_metrics.get("collateral_art_damage_delta", 0),
                "proof_scope": proof_metrics.get("proof_scope", "component_partition"),
                "proof_scope_bbox": proof_metrics.get("proof_scope_bbox", list(partition_bbox or [])),
                "parent_cleanup_unit_id": parent_cleanup_unit_id,
                "parent_expected_erasure_bbox": list(parent_expected or []),
                "proof_scope_excludes_parent_expected_bbox": bool(
                    proof_metrics.get("proof_scope_excludes_parent_expected_bbox", True)
                ),
                "adaptive_attempt_count": len(attempt_records),
                "adaptive_attempts": _json_safe(attempt_records),
                "selected_attempt_index": selected_attempt.get("attempt_index") if selected_attempt else None,
                "selected_attempt_strategy": selected_attempt.get("strategy") if selected_attempt else "",
                "partitioned_cleanup": True,
                "partition_index": index,
                "partition_count": partition_count,
                "partition_parent_rejection_reason": parent_rejection_reason,
                "target_region_ids": list(target_region_ids),
                "represented_source_glyph_ids": list(partition_mask.consumed_source_glyph_mask_ids or []),
                "consumed_source_glyph_mask_ids": list(partition_mask.consumed_source_glyph_mask_ids or []),
                "missing_source_glyph_mask_ids": list(partition_mask.missing_source_glyph_mask_ids or []),
                "render_consumption_decision_if_consumed": proof.render_consumption_decision_if_consumed,
                "renderer_consumed": False,
            }
        statuses.append(child_status)
        selected_child_records.append((result, proof, child_status))
    aggregate_record = _parent_cleanup_unit_aggregate_record(
        base_record=base_record,
        statuses=statuses,
        partition_masks=partition_masks,
        parent_rejection_reason=parent_rejection_reason,
        parent_cleanup_unit_id=str(
            getattr(job, "cleanup_unit_id", "")
            or (base_record.get("cleanup_unit_id") if isinstance(base_record, Mapping) else "")
            or (base_record.get("cleanup_job_id") if isinstance(base_record, Mapping) else "")
            or ""
        ),
    )
    if aggregate_record:
        aggregate_records.append(aggregate_record)
        for status in statuses:
            status["parent_cleanup_unit_aggregate_status"] = aggregate_record.get("aggregate_status", "")
            status["parent_cleanup_unit_aggregate_complete"] = bool(
                aggregate_record.get("aggregate_complete", False)
            )
            status["parent_cleanup_unit_component_attempt_count"] = aggregate_record.get(
                "component_attempt_count",
                0,
            )
    parent_plan = _runtime_plan_from_formal_plan(
        formal_plan=formal_plan,
        page_id=page_id,
        job=job,
        inpaint_mode=inpaint_mode,
        cleanup_class=cleanup_class,
        cleanup_mask=parent_cleanup_mask,
    )
    parent_params = dict(parent_plan.backend_parameters or {})
    parent_params.update(
        {
            "partitioned_cleanup": True,
            "partition_count": partition_count,
            "partition_backend_result_count": child_result_count,
            "partition_backend_proof_count": child_proof_count,
            "partition_parent_rejection_reason": parent_rejection_reason,
            "partition_parent_full_mask_first_attempt": True,
            "partition_parent_full_mask_first_failed_reason": (
                parent_full_failed_proof.failure_reason
                if parent_full_failed_proof is not None
                else (
                    parent_full_failed_result.failure_reason
                    if parent_full_failed_result is not None
                    else "no_parent_full_mask_result"
                )
            ),
            "partition_child_statuses": _json_safe(statuses),
            "parent_cleanup_unit_aggregate_status": aggregate_record.get("aggregate_status", ""),
            "parent_cleanup_unit_aggregate_complete": bool(aggregate_record.get("aggregate_complete", False)),
            "cleanup_obligation_id": parent_cleanup_mask_id,
            "parent_cleanup_mask_id": parent_cleanup_mask_id,
            "text_block_root_id": text_block_root_id,
            "mask_space": "page",
            "bbox_format": "xyxy_exclusive",
        }
    )
    parent_plan = CleanupPlan(
        cleanup_plan_id=f"{parent_plan.cleanup_plan_id}_partition_parent",
        cleanup_job_id=parent_plan.cleanup_job_id,
        cleanup_mask_id=str(parent_cleanup_mask.cleanup_mask_id),
        cleanup_class=parent_plan.cleanup_class,
        selected_backend=parent_plan.selected_backend,
        cleanup_method=f"{parent_plan.cleanup_method}_partition_parent_merged",
        parent_execution_bundle_id=parent_plan.parent_execution_bundle_id,
        parent_logical_text_unit_id=parent_plan.parent_logical_text_unit_id,
        text_block_root_id=parent_plan.text_block_root_id,
        target_region_ids=list(parent_plan.target_region_ids or []),
        backend_parameters=parent_params,
        inpaint_mode=parent_plan.inpaint_mode,
        crop_context_bbox=list(parent_cleanup_mask.erase_mask_bbox or parent_plan.crop_context_bbox or []),
        fallback_policy=list(parent_plan.fallback_policy or []),
        expected_runtime_class=parent_plan.expected_runtime_class,
        proof_thresholds=dict(parent_plan.proof_thresholds or {}),
    )
    parent_result, parent_merge_failure = _partition_parent_cleanup_result(
        page_id=page_id,
        region_id=region_id,
        image=image,
        parent_plan=parent_plan,
        parent_cleanup_mask=parent_cleanup_mask,
        selected_child_records=selected_child_records,
        aggregate_record=aggregate_record,
    )
    parent_proof = prove_cleanup_result(
        source_image=source_image,
        before_image=image,
        cleanup_result=parent_result,
        cleanup_plan=parent_plan,
        cleanup_mask=parent_cleanup_mask,
    )
    if parent_merge_failure:
        parent_proof = _proof(
            parent_result,
            parent_plan,
            status=ProofStatus.FAILED,
            failure_reason=parent_merge_failure,
            metrics={
                "partition_parent_merge_failure": parent_merge_failure,
                "proof_quality_state": "partition_parent_merge_failed",
            },
        )
    aggregate_complete = bool(aggregate_record.get("aggregate_complete", False))
    parent_full_mask_proof_complete = (
        not bool(parent_merge_failure)
        and parent_proof.proof_status == ProofStatus.PASSED
        and bool(parent_result.pixel_changed)
    )
    parent_passed = parent_full_mask_proof_complete
    first_child_failure = next(
        (
            str(status.get("failure_reason") or "")
            for status in statuses
            if str(status.get("failure_reason") or "") and str(status.get("failure_reason") or "") != "passed"
        ),
        "",
    )
    parent_failure_reason = (
        "passed"
        if parent_passed
        else parent_merge_failure
        or first_child_failure
        or parent_proof.failure_reason
        or parent_rejection_reason
        or "parent_cleanup_unit_aggregate_incomplete"
    )
    parent_status = {
        **dict(base_record),
        "runtime_status": "passed" if parent_passed else "failed",
        "cleanup_plan_id": parent_result.cleanup_plan_id,
        "cleanup_mask_id": str(parent_cleanup_mask.cleanup_mask_id),
        "cleanup_obligation_id": parent_cleanup_mask_id,
        "parent_cleanup_mask_id": parent_cleanup_mask_id,
        "text_block_root_id": text_block_root_id,
        "cleanup_result_id": parent_result.cleanup_result_id,
        "cleanup_proof_id": parent_proof.cleanup_proof_id,
        "proof_status": _enum_value(parent_proof.proof_status),
        "failure_reason": parent_failure_reason,
        "cleanup_outcome_state": _cleanup_outcome_state_for_failure(
            parent_failure_reason,
            runtime_status="passed" if parent_passed else "failed",
        ),
        "pixel_changed": bool(parent_result.pixel_changed),
        "changed_pixel_count": int(parent_result.changed_pixel_count or 0),
        "cleaned_image_ref": parent_result.cleaned_image_ref,
        "diff_ref": parent_result.diff_ref,
        "mask_ref": parent_result.mask_ref,
        "proof_metrics": _json_safe(parent_proof.metrics or {}),
        "proof_quality_state": (parent_proof.metrics or {}).get("proof_quality_state", ""),
        "degraded_commit_allowed": False,
        "partitioned_cleanup": True,
        "partition_count": partition_count,
        "partition_backend_result_count": child_result_count,
        "partition_backend_proof_count": child_proof_count,
        "partition_parent_rejection_reason": parent_rejection_reason,
        "partition_child_statuses": _json_safe(statuses),
        "parent_cleanup_unit_id": parent_cleanup_unit_id,
        "parent_cleanup_unit_aggregate_status": aggregate_record.get("aggregate_status", ""),
        "parent_cleanup_unit_aggregate_complete": aggregate_complete,
        "partition_parent_full_mask_proof_complete": parent_full_mask_proof_complete,
        "parent_cleanup_unit_component_attempt_count": aggregate_record.get("component_attempt_count", 0),
        "target_region_ids": list(target_region_ids),
        "render_consumption_decision_if_consumed": parent_proof.render_consumption_decision_if_consumed,
        "renderer_consumed": False,
    }
    return [parent_result], [parent_proof], [parent_status], aggregate_records


def _partition_parent_cleanup_result(
    *,
    page_id: str,
    region_id: str,
    image: Any,
    parent_plan: CleanupPlan,
    parent_cleanup_mask: CleanupMask,
    selected_child_records: Sequence[tuple[CleanupResult, CleanupProof, Mapping[str, Any]]],
    aggregate_record: Mapping[str, Any],
) -> tuple[CleanupResult, str]:
    identity = _cleanup_identity_fields(parent_plan, parent_cleanup_mask, fallback_region_id=region_id)
    canonical_region_id = _canonical_parent_region_id(parent_plan, parent_cleanup_mask, fallback_region_id=region_id)
    if np is None or Image is None:
        return (
            CleanupResult(
                cleanup_result_id=f"cres_{_safe_id(parent_plan.cleanup_plan_id)}",
                cleanup_plan_id=str(parent_plan.cleanup_plan_id),
                cleanup_job_id=str(parent_plan.cleanup_job_id),
                cleanup_mask_id=str(parent_cleanup_mask.cleanup_mask_id),
                operation_bbox=_valid_bbox(parent_cleanup_mask.erase_mask_bbox),
                page=page_id,
                region_id=canonical_region_id,
                parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
                parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
                text_block_root_id=str(identity.get("text_block_root_id") or ""),
                target_region_ids=list(identity.get("target_region_ids") or []),
                cleanup_class=parent_plan.cleanup_class,
                pixel_changed=False,
                changed_pixel_count=0,
                execution_status="failed",
                failure_reason="pillow_or_numpy_unavailable",
                backend_name="partition_parent_merge",
                backend_parameters=dict(parent_plan.backend_parameters or {}),
                cleaned_image=image,
            ),
            "pillow_or_numpy_unavailable",
        )
    try:
        source_np = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        parent_np = source_np.copy()
    except Exception as exc:
        return (
            CleanupResult(
                cleanup_result_id=f"cres_{_safe_id(parent_plan.cleanup_plan_id)}",
                cleanup_plan_id=str(parent_plan.cleanup_plan_id),
                cleanup_job_id=str(parent_plan.cleanup_job_id),
                cleanup_mask_id=str(parent_cleanup_mask.cleanup_mask_id),
                operation_bbox=_valid_bbox(parent_cleanup_mask.erase_mask_bbox),
                page=page_id,
                region_id=canonical_region_id,
                parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
                parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
                text_block_root_id=str(identity.get("text_block_root_id") or ""),
                target_region_ids=list(identity.get("target_region_ids") or []),
                cleanup_class=parent_plan.cleanup_class,
                pixel_changed=False,
                changed_pixel_count=0,
                execution_status="failed",
                failure_reason=f"partition_parent_source_array_error:{type(exc).__name__}: {exc}",
                backend_name="partition_parent_merge",
                backend_parameters=dict(parent_plan.backend_parameters or {}),
                cleaned_image=image,
            ),
            "partition_parent_source_array_error",
        )
    changed_total = 0
    merge_failure = ""
    for child_result, child_proof, child_status in selected_child_records:
        if child_proof.proof_status != ProofStatus.PASSED or not bool(child_result.pixel_changed):
            continue
        cleaned_image = _cleanup_result_page_payload(child_result, image)
        if cleaned_image is None:
            merge_failure = "partition_child_cleaned_image_missing"
            continue
        committed_pixels, _commit_mask = _commit_runtime_result_pixels(
            working_np=parent_np,
            source_np=source_np,
            cleaned_image=cleaned_image,
            operation_bbox=child_result.operation_bbox,
        )
        changed_total += int(committed_pixels or 0)
    cleaned_image = Image.fromarray(parent_np.astype("uint8"), mode="RGB")
    params = dict(parent_plan.backend_parameters or {})
    params.update(
        {
            "candidate_status": "completed" if changed_total > 0 else "no_pixel_change",
            "partition_parent_candidate": True,
            "parent_cleanup_unit_aggregate_status": aggregate_record.get("aggregate_status", ""),
            "parent_cleanup_unit_aggregate_complete": bool(aggregate_record.get("aggregate_complete", False)),
            "parent_cleanup_unit_component_attempt_count": aggregate_record.get("component_attempt_count", 0),
        }
    )
    if merge_failure:
        params["partition_parent_merge_failure"] = merge_failure
    return (
        CleanupResult(
            cleanup_result_id=f"cres_{_safe_id(parent_plan.cleanup_plan_id)}",
            cleanup_plan_id=str(parent_plan.cleanup_plan_id),
            cleanup_job_id=str(parent_plan.cleanup_job_id),
            cleanup_mask_id=str(parent_cleanup_mask.cleanup_mask_id),
            operation_bbox=_valid_bbox(parent_cleanup_mask.erase_mask_bbox),
            page=page_id,
            region_id=canonical_region_id,
            parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
            parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
            text_block_root_id=str(identity.get("text_block_root_id") or ""),
            target_region_ids=list(identity.get("target_region_ids") or []),
            cleanup_class=parent_plan.cleanup_class,
            pixel_changed=changed_total > 0,
            changed_pixel_count=changed_total,
            execution_status="completed" if changed_total > 0 else "completed_no_pixel_change",
            failure_reason="" if changed_total > 0 else "no_pixels_changed",
            backend_name="partition_parent_merge",
            backend_parameters=params,
            mask_stats=mask_stats(parent_cleanup_mask.erase_mask) or {},
            cleaned_image=cleaned_image,
        ),
        merge_failure,
    )


def _parent_cleanup_unit_aggregate_record(
    *,
    base_record: Mapping[str, Any],
    statuses: Sequence[Mapping[str, Any]],
    partition_masks: Sequence[CleanupMask],
    parent_rejection_reason: str,
    parent_cleanup_unit_id: str,
) -> dict[str, Any]:
    if not statuses:
        return {}
    component_attempt_ids = [
        str(status.get("cleanup_result_id") or "")
        for status in statuses
        if str(status.get("cleanup_result_id") or "")
    ]
    component_proof_ids = [
        str(status.get("cleanup_proof_id") or "")
        for status in statuses
        if str(status.get("cleanup_proof_id") or "")
    ]
    component_statuses = [
        {
            "cleanup_result_id": status.get("cleanup_result_id", ""),
            "cleanup_proof_id": status.get("cleanup_proof_id", ""),
            "runtime_status": status.get("runtime_status", ""),
            "proof_status": status.get("proof_status", ""),
            "failure_reason": status.get("failure_reason", ""),
            "pixel_changed": bool(status.get("pixel_changed")),
            "partition_index": status.get("partition_index", ""),
            "proof_scope": status.get("proof_scope", ""),
            "proof_scope_bbox": status.get("proof_scope_bbox", []),
        }
        for status in statuses
    ]
    expected_count = len(partition_masks)
    passed = [
        status
        for status in statuses
        if str(status.get("runtime_status") or "") == "passed" and bool(status.get("pixel_changed"))
    ]
    warning = [
        status
        for status in statuses
        if str(status.get("runtime_status") or "") == "warning" and bool(status.get("pixel_changed"))
    ]
    commit_candidates = passed
    failure_reasons = [
        str(status.get("failure_reason") or "")
        for status in statuses
        if str(status.get("failure_reason") or "") and str(status.get("failure_reason") or "") != "passed"
    ]
    aggregate_status = _parent_cleanup_unit_aggregate_status(
        expected_count=expected_count,
        attempted_count=len(statuses),
        passed_count=len(passed),
        warning_count=len(warning),
        failure_reasons=failure_reasons,
    )
    aggregate_complete = expected_count > 0 and len(commit_candidates) == expected_count and len(statuses) == expected_count
    return {
        "page_id": base_record.get("page_id", ""),
        "cleanup_job_id": base_record.get("cleanup_job_id", ""),
        "parent_cleanup_unit_id": parent_cleanup_unit_id,
        "cleanup_unit_id": base_record.get("cleanup_unit_id", parent_cleanup_unit_id),
        "cleanup_unit_level": base_record.get("cleanup_unit_level", ""),
        "cleanup_unit_anchor_region_id": base_record.get("cleanup_unit_anchor_region_id", ""),
        "cleanup_unit_child_region_ids": list(base_record.get("cleanup_unit_child_region_ids", []) or []),
        "target_region_ids": list(base_record.get("target_region_ids", []) or []),
        "required_source_glyph_mask_ids": list(
            base_record.get("cleanup_unit_required_source_glyph_mask_ids", []) or []
        ),
        "component_attempt_ids": component_attempt_ids,
        "component_proof_ids": component_proof_ids,
        "component_attempt_count": len(statuses),
        "component_expected_count": expected_count,
        "component_passed_count": len(passed),
        "component_warning_count": len(warning),
        "component_commit_candidate_count": len(commit_candidates),
        "component_uncommitted_warning_count": len(warning),
        "component_failed_count": max(0, len(statuses) - len(commit_candidates)),
        "component_statuses": component_statuses,
        "aggregate_status": aggregate_status,
        "aggregate_complete": aggregate_complete,
        "aggregate_completion_claimed": aggregate_complete,
        "aggregate_commit_status": (
            "existing_upstream_commit_path_required"
            if aggregate_complete
            else "unresolved_components_after_bounded_attempts"
        ),
        "parent_rejection_reason": parent_rejection_reason,
        "renderer_consumed": False,
    }


def _parent_cleanup_unit_aggregate_status(
    *,
    expected_count: int,
    attempted_count: int,
    passed_count: int,
    warning_count: int = 0,
    failure_reasons: Sequence[str],
) -> str:
    if expected_count <= 0 or attempted_count <= 0:
        return "cleanup_contract_error_invalid_input"
    if passed_count == expected_count and attempted_count == expected_count:
        return "cleanup_completed"
    if passed_count + warning_count == expected_count and attempted_count == expected_count:
        return "cleanup_partially_completed_with_warnings"
    joined = " ".join(str(reason or "").lower() for reason in failure_reasons)
    if "sourceglyph" in joined or "source_glyph" in joined:
        return "cleanup_contract_error_missing_sourceglyph"
    if "unsupported" in joined or "non_phase5" in joined:
        return "cleanup_contract_error_invalid_input"
    if "backend_noop" in joined or "no_pixels_changed" in joined:
        return "cleanup_failed_no_safe_pixel_change"
    if "protected" in joined or "preserve" in joined or "sfx" in joined or "decorative" in joined:
        return "cleanup_contract_error_invalid_input"
    if "source_residual" in joined:
        return "cleanup_partially_completed_with_warnings"
    if "mask" in joined or "scope" in joined or "erase" in joined:
        return "cleanup_partially_completed_with_warnings"
    return "cleanup_partially_completed_with_warnings"


def _run_adaptive_cleanup_attempts(
    *,
    image: Any,
    source_image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    backend_context: Mapping[str, Any] | None = None,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
    proof_context: Mapping[str, Any] | None = None,
) -> tuple[list[CleanupResult], list[CleanupProof], list[dict[str, Any]]]:
    adaptive_started = time.time()
    region_id = _canonical_parent_region_id(cleanup_plan, cleanup_mask, fallback_region_id=region_id) or region_id
    results: list[CleanupResult] = []
    proofs: list[CleanupProof] = []
    attempts: list[dict[str, Any]] = []

    model_required = _model_required_for_plan(cleanup_plan, cleanup_mask)
    if model_required:
        model_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=0,
            strategy="model_required_full_mask_inpaint",
            cleanup_tag="authorized_ai_inpaint_full_mask",
            retry_reason="model_required_by_ai_mode",
            suffix="",
        )
        model_plan = replace(model_plan, inpaint_mode="ai")
        model_plan.backend_parameters["inpaint_mode"] = "ai"
        model_plan.backend_parameters["model_required"] = True
        model_plan.backend_parameters["model_required_reason"] = (
            "inpaint_mode_ai_requires_model_for_accepted_normal_text"
        )
        model_plan.backend_parameters["authorized_ai_retry_source"] = (
            "formal_cleanup_plan_exact_cleanup_mask"
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(model_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=0,
            strategy="model_required_full_mask_inpaint",
            retry_reason="model_required_by_ai_mode",
        )
        model_result, model_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=model_plan,
            cleanup_mask=cleanup_mask,
            backend_context=None,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=0,
            strategy="model_required_full_mask_inpaint",
            retry_reason="model_required_by_ai_mode",
            proof_context=proof_context,
        )
        results.append(model_result)
        proofs.append(model_proof)
        attempts.append(
            _adaptive_attempt_record(
                model_result,
                model_proof,
                0,
                "model_required_full_mask_inpaint",
                "model_required_by_ai_mode",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(model_plan.cleanup_plan_id),
            attempt_index=0,
            strategy="model_required_full_mask_inpaint",
            proof_status=_enum_value(model_proof.proof_status),
            failure_reason=model_proof.failure_reason or model_result.failure_reason,
            runtime_ms=model_result.runtime_ms,
        )
        if _proof_passed_with_pixels(model_result, model_proof):
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "early_pass",
                page_id=page_id,
                region_id=region_id,
                attempt_count=len(attempts),
                elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
            )
            return results, proofs, attempts
        last_result = model_result
        last_proof = model_proof
    else:
        first_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=0,
            strategy="current_conservative",
            cleanup_tag=str((cleanup_plan.backend_parameters or {}).get("cleanup_tag") or "cleanup_effectiveness_retry"),
            retry_reason="initial_attempt",
            suffix="",
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(first_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=0,
            strategy="current_conservative",
            retry_reason="initial_attempt",
        )
        first_result, first_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=first_plan,
            cleanup_mask=cleanup_mask,
            backend_context=backend_context,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=0,
            strategy="current_conservative",
            retry_reason="initial_attempt",
            proof_context=proof_context,
        )
        results.append(first_result)
        proofs.append(first_proof)
        attempts.append(_adaptive_attempt_record(first_result, first_proof, 0, "current_conservative", "initial_attempt"))
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(first_plan.cleanup_plan_id),
            attempt_index=0,
            strategy="current_conservative",
            proof_status=_enum_value(first_proof.proof_status),
            failure_reason=first_proof.failure_reason or first_result.failure_reason,
            runtime_ms=first_result.runtime_ms,
        )
        if _proof_passed_with_pixels(first_result, first_proof):
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "early_pass",
                page_id=page_id,
                region_id=region_id,
                attempt_count=len(attempts),
                elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
            )
            return results, proofs, attempts
        last_result = first_result
        last_proof = first_proof
    attempt_index = 1

    if (not model_required) and _should_run_backend_noop_fallback(last_result, last_proof):
        for strategy, cleanup_tag in (
            ("backend_noop_deterministic_fallback", "deterministic_local_fill"),
            ("minimal_erasure_fallback", "minimal_erasure"),
        ):
            noop_plan = _adaptive_plan_variant(
                cleanup_plan,
                attempt_index=attempt_index,
                strategy=strategy,
                cleanup_tag=cleanup_tag,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
                suffix=f"_attempt{attempt_index}_{strategy}",
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "start",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(noop_plan.cleanup_plan_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                attempt_index=attempt_index,
                strategy=strategy,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
            )
            noop_result, noop_proof = _execute_cleanup_attempt(
                image=image,
                proof_before=image,
                source_image=source_image,
                cleanup_plan=noop_plan,
                cleanup_mask=cleanup_mask,
                backend_context=backend_context,
                page_id=page_id,
                region_id=region_id,
                use_gpu=use_gpu,
                model_id=model_id,
                artifact_dir=artifact_dir,
                attempt_index=attempt_index,
                strategy=strategy,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
                proof_context=proof_context,
            )
            results.append(noop_result)
            proofs.append(noop_proof)
            attempts.append(
                _adaptive_attempt_record(
                    noop_result,
                    noop_proof,
                    attempt_index,
                    strategy,
                    last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
                )
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "end",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(noop_plan.cleanup_plan_id),
                attempt_index=attempt_index,
                strategy=strategy,
                proof_status=_enum_value(noop_proof.proof_status),
                failure_reason=noop_proof.failure_reason or noop_result.failure_reason,
                runtime_ms=noop_result.runtime_ms,
            )
            last_result = noop_result
            last_proof = noop_proof
            attempt_index += 1
            if _proof_passed_with_pixels(noop_result, noop_proof):
                _page014_timeout_checkpoint(
                    "adaptive_cleanup_attempts",
                    "early_pass",
                    page_id=page_id,
                    region_id=region_id,
                    attempt_count=len(attempts),
                    elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
                )
                return results, proofs, attempts
            if not _should_run_backend_noop_fallback(noop_result, noop_proof):
                break

    if _should_run_mask_faithful_inpaint_retry(cleanup_plan, last_proof):
        inpaint_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=attempt_index,
            strategy="mask_faithful_root_inpaint_retry",
            cleanup_tag="background_narration_root_inpaint",
            retry_reason=last_proof.failure_reason or last_result.failure_reason or "background_cleanup_incomplete",
            suffix=f"_attempt{attempt_index}_mask_faithful_root_inpaint",
        )
        inpaint_plan.backend_parameters["mask_faithful_root_inpaint_source"] = (
            "formal_cleanup_plan_exact_cleanup_mask"
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(inpaint_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=attempt_index,
            strategy="mask_faithful_root_inpaint_retry",
            retry_reason=last_proof.failure_reason or last_result.failure_reason or "background_cleanup_incomplete",
        )
        inpaint_result, inpaint_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=inpaint_plan,
            cleanup_mask=cleanup_mask,
            backend_context=backend_context,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=attempt_index,
            strategy="mask_faithful_root_inpaint_retry",
            retry_reason=last_proof.failure_reason or last_result.failure_reason or "background_cleanup_incomplete",
            proof_context=proof_context,
        )
        results.append(inpaint_result)
        proofs.append(inpaint_proof)
        attempts.append(
            _adaptive_attempt_record(
                inpaint_result,
                inpaint_proof,
                attempt_index,
                "mask_faithful_root_inpaint_retry",
                last_proof.failure_reason or last_result.failure_reason or "background_cleanup_incomplete",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(inpaint_plan.cleanup_plan_id),
            attempt_index=attempt_index,
            strategy="mask_faithful_root_inpaint_retry",
            proof_status=_enum_value(inpaint_proof.proof_status),
            failure_reason=inpaint_proof.failure_reason or inpaint_result.failure_reason,
            runtime_ms=inpaint_result.runtime_ms,
        )
        last_result = inpaint_result
        last_proof = inpaint_proof
        attempt_index += 1
        if _proof_passed_with_pixels(inpaint_result, inpaint_proof):
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "early_pass",
                page_id=page_id,
                region_id=region_id,
                attempt_count=len(attempts),
                elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
            )
            return results, proofs, attempts

    if _should_run_residual_retry(last_proof):
        residual_retry_mask, residual_retry_skip_reason = _residual_retry_cleanup_mask(
            cleanup_mask,
            last_proof,
        )
        if residual_retry_mask is None:
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "strategy": "residual_retry",
                    "retry_reason": last_proof.failure_reason or "source_residual_remaining",
                    "attempt_skipped": True,
                    "skip_reason": residual_retry_skip_reason,
                }
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "skipped",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                attempt_index=attempt_index,
                strategy="residual_retry",
                reason=residual_retry_skip_reason,
            )
        else:
            residual_plan = _adaptive_plan_variant(
                cleanup_plan,
                attempt_index=attempt_index,
                strategy="residual_retry",
                cleanup_tag="executable_residual_tone_fill",
                retry_reason=last_proof.failure_reason or "source_residual_remaining",
                suffix=f"_attempt{attempt_index}_residual_retry",
            )
            retry_pixels = int(getattr(residual_retry_mask, "foreground_mask_pixels", 0) or 0)
            residual_plan.backend_parameters["residual_retry_source"] = "executable_text_residual_mask"
            residual_plan.backend_parameters["residual_retry_source_context"] = (
                "previous_cleanup_candidate"
            )
            residual_plan.backend_parameters["residual_retry_foreground_pixels"] = retry_pixels
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "start",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(residual_plan.cleanup_plan_id),
                cleanup_mask_id=str(residual_retry_mask.cleanup_mask_id),
                attempt_index=attempt_index,
                strategy="residual_retry",
                retry_reason=last_proof.failure_reason or "source_residual_remaining",
            )
            # Run the residual mask against the previous candidate, then merge
            # that residual change back into the primary candidate. This keeps
            # any texture or line-art recovery produced by the first attempt
            # instead of replacing it with a source-image white glyph fill.
            residual_image = _cleanup_result_page_payload(last_result, source_image) or source_image
            residual_result, residual_proof = _execute_cleanup_attempt(
                image=residual_image,
                proof_before=residual_image,
                source_image=source_image,
                cleanup_plan=residual_plan,
                cleanup_mask=residual_retry_mask,
                backend_context=backend_context,
                page_id=page_id,
                region_id=region_id,
                use_gpu=use_gpu,
                model_id=model_id,
                artifact_dir=artifact_dir,
                attempt_index=attempt_index,
                strategy="residual_retry",
                retry_reason=last_proof.failure_reason or "source_residual_remaining",
                proof_context=proof_context,
            )
            results.append(residual_result)
            proofs.append(residual_proof)
            attempts.append(
                _adaptive_attempt_record(
                    residual_result,
                    residual_proof,
                    attempt_index,
                    "residual_retry",
                    last_proof.failure_reason or "source_residual_remaining",
                )
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "end",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(residual_plan.cleanup_plan_id),
                attempt_index=attempt_index,
                strategy="residual_retry",
                proof_status=_enum_value(residual_proof.proof_status),
                failure_reason=residual_proof.failure_reason or residual_result.failure_reason,
                runtime_ms=residual_result.runtime_ms,
            )
            combined_plan = _adaptive_plan_variant(
                cleanup_plan,
                attempt_index=attempt_index + 1,
                strategy="residual_retry_combined_parent_proof",
                cleanup_tag=str((cleanup_plan.backend_parameters or {}).get("cleanup_tag") or "cleanup_effectiveness_retry"),
                retry_reason=last_proof.failure_reason or "source_residual_remaining",
                suffix=f"_attempt{attempt_index}_residual_retry_combined",
            )
            combined_result = _combined_residual_retry_result(
                base_image=image,
                cleanup_plan=combined_plan,
                cleanup_mask=cleanup_mask,
                residual_result=residual_result,
                primary_result=last_result,
                page_id=page_id,
                region_id=region_id,
            )
            combined_proof = prove_cleanup_result(
                source_image=source_image,
                before_image=image,
                cleanup_result=combined_result,
                cleanup_plan=combined_plan,
                cleanup_mask=cleanup_mask,
                proof_context=proof_context,
            )
            results.append(combined_result)
            proofs.append(combined_proof)
            attempts.append(
                _adaptive_attempt_record(
                    combined_result,
                    combined_proof,
                    attempt_index + 1,
                    "residual_retry_combined_parent_proof",
                    last_proof.failure_reason or "source_residual_remaining",
                )
            )
            last_result = combined_result
            last_proof = combined_proof
            attempt_index += 2
            if _proof_passed_with_pixels(combined_result, combined_proof):
                _page014_timeout_checkpoint(
                    "adaptive_cleanup_attempts",
                    "early_pass",
                    page_id=page_id,
                    region_id=region_id,
                    attempt_count=len(attempts),
                    elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
                )
                return results, proofs, attempts

    if _should_run_restoration_retry(last_proof):
        restoration_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            cleanup_tag="speech_strong",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
            suffix=f"_attempt{attempt_index}_restoration_retry",
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(restoration_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
        )
        restoration_result, restoration_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=restoration_plan,
            cleanup_mask=cleanup_mask,
            backend_context=backend_context,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
            proof_context=proof_context,
        )
        results.append(restoration_result)
        proofs.append(restoration_proof)
        attempts.append(
            _adaptive_attempt_record(
                restoration_result,
                restoration_proof,
                attempt_index,
                "restoration_scale_back_retry",
                last_proof.failure_reason or "collateral_or_destructive_fill",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(restoration_plan.cleanup_plan_id),
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            proof_status=_enum_value(restoration_proof.proof_status),
            failure_reason=restoration_proof.failure_reason or restoration_result.failure_reason,
            runtime_ms=restoration_result.runtime_ms,
        )
        last_result = restoration_result
        last_proof = restoration_proof
        attempt_index += 1

    if _should_run_authorized_ai_inpaint_retry(cleanup_plan, last_proof):
        ai_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=attempt_index,
            strategy="authorized_ai_inpaint_full_mask_retry",
            cleanup_tag="authorized_ai_inpaint_full_mask",
            retry_reason=last_proof.failure_reason or "caption_background_cleanup_incomplete",
            suffix=f"_attempt{attempt_index}_authorized_ai_inpaint",
        )
        ai_plan = replace(ai_plan, inpaint_mode="ai")
        ai_plan.backend_parameters["inpaint_mode"] = "ai"
        ai_plan.backend_parameters["authorized_ai_retry_source"] = "formal_cleanup_plan_exact_cleanup_mask"
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(ai_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=attempt_index,
            strategy="authorized_ai_inpaint_full_mask_retry",
            retry_reason=last_proof.failure_reason or "caption_background_cleanup_incomplete",
        )
        ai_result, ai_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=ai_plan,
            cleanup_mask=cleanup_mask,
            backend_context=backend_context,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=attempt_index,
            strategy="authorized_ai_inpaint_full_mask_retry",
            retry_reason=last_proof.failure_reason or "caption_background_cleanup_incomplete",
            proof_context=proof_context,
        )
        results.append(ai_result)
        proofs.append(ai_proof)
        attempts.append(
            _adaptive_attempt_record(
                ai_result,
                ai_proof,
                attempt_index,
                "authorized_ai_inpaint_full_mask_retry",
                last_proof.failure_reason or "caption_background_cleanup_incomplete",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(ai_plan.cleanup_plan_id),
            attempt_index=attempt_index,
            strategy="authorized_ai_inpaint_full_mask_retry",
            proof_status=_enum_value(ai_proof.proof_status),
            failure_reason=ai_proof.failure_reason or ai_result.failure_reason,
            runtime_ms=ai_result.runtime_ms,
        )
        if _proof_passed_with_pixels(ai_result, ai_proof):
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "early_pass",
                page_id=page_id,
                region_id=region_id,
                attempt_count=len(attempts),
                elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
            )
            return results, proofs, attempts
        if _should_run_residual_retry(ai_proof):
            residual_retry_mask, residual_retry_skip_reason = _residual_retry_cleanup_mask(
                cleanup_mask,
                ai_proof,
            )
            if residual_retry_mask is None:
                attempts.append(
                    {
                        "attempt_index": attempt_index + 1,
                        "strategy": "authorized_ai_residual_retry",
                        "retry_reason": ai_proof.failure_reason or "source_residual_remaining_after_ai",
                        "attempt_skipped": True,
                        "skip_reason": residual_retry_skip_reason,
                    }
                )
                _page014_timeout_checkpoint(
                    "adaptive_attempt",
                    "skipped",
                    page_id=page_id,
                    region_id=region_id,
                    cleanup_plan_id=str(ai_plan.cleanup_plan_id),
                    cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                    attempt_index=attempt_index + 1,
                    strategy="authorized_ai_residual_retry",
                    reason=residual_retry_skip_reason,
                )
            else:
                residual_plan = _adaptive_plan_variant(
                    cleanup_plan,
                    attempt_index=attempt_index + 1,
                    strategy="authorized_ai_residual_retry",
                    cleanup_tag="executable_residual_tone_fill",
                    retry_reason=ai_proof.failure_reason or "source_residual_remaining_after_ai",
                    suffix=f"_attempt{attempt_index + 1}_authorized_ai_residual_retry",
                )
                residual_pixels = int(getattr(residual_retry_mask, "foreground_mask_pixels", 0) or 0)
                residual_plan.backend_parameters["residual_retry_source"] = (
                    "authorized_ai_executable_text_residual_mask"
                )
                residual_plan.backend_parameters["residual_retry_source_context"] = (
                    "previous_cleanup_candidate"
                )
                residual_plan.backend_parameters["residual_retry_foreground_pixels"] = residual_pixels
                _page014_timeout_checkpoint(
                    "adaptive_attempt",
                    "start",
                    page_id=page_id,
                    region_id=region_id,
                    cleanup_plan_id=str(residual_plan.cleanup_plan_id),
                    cleanup_mask_id=str(residual_retry_mask.cleanup_mask_id),
                    attempt_index=attempt_index + 1,
                    strategy="authorized_ai_residual_retry",
                    retry_reason=ai_proof.failure_reason or "source_residual_remaining_after_ai",
                )
                residual_image = _cleanup_result_page_payload(ai_result, source_image) or source_image
                residual_result, residual_proof = _execute_cleanup_attempt(
                    image=residual_image,
                    proof_before=residual_image,
                    source_image=source_image,
                    cleanup_plan=residual_plan,
                    cleanup_mask=residual_retry_mask,
                    backend_context=backend_context,
                    page_id=page_id,
                    region_id=region_id,
                    use_gpu=use_gpu,
                    model_id=model_id,
                    artifact_dir=artifact_dir,
                    attempt_index=attempt_index + 1,
                    strategy="authorized_ai_residual_retry",
                    retry_reason=ai_proof.failure_reason or "source_residual_remaining_after_ai",
                    proof_context=proof_context,
                )
                results.append(residual_result)
                proofs.append(residual_proof)
                attempts.append(
                    _adaptive_attempt_record(
                        residual_result,
                        residual_proof,
                        attempt_index + 1,
                        "authorized_ai_residual_retry",
                        ai_proof.failure_reason or "source_residual_remaining_after_ai",
                    )
                )
                _page014_timeout_checkpoint(
                    "adaptive_attempt",
                    "end",
                    page_id=page_id,
                    region_id=region_id,
                    cleanup_plan_id=str(residual_plan.cleanup_plan_id),
                    attempt_index=attempt_index + 1,
                    strategy="authorized_ai_residual_retry",
                    proof_status=_enum_value(residual_proof.proof_status),
                    failure_reason=residual_proof.failure_reason or residual_result.failure_reason,
                    runtime_ms=residual_result.runtime_ms,
                )
                combined_plan = _adaptive_plan_variant(
                    cleanup_plan,
                    attempt_index=attempt_index + 2,
                    strategy="authorized_ai_residual_retry_combined_parent_proof",
                    cleanup_tag=str(
                        (cleanup_plan.backend_parameters or {}).get("cleanup_tag")
                        or "cleanup_effectiveness_retry"
                    ),
                    retry_reason=ai_proof.failure_reason or "source_residual_remaining_after_ai",
                    suffix=f"_attempt{attempt_index + 1}_authorized_ai_residual_retry_combined",
                )
                combined_result = _combined_residual_retry_result(
                    base_image=image,
                    cleanup_plan=combined_plan,
                    cleanup_mask=cleanup_mask,
                    residual_result=residual_result,
                    primary_result=ai_result,
                    page_id=page_id,
                    region_id=region_id,
                )
                combined_proof = prove_cleanup_result(
                    source_image=source_image,
                    before_image=image,
                    cleanup_result=combined_result,
                    cleanup_plan=combined_plan,
                    cleanup_mask=cleanup_mask,
                    proof_context=proof_context,
                )
                results.append(combined_result)
                proofs.append(combined_proof)
                attempts.append(
                    _adaptive_attempt_record(
                        combined_result,
                        combined_proof,
                        attempt_index + 2,
                        "authorized_ai_residual_retry_combined_parent_proof",
                        ai_proof.failure_reason or "source_residual_remaining_after_ai",
                    )
                )
                if _proof_passed_with_pixels(combined_result, combined_proof):
                    _page014_timeout_checkpoint(
                        "adaptive_cleanup_attempts",
                        "early_pass",
                        page_id=page_id,
                        region_id=region_id,
                        attempt_count=len(attempts),
                        elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
                    )
                    return results, proofs, attempts

    _page014_timeout_checkpoint(
        "adaptive_cleanup_attempts",
        "end",
        page_id=page_id,
        region_id=region_id,
        attempt_count=len(attempts),
        elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
    )
    return results, proofs, attempts


def _execute_cleanup_attempt(
    *,
    image: Any,
    proof_before: Any,
    source_image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    backend_context: Mapping[str, Any] | None = None,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
    attempt_index: int,
    strategy: str,
    retry_reason: str,
    proof_context: Mapping[str, Any] | None = None,
) -> tuple[CleanupResult, CleanupProof]:
    attempt_started = time.time()
    canonical_region_id = _canonical_parent_region_id(cleanup_plan, cleanup_mask, fallback_region_id=region_id)
    _page014_timeout_checkpoint(
        "cleanup_attempt_execute",
        "start",
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        attempt_index=attempt_index,
        strategy=strategy,
        retry_reason=retry_reason,
    )
    result = execute_cleanup_runtime_plan(
        image=image,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
        backend_context=backend_context,
        page_id=page_id,
        region_id=canonical_region_id,
        use_gpu=use_gpu,
        model_id=model_id,
        artifact_dir=artifact_dir,
    )
    result.backend_parameters.setdefault("attempt_index", attempt_index)
    result.backend_parameters.setdefault("strategy", strategy)
    result.backend_parameters.setdefault("retry_reason", retry_reason)
    result.backend_parameters.setdefault("adaptive_cleanup_bounded", True)
    for key in (
        "proof_scope",
        "proof_scope_bbox",
        "parent_cleanup_unit_id",
        "parent_expected_erasure_bbox",
        "source_glyph_erasure_bbox",
        "source_glyph_erasure_expected_area_bbox",
        "partitioned_cleanup",
        "partition_index",
        "partition_count",
        "partition_parent_rejection_reason",
        "partition_parent_cleanup_job_id",
        "represented_source_glyph_ids",
        "consumed_source_glyph_mask_ids",
        "missing_source_glyph_mask_ids",
        "formal_cleanup_plan_id",
        "formal_cleanup_mask_id",
        "cleanup_plan_authority",
        "runtime_plan_derived_from_formal_plan",
        "formal_cleanup_plan",
        "component_projection_method",
        "mask_contract_exception_reason",
        "owned_segmentation_pixels",
        "executable_foreground_pixels",
        "owned_segmentation_to_executable_ratio",
        "inpaint_mode",
        "model_required",
        "model_required_reason",
        "allow_non_model_method",
        "non_model_method_allowed",
        "allow_non_model_method_reason",
        "non_model_method_allowed_reason",
        "authorized_ai_retry_source",
    ):
        if key in cleanup_plan.backend_parameters:
            result.backend_parameters.setdefault(key, cleanup_plan.backend_parameters.get(key))
    _page014_timeout_checkpoint(
        "cleanup_attempt_proof",
        "start",
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_result_id=str(result.cleanup_result_id),
        attempt_index=attempt_index,
        strategy=strategy,
    )
    proof_started = time.time()
    proof = prove_cleanup_result(
        source_image=source_image,
        before_image=proof_before,
        cleanup_result=result,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
        proof_context=proof_context,
    )
    proof_elapsed_ms = round((time.time() - proof_started) * 1000.0, 3)
    result.backend_parameters["proof_elapsed_ms"] = proof_elapsed_ms
    try:
        proof.metrics["proof_elapsed_ms"] = proof_elapsed_ms
    except Exception:
        pass
    _page014_timeout_checkpoint(
        "cleanup_attempt_execute",
        "end",
        page_id=page_id,
        region_id=canonical_region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_result_id=str(result.cleanup_result_id),
        cleanup_proof_id=str(proof.cleanup_proof_id),
        attempt_index=attempt_index,
        strategy=strategy,
        proof_status=_enum_value(proof.proof_status),
        failure_reason=proof.failure_reason or result.failure_reason,
        elapsed_ms=round((time.time() - attempt_started) * 1000.0, 3),
    )
    return result, proof


def _combined_residual_retry_result(
    *,
    base_image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    residual_result: CleanupResult,
    primary_result: CleanupResult | None = None,
    page_id: str,
    region_id: str,
) -> CleanupResult:
    residual_image = _cleanup_result_page_payload(residual_result, base_image)
    primary_image = (
        _cleanup_result_page_payload(primary_result, base_image)
        if primary_result is not None
        else None
    )
    if residual_image is None:
        residual_image = base_image
    if primary_image is None:
        primary_image = base_image
    try:
        base_np = np.asarray(base_image.convert("RGB") if hasattr(base_image, "convert") else base_image)
        primary_np = np.asarray(primary_image.convert("RGB") if hasattr(primary_image, "convert") else primary_image).copy()
        residual_np = np.asarray(residual_image.convert("RGB") if hasattr(residual_image, "convert") else residual_image)
        if base_np.shape == primary_np.shape == residual_np.shape:
            diff = np.abs(residual_np.astype(np.int16) - base_np.astype(np.int16))
            changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
            primary_np[changed] = residual_np[changed]
            cleaned_image = Image.fromarray(primary_np.astype("uint8"), mode="RGB")
        else:
            cleaned_image = residual_image
    except Exception:
        cleaned_image = residual_image
    changed_pixels = _changed_pixel_count(base_image, cleaned_image)
    params = dict(cleanup_plan.backend_parameters or {})
    params.update(dict(residual_result.backend_parameters or {}))
    params.update(
        {
            "candidate_status": "completed" if changed_pixels > 0 else "no_pixel_change",
            "residual_retry_combined_parent_candidate": True,
            "residual_retry_source_context": str(
                (residual_result.backend_parameters or {}).get("residual_retry_source_context")
                or "previous_cleanup_candidate"
            ),
            "residual_retry_merged_with_primary_candidate": primary_result is not None,
            "primary_cleanup_result_id": str(getattr(primary_result, "cleanup_result_id", "") or ""),
            "residual_retry_result_id": str(residual_result.cleanup_result_id),
            "residual_retry_mask_id": str(residual_result.cleanup_mask_id),
            "formal_cleanup_mask_id": str(cleanup_mask.cleanup_mask_id),
            "cleanup_obligation_id": str(cleanup_mask.cleanup_mask_id),
        }
    )
    backend_name = str(residual_result.backend_name or residual_result.execution_backend or "residual_retry")
    identity = _cleanup_identity_fields(cleanup_plan, cleanup_mask, residual_result, fallback_region_id=region_id)
    canonical_region_id = _canonical_parent_region_id(cleanup_plan, cleanup_mask, residual_result, fallback_region_id=region_id)
    return CleanupResult(
        cleanup_result_id=f"cres_{_safe_id(cleanup_plan.cleanup_plan_id)}",
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_job_id=str(cleanup_plan.cleanup_job_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        operation_bbox=_valid_bbox(cleanup_mask.erase_mask_bbox),
        page=page_id,
        region_id=canonical_region_id,
        parent_execution_bundle_id=str(identity.get("parent_execution_bundle_id") or ""),
        parent_logical_text_unit_id=str(identity.get("parent_logical_text_unit_id") or ""),
        text_block_root_id=str(identity.get("text_block_root_id") or ""),
        target_region_ids=list(identity.get("target_region_ids") or []),
        cleanup_class=cleanup_plan.cleanup_class,
        pixel_changed=changed_pixels > 0,
        changed_pixel_count=changed_pixels,
        diff_ref="",
        mask_ref=(primary_result.mask_ref if primary_result is not None and primary_result.mask_ref else residual_result.mask_ref),
        renderer_consumed=False,
        execution_backend=f"{backend_name}|combined_parent_mask",
        execution_status="completed" if changed_pixels > 0 else "completed_no_pixel_change",
        failure_reason="" if changed_pixels > 0 else "no_pixels_changed",
        mask_stats=mask_stats(cleanup_mask.erase_mask) or {},
        backend_name=f"{backend_name}|combined_parent_mask",
        backend_parameters=params,
        runtime_ms=residual_result.runtime_ms,
        fallback_status=residual_result.fallback_status,
        errors=list(residual_result.errors or []),
        result_space="page",
        cleaned_image_ref="",
        cleaned_crop_ref="",
        cleaned_image=cleaned_image,
        cleaned_crop=None,
    )


def _adaptive_plan_variant(
    base_plan: CleanupPlan,
    *,
    attempt_index: int,
    strategy: str,
    cleanup_tag: str,
    retry_reason: str,
    suffix: str,
) -> CleanupPlan:
    params = dict(base_plan.backend_parameters or {})
    params.update(
        {
            "adaptive_cleanup_bounded": True,
            "attempt_index": attempt_index,
            "strategy": strategy,
            "cleanup_tag": cleanup_tag,
            "retry_reason": retry_reason,
            "residual_retry_max": 1,
            "restoration_retry_max": 1,
            "allowed_area_expanded": False,
            "erase_limits_unchanged": True,
        }
    )
    return CleanupPlan(
        cleanup_plan_id=f"{base_plan.cleanup_plan_id}{suffix}",
        cleanup_job_id=base_plan.cleanup_job_id,
        cleanup_mask_id=base_plan.cleanup_mask_id,
        cleanup_class=base_plan.cleanup_class,
        selected_backend=base_plan.selected_backend,
        cleanup_method=base_plan.cleanup_method,
        parent_execution_bundle_id=base_plan.parent_execution_bundle_id,
        parent_logical_text_unit_id=base_plan.parent_logical_text_unit_id,
        text_block_root_id=base_plan.text_block_root_id,
        target_region_ids=list(base_plan.target_region_ids or []),
        backend_parameters=params,
        inpaint_mode=base_plan.inpaint_mode,
        crop_context_bbox=list(base_plan.crop_context_bbox or []),
        fallback_policy=list(base_plan.fallback_policy or []),
        expected_runtime_class=base_plan.expected_runtime_class,
        proof_thresholds=dict(base_plan.proof_thresholds or {}),
    )


def _proof_passed_with_pixels(result: CleanupResult, proof: CleanupProof) -> bool:
    return bool(result.pixel_changed and proof.proof_status == ProofStatus.PASSED)


def _proof_allows_degraded_commit(
    proof: CleanupProof | None,
    result: CleanupResult | None = None,
    status: Mapping[str, Any] | None = None,
) -> bool:
    # Degraded/warning cleanup is diagnostic-only unless a future contract
    # explicitly authorizes non-passed proof commits.
    return False


def _proof_commit_candidate(result: CleanupResult, proof: CleanupProof) -> bool:
    return _proof_passed_with_pixels(result, proof)


def _is_component_projected_cleanup_mask(cleanup_mask: CleanupMask, visual_scope: str | None = None) -> bool:
    scope = _normalise_cleanup_visual_scope(visual_scope or cleanup_mask.visual_scope)
    method = str(getattr(cleanup_mask, "component_projection_method", "") or "")
    return scope == "segmentation_component" or method in COMPONENT_PROJECTION_METHODS


def _state_token_set(value: Any) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _optional_ratio_at_least(value: Any, minimum: float) -> bool:
    if value is None:
        return True
    try:
        return float(value) >= minimum
    except Exception:
        return False


def _component_projection_outside_warning_is_audit_only(
    cleanup_mask: CleanupMask,
    rejection_reason: str = "",
) -> bool:
    if not _is_component_projected_cleanup_mask(cleanup_mask):
        return False
    reason = str(rejection_reason or getattr(cleanup_mask, "rejection_reason", "") or "")
    if reason and reason not in COMPONENT_PROJECTION_AUDIT_ONLY_REJECTION_REASONS:
        return False
    if int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0) <= 0:
        return False
    if int(getattr(cleanup_mask, "erase_mask_pixels", 0) or 0) <= 0:
        return False
    if bool(getattr(cleanup_mask, "protected", False)):
        return False
    if getattr(cleanup_mask, "foreground_mask", None) is None or getattr(cleanup_mask, "erase_mask", None) is None:
        return False
    if bool(getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "segmentation_contract_override_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "bbox_executable_foreground_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "page_level_executable_foreground_detected", False)):
        return False

    protected_ids = [item for item in getattr(cleanup_mask, "protected_component_ids", []) or [] if str(item)]
    unowned_ids = [item for item in getattr(cleanup_mask, "unowned_component_ids", []) or [] if str(item)]
    if protected_ids or unowned_ids:
        return False
    if int(getattr(cleanup_mask, "protected_component_pixel_count", 0) or 0) > 0:
        return False
    if int(getattr(cleanup_mask, "unowned_component_pixels", 0) or 0) > 0:
        return False

    owned_ids = [item for item in getattr(cleanup_mask, "owned_component_ids", []) or [] if str(item)]
    owned_component_pixels = int(getattr(cleanup_mask, "owned_component_pixel_count", 0) or 0)
    if not owned_ids and owned_component_pixels <= 0:
        return False
    owned_pixels = int(
        owned_component_pixels
        or getattr(cleanup_mask, "owned_segmentation_pixels", None)
        or getattr(cleanup_mask, "foreground_mask_pixels", 0)
        or 0
    )

    projection_tokens = _state_token_set(getattr(cleanup_mask, "projection_quality_state", ""))
    if projection_tokens:
        allowed_projection_tokens = {PROJECTION_READY_STATE, PROJECTION_OUTSIDE_AUTHORIZED_AREA_STATE}
        if PROJECTION_READY_STATE not in projection_tokens:
            return False
        if not projection_tokens.issubset(allowed_projection_tokens):
            return False

    readiness_tokens = _state_token_set(getattr(cleanup_mask, "mask_readiness_state", ""))
    if readiness_tokens:
        allowed_readiness_tokens = {MASK_READY_STATE, MASK_NOT_READY_STATE}
        if MASK_READY_STATE not in readiness_tokens:
            return False
        if not readiness_tokens.issubset(allowed_readiness_tokens):
            return False

    failure_tokens = _state_token_set(getattr(cleanup_mask, "mask_readiness_failure_reason", ""))
    if failure_tokens and not failure_tokens.issubset({PROJECTION_OUTSIDE_AUTHORIZED_AREA_STATE}):
        return False

    rejected_component_reasons = {
        str(item)
        for item in getattr(cleanup_mask, "rejected_component_reasons", []) or []
        if str(item)
    }
    if rejected_component_reasons and not rejected_component_reasons.issubset(
        COMPONENT_PROJECTION_AUDIT_ONLY_REJECTED_COMPONENT_REASONS
    ):
        return False

    ambiguous_pixels = int(
        getattr(cleanup_mask, "ambiguous_component_pixel_count", None)
        or getattr(cleanup_mask, "ambiguous_component_pixels", None)
        or 0
    )
    if ambiguous_pixels > 0 and ambiguous_pixels / float(max(1, owned_pixels)) > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO:
        return False

    if not _optional_ratio_at_least(
        getattr(cleanup_mask, "effective_coverage_ratio", None),
        PROOF_MASK_SOURCE_OVERLAP_RATIO,
    ):
        return False
    if not _optional_ratio_at_least(
        getattr(cleanup_mask, "text_block_coverage_estimate", None),
        PROOF_MASK_SOURCE_OVERLAP_RATIO,
    ):
        return False
    return True


def _is_component_projection_ready_audit_rejection(cleanup_mask: CleanupMask, rejection_reason: str) -> bool:
    if str(rejection_reason or "") not in {
        "effective_mask_not_ready",
        "effective_mask_incomplete_under_coverage",
        "effective_mask_fragment_only",
        "effective_mask_protected_overlap_removed",
        "upstream_container_mismatch",
    }:
        return False
    if not _is_component_projected_cleanup_mask(cleanup_mask):
        return False
    if _component_projection_outside_warning_is_audit_only(cleanup_mask, rejection_reason):
        return True
    if str(getattr(cleanup_mask, "projection_quality_state", "") or "") != "projection_ready":
        return False
    if str(getattr(cleanup_mask, "mask_readiness_state", "") or "") != "mask_ready":
        return False
    if int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0) <= 0:
        return False
    if int(getattr(cleanup_mask, "erase_mask_pixels", 0) or 0) <= 0:
        return False
    return True


def _should_run_backend_noop_fallback(result: CleanupResult, proof: CleanupProof | None) -> bool:
    if not result.pixel_changed or int(result.changed_pixel_count or 0) <= 0:
        return True
    candidate_status = str((result.backend_parameters or {}).get("candidate_status") or "")
    if candidate_status in {"no_pixel_change", "completed_no_pixel_change"}:
        return True
    reason = str((proof.failure_reason if proof is not None else "") or result.failure_reason or "")
    return reason in {"backend_noop_or_error", "no_pixels_changed"}


def _residual_retry_cleanup_mask(
    cleanup_mask: CleanupMask,
    proof: CleanupProof,
) -> tuple[CleanupMask | None, str]:
    if np is None or Image is None:
        return None, "numpy_or_pillow_unavailable"
    metrics = proof.metrics or {}
    residual_ref = str(metrics.get("residual_source_mask_ref") or "")
    if not residual_ref:
        return None, "residual_retry_skipped_no_executable_text_residual_ref"
    if not os.path.isfile(residual_ref):
        return None, "residual_retry_skipped_residual_text_mask_missing"
    foreground = _binary_cleanup_mask_array(cleanup_mask.foreground_mask)
    erase = _binary_cleanup_mask_array(cleanup_mask.erase_mask)
    if foreground is None or not np.any(foreground > 0):
        return None, "residual_retry_skipped_missing_cleanup_foreground"
    try:
        with Image.open(residual_ref) as img:
            residual_arr = np.asarray(img.convert("L"))
    except Exception as exc:
        return None, f"residual_retry_skipped_residual_text_mask_load_error:{type(exc).__name__}"
    residual = _binary_mask(residual_arr, foreground.shape)
    if residual is None:
        return None, "residual_retry_skipped_invalid_residual_text_mask"
    residual = (residual > 0) & (foreground > 0)
    if erase is not None and erase.shape == foreground.shape and np.any(erase > 0):
        residual = residual & (erase > 0)
    residual_u8 = residual.astype(np.uint8)
    residual_pixels = int(np.count_nonzero(residual_u8))
    if residual_pixels <= 0:
        return None, "residual_retry_skipped_no_executable_text_residual"
    residual_bbox = _mask_bbox(residual_u8)
    if residual_bbox is None:
        return None, "residual_retry_skipped_empty_residual_text_bbox"
    return (
        replace(
            cleanup_mask,
            cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_residual_retry",
            foreground_mask_bbox=residual_bbox,
            foreground_mask_pixels=residual_pixels,
            erase_mask_bbox=residual_bbox,
            erase_mask_pixels=residual_pixels,
            growth_ratio=1.0,
            mask_method=f"{cleanup_mask.mask_method}|executable_text_residual_retry",
            rejection_reason="",
            visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
            foreground_mask=residual_u8,
            erase_mask=residual_u8.copy(),
        ),
        "",
    )


def _should_run_residual_retry(proof: CleanupProof) -> bool:
    reason = str(proof.failure_reason or "")
    if reason not in {
        "source_residual_remaining",
        "source_residual_remaining_in_source_glyph_erasure_bbox",
        "source_residual_remaining_in_component_partition_bbox",
    }:
        return False
    metrics = proof.metrics or {}
    residual_pixels = _first_metric_int(
        metrics,
        "executable_text_residual_pixels",
        "residual_dark_source_pixels",
        "residual_source_pixels",
        default=0,
    )
    if residual_pixels <= 0:
        return False
    residual_ref = str(metrics.get("residual_source_mask_ref") or "")
    if not residual_ref:
        return False
    overlap = _float_or_none(metrics.get("mask_source_overlap_ratio"))
    if overlap is not None and overlap < PROOF_MASK_SOURCE_OVERLAP_RATIO:
        return False
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    changed_outside_ratio = _float_or_none(metrics.get("changed_outside_allowed_ratio")) or 0.0
    collateral = (
        _float_or_none(metrics.get("accepted_mask_collateral_ratio"))
        or _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio"))
        or 0.0
    )
    return (
        changed_outside <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
        and changed_outside_ratio <= PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        and collateral <= PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
    )


def _first_metric_int(metrics: Mapping[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = _int_or_none(metrics.get(key))
        if value is not None:
            return int(value)
    return int(default)


def _should_run_restoration_retry(proof: CleanupProof) -> bool:
    reason = str(proof.failure_reason or "")
    if reason in {"changed_outside_accepted_mask", "changed_outside_allowed_area", "collateral_change_too_broad"}:
        return True
    metrics = proof.metrics or {}
    changed_outside = (
        _int_or_none(metrics.get("changed_outside_accepted_mask_pixels"))
        or _int_or_none(metrics.get("changed_outside_allowed_pixels"))
        or 0
    )
    changed_outside_ratio = (
        _float_or_none(metrics.get("changed_outside_accepted_mask_ratio"))
        or _float_or_none(metrics.get("changed_outside_allowed_ratio"))
        or 0.0
    )
    collateral = (
        _float_or_none(metrics.get("accepted_mask_collateral_ratio"))
        or _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio"))
        or 0.0
    )
    return (
        changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
        or changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        or collateral > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
    )


def _should_run_authorized_ai_inpaint_retry(
    cleanup_plan: CleanupPlan,
    proof: CleanupProof,
) -> bool:
    if proof is None or proof.proof_status == ProofStatus.PASSED:
        return False
    cleanup_value = _enum_value(getattr(cleanup_plan, "cleanup_class", ""))
    if cleanup_value not in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }:
        return False
    reason = str(proof.failure_reason or "")
    if reason in {
        "source_residual_remaining",
        "source_residual_outside_accepted_mask_boundary",
        "broad_white_patch_risk",
    }:
        return True
    metrics = proof.metrics or {}
    return bool(
        metrics.get("outside_accepted_mask_boundary_residual_remaining")
        or metrics.get("broad_white_patch_risk")
    )


def _should_run_mask_faithful_inpaint_retry(
    cleanup_plan: CleanupPlan,
    proof: CleanupProof,
) -> bool:
    if proof is None or proof.proof_status == ProofStatus.PASSED:
        return False
    cleanup_value = _enum_value(getattr(cleanup_plan, "cleanup_class", ""))
    if cleanup_value not in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }:
        return False
    reason = str(proof.failure_reason or "")
    if reason in {
        "source_residual_remaining",
        "source_residual_outside_accepted_mask_boundary",
        "broad_white_patch_risk",
        "backend_noop_or_error",
    }:
        return True
    metrics = proof.metrics or {}
    return bool(
        metrics.get("dark_source_residual_remaining")
        or metrics.get("light_source_residual_remaining")
        or metrics.get("cleaned_light_source_residual_remaining")
        or metrics.get("broad_white_patch_risk")
    )


def _adaptive_attempt_record(
    result: CleanupResult,
    proof: CleanupProof,
    attempt_index: int,
    strategy: str,
    retry_reason: str,
) -> dict[str, Any]:
    metrics = proof.metrics or {}
    return {
        "attempt_index": attempt_index,
        "strategy": strategy,
        "backend": result.backend_name or result.execution_backend,
        "cleanup_result_id": result.cleanup_result_id,
        "cleanup_proof_id": proof.cleanup_proof_id,
        "proof_status": _enum_value(proof.proof_status),
        "failure_reason": proof.failure_reason or result.failure_reason,
        "pixel_changed": bool(result.pixel_changed),
        "cleaned_image_ref": result.cleaned_image_ref,
        "diff_ref": result.diff_ref,
        "mask_ref": result.mask_ref,
        "runtime_ms": result.runtime_ms,
        "residual_status": _residual_status(metrics, proof),
        "collateral_status": _collateral_status(metrics, proof),
        "allowed_area_mutation_status": _allowed_area_status(metrics),
        "proof_scope": metrics.get("proof_scope", ""),
        "proof_scope_bbox": metrics.get("proof_scope_bbox", []),
        "parent_cleanup_unit_id": metrics.get("parent_cleanup_unit_id", ""),
        "parent_expected_erasure_bbox": metrics.get("parent_expected_erasure_bbox", []),
        "proof_quality_state": metrics.get("proof_quality_state", ""),
        "degraded_commit_allowed": False,
        "degraded_cleanup_diagnostic": bool(metrics.get("degraded_cleanup_diagnostic", False)),
        "source_residual_delta_dark_pixels": metrics.get("source_residual_delta_dark_pixels", 0),
        "source_residual_delta_dark_ratio": metrics.get("source_residual_delta_dark_ratio", 0),
        "collateral_art_damage_delta": metrics.get("collateral_art_damage_delta", 0),
        "broad_white_patch_risk": bool(metrics.get("broad_white_patch_risk", False)),
        "commit_candidate": _proof_commit_candidate(result, proof),
        "retry_reason": retry_reason,
    }


def _selected_attempt_record(
    attempt_records: Sequence[dict[str, Any]],
    cleanup_result: CleanupResult | None,
) -> dict[str, Any]:
    if cleanup_result is None:
        return {}
    result_id = str(cleanup_result.cleanup_result_id)
    for record in attempt_records:
        if str(record.get("cleanup_result_id") or "") == result_id:
            return dict(record)
    return {}


def _residual_status(metrics: Mapping[str, Any], proof: CleanupProof) -> str:
    if proof.proof_status == ProofStatus.PASSED:
        return "passed"
    reason = str(proof.failure_reason or "")
    if "source_residual" in reason:
        return "failed"
    return "not_primary_failure"


def _white_patch_context_metrics(
    *,
    source_gray: Any,
    white_patch_mask: Any,
    erase_mask: Any,
) -> dict[str, Any]:
    if np is None or source_gray is None or white_patch_mask is None:
        return {
            "median_luma": 0.0,
            "light_ratio": 0.0,
            "dark_ratio": 0.0,
            "light_context": False,
        }
    try:
        source_arr = np.asarray(source_gray)
        white_patch = np.asarray(white_patch_mask) > 0
        erase = np.asarray(erase_mask) > 0 if erase_mask is not None else white_patch
    except Exception:
        return {
            "median_luma": 0.0,
            "light_ratio": 0.0,
            "dark_ratio": 0.0,
            "light_context": False,
        }
    if source_arr.ndim != 2 or white_patch.shape != source_arr.shape or not np.any(white_patch):
        return {
            "median_luma": 0.0,
            "light_ratio": 0.0,
            "dark_ratio": 0.0,
            "light_context": False,
        }
    if erase.shape != source_arr.shape:
        erase = white_patch
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        near = cv2.dilate(white_patch.astype(np.uint8), kernel, iterations=1).astype(bool)
        far = cv2.dilate(white_patch.astype(np.uint8), kernel, iterations=3).astype(bool)
        context_mask = far & ~near & ~erase
    else:
        y_idx, x_idx = np.where(white_patch)
        y0 = max(0, int(y_idx.min()) - 3)
        y1 = min(source_arr.shape[0], int(y_idx.max()) + 4)
        x0 = max(0, int(x_idx.min()) - 3)
        x1 = min(source_arr.shape[1], int(x_idx.max()) + 4)
        context_mask = np.zeros(source_arr.shape, dtype=bool)
        context_mask[y0:y1, x0:x1] = True
        context_mask &= ~white_patch & ~erase
    values = source_arr[context_mask]
    if values.size < 12:
        values = source_arr[(~erase) & (~white_patch)]
    if values.size <= 0:
        return {
            "median_luma": 0.0,
            "light_ratio": 0.0,
            "dark_ratio": 0.0,
            "light_context": False,
        }
    median_luma = float(np.median(values))
    light_ratio = float(np.count_nonzero(values >= 225) / max(1, values.size))
    dark_ratio = float(np.count_nonzero(values < 120) / max(1, values.size))
    light_context = median_luma >= 235.0 and light_ratio >= 0.50 and dark_ratio <= 0.35
    return {
        "median_luma": round(median_luma, 3),
        "light_ratio": round(light_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
        "light_context": bool(light_context),
    }


def _collateral_status(metrics: Mapping[str, Any], proof: CleanupProof) -> str:
    reason = str(proof.failure_reason or "")
    if reason in {"changed_outside_accepted_mask", "changed_outside_allowed_area", "collateral_change_too_broad"}:
        return "failed"
    changed_outside = (
        _int_or_none(metrics.get("changed_outside_accepted_mask_pixels"))
        or _int_or_none(metrics.get("changed_outside_allowed_pixels"))
        or 0
    )
    collateral = (
        _float_or_none(metrics.get("accepted_mask_collateral_ratio"))
        or _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio"))
        or 0.0
    )
    if changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS or collateral > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO:
        return "failed"
    return "passed_or_not_primary"


def _allowed_area_status(metrics: Mapping[str, Any]) -> str:
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    changed_outside_ratio = _float_or_none(metrics.get("changed_outside_allowed_ratio")) or 0.0
    if changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS or changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO:
        return "changed_outside_allowed_area"
    return "contained"


def _cleanup_outcome_state_for_failure(failure_reason: str, *, runtime_status: str = "blocked") -> str:
    reason = str(failure_reason or "").lower()
    if "segmentation_foreground_contract_defect" in reason:
        return "segmentation_foreground_contract_defect"
    if "sourceglyph_generation_or_foreground_contract_defect" in reason:
        return "sourceglyph_generation_or_foreground_contract_defect"
    if runtime_status == "passed" or reason == "passed":
        return "cleanup_completed"
    if runtime_status == "warning":
        if "source_residual" in reason:
            return "safe_partial_attempt_uncommitted"
        return "cleanup_partially_completed_with_warnings"
    if reason.startswith("cleanup_contract_error_"):
        return reason
    if "component_limit" in reason:
        return "cleanup_contract_error_component_limit"
    if "invalid" in reason or "expected_exactly_one" in reason or "missing_result_or_proof" in reason:
        return "cleanup_contract_error_invalid_input"
    if "source_residual" in reason:
        return "cleanup_restoration_attempted_with_warning"
    if "backend_noop" in reason or "no_pixels_changed" in reason or "no_pixel_change" in reason:
        return "cleanup_failed_no_safe_pixel_change"
    if "sourceglyph" in reason or "source_glyph" in reason or "source_ungrounded" in reason:
        return "cleanup_contract_error_missing_sourceglyph"
    if "erase" in reason or "too_large" in reason:
        return "cleanup_degraded_minimal_erasure_applied"
    if "mask" in reason and "source_residual" not in reason:
        return "cleanup_contract_error_invalid_input"
    if "collateral" in reason or "destructive" in reason or "outside_allowed" in reason:
        return "cleanup_restoration_attempted_with_warning"
    if "commit" in reason:
        return "cleanup_partially_completed_with_warnings"
    if "non_phase5" in reason or "unsupported" in reason or "runtime_supported" in reason:
        return "cleanup_contract_error_invalid_input"
    if "protected" in reason or "preserve" in reason or "sfx" in reason or "decorative" in reason:
        return "cleanup_contract_error_invalid_input"
    if "artifact_risk" in reason:
        return "cleanup_partially_completed_with_warnings"
    return "cleanup_partially_completed_with_warnings"


def _contract_error_for_reason(reason: str) -> str:
    text = str(reason or "").lower()
    if "segmentation_foreground_contract_defect" in text:
        return "segmentation_foreground_contract_defect"
    if "sourceglyph_generation_or_foreground_contract_defect" in text:
        return "sourceglyph_generation_or_foreground_contract_defect"
    if "component_limit" in text:
        return "cleanup_contract_error_component_limit"
    if (
        "sourceglyph" in text
        or "source_glyph" in text
        or "source_ungrounded" in text
        or "source_text_missing" in text
        or "consumable_sourceglyph" in text
    ):
        return "cleanup_contract_error_missing_sourceglyph"
    if "empty" in text or "missing" in text or "invalid" in text or "expected_exactly_one" in text:
        return "cleanup_contract_error_invalid_input"
    if "bbox" in text or "allowed_area" in text or "dimensions" in text:
        return "cleanup_contract_error_invalid_input"
    if "preserve" in text or "sfx" in text or "decorative" in text or "protected" in text:
        return "cleanup_contract_error_invalid_input"
    if "unsupported" in text or "non_phase5" in text or "runtime_supported" in text:
        return "cleanup_contract_error_invalid_input"
    return "cleanup_contract_error_invalid_input"


def _changed_pixel_count(before: Any, after: Any) -> int:
    if np is None:
        return 0
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
        if before_np.shape != after_np.shape:
            return 0
        diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
        return int(np.count_nonzero(np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8))
    except Exception:
        return 0


def _write_runtime_artifacts(
    *,
    artifact_dir: str | None,
    page_id: str,
    region_id: str,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    before: Any,
    backend_input_mask: Any,
    raw_cleaned: Any,
    cleaned: Any,
    commit_mask: Any | None = None,
) -> dict[str, str]:
    if not artifact_dir or Image is None:
        return {}
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        stem = _short_artifact_stem(page_id, region_id, cleanup_plan.cleanup_plan_id)
        cleaned_path = os.path.join(artifact_dir, f"{stem}_cleaned.png")
        diff_path = os.path.join(artifact_dir, f"{stem}_diff.png")
        mask_path = os.path.join(artifact_dir, f"{stem}_mask.png")
        backend_input_mask_path = os.path.join(artifact_dir, f"{stem}_backend_input_mask.png")
        model_input_image_path = os.path.join(artifact_dir, f"{stem}_model_input_image.png")
        model_input_mask_path = os.path.join(artifact_dir, f"{stem}_model_input_mask.png")
        raw_backend_output_path = os.path.join(artifact_dir, f"{stem}_raw_backend_output.png")
        final_clipped_output_path = os.path.join(artifact_dir, f"{stem}_final_clipped_output.png")
        crop_path = os.path.join(artifact_dir, f"{stem}_cleaned_crop.png")
        _save_image(before, model_input_image_path)
        _save_mask(backend_input_mask, backend_input_mask_path)
        _save_mask(backend_input_mask, model_input_mask_path)
        _save_image(raw_cleaned, raw_backend_output_path)
        _save_image(cleaned, cleaned_path)
        _save_image(cleaned, final_clipped_output_path)
        _save_diff(before, cleaned, diff_path)
        _save_mask(commit_mask if commit_mask is not None else cleanup_mask.erase_mask, mask_path)
        _save_crop(cleaned, cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area, crop_path)
        return {
            "cleaned_image_ref": cleaned_path,
            "cleaned_crop_ref": crop_path,
            "diff_ref": diff_path,
            "mask_ref": mask_path,
            "backend_input_mask_ref": backend_input_mask_path,
            "model_input_image_ref": model_input_image_path,
            "model_input_mask_ref": model_input_mask_path,
            "raw_backend_output_ref": raw_backend_output_path,
            "final_clipped_output_ref": final_clipped_output_path,
        }
    except Exception:
        return {}


def _save_image(image: Any, path: str) -> None:
    write_debug_image_file(path, image)


def _save_diff(before: Any, after: Any, path: str) -> None:
    if np is None or Image is None:
        return
    before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
    after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
    if before_np.shape != after_np.shape:
        return
    diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
    if diff.ndim == 3:
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
    else:
        diff_gray = diff.astype(np.uint8)
    write_debug_image_file(path, Image.fromarray(diff_gray))


def _save_mask(mask: Any, path: str) -> None:
    if np is None or Image is None or mask is None:
        return
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return
    write_debug_image_file(path, Image.fromarray((arr > 0).astype(np.uint8) * 255))


def _save_crop(image: Any, bbox: Sequence[int] | None, path: str) -> None:
    box = _valid_bbox(bbox)
    if box is None or not hasattr(image, "crop"):
        return
    write_debug_image_file(path, image.crop(tuple(box)))


def _clip_mask_to_valid_bbox(mask: Any, bbox: Sequence[int]) -> Any:
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    box = _valid_bbox(bbox)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    height, width = out.shape[:2]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _job_exclusion_reason(job: CleanupJob) -> str:
    if bool(job.protected):
        return job.protection_reason or "job_protected"
    combined = " ".join(
        str(value or "")
        for value in (
            _enum_value(job.cleanup_class),
            job.route_intent,
            job.semantic_class,
            job.cleanup_mode,
            job.container_type,
            " ".join(job.source_glyph_mask_ids or []),
        )
    ).lower()
    for blocked in (
        "preserve",
        "sfx",
        "decorative",
        "non_text",
        "non_translation_art",
        "art_only",
        "source_grounding_protected",
        "source_ungrounded",
    ):
        if blocked in combined:
            return f"cleanup_excludes_{blocked}"
    return ""


def _mask_exclusion_reason(cleanup_mask: CleanupMask, image_size: tuple[int, int] | None) -> str:
    if cleanup_mask.protected:
        return cleanup_mask.protection_reason or "mask_protected"
    if _is_diagnostic_non_segmentation_cleanup_mask(cleanup_mask):
        return "diagnostic_non_segmentation_fallback_not_executable"
    protected_conflict = _mask_protected_conflict_reason(cleanup_mask)
    if protected_conflict:
        return protected_conflict
    visual_scope = _normalise_cleanup_visual_scope(cleanup_mask.visual_scope)
    if not visual_scope:
        return "visual_scope_missing"
    if visual_scope not in VALID_CLEANUP_VISUAL_SCOPES:
        return f"unsupported_visual_scope_{_safe_reason_token(visual_scope)}"
    if not str(cleanup_mask.cleanup_job_id or "").strip():
        return "cleanup_job_id_missing"
    consumed_source_ids = _unique_texts(cleanup_mask.consumed_source_glyph_mask_ids)
    missing_source_ids = _unique_texts(cleanup_mask.missing_source_glyph_mask_ids)
    component_projected = _is_component_projected_cleanup_mask(cleanup_mask, visual_scope)
    rejection_reason = str(getattr(cleanup_mask, "rejection_reason", "") or "")
    if rejection_reason and not _is_component_projection_ready_audit_rejection(cleanup_mask, rejection_reason):
        return f"mask_rejected_{rejection_reason}"
    if missing_source_ids and not component_projected:
        return "missing_required_source_glyph_evidence"
    if not consumed_source_ids and not component_projected:
        return "consumed_source_glyph_evidence_missing"
    if visual_scope == "source_glyph_local" and len(consumed_source_ids) != 1 and not component_projected:
        return "source_glyph_local_membership_count_mismatch"
    if visual_scope == "source_glyph_union" and len(consumed_source_ids) <= 1 and not component_projected:
        return "source_glyph_union_membership_count_mismatch"
    if visual_scope == "source_glyph_union_partition" and len(consumed_source_ids) <= 1 and not component_projected:
        return "source_glyph_union_partition_membership_count_mismatch"
    if cleanup_mask.foreground_mask is None or cleanup_mask.erase_mask is None:
        return "mask_raw_arrays_missing"
    if not _valid_bbox(cleanup_mask.allowed_area):
        return "allowed_area_missing_or_invalid"
    if not _valid_bbox(cleanup_mask.erase_mask_bbox):
        return "erase_mask_bbox_missing_or_invalid"
    if bool(getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)):
        return "sourceglyph_executable_influence_detected"
    if bool(getattr(cleanup_mask, "segmentation_contract_override_detected", False)):
        return "segmentation_contract_override_detected"
    # Sparse component coverage is a cleanup-quality warning. It must not
    # prevent an otherwise executable parent mask from reaching the backend.
    rejection_reason = str(getattr(cleanup_mask, "rejection_reason", "") or "")
    if rejection_reason in {
        "effective_mask_incomplete_under_coverage",
        "effective_mask_fragment_only",
        "segmentation_mask_wrong_owner_broad_background_capture",
        "effective_mask_failed_unsafe_background_capture",
    }:
        if not _is_component_projection_ready_audit_rejection(cleanup_mask, rejection_reason):
            return rejection_reason
    if (
        str(getattr(cleanup_mask, "mask_source", "") or "") != "cleanup_mask_from_text_foreground_segmentation"
        and component_projected is False
    ):
        return "non_segmentation_cleanup_mask_not_production_authority"
    if cleanup_mask.artifact_risk and not _is_partition_component_mask(cleanup_mask):
        return ARTIFACT_RISK_PARTITION_REQUIRED_REASON
    growth = float(cleanup_mask.growth_ratio if cleanup_mask.growth_ratio is not None else 999.0)
    erase_pixels = int(cleanup_mask.erase_mask_pixels or 0)
    allowed_area = _bbox_area(cleanup_mask.allowed_area)
    erase_bbox_area = _bbox_area(cleanup_mask.erase_mask_bbox)
    growth_exception = str(getattr(cleanup_mask, "mask_contract_exception_reason", "") or "") in {
        "caption_flat_small_mask_growth_exception",
        "speech_flat_runtime_dark_pixel_growth_exception",
        "text_area_component_authorization_map_growth_exception",
    }
    if growth > MAX_PLAN_GROWTH_RATIO and not growth_exception:
        return "growth_ratio_too_high"
    if erase_pixels <= 0:
        return "erase_mask_empty"
    if erase_pixels > MAX_ERASE_MASK_PIXELS:
        return "erase_mask_too_large"
    if erase_bbox_area > MAX_ERASE_BBOX_AREA:
        return "erase_bbox_too_broad"
    if erase_pixels / max(1, allowed_area) > MAX_ERASE_ALLOWED_RATIO:
        return "erase_allowed_ratio_too_high"
    if image_size and image_size[0] > 0 and image_size[1] > 0:
        page_area = int(image_size[0]) * int(image_size[1])
        if allowed_area / max(1, page_area) > MAX_ALLOWED_PAGE_RATIO:
            return "allowed_area_too_broad"
    return ""


def _runtime_structural_mask_error(cleanup_mask: CleanupMask) -> str:
    """Return only hard contract errors that prevent executing an accepted mask.

    Risk signals such as artifact risk, growth ratio, broad bbox, and allowed
    area size are proof/audit concerns for an already accepted CleanupMask.
    They must not turn a formal cleanup obligation into a hidden partition or a
    dropped runtime task before the exact mask reaches the backend.
    """

    if cleanup_mask.protected:
        return cleanup_mask.protection_reason or "mask_protected"
    if _is_diagnostic_non_segmentation_cleanup_mask(cleanup_mask):
        return "diagnostic_non_segmentation_fallback_not_executable"
    protected_conflict = _mask_protected_conflict_reason(cleanup_mask)
    if protected_conflict:
        return protected_conflict
    visual_scope = _normalise_cleanup_visual_scope(cleanup_mask.visual_scope)
    if not visual_scope:
        return "visual_scope_missing"
    if visual_scope not in VALID_CLEANUP_VISUAL_SCOPES:
        return f"unsupported_visual_scope_{_safe_reason_token(visual_scope)}"
    if not str(cleanup_mask.cleanup_job_id or "").strip():
        return "cleanup_job_id_missing"
    component_projected = _is_component_projected_cleanup_mask(cleanup_mask, visual_scope)
    rejection_reason = str(getattr(cleanup_mask, "rejection_reason", "") or "")
    if rejection_reason and not _is_component_projection_ready_audit_rejection(cleanup_mask, rejection_reason):
        return f"mask_rejected_{rejection_reason}"
    consumed_source_ids = _unique_texts(cleanup_mask.consumed_source_glyph_mask_ids)
    missing_source_ids = _unique_texts(cleanup_mask.missing_source_glyph_mask_ids)
    if missing_source_ids and not component_projected:
        return "missing_required_source_glyph_evidence"
    if not consumed_source_ids and not component_projected:
        return "consumed_source_glyph_evidence_missing"
    if visual_scope == "source_glyph_local" and len(consumed_source_ids) != 1 and not component_projected:
        return "source_glyph_local_membership_count_mismatch"
    if visual_scope == "source_glyph_union" and len(consumed_source_ids) <= 1 and not component_projected:
        return "source_glyph_union_membership_count_mismatch"
    if visual_scope == "source_glyph_union_partition" and len(consumed_source_ids) <= 1 and not component_projected:
        return "source_glyph_union_partition_membership_count_mismatch"
    if cleanup_mask.foreground_mask is None or cleanup_mask.erase_mask is None:
        return "mask_raw_arrays_missing"
    if not _valid_bbox(cleanup_mask.allowed_area):
        return "allowed_area_missing_or_invalid"
    if not _valid_bbox(cleanup_mask.erase_mask_bbox):
        return "erase_mask_bbox_missing_or_invalid"
    if bool(getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)):
        return "sourceglyph_executable_influence_detected"
    if bool(getattr(cleanup_mask, "segmentation_contract_override_detected", False)):
        return "segmentation_contract_override_detected"
    # Sparse component coverage is recorded on the mask/plan for audit, but it
    # is not a structural runtime defect.
    if int(cleanup_mask.erase_mask_pixels or 0) <= 0:
        return "erase_mask_empty"
    return ""


def _accepted_executable_cleanup_mask(cleanup_mask: CleanupMask) -> bool:
    if cleanup_mask is None:
        return False
    if bool(getattr(cleanup_mask, "protected", False)):
        return False
    if _is_diagnostic_non_segmentation_cleanup_mask(cleanup_mask):
        return False
    if getattr(cleanup_mask, "foreground_mask", None) is None or getattr(cleanup_mask, "erase_mask", None) is None:
        return False
    if int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0) <= 0:
        return False
    if int(getattr(cleanup_mask, "erase_mask_pixels", 0) or 0) <= 0:
        return False
    rejection_reason = str(getattr(cleanup_mask, "rejection_reason", "") or "")
    audit_only_projection_warning = _component_projection_outside_warning_is_audit_only(
        cleanup_mask,
        rejection_reason,
    )
    projection_state = str(getattr(cleanup_mask, "projection_quality_state", "") or "")
    if projection_state and projection_state != PROJECTION_READY_STATE and not audit_only_projection_warning:
        return False
    readiness_state = str(getattr(cleanup_mask, "mask_readiness_state", "") or "")
    if readiness_state and readiness_state != MASK_READY_STATE and not audit_only_projection_warning:
        return False
    if rejection_reason and not _is_component_projection_ready_audit_rejection(cleanup_mask, rejection_reason):
        return False
    if bool(getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "segmentation_contract_override_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "bbox_executable_foreground_detected", False)):
        return False
    if bool(getattr(cleanup_mask, "page_level_executable_foreground_detected", False)):
        return False
    return True


def _cleanup_mask_non_executable_defect(cleanup_mask: CleanupMask) -> str:
    if cleanup_mask is None:
        return "cleanup_mask_missing"
    if bool(getattr(cleanup_mask, "protected", False)):
        return getattr(cleanup_mask, "protection_reason", "") or "cleanup_mask_protected"
    if _is_diagnostic_non_segmentation_cleanup_mask(cleanup_mask):
        return "diagnostic_non_segmentation_fallback_not_executable"
    if getattr(cleanup_mask, "foreground_mask", None) is None or getattr(cleanup_mask, "erase_mask", None) is None:
        return "mask_raw_arrays_missing"
    if int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0) <= 0:
        return "foreground_mask_empty"
    if int(getattr(cleanup_mask, "erase_mask_pixels", 0) or 0) <= 0:
        return "erase_mask_empty"
    rejection_reason = str(getattr(cleanup_mask, "rejection_reason", "") or "")
    audit_only_projection_warning = _component_projection_outside_warning_is_audit_only(
        cleanup_mask,
        rejection_reason,
    )
    projection_state = str(getattr(cleanup_mask, "projection_quality_state", "") or "")
    if projection_state and projection_state != PROJECTION_READY_STATE and not audit_only_projection_warning:
        return f"projection_not_ready_{_safe_reason_token(projection_state)}"
    readiness_state = str(getattr(cleanup_mask, "mask_readiness_state", "") or "")
    if readiness_state and readiness_state != MASK_READY_STATE and not audit_only_projection_warning:
        return f"mask_not_ready_{_safe_reason_token(readiness_state)}"
    if rejection_reason and not _is_component_projection_ready_audit_rejection(cleanup_mask, rejection_reason):
        return f"mask_rejected_{rejection_reason}"
    if bool(getattr(cleanup_mask, "sourceglyph_executable_influence_detected", False)):
        return "sourceglyph_executable_influence_detected"
    if bool(getattr(cleanup_mask, "segmentation_contract_override_detected", False)):
        return "segmentation_contract_override_detected"
    if bool(getattr(cleanup_mask, "bbox_executable_foreground_detected", False)):
        return "bbox_executable_foreground_detected"
    if bool(getattr(cleanup_mask, "page_level_executable_foreground_detected", False)):
        return "page_level_executable_foreground_detected"
    return "cleanup_mask_not_executable"


def _is_diagnostic_non_segmentation_cleanup_mask(cleanup_mask: CleanupMask) -> bool:
    """Diagnostic/local reconstruction evidence must never become executable authority."""

    if cleanup_mask is None:
        return False
    if bool(getattr(cleanup_mask, "non_segmentation_or_local_fallback_used", False)):
        return True
    clean_authority = str(getattr(cleanup_mask, "clean_mask_authority", "") or "").lower()
    if clean_authority == "diagnostic_non_segmentation_fallback":
        return True
    completion_method = str(getattr(cleanup_mask, "mask_completion_method", "") or "").lower()
    if completion_method.startswith("local_contrast_fallback_after_"):
        return True
    return False


def _is_artifact_risk_partition_reason(reason: str) -> bool:
    reason_text = str(reason or "").lower()
    return reason_text == ARTIFACT_RISK_PARTITION_REQUIRED_REASON or reason_text.startswith("mask_artifact_risk_")


def _is_partition_component_mask(cleanup_mask: CleanupMask) -> bool:
    visual_scope = _normalise_cleanup_visual_scope(getattr(cleanup_mask, "visual_scope", ""))
    if visual_scope == "source_glyph_union_partition":
        return True
    method = str(getattr(cleanup_mask, "mask_method", "") or "").lower()
    source = str(getattr(cleanup_mask, "mask_source", "") or "").lower()
    return "bounded_connected_component_partition" in method or "partitioned_cleanup_component" in source


def _normalise_cleanup_visual_scope(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return LEGACY_CLEANUP_VISUAL_SCOPE_MAP.get(text, text)


def _partition_visual_scope(value: Any) -> str:
    visual_scope = _normalise_cleanup_visual_scope(value)
    if visual_scope == "source_glyph_union":
        return "source_glyph_union_partition"
    return visual_scope


def _unique_texts(values: Sequence[Any] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _safe_reason_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    output = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return output.strip("_") or "unknown"


def _short_artifact_stem(*parts: Any) -> str:
    tokens = [_safe_id(part) for part in parts if str(part or "").strip()]
    raw = "_".join(tokens) or "cleanup_artifact"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    compact = "_".join(token[:24].strip("_") for token in tokens if token) or "cleanup_artifact"
    return f"{compact[:80].strip('_')}_{digest}"


def _mask_protected_conflict_reason(cleanup_mask: CleanupMask) -> str:
    combined = " ".join(
        str(value or "")
        for value in (
            cleanup_mask.mask_method,
            cleanup_mask.mask_source,
            cleanup_mask.rejection_reason,
            cleanup_mask.protection_reason,
        )
    ).lower()
    protected_markers = (
        ("preserve", "mask_protected_preserve"),
        ("sfx", "mask_protected_sfx"),
        ("decorative", "mask_protected_decorative"),
        ("non_translation_art", "mask_protected_non_translation_art"),
        ("art_only", "mask_protected_art_only"),
    )
    for marker, reason in protected_markers:
        if marker in combined:
            return reason
    return ""


def _source_image_array(*, source_image: Any | None, source_image_path: str | None) -> tuple[Any | None, str]:
    if np is None:
        return None, "numpy_unavailable"
    if source_image is not None:
        try:
            if hasattr(source_image, "convert"):
                return np.asarray(source_image.convert("RGB")), ""
            arr = np.asarray(source_image)
            if arr.ndim == 2:
                return arr, ""
            if arr.ndim == 3:
                return arr[:, :, :3], ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    if source_image_path:
        if Image is None:
            return None, "pillow_unavailable"
        try:
            with Image.open(source_image_path) as img:
                return np.asarray(img.convert("RGB")), ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    return None, "source_image_path_missing"


def _flat_background_pre_execution_rejection(
    *,
    cleanup_mask: CleanupMask,
    source_np: Any | None,
    source_error: str,
) -> tuple[str, dict[str, Any]]:
    if source_np is None:
        return "cleanup_source_image_evidence_unavailable", {"source_image_error": source_error}
    metrics = _background_flatness_metrics(source_np, cleanup_mask.allowed_area)
    if metrics.get("error"):
        return "cleanup_source_image_evidence_unavailable", metrics

    allowed_area = int(metrics.get("allowed_area") or 0)
    edge_density = float(metrics.get("edge_density") or 0.0)
    luma_span = float(metrics.get("luma_span") or 0.0)
    luma_std = float(metrics.get("luma_std") or 0.0)
    page_ratio = float(metrics.get("allowed_page_ratio") or 0.0)

    broad_textured = allowed_area > 60_000 and edge_density >= 0.18 and luma_span >= 160.0
    page_broad_textured = page_ratio > 0.025 and edge_density >= 0.20 and luma_span >= 180.0
    very_broad_variable = allowed_area > 120_000 and (edge_density >= 0.12 or luma_std >= 45.0)
    if broad_textured or page_broad_textured or very_broad_variable:
        return "cleanup_background_not_flat_or_art_risk", metrics
    return "", metrics


def _background_flatness_metrics(source_np: Any, allowed_area: Sequence[int] | None) -> dict[str, Any]:
    if np is None:
        return {"error": "numpy_unavailable"}
    allowed = _valid_bbox(allowed_area)
    if allowed is None:
        return {"error": "allowed_area_missing_or_invalid"}
    arr = np.asarray(source_np)
    if arr.ndim == 3:
        gray = _gray(arr[:, :, :3])
    elif arr.ndim == 2:
        gray = arr.astype(np.uint8)
    else:
        return {"error": "unsupported_source_image_shape"}
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = allowed
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return {"error": "allowed_area_outside_source_image"}
    crop = gray[y0:y1, x0:x1]
    if crop.size <= 0:
        return {"error": "allowed_area_empty_crop"}
    if cv2 is not None:
        edges = cv2.Canny(crop, 48, 128)
        edge_density = float(np.count_nonzero(edges)) / max(1, int(edges.size))
    else:
        gy, gx = np.gradient(crop.astype(np.float32))
        edge_density = float(np.count_nonzero((np.abs(gx) + np.abs(gy)) > 32.0)) / max(1, int(crop.size))
    p10 = float(np.percentile(crop, 10))
    p90 = float(np.percentile(crop, 90))
    allowed_pixels = int(crop.size)
    page_area = max(1, int(width) * int(height))
    return {
        "allowed_area": allowed_pixels,
        "allowed_page_ratio": round(float(allowed_pixels) / float(page_area), 6),
        "luma_mean": round(float(np.mean(crop)), 3),
        "luma_std": round(float(np.std(crop)), 3),
        "luma_p10": round(p10, 3),
        "luma_p90": round(p90, 3),
        "luma_span": round(p90 - p10, 3),
        "edge_density": round(edge_density, 6),
    }


def _active_mask_bbox(mask: Any, shape: tuple[int, int], *, pad: int = 0) -> tuple[int, int, int, int] | None:
    try:
        active = np.asarray(mask) > 0
    except Exception:
        return None
    if active.shape[:2] != shape[:2] or not np.any(active):
        return None
    ys, xs = np.where(active)
    height, width = shape[:2]
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(width, int(xs.max()) + 1 + int(pad))
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(height, int(ys.max()) + 1 + int(pad))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _source_residual_mask(source_gray: Any, cleaned_gray: Any, foreground: Any) -> Any:
    shape = source_gray.shape[:2]
    output = np.zeros(shape, dtype=bool)
    box = _active_mask_bbox(foreground, shape)
    if box is None:
        return output
    x0, y0, x1, y1 = box
    source_crop = source_gray[y0:y1, x0:x1]
    cleaned_crop = cleaned_gray[y0:y1, x0:x1]
    foreground_crop = np.asarray(foreground)[y0:y1, x0:x1] > 0
    diff = np.abs(source_crop.astype(np.int16) - cleaned_crop.astype(np.int16))
    source_dark = source_crop < 210
    cleaned_still_dark = cleaned_crop < 230
    source_light = source_crop > 190
    cleaned_still_light = cleaned_crop > 170
    unchanged = diff < 45
    output[y0:y1, x0:x1] = foreground_crop & unchanged & (
        (source_dark & cleaned_still_dark) | (source_light & cleaned_still_light)
    )
    return output


def _dark_source_residual_mask(source_gray: Any, cleaned_gray: Any, foreground: Any) -> Any:
    shape = source_gray.shape[:2]
    output = np.zeros(shape, dtype=bool)
    box = _active_mask_bbox(foreground, shape)
    if box is None:
        return output
    x0, y0, x1, y1 = box
    source_crop = source_gray[y0:y1, x0:x1]
    cleaned_crop = cleaned_gray[y0:y1, x0:x1]
    foreground_crop = np.asarray(foreground)[y0:y1, x0:x1] > 0
    diff = np.abs(source_crop.astype(np.int16) - cleaned_crop.astype(np.int16))
    source_dark = source_crop < 210
    cleaned_still_dark = cleaned_crop < 230
    source_like = diff < 45
    output[y0:y1, x0:x1] = foreground_crop & source_dark & cleaned_still_dark & source_like
    return output


def _source_light_foreground_context(source_gray: Any, foreground: Any) -> dict[str, Any]:
    """Identify when unchanged light foreground pixels are likely light text, not paper."""
    if source_gray is None or foreground is None:
        return {
            "dark_context": False,
            "median_luma": None,
            "dark_ratio": 0.0,
        }
    try:
        source_arr = np.asarray(source_gray)
        foreground_bool = np.asarray(foreground) > 0
        if not np.any(foreground_bool):
            return {
                "dark_context": False,
                "median_luma": None,
                "dark_ratio": 0.0,
            }
        box = _active_mask_bbox(foreground_bool, source_arr.shape[:2], pad=16)
        if box is None:
            return {
                "dark_context": False,
                "median_luma": None,
                "dark_ratio": 0.0,
            }
        x0, y0, x1, y1 = box
        foreground_crop = foreground_bool[y0:y1, x0:x1]
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            context = cv2.dilate(foreground_crop.astype(np.uint8), kernel, iterations=2) > 0
            context = context & ~foreground_crop
        else:
            context = np.zeros_like(foreground_crop, dtype=bool)
        if not np.any(context):
            return {
                "dark_context": False,
                "median_luma": None,
                "dark_ratio": 0.0,
            }
        values = source_arr[y0:y1, x0:x1][context]
        if values.size <= 0:
            return {
                "dark_context": False,
                "median_luma": None,
                "dark_ratio": 0.0,
            }
        median_luma = float(np.median(values))
        dark_ratio = float(np.count_nonzero(values < 170) / max(1, values.size))
        return {
            "dark_context": bool(median_luma < 180 and dark_ratio > 0.15),
            "median_luma": round(median_luma, 3),
            "dark_ratio": round(dark_ratio, 4),
        }
    except Exception:
        return {
            "dark_context": False,
            "median_luma": None,
            "dark_ratio": 0.0,
        }


def _outside_accepted_mask_boundary_residual_mask(
    *,
    source_gray: Any,
    cleaned_gray: Any,
    foreground: Any,
    erase: Any,
    allowed: Any,
    source_light_foreground_ratio: float,
) -> Any:
    if np is None or source_gray is None or cleaned_gray is None:
        return np.zeros((0, 0), dtype=bool) if np is not None else []
    try:
        source_arr = np.asarray(source_gray)
        cleaned_arr = np.asarray(cleaned_gray)
        foreground_bool = np.asarray(foreground) > 0
        erase_bool = np.asarray(erase) > 0
        allowed_bool = np.asarray(allowed) > 0
    except Exception:
        return np.zeros(np.asarray(source_gray).shape[:2], dtype=bool)
    if (
        source_arr.ndim != 2
        or cleaned_arr.shape != source_arr.shape
        or foreground_bool.shape != source_arr.shape
        or erase_bool.shape != source_arr.shape
        or allowed_bool.shape != source_arr.shape
        or not np.any(foreground_bool)
        or not np.any(erase_bool)
    ):
        return np.zeros(source_arr.shape[:2], dtype=bool)
    if source_light_foreground_ratio < 0.25:
        return np.zeros(source_arr.shape[:2], dtype=bool)
    output = np.zeros(source_arr.shape[:2], dtype=bool)
    box = _active_mask_bbox(foreground_bool, source_arr.shape[:2], pad=8)
    if box is None:
        return output
    x0, y0, x1, y1 = box
    source_crop = source_arr[y0:y1, x0:x1]
    cleaned_crop = cleaned_arr[y0:y1, x0:x1]
    foreground_crop = foreground_bool[y0:y1, x0:x1]
    erase_crop = erase_bool[y0:y1, x0:x1]
    allowed_crop = allowed_bool[y0:y1, x0:x1]
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        near_foreground = cv2.dilate(foreground_crop.astype(np.uint8), kernel, iterations=2) > 0
        boundary = near_foreground & ~erase_crop & allowed_crop
    else:
        boundary = np.zeros(foreground_crop.shape[:2], dtype=bool)
    if not np.any(boundary):
        return output
    unchanged = np.abs(cleaned_crop.astype(np.int16) - source_crop.astype(np.int16)) < 45
    output[y0:y1, x0:x1] = boundary & (source_crop < 120) & (cleaned_crop < 140) & unchanged
    return output


def _source_erasure_box_residual_mask(source_gray: Any, cleaned_gray: Any, bbox: Sequence[int]) -> Any:
    residual = np.zeros(source_gray.shape[:2], dtype=bool)
    box = _valid_bbox(bbox)
    if box is None:
        return residual
    height, width = source_gray.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return residual
    source_crop = source_gray[y0:y1, x0:x1]
    cleaned_crop = cleaned_gray[y0:y1, x0:x1]
    source_dark = source_crop < 135
    cleaned_still_dark = cleaned_crop < 150
    unchanged = np.abs(cleaned_crop.astype(np.int16) - source_crop.astype(np.int16)) < 35
    residual[y0:y1, x0:x1] = source_dark & cleaned_still_dark & unchanged
    return residual


def _write_residual_source_mask(
    *,
    cleanup_result: CleanupResult,
    residual_mask: Any,
) -> tuple[str, str]:
    if Image is None or np is None:
        return "", "pillow_or_numpy_unavailable"
    if residual_mask is None:
        return "", "residual_mask_missing"
    base_ref = cleanup_result.mask_ref or cleanup_result.diff_ref or cleanup_result.cleaned_image_ref
    if not base_ref:
        return "", "cleanup_artifact_ref_unavailable"
    try:
        artifact_dir = os.path.dirname(str(base_ref))
        if not artifact_dir or not os.path.isdir(artifact_dir):
            return "", "cleanup_artifact_dir_unavailable"
        filename = f"{_safe_id(cleanup_result.cleanup_result_id)}_residual_source_mask.png"
        path = os.path.join(artifact_dir, filename)
        if len(path) > 240:
            digest = hashlib.sha1(str(cleanup_result.cleanup_result_id).encode("utf-8")).hexdigest()[:16]
            filename = f"cres_{digest}_residual_source_mask.png"
            path = os.path.join(artifact_dir, filename)
        arr = (np.asarray(residual_mask) > 0).astype(np.uint8) * 255
        written = write_debug_image_file(path, Image.fromarray(arr, mode="L"))
        return written, "" if written else "cleanup_artifact_write_failed"
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _gray(image_np: Any) -> Any:
    if image_np.ndim == 2:
        return image_np
    return (
        0.299 * image_np[:, :, 0]
        + 0.587 * image_np[:, :, 1]
        + 0.114 * image_np[:, :, 2]
    ).astype(np.uint8)


def _binary_mask(value: Any, shape: tuple[int, int]) -> Any:
    if np is None or value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    output = np.zeros(shape, dtype=np.uint8)
    height = min(shape[0], arr.shape[0])
    width = min(shape[1], arr.shape[1])
    if height <= 0 or width <= 0:
        return output
    output[:height, :width] = (arr[:height, :width] > 0).astype(np.uint8)
    return output


def _bbox_to_mask(bbox: Sequence[int] | None, shape: tuple[int, int]) -> Any:
    box = _valid_bbox(bbox)
    if np is None or box is None:
        return None
    output = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = box
    x0 = max(0, min(shape[1], x0))
    x1 = max(0, min(shape[1], x1))
    y0 = max(0, min(shape[0], y0))
    y1 = max(0, min(shape[0], y1))
    if x1 > x0 and y1 > y0:
        output[y0:y1, x0:x1] = 1
    return output


def _mask_bbox(mask: Any) -> list[int] | None:
    if np is None or mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return None
    ys, xs = np.nonzero(arr > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _extract_jobs(value: Any) -> list[CleanupJob]:
    if value is None:
        return []
    if hasattr(value, "jobs"):
        return [item for item in getattr(value, "jobs") or [] if isinstance(item, CleanupJob)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupJob)]
    return []


def _extract_masks(value: Any) -> list[CleanupMask]:
    if value is None:
        return []
    if hasattr(value, "masks"):
        return [item for item in getattr(value, "masks") or [] if isinstance(item, CleanupMask)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupMask)]
    return []


def _extract_plans(value: Any) -> list[CleanupPlan]:
    if value is None:
        return []
    if hasattr(value, "plans"):
        return [item for item in getattr(value, "plans") or [] if isinstance(item, CleanupPlan)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupPlan)]
    return []


def _plans_by_job_and_mask_id(plans: Sequence[CleanupPlan]) -> dict[tuple[str, str], list[CleanupPlan]]:
    output: dict[tuple[str, str], list[CleanupPlan]] = {}
    for plan in plans or []:
        key = (str(plan.cleanup_job_id), str(plan.cleanup_mask_id))
        output.setdefault(key, []).append(plan)
    return output


def _plans_by_cleanup_mask_id(plans: Sequence[CleanupPlan]) -> dict[str, list[CleanupPlan]]:
    output: dict[str, list[CleanupPlan]] = {}
    for plan in plans or []:
        output.setdefault(str(plan.cleanup_mask_id), []).append(plan)
    return output


def _select_formal_cleanup_plan(plans: Sequence[CleanupPlan]) -> CleanupPlan | None:
    for plan in plans or []:
        if bool((plan.backend_parameters or {}).get("formal_cleanup_plan", False)):
            return plan
    for plan in plans or []:
        return plan
    return None


def _job_base_record(page_id: str, job: CleanupJob) -> dict[str, Any]:
    identity = _cleanup_identity_fields(job, fallback_region_id=str(job.cleanup_job_id or ""))
    return {
        "page_id": page_id,
        "cleanup_job_id": str(job.cleanup_job_id),
        "parent_execution_bundle_id": str(identity.get("parent_execution_bundle_id") or ""),
        "parent_logical_text_unit_id": str(identity.get("parent_logical_text_unit_id") or ""),
        "text_block_root_id": str(identity.get("text_block_root_id") or getattr(job, "text_block_root_id", "") or ""),
        "target_region_ids": list(identity.get("target_region_ids") or job.target_region_ids or []),
        "cleanup_class": _enum_value(job.cleanup_class),
        "route_intent": str(job.route_intent or ""),
        "semantic_class": str(job.semantic_class or ""),
        "cleanup_mode": str(job.cleanup_mode or ""),
        "classification_reason": str(job.classification_reason or ""),
    }


def _cleanup_identity_fields(*sources: Any, fallback_region_id: str = "") -> dict[str, Any]:
    parent_execution_bundle_id = _first_identity_value(
        sources,
        "parent_execution_bundle_id",
        "parent_id",
    )
    parent_logical_text_unit_id = _first_identity_value(
        sources,
        "parent_logical_text_unit_id",
        "parent_id",
    )
    if not parent_execution_bundle_id:
        parent_execution_bundle_id = parent_logical_text_unit_id
    if not parent_logical_text_unit_id:
        parent_logical_text_unit_id = parent_execution_bundle_id
    text_block_root_id = _first_identity_value(
        sources,
        "text_block_root_id",
        "root_id",
    )
    target_region_ids = _first_identity_list(sources, "target_region_ids")
    fallback = str(fallback_region_id or "").strip()
    if not target_region_ids and fallback:
        target_region_ids = [fallback]
    parent = parent_execution_bundle_id or parent_logical_text_unit_id
    if parent and parent not in target_region_ids:
        target_region_ids = [parent] + target_region_ids
    return {
        "parent_execution_bundle_id": parent_execution_bundle_id,
        "parent_logical_text_unit_id": parent_logical_text_unit_id,
        "text_block_root_id": text_block_root_id,
        "target_region_ids": target_region_ids,
    }


def _canonical_parent_region_id(*sources: Any, fallback_region_id: str = "") -> str:
    identity = _cleanup_identity_fields(*sources, fallback_region_id=fallback_region_id)
    return str(
        identity.get("parent_execution_bundle_id")
        or identity.get("parent_logical_text_unit_id")
        or fallback_region_id
        or ""
    )


def _first_identity_value(sources: Sequence[Any], *keys: str) -> str:
    for source in sources:
        if source is None:
            continue
        for key in keys:
            value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
            if value in (None, "", [], {}):
                continue
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _first_identity_list(sources: Sequence[Any], key: str) -> list[str]:
    for source in sources:
        if source is None:
            continue
        value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = []
            for item in value:
                text = str(item or "").strip()
                if text and text not in items:
                    items.append(text)
            if items:
                return items
    return []


def _render_eligibility_by_region_id(render_eligibility: Any | None) -> dict[str, Any]:
    if render_eligibility is None:
        return {}
    decisions = getattr(render_eligibility, "decisions_by_region_id", None)
    if isinstance(decisions, Mapping):
        return {str(key): value for key, value in decisions.items()}
    if isinstance(render_eligibility, Mapping):
        raw = render_eligibility.get("decisions_by_region_id")
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items()}
        raw_list = render_eligibility.get("decisions")
        if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes)):
            output: dict[str, Any] = {}
            for item in raw_list:
                rid = str(_decision_value(item, "region_id") or "")
                if rid:
                    output[rid] = item
            return output
    return {}


def _source_grounding_suppressed_decision(
    job: CleanupJob,
    decisions_by_region_id: Mapping[str, Any],
) -> Any | None:
    for region_id in job.target_region_ids or []:
        decision = decisions_by_region_id.get(str(region_id or ""))
        if str(_decision_value(decision, "status") or "") == "suppressed_source_ungrounded":
            return decision
    return None


def _decision_value(decision: Any, key: str) -> Any:
    if decision is None:
        return None
    if isinstance(decision, Mapping):
        return decision.get(key)
    value = getattr(decision, key, None)
    return getattr(value, "value", value)


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


def _union_valid_bboxes(*values: Any) -> list[int] | None:
    boxes = [box for value in values if (box := _valid_bbox(value)) is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _bbox_area(value: Sequence[int] | None) -> int:
    box = _valid_bbox(value)
    if box is None:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _growth_ratio(erase_pixels: int, foreground_pixels: int) -> float:
    if foreground_pixels <= 0:
        return float("inf")
    return round(float(erase_pixels) / float(foreground_pixels), 4)


def _safe_id(value: Any) -> str:
    text = str(value or "none")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "none"


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value or "")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_audit_dict"):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items() if not _raw_payload_key(key)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return str(value)


def _raw_payload_key(key: Any) -> bool:
    return str(key).lower() in {
        "image",
        "cleaned_image",
        "cleaned_crop",
        "mask",
        "foreground_mask",
        "erase_mask",
        "array",
        "bitmap",
        "crop",
    }
