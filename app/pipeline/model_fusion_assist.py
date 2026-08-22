"""Debug-only Phase 4 model-fusion assist enrichment.

This module is quarantined from production behavior. Historical page-specific
diagnostic helpers remain importable only when both the legacy quarantine flag
and the requested diagnostic flag are enabled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from app.pipeline.bubble_detection import (
    BUBBLE_DETECTION_VERSION,
    BubbleDetectionInput,
    BubbleDetectionResult,
    draw_bubble_detection_overlay,
    run_bubble_detection,
)


MODEL_FUSION_ASSIST_VERSION = "phase4b6_model_fusion_assist_v1"
MODEL_FUSION_ASSIST_FLAG = "MT_MODEL_FUSION_ASSIST"
MODEL_FUSION_OWNERSHIP_PROOF_VERSION = "phase4b9_model_fusion_ownership_proof_v1"
MODEL_FUSION_OWNERSHIP_PROOF_FLAG = "MT_MODEL_FUSION_OWNERSHIP_PROOF"
MODEL_FUSION_TEXT_CONSERVATION_PROOF_VERSION = "phase4b10_model_fusion_text_conservation_proof_v1"
MODEL_FUSION_TEXT_CONSERVATION_PROOF_FLAG = "MT_MODEL_FUSION_TEXT_CONSERVATION_PROOF"
MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION_VERSION = "phase4b19_model_fusion_text_conservation_integration_v1"
MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION_FLAG = "MT_MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION"
MODEL_FUSION_ROUTE_GUARD_PROOF_VERSION = "phase4b11_model_fusion_route_guard_proof_v1"
MODEL_FUSION_ROUTE_GUARD_PROOF_FLAG = "MT_MODEL_FUSION_ROUTE_GUARD_PROOF"
MODEL_FUSION_CONFIDENCE_POLICY_VERSION = "phase4b11_model_fusion_confidence_policy_v1"
MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION_VERSION = "phase4b12_model_fusion_render_constraint_calibration_v1"
MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION_FLAG = "MT_MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION"
MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION_VERSION = "phase4b13_model_fusion_cleanup_boundary_calibration_v1"
MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION_FLAG = "MT_MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION"
HIGH_ACCURACY_BUBBLE_MODE_VERSION = "phase4b14_high_accuracy_bubble_mode_v1"
HIGH_ACCURACY_BUBBLE_MODE_FLAG = "MT_HIGH_ACCURACY_BUBBLE_MODE"
LEGACY_PAGE_SPECIFIC_ASSIST_FLAG = "MT_LEGACY_PAGE_SPECIFIC_ASSIST"

ROOT = Path(__file__).resolve().parents[2]

SAFE_FUTURE_CLASSES = {
    "safe_future_text_container_assist",
    "safe_future_text_ownership_assist",
    "safe_future_render_constraint_hint",
    "safe_future_missed_text_hint",
}

OWNERSHIP_PROOF_ALLOWLIST: dict[str, set[str]] = {}
TEXT_CONSERVATION_PROOF_ALLOWLIST: dict[str, set[str]] = {}
ROUTE_GUARD_PROOF_REFERENCES: dict[str, dict[str, str]] = {}

BUBBLE_LOCAL_OWNERSHIP_REASON = "bubble_local_nested_speech_fragment_ownership"
ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON = "adjacent_vertical_speech_text_conservation_recovery"

MODEL_FUSION_CONFIDENCE_POLICY_TIERS = {
    "strong_model_container": {
        "required_evidence": [
            "kitsumed_speech_mask",
            "ogkalu_bubble_or_text_bubble_same_local_container",
            "no_current_sfx_decorative_or_caption_conflict",
        ],
        "may_support": ["speech_container", "text_ownership", "render_constraint_hint"],
        "may_not_support": ["override_current_sfx_decorative_preserve", "automatic_text_transfer", "automatic_cleanup_or_render_mutation"],
        "human_review_required": False,
        "fallback_behavior": "current_pipeline_route",
        "audit_fields": ["model_fusion_route_guard_confidence_tier", "model_fusion_route_guard_reason_codes"],
    },
    "mask_primary_container": {
        "required_evidence": ["kitsumed_speech_mask", "ogkalu_absent_or_weak", "current_region_not_preserve_conflict"],
        "may_support": ["speech_container", "render_constraint_hint"],
        "may_not_support": ["override_current_sfx_decorative_preserve", "automatic_text_ownership_transfer"],
        "human_review_required": True,
        "fallback_behavior": "current_pipeline_route",
        "audit_fields": ["model_fusion_route_guard_confidence_tier", "model_fusion_route_guard_kitsumed_mask_ids"],
    },
    "text_bubble_review_container": {
        "required_evidence": ["ogkalu_bubble_or_text_bubble", "no_kitsumed_speech_mask"],
        "may_support": ["missed_text_review", "ownership_review"],
        "may_not_support": ["automatic_speech_route_mutation", "automatic_translation_or_cleanup"],
        "human_review_required": True,
        "fallback_behavior": "current_pipeline_route",
        "audit_fields": ["model_fusion_route_guard_confidence_tier", "model_fusion_route_guard_ogkalu_detection_ids"],
    },
    "text_free_review_only": {
        "required_evidence": ["ogkalu_text_free"],
        "may_support": ["free_text_review"],
        "may_not_support": ["speech_decision", "non_speech_decision", "automatic_translate_or_preserve"],
        "human_review_required": True,
        "fallback_behavior": "current_pipeline_route",
        "audit_fields": ["model_fusion_route_guard_confidence_tier", "model_fusion_route_guard_human_review_required"],
    },
    "conflict_preserve_wins": {
        "required_evidence": ["current_decorative_sfx_preserve_route", "overlapping_or_conflicting_model_evidence"],
        "may_support": ["preserve_guard", "review_conflict"],
        "may_not_support": ["override_current_preserve_route"],
        "human_review_required": True,
        "fallback_behavior": "preserve_current_route",
        "audit_fields": ["model_fusion_route_guard_result", "model_fusion_route_guard_reason_codes"],
    },
    "unsupported_by_model_but_deterministic_valid": {
        "required_evidence": ["current_deterministic_route_valid", "missing_or_weak_model_evidence"],
        "may_support": ["do_not_penalize_current_route"],
        "may_not_support": ["automatic_model_based_mutation"],
        "human_review_required": True,
        "fallback_behavior": "current_pipeline_route",
        "audit_fields": ["model_fusion_route_guard_result", "model_fusion_route_guard_reason_codes"],
    },
}

CLASS_TO_ASSIST_TYPE = {
    "safe_future_text_container_assist": "text_container_hint",
    "safe_future_text_ownership_assist": "text_ownership_hint",
    "safe_future_render_constraint_hint": "render_constraint_hint",
    "safe_future_missed_text_hint": "missed_text_review_hint",
    "review_only_ambiguous": "review_only",
    "review_only_text_free": "review_only",
    "review_only_caption_or_background": "review_only",
    "review_only_sfx_decorative_conflict": "review_only",
    "noisy_false_positive": "review_only",
    "missed_useful_evidence": "review_only",
    "ignore_low_value": "review_only",
}


def model_fusion_assist_enabled() -> bool:
    return _legacy_page_specific_assist_enabled() and os.environ.get(MODEL_FUSION_ASSIST_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def high_accuracy_bubble_mode_enabled() -> bool:
    return _legacy_page_specific_assist_enabled() and os.environ.get(HIGH_ACCURACY_BUBBLE_MODE_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _legacy_page_specific_assist_enabled() -> bool:
    return os.environ.get(LEGACY_PAGE_SPECIFIC_ASSIST_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def effective_model_fusion_assist_enabled() -> bool:
    return model_fusion_assist_enabled() or high_accuracy_bubble_mode_enabled()


def model_fusion_ownership_proof_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    proof_enabled = os.environ.get(MODEL_FUSION_OWNERSHIP_PROOF_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and (proof_enabled or high_accuracy_bubble_mode_enabled())


def model_fusion_text_conservation_proof_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    proof_enabled = os.environ.get(MODEL_FUSION_TEXT_CONSERVATION_PROOF_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and (proof_enabled or high_accuracy_bubble_mode_enabled())


def model_fusion_text_conservation_integration_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    integration_enabled = os.environ.get(MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and integration_enabled


def model_fusion_route_guard_proof_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    proof_enabled = os.environ.get(MODEL_FUSION_ROUTE_GUARD_PROOF_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and (proof_enabled or high_accuracy_bubble_mode_enabled())


def model_fusion_render_constraint_calibration_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    calibration_enabled = os.environ.get(MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and (calibration_enabled or high_accuracy_bubble_mode_enabled())


def model_fusion_cleanup_boundary_calibration_enabled() -> bool:
    assist_enabled = effective_model_fusion_assist_enabled()
    calibration_enabled = os.environ.get(MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}
    return assist_enabled and (calibration_enabled or high_accuracy_bubble_mode_enabled())


def enrich_audit_with_model_fusion_assist(
    audit: Mapping[str, Any],
    *,
    page_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Return an audit copy with diagnostic model-fusion evidence attached.

    The function never mutates routing, cleanup, OCR, translation, render boxes,
    output images, or project records. On any failure it records a diagnostic
    error and returns a usable audit payload.
    """

    started = time.perf_counter()
    enriched = _copy_audit(audit)
    _initialize_top_level(enriched)

    high_accuracy_enabled = high_accuracy_bubble_mode_enabled()
    if not effective_model_fusion_assist_enabled():
        enriched["model_fusion_assist_enabled"] = False
        enriched["model_fusion_assist_generated"] = False
        enriched["model_fusion_assist_runtime_sec"] = round(time.perf_counter() - started, 6)
        return enriched

    enriched["model_fusion_assist_enabled"] = True
    enriched["high_accuracy_bubble_mode_enabled"] = high_accuracy_enabled
    page_path = Path(page_dir) if page_dir is not None else None

    try:
        image_path = _source_image_path(enriched)
        if image_path is None:
            raise FileNotFoundError("audit has no readable source_path/original_path/image_path")

        page_id = str(enriched.get("page_id") or image_path.stem)
        service_result = run_bubble_detection(
            BubbleDetectionInput(
                page_id=page_id,
                image_path=image_path,
                regions=enriched.get("regions", []),
                debug_page_dir=page_path,
                mode="high_accuracy" if high_accuracy_enabled else "assist",
            )
        )
        _attach_bubble_detection_fields(enriched, service_result)
        _attach_bubble_detection_consumer_metadata(enriched, service_result)

        if not service_result.generated:
            enriched["model_fusion_assist_generated"] = False
            enriched["model_fusion_assist_error"] = service_result.error or "bubble detection service did not generate evidence"
            enriched["model_fusion_evidence"] = service_result.legacy_model_fusion_evidence()
            enriched["model_fusion_assist_candidates"] = []
            enriched["model_fusion_conflicts"] = []
            enriched["model_fusion_assist_runtime_sec"] = round(time.perf_counter() - started, 6)
            _attach_high_accuracy_bubble_summary_fields(enriched, [], [])
            if page_path is not None:
                _write_page_artifacts(page_path=page_path, page_id=page_id, service_result=service_result, audit=enriched)
            return enriched

        calibration_module = _get_calibration_module()
        region_links = service_result.region_model_links
        fused_containers = service_result.fused_containers

        candidates = _build_candidates(
            page_id=page_id,
            fused_containers=fused_containers,
            region_links=region_links,
            regions=enriched.get("regions", []),
            calibration_module=calibration_module,
        )
        conflicts = _build_conflicts(candidates, fused_containers)
        _attach_region_fields(enriched.get("regions", []), candidates)
        _attach_ownership_proof_fields(enriched, candidates)
        _attach_text_conservation_proof_fields(enriched, candidates)
        _attach_text_conservation_integration_fields(enriched, candidates)
        _attach_route_guard_proof_fields(enriched, candidates)
        _attach_render_constraint_calibration_fields(enriched, candidates, fused_containers)
        _attach_cleanup_boundary_calibration_fields(
            enriched,
            candidates,
            fused_containers,
            service_result.bubble_model_evidence,
            service_result.text_area_model_evidence,
        )

        enriched["model_fusion_assist_generated"] = True
        enriched["model_fusion_assist_error"] = None
        enriched["model_fusion_evidence"] = service_result.legacy_model_fusion_evidence()
        enriched["model_fusion_assist_candidates"] = candidates
        enriched["model_fusion_conflicts"] = conflicts
        enriched["model_fusion_assist_runtime_sec"] = round(time.perf_counter() - started, 6)
        _attach_high_accuracy_bubble_summary_fields(enriched, candidates, conflicts)

        if page_path is not None:
            _write_page_artifacts(
                page_path=page_path,
                page_id=page_id,
                service_result=service_result,
                audit=enriched,
            )

    except Exception as exc:
        enriched["model_fusion_assist_generated"] = False
        enriched["model_fusion_assist_error"] = f"{type(exc).__name__}: {exc}"
        enriched["model_fusion_evidence"] = {}
        enriched["model_fusion_assist_candidates"] = []
        enriched["model_fusion_conflicts"] = []
        enriched["model_fusion_assist_runtime_sec"] = round(time.perf_counter() - started, 6)
        enriched["bubble_detection_version"] = BUBBLE_DETECTION_VERSION
        enriched["bubble_detection_generated"] = False
        enriched["bubble_detection_error"] = enriched["model_fusion_assist_error"]
        enriched["bubble_detection_runtime_sec"] = 0.0
        enriched["bubble_detection_fallback_used"] = True
        enriched["bubble_detection_cache_enabled"] = False
        enriched["bubble_detection_cache_key"] = None
        enriched["bubble_detection_cache_hit"] = False
        enriched["bubble_detection_cache_read_path"] = None
        enriched["bubble_detection_cache_write_path"] = None
        enriched["bubble_detection_cache_error"] = enriched["model_fusion_assist_error"]
        enriched["bubble_detection_cache_invalidation_reason"] = "model_fusion_assist_exception"
        try:
            from app.pipeline.text_area_diagnostics import attach_bubble_detection_consumer_fields

            attach_bubble_detection_consumer_fields(
                enriched,
                {"generated": False, "error": enriched["model_fusion_assist_error"]},
                consumer_source="model_fusion_assist",
            )
        except Exception:
            enriched["bubble_detection_consumer_version"] = "phase4b17_bubble_detection_consumer_v1"
            enriched["bubble_detection_consumer_generated"] = False
            enriched["bubble_detection_consumer_error"] = enriched["model_fusion_assist_error"]
            enriched["bubble_detection_consumer_sources"] = ["model_fusion_assist", "bubble_detection_service"]
        _attach_high_accuracy_bubble_summary_fields(enriched, [], [])

    enriched["model_fusion_assist_runtime_sec"] = round(time.perf_counter() - started, 6)
    if enriched.get("high_accuracy_bubble_mode_enabled"):
        enriched["high_accuracy_bubble_mode_runtime_sec"] = enriched["model_fusion_assist_runtime_sec"]
    return enriched


def _initialize_top_level(audit: MutableMapping[str, Any]) -> None:
    audit.setdefault("model_fusion_assist_version", MODEL_FUSION_ASSIST_VERSION)
    audit.setdefault("model_fusion_assist_enabled", False)
    audit.setdefault("model_fusion_assist_generated", False)
    audit.setdefault("model_fusion_assist_error", None)
    audit.setdefault("model_fusion_assist_runtime_sec", 0.0)
    audit.setdefault("model_fusion_evidence", {})
    audit.setdefault("model_fusion_assist_candidates", [])
    audit.setdefault("model_fusion_conflicts", [])
    audit.setdefault("bubble_detection_version", BUBBLE_DETECTION_VERSION)
    audit.setdefault("bubble_detection_generated", False)
    audit.setdefault("bubble_detection_error", None)
    audit.setdefault("bubble_detection_runtime_sec", 0.0)
    audit.setdefault("bubble_detection_fallback_used", False)
    audit.setdefault("bubble_detection_result", {})
    audit.setdefault("bubble_detection_cache_enabled", False)
    audit.setdefault("bubble_detection_cache_key", None)
    audit.setdefault("bubble_detection_cache_hit", False)
    audit.setdefault("bubble_detection_cache_read_path", None)
    audit.setdefault("bubble_detection_cache_write_path", None)
    audit.setdefault("bubble_detection_cache_error", None)
    audit.setdefault("bubble_detection_cache_invalidation_reason", None)
    audit.setdefault("bubble_detection_result_path", None)
    audit.setdefault("bubble_detection_overlay_path", None)
    audit.setdefault("bubble_detection_consumer_version", "phase4b17_bubble_detection_consumer_v1")
    audit.setdefault("bubble_detection_consumer_generated", False)
    audit.setdefault("bubble_detection_consumer_error", None)
    audit.setdefault("bubble_detection_consumer_sources", [])
    audit.setdefault("model_fusion_confidence_policy_version", MODEL_FUSION_CONFIDENCE_POLICY_VERSION)
    audit.setdefault("model_fusion_confidence_policy_tiers", MODEL_FUSION_CONFIDENCE_POLICY_TIERS)
    audit.setdefault("model_fusion_ownership_proof_enabled", False)
    audit.setdefault("model_fusion_ownership_proof_applied", False)
    audit.setdefault("model_fusion_ownership_proof_version", MODEL_FUSION_OWNERSHIP_PROOF_VERSION)
    audit.setdefault("model_fusion_ownership_proof_errors", [])
    audit.setdefault("model_fusion_text_conservation_proof_enabled", False)
    audit.setdefault("model_fusion_text_conservation_proof_applied", False)
    audit.setdefault("model_fusion_text_conservation_proof_version", MODEL_FUSION_TEXT_CONSERVATION_PROOF_VERSION)
    audit.setdefault("model_fusion_text_conservation_proof_errors", [])
    audit.setdefault("model_fusion_text_conservation_integration_version", MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION_VERSION)
    audit.setdefault("model_fusion_text_conservation_integration_enabled", False)
    audit.setdefault("model_fusion_text_conservation_integration_generated", False)
    audit.setdefault("model_fusion_text_conservation_integration_errors", [])
    audit.setdefault("model_fusion_text_conservation_integration_candidates", [])
    audit.setdefault("model_fusion_text_conservation_integration_blocked_references", [])
    audit.setdefault("model_fusion_text_conservation_integration_summary", {})
    audit.setdefault("model_fusion_text_conservation_integration_mutation_count", 0)
    audit.setdefault("model_fusion_route_guard_proof_enabled", False)
    audit.setdefault("model_fusion_route_guard_proof_applied", False)
    audit.setdefault("model_fusion_route_guard_proof_version", MODEL_FUSION_ROUTE_GUARD_PROOF_VERSION)
    audit.setdefault("model_fusion_route_guard_proof_errors", [])
    audit.setdefault("model_fusion_render_constraint_calibration_enabled", False)
    audit.setdefault("model_fusion_render_constraint_calibration_generated", False)
    audit.setdefault("model_fusion_render_constraint_calibration_version", MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION_VERSION)
    audit.setdefault("model_fusion_render_constraint_calibration_errors", [])
    audit.setdefault("model_fusion_render_constraint_calibration_candidates", [])
    audit.setdefault("model_fusion_cleanup_boundary_calibration_enabled", False)
    audit.setdefault("model_fusion_cleanup_boundary_calibration_generated", False)
    audit.setdefault("model_fusion_cleanup_boundary_calibration_version", MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION_VERSION)
    audit.setdefault("model_fusion_cleanup_boundary_calibration_errors", [])
    audit.setdefault("model_fusion_cleanup_boundary_calibration_candidates", [])
    audit.setdefault("high_accuracy_bubble_mode_enabled", False)
    audit.setdefault("high_accuracy_bubble_mode_version", HIGH_ACCURACY_BUBBLE_MODE_VERSION)
    audit.setdefault("high_accuracy_bubble_mode_generated", False)
    audit.setdefault("high_accuracy_bubble_mode_error", None)
    audit.setdefault("high_accuracy_bubble_mode_runtime_sec", 0.0)
    audit.setdefault("high_accuracy_bubble_mode_mutation_allowed", False)
    audit.setdefault("high_accuracy_bubble_mode_components", {})
    audit.setdefault("high_accuracy_bubble_mode_candidate_counts", {})
    audit.setdefault("high_accuracy_bubble_mode_conflict_counts", {})
    audit.setdefault("high_accuracy_bubble_mode_fallback_used", False)


def _copy_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(audit)
    copied["regions"] = [dict(region) for region in audit.get("regions", [])]
    return copied


def _attach_bubble_detection_fields(audit: MutableMapping[str, Any], result: BubbleDetectionResult) -> None:
    audit["bubble_detection_version"] = result.version
    audit["bubble_detection_generated"] = result.generated
    audit["bubble_detection_error"] = result.error
    audit["bubble_detection_runtime_sec"] = result.runtime_sec
    audit["bubble_detection_fallback_used"] = result.fallback_used
    audit["bubble_detection_result"] = result.to_dict()
    audit["bubble_detection_cache_enabled"] = result.cache_enabled
    audit["bubble_detection_cache_key"] = result.cache_key
    audit["bubble_detection_cache_hit"] = result.cache_hit
    audit["bubble_detection_cache_read_path"] = result.cache_read_path
    audit["bubble_detection_cache_write_path"] = result.cache_write_path
    audit["bubble_detection_cache_error"] = result.cache_error
    audit["bubble_detection_cache_invalidation_reason"] = result.cache_invalidation_reason


def _attach_bubble_detection_consumer_metadata(
    audit: MutableMapping[str, Any],
    result: BubbleDetectionResult,
) -> None:
    try:
        from app.pipeline.text_area_diagnostics import attach_bubble_detection_consumer_fields

        attach_bubble_detection_consumer_fields(
            audit,
            result.to_dict(),
            consumer_source="model_fusion_assist",
        )
    except Exception as exc:  # pragma: no cover - metadata bridge must fail closed
        audit["bubble_detection_consumer_version"] = "phase4b17_bubble_detection_consumer_v1"
        audit["bubble_detection_consumer_generated"] = False
        audit["bubble_detection_consumer_error"] = f"{type(exc).__name__}: {exc}"
        audit["bubble_detection_consumer_sources"] = ["model_fusion_assist", "bubble_detection_service"]


def _get_calibration_module() -> Any:
    from app.pipeline import legacy_model_fusion_calibration

    return legacy_model_fusion_calibration


def _source_image_path(audit: Mapping[str, Any]) -> Optional[Path]:
    for key in ("source_path", "original_path", "image_path"):
        value = audit.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    return None


def _build_candidates(
    *,
    page_id: str,
    fused_containers: Sequence[Mapping[str, Any]],
    region_links: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    calibration_module: Any,
) -> List[Dict[str, Any]]:
    page_record = {
        "region_model_links": region_links,
        "fused_containers": fused_containers,
    }
    region_info_by_id = calibration_module.region_maps(page_record)
    current_by_id = {str(region.get("region_id")): region for region in regions if region.get("region_id") is not None}

    candidates: List[Dict[str, Any]] = []
    for index, fused in enumerate(fused_containers):
        classification, allowed, verdict, policy, additional_evidence = calibration_module.classify_fused_record(
            page_id,
            fused,
            region_info_by_id,
        )
        region_ids = [str(rid) for rid in fused.get("affected_current_region_ids", []) if rid is not None]
        confidence = str(fused.get("confidence") or "low")
        conflict_flags = _as_list(fused.get("conflict_flags"))
        reason_codes = sorted(set(_as_list(fused.get("reason_codes")) + _reason_codes_from_policy(policy)))
        safe_high_confidence = (
            classification in SAFE_FUTURE_CLASSES
            and confidence == "high"
            and bool(allowed)
            and not conflict_flags
        )
        review_only = not safe_high_confidence or classification == "safe_future_missed_text_hint"
        assist_type = CLASS_TO_ASSIST_TYPE.get(classification, "review_only")

        current_semantic_classes = {
            rid: current_by_id.get(rid, {}).get("semantic_class")
            for rid in region_ids
            if rid in current_by_id
        }
        current_cleanup_modes = {
            rid: current_by_id.get(rid, {}).get("cleanup_mode")
            for rid in region_ids
            if rid in current_by_id
        }

        candidate = {
            "candidate_id": f"{page_id}:mf{index:03d}",
            "page_id": page_id,
            "fused_container_id": fused.get("fused_container_id"),
            "linked_kitsumed_mask_ids": _as_list(fused.get("linked_kitsumed_mask_ids")),
            "linked_ogkalu_detection_ids": _as_list(fused.get("linked_ogkalu_detection_ids")),
            "linked_current_region_ids": region_ids,
            "fusion_classification": classification,
            "assist_type": assist_type,
            "confidence": confidence,
            "visual_verdict": verdict,
            "assist_policy": policy,
            "current_semantic_classes": current_semantic_classes,
            "current_cleanup_modes": current_cleanup_modes,
            "reason_codes": reason_codes,
            "conflict_flags": conflict_flags,
            "model_agreement": calibration_module.describe_agreement(fused),
            "model_disagreement": calibration_module.describe_disagreement(fused),
            "required_evidence": additional_evidence,
            "contraindications": _contraindications(classification, confidence, conflict_flags, safe_high_confidence),
            "allowed_future_assist": safe_high_confidence,
            "eligible_for_future_mutation": safe_high_confidence and classification != "safe_future_missed_text_hint",
            "review_only": review_only,
            "would_change_behavior": False,
            "phase4_status": "assist_candidate_only",
            "human_review_required": review_only,
        }
        candidates.append(candidate)
    return candidates


def _build_conflicts(
    candidates: Sequence[Mapping[str, Any]],
    fused_containers: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    fused_by_id = {fused.get("fused_container_id"): fused for fused in fused_containers}
    conflicts: List[Dict[str, Any]] = []
    for candidate in candidates:
        conflict_flags = _as_list(candidate.get("conflict_flags"))
        classification = str(candidate.get("fusion_classification") or "")
        if not conflict_flags and classification not in {"review_only_sfx_decorative_conflict", "noisy_false_positive"}:
            continue
        fused = fused_by_id.get(candidate.get("fused_container_id"), {})
        conflicts.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "fused_container_id": candidate.get("fused_container_id"),
                "linked_current_region_ids": candidate.get("linked_current_region_ids", []),
                "fusion_classification": classification,
                "confidence": candidate.get("confidence"),
                "conflict_flags": conflict_flags,
                "reason_codes": candidate.get("reason_codes", []),
                "fused_container_type": fused.get("fused_container_type"),
                "would_change_behavior": False,
                "phase4_status": "review_only_conflict",
            }
        )
    return conflicts


def _attach_region_fields(regions: Sequence[MutableMapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> None:
    by_region: Dict[str, List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        for region_id in candidate.get("linked_current_region_ids", []) or []:
            by_region.setdefault(str(region_id), []).append(candidate)

    for region in regions:
        region_id = str(region.get("region_id"))
        linked = by_region.get(region_id)
        if not linked:
            continue
        container_ids = sorted({str(item.get("fused_container_id")) for item in linked if item.get("fused_container_id")})
        candidate_types = sorted({str(item.get("fusion_classification")) for item in linked if item.get("fusion_classification")})
        conflict_flags = sorted({flag for item in linked for flag in _as_list(item.get("conflict_flags"))})
        reason_codes = sorted({code for item in linked for code in _as_list(item.get("reason_codes"))})
        region["diagnostic_model_fusion_container_ids"] = container_ids
        region["diagnostic_model_fusion_candidate_types"] = candidate_types
        region["diagnostic_model_fusion_conflict_flags"] = conflict_flags
        region["diagnostic_model_fusion_reason_codes"] = reason_codes
        region["diagnostic_model_fusion_allowed_future_assist"] = any(bool(item.get("allowed_future_assist")) for item in linked)
        region["diagnostic_model_fusion_review_only"] = any(bool(item.get("review_only")) for item in linked)


def _attach_high_accuracy_bubble_summary_fields(
    audit: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> None:
    audit["high_accuracy_bubble_mode_version"] = HIGH_ACCURACY_BUBBLE_MODE_VERSION
    audit["high_accuracy_bubble_mode_enabled"] = high_accuracy_bubble_mode_enabled()
    audit["high_accuracy_bubble_mode_mutation_allowed"] = False

    if not audit["high_accuracy_bubble_mode_enabled"]:
        audit["high_accuracy_bubble_mode_generated"] = False
        return

    generated = bool(audit.get("model_fusion_assist_generated")) and not bool(audit.get("model_fusion_assist_error"))
    audit["high_accuracy_bubble_mode_generated"] = generated
    audit["high_accuracy_bubble_mode_error"] = audit.get("model_fusion_assist_error")
    audit["high_accuracy_bubble_mode_runtime_sec"] = audit.get("model_fusion_assist_runtime_sec", 0.0)
    audit["high_accuracy_bubble_mode_components"] = _high_accuracy_component_status(audit)
    audit["high_accuracy_bubble_mode_candidate_counts"] = _high_accuracy_candidate_counts(audit, candidates)
    audit["high_accuracy_bubble_mode_conflict_counts"] = _high_accuracy_conflict_counts(conflicts)
    audit["high_accuracy_bubble_mode_fallback_used"] = _high_accuracy_fallback_used(audit)
    _attach_high_accuracy_region_fields(audit.get("regions", []) or [], candidates)


def _high_accuracy_component_status(audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_fusion_evidence": bool(audit.get("model_fusion_assist_generated")),
        "bubble_detection_cache": {
            "enabled": bool(audit.get("bubble_detection_cache_enabled")),
            "hit": bool(audit.get("bubble_detection_cache_hit")),
            "error": audit.get("bubble_detection_cache_error"),
            "invalidation_reason": audit.get("bubble_detection_cache_invalidation_reason"),
        },
        "confidence_policy": True,
        "route_guard_proof_metadata": {
            "enabled": bool(audit.get("model_fusion_route_guard_proof_enabled")),
            "applied": bool(audit.get("model_fusion_route_guard_proof_applied")),
            "errors": audit.get("model_fusion_route_guard_proof_errors", []),
        },
        "ownership_proof_metadata": {
            "enabled": bool(audit.get("model_fusion_ownership_proof_enabled")),
            "applied": bool(audit.get("model_fusion_ownership_proof_applied")),
            "errors": audit.get("model_fusion_ownership_proof_errors", []),
        },
        "text_conservation_proof_metadata": {
            "enabled": bool(audit.get("model_fusion_text_conservation_proof_enabled")),
            "applied": bool(audit.get("model_fusion_text_conservation_proof_applied")),
            "errors": audit.get("model_fusion_text_conservation_proof_errors", []),
        },
        "text_conservation_integration_metadata": {
            "enabled": bool(audit.get("model_fusion_text_conservation_integration_enabled")),
            "generated": bool(audit.get("model_fusion_text_conservation_integration_generated")),
            "errors": audit.get("model_fusion_text_conservation_integration_errors", []),
            "mutation_count": audit.get("model_fusion_text_conservation_integration_mutation_count", 0),
        },
        "render_constraint_calibration_metadata": {
            "enabled": bool(audit.get("model_fusion_render_constraint_calibration_enabled")),
            "generated": bool(audit.get("model_fusion_render_constraint_calibration_generated")),
            "errors": audit.get("model_fusion_render_constraint_calibration_errors", []),
        },
        "cleanup_boundary_calibration_metadata": {
            "enabled": bool(audit.get("model_fusion_cleanup_boundary_calibration_enabled")),
            "generated": bool(audit.get("model_fusion_cleanup_boundary_calibration_generated")),
            "errors": audit.get("model_fusion_cleanup_boundary_calibration_errors", []),
        },
        "mutation_proof": {
            "enabled": False,
            "reason": "high_accuracy_bubble_mode_does_not_enable_mutation_flags",
        },
    }


def _high_accuracy_candidate_counts(
    audit: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_class: Dict[str, int] = {}
    by_candidate_confidence: Dict[str, int] = {}
    by_policy_tier: Dict[str, int] = {}
    for candidate in candidates:
        classification = str(candidate.get("fusion_classification") or "unknown")
        confidence = str(candidate.get("confidence") or "unknown")
        by_class[classification] = by_class.get(classification, 0) + 1
        by_candidate_confidence[confidence] = by_candidate_confidence.get(confidence, 0) + 1

    candidates_by_region = _candidates_by_region(candidates)
    for region in audit.get("regions", []) or []:
        region_id = str(region.get("region_id") or "")
        candidate = _select_route_guard_candidate(region, candidates_by_region.get(region_id, []))
        tier = _route_guard_confidence_tier(region, candidate)
        by_policy_tier[tier] = by_policy_tier.get(tier, 0) + 1

    return {
        "total_candidates": len(candidates),
        "by_fusion_classification": dict(sorted(by_class.items())),
        "by_candidate_confidence": dict(sorted(by_candidate_confidence.items())),
        "by_confidence_tier": dict(sorted(by_policy_tier.items())),
    }


def _high_accuracy_conflict_counts(conflicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_flag: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for conflict in conflicts:
        classification = str(conflict.get("fusion_classification") or "unknown")
        by_class[classification] = by_class.get(classification, 0) + 1
        for flag in _as_list(conflict.get("conflict_flags")):
            flag_text = str(flag)
            by_flag[flag_text] = by_flag.get(flag_text, 0) + 1
    return {
        "total_conflicts": len(conflicts),
        "by_fusion_classification": dict(sorted(by_class.items())),
        "by_conflict_flag": dict(sorted(by_flag.items())),
    }


def _high_accuracy_fallback_used(audit: Mapping[str, Any]) -> bool:
    evidence = audit.get("model_fusion_evidence") or {}
    backend = evidence.get("backend") if isinstance(evidence, Mapping) else {}
    if not isinstance(backend, Mapping):
        return bool(audit.get("model_fusion_assist_error"))
    providers = [
        str(backend.get("kitsumed_provider_used") or ""),
        str(backend.get("ogkalu_provider_used") or ""),
    ]
    return bool(audit.get("model_fusion_assist_error")) or any(
        provider and provider != "CUDAExecutionProvider" for provider in providers
    )


def _attach_high_accuracy_region_fields(
    regions: Sequence[MutableMapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    candidates_by_region = _candidates_by_region(candidates)
    for region in regions:
        region_id = str(region.get("region_id") or "")
        candidate = _select_route_guard_candidate(region, candidates_by_region.get(region_id, []))
        tier = _route_guard_confidence_tier(region, candidate)
        supported_actions = _high_accuracy_supported_actions(candidate)
        region["high_accuracy_bubble_container_id"] = (candidate or {}).get("fused_container_id")
        region["high_accuracy_bubble_confidence_tier"] = tier
        region["high_accuracy_bubble_supported_actions"] = supported_actions
        region["high_accuracy_bubble_review_only"] = bool(
            (candidate or {}).get("review_only")
            or tier in {"text_bubble_review_container", "text_free_review_only", "conflict_preserve_wins", "unsupported_by_model_but_deterministic_valid"}
        )
        region["high_accuracy_bubble_conflict_flags"] = _as_list((candidate or {}).get("conflict_flags"))
        region["high_accuracy_bubble_reason_codes"] = sorted(
            set(
                [
                    f"confidence_tier:{tier}",
                    "high_accuracy_bubble_mode_metadata_only",
                ]
                + [f"candidate_reason:{reason}" for reason in _as_list((candidate or {}).get("reason_codes"))]
                + [f"candidate_conflict:{flag}" for flag in _as_list((candidate or {}).get("conflict_flags"))]
            )
        )
        region["high_accuracy_bubble_would_change_behavior"] = False


def _high_accuracy_supported_actions(candidate: Mapping[str, Any] | None) -> List[str]:
    if candidate is None:
        return []
    classification = str(candidate.get("fusion_classification") or "")
    action = CLASS_TO_ASSIST_TYPE.get(classification)
    if bool(candidate.get("allowed_future_assist")) and action:
        return [action]
    if action == "review_only":
        return ["review_only"]
    return []


def _attach_ownership_proof_fields(audit: MutableMapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    audit["model_fusion_ownership_proof_version"] = MODEL_FUSION_OWNERSHIP_PROOF_VERSION
    audit["model_fusion_ownership_proof_enabled"] = model_fusion_ownership_proof_enabled()
    audit["model_fusion_ownership_proof_applied"] = False
    audit["model_fusion_ownership_proof_errors"] = []

    if not audit["model_fusion_ownership_proof_enabled"]:
        return

    page_id = str(audit.get("page_id") or "")
    allowlisted_regions = OWNERSHIP_PROOF_ALLOWLIST.get(page_id)
    if not allowlisted_regions:
        return

    regions = audit.get("regions", []) or []
    region_by_id: Dict[str, MutableMapping[str, Any]] = {
        str(region.get("region_id")): region
        for region in regions
        if isinstance(region, MutableMapping) and region.get("region_id") is not None
    }
    missing_regions = sorted(rid for rid in allowlisted_regions if rid not in region_by_id)
    if missing_regions:
        audit["model_fusion_ownership_proof_errors"] = [f"allowlisted_regions_missing:{','.join(missing_regions)}"]
        _stamp_ownership_proof_failure(region_by_id, allowlisted_regions, "failed_closed", ["allowlisted_regions_missing"])
        return

    candidate = _ownership_proof_candidate(candidates, page_id, allowlisted_regions)
    if candidate is None:
        audit["model_fusion_ownership_proof_errors"] = ["no_matching_model_fusion_ownership_candidate"]
        _stamp_ownership_proof_failure(
            region_by_id,
            allowlisted_regions,
            "failed_closed",
            ["no_matching_model_fusion_ownership_candidate"],
        )
        return

    result, reason_codes = _evaluate_ownership_proof_candidate(candidate, region_by_id, allowlisted_regions)
    if result != "agreed":
        audit["model_fusion_ownership_proof_errors"] = reason_codes

    linked_region_ids = sorted(allowlisted_regions)
    kitsumed_ids = _as_list(candidate.get("linked_kitsumed_mask_ids"))
    ogkalu_ids = _as_list(candidate.get("linked_ogkalu_detection_ids"))
    deterministic_reason = _deterministic_ownership_reason(region_by_id, linked_region_ids)
    payload = {
        "model_fusion_ownership_proof_candidate_id": candidate.get("candidate_id"),
        "model_fusion_ownership_proof_container_id": candidate.get("fused_container_id"),
        "model_fusion_ownership_proof_linked_region_ids": linked_region_ids,
        "model_fusion_ownership_proof_result": result,
        "model_fusion_ownership_proof_reason_codes": reason_codes,
        "model_fusion_ownership_proof_kitsumed_mask_id": kitsumed_ids[0] if kitsumed_ids else None,
        "model_fusion_ownership_proof_ogkalu_detection_ids": ogkalu_ids,
        "model_fusion_ownership_proof_current_deterministic_reason": deterministic_reason,
        "would_change_behavior": False,
    }
    for region_id in linked_region_ids:
        region_by_id[region_id].update(payload)

    audit["model_fusion_ownership_proof_applied"] = result == "agreed"


def _ownership_proof_candidate(
    candidates: Sequence[Mapping[str, Any]],
    page_id: str,
    allowlisted_regions: set[str],
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("page_id") or "") != page_id:
            continue
        if candidate.get("fusion_classification") != "safe_future_text_ownership_assist":
            continue
        linked = {str(rid) for rid in candidate.get("linked_current_region_ids", []) or []}
        if allowlisted_regions.issubset(linked):
            return candidate
    return None


def _evaluate_ownership_proof_candidate(
    candidate: Mapping[str, Any],
    region_by_id: Mapping[str, Mapping[str, Any]],
    allowlisted_regions: set[str],
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    linked = {str(rid) for rid in candidate.get("linked_current_region_ids", []) or []}
    if not allowlisted_regions.issubset(linked):
        reasons.append("model_candidate_does_not_link_all_allowlisted_regions")
    if candidate.get("fusion_classification") != "safe_future_text_ownership_assist":
        reasons.append("candidate_not_text_ownership_assist")
    if candidate.get("confidence") != "high":
        reasons.append("candidate_not_high_confidence")
    if candidate.get("conflict_flags"):
        reasons.append("model_fusion_conflict_present")
    if not candidate.get("linked_kitsumed_mask_ids"):
        reasons.append("no_kitsumed_speech_mask")
    if "ogkalu_text_bubble_strengthens_ownership" not in _as_list(candidate.get("reason_codes")):
        reasons.append("no_ogkalu_text_bubble_support")

    group_ids = set()
    deterministic_reasons = set()
    for region_id in sorted(allowlisted_regions):
        region = region_by_id.get(region_id) or {}
        if region.get("semantic_class") != "speech_bubble":
            reasons.append(f"{region_id}:not_speech_bubble")
        if region.get("is_decorative") or region.get("is_sfx") or region.get("is_background") or region.get("is_sign"):
            reasons.append(f"{region_id}:non_speech_visual_role")
        if not _meaningful_japanese(region.get("ocr_text")):
            reasons.append(f"{region_id}:no_meaningful_japanese_ocr")
        reason = str(region.get("classification_reason") or "").strip()
        if reason:
            deterministic_reasons.add(reason)
        if reason != BUBBLE_LOCAL_OWNERSHIP_REASON:
            reasons.append(f"{region_id}:deterministic_reason_not_bubble_local_ownership")
        group_id = str(region.get("group_id") or "").strip()
        if group_id:
            group_ids.add(group_id)
    if len(group_ids) != 1:
        reasons.append("regions_do_not_share_single_group_id")
    if BUBBLE_LOCAL_OWNERSHIP_REASON not in deterministic_reasons:
        reasons.append("deterministic_bubble_local_ownership_missing")

    if reasons:
        return "disagreed", sorted(set(reasons))

    return (
        "agreed",
        sorted(
            {
                "allowlist:020_r017_r018_model_fusion_ownership_proof",
                "current_deterministic_bubble_local_ownership_agrees",
                "kitsumed_same_speech_mask",
                "ogkalu_text_bubble_support",
                "same_fused_speech_container",
                "same_group_id",
                "no_sfx_decorative_or_caption_conflict",
                "meaningful_japanese_source_text",
                "metadata_only_no_behavior_change",
            }
        ),
    )


def _stamp_ownership_proof_failure(
    region_by_id: Mapping[str, MutableMapping[str, Any]],
    region_ids: set[str],
    result: str,
    reason_codes: Sequence[str],
) -> None:
    for region_id in sorted(region_ids):
        region = region_by_id.get(region_id)
        if region is None:
            continue
        region.update(
            {
                "model_fusion_ownership_proof_candidate_id": None,
                "model_fusion_ownership_proof_container_id": None,
                "model_fusion_ownership_proof_linked_region_ids": sorted(region_ids),
                "model_fusion_ownership_proof_result": result,
                "model_fusion_ownership_proof_reason_codes": list(reason_codes),
                "model_fusion_ownership_proof_kitsumed_mask_id": None,
                "model_fusion_ownership_proof_ogkalu_detection_ids": [],
                "model_fusion_ownership_proof_current_deterministic_reason": None,
                "would_change_behavior": False,
            }
        )


def _deterministic_ownership_reason(
    region_by_id: Mapping[str, Mapping[str, Any]],
    region_ids: Sequence[str],
) -> str:
    reasons = {
        str((region_by_id.get(region_id) or {}).get("classification_reason") or "").strip()
        for region_id in region_ids
    }
    reasons.discard("")
    if not reasons:
        return ""
    return "+".join(sorted(reasons))


def _meaningful_japanese(text: Any) -> bool:
    value = str(text or "")
    has_japanese = any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in value)
    body = "".join(
        ch
        for ch in value
        if ch.strip()
        and ch
        not in "。，、！？：；…‥・･ー-—―－〜～「」『』（）()[]【】<>〈〉《》“”‘’\"' ."
    )
    return has_japanese and len(body) >= 2


def _attach_text_conservation_proof_fields(audit: MutableMapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    audit["model_fusion_text_conservation_proof_version"] = MODEL_FUSION_TEXT_CONSERVATION_PROOF_VERSION
    audit["model_fusion_text_conservation_proof_enabled"] = model_fusion_text_conservation_proof_enabled()
    audit["model_fusion_text_conservation_proof_applied"] = False
    audit["model_fusion_text_conservation_proof_errors"] = []

    if not audit["model_fusion_text_conservation_proof_enabled"]:
        return

    page_id = str(audit.get("page_id") or "")
    allowlisted_regions = TEXT_CONSERVATION_PROOF_ALLOWLIST.get(page_id)
    if not allowlisted_regions:
        return

    regions = audit.get("regions", []) or []
    region_by_id: Dict[str, MutableMapping[str, Any]] = {
        str(region.get("region_id")): region
        for region in regions
        if isinstance(region, MutableMapping) and region.get("region_id") is not None
    }
    missing_regions = sorted(rid for rid in allowlisted_regions if rid not in region_by_id)
    if missing_regions:
        reason_codes = [f"allowlisted_regions_missing:{','.join(missing_regions)}"]
        audit["model_fusion_text_conservation_proof_errors"] = reason_codes
        _stamp_text_conservation_proof_failure(region_by_id, allowlisted_regions, "failed_closed", reason_codes)
        return

    proof = _evaluate_text_conservation_proof(candidates, region_by_id, allowlisted_regions)
    if proof["result"] not in {"agreed", "partial_agreement"}:
        audit["model_fusion_text_conservation_proof_errors"] = proof["reason_codes"]

    linked_region_ids = sorted(allowlisted_regions)
    phrase_status = proof["phrase_status"]
    for region_id in linked_region_ids:
        region_payload = proof["per_region"].get(region_id, {})
        region_by_id[region_id].update(
            {
                "model_fusion_text_conservation_proof_candidate_id": region_payload.get("candidate_id"),
                "model_fusion_text_conservation_container_id": region_payload.get("container_id"),
                "model_fusion_text_conservation_linked_region_ids": linked_region_ids,
                "model_fusion_text_conservation_result": proof["result"],
                "model_fusion_text_conservation_reason_codes": proof["reason_codes"],
                "model_fusion_text_conservation_kitsumed_mask_id": region_payload.get("kitsumed_mask_id"),
                "model_fusion_text_conservation_ogkalu_detection_ids": region_payload.get("ogkalu_detection_ids", []),
                "model_fusion_text_conservation_deterministic_reason": proof["deterministic_reason"],
                "model_fusion_text_conservation_phrase_status": phrase_status,
                "would_change_behavior": False,
            }
        )

    audit["model_fusion_text_conservation_proof_applied"] = proof["result"] in {"agreed", "partial_agreement"}


def _evaluate_text_conservation_proof(
    candidates: Sequence[Mapping[str, Any]],
    region_by_id: Mapping[str, Mapping[str, Any]],
    allowlisted_regions: set[str],
) -> Dict[str, Any]:
    upper_candidate = _text_conservation_candidate(candidates, {"r004", "r006"}, preferred_class="safe_future_missed_text_hint")
    if upper_candidate is None:
        upper_candidate = _text_conservation_candidate(candidates, {"r004", "r006"})
    lower_candidate = _text_conservation_candidate(candidates, {"r005"})

    phrase_status = _build_030_phrase_status(region_by_id)
    phrase_passed = all(item.get("exactly_once") == "yes" for item in phrase_status)
    reasons: List[str] = [
        "allowlist:030_lower_left_model_fusion_text_conservation_proof",
        "metadata_only_no_behavior_change",
    ]
    errors: List[str] = []

    if upper_candidate is None:
        errors.append("no_model_evidence_for_recovered_upper_fragments")
    else:
        if upper_candidate.get("fusion_classification") == "safe_future_missed_text_hint":
            reasons.append("ogkalu_text_bubble_supports_recovered_upper_fragments")
        else:
            reasons.append("ogkalu_bubble_or_ambiguous_evidence_supports_recovered_upper_fragments")
        if upper_candidate.get("linked_kitsumed_mask_ids"):
            reasons.append("kitsumed_mask_supports_upper_fragments")
        else:
            reasons.append("upper_fragments_are_ogkalu_only_review_supported")
        if upper_candidate.get("conflict_flags"):
            errors.append("upper_fragment_model_conflict_present")

    if lower_candidate is None:
        errors.append("no_model_evidence_for_lower_food_line")
    else:
        if lower_candidate.get("linked_kitsumed_mask_ids"):
            reasons.append("kitsumed_mask_supports_lower_food_line")
        if lower_candidate.get("linked_ogkalu_detection_ids"):
            reasons.append("ogkalu_supports_lower_food_line")
        if lower_candidate.get("conflict_flags"):
            errors.append("lower_food_line_model_conflict_present")

    deterministic_ok = _text_conservation_deterministic_ok(region_by_id)
    if deterministic_ok:
        reasons.append("deterministic_text_conservation_recovery_agrees")
    else:
        errors.append("deterministic_text_conservation_recovery_missing_or_changed")

    if phrase_passed:
        reasons.append("phrase_conservation_passed")
    else:
        errors.append("phrase_conservation_failed")

    if _regions_have_sfx_or_background_role(region_by_id, allowlisted_regions):
        errors.append("target_regions_have_non_speech_visual_role")
    else:
        reasons.append("no_sfx_decorative_or_caption_conflict")

    if upper_candidate and lower_candidate and upper_candidate.get("fused_container_id") != lower_candidate.get("fused_container_id"):
        reasons.append("model_evidence_split_across_upper_and_lower_containers_review_only")

    if errors:
        result = "disagreed"
        reasons.extend(errors)
    elif upper_candidate and lower_candidate and upper_candidate.get("fused_container_id") == lower_candidate.get("fused_container_id"):
        result = "agreed"
        reasons.append("same_model_fusion_container_for_all_target_regions")
    else:
        result = "partial_agreement"

    per_region = {
        "r004": _text_conservation_region_payload(upper_candidate),
        "r006": _text_conservation_region_payload(upper_candidate),
        "r005": _text_conservation_region_payload(lower_candidate),
    }
    return {
        "result": result,
        "reason_codes": sorted(set(reasons)),
        "deterministic_reason": ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON if deterministic_ok else "",
        "phrase_status": phrase_status,
        "per_region": per_region,
    }


def _text_conservation_candidate(
    candidates: Sequence[Mapping[str, Any]],
    required_regions: set[str],
    *,
    preferred_class: str | None = None,
) -> Mapping[str, Any] | None:
    matches: List[Mapping[str, Any]] = []
    for candidate in candidates:
        linked = {str(rid) for rid in candidate.get("linked_current_region_ids", []) or []}
        if not required_regions.issubset(linked):
            continue
        if preferred_class and candidate.get("fusion_classification") == preferred_class:
            return candidate
        matches.append(candidate)
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (
            item.get("fusion_classification") != "safe_future_missed_text_hint",
            item.get("confidence") != "high",
            str(item.get("candidate_id") or ""),
        ),
    )[0]


def _text_conservation_region_payload(candidate: Mapping[str, Any] | None) -> Dict[str, Any]:
    if candidate is None:
        return {
            "candidate_id": None,
            "container_id": None,
            "kitsumed_mask_id": None,
            "ogkalu_detection_ids": [],
        }
    kitsumed_ids = _as_list(candidate.get("linked_kitsumed_mask_ids"))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "container_id": candidate.get("fused_container_id"),
        "kitsumed_mask_id": kitsumed_ids[0] if kitsumed_ids else None,
        "ogkalu_detection_ids": _as_list(candidate.get("linked_ogkalu_detection_ids")),
    }


def _text_conservation_deterministic_ok(region_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    r004 = region_by_id.get("r004") or {}
    r006 = region_by_id.get("r006") or {}
    r005 = region_by_id.get("r005") or {}
    return (
        r004.get("semantic_class") == "speech_bubble"
        and r006.get("semantic_class") == "speech_bubble"
        and r005.get("semantic_class") == "speech_bubble"
        and r004.get("classification_reason") == ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON
        and r006.get("classification_reason") == ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON
        and r006.get("cleanup_mode") == "transferred_to_speech_text_conservation_anchor"
        and bool(str(r004.get("translated_text") or "").strip())
        and not bool(str(r006.get("translated_text") or "").strip())
    )


def _regions_have_sfx_or_background_role(region_by_id: Mapping[str, Mapping[str, Any]], region_ids: set[str]) -> bool:
    for region_id in region_ids:
        region = region_by_id.get(region_id) or {}
        if region.get("is_decorative") or region.get("is_sfx") or region.get("is_background") or region.get("is_sign"):
            return True
    return False


def _build_030_phrase_status(region_by_id: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    _ = region_by_id
    return []


def _text_marker_present(text: str, marker: Any) -> bool:
    if isinstance(marker, (list, tuple, set)):
        return any(_text_marker_present(text, item) for item in marker)
    return str(marker) in str(text or "")


def _phrase_source_marker_present(source_phrase: str, ocr_text: str) -> bool:
    compact_phrase = re.sub(r"[.\s…‥・･]+", "", str(source_phrase or ""))
    compact_text = re.sub(r"[.\s…‥・･]+", "", str(ocr_text or ""))
    return bool(compact_phrase and compact_phrase in compact_text)


def _phrase_duplicate_rendered(spec: Mapping[str, Any], regions: Mapping[str, Mapping[str, Any]]) -> bool:
    expected_region_id = str(spec.get("translation_region_id") or "")
    markers = _flatten_text_markers(spec.get("render_markers", []) or [])
    if not markers:
        return False
    rendered_regions = 0
    for region_id, region in regions.items():
        wrapped_text = "".join(str(line or "") for line in region.get("wrapped_lines") or [])
        translated = str(region.get("translated_text") or "")
        if all(marker in wrapped_text or marker in translated for marker in markers):
            rendered_regions += 1
            if region_id != expected_region_id:
                return True
    return rendered_regions > 1


def _flatten_text_markers(markers: Sequence[Any]) -> List[str]:
    flattened: List[str] = []
    for marker in markers:
        if isinstance(marker, (list, tuple, set)):
            flattened.extend(_flatten_text_markers(list(marker)))
        else:
            flattened.append(str(marker))
    return flattened


def _stamp_text_conservation_proof_failure(
    region_by_id: Mapping[str, MutableMapping[str, Any]],
    region_ids: set[str],
    result: str,
    reason_codes: Sequence[str],
) -> None:
    for region_id in sorted(region_ids):
        region = region_by_id.get(region_id)
        if region is None:
            continue
        region.update(
            {
                "model_fusion_text_conservation_proof_candidate_id": None,
                "model_fusion_text_conservation_container_id": None,
                "model_fusion_text_conservation_linked_region_ids": sorted(region_ids),
                "model_fusion_text_conservation_result": result,
                "model_fusion_text_conservation_reason_codes": list(reason_codes),
                "model_fusion_text_conservation_kitsumed_mask_id": None,
                "model_fusion_text_conservation_ogkalu_detection_ids": [],
                "model_fusion_text_conservation_deterministic_reason": None,
                "model_fusion_text_conservation_phrase_status": [],
                "would_change_behavior": False,
            }
        )


def _attach_text_conservation_integration_fields(
    audit: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    audit["model_fusion_text_conservation_integration_version"] = MODEL_FUSION_TEXT_CONSERVATION_INTEGRATION_VERSION
    audit["model_fusion_text_conservation_integration_enabled"] = model_fusion_text_conservation_integration_enabled()
    audit["model_fusion_text_conservation_integration_generated"] = False
    audit["model_fusion_text_conservation_integration_errors"] = []
    audit["model_fusion_text_conservation_integration_candidates"] = []
    audit["model_fusion_text_conservation_integration_blocked_references"] = []
    audit["model_fusion_text_conservation_integration_summary"] = {}
    audit["model_fusion_text_conservation_integration_mutation_count"] = 0

    for region in audit.get("regions", []) or []:
        if not isinstance(region, MutableMapping):
            continue
        region["model_fusion_text_conservation_integration_candidate_ids"] = []
        region["model_fusion_text_conservation_integration_status"] = "disabled"
        region["model_fusion_text_conservation_integration_action"] = None
        region["model_fusion_text_conservation_integration_reason_codes"] = []
        region["model_fusion_text_conservation_integration_would_change_behavior"] = False

    if not audit["model_fusion_text_conservation_integration_enabled"]:
        return

    try:
        page_id = str(audit.get("page_id") or "")
        regions = audit.get("regions", []) or []
        region_by_id: Dict[str, MutableMapping[str, Any]] = {
            str(region.get("region_id")): region
            for region in regions
            if isinstance(region, MutableMapping) and region.get("region_id") is not None
        }
        records: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, ...]] = set()

        def add_record(record: Dict[str, Any] | None) -> None:
            if not record:
                return
            key = tuple([str(record.get("candidate_type") or ""), *sorted(str(rid) for rid in record.get("region_ids", []) or [])])
            if key in seen_keys:
                return
            seen_keys.add(key)
            record["candidate_id"] = f"{page_id}:text_conservation_integration:{len(records):03d}"
            records.append(record)

        add_record(_shared_ownership_integration_record(page_id, region_by_id, candidates))
        add_record(_adjacent_vertical_text_conservation_integration_record(page_id, region_by_id, candidates))
        for region in regions:
            if not isinstance(region, Mapping):
                continue
            add_record(_recovered_speech_integration_record(page_id, region, candidates))
        for candidate in candidates:
            add_record(_model_review_text_conservation_record(page_id, candidate, region_by_id))
            add_record(_model_ocr_gap_record(page_id, candidate))

        blocked = _blocked_text_conservation_references(page_id, region_by_id)
        _stamp_text_conservation_integration_regions(region_by_id, records)

        audit["model_fusion_text_conservation_integration_candidates"] = records
        audit["model_fusion_text_conservation_integration_blocked_references"] = blocked
        audit["model_fusion_text_conservation_integration_summary"] = _text_conservation_integration_summary(records, blocked)
        audit["model_fusion_text_conservation_integration_generated"] = True
    except Exception as exc:  # pragma: no cover - debug isolation
        audit["model_fusion_text_conservation_integration_generated"] = False
        audit["model_fusion_text_conservation_integration_errors"] = [f"{type(exc).__name__}: {exc}"]


def _shared_ownership_integration_record(
    page_id: str,
    region_by_id: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    target_ids = OWNERSHIP_PROOF_ALLOWLIST.get(page_id)
    if not target_ids or not target_ids.issubset(set(region_by_id)):
        return None
    if not any(
        (region_by_id.get(rid) or {}).get("classification_reason") == BUBBLE_LOCAL_OWNERSHIP_REASON
        for rid in target_ids
    ):
        return None
    candidate = _best_candidate_for_regions(candidates, target_ids, preferred_class="safe_future_text_ownership_assist")
    return _text_conservation_record(
        page_id=page_id,
        candidate_type="shared_bubble_ownership_gap",
        region_ids=sorted(target_ids),
        regions=region_by_id,
        candidate=candidate,
        deterministic_reason=BUBBLE_LOCAL_OWNERSHIP_REASON,
        model_evidence_status=_model_evidence_status_for_regions(region_by_id, target_ids, candidate),
        ocr_source_status=_ocr_source_status(region_by_id, target_ids),
        phrase_conservation_status="not_evaluated",
        phrase_status=[],
        proposed_action="support_existing_recovery",
        reason_codes=[
            "existing_deterministic_shared_bubble_ownership",
            "metadata_only_no_merge_or_suppress_mutation",
        ],
    )


def _adjacent_vertical_text_conservation_integration_record(
    page_id: str,
    region_by_id: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    target_ids = TEXT_CONSERVATION_PROOF_ALLOWLIST.get(page_id)
    if not target_ids or not target_ids.issubset(set(region_by_id)):
        return None
    if not any(
        (region_by_id.get(rid) or {}).get("classification_reason") == ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON
        for rid in target_ids
    ):
        return None
    upper_candidate = _text_conservation_candidate(candidates, {"r004", "r006"}, preferred_class="safe_future_missed_text_hint")
    if upper_candidate is None:
        upper_candidate = _text_conservation_candidate(candidates, {"r004", "r006"})
    lower_candidate = _text_conservation_candidate(candidates, {"r005"})
    merged_candidate = _merge_text_conservation_candidates(page_id, upper_candidate, lower_candidate)
    phrase_status = _build_030_phrase_status(region_by_id)
    phrase_conservation_status = _phrase_conservation_status_from_items(phrase_status)
    model_status = "partial_agreement"
    if upper_candidate and lower_candidate and upper_candidate.get("fused_container_id") == lower_candidate.get("fused_container_id"):
        model_status = "strong_agreement"
    elif not upper_candidate and not lower_candidate:
        model_status = "model_missing_but_deterministic_valid"
    return _text_conservation_record(
        page_id=page_id,
        candidate_type="adjacent_vertical_speech_fragments",
        region_ids=sorted(target_ids),
        regions=region_by_id,
        candidate=merged_candidate,
        deterministic_reason=ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON,
        model_evidence_status=model_status,
        ocr_source_status=_ocr_source_status(region_by_id, target_ids),
        phrase_conservation_status=phrase_conservation_status,
        phrase_status=phrase_status,
        proposed_action="support_existing_recovery",
        reason_codes=[
            "existing_deterministic_adjacent_vertical_text_conservation",
            "metadata_only_no_auto_recovery_from_model_evidence",
        ],
    )


def _recovered_speech_integration_record(
    page_id: str,
    region: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    if region.get("classification_reason") != "speech_bubble_missed_text_recovery":
        return None
    region_id = str(region.get("region_id") or "")
    if not region_id:
        return None
    candidate = _best_candidate_for_regions(candidates, {region_id})
    region_map = {region_id: region}
    return _text_conservation_record(
        page_id=page_id,
        candidate_type="ocr_coverage_gap_inside_speech_container",
        region_ids=[region_id],
        regions=region_map,
        candidate=candidate,
        deterministic_reason="speech_bubble_missed_text_recovery",
        model_evidence_status=_model_evidence_status_for_regions(region_map, {region_id}, candidate),
        ocr_source_status=_ocr_source_status(region_map, {region_id}),
        phrase_conservation_status="not_evaluated",
        phrase_status=[],
        proposed_action="support_existing_recovery",
        reason_codes=[
            "existing_deterministic_missed_speech_recovery",
            "model_evidence_must_not_block_valid_deterministic_recovery",
        ],
    )


def _model_review_text_conservation_record(
    page_id: str,
    candidate: Mapping[str, Any],
    region_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any] | None:
    if candidate.get("fusion_classification") != "safe_future_missed_text_hint":
        return None
    region_ids = [str(rid) for rid in candidate.get("linked_current_region_ids", []) or [] if str(rid) in region_by_id]
    if not region_ids:
        return None
    region_set = set(region_ids)
    if any(_route_guard_is_preserve_decorative_sfx(region_by_id[rid]) or _route_guard_is_caption_or_background(region_by_id[rid]) for rid in region_ids):
        return None
    return _text_conservation_record(
        page_id=page_id,
        candidate_type="model_supported_text_bubble_review",
        region_ids=sorted(region_ids),
        regions=region_by_id,
        candidate=candidate,
        deterministic_reason=_combined_deterministic_reason(region_by_id, region_ids),
        model_evidence_status="review_only",
        ocr_source_status=_ocr_source_status(region_by_id, region_set),
        phrase_conservation_status="not_evaluated",
        phrase_status=[],
        proposed_action="review_only",
        reason_codes=[
            "ogkalu_text_bubble_evidence_is_review_only",
            "model_only_evidence_does_not_trigger_recovery",
        ],
    )


def _model_ocr_gap_record(page_id: str, candidate: Mapping[str, Any]) -> Dict[str, Any] | None:
    if candidate.get("fusion_classification") != "safe_future_text_container_assist":
        return None
    if _as_list(candidate.get("linked_current_region_ids")):
        return None
    if candidate.get("confidence") != "high":
        return None
    if _as_list(candidate.get("conflict_flags")):
        return None
    if not _as_list(candidate.get("linked_kitsumed_mask_ids")):
        return None
    return _text_conservation_record(
        page_id=page_id,
        candidate_type="ocr_coverage_gap_inside_strong_bubble_container",
        region_ids=[],
        regions={},
        candidate=candidate,
        deterministic_reason="",
        model_evidence_status="review_only",
        ocr_source_status="reocr_required",
        phrase_conservation_status="not_evaluated",
        phrase_status=[],
        proposed_action="request_reocr",
        reason_codes=[
            "model_container_has_no_linked_current_ocr_region",
            "source_text_must_come_from_reocr_not_model_output",
        ],
    )


def _text_conservation_record(
    *,
    page_id: str,
    candidate_type: str,
    region_ids: Sequence[str],
    regions: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Any] | None,
    deterministic_reason: str,
    model_evidence_status: str,
    ocr_source_status: str,
    phrase_conservation_status: str,
    phrase_status: Sequence[Mapping[str, Any]],
    proposed_action: str,
    reason_codes: Sequence[str],
) -> Dict[str, Any]:
    region_ids_list = [str(rid) for rid in region_ids]
    tiers = sorted(
        {
            str((regions.get(rid) or {}).get("diagnostic_bubble_confidence_tier"))
            for rid in region_ids_list
            if (regions.get(rid) or {}).get("diagnostic_bubble_confidence_tier")
        }
    )
    confidence_tier = "+".join(tiers) if tiers else _candidate_confidence_tier(candidate)
    container_ids = sorted(
        {
            str((regions.get(rid) or {}).get("diagnostic_bubble_container_id"))
            for rid in region_ids_list
            if (regions.get(rid) or {}).get("diagnostic_bubble_container_id")
        }
    )
    container_id = "+".join(container_ids) if container_ids else ((candidate or {}).get("fused_container_id"))
    container_types = sorted(
        {
            str((regions.get(rid) or {}).get("diagnostic_bubble_container_type"))
            for rid in region_ids_list
            if (regions.get(rid) or {}).get("diagnostic_bubble_container_type")
        }
    )
    container_type = "+".join(container_types) if container_types else "speech_bubble"
    candidate_reason_codes = _as_list((candidate or {}).get("reason_codes"))
    return {
        "candidate_id": None,
        "candidate_type": candidate_type,
        "page_id": page_id,
        "region_ids": region_ids_list,
        "container_id": container_id,
        "container_type": container_type,
        "confidence_tier": confidence_tier,
        "kitsumed_mask_ids": _as_list((candidate or {}).get("linked_kitsumed_mask_ids")),
        "ogkalu_detection_ids": _as_list((candidate or {}).get("linked_ogkalu_detection_ids")),
        "deterministic_route_recovery_reason": deterministic_reason,
        "model_evidence_status": model_evidence_status,
        "ocr_source_status": ocr_source_status,
        "phrase_conservation_status": phrase_conservation_status,
        "phrase_status": list(phrase_status),
        "proposed_action": proposed_action,
        "reason_codes": sorted(set(list(reason_codes) + candidate_reason_codes)),
        "would_change_behavior": False,
        "phase4b19_status": "metadata_only",
        "human_review_required": proposed_action != "support_existing_recovery" or model_evidence_status != "strong_agreement",
    }


def _merge_text_conservation_candidates(
    page_id: str,
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    if first is None and second is None:
        return None
    linked_regions = sorted(
        set(_as_list((first or {}).get("linked_current_region_ids")) + _as_list((second or {}).get("linked_current_region_ids")))
    )
    kitsumed = sorted(set(_as_list((first or {}).get("linked_kitsumed_mask_ids")) + _as_list((second or {}).get("linked_kitsumed_mask_ids"))))
    ogkalu = sorted(set(_as_list((first or {}).get("linked_ogkalu_detection_ids")) + _as_list((second or {}).get("linked_ogkalu_detection_ids"))))
    containers = [str(item.get("fused_container_id")) for item in (first, second) if item and item.get("fused_container_id")]
    reason_codes = sorted(set(_as_list((first or {}).get("reason_codes")) + _as_list((second or {}).get("reason_codes"))))
    return {
        "candidate_id": f"{page_id}:merged_text_conservation_model_evidence",
        "fused_container_id": "+".join(sorted(set(containers))) if containers else None,
        "linked_current_region_ids": linked_regions,
        "linked_kitsumed_mask_ids": kitsumed,
        "linked_ogkalu_detection_ids": ogkalu,
        "reason_codes": reason_codes,
        "confidence": "medium",
    }


def _best_candidate_for_regions(
    candidates: Sequence[Mapping[str, Any]],
    region_ids: set[str],
    *,
    preferred_class: str | None = None,
) -> Mapping[str, Any] | None:
    matches = []
    for candidate in candidates:
        linked = {str(rid) for rid in candidate.get("linked_current_region_ids", []) or []}
        if not region_ids.issubset(linked):
            continue
        if preferred_class and candidate.get("fusion_classification") == preferred_class:
            return candidate
        matches.append(candidate)
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda item: (
            bool(_as_list(item.get("conflict_flags"))),
            item.get("fusion_classification") not in SAFE_FUTURE_CLASSES,
            item.get("confidence") != "high",
            str(item.get("candidate_id") or ""),
        ),
    )[0]


def _model_evidence_status_for_regions(
    region_by_id: Mapping[str, Mapping[str, Any]],
    region_ids: set[str],
    candidate: Mapping[str, Any] | None,
) -> str:
    if any(_route_guard_is_preserve_decorative_sfx(region_by_id.get(rid) or {}) for rid in region_ids):
        return "blocked_conflict"
    if any(_route_guard_is_caption_or_background(region_by_id.get(rid) or {}) for rid in region_ids):
        return "blocked_conflict"
    tiers = {
        str((region_by_id.get(rid) or {}).get("diagnostic_bubble_confidence_tier") or "")
        for rid in region_ids
    }
    tiers.discard("")
    if "conflict_preserve_wins" in tiers:
        return "blocked_conflict"
    if "strong_model_container" in tiers:
        return "strong_agreement"
    if "mask_primary_container" in tiers or "text_bubble_review_container" in tiers:
        return "partial_agreement"
    if "unsupported_by_model_but_deterministic_valid" in tiers or candidate is None:
        return "model_missing_but_deterministic_valid"
    return "review_only"


def _ocr_source_status(region_by_id: Mapping[str, Mapping[str, Any]], region_ids: set[str]) -> str:
    if not region_ids:
        return "reocr_required"
    existing = [region_by_id.get(rid) or {} for rid in region_ids]
    if all(str(region.get("ocr_text") or "").strip() for region in existing):
        return "existing_ocr_only"
    if any(str(region.get("ocr_text") or "").strip() for region in existing):
        return "reocr_available"
    return "no_text_source_available"


def _phrase_conservation_status_from_items(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "not_evaluated"
    if any(item.get("duplicate_status") == "duplicate" for item in items):
        return "duplicate_phrase"
    if any(item.get("exactly_once") != "yes" for item in items):
        return "missing_phrase"
    return "complete"


def _candidate_confidence_tier(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return "unsupported_by_model_but_deterministic_valid"
    classification = str(candidate.get("fusion_classification") or "")
    confidence = str(candidate.get("confidence") or "")
    if classification == "safe_future_text_ownership_assist" and confidence == "high":
        return "strong_model_container"
    if _as_list(candidate.get("linked_kitsumed_mask_ids")):
        return "mask_primary_container"
    if _as_list(candidate.get("linked_ogkalu_detection_ids")):
        return "text_bubble_review_container"
    return "review_only"


def _combined_deterministic_reason(region_by_id: Mapping[str, Mapping[str, Any]], region_ids: Sequence[str]) -> str:
    reasons = {
        str((region_by_id.get(rid) or {}).get("classification_reason") or "")
        for rid in region_ids
    }
    reasons.discard("")
    return "+".join(sorted(reasons))


def _blocked_text_conservation_references(
    page_id: str,
    region_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    blocked: List[Dict[str, Any]] = []
    for region_id, region in sorted(region_by_id.items()):
        if not (_route_guard_is_preserve_decorative_sfx(region) or _route_guard_is_caption_or_background(region)):
            continue
        reason_codes = ["blocked_from_missed_text_recovery"]
        if _route_guard_is_preserve_decorative_sfx(region):
            reason_codes.append("sfx_decorative_preserve_wins")
        if _route_guard_is_caption_or_background(region):
            reason_codes.append("caption_background_not_missed_speech")
        blocked.append(
            {
                "page_id": page_id,
                "region_id": region_id,
                "semantic_class": region.get("semantic_class"),
                "cleanup_mode": region.get("cleanup_mode"),
                "classification_reason": region.get("classification_reason"),
                "confidence_tier": region.get("diagnostic_bubble_confidence_tier"),
                "reason_codes": reason_codes,
                "proposed_action": "block_auto_recovery",
                "would_change_behavior": False,
            }
        )
    return blocked


def _stamp_text_conservation_integration_regions(
    region_by_id: Mapping[str, MutableMapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    by_region: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        for region_id in record.get("region_ids", []) or []:
            by_region.setdefault(str(region_id), []).append(record)
    for region_id, region in region_by_id.items():
        linked = by_region.get(region_id, [])
        if not linked:
            region["model_fusion_text_conservation_integration_status"] = "not_applicable"
            continue
        region["model_fusion_text_conservation_integration_candidate_ids"] = [
            str(item.get("candidate_id")) for item in linked if item.get("candidate_id")
        ]
        statuses = sorted({str(item.get("model_evidence_status") or "") for item in linked if item.get("model_evidence_status")})
        actions = sorted({str(item.get("proposed_action") or "") for item in linked if item.get("proposed_action")})
        region["model_fusion_text_conservation_integration_status"] = "+".join(statuses) if statuses else "review_only"
        region["model_fusion_text_conservation_integration_action"] = "+".join(actions) if actions else "review_only"
        region["model_fusion_text_conservation_integration_reason_codes"] = sorted(
            {str(reason) for item in linked for reason in _as_list(item.get("reason_codes"))}
        )
        region["model_fusion_text_conservation_integration_would_change_behavior"] = False


def _text_conservation_integration_summary(
    records: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_status: Dict[str, int] = {}
    by_action: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for record in records:
        status = str(record.get("model_evidence_status") or "unknown")
        action = str(record.get("proposed_action") or "unknown")
        ctype = str(record.get("candidate_type") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
        by_type[ctype] = by_type.get(ctype, 0) + 1
    return {
        "total_candidates": len(records),
        "blocked_reference_count": len(blocked),
        "by_model_evidence_status": by_status,
        "by_proposed_action": by_action,
        "by_candidate_type": by_type,
        "mutation_count": 0,
        "proof_mode": "metadata_only",
    }


def _attach_route_guard_proof_fields(audit: MutableMapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    audit["model_fusion_route_guard_proof_version"] = MODEL_FUSION_ROUTE_GUARD_PROOF_VERSION
    audit["model_fusion_route_guard_proof_enabled"] = model_fusion_route_guard_proof_enabled()
    audit["model_fusion_route_guard_proof_applied"] = False
    audit["model_fusion_route_guard_proof_errors"] = []

    if not audit["model_fusion_route_guard_proof_enabled"]:
        return

    page_id = str(audit.get("page_id") or "")
    reference_regions = ROUTE_GUARD_PROOF_REFERENCES.get(page_id, {})
    if not reference_regions:
        return

    regions = audit.get("regions", []) or []
    region_by_id: Dict[str, MutableMapping[str, Any]] = {
        str(region.get("region_id")): region
        for region in regions
        if isinstance(region, MutableMapping) and region.get("region_id") is not None
    }
    missing_regions = sorted(rid for rid in reference_regions if rid not in region_by_id)
    errors: List[str] = []
    if missing_regions:
        errors.append(f"allowlisted_route_guard_regions_missing:{','.join(missing_regions)}")

    candidates_by_region = _candidates_by_region(candidates)
    stamped = 0
    for region_id, case_id in sorted(reference_regions.items()):
        region = region_by_id.get(region_id)
        if region is None:
            continue
        linked_candidates = candidates_by_region.get(region_id, [])
        proof = _evaluate_route_guard_region(region, linked_candidates, case_id)
        if proof["result"] == "failed_closed":
            errors.extend(proof["reason_codes"])
        region.update(_route_guard_region_payload(proof, region))
        stamped += 1

    audit["model_fusion_route_guard_proof_errors"] = sorted(set(errors))
    audit["model_fusion_route_guard_proof_applied"] = stamped > 0 and not errors


def _candidates_by_region(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for candidate in candidates:
        for region_id in candidate.get("linked_current_region_ids", []) or []:
            out.setdefault(str(region_id), []).append(candidate)
    return out


def _evaluate_route_guard_region(
    region: Mapping[str, Any],
    linked_candidates: Sequence[Mapping[str, Any]],
    case_id: str,
) -> Dict[str, Any]:
    candidate = _select_route_guard_candidate(region, linked_candidates)
    tier = _route_guard_confidence_tier(region, candidate)
    current_role = _route_guard_current_role(region)
    result, suggested_guard, human_review, evaluation_reasons = _route_guard_result_for_region(
        current_role=current_role,
        tier=tier,
        candidate=candidate,
    )
    reason_codes = sorted(
        set(
            [
                f"reference_case:{case_id}",
                f"confidence_tier:{tier}",
                f"current_role:{current_role}",
                "metadata_only_no_behavior_change",
            ]
            + evaluation_reasons
            + [f"candidate_reason:{reason}" for reason in _as_list((candidate or {}).get("reason_codes"))]
            + [f"candidate_conflict:{flag}" for flag in _as_list((candidate or {}).get("conflict_flags"))]
        )
    )
    return {
        "candidate": candidate,
        "confidence_tier": tier,
        "result": result,
        "suggested_guard": suggested_guard,
        "human_review_required": human_review,
        "reason_codes": reason_codes,
    }


def _select_route_guard_candidate(
    region: Mapping[str, Any],
    linked_candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not linked_candidates:
        return None
    preserve_region = _route_guard_current_role(region) == "preserve_decorative_sfx"

    def priority(candidate: Mapping[str, Any]) -> tuple[int, int, int, str]:
        classification = str(candidate.get("fusion_classification") or "")
        confidence = str(candidate.get("confidence") or "")
        if preserve_region and _candidate_has_preserve_conflict(candidate):
            primary = 0
        elif _candidate_has_kitsumed_and_ogkalu_bubble(candidate):
            primary = 1
        elif _candidate_has_kitsumed_mask(candidate):
            primary = 2
        elif classification == "safe_future_missed_text_hint":
            primary = 3
        elif _candidate_has_ogkalu_bubble_or_text_bubble(candidate):
            primary = 4
        elif _candidate_has_text_free(candidate):
            primary = 5
        elif _candidate_has_preserve_conflict(candidate):
            primary = 6
        else:
            primary = 7
        confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(confidence, 3)
        review_rank = 1 if candidate.get("review_only") else 0
        return (primary, confidence_rank, review_rank, str(candidate.get("candidate_id") or ""))

    return sorted(linked_candidates, key=priority)[0]


def _route_guard_confidence_tier(region: Mapping[str, Any], candidate: Mapping[str, Any] | None) -> str:
    if candidate is None:
        return "unsupported_by_model_but_deterministic_valid"
    if _route_guard_current_role(region) == "preserve_decorative_sfx" and (
        _candidate_has_preserve_conflict(candidate)
        or _candidate_has_kitsumed_mask(candidate)
        or _candidate_has_ogkalu_bubble_or_text_bubble(candidate)
        or _candidate_has_text_free(candidate)
    ):
        return "conflict_preserve_wins"
    if _candidate_has_kitsumed_and_ogkalu_bubble(candidate):
        return "strong_model_container"
    if _candidate_has_kitsumed_mask(candidate):
        return "mask_primary_container"
    if _candidate_has_ogkalu_bubble_or_text_bubble(candidate):
        return "text_bubble_review_container"
    if _candidate_has_text_free(candidate):
        return "text_free_review_only"
    return "unsupported_by_model_but_deterministic_valid"


def _route_guard_result_for_region(
    *,
    current_role: str,
    tier: str,
    candidate: Mapping[str, Any] | None,
) -> tuple[str, str, bool, List[str]]:
    if current_role == "preserve_decorative_sfx":
        if tier == "conflict_preserve_wins":
            return (
                "conflicts_preserve_wins",
                "keep_current_preserve_decorative_sfx",
                True,
                ["current_preserve_route_wins_over_model_evidence"],
            )
        return (
            "unsupported_but_deterministic_valid",
            "keep_current_preserve_decorative_sfx",
            True,
            ["current_preserve_route_valid_without_model_support"],
        )

    if current_role == "caption_or_background":
        if tier in {"strong_model_container", "mask_primary_container", "text_bubble_review_container"}:
            return (
                "failed_closed",
                "keep_current_caption_background_review_model_conflict",
                True,
                ["speech_container_model_evidence_conflicts_with_current_caption_background_route"],
            )
        if tier == "text_free_review_only":
            return (
                "review_only",
                "keep_current_caption_background",
                True,
                ["text_free_is_not_caption_or_speech_authority"],
            )
        return (
            "unsupported_but_deterministic_valid",
            "keep_current_caption_background",
            True,
            ["current_caption_background_route_valid_without_model_support"],
        )

    if current_role == "speech":
        if tier in {"strong_model_container", "mask_primary_container"}:
            return (
                "supports_current_route",
                "keep_current_speech_route",
                MODEL_FUSION_CONFIDENCE_POLICY_TIERS[tier]["human_review_required"],
                ["model_container_evidence_supports_current_speech_route"],
            )
        if tier == "text_bubble_review_container":
            return (
                "review_only",
                "keep_current_speech_route_with_review_supported_model_hint",
                True,
                ["ogkalu_only_text_bubble_is_review_support_not_route_authority"],
            )
        if tier == "text_free_review_only":
            return (
                "review_only",
                "keep_current_speech_route_review_text_free_noise",
                True,
                ["text_free_does_not_override_current_speech_route"],
            )
        return (
            "unsupported_but_deterministic_valid",
            "keep_current_speech_route",
            True,
            ["current_speech_route_valid_without_model_support"],
        )

    return (
        "not_applicable",
        "no_route_guard_action",
        True,
        ["current_route_not_in_route_guard_reference_policy"],
    )


def _route_guard_current_role(region: Mapping[str, Any]) -> str:
    if _route_guard_is_preserve_decorative_sfx(region):
        return "preserve_decorative_sfx"
    if _route_guard_is_caption_or_background(region):
        return "caption_or_background"
    if str(region.get("semantic_class") or "") == "speech_bubble":
        return "speech"
    return "unknown"


def _route_guard_is_preserve_decorative_sfx(region: Mapping[str, Any]) -> bool:
    semantic = str(region.get("semantic_class") or "")
    reason = str(region.get("classification_reason") or "").lower()
    cleanup = str(region.get("cleanup_mode") or "").lower()
    skip_reason = str(region.get("skip_reason") or "").lower()
    return bool(
        semantic in {"decorative_text", "sfx"}
        or region.get("is_decorative")
        or region.get("is_sfx")
        or cleanup == "preserve"
        or "sfx" in reason
        or "decorative" in reason
        or "art_text" in reason
        or "preserve" in skip_reason
    )


def _route_guard_is_caption_or_background(region: Mapping[str, Any]) -> bool:
    semantic = str(region.get("semantic_class") or "")
    reason = str(region.get("classification_reason") or "").lower()
    cleanup = str(region.get("cleanup_mode") or "").lower()
    return bool(
        semantic in {"background_text", "sign_text"}
        or "caption" in reason
        or "background" in reason
        or "caption" in cleanup
    )


def _candidate_has_preserve_conflict(candidate: Mapping[str, Any]) -> bool:
    return (
        candidate.get("fusion_classification") == "review_only_sfx_decorative_conflict"
        or bool(candidate.get("conflict_flags"))
        or "current_pipeline_preserve_role_takes_precedence" in _as_list(candidate.get("reason_codes"))
    )


def _candidate_has_kitsumed_mask(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("linked_kitsumed_mask_ids"))


def _candidate_has_kitsumed_and_ogkalu_bubble(candidate: Mapping[str, Any]) -> bool:
    reasons = _as_list(candidate.get("reason_codes"))
    return bool(candidate.get("linked_kitsumed_mask_ids")) and (
        "ogkalu_text_bubble_strengthens_ownership" in reasons
        or "ogkalu_bubble_support" in reasons
    )


def _candidate_has_ogkalu_bubble_or_text_bubble(candidate: Mapping[str, Any]) -> bool:
    reasons = _as_list(candidate.get("reason_codes"))
    return any(
        reason in reasons
        for reason in (
            "ogkalu_text_bubble_strengthens_ownership",
            "ogkalu_text_bubble_without_kitsumed_mask",
            "ogkalu_bubble_support",
            "ogkalu_bubble_without_kitsumed_mask",
        )
    )


def _candidate_has_text_free(candidate: Mapping[str, Any]) -> bool:
    reasons = _as_list(candidate.get("reason_codes"))
    return any(
        reason in reasons
        for reason in (
            "ogkalu_text_free_without_kitsumed_mask",
            "ogkalu_text_free_inside_mask_conflict_or_annotation_noise",
        )
    )


def _route_guard_region_payload(proof: Mapping[str, Any], region: Mapping[str, Any]) -> Dict[str, Any]:
    candidate = proof.get("candidate") or {}
    return {
        "model_fusion_route_guard_candidate_id": candidate.get("candidate_id"),
        "model_fusion_route_guard_confidence_tier": proof.get("confidence_tier"),
        "model_fusion_route_guard_result": proof.get("result"),
        "model_fusion_route_guard_current_semantic_class": region.get("semantic_class"),
        "model_fusion_route_guard_current_cleanup_mode": region.get("cleanup_mode"),
        "model_fusion_route_guard_suggested_route_guard": proof.get("suggested_guard"),
        "model_fusion_route_guard_reason_codes": proof.get("reason_codes", []),
        "model_fusion_route_guard_kitsumed_mask_ids": _as_list(candidate.get("linked_kitsumed_mask_ids")),
        "model_fusion_route_guard_ogkalu_detection_ids": _as_list(candidate.get("linked_ogkalu_detection_ids")),
        "model_fusion_route_guard_human_review_required": bool(proof.get("human_review_required")),
        "would_change_behavior": False,
    }


def _attach_render_constraint_calibration_fields(
    audit: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    fused_containers: Sequence[Mapping[str, Any]],
) -> None:
    audit["model_fusion_render_constraint_calibration_version"] = MODEL_FUSION_RENDER_CONSTRAINT_CALIBRATION_VERSION
    audit["model_fusion_render_constraint_calibration_enabled"] = model_fusion_render_constraint_calibration_enabled()
    audit["model_fusion_render_constraint_calibration_generated"] = False
    audit["model_fusion_render_constraint_calibration_errors"] = []
    audit["model_fusion_render_constraint_calibration_candidates"] = []

    if not audit["model_fusion_render_constraint_calibration_enabled"]:
        return

    regions = audit.get("regions", []) or []
    region_by_id: Dict[str, MutableMapping[str, Any]] = {
        str(region.get("region_id")): region
        for region in regions
        if isinstance(region, MutableMapping) and region.get("region_id") is not None
    }
    fused_by_id = {
        str(container.get("fused_container_id")): container
        for container in fused_containers
        if container.get("fused_container_id") is not None
    }
    candidates_by_region = _candidates_by_region(candidates)
    calibration_records: List[Dict[str, Any]] = []
    errors: List[str] = []

    for region_id, region in sorted(region_by_id.items()):
        if str(region.get("semantic_class") or "") != "speech_bubble":
            continue
        linked_candidates = candidates_by_region.get(region_id, [])
        candidate = _select_render_constraint_candidate(region, linked_candidates)
        if candidate is None:
            if _meaningful_japanese(region.get("ocr_text")) and _region_has_render_output(region):
                record = _render_constraint_record(
                    page_id=str(audit.get("page_id") or ""),
                    region=region,
                    candidate=None,
                    fused_container=None,
                    status="unsupported_by_model_but_current_valid",
                    extra_reason_codes=["no_model_fusion_render_container_candidate"],
                )
            else:
                continue
        else:
            fused_container = fused_by_id.get(str(candidate.get("fused_container_id") or ""))
            status, extra_reasons = _render_constraint_status(region, candidate, fused_container)
            record = _render_constraint_record(
                page_id=str(audit.get("page_id") or ""),
                region=region,
                candidate=candidate,
                fused_container=fused_container,
                status=status,
                extra_reason_codes=extra_reasons,
            )
        region.update(_render_constraint_region_payload(record))
        calibration_records.append(record)

    audit["model_fusion_render_constraint_calibration_candidates"] = calibration_records
    audit["model_fusion_render_constraint_calibration_errors"] = sorted(set(errors))
    audit["model_fusion_render_constraint_calibration_generated"] = True


def _select_render_constraint_candidate(
    region: Mapping[str, Any],
    linked_candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    usable = [
        candidate
        for candidate in linked_candidates
        if _candidate_has_kitsumed_mask(candidate)
        or candidate.get("fusion_classification") == "safe_future_render_constraint_hint"
    ]
    if not usable:
        return None

    def priority(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
        classification = str(candidate.get("fusion_classification") or "")
        confidence = str(candidate.get("confidence") or "")
        if classification == "safe_future_render_constraint_hint":
            primary = 0
        elif _candidate_has_kitsumed_and_ogkalu_bubble(candidate):
            primary = 1
        elif _candidate_has_kitsumed_mask(candidate):
            primary = 2
        else:
            primary = 3
        confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(confidence, 3)
        return (primary, confidence_rank, str(candidate.get("candidate_id") or ""))

    return sorted(usable, key=priority)[0]


def _render_constraint_status(
    region: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fused_container: Mapping[str, Any] | None,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    page_id = str(candidate.get("page_id") or "")
    region_id = str(region.get("region_id") or "")
    tier = _render_constraint_confidence_tier(candidate)
    source_bbox = _region_source_bbox_xyxy(region)
    render_bbox = _bbox_list(region.get("final_render_bbox"))
    container_bbox = _bbox_list((fused_container or {}).get("bbox"))
    source_inside = _bbox_inside_ratio(source_bbox, container_bbox)
    render_outside = _bbox_outside_ratio(render_bbox, container_bbox)
    fit = _simulate_render_fit(region, container_bbox)

    if _route_guard_is_preserve_decorative_sfx(region):
        return "blocked_by_sfx_or_caption_conflict", ["current_region_is_preserve_or_decorative"]
    if _route_guard_is_caption_or_background(region):
        return "blocked_by_sfx_or_caption_conflict", ["current_region_is_caption_or_background"]
    if candidate.get("conflict_flags"):
        return "blocked_by_sfx_or_caption_conflict", ["model_fusion_conflict_present"]
    if not _meaningful_japanese(region.get("ocr_text")):
        return "review_only", ["source_text_not_meaningful_japanese_speech"]
    if not container_bbox:
        return "unsupported_by_model_but_current_valid", ["missing_model_container_bbox"]
    if _bbox_area(container_bbox) < 2500:
        return "noisy_false_positive", ["model_container_too_small"]
    if source_inside < 0.45:
        return "noisy_false_positive", [f"source_inside_ratio_low:{source_inside:.3f}"]
    if fit.get("translated_text_completeness") != "complete":
        return "blocked_by_fit_risk", ["fit_simulation_text_completeness_failed"]
    if float(fit.get("estimated_fit_ratio") or 0.0) > 0.98:
        return "blocked_by_fit_risk", [f"fit_ratio_high:{float(fit.get('estimated_fit_ratio') or 0.0):.3f}"]
    if tier == "strong_model_container" and render_outside >= 0.30 and source_inside >= 0.70:
        return "safe_future_candidate", ["strong_model_container_with_render_outside_container"]
    if tier == "mask_primary_container" and render_outside >= 0.45 and source_inside >= 0.70:
        return "watch_only", ["mask_primary_container_with_render_outside_container_review_needed"]
    if render_outside >= 0.20:
        return "watch_only", [f"render_outside_ratio_watch:{render_outside:.3f}"]
    return "unsupported_by_model_but_current_valid", ["current_render_inside_or_model_constraint_not_needed"]


def _render_constraint_record(
    *,
    page_id: str,
    region: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    fused_container: Mapping[str, Any] | None,
    status: str,
    extra_reason_codes: Sequence[str],
) -> Dict[str, Any]:
    source_bbox = _region_source_bbox_xyxy(region)
    render_bbox = _bbox_list(region.get("final_render_bbox"))
    container_bbox = _bbox_list((fused_container or {}).get("bbox"))
    allowed_area = _proposed_allowed_area(container_bbox)
    allowed_size = _allowed_area_size(allowed_area)
    tier = _render_constraint_confidence_tier(candidate)
    fit = _simulate_render_fit(region, allowed_area)
    reason_codes = sorted(
        set(
            [
                f"region:{region.get('region_id')}",
                f"status:{status}",
                f"confidence_tier:{tier}",
                "metadata_only_no_behavior_change",
            ]
            + list(extra_reason_codes)
            + [f"candidate_reason:{reason}" for reason in _as_list((candidate or {}).get("reason_codes"))]
            + [f"candidate_conflict:{flag}" for flag in _as_list((candidate or {}).get("conflict_flags"))]
        )
    )
    return {
        "candidate_id": f"{page_id}:{region.get('region_id')}:render_constraint_calibration",
        "page_id": page_id,
        "region_id": region.get("region_id"),
        "model_fusion_candidate_id": (candidate or {}).get("candidate_id"),
        "confidence_tier": tier,
        "status": status,
        "reason_codes": reason_codes,
        "kitsumed_mask_id": _first_or_none(_as_list((candidate or {}).get("linked_kitsumed_mask_ids"))),
        "ogkalu_detection_ids": _as_list((candidate or {}).get("linked_ogkalu_detection_ids")),
        "source_bbox": source_bbox,
        "current_render_bbox": render_bbox,
        "model_container_bbox": container_bbox,
        "model_container_area": round(_bbox_area(container_bbox), 2),
        "source_inside_ratio": round(_bbox_inside_ratio(source_bbox, container_bbox), 6),
        "render_outside_ratio": round(_bbox_outside_ratio(render_bbox, container_bbox), 6),
        "proposed_allowed_area": allowed_area,
        "allowed_area_size": allowed_size,
        "obstacle_conflict": bool((candidate or {}).get("conflict_flags")),
        "caption_background_conflict": _route_guard_is_caption_or_background(region),
        "meaningful_speech": _meaningful_japanese(region.get("ocr_text")),
        "visual_acceptability": _render_constraint_visual_acceptability(status),
        "fit_simulation": fit,
        "human_review_required": status != "proven_existing_allowlist",
        "would_change_behavior": False,
    }


def _render_constraint_region_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_fusion_render_constraint_candidate_id": record.get("candidate_id"),
        "model_fusion_render_constraint_confidence_tier": record.get("confidence_tier"),
        "model_fusion_render_constraint_status": record.get("status"),
        "model_fusion_render_constraint_reason_codes": record.get("reason_codes", []),
        "model_fusion_render_constraint_kitsumed_mask_id": record.get("kitsumed_mask_id"),
        "model_fusion_render_constraint_ogkalu_detection_ids": record.get("ogkalu_detection_ids", []),
        "model_fusion_render_constraint_source_inside_ratio": record.get("source_inside_ratio"),
        "model_fusion_render_constraint_render_outside_ratio": record.get("render_outside_ratio"),
        "model_fusion_render_constraint_proposed_allowed_area": record.get("proposed_allowed_area"),
        "model_fusion_render_constraint_allowed_area_size": record.get("allowed_area_size"),
        "model_fusion_render_constraint_obstacle_conflict": record.get("obstacle_conflict"),
        "model_fusion_render_constraint_fit_simulation": record.get("fit_simulation"),
        "model_fusion_render_constraint_human_review_required": record.get("human_review_required"),
        "would_change_behavior": False,
    }


def _attach_cleanup_boundary_calibration_fields(
    audit: MutableMapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    fused_containers: Sequence[Mapping[str, Any]],
    kitsumed_evidence: Sequence[Mapping[str, Any]],
    ogkalu_evidence: Sequence[Mapping[str, Any]],
) -> None:
    audit["model_fusion_cleanup_boundary_calibration_version"] = MODEL_FUSION_CLEANUP_BOUNDARY_CALIBRATION_VERSION
    audit["model_fusion_cleanup_boundary_calibration_enabled"] = model_fusion_cleanup_boundary_calibration_enabled()
    audit["model_fusion_cleanup_boundary_calibration_generated"] = False
    audit["model_fusion_cleanup_boundary_calibration_errors"] = []
    audit["model_fusion_cleanup_boundary_calibration_candidates"] = []

    if not audit["model_fusion_cleanup_boundary_calibration_enabled"]:
        return

    page_id = str(audit.get("page_id") or "")
    regions = audit.get("regions", []) or []
    region_by_id: Dict[str, MutableMapping[str, Any]] = {
        str(region.get("region_id")): region
        for region in regions
        if isinstance(region, MutableMapping) and region.get("region_id") is not None
    }
    candidates_by_region = _candidates_by_region(candidates)
    fused_by_id = {
        str(container.get("fused_container_id")): container
        for container in fused_containers
        if container.get("fused_container_id") is not None
    }
    kitsumed_by_id = {
        str(item.get("evidence_id")): item
        for item in kitsumed_evidence
        if item.get("evidence_id") is not None
    }
    ogkalu_by_id = {
        str(item.get("evidence_id")): item
        for item in ogkalu_evidence
        if item.get("evidence_id") is not None
    }
    preserved_regions = [
        region
        for region in region_by_id.values()
        if _route_guard_is_preserve_decorative_sfx(region)
    ]
    caption_regions = [
        region
        for region in region_by_id.values()
        if _route_guard_is_caption_or_background(region)
    ]

    calibration_records: List[Dict[str, Any]] = []
    errors: List[str] = []
    for region_id, region in sorted(region_by_id.items()):
        if not _cleanup_boundary_region_is_relevant(page_id, region):
            continue
        try:
            linked_candidates = candidates_by_region.get(region_id, [])
            candidate = _select_route_guard_candidate(region, linked_candidates)
            fused_container = fused_by_id.get(str((candidate or {}).get("fused_container_id") or ""))
            record = _cleanup_boundary_record(
                page_id=page_id,
                region=region,
                candidate=candidate,
                fused_container=fused_container,
                kitsumed_by_id=kitsumed_by_id,
                ogkalu_by_id=ogkalu_by_id,
                preserved_regions=preserved_regions,
                caption_regions=caption_regions,
            )
            region.update(_cleanup_boundary_region_payload(record))
            calibration_records.append(record)
        except Exception as exc:
            errors.append(f"{page_id}:{region_id}:{type(exc).__name__}:{exc}")

    audit["model_fusion_cleanup_boundary_calibration_candidates"] = calibration_records
    audit["model_fusion_cleanup_boundary_calibration_errors"] = sorted(set(errors))
    audit["model_fusion_cleanup_boundary_calibration_generated"] = True


def _cleanup_boundary_region_is_relevant(page_id: str, region: Mapping[str, Any]) -> bool:
    return bool(
        _region_has_cleanup_mask(region)
        or _region_cleanup_applied(region)
        or _route_guard_is_preserve_decorative_sfx(region)
        or _route_guard_is_caption_or_background(region)
        or _cleanup_boundary_group_id(page_id, region)
    )


def _cleanup_boundary_record(
    *,
    page_id: str,
    region: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    fused_container: Mapping[str, Any] | None,
    kitsumed_by_id: Mapping[str, Mapping[str, Any]],
    ogkalu_by_id: Mapping[str, Mapping[str, Any]],
    preserved_regions: Sequence[Mapping[str, Any]],
    caption_regions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    cleanup_bbox = _region_cleanup_bbox_xyxy(region)
    source_bbox = _region_source_bbox_xyxy(region)
    tier = _route_guard_confidence_tier(region, candidate)
    kitsumed_ids = [str(value) for value in _as_list((candidate or {}).get("linked_kitsumed_mask_ids")) if value is not None]
    ogkalu_ids = [str(value) for value in _as_list((candidate or {}).get("linked_ogkalu_detection_ids")) if value is not None]
    kitsumed_overlaps = _cleanup_kitsumed_overlaps(cleanup_bbox, kitsumed_ids, kitsumed_by_id)
    ogkalu_overlaps = _cleanup_ogkalu_overlaps(cleanup_bbox or source_bbox, ogkalu_ids, ogkalu_by_id)
    cleanup_inside = max((item.get("cleanup_inside_ratio", 0.0) for item in kitsumed_overlaps), default=0.0)
    cleanup_outside = 1.0 - cleanup_inside if cleanup_bbox and kitsumed_ids else 0.0
    preserved_overlap = _cleanup_overlap_with_regions(cleanup_bbox, preserved_regions, exclude_region_id=str(region.get("region_id") or ""))
    caption_overlap = _cleanup_overlap_with_regions(cleanup_bbox, caption_regions, exclude_region_id=str(region.get("region_id") or ""))
    expansion = _cleanup_bbox_expansion_ratio(cleanup_bbox, source_bbox)
    group_id = _cleanup_boundary_group_id(page_id, region)
    status, status_reasons, human_review = _cleanup_boundary_status(
        region=region,
        candidate=candidate,
        tier=tier,
        cleanup_bbox=cleanup_bbox,
        cleanup_inside_ratio=cleanup_inside,
        cleanup_outside_ratio=cleanup_outside,
        preserved_overlap_ratio=preserved_overlap,
        caption_overlap_ratio=caption_overlap,
        group_id=group_id,
    )
    reason_codes = sorted(
        set(
            [
                f"region:{region.get('region_id')}",
                f"status:{status}",
                f"confidence_tier:{tier}",
                "metadata_only_no_behavior_change",
            ]
            + status_reasons
            + [f"candidate_reason:{reason}" for reason in _as_list((candidate or {}).get("reason_codes"))]
            + [f"candidate_conflict:{flag}" for flag in _as_list((candidate or {}).get("conflict_flags"))]
        )
    )
    return {
        "candidate_id": f"{page_id}:{region.get('region_id')}:cleanup_boundary_calibration",
        "page_id": page_id,
        "region_id": region.get("region_id"),
        "model_fusion_candidate_id": (candidate or {}).get("candidate_id"),
        "current_semantic_class": region.get("semantic_class"),
        "cleanup_mode": region.get("cleanup_mode"),
        "cleanup_applied": _region_cleanup_applied(region),
        "cleanup_mask_bbox": cleanup_bbox,
        "cleanup_mask_area": round(_bbox_area(cleanup_bbox), 2),
        "source_bbox": source_bbox,
        "linked_model_fusion_container_id": (candidate or {}).get("fused_container_id"),
        "linked_model_container_bbox": _bbox_list((fused_container or {}).get("bbox")),
        "confidence_tier": tier,
        "status": status,
        "reason_codes": reason_codes,
        "kitsumed_mask_ids": kitsumed_ids,
        "ogkalu_detection_ids": ogkalu_ids,
        "kitsumed_mask_overlaps": kitsumed_overlaps,
        "ogkalu_detection_overlaps": ogkalu_overlaps,
        "cleanup_inside_mask_ratio": round(cleanup_inside, 6),
        "cleanup_outside_mask_ratio": round(max(0.0, cleanup_outside), 6),
        "preserved_overlap_ratio": round(preserved_overlap, 6),
        "caption_overlap_ratio": round(caption_overlap, 6),
        "bbox_expansion_ratio": round(expansion, 6) if expansion is not None else None,
        "group_id": group_id,
        "visual_role": _route_guard_current_role(region),
        "human_review_required": human_review,
        "would_change_behavior": False,
    }


def _cleanup_boundary_status(
    *,
    region: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    tier: str,
    cleanup_bbox: Sequence[float],
    cleanup_inside_ratio: float,
    cleanup_outside_ratio: float,
    preserved_overlap_ratio: float,
    caption_overlap_ratio: float,
    group_id: str | None,
) -> tuple[str, List[str], bool]:
    role = _route_guard_current_role(region)
    if group_id:
        return "transferred_group_cleanup_valid", [f"cleanup_interpreted_with_group:{group_id}"], True
    if role == "preserve_decorative_sfx":
        if _region_cleanup_applied(region) and cleanup_bbox:
            return "possible_destructive_cleanup_risk", ["preserve_region_has_cleanup_mask"], True
        if candidate is not None and tier == "conflict_preserve_wins":
            return "review_only_model_conflict", ["current_preserve_route_wins_over_model_cleanup_evidence"], True
        return "preserve_no_cleanup_valid", ["current_preserve_route_has_no_cleanup"], False
    if role == "caption_or_background":
        if preserved_overlap_ratio >= 0.15:
            return "possible_cleanup_leak", [f"caption_cleanup_overlaps_preserved_region:{preserved_overlap_ratio:.3f}"], True
        return "caption_cleanup_valid", ["current_caption_background_cleanup_policy_retained"], True
    if _candidate_has_preserve_conflict(candidate or {}):
        return "review_only_model_conflict", ["model_fusion_preserve_conflict_blocks_cleanup_assist"], True
    if preserved_overlap_ratio >= 0.10:
        return "possible_destructive_cleanup_risk", [f"cleanup_overlaps_preserved_region:{preserved_overlap_ratio:.3f}"], True
    if caption_overlap_ratio >= 0.20:
        return "possible_cleanup_leak", [f"cleanup_overlaps_caption_background_region:{caption_overlap_ratio:.3f}"], True
    if not cleanup_bbox:
        if candidate is None:
            return "blocked_by_missing_model_evidence", ["no_cleanup_mask_and_no_model_boundary_evidence"], True
        return "valid_cleanup_unsupported_by_model", ["no_cleanup_mask_for_current_region"], True
    if not _as_list((candidate or {}).get("linked_kitsumed_mask_ids")):
        return "valid_cleanup_unsupported_by_model", ["no_kitsumed_mask_boundary_for_cleanup_comparison"], True
    if cleanup_inside_ratio >= 0.60 and cleanup_outside_ratio <= 0.40:
        return "valid_cleanup_inside_container", [f"cleanup_inside_speech_mask:{cleanup_inside_ratio:.3f}"], False
    if cleanup_outside_ratio >= 0.45:
        return "possible_cleanup_leak", [f"cleanup_outside_speech_mask:{cleanup_outside_ratio:.3f}"], True
    return "valid_cleanup_unsupported_by_model", [f"cleanup_boundary_overlap_ambiguous:{cleanup_inside_ratio:.3f}"], True


def _cleanup_boundary_region_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model_fusion_cleanup_boundary_candidate_id": record.get("candidate_id"),
        "model_fusion_cleanup_boundary_status": record.get("status"),
        "model_fusion_cleanup_boundary_confidence_tier": record.get("confidence_tier"),
        "model_fusion_cleanup_boundary_reason_codes": record.get("reason_codes", []),
        "model_fusion_cleanup_boundary_kitsumed_mask_ids": record.get("kitsumed_mask_ids", []),
        "model_fusion_cleanup_boundary_ogkalu_detection_ids": record.get("ogkalu_detection_ids", []),
        "model_fusion_cleanup_boundary_cleanup_inside_mask_ratio": record.get("cleanup_inside_mask_ratio"),
        "model_fusion_cleanup_boundary_cleanup_outside_mask_ratio": record.get("cleanup_outside_mask_ratio"),
        "model_fusion_cleanup_boundary_preserved_overlap_ratio": record.get("preserved_overlap_ratio"),
        "model_fusion_cleanup_boundary_caption_overlap_ratio": record.get("caption_overlap_ratio"),
        "model_fusion_cleanup_boundary_bbox_expansion_ratio": record.get("bbox_expansion_ratio"),
        "model_fusion_cleanup_boundary_group_id": record.get("group_id"),
        "model_fusion_cleanup_boundary_human_review_required": record.get("human_review_required"),
        "would_change_behavior": False,
    }


def _region_cleanup_bbox_xyxy(region: Mapping[str, Any]) -> List[float]:
    bbox = _bbox_list(region.get("cleanup_mask_bbox"))
    if bbox:
        return bbox
    cleanup_mask = region.get("cleanup_mask")
    if isinstance(cleanup_mask, Mapping):
        return _bbox_list(cleanup_mask.get("bbox"))
    return []


def _region_has_cleanup_mask(region: Mapping[str, Any]) -> bool:
    return bool(_region_cleanup_bbox_xyxy(region))


def _region_cleanup_applied(region: Mapping[str, Any]) -> bool:
    return bool(region.get("cleanup_applied")) or bool(_region_cleanup_bbox_xyxy(region))


def _cleanup_boundary_group_id(page_id: str, region: Mapping[str, Any]) -> str | None:
    region_id = str(region.get("region_id") or "")
    cleanup_mode = str(region.get("cleanup_mode") or "")
    reason = str(region.get("classification_reason") or "")
    if cleanup_mode.startswith("transferred_to_"):
        return cleanup_mode
    if reason == BUBBLE_LOCAL_OWNERSHIP_REASON:
        return "bubble_local_nested_speech_fragment_ownership"
    if reason == ADJACENT_VERTICAL_TEXT_CONSERVATION_REASON:
        return "adjacent_vertical_speech_text_conservation_recovery"
    return None


def _cleanup_kitsumed_overlaps(
    cleanup_bbox: Sequence[float],
    kitsumed_ids: Sequence[str],
    kitsumed_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    overlaps: List[Dict[str, Any]] = []
    for mask_id in kitsumed_ids:
        item = kitsumed_by_id.get(str(mask_id)) or {}
        mask_bbox = _bbox_list(item.get("mask_bbox") or item.get("bbox"))
        overlaps.append(
            {
                "mask_id": mask_id,
                "mask_bbox": mask_bbox,
                "cleanup_inside_ratio": round(_bbox_inside_ratio(cleanup_bbox, mask_bbox), 6),
                "cleanup_overlap_area": round(_bbox_intersection_area(cleanup_bbox, mask_bbox), 2),
            }
        )
    return overlaps


def _cleanup_ogkalu_overlaps(
    comparison_bbox: Sequence[float],
    ogkalu_ids: Sequence[str],
    ogkalu_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    overlaps: List[Dict[str, Any]] = []
    for detection_id in ogkalu_ids:
        item = ogkalu_by_id.get(str(detection_id)) or {}
        bbox = _bbox_list(item.get("bbox"))
        overlaps.append(
            {
                "detection_id": detection_id,
                "class_name": item.get("class_name"),
                "bbox": bbox,
                "overlap_ratio": round(_bbox_inside_ratio(comparison_bbox, bbox), 6),
                "overlap_area": round(_bbox_intersection_area(comparison_bbox, bbox), 2),
            }
        )
    return overlaps


def _cleanup_overlap_with_regions(
    cleanup_bbox: Sequence[float],
    regions: Sequence[Mapping[str, Any]],
    *,
    exclude_region_id: str,
) -> float:
    cleanup_area = _bbox_area(cleanup_bbox)
    if cleanup_area <= 0.0:
        return 0.0
    overlap = 0.0
    for region in regions:
        if str(region.get("region_id") or "") == exclude_region_id:
            continue
        overlap += _bbox_intersection_area(cleanup_bbox, _region_source_bbox_xyxy(region))
    return min(1.0, overlap / cleanup_area)


def _cleanup_bbox_expansion_ratio(cleanup_bbox: Sequence[float], source_bbox: Sequence[float]) -> float | None:
    source_area = _bbox_area(source_bbox)
    if source_area <= 0.0 or not cleanup_bbox:
        return None
    return _bbox_area(cleanup_bbox) / source_area


def _render_constraint_confidence_tier(candidate: Mapping[str, Any] | None) -> str:
    if candidate is None:
        return "unsupported_by_model_but_deterministic_valid"
    if _candidate_has_preserve_conflict(candidate):
        return "conflict_preserve_wins"
    if _candidate_has_kitsumed_and_ogkalu_bubble(candidate):
        return "strong_model_container"
    if _candidate_has_kitsumed_mask(candidate):
        return "mask_primary_container"
    if _candidate_has_ogkalu_bubble_or_text_bubble(candidate):
        return "text_bubble_review_container"
    if _candidate_has_text_free(candidate):
        return "text_free_review_only"
    return "unsupported_by_model_but_deterministic_valid"


def _region_has_render_output(region: Mapping[str, Any]) -> bool:
    return bool(_bbox_list(region.get("final_render_bbox")) or region.get("wrapped_lines"))


def _region_source_bbox_xyxy(region: Mapping[str, Any]) -> List[float]:
    bbox = region.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
        return [x, y, x + max(0.0, w), y + max(0.0, h)]
    return []


def _bbox_list(value: Any) -> List[float]:
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return []
    return []


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_inside_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = _bbox_area(inner)
    if area <= 0.0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / area


def _bbox_outside_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    area = _bbox_area(inner)
    if area <= 0.0:
        return 0.0
    return max(0.0, 1.0 - (_bbox_intersection_area(inner, outer) / area))


def _proposed_allowed_area(container_bbox: Sequence[float]) -> List[float]:
    if len(container_bbox) != 4:
        return []
    x1, y1, x2, y2 = [float(v) for v in container_bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    pad = max(4.0, min(width, height) * 0.08)
    if width <= pad * 2 or height <= pad * 2:
        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    return [round(x1 + pad, 2), round(y1 + pad, 2), round(x2 - pad, 2), round(y2 - pad, 2)]


def _allowed_area_size(bbox: Sequence[float]) -> Dict[str, float]:
    if len(bbox) != 4:
        return {"width": 0.0, "height": 0.0, "aspect_ratio": 0.0, "area": 0.0}
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return {
        "width": round(width, 2),
        "height": round(height, 2),
        "aspect_ratio": round(width / height, 4) if height > 0.0 else 0.0,
        "area": round(width * height, 2),
    }


def _simulate_render_fit(region: Mapping[str, Any], allowed_area: Sequence[float]) -> Dict[str, Any]:
    translated = str(region.get("translated_text") or "")
    wrapped = [str(line or "") for line in (region.get("wrapped_lines") or [])]
    allowed = _allowed_area_size(allowed_area)
    non_punct_translated = _non_punctuation_chars(translated)
    non_punct_wrapped = _non_punctuation_chars("".join(wrapped))
    completeness = "complete" if all(ch in non_punct_wrapped for ch in non_punct_translated) else "missing_from_current_wrapped_lines"
    current_fit_ratio = _safe_float(region.get("fit_ratio"))
    text_len = max(1, len(non_punct_translated))
    area = max(1.0, float(allowed.get("area") or 0.0))
    density_ratio = min(1.5, (text_len * 430.0) / area)
    estimated_fit_ratio = current_fit_ratio if current_fit_ratio is not None else density_ratio
    simulated_font_size = _safe_float(region.get("selected_font_size") or region.get("font_size"))
    if simulated_font_size is None:
        min_side = min(float(allowed.get("width") or 0.0), float(allowed.get("height") or 0.0))
        simulated_font_size = max(10.0, min(32.0, min_side / max(4.0, len(wrapped) or 4)))
    failure_reason = None
    if completeness != "complete":
        failure_reason = "current_wrapped_lines_omit_translated_text"
    elif area < 2500:
        failure_reason = "allowed_area_too_small"
    elif estimated_fit_ratio > 0.98:
        failure_reason = "estimated_fit_ratio_too_high"
    return {
        "simulation_type": "metadata_only_estimate",
        "simulated_font_size": round(float(simulated_font_size), 2),
        "simulated_wrapped_lines": wrapped,
        "translated_text_completeness": completeness,
        "estimated_fit_ratio": round(float(estimated_fit_ratio), 6),
        "failure_reason": failure_reason,
        "would_change_behavior": False,
    }


def _non_punctuation_chars(text: str) -> List[str]:
    return [
        ch
        for ch in str(text or "")
        if ch.strip()
        and ch
        not in "。，、！？：；…‥・･ー-—―－〜～「」『』（）()[]【】<>〈〉《》“”‘’\"' .\n\r\t"
    ]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_constraint_visual_acceptability(status: str) -> str:
    if status in {"proven_existing_allowlist", "safe_future_candidate"}:
        return "failed_or_candidate_for_visual_review"
    if status in {"watch_only", "blocked_by_fit_risk"}:
        return "watch"
    if status.startswith("blocked"):
        return "blocked"
    return "current_valid_or_review_only"


def _first_or_none(values: Sequence[Any]) -> Any:
    return values[0] if values else None


def _write_page_artifacts(
    *,
    page_path: Path,
    page_id: str,
    service_result: BubbleDetectionResult,
    audit: Mapping[str, Any],
) -> None:
    page_path.mkdir(parents=True, exist_ok=True)

    service_overlay_path = page_path / "bubble_detection_overlay.jpg"
    overlay_path = page_path / "model_fusion_overlay.jpg"
    try:
        draw_bubble_detection_overlay(
            service_result,
            service_overlay_path,
            f"BubbleDetection service {page_id}",
        )
        audit["bubble_detection_overlay_path"] = str(service_overlay_path)
        try:
            shutil.copyfile(service_overlay_path, overlay_path)
        except Exception:
            draw_bubble_detection_overlay(
                service_result,
                overlay_path,
                f"Phase 4 model fusion {page_id}",
            )
        audit["model_fusion_overlay_path"] = str(overlay_path)
    except Exception as exc:
        audit["model_fusion_overlay_error"] = f"{type(exc).__name__}: {exc}"
        audit["bubble_detection_overlay_error"] = f"{type(exc).__name__}: {exc}"

    service_payload_path = page_path / "bubble_detection_result.json"
    try:
        service_payload_path.write_text(
            json.dumps(service_result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit["bubble_detection_result_path"] = str(service_payload_path)
    except Exception as exc:
        audit["bubble_detection_result_error"] = f"{type(exc).__name__}: {exc}"

    payload_path = page_path / "model_fusion_assist.json"
    payload = {
        "model_fusion_assist_version": audit.get("model_fusion_assist_version"),
        "model_fusion_assist_enabled": audit.get("model_fusion_assist_enabled"),
        "model_fusion_assist_generated": audit.get("model_fusion_assist_generated"),
        "model_fusion_assist_error": audit.get("model_fusion_assist_error"),
        "bubble_detection_version": audit.get("bubble_detection_version"),
        "bubble_detection_generated": audit.get("bubble_detection_generated"),
        "bubble_detection_error": audit.get("bubble_detection_error"),
        "bubble_detection_runtime_sec": audit.get("bubble_detection_runtime_sec"),
        "bubble_detection_fallback_used": audit.get("bubble_detection_fallback_used"),
        "bubble_detection_result": audit.get("bubble_detection_result", {}),
        "bubble_detection_cache_enabled": audit.get("bubble_detection_cache_enabled"),
        "bubble_detection_cache_key": audit.get("bubble_detection_cache_key"),
        "bubble_detection_cache_hit": audit.get("bubble_detection_cache_hit"),
        "bubble_detection_cache_read_path": audit.get("bubble_detection_cache_read_path"),
        "bubble_detection_cache_write_path": audit.get("bubble_detection_cache_write_path"),
        "bubble_detection_cache_error": audit.get("bubble_detection_cache_error"),
        "bubble_detection_cache_invalidation_reason": audit.get("bubble_detection_cache_invalidation_reason"),
        "bubble_detection_result_path": audit.get("bubble_detection_result_path"),
        "bubble_detection_overlay_path": audit.get("bubble_detection_overlay_path"),
        "bubble_detection_consumer_version": audit.get("bubble_detection_consumer_version"),
        "bubble_detection_consumer_generated": audit.get("bubble_detection_consumer_generated"),
        "bubble_detection_consumer_error": audit.get("bubble_detection_consumer_error"),
        "bubble_detection_consumer_sources": audit.get("bubble_detection_consumer_sources", []),
        "model_fusion_ownership_proof_enabled": audit.get("model_fusion_ownership_proof_enabled"),
        "model_fusion_ownership_proof_applied": audit.get("model_fusion_ownership_proof_applied"),
        "model_fusion_ownership_proof_version": audit.get("model_fusion_ownership_proof_version"),
        "model_fusion_ownership_proof_errors": audit.get("model_fusion_ownership_proof_errors"),
        "model_fusion_text_conservation_proof_enabled": audit.get("model_fusion_text_conservation_proof_enabled"),
        "model_fusion_text_conservation_proof_applied": audit.get("model_fusion_text_conservation_proof_applied"),
        "model_fusion_text_conservation_proof_version": audit.get("model_fusion_text_conservation_proof_version"),
        "model_fusion_text_conservation_proof_errors": audit.get("model_fusion_text_conservation_proof_errors"),
        "model_fusion_text_conservation_integration_enabled": audit.get("model_fusion_text_conservation_integration_enabled"),
        "model_fusion_text_conservation_integration_generated": audit.get("model_fusion_text_conservation_integration_generated"),
        "model_fusion_text_conservation_integration_version": audit.get("model_fusion_text_conservation_integration_version"),
        "model_fusion_text_conservation_integration_errors": audit.get("model_fusion_text_conservation_integration_errors"),
        "model_fusion_text_conservation_integration_candidates": audit.get("model_fusion_text_conservation_integration_candidates", []),
        "model_fusion_text_conservation_integration_blocked_references": audit.get("model_fusion_text_conservation_integration_blocked_references", []),
        "model_fusion_text_conservation_integration_summary": audit.get("model_fusion_text_conservation_integration_summary", {}),
        "model_fusion_text_conservation_integration_mutation_count": audit.get("model_fusion_text_conservation_integration_mutation_count"),
        "model_fusion_route_guard_proof_enabled": audit.get("model_fusion_route_guard_proof_enabled"),
        "model_fusion_route_guard_proof_applied": audit.get("model_fusion_route_guard_proof_applied"),
        "model_fusion_route_guard_proof_version": audit.get("model_fusion_route_guard_proof_version"),
        "model_fusion_route_guard_proof_errors": audit.get("model_fusion_route_guard_proof_errors"),
        "model_fusion_confidence_policy_version": audit.get("model_fusion_confidence_policy_version"),
        "model_fusion_confidence_policy_tiers": audit.get("model_fusion_confidence_policy_tiers"),
        "model_fusion_render_constraint_calibration_enabled": audit.get("model_fusion_render_constraint_calibration_enabled"),
        "model_fusion_render_constraint_calibration_generated": audit.get("model_fusion_render_constraint_calibration_generated"),
        "model_fusion_render_constraint_calibration_version": audit.get("model_fusion_render_constraint_calibration_version"),
        "model_fusion_render_constraint_calibration_errors": audit.get("model_fusion_render_constraint_calibration_errors"),
        "model_fusion_render_constraint_calibration_candidates": audit.get("model_fusion_render_constraint_calibration_candidates", []),
        "model_fusion_cleanup_boundary_calibration_enabled": audit.get("model_fusion_cleanup_boundary_calibration_enabled"),
        "model_fusion_cleanup_boundary_calibration_generated": audit.get("model_fusion_cleanup_boundary_calibration_generated"),
        "model_fusion_cleanup_boundary_calibration_version": audit.get("model_fusion_cleanup_boundary_calibration_version"),
        "model_fusion_cleanup_boundary_calibration_errors": audit.get("model_fusion_cleanup_boundary_calibration_errors"),
        "model_fusion_cleanup_boundary_calibration_candidates": audit.get("model_fusion_cleanup_boundary_calibration_candidates", []),
        "high_accuracy_bubble_mode_enabled": audit.get("high_accuracy_bubble_mode_enabled"),
        "high_accuracy_bubble_mode_version": audit.get("high_accuracy_bubble_mode_version"),
        "high_accuracy_bubble_mode_generated": audit.get("high_accuracy_bubble_mode_generated"),
        "high_accuracy_bubble_mode_error": audit.get("high_accuracy_bubble_mode_error"),
        "high_accuracy_bubble_mode_runtime_sec": audit.get("high_accuracy_bubble_mode_runtime_sec"),
        "high_accuracy_bubble_mode_mutation_allowed": audit.get("high_accuracy_bubble_mode_mutation_allowed"),
        "high_accuracy_bubble_mode_components": audit.get("high_accuracy_bubble_mode_components", {}),
        "high_accuracy_bubble_mode_candidate_counts": audit.get("high_accuracy_bubble_mode_candidate_counts", {}),
        "high_accuracy_bubble_mode_conflict_counts": audit.get("high_accuracy_bubble_mode_conflict_counts", {}),
        "high_accuracy_bubble_mode_fallback_used": audit.get("high_accuracy_bubble_mode_fallback_used"),
        "model_fusion_evidence": audit.get("model_fusion_evidence", {}),
        "model_fusion_assist_candidates": audit.get("model_fusion_assist_candidates", []),
        "model_fusion_conflicts": audit.get("model_fusion_conflicts", []),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _reason_codes_from_policy(policy: Any) -> List[str]:
    if not policy:
        return []
    policy_text = str(policy)
    return [f"calibration_policy:{policy_text}"]


def _contraindications(
    classification: str,
    confidence: str,
    conflict_flags: Sequence[Any],
    safe_high_confidence: bool,
) -> List[str]:
    contraindications: List[str] = []
    if confidence != "high":
        contraindications.append("not_high_confidence")
    if conflict_flags:
        contraindications.append("model_or_current_evidence_conflict")
    if classification not in SAFE_FUTURE_CLASSES:
        contraindications.append("review_only_classification")
    if classification == "safe_future_missed_text_hint":
        contraindications.append("missed_text_requires_human_review_or_recovery_harness")
    if safe_high_confidence:
        return contraindications
    return contraindications or ["not_eligible_for_assist"]
