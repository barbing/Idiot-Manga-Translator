# -*- coding: utf-8 -*-
"""Semantic route advisor and experimental opt-in route assist.

The advisor consumes existing text-area diagnostics and current audit metadata,
then emits dry-run route suggestions for review. By default it does not mutate
production region routing, cleanup, translation, rendering, or project output.
"""
from __future__ import annotations

import os
import time
from typing import Any


ROUTE_ADVISOR_VERSION = "text_area_route_advisor_phase2a_v1"
ROUTE_ASSIST_VERSION = "text_area_route_assist_phase2c_v1"
ROUTE_ASSIST_FLAG = "MT_TEXT_AREA_ROUTE_ASSIST"
ROUTE_CONSUMPTION_PROOF_VERSION = "phase4b18_route_consumption_proof_v1"
ROUTE_CONSUMPTION_PROOF_FLAG = "MT_MODEL_FUSION_ROUTE_CONSUMPTION_PROOF"
MODEL_FUSION_ASSIST_FLAG = "MT_MODEL_FUSION_ASSIST"
HIGH_ACCURACY_BUBBLE_MODE_FLAG = "MT_HIGH_ACCURACY_BUBBLE_MODE"
LEGACY_PAGE_SPECIFIC_ASSIST_FLAG = "MT_LEGACY_PAGE_SPECIFIC_ASSIST"
PHASE2_STATUS = "advisory_only"

SPEECH_REASONS = {
    "bubble_contained_short_laugh_speech",
    "speech_bubble_missed_text_recovery",
    "bubble_local_nested_speech_fragment_ownership",
    "adjacent_vertical_speech_text_conservation_recovery",
}

SHARED_SPEECH_OWNERSHIP_REASONS = {
    "bubble_local_nested_speech_fragment_ownership",
    "adjacent_vertical_speech_text_conservation_recovery",
}

CAPTION_REASONS = {
    "top_row_background_caption_candidate",
    "top_row_caption_fragment_candidate",
}

DECORATIVE_REASONS = {
    "nonbubble_short_kana_art_text_candidate",
    "nonbubble_short_reaction_art_text_candidate",
    "short_reaction_without_visual_speech_ownership",
    "nonbubble_short_reaction_art_sfx_candidate",
    "nonbubble_breath_sfx_art_text_candidate",
    "large_low_confidence_nonbubble_sfx_candidate",
    "large_short_decorative_sfx_candidate",
    "medium_large_katakana_sfx_candidate",
    "low_conf_dark_short_art_sfx_candidate",
}

ELIGIBLE_ROUTE_ASSIST_TYPES = {
    "probable_sfx_decorative_preserve",
    "probable_bubble_contained_short_speech",
    "probable_caption_not_speech",
}


def enrich_audit_with_route_advisor(audit: dict[str, Any]) -> dict[str, Any]:
    """Attach advisory route suggestions to an enriched debug audit.

    Failures are recorded in advisor status fields and never propagate.
    """
    start = time.time()
    enriched = dict(audit)
    enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
    enriched["route_advisor_version"] = ROUTE_ADVISOR_VERSION
    try:
        if not enriched.get("diagnostic_generated"):
            raise ValueError("text-area diagnostics unavailable")
        suggestions = build_route_suggestions(enriched)
        _attach_route_suggestions_to_regions(enriched, suggestions)
        enriched["route_suggestions"] = suggestions
        enriched["route_advisor_generated"] = True
        enriched["route_advisor_error"] = None
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched["route_suggestions"] = []
        enriched["route_advisor_generated"] = False
        enriched["route_advisor_error"] = str(exc)
        for region in enriched.get("regions", []) or []:
            region["diagnostic_route_suggestions"] = []
    _attach_route_consumption_proof(enriched, enriched.get("route_suggestions", []) or [])
    enriched["route_advisor_runtime_sec"] = round(time.time() - start, 6)
    return enriched


def route_assist_enabled() -> bool:
    return False


def route_consumption_proof_enabled() -> bool:
    """Return whether Phase 4b-18 dry-run proof metadata should be generated."""
    return _truthy_env(ROUTE_CONSUMPTION_PROOF_FLAG) and (
        _truthy_env(LEGACY_PAGE_SPECIFIC_ASSIST_FLAG)
        and (
            _truthy_env(MODEL_FUSION_ASSIST_FLAG)
            or _truthy_env(HIGH_ACCURACY_BUBBLE_MODE_FLAG)
        )
    )


def apply_route_assist_to_regions(
    *,
    page_id: str,
    source_path: str,
    output_path: str | None,
    page_class: str | None,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply eligible high-confidence route suggestions when explicitly enabled.

    This function is intentionally fail-closed. When the experimental flag is
    off it returns without inspecting diagnostics. When the flag is on, failures
    are reported in the returned status and no exception is propagated.
    """
    start = time.time()
    status: dict[str, Any] = {
        "route_assist_version": ROUTE_ASSIST_VERSION,
        "route_assist_enabled": route_assist_enabled(),
        "route_assist_generated": False,
        "route_assist_error": None,
        "route_assist_suggestions_considered": 0,
        "route_assist_eligible_count": 0,
        "route_assist_applied_count": 0,
        "route_assist_applied": [],
    }
    if not status["route_assist_enabled"]:
        status["route_assist_runtime_sec"] = round(time.time() - start, 6)
        return status
    try:
        from app.pipeline.text_area_diagnostics import enrich_audit_with_text_area_diagnostics

        audit = _build_route_assist_audit(page_id, source_path, output_path, page_class, regions)
        project_regions = {
            str(region.get("region_id") or ""): region
            for region in regions
            if str(region.get("region_id") or "")
        }
        enriched = enrich_audit_with_text_area_diagnostics(
            audit,
            audit_path=None,
            project_regions=project_regions,
        )
        if not enriched.get("diagnostic_generated"):
            raise ValueError(str(enriched.get("diagnostic_error") or "text-area diagnostics unavailable"))
        suggestions = build_route_suggestions(enriched)
        status["route_assist_suggestions_considered"] = len(suggestions)
        regions_by_id = {
            str(region.get("region_id") or ""): region
            for region in regions
            if str(region.get("region_id") or "")
        }
        eligible = [item for item in suggestions if _eligible_route_assist_suggestion(item)]
        status["route_assist_eligible_count"] = len(eligible)
        for suggestion in eligible:
            region = regions_by_id.get(str(suggestion.get("region_id") or ""))
            if not region:
                continue
            applied = _apply_route_assist_suggestion(region, suggestion)
            if applied:
                status["route_assist_applied"].append(applied)
        status["route_assist_applied_count"] = len(status["route_assist_applied"])
        status["route_assist_generated"] = True
    except Exception as exc:  # pragma: no cover - experimental path fails closed
        status["route_assist_generated"] = False
        status["route_assist_error"] = str(exc)
    status["route_assist_runtime_sec"] = round(time.time() - start, 6)
    return status


def build_route_suggestions(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Build diagnostic-only route suggestions from an enriched audit."""
    regions = [dict(region) for region in audit.get("regions", []) or []]
    regions_by_id = {str(region.get("region_id") or ""): region for region in regions}
    ownership_by_rid = {
        str(item.get("region_id") or ""): item
        for item in audit.get("text_ownership", []) or []
    }
    containers_by_id = {
        str(item.get("container_id") or ""): item
        for item in audit.get("text_containers", []) or []
    }
    evidence_by_rid = _visual_evidence_by_region(audit)
    blocks_by_rid = _blocks_by_region(audit)

    suggestions: list[dict[str, Any]] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        link = ownership_by_rid.get(rid) or {}
        container = containers_by_id.get(str(link.get("container_id") or "")) or {}
        evidence = evidence_by_rid.get(rid) or region.get("diagnostic_role_evidence") or {}
        block = blocks_by_rid.get(rid)

        suggestion = (
            _shared_speech_ownership_suggestion(region, link, container, block)
            or _sfx_decorative_preserve_suggestion(region, link, container, evidence)
            or _caption_not_speech_suggestion(region, link, container, evidence)
            or _bubble_short_speech_suggestion(region, link, container, evidence)
            or _uncertain_review_suggestion(region, link, container, evidence)
        )
        if suggestion:
            suggestion["suggestion_id"] = f"route_suggestion_{len(suggestions):03d}"
            suggestions.append(suggestion)
    return suggestions


def _build_route_assist_audit(
    page_id: str,
    source_path: str,
    output_path: str | None,
    page_class: str | None,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_regions = []
    for order_idx, region in enumerate(regions):
        flags = region.get("flags", {}) or {}
        confidence = region.get("confidence", {}) or {}
        render = region.get("render", {}) or {}
        semantic = str(region.get("type") or "unknown")
        audit_regions.append(
            {
                "page_id": page_id,
                "region_id": str(region.get("region_id") or ""),
                "bbox": region.get("bbox"),
                "polygon": region.get("polygon"),
                "detection_confidence": confidence.get("det"),
                "ocr_confidence": confidence.get("ocr"),
                "ocr_text": region.get("ocr_text", ""),
                "normalized_ocr_text": _normalize_for_assist_audit(region.get("ocr_text", "")),
                "reading_order_index": order_idx,
                "group_id": region.get("group_id"),
                "bubble_id": region.get("bubble_id"),
                "semantic_class": semantic,
                "is_speech_bubble": semantic == "speech_bubble",
                "is_background": semantic == "background_text" or bool(flags.get("bg_text")),
                "is_decorative": semantic == "decorative_text",
                "is_sfx": semantic in {"decorative_text", "sfx"} or bool(flags.get("sfx")),
                "classification_reason": render.get("classification_reason"),
                "cleanup_mode": render.get("cleanup_mode"),
                "translated_text": region.get("translation", ""),
                "skip_reason": "ignored_by_pipeline" if flags.get("ignore") else None,
                "source_orientation": render.get("source_orientation"),
                "source_size_hint": render.get("source_size_hint"),
                "source_size_min": render.get("source_size_min"),
                "source_size_max": render.get("source_size_max"),
                "flags": flags,
            }
        )
    return {
        "page_id": page_id,
        "page_class": page_class,
        "source_path": source_path,
        "output_path": output_path,
        "regions": audit_regions,
    }


def _normalize_for_assist_audit(text: Any) -> str:
    return "".join(str(text or "").split())


def _eligible_route_assist_suggestion(suggestion: dict[str, Any]) -> bool:
    if suggestion.get("confidence") != "high":
        return False
    if suggestion.get("suggestion_type") not in ELIGIBLE_ROUTE_ASSIST_TYPES:
        return False
    if suggestion.get("contraindications"):
        return False
    reason_codes = set(str(item) for item in (suggestion.get("reason_codes") or []))
    suggestion_type = suggestion.get("suggestion_type")
    if suggestion_type == "probable_sfx_decorative_preserve":
        return (
            suggestion.get("suggested_semantic_class") == "decorative_text"
            and suggestion.get("suggested_cleanup_mode") == "preserve"
            and "container_type:sfx_decorative" in reason_codes
            and "non_speech_decorative_container" in reason_codes
            and "preserve_policy_candidate" in reason_codes
        )
    if suggestion_type == "probable_bubble_contained_short_speech":
        return (
            suggestion.get("suggested_semantic_class") == "speech_bubble"
            and "container_type:speech_bubble" in reason_codes
            and "bubble_boundary_evidence" in reason_codes
            and "short_kana_laugh_or_reaction_text" in reason_codes
            and "high_ocr_confidence" in reason_codes
        )
    if suggestion_type == "probable_caption_not_speech":
        return (
            suggestion.get("suggested_semantic_class") == "background_text"
            and "container_type:caption" in reason_codes
            and "caption_band_evidence" in reason_codes
            and any(str(item).startswith("current_reason:") for item in reason_codes)
        )
    return False


def _apply_route_assist_suggestion(region: dict[str, Any], suggestion: dict[str, Any]) -> dict[str, Any] | None:
    previous_semantic = str(region.get("type") or "")
    flags = region.setdefault("flags", {})
    render = region.setdefault("render", {})
    previous_cleanup = render.get("cleanup_mode")
    previous_ignore = bool(flags.get("ignore"))
    previous_bg = bool(flags.get("bg_text"))

    target_semantic = str(suggestion.get("suggested_semantic_class") or previous_semantic)
    target_cleanup = _route_assist_cleanup_mode(suggestion, target_semantic, previous_cleanup)
    target_ignore = previous_ignore
    target_bg = previous_bg
    target_review = bool(flags.get("needs_review"))

    suggestion_type = str(suggestion.get("suggestion_type") or "")
    if suggestion_type == "probable_sfx_decorative_preserve":
        target_ignore = True
        target_bg = True
        target_review = False
    elif suggestion_type == "probable_bubble_contained_short_speech":
        target_ignore = False
        target_bg = False
    elif suggestion_type == "probable_caption_not_speech":
        target_ignore = False
        target_bg = True

    changed = (
        previous_semantic != target_semantic
        or previous_cleanup != target_cleanup
        or previous_ignore != target_ignore
        or previous_bg != target_bg
    )
    if not changed:
        return None

    region["type"] = target_semantic
    render["cleanup_mode"] = target_cleanup
    flags["ignore"] = target_ignore
    flags["bg_text"] = target_bg
    flags["needs_review"] = target_review
    if target_ignore:
        region["translation"] = ""
    if target_semantic == "speech_bubble":
        flags.pop("hard_fail", None)
        if str(render.get("source_orientation") or "").strip().lower() == "vertical":
            render["wrap_mode"] = "vertical"

    audit_payload = {
        "route_assist_applied": True,
        "route_assist_version": ROUTE_ASSIST_VERSION,
        "route_assist_suggestion_type": suggestion_type,
        "route_assist_reason_codes": suggestion.get("reason_codes", []),
        "route_assist_previous_semantic_class": previous_semantic,
        "route_assist_previous_cleanup_mode": previous_cleanup,
        "route_assist_previous_ignore": previous_ignore,
        "route_assist_previous_bg_text": previous_bg,
        "route_assist_new_semantic_class": target_semantic,
        "route_assist_new_cleanup_mode": target_cleanup,
        "route_assist_new_ignore": target_ignore,
        "route_assist_new_bg_text": target_bg,
        "route_assist_confidence": suggestion.get("confidence"),
        "route_assist_linked_container_id": suggestion.get("linked_container_id"),
        "route_assist_linked_ownership_id": suggestion.get("linked_ownership_id"),
    }
    render.update(audit_payload)
    return {
        "region_id": str(region.get("region_id") or ""),
        "suggestion_type": suggestion_type,
        "previous_semantic_class": previous_semantic,
        "previous_cleanup_mode": previous_cleanup,
        "previous_ignore": previous_ignore,
        "previous_bg_text": previous_bg,
        "new_semantic_class": target_semantic,
        "new_cleanup_mode": target_cleanup,
        "new_ignore": target_ignore,
        "new_bg_text": target_bg,
        "confidence": suggestion.get("confidence"),
        "reason_codes": suggestion.get("reason_codes", []),
        "linked_container_id": suggestion.get("linked_container_id"),
        "linked_ownership_id": suggestion.get("linked_ownership_id"),
    }


def _route_assist_cleanup_mode(
    suggestion: dict[str, Any],
    target_semantic: str,
    previous_cleanup: Any,
) -> str:
    suggested = suggestion.get("suggested_cleanup_mode")
    if suggested:
        return str(suggested)
    if target_semantic == "speech_bubble":
        return "bubble"
    if target_semantic == "background_text":
        return "local_text_mask"
    if target_semantic in {"decorative_text", "sfx"}:
        return "preserve"
    return str(previous_cleanup or "")


def summarize_route_suggestions(suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact counts for reports."""
    by_type: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    for item in suggestions:
        by_type[str(item.get("suggestion_type") or "unknown")] = by_type.get(str(item.get("suggestion_type") or "unknown"), 0) + 1
        by_confidence[str(item.get("confidence") or "unknown")] = by_confidence.get(str(item.get("confidence") or "unknown"), 0) + 1
    return {
        "total": len(suggestions),
        "by_type": by_type,
        "by_confidence": by_confidence,
    }


def _attach_route_consumption_proof(audit: dict[str, Any], suggestions: list[dict[str, Any]]) -> None:
    start = time.time()
    requested = _truthy_env(ROUTE_CONSUMPTION_PROOF_FLAG)
    source_enabled = _truthy_env(LEGACY_PAGE_SPECIFIC_ASSIST_FLAG) and (
        _truthy_env(MODEL_FUSION_ASSIST_FLAG) or _truthy_env(HIGH_ACCURACY_BUBBLE_MODE_FLAG)
    )
    enabled = requested and source_enabled
    audit["model_fusion_route_consumption_proof_version"] = ROUTE_CONSUMPTION_PROOF_VERSION
    audit["model_fusion_route_consumption_proof_enabled"] = enabled
    audit["model_fusion_route_consumption_proof_requested"] = requested
    audit["model_fusion_route_consumption_proof_generated"] = False
    audit["model_fusion_route_consumption_proof_error"] = None
    audit["route_consumption_candidates"] = []
    audit["route_consumption_candidate_summary"] = _route_consumption_summary([])
    audit["route_consumption_mutation_count"] = 0
    audit["route_consumption_mutations"] = []
    audit["route_consumption_proof_status"] = "disabled"
    for region in audit.get("regions", []) or []:
        region["diagnostic_route_consumption_candidates"] = []
        region["diagnostic_route_consumption_candidate_types"] = []
        region["diagnostic_route_consumption_status"] = "disabled"
        region["diagnostic_route_consumption_reason_codes"] = []
        region["diagnostic_route_consumption_would_change_behavior"] = False

    if not requested:
        audit["model_fusion_route_consumption_proof_runtime_sec"] = round(time.time() - start, 6)
        return
    if not source_enabled:
        audit["model_fusion_route_consumption_proof_error"] = (
            "MT_MODEL_FUSION_ROUTE_CONSUMPTION_PROOF requires "
            "MT_LEGACY_PAGE_SPECIFIC_ASSIST=1 plus MT_MODEL_FUSION_ASSIST=1 "
            "or MT_HIGH_ACCURACY_BUBBLE_MODE=1"
        )
        audit["route_consumption_proof_status"] = "failed_closed_missing_bubble_evidence_mode"
        audit["model_fusion_route_consumption_proof_runtime_sec"] = round(time.time() - start, 6)
        return
    if audit.get("route_advisor_error"):
        audit["model_fusion_route_consumption_proof_error"] = str(audit.get("route_advisor_error"))
        audit["route_consumption_proof_status"] = "failed_closed_route_advisor_unavailable"
        audit["model_fusion_route_consumption_proof_runtime_sec"] = round(time.time() - start, 6)
        return

    try:
        regions_by_id = {
            str(region.get("region_id") or ""): region
            for region in audit.get("regions", []) or []
            if str(region.get("region_id") or "")
        }
        candidates: list[dict[str, Any]] = []
        for suggestion in suggestions:
            evidence = suggestion.get("bubble_detection_evidence") or {}
            if not evidence.get("available"):
                continue
            region = regions_by_id.get(str(suggestion.get("region_id") or ""))
            candidate = _route_consumption_candidate(suggestion, region)
            if candidate:
                candidate["route_consumption_candidate_id"] = f"route_consumption_{len(candidates):03d}"
                candidates.append(candidate)
        _attach_route_consumption_candidates_to_regions(audit, candidates)
        audit["route_consumption_candidates"] = candidates
        audit["route_consumption_candidate_summary"] = _route_consumption_summary(candidates)
        audit["model_fusion_route_consumption_proof_generated"] = True
        audit["route_consumption_proof_status"] = "dry_run_only_no_real_mutation_justified"
    except Exception as exc:  # pragma: no cover - debug isolation
        audit["model_fusion_route_consumption_proof_generated"] = False
        audit["model_fusion_route_consumption_proof_error"] = str(exc)
        audit["route_consumption_proof_status"] = "failed_closed"
    audit["model_fusion_route_consumption_proof_runtime_sec"] = round(time.time() - start, 6)


def _route_consumption_candidate(suggestion: dict[str, Any], region: dict[str, Any] | None) -> dict[str, Any] | None:
    evidence = suggestion.get("bubble_detection_evidence") or {}
    if not evidence.get("available"):
        return None
    classification, status, future_allowed, reason_codes = _route_consumption_classification(
        suggestion,
        region,
        evidence,
    )
    if classification is None:
        return None
    return {
        "route_consumption_candidate_id": None,
        "region_id": str(suggestion.get("region_id") or ""),
        "suggestion_id": suggestion.get("suggestion_id"),
        "suggestion_type": suggestion.get("suggestion_type"),
        "route_consumption_class": classification,
        "current_semantic_class": suggestion.get("current_semantic_class"),
        "suggested_semantic_class": suggestion.get("suggested_semantic_class"),
        "current_cleanup_mode": suggestion.get("current_cleanup_mode"),
        "suggested_cleanup_mode": suggestion.get("suggested_cleanup_mode"),
        "diagnostic_bubble_container_id": evidence.get("container_id"),
        "diagnostic_bubble_confidence_tier": evidence.get("confidence_tier"),
        "diagnostic_bubble_supported_actions": evidence.get("supported_actions", []),
        "diagnostic_bubble_blocked_actions": evidence.get("blocked_actions", []),
        "diagnostic_bubble_conflict_flags": evidence.get("conflict_flags", []),
        "diagnostic_bubble_review_only": bool(evidence.get("review_only")),
        "evidence_source": evidence.get("source") or "bubble_detection_service",
        "reason_codes": sorted(set(reason_codes + list(suggestion.get("reason_codes", []) or []))),
        "would_change_behavior": False,
        "phase4b18_status": status,
        "future_assist_allowed": future_allowed,
        "human_review_required": True,
    }


def _route_consumption_classification(
    suggestion: dict[str, Any],
    region: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> tuple[str | None, str, bool, list[str]]:
    suggestion_type = str(suggestion.get("suggestion_type") or "")
    tier = str(evidence.get("confidence_tier") or "")
    supported = set(_string_list(evidence.get("supported_actions")))
    blocked = set(_string_list(evidence.get("blocked_actions")))
    conflicts = set(_string_list(evidence.get("conflict_flags")))
    review_only = bool(evidence.get("review_only"))
    current_semantic = str(suggestion.get("current_semantic_class") or "")
    suggested_semantic = str(suggestion.get("suggested_semantic_class") or "")
    current_cleanup = str(suggestion.get("current_cleanup_mode") or "")
    suggested_cleanup = str(suggestion.get("suggested_cleanup_mode") or "")
    reason = _classification_reason(region or {})
    reason_codes = [
        "phase4b18_route_consumption_dry_run",
        f"bubble_confidence_tier:{tier}",
    ]
    if suggestion_type == "probable_bubble_contained_short_speech":
        reason_codes.append("route_consumption_class:model_supported_bubble_contained_short_speech")
        safe = (
            current_semantic == "speech_bubble"
            and suggested_semantic == "speech_bubble"
            and tier in {"strong_model_container", "mask_primary_container"}
            and not review_only
            and not conflicts
            and "preserve_wins" not in blocked
            and "sfx_conflict" not in blocked
        )
        status = "dry_run_only" if safe else "dry_run_blocked_review_only"
        return "model_supported_bubble_contained_short_speech", status, False, reason_codes
    if suggestion_type == "probable_caption_not_speech":
        reason_codes.append("route_consumption_class:model_supported_caption_background_guard")
        caption_route_agrees = (
            current_semantic == "background_text"
            and suggested_semantic == "background_text"
            and (reason in CAPTION_REASONS or "caption" in current_cleanup or "caption" in suggested_cleanup)
        )
        if tier in {"strong_model_container", "mask_primary_container"}:
            reason_codes.append("model_speech_evidence_does_not_override_caption_route")
            return "model_supported_caption_background_guard", "dry_run_blocked_caption_speech_conflict", False, reason_codes
        if caption_route_agrees:
            reason_codes.append("deterministic_caption_route_agrees")
            return "model_supported_caption_background_guard", "dry_run_only", False, reason_codes
        return "model_supported_caption_background_guard", "dry_run_blocked_review_only", False, reason_codes
    if suggestion_type == "probable_sfx_decorative_preserve":
        reason_codes.append("route_consumption_class:conflict_preserve_wins_guard")
        preserve_route_agrees = (
            current_semantic in {"decorative_text", "sfx"}
            or suggested_semantic in {"decorative_text", "sfx"}
            or current_cleanup == "preserve"
            or suggested_cleanup == "preserve"
            or reason in DECORATIVE_REASONS
        )
        preserve_wins = (
            tier == "conflict_preserve_wins"
            or preserve_route_agrees
            or bool(conflicts)
            or bool(blocked)
        )
        if preserve_wins:
            reason_codes.append("deterministic_preserve_route_wins")
            return "conflict_preserve_wins_guard", "dry_run_only_preserve_authoritative", False, reason_codes
        return "conflict_preserve_wins_guard", "dry_run_blocked_review_only", False, reason_codes
    if suggestion_type in {"probable_shared_speech_ownership", "route_uncertain_review_only"}:
        reason_codes.append("route_consumption_class:blocked_review_only")
        return "blocked_review_only", "dry_run_blocked_class_not_eligible", False, reason_codes
    if "text_free" in tier or review_only or not supported:
        reason_codes.append("route_consumption_class:blocked_review_only")
        return "blocked_review_only", "dry_run_blocked_review_only", False, reason_codes
    return None, "not_applicable", False, reason_codes


def _attach_route_consumption_candidates_to_regions(audit: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_rid.setdefault(str(candidate.get("region_id") or ""), []).append(candidate)
    for region in audit.get("regions", []) or []:
        rid = str(region.get("region_id") or "")
        items = by_rid.get(rid, [])
        region["diagnostic_route_consumption_candidates"] = items
        region["diagnostic_route_consumption_candidate_types"] = sorted(
            set(str(item.get("route_consumption_class") or "unknown") for item in items)
        )
        region["diagnostic_route_consumption_status"] = "dry_run_only" if items else "not_applicable"
        region["diagnostic_route_consumption_reason_codes"] = sorted(
            set(str(reason) for item in items for reason in (item.get("reason_codes") or []))
        )
        region["diagnostic_route_consumption_would_change_behavior"] = False
        if items and region.get("diagnostic_bubble_confidence_tier") is not None:
            _append_bubble_consumer_source(region, "text_area_route_consumption_proof")


def _route_consumption_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for item in candidates:
        by_class[str(item.get("route_consumption_class") or "unknown")] = by_class.get(
            str(item.get("route_consumption_class") or "unknown"),
            0,
        ) + 1
        by_status[str(item.get("phase4b18_status") or "unknown")] = by_status.get(
            str(item.get("phase4b18_status") or "unknown"),
            0,
        ) + 1
        by_tier[str(item.get("diagnostic_bubble_confidence_tier") or "unknown")] = by_tier.get(
            str(item.get("diagnostic_bubble_confidence_tier") or "unknown"),
            0,
        ) + 1
    return {
        "total": len(candidates),
        "by_class": by_class,
        "by_status": by_status,
        "by_confidence_tier": by_tier,
        "mutation_count": 0,
        "proof_mode": "dry_run_only",
    }


def _attach_route_suggestions_to_regions(audit: dict[str, Any], suggestions: list[dict[str, Any]]) -> None:
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for item in suggestions:
        by_rid.setdefault(str(item.get("region_id") or ""), []).append(item)
    for region in audit.get("regions", []) or []:
        rid = str(region.get("region_id") or "")
        region["diagnostic_route_suggestions"] = by_rid.get(rid, [])
        if region.get("diagnostic_bubble_confidence_tier") is not None:
            _append_bubble_consumer_source(region, "text_area_route_advisor")


def _bubble_route_evidence(region: dict[str, Any]) -> dict[str, Any]:
    tier = region.get("diagnostic_bubble_confidence_tier")
    if tier is None:
        return {
            "available": False,
            "source": "deterministic_only",
            "would_change_behavior": False,
        }
    supported_actions = _string_list(region.get("diagnostic_bubble_supported_actions"))
    blocked_actions = _string_list(region.get("diagnostic_bubble_blocked_actions"))
    return {
        "available": True,
        "source": "bubble_detection_service",
        "container_id": region.get("diagnostic_bubble_container_id"),
        "container_type": region.get("diagnostic_bubble_container_type"),
        "membership_type": region.get("diagnostic_bubble_membership_type"),
        "membership_confidence": region.get("diagnostic_bubble_membership_confidence"),
        "confidence_tier": tier,
        "decision_status": "supported" if supported_actions and not region.get("diagnostic_bubble_review_only") else "review_only",
        "supported_actions": supported_actions,
        "container_suggested_actions": _string_list(region.get("diagnostic_bubble_container_suggested_actions")),
        "blocked_actions": blocked_actions,
        "conflict_flags": _string_list(region.get("diagnostic_bubble_conflict_flags")),
        "source_model_ids": _string_list(region.get("diagnostic_bubble_source_model_ids")),
        "review_only": bool(region.get("diagnostic_bubble_review_only")),
        "consumer_sources": _string_list(region.get("diagnostic_bubble_consumer_sources")),
        "would_change_behavior": False,
    }


def _bubble_route_reason_codes(region: dict[str, Any]) -> list[str]:
    tier = region.get("diagnostic_bubble_confidence_tier")
    if tier is None:
        return []
    reasons = [
        "bubble_detection_service_evidence_available",
        f"bubble_confidence_tier:{tier}",
    ]
    container_id = region.get("diagnostic_bubble_container_id")
    container_type = region.get("diagnostic_bubble_container_type")
    if container_id:
        reasons.append(f"bubble_container:{container_id}")
    if container_type:
        reasons.append(f"bubble_container_type:{container_type}")
    for action in _string_list(region.get("diagnostic_bubble_supported_actions")):
        reasons.append(f"bubble_supported_action:{action}")
    for action in _string_list(region.get("diagnostic_bubble_blocked_actions")):
        reasons.append(f"bubble_blocked_action:{action}")
    for flag in _string_list(region.get("diagnostic_bubble_conflict_flags")):
        reasons.append(f"bubble_conflict:{flag}")
    return reasons


def _append_bubble_consumer_source(region: dict[str, Any], source: str) -> None:
    try:
        from app.pipeline.text_area_diagnostics import append_bubble_detection_consumer_source

        append_bubble_detection_consumer_source(region, source)
    except Exception:
        existing = set(_string_list(region.get("diagnostic_bubble_consumer_sources")))
        existing.add(source)
        region["diagnostic_bubble_consumer_sources"] = sorted(existing)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _base_suggestion(
    *,
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    suggestion_type: str,
    suggested_semantic_class: str | None,
    suggested_cleanup_mode: str | None,
    confidence: str,
    reason_codes: list[str],
    required_evidence: list[str],
    contraindications: list[str] | None = None,
    human_review_required: bool | None = None,
) -> dict[str, Any]:
    bubble_evidence = _bubble_route_evidence(region)
    all_reason_codes = sorted(set(reason_codes + _bubble_route_reason_codes(region)))
    return {
        "region_id": str(region.get("region_id") or ""),
        "current_semantic_class": _semantic_class(region),
        "current_cleanup_mode": _cleanup_mode(region),
        "suggested_semantic_class": suggested_semantic_class,
        "suggested_cleanup_mode": suggested_cleanup_mode,
        "suggestion_type": suggestion_type,
        "confidence": confidence,
        "reason_codes": all_reason_codes,
        "required_evidence": required_evidence,
        "contraindications": contraindications or [],
        "would_change_behavior": False,
        "phase2_status": PHASE2_STATUS,
        "human_review_required": bool(human_review_required if human_review_required is not None else confidence != "high"),
        "linked_container_id": container.get("container_id"),
        "linked_ownership_id": _ownership_id(link),
        "bubble_detection_evidence": bubble_evidence,
        "bubble_detection_decision_status": bubble_evidence.get("decision_status"),
        "bubble_detection_confidence_tier": bubble_evidence.get("confidence_tier"),
        "bubble_detection_supported_actions": bubble_evidence.get("supported_actions", []),
        "bubble_detection_blocked_actions": bubble_evidence.get("blocked_actions", []),
        "bubble_detection_conflict_flags": bubble_evidence.get("conflict_flags", []),
        "bubble_detection_consumer_sources": bubble_evidence.get("consumer_sources", []),
        "current_route_matches_suggestion": (
            suggested_semantic_class is not None
            and _semantic_class(region) == suggested_semantic_class
            and (suggested_cleanup_mode is None or _cleanup_mode(region) == suggested_cleanup_mode)
        ),
    }


def _shared_speech_ownership_suggestion(
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    block: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not block or container.get("container_type") != "speech_bubble":
        return None
    region_ids = [str(rid) for rid in block.get("region_ids", []) or []]
    if len(region_ids) < 2 or str(region.get("region_id") or "") not in region_ids:
        return None
    conservation = block.get("text_conservation", {}) or {}
    transfer_evidence = conservation.get("transferred_evidence", []) or []
    reason = _classification_reason(region)
    transfer_region_ids = _transfer_region_ids(transfer_evidence)
    rid = str(region.get("region_id") or "")
    if reason not in SHARED_SPEECH_OWNERSHIP_REASONS and rid not in transfer_region_ids:
        return None
    reason_codes = _diagnostic_reason_codes(link, container)
    reason_codes.append("shared_speech_container")
    if reason:
        reason_codes.append(f"current_reason:{reason}")
    if transfer_evidence:
        reason_codes.append("text_transfer_evidence_present")
    return _base_suggestion(
        region=region,
        link=link,
        container=container,
        suggestion_type="probable_shared_speech_ownership",
        suggested_semantic_class="speech_bubble",
        suggested_cleanup_mode=None,
        confidence="high" if reason in SHARED_SPEECH_OWNERSHIP_REASONS or rid in transfer_region_ids else "medium",
        reason_codes=reason_codes,
        required_evidence=[
            "speech_bubble_container",
            "multiple_text_instances_in_container",
            "shared_ownership_or_transfer_evidence",
        ],
        contraindications=_route_contraindications(region, container),
        human_review_required=False,
    )


def _sfx_decorative_preserve_suggestion(
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    reason = _classification_reason(region)
    semantic = _semantic_class(region)
    cleanup = _cleanup_mode(region)
    if reason in SPEECH_REASONS or container.get("container_type") == "speech_bubble":
        return None
    sfx_score = float(evidence.get("sfx_decorative_score", 0.0) or 0.0)
    role_agrees = (
        container.get("container_type") == "sfx_decorative"
        and _ownership_strong(link)
        and (reason in DECORATIVE_REASONS or cleanup == "preserve" or semantic in {"decorative_text", "sfx"} or sfx_score >= 0.75)
    )
    if not role_agrees:
        return None
    confidence = "high" if reason in DECORATIVE_REASONS else "medium"
    reason_codes = _diagnostic_reason_codes(link, container)
    reason_codes.extend(["non_speech_decorative_container", "preserve_policy_candidate"])
    if reason:
        reason_codes.append(f"current_reason:{reason}")
    return _base_suggestion(
        region=region,
        link=link,
        container=container,
        suggestion_type="probable_sfx_decorative_preserve",
        suggested_semantic_class="decorative_text",
        suggested_cleanup_mode="preserve",
        confidence=confidence,
        reason_codes=reason_codes,
        required_evidence=[
            "sfx_decorative_container",
            "inside_or_overlaps_container",
            "decorative_preserve_or_sfx_role_evidence",
            "no_speech_container_ownership",
        ],
        contraindications=_route_contraindications(region, container),
        human_review_required=confidence != "high",
    )


def _caption_not_speech_suggestion(
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    reason = _classification_reason(region)
    if reason in SPEECH_REASONS or reason in DECORATIVE_REASONS:
        return None
    if container.get("container_type") not in {"caption", "sign/background_text_area"}:
        return None
    if not _ownership_strong(link):
        return None
    reason_codes = _diagnostic_reason_codes(link, container)
    caption_score = float(evidence.get("caption_band_score", 0.0) or 0.0)
    if reason in CAPTION_REASONS:
        reason_codes.append(f"current_reason:{reason}")
    if caption_score >= 0.5:
        reason_codes.append("caption_band_evidence")
    if reason not in CAPTION_REASONS and caption_score < 0.5 and _semantic_class(region) != "background_text":
        return None
    return _base_suggestion(
        region=region,
        link=link,
        container=container,
        suggestion_type="probable_caption_not_speech",
        suggested_semantic_class="background_text",
        suggested_cleanup_mode=None,
        confidence="high" if reason in CAPTION_REASONS else "medium",
        reason_codes=reason_codes,
        required_evidence=[
            "caption_or_background_container",
            "inside_or_overlaps_container",
            "caption_band_or_current_caption_reason",
        ],
        contraindications=_route_contraindications(region, container),
        human_review_required=reason not in CAPTION_REASONS,
    )


def _bubble_short_speech_suggestion(
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    reason = _classification_reason(region)
    if reason not in SPEECH_REASONS:
        return None
    if not _short_kana_laugh_or_reaction_text(region):
        return None
    if not _high_ocr_confidence(region):
        return None
    if container.get("container_type") != "speech_bubble" or not _ownership_strong(link):
        return None
    if reason in DECORATIVE_REASONS:
        return None
    reason_codes = _diagnostic_reason_codes(link, container)
    reason_codes.append(f"current_reason:{reason}")
    reason_codes.extend(["short_kana_laugh_or_reaction_text", "high_ocr_confidence"])
    if evidence.get("bubble_boundary_evidence"):
        reason_codes.append("bubble_boundary_evidence")
    return _base_suggestion(
        region=region,
        link=link,
        container=container,
        suggestion_type="probable_bubble_contained_short_speech",
        suggested_semantic_class="speech_bubble",
        suggested_cleanup_mode=None,
        confidence="high",
        reason_codes=reason_codes,
        required_evidence=[
            "speech_bubble_container",
            "inside_or_overlaps_container",
            "speech_ownership_or_recovery_reason",
            "no_decorative_preserve_reason",
        ],
        contraindications=_route_contraindications(region, container),
        human_review_required=False,
    )


def _uncertain_review_suggestion(
    region: dict[str, Any],
    link: dict[str, Any],
    container: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    semantic = _semantic_class(region)
    ctype = str(container.get("container_type") or "")
    if not ctype or ctype == "unknown":
        return None
    conflicts = _route_contraindications(region, container)
    evidence_reasons = evidence.get("reason_codes", []) or []
    has_conflict = bool(conflicts)
    if not has_conflict:
        if semantic == "speech_bubble" and ctype in {"caption", "sign/background_text_area", "sfx_decorative"}:
            has_conflict = True
        elif semantic in {"decorative_text", "sfx"} and ctype == "speech_bubble":
            has_conflict = True
        elif semantic == "background_text" and ctype == "speech_bubble":
            has_conflict = True
    if not has_conflict:
        return None
    reason_codes = _diagnostic_reason_codes(link, container)
    reason_codes.extend([f"evidence:{item}" for item in evidence_reasons])
    return _base_suggestion(
        region=region,
        link=link,
        container=container,
        suggestion_type="route_uncertain_review_only",
        suggested_semantic_class=None,
        suggested_cleanup_mode=None,
        confidence="low",
        reason_codes=reason_codes,
        required_evidence=[
            "conflicting_current_route_and_diagnostic_container_evidence",
        ],
        contraindications=conflicts,
        human_review_required=True,
    )


def _visual_evidence_by_region(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_rid: dict[str, dict[str, Any]] = {}
    for item in audit.get("visual_role_evidence", []) or []:
        rid = str(item.get("region_id") or "")
        if rid:
            by_rid[rid] = item.get("evidence", {}) or {}
    for region in audit.get("regions", []) or []:
        rid = str(region.get("region_id") or "")
        if rid and rid not in by_rid:
            by_rid[rid] = region.get("diagnostic_role_evidence", {}) or {}
    return by_rid


def _transfer_region_ids(transfer_evidence: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for item in transfer_evidence:
        source = str(item.get("region_id") or "").strip()
        target = str(item.get("transfer_to_region_id") or "").strip()
        if source:
            ids.add(source)
        if target:
            ids.add(target)
    return ids


def _blocks_by_region(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_rid: dict[str, dict[str, Any]] = {}
    for block in audit.get("logical_text_blocks", []) or []:
        for rid in block.get("region_ids", []) or []:
            by_rid[str(rid)] = block
    return by_rid


def _semantic_class(region: dict[str, Any]) -> str:
    return str(region.get("semantic_class") or region.get("type") or "unknown")


def _cleanup_mode(region: dict[str, Any]) -> str | None:
    value = region.get("cleanup_mode")
    if value is None:
        value = (region.get("render", {}) or {}).get("cleanup_mode")
    return str(value) if value is not None else None


def _classification_reason(region: dict[str, Any]) -> str:
    value = region.get("classification_reason")
    if not value:
        value = (region.get("render", {}) or {}).get("classification_reason")
    return str(value or "").strip()


def _ownership_strong(link: dict[str, Any]) -> bool:
    return (
        str(link.get("ownership_type") or "") in {"inside", "overlaps"}
        and float(link.get("confidence", 0.0) or 0.0) >= 0.70
    )


def _ownership_id(link: dict[str, Any]) -> str | None:
    rid = str(link.get("region_id") or "")
    cid = str(link.get("container_id") or "")
    if not rid or not cid:
        return None
    return f"{rid}->{cid}"


def _diagnostic_reason_codes(link: dict[str, Any], container: dict[str, Any]) -> list[str]:
    reasons = []
    ctype = str(container.get("container_type") or "")
    if ctype:
        reasons.append(f"container_type:{ctype}")
    relation = str(link.get("ownership_type") or "")
    if relation:
        reasons.append(f"ownership:{relation}")
    for reason in link.get("reason_codes", []) or []:
        reasons.append(f"ownership_reason:{reason}")
    for reason in (container.get("evidence", {}) or {}).get("reason_codes", []) or []:
        reasons.append(f"container_evidence:{reason}")
    return reasons


def _route_contraindications(region: dict[str, Any], container: dict[str, Any]) -> list[str]:
    semantic = _semantic_class(region)
    reason = _classification_reason(region)
    ctype = str(container.get("container_type") or "")
    contraindications = []
    if ctype == "sfx_decorative" and semantic == "speech_bubble" and reason in SPEECH_REASONS:
        contraindications.append("current_speech_reason_conflicts_with_decorative_container")
    if ctype == "speech_bubble" and reason in DECORATIVE_REASONS:
        contraindications.append("current_decorative_reason_conflicts_with_speech_container")
    if ctype in {"caption", "sign/background_text_area"} and reason in SPEECH_REASONS:
        contraindications.append("current_speech_reason_conflicts_with_caption_container")
    return contraindications


def _short_kana_laugh_or_reaction_text(region: dict[str, Any]) -> bool:
    text = "".join(str(region.get("ocr_text") or "").split())
    if not text:
        return False
    chars = [ch for ch in text if not _is_reaction_punctuation(ch)]
    if not chars or len(chars) > 8:
        return False
    return all(_is_kana_or_kana_mark(ch) for ch in chars)


def _is_kana_or_kana_mark(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xFF66 <= code <= 0xFF9F
    )


def _is_reaction_punctuation(ch: str) -> bool:
    code = ord(ch)
    return ch in {".", ",", "!", "?", "~", "-", "_"} or code in {
        0x3000,
        0x3001,
        0x3002,
        0x30FB,
        0xFF01,
        0xFF1F,
        0xFF5E,
        0x2026,
        0x22EF,
    }


def _high_ocr_confidence(region: dict[str, Any]) -> bool:
    value = region.get("ocr_confidence")
    if value is None:
        value = (region.get("confidence", {}) or {}).get("ocr")
    try:
        return float(value) >= 0.75
    except (TypeError, ValueError):
        return False
