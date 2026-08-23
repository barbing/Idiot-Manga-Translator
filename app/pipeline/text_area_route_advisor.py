# -*- coding: utf-8 -*-
"""Semantic route advisor and experimental opt-in route assist.

The advisor consumes existing text-area diagnostics and current audit metadata,
then emits dry-run route suggestions for review. By default it does not mutate
production region routing, cleanup, translation, rendering, or project output.
"""
from __future__ import annotations

import time
from typing import Any


ROUTE_ADVISOR_VERSION = "text_area_route_advisor_phase2a_v1"
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
    enriched["route_advisor_runtime_sec"] = round(time.time() - start, 6)
    return enriched


def route_assist_enabled() -> bool:
    """Compatibility probe: route suggestions never have mutation authority."""

    return False


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
