# -*- coding: utf-8 -*-
"""Offline text-area ownership diagnostics.

This module is intentionally read-only: it consumes existing debug/project
artifacts and original/output images, then writes or returns diagnostic
artifacts. Production OCR, detection, translation, cleanup, inpainting, routing,
and rendering decisions must not consume this output.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:
    from PIL import Image, ImageDraw, ImageFont, ImageStat
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageStat = None


SPEECH_REASONS = {
    "bubble_contained_short_laugh_speech",
    "speech_bubble_missed_text_recovery",
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

DIAGNOSTIC_VERSION = "text_area_diagnostics_phase1_v1"
BUBBLE_DETECTION_CONSUMER_VERSION = "phase4b17_bubble_detection_consumer_v1"


@dataclass
class PageArtifacts:
    page_id: str
    audit_path: Path
    audit: dict[str, Any]
    project_regions: dict[str, dict[str, Any]]


def run_diagnostic_report(
    *,
    debug_dirs: list[Path],
    project_jsons: list[Path],
    pages: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Run offline diagnostics for pages and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    project_index = _load_project_index(project_jsons)
    summary: dict[str, Any] = {
        "output_dir": str(output_dir.resolve()),
        "pages": [],
        "missing_pages": [],
        "reference_case_results": [],
        "total_runtime_seconds": 0.0,
    }
    total_start = time.time()
    for page_id in pages:
        start = time.time()
        try:
            artifacts = _load_page_artifacts(page_id, debug_dirs, project_index)
        except FileNotFoundError as exc:
            summary["missing_pages"].append({"page_id": page_id, "missing": str(exc)})
            continue
        page_dir = output_dir / page_id
        page_dir.mkdir(parents=True, exist_ok=True)
        page_result = analyze_page(artifacts, page_dir)
        runtime = time.time() - start
        page_result["runtime_seconds"] = runtime
        summary["pages"].append(
            {
                "page_id": page_id,
                "runtime_seconds": runtime,
                "audit_path": str(artifacts.audit_path),
                "containers": len(page_result["containers"]),
                "text_instances": len(page_result["text_instances"]),
                "ownership": len(page_result["ownership"]),
                "render_constraints": len(page_result["render_constraints"]),
                "route_suggestions": len(page_result.get("route_suggestions", [])),
                "route_suggestion_summary": page_result.get("route_suggestion_summary", {}),
                "render_plan_suggestions": len(page_result.get("render_plan_suggestions", [])),
                "render_plan_suggestion_summary": page_result.get("render_plan_suggestion_summary", {}),
                "overlay_path": page_result.get("overlay_path"),
                "compact_overlay_path": page_result.get("compact_overlay_path"),
                "warnings_overlay_path": page_result.get("warnings_overlay_path"),
                "case_summary_path": page_result.get("case_summary_path"),
            }
        )
        summary["reference_case_results"].extend(page_result["reference_case_results"])
    summary["total_runtime_seconds"] = time.time() - total_start
    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_reference_summary(output_dir / "reference_case_results.md", summary["reference_case_results"])
    return summary


def build_text_area_diagnostics(
    *,
    page_id: str,
    audit: dict[str, Any],
    audit_path: str | Path | None = None,
    project_regions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build diagnostic-only text-area metadata for one page.

    This is the shared core used by the standalone Phase 0 report and by debug
    artifact enrichment. It reads only existing region/audit/image metadata.
    """
    artifacts = PageArtifacts(
        str(page_id),
        Path(audit_path) if audit_path else Path(""),
        audit,
        project_regions or {},
    )
    return _analyze_artifacts(artifacts)


def enrich_audit_with_text_area_diagnostics(
    audit: dict[str, Any],
    *,
    audit_path: str | Path | None = None,
    project_regions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach diagnostic text-area metadata to a page audit.

    Failures are recorded in diagnostic status fields and do not propagate.
    """
    start = time.time()
    enriched = dict(audit)
    enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
    enriched["diagnostic_version"] = DIAGNOSTIC_VERSION
    try:
        diagnostics = build_text_area_diagnostics(
            page_id=str(enriched.get("page_id") or "page"),
            audit=enriched,
            audit_path=audit_path,
            project_regions=project_regions,
        )
        _attach_diagnostics_to_audit(enriched, diagnostics)
        enriched["diagnostic_generated"] = True
        enriched["diagnostic_error"] = None
    except Exception as exc:  # pragma: no cover - debug safety net
        enriched["diagnostic_generated"] = False
        enriched["diagnostic_error"] = str(exc)
        enriched.setdefault("text_containers", [])
        enriched.setdefault("text_instances", [])
        enriched.setdefault("text_ownership", [])
        enriched.setdefault("logical_text_blocks", [])
        enriched.setdefault("render_constraints", [])
        enriched.setdefault("visual_role_evidence", [])
    enriched["diagnostic_runtime_sec"] = round(time.time() - start, 6)
    return enriched


def attach_bubble_detection_consumer_fields(
    audit: MutableMapping[str, Any],
    bubble_detection_result: Mapping[str, Any] | None = None,
    *,
    consumer_source: str = "text_area_diagnostics",
) -> MutableMapping[str, Any]:
    """Attach normalized BubbleDetection service evidence to audit regions.

    This is a metadata-only bridge for diagnostics, route advice, render
    planning, and review artifacts. It never changes deterministic text-area
    diagnostics or any production routing/rendering behavior.
    """

    audit["bubble_detection_consumer_version"] = BUBBLE_DETECTION_CONSUMER_VERSION
    audit["bubble_detection_consumer_generated"] = False
    audit["bubble_detection_consumer_error"] = None
    audit["bubble_detection_consumer_sources"] = _dedupe_strings(
        [consumer_source, "bubble_detection_service"]
        + _as_string_list(audit.get("bubble_detection_consumer_sources"))
    )

    try:
        result = bubble_detection_result
        if not isinstance(result, Mapping):
            result = audit.get("bubble_detection_result")
        if not isinstance(result, Mapping) or not result.get("generated"):
            reason = None
            if isinstance(result, Mapping):
                reason = result.get("error") or "bubble_detection_result_not_generated"
            _stamp_empty_bubble_consumer_fields(audit, reason or "bubble_detection_result_unavailable")
            return audit

        containers_by_id = {
            str(item.get("fused_container_id") or item.get("container_id") or ""): item
            for item in result.get("fused_containers", []) or []
            if isinstance(item, Mapping)
        }
        memberships_by_rid = {
            str(item.get("region_id") or ""): item
            for item in result.get("memberships", []) or []
            if isinstance(item, Mapping)
        }
        decisions_by_rid = {
            str(item.get("region_id") or ""): item
            for item in result.get("decisions", []) or []
            if isinstance(item, Mapping)
        }

        for region in audit.get("regions", []) or []:
            if not isinstance(region, MutableMapping):
                continue
            rid = str(region.get("region_id") or "")
            membership = memberships_by_rid.get(rid) or {}
            container_id = str(membership.get("container_id") or "")
            container = containers_by_id.get(container_id) or {}
            decision = decisions_by_rid.get(rid) or {}
            _attach_region_bubble_consumer_fields(region, container, membership, decision, consumer_source)

        audit["bubble_detection_consumer_generated"] = True
    except Exception as exc:  # pragma: no cover - diagnostic safety net
        audit["bubble_detection_consumer_generated"] = False
        audit["bubble_detection_consumer_error"] = f"{type(exc).__name__}: {exc}"
        _stamp_empty_bubble_consumer_fields(audit, audit["bubble_detection_consumer_error"])
    return audit


def write_text_area_diagnostic_overlays(
    audit: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write compact diagnostic overlays from an enriched audit payload."""
    if Image is None or ImageDraw is None:
        return {"generated": False, "error": "Pillow is unavailable"}
    image_path = _resolve_existing_path(audit.get("source_path")) or _resolve_existing_path(audit.get("output_path"))
    if image_path is None:
        return {"generated": False, "error": "source/output image path is unavailable"}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    containers = audit.get("text_containers", []) or []
    instances = audit.get("text_instances", []) or []
    ownership = audit.get("text_ownership", []) or []
    constraints = audit.get("render_constraints", []) or []

    compact_path = output_path / "container_overlay_compact.jpg"
    warnings_path = output_path / "container_overlay_warnings.jpg"
    _write_compact_container_overlay(image, containers, instances, ownership, constraints, compact_path)
    warning_count = _write_warning_container_overlay(image, containers, instances, ownership, constraints, warnings_path)
    crop_paths = _write_warning_crops(image, containers, constraints, output_path / "warning_crops")
    return {
        "generated": True,
        "container_overlay_compact": str(compact_path),
        "container_overlay_warnings": str(warnings_path) if warning_count else None,
        "warning_crops": crop_paths,
        "warning_count": warning_count,
    }


def analyze_page(artifacts: PageArtifacts, output_dir: Path) -> dict[str, Any]:
    page_result = _analyze_artifacts(artifacts)
    page_id = artifacts.page_id
    image_path = _resolve_existing_path(artifacts.audit.get("source_path"))
    output_path = _resolve_existing_path(artifacts.audit.get("output_path"))
    image = Image.open(image_path).convert("RGB") if Image is not None and image_path else None

    containers = page_result["containers"]
    text_instances = page_result["text_instances"]
    ownership = page_result["ownership"]
    logical_blocks = page_result["logical_text_blocks"]
    render_constraints = page_result["render_constraints"]
    reference_results = page_result["reference_case_results"]
    route_suggestions = _build_route_advisor_payload(artifacts, page_result)
    route_suggestion_summary = _summarize_route_suggestions(route_suggestions)
    render_plan_suggestions = _build_render_planner_payload(artifacts, page_result)
    render_plan_suggestion_summary = _summarize_render_plan_suggestions(render_plan_suggestions)

    containers_payload = {
        "page_id": page_id,
        "audit_path": str(artifacts.audit_path),
        "source_path": str(image_path) if image_path else None,
        "output_path": str(output_path) if output_path else None,
        "text_instances": text_instances,
        "containers": containers,
        "logical_text_blocks": logical_blocks,
        "visual_role_evidence": page_result["visual_role_evidence"],
    }
    ownership_payload = {"page_id": page_id, "ownership": ownership}
    constraints_payload = {
        "page_id": page_id,
        "render_constraints": render_constraints,
        "preserved_obstacles": [
            container
            for container in containers
            if container.get("container_type") == "sfx_decorative"
        ],
    }

    (output_dir / "containers.json").write_text(json.dumps(containers_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "ownership.json").write_text(json.dumps(ownership_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "render_constraints.json").write_text(json.dumps(constraints_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "route_suggestions.json").write_text(
        json.dumps(
            {
                "page_id": page_id,
                "route_suggestions": route_suggestions,
                "summary": route_suggestion_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "render_plan_suggestions.json").write_text(
        json.dumps(
            {
                "page_id": page_id,
                "render_plan_suggestions": render_plan_suggestions,
                "summary": render_plan_suggestion_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    overlay_path = output_dir / "container_overlay.jpg"
    _write_container_overlay(image, containers, text_instances, render_constraints, overlay_path)
    overlay_paths = write_text_area_diagnostic_overlays(
        {
            "source_path": str(image_path) if image_path else None,
            "output_path": str(output_path) if output_path else None,
            "text_containers": containers,
            "text_instances": text_instances,
            "text_ownership": ownership,
            "render_constraints": render_constraints,
            "visual_role_evidence": page_result["visual_role_evidence"],
        },
        output_dir,
    )
    summary_path = output_dir / "case_summary.md"
    _write_case_summary(
        summary_path,
        page_id,
        containers,
        text_instances,
        ownership,
        logical_blocks,
        render_constraints,
        reference_results,
        route_suggestions,
        route_suggestion_summary,
        render_plan_suggestions,
        render_plan_suggestion_summary,
        overlay_paths,
    )

    page_result["overlay_path"] = str(overlay_path)
    page_result["compact_overlay_path"] = overlay_paths.get("container_overlay_compact")
    page_result["warnings_overlay_path"] = overlay_paths.get("container_overlay_warnings")
    page_result["warning_crop_paths"] = overlay_paths.get("warning_crops", [])
    page_result["case_summary_path"] = str(summary_path)
    page_result["route_suggestions"] = route_suggestions
    page_result["route_suggestion_summary"] = route_suggestion_summary
    page_result["render_plan_suggestions"] = render_plan_suggestions
    page_result["render_plan_suggestion_summary"] = render_plan_suggestion_summary
    return page_result


def _analyze_artifacts(artifacts: PageArtifacts) -> dict[str, Any]:
    audit = artifacts.audit
    page_id = artifacts.page_id
    regions = [_merge_project_region(r, artifacts.project_regions.get(str(r.get("region_id", "")))) for r in audit.get("regions", [])]
    image_path = _resolve_existing_path(audit.get("source_path"))
    image = Image.open(image_path).convert("RGB") if Image is not None and image_path else None
    page_size = image.size if image is not None else _page_size_from_regions(regions)

    text_instances = [_build_text_instance(page_id, idx, region, image, page_size) for idx, region in enumerate(regions)]
    containers = _build_containers(page_id, regions, text_instances, page_size)
    ownership = _build_ownership(page_id, text_instances, containers)
    logical_blocks = _build_logical_blocks(page_id, text_instances, containers, ownership, regions)
    render_constraints = _build_render_constraints(page_id, regions, containers)
    reference_results = _reference_case_results(page_id, text_instances, containers, ownership, logical_blocks, render_constraints)

    return {
        "page_id": page_id,
        "text_instances": text_instances,
        "containers": containers,
        "ownership": ownership,
        "logical_text_blocks": logical_blocks,
        "visual_role_evidence": [
            {
                "instance_id": item["instance_id"],
                "region_id": item["region_id"],
                "evidence": item["visual_role_evidence"],
            }
            for item in text_instances
        ],
        "render_constraints": render_constraints,
        "reference_case_results": reference_results,
    }


def _attach_diagnostics_to_audit(audit: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    audit["text_containers"] = diagnostics.get("containers", [])
    audit["text_instances"] = diagnostics.get("text_instances", [])
    audit["text_ownership"] = diagnostics.get("ownership", [])
    audit["logical_text_blocks"] = diagnostics.get("logical_text_blocks", [])
    audit["render_constraints"] = diagnostics.get("render_constraints", [])
    audit["visual_role_evidence"] = diagnostics.get("visual_role_evidence", [])

    containers_by_id = {item.get("container_id"): item for item in audit["text_containers"]}
    ownership_by_rid = {item.get("region_id"): item for item in audit["text_ownership"]}
    evidence_by_rid = {item.get("region_id"): item.get("evidence", {}) for item in audit["visual_role_evidence"]}
    constraints_by_rid: dict[str, list[dict[str, Any]]] = {}
    for constraint in audit["render_constraints"]:
        if constraint.get("constraint_status") == "satisfied":
            continue
        constraints_by_rid.setdefault(str(constraint.get("region_id") or ""), []).append(
            {
                "constraint_status": constraint.get("constraint_status"),
                "reason_codes": constraint.get("reason_codes", []),
                "outside_allowed_ratio": constraint.get("outside_allowed_ratio"),
                "obstacle_container_ids": constraint.get("newly_introduced_obstacle_container_ids", []),
            }
        )
    phrase_warnings_by_rid: dict[str, list[dict[str, Any]]] = {}
    for block in audit["logical_text_blocks"]:
        conservation = block.get("text_conservation", {}) or {}
        warnings = []
        if not conservation.get("represented_once", True):
            warnings.append("text_not_represented_once")
        if conservation.get("missing"):
            warnings.append("missing_source_phrase")
        if conservation.get("duplicates"):
            warnings.append("duplicate_source_phrase")
        if not warnings:
            continue
        payload = {
            "block_id": block.get("block_id"),
            "warnings": warnings,
            "missing": conservation.get("missing", []),
            "duplicates": conservation.get("duplicates", []),
        }
        for rid in block.get("region_ids", []) or []:
            phrase_warnings_by_rid.setdefault(str(rid), []).append(payload)

    for region in audit.get("regions", []) or []:
        rid = str(region.get("region_id") or "")
        link = ownership_by_rid.get(rid) or {}
        container = containers_by_id.get(link.get("container_id")) or {}
        region["diagnostic_text_container_id"] = link.get("container_id")
        region["diagnostic_container_type"] = container.get("container_type")
        region["diagnostic_ownership_type"] = link.get("ownership_type")
        region["diagnostic_ownership_confidence"] = link.get("confidence")
        region["diagnostic_ownership_reason_codes"] = link.get("reason_codes", [])
        region["diagnostic_role_evidence"] = evidence_by_rid.get(rid, {})
        region["diagnostic_render_constraint_warnings"] = constraints_by_rid.get(rid, [])
        region["diagnostic_phrase_conservation_warnings"] = phrase_warnings_by_rid.get(rid, [])


def append_bubble_detection_consumer_source(region: MutableMapping[str, Any], source: str) -> None:
    """Record that a diagnostic consumer inspected BubbleDetection metadata."""

    current = _as_string_list(region.get("diagnostic_bubble_consumer_sources"))
    region["diagnostic_bubble_consumer_sources"] = _dedupe_strings(current + [source])


def _attach_region_bubble_consumer_fields(
    region: MutableMapping[str, Any],
    container: Mapping[str, Any],
    membership: Mapping[str, Any],
    decision: Mapping[str, Any],
    consumer_source: str,
) -> None:
    container_id = str(membership.get("container_id") or container.get("fused_container_id") or container.get("container_id") or "")
    container_type = str(container.get("fused_container_type") or container.get("container_type") or "")
    conflict_flags = _dedupe_strings(
        _as_string_list(container.get("conflict_flags"))
        + _as_string_list(decision.get("conflict_flags"))
    )
    confidence_tier = _bubble_confidence_tier(region, container, membership, decision, conflict_flags)
    decision_supported_actions = _as_string_list(decision.get("supported_actions"))
    container_suggested_actions = _as_string_list(container.get("suggested_downstream_use"))
    supported_actions = _dedupe_strings(decision_supported_actions or container_suggested_actions)
    blocked_actions = _dedupe_strings(
        _as_string_list(decision.get("blocked_actions"))
        + _bubble_consumer_blocked_actions(region, container, confidence_tier, conflict_flags)
    )
    reason_codes = _dedupe_strings(
        [f"confidence_tier:{confidence_tier}"]
        + [f"container_reason:{item}" for item in _as_string_list(container.get("reason_codes"))]
        + [f"membership_reason:{item}" for item in _as_string_list(membership.get("ownership_reason_codes"))]
        + [f"decision_reason:{item}" for item in _as_string_list(decision.get("reason_codes"))]
        + [f"bubble_conflict:{item}" for item in conflict_flags]
    )
    review_only = _bubble_consumer_review_only(region, container, decision, confidence_tier, conflict_flags)

    region["diagnostic_bubble_container_id"] = container_id or None
    region["diagnostic_bubble_container_type"] = container_type or None
    region["diagnostic_bubble_membership_type"] = membership.get("membership_type") if membership else None
    region["diagnostic_bubble_membership_confidence"] = membership.get("ownership_confidence") if membership else None
    region["diagnostic_bubble_confidence_tier"] = confidence_tier
    region["diagnostic_bubble_conflict_flags"] = conflict_flags
    region["diagnostic_bubble_review_only"] = review_only
    region["diagnostic_bubble_source_model_ids"] = _bubble_source_model_ids(container)
    region["diagnostic_bubble_supported_actions"] = supported_actions
    region["diagnostic_bubble_blocked_actions"] = blocked_actions
    region["diagnostic_bubble_container_suggested_actions"] = _dedupe_strings(container_suggested_actions)
    region["diagnostic_bubble_reason_codes"] = reason_codes
    region["diagnostic_bubble_inside_ratio"] = membership.get("inside_ratio") if membership else None
    region["diagnostic_bubble_center_inside"] = membership.get("center_inside") if membership else None
    region["diagnostic_bubble_would_change_behavior"] = False
    append_bubble_detection_consumer_source(region, consumer_source)
    append_bubble_detection_consumer_source(region, "bubble_detection_service")


def _stamp_empty_bubble_consumer_fields(audit: MutableMapping[str, Any], reason: str) -> None:
    audit["bubble_detection_consumer_error"] = reason
    for region in audit.get("regions", []) or []:
        if not isinstance(region, MutableMapping):
            continue
        region["diagnostic_bubble_container_id"] = None
        region["diagnostic_bubble_container_type"] = None
        region["diagnostic_bubble_membership_type"] = None
        region["diagnostic_bubble_membership_confidence"] = None
        region["diagnostic_bubble_confidence_tier"] = "unsupported_by_model_but_deterministic_valid"
        region["diagnostic_bubble_conflict_flags"] = []
        region["diagnostic_bubble_review_only"] = True
        region["diagnostic_bubble_source_model_ids"] = []
        region["diagnostic_bubble_supported_actions"] = []
        region["diagnostic_bubble_blocked_actions"] = ["bubble_detection_unavailable", "automatic_mutation_without_explicit_proof_flag"]
        region["diagnostic_bubble_container_suggested_actions"] = []
        region["diagnostic_bubble_reason_codes"] = [str(reason)]
        region["diagnostic_bubble_inside_ratio"] = None
        region["diagnostic_bubble_center_inside"] = None
        region["diagnostic_bubble_would_change_behavior"] = False
        append_bubble_detection_consumer_source(region, "bubble_detection_service")


def _bubble_confidence_tier(
    region: Mapping[str, Any],
    container: Mapping[str, Any],
    membership: Mapping[str, Any],
    decision: Mapping[str, Any],
    conflict_flags: list[str],
) -> str:
    if conflict_flags:
        return "conflict_preserve_wins"
    if not container:
        return "unsupported_by_model_but_deterministic_valid"
    container_type = str(container.get("fused_container_type") or container.get("container_type") or "")
    confidence = str(container.get("confidence") or decision.get("decision_status") or "").lower()
    kitsumed_ids = _as_string_list(container.get("linked_kitsumed_mask_ids"))
    ogkalu_ids = _as_string_list(container.get("linked_ogkalu_detection_ids"))
    suggested_use = set(_as_string_list(container.get("suggested_downstream_use")))
    if _is_preserve_or_decorative(region) and container_type == "speech_bubble":
        return "conflict_preserve_wins"
    if container_type == "free_text":
        return "text_free_review_only"
    if container_type == "speech_bubble" and kitsumed_ids and ogkalu_ids and confidence == "high":
        return "strong_model_container"
    if container_type == "speech_bubble" and kitsumed_ids:
        return "mask_primary_container"
    if ogkalu_ids and "ownership_hint" in suggested_use:
        return "text_bubble_review_container"
    if ogkalu_ids and not kitsumed_ids:
        return "text_bubble_review_container"
    return "unsupported_by_model_but_deterministic_valid"


def _bubble_consumer_review_only(
    region: Mapping[str, Any],
    container: Mapping[str, Any],
    decision: Mapping[str, Any],
    confidence_tier: str,
    conflict_flags: list[str],
) -> bool:
    if conflict_flags:
        return True
    if _is_preserve_or_decorative(region) and container:
        return True
    if confidence_tier in {
        "text_bubble_review_container",
        "text_free_review_only",
        "conflict_preserve_wins",
        "unsupported_by_model_but_deterministic_valid",
    }:
        return True
    if str(decision.get("decision_status") or "") != "supported":
        return True
    return bool(container.get("human_review_required"))


def _bubble_consumer_blocked_actions(
    region: Mapping[str, Any],
    container: Mapping[str, Any],
    confidence_tier: str,
    conflict_flags: list[str],
) -> list[str]:
    blocked = ["automatic_mutation_without_explicit_proof_flag"]
    if conflict_flags:
        blocked.extend(["automatic_route_mutation", "automatic_cleanup_mutation", "automatic_render_mutation"])
    if confidence_tier in {"text_bubble_review_container", "text_free_review_only", "unsupported_by_model_but_deterministic_valid"}:
        blocked.append("model_evidence_review_only")
    if _is_preserve_or_decorative(region):
        blocked.append("deterministic_preserve_or_decorative_route_wins")
    if not container:
        blocked.append("no_model_backed_container")
    return blocked


def _bubble_source_model_ids(container: Mapping[str, Any]) -> list[str]:
    return _dedupe_strings(
        [f"kitsumed:{item}" for item in _as_string_list(container.get("linked_kitsumed_mask_ids"))]
        + [f"ogkalu:{item}" for item in _as_string_list(container.get("linked_ogkalu_detection_ids"))]
    )


def _is_preserve_or_decorative(region: Mapping[str, Any]) -> bool:
    semantic = str(region.get("semantic_class") or region.get("type") or "").strip()
    cleanup = str(region.get("cleanup_mode") or (region.get("render", {}) or {}).get("cleanup_mode") or "").strip()
    reason = str(region.get("classification_reason") or (region.get("render", {}) or {}).get("classification_reason") or "").strip()
    return (
        semantic in {"decorative_text", "sfx"}
        or cleanup == "preserve"
        or reason in DECORATIVE_REASONS
        or bool(region.get("is_decorative"))
        or bool(region.get("is_sfx"))
    )


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _dedupe_strings(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _build_route_advisor_payload(artifacts: PageArtifacts, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from app.pipeline.text_area_route_advisor import build_route_suggestions

        advisor_audit = dict(artifacts.audit)
        advisor_audit["regions"] = [dict(region) for region in artifacts.audit.get("regions", []) or []]
        advisor_audit["diagnostic_generated"] = True
        advisor_audit["text_containers"] = diagnostics.get("containers", [])
        advisor_audit["text_instances"] = diagnostics.get("text_instances", [])
        advisor_audit["text_ownership"] = diagnostics.get("ownership", [])
        advisor_audit["logical_text_blocks"] = diagnostics.get("logical_text_blocks", [])
        advisor_audit["render_constraints"] = diagnostics.get("render_constraints", [])
        advisor_audit["visual_role_evidence"] = diagnostics.get("visual_role_evidence", [])
        return build_route_suggestions(advisor_audit)
    except Exception as exc:  # pragma: no cover - offline report should continue
        return [
            {
                "suggestion_type": "route_advisor_error",
                "confidence": "low",
                "reason_codes": ["route_advisor_exception"],
                "required_evidence": [],
                "contraindications": [str(exc)],
                "would_change_behavior": False,
                "phase2_status": "advisory_only",
                "human_review_required": True,
            }
        ]


def _build_render_planner_payload(artifacts: PageArtifacts, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from app.pipeline.text_area_render_planner import build_render_plan_suggestions

        planner_audit = dict(artifacts.audit)
        planner_audit["regions"] = [dict(region) for region in artifacts.audit.get("regions", []) or []]
        planner_audit["diagnostic_generated"] = True
        planner_audit["text_containers"] = diagnostics.get("containers", [])
        planner_audit["text_instances"] = diagnostics.get("text_instances", [])
        planner_audit["text_ownership"] = diagnostics.get("ownership", [])
        planner_audit["logical_text_blocks"] = diagnostics.get("logical_text_blocks", [])
        planner_audit["render_constraints"] = diagnostics.get("render_constraints", [])
        planner_audit["visual_role_evidence"] = diagnostics.get("visual_role_evidence", [])
        return build_render_plan_suggestions(planner_audit)
    except Exception as exc:  # pragma: no cover - offline report should continue
        return [
            {
                "suggestion_type": "render_planner_error",
                "severity": "watch",
                "confidence": "low",
                "reason_codes": ["render_planner_exception"],
                "would_change_behavior": False,
                "phase3_status": "diagnostic_only",
                "human_review_required": True,
                "proposed_action": "review_only_no_action",
                "error": str(exc),
            }
        ]


def _summarize_route_suggestions(route_suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from app.pipeline.text_area_route_advisor import summarize_route_suggestions

        return summarize_route_suggestions(route_suggestions)
    except Exception:
        by_type: dict[str, int] = {}
        for item in route_suggestions:
            key = str(item.get("suggestion_type") or "unknown")
            by_type[key] = by_type.get(key, 0) + 1
        return {"total": len(route_suggestions), "by_type": by_type, "by_confidence": {}}


def _summarize_render_plan_suggestions(render_plan_suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from app.pipeline.text_area_render_planner import summarize_render_plan_suggestions

        return summarize_render_plan_suggestions(render_plan_suggestions)
    except Exception:
        by_type: dict[str, int] = {}
        for item in render_plan_suggestions:
            key = str(item.get("suggestion_type") or "unknown")
            by_type[key] = by_type.get(key, 0) + 1
        return {"total": len(render_plan_suggestions), "by_type": by_type, "by_severity": {}, "by_confidence": {}}



def _load_page_artifacts(page_id: str, debug_dirs: list[Path], project_index: dict[tuple[str, str], dict[str, Any]]) -> PageArtifacts:
    normalized = _normalize_page_id(page_id)
    for debug_dir in debug_dirs:
        audit_path = debug_dir / normalized / f"{normalized}_region_audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            page_key = str(audit.get("page_id") or normalized)
            project_regions = {
                rid: region
                for (pid, rid), region in project_index.items()
                if pid == page_key
            }
            return PageArtifacts(page_key, audit_path, audit, project_regions)
    searched = ", ".join(str(path) for path in debug_dirs)
    raise FileNotFoundError(f"{normalized}_region_audit.json in {searched}")


def _load_project_index(project_jsons: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in project_jsons:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for page in data.get("pages", []) or []:
            page_id = str(page.get("page_id", "") or "")
            for region in page.get("regions", []) or []:
                rid = str(region.get("region_id", "") or "")
                if page_id and rid and (page_id, rid) not in index:
                    index[(page_id, rid)] = region
    return index


def _merge_project_region(audit_region: dict[str, Any], project_region: dict[str, Any] | None) -> dict[str, Any]:
    region = dict(audit_region)
    if not project_region:
        return region
    render = dict(project_region.get("render", {}) or {})
    render.update(region.get("_render", {}) or {})
    # Audit fields are flattened; keep project render metadata for transfer evidence.
    region["project_region"] = project_region
    region["render"] = render
    if project_region.get("group_id") and not region.get("group_id"):
        region["group_id"] = project_region.get("group_id")
    if project_region.get("bubble_id") and not region.get("bubble_id"):
        region["bubble_id"] = project_region.get("bubble_id")
    return region


def _build_text_instance(
    page_id: str,
    idx: int,
    region: dict[str, Any],
    image,
    page_size: tuple[int, int],
) -> dict[str, Any]:
    rid = str(region.get("region_id", "") or f"r{idx:03d}")
    bbox = _bbox_xywh(region.get("bbox"))
    evidence = _visual_evidence(region, image, bbox, page_size)
    role_hint = _visual_role_hint(region, evidence, page_size)
    return {
        "instance_id": f"i{idx:03d}",
        "page_id": page_id,
        "region_id": rid,
        "raw_detector_box_id": f"d{idx:03d}",
        "ocr_crop_id": f"o{idx:03d}",
        "bbox": bbox,
        "polygon": region.get("polygon"),
        "ocr_text": region.get("ocr_text", ""),
        "ocr_confidence": region.get("ocr_confidence"),
        "detection_confidence": region.get("detection_confidence"),
        "semantic_class": region.get("semantic_class"),
        "classification_reason": region.get("classification_reason") or (region.get("render", {}) or {}).get("classification_reason"),
        "cleanup_mode": region.get("cleanup_mode") or (region.get("render", {}) or {}).get("cleanup_mode"),
        "translated_text": region.get("translated_text", ""),
        "sent_to_translation": region.get("sent_to_translation"),
        "skip_reason": region.get("skip_reason"),
        "source_orientation": region.get("source_orientation") or (region.get("render", {}) or {}).get("source_orientation"),
        "visual_role_hint": role_hint,
        "visual_role_evidence": evidence,
    }


def _build_containers(
    page_id: str,
    regions: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    page_size: tuple[int, int],
) -> list[dict[str, Any]]:
    instance_by_rid = {item["region_id"]: item for item in instances}
    groups = _initial_container_groups(regions, instances)
    groups = _merge_nearby_speech_groups(groups, instance_by_rid)
    containers: list[dict[str, Any]] = []
    for idx, group in enumerate(groups):
        group_instances = [instance_by_rid[rid] for rid in group["region_ids"] if rid in instance_by_rid]
        if not group_instances:
            continue
        container_type = group["container_type"]
        source_boxes = [item["bbox"] for item in group_instances]
        if container_type == "speech_bubble":
            allowed_boxes = []
            for item in group_instances:
                region = _region_for_instance(regions, item["region_id"])
                cleanup = _xyxy_to_xywh(region.get("cleanup_mask_bbox"))
                if cleanup:
                    allowed_boxes.append(cleanup)
            bbox = _expand_xywh(_union_xywh(allowed_boxes or source_boxes), 8, page_size)
        else:
            bbox = _expand_xywh(_union_xywh(source_boxes), 4, page_size)
        confidence = _container_confidence(container_type, group_instances)
        evidence = _combine_evidence(group_instances)
        containers.append(
            {
                "container_id": f"c{idx:03d}",
                "page_id": page_id,
                "container_type": container_type,
                "bbox": bbox,
                "polygon": _bbox_polygon(bbox),
                "source": group.get("source", "region_audit_deterministic"),
                "confidence": confidence,
                "instance_region_ids": [item["region_id"] for item in group_instances],
                "evidence": evidence,
                "failure_modes": _container_failure_modes(container_type, group_instances, bbox),
            }
        )
    return containers


def _initial_container_groups(regions: list[dict[str, Any]], instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    claimed: set[str] = set()
    region_by_id = {str(region.get("region_id", "") or ""): region for region in regions}

    explicit_groups: dict[str, list[str]] = {}
    for region in regions:
        rid = str(region.get("region_id", "") or "")
        if not rid:
            continue
        group_id = str(region.get("group_id") or "").strip()
        render = region.get("render", {}) or {}
        transfer_to = str(
            render.get("bubble_local_transfer_to_region_id")
            or render.get("speech_text_conservation_transfer_to_region_id")
            or ""
        ).strip()
        if group_id:
            explicit_groups.setdefault(group_id, []).append(rid)
        elif transfer_to:
            explicit_groups.setdefault(f"transfer:{transfer_to}", []).extend([transfer_to, rid])
    for group_id, rids in explicit_groups.items():
        unique = []
        for rid in rids:
            if rid and rid in region_by_id and rid not in unique:
                unique.append(rid)
        if len(unique) < 2:
            continue
        groups.append(
            {
                "container_type": "speech_bubble",
                "region_ids": unique,
                "source": f"explicit_ownership:{group_id}",
            }
        )
        claimed.update(unique)

    for instance in instances:
        rid = instance["region_id"]
        if rid in claimed:
            continue
        role = instance.get("visual_role_hint") or "unknown"
        if role == "speech":
            ctype = "speech_bubble"
        elif role == "caption":
            ctype = "caption"
        elif role == "background":
            ctype = "sign/background_text_area"
        elif role in {"sfx", "decorative"}:
            ctype = "sfx_decorative"
        else:
            ctype = "unknown"
        groups.append({"container_type": ctype, "region_ids": [rid], "source": "region_role_hint"})
    return groups


def _merge_nearby_speech_groups(
    groups: list[dict[str, Any]],
    instance_by_rid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changed = True
    groups = [dict(group) for group in groups]
    while changed:
        changed = False
        for i in range(len(groups)):
            if groups[i].get("container_type") != "speech_bubble":
                continue
            for j in range(i + 1, len(groups)):
                if groups[j].get("container_type") != "speech_bubble":
                    continue
                if _speech_groups_should_merge(groups[i], groups[j], instance_by_rid):
                    groups[i]["region_ids"] = list(dict.fromkeys(groups[i]["region_ids"] + groups[j]["region_ids"]))
                    groups[i]["source"] = f"{groups[i].get('source')};nearby_speech_text_area"
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def _speech_groups_should_merge(
    a: dict[str, Any],
    b: dict[str, Any],
    instance_by_rid: dict[str, dict[str, Any]],
) -> bool:
    a_boxes = [instance_by_rid[rid]["bbox"] for rid in a["region_ids"] if rid in instance_by_rid]
    b_boxes = [instance_by_rid[rid]["bbox"] for rid in b["region_ids"] if rid in instance_by_rid]
    if not a_boxes or not b_boxes:
        return False
    box_a = _union_xywh(a_boxes)
    box_b = _union_xywh(b_boxes)
    gap_y = _vertical_gap(box_a, box_b)
    gap_x = _horizontal_gap(box_a, box_b)
    center_dx = abs(_center(box_a)[0] - _center(box_b)[0])
    union = _union_xywh([box_a, box_b])
    if union[2] > 330 or union[3] > 720:
        return False
    if center_dx <= 140 and 0 <= gap_y <= 260:
        return True
    if center_dx <= 90 and gap_x <= 80 and _overlap_1d(box_a[1], box_a[1] + box_a[3], box_b[1], box_b[1] + box_b[3]) > 0.4:
        return True
    return False


def _build_ownership(
    page_id: str,
    instances: list[dict[str, Any]],
    containers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ownership = []
    for instance in instances:
        best = None
        ibox = instance["bbox"]
        for container in containers:
            cbox = container["bbox"]
            containment = _intersection_area_xywh(ibox, cbox) / max(1, _area_xywh(ibox))
            overlap = _intersection_area_xywh(ibox, cbox) / max(1, min(_area_xywh(ibox), _area_xywh(cbox)))
            if instance["region_id"] in container.get("instance_region_ids", []):
                score = 1.0 + containment
            else:
                score = containment + overlap * 0.25
            if best is None or score > best[0]:
                best = (score, container, containment, overlap)
        if best is None:
            continue
        _score, container, containment, overlap = best
        if instance["region_id"] in container.get("instance_region_ids", []):
            relation = "inside" if containment >= 0.80 else "overlaps"
        elif containment >= 0.60:
            relation = "ambiguous"
        elif _distance_xywh(ibox, container["bbox"]) <= 30:
            relation = "nearby"
        else:
            relation = "excluded"
        reasons = []
        if relation in {"inside", "overlaps"}:
            reasons.append("container_claims_instance")
        if container["container_type"] == "speech_bubble" and instance["visual_role_hint"] == "speech":
            reasons.append("speech_role_agreement")
        if container["container_type"] == "sfx_decorative":
            reasons.append("preserved_decorative_obstacle")
        ownership.append(
            {
                "page_id": page_id,
                "instance_id": instance["instance_id"],
                "region_id": instance["region_id"],
                "container_id": container["container_id"],
                "ownership_type": relation,
                "overlap_ratio": round(overlap, 4),
                "containment_ratio": round(containment, 4),
                "confidence": round(min(1.0, 0.45 + containment * 0.5), 4),
                "reason_codes": reasons,
            }
        )
    return ownership


def _build_logical_blocks(
    page_id: str,
    instances: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_container: dict[str, list[dict[str, Any]]] = {}
    instance_by_id = {item["instance_id"]: item for item in instances}
    for link in ownership:
        if link["ownership_type"] not in {"inside", "overlaps"}:
            continue
        instance = instance_by_id.get(link["instance_id"])
        if instance:
            by_container.setdefault(link["container_id"], []).append(instance)
    container_by_id = {c["container_id"]: c for c in containers}
    blocks = []
    for idx, (container_id, items) in enumerate(by_container.items()):
        container = container_by_id.get(container_id, {})
        if container.get("container_type") not in {"speech_bubble", "caption", "sign/background_text_area"}:
            continue
        items.sort(key=lambda item: (item["bbox"][1], -item["bbox"][0]))
        source_text = " | ".join(str(item.get("ocr_text", "") or "") for item in items if str(item.get("ocr_text", "") or "").strip())
        transferred = [_transfer_evidence(_region_for_instance(regions, item["region_id"])) for item in items]
        transferred = [item for item in transferred if item]
        translated_instances = [item for item in items if str(item.get("translated_text", "") or "").strip()]
        ignored_transfers = [
            item
            for item in items
            if str(item.get("cleanup_mode", "") or "").startswith("transferred_to_")
        ]
        represented_once = bool(translated_instances) or not ignored_transfers
        if ignored_transfers and not translated_instances:
            represented_once = False
        blocks.append(
            {
                "block_id": f"b{idx:03d}",
                "page_id": page_id,
                "container_id": container_id,
                "instance_ids": [item["instance_id"] for item in items],
                "region_ids": [item["region_id"] for item in items],
                "source_text": source_text,
                "reading_order": [item["instance_id"] for item in items],
                "translation_unit_policy": "single_translation" if transferred or len(items) > 1 else "separate_translation",
                "render_policy": "single_render_box" if transferred else ("separate_render_boxes" if len(items) > 1 else "single_render_box"),
                "text_conservation": {
                    "source_phrase_count": len([item for item in items if str(item.get("ocr_text", "") or "").strip()]),
                    "represented_once": represented_once,
                    "transferred_evidence": transferred,
                    "missing": [] if represented_once else ["no translated anchor found for transferred text"],
                    "duplicates": [],
                },
            }
        )
    return blocks


def _build_render_constraints(
    page_id: str,
    regions: list[dict[str, Any]],
    containers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    constraints = []
    container_by_rid = {}
    obstacles = [
        c
        for c in containers
        if c.get("container_type") == "sfx_decorative"
    ]
    for container in containers:
        for rid in container.get("instance_region_ids", []):
            container_by_rid[rid] = container
    for region in regions:
        rid = str(region.get("region_id", "") or "")
        final_box = _xyxy(region.get("final_render_bbox"))
        if not rid or final_box is None:
            continue
        container = container_by_rid.get(rid)
        if not container:
            continue
        allowed = _xyxy_from_xywh(container["bbox"])
        outside = _outside_ratio(final_box, allowed)
        source_box = _xyxy_from_xywh(_bbox_xywh(region.get("bbox")))
        cleanup_box = _xyxy(region.get("cleanup_mask_bbox"))
        obstacle_hits = []
        introduced_hits = []
        for obstacle in obstacles:
            if rid in obstacle.get("instance_region_ids", []):
                continue
            obox = _xyxy_from_xywh(obstacle["bbox"])
            final_overlap = _overlap_ratio_xyxy(final_box, obox)
            if final_overlap <= 0.08:
                continue
            obstacle_hits.append(obstacle["container_id"])
            inherited = _overlap_ratio_xyxy(source_box, obox) > 0.08 or (cleanup_box is not None and _overlap_ratio_xyxy(cleanup_box, obox) > 0.08)
            if not inherited:
                introduced_hits.append(obstacle["container_id"])
        status = "satisfied"
        reasons = []
        if outside >= 0.20:
            status = "violates_container"
            reasons.append("final_render_bbox_outside_inferred_container")
        if introduced_hits:
            status = "violates_obstacle"
            reasons.append("final_render_bbox_newly_overlaps_preserved_obstacle")
        constraints.append(
            {
                "page_id": page_id,
                "region_id": rid,
                "container_id": container["container_id"],
                "allowed_bbox": list(allowed),
                "current_final_render_bbox": list(final_box),
                "source_bbox": list(source_box),
                "cleanup_mask_bbox": list(cleanup_box) if cleanup_box else None,
                "outside_allowed_ratio": round(outside, 4),
                "obstacle_container_ids": obstacle_hits,
                "newly_introduced_obstacle_container_ids": introduced_hits,
                "fit_policy": "diagnostic_only",
                "constraint_status": status,
                "reason_codes": reasons,
            }
        )
    return constraints


def _reference_case_results(
    page_id: str,
    instances: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _ = page_id, instances, containers, ownership, blocks, constraints
    return []


def _case_result(page_id: str, case: str, ok: bool, result: str, evidence: Any) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "case": case,
        "result": result if ok else "partially" if result == "partially" else "not_explained",
        "confidence": "medium" if ok else "low",
        "evidence": evidence,
    }


def _write_container_overlay(
    image,
    containers: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if Image is None or ImageDraw is None:
        return
    if image is None:
        return
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _overlay_font()
    colors = {
        "speech_bubble": (40, 110, 255),
        "caption": (230, 190, 20),
        "sign/background_text_area": (0, 180, 200),
        "sfx_decorative": (220, 30, 190),
        "unknown": (150, 150, 150),
    }
    draw.rectangle((8, 8, 760, 82), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((16, 14), "Text-area diagnostic overlay: blue=speech, yellow=caption, cyan=background, magenta=SFX, orange=instance, green=render, red=violation", fill=(0, 0, 0), font=font)
    draw.text((16, 42), "Diagnostic only. No production route/render behavior consumed this output.", fill=(0, 0, 0), font=font)
    for container in containers:
        box = _xyxy_from_xywh(container["bbox"])
        color = colors.get(container.get("container_type"), (150, 150, 150))
        width = 4 if container.get("confidence", 0) >= 0.7 else 2
        draw.rectangle(box, outline=color, width=width)
        label = f"{container['container_id']} {container['container_type']} {container.get('confidence', 0):.2f}"
        _draw_label(draw, box[0], box[1] - 18, label, font, fill=(255, 255, 230))
    for instance in instances:
        box = _xyxy_from_xywh(instance["bbox"])
        draw.rectangle(box, outline=(255, 140, 0), width=2)
        _draw_label(draw, box[0], box[3] + 1, f"{instance['region_id']} {instance['visual_role_hint']}", font, fill=(255, 245, 220))
    for constraint in constraints:
        final_box = tuple(constraint["current_final_render_bbox"])
        status = constraint.get("constraint_status")
        color = (230, 0, 0) if status != "satisfied" else (0, 180, 0)
        draw.rectangle(final_box, outline=color, width=3)
        if status != "satisfied":
            _draw_label(draw, final_box[0], final_box[1] - 18, f"{constraint['region_id']} {status}", font, fill=(255, 220, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _write_compact_container_overlay(
    image,
    containers: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    output_path: Path,
) -> None:
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _overlay_font(12)
    small_font = _overlay_font(10)
    ownership_by_rid = {item.get("region_id"): item for item in ownership}
    constraints_by_rid: dict[str, list[dict[str, Any]]] = {}
    for constraint in constraints:
        if constraint.get("constraint_status") != "satisfied":
            constraints_by_rid.setdefault(str(constraint.get("region_id") or ""), []).append(constraint)

    _draw_compact_legend(draw, font, "Legend: blue speech | cyan caption/sign | magenta SFX | yellow instance | red warn")
    for container in containers:
        box = _xyxy_from_xywh(container.get("bbox"))
        color = _container_color(container)
        width = 3 if float(container.get("confidence", 0.0) or 0.0) >= 0.7 else 2
        draw.rectangle(box, outline=color, width=width)
        if container.get("container_type") == "sfx_decorative":
            _draw_corner_marker(draw, box, color)
        label = f"{container.get('container_id')} {_compact_container_type(container.get('container_type'))}"
        if _container_ambiguous(container):
            label += " amb"
        _draw_small_label(draw, box[0], box[1] - 14, label, font, fill=(255, 255, 230, 220))

    for instance in instances:
        box = _xyxy_from_xywh(instance.get("bbox"))
        rid = str(instance.get("region_id") or "")
        link = ownership_by_rid.get(rid) or {}
        warn = rid in constraints_by_rid
        color = (255, 208, 0, 225) if not warn else (255, 80, 20, 245)
        draw.rectangle(box, outline=color, width=2)
        label = f"{rid}->{link.get('container_id', '?')}"
        relation = str(link.get("ownership_type") or "")
        if relation and relation not in {"inside"}:
            label += f" {relation[:3]}"
        _draw_small_label(draw, box[0], box[3] + 1, label, small_font, fill=(255, 255, 210, 210))

    for constraint in constraints:
        if constraint.get("constraint_status") == "satisfied":
            continue
        final_box = _xyxy(constraint.get("current_final_render_bbox"))
        if not final_box:
            continue
        draw.rectangle(final_box, outline=(255, 35, 20, 245), width=4)
        label = _compact_warning_label(constraint)
        _draw_small_label(draw, final_box[0], final_box[1] - 14, label, font, fill=(255, 220, 190, 235))

    canvas = Image.alpha_composite(canvas, layer).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _write_warning_container_overlay(
    image,
    containers: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    output_path: Path,
) -> int:
    warning_constraints = [item for item in constraints if item.get("constraint_status") != "satisfied"]
    ambiguous_containers = [item for item in containers if _container_ambiguous(item)]
    warning_count = len(warning_constraints) + len(ambiguous_containers)
    if warning_count <= 0:
        return 0
    canvas = image.convert("RGBA")
    veil = Image.new("RGBA", canvas.size, (255, 255, 255, 70))
    canvas = Image.alpha_composite(canvas, veil)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _overlay_font(12)
    instance_by_rid = {item.get("region_id"): item for item in instances}
    container_by_id = {item.get("container_id"): item for item in containers}

    _draw_compact_legend(draw, font, "Warnings: red constraint | orange ambiguous | magenta SFX obstacle")
    for container in containers:
        if container.get("container_type") == "sfx_decorative" or _container_ambiguous(container):
            box = _xyxy_from_xywh(container.get("bbox"))
            color = (220, 30, 190, 230) if container.get("container_type") == "sfx_decorative" else (255, 135, 0, 235)
            draw.rectangle(box, outline=color, width=3)
            _draw_small_label(draw, box[0], box[1] - 14, f"{container.get('container_id')} {_compact_container_type(container.get('container_type'))}", font, fill=(255, 235, 190, 230))

    for constraint in warning_constraints:
        final_box = _xyxy(constraint.get("current_final_render_bbox"))
        if not final_box:
            continue
        container = container_by_id.get(constraint.get("container_id")) or {}
        cbox = _xyxy_from_xywh(container.get("bbox")) if container else None
        if cbox:
            draw.rectangle(cbox, outline=(40, 110, 255, 225), width=2)
        draw.rectangle(final_box, outline=(255, 20, 20, 245), width=5)
        rid = str(constraint.get("region_id") or "")
        instance = instance_by_rid.get(rid)
        if instance:
            ibox = _xyxy_from_xywh(instance.get("bbox"))
            draw.rectangle(ibox, outline=(255, 208, 0, 235), width=2)
        _draw_small_label(draw, final_box[0], final_box[1] - 16, _compact_warning_label(constraint), font, fill=(255, 215, 190, 240))

    canvas = Image.alpha_composite(canvas, layer).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return warning_count


def _write_warning_crops(
    image,
    containers: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    warning_constraints = [item for item in constraints if item.get("constraint_status") != "satisfied"]
    if not warning_constraints:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    container_by_id = {item.get("container_id"): item for item in containers}
    paths = []
    for constraint in warning_constraints:
        final_box = _xyxy(constraint.get("current_final_render_bbox"))
        if not final_box:
            continue
        container = container_by_id.get(constraint.get("container_id")) or {}
        cbox = _xyxy_from_xywh(container.get("bbox")) if container else None
        crop_box = _expand_xyxy(_union_xyxy([box for box in (final_box, cbox) if box]), 90, image.size)
        crop = image.crop(crop_box).convert("RGBA")
        layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = _overlay_font(12)
        if cbox:
            draw.rectangle(_shift_xyxy(cbox, crop_box), outline=(40, 110, 255, 230), width=3)
        draw.rectangle(_shift_xyxy(final_box, crop_box), outline=(255, 20, 20, 245), width=4)
        _draw_small_label(draw, 4, 4, _compact_warning_label(constraint), font, fill=(255, 220, 190, 240))
        crop = Image.alpha_composite(crop, layer).convert("RGB")
        rid = str(constraint.get("region_id") or "region")
        status = str(constraint.get("constraint_status") or "warning")
        path = output_dir / f"{rid}_{status}.jpg"
        crop.save(path, quality=92)
        paths.append(str(path))
    return paths


def _write_case_summary(
    path: Path,
    page_id: str,
    containers: list[dict[str, Any]],
    instances: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    reference_results: list[dict[str, Any]],
    route_suggestions: list[dict[str, Any]],
    route_suggestion_summary: dict[str, Any],
    render_plan_suggestions: list[dict[str, Any]],
    render_plan_suggestion_summary: dict[str, Any],
    overlay_paths: dict[str, Any] | None = None,
) -> None:
    lines = [f"# Page {page_id} Text-Area Diagnostics", ""]
    lines.append(f"- text instances: {len(instances)}")
    lines.append(f"- containers: {len(containers)}")
    lines.append(f"- ownership links: {len(ownership)}")
    lines.append(f"- logical text blocks: {len(blocks)}")
    lines.append(f"- render constraints: {len(constraints)}")
    if overlay_paths:
        lines.append(f"- compact overlay: {overlay_paths.get('container_overlay_compact')}")
        if overlay_paths.get("container_overlay_warnings"):
            lines.append(f"- warning overlay: {overlay_paths.get('container_overlay_warnings')}")
        if overlay_paths.get("warning_crops"):
            lines.append(f"- warning crops: {len(overlay_paths.get('warning_crops') or [])}")
    lines.append("")
    lines.append("## Reference Cases")
    for item in reference_results:
        lines.append(f"- {item['case']}: {item['result']} ({item['confidence']})")
    lines.append("")
    lines.append("## Route Advisor Suggestions")
    lines.append(f"- total: {route_suggestion_summary.get('total', len(route_suggestions))}")
    by_type = route_suggestion_summary.get("by_type", {}) or {}
    if by_type:
        for key, count in sorted(by_type.items()):
            lines.append(f"- {key}: {count}")
    if route_suggestions:
        lines.append("")
        for item in route_suggestions:
            lines.append(
                f"- {item.get('region_id', '?')}: {item.get('suggestion_type')} "
                f"conf={item.get('confidence')} current={item.get('current_semantic_class')}/"
                f"{item.get('current_cleanup_mode')} suggested={item.get('suggested_semantic_class')}/"
                f"{item.get('suggested_cleanup_mode')}"
            )
    lines.append("")
    lines.append("## Render Planner Suggestions")
    lines.append(f"- total: {render_plan_suggestion_summary.get('total', len(render_plan_suggestions))}")
    render_by_type = render_plan_suggestion_summary.get("by_type", {}) or {}
    if render_by_type:
        for key, count in sorted(render_by_type.items()):
            lines.append(f"- {key}: {count}")
    if render_plan_suggestions:
        lines.append("")
        for item in render_plan_suggestions:
            lines.append(
                f"- {item.get('region_id', '?')}: {item.get('suggestion_type')} "
                f"severity={item.get('severity')} confidence={item.get('confidence')} "
                f"action={item.get('proposed_action')}"
            )
    lines.append("")
    lines.append("## Containers")
    for container in containers:
        lines.append(
            f"- {container['container_id']} {container['container_type']} "
            f"conf={container['confidence']:.2f} regions={','.join(container.get('instance_region_ids', []))} "
            f"bbox={container['bbox']}"
        )
    lines.append("")
    lines.append("## Constraint Findings")
    for constraint in constraints:
        if constraint["constraint_status"] != "satisfied":
            lines.append(
                f"- {constraint['region_id']}: {constraint['constraint_status']} "
                f"outside={constraint['outside_allowed_ratio']} obstacles={constraint['newly_introduced_obstacle_container_ids']}"
            )
    if not any(c["constraint_status"] != "satisfied" for c in constraints):
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reference_summary(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# Reference Case Results", "", "| page | case | result | confidence |", "|---|---|---|---|"]
    for item in results:
        lines.append(f"| {item['page_id']} | {item['case']} | {item['result']} | {item['confidence']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _visual_evidence(region: dict[str, Any], image, bbox: list[int], page_size: tuple[int, int]) -> dict[str, Any]:
    stats = _luma_stats(image, bbox)
    padded_stats = _luma_stats(image, _expand_xywh(bbox, 36, page_size))
    reason = str(region.get("classification_reason") or (region.get("render", {}) or {}).get("classification_reason") or "").strip()
    cleanup_mode = str(region.get("cleanup_mode") or (region.get("render", {}) or {}).get("cleanup_mode") or "").strip()
    y_mid = bbox[1] + bbox[3] / 2.0
    topish = y_mid <= max(1, page_size[1]) * 0.28
    reason_codes = []
    if _bright_bubble_like(padded_stats):
        reason_codes.append("bright_bubble_like_context")
    if stats["dark_ratio"] >= 0.18 and not _bright_bubble_like(padded_stats):
        reason_codes.append("art_or_nonuniform_context")
    if topish and (reason in CAPTION_REASONS or str(region.get("semantic_class")) == "background_text"):
        reason_codes.append("top_caption_band")
    if cleanup_mode == "preserve" or reason in DECORATIVE_REASONS:
        reason_codes.append("preserved_or_decorative_route")
    if reason in SPEECH_REASONS:
        reason_codes.append("speech_ownership_or_recovery_reason")
    return {
        "mean_luma": stats["mean_luma"],
        "bright_ratio": stats["bright_ratio"],
        "dark_ratio": stats["dark_ratio"],
        "local_uniformity": stats["uniformity"],
        "padded_mean_luma": padded_stats["mean_luma"],
        "padded_bright_ratio": padded_stats["bright_ratio"],
        "padded_dark_ratio": padded_stats["dark_ratio"],
        "enclosure_score": _enclosure_score(padded_stats),
        "bubble_boundary_evidence": _bright_bubble_like(padded_stats),
        "art_context_score": _art_context_score(stats, padded_stats),
        "caption_band_score": 0.85 if topish else 0.0,
        "sfx_decorative_score": 0.90 if cleanup_mode == "preserve" or reason in DECORATIVE_REASONS else 0.0,
        "nearby_preserved_sfx_ids": [],
        "reason_codes": reason_codes,
    }


def _visual_role_hint(region: dict[str, Any], evidence: dict[str, Any], page_size: tuple[int, int]) -> str:
    semantic = str(region.get("semantic_class") or "").strip()
    reason = str(region.get("classification_reason") or (region.get("render", {}) or {}).get("classification_reason") or "").strip()
    cleanup = str(region.get("cleanup_mode") or (region.get("render", {}) or {}).get("cleanup_mode") or "").strip()
    if reason in CAPTION_REASONS:
        return "caption"
    if reason == "bubble_contained_short_laugh_speech" and evidence.get("bubble_boundary_evidence"):
        return "speech"
    if reason in DECORATIVE_REASONS or cleanup == "preserve" or semantic in {"decorative_text", "sfx"}:
        return "sfx" if "sfx" in reason or semantic == "sfx" else "decorative"
    if semantic == "speech_bubble":
        return "speech"
    if semantic == "background_text":
        y_mid = _bbox_xywh(region.get("bbox"))[1] + _bbox_xywh(region.get("bbox"))[3] / 2.0
        if y_mid <= max(1, page_size[1]) * 0.28:
            return "caption"
        return "background"
    return "unknown"


def _container_confidence(container_type: str, instances: list[dict[str, Any]]) -> float:
    if not instances:
        return 0.0
    evidence = [item["visual_role_evidence"] for item in instances]
    if container_type == "speech_bubble":
        base = 0.58
        if any("bright_bubble_like_context" in ev["reason_codes"] for ev in evidence):
            base += 0.20
        if any("speech_ownership_or_recovery_reason" in ev["reason_codes"] for ev in evidence):
            base += 0.18
        return round(min(0.96, base), 4)
    if container_type == "caption":
        return 0.86
    if container_type == "sfx_decorative":
        return 0.90
    if container_type == "sign/background_text_area":
        return 0.66
    return 0.35


def _combine_evidence(instances: list[dict[str, Any]]) -> dict[str, Any]:
    if not instances:
        return {}
    keys = ["mean_luma", "bright_ratio", "dark_ratio", "local_uniformity", "enclosure_score", "art_context_score", "caption_band_score", "sfx_decorative_score"]
    combined = {}
    for key in keys:
        vals = [float(item["visual_role_evidence"].get(key, 0.0) or 0.0) for item in instances]
        combined[key] = round(sum(vals) / max(1, len(vals)), 4)
    reason_codes = []
    for item in instances:
        reason_codes.extend(item["visual_role_evidence"].get("reason_codes", []))
    combined["reason_codes"] = sorted(set(reason_codes))
    return combined


def _container_failure_modes(container_type: str, instances: list[dict[str, Any]], bbox: list[int]) -> list[str]:
    modes = []
    if container_type == "speech_bubble" and len(instances) > 1:
        modes.append("multi_instance_container")
    if container_type == "speech_bubble" and not any("bright_bubble_like_context" in item["visual_role_evidence"].get("reason_codes", []) for item in instances):
        modes.append("weak_bubble_boundary_evidence")
    if bbox[2] > 420 or bbox[3] > 760:
        modes.append("possible_merged_container")
    return modes


def _transfer_evidence(region: dict[str, Any]) -> dict[str, Any] | None:
    render = region.get("render", {}) or {}
    transfer_to = (
        render.get("bubble_local_transfer_to_region_id")
        or render.get("speech_text_conservation_transfer_to_region_id")
    )
    if not transfer_to:
        return None
    return {
        "region_id": region.get("region_id"),
        "transfer_to_region_id": transfer_to,
        "transfer_text": render.get("bubble_local_transfer_text") or render.get("speech_text_conservation_transfer_text"),
        "reason": render.get("classification_reason") or region.get("classification_reason"),
    }


def _constraint_for(constraints: list[dict[str, Any]], rid: str) -> dict[str, Any] | None:
    return next((item for item in constraints if item.get("region_id") == rid), None)


def _ownership_for(ownership: list[dict[str, Any]], rid: str) -> dict[str, Any] | None:
    return next((item for item in ownership if item.get("region_id") == rid), None)


def _same_container(ownership: list[dict[str, Any]], a: str, b: str) -> bool:
    ca = _ownership_for(ownership, a)
    cb = _ownership_for(ownership, b)
    return bool(ca and cb and ca.get("container_id") == cb.get("container_id"))


def _block_with_regions(blocks: list[dict[str, Any]], rids: set[str]) -> dict[str, Any] | None:
    for block in blocks:
        if rids.issubset(set(block.get("region_ids", []))):
            return block
    return None


def _region_for_instance(regions: list[dict[str, Any]], rid: str) -> dict[str, Any]:
    return next((region for region in regions if str(region.get("region_id", "") or "") == rid), {})


def _luma_stats(image, bbox: list[int]) -> dict[str, float]:
    if image is None or ImageStat is None:
        return {"mean_luma": 0.0, "bright_ratio": 0.0, "dark_ratio": 0.0, "uniformity": 0.0}
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return {"mean_luma": 0.0, "bright_ratio": 0.0, "dark_ratio": 0.0, "uniformity": 0.0}
    crop = image.crop((x, y, x + w, y + h)).convert("L")
    stat = ImageStat.Stat(crop)
    hist = crop.histogram()
    total = max(1, sum(hist))
    mean = float(stat.mean[0]) if stat.mean else 0.0
    std = float(stat.stddev[0]) if stat.stddev else 0.0
    return {
        "mean_luma": round(mean, 4),
        "bright_ratio": round(sum(hist[230:]) / total, 4),
        "dark_ratio": round(sum(hist[:80]) / total, 4),
        "uniformity": round(max(0.0, min(1.0, 1.0 - (std / 128.0))), 4),
    }


def _bright_bubble_like(stats: dict[str, float]) -> bool:
    return (
        stats.get("mean_luma", 0.0) >= 225.0
        and stats.get("bright_ratio", 0.0) >= 0.78
        and stats.get("dark_ratio", 1.0) <= 0.12
    )


def _enclosure_score(stats: dict[str, float]) -> float:
    score = 0.0
    score += min(0.45, stats.get("bright_ratio", 0.0) * 0.45)
    score += min(0.35, max(0.0, (stats.get("mean_luma", 0.0) - 170.0) / 90.0) * 0.35)
    score += min(0.20, stats.get("uniformity", 0.0) * 0.20)
    return round(min(1.0, score), 4)


def _art_context_score(stats: dict[str, float], padded: dict[str, float]) -> float:
    score = 0.0
    score += min(0.40, stats.get("dark_ratio", 0.0) * 1.2)
    score += min(0.30, max(0.0, 1.0 - padded.get("uniformity", 0.0)) * 0.45)
    if not _bright_bubble_like(padded):
        score += 0.25
    return round(min(1.0, score), 4)


def _bbox_xywh(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return [0, 0, 1, 1]
    try:
        x, y, w, h = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return [0, 0, 1, 1]
    return [max(0, x), max(0, y), max(1, w), max(1, h)]


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


def _xyxy_to_xywh(value: Any) -> list[int] | None:
    box = _xyxy(value)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _xyxy_from_xywh(box: list[int]) -> tuple[int, int, int, int]:
    x, y, w, h = _bbox_xywh(box)
    return (x, y, x + w, y + h)


def _bbox_polygon(box: list[int]) -> list[list[int]]:
    x, y, w, h = _bbox_xywh(box)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _union_xywh(boxes: list[list[int]]) -> list[int]:
    valid = [_bbox_xywh(box) for box in boxes if box]
    if not valid:
        return [0, 0, 1, 1]
    x0 = min(box[0] for box in valid)
    y0 = min(box[1] for box in valid)
    x1 = max(box[0] + box[2] for box in valid)
    y1 = max(box[1] + box[3] for box in valid)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _expand_xywh(box: list[int], pad: int, page_size: tuple[int, int]) -> list[int]:
    x, y, w, h = _bbox_xywh(box)
    max_w, max_h = page_size
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(max_w, x + w + pad)
    y1 = min(max_h, y + h + pad)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _area_xywh(box: list[int]) -> int:
    _x, _y, w, h = _bbox_xywh(box)
    return max(0, w) * max(0, h)


def _intersection_area_xywh(a: list[int], b: list[int]) -> int:
    ax0, ay0, ax1, ay1 = _xyxy_from_xywh(a)
    bx0, by0, bx1, by1 = _xyxy_from_xywh(b)
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))


def _overlap_ratio_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / max(1, min(a_area, b_area))


def _outside_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    inter = max(0, min(ix1, ox1) - max(ix0, ox0)) * max(0, min(iy1, oy1) - max(iy0, oy0))
    area = max(1, (ix1 - ix0) * (iy1 - iy0))
    return max(0.0, 1.0 - (inter / area))


def _distance_xywh(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = _xyxy_from_xywh(a)
    bx0, by0, bx1, by1 = _xyxy_from_xywh(b)
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return math.sqrt(dx * dx + dy * dy)


def _center(box: list[int]) -> tuple[float, float]:
    x, y, w, h = _bbox_xywh(box)
    return x + w / 2.0, y + h / 2.0


def _vertical_gap(a: list[int], b: list[int]) -> int:
    ay0, ay1 = a[1], a[1] + a[3]
    by0, by1 = b[1], b[1] + b[3]
    return max(by0 - ay1, ay0 - by1, 0)


def _horizontal_gap(a: list[int], b: list[int]) -> int:
    ax0, ax1 = a[0], a[0] + a[2]
    bx0, bx1 = b[0], b[0] + b[2]
    return max(bx0 - ax1, ax0 - bx1, 0)


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / max(1, min(a1 - a0, b1 - b0))


def _page_size_from_regions(regions: list[dict[str, Any]]) -> tuple[int, int]:
    max_x = 1
    max_y = 1
    for region in regions:
        x, y, w, h = _bbox_xywh(region.get("bbox"))
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    return max_x, max_y


def _resolve_existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    return None


def _normalize_page_id(value: str) -> str:
    stem = os.path.splitext(os.path.basename(str(value).strip()))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits.zfill(3) if digits else stem


def _overlay_font(size: int = 14):
    if ImageFont is None:
        return None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        try:
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_label(draw, x: int, y: int, text: str, font, fill=(255, 255, 255)) -> None:
    if font is None:
        return
    x = max(0, int(x))
    y = max(0, int(y))
    width = min(900, max(40, len(text) * 8 + 8))
    draw.rectangle((x, y, x + width, y + 18), fill=fill, outline=(0, 0, 0))
    draw.text((x + 4, y + 2), text, fill=(0, 0, 0), font=font)


def _draw_small_label(draw, x: int, y: int, text: str, font, fill=(255, 255, 255, 225)) -> None:
    if font is None:
        return
    x = max(0, int(x))
    y = max(0, int(y))
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        width = max(34, right - left + 6)
        height = max(14, bottom - top + 4)
    except Exception:
        width = min(260, max(34, len(text) * 7 + 6))
        height = 15
    draw.rectangle((x, y, x + width, y + height), fill=fill, outline=(40, 40, 40, 210))
    draw.text((x + 3, y + 1), text, fill=(0, 0, 0, 255), font=font)


def _draw_compact_legend(draw, font, text: str) -> None:
    if font is None:
        return
    width = min(920, max(260, len(text) * 7 + 18))
    draw.rectangle((8, 8, width, 44), fill=(255, 255, 255, 225), outline=(0, 0, 0, 230))
    draw.text((16, 15), text, fill=(0, 0, 0, 255), font=font)


def _container_color(container: dict[str, Any]) -> tuple[int, int, int, int]:
    ctype = container.get("container_type")
    if ctype == "speech_bubble":
        return (40, 110, 255, 230)
    if ctype in {"caption", "sign/background_text_area"}:
        return (0, 185, 215, 230)
    if ctype == "sfx_decorative":
        return (220, 30, 190, 230)
    return (135, 135, 135, 220)


def _compact_container_type(value: Any) -> str:
    mapping = {
        "speech_bubble": "speech",
        "caption": "caption",
        "sign/background_text_area": "bg/sign",
        "sfx_decorative": "sfx",
        "unknown": "unknown",
    }
    return mapping.get(str(value or "unknown"), str(value or "unknown")[:8])


def _container_ambiguous(container: dict[str, Any]) -> bool:
    failure_modes = set(container.get("failure_modes", []) or [])
    return (
        container.get("container_type") == "unknown"
        or float(container.get("confidence", 0.0) or 0.0) < 0.5
        or bool(failure_modes & {"weak_bubble_boundary_evidence", "possible_merged_container"})
    )


def _compact_warning_label(constraint: dict[str, Any]) -> str:
    reasons = constraint.get("reason_codes", []) or []
    if "final_render_bbox_newly_overlaps_preserved_obstacle" in reasons:
        kind = "obstacle"
    elif "final_render_bbox_outside_inferred_container" in reasons:
        kind = "outside"
    else:
        kind = str(constraint.get("constraint_status") or "warn")
    return f"{constraint.get('region_id')} {kind}"


def _draw_corner_marker(draw, box: tuple[int, int, int, int], color: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    size = min(16, max(6, (x1 - x0) // 8), max(6, (y1 - y0) // 8))
    draw.line((x0, y0, x0 + size, y0), fill=color, width=4)
    draw.line((x0, y0, x0, y0 + size), fill=color, width=4)
    draw.line((x1, y0, x1 - size, y0), fill=color, width=4)
    draw.line((x1, y0, x1, y0 + size), fill=color, width=4)


def _union_xyxy(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    valid = [box for box in boxes if box]
    if not valid:
        return (0, 0, 1, 1)
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def _expand_xyxy(box: tuple[int, int, int, int], pad: int, page_size: tuple[int, int]) -> tuple[int, int, int, int]:
    max_w, max_h = page_size
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(max_w, box[2] + pad),
        min(max_h, box[3] + pad),
    )


def _shift_xyxy(box: tuple[int, int, int, int], origin: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ox, oy, _x1, _y1 = origin
    return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
