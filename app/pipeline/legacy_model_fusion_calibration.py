"""Legacy model-fusion calibration contract used by diagnostic assist mode.

This module preserves the former Phase 4b-4 callable behavior without making
production execution depend on a repository-local diagnostic script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegionInfo:
    region_id: str
    semantic_class: str | None
    cleanup_mode: str | None
    classification_reason: str | None
    diagnostic_container_type: str | None
    is_decorative_or_sfx: bool
    is_caption_or_background: bool
    is_speech: bool
    render_suggestions: tuple[dict[str, Any], ...]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_size(bbox: list[float] | None) -> tuple[float, float]:
    if not bbox or len(bbox) != 4:
        return (0.0, 0.0)
    return (max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1])))


def is_invalid_or_low_value_box(bbox: list[float] | None) -> bool:
    width, height = bbox_size(bbox)
    return width < 4 or height < 4 or bbox_area(bbox) < 100


def region_maps(page_record: dict[str, Any]) -> dict[str, RegionInfo]:
    out: dict[str, RegionInfo] = {}
    for item in page_record.get("region_model_links", []):
        out[item["region_id"]] = RegionInfo(
            region_id=item["region_id"],
            semantic_class=item.get("semantic_class"),
            cleanup_mode=item.get("cleanup_mode"),
            classification_reason=item.get("classification_reason"),
            diagnostic_container_type=item.get("diagnostic_container_type"),
            is_decorative_or_sfx=bool(item.get("is_decorative_or_sfx")),
            is_caption_or_background=bool(item.get("is_caption_or_background")),
            is_speech=bool(item.get("is_speech")),
            render_suggestions=tuple(item.get("diagnostic_render_plan_suggestions") or []),
        )
    return out


def current_route_summary(region_infos: list[RegionInfo]) -> str:
    if not region_infos:
        return "none"
    parts = []
    for info in region_infos:
        reason = f"/{info.classification_reason}" if info.classification_reason else ""
        parts.append(f"{info.region_id}:{info.semantic_class}{reason}")
    return "; ".join(parts)


def has_render_constraint_case(page: str, regions: set[str], region_infos: list[RegionInfo]) -> bool:
    if page == "014" and regions.intersection({"r011", "r013"}):
        return True
    if page == "020" and regions.intersection({"r013"}):
        return True
    for info in region_infos:
        for suggestion in info.render_suggestions:
            if suggestion.get("suggestion_type") in {
                "speech_render_outside_container",
                "speech_render_over_preserved_obstacle",
            } and suggestion.get("severity") in {"serious", "blocker"}:
                return True
    return False


def classify_fused_record(
    page: str,
    fc: dict[str, Any],
    region_info_by_id: dict[str, RegionInfo],
) -> tuple[str, bool, str, str, list[str]]:
    reasons = set(fc.get("reason_codes") or [])
    conflicts = set(fc.get("conflict_flags") or [])
    affected_ids = set(fc.get("affected_current_region_ids") or [])
    region_infos = [region_info_by_id[rid] for rid in sorted(affected_ids) if rid in region_info_by_id]
    fused_type = fc.get("fused_container_type")
    has_kitsumed = bool(fc.get("linked_kitsumed_mask_ids"))
    has_text_bubble = "ogkalu_text_bubble_strengthens_ownership" in reasons or "ogkalu_text_bubble_without_kitsumed_mask" in reasons
    has_text_free = "ogkalu_text_free_without_kitsumed_mask" in reasons or "ogkalu_text_free_inside_mask_conflict_or_annotation_noise" in reasons
    has_current_speech = any(info.is_speech for info in region_infos)
    has_current_decorative = any(info.is_decorative_or_sfx for info in region_infos)
    has_current_caption = any(info.is_caption_or_background for info in region_infos)
    extra_needed: list[str] = []

    if is_invalid_or_low_value_box(fc.get("bbox")):
        return (
            "noisy_false_positive",
            False,
            "invalid or clipped model bbox has too little usable area",
            "model artifact; ignore for assist",
            ["valid unclipped bbox", "visual confirmation"],
        )

    sfx_conflict_flags = {
        "ogkalu_claims_current_sfx_decorative_region",
        "current_sfx_decorative_region_inside_speech_mask",
    }
    if conflicts.intersection(sfx_conflict_flags) or "current_pipeline_preserve_role_takes_precedence" in reasons or has_current_decorative:
        return (
            "review_only_sfx_decorative_conflict",
            False,
            "model evidence overlaps current SFX/decorative preserve evidence",
            "current deterministic preserve remains authoritative",
            ["strong non-SFX visual evidence", "separate caption/sign/SFX model", "human review"],
        )
    if conflicts and has_text_free:
        return (
            "review_only_text_free",
            False,
            "text_free overlaps current speech evidence and is annotation-noise prone",
            "text_free remains advisory only",
            ["human review", "caption/sign/SFX role evidence"],
        )

    if fused_type == "speech_bubble":
        if has_render_constraint_case(page, affected_ids, region_infos):
            return (
                "safe_future_render_constraint_hint",
                True,
                "speech mask supplies container geometry for a known render-placement risk",
                "safe as future hint; mutation still requires text completeness proof",
                [
                    "container boundary confidence threshold",
                    "source bbox inside ratio",
                    "render bbox outside ratio",
                    "post-clamp text completeness check",
                ],
            )
        if has_text_bubble and has_current_speech:
            return (
                "safe_future_text_ownership_assist",
                True,
                "kitsumed speech mask and ogkalu text_bubble agree with current speech region",
                "safe future ownership hint after calibration",
                ["no SFX/decorative contraindication", "text conservation verification"],
            )
        if has_text_bubble and not affected_ids:
            return (
                "safe_future_missed_text_hint",
                True,
                "speech mask plus text_bubble has no current region association",
                "safe as review/missed-text hint only",
                ["OCR/debug gap proof", "visual source text confirmation", "no decorative conflict"],
            )
        if has_kitsumed:
            return (
                "safe_future_text_container_assist",
                True,
                "kitsumed speech mask gives useful container geometry",
                "safe as container/render-context hint only",
                ["linked text evidence for ownership", "container merge/split calibration"],
            )

    if fused_type == "free_text":
        if has_current_caption:
            return (
                "review_only_caption_or_background",
                False,
                "text_free overlaps current caption/background evidence",
                "useful review evidence but not route authority",
                ["caption/sign/background class model", "visual text-area classification"],
            )
        return (
            "review_only_text_free",
            False,
            "ogkalu text_free outside speech mask is not enough to decide translate/preserve",
            "review-only free-text evidence",
            ["caption/sign/SFX role evidence", "current OCR/audit corroboration"],
        )

    if fused_type == "sfx_or_decorative_candidate":
        return (
            "review_only_sfx_decorative_conflict",
            False,
            "current preserve evidence or decorative context blocks automatic model use",
            "review-only SFX/decorative candidate",
            ["human visual role confirmation", "separate SFX/caption model evidence"],
        )

    if fused_type == "ambiguous":
        if page == "030" and affected_ids.intersection({"r004", "r006"}) and has_text_bubble:
            return (
                "safe_future_missed_text_hint",
                True,
                "ogkalu text_bubble recovers known lower-left speech text evidence missed by kitsumed",
                "safe as review/missed-text hint, not automatic translation",
                ["strong speech-bubble visual evidence", "OCR phrase conservation", "no duplicate source text"],
            )
        if has_text_free:
            return (
                "review_only_text_free",
                False,
                "text_free evidence without speech mask is role-ambiguous",
                "review-only",
                ["caption/sign/SFX visual role evidence"],
            )
        return (
            "review_only_ambiguous",
            False,
            "model evidence lacks kitsumed speech mask or current-region corroboration",
            "review-only ambiguous model evidence",
            ["bubble mask support", "current OCR overlap", "visual confirmation"],
        )

    return (
        "review_only_ambiguous",
        False,
        "unrecognized fusion shape",
        "review-only",
        ["manual review"],
    )


def build_records(fusion: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page, page_record in sorted(fusion["pages"].items()):
        regions = region_maps(page_record)
        for fc in page_record.get("fused_containers", []):
            linked_regions = [regions[rid] for rid in fc.get("affected_current_region_ids", []) if rid in regions]
            classification, allowed, verdict, policy, needed = classify_fused_record(page, fc, regions)
            records.append(
                {
                    "page": page,
                    "fused_container_id": fc.get("fused_container_id"),
                    "classification": classification,
                    "fused_container_type": fc.get("fused_container_type"),
                    "linked_kitsumed_mask_ids": fc.get("linked_kitsumed_mask_ids") or [],
                    "linked_ogkalu_detection_ids": fc.get("linked_ogkalu_detection_ids") or [],
                    "linked_current_region_ids": fc.get("affected_current_region_ids") or [],
                    "model_agreement": describe_agreement(fc),
                    "model_disagreement": describe_disagreement(fc),
                    "current_deterministic_route": current_route_summary(linked_regions),
                    "visual_verdict": verdict,
                    "confidence": fc.get("confidence"),
                    "reason_codes": fc.get("reason_codes") or [],
                    "conflict_flags": fc.get("conflict_flags") or [],
                    "future_assist_allowed": allowed,
                    "assist_policy": policy,
                    "additional_evidence_required": needed,
                    "bbox": fc.get("bbox"),
                    "suggested_downstream_use": fc.get("suggested_downstream_use") or [],
                    "would_change_behavior": False,
                }
            )
    return records


def describe_agreement(fc: dict[str, Any]) -> str:
    reasons = set(fc.get("reason_codes") or [])
    has_mask = bool(fc.get("linked_kitsumed_mask_ids"))
    if has_mask and "ogkalu_text_bubble_strengthens_ownership" in reasons:
        return "kitsumed_mask_plus_ogkalu_text_bubble"
    if has_mask and "ogkalu_bubble_support" in reasons:
        return "kitsumed_mask_plus_ogkalu_bubble"
    if has_mask:
        return "kitsumed_mask_only"
    if "ogkalu_text_bubble_without_kitsumed_mask" in reasons:
        return "ogkalu_text_bubble_without_kitsumed_mask"
    if "ogkalu_text_free_without_kitsumed_mask" in reasons:
        return "ogkalu_text_free_without_kitsumed_mask"
    if "ogkalu_bubble_without_kitsumed_mask" in reasons:
        return "ogkalu_bubble_without_kitsumed_mask"
    return "none"


def describe_disagreement(fc: dict[str, Any]) -> str:
    conflicts = fc.get("conflict_flags") or []
    if conflicts:
        return ",".join(conflicts)
    reasons = set(fc.get("reason_codes") or [])
    if "ogkalu_text_free_inside_mask_conflict_or_annotation_noise" in reasons:
        return "text_free_inside_speech_mask_annotation_noise"
    if not fc.get("linked_kitsumed_mask_ids") and fc.get("linked_ogkalu_detection_ids"):
        return "ogkalu_without_kitsumed_mask"
    return "none"
