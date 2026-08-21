"""Source-grounded render eligibility contracts.

This module decides whether translated text is sufficiently source-grounded to
enter final composition. It does not run OCR, routing, translation, cleanup, or
model inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence
import unicodedata

from app.pipeline.parent_execution_bundle import parent_execution_region_records


RENDER_ELIGIBILITY_CONTRACT_VERSION = "render_eligibility_source_grounding_v1"
UNSAFE_CLEANUP_RENDER_CONTRACT_VERSION = "render_eligibility_unsafe_cleanup_render_v1"

LOW_CONFIDENCE_THRESHOLD = 0.35
COMPACT_GLYPH_MAX_BBOX_AREA = 30_000
WEAK_SOURCE_MAX_SEMANTIC_CHARS = 4

RISKY_RECOVERY_MARKERS = {
    "caption_container_text_instance_recovery",
    "deterministic_top_band_caption_search",
    "deterministic_vertical_side_caption_search",
    "caption_component_recovery",
}

LOW_CONFIDENCE_TIER_MARKERS = {
    "low_confidence",
    "review_only",
    "text_free_review_only",
}

RAW_AUDIT_KEYS = {
    "mask",
    "foreground_mask",
    "erase_mask",
    "image",
    "source_image",
    "cleaned_image",
    "crop",
    "cleaned_crop",
}


class RenderEligibilityStatus(str, Enum):
    """Final composition eligibility status."""

    ELIGIBLE = "eligible"
    REVIEW_ALLOWED = "review_allowed"
    SUPPRESSED_SOURCE_UNGROUNDED = "suppressed_source_ungrounded"
    SUPPRESSED_UNSAFE_CLEANUP_RENDER = "suppressed_unsafe_cleanup_render"


@dataclass(frozen=True)
class RenderEligibilityDecision:
    """Audit-safe decision for one render region."""

    page_id: str
    region_id: str
    status: RenderEligibilityStatus
    reason: str = ""
    translated_text_present: bool = False
    source_text: str = ""
    translated_text: str = ""
    ocr_confidence: float | None = None
    risky_detection_source: str = ""
    hard_contradictions: list[str] = field(default_factory=list)
    preservation_reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "region_id": self.region_id,
            "parent_execution_bundle_id": str(self.evidence.get("parent_execution_bundle_id") or self.region_id or ""),
            "parent_logical_text_unit_id": str(self.evidence.get("parent_logical_text_unit_id") or self.region_id or ""),
            "text_block_root_id": str(self.evidence.get("text_block_root_id") or ""),
            "status": _enum_value(self.status),
            "reason": self.reason,
            "translated_text_present": self.translated_text_present,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "ocr_confidence": self.ocr_confidence,
            "risky_detection_source": self.risky_detection_source,
            "hard_contradictions": list(self.hard_contradictions),
            "preservation_reason": self.preservation_reason,
            "evidence": _json_safe(self.evidence),
        }


@dataclass(frozen=True)
class RenderEligibilityResult:
    """Page-level render eligibility contract."""

    page_id: str
    version: str
    decisions: list[RenderEligibilityDecision] = field(default_factory=list)
    decisions_by_region_id: dict[str, RenderEligibilityDecision] = field(default_factory=dict)
    suppressed_records: list[dict[str, Any]] = field(default_factory=list)
    review_allowed_records: list[dict[str, Any]] = field(default_factory=list)
    eligible_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def decision_for_region(self, region_id: str) -> RenderEligibilityDecision | None:
        return self.decisions_by_region_id.get(str(region_id or ""))

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "decisions": [decision.to_audit_dict() for decision in self.decisions],
            "decisions_by_region_id": {
                region_id: decision.to_audit_dict()
                for region_id, decision in sorted(self.decisions_by_region_id.items())
            },
            "suppressed_records": _json_safe(self.suppressed_records),
            "review_allowed_records": _json_safe(self.review_allowed_records),
            "eligible_records": _json_safe(self.eligible_records),
            "errors": list(self.errors),
            "summary": {
                "decision_count": len(self.decisions),
                "suppressed_count": len(self.suppressed_records),
                "review_allowed_count": len(self.review_allowed_records),
                "eligible_count": len(self.eligible_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
            },
        }


def build_render_eligibility_decisions(
    *,
    page_id: str,
    regions: Sequence[Mapping[str, Any]] | Any,
    source_glyph_masks: Any = None,
    cleanup_job_contracts: Any = None,
    cleanup_mask_contracts: Any = None,
    source_image_path: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> RenderEligibilityResult:
    """Build source-grounding render decisions from existing audit evidence."""

    source_records_by_region = _source_records_by_region(source_glyph_masks)
    cleanup_records_by_region = _cleanup_records_by_region(cleanup_job_contracts)
    mask_records_by_region = _mask_records_by_region(
        cleanup_mask_contracts,
        cleanup_records_by_region,
    )
    decisions: list[RenderEligibilityDecision] = []
    decisions_by_region_id: dict[str, RenderEligibilityDecision] = {}
    suppressed_records: list[dict[str, Any]] = []
    review_allowed_records: list[dict[str, Any]] = []
    eligible_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, region in enumerate(regions or []):
        if not isinstance(region, Mapping):
            errors.append(f"region index {index} is not mapping-like")
            continue
        try:
            decision = _decision_for_region(
                page_id=str(page_id or ""),
                region=region,
                source_records=source_records_by_region.get(_region_id(region), []),
                cleanup_records=cleanup_records_by_region.get(_region_id(region), []),
                mask_records=mask_records_by_region.get(_region_id(region), []),
                source_image_path=source_image_path,
                image_size=image_size,
            )
            decisions.append(decision)
            decisions_by_region_id[decision.region_id] = decision
            audit = decision.to_audit_dict()
            if decision.status == RenderEligibilityStatus.SUPPRESSED_SOURCE_UNGROUNDED:
                suppressed_records.append(audit)
            elif decision.status == RenderEligibilityStatus.REVIEW_ALLOWED:
                review_allowed_records.append(audit)
            else:
                eligible_records.append(audit)
        except Exception as exc:
            rid = _region_id(region) or f"region_{index}"
            errors.append(f"{rid}:{type(exc).__name__}: {exc}")
            decision = RenderEligibilityDecision(
                page_id=str(page_id or ""),
                region_id=rid,
                status=RenderEligibilityStatus.REVIEW_ALLOWED,
                reason="render_eligibility_error_review_degraded",
                evidence={"error": f"{type(exc).__name__}: {exc}"},
            )
            decisions.append(decision)
            decisions_by_region_id[decision.region_id] = decision
            review_allowed_records.append(decision.to_audit_dict())

    return RenderEligibilityResult(
        page_id=str(page_id or ""),
        version=RENDER_ELIGIBILITY_CONTRACT_VERSION,
        decisions=decisions,
        decisions_by_region_id=decisions_by_region_id,
        suppressed_records=suppressed_records,
        review_allowed_records=review_allowed_records,
        eligible_records=eligible_records,
        errors=errors,
    )


def build_unsafe_cleanup_render_decisions(
    *,
    page_id: str,
    regions: Sequence[Mapping[str, Any]] | Any,
    region_audit_fields: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | Any,
) -> RenderEligibilityResult:
    """Build post-proof unsafe cleanup render decisions.

    This consumes renderer proof/audit fields after pre-render source-erasure
    proof has run. It does not run OCR, route policy, cleanup, or mask logic.
    """

    audit_by_region = _region_audit_fields_by_region(region_audit_fields)
    regions_by_id = {
        _region_id(region): region
        for region in (regions or [])
        if isinstance(region, Mapping) and _region_id(region)
    }
    decisions: list[RenderEligibilityDecision] = []
    decisions_by_region_id: dict[str, RenderEligibilityDecision] = {}
    suppressed_records: list[dict[str, Any]] = []
    review_allowed_records: list[dict[str, Any]] = []
    eligible_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for region_id, audit in sorted(audit_by_region.items()):
        try:
            region = regions_by_id.get(region_id, {})
            decision = _unsafe_cleanup_decision_for_region(
                page_id=str(page_id or ""),
                region_id=region_id,
                region=region,
                audit=audit,
            )
        except Exception as exc:
            errors.append(f"{region_id}:{type(exc).__name__}: {exc}")
            decision = RenderEligibilityDecision(
                page_id=str(page_id or ""),
                region_id=region_id,
                status=RenderEligibilityStatus.REVIEW_ALLOWED,
                reason="unsafe_cleanup_render_decision_error_review_allowed",
                evidence={"error": f"{type(exc).__name__}: {exc}"},
            )
        decisions.append(decision)
        decisions_by_region_id[decision.region_id] = decision
        audit_record = decision.to_audit_dict()
        if decision.status == RenderEligibilityStatus.SUPPRESSED_UNSAFE_CLEANUP_RENDER:
            suppressed_records.append(audit_record)
        elif decision.status == RenderEligibilityStatus.REVIEW_ALLOWED:
            review_allowed_records.append(audit_record)
        else:
            eligible_records.append(audit_record)

    return RenderEligibilityResult(
        page_id=str(page_id or ""),
        version=UNSAFE_CLEANUP_RENDER_CONTRACT_VERSION,
        decisions=decisions,
        decisions_by_region_id=decisions_by_region_id,
        suppressed_records=suppressed_records,
        review_allowed_records=review_allowed_records,
        eligible_records=eligible_records,
        errors=errors,
    )


def build_render_eligibility_decisions_for_parent_bundles(
    *,
    page_id: str,
    parent_execution_bundles: Sequence[Any],
    source_glyph_masks: Any = None,
    cleanup_job_contracts: Any = None,
    cleanup_mask_contracts: Any = None,
    source_image_path: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> RenderEligibilityResult:
    """Build render eligibility decisions from finalized parent bundles."""

    return build_render_eligibility_decisions(
        page_id=page_id,
        regions=parent_execution_region_records(parent_execution_bundles),
        source_glyph_masks=source_glyph_masks,
        cleanup_job_contracts=cleanup_job_contracts,
        cleanup_mask_contracts=cleanup_mask_contracts,
        source_image_path=source_image_path,
        image_size=image_size,
    )


def _decision_for_region(
    *,
    page_id: str,
    region: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
    source_image_path: str | None,
    image_size: tuple[int, int] | None,
) -> RenderEligibilityDecision:
    rid = _region_id(region)
    render = _mapping_or_empty(region.get("render"))
    source_text = _text_from(region, render, keys=("ocr_text", "source_text", "text"))
    translated_text = _text_from(region, render, keys=("translation", "translated_text"))
    translated_meaningful = _meaningful_text(translated_text)
    confidence_values = _confidence_values(region, render)
    min_confidence = min(confidence_values) if confidence_values else None
    low_confidence = _low_confidence(region, render, confidence_values)
    semantic_class = _text_from(region, render, keys=("semantic_class", "type", "region_type"))
    cleanup_classes = _cleanup_classes(cleanup_records, mask_records)
    risky_source = _risky_detection_source(region, render)
    source_glyph_erasure_coverage_proven = _source_glyph_erasure_coverage_proven(source_records)
    source_erasure_unproven = _source_erasure_required_but_unproven(
        source_records=source_records,
        cleanup_records=cleanup_records,
        mask_records=mask_records,
    )
    risky_caption_background_recovery = _risky_caption_background_recovery_target(
        region=region,
        render=render,
        semantic_class=semantic_class,
        cleanup_classes=cleanup_classes,
        risky_source=risky_source,
    )
    hard_contradictions = _hard_contradictions(
        region=region,
        render=render,
        source_text=source_text,
        translated_text=translated_text,
        source_records=source_records,
        cleanup_records=cleanup_records,
        mask_records=mask_records,
        risky_source=risky_source,
        source_erasure_unproven=source_erasure_unproven,
        risky_caption_background_recovery=risky_caption_background_recovery,
    )
    parent_execution_bundle_id = str(
        _get(region, "parent_execution_bundle_id")
        or _get(render, "parent_execution_bundle_id")
        or rid
        or ""
    )
    parent_logical_text_unit_id = str(
        _get(region, "parent_logical_text_unit_id")
        or _get(render, "parent_logical_text_unit_id")
        or rid
        or ""
    )
    text_block_root_id = str(
        _get(region, "text_block_root_id")
        or _get(render, "text_block_root_id")
        or ""
    )

    evidence = {
        "parent_execution_bundle_id": parent_execution_bundle_id,
        "parent_logical_text_unit_id": parent_logical_text_unit_id,
        "text_block_root_id": text_block_root_id,
        "semantic_class": semantic_class,
        "cleanup_classes": cleanup_classes,
        "text_area_detection_source": _text_from(region, render, keys=("text_area_detection_source", "detection_source")),
        "text_area_confidence_tier": _text_from(region, render, keys=("text_area_confidence_tier",)),
        "text_area_reason_codes": _reason_codes(region, render),
        "confidence_values": confidence_values,
        "low_confidence": low_confidence,
        "source_semantic_char_count": _semantic_char_count(source_text),
        "translated_semantic_char_count": _semantic_char_count(translated_text),
        "source_glyph_statuses": [
            _get(record, "source_glyph_mask_generation_status", "generation_status")
            for record in source_records
        ],
        "source_glyph_generated": any(_truthy(_get(record, "source_glyph_mask_generated", "generated")) for record in source_records),
        "source_glyph_compact": _has_compact_source_glyph_grounding(source_records),
        "source_glyph_erasure_coverage_proven": source_glyph_erasure_coverage_proven,
        "source_erasure_required_but_unproven": source_erasure_unproven,
        "risky_caption_background_recovery": risky_caption_background_recovery,
        "valid_cleanup_mask_exists": _valid_cleanup_mask_exists(mask_records),
        "source_image_path_provided": bool(source_image_path),
        "image_size": list(image_size) if image_size else None,
    }

    if _failed_closed_root(region, render):
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=RenderEligibilityStatus.REVIEW_ALLOWED,
            reason="root_transaction_failure_review_diagnostic",
            translated_text_present=bool(str(translated_text or "").strip()),
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=_dedupe(list(hard_contradictions) + ["root_transaction_failed_closed"]),
            evidence=evidence,
        )

    if source_erasure_unproven and translated_meaningful:
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=RenderEligibilityStatus.REVIEW_ALLOWED,
            reason="source_erasure_unproven_review_allowed",
            translated_text_present=True,
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=hard_contradictions,
            evidence=evidence,
        )

    preserve_reason = _preservation_reason(page_id, rid, semantic_class, source_text, low_confidence, cleanup_classes)
    if preserve_reason:
        status = (
            RenderEligibilityStatus.ELIGIBLE
            if preserve_reason == "normal_high_confidence_speech_preserved"
            else RenderEligibilityStatus.REVIEW_ALLOWED
        )
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=status,
            reason=preserve_reason,
            translated_text_present=translated_meaningful,
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=hard_contradictions,
            preservation_reason=preserve_reason,
            evidence=evidence,
        )

    if not translated_meaningful:
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=RenderEligibilityStatus.ELIGIBLE,
            reason="no_meaningful_translated_text",
            translated_text_present=False,
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=hard_contradictions,
            evidence=evidence,
        )

    if _high_confidence_source_backed_speech(semantic_class, source_text, low_confidence):
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=RenderEligibilityStatus.ELIGIBLE,
            reason="high_confidence_source_backed_speech",
            translated_text_present=True,
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=hard_contradictions,
            evidence=evidence,
        )

    if _should_suppress(
        page_id=page_id,
        region_id=rid,
        semantic_class=semantic_class,
        source_text=source_text,
        translated_text=translated_text,
        low_confidence=low_confidence,
        risky_source=risky_source,
        hard_contradictions=hard_contradictions,
    ):
        return RenderEligibilityDecision(
            page_id=page_id,
            region_id=rid,
            status=RenderEligibilityStatus.REVIEW_ALLOWED,
            reason="source_grounding_risk_review_allowed",
            translated_text_present=True,
            source_text=source_text,
            translated_text=translated_text,
            ocr_confidence=min_confidence,
            risky_detection_source=risky_source,
            hard_contradictions=hard_contradictions,
            evidence=evidence,
        )

    if "art_entangled_ambiguous" in cleanup_classes:
        status = RenderEligibilityStatus.REVIEW_ALLOWED
        reason = "art_entangled_cleanup_target_preserved_for_review"
    elif hard_contradictions:
        status = RenderEligibilityStatus.REVIEW_ALLOWED
        reason = "source_grounding_contradiction_review_allowed"
    else:
        status = RenderEligibilityStatus.ELIGIBLE
        reason = "source_grounding_not_blocked"
    return RenderEligibilityDecision(
        page_id=page_id,
        region_id=rid,
        status=status,
        reason=reason,
        translated_text_present=True,
        source_text=source_text,
        translated_text=translated_text,
        ocr_confidence=min_confidence,
        risky_detection_source=risky_source,
        hard_contradictions=hard_contradictions,
        evidence=evidence,
    )


def _unsafe_cleanup_decision_for_region(
    *,
    page_id: str,
    region_id: str,
    region: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> RenderEligibilityDecision:
    render = _mapping_or_empty(region.get("render"))
    source_text = _text_from(audit, region, render, keys=("ocr_text", "source_text", "text"))
    translated_text = _text_from(audit, region, render, keys=("translation", "translated_text"))
    semantic_class = _text_from(audit, region, render, keys=("semantic_class", "type", "region_type"))
    confidence_values = _confidence_values(audit, render) + [
        value for value in _confidence_values(region, render) if value not in _confidence_values(audit, render)
    ]
    min_confidence = min(confidence_values) if confidence_values else None
    hard_contradictions = _unsafe_cleanup_contradictions(audit)
    protected_reason = _unsafe_cleanup_protected_reason(audit, region, semantic_class)
    parent_execution_bundle_id = str(
        _first_present(audit, "parent_execution_bundle_id", default=None)
        or _first_present(region, "parent_execution_bundle_id", default=None)
        or _first_present(render, "parent_execution_bundle_id", default=None)
        or region_id
        or ""
    )
    parent_logical_text_unit_id = str(
        _first_present(audit, "parent_logical_text_unit_id", default=None)
        or _first_present(region, "parent_logical_text_unit_id", default=None)
        or _first_present(render, "parent_logical_text_unit_id", default=None)
        or region_id
        or ""
    )
    text_block_root_id = str(
        _first_present(audit, "text_block_root_id", default=None)
        or _first_present(region, "text_block_root_id", default=None)
        or _first_present(render, "text_block_root_id", default=None)
        or ""
    )
    evidence = {
        "parent_execution_bundle_id": parent_execution_bundle_id,
        "parent_logical_text_unit_id": parent_logical_text_unit_id,
        "text_block_root_id": text_block_root_id,
        "semantic_class": semantic_class,
        "pre_render_source_erasure_status": _text_from(
            audit,
            region,
            keys=("pre_render_source_erasure_status",),
        ),
        "pre_render_source_erasure_failure_reason": _text_from(
            audit,
            region,
            keys=("pre_render_source_erasure_failure_reason",),
        ),
        "pre_render_source_erasure_residual_ratio": _first_present(
            audit,
            "pre_render_source_erasure_residual_ratio",
            "pre_render_residual_score",
            default=None,
        ),
        "pre_render_source_erasure_residual_pixels": _first_present(
            audit,
            "pre_render_source_erasure_residual_pixels",
            default=None,
        ),
        "cleanup_artifact_risk": _truthy(_first_present(audit, "cleanup_artifact_risk", default=False)),
        "cleanup_visual_validation_failed": _truthy(
            _first_present(audit, "cleanup_visual_validation_failed", default=False)
        ),
        "render_visual_contract_blocker_reason": _text_from(
            audit,
            region,
            keys=("render_visual_contract_blocker_reason",),
        ),
        "source_erasure_warning_reason": _text_from(audit, region, keys=("source_erasure_warning_reason",)),
        "source_erasure_failure_reason": _text_from(audit, region, keys=("source_erasure_failure_reason",)),
        "cleanup_source_erasure_failure_reason": _text_from(
            audit,
            region,
            keys=("cleanup_source_erasure_failure_reason",),
        ),
    }

    if not _meaningful_text(translated_text):
        status = RenderEligibilityStatus.ELIGIBLE
        reason = "no_meaningful_translated_text"
    elif protected_reason:
        status = RenderEligibilityStatus.REVIEW_ALLOWED
        reason = protected_reason
    elif _should_suppress_unsafe_cleanup_render(audit, semantic_class, hard_contradictions):
        status = RenderEligibilityStatus.REVIEW_ALLOWED
        reason = _unsafe_cleanup_suppression_reason(hard_contradictions)
    elif hard_contradictions:
        status = RenderEligibilityStatus.REVIEW_ALLOWED
        reason = "unsafe_cleanup_render_review_allowed"
    else:
        status = RenderEligibilityStatus.ELIGIBLE
        reason = "unsafe_cleanup_render_not_blocked"

    return RenderEligibilityDecision(
        page_id=page_id,
        region_id=region_id,
        status=status,
        reason=reason,
        translated_text_present=_meaningful_text(translated_text),
        source_text=source_text,
        translated_text=translated_text,
        ocr_confidence=min_confidence,
        risky_detection_source="",
        hard_contradictions=hard_contradictions,
        preservation_reason=protected_reason,
        evidence=evidence,
    )


def _should_suppress(
    *,
    page_id: str,
    region_id: str,
    semantic_class: str,
    source_text: str,
    translated_text: str,
    low_confidence: bool,
    risky_source: str,
    hard_contradictions: Sequence[str],
) -> bool:
    if not hard_contradictions or not _meaningful_text(translated_text):
        return False
    contradictions = set(hard_contradictions)
    source_erasure_block = bool(
        contradictions
        & {
            "mislocalized_source_erasure_required",
            "source_present_cleanup_unproven",
        }
    )
    if source_erasure_block and risky_source and low_confidence:
        return True
    if risky_source and low_confidence:
        return True
    if "punctuation_only_source_meaningful_translation" in contradictions and low_confidence:
        return True
    if (
        "speech" in semantic_class.lower()
        and low_confidence
        and "source_glyph_missing_failed_or_quality_failed" in contradictions
        and _weak_source_text(source_text)
    ):
        return True
    return False


def _should_suppress_unsafe_cleanup_render(
    audit: Mapping[str, Any],
    semantic_class: str,
    hard_contradictions: Sequence[str],
) -> bool:
    if not hard_contradictions:
        return False
    if str(_first_present(audit, "pre_render_source_erasure_status", default="") or "") != "failed_audit_only":
        return False
    if _unsafe_cleanup_protected_reason(audit, {}, semantic_class):
        return False
    contradictions = set(hard_contradictions)
    residual_ratio = _float_or_none(
        _first_present(audit, "pre_render_source_erasure_residual_ratio", default=None)
    ) or 0.0
    residual_pixels = _float_or_none(
        _first_present(audit, "pre_render_source_erasure_residual_pixels", default=None)
    ) or 0.0
    source_text = _text_from(audit, keys=("ocr_text", "source_text", "text"))
    source_semantic_chars = _semantic_char_count(source_text)
    role = _mapping_or_empty(_first_present(audit, "diagnostic_role_evidence", default={}))
    art_context_score = _float_or_none(role.get("art_context_score")) or 0.0
    expected_overlap = _float_or_none(
        _first_present(audit, "source_glyph_mask_expected_overlap_ratio", default=None)
    )
    outside_allowed_ratio = _float_or_none(
        _first_present(audit, "cleanup_outside_allowed_area_ratio", default=None)
    )
    if "unsafe_cleanup_artifact_risk_with_source_residual" in contradictions:
        return (
            (residual_ratio >= 0.35 and residual_pixels >= 1000)
            or (art_context_score >= 0.75 and 0 < source_semantic_chars <= 4)
            or (source_semantic_chars <= 2 and 0 < residual_pixels <= 32 and residual_ratio >= 0.20)
            or (
                0 < source_semantic_chars <= 4
                and expected_overlap is not None
                and expected_overlap >= 0.99
                and outside_allowed_ratio is not None
                and outside_allowed_ratio <= 0.02
                and residual_pixels > 0
            )
        )
    if (
        "unsafe_cleanup_visual_validation_failed" in contradictions
        and residual_ratio >= 0.35
        and residual_pixels >= 2000
    ):
        return True
    return False


def _hard_contradictions(
    *,
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    source_text: str,
    translated_text: str,
    source_records: Sequence[Mapping[str, Any]],
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
    risky_source: str,
    source_erasure_unproven: bool,
    risky_caption_background_recovery: bool,
) -> list[str]:
    contradictions: list[str] = []
    if _source_glyph_missing_failed_or_quality_failed(source_records):
        contradictions.append("source_glyph_missing_failed_or_quality_failed")
    if _expected_source_mask_missing(cleanup_records, mask_records):
        contradictions.append("expected_source_mask_missing")
    if source_erasure_unproven:
        if risky_caption_background_recovery and risky_source:
            contradictions.append("mislocalized_source_erasure_required")
        else:
            contradictions.append("source_present_cleanup_unproven")
    if _caption_flat_cleanup_art_risk(cleanup_records, mask_records):
        contradictions.append("cleanup_background_not_flat_or_art_risk")
    if _punctuation_only(source_text) and _meaningful_text(translated_text):
        contradictions.append("punctuation_only_source_meaningful_translation")
    if risky_source and not _has_compact_source_glyph_grounding(source_records):
        contradictions.append("risky_recovery_no_compact_source_glyph_grounding")
    if _artifact_or_art_risk(cleanup_records, mask_records, region, render):
        contradictions.append("artifact_or_art_risk_evidence")
    return _dedupe(contradictions)


def _unsafe_cleanup_contradictions(audit: Mapping[str, Any]) -> list[str]:
    contradictions: list[str] = []
    status = str(_first_present(audit, "pre_render_source_erasure_status", default="") or "")
    if status != "failed_audit_only":
        return contradictions
    combined = " ".join(
        str(_first_present(audit, key, default="") or "").lower()
        for key in (
            "pre_render_source_erasure_failure_reason",
            "source_erasure_warning_reason",
            "source_erasure_failure_reason",
            "cleanup_source_erasure_failure_reason",
            "render_visual_contract_blocker_reason",
        )
    )
    if "true_source_residual_blocker" in combined:
        contradictions.append("true_source_residual_blocker")
    if "source_pixels_remain_in_pre_render_cleaned_image" in combined:
        contradictions.append("source_pixels_remain_in_pre_render_cleaned_image")
    if "cleanup_mask_misses_source_glyphs" in combined:
        contradictions.append("cleanup_mask_misses_source_glyphs")
    if "cleanup_mask_too_large_for_source_glyphs" in combined:
        contradictions.append("cleanup_mask_too_large_for_source_glyphs")
    if _truthy(_first_present(audit, "cleanup_visual_validation_failed", default=False)):
        contradictions.append("unsafe_cleanup_visual_validation_failed")
    if _truthy(_first_present(audit, "cleanup_artifact_risk", default=False)) and (
        "source_pixels_remain_in_pre_render_cleaned_image" in combined
        or "true_source_residual_blocker" in combined
        or "cleanup_mask_too_large_for_source_glyphs" in combined
    ):
        contradictions.append("unsafe_cleanup_artifact_risk_with_source_residual")
    return _dedupe(contradictions)


def _unsafe_cleanup_protected_reason(
    audit: Mapping[str, Any],
    region: Mapping[str, Any],
    semantic_class: str,
) -> str:
    combined = " ".join(
        [
            str(semantic_class or ""),
            _text_from(audit, region, keys=("cleanup_mode", "cleanup_class")),
            _text_from(audit, region, keys=("classification_reason", "text_area_route_intent")),
            " ".join(_reason_codes(audit, region)),
            str(_first_present(audit, "source_grounding_status", default="") or ""),
        ]
    ).lower()
    flags = _mapping_or_empty(_first_present(audit, "flags", default={}))
    if _truthy(flags.get("ignore")):
        return "region_ignored"
    if any(marker in combined for marker in ("sfx", "decorative", "preserve")):
        return "protected_sfx_decorative_or_preserve"
    if "art_entangled" in combined:
        return "future_art_entangled_cleanup"
    if "speech" not in str(semantic_class or "").lower():
        role = _mapping_or_empty(_first_present(audit, "diagnostic_role_evidence", default={}))
        art_score = _float_or_none(role.get("art_context_score")) or 0.0
        erase_art_risk = _truthy(_first_present(audit, "erase_mask_artifact_risk", default=False))
        mixed_art_review = _truthy(
            _first_present(audit, "caption_component_v4_nonblocking_mixed_art_review", default=False)
        )
        if art_score >= 0.80 or erase_art_risk or mixed_art_review or "dark_or_art_context" in combined:
            return "future_art_entangled_cleanup"
    return ""


def _preservation_reason(
    page_id: str,
    region_id: str,
    semantic_class: str,
    source_text: str,
    low_confidence: bool,
    cleanup_classes: Sequence[str],
) -> str:
    if _high_confidence_source_backed_speech(semantic_class, source_text, low_confidence):
        return "normal_high_confidence_speech_preserved"
    if "art_entangled_ambiguous" in cleanup_classes:
        return "art_entangled_source_text_preserved_for_review"
    return ""


def _high_confidence_source_backed_speech(semantic_class: str, source_text: str, low_confidence: bool) -> bool:
    return "speech" in str(semantic_class or "").lower() and not low_confidence and _semantic_char_count(source_text) > 0


def _source_records_by_region(source_glyph_masks: Any) -> dict[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    masks_by_region = _get(source_glyph_masks, "masks_by_region")
    if isinstance(masks_by_region, Mapping):
        for region_id, value in masks_by_region.items():
            for item in _sequence_or_single(value):
                records.append(_audit_mapping(item, region_id=str(region_id or "")))
    for key in ("source_glyph_masks", "source_glyph_mask_coverage_records", "coverage_records", "masks"):
        value = _get(source_glyph_masks, key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                records.append(_audit_mapping(item))
    output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        rid = str(_first_present(record, "region_id", "target_region_id", default="") or "")
        if rid:
            output.setdefault(rid, []).append(record)
    return output


def _cleanup_records_by_region(cleanup_job_contracts: Any) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for section in ("jobs", "skipped_records", "protected_records", "rejected_records"):
        for item in _records_from(cleanup_job_contracts, section):
            record = _audit_mapping(item)
            record.setdefault("record_section", section)
            for rid in _record_region_ids(record, item):
                output.setdefault(rid, []).append(record)
    return output


def _mask_records_by_region(
    cleanup_mask_contracts: Any,
    cleanup_records_by_region: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    job_to_regions: dict[str, list[str]] = {}
    for region_id, records in cleanup_records_by_region.items():
        for record in records:
            job_id = str(_first_present(record, "cleanup_job_id", default="") or "")
            if job_id:
                job_to_regions.setdefault(job_id, []).append(region_id)

    output: dict[str, list[dict[str, Any]]] = {}
    # Production render eligibility consumes accepted CleanupMask records only.
    # Historical local recovery artifacts are no longer accepted here.
    for source_name, contracts in (
        ("cleanup_mask_contracts", cleanup_mask_contracts),
    ):
        for section in ("masks", "rejected_records", "protected_records", "skipped_records", "errors"):
            for item in _records_from(contracts, section):
                record = _audit_mapping(item)
                record.setdefault("record_source", source_name)
                record.setdefault("record_section", section)
                region_ids = _record_region_ids(record, item)
                job_id = str(_first_present(record, "cleanup_job_id", default="") or "")
                if job_id and not region_ids:
                    region_ids = job_to_regions.get(job_id, [])
                for region_id in region_ids:
                    output.setdefault(region_id, []).append(record)
    return output


def _records_from(value: Any, section: str) -> list[Any]:
    if value is None:
        return []
    raw = _get(value, section)
    if isinstance(raw, Mapping):
        return list(raw.values())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _record_region_ids(record: Mapping[str, Any], raw: Any) -> list[str]:
    region_ids: list[str] = []
    target_region_ids = _first_present(record, "target_region_ids", "region_ids", default=None)
    if isinstance(target_region_ids, Sequence) and not isinstance(target_region_ids, (str, bytes)):
        region_ids.extend(str(rid) for rid in target_region_ids if str(rid or ""))
    region_id = str(_first_present(record, "region_id", "target_region_id", default="") or "")
    if region_id:
        region_ids.append(region_id)
    if not region_ids and hasattr(raw, "target_region_ids"):
        region_ids.extend(str(rid) for rid in getattr(raw, "target_region_ids", []) if str(rid or ""))
    return _dedupe(region_ids)


def _risky_detection_source(region: Mapping[str, Any], render: Mapping[str, Any]) -> str:
    fields = [
        _text_from(region, render, keys=("text_area_detection_source", "detection_source", "source")),
        _text_from(region, render, keys=("text_area_confidence_tier",)),
        " ".join(_reason_codes(region, render)),
    ]
    combined = " ".join(fields).lower()
    for marker in RISKY_RECOVERY_MARKERS:
        if marker in combined:
            return marker
    return ""


def _low_confidence(region: Mapping[str, Any], render: Mapping[str, Any], values: Sequence[float]) -> bool:
    if values and min(values) < LOW_CONFIDENCE_THRESHOLD:
        return True
    combined = " ".join(
        [
            _text_from(region, render, keys=("text_area_confidence_tier",)),
            " ".join(_reason_codes(region, render)),
        ]
    ).lower()
    return any(marker in combined for marker in LOW_CONFIDENCE_TIER_MARKERS)


def _confidence_values(region: Mapping[str, Any], render: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    confidence = _mapping_or_empty(region.get("confidence"))
    for mapping in (region, render, confidence):
        for key in (
            "ocr_confidence",
            "route_owned_ocr_retry_confidence",
            "route_owned_ocr_retry_original_confidence",
            "logical_text_source_reconstruction_ocr_confidence",
            "confidence",
            "ocr",
        ):
            raw = mapping.get(key)
            try:
                if raw is not None and raw != "":
                    values.append(float(raw))
            except Exception:
                pass
    return values


def _source_glyph_missing_failed_or_quality_failed(records: Sequence[Mapping[str, Any]]) -> bool:
    if not records:
        return True
    generated = False
    for record in records:
        status = str(_get(record, "source_glyph_mask_generation_status", "generation_status") or "").lower()
        if _truthy(_get(record, "source_glyph_mask_generated", "generated")) or status.startswith("generated"):
            generated = True
        if "failed" in status or "quality_failed" in status:
            return True
        if _get(record, "source_glyph_mask_missing_reason", "source_glyph_mask_not_generated_reason", "failure_reason"):
            return True
    return not generated


def _source_glyph_erasure_coverage_proven(records: Sequence[Mapping[str, Any]]) -> bool:
    """True when source-glyph erasure proof is already bound to this region.

    This is not cleanup authorization and does not create a render target. It
    only prevents fail-closed render suppression when a generated source-glyph
    proof was already consumed and reports full cleanup coverage for the same
    region.
    """

    for record in records:
        status = str(_get(record, "source_glyph_mask_generation_status", "generation_status") or "").lower()
        generated = _truthy(_get(record, "source_glyph_mask_generated", "generated")) or status.startswith("generated")
        consumed = _truthy(_get(record, "source_glyph_mask_consumed_by_renderer", "consumed_by_renderer")) or "consumed" in status
        if not (generated and consumed):
            continue
        if "failed" in status or "quality_failed" in status:
            continue
        if _get(record, "source_glyph_mask_missing_reason", "source_glyph_mask_not_generated_reason", "failure_reason"):
            continue
        coverage_ratio = max(
            value
            for value in (
                _float_or_none(_first_present(record, "source_glyph_erasure_coverage_ratio", default=None)),
                _float_or_none(_first_present(record, "source_glyph_mask_cleanup_overlap_ratio", default=None)),
                _float_or_none(_first_present(record, "cleanup_source_glyph_coverage_ratio", default=None)),
                _float_or_none(_first_present(record, "source_glyph_mask_expected_overlap_ratio", default=None)),
                0.0,
            )
            if value is not None
        )
        covers_source = _truthy(_first_present(record, "cleanup_covers_source_glyphs", default=False))
        if covers_source and coverage_ratio >= 0.98:
            return True
    return False


def _expected_source_mask_missing(
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
) -> bool:
    reasons = _record_reasons(cleanup_records) + _record_reasons(mask_records)
    for reason in reasons:
        if any(marker in reason for marker in ("missing", "not_generated", "empty", "invalid")):
            return True
    return False


def _source_erasure_required_but_unproven(
    *,
    source_records: Sequence[Mapping[str, Any]],
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
) -> bool:
    """True when source cleanup is required but no executable mask proof exists."""

    cleanup_erasure_required = _cleanup_records_require_source_erasure(cleanup_records)
    valid_cleanup_mask_exists = _valid_cleanup_mask_exists(mask_records)
    source_glyph_erasure_proven = _source_glyph_erasure_coverage_proven(source_records)

    if cleanup_erasure_required and not valid_cleanup_mask_exists and not source_glyph_erasure_proven:
        return True

    source_required = any(
        _truthy(
            _get(
                record,
                "source_glyph_mask_required",
                "source_glyph_mask_required_for_cleanup",
                "cleanup_required",
            )
        )
        for record in source_records
    )
    if not source_required:
        return False
    source_unproven = _source_glyph_missing_failed_or_quality_failed(source_records)
    expected_mask_missing = _expected_source_mask_missing(cleanup_records, mask_records)
    return (source_unproven or expected_mask_missing) and not (
        valid_cleanup_mask_exists or source_glyph_erasure_proven
    )


def _cleanup_records_require_source_erasure(cleanup_records: Sequence[Mapping[str, Any]]) -> bool:
    for record in cleanup_records or []:
        section = str(_get(record, "record_section") or "").lower()
        if section and section not in {"jobs", "obligation_records"}:
            continue
        cleanup_class = str(_get(record, "cleanup_class") or "").lower()
        route_intent = str(_get(record, "route_intent") or "").lower()
        semantic_class = str(_get(record, "semantic_class") or "").lower()
        cleanup_mode = str(_get(record, "cleanup_mode") or "").lower()
        combined = " ".join([cleanup_class, route_intent, semantic_class, cleanup_mode])
        if any(marker in combined for marker in ("preserve", "sfx", "decorative", "art_entangled")):
            continue
        if any(
            marker in combined
            for marker in (
                "speech",
                "caption",
                "background",
                "narration",
                "title",
                "sign",
                "translate",
                "small_reaction",
            )
        ):
            return True
    return False


def _valid_cleanup_mask_exists(mask_records: Sequence[Mapping[str, Any]]) -> bool:
    for record in mask_records:
        section = str(_get(record, "record_section") or "").lower()
        if section and section != "masks":
            continue
        reason = " ".join(_record_reasons([record])).lower()
        if any(marker in reason for marker in ("missing", "invalid", "empty", "rejected", "failed")):
            continue
        if _truthy(_get(record, "erase_mask_empty", "mask_empty", "rejected")):
            continue
        if section == "masks" or _get(record, "cleanup_mask_id", "erase_mask_bbox", "foreground_mask_bbox"):
            return True
    return False


def _risky_caption_background_recovery_target(
    *,
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    semantic_class: str,
    cleanup_classes: Sequence[str],
    risky_source: str,
) -> bool:
    if not risky_source:
        return False
    combined = " ".join(
        [
            str(semantic_class or ""),
            _text_from(region, render, keys=("text_area_route_intent", "route_intent")),
            _text_from(region, render, keys=("text_area_detection_source", "detection_source")),
            " ".join(_reason_codes(region, render)),
            " ".join(cleanup_classes),
        ]
    ).lower()
    if any(blocked in combined for blocked in ("sfx", "decorative", "preserve")):
        return False
    if "art_entangled_ambiguous" in combined:
        return False
    if "speech" in str(semantic_class or "").lower() and not any(
        marker in combined for marker in ("caption", "background", "side_caption")
    ):
        return False
    return any(marker in combined for marker in ("caption", "background", "side_caption"))


def _caption_flat_cleanup_art_risk(
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
) -> bool:
    classes = _cleanup_classes(cleanup_records, mask_records)
    reasons = _record_reasons(mask_records)
    return "caption_flat_background" in classes and "cleanup_background_not_flat_or_art_risk" in reasons


def _artifact_or_art_risk(
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
    region: Mapping[str, Any],
    render: Mapping[str, Any],
) -> bool:
    combined = " ".join(
        _cleanup_classes(cleanup_records, mask_records)
        + _record_reasons(cleanup_records)
        + _record_reasons(mask_records)
        + _reason_codes(region, render)
    ).lower()
    if any(marker in combined for marker in ("artifact", "art_entangled", "dark_or_art", "screentone")):
        return True
    for record in mask_records:
        for key in ("erase_mask_artifact_risk", "cleanup_visual_artifact_risk", "artifact_risk"):
            if _truthy(_get(record, key)):
                return True
    return False


def _has_compact_source_glyph_grounding(records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        status = str(_get(record, "source_glyph_mask_generation_status", "generation_status") or "").lower()
        if not (_truthy(_get(record, "source_glyph_mask_generated", "generated")) or status.startswith("generated")):
            continue
        if _truthy(_get(record, "erase_mask_artifact_risk", "cleanup_visual_artifact_risk")):
            continue
        bbox = _valid_bbox(
            _first_present(
                record,
                "source_glyph_mask_actual_bbox",
                "foreground_mask_bbox",
                "source_glyph_erasure_bbox",
                "mask_bbox",
                default=None,
            )
        )
        if bbox is not None and _bbox_area(bbox) <= COMPACT_GLYPH_MAX_BBOX_AREA:
            return True
    return False


def _cleanup_classes(
    cleanup_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    values: list[str] = []
    for record in list(cleanup_records) + list(mask_records):
        raw = _first_present(record, "cleanup_class", "semantic_class", default="")
        if raw:
            values.append(_enum_value(raw))
    return _dedupe(values)


def _record_reasons(records: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for record in records:
        for key in (
            "reason",
            "rejection_reason",
            "protection_reason",
            "source_glyph_mask_missing_reason",
            "source_glyph_mask_not_generated_reason",
            "erase_mask_rejected_reason",
        ):
            raw = _first_present(record, key, default="")
            if raw:
                reasons.append(str(raw).lower())
    return _dedupe(reasons)


def _suppression_reason(hard_contradictions: Sequence[str]) -> str:
    if "root_transaction_failed_closed" in hard_contradictions:
        return "root_transaction_failed_closed"
    if "mislocalized_source_erasure_required" in hard_contradictions:
        return "mislocalized_source_erasure_required"
    if "source_present_cleanup_unproven" in hard_contradictions:
        return "source_present_cleanup_unproven"
    if "punctuation_only_source_meaningful_translation" in hard_contradictions:
        return "punctuation_only_source_meaningful_translation"
    if "cleanup_background_not_flat_or_art_risk" in hard_contradictions:
        return "cleanup_background_not_flat_or_art_risk"
    if "source_glyph_missing_failed_or_quality_failed" in hard_contradictions:
        return "source_glyph_missing_failed_or_quality_failed"
    if hard_contradictions:
        return str(hard_contradictions[0])
    return "source_ungrounded"


def _unsafe_cleanup_suppression_reason(hard_contradictions: Sequence[str]) -> str:
    if "unsafe_cleanup_artifact_risk_with_source_residual" in hard_contradictions:
        return "unsafe_cleanup_artifact_risk_with_source_residual"
    if "unsafe_cleanup_visual_validation_failed" in hard_contradictions:
        return "unsafe_cleanup_visual_validation_failed"
    if hard_contradictions:
        return str(hard_contradictions[0])
    return "unsafe_cleanup_render_failed_audit_only"


def _region_audit_fields_by_region(
    region_audit_fields: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | Any,
) -> dict[str, dict[str, Any]]:
    if region_audit_fields is None:
        return {}
    if isinstance(region_audit_fields, Mapping):
        output: dict[str, dict[str, Any]] = {}
        for key, value in region_audit_fields.items():
            if isinstance(value, Mapping):
                record = dict(value)
                rid = str(record.get("region_id") or key or "")
                if rid:
                    record.setdefault("region_id", rid)
                    output[rid] = record
        return output
    if isinstance(region_audit_fields, Sequence) and not isinstance(region_audit_fields, (str, bytes)):
        output = {}
        for item in region_audit_fields:
            if not isinstance(item, Mapping):
                continue
            rid = str(item.get("region_id") or "")
            if rid:
                output[rid] = dict(item)
        return output
    return {}


def _text_from(*mappings: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        for mapping in mappings:
            raw = mapping.get(key)
            if raw is not None and raw != "":
                return str(raw)
    return ""


def _reason_codes(region: Mapping[str, Any], render: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for mapping in (region, render):
        for key in ("text_area_reason_codes", "reason_codes", "route_reason_codes"):
            raw = mapping.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                values.extend(str(item) for item in raw if str(item or ""))
            elif raw:
                values.append(str(raw))
    return _dedupe(values)


def _meaningful_text(text: str) -> bool:
    return _semantic_char_count(text) > 0


def _weak_source_text(text: str) -> bool:
    text = str(text or "").strip()
    return _punctuation_only(text) or _semantic_char_count(text) <= WEAK_SOURCE_MAX_SEMANTIC_CHARS


def _failed_closed_root(region: Mapping[str, Any], render: Mapping[str, Any]) -> bool:
    state = _text_from(
        region,
        render,
        keys=("root_final_state", "text_block_root_final_state"),
    )
    reason = _text_from(
        region,
        render,
        keys=("root_final_state_reason", "text_block_root_final_state_reason"),
    )
    skip = _text_from(region, render, keys=("skip_reason",))
    action = _text_from(
        region,
        render,
        keys=("source_coherence_action", "root_source_coherence_action"),
    )
    combined = " ".join(value for value in (state, reason, skip, action) if value).lower()
    if state == "unresolved_meaningful_blocker":
        return True
    markers = {
        "root_transaction_failed_closed",
        "ocr_punctuation_only_blocker",
        "root_source_coherence_rejected_parent",
        "block_review_only",
        "unresolved_review_only",
        "no_accepted_parent_unit",
    }
    return any(marker in combined for marker in markers)


def _punctuation_only(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped) and _semantic_char_count(stripped) == 0


def _semantic_char_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if _is_semantic_char(ch))


def _is_semantic_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    if 0x3040 <= code <= 0x30FF:
        return True
    if 0x3400 <= code <= 0x9FFF:
        return True
    category = unicodedata.category(ch)
    return category[0] in {"L", "N"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _audit_mapping(value: Any, *, region_id: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        audit = dict(value)
    elif hasattr(value, "to_audit_dict"):
        try:
            raw = value.to_audit_dict()
            audit = dict(raw) if isinstance(raw, Mapping) else {}
        except Exception:
            audit = {}
    elif is_dataclass(value):
        audit = {
            key: getattr(value, key)
            for key in getattr(value, "__dataclass_fields__", {})
            if key not in RAW_AUDIT_KEYS
        }
    else:
        audit = {}
    if region_id and not audit.get("region_id"):
        audit["region_id"] = region_id
    for attr in ("cleanup_job_id", "target_region_ids", "cleanup_class"):
        if attr not in audit and hasattr(value, attr):
            audit[attr] = getattr(value, attr)
    return _json_safe(audit)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if _raw_payload_key(key):
                continue
            output[str(key)] = _json_safe(item)
        return output
    if is_dataclass(value):
        return _audit_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _raw_payload_key(key: Any) -> bool:
    normalized = str(key or "").lower()
    return normalized in RAW_AUDIT_KEYS or normalized.endswith("_array") or normalized.endswith("_image")


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_single(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _get(value: Any, *keys: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                return value[key]
        return None
    for key in keys:
        if hasattr(value, key):
            return getattr(value, key)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "generated", "generated_consumed"}
    return bool(value)


def _valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [int(float(item)) for item in list(value)[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_area(value: Sequence[int] | None) -> int:
    bbox = _valid_bbox(value)
    if bbox is None:
        return 0
    return max(0, int(bbox[2] - bbox[0])) * max(0, int(bbox[3] - bbox[1]))


def _region_id(region: Mapping[str, Any]) -> str:
    return str(_first_present(region, "region_id", "id", default="") or "")


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _dedupe(values: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
