"""Diagnostic render-readiness observations for finalized parent bundles.

This module never decides whether a finalized parent enters composition. The
root-parent graph and ParentExecutionBundle own that obligation; observations
collected here are audit metadata for developers and the review UI only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from app.pipeline.parent_execution_bundle import parent_execution_region_records


RENDER_ELIGIBILITY_CONTRACT_VERSION = "render_readiness_diagnostics_v2"
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
    """Diagnostic state; neither value can cancel rendering."""

    ELIGIBLE = "eligible"
    VALID_WITH_DIAGNOSTICS = "valid_with_diagnostics"
    REVIEW_ALLOWED = "valid_with_diagnostics"


@dataclass(frozen=True)
class RenderEligibilityDecision:
    """Audit-safe observation for one finalized parent."""

    page_id: str
    region_id: str
    status: RenderEligibilityStatus
    reason: str = ""
    translated_text_present: bool = False
    source_text: str = ""
    translated_text: str = ""
    ocr_confidence: float | None = None
    hard_contradictions: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "region_id": self.region_id,
            "parent_execution_bundle_id": str(
                self.evidence.get("parent_execution_bundle_id") or self.region_id
            ),
            "parent_logical_text_unit_id": str(
                self.evidence.get("parent_logical_text_unit_id") or self.region_id
            ),
            "text_block_root_id": str(self.evidence.get("text_block_root_id") or ""),
            "status": self.status.value,
            "reason": self.reason,
            "diagnostic_only": True,
            "render_required": True,
            "translated_text_present": self.translated_text_present,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "ocr_confidence": self.ocr_confidence,
            "hard_contradictions": list(self.hard_contradictions),
            "evidence": _json_safe(self.evidence),
        }


@dataclass(frozen=True)
class RenderEligibilityResult:
    """Page-level diagnostic contract for all finalized parents."""

    page_id: str
    version: str
    decisions: list[RenderEligibilityDecision] = field(default_factory=list)
    decisions_by_region_id: dict[str, RenderEligibilityDecision] = field(default_factory=dict)
    diagnostic_records: list[dict[str, Any]] = field(default_factory=list)
    eligible_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def decision_for_region(self, region_id: str) -> RenderEligibilityDecision | None:
        return self.decisions_by_region_id.get(str(region_id or ""))

    @property
    def review_allowed_records(self) -> list[dict[str, Any]]:
        """Compatibility alias for diagnostic-only readiness records."""

        return self.diagnostic_records

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "diagnostic_only": True,
            "decisions": [decision.to_audit_dict() for decision in self.decisions],
            "decisions_by_region_id": {
                region_id: decision.to_audit_dict()
                for region_id, decision in sorted(self.decisions_by_region_id.items())
            },
            "diagnostic_records": _json_safe(self.diagnostic_records),
            "eligible_records": _json_safe(self.eligible_records),
            "errors": list(self.errors),
            "summary": {
                "decision_count": len(self.decisions),
                "diagnostic_count": len(self.diagnostic_records),
                "eligible_count": len(self.eligible_records),
                "error_count": len(self.errors),
                "render_required_count": len(self.decisions),
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
    """Record render-readiness diagnostics without changing render admission."""

    decisions: list[RenderEligibilityDecision] = []
    decisions_by_region_id: dict[str, RenderEligibilityDecision] = {}
    diagnostic_records: list[dict[str, Any]] = []
    eligible_records: list[dict[str, Any]] = []
    errors: list[str] = []
    source_summary = _contract_summary(source_glyph_masks)
    cleanup_job_summary = _contract_summary(cleanup_job_contracts)
    cleanup_mask_summary = _contract_summary(cleanup_mask_contracts)

    for index, region in enumerate(regions or []):
        if not isinstance(region, Mapping):
            errors.append(f"region index {index} is not mapping-like")
            continue
        region_id = str(
            region.get("parent_execution_bundle_id")
            or region.get("parent_logical_text_unit_id")
            or region.get("region_id")
            or f"parent_{index}"
        )
        try:
            decision = _decision_for_region(
                page_id=str(page_id or ""),
                region_id=region_id,
                region=region,
                source_summary=source_summary,
                cleanup_job_summary=cleanup_job_summary,
                cleanup_mask_summary=cleanup_mask_summary,
                source_image_path=source_image_path,
                image_size=image_size,
                source_erasure_state=_source_erasure_state(
                    source_glyph_masks,
                    region_id,
                ),
            )
        except Exception as exc:
            errors.append(f"{region_id}:{type(exc).__name__}: {exc}")
            decision = RenderEligibilityDecision(
                page_id=str(page_id or ""),
                region_id=region_id,
                status=RenderEligibilityStatus.VALID_WITH_DIAGNOSTICS,
                reason="render_eligibility_error_review_degraded",
                hard_contradictions=[f"{type(exc).__name__}: {exc}"],
                evidence={
                    "parent_execution_bundle_id": region_id,
                    "parent_logical_text_unit_id": region_id,
                },
            )
        decisions.append(decision)
        decisions_by_region_id[decision.region_id] = decision
        audit = decision.to_audit_dict()
        if decision.status == RenderEligibilityStatus.VALID_WITH_DIAGNOSTICS:
            diagnostic_records.append(audit)
        else:
            eligible_records.append(audit)

    return RenderEligibilityResult(
        page_id=str(page_id or ""),
        version=RENDER_ELIGIBILITY_CONTRACT_VERSION,
        decisions=decisions,
        decisions_by_region_id=decisions_by_region_id,
        diagnostic_records=diagnostic_records,
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
    """Build diagnostics for the canonical finalized-parent denominator."""

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
    region_id: str,
    region: Mapping[str, Any],
    source_summary: dict[str, Any],
    cleanup_job_summary: dict[str, Any],
    cleanup_mask_summary: dict[str, Any],
    source_image_path: str | None,
    image_size: tuple[int, int] | None,
    source_erasure_state: tuple[bool, bool],
) -> RenderEligibilityDecision:
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    source_text = _first_text(region, render, "ocr_text", "source_text", "text")
    translated_text = _first_text(region, render, "translation", "translated_text")
    diagnostics: list[str] = []
    if not source_text.strip():
        diagnostics.append("source_text_missing")
    if not translated_text.strip():
        diagnostics.append("translated_text_missing")
    if bool(region.get("needs_review") or render.get("needs_review")):
        diagnostics.append("upstream_review_recommended")
    confidence = _first_float(region, render, "ocr_confidence", "confidence")
    if confidence is not None and confidence < 0.35:
        diagnostics.append("low_ocr_confidence")
    source_evidence_present, source_erasure_proven = source_erasure_state
    source_erasure_unproven = bool(source_text.strip() and not source_erasure_proven)
    if source_erasure_unproven:
        diagnostics.append("source_present_cleanup_unproven")

    evidence = {
        "parent_execution_bundle_id": str(
            region.get("parent_execution_bundle_id") or region_id
        ),
        "parent_logical_text_unit_id": str(
            region.get("parent_logical_text_unit_id") or region_id
        ),
        "text_block_root_id": str(region.get("text_block_root_id") or ""),
        "execution_region_authority": str(region.get("execution_region_authority") or ""),
        "source_glyph_contract_summary": source_summary,
        "cleanup_job_contract_summary": cleanup_job_summary,
        "cleanup_mask_contract_summary": cleanup_mask_summary,
        "valid_cleanup_mask_exists": bool(
            cleanup_mask_summary.get("accepted_count")
            or cleanup_mask_summary.get("mask_count")
            or cleanup_mask_summary.get("cleanup_mask_count")
        ),
        "source_image_path_provided": bool(source_image_path),
        "image_size": list(image_size) if image_size else None,
        "source_glyph_evidence_present": source_evidence_present,
        "source_glyph_erasure_coverage_proven": source_erasure_proven,
        "source_erasure_required_but_unproven": source_erasure_unproven,
    }
    status = (
        RenderEligibilityStatus.VALID_WITH_DIAGNOSTICS
        if diagnostics
        else RenderEligibilityStatus.ELIGIBLE
    )
    return RenderEligibilityDecision(
        page_id=page_id,
        region_id=region_id,
        status=status,
        reason=(
            "source_erasure_unproven_review_allowed"
            if source_erasure_unproven
            else "normal_high_confidence_speech_preserved"
            if source_erasure_proven
            else diagnostics[0]
            if diagnostics
            else "finalized_parent_render_required"
        ),
        translated_text_present=bool(translated_text.strip()),
        source_text=source_text,
        translated_text=translated_text,
        ocr_confidence=confidence,
        hard_contradictions=diagnostics,
        evidence=evidence,
    )


def _preservation_reason(*_args: Any, **_kwargs: Any) -> str:
    """Compatibility probe: no page or region can cancel parent rendering."""

    return ""


def _first_text(*sources_and_keys: Any) -> str:
    sources = [value for value in sources_and_keys if isinstance(value, Mapping)]
    keys = [value for value in sources_and_keys if isinstance(value, str)]
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _first_float(*sources_and_keys: Any) -> float | None:
    sources = [value for value in sources_and_keys if isinstance(value, Mapping)]
    keys = [value for value in sources_and_keys if isinstance(value, str)]
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _contract_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    to_audit = getattr(value, "to_audit_dict", None)
    if callable(to_audit):
        try:
            payload = to_audit()
        except Exception:
            payload = {}
    elif isinstance(value, Mapping):
        payload = value
    else:
        payload = {}
    if not isinstance(payload, Mapping):
        return {}
    summary = payload.get("summary")
    return _json_safe(summary) if isinstance(summary, Mapping) else {}


def _source_erasure_state(value: Any, region_id: str) -> tuple[bool, bool]:
    if value is None:
        return False, False
    if isinstance(value, Mapping):
        records = value.get("source_glyph_masks") or value.get("masks") or []
    else:
        records = getattr(value, "masks", None) or getattr(value, "source_glyph_masks", None) or []
    present = False
    proven = False
    for record in records:
        if not isinstance(record, Mapping):
            to_audit = getattr(record, "to_audit_dict", None)
            record = to_audit() if callable(to_audit) else {}
        if not isinstance(record, Mapping):
            continue
        record_region_id = str(
            record.get("parent_execution_bundle_id")
            or record.get("parent_logical_text_unit_id")
            or record.get("region_id")
            or ""
        )
        if record_region_id != str(region_id or ""):
            continue
        present = True
        consumed = bool(
            record.get("source_glyph_mask_consumed_by_renderer")
            or record.get("source_glyph_mask_consumed_by_cleanup")
            or record.get("consumed_by_cleanup")
        )
        covers = record.get("cleanup_covers_source_glyphs") is not False
        try:
            coverage = float(record.get("source_glyph_erasure_coverage_ratio") or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        if consumed and covers and coverage > 0.0:
            proven = True
    return present, proven


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if str(key) not in RAW_AUDIT_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(value.__dict__)
    return str(value)
