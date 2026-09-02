# -*- coding: utf-8 -*-
"""BubbleDetection-first text-area planning contracts and helpers.

This module is intentionally model-runtime free. BubbleDetection owns visual
model inference; the controller and debug artifacts use this module to turn the
service result into a page-level plan for scoped detection/OCR decisions.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from app.pipeline.oriented_layout import oriented_frame_from_speech_container
from app.pipeline.status_contracts import PipelineStage, PipelineStageTechnicalError

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None


TEXT_AREA_PLAN_VERSION = "bubble_detection_default_text_area_plan_v1"
TEXT_AREA_ROOT_PARENT_CHILD_PLAN_VERSION = "text_area_root_parent_child_plan_v1"

CONTAINER_SPEECH = "speech_bubble"
CONTAINER_CAPTION = "caption_background"
CONTAINER_SFX = "sfx_decorative_art"
CONTAINER_UNKNOWN = "unknown_fallback"

ROUTE_TRANSLATE_SPEECH = "translate_speech"
ROUTE_TRANSLATE_CAPTION = "translate_caption_background"
ROUTE_PRESERVE_SFX = "preserve_sfx_decorative"
ROUTE_REVIEW_FALLBACK = "review_or_fallback"

TEXT_AREA_GRAPH_WORKFLOW_DISPOSITION = "workflow_root"
TEXT_AREA_GRAPH_EXCLUDED_DISPOSITION = "excluded_nonworkflow"
TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT = "text_unit_evidence_bbox"
TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT_REFINED_ISLAND = "text_unit_evidence_visual_refined_island"
TEXT_AREA_GRAPH_PARENT_SOURCE_ROOT_VISUAL_TEXT_ISLAND = "root_local_visual_text_island"
TEXT_AREA_GRAPH_PARENT_SOURCE_SINGLE_CLAIM = "single_parent_claim"
TEXT_AREA_GRAPH_CHILD_SCOPE_PARENT_BOUNDARY = "parent_boundary_bbox"

AUTH_CLEANUP_TRANSLATE_SPEECH = "cleanup_translate_speech"
AUTH_CLEANUP_TRANSLATE_BACKGROUND = "cleanup_translate_background"
AUTH_CLEANUP_TRANSLATE_CAPTION = "cleanup_translate_caption"
AUTH_PROTECT_SFX_DECORATIVE = "protect_sfx_decorative"
AUTH_PROTECT_ART_OR_NON_TEXT = "protect_art_or_non_text"
AUTH_REVIEW_UNKNOWN_NOT_CLEANUP = "review_unknown_not_cleanup"
AUTH_OUTSIDE_CLEANUP_SCOPE = "outside_cleanup_scope"
AUTH_AMBIGUOUS_COMPONENT_OWNER = "ambiguous_component_owner"

SEMANTIC_KIND_SPEECH = "speech"
SEMANTIC_KIND_BACKGROUND_NARRATION = "background_narration"
SEMANTIC_KIND_TITLE = "title"
SEMANTIC_KIND_CAPTION = "caption"
SEMANTIC_KIND_SFX = "sfx"
SEMANTIC_KIND_DECORATIVE = "decorative"
SEMANTIC_KIND_ART_OR_NON_TEXT = "art_or_non_text"
SEMANTIC_KIND_UNKNOWN = "unknown"

TEXT_AREA_COMPONENT_AUTHORIZATION_MAP_VERSION = "text_area_component_authorization_map_v1"
OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE = 0.85
OGKALU_RETAINED_PAIRED_TEXT_EVIDENCE_CONFIDENCE = 0.50
OGKALU_TEXT_FREE_BACKGROUND_ROOT_CONFIDENCE = 0.70

PROJECTION_READY = "projection_ready"
PROJECTION_NO_SEMANTIC_AUTHORITY = "projection_no_semantic_authority"
PROJECTION_UNDERCOVERAGE = "segmentation_undercoverage"
PROJECTION_COMPONENT_MISSING = "segmentation_component_missing"
PROJECTION_COMPONENT_FRAGMENTED = "segmentation_component_fragmented"
PROJECTION_COMPONENT_MERGED = "segmentation_component_merged"
PROJECTION_AMBIGUOUS_COMPONENT = "projection_ambiguous_component"
PROJECTION_OUTSIDE_AUTHORIZED_AREA = "projection_outside_authorized_area"
EFFECTIVE_MASK_NOT_READY = "effective_mask_not_ready"

MASK_READY = "mask_ready"
MASK_NOT_READY = "mask_not_ready"
MASK_NOT_APPLICABLE = "mask_not_applicable"
COMPONENT_FINAL_MASK_AUTHORITY_WITHHELD_REASON = "component_final_mask_authority_withheld_until_projection_ready"
COMPONENT_PARENT_EXECUTION_BOUNDARY_REQUIRED_REASON = "component_cleanup_authority_requires_parent_execution_boundary"
COMPONENT_PARENT_EXECUTION_BOUNDARY_PREFERRED_REASON = "component_cleanup_authority_prefers_parent_execution_boundary"
COMPONENT_PARENT_EXECUTION_BOUNDARY_BLOCKED_REASON = "component_cleanup_authority_blocked_by_parent_execution_boundary"

COMPONENT_AUTHORIZATION_STATES = {
    AUTH_CLEANUP_TRANSLATE_SPEECH,
    AUTH_CLEANUP_TRANSLATE_BACKGROUND,
    AUTH_CLEANUP_TRANSLATE_CAPTION,
    AUTH_PROTECT_SFX_DECORATIVE,
    AUTH_PROTECT_ART_OR_NON_TEXT,
    AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
    AUTH_OUTSIDE_CLEANUP_SCOPE,
    AUTH_AMBIGUOUS_COMPONENT_OWNER,
}

SEMANTIC_AUTHORIZATION_FINAL_STATES = set(COMPONENT_AUTHORIZATION_STATES)

COMPONENT_AUTHORIZATION_COLORS = {
    AUTH_CLEANUP_TRANSLATE_SPEECH: "green",
    AUTH_CLEANUP_TRANSLATE_BACKGROUND: "green",
    AUTH_CLEANUP_TRANSLATE_CAPTION: "green",
    AUTH_PROTECT_SFX_DECORATIVE: "red",
    AUTH_PROTECT_ART_OR_NON_TEXT: "purple",
    AUTH_REVIEW_UNKNOWN_NOT_CLEANUP: "gray",
    AUTH_OUTSIDE_CLEANUP_SCOPE: "blue",
    AUTH_AMBIGUOUS_COMPONENT_OWNER: "orange",
}

FRESH_AUTHORIZATION_FIELD_ORIGIN = "fresh_text_area_plan"
REPLAY_AUTHORIZATION_FIELD_ORIGINS = {
    "replay_old_artifact_baseline",
    "old_artifact_baseline",
    "stale_artifact",
    "fixture",
}

SEMANTIC_EVIDENCE_PROVIDER_VERSION = "semantic_evidence_provider_v1"
PROVIDER_KITSUMED_SPEECH_MASK = "kitsumed_speech_mask_evidence"
PROVIDER_OGKALU_TEXT_BUBBLE = "ogkalu_text_bubble_evidence"
PROVIDER_OGKALU_TEXT_FREE_BACKGROUND = "ogkalu_text_free_background_evidence"
PROVIDER_OGKALU_SFX_DECORATIVE = "ogkalu_sfx_decorative_evidence"
PROVIDER_COMIC_TEXT_DETECTOR_PARENT_BOUNDARY = "comic_text_detector_parent_boundary_evidence"
PROVIDER_CURRENT_REGION_SEMANTIC = "current_region_semantic_evidence"
PROVIDER_TEXTAREA_DETERMINISTIC_TOP_BAND = "textarea_deterministic_top_band_background_evidence"
PROVIDER_TEXTAREA_DETERMINISTIC_SIDE_NARRATION = "textarea_deterministic_side_narration_evidence"
PROVIDER_TEXTAREA_DETERMINISTIC_LARGE_SFX = "textarea_deterministic_large_sfx_evidence"

APPROVED_SEMANTIC_EVIDENCE_PROVIDERS = {
    PROVIDER_KITSUMED_SPEECH_MASK,
    PROVIDER_OGKALU_TEXT_BUBBLE,
    PROVIDER_OGKALU_TEXT_FREE_BACKGROUND,
    PROVIDER_OGKALU_SFX_DECORATIVE,
    PROVIDER_COMIC_TEXT_DETECTOR_PARENT_BOUNDARY,
    PROVIDER_CURRENT_REGION_SEMANTIC,
    PROVIDER_TEXTAREA_DETERMINISTIC_TOP_BAND,
    PROVIDER_TEXTAREA_DETERMINISTIC_SIDE_NARRATION,
    PROVIDER_TEXTAREA_DETERMINISTIC_LARGE_SFX,
}

SEMANTIC_TARGET_SFX_DECORATIVE = "sfx_decorative"
SEMANTIC_TARGET_REVIEW_UNKNOWN = "review_unknown"

WEAK_BACKGROUND_AUTHORITY_REASON_TOKENS = (
    "typed_deterministic_side_narration_background_authority",
    "typed_side_narration_background_authority",
    "typed_deterministic_top_band_background_authority",
    "deterministic_top_band_background_authority",
    "typed_text_free_background_model_authority",
    "deterministic_side_narration_background_authority",
    "side_narration_background_authority",
)

WEAK_SIDE_BACKGROUND_AUTHORITY_REASON_TOKENS = (
    "typed_deterministic_side_narration_background_authority",
    "typed_side_narration_background_authority",
    "deterministic_side_narration_background_authority",
    "side_narration_background_authority",
)

PROTECTED_BLOCKING_COMPONENT_DEFECT_CODES = {
    "sfx_group_conflicts_with_cleanup_authorization",
    "large_decorative_component_conflicts_with_cleanup_authority",
    "parent_execution_cleanup_binding_blocked_by_protected_projection",
}

REVIEW_BLOCKING_COMPONENT_DEFECT_CODES = {
    "large_review_only_caption_component_conflicts_with_cleanup_authority",
    "large_review_only_caption_group_conflicts_with_cleanup_authority",
    "unowned_display_neighbor_conflicts_with_cleanup_authority",
}

SIDE_CAPTION_TEXT_CHAIN_SUPPORT_DEFECT = "side_caption_projection_without_text_chain_support"

OGKALU_ONLY_SPEECH_AUTHORITY_REASON_TOKENS = (
    "typed_bright_ogkalu_bubble_speech_authority",
    "typed_ogkalu_text_bubble_speech_authority",
)

UNRESOLVED_OGKALU_SPEECH_RISK_REASON = "unresolved_ogkalu_speech_risk_candidate"
OGKALU_BUBBLE_TEXT_PAIR_REASON = "ogkalu_text_bubble_inside_unmasked_bubble_support"
OGKALU_BUBBLE_TEXT_PAIR_AUTHORITY_REASON = "ogkalu_bubble_text_pair_speech_authority"
OGKALU_TEXT_EVIDENCE_ATTACHED_REASON = "ogkalu_text_bubble_attached_to_unmasked_bubble_container"

DETECTION_SCOPED = "scoped"
DETECTION_COMPATIBILITY_FALLBACK = "compatibility_fallback"
DETECTION_BLOCKED = "blocked_by_text_area_plan"
DETECTION_CAPTION_RECOVERY = "caption_container_text_instance_recovery"

TOP_BAND_CAPTION_SEARCH_REASON = "text_area_plan:deterministic_top_band_caption_search"
SIDE_CAPTION_SEARCH_REASON = "text_area_plan:deterministic_vertical_side_caption_search"

DECORATIVE_REGION_REASONS = {
    "nonbubble_short_kana_art_text_candidate",
    "nonbubble_short_reaction_art_text_candidate",
    "nonbubble_short_reaction_art_sfx_candidate",
    "nonbubble_breath_sfx_art_text_candidate",
    "short_reaction_without_visual_speech_ownership",
    "medium_large_katakana_sfx_candidate",
    "low_conf_dark_short_art_sfx_candidate",
    "large_low_confidence_nonbubble_sfx_candidate",
    "large_short_decorative_sfx_candidate",
    "text_area_review_only_unknown_not_auto_translated",
}

CAPTION_REGION_REASONS = {
    "top_row_background_caption_candidate",
    "top_row_caption_fragment_candidate",
}

SPEECH_REGION_REASONS = {
    "bubble_contained_short_laugh_speech",
    "speech_bubble_missed_text_recovery",
    "bubble_local_nested_speech_fragment_ownership",
    "adjacent_vertical_speech_text_conservation_recovery",
}


@dataclass
class TextAreaPlanEvidence:
    source: str
    source_ids: List[str] = field(default_factory=list)
    confidence: float | str | None = None
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "conflict_flags": list(self.conflict_flags),
        }


@dataclass
class TextAreaRouteIntent:
    route_intent: str
    translation_eligible: bool
    preserve_source: bool
    human_review_required: bool
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaFallbackReason:
    reason: str
    detail: str = ""
    safe_to_ocr: bool = False
    safe_to_translate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaContainer:
    container_id: str
    page_id: str
    container_type: str
    bbox: List[int]
    mask_ref: Optional[str] = None
    mask_summary: Dict[str, Any] = field(default_factory=dict)
    source_model_ids: List[str] = field(default_factory=list)
    confidence: float | str | None = None
    confidence_tier: str = "low"
    route_intent: str = ROUTE_REVIEW_FALLBACK
    ocr_eligible: bool = True
    comic_text_detector_scope_eligible: bool = True
    fallback_reason: Optional[str] = None
    evidence_reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    debug_behavior_marker: str = "text_area_plan_default_owner"
    would_change_behavior: bool = False
    human_review_required: bool = False
    ocr_eligibility_reason: str = ""
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    semantic_kind: str = ""
    semantic_role_evidence: Dict[str, Any] = field(default_factory=dict)
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"
    authorization_source_stage: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    semantic_authorization_state: str = ""
    parent_source_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaScope:
    scope_id: str
    page_id: str
    container_id: str
    bbox: List[int]
    container_type: str
    route_intent: str
    ocr_eligible: bool
    comic_text_detector_scope_eligible: bool
    fallback_reason: Optional[str] = None
    source: str = "text_area_plan"
    ocr_eligibility_reason: str = ""
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"
    authorization_source_stage: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    semantic_authorization_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaSemanticAuthorizationRecord:
    semantic_unit_id: str
    page_id: str
    bbox: List[int]
    polygon: List[List[float]] = field(default_factory=list)
    semantic_kind: str = "unknown"
    cleanup_authorization: str = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    authorization_source_stage: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    source_evidence_ids: List[str] = field(default_factory=list)
    source_model_ids: List[str] = field(default_factory=list)
    evidence_source_list: List[str] = field(default_factory=list)
    confidence_tier: str = "low"
    reason_codes: List[str] = field(default_factory=list)
    evidence_reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    semantic_role_evidence: Dict[str, Any] = field(default_factory=dict)
    semantic_evidence_providers: List[str] = field(default_factory=list)
    semantic_evidence_ids: List[str] = field(default_factory=list)
    semantic_evidence_trace: List[Dict[str, Any]] = field(default_factory=list)
    must_not_mutate: bool = False
    review_required: bool = False
    ocr_eligible: bool = False
    comic_text_detector_scope_eligible: bool = False
    translation_eligible: bool = False
    render_eligible: bool = False
    cleanup_executable: bool = False
    semantic_authorization_state: str = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    semantic_authority_owner: str = "TextAreaPlan/BubbleDetection"
    container_id: str = ""
    route_intent: str = ROUTE_REVIEW_FALLBACK
    container_type: str = CONTAINER_UNKNOWN
    protection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaSemanticAdjudication:
    cleanup_authorization: str
    semantic_authorization_state: str
    semantic_kind: str
    must_not_mutate: bool
    protection_reason: str = ""
    authorization_source_stage: str = "text_area_plan_pre_ocr"
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = FRESH_AUTHORIZATION_FIELD_ORIGIN
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class ScopedDetectionCandidate:
    detection_id: str
    page_id: str
    bbox: List[int]
    polygon: List[List[float]]
    confidence: float
    text_area_container_id: Optional[str]
    container_type: str
    route_intent: str
    ocr_eligible: bool
    detection_source: str
    fallback_reason: Optional[str]
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    would_change_behavior: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"
    authorization_source_stage: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    semantic_authorization_state: str = ""
    ctd_scope_eligible: bool = False
    translation_eligible: bool = False
    render_eligible: bool = False
    cleanup_executable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ScopedOcrCandidate:
    ocr_candidate_id: str
    page_id: str
    region_id: str
    bbox: List[int]
    text_area_container_id: Optional[str]
    container_type: str
    route_intent: str
    crop_source: str
    fallback_reason: Optional[str]
    ocr_eligible: bool
    ctd_scope_eligible: bool
    ocr_text: str
    ocr_confidence: float
    accepted: bool
    reason_codes: List[str] = field(default_factory=list)
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    would_change_behavior: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"
    authorization_source_stage: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    semantic_authorization_state: str = ""
    translation_eligible: bool = False
    render_eligible: bool = False
    cleanup_executable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaPlanRuntime:
    version: str = TEXT_AREA_PLAN_VERSION
    generated: bool = False
    runtime_sec: float = 0.0
    bubble_detection_generated: bool = False
    bubble_detection_cache_hit: Optional[bool] = None
    provider_fallback_used: Optional[bool] = None
    true_scoped_detector_available: bool = False
    compatibility_mode: str = "scoped_detector_by_text_area_plan"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _records_to_dict(records: Sequence[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in records or []:
        if hasattr(record, "to_dict"):
            output.append(record.to_dict())
        elif isinstance(record, Mapping):
            output.append(dict(record))
    return output


@dataclass
class TextAreaRootNode:
    root_node_id: str
    page_id: str
    container_id: str
    semantic_role: str
    workflow_disposition: str
    bbox: List[int] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    support_geometry_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_node_id": self.root_node_id,
            "page_id": self.page_id,
            "container_id": self.container_id,
            "semantic_role": self.semantic_role,
            "workflow_disposition": self.workflow_disposition,
            "bbox": list(self.bbox),
            "reason_codes": list(self.reason_codes),
            "source_evidence_ids": list(self.source_evidence_ids),
            "support_geometry_ids": list(self.support_geometry_ids),
        }


@dataclass
class TextAreaParentNode:
    parent_node_id: str
    root_node_id: str
    page_id: str
    container_id: str
    parent_boundary_id: str
    parent_kind: str
    bbox: List[int] = field(default_factory=list)
    container_local_bbox: List[int] = field(default_factory=list)
    boundary_source: str = ""
    is_explicit_parent_obligation: bool = False
    initial_state: str = "pending_source_attachment"
    reason_codes: List[str] = field(default_factory=list)
    support_geometry_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_node_id": self.parent_node_id,
            "root_node_id": self.root_node_id,
            "page_id": self.page_id,
            "container_id": self.container_id,
            "parent_boundary_id": self.parent_boundary_id,
            "parent_kind": self.parent_kind,
            "bbox": list(self.bbox),
            "container_local_bbox": list(self.container_local_bbox),
            "boundary_source": self.boundary_source,
            "is_explicit_parent_obligation": bool(self.is_explicit_parent_obligation),
            "initial_state": self.initial_state,
            "reason_codes": list(self.reason_codes),
            "support_geometry_ids": list(self.support_geometry_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
        }


@dataclass
class TextAreaParentBoundaryCandidate:
    candidate_id: str
    page_id: str
    root_node_id: str
    container_id: str
    candidate_boundary_id: str
    candidate_kind: str
    bbox: List[int] = field(default_factory=list)
    container_local_bbox: List[int] = field(default_factory=list)
    candidate_source: str = ""
    is_explicit_parent_boundary_candidate: bool = False
    adjudication_state: str = "proposed"
    blocker_kind: str = ""
    reason_codes: List[str] = field(default_factory=list)
    support_geometry_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    cross_container_owner_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "page_id": self.page_id,
            "root_node_id": self.root_node_id,
            "container_id": self.container_id,
            "candidate_boundary_id": self.candidate_boundary_id,
            "candidate_kind": self.candidate_kind,
            "bbox": list(self.bbox),
            "container_local_bbox": list(self.container_local_bbox),
            "candidate_source": self.candidate_source,
            "is_explicit_parent_boundary_candidate": bool(self.is_explicit_parent_boundary_candidate),
            "adjudication_state": self.adjudication_state,
            "blocker_kind": self.blocker_kind,
            "reason_codes": list(self.reason_codes),
            "support_geometry_ids": list(self.support_geometry_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
            "cross_container_owner_ids": list(self.cross_container_owner_ids),
        }


@dataclass
class TextAreaChildEvidenceSlot:
    child_slot_id: str
    parent_node_id: str
    root_node_id: str
    page_id: str
    allowed_attachment_scope: str
    bbox: List[int] = field(default_factory=list)
    support_geometry_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "child_slot_id": self.child_slot_id,
            "parent_node_id": self.parent_node_id,
            "root_node_id": self.root_node_id,
            "page_id": self.page_id,
            "allowed_attachment_scope": self.allowed_attachment_scope,
            "bbox": list(self.bbox),
            "support_geometry_ids": list(self.support_geometry_ids),
            "source_evidence_ids": list(self.source_evidence_ids),
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class TextAreaGraphBlocker:
    blocker_id: str
    page_id: str
    blocker_kind: str
    container_id: str = ""
    root_node_id: str = ""
    parent_node_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    diagnostic_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocker_id": self.blocker_id,
            "page_id": self.page_id,
            "container_id": self.container_id,
            "root_node_id": self.root_node_id,
            "parent_node_id": self.parent_node_id,
            "blocker_kind": self.blocker_kind,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "diagnostic_message": self.diagnostic_message,
        }


@dataclass
class TextAreaExcludedInventoryRecord:
    excluded_id: str
    page_id: str
    container_id: str
    exclusion_kind: str
    bbox: List[int] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    support_geometry_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excluded_id": self.excluded_id,
            "page_id": self.page_id,
            "container_id": self.container_id,
            "exclusion_kind": self.exclusion_kind,
            "bbox": list(self.bbox),
            "reason_codes": list(self.reason_codes),
            "source_evidence_ids": list(self.source_evidence_ids),
            "support_geometry_ids": list(self.support_geometry_ids),
        }


@dataclass
class TextAreaSourceEvidencePayload:
    source_evidence_id: str
    page_id: str
    source_kind: str
    container_id: str = ""
    bbox: List[int] = field(default_factory=list)
    source_ref: str = ""
    reason_codes: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_evidence_id": self.source_evidence_id,
            "page_id": self.page_id,
            "source_kind": self.source_kind,
            "container_id": self.container_id,
            "bbox": list(self.bbox),
            "source_ref": self.source_ref,
            "reason_codes": list(self.reason_codes),
            "payload": dict(self.payload),
        }


@dataclass
class TextAreaRootParentChildPlan:
    page_id: str
    version: str = TEXT_AREA_ROOT_PARENT_CHILD_PLAN_VERSION
    root_nodes: List[TextAreaRootNode | Mapping[str, Any]] = field(default_factory=list)
    parent_boundary_candidates: List[TextAreaParentBoundaryCandidate | Mapping[str, Any]] = field(default_factory=list)
    parent_nodes: List[TextAreaParentNode | Mapping[str, Any]] = field(default_factory=list)
    child_evidence_slots: List[TextAreaChildEvidenceSlot | Mapping[str, Any]] = field(default_factory=list)
    excluded_inventory: List[TextAreaExcludedInventoryRecord | Mapping[str, Any]] = field(default_factory=list)
    graph_blockers: List[TextAreaGraphBlocker | Mapping[str, Any]] = field(default_factory=list)
    source_evidence_payloads: List[TextAreaSourceEvidencePayload | Mapping[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "version": self.version,
            "root_nodes": _records_to_dict(self.root_nodes),
            "parent_boundary_candidates": _records_to_dict(self.parent_boundary_candidates),
            "parent_nodes": _records_to_dict(self.parent_nodes),
            "child_evidence_slots": _records_to_dict(self.child_evidence_slots),
            "excluded_inventory": _records_to_dict(self.excluded_inventory),
            "graph_blockers": _records_to_dict(self.graph_blockers),
            "source_evidence_payloads": _records_to_dict(self.source_evidence_payloads),
            "diagnostics": dict(self.diagnostics),
        }


def _root_parent_child_plan_to_dict(plan: Any) -> Dict[str, Any]:
    if hasattr(plan, "to_dict"):
        return plan.to_dict()
    if isinstance(plan, Mapping):
        return dict(plan)
    return {}


def build_text_area_root_parent_child_plan(plan: "TextAreaPlan | Mapping[str, Any]") -> TextAreaRootParentChildPlan:
    """Create the Phase 2 source-text-free graph plan from TextAreaPlan evidence."""

    page_id = _text_area_graph_plan_page_id(plan)
    containers = _text_area_graph_plan_containers(plan)
    image_size = _text_area_graph_plan_image_size(plan)
    luma_image = _text_area_graph_plan_luma_image(plan)
    root_nodes: List[TextAreaRootNode] = []
    parent_boundary_candidates: List[TextAreaParentBoundaryCandidate] = []
    parent_nodes: List[TextAreaParentNode] = []
    child_slots: List[TextAreaChildEvidenceSlot] = []
    excluded_inventory: List[TextAreaExcludedInventoryRecord] = []
    graph_blockers: List[TextAreaGraphBlocker] = []
    source_payloads: List[TextAreaSourceEvidencePayload] = []

    roots_by_container_id: Dict[str, TextAreaRootNode] = {}
    root_container_ids_by_root_id: Dict[str, List[str]] = {}
    containers_by_id: Dict[str, TextAreaContainer | Mapping[str, Any]] = {}
    candidate_entries_by_container_id: Dict[str, List[Dict[str, Any]]] = {}
    boundary_owners: Dict[str, List[str]] = {}
    boundary_owner_containers: Dict[str, List[str]] = {}

    workflow_containers: List[TextAreaContainer | Mapping[str, Any]] = []
    for container in containers:
        container_id = _text_area_graph_container_id(container)
        if not container_id:
            continue
        containers_by_id[container_id] = container
        if _text_area_graph_is_workflow_container(container):
            workflow_containers.append(container)

    root_groups, demoted_root_containers = _text_area_graph_normalized_root_groups(
        workflow_containers,
        image_size=image_size,
    )
    demoted_root_container_ids = {
        _text_area_graph_container_id(container)
        for container in demoted_root_containers
        if _text_area_graph_container_id(container)
    }

    for container in containers:
        container_id = _text_area_graph_container_id(container)
        if not container_id:
            continue
        bbox = _text_area_graph_container_bbox(container)
        source_ids = _text_area_graph_container_source_ids(container)
        support_ids = _text_area_graph_support_geometry_ids(container)
        reasons = _text_area_graph_reason_codes(container)
        if container_id in demoted_root_container_ids:
            excluded_inventory.append(
                TextAreaExcludedInventoryRecord(
                    excluded_id=_text_area_graph_node_id("tap_excluded", page_id, container_id),
                    page_id=page_id,
                    container_id=container_id,
                    exclusion_kind="duplicate_background_candidate",
                    bbox=bbox,
                    reason_codes=_text_area_graph_unique_strings(
                        reasons
                        + [
                            "text_area_plan:phase2_root_normalization_demoted",
                            "text_area_plan:phase2_background_evidence_reused_by_speech_root",
                        ]
                    ),
                    source_evidence_ids=source_ids,
                    support_geometry_ids=support_ids,
                )
            )
        elif _text_area_graph_is_workflow_container(container):
            continue
        elif _text_area_graph_is_excluded_container(container):
            excluded_inventory.append(
                TextAreaExcludedInventoryRecord(
                    excluded_id=_text_area_graph_node_id("tap_excluded", page_id, container_id),
                    page_id=page_id,
                    container_id=container_id,
                    exclusion_kind=_text_area_graph_exclusion_kind(container),
                    bbox=bbox,
                    reason_codes=_text_area_graph_unique_strings(
                        reasons + ["text_area_plan:phase2_excluded_inventory"]
                    ),
                    source_evidence_ids=source_ids,
                    support_geometry_ids=support_ids,
                )
            )
        else:
            graph_blockers.append(
                TextAreaGraphBlocker(
                    blocker_id=_text_area_graph_node_id("tap_blocker_root", page_id, container_id),
                    page_id=page_id,
                    container_id=container_id,
                    blocker_kind="missing_root_workflow_container",
                    evidence_ids=_text_area_graph_unique_strings(source_ids + support_ids),
                    reason_codes=_text_area_graph_unique_strings(
                        reasons + ["text_area_plan:phase2_unclassified_container"]
                    ),
                    diagnostic_message="TextAreaPlan container is neither workflow-root nor excluded inventory.",
                )
            )

    root_groups = _text_area_graph_sort_root_groups_for_reading_order(root_groups)
    root_sequence_keys_by_root_id: Dict[str, str] = {}

    for root_index, group in enumerate(root_groups):
        group_containers = list(group.get("containers") or [])
        if not group_containers:
            continue
        primary = group_containers[0]
        primary_container_id = _text_area_graph_container_id(primary)
        if not primary_container_id:
            continue
        root_sequence_key = f"r{root_index:03d}"
        root = TextAreaRootNode(
            root_node_id=_text_area_graph_node_id("tap_root", page_id, root_sequence_key),
            page_id=page_id,
            container_id=primary_container_id,
            semantic_role=str(group.get("semantic_role") or _text_area_graph_semantic_role(primary)),
            workflow_disposition=TEXT_AREA_GRAPH_WORKFLOW_DISPOSITION,
            bbox=list(group.get("bbox") or _text_area_graph_container_bbox(primary)),
            reason_codes=_text_area_graph_unique_strings(
                list(group.get("reason_codes") or [])
                + [
                    "text_area_plan:phase2_workflow_root_node",
                    "text_area_plan:phase2_normalized_root_boundary",
                ]
            ),
            source_evidence_ids=_text_area_graph_unique_strings(list(group.get("source_evidence_ids") or [])),
            support_geometry_ids=_text_area_graph_unique_strings(list(group.get("support_geometry_ids") or [])),
        )
        root_nodes.append(root)
        root_sequence_keys_by_root_id[root.root_node_id] = root_sequence_key
        group_container_ids: List[str] = []
        for member in group_containers:
            member_id = _text_area_graph_container_id(member)
            if not member_id:
                continue
            roots_by_container_id[member_id] = root
            group_container_ids.append(member_id)
            entries: List[Dict[str, Any]] = []
            for index, raw_entry in enumerate(
                _text_area_graph_parent_boundary_entries(
                    member,
                    image_size=image_size,
                    luma_image=luma_image,
                )
            ):
                evidence_id = str(raw_entry.get("evidence_id") or f"boundary_{index:02d}")
                entry = {
                    **raw_entry,
                    "candidate_id": _text_area_graph_node_id("tap_candidate", page_id, member_id, evidence_id),
                    "candidate_boundary_id": _text_area_graph_node_id("tap_candidate_boundary", page_id, member_id, evidence_id),
                }
                entries.append(entry)
                key = _text_area_graph_parent_boundary_key(entry)
                if key:
                    boundary_owners.setdefault(key, []).append(root.root_node_id)
                    boundary_owner_containers.setdefault(key, []).append(member_id)
            candidate_entries_by_container_id[member_id] = entries
        root_container_ids_by_root_id[root.root_node_id] = group_container_ids

    for root in root_nodes:
        group_container_ids = root_container_ids_by_root_id.get(root.root_node_id) or [root.container_id]
        accepted_candidates: List[TextAreaParentBoundaryCandidate] = []
        ambiguous_entries: List[Dict[str, Any]] = []
        blocked_entries: List[Dict[str, Any]] = []
        seen_root_candidate_keys: set[str] = set()
        for container_id in group_container_ids:
            container = containers_by_id.get(container_id)
            if container is None:
                continue
            for entry in candidate_entries_by_container_id.get(container_id, []):
                key = _text_area_graph_parent_boundary_key(entry)
                owners = _text_area_graph_unique_strings(boundary_owners.get(key, []))
                if key and len(owners) <= 1 and key in seen_root_candidate_keys:
                    continue
                if key:
                    seen_root_candidate_keys.add(key)
                explicit = bool(entry.get("is_explicit_parent_boundary_evidence"))
                blocker_kind = ""
                state = "accepted"
                state_reasons = ["text_area_plan:phase2_parent_boundary_candidate_accepted"]
                if len(owners) > 1:
                    state = "ambiguous"
                    blocker_kind = "cross_container_parent_boundary_evidence"
                    state_reasons = ["text_area_plan:phase2_cross_container_parent_boundary_evidence"]
                    ambiguous_entries.append({**entry, "cross_container_owners": boundary_owner_containers.get(key, [])})
                elif not explicit:
                    state = "blocked"
                    blocker_kind = "missing_explicit_parent_boundary_marker"
                    state_reasons = ["text_area_plan:phase2_missing_explicit_parent_boundary_marker"]
                    blocked_entries.append(entry)
                evidence_id = str(entry.get("evidence_id") or "")
                bbox = _text_area_graph_bbox_from_entry(entry)
                candidate = TextAreaParentBoundaryCandidate(
                    candidate_id=str(entry.get("candidate_id") or _text_area_graph_node_id("tap_candidate", page_id, container_id, evidence_id)),
                    page_id=page_id,
                    root_node_id=root.root_node_id,
                    container_id=container_id,
                    candidate_boundary_id=str(
                        entry.get("candidate_boundary_id")
                        or _text_area_graph_node_id("tap_candidate_boundary", page_id, container_id, evidence_id)
                    ),
                    candidate_kind=_text_area_graph_semantic_role(container),
                    bbox=bbox,
                    container_local_bbox=_text_area_graph_container_local_bbox(bbox, root.bbox),
                    candidate_source=str(entry.get("boundary_source") or TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT),
                    is_explicit_parent_boundary_candidate=explicit,
                    adjudication_state=state,
                    blocker_kind=blocker_kind,
                    reason_codes=_text_area_graph_unique_strings(
                        [
                            "text_area_plan:phase2_parent_boundary_candidate",
                            f"text_area_plan:parent_boundary_class:{entry.get('class_name') or 'unknown'}",
                        ]
                        + state_reasons
                    ),
                    support_geometry_ids=_text_area_graph_unique_strings([evidence_id] if evidence_id else []),
                    source_evidence_ids=_text_area_graph_unique_strings(
                        _text_area_graph_container_source_ids(container) + ([evidence_id] if evidence_id else [])
                    ),
                    cross_container_owner_ids=_text_area_graph_unique_strings(
                        boundary_owner_containers.get(key, [])
                    )
                    if state == "ambiguous"
                    else [],
                )
                parent_boundary_candidates.append(candidate)
                if state == "accepted":
                    accepted_candidates.append(candidate)
        if ambiguous_entries:
            graph_blockers.append(
                TextAreaGraphBlocker(
                    blocker_id=_text_area_graph_node_id("tap_blocker_cross_container_parent", page_id, root.container_id),
                    page_id=page_id,
                    container_id=root.container_id,
                    root_node_id=root.root_node_id,
                    blocker_kind="cross_container_parent_boundary_evidence",
                    evidence_ids=_text_area_graph_unique_strings(
                        [str(entry.get("evidence_id") or "") for entry in ambiguous_entries]
                    ),
                    reason_codes=["text_area_plan:phase2_cross_container_parent_boundary_evidence"],
                    diagnostic_message="One or more parent-boundary evidence records are owned by multiple workflow roots.",
                )
            )
        if blocked_entries:
            graph_blockers.append(
                TextAreaGraphBlocker(
                    blocker_id=_text_area_graph_node_id("tap_blocker_candidate_marker", page_id, root.container_id),
                    page_id=page_id,
                    container_id=root.container_id,
                    root_node_id=root.root_node_id,
                    blocker_kind="missing_explicit_parent_boundary_marker",
                    evidence_ids=_text_area_graph_unique_strings(
                        [str(entry.get("evidence_id") or "") for entry in blocked_entries]
                    ),
                    reason_codes=["text_area_plan:phase2_missing_explicit_parent_boundary_marker"],
                    diagnostic_message="One or more candidate records lacked an explicit source-text-free parent-boundary marker.",
                )
            )
        if not accepted_candidates:
            source_ids = list(root.source_evidence_ids)
            support_ids = list(root.support_geometry_ids)
            blocker_kind = "source_evidence_without_parent_boundary" if source_ids or support_ids else "missing_parent_boundary_evidence"
            graph_blockers.append(
                TextAreaGraphBlocker(
                    blocker_id=_text_area_graph_node_id("tap_blocker_parent_boundary", page_id, root.container_id),
                    page_id=page_id,
                    container_id=root.container_id,
                    root_node_id=root.root_node_id,
                    blocker_kind=blocker_kind,
                    evidence_ids=_text_area_graph_unique_strings(source_ids + support_ids),
                    reason_codes=[
                        "text_area_plan:phase2_missing_explicit_parent_boundary",
                        "text_area_plan:single_container_bbox_not_parent_proof",
                    ],
                    diagnostic_message="Workflow root has no unique explicit source-text-free parent-boundary evidence.",
                )
            )
            continue
        root_sequence_key = root_sequence_keys_by_root_id.get(root.root_node_id) or root.container_id
        for parent_index, candidate in enumerate(
            _text_area_graph_sort_parent_candidates_for_reading_order(accepted_candidates)
        ):
            container_id = candidate.container_id
            parent_sequence_key = f"p{parent_index:03d}"
            parent_id = _text_area_graph_node_id("tap_parent", page_id, root_sequence_key, parent_sequence_key)
            boundary_id = _text_area_graph_node_id("tap_boundary", page_id, root_sequence_key, parent_sequence_key)
            bbox = list(candidate.bbox)
            parent = TextAreaParentNode(
                parent_node_id=parent_id,
                root_node_id=root.root_node_id,
                page_id=page_id,
                container_id=container_id,
                parent_boundary_id=boundary_id,
                parent_kind=candidate.candidate_kind,
                bbox=bbox,
                container_local_bbox=list(candidate.container_local_bbox),
                boundary_source=candidate.candidate_source,
                is_explicit_parent_obligation=True,
                initial_state="pending_source_attachment",
                reason_codes=_text_area_graph_unique_strings(
                    [
                        "text_area_plan:phase2_explicit_parent_boundary",
                        f"text_area_plan:accepted_candidate:{candidate.candidate_id}",
                    ]
                ),
                support_geometry_ids=list(candidate.support_geometry_ids),
                source_evidence_ids=list(candidate.source_evidence_ids),
            )
            parent_nodes.append(parent)
            child_slots.append(
                TextAreaChildEvidenceSlot(
                    child_slot_id=_text_area_graph_node_id("tap_child_slot", page_id, root_sequence_key, parent_sequence_key),
                    parent_node_id=parent_id,
                    root_node_id=root.root_node_id,
                    page_id=page_id,
                    allowed_attachment_scope=TEXT_AREA_GRAPH_CHILD_SCOPE_PARENT_BOUNDARY,
                    bbox=bbox,
                    support_geometry_ids=list(candidate.support_geometry_ids),
                    source_evidence_ids=parent.source_evidence_ids,
                    reason_codes=["text_area_plan:phase2_child_slot_for_parent_boundary"],
                )
            )
            source_payloads.append(
                TextAreaSourceEvidencePayload(
                    source_evidence_id=_text_area_graph_node_id("tap_source_payload", page_id, root_sequence_key, parent_sequence_key),
                    page_id=page_id,
                    source_kind="parent_boundary_reference",
                    container_id=container_id,
                    bbox=bbox,
                    source_ref=candidate.candidate_id,
                    reason_codes=["text_area_plan:phase2_source_payload_reference"],
                    payload={
                        "parent_node_id": parent_id,
                        "parent_boundary_id": boundary_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_boundary_id": candidate.candidate_boundary_id,
                        "boundary_source": candidate.candidate_source,
                        "text_payload_included": False,
                    },
                )
            )

    diagnostics = {
        "schema_version": TEXT_AREA_ROOT_PARENT_CHILD_PLAN_VERSION,
        "phase": "phase2_text_area_graph_production",
        "text_payload_free_topology": True,
        "root_node_count": len(root_nodes),
        "normalized_root_group_count": len(root_nodes),
        "root_normalized_container_count": sum(len(ids) for ids in root_container_ids_by_root_id.values()),
        "root_demoted_background_candidate_count": len(demoted_root_container_ids),
        "parent_boundary_candidate_count": len(parent_boundary_candidates),
        "parent_node_count": len(parent_nodes),
        "child_evidence_slot_count": len(child_slots),
        "excluded_inventory_count": len(excluded_inventory),
        "graph_blocker_count": len(graph_blockers),
        "source_evidence_payload_count": len(source_payloads),
        "parent_boundary_source_counts": _text_area_graph_parent_source_counts(parent_nodes),
        "parent_boundary_candidate_state_counts": _text_area_graph_candidate_state_counts(parent_boundary_candidates),
        "blocker_counts": _text_area_graph_blocker_counts(graph_blockers),
        "accepted_parent_boundary_requires_explicit_evidence": True,
        "single_container_bbox_parent_claims_created": 0,
        "production_topology_forbidden_inputs": [
            "task0_ledger",
            "ocr_payload_text",
            "translated_payload_text",
            "downstream_graph_output",
            "cleanup_render_metadata",
            "route_retry_recovery_state",
        ],
    }
    return TextAreaRootParentChildPlan(
        page_id=page_id,
        root_nodes=root_nodes,
        parent_boundary_candidates=parent_boundary_candidates,
        parent_nodes=parent_nodes,
        child_evidence_slots=child_slots,
        excluded_inventory=excluded_inventory,
        graph_blockers=graph_blockers,
        source_evidence_payloads=source_payloads,
        diagnostics=diagnostics,
    )


def _text_area_graph_normalized_root_groups(
    workflow_containers: Sequence[TextAreaContainer | Mapping[str, Any]],
    *,
    image_size: Tuple[int, int] = (1, 1),
) -> Tuple[List[Dict[str, Any]], List[TextAreaContainer | Mapping[str, Any]]]:
    """Normalize workflow root evidence before parent topology is built."""

    speech_evidence_ids: set[str] = set()
    for container in workflow_containers:
        if _text_area_graph_semantic_role(container) != SEMANTIC_KIND_SPEECH:
            continue
        speech_evidence_ids.update(_text_area_graph_root_evidence_ids(container))

    demoted: List[TextAreaContainer | Mapping[str, Any]] = []
    active: List[TextAreaContainer | Mapping[str, Any]] = []
    for container in workflow_containers:
        if (
            _text_area_graph_is_deterministic_background_root(container)
            and bool(_text_area_graph_root_evidence_ids(container) & speech_evidence_ids)
        ):
            demoted.append(container)
        else:
            active.append(container)

    if not active:
        return [], demoted

    parents = list(range(len(active)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index in range(len(active)):
        for second_index in range(first_index + 1, len(active)):
            if _text_area_graph_workflow_roots_should_merge(active[first_index], active[second_index]):
                union(first_index, second_index)

    grouped_indexes: Dict[int, List[int]] = {}
    for index in range(len(active)):
        grouped_indexes.setdefault(find(index), []).append(index)

    groups: List[Dict[str, Any]] = []
    for indexes in grouped_indexes.values():
        members = [active[index] for index in sorted(indexes)]
        if not members:
            continue
        root_bbox, root_bbox_reasons = _text_area_graph_physical_root_bbox(
            members,
            active_containers=active,
            image_size=image_size,
        )
        role = _text_area_graph_semantic_role(members[0])
        reasons: List[str] = []
        source_ids: List[str] = []
        support_ids: List[str] = []
        for member in members:
            reasons.extend(_text_area_graph_reason_codes(member))
            source_ids.extend(_text_area_graph_container_source_ids(member))
            support_ids.extend(_text_area_graph_support_geometry_ids(member))
        if len(members) > 1:
            reasons.append("text_area_plan:phase2_merged_workflow_root_fragments")
        reasons.extend(root_bbox_reasons)
        groups.append(
            {
                "containers": members,
                "semantic_role": role,
                "bbox": root_bbox,
                "reason_codes": _text_area_graph_unique_strings(reasons),
                "source_evidence_ids": _text_area_graph_unique_strings(source_ids),
                "support_geometry_ids": _text_area_graph_unique_strings(support_ids),
            }
        )
    return groups, demoted


def _text_area_graph_physical_root_bbox(
    members: Sequence[TextAreaContainer | Mapping[str, Any]],
    *,
    active_containers: Sequence[TextAreaContainer | Mapping[str, Any]],
    image_size: Tuple[int, int] = (1, 1),
) -> Tuple[List[int], List[str]]:
    member_bboxes = [_text_area_graph_container_bbox(member) for member in members]
    root_bbox = _text_area_graph_union_xywh(member_bboxes)
    if not root_bbox:
        return [], []
    if image_size != (1, 1):
        root_bbox = _normalize_xywh(root_bbox, image_size)
    typed_bbox, typed_reasons = _text_area_graph_typed_speech_root_bbox(
        members,
        root_bbox,
        image_size=image_size,
    )
    if typed_bbox:
        return typed_bbox, typed_reasons
    return root_bbox, []


def _text_area_graph_typed_speech_root_bbox(
    members: Sequence[TextAreaContainer | Mapping[str, Any]],
    fallback_root_bbox: Sequence[Any],
    *,
    image_size: Tuple[int, int] = (1, 1),
) -> Tuple[List[int], List[str]]:
    if not members or not fallback_root_bbox:
        return [], []
    if any(_text_area_graph_semantic_role(member) != SEMANTIC_KIND_SPEECH for member in members):
        return [], []

    root_bbox = list(fallback_root_bbox)
    if image_size != (1, 1):
        root_bbox = _normalize_xywh(root_bbox, image_size)
    root_area = _bbox_area_xywh(root_bbox)
    if root_area <= 0:
        return [], []

    text_bboxes: List[List[int]] = []
    bubble_bboxes: List[List[int]] = []
    seen_text: set[Tuple[int, int, int, int]] = set()
    seen_bubbles: set[Tuple[int, int, int, int]] = set()
    for member in members:
        role_evidence = _container_semantic_role_evidence(member)
        for field_name in ("text_unit_evidence_bboxes", "model_evidence_bboxes"):
            entries = role_evidence.get(field_name) or []
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                class_name = str(entry.get("class_name") or "").strip().lower()
                if class_name not in {"text_bubble", "bubble"}:
                    continue
                bbox = _text_evidence_bbox_xywh(entry, image_size)
                x, y, w, h = _coerce_xywh(bbox)
                if w <= 0 or h <= 0:
                    continue
                normalized = [x, y, w, h]
                key = (x, y, w, h)
                if class_name == "text_bubble":
                    if key not in seen_text:
                        seen_text.add(key)
                        text_bboxes.append(normalized)
                elif key not in seen_bubbles:
                    seen_bubbles.add(key)
                    bubble_bboxes.append(normalized)

    if not text_bboxes or not bubble_bboxes:
        return [], []

    accepted_bubbles: List[List[int]] = []
    for bubble_bbox in bubble_bboxes:
        if _inside_ratio_xywh(bubble_bbox, root_bbox) < 0.60 and not _text_area_graph_bbox_center_inside_xywh(
            bubble_bbox,
            root_bbox,
        ):
            continue
        owned_text = [
            text_bbox
            for text_bbox in text_bboxes
            if _inside_ratio_xywh(text_bbox, bubble_bbox) >= 0.70
            or _text_area_graph_bbox_center_inside_xywh(text_bbox, bubble_bbox)
        ]
        if owned_text:
            accepted_bubbles.append(bubble_bbox)

    if not accepted_bubbles:
        return [], []

    candidate = _text_area_graph_union_xywh(accepted_bubbles)
    if image_size != (1, 1) and candidate:
        candidate = _normalize_xywh(candidate, image_size)
    candidate_area = _bbox_area_xywh(candidate)
    if candidate_area <= 0 or candidate_area >= root_area * 0.96:
        return [], []

    for text_bbox in text_bboxes:
        if _inside_ratio_xywh(text_bbox, candidate) < 0.70 and not _text_area_graph_bbox_center_inside_xywh(
            text_bbox,
            candidate,
        ):
            return [], []

    return candidate, ["text_area_plan:physical_root_bbox_from_typed_bubble_evidence"]


def _text_area_graph_sort_root_groups_for_reading_order(
    groups: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Sort root containers in Japanese manga page order.

    The graph ID sequence is canonical identity, so it must be based on page
    layout rather than detector/container ids. Roots are ordered by upper page
    bands first, then right-to-left within each band.
    """

    records: List[Tuple[int, Mapping[str, Any], List[int]]] = []
    for index, group in enumerate(groups or []):
        bbox = _text_area_graph_bbox_from_any(group.get("bbox") if isinstance(group, Mapping) else [])
        if not bbox and isinstance(group, Mapping):
            bbox = _text_area_graph_union_xywh(
                [
                    _text_area_graph_container_bbox(container)
                    for container in list(group.get("containers") or [])
                ]
            )
        records.append((index, group, bbox))
    if not records:
        return []

    heights = [float(bbox[3]) for _index, _group, bbox in records if _text_area_graph_valid_bbox(bbox)]
    row_threshold = max(32.0, min(160.0, _median_float(heights) * 0.45)) if heights else 64.0

    rows: List[Dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            _text_area_graph_bbox_top(item[2]),
            -_text_area_graph_bbox_right(item[2]),
            item[0],
        ),
    ):
        y = _text_area_graph_bbox_top(record[2])
        target = None
        for row in rows:
            if abs(y - float(row["anchor_y"])) <= row_threshold:
                target = row
                break
        if target is None:
            rows.append({"anchor_y": y, "records": [record]})
        else:
            target["records"].append(record)

    ordered: List[Mapping[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["anchor_y"])):
        row_records = sorted(
            row["records"],
            key=lambda item: (
                -_text_area_graph_bbox_right(item[2]),
                _text_area_graph_bbox_top(item[2]),
                item[0],
            ),
        )
        ordered.extend(record[1] for record in row_records)
    return ordered


def _text_area_graph_sort_parent_candidates_for_reading_order(
    candidates: Sequence[TextAreaParentBoundaryCandidate],
) -> List[TextAreaParentBoundaryCandidate]:
    """Sort accepted parent obligations inside one root.

    Vertical manga text is read by columns: right-side columns before left-side
    columns, and top-to-bottom within a column. Candidate ids and evidence ids
    remain provenance only; the canonical parent sequence follows this order.
    """

    records: List[Tuple[int, TextAreaParentBoundaryCandidate, List[int]]] = [
        (index, candidate, list(candidate.bbox or []))
        for index, candidate in enumerate(candidates or [])
    ]
    if not records:
        return []

    widths = [float(bbox[2]) for _index, _candidate, bbox in records if _text_area_graph_valid_bbox(bbox)]
    column_threshold = max(12.0, min(96.0, _median_float(widths) * 0.60)) if widths else 24.0

    columns: List[Dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            -_text_area_graph_bbox_center_x(item[2]),
            _text_area_graph_bbox_top(item[2]),
            item[0],
        ),
    ):
        cx = _text_area_graph_bbox_center_x(record[2])
        target = None
        for column in columns:
            if abs(cx - float(column["anchor_x"])) <= column_threshold:
                target = column
                break
        if target is None:
            columns.append({"anchor_x": cx, "records": [record]})
        else:
            target["records"].append(record)

    ordered: List[TextAreaParentBoundaryCandidate] = []
    for column in sorted(columns, key=lambda item: -float(item["anchor_x"])):
        column_records = sorted(
            column["records"],
            key=lambda item: (
                _text_area_graph_bbox_top(item[2]),
                -_text_area_graph_bbox_right(item[2]),
                item[0],
            ),
        )
        ordered.extend(record[1] for record in column_records)
    return ordered


def _text_area_graph_bbox_from_any(value: Any) -> List[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return []
    x, y, w, h = _coerce_xywh(value)
    if w <= 0 or h <= 0:
        return []
    return [x, y, w, h]


def _text_area_graph_valid_bbox(bbox: Sequence[Any]) -> bool:
    _x, _y, w, h = _coerce_xywh(bbox)
    return w > 0 and h > 0


def _text_area_graph_bbox_top(bbox: Sequence[Any]) -> float:
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return float("inf")
    return float(y)


def _text_area_graph_bbox_right(bbox: Sequence[Any]) -> float:
    x, _y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return float("-inf")
    return float(x + w)


def _text_area_graph_bbox_center_x(bbox: Sequence[Any]) -> float:
    x, _y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return float("-inf")
    return float(x + (w / 2.0))


def _text_area_graph_owned_text_unit_physical_bboxes(
    container: TextAreaContainer | Mapping[str, Any],
    *,
    active_containers: Sequence[TextAreaContainer | Mapping[str, Any]],
    group_container_ids: set[str],
) -> List[List[int]]:
    role_evidence = _container_semantic_role_evidence(container)
    entries = role_evidence.get("text_unit_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    preferred_classes = _text_area_graph_root_text_unit_classes(container)
    if not preferred_classes:
        return []
    output: List[List[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        class_name = str(entry.get("class_name") or "")
        if class_name not in preferred_classes:
            continue
        bbox = _semantic_unit_bbox_from_evidence(entry)
        if not bbox:
            continue
        if not _text_area_graph_container_can_claim_root_text_unit(container, bbox):
            continue
        if _text_area_graph_text_unit_better_owned_outside_group(
            bbox,
            container,
            active_containers=active_containers,
            group_container_ids=group_container_ids,
        ):
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        output.append(bbox)
    return output


def _text_area_graph_root_text_unit_classes(container: TextAreaContainer | Mapping[str, Any]) -> set[str]:
    role = _text_area_graph_semantic_role(container)
    if role == SEMANTIC_KIND_SPEECH:
        return {"text_bubble"}
    if role in {SEMANTIC_KIND_BACKGROUND_NARRATION, SEMANTIC_KIND_CAPTION}:
        return {"text_free", "text_bubble"}
    return set()


def _text_area_graph_container_can_claim_root_text_unit(
    container: TextAreaContainer | Mapping[str, Any],
    bbox: Sequence[Any],
) -> bool:
    container_bbox = _text_area_graph_container_bbox(container)
    return (
        _inside_ratio_xywh(bbox, container_bbox) >= 0.25
        or _text_area_graph_bbox_center_inside_xywh(bbox, container_bbox)
    )


def _text_area_graph_text_unit_better_owned_outside_group(
    bbox: Sequence[Any],
    container: TextAreaContainer | Mapping[str, Any],
    *,
    active_containers: Sequence[TextAreaContainer | Mapping[str, Any]],
    group_container_ids: set[str],
) -> bool:
    owner_score = _text_area_graph_root_text_unit_owner_score(container, bbox)
    if owner_score <= 0.0:
        return True
    owner_id = _text_area_graph_container_id(container)
    owner_role = _text_area_graph_semantic_role(container)
    for other in active_containers:
        other_id = _text_area_graph_container_id(other)
        if not other_id or other_id == owner_id or other_id in group_container_ids:
            continue
        if _text_area_graph_semantic_role(other) != owner_role:
            continue
        other_score = _text_area_graph_root_text_unit_owner_score(other, bbox)
        if other_score > owner_score + 0.05:
            return True
    return False


def _text_area_graph_root_text_unit_owner_score(
    container: TextAreaContainer | Mapping[str, Any],
    bbox: Sequence[Any],
) -> float:
    container_bbox = _text_area_graph_container_bbox(container)
    overlap = _inside_ratio_xywh(bbox, container_bbox)
    center_bonus = 1.0 if _text_area_graph_bbox_center_inside_xywh(bbox, container_bbox) else 0.0
    return center_bonus + overlap


def _text_area_graph_bbox_center_inside_xywh(inner: Sequence[Any], outer: Sequence[Any]) -> bool:
    ix, iy, iw, ih = _coerce_xywh(inner)
    ox, oy, ow, oh = _coerce_xywh(outer)
    if iw <= 0 or ih <= 0 or ow <= 0 or oh <= 0:
        return False
    cx = ix + iw / 2.0
    cy = iy + ih / 2.0
    return float(ox) <= cx <= float(ox + ow) and float(oy) <= cy <= float(oy + oh)


def _text_area_graph_is_deterministic_background_root(container: TextAreaContainer | Mapping[str, Any]) -> bool:
    role = _text_area_graph_semantic_role(container)
    if role not in {SEMANTIC_KIND_BACKGROUND_NARRATION, SEMANTIC_KIND_CAPTION}:
        return False
    evidence = _container_semantic_role_evidence(container)
    reasons = _text_area_graph_reason_codes(container)
    authority_kind = str(evidence.get("authority_evidence_kind") or "")
    return (
        "typed_deterministic_side_narration_background_authority" in authority_kind
        or "text_area_plan:deterministic_vertical_side_caption_search" in reasons
        or "text_area_plan:vertical_side_caption_authorized_columns_grouped_root" in reasons
    )


def _text_area_graph_root_evidence_ids(container: TextAreaContainer | Mapping[str, Any]) -> set[str]:
    evidence_ids = set(_text_area_graph_container_source_ids(container))
    evidence_ids.update(_text_area_graph_support_geometry_ids(container))
    role_evidence = _container_semantic_role_evidence(container)
    for key in ("text_unit_evidence_bboxes", "model_evidence_bboxes", "speech_mask_polygons"):
        entries = role_evidence.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            evidence_id = str(entry.get("evidence_id") or entry.get("model_evidence_id") or "")
            if evidence_id:
                evidence_ids.add(evidence_id)
    return {item for item in evidence_ids if item}


def _text_area_graph_workflow_roots_should_merge(
    first: TextAreaContainer | Mapping[str, Any],
    second: TextAreaContainer | Mapping[str, Any],
) -> bool:
    first_role = _text_area_graph_semantic_role(first)
    second_role = _text_area_graph_semantic_role(second)
    if first_role != second_role:
        return False
    if first_role != SEMANTIC_KIND_SPEECH:
        return False
    first_bbox = _text_area_graph_container_bbox(first)
    second_bbox = _text_area_graph_container_bbox(second)
    metrics = _text_area_graph_axis_overlap_metrics(first_bbox, second_bbox)
    if not metrics:
        return False
    shared_evidence = bool(_text_area_graph_root_evidence_ids(first) & _text_area_graph_root_evidence_ids(second))
    if not shared_evidence and not _text_area_graph_has_typed_speech_mask_seam(first, second):
        return False
    if metrics["intersection_min_ratio"] >= 0.55:
        return True
    if shared_evidence and metrics["intersection_min_ratio"] >= 0.12:
        return True
    if metrics["x_overlap_min_ratio"] >= 0.55 and metrics["y_overlap"] > 0:
        return True
    if (
        metrics["y_overlap_min_ratio"] >= 0.58
        and metrics["x_overlap_min_ratio"] >= 0.08
    ):
        first_text_ids = _text_area_graph_text_unit_evidence_ids(first)
        second_text_ids = _text_area_graph_text_unit_evidence_ids(second)
        if first_text_ids and second_text_ids and not (first_text_ids & second_text_ids) and not shared_evidence:
            first_area = _bbox_area_xywh(first_bbox)
            second_area = _bbox_area_xywh(second_bbox)
            area_ratio = min(first_area, second_area) / max(1.0, float(max(first_area, second_area)))
            if area_ratio < 0.35:
                return False
        if _text_area_graph_shared_text_unit_evidence_ids(first, second):
            return False
        return True
    if shared_evidence and metrics["x_overlap_min_ratio"] >= 0.55 and metrics["y_gap"] <= metrics["small_axis_gap"]:
        return True
    return False


def _text_area_graph_has_text_unit_evidence(container: TextAreaContainer | Mapping[str, Any]) -> bool:
    entries = _container_semantic_role_evidence(container).get("text_unit_evidence_bboxes") or []
    return isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)) and bool(entries)


def _text_area_graph_shared_text_unit_evidence_ids(
    first: TextAreaContainer | Mapping[str, Any],
    second: TextAreaContainer | Mapping[str, Any],
) -> set[str]:
    return _text_area_graph_text_unit_evidence_ids(first) & _text_area_graph_text_unit_evidence_ids(second)


def _text_area_graph_text_unit_evidence_ids(container: TextAreaContainer | Mapping[str, Any]) -> set[str]:
    entries = _container_semantic_role_evidence(container).get("text_unit_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return set()
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        evidence_id = str(entry.get("evidence_id") or entry.get("model_evidence_id") or "")
        if evidence_id:
            ids.add(evidence_id)
    return ids


def _text_area_graph_axis_overlap_metrics(
    first_bbox: Sequence[Any],
    second_bbox: Sequence[Any],
) -> Dict[str, float]:
    ax, ay, aw, ah = _coerce_xywh(first_bbox)
    bx, by, bw, bh = _coerce_xywh(second_bbox)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return {}
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    x_overlap = max(0, min(ax1, bx1) - max(ax, bx))
    y_overlap = max(0, min(ay1, by1) - max(ay, by))
    x_gap = max(0, max(ax, bx) - min(ax1, bx1))
    y_gap = max(0, max(ay, by) - min(ay1, by1))
    min_width = float(max(1, min(aw, bw)))
    min_height = float(max(1, min(ah, bh)))
    inter_area = float(x_overlap * y_overlap)
    min_area = float(max(1, min(aw * ah, bw * bh)))
    return {
        "x_overlap": float(x_overlap),
        "y_overlap": float(y_overlap),
        "x_gap": float(x_gap),
        "y_gap": float(y_gap),
        "x_overlap_min_ratio": float(x_overlap) / min_width,
        "y_overlap_min_ratio": float(y_overlap) / min_height,
        "intersection_min_ratio": inter_area / min_area,
        "small_axis_gap": max(4.0, min(min_width, min_height) * 0.04),
    }


def _text_area_graph_plan_page_id(plan: "TextAreaPlan | Mapping[str, Any]") -> str:
    if isinstance(plan, Mapping):
        return str(plan.get("page_id") or "")
    return str(getattr(plan, "page_id", "") or "")


def _text_area_graph_plan_containers(plan: "TextAreaPlan | Mapping[str, Any]") -> List[TextAreaContainer | Mapping[str, Any]]:
    if isinstance(plan, Mapping):
        return [item for item in plan.get("containers", []) or [] if isinstance(item, Mapping)]
    return [item for item in getattr(plan, "containers", []) or []]


def _text_area_graph_plan_image_size(plan: "TextAreaPlan | Mapping[str, Any]") -> Tuple[int, int]:
    raw = plan.get("image_size") if isinstance(plan, Mapping) else getattr(plan, "image_size", None)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        try:
            return max(1, int(raw[0])), max(1, int(raw[1]))
        except Exception:
            pass
    return 1, 1


def _text_area_graph_plan_luma_image(plan: "TextAreaPlan | Mapping[str, Any]") -> Any:
    image_path = plan.get("image_path") if isinstance(plan, Mapping) else getattr(plan, "image_path", "")
    if not image_path:
        return None
    return _load_luma_image(str(image_path))


def _text_area_graph_node_id(prefix: str, *parts: Any) -> str:
    tokens = [_text_area_graph_token(part) for part in parts if str(part)]
    return "_".join([prefix] + tokens)


def _text_area_graph_token(value: Any) -> str:
    text = str(value or "")
    output = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            output.append(char)
        else:
            output.append("_")
    token = "".join(output).strip("_")
    return token or "unknown"


def _text_area_graph_container_id(container: TextAreaContainer | Mapping[str, Any]) -> str:
    return str(_container_value(container, "container_id", "") or "")


def _text_area_graph_container_bbox(container: TextAreaContainer | Mapping[str, Any]) -> List[int]:
    return list(_container_value(container, "bbox", []) or [])


def _text_area_graph_container_source_ids(container: TextAreaContainer | Mapping[str, Any]) -> List[str]:
    role_evidence = _container_semantic_role_evidence(container)
    semantic_ids = [
        str(record.get("evidence_id") or "")
        for record in _semantic_evidence_records(role_evidence)
        if str(record.get("evidence_id") or "")
    ]
    return _text_area_graph_unique_strings(
        _container_list_value(container, "source_model_ids")
        + _container_list_value(container, "semantic_evidence_ids")
        + semantic_ids
    )


def _text_area_graph_support_geometry_ids(container: TextAreaContainer | Mapping[str, Any]) -> List[str]:
    output: List[str] = []
    role_evidence = _container_semantic_role_evidence(container)
    for key in (
        "text_unit_evidence_bboxes",
        "speech_mask_polygons",
        "component_evidence_ids",
        "assigned_component_ids",
        "assigned_region_ids",
    ):
        value = role_evidence.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for entry in value:
                if isinstance(entry, Mapping):
                    evidence_id = str(entry.get("evidence_id") or entry.get("component_id") or entry.get("region_id") or "")
                    if evidence_id:
                        output.append(evidence_id)
                elif str(entry):
                    output.append(str(entry))
        elif value:
            output.append(str(value))
    return _text_area_graph_unique_strings(output)


def _text_area_graph_reason_codes(container: TextAreaContainer | Mapping[str, Any]) -> List[str]:
    role_evidence = _container_semantic_role_evidence(container)
    return _text_area_graph_unique_strings(
        _container_list_value(container, "evidence_reason_codes")
        + _container_list_value(container, "conflict_flags")
        + _semantic_role_values(role_evidence, "typed_authority_reason_codes")
    )


def _text_area_graph_is_workflow_container(container: TextAreaContainer | Mapping[str, Any]) -> bool:
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    route = str(_container_value(container, "route_intent", "") or "")
    ctype = str(_container_value(container, "container_type", "") or "")
    if _component_auth_family(auth) == "cleanup":
        return True
    if auth and _component_auth_family(auth) != "cleanup":
        return False
    if route in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION} and ctype in {CONTAINER_SPEECH, CONTAINER_CAPTION}:
        return True
    return False


def _text_area_graph_is_excluded_container(container: TextAreaContainer | Mapping[str, Any]) -> bool:
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    route = str(_container_value(container, "route_intent", "") or "")
    ctype = str(_container_value(container, "container_type", "") or "")
    return (
        _component_auth_family(auth) in {"protected", "outside"}
        or route == ROUTE_PRESERVE_SFX
        or ctype == CONTAINER_SFX
        or bool(_container_value(container, "must_not_mutate", False))
    )


def _text_area_graph_exclusion_kind(container: TextAreaContainer | Mapping[str, Any]) -> str:
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    if auth == AUTH_PROTECT_ART_OR_NON_TEXT:
        return "art_or_non_text"
    if auth == AUTH_OUTSIDE_CLEANUP_SCOPE:
        return "outside_cleanup_scope"
    if auth == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        return "ambiguous_component_owner"
    return "sfx_decorative_or_protected"


def _text_area_graph_semantic_role(container: TextAreaContainer | Mapping[str, Any]) -> str:
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    ctype = str(_container_value(container, "container_type", "") or "")
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH or ctype == CONTAINER_SPEECH:
        return SEMANTIC_KIND_SPEECH
    if auth == AUTH_CLEANUP_TRANSLATE_CAPTION:
        return SEMANTIC_KIND_CAPTION
    if auth == AUTH_CLEANUP_TRANSLATE_BACKGROUND or ctype == CONTAINER_CAPTION:
        return SEMANTIC_KIND_BACKGROUND_NARRATION
    return SEMANTIC_KIND_UNKNOWN


def _text_area_graph_parent_boundary_entries(
    container: TextAreaContainer | Mapping[str, Any],
    *,
    image_size: Tuple[int, int] = (1, 1),
    luma_image: Any = None,
) -> List[Dict[str, Any]]:
    ctd_entries = _text_area_graph_ctd_parent_boundary_entries(container)
    if ctd_entries:
        entries = ctd_entries
    elif isinstance(container, TextAreaContainer):
        entries = _semantic_unit_evidence_bboxes_for_container(container)
    else:
        entries = _text_area_graph_mapping_parent_boundary_entries(container)
    if not entries:
        entries = _linked_text_free_speech_parent_boundary_entries(container)
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        bbox = _text_area_graph_bbox_from_entry(entry)
        if not bbox:
            continue
        evidence_id = str(entry.get("evidence_id") or "")
        key = evidence_id or ",".join(str(item) for item in bbox)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "evidence_id": evidence_id,
                "class_name": str(entry.get("class_name") or ""),
                "bbox": bbox,
                "boundary_source": str(entry.get("boundary_source") or TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT),
                "container_overlap_ratio": float(entry.get("container_overlap_ratio") or 0.0),
                "is_explicit_parent_boundary_evidence": bool(
                    entry.get("is_explicit_parent_boundary_evidence", False)
                ),
            }
        )
    return output


def _text_area_graph_refine_semantic_parent_boundary_entries(
    container: TextAreaContainer | Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    image_size: Tuple[int, int],
    luma_image: Any,
) -> List[Dict[str, Any]]:
    """Split a broad model text envelope only when internal ink proves islands."""

    output = [dict(entry) for entry in entries if isinstance(entry, Mapping)]
    if len(output) != 1 or luma_image is None or np is None:
        return output
    entry = output[0]
    if not bool(entry.get("is_explicit_parent_boundary_evidence", False)):
        return output
    if str(entry.get("class_name") or "") != "text_bubble":
        return output
    if _text_area_graph_semantic_role(container) != SEMANTIC_KIND_SPEECH:
        return output
    bbox = _text_area_graph_bbox_from_entry(entry)
    if not bbox:
        return output
    split_bboxes = _text_area_graph_split_text_unit_bbox_by_horizontal_ink_gap(
        luma_image=luma_image,
        root_bbox=_text_area_graph_container_bbox(container),
        bbox=bbox,
        image_size=image_size,
    )
    if len(split_bboxes) < 2:
        return output
    evidence_id = str(entry.get("evidence_id") or "text_unit")
    refined: List[Dict[str, Any]] = []
    for index, split_bbox in enumerate(split_bboxes):
        refined_entry = dict(entry)
        refined_entry["bbox"] = list(split_bbox)
        refined_entry["evidence_id"] = f"{evidence_id}_island_{index:02d}"
        refined_entry["source_text_unit_evidence_id"] = evidence_id
        refined_entry["boundary_source"] = TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT_REFINED_ISLAND
        refined_entry["is_explicit_parent_boundary_evidence"] = True
        refined.append(refined_entry)
    return refined


def _text_area_graph_split_text_unit_bbox_by_horizontal_ink_gap(
    *,
    luma_image: Any,
    root_bbox: Sequence[Any],
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> List[List[int]]:
    if luma_image is None or np is None:
        return []
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    rx, ry, rw, rh = _coerce_xywh(_normalize_xywh(root_bbox, image_size))
    bx, by, bw, bh = _coerce_xywh(_normalize_xywh(bbox, image_size))
    if rw <= 0 or rh <= 0 or bw <= 0 or bh <= 0:
        return []
    x0 = max(0, rx, bx)
    y0 = max(0, ry, by)
    x1 = min(width, rx + rw, bx + bw)
    y1 = min(height, ry + rh, by + bh)
    crop_w, crop_h = int(x1 - x0), int(y1 - y0)
    if crop_w < 36 or crop_h < 96:
        return []
    try:
        crop = luma_image.crop((x0, y0, x1, y1)).convert("L")
        arr = np.asarray(crop)
    except Exception:
        return []
    if arr.size <= 0:
        return []
    mean_luma = float(arr.mean())
    dark_threshold = 150 if mean_luma >= 135.0 else 112
    mask = arr <= dark_threshold
    ink_count = int(mask.sum())
    if ink_count < max(64, int(round(crop_w * crop_h * 0.002))):
        return []

    row_counts = mask.sum(axis=1)
    row_threshold = max(2, int(round(crop_w * 0.012)))
    active = [bool(value > row_threshold) for value in row_counts]
    bands: List[Tuple[int, int]] = []
    start: int | None = None
    for row_index, value in enumerate(active):
        if value and start is None:
            start = row_index
        if (not value or row_index == len(active) - 1) and start is not None:
            end = row_index - 1 if not value else row_index
            if end >= start:
                bands.append((start, end))
            start = None
    if len(bands) < 2:
        return []

    small_gap = max(4, int(round(crop_h * 0.014)))
    merged_bands: List[Tuple[int, int]] = []
    for band_start, band_end in bands:
        if merged_bands and band_start - merged_bands[-1][1] - 1 <= small_gap:
            prev_start, _prev_end = merged_bands[-1]
            merged_bands[-1] = (prev_start, band_end)
        else:
            merged_bands.append((band_start, band_end))
    if len(merged_bands) < 2:
        return []

    min_split_gap = max(18, int(round(crop_h * 0.055)))
    min_side_ink = max(60, int(round(ink_count * 0.08)))
    min_side_height = max(28, int(round(crop_h * 0.10)))
    split_gaps: List[Tuple[int, int]] = []
    for current, nxt in zip(merged_bands, merged_bands[1:]):
        gap_start = current[1] + 1
        gap_end = nxt[0] - 1
        gap = gap_end - gap_start + 1
        if gap < min_split_gap:
            continue
        upper = mask[:gap_start, :]
        lower = mask[gap_end + 1 :, :]
        if int(upper.sum()) < min_side_ink or int(lower.sum()) < min_side_ink:
            continue
        if gap_start < min_side_height or crop_h - gap_end - 1 < min_side_height:
            continue
        split_gaps.append((gap_start, gap_end))
    if not split_gaps:
        return []

    ranges: List[Tuple[int, int]] = []
    current_start = 0
    for gap_start, gap_end in split_gaps:
        ranges.append((current_start, gap_start - 1))
        current_start = gap_end + 1
    ranges.append((current_start, crop_h - 1))

    pad_x = max(3, min(12, int(round(crop_w * 0.04))))
    pad_y = max(3, min(14, int(round(crop_h * 0.025))))
    min_candidate_width = max(32, int(round(crop_w * 0.18)))
    min_candidate_height = max(52, int(round(crop_h * 0.16)))
    split_bboxes: List[List[int]] = []
    for seg_start, seg_end in ranges:
        if seg_end <= seg_start:
            continue
        segment_mask = mask[seg_start : seg_end + 1, :]
        if int(segment_mask.sum()) < min_side_ink:
            continue
        ys, xs = np.where(segment_mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        local_x0 = max(0, int(xs.min()) - pad_x)
        local_y0 = max(0, seg_start + int(ys.min()) - pad_y)
        local_x1 = min(crop_w, int(xs.max()) + 1 + pad_x)
        local_y1 = min(crop_h, seg_start + int(ys.max()) + 1 + pad_y)
        if local_x1 <= local_x0 or local_y1 <= local_y0:
            continue
        candidate = [
            int(x0 + local_x0),
            int(y0 + local_y0),
            int(local_x1 - local_x0),
            int(local_y1 - local_y0),
        ]
        if candidate[2] < min_candidate_width or candidate[3] < min_candidate_height:
            continue
        split_bboxes.append(candidate)

    if len(split_bboxes) < 2:
        return []
    return split_bboxes


def _text_area_graph_visual_parent_boundary_entries(
    container: TextAreaContainer | Mapping[str, Any],
    *,
    image_size: Tuple[int, int],
    luma_image: Any,
) -> List[Dict[str, Any]]:
    if luma_image is None or np is None or cv2 is None:
        return []
    if not _text_area_graph_visual_parent_provider_allowed(container):
        return []
    root_bbox = _text_area_graph_container_bbox(container)
    role = _text_area_graph_semantic_role(container)
    island_bboxes = _text_area_graph_root_local_visual_text_island_bboxes(
        luma_image=luma_image,
        root_bbox=root_bbox,
        image_size=image_size,
    )
    island_bboxes = _text_area_graph_filter_visual_parent_islands(root_bbox, island_bboxes)
    if not island_bboxes:
        return []
    container_id = _text_area_graph_container_id(container)
    class_name = "text_bubble" if role == SEMANTIC_KIND_SPEECH else "text_free"
    explicit = len(island_bboxes) == 1
    entries: List[Dict[str, Any]] = []
    for index, bbox in enumerate(island_bboxes):
        entries.append(
            {
                "bbox": bbox,
                "evidence_id": f"{container_id}_visual_text_island_{index:02d}" if container_id else f"visual_text_island_{index:02d}",
                "class_name": class_name,
                "boundary_source": TEXT_AREA_GRAPH_PARENT_SOURCE_ROOT_VISUAL_TEXT_ISLAND,
                "container_overlap_ratio": 1.0,
                "is_explicit_parent_boundary_evidence": explicit,
            }
        )
    return entries


def _text_area_graph_filter_visual_parent_islands(
    root_bbox: Sequence[Any],
    island_bboxes: Sequence[Sequence[Any]],
) -> List[List[int]]:
    rx, ry, rw, rh = _coerce_xywh(root_bbox)
    if rw <= 0 or rh <= 0:
        return []
    root_area = float(max(1, rw * rh))
    filtered: List[List[int]] = []
    for raw_bbox in island_bboxes:
        bbox = _text_area_graph_bbox_from_entry({"bbox": raw_bbox})
        if not bbox:
            continue
        x, y, w, h = _coerce_xywh(bbox)
        area_ratio = (w * h) / root_area
        width_ratio = w / float(max(1, rw))
        height_ratio = h / float(max(1, rh))
        touches_left_border = x <= rx + 1
        touches_right_border = x + w >= rx + rw - 1
        touches_top_border = y <= ry + 1
        touches_bottom_border = y + h >= ry + rh - 1
        touches_vertical_border = touches_left_border or touches_right_border
        touches_horizontal_border = touches_top_border or touches_bottom_border
        touches_border_count = sum(
            bool(value)
            for value in (
                touches_left_border,
                touches_right_border,
                touches_top_border,
                touches_bottom_border,
            )
        )
        if area_ratio < 0.018:
            continue
        if h < max(28, int(round(rh * 0.12))):
            continue
        if (
            touches_border_count >= 2
            and area_ratio >= 0.10
            and (height_ratio >= 0.72 or width_ratio >= 0.72)
        ):
            continue
        if touches_vertical_border and height_ratio >= 0.72 and width_ratio >= 0.24:
            continue
        if touches_horizontal_border and width_ratio >= 0.72 and height_ratio >= 0.24:
            continue
        if touches_vertical_border and width_ratio < 0.18:
            continue
        if touches_horizontal_border and height_ratio < 0.18:
            continue
        filtered.append(bbox)
    return filtered


def _text_area_graph_visual_parent_provider_allowed(container: TextAreaContainer | Mapping[str, Any]) -> bool:
    if bool(_container_value(container, "must_not_mutate", False)):
        return False
    ctype = str(_container_value(container, "container_type", "") or "")
    if ctype != CONTAINER_SPEECH:
        return False
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    route = str(_container_value(container, "route_intent", "") or "")
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH:
        return True
    return route == ROUTE_TRANSLATE_SPEECH


def _text_area_graph_root_local_visual_text_island_bboxes(
    *,
    luma_image: Any,
    root_bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> List[List[int]]:
    root = _normalize_xywh(root_bbox, image_size)
    rx, ry, rw, rh = _coerce_xywh(root)
    if rw <= 4 or rh <= 4:
        return []
    try:
        crop = luma_image.crop((rx, ry, rx + rw, ry + rh)).convert("L")
        arr = np.asarray(crop)
        crop_h, crop_w = int(arr.shape[0]), int(arr.shape[1])
        if crop_w <= 4 or crop_h <= 4:
            return []
        mean_luma = float(arr.mean())
        dark_mask = arr <= (150 if mean_luma >= 135.0 else 112)
        if float(dark_mask.sum()) / float(max(1, crop_w * crop_h)) < 0.0015 and mean_luma <= 160.0:
            mask = (arr >= 190).astype("uint8")
        else:
            mask = dark_mask.astype("uint8")
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return []

    root_area = max(1, crop_w * crop_h)
    components: List[Dict[str, Any]] = []
    min_area = max(4, int(root_area * 0.00008))
    max_area = max(140, int(root_area * 0.16))
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, area = [int(value) for value in stats[label][:5]]
        if area < min_area or area > max_area:
            continue
        if cw < 2 or ch < 2:
            continue
        density = area / float(max(1, cw * ch))
        if density < 0.035:
            continue
        span_w = cw / float(max(1, crop_w))
        span_h = ch / float(max(1, crop_h))
        touches_border = cx <= 1 or cy <= 1 or cx + cw >= crop_w - 1 or cy + ch >= crop_h - 1
        if span_w >= 0.72 and span_h >= 0.72:
            continue
        if touches_border and (span_w >= 0.48 or span_h >= 0.48):
            continue
        if span_w >= 0.62 and ch <= max(4, int(crop_h * 0.12)):
            continue
        if span_h >= 0.78 and cw <= max(4, int(crop_w * 0.08)):
            continue
        page_bbox = [rx + cx, ry + cy, cw, ch]
        components.append(
            {
                "bbox": page_bbox,
                "center_x": rx + cx + cw / 2.0,
                "center_y": ry + cy + ch / 2.0,
                "width": cw,
                "height": ch,
                "area": area,
            }
        )
    if not components:
        return []

    median_width = _median_float([float(item["width"]) for item in components]) or 1.0
    median_height = _median_float([float(item["height"]) for item in components]) or 1.0
    column_threshold = max(8.0, median_width * 2.4, rw * 0.025)
    columns: List[Dict[str, Any]] = []
    for component in sorted(components, key=lambda item: float(item["center_x"]), reverse=True):
        placed = False
        for column in columns:
            if abs(float(component["center_x"]) - float(column["center_x"])) <= column_threshold:
                column["items"].append(component)
                centers = [float(item["center_x"]) for item in column["items"]]
                column["center_x"] = sum(centers) / float(len(centers))
                placed = True
                break
        if not placed:
            columns.append({"center_x": float(component["center_x"]), "items": [component]})

    segment_gap_y = max(median_height * 2.8, rh * 0.10)
    segments: List[Dict[str, Any]] = []
    for column_index, column in enumerate(columns):
        items = sorted(list(column.get("items") or []), key=lambda item: _coerce_xywh(item["bbox"])[1])
        if not items:
            continue
        current = [items[0]]
        for item in items[1:]:
            _px, py, _pw, ph = _coerce_xywh(current[-1]["bbox"])
            _ix, iy, _iw, _ih = _coerce_xywh(item["bbox"])
            y_gap = iy - (py + ph)
            if y_gap > segment_gap_y:
                segments.append(_text_area_graph_segment_from_components(current, column_index))
                current = [item]
            else:
                current.append(item)
        if current:
            segments.append(_text_area_graph_segment_from_components(current, column_index))
    if not segments:
        return []

    merge_x_gap = max(median_width * 3.0, rw * 0.12)
    merge_y_gap = max(median_height * 4.0, rh * 0.08)
    parents = list(range(len(segments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for left_index in range(len(segments)):
        for right_index in range(left_index + 1, len(segments)):
            if _text_area_graph_segments_should_merge(
                segments[left_index],
                segments[right_index],
                max_x_gap=merge_x_gap,
                max_y_gap=merge_y_gap,
            ):
                union(left_index, right_index)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for index, segment in enumerate(segments):
        grouped.setdefault(find(index), []).append(segment)

    root_x1, root_y1 = rx + rw, ry + rh
    bboxes: List[List[int]] = []
    for group_segments in grouped.values():
        component_bboxes: List[List[int]] = []
        for segment in group_segments:
            for component in segment.get("components") or []:
                bbox = component.get("bbox") if isinstance(component, Mapping) else None
                if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
                    component_bboxes.append(list(bbox))
        if not component_bboxes:
            continue
        bbox = _text_area_graph_union_xywh(component_bboxes)
        if not bbox:
            continue
        pad_x = max(2, int(round(median_width * 0.75)))
        pad_y = max(2, int(round(median_height * 0.75)))
        x, y, w, h = _coerce_xywh(bbox)
        x0 = max(rx, x - pad_x)
        y0 = max(ry, y - pad_y)
        x1 = min(root_x1, x + w + pad_x)
        y1 = min(root_y1, y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            continue
        candidate = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
        coverage_w = candidate[2] / float(max(1, rw))
        coverage_h = candidate[3] / float(max(1, rh))
        if coverage_w >= 0.84 and coverage_h >= 0.84:
            continue
        bboxes.append(candidate)

    bboxes.sort(key=lambda item: (-(item[0] + item[2] / 2.0), item[1], item[2] * item[3]))
    deduped: List[List[int]] = []
    for bbox in bboxes:
        if any(_intersection_ratio_xywh(bbox, existing) >= 0.72 for existing in deduped):
            continue
        deduped.append(bbox)
    return deduped


def _text_area_graph_segment_from_components(
    components: Sequence[Mapping[str, Any]],
    column_index: int,
) -> Dict[str, Any]:
    bboxes = [list(item.get("bbox") or []) for item in components if isinstance(item.get("bbox"), Sequence)]
    bbox = _text_area_graph_union_xywh(bboxes)
    return {
        "bbox": bbox,
        "components": list(components),
        "column_index": column_index,
    }


def _text_area_graph_segments_should_merge(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    max_x_gap: float,
    max_y_gap: float,
) -> bool:
    ax, ay, aw, ah = _coerce_xywh(first.get("bbox") or [])
    bx, by, bw, bh = _coerce_xywh(second.get("bbox") or [])
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    x_gap = max(0, max(ax, bx) - min(ax1, bx1))
    if x_gap > max_x_gap:
        return False
    y_overlap = max(0, min(ay1, by1) - max(ay, by))
    y_overlap_ratio = y_overlap / float(max(1, min(ah, bh)))
    y_gap = max(0, max(ay, by) - min(ay1, by1))
    return y_overlap_ratio >= 0.25 or y_gap <= max_y_gap


def _text_area_graph_union_xywh(bboxes: Sequence[Sequence[Any]]) -> List[int]:
    normalized: List[Tuple[int, int, int, int]] = []
    for bbox in bboxes:
        x, y, w, h = _coerce_xywh(bbox)
        if w <= 0 or h <= 0:
            continue
        normalized.append((x, y, x + w, y + h))
    if not normalized:
        return []
    x0 = min(item[0] for item in normalized)
    y0 = min(item[1] for item in normalized)
    x1 = max(item[2] for item in normalized)
    y1 = max(item[3] for item in normalized)
    return [int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0))]


def _median_float(values: Sequence[float]) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return 0.0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def _text_area_graph_mapping_parent_boundary_entries(container: Mapping[str, Any]) -> List[Dict[str, Any]]:
    role_evidence = _container_semantic_role_evidence(container)
    entries = role_evidence.get("text_unit_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH:
        preferred_classes = {"text_bubble"}
    elif auth in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        preferred_classes = {"text_free", "text_bubble"}
    else:
        preferred_classes = set()
    output: List[Dict[str, Any]] = []
    container_bbox = _text_area_graph_container_bbox(container)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        class_name = str(entry.get("class_name") or "")
        if preferred_classes and class_name not in preferred_classes:
            continue
        bbox = _semantic_unit_bbox_from_evidence(entry)
        if not bbox:
            continue
        overlap_ratio = _inside_ratio_xywh(bbox, container_bbox)
        if overlap_ratio < 0.25:
            continue
        output.append(
            {
                "bbox": bbox,
                "evidence_id": str(entry.get("evidence_id") or ""),
                "class_name": class_name,
                "container_overlap_ratio": round(overlap_ratio, 6),
                "is_explicit_parent_boundary_evidence": _text_area_graph_provider_authored_parent_boundary(
                    container,
                    entry,
                ),
            }
        )
    return output


def _text_area_graph_ctd_parent_boundary_entries(
    container: TextAreaContainer | Mapping[str, Any],
) -> List[Dict[str, Any]]:
    role_evidence = _container_semantic_role_evidence(container)
    entries = role_evidence.get("ctd_parent_boundary_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    output: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        bbox = _text_area_graph_bbox_from_entry(entry)
        if not bbox:
            continue
        output.append(
            {
                "bbox": bbox,
                "evidence_id": str(entry.get("evidence_id") or ""),
                "class_name": "ctd_text_block",
                "confidence": float(entry.get("confidence") or 0.0),
                "boundary_source": "ctd_text_block_evidence",
                "is_explicit_parent_boundary_evidence": True,
            }
        )
    return output


def _text_area_graph_provider_authored_parent_boundary(
    container: TextAreaContainer | Mapping[str, Any],
    entry: Mapping[str, Any],
) -> bool:
    if bool(entry.get("is_explicit_parent_boundary_evidence", False)):
        return True
    evidence_id = str(entry.get("evidence_id") or "").strip()
    class_name = str(entry.get("class_name") or "").strip()
    if not evidence_id or class_name not in {"text_bubble", "text_free"}:
        return False
    expected_provider = {
        "text_bubble": "ogkalu_text_bubble_evidence",
        "text_free": "ogkalu_text_free_background_evidence",
    }[class_name]
    role_evidence = _container_semantic_role_evidence(container)
    for record in _semantic_evidence_records(role_evidence):
        provider = str(record.get("provider") or "").strip()
        if provider != expected_provider:
            continue
        record_id = str(record.get("evidence_id") or "").strip()
        source_ids = {str(value) for value in record.get("source_model_ids") or [] if str(value)}
        if evidence_id in source_ids or record_id == f"{expected_provider}:{evidence_id}":
            return True
    return False


def _text_area_graph_has_typed_speech_mask_seam(
    first: TextAreaContainer | Mapping[str, Any],
    second: TextAreaContainer | Mapping[str, Any],
) -> bool:
    shared_context = _text_area_graph_shared_speech_context_ids(first, second)
    if len(shared_context) < 2:
        return False
    first_masks = _text_area_graph_typed_speech_mask_polygons(first)
    second_masks = _text_area_graph_typed_speech_mask_polygons(second)
    if not first_masks or not second_masks:
        return False
    for first_polygon in first_masks:
        for second_polygon in second_masks:
            first_bbox = polygon_to_bbox(first_polygon)
            second_bbox = polygon_to_bbox(second_polygon)
            dimensions = [
                first_bbox[2],
                first_bbox[3],
                second_bbox[2],
                second_bbox[3],
            ]
            seam_limit = max(4.0, min(float(value) for value in dimensions if value > 0) * 0.04)
            if _text_area_graph_polygon_distance(first_polygon, second_polygon) <= seam_limit:
                return True
    return False


def _text_area_graph_shared_speech_context_ids(
    first: TextAreaContainer | Mapping[str, Any],
    second: TextAreaContainer | Mapping[str, Any],
) -> set[str]:
    first_ids = set(
        _semantic_role_values(
            _container_semantic_role_evidence(first),
            "neighboring_speech_context_ids",
        )
    )
    second_ids = set(
        _semantic_role_values(
            _container_semantic_role_evidence(second),
            "neighboring_speech_context_ids",
        )
    )
    return first_ids & second_ids


def _text_area_graph_typed_speech_mask_polygons(
    container: TextAreaContainer | Mapping[str, Any],
) -> List[List[List[float]]]:
    role_evidence = _container_semantic_role_evidence(container)
    typed_ids = {
        str(item)
        for record in _semantic_evidence_records(role_evidence)
        if str(record.get("provider") or "") == PROVIDER_KITSUMED_SPEECH_MASK
        and str(record.get("semantic_target") or "") == SEMANTIC_KIND_SPEECH
        and str(record.get("authority_state") or "") == AUTH_CLEANUP_TRANSLATE_SPEECH
        and not bool(record.get("requires_review"))
        for item in record.get("source_model_ids") or []
        if str(item)
    }
    if not typed_ids:
        return []
    output: List[List[List[float]]] = []
    entries = role_evidence.get("speech_mask_polygons") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    for entry in entries:
        if not isinstance(entry, Mapping) or str(entry.get("evidence_id") or "") not in typed_ids:
            continue
        polygon = _text_area_graph_polygon_points(entry.get("polygon") or [])
        if len(polygon) >= 3:
            output.append(polygon)
    return output


def _text_area_graph_polygon_points(value: Any) -> List[List[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: List[List[float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            continue
        try:
            output.append([float(point[0]), float(point[1])])
        except Exception:
            continue
    return output


def _text_area_graph_polygon_distance(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    first_points = _text_area_graph_polygon_points(first)
    second_points = _text_area_graph_polygon_points(second)
    if len(first_points) < 2 or len(second_points) < 2:
        return float("inf")
    first_segments = list(zip(first_points, first_points[1:] + first_points[:1]))
    second_segments = list(zip(second_points, second_points[1:] + second_points[:1]))
    best = float("inf")
    for first_start, first_end in first_segments:
        for second_start, second_end in second_segments:
            if _text_area_graph_segments_intersect(first_start, first_end, second_start, second_end):
                return 0.0
            best = min(
                best,
                _text_area_graph_point_segment_distance(first_start, second_start, second_end),
                _text_area_graph_point_segment_distance(first_end, second_start, second_end),
                _text_area_graph_point_segment_distance(second_start, first_start, first_end),
                _text_area_graph_point_segment_distance(second_end, first_start, first_end),
            )
    return best


def _text_area_graph_segments_intersect(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> bool:
    def cross(origin: Sequence[float], first_point: Sequence[float], second_point: Sequence[float]) -> float:
        return (
            (float(first_point[0]) - float(origin[0])) * (float(second_point[1]) - float(origin[1]))
            - (float(first_point[1]) - float(origin[1])) * (float(second_point[0]) - float(origin[0]))
        )

    first_cross = cross(first_start, first_end, second_start)
    second_cross = cross(first_start, first_end, second_end)
    third_cross = cross(second_start, second_end, first_start)
    fourth_cross = cross(second_start, second_end, first_end)
    epsilon = 1e-6
    if (
        ((first_cross > epsilon and second_cross < -epsilon) or (first_cross < -epsilon and second_cross > epsilon))
        and ((third_cross > epsilon and fourth_cross < -epsilon) or (third_cross < -epsilon and fourth_cross > epsilon))
    ):
        return True
    return any(
        _text_area_graph_point_segment_distance(point, start, end) <= epsilon
        for point, start, end in (
            (second_start, first_start, first_end),
            (second_end, first_start, first_end),
            (first_start, second_start, second_end),
            (first_end, second_start, second_end),
        )
    )


def _text_area_graph_point_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(px - sx, py - sy)
    position = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_squared))
    nearest_x = sx + position * dx
    nearest_y = sy + position * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _text_area_graph_parent_boundary_key(entry: Mapping[str, Any]) -> str:
    evidence_id = str(entry.get("evidence_id") or "")
    bbox = _text_area_graph_bbox_from_entry(entry)
    if evidence_id:
        return "evidence:{evidence}|bbox:{bbox}".format(
            evidence=evidence_id,
            bbox=",".join(str(item) for item in bbox),
        )
    return "bbox:" + ",".join(str(item) for item in bbox)


def _text_area_graph_bbox_from_entry(entry: Mapping[str, Any]) -> List[int]:
    bbox = entry.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) < 4:
        return []
    try:
        x, y, w, h = [int(round(float(item or 0))) for item in bbox[:4]]
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    return [max(0, x), max(0, y), max(1, w), max(1, h)]


def _text_area_graph_container_local_bbox(child_bbox: Sequence[int], container_bbox: Sequence[int]) -> List[int]:
    x, y, w, h = _coerce_xywh(child_bbox)
    cx, cy, _, _ = _coerce_xywh(container_bbox)
    if w <= 0 or h <= 0:
        return []
    return [max(0, x - cx), max(0, y - cy), w, h]


def _text_area_graph_blocker_counts(blockers: Sequence[TextAreaGraphBlocker]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for blocker in blockers:
        counts[blocker.blocker_kind] = counts.get(blocker.blocker_kind, 0) + 1
    return counts


def _text_area_graph_candidate_state_counts(candidates: Sequence[TextAreaParentBoundaryCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        state = str(candidate.adjudication_state or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def _text_area_graph_parent_source_counts(parent_nodes: Sequence[TextAreaParentNode]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for parent in parent_nodes:
        source = str(parent.boundary_source or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _text_area_graph_unique_strings(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


@dataclass
class TextAreaPlan:
    page_id: str
    image_path: str
    image_size: Tuple[int, int]
    version: str = TEXT_AREA_PLAN_VERSION
    generated: bool = False
    containers: List[TextAreaContainer] = field(default_factory=list)
    scopes: List[TextAreaScope] = field(default_factory=list)
    semantic_units: List[TextAreaSemanticAuthorizationRecord] = field(default_factory=list)
    fallback_reasons: List[TextAreaFallbackReason] = field(default_factory=list)
    evidence: List[TextAreaPlanEvidence] = field(default_factory=list)
    runtime: TextAreaPlanRuntime = field(default_factory=TextAreaPlanRuntime)
    summary: Dict[str, Any] = field(default_factory=dict)
    stage: str = "pre_ocr"
    root_parent_child_plan: Optional[TextAreaRootParentChildPlan | Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "page_id": self.page_id,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "version": self.version,
            "generated": self.generated,
            "containers": [item.to_dict() for item in self.containers],
            "scopes": [item.to_dict() for item in self.scopes],
            "semantic_units": [item.to_dict() for item in self.semantic_units],
            "fallback_reasons": [item.to_dict() for item in self.fallback_reasons],
            "evidence": [item.to_dict() for item in self.evidence],
            "runtime": self.runtime.to_dict(),
            "summary": dict(self.summary),
            "stage": self.stage,
        }
        if self.root_parent_child_plan is not None:
            payload["root_parent_child_plan"] = _root_parent_child_plan_to_dict(self.root_parent_child_plan)
        return payload


@dataclass
class TextAreaComponentAuthorizationRecord:
    page_id: str
    component_id: str
    bbox: List[int]
    component_bbox: List[int]
    pixel_count: int
    component_pixel_count: int
    authorization_state: str
    cleanup_authorization: str
    route_intent: str
    must_not_mutate: bool
    owning_container_ids: List[str] = field(default_factory=list)
    protection_container_ids: List[str] = field(default_factory=list)
    cleanup_owner_ids: List[str] = field(default_factory=list)
    protection_owner_ids: List[str] = field(default_factory=list)
    source_stage: str = "text_area_plan_component_authorization"
    confidence_tier: str = "low"
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    review_required: bool = False
    visual_debug_color: str = "gray"
    group_id: str = ""
    group_authorization_state: str = ""
    label: int = 0
    owner_cleanup_job_id: str = ""
    scope_cleanup_job_ids: List[str] = field(default_factory=list)
    candidate_cleanup_job_ids: List[str] = field(default_factory=list)
    owning_region_ids: List[str] = field(default_factory=list)
    protection_region_ids: List[str] = field(default_factory=list)
    overlap_pixels: int = 0
    overlap_ratio: float = 0.0
    protected_overlap_pixels: int = 0
    protected_overlap_ratio: float = 0.0
    centroid: List[float] = field(default_factory=list)
    candidate_container_ids: List[str] = field(default_factory=list)
    candidate_region_ids: List[str] = field(default_factory=list)
    sourceglyph_overlap_pixels: int = 0
    sourceglyph_missing: bool = True
    ambiguity_reasons: List[str] = field(default_factory=list)
    unresolved_reason_codes: List[str] = field(default_factory=list)
    protected_reason_codes: List[str] = field(default_factory=list)
    ambiguous_reason_codes: List[str] = field(default_factory=list)
    semantic_authorization_state: str = ""
    semantic_visual_color: str = ""
    job_binding_state: str = ""
    job_binding_failure_reason: str = ""
    final_mask_authorization_state: str = ""
    authorization_warning_codes: List[str] = field(default_factory=list)
    component_contract_defect_codes: List[str] = field(default_factory=list)
    requires_visual_review: bool = False
    candidate_conflict_reason: str = ""
    explicit_cleanup_authority: bool = False
    explicit_protected_authority: bool = False
    explicit_authority_source: str = ""
    authorization_basis: str = ""
    authorization_explicit: bool = False
    authorization_field_origin: str = ""
    semantic_unit_ids: List[str] = field(default_factory=list)
    semantic_unit_states: List[str] = field(default_factory=list)
    semantic_kind: str = ""
    semantic_kinds: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    semantic_evidence_providers: List[str] = field(default_factory=list)
    semantic_evidence_ids: List[str] = field(default_factory=list)
    semantic_evidence_trace: List[Dict[str, Any]] = field(default_factory=list)
    semantic_authority_owner: str = "TextAreaPlan/BubbleDetection"
    projection_owner: str = "TextForegroundSegmentationMask/TextAreaPlan projection"
    projection_quality_state: str = PROJECTION_NO_SEMANTIC_AUTHORITY
    projection_quality_reasons: List[str] = field(default_factory=list)
    mask_readiness_state: str = MASK_NOT_READY
    mask_readiness_failure_reason: str = ""
    projected_label_ids: List[int] = field(default_factory=list)
    projection_overlap_pixels: int = 0
    projection_overlap_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaComponentAuthorizationMap:
    page_id: str
    image_size: Tuple[int, int] | List[int] | None
    version: str = TEXT_AREA_COMPONENT_AUTHORIZATION_MAP_VERSION
    components: List[TextAreaComponentAuthorizationRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "image_size": list(self.image_size) if self.image_size else None,
            "components": [record.to_dict() for record in self.components],
            "errors": list(self.errors),
            "summary": dict(self.summary),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_audit_dict()


def build_text_area_component_authorization_map(
    *,
    page_id: str,
    text_foreground_segmentation: Any,
    text_area_plan: Any = None,
    page_region_records: Sequence[Mapping[str, Any]] | None = None,
    cleanup_jobs: Sequence[Any] | None = None,
) -> TextAreaComponentAuthorizationMap:
    """Project TextAreaPlan semantic authorization onto CTD components.

    This is the only production handoff that decides whether a refined CTD
    component is cleanup-owned, protected, review-only, outside-scope, or
    ambiguous. CleanupMask consumes this map without reinterpreting semantics.
    """

    mask, image_size, mask_error = _component_auth_segmentation_mask(text_foreground_segmentation)
    errors: List[str] = []
    if mask_error:
        errors.append(mask_error)
    if mask is None or np is None:
        return TextAreaComponentAuthorizationMap(
            page_id=str(page_id),
            image_size=image_size,
            components=[],
            errors=errors or ["text_foreground_segmentation_missing"],
            summary={
                "component_count": 0,
                "state_counts": {},
                "semantic_authority": "text_area_plan",
                "segmentation_source": "text_foreground_segmentation",
                "component_classification_complete": False,
            },
        )

    binary = (mask > 0).astype(np.uint8)
    labels, stats, centroids = _component_auth_connected_components(binary)
    plan_dict = _component_auth_plan_dict(text_area_plan)
    scopes = _component_auth_scopes(
        plan=plan_dict,
        page_region_records=page_region_records or [],
        cleanup_jobs=cleanup_jobs or [],
        mask_shape=binary.shape,
    )
    cleanup_job_areas: List[Dict[str, Any]] = []
    for job in cleanup_jobs or []:
        cleanup_job_areas.extend(_component_auth_cleanup_job_area_records(job, binary.shape))

    components: List[TextAreaComponentAuthorizationRecord] = []
    for label in range(1, int(len(stats))):
        x, y, w, h, pixel_count = [int(item) for item in stats[label][:5]]
        if pixel_count <= 0:
            continue
        bbox = [x, y, x + w, y + h]
        candidates = _component_auth_candidates(labels, label, pixel_count, centroids[label], scopes, bbox)
        record = _component_auth_record(
            page_id=str(page_id),
            component_index=len(components),
            label=label,
            bbox=bbox,
            pixel_count=pixel_count,
            centroid=centroids[label],
            candidates=candidates,
        )
        components.append(record)

    _component_auth_apply_vertical_review_text_groups(components)
    _component_auth_apply_orphan_text_near_cleanup_groups(components)
    _component_auth_apply_ogkalu_speech_text_group_completion(components)
    _component_auth_apply_protected_sfx_grouping(components)
    _component_auth_apply_large_decorative_review_rule(components)
    _component_auth_apply_large_decorative_review_groups(components)
    _component_auth_apply_unowned_display_neighbor_conflicts(components)
    _component_auth_apply_protected_sfx_grouping(components)
    _component_auth_apply_review_only_caption_guard(components)
    _component_auth_apply_cleanup_obligation_area_bindings(components, cleanup_job_areas)
    _component_auth_apply_terminal_authority_guards(components)
    _component_auth_apply_side_caption_projection_guards(components)
    _component_auth_assign_groups(str(page_id), components)
    state_counts: Dict[str, int] = {}
    for record in components:
        state_counts[record.authorization_state] = state_counts.get(record.authorization_state, 0) + 1
    return TextAreaComponentAuthorizationMap(
        page_id=str(page_id),
        image_size=image_size or [int(binary.shape[1]), int(binary.shape[0])],
        components=components,
        errors=errors,
        summary={
            "component_count": len(components),
            "state_counts": state_counts,
            "semantic_authority": "text_area_plan",
            "segmentation_source": "text_foreground_segmentation",
            "component_classification_complete": True,
            "component_record_count_matches_ctd": True,
            "text_area_scope_count": len(scopes),
            "cleanup_job_count": len(cleanup_jobs or []),
        },
    )


def _component_auth_segmentation_mask(segmentation: Any) -> tuple[Any | None, List[int] | None, str]:
    if np is None:
        return None, None, "numpy_unavailable_for_component_authorization"
    image_size = _component_auth_image_size(segmentation)
    mask = _component_auth_get_value(segmentation, "refined_mask")
    if mask is None:
        mask = _component_auth_get_value(segmentation, "mask")
    if mask is None:
        mask_ref = str(
            _component_auth_get_value(segmentation, "refined_mask_ref")
            or _component_auth_get_value(segmentation, "mask_ref")
            or ""
        )
        if mask_ref:
            try:
                from PIL import Image

                path = Path(mask_ref)
                if path.exists():
                    mask = np.asarray(Image.open(path).convert("L"))
            except Exception:
                mask = None
    if mask is None:
        return None, image_size, "text_foreground_segmentation_missing_refined_mask"
    try:
        arr = np.asarray(mask)
    except Exception:
        return None, image_size, "text_foreground_segmentation_mask_unreadable"
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size <= 0:
        return None, image_size, "text_foreground_segmentation_mask_empty"
    if image_size is None:
        image_size = [int(arr.shape[1]), int(arr.shape[0])]
    return (arr > 0).astype(np.uint8), image_size, ""


def _component_auth_image_size(segmentation: Any) -> List[int] | None:
    value = _component_auth_get_value(segmentation, "image_size")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except Exception:
            return None
    return None


def _component_auth_get_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _component_auth_connected_components(binary: Any) -> tuple[Any, Any, Any]:
    if cv2 is not None:
        _label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return labels.astype(np.int32), stats, centroids
    height, width = binary.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    stats = [[0, 0, 0, 0, 0]]
    centroids = [[0.0, 0.0]]
    label = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x] <= 0 or labels[y, x] != 0:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            xs: List[int] = []
            ys: List[int] = []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx)
                ys.append(sy)
                for ny in range(max(0, sy - 1), min(height, sy + 2)):
                    for nx in range(max(0, sx - 1), min(width, sx + 2)):
                        if binary[ny, nx] > 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((nx, ny))
            stats.append([min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(xs)])
            centroids.append([sum(xs) / float(max(1, len(xs))), sum(ys) / float(max(1, len(ys)))])
    return labels, np.asarray(stats, dtype=np.int32), np.asarray(centroids, dtype=float)


def _component_auth_plan_dict(text_area_plan: Any) -> Dict[str, Any]:
    if text_area_plan is None:
        return {}
    if isinstance(text_area_plan, Mapping):
        return dict(text_area_plan)
    if hasattr(text_area_plan, "to_dict"):
        try:
            value = text_area_plan.to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            return {}
    return {}


def _component_auth_scopes(
    *,
    plan: Mapping[str, Any],
    page_region_records: Sequence[Mapping[str, Any]],
    cleanup_jobs: Sequence[Any],
    mask_shape: tuple[int, int],
) -> List[Dict[str, Any]]:
    region_jobs: Dict[str, List[str]] = {}
    container_jobs: Dict[str, List[str]] = {}
    source_evidence_jobs: Dict[str, List[Dict[str, Any]]] = {}
    cleanup_job_areas: List[Dict[str, Any]] = []
    for job in cleanup_jobs or []:
        job_id = str(getattr(job, "cleanup_job_id", "") or "")
        if not job_id:
            continue
        cleanup_job_areas.extend(_component_auth_cleanup_job_area_records(job, mask_shape))
        for region_id in getattr(job, "target_region_ids", []) or []:
            region_jobs.setdefault(str(region_id), []).append(job_id)
        container_id = str(getattr(job, "text_area_container_id", "") or "")
        if container_id:
            container_jobs.setdefault(container_id, []).append(job_id)
        for evidence in _component_auth_cleanup_job_source_evidence(job):
            evidence_container_id = str(
                _component_auth_mapping_get(
                    evidence,
                    "text_area_container_id",
                    "container_id",
                    "source_glyph_mask_text_area_container_id",
                )
                or ""
            )
            if not evidence_container_id:
                continue
            evidence_bbox = _component_auth_source_evidence_bbox(evidence, mask_shape)
            if not evidence_bbox:
                continue
            source_evidence_jobs.setdefault(evidence_container_id, []).append(
                {
                    "cleanup_job_id": job_id,
                    "bbox": evidence_bbox,
                    "region_id": str(_component_auth_mapping_get(evidence, "region_id", "target_region_id") or ""),
                    "source_glyph_mask_id": str(
                        _component_auth_mapping_get(evidence, "source_glyph_mask_id", "mask_id", "id") or ""
                    ),
                }
            )

    scopes: List[Dict[str, Any]] = []
    parent_execution_region_records = [
        region
        for region in page_region_records or []
        if isinstance(region, Mapping) and _component_auth_is_parent_execution_region(region)
    ]
    legacy_region_records = [
        region
        for region in page_region_records or []
        if isinstance(region, Mapping) and not _component_auth_is_parent_execution_region(region)
    ]
    semantic_unit_records = list(plan.get("semantic_units") or [])
    for index, semantic_unit in enumerate(semantic_unit_records):
        if isinstance(semantic_unit, Mapping):
            scope = _component_auth_scope(
                source=semantic_unit,
                source_kind="text_area_semantic_unit",
                fallback_id=f"semantic_unit_{index:04d}",
                bbox_mode="xywh",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
                source_evidence_jobs=source_evidence_jobs,
                cleanup_job_areas=cleanup_job_areas,
            )
            if scope:
                scopes.append(scope)
    for index, region in enumerate(parent_execution_region_records):
        scope = _component_auth_scope(
            source=region,
            source_kind="parent_execution_bundle",
            fallback_id=f"parent_execution_region_{index:04d}",
            bbox_mode="xywh",
            mask_shape=mask_shape,
            region_jobs=region_jobs,
            container_jobs=container_jobs,
            source_evidence_jobs=source_evidence_jobs,
            cleanup_job_areas=cleanup_job_areas,
        )
        if scope:
            scopes.append(scope)
    if semantic_unit_records:
        return scopes
    for index, container in enumerate(plan.get("containers") or []):
        if isinstance(container, Mapping):
            scope = _component_auth_scope(
                source=container,
                source_kind="text_area_container",
                fallback_id=f"container_{index:04d}",
                bbox_mode="xywh",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
                source_evidence_jobs=source_evidence_jobs,
                cleanup_job_areas=cleanup_job_areas,
            )
            if scope:
                scopes.append(scope)
    for index, scope_record in enumerate(plan.get("scopes") or []):
        if isinstance(scope_record, Mapping):
            scope = _component_auth_scope(
                source=scope_record,
                source_kind="text_area_scope",
                fallback_id=f"scope_{index:04d}",
                bbox_mode="xywh",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
                source_evidence_jobs=source_evidence_jobs,
                cleanup_job_areas=cleanup_job_areas,
            )
            if scope:
                scopes.append(scope)
    for index, region in enumerate(legacy_region_records):
        scope = _component_auth_scope(
            source=region,
            source_kind="region_record",
            fallback_id=f"region_{index:04d}",
            bbox_mode="region",
            mask_shape=mask_shape,
            region_jobs=region_jobs,
            container_jobs=container_jobs,
            source_evidence_jobs=source_evidence_jobs,
            cleanup_job_areas=cleanup_job_areas,
        )
        if scope:
            scopes.append(scope)
    return scopes


def _component_auth_is_parent_execution_region(region: Mapping[str, Any]) -> bool:
    return bool(
        region.get("parent_execution_bundle_id")
        or region.get("parent_execution_bundle_version")
    )


def _component_auth_binding_unit_id(
    *,
    source: Mapping[str, Any],
    source_kind: str,
    container_id: str,
    region_id: str,
    fallback_id: str,
) -> str:
    """Keep semantic-container identity separate from executable parent identity."""

    if source_kind == "parent_execution_bundle":
        return str(
            source.get("parent_execution_bundle_id")
            or source.get("parent_logical_text_unit_id")
            or source.get("logical_text_block_id")
            or region_id
            or fallback_id
        )
    return str(
        source.get("semantic_unit_id")
        or container_id
        or region_id
        or fallback_id
    )


def _component_auth_scope(
    *,
    source: Mapping[str, Any],
    source_kind: str,
    fallback_id: str,
    bbox_mode: str,
    mask_shape: tuple[int, int],
    region_jobs: Mapping[str, Sequence[str]],
    container_jobs: Mapping[str, Sequence[str]],
    source_evidence_jobs: Mapping[str, Sequence[Mapping[str, Any]]],
    cleanup_job_areas: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    bbox_value = source.get("bbox") or source.get("xyxy") or source.get("bounds")
    bbox = (
        _component_auth_xywh_to_xyxy(bbox_value, mask_shape)
        if bbox_mode == "xywh"
        else _component_auth_valid_or_xywh_bbox(bbox_value, mask_shape)
    )
    if not bbox:
        return None
    scope_mask = _component_auth_polygon_mask(source.get("polygon") or source.get("mask_polygon"), mask_shape)
    raw_cleanup_authorization = source.get("cleanup_authorization")
    raw_text_area_cleanup_authorization = source.get("text_area_cleanup_authorization")
    raw_auth = str(raw_cleanup_authorization or raw_text_area_cleanup_authorization or "")
    explicit_authority_source = ""
    if raw_cleanup_authorization:
        explicit_authority_source = "cleanup_authorization"
    elif raw_text_area_cleanup_authorization:
        explicit_authority_source = "text_area_cleanup_authorization"
    authorization_field_origin = str(
        source.get("authorization_field_origin")
        or source.get("text_area_authorization_field_origin")
        or ""
    )
    explicit_flag_value = source.get("authorization_explicit")
    if explicit_flag_value is None:
        explicit_flag_value = source.get("text_area_authorization_explicit")
    authorization_explicit = bool(explicit_flag_value)
    replay_authority = authorization_field_origin in REPLAY_AUTHORIZATION_FIELD_ORIGINS
    if raw_auth and authorization_explicit and not replay_authority:
        auth = raw_auth
    else:
        auth = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        explicit_authority_source = ""
    protection_reason = str(source.get("protection_reason") or source.get("text_area_protection_reason") or "")
    must_not_mutate = bool(source.get("must_not_mutate") or source.get("text_area_must_not_mutate"))
    container_id = str(source.get("container_id") or source.get("text_area_container_id") or source.get("scope_id") or fallback_id)
    region_id = str(source.get("region_id") or source.get("id") or "")
    job_ids = []
    direct_region_job_ids: List[str] = []
    for job_id in region_jobs.get(region_id, []) if region_id else []:
        if job_id not in job_ids:
            job_ids.append(str(job_id))
            direct_region_job_ids.append(str(job_id))
    if not (source_kind in {"region_record", "parent_execution_bundle"} and direct_region_job_ids):
        for job_id in container_jobs.get(container_id, []) if container_id else []:
            if job_id not in job_ids:
                job_ids.append(str(job_id))
    evidence_job_ids: List[str] = []
    if source_kind in {"parent_execution_bundle", "region_record"}:
        evidence_job_ids = _component_auth_source_evidence_job_ids(
            container_id=container_id,
            bbox=bbox,
            source_evidence_jobs=source_evidence_jobs,
        )
    if evidence_job_ids:
        if not job_ids or (len(job_ids) != 1 and len(evidence_job_ids) == 1):
            job_ids = list(evidence_job_ids)
        else:
            for job_id in evidence_job_ids:
                if job_id not in job_ids:
                    job_ids.append(str(job_id))
    cleanup_area_job_ids = _component_auth_cleanup_area_job_ids(
        authorization_state=auth,
        route_intent=str(source.get("route_intent") or source.get("text_area_route_intent") or source.get("intent") or ""),
        semantic_kind=str(source.get("semantic_kind") or source.get("container_type") or source.get("text_area_container_type") or ""),
        bbox=bbox,
        cleanup_job_areas=cleanup_job_areas,
    )
    cleanup_area_binding_used = False
    if cleanup_area_job_ids and not job_ids and source_kind in {"parent_execution_bundle", "region_record"}:
        job_ids = list(cleanup_area_job_ids)
        cleanup_area_binding_used = True
    reason_codes = _component_auth_list(
        source.get("reason_codes")
        or source.get("evidence_reason_codes")
        or source.get("text_area_reason_codes")
        or []
    )
    if evidence_job_ids and "cleanup_job_binding_from_source_glyph_evidence" not in reason_codes:
        reason_codes.append("cleanup_job_binding_from_source_glyph_evidence")
    if cleanup_area_binding_used and "cleanup_job_binding_from_cleanup_obligation_area" not in reason_codes:
        reason_codes.append("cleanup_job_binding_from_cleanup_obligation_area")
    if protection_reason and protection_reason not in reason_codes:
        reason_codes.append(protection_reason)
    conflict_flags = _component_auth_list(source.get("conflict_flags") or source.get("text_area_conflict_flags") or [])
    if raw_auth and not authorization_explicit:
        reason_codes.append("missing_explicit_authorization_field")
    if replay_authority:
        reason_codes.append("replay_artifact_authorization_ignored")
    if auth not in COMPONENT_AUTHORIZATION_STATES:
        auth = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        explicit_authority_source = ""
    explicit_cleanup_authority = _component_auth_family(auth) == "cleanup" and bool(explicit_authority_source)
    explicit_protected_authority = _component_auth_family(auth) == "protected" and bool(explicit_authority_source)
    family = _component_auth_family(auth)
    role_evidence = source.get("semantic_role_evidence") if isinstance(source.get("semantic_role_evidence"), Mapping) else {}
    semantic_evidence_trace = list(source.get("semantic_evidence_trace") or []) if isinstance(source.get("semantic_evidence_trace"), Sequence) and not isinstance(source.get("semantic_evidence_trace"), (str, bytes)) else []
    if role_evidence:
        semantic_evidence_trace.extend(_semantic_evidence_records(role_evidence))
    normalized_trace: List[Dict[str, Any]] = []
    seen_trace: set[tuple[str, str]] = set()
    for record in semantic_evidence_trace:
        if not isinstance(record, Mapping):
            continue
        provider = str(record.get("provider") or "")
        evidence_id = str(record.get("evidence_id") or "")
        if not provider:
            continue
        key = (provider, evidence_id)
        if key in seen_trace:
            continue
        seen_trace.add(key)
        normalized_trace.append(dict(record))
    semantic_evidence_providers = sorted(
        set(_component_auth_list(source.get("semantic_evidence_providers") or []))
        | {str(record.get("provider") or "") for record in normalized_trace if str(record.get("provider") or "")}
    )
    semantic_evidence_ids = sorted(
        set(_component_auth_list(source.get("semantic_evidence_ids") or []))
        | {str(record.get("evidence_id") or "") for record in normalized_trace if str(record.get("evidence_id") or "")}
    )
    return {
        "source_kind": source_kind,
        "bbox": bbox,
        "scope_mask": scope_mask,
        "authorization_state": auth,
        "family": family,
        "container_id": container_id,
        "region_id": region_id,
        "semantic_unit_id": _component_auth_binding_unit_id(
            source=source,
            source_kind=source_kind,
            container_id=container_id,
            region_id=region_id,
            fallback_id=fallback_id,
        ),
        "semantic_kind": str(source.get("semantic_kind") or source.get("container_type") or source.get("text_area_container_type") or ""),
        "route_intent": str(source.get("route_intent") or source.get("text_area_route_intent") or source.get("intent") or ""),
        "must_not_mutate": must_not_mutate or family in {"protected", "review", "outside", "ambiguous"},
        "source_stage": str(source.get("source_stage") or source.get("text_area_authorization_source_stage") or source_kind),
        "confidence_tier": str(source.get("confidence_tier") or source.get("text_area_confidence_tier") or "low"),
        "reason_codes": reason_codes,
        "conflict_flags": conflict_flags,
        "source_evidence_ids": _component_auth_list(
            source.get("source_evidence_ids")
            or source.get("source_model_ids")
            or source.get("text_area_source_evidence_ids")
            or []
        ),
        "cleanup_job_ids": job_ids,
        "source_evidence_job_records": list(source_evidence_jobs.get(container_id, []) or []),
        "cleanup_job_area_records": list(cleanup_job_areas or []),
        "explicit_cleanup_authority": explicit_cleanup_authority,
        "explicit_protected_authority": explicit_protected_authority,
        "explicit_authority_source": explicit_authority_source,
        "authorization_basis": str(source.get("authorization_basis") or source.get("text_area_authorization_basis") or ""),
        "authorization_explicit": authorization_explicit and not replay_authority,
        "authorization_field_origin": authorization_field_origin,
        "semantic_authorization_state": str(
            source.get("semantic_authorization_state")
            or source.get("text_area_semantic_authorization_state")
            or auth
        ),
        "semantic_role_evidence": dict(role_evidence or {}),
        "semantic_evidence_providers": semantic_evidence_providers,
        "semantic_evidence_ids": semantic_evidence_ids,
        "semantic_evidence_trace": normalized_trace,
    }


def _component_auth_cleanup_job_source_evidence(job: Any) -> List[Mapping[str, Any]]:
    raw = getattr(job, "source_glyph_evidence", None)
    if raw is None and isinstance(job, Mapping):
        raw = job.get("source_glyph_evidence")
    output: List[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                output.append(item)
    if len(output) <= 1:
        return output
    anchor_region_id = str(
        getattr(job, "cleanup_unit_anchor_region_id", "")
        or (job.get("cleanup_unit_anchor_region_id") if isinstance(job, Mapping) else "")
        or ""
    )
    target_region_ids = [
        str(item)
        for item in (
            getattr(job, "target_region_ids", None)
            if not isinstance(job, Mapping)
            else job.get("target_region_ids")
        )
        or []
        if str(item)
    ]
    if not anchor_region_id and len(target_region_ids) == 1:
        anchor_region_id = target_region_ids[0]
    if anchor_region_id:
        anchored: List[Mapping[str, Any]] = [
            item
            for item in output
            if anchor_region_id in _component_auth_region_ids_from_evidence(item)
        ]
        if anchored:
            return anchored
    expected_bbox = _component_auth_first_job_bbox(
        job,
        (
            "source_glyph_erasure_expected_area_bbox",
            "source_glyph_erasure_bbox",
            "allowed_cleanup_area",
        ),
    )
    if expected_bbox:
        expected_box = _component_auth_box_tuple(expected_bbox)
        if expected_box:
            spatial_matches: List[Mapping[str, Any]] = []
            best_score = 0.0
            for item in output:
                item_box = _component_auth_box_tuple(
                    item.get("source_glyph_erasure_expected_area_bbox")
                    or item.get("source_glyph_erasure_bbox")
                    or item.get("bbox")
                    or []
                )
                if not item_box:
                    continue
                intersection = _intersection_area(expected_box, item_box)
                if intersection <= 0:
                    continue
                score = intersection / max(1.0, min(_area(expected_box), _area(item_box)))
                if score > best_score:
                    best_score = score
                    spatial_matches = [item]
                elif score > 0 and score >= best_score * 0.90:
                    spatial_matches.append(item)
            if spatial_matches and best_score >= 0.90:
                return spatial_matches
    return output


def _component_auth_cleanup_job_area_records(job: Any, mask_shape: tuple[int, int]) -> List[Dict[str, Any]]:
    job_id = str(getattr(job, "cleanup_job_id", "") or "")
    if not job_id:
        return []
    if bool(getattr(job, "protected", False)):
        return []
    if getattr(job, "source_text_present", True) is False or getattr(job, "translated_text_present", True) is False:
        return []
    area_bbox: List[int] | None = None
    for attr in (
        "allowed_cleanup_area",
        "source_glyph_erasure_expected_area_bbox",
        "source_glyph_erasure_bbox",
    ):
        bbox = _component_auth_valid_or_xywh_bbox(getattr(job, attr, None), mask_shape)
        if bbox:
            area_bbox = bbox
            break
    cleanup_class = getattr(job, "cleanup_class", "")
    cleanup_class_value = str(getattr(cleanup_class, "value", cleanup_class) or "")
    return [
        {
            "cleanup_job_id": job_id,
            "bbox": area_bbox,
            "route_intent": str(getattr(job, "route_intent", "") or ""),
            "cleanup_class": cleanup_class_value,
            "semantic_class": str(getattr(job, "semantic_class", "") or ""),
            "text_block_root_id": str(getattr(job, "text_block_root_id", "") or ""),
            "cleanup_unit_id": str(getattr(job, "cleanup_unit_id", "") or ""),
            "text_area_container_id": str(getattr(job, "text_area_container_id", "") or ""),
        }
    ]


def _component_auth_first_job_bbox(job: Any, attrs: Sequence[str]) -> List[int] | None:
    for attr in attrs:
        value = getattr(job, attr, None) if not isinstance(job, Mapping) else job.get(attr)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
            try:
                bbox = [int(round(float(item))) for item in value[:4]]
            except Exception:
                continue
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return bbox
    return None


def _component_auth_region_ids_from_evidence(evidence: Mapping[str, Any]) -> List[str]:
    ids: set[str] = set()
    for key in ("region_id", "target_region_id"):
        value = evidence.get(key)
        if value is not None and str(value):
            ids.add(str(value))
    for key in ("text_instance_ids", "target_region_ids", "represented_region_ids"):
        value = evidence.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if item is not None and str(item):
                    ids.add(str(item))
        elif value is not None and str(value):
            ids.add(str(value))
    compatibility = evidence.get("compatibility_source_fields")
    if isinstance(compatibility, Mapping):
        ids.update(_component_auth_region_ids_from_evidence(compatibility))
    return sorted(ids)


def _component_auth_cleanup_area_job_ids(
    *,
    authorization_state: str,
    route_intent: str,
    semantic_kind: str,
    bbox: Sequence[int],
    cleanup_job_areas: Sequence[Mapping[str, Any]],
) -> List[str]:
    if _component_auth_family(authorization_state) != "cleanup":
        return []
    scope_box = _component_auth_box_tuple(bbox)
    if not scope_box:
        return []
    scope_area = max(1.0, _area(scope_box))
    job_scores: Dict[str, Tuple[float, float]] = {}
    for record in cleanup_job_areas or []:
        job_id = str(record.get("cleanup_job_id") or "")
        job_box = _component_auth_box_tuple(record.get("bbox") or [])
        if not job_id or not job_box:
            continue
        if not _component_auth_cleanup_job_route_compatible(
            authorization_state=authorization_state,
            route_intent=route_intent,
            semantic_kind=semantic_kind,
            job_record=record,
        ):
            continue
        intersection = _intersection_area(scope_box, job_box)
        if intersection <= 0:
            continue
        containment = intersection / scope_area
        if containment < 0.60:
            continue
        job_area = _area(job_box)
        current = job_scores.get(job_id)
        if current is None or containment > current[0] or (
            abs(containment - current[0]) < 1e-6 and job_area < current[1]
        ):
            job_scores[job_id] = (float(containment), float(job_area))
    if not job_scores:
        return []
    best_score = max(score for score, _area_value in job_scores.values())
    top = {
        job_id: (score, area_value)
        for job_id, (score, area_value) in job_scores.items()
        if score >= 0.60 and score >= best_score * 0.90
    }
    if not top:
        return []
    smallest_area = min(area_value for _score, area_value in top.values())
    return [
        job_id
        for job_id, (_score, area_value) in sorted(top.items())
        if area_value <= smallest_area * 1.05
    ]


def _component_auth_cleanup_job_route_compatible(
    *,
    authorization_state: str,
    route_intent: str,
    semantic_kind: str,
    job_record: Mapping[str, Any],
) -> bool:
    del route_intent, semantic_kind
    text = " ".join(
        str(value or "").lower()
        for value in (
            job_record.get("route_intent"),
            job_record.get("cleanup_class"),
            job_record.get("semantic_class"),
        )
    )
    if authorization_state == AUTH_CLEANUP_TRANSLATE_SPEECH:
        return "speech" in text or "bubble" in text
    if authorization_state in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        return any(token in text for token in ("caption", "background", "title", "sign", "side"))
    return False


def _component_auth_mapping_get(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source and source[key] not in (None, ""):
            return source[key]
    return None


def _component_auth_source_evidence_bbox(
    evidence: Mapping[str, Any],
    mask_shape: tuple[int, int],
) -> List[int] | None:
    for key in (
        "bbox",
        "source_glyph_erasure_bbox",
        "mask_bbox",
        "source_glyph_mask_actual_bbox",
        "source_glyph_mask_expected_bbox",
    ):
        bbox = _component_auth_valid_or_xywh_bbox(evidence.get(key), mask_shape)
        if bbox:
            return bbox
    return None


def _component_auth_source_evidence_job_ids(
    *,
    container_id: str,
    bbox: Sequence[int],
    source_evidence_jobs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[str]:
    if not container_id or not bbox:
        return []
    scope_box = _component_auth_box_tuple(bbox)
    if not scope_box:
        return []
    job_scores: Dict[str, Tuple[float, float]] = {}
    for record in source_evidence_jobs.get(container_id, []) or []:
        job_id = str(record.get("cleanup_job_id") or "")
        evidence_box = _component_auth_box_tuple(record.get("bbox") or [])
        if not job_id or not evidence_box:
            continue
        intersection = _intersection_area(scope_box, evidence_box)
        if intersection <= 0:
            continue
        score = intersection / max(1.0, min(_area(scope_box), _area(evidence_box)))
        if score <= 0:
            continue
        evidence_area = _area(evidence_box)
        current = job_scores.get(job_id)
        if current is None or score > current[0] or (
            abs(score - current[0]) < 1e-6 and evidence_area < current[1]
        ):
            job_scores[job_id] = (float(score), float(evidence_area))
    if not job_scores:
        return []
    best_score = max(score for score, _area_value in job_scores.values())
    if best_score < 0.20:
        return []
    top = {
        job_id: (score, area_value)
        for job_id, (score, area_value) in job_scores.items()
        if score >= 0.20 and score >= best_score * 0.90
    }
    if not top:
        return []
    smallest_area = min(area_value for _score, area_value in top.values())
    return [
        job_id
        for job_id, (_score, area_value) in sorted(top.items())
        if area_value <= smallest_area * 1.05
    ]


def _component_auth_box_tuple(value: Any) -> Tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in value[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _component_auth_polygon_mask(polygon: Any, mask_shape: tuple[int, int]) -> Any:
    if np is None or cv2 is None:
        return None
    if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)) or len(polygon) < 3:
        return None
    points: List[List[int]] = []
    height, width = int(mask_shape[0]), int(mask_shape[1])
    for item in polygon:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        try:
            x = max(0, min(width - 1, int(round(float(item[0])))))
            y = max(0, min(height - 1, int(round(float(item[1])))))
        except Exception:
            continue
        points.append([x, y])
    if len(points) < 3:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    except Exception:
        return None
    return mask


def _component_auth_infer_authorization(source: Mapping[str, Any]) -> str:
    """Legacy marker inference is fenced off from executable authority paths."""

    del source
    return AUTH_REVIEW_UNKNOWN_NOT_CLEANUP


def _component_auth_family(auth: str) -> str:
    if auth in {AUTH_CLEANUP_TRANSLATE_SPEECH, AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        return "cleanup"
    if auth in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT}:
        return "protected"
    if auth == AUTH_OUTSIDE_CLEANUP_SCOPE:
        return "outside"
    if auth == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        return "ambiguous"
    return "review"


def _component_auth_candidates(
    labels: Any,
    component_label: int,
    pixel_count: int,
    centroid: Sequence[float],
    scopes: Sequence[Mapping[str, Any]],
    component_bbox: Sequence[int],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    cx = float(centroid[0])
    cy = float(centroid[1])
    for scope in scopes or []:
        bbox = scope.get("bbox")
        if not isinstance(bbox, Sequence) or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [int(item) for item in bbox[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        overlap, centroid_inside = _component_auth_overlap_for_label(
            labels=labels,
            component_label=component_label,
            component_bbox=component_bbox,
            scope_bbox=(x0, y0, x1, y1),
            scope_mask=scope.get("scope_mask"),
            centroid=(cx, cy),
        )
        if overlap <= 0 and not centroid_inside:
            continue
        ratio = overlap / float(max(1, pixel_count))
        eligible = _component_auth_projection_candidate_eligible(
            overlap_pixels=overlap,
            pixel_count=pixel_count,
            overlap_ratio=ratio,
        )
        refined_scope = dict(scope)
        refined_job_ids, refinement_reasons = _component_auth_refined_component_job_ids(
            scope=scope,
            component_bbox=component_bbox,
        )
        if refined_job_ids != list(scope.get("cleanup_job_ids") or []):
            refined_scope["cleanup_job_ids"] = refined_job_ids
            reason_codes = _component_auth_list(refined_scope.get("reason_codes") or [])
            for reason in refinement_reasons:
                if reason not in reason_codes:
                    reason_codes.append(reason)
            refined_scope["reason_codes"] = reason_codes
        candidates.append(
            {
                **refined_scope,
                "overlap_pixels": overlap,
                "overlap_ratio": ratio,
                "centroid_inside": centroid_inside,
                "eligible": eligible,
            }
        )
    candidates.sort(
        key=lambda item: (
            int(item.get("overlap_pixels") or 0),
            bool(item.get("centroid_inside")),
            _component_auth_source_priority(str(item.get("source_kind") or "")),
            bool(item.get("cleanup_job_ids")),
        ),
        reverse=True,
    )
    return candidates


def _component_auth_refined_component_job_ids(
    *,
    scope: Mapping[str, Any],
    component_bbox: Sequence[int],
) -> Tuple[List[str], List[str]]:
    existing = [str(job_id) for job_id in scope.get("cleanup_job_ids") or [] if str(job_id)]
    container_id = str(scope.get("container_id") or "")
    reasons: List[str] = []
    refined = list(existing)
    evidence_records = scope.get("source_evidence_job_records") or []
    if evidence_records:
        evidence_ids = _component_auth_source_evidence_job_ids(
            container_id=container_id,
            bbox=component_bbox,
            source_evidence_jobs={container_id: evidence_records},
        )
        if refined:
            refined_by_evidence = _component_auth_refine_existing_job_ids(refined, evidence_ids)
            if refined_by_evidence != refined:
                refined = refined_by_evidence
                reasons.append("cleanup_job_binding_refined_by_component_source_glyph_evidence")
        elif evidence_ids:
            refined = list(evidence_ids)
            reasons.append("cleanup_job_binding_from_component_source_glyph_evidence")
    if not refined or len(refined) > 1:
        area_records = scope.get("cleanup_job_area_records") or []
        if area_records:
            area_ids = _component_auth_cleanup_area_job_ids(
                authorization_state=str(scope.get("authorization_state") or ""),
                route_intent=str(scope.get("route_intent") or ""),
                semantic_kind=str(scope.get("semantic_kind") or ""),
                bbox=component_bbox,
                cleanup_job_areas=area_records,
            )
            if not refined and area_ids:
                refined = list(area_ids)
                reasons.append("cleanup_job_binding_from_cleanup_obligation_area")
                reasons.append("cleanup_job_binding_refined_by_component_cleanup_obligation_area")
            elif refined:
                refined_by_area = _component_auth_refine_existing_job_ids(refined, area_ids)
                if refined_by_area != refined:
                    refined = refined_by_area
                    reasons.append("cleanup_job_binding_from_cleanup_obligation_area")
                    reasons.append("cleanup_job_binding_refined_by_component_cleanup_obligation_area")
    return refined, reasons


def _component_auth_refine_existing_job_ids(existing: Sequence[str], preferred: Sequence[str]) -> List[str]:
    current = [str(job_id) for job_id in existing or [] if str(job_id)]
    selected = [str(job_id) for job_id in preferred or [] if str(job_id)]
    if not current or not selected:
        return current
    intersection = [job_id for job_id in current if job_id in set(selected)]
    if intersection:
        return intersection
    return current


def _component_auth_projection_candidate_eligible(
    *,
    overlap_pixels: int,
    pixel_count: int,
    overlap_ratio: float,
) -> bool:
    if overlap_ratio >= 0.25:
        return True
    # Tiny punctuation/caption fragments can fall just under the generic ratio
    # because the CTD component includes antialiased edge pixels. This only
    # affects projection readiness after explicit semantic authority exists.
    if pixel_count <= 64 and overlap_pixels >= 8 and overlap_ratio >= 0.18:
        return True
    return False


def _component_auth_source_priority(source_kind: str) -> int:
    if source_kind == "parent_execution_bundle":
        return 5
    if source_kind == "text_area_semantic_unit":
        return 4
    if source_kind == "region_record":
        return 3
    if source_kind == "text_area_container":
        return 2
    if source_kind == "text_area_scope":
        return 1
    return 0


def _component_auth_semantic_unit_key(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("semantic_unit_id")
        or candidate.get("container_id")
        or candidate.get("region_id")
        or candidate.get("source_kind")
        or ""
    )


def _component_auth_semantic_units(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    units: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates or []:
        if not (int(candidate.get("overlap_pixels") or 0) > 0 or bool(candidate.get("centroid_inside"))):
            continue
        key = _component_auth_semantic_unit_key(candidate)
        if not key:
            continue
        existing = units.get(key)
        if existing is None:
            units[key] = dict(candidate)
            continue
        existing_overlap = int(existing.get("overlap_pixels") or 0)
        candidate_overlap = int(candidate.get("overlap_pixels") or 0)
        if candidate_overlap > existing_overlap or (
            candidate_overlap == existing_overlap
            and _component_auth_source_priority(str(candidate.get("source_kind") or ""))
            > _component_auth_source_priority(str(existing.get("source_kind") or ""))
        ):
            merged = dict(candidate)
            merged["cleanup_job_ids"] = sorted(
                {
                    str(job_id)
                    for source in (existing, candidate)
                    for job_id in (source.get("cleanup_job_ids") or [])
                    if str(job_id)
                }
            )
            merged["reason_codes"] = _component_auth_merged_values([existing, candidate], "reason_codes")
            merged["conflict_flags"] = _component_auth_merged_values([existing, candidate], "conflict_flags")
            merged["source_evidence_ids"] = _component_auth_merged_values([existing, candidate], "source_evidence_ids")
            units[key] = merged
        else:
            existing["cleanup_job_ids"] = sorted(
                {
                    str(job_id)
                    for source in (existing, candidate)
                    for job_id in (source.get("cleanup_job_ids") or [])
                    if str(job_id)
                }
            )
            existing["reason_codes"] = _component_auth_merged_values([existing, candidate], "reason_codes")
            existing["conflict_flags"] = _component_auth_merged_values([existing, candidate], "conflict_flags")
            existing["source_evidence_ids"] = _component_auth_merged_values([existing, candidate], "source_evidence_ids")
            existing["eligible"] = bool(existing.get("eligible") or candidate.get("eligible"))
    values = list(units.values())
    values.sort(
        key=lambda item: (
            bool(item.get("eligible")),
            int(item.get("overlap_pixels") or 0),
            _component_auth_source_priority(str(item.get("source_kind") or "")),
        ),
        reverse=True,
    )
    return values


def _component_auth_candidate_is_parent_execution_scope(candidate: Mapping[str, Any]) -> bool:
    source_kind = str(candidate.get("source_kind") or "").lower()
    if source_kind == "parent_execution_bundle":
        return True
    origin = str(candidate.get("authorization_field_origin") or "").lower()
    if "parent_execution_bundle" in origin:
        return True
    basis = str(candidate.get("authorization_basis") or "").lower()
    if "parent_execution_bundle" in basis or "finalized_parent_execution_bundle" in basis:
        return True
    region_id = str(candidate.get("region_id") or "")
    semantic_unit_id = str(candidate.get("semantic_unit_id") or "")
    return (
        region_id.startswith("tap_parent_")
        or region_id.startswith("parent_")
        or semantic_unit_id.startswith("parent_execution_region_")
    )


def _component_auth_cleanup_candidate_requires_parent_execution_boundary(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("family") != "cleanup" or not bool(candidate.get("explicit_cleanup_authority")):
        return False
    if _component_auth_candidate_is_parent_execution_scope(candidate):
        return True
    for key in ("cleanup_job_ids", "scope_cleanup_job_ids", "candidate_cleanup_job_ids"):
        for job_id in candidate.get(key) or []:
            text = str(job_id or "")
            if "tap_parent_" in text or "parent_" in text:
                return True
    return False


def _component_auth_parent_execution_candidate_supports_cleanup(candidate: Mapping[str, Any]) -> bool:
    if not _component_auth_candidate_is_parent_execution_scope(candidate):
        return False
    overlap_pixels = int(candidate.get("overlap_pixels") or 0)
    overlap_ratio = float(candidate.get("overlap_ratio") or 0.0)
    centroid_inside = bool(candidate.get("centroid_inside"))
    if overlap_pixels <= 0:
        return False
    if overlap_ratio >= 0.45:
        return True
    if centroid_inside and overlap_ratio >= 0.25:
        return True
    return False


def _component_auth_parent_execution_boundary_filtered_units(
    semantic_units: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    accepted_units = [dict(item) for item in semantic_units if bool(item.get("eligible"))]
    cleanup_units = [
        item
        for item in accepted_units
        if item.get("family") == "cleanup" and bool(item.get("explicit_cleanup_authority"))
    ]
    if not cleanup_units:
        return accepted_units, []

    parent_cleanup_units = [
        item
        for item in cleanup_units
        if _component_auth_parent_execution_candidate_supports_cleanup(item)
    ]
    parent_boundary_required = any(
        _component_auth_cleanup_candidate_requires_parent_execution_boundary(item)
        for item in semantic_units
    )
    if parent_cleanup_units:
        filtered: List[Dict[str, Any]] = []
        for item in accepted_units:
            if (
                item.get("family") == "cleanup"
                and bool(item.get("explicit_cleanup_authority"))
                and not _component_auth_candidate_is_parent_execution_scope(item)
                and _component_auth_cleanup_candidate_requires_parent_execution_boundary(item)
            ):
                continue
            filtered.append(item)
        if len(filtered) != len(accepted_units):
            return filtered, [COMPONENT_PARENT_EXECUTION_BOUNDARY_PREFERRED_REASON]
        return accepted_units, []

    if parent_boundary_required:
        filtered = [
            item
            for item in accepted_units
            if not (
                item.get("family") == "cleanup"
                and bool(item.get("explicit_cleanup_authority"))
                and _component_auth_cleanup_candidate_requires_parent_execution_boundary(item)
            )
        ]
        return filtered, [COMPONENT_PARENT_EXECUTION_BOUNDARY_REQUIRED_REASON]

    return accepted_units, []


def _component_auth_projection_quality(
    *,
    semantic_units: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
) -> tuple[str, List[str]]:
    if not semantic_units:
        return PROJECTION_NO_SEMANTIC_AUTHORITY, ["no_upstream_text_area_authorization"]
    if len(semantic_units) > 1:
        return PROJECTION_COMPONENT_MERGED, [PROJECTION_AMBIGUOUS_COMPONENT, PROJECTION_COMPONENT_MERGED]
    if selected is None:
        return PROJECTION_COMPONENT_MISSING, [PROJECTION_COMPONENT_MISSING]
    if not bool(selected.get("eligible")):
        if int(selected.get("overlap_pixels") or 0) <= 0:
            return PROJECTION_COMPONENT_MISSING, [PROJECTION_COMPONENT_MISSING]
        return PROJECTION_UNDERCOVERAGE, [PROJECTION_UNDERCOVERAGE]
    return PROJECTION_READY, []


def _component_auth_projection_ready_for_mask(
    *,
    semantic_authorization_state: str,
    projection_quality_state: str,
    job_binding_state: str,
) -> tuple[str, str]:
    family = _component_auth_family(semantic_authorization_state)
    if family != "cleanup":
        return MASK_NOT_APPLICABLE, ""
    if projection_quality_state != PROJECTION_READY:
        return MASK_NOT_READY, projection_quality_state or EFFECTIVE_MASK_NOT_READY
    if job_binding_state != "bound_unique":
        return MASK_NOT_READY, job_binding_state or "cleanup_job_binding_contract_error"
    return MASK_READY, ""


def _component_auth_projects_outside_authorized_area(
    *,
    semantic_authorization_state: str,
    selected: Mapping[str, Any] | None,
    pixel_count: int,
) -> bool:
    if selected is None:
        return False
    if _component_auth_family(semantic_authorization_state) not in {"cleanup", "protected"}:
        return False
    if bool(selected.get("eligible")):
        return False
    overlap_pixels = int(selected.get("overlap_pixels") or 0)
    overlap_ratio = float(selected.get("overlap_ratio") or 0.0)
    if overlap_pixels <= 0:
        return False
    return bool(pixel_count >= 512 and overlap_ratio < 0.05)


def _component_auth_record(
    *,
    page_id: str,
    component_index: int,
    label: int,
    bbox: List[int],
    pixel_count: int,
    centroid: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
) -> TextAreaComponentAuthorizationRecord:
    semantic_units = _component_auth_semantic_units(candidates)
    accepted_semantic_units, parent_boundary_reason_codes = _component_auth_parent_execution_boundary_filtered_units(
        semantic_units
    )
    cleanup_candidates = [
        item
        for item in accepted_semantic_units
        if item.get("family") == "cleanup" and bool(item.get("explicit_cleanup_authority"))
    ]
    protected_candidates = [
        item
        for item in accepted_semantic_units
        if item.get("family") == "protected" and bool(item.get("explicit_protected_authority"))
    ]
    ambiguous_candidates = [
        item
        for item in accepted_semantic_units
        if item.get("family") == "ambiguous" and bool(item.get("explicit_authority_source"))
    ]
    review_candidates = [item for item in semantic_units if item.get("family") == "review"]
    outside_candidates = [item for item in semantic_units if item.get("family") == "outside"]
    explicit_projection_candidates = [
        item
        for item in semantic_units
        if (
            (item.get("family") == "cleanup" and bool(item.get("explicit_cleanup_authority")))
            or (item.get("family") == "protected" and bool(item.get("explicit_protected_authority")))
        )
    ]
    explicit_units = cleanup_candidates + protected_candidates + ambiguous_candidates
    explicit_projection_families = {
        _component_auth_family(str(item.get("authorization_state") or ""))
        for item in explicit_projection_candidates
        if str(item.get("authorization_state") or "")
    }
    has_projection_authority_conflict = bool(
        "cleanup" in explicit_projection_families
        and "protected" in explicit_projection_families
    )
    best_cleanup = cleanup_candidates[0] if cleanup_candidates else None
    best_protected = protected_candidates[0] if protected_candidates else None
    ambiguity_reasons: List[str] = []

    selected: Mapping[str, Any] | None = None
    projection_candidate: Mapping[str, Any] | None = None
    semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    if ambiguous_candidates:
        selected = ambiguous_candidates[0]
        projection_candidate = selected
        semantic_authorization_state = AUTH_AMBIGUOUS_COMPONENT_OWNER
        ambiguity_reasons = ["explicit_cleanup_protected_conflict"]
    elif has_projection_authority_conflict:
        selected = None
        projection_candidate = explicit_projection_candidates[0] if explicit_projection_candidates else None
        semantic_authorization_state = AUTH_AMBIGUOUS_COMPONENT_OWNER
        ambiguity_reasons = [PROJECTION_COMPONENT_MERGED, PROJECTION_AMBIGUOUS_COMPONENT]
    elif len(explicit_units) > 1:
        selected = best_cleanup or best_protected or explicit_units[0]
        projection_candidate = selected
        semantic_authorization_state = str(selected.get("authorization_state") or AUTH_REVIEW_UNKNOWN_NOT_CLEANUP)
    elif best_cleanup:
        selected = best_cleanup
        projection_candidate = selected
        semantic_authorization_state = str(best_cleanup.get("authorization_state") or AUTH_CLEANUP_TRANSLATE_SPEECH)
    elif best_protected:
        selected = best_protected
        projection_candidate = selected
        semantic_authorization_state = str(best_protected.get("authorization_state") or AUTH_PROTECT_SFX_DECORATIVE)
    elif explicit_projection_candidates:
        projection_candidate = explicit_projection_candidates[0]
        projection_state = str(projection_candidate.get("authorization_state") or AUTH_REVIEW_UNKNOWN_NOT_CLEANUP)
        if _component_auth_family(projection_state) in {"cleanup", "protected", "outside", "ambiguous"}:
            semantic_authorization_state = projection_state
        else:
            semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    elif review_candidates:
        selected = review_candidates[0]
        projection_candidate = selected
        semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    elif outside_candidates:
        selected = outside_candidates[0]
        projection_candidate = selected
        semantic_authorization_state = AUTH_OUTSIDE_CLEANUP_SCOPE
    else:
        selected = None
        projection_candidate = None
        semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP

    projection_quality_state, projection_quality_reasons = _component_auth_projection_quality(
        semantic_units=semantic_units,
        selected=projection_candidate,
    )
    if has_projection_authority_conflict:
        projection_quality_state = PROJECTION_COMPONENT_MERGED
        for reason in (PROJECTION_AMBIGUOUS_COMPONENT, PROJECTION_COMPONENT_MERGED):
            if reason not in projection_quality_reasons:
                projection_quality_reasons.append(reason)
    elif (
        selected is not None
        and not has_projection_authority_conflict
        and bool(selected.get("eligible"))
    ):
        projection_quality_state = PROJECTION_READY
        projection_quality_reasons = [
            reason
            for reason in projection_quality_reasons
            if reason not in {PROJECTION_AMBIGUOUS_COMPONENT, PROJECTION_COMPONENT_MERGED}
        ]
    elif (
        projection_candidate is not None
        and selected is None
        and _component_auth_projects_outside_authorized_area(
            semantic_authorization_state=str(projection_candidate.get("authorization_state") or ""),
            selected=projection_candidate,
            pixel_count=pixel_count,
        )
    ):
        projection_quality_state = PROJECTION_OUTSIDE_AUTHORIZED_AREA
        if PROJECTION_OUTSIDE_AUTHORIZED_AREA not in projection_quality_reasons:
            projection_quality_reasons.append(PROJECTION_OUTSIDE_AUTHORIZED_AREA)
    semantic_family = _component_auth_family(semantic_authorization_state)
    reason_codes = _component_auth_merged_values(semantic_units, "reason_codes")
    conflict_flags = _component_auth_merged_values(semantic_units, "conflict_flags")
    if not semantic_units and "no_upstream_text_area_authorization" not in reason_codes:
        reason_codes.append("no_upstream_text_area_authorization")
    for reason in ambiguity_reasons + projection_quality_reasons:
        if reason and reason not in reason_codes:
            reason_codes.append(reason)
    for reason in parent_boundary_reason_codes:
        if reason and reason not in reason_codes:
            reason_codes.append(reason)
    selected_cleanup_candidates = (
        [selected]
        if selected and selected.get("family") == "cleanup" and semantic_family == "cleanup"
        else cleanup_candidates
        if semantic_family == "cleanup"
        else []
    )
    if (
        semantic_family == "cleanup"
        and not selected_cleanup_candidates
        and projection_candidate is not None
        and projection_candidate.get("family") == "cleanup"
    ):
        selected_cleanup_candidates = [projection_candidate]
    binding_state, binding_failure_reason, owner_cleanup_job_id, scope_cleanup_job_ids = _component_auth_job_binding(
        semantic_authorization_state,
        selected_cleanup_candidates,
    )
    mask_readiness_state, mask_readiness_failure_reason = _component_auth_projection_ready_for_mask(
        semantic_authorization_state=semantic_authorization_state,
        projection_quality_state=projection_quality_state,
        job_binding_state=binding_state,
    )
    final_mask_authorization_state = semantic_authorization_state
    if semantic_family == "cleanup" and mask_readiness_state != MASK_READY:
        final_mask_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        owner_cleanup_job_id = ""
        scope_cleanup_job_ids = []
        binding_state = "not_applicable_projection_not_ready"
        binding_failure_reason = ""
        mask_readiness_state = MASK_NOT_APPLICABLE
        mask_readiness_failure_reason = ""
        if COMPONENT_FINAL_MASK_AUTHORITY_WITHHELD_REASON not in reason_codes:
            reason_codes.append(COMPONENT_FINAL_MASK_AUTHORITY_WITHHELD_REASON)
    family = _component_auth_family(final_mask_authorization_state)
    component_id = f"tauthcomp_{_component_auth_safe_id(page_id)}_{component_index:04d}"
    owning_candidates = cleanup_candidates if family == "cleanup" else []
    protection_candidates = protected_candidates if family == "protected" else []
    if semantic_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        owning_candidates = cleanup_candidates
        protection_candidates = protected_candidates
    if binding_failure_reason and binding_failure_reason not in reason_codes:
        reason_codes.append(binding_failure_reason)
    if binding_state in {"missing_cleanup_job", "non_unique_cleanup_job"} and binding_state not in reason_codes:
        reason_codes.append(binding_state)
    if mask_readiness_failure_reason and mask_readiness_failure_reason not in reason_codes:
        reason_codes.append(mask_readiness_failure_reason)
    explicit_context = selected or projection_candidate
    explicit_cleanup_authority = bool(best_cleanup) if final_mask_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER else (
        bool(explicit_context.get("explicit_cleanup_authority")) if explicit_context else False
    )
    explicit_protected_authority = bool(best_protected) if final_mask_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER else (
        bool(explicit_context.get("explicit_protected_authority")) if explicit_context else False
    )
    explicit_authority_source = ",".join(
        sorted(
            {
                str(item.get("explicit_authority_source") or "")
                for item in (explicit_context, best_protected)
                if item and str(item.get("explicit_authority_source") or "")
            }
        )
    )
    authorization_field_origin = ",".join(
        sorted(
            {
                str(item.get("authorization_field_origin") or "")
                for item in (explicit_context, best_protected)
                if item and str(item.get("authorization_field_origin") or "")
            }
        )
    )
    authorization_basis = " | ".join(
        str(item.get("authorization_basis") or "")
        for item in (explicit_context, best_protected)
        if item and str(item.get("authorization_basis") or "")
    )
    semantic_unit_ids = _component_auth_unique_ids(semantic_units, "semantic_unit_id")
    semantic_unit_states = sorted(
        {
            str(item.get("authorization_state") or "")
            for item in semantic_units
            if str(item.get("authorization_state") or "")
        }
    )
    semantic_kinds = sorted({str(item.get("semantic_kind") or "") for item in semantic_units if str(item.get("semantic_kind") or "")})
    source_evidence_ids = _component_auth_merged_values(semantic_units, "source_evidence_ids")
    semantic_evidence_trace: List[Dict[str, Any]] = []
    seen_evidence_trace: set[tuple[str, str]] = set()
    for item in semantic_units:
        for record in item.get("semantic_evidence_trace") or []:
            if not isinstance(record, Mapping):
                continue
            provider = str(record.get("provider") or "")
            evidence_id = str(record.get("evidence_id") or "")
            if not provider:
                continue
            key = (provider, evidence_id)
            if key in seen_evidence_trace:
                continue
            seen_evidence_trace.add(key)
            semantic_evidence_trace.append(dict(record))
    semantic_evidence_providers = sorted(
        set(_component_auth_merged_values(semantic_units, "semantic_evidence_providers"))
        | {str(record.get("provider") or "") for record in semantic_evidence_trace if str(record.get("provider") or "")}
    )
    semantic_evidence_ids = sorted(
        set(_component_auth_merged_values(semantic_units, "semantic_evidence_ids"))
        | {str(record.get("evidence_id") or "") for record in semantic_evidence_trace if str(record.get("evidence_id") or "")}
    )
    projection_overlap_pixels = int(projection_candidate.get("overlap_pixels") or 0) if projection_candidate else 0
    projection_overlap_ratio = round(float(projection_candidate.get("overlap_ratio") or 0.0), 4) if projection_candidate else 0.0
    selected_context = selected or projection_candidate

    return TextAreaComponentAuthorizationRecord(
        page_id=page_id,
        component_id=component_id,
        bbox=[int(item) for item in bbox],
        component_bbox=[int(item) for item in bbox],
        pixel_count=int(pixel_count),
        component_pixel_count=int(pixel_count),
        authorization_state=final_mask_authorization_state,
        cleanup_authorization=final_mask_authorization_state if family in {"cleanup", "protected", "ambiguous"} else "",
        route_intent=str(selected_context.get("route_intent") or "") if selected_context else "",
        must_not_mutate=family != "cleanup",
        owning_container_ids=_component_auth_unique_ids(owning_candidates, "container_id"),
        protection_container_ids=_component_auth_unique_ids(protection_candidates, "container_id"),
        cleanup_owner_ids=_component_auth_unique_ids(owning_candidates, "container_id"),
        protection_owner_ids=_component_auth_unique_ids(protection_candidates, "container_id"),
        source_stage=str(selected_context.get("source_stage") or "text_area_plan_component_authorization") if selected_context else "text_area_plan_component_authorization",
        confidence_tier=str(selected_context.get("confidence_tier") or "low") if selected_context else "low",
        reason_codes=reason_codes,
        conflict_flags=conflict_flags,
        review_required=family in {"review", "outside", "ambiguous"},
        visual_debug_color=COMPONENT_AUTHORIZATION_COLORS.get(final_mask_authorization_state, "gray"),
        label=int(label),
        owner_cleanup_job_id=owner_cleanup_job_id,
        scope_cleanup_job_ids=scope_cleanup_job_ids,
        candidate_cleanup_job_ids=_component_auth_unique_jobs(semantic_units),
        owning_region_ids=_component_auth_unique_ids(owning_candidates, "region_id"),
        protection_region_ids=_component_auth_unique_ids(protection_candidates, "region_id"),
        overlap_pixels=int(selected.get("overlap_pixels") or 0) if selected else 0,
        overlap_ratio=round(float(selected.get("overlap_ratio") or 0.0), 4) if selected else 0.0,
        protected_overlap_pixels=int(best_protected.get("overlap_pixels") or 0) if best_protected else 0,
        protected_overlap_ratio=round(float(best_protected.get("overlap_ratio") or 0.0), 4) if best_protected else 0.0,
        centroid=[round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
        candidate_container_ids=_component_auth_unique_ids(semantic_units, "container_id"),
        candidate_region_ids=_component_auth_unique_ids(semantic_units, "region_id"),
        ambiguity_reasons=ambiguity_reasons,
        unresolved_reason_codes=reason_codes if family in {"review", "outside"} else [],
        protected_reason_codes=reason_codes if family == "protected" else [],
        ambiguous_reason_codes=reason_codes if family == "ambiguous" else [],
        semantic_authorization_state=semantic_authorization_state,
        semantic_visual_color=COMPONENT_AUTHORIZATION_COLORS.get(semantic_authorization_state, "gray"),
        job_binding_state=binding_state,
        job_binding_failure_reason=binding_failure_reason,
        final_mask_authorization_state=final_mask_authorization_state,
        explicit_cleanup_authority=explicit_cleanup_authority,
        explicit_protected_authority=explicit_protected_authority,
        explicit_authority_source=explicit_authority_source,
        authorization_basis=authorization_basis,
        authorization_explicit=bool(explicit_cleanup_authority or explicit_protected_authority),
        authorization_field_origin=authorization_field_origin,
        semantic_unit_ids=semantic_unit_ids,
        semantic_unit_states=semantic_unit_states,
        semantic_kind=str(selected_context.get("semantic_kind") or "") if selected_context else "",
        semantic_kinds=semantic_kinds,
        source_evidence_ids=source_evidence_ids,
        semantic_evidence_providers=semantic_evidence_providers,
        semantic_evidence_ids=semantic_evidence_ids,
        semantic_evidence_trace=semantic_evidence_trace,
        semantic_authority_owner="TextAreaPlan/BubbleDetection",
        projection_owner="TextForegroundSegmentationMask/TextAreaPlan projection",
        projection_quality_state=projection_quality_state,
        projection_quality_reasons=projection_quality_reasons,
        mask_readiness_state=mask_readiness_state,
        mask_readiness_failure_reason=mask_readiness_failure_reason,
        projected_label_ids=[int(label)],
        projection_overlap_pixels=projection_overlap_pixels,
        projection_overlap_ratio=projection_overlap_ratio,
    )


def _component_auth_assign_groups(page_id: str, records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    groups: Dict[tuple[str, str], List[TextAreaComponentAuthorizationRecord]] = {}
    for record in records:
        if record.owning_container_ids:
            key = ("cleanup", record.owning_container_ids[0])
        elif record.protection_container_ids:
            key = ("protected", record.protection_container_ids[0])
        elif record.candidate_container_ids:
            key = ("candidate", record.candidate_container_ids[0])
        else:
            key = ("component", record.component_id)
        groups.setdefault(key, []).append(record)
    for index, (_key, group) in enumerate(groups.items()):
        states = {record.authorization_state for record in group}
        cleanup_states = sorted(state for state in states if _component_auth_family(state) == "cleanup")
        protected_states = sorted(state for state in states if _component_auth_family(state) == "protected")
        if AUTH_AMBIGUOUS_COMPONENT_OWNER in states or (cleanup_states and protected_states):
            group_state = AUTH_AMBIGUOUS_COMPONENT_OWNER
        elif cleanup_states:
            group_state = cleanup_states[0]
        elif protected_states:
            group_state = protected_states[0]
        elif AUTH_OUTSIDE_CLEANUP_SCOPE in states:
            group_state = AUTH_OUTSIDE_CLEANUP_SCOPE
        else:
            group_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        group_id = f"tauthgrp_{_component_auth_safe_id(page_id)}_{index:04d}"
        for record in group:
            record.group_id = group_id
            record.group_authorization_state = group_state
            if record.authorization_state in {AUTH_REVIEW_UNKNOWN_NOT_CLEANUP, AUTH_OUTSIDE_CLEANUP_SCOPE} and group_state not in {
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                AUTH_OUTSIDE_CLEANUP_SCOPE,
                AUTH_AMBIGUOUS_COMPONENT_OWNER,
            }:
                _component_auth_add_contract_diagnostic(
                    record,
                    "group_authorization_candidate_not_semantic_authority",
                    warning=True,
                    visual_review=True,
                )


def _component_auth_apply_protected_sfx_grouping(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    seeds = [
        record
        for record in records
        if _component_auth_is_protected_sfx_group_seed(record)
    ]
    if not seeds:
        return
    for record in records:
        if record.authorization_state == AUTH_PROTECT_SFX_DECORATIVE:
            continue
        is_speech_record = _component_auth_is_speech_record(record)
        matching_seeds = [
            seed
            for seed in seeds
            if _component_auth_same_sfx_band(record, seed)
            or (
                not is_speech_record
                and _component_auth_component_inside_protected_seed(record, seed)
            )
        ]
        if not matching_seeds:
            continue
        contained_protected_seeds = [
            seed
            for seed in matching_seeds
            if not is_speech_record and _component_auth_component_inside_protected_seed(record, seed)
        ]
        local_sfx_seeds = [
            seed
            for seed in matching_seeds
            if _component_auth_is_local_sfx_continuation(record, seed)
        ]
        matching_seed = (contained_protected_seeds or local_sfx_seeds or matching_seeds)[0]
        is_local_sfx_continuation = bool(local_sfx_seeds)
        protected_seed_overlap = bool(contained_protected_seeds)
        if (
            is_speech_record
            and not _component_auth_has_decorative_evidence(record)
            and not (_component_auth_is_large_display_fragment(record) and is_local_sfx_continuation)
        ):
            continue
        if record.authorization_state in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            if not (
                _component_auth_has_decorative_evidence(record)
                or is_local_sfx_continuation
                or protected_seed_overlap
            ):
                continue
            _component_auth_add_contract_diagnostic(
                record,
                "sfx_group_conflicts_with_cleanup_authorization",
                defect=True,
                visual_review=True,
                candidate_conflict=True,
            )
        else:
            _component_auth_add_contract_diagnostic(
                record,
                "sfx_group_propagated_from_protected_neighbor_candidate",
                warning=True,
                visual_review=True,
            )
        owner_seed_keys: set[str] = set()
        owner_seed_candidates: List[TextAreaComponentAuthorizationRecord] = []
        for seed in [*contained_protected_seeds, *local_sfx_seeds, matching_seed]:
            key = str(seed.component_id or id(seed))
            if key in owner_seed_keys:
                continue
            owner_seed_keys.add(key)
            owner_seed_candidates.append(seed)
        for seed in owner_seed_candidates:
            for container_id in _component_auth_protected_seed_owner_ids(seed):
                if container_id and container_id not in record.protection_container_ids:
                    record.protection_container_ids.append(container_id)


def _component_auth_is_protected_sfx_group_seed(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.component_pixel_count < 20:
        return False
    if record.authorization_state == AUTH_PROTECT_SFX_DECORATIVE:
        return True
    if set(record.component_contract_defect_codes or []) & PROTECTED_BLOCKING_COMPONENT_DEFECT_CODES:
        return True
    marker = _component_auth_record_marker(record)
    if not any(
        token in marker
        for token in (
            "deterministic_large_sfx",
            "semantic_sfx_decorative_authority",
            "explicit_sfx_decorative_authorization",
            "protect_sfx_decorative",
            "sfx_decorative",
        )
    ):
        return False
    if record.authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        return True
    if record.explicit_protected_authority:
        return True
    return AUTH_PROTECT_SFX_DECORATIVE in set(record.semantic_unit_states or [])


def _component_auth_protected_seed_owner_ids(record: TextAreaComponentAuthorizationRecord) -> List[str]:
    ids: List[str] = []
    for source, force_include in (
        (record.protection_container_ids, True),
        (record.protection_owner_ids, True),
        (record.candidate_container_ids, False),
        (record.owning_container_ids, False),
    ):
        for item in source or []:
            value = str(item or "")
            marker = value.lower()
            if not value:
                continue
            if value in ids:
                continue
            if force_include or "sfx" in marker or "decorative" in marker:
                ids.append(value)
    if not ids and _component_auth_is_protected_sfx_group_seed(record):
        ids.append(str(record.component_id or "protected_sfx_component"))
    return ids


def _component_auth_component_inside_protected_seed(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    if len(record.component_bbox) < 4 or len(seed.component_bbox) < 4:
        return False
    ax0, ay0, ax1, ay1 = [int(item) for item in record.component_bbox[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in seed.component_bbox[:4]]
    if ax1 <= ax0 or ay1 <= ay0 or bx1 <= bx0 or by1 <= by0:
        return False
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    overlap_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    if overlap_area / float(area) >= 0.35:
        return True
    cx = (ax0 + ax1) / 2.0
    cy = (ay0 + ay1) / 2.0
    return bool(bx0 <= cx < bx1 and by0 <= cy < by1)


def _component_auth_apply_vertical_review_text_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if _component_auth_is_vertical_review_text_candidate(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        if not _component_auth_is_vertical_review_text_group(group):
            continue
        for item in group:
            _component_auth_add_contract_diagnostic(
                item,
                "vertical_review_text_group_candidate_missing_upstream_cleanup_authority",
                defect=True,
                visual_review=True,
            )


def _component_auth_is_vertical_review_text_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state in {
        AUTH_CLEANUP_TRANSLATE_SPEECH,
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
        AUTH_OUTSIDE_CLEANUP_SCOPE,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    if width > 70 or height > 90 or record.component_pixel_count > 1200:
        return False
    return True


def _component_auth_is_vertical_review_text_group(group: Sequence[TextAreaComponentAuthorizationRecord]) -> bool:
    if len(group) < 6:
        return False
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if height < 140 or width < 35 or width > 220:
        return False
    if height < width * 1.25:
        return False
    seed_count = sum(1 for item in group if _component_auth_has_review_text_seed(item))
    if seed_count <= 0:
        return False
    large_items = [
        item
        for item in group
        if (
            max(1, int(item.component_bbox[2]) - int(item.component_bbox[0])) > 70
            or max(1, int(item.component_bbox[3]) - int(item.component_bbox[1])) > 90
            or item.component_pixel_count > 1200
        )
    ]
    return not large_items


def _component_auth_has_review_text_seed(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return bool(
        record.candidate_container_ids
        and any(token in marker for token in ("review_or_fallback", "unknown_fallback", "bubble_without_kitsumed", "dark_or_art_context"))
    )


def _component_auth_apply_orphan_text_near_cleanup_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if _component_auth_is_orphan_text_candidate(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        anchor = _component_auth_cleanup_anchor_for_orphan_group(group, records)
        if anchor is None:
            continue
        state = AUTH_CLEANUP_TRANSLATE_SPEECH if str(anchor.route_intent or "") == "translate_speech" else AUTH_CLEANUP_TRANSLATE_BACKGROUND
        route = "translate_speech" if state == AUTH_CLEANUP_TRANSLATE_SPEECH else "translate_caption_background"
        for item in group:
            _component_auth_add_contract_diagnostic(
                item,
                "small_orphan_text_group_near_cleanup_missing_upstream_authority",
                defect=True,
                visual_review=True,
            )


def _component_auth_is_orphan_text_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state != AUTH_REVIEW_UNKNOWN_NOT_CLEANUP:
        return False
    if record.candidate_container_ids or record.candidate_cleanup_job_ids or record.owning_container_ids or record.protection_container_ids:
        return False
    if _component_auth_has_decorative_evidence(record) or _component_auth_is_speech_record(record):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    return bool(15 <= record.component_pixel_count <= 600 and width <= 50 and height <= 55)


def _component_auth_cleanup_anchor_for_orphan_group(
    group: Sequence[TextAreaComponentAuthorizationRecord],
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> TextAreaComponentAuthorizationRecord | None:
    if len(group) < 2:
        return None
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    pixels = sum(int(item.component_pixel_count) for item in group)
    if pixels < 120 or width > 100 or height > 100:
        return None
    group_box = [x0, y0, x1, y1]
    for protected in records:
        if protected.authorization_state not in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT, AUTH_AMBIGUOUS_COMPONENT_OWNER}:
            continue
        if protected.component_pixel_count >= 200 and _component_auth_bbox_gap(group_box, protected.component_bbox) <= 120:
            return None
    anchors = [
        record
        for record in records
        if _component_auth_family(record.authorization_state) == "cleanup"
        and not _component_auth_has_decorative_evidence(record)
        and _component_auth_bbox_gap(group_box, record.component_bbox) <= 180
    ]
    if not anchors:
        return None
    anchors.sort(key=lambda item: _component_auth_bbox_gap(group_box, item.component_bbox))
    return anchors[0]


def _component_auth_apply_ogkalu_speech_text_group_completion(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    """Complete plan-bound Ogkalu speech text groups at the component boundary.

    Ogkalu-only text-bubble authority is a text-unit authority, not permission
    to wipe an entire detected container. Once one projected component is bound
    to that formal speech obligation, adjacent character-sized components in
    the same text run must follow that obligation unless they carry strong
    protected/SFX authority.
    """

    anchors = [
        record
        for record in records
        if _component_auth_is_ogkalu_speech_text_anchor(record)
    ]
    if not anchors:
        return
    candidates = [
        record
        for record in records
        if _component_auth_is_ogkalu_speech_text_completion_candidate(record)
    ]
    for anchor in anchors:
        group: List[TextAreaComponentAuthorizationRecord] = [anchor]
        changed = True
        while changed:
            changed = False
            for candidate in candidates:
                if candidate in group:
                    continue
                if _component_auth_text_component_adjacent_to_group(candidate, group):
                    group.append(candidate)
                    changed = True
        if len(group) < 2 or not _component_auth_group_forms_text_run(group):
            continue
        for item in group:
            _component_auth_promote_to_anchor_cleanup(
                item,
                anchor,
                reason="ogkalu_speech_text_group_completion",
            )


def _component_auth_is_ogkalu_speech_text_anchor(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state != AUTH_CLEANUP_TRANSLATE_SPEECH:
        return False
    if str(record.route_intent or "") != ROUTE_TRANSLATE_SPEECH:
        return False
    if record.job_binding_state != "bound_unique" or not record.owner_cleanup_job_id:
        return False
    marker = _component_auth_record_marker(record)
    if "ogkalu_text_bubble_without_kitsumed_mask" not in marker:
        return False
    if "kitsumed_mask_primary_geometry" in marker or "speech_mask_container" in marker:
        return False
    if _component_auth_has_strong_protected_text_group_block(record):
        return False
    return _component_auth_is_character_sized_text_component(record)


def _component_auth_is_ogkalu_speech_text_completion_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if COMPONENT_FINAL_MASK_AUTHORITY_WITHHELD_REASON in set(record.reason_codes or []):
        return False
    if _component_auth_family(record.authorization_state) == "cleanup":
        return False
    if record.authorization_state in {AUTH_AMBIGUOUS_COMPONENT_OWNER, AUTH_OUTSIDE_CLEANUP_SCOPE}:
        return False
    if not _component_auth_is_character_sized_text_component(record):
        return False
    if _component_auth_has_strong_protected_text_group_block(record):
        return False
    marker = _component_auth_record_marker(record)
    if any(
        token in marker
        for token in (
            "ogkalu_text_bubble_without_kitsumed_mask",
            "ogkalu_bubble_without_kitsumed_mask",
            "no_upstream_text_area_authorization",
            "ogkalu_text_free_without_kitsumed_mask",
        )
    ):
        return True
    return False


def _component_auth_record_marker(record: TextAreaComponentAuthorizationRecord) -> str:
    return " ".join(
        [
            str(record.route_intent or ""),
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.conflict_flags or []),
            " ".join(record.owning_container_ids or []),
            " ".join(record.candidate_container_ids or []),
            " ".join(record.protection_container_ids or []),
            " ".join(record.semantic_unit_states or []),
            " ".join(record.protected_reason_codes or []),
            " ".join(record.semantic_evidence_providers or []),
        ]
    ).lower()


def _component_auth_has_strong_protected_text_group_block(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = _component_auth_record_marker(record)
    strong_tokens = (
        "deterministic_large_sfx",
        "recognized_sfx_decorative_typed_evidence",
        "typed_ogkalu_sfx_decorative",
        "explicit_sfx_decorative_authorization",
        "protect_art_or_non_text",
        "art_or_non_text",
        "non_text",
        "non-text",
    )
    if any(token in marker for token in strong_tokens):
        weak_text_free_only = (
            "ogkalu_text_free_without_kitsumed_mask" in marker
            and "deterministic_large_sfx" not in marker
            and "typed_ogkalu_sfx_decorative" not in marker
            and "recognized_sfx_decorative_typed_evidence" not in marker
        )
        if not weak_text_free_only:
            return True
    return False


def _component_auth_is_character_sized_text_component(record: TextAreaComponentAuthorizationRecord) -> bool:
    if len(record.component_bbox) < 4:
        return False
    x0, y0, x1, y1 = [int(item) for item in record.component_bbox[:4]]
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    area = width * height
    pixels = int(record.component_pixel_count or 0)
    if pixels < 12 or pixels > 1600:
        return False
    if width < 4 or height < 6 or width > 64 or height > 92:
        return False
    density = pixels / float(max(1, area))
    if density < 0.035 or density > 0.88:
        return False
    if width >= 58 and width >= height * 2.6:
        return False
    if height <= 6 and width >= 18:
        return False
    return True


def _component_auth_text_component_adjacent_to_group(
    record: TextAreaComponentAuthorizationRecord,
    group: Sequence[TextAreaComponentAuthorizationRecord],
) -> bool:
    return any(_component_auth_text_components_adjacent(record, item) for item in group)


def _component_auth_text_components_adjacent(
    a: TextAreaComponentAuthorizationRecord,
    b: TextAreaComponentAuthorizationRecord,
) -> bool:
    if len(a.component_bbox) < 4 or len(b.component_bbox) < 4:
        return False
    ax0, ay0, ax1, ay1 = [int(item) for item in a.component_bbox[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in b.component_bbox[:4]]
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    acx, acy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0, max(ay0, by0) - min(ay1, by1))
    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
    x_overlap_ratio = x_overlap / float(max(1, min(aw, bw)))
    y_overlap_ratio = y_overlap / float(max(1, min(ah, bh)))
    vertical_column = (
        abs(acx - bcx) <= max(24.0, min(aw, bw) * 1.25)
        and y_gap <= max(34, int(max(ah, bh) * 1.25))
    )
    horizontal_run = (
        abs(acy - bcy) <= max(22.0, min(ah, bh) * 1.25)
        and x_gap <= max(34, int(max(aw, bw) * 1.25))
    )
    return bool(vertical_column or horizontal_run or (x_overlap_ratio >= 0.35 and y_gap <= 42) or (y_overlap_ratio >= 0.35 and x_gap <= 42))


def _component_auth_group_forms_text_run(group: Sequence[TextAreaComponentAuthorizationRecord]) -> bool:
    if len(group) < 2:
        return False
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if width > 180 and height > 180:
        return False
    centers_x = [(int(item.component_bbox[0]) + int(item.component_bbox[2])) / 2.0 for item in group]
    centers_y = [(int(item.component_bbox[1]) + int(item.component_bbox[3])) / 2.0 for item in group]
    vertical = height >= width * 1.15 and (max(centers_x) - min(centers_x)) <= max(42.0, width * 0.72)
    horizontal = width >= height * 1.15 and (max(centers_y) - min(centers_y)) <= max(36.0, height * 0.72)
    compact = width <= 96 and height <= 140
    return bool(vertical or horizontal or compact)


def _component_auth_promote_to_anchor_cleanup(
    record: TextAreaComponentAuthorizationRecord,
    anchor: TextAreaComponentAuthorizationRecord,
    *,
    reason: str,
) -> None:
    _component_auth_set_state(record, anchor.authorization_state, reason=reason)
    record.route_intent = anchor.route_intent
    record.source_stage = anchor.source_stage
    record.confidence_tier = anchor.confidence_tier
    record.owning_container_ids = list(anchor.owning_container_ids or anchor.candidate_container_ids or [])
    record.cleanup_owner_ids = list(anchor.cleanup_owner_ids or anchor.owning_container_ids or anchor.candidate_container_ids or [])
    record.owner_cleanup_job_id = str(anchor.owner_cleanup_job_id or "")
    job_ids = list(anchor.scope_cleanup_job_ids or anchor.candidate_cleanup_job_ids or [])
    if anchor.owner_cleanup_job_id and anchor.owner_cleanup_job_id not in job_ids:
        job_ids.append(anchor.owner_cleanup_job_id)
    record.scope_cleanup_job_ids = [str(item) for item in job_ids if str(item)]
    record.candidate_cleanup_job_ids = list(record.scope_cleanup_job_ids)
    record.job_binding_state = anchor.job_binding_state or "bound_unique"
    record.job_binding_failure_reason = ""
    record.projection_quality_state = PROJECTION_READY
    record.projection_quality_reasons = []
    record.mask_readiness_state = MASK_READY
    record.mask_readiness_failure_reason = ""
    record.semantic_unit_ids = list(anchor.semantic_unit_ids or [])
    record.semantic_unit_states = list(anchor.semantic_unit_states or [anchor.authorization_state])
    record.semantic_kind = anchor.semantic_kind
    record.semantic_kinds = list(anchor.semantic_kinds or [anchor.semantic_kind])
    record.source_evidence_ids = list(anchor.source_evidence_ids or [])
    record.semantic_evidence_providers = list(anchor.semantic_evidence_providers or [])
    record.semantic_evidence_ids = list(anchor.semantic_evidence_ids or [])
    record.semantic_evidence_trace = list(anchor.semantic_evidence_trace or [])
    record.semantic_authority_owner = anchor.semantic_authority_owner
    record.projection_owner = anchor.projection_owner
    record.authorization_basis = anchor.authorization_basis
    record.authorization_explicit = anchor.authorization_explicit
    record.authorization_field_origin = anchor.authorization_field_origin
    record.explicit_cleanup_authority = True
    record.explicit_protected_authority = False


def _component_auth_bbox_gap(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < 4 or len(b) < 4:
        return 10**9
    ax0, ay0, ax1, ay1 = [int(item) for item in a[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in b[:4]]
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0, max(ay0, by0) - min(ay1, by1))
    return int(max(x_gap, y_gap))


def _component_auth_same_sfx_band(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    box = record.component_bbox
    seed_box = seed.component_bbox
    if len(box) < 4 or len(seed_box) < 4:
        return False
    width = max(1, int(box[2]) - int(box[0]))
    height = max(1, int(box[3]) - int(box[1]))
    seed_width = max(1, int(seed_box[2]) - int(seed_box[0]))
    seed_height = max(1, int(seed_box[3]) - int(seed_box[1]))
    if record.component_pixel_count < 20 and width < 8 and height < 8:
        return False
    y_overlap = max(0, min(int(box[3]), int(seed_box[3])) - max(int(box[1]), int(seed_box[1])))
    y_ratio = y_overlap / float(max(1, min(height, seed_height)))
    x_overlap = max(0, min(int(box[2]), int(seed_box[2])) - max(int(box[0]), int(seed_box[0])))
    x_ratio = x_overlap / float(max(1, min(width, seed_width)))
    if int(box[2]) < int(seed_box[0]):
        gap = int(seed_box[0]) - int(box[2])
    elif int(seed_box[2]) < int(box[0]):
        gap = int(box[0]) - int(seed_box[2])
    else:
        gap = 0
    if int(box[3]) < int(seed_box[1]):
        vertical_gap = int(seed_box[1]) - int(box[3])
    elif int(seed_box[3]) < int(box[1]):
        vertical_gap = int(box[1]) - int(seed_box[3])
    else:
        vertical_gap = 0
    if y_ratio < 0.28 and not (x_ratio >= 0.35 and vertical_gap <= max(120, int(max(height, seed_height) * 1.2))):
        return False
    if gap > max(120, int(max(width, seed_width) * 2.0)):
        return False
    center_y = (int(box[1]) + int(box[3])) / 2.0
    seed_center_y = (int(seed_box[1]) + int(seed_box[3])) / 2.0
    if abs(center_y - seed_center_y) > max(160.0, float(max(height, seed_height)) * 1.4):
        return False
    return bool(record.component_pixel_count >= 20 or seed_width >= 40)


def _component_auth_is_speech_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.owning_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return any(token in marker for token in ("translate_speech", "speech_bubble", "speech_mask_container"))


def _component_auth_has_decorative_evidence(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.conflict_flags or []),
            " ".join(record.protection_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return any(token in marker for token in ("sfx", "decorative", "preserve", "art_text", "art_sfx", "non_text", "non-text"))


def _component_auth_is_local_sfx_continuation(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    box = record.component_bbox
    seed_box = seed.component_bbox
    if len(box) < 4 or len(seed_box) < 4:
        return False
    width = max(1, int(box[2]) - int(box[0]))
    height = max(1, int(box[3]) - int(box[1]))
    seed_width = max(1, int(seed_box[2]) - int(seed_box[0]))
    seed_height = max(1, int(seed_box[3]) - int(seed_box[1]))
    if int(box[2]) < int(seed_box[0]):
        gap = int(seed_box[0]) - int(box[2])
    elif int(seed_box[2]) < int(box[0]):
        gap = int(box[0]) - int(seed_box[2])
    else:
        gap = 0
    y_overlap = max(0, min(int(box[3]), int(seed_box[3])) - max(int(box[1]), int(seed_box[1])))
    y_ratio = y_overlap / float(max(1, min(height, seed_height)))
    x_overlap = max(0, min(int(box[2]), int(seed_box[2])) - max(int(box[0]), int(seed_box[0])))
    x_ratio = x_overlap / float(max(1, min(width, seed_width)))
    if int(box[3]) < int(seed_box[1]):
        vertical_gap = int(seed_box[1]) - int(box[3])
    elif int(seed_box[3]) < int(box[1]):
        vertical_gap = int(box[1]) - int(seed_box[3])
    else:
        vertical_gap = 0
    if y_ratio < 0.35 and not (x_ratio >= 0.35 and vertical_gap <= 120):
        return False
    horizontal_stroke = gap <= 80 and width >= 45 and width >= height * 2.0
    compact_neighbor = gap <= 80 and width >= 18 and seed_width >= 60
    large_sfx_cluster_neighbor = (
        gap <= 220
        and width >= 80
        and height >= 80
        and seed_width >= 80
        and seed_height >= 80
    )
    vertical_sfx_stack = (
        vertical_gap <= 120
        and x_ratio >= 0.35
        and width >= 45
        and height >= 45
        and seed_width >= 45
        and seed_height >= 45
    )
    return bool(horizontal_stroke or compact_neighbor or large_sfx_cluster_neighbor or vertical_sfx_stack)


def _component_auth_apply_review_only_caption_guard(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    guarded: List[TextAreaComponentAuthorizationRecord] = []
    for record in records:
        if not _component_auth_is_large_review_only_caption_seed(record):
            continue
        _component_auth_add_contract_diagnostic(
            record,
            "large_review_only_caption_component_conflicts_with_cleanup_authority",
            defect=True,
            visual_review=True,
            candidate_conflict=True,
        )
        guarded.append(record)
    for group in _component_auth_review_only_caption_groups(records):
        if not _component_auth_is_large_review_only_caption_group(group):
            continue
        for record in group:
            _component_auth_add_contract_diagnostic(
                record,
                "large_review_only_caption_group_conflicts_with_cleanup_authority",
                defect=True,
                visual_review=True,
                candidate_conflict=True,
            )
            if record not in guarded:
                guarded.append(record)
    if not guarded:
        return

    for record in records:
        if record in guarded:
            continue
        if record.authorization_state not in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            continue
        if _component_auth_is_speech_record(record):
            continue
        if str(record.route_intent or "") != "translate_caption_background":
            continue
        if not any(_component_auth_shares_local_caption_guard(record, seed) for seed in guarded):
            continue
        _component_auth_add_contract_diagnostic(
            record,
            "large_review_only_caption_group_conflicts_with_cleanup_authority",
            defect=True,
            visual_review=True,
            candidate_conflict=True,
        )


def _component_auth_is_large_review_only_caption_component(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
    }:
        return False
    return _component_auth_is_large_review_only_caption_seed(record)


def _component_auth_is_large_review_only_caption_seed(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if str(record.route_intent or "") != "translate_caption_background":
        return False
    marker = " ".join(
        [
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.conflict_flags or []),
        ]
    ).lower()
    if not any(
        token in marker
        for token in (
            "text_free_review_only",
            "ogkalu_text_free_without_kitsumed_mask",
            "deterministic_side_caption_requires_review",
            "deterministic_top_band_far_right_caption_search",
            "sfx_group_conflicts_with_cleanup_authorization",
        )
    ):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    if record.component_pixel_count < 1000:
        return False
    return bool(width >= 120 or (width >= 90 and height >= 90))


def _component_auth_review_only_caption_groups(
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> List[List[TextAreaComponentAuthorizationRecord]]:
    groups: Dict[str, List[TextAreaComponentAuthorizationRecord]] = {}
    for record in records:
        if not _component_auth_is_review_only_caption_group_candidate(record):
            continue
        ids = list(record.owning_container_ids or []) + list(record.candidate_container_ids or [])
        key = ids[0] if ids else record.component_id
        groups.setdefault(key, []).append(record)
    return [group for group in groups.values() if len(group) >= 2]


def _component_auth_is_review_only_caption_group_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if str(record.route_intent or "") != "translate_caption_background":
        return False
    marker = " ".join(
        [
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.owning_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    has_text_free_marker = any(token in marker for token in ("text_free_review_only", "ogkalu_text_free_without_kitsumed_mask"))
    if (
        "deterministic_top_band_far_right_caption_search" in marker
        and not has_text_free_marker
        and len(record.component_bbox) >= 4
        and int(record.component_bbox[1]) > 280
    ):
        return False
    return any(
        token in marker
        for token in (
            "text_free_review_only",
            "ogkalu_text_free_without_kitsumed_mask",
            "deterministic_top_band_far_right_caption_search",
            "deterministic_top_band_day_caption_search",
            "sfx_group_conflicts_with_cleanup_authorization",
        )
    )


def _component_auth_is_large_review_only_caption_group(group: Sequence[TextAreaComponentAuthorizationRecord]) -> bool:
    if len(group) < 2:
        return False
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    pixels = sum(int(item.component_pixel_count) for item in group)
    if pixels < 2500 or height < 80 or width < 45:
        return False
    if any(item.authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER for item in group):
        return True
    display_like_components = 0
    for item in group:
        item_width = max(1, int(item.component_bbox[2]) - int(item.component_bbox[0]))
        item_height = max(1, int(item.component_bbox[3]) - int(item.component_bbox[1]))
        if item.component_pixel_count >= 1200 or (item_width >= 45 and item_height >= 80):
            display_like_components += 1
    if display_like_components < 2:
        return False
    return bool((len(group) >= 3 and (width >= 75 or height >= 160)) or (len(group) >= 2 and pixels >= 5000 and height >= 160))


def _component_auth_shares_local_caption_guard(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    shared_containers = set(record.owning_container_ids or []) & set(seed.owning_container_ids or [])
    shared_candidates = set(record.candidate_container_ids or []) & set(seed.candidate_container_ids or [])
    if not shared_containers and not shared_candidates:
        return False
    if len(record.component_bbox) < 4 or len(seed.component_bbox) < 4:
        return False
    box = [int(item) for item in record.component_bbox[:4]]
    seed_box = [int(item) for item in seed.component_bbox[:4]]
    margin = 35
    return not (
        box[2] < seed_box[0] - margin
        or box[0] > seed_box[2] + margin
        or box[3] < seed_box[1] - margin
        or box[1] > seed_box[3] + margin
    )


def _component_auth_set_state(
    record: TextAreaComponentAuthorizationRecord,
    state: str,
    *,
    reason: str = "",
    ambiguity: bool = False,
) -> None:
    family = _component_auth_family(state)
    record.authorization_state = state
    record.semantic_authorization_state = state
    record.final_mask_authorization_state = state
    record.cleanup_authorization = state if family in {"cleanup", "protected", "ambiguous"} else ""
    record.must_not_mutate = family != "cleanup"
    record.review_required = family in {"review", "outside", "ambiguous"}
    color = COMPONENT_AUTHORIZATION_COLORS.get(state, "gray")
    record.visual_debug_color = color
    record.semantic_visual_color = color
    if family != "cleanup":
        record.job_binding_state = "not_applicable_semantic_conflict" if state == AUTH_AMBIGUOUS_COMPONENT_OWNER else "not_applicable_non_cleanup"
        record.job_binding_failure_reason = ""
        record.owner_cleanup_job_id = ""
        record.scope_cleanup_job_ids = []
        record.mask_readiness_state = MASK_NOT_APPLICABLE
        record.mask_readiness_failure_reason = ""
        record.reason_codes = [
            item
            for item in record.reason_codes
            if item not in {"cleanup_job_binding_contract_error", "missing_cleanup_job", "non_unique_cleanup_job"}
        ]
    if reason and reason not in record.reason_codes:
        record.reason_codes.append(reason)
    if ambiguity and reason and reason not in record.ambiguity_reasons:
        record.ambiguity_reasons.append(reason)


def _component_auth_apply_cleanup_obligation_area_bindings(
    records: Sequence[TextAreaComponentAuthorizationRecord],
    cleanup_job_areas: Sequence[Mapping[str, Any]],
) -> None:
    if not cleanup_job_areas:
        return
    for record in records:
        if _component_auth_family(record.authorization_state) != "cleanup":
            continue
        if record.job_binding_state == "bound_unique" and record.owner_cleanup_job_id:
            continue
        job_ids = _component_auth_cleanup_area_job_ids(
            authorization_state=record.authorization_state,
            route_intent=record.route_intent,
            semantic_kind=record.semantic_kind or " ".join(record.semantic_kinds or []),
            bbox=record.component_bbox,
            cleanup_job_areas=cleanup_job_areas,
        )
        if not job_ids:
            continue
        existing_job_ids = _component_auth_record_cleanup_job_ids(record)
        selected_job_ids = list(job_ids)
        if existing_job_ids:
            selected_job_ids = [job_id for job_id in job_ids if job_id in existing_job_ids]
            if len(selected_job_ids) != 1:
                continue
        record.scope_cleanup_job_ids = list(selected_job_ids)
        record.candidate_cleanup_job_ids = list(selected_job_ids)
        if len(selected_job_ids) == 1:
            record.owner_cleanup_job_id = selected_job_ids[0]
        if "cleanup_job_binding_from_cleanup_obligation_area" not in record.reason_codes:
            record.reason_codes.append("cleanup_job_binding_from_cleanup_obligation_area")


def _component_auth_record_cleanup_job_ids(record: TextAreaComponentAuthorizationRecord) -> List[str]:
    output: List[str] = []
    for value in (
        [record.owner_cleanup_job_id] if record.owner_cleanup_job_id else [],
        record.scope_cleanup_job_ids or [],
        record.candidate_cleanup_job_ids or [],
    ):
        for item in value:
            text = str(item or "")
            if text and text not in output:
                output.append(text)
    return output


def _component_auth_record_requires_parent_execution_boundary(record: TextAreaComponentAuthorizationRecord) -> bool:
    reasons = set(record.reason_codes or [])
    if COMPONENT_PARENT_EXECUTION_BOUNDARY_REQUIRED_REASON in reasons:
        return True
    if COMPONENT_PARENT_EXECUTION_BOUNDARY_PREFERRED_REASON in reasons:
        return True
    return False


def _component_auth_record_has_supported_parent_execution_boundary(record: TextAreaComponentAuthorizationRecord) -> bool:
    source_marker = " ".join(
        [
            str(record.source_stage or ""),
            str(record.authorization_field_origin or ""),
            " ".join(record.owning_region_ids or []),
        ]
    ).lower()
    if "parent_execution_bundle" not in source_marker and not any(
        str(item or "").startswith("tap_parent_") or str(item or "").startswith("parent_")
        for item in record.owning_region_ids or []
    ):
        return False
    if int(record.overlap_pixels or 0) <= 0:
        return False
    ratio = float(record.overlap_ratio or 0.0)
    return ratio >= 0.45


def _component_auth_apply_terminal_authority_guards(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    """Resolve blocking contract defects before CleanupMask consumption.

    Orange/debug ambiguity is useful during review, but green/executable is not
    allowed to survive a proven protected/decorative conflict on the same
    component. A sibling/root-level protected conflict is audit evidence only;
    it must not demote otherwise authorized text in the same cleanup task.
    """

    _component_auth_apply_ogkalu_speech_unit_protected_blocks(records)
    for record in records:
        if record.authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
            _component_auth_resolve_ambiguous_component(record)
            continue
        if _component_auth_family(record.authorization_state) != "cleanup":
            continue
        if (
            _component_auth_record_requires_parent_execution_boundary(record)
            and not _component_auth_record_has_supported_parent_execution_boundary(record)
        ):
            _component_auth_set_state(
                record,
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                reason=COMPONENT_PARENT_EXECUTION_BOUNDARY_BLOCKED_REASON,
            )
            record.explicit_cleanup_authority = False
            continue
        if _component_auth_has_terminal_protected_projection(record):
            _component_auth_fail_closed_to_protected_projection(
                record,
                reason="parent_execution_cleanup_binding_blocked_by_protected_projection",
            )
            continue
        defect_codes = set(record.component_contract_defect_codes or [])
        if defect_codes & PROTECTED_BLOCKING_COMPONENT_DEFECT_CODES:
            if _component_auth_cleanup_overrides_protected_conflict(record):
                continue
            if _component_auth_record_has_protected_conflict(record):
                _component_auth_set_state(
                    record,
                    AUTH_PROTECT_SFX_DECORATIVE,
                    reason="component_authority_blocked_by_protected_conflict",
                )
                record.explicit_cleanup_authority = False
                record.explicit_protected_authority = True
                record.protection_owner_ids = list(record.protection_container_ids or record.protection_owner_ids or [])
                record.protected_reason_codes = list(record.reason_codes)
                continue
            _component_auth_set_state(
                record,
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                reason="component_authority_blocked_by_unresolved_protected_conflict",
            )
            record.explicit_cleanup_authority = False
            continue
        if defect_codes & REVIEW_BLOCKING_COMPONENT_DEFECT_CODES and _component_auth_is_weak_background_authority_record(record):
            _component_auth_set_state(
                record,
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                reason="component_authority_blocked_by_contract_defect",
            )
            record.explicit_cleanup_authority = False
            continue
        if record.job_binding_state in {"", "missing_cleanup_job", "non_unique_cleanup_job"}:
            _component_auth_bind_cleanup_job_from_candidates(record)


def _component_auth_apply_side_caption_projection_guards(
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> None:
    """Reject isolated CTD fragments inside broad deterministic side-caption scopes.

    Deterministic side-caption authority is a weak background/caption authority:
    it tells cleanup which caption text area is executable, but CTD can still
    report nearby art or background edges inside that broad area. Those pixels
    need structural support from the vertical text chain before they can remain
    executable cleanup components.
    """

    groups: Dict[str, List[TextAreaComponentAuthorizationRecord]] = {}
    for record in records:
        if not _component_auth_is_side_caption_projection_guard_candidate(record):
            continue
        for key in _component_auth_side_caption_projection_group_keys(record):
            groups.setdefault(key, []).append(record)

    for group in groups.values():
        columns = _component_auth_side_caption_supported_columns(group)
        if not columns:
            continue
        supported_record_ids = {
            id(record)
            for record in group
            if _component_auth_is_side_caption_projection_guard_candidate(record)
            and _component_auth_side_caption_record_has_column_support(record, columns)
        }
        changed = True
        while changed:
            changed = False
            for record in group:
                if id(record) in supported_record_ids:
                    continue
                if not _component_auth_is_side_caption_projection_guard_candidate(record):
                    continue
                if _component_auth_side_caption_record_has_supported_local_glyph_support(
                    record,
                    group,
                    supported_record_ids,
                ):
                    supported_record_ids.add(id(record))
                    changed = True
        for record in group:
            if not _component_auth_is_side_caption_projection_guard_candidate(record):
                continue
            if id(record) in supported_record_ids:
                continue
            _component_auth_add_contract_diagnostic(
                record,
                SIDE_CAPTION_TEXT_CHAIN_SUPPORT_DEFECT,
                warning=True,
                defect=True,
                visual_review=True,
                candidate_conflict=True,
            )
            _component_auth_set_state(
                record,
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                reason=SIDE_CAPTION_TEXT_CHAIN_SUPPORT_DEFECT,
            )
            record.explicit_cleanup_authority = False
            record.explicit_protected_authority = False


def _component_auth_is_side_caption_projection_guard_candidate(
    record: TextAreaComponentAuthorizationRecord,
) -> bool:
    if _component_auth_family(record.authorization_state) != "cleanup":
        return False
    if record.sourceglyph_overlap_pixels > 0:
        return False
    marker = _component_auth_record_marker(record)
    if "det_side_caption_" not in marker and "vertical_side_caption" not in marker:
        return False
    if not _component_auth_is_weak_background_authority_record(record):
        return False
    if _component_auth_has_terminal_protected_projection(record):
        return False
    return True


def _component_auth_side_caption_projection_group_keys(record: TextAreaComponentAuthorizationRecord) -> List[str]:
    keys: List[str] = []
    for source in (
        record.owning_container_ids,
        record.cleanup_owner_ids,
        record.candidate_container_ids,
        record.semantic_unit_ids,
    ):
        for item in source or []:
            value = str(item or "")
            if value.startswith("det_side_caption_") and value not in keys:
                keys.append(value)
    if keys:
        return keys
    if record.owner_cleanup_job_id:
        return [f"cleanup_job:{record.owner_cleanup_job_id}"]
    return []


def _component_auth_side_caption_supported_columns(
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> List[Dict[str, float]]:
    metrics = [
        metric
        for metric in (_component_auth_side_caption_metric(record) for record in records)
        if metric is not None
    ]
    if len(metrics) < 4:
        return []
    seed_metrics = [
        metric
        for metric in metrics
        if metric["pixels"] >= 80.0 and metric["width"] >= 3.0 and metric["height"] >= 8.0
    ]
    if len(seed_metrics) < 4:
        return []

    median_width = _component_auth_median([metric["width"] for metric in seed_metrics]) or 16.0
    median_height = _component_auth_median([metric["height"] for metric in seed_metrics]) or 24.0
    center_tolerance = max(10.0, min(18.0, median_width * 0.65))

    raw_clusters: List[List[Dict[str, float]]] = []
    for metric in sorted(seed_metrics, key=lambda item: item["cx"]):
        selected: List[Dict[str, float]] | None = None
        best_distance = float("inf")
        for cluster in raw_clusters:
            cluster_center = _component_auth_median([item["cx"] for item in cluster]) or metric["cx"]
            distance = abs(metric["cx"] - cluster_center)
            if distance <= center_tolerance and distance < best_distance:
                selected = cluster
                best_distance = distance
        if selected is None:
            raw_clusters.append([metric])
        else:
            selected.append(metric)

    columns: List[Dict[str, float]] = []
    for cluster in raw_clusters:
        if len(cluster) < 3:
            continue
        y0 = min(item["y0"] for item in cluster)
        y1 = max(item["y1"] for item in cluster)
        y_span = y1 - y0
        if y_span < max(48.0, median_height * 2.2):
            continue
        cluster_width = _component_auth_median([item["width"] for item in cluster]) or median_width
        columns.append(
            {
                "cx": _component_auth_median([item["cx"] for item in cluster]) or cluster[0]["cx"],
                "x_tolerance": max(12.0, min(22.0, cluster_width * 0.75)),
                "y0": y0,
                "y1": y1,
                "y_pad": max(24.0, min(42.0, median_height * 1.35)),
            }
        )
    return columns


def _component_auth_side_caption_record_has_column_support(
    record: TextAreaComponentAuthorizationRecord,
    columns: Sequence[Mapping[str, float]],
) -> bool:
    metric = _component_auth_side_caption_metric(record)
    if metric is None:
        return False
    for column in columns:
        center = float(column.get("cx") or 0.0)
        tolerance = float(column.get("x_tolerance") or 0.0)
        y0 = float(column.get("y0") or 0.0) - float(column.get("y_pad") or 0.0)
        y1 = float(column.get("y1") or 0.0) + float(column.get("y_pad") or 0.0)
        if abs(metric["cx"] - center) > tolerance:
            continue
        if metric["y1"] < y0 or metric["y0"] > y1:
            continue
        return True
    return False


def _component_auth_side_caption_record_has_supported_local_glyph_support(
    record: TextAreaComponentAuthorizationRecord,
    group: Sequence[TextAreaComponentAuthorizationRecord],
    supported_record_ids: set[int],
) -> bool:
    metric = _component_auth_side_caption_metric(record)
    if metric is None:
        return False
    if metric["pixels"] > 80.0 and metric["width"] > 10.0:
        return False
    for other in group:
        if other is record:
            continue
        if id(other) not in supported_record_ids:
            continue
        other_metric = _component_auth_side_caption_metric(other)
        if other_metric is None:
            continue
        if other_metric["pixels"] > 120.0 and other_metric["width"] > 14.0:
            continue
        horizontal_gap = abs(metric["cx"] - other_metric["cx"])
        vertical_gap = max(0.0, max(other_metric["y0"] - metric["y1"], metric["y0"] - other_metric["y1"]))
        if horizontal_gap <= 18.0 and vertical_gap <= 24.0:
            return True
    return False


def _component_auth_side_caption_metric(
    record: TextAreaComponentAuthorizationRecord,
) -> Dict[str, float] | None:
    if len(record.component_bbox or []) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in record.component_bbox[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": x1 - x0,
        "height": y1 - y0,
        "cx": (x0 + x1) * 0.5,
        "cy": (y0 + y1) * 0.5,
        "pixels": float(record.component_pixel_count or record.pixel_count or 0),
    }


def _component_auth_median(values: Sequence[float]) -> float:
    clean: List[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            clean.append(number)
    clean.sort()
    if not clean:
        return 0.0
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) * 0.5


def _component_auth_resolve_ambiguous_component(record: TextAreaComponentAuthorizationRecord) -> None:
    """Resolve orange/debug ownership before CleanupMask consumption.

    Orange is a review/debug signal for a component-level conflict, not a
    stable production state. Before the cleanup contract consumes the map, a
    component must either be a cleanup obligation, a protected component, or a
    review-only component. Cleanup may win only when the record carries a
    bounded text-unit authority that is already job-bindable; otherwise
    protected conflicts fail closed as protected SFX/decorative.
    """

    if _component_auth_has_terminal_protected_projection(record):
        _component_auth_fail_closed_to_protected_projection(
            record,
            reason="parent_execution_cleanup_binding_blocked_by_protected_projection",
        )
        return

    if _component_auth_is_completed_ogkalu_speech_text_run_record(record):
        cleanup_state = _component_auth_cleanup_state_for_record(record)
        _component_auth_set_state(
            record,
            cleanup_state,
            reason="ambiguous_component_resolved_to_cleanup_text_authority",
        )
        _component_auth_bind_cleanup_job_from_candidates(record)
        record.explicit_cleanup_authority = True
        record.explicit_protected_authority = False
        return

    if _component_auth_is_parent_execution_bound_cleanup_record(record):
        cleanup_state = _component_auth_cleanup_state_for_record(record)
        _component_auth_set_state(
            record,
            cleanup_state,
            reason="ambiguous_parent_execution_component_resolved_to_cleanup_authority",
        )
        _component_auth_bind_cleanup_job_from_candidates(record)
        record.explicit_cleanup_authority = True
        record.explicit_protected_authority = False
        return

    if _component_auth_record_has_protected_conflict(record) or _component_auth_has_decorative_evidence(record):
        _component_auth_set_state(
            record,
            AUTH_PROTECT_SFX_DECORATIVE,
            reason="ambiguous_protected_component_without_cleanup_owner",
        )
        record.explicit_cleanup_authority = False
        record.explicit_protected_authority = True
        if not record.protection_owner_ids:
            record.protection_owner_ids = list(record.protection_container_ids or [])
        record.protected_reason_codes = list(record.reason_codes)
        return

    _component_auth_set_state(
        record,
        AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
        reason="ambiguous_component_requires_review_not_cleanup",
        ambiguity=True,
    )
    record.explicit_cleanup_authority = False
    record.explicit_protected_authority = False


def _component_auth_cleanup_overrides_protected_conflict(record: TextAreaComponentAuthorizationRecord) -> bool:
    if _component_auth_has_terminal_protected_projection(record):
        return False
    if _component_auth_is_parent_execution_bound_cleanup_record(record):
        return True
    if not _component_auth_is_speech_record(record):
        return False
    if _component_auth_is_weak_background_authority_record(record):
        return False
    if _component_auth_is_completed_ogkalu_speech_text_run_record(record):
        return True
    if (
        _component_auth_is_strong_speech_authority_record(record)
        and not _component_auth_has_nondeterministic_protected_authority(record)
    ):
        return True
    return False


def _component_auth_has_terminal_protected_projection(record: TextAreaComponentAuthorizationRecord) -> bool:
    if not _component_auth_record_has_protected_conflict(record):
        return False
    if not _component_auth_has_decorative_evidence(record):
        return False
    semantic_kind = str(record.semantic_kind or "").lower()
    route_intent = str(record.route_intent or "").lower()
    source_stage = str(record.source_stage or "").lower()
    confidence_tier = str(record.confidence_tier or "").lower()
    if semantic_kind in {SEMANTIC_KIND_SFX, SEMANTIC_KIND_DECORATIVE, CONTAINER_SFX}:
        return True
    if route_intent == ROUTE_PRESERVE_SFX:
        return True
    if (
        source_stage == "text_area_semantic_unit"
        and "deterministic_large_sfx_visual_authority" in confidence_tier
    ):
        return True
    return False


def _component_auth_fail_closed_to_protected_projection(
    record: TextAreaComponentAuthorizationRecord,
    *,
    reason: str,
) -> None:
    _component_auth_add_contract_diagnostic(
        record,
        reason,
        defect=True,
        visual_review=True,
        candidate_conflict=True,
    )
    _component_auth_set_state(
        record,
        AUTH_PROTECT_SFX_DECORATIVE,
        reason=reason,
    )
    record.explicit_cleanup_authority = False
    record.explicit_protected_authority = True
    if not record.protection_owner_ids:
        record.protection_owner_ids = list(record.protection_container_ids or [])
    record.protected_reason_codes = list(record.reason_codes)


def _component_auth_is_parent_execution_bound_cleanup_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    states = {
        str(record.authorization_state or ""),
        str(record.semantic_authorization_state or ""),
        str(record.cleanup_authorization or ""),
        str(record.final_mask_authorization_state or ""),
        *(str(item or "") for item in (record.semantic_unit_states or [])),
    }
    if not any(_component_auth_family(state) == "cleanup" for state in states):
        return False
    job_ids = [
        str(item or "")
        for item in (
            ([record.owner_cleanup_job_id] if record.owner_cleanup_job_id else [])
            + list(record.candidate_cleanup_job_ids or [])
            + list(record.scope_cleanup_job_ids or [])
        )
        if str(item or "")
    ]
    job_ids = list(dict.fromkeys(job_ids))
    if len(job_ids) != 1:
        return False
    marker = _component_auth_record_marker(record)
    source_marker = " ".join(
        [
            str(record.source_stage or ""),
            str(record.authorization_field_origin or ""),
            *[str(item or "") for item in (record.reason_codes or [])],
        ]
    ).lower()
    if "parent_execution_bundle" not in source_marker and "parent_execution_region" not in marker and not any(
        str(item or "").startswith("parent_") or str(item or "").startswith("tap_parent_")
        for item in (record.owning_region_ids or record.candidate_region_ids or [])
    ):
        return False
    if not any(
        str(item or "").startswith("parent_") or str(item or "").startswith("tap_parent_")
        for item in (record.owning_region_ids or record.candidate_region_ids or [])
    ):
        return False
    return True


def _component_auth_is_completed_ogkalu_speech_text_run_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = _component_auth_record_marker(record)
    if "ogkalu_speech_text_group_completion" not in marker:
        return False
    if AUTH_CLEANUP_TRANSLATE_SPEECH not in set(record.semantic_unit_states or []):
        return False
    if len(record.candidate_cleanup_job_ids or []) > 1:
        return False
    return bool(
        record.owner_cleanup_job_id
        or record.candidate_cleanup_job_ids
        or record.scope_cleanup_job_ids
        or record.owning_container_ids
        or record.cleanup_owner_ids
    )


def _component_auth_cleanup_state_for_record(record: TextAreaComponentAuthorizationRecord) -> str:
    states = set(record.semantic_unit_states or [])
    if AUTH_CLEANUP_TRANSLATE_SPEECH in states or _component_auth_is_speech_record(record):
        return AUTH_CLEANUP_TRANSLATE_SPEECH
    if AUTH_CLEANUP_TRANSLATE_BACKGROUND in states:
        return AUTH_CLEANUP_TRANSLATE_BACKGROUND
    if AUTH_CLEANUP_TRANSLATE_CAPTION in states:
        return AUTH_CLEANUP_TRANSLATE_CAPTION
    return AUTH_CLEANUP_TRANSLATE_SPEECH


def _component_auth_bind_cleanup_job_from_candidates(record: TextAreaComponentAuthorizationRecord) -> None:
    job_ids = [str(item) for item in (record.scope_cleanup_job_ids or record.candidate_cleanup_job_ids or []) if str(item)]
    if record.owner_cleanup_job_id and record.owner_cleanup_job_id not in job_ids:
        job_ids.insert(0, str(record.owner_cleanup_job_id))
    job_ids = list(dict.fromkeys(job_ids))
    if len(job_ids) == 1:
        record.owner_cleanup_job_id = job_ids[0]
        record.scope_cleanup_job_ids = list(job_ids)
        record.candidate_cleanup_job_ids = list(job_ids)
        record.job_binding_state = "bound_unique"
        record.job_binding_failure_reason = ""
        record.reason_codes = [
            reason
            for reason in (record.reason_codes or [])
            if reason not in {"cleanup_job_binding_contract_error", "missing_cleanup_job", "non_unique_cleanup_job"}
        ]
        record.projection_quality_state = PROJECTION_READY
        record.projection_quality_reasons = [
            reason
            for reason in (record.projection_quality_reasons or [])
            if reason not in {PROJECTION_AMBIGUOUS_COMPONENT, PROJECTION_COMPONENT_MERGED}
        ]
        record.mask_readiness_state = MASK_READY
        record.mask_readiness_failure_reason = ""
    elif not job_ids:
        record.job_binding_state = "missing_cleanup_job"
        record.mask_readiness_state = MASK_NOT_READY
        record.mask_readiness_failure_reason = "missing_cleanup_job"
    else:
        record.job_binding_state = "non_unique_cleanup_job"
        record.scope_cleanup_job_ids = job_ids
        record.candidate_cleanup_job_ids = job_ids
        record.mask_readiness_state = MASK_NOT_READY
        record.mask_readiness_failure_reason = "non_unique_cleanup_job"


def _component_auth_has_nondeterministic_protected_authority(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = _component_auth_record_marker(record)
    strong_tokens = (
        "typed_ogkalu_sfx_decorative",
        "recognized_sfx_decorative_typed_evidence",
        "current_region_sfx_decorative_evidence",
        "current_sfx_decorative_region",
        "protect_art_or_non_text",
        "art_or_non_text",
        "non_text",
        "non-text",
    )
    if any(token in marker for token in strong_tokens):
        return True
    return False


def _component_auth_apply_ogkalu_speech_unit_protected_blocks(
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> None:
    blocked_keys: set[str] = set()
    for record in records:
        marker = _component_auth_record_marker(record)
        if "ogkalu_text_bubble_without_kitsumed_mask" not in marker:
            continue
        if not (
            "deterministic_large_sfx" in marker
            or "typed_ogkalu_sfx_decorative" in marker
            or "recognized_sfx_decorative_typed_evidence" in marker
        ):
            continue
        if not (
            record.authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER
            or _component_auth_family(record.authorization_state) == "protected"
            or _component_auth_has_decorative_evidence(record)
        ):
            continue
        for key in _component_auth_cleanup_group_keys(record):
            blocked_keys.add(key)
    if not blocked_keys:
        return

    for record in records:
        if _component_auth_family(record.authorization_state) != "cleanup":
            continue
        if not _component_auth_is_speech_record(record):
            continue
        if _component_auth_is_completed_ogkalu_speech_text_run_record(record):
            continue
        if not (blocked_keys & set(_component_auth_cleanup_group_keys(record))):
            continue
        if not _component_auth_is_large_display_fragment(record):
            continue
        _component_auth_add_contract_diagnostic(
            record,
            "ogkalu_speech_unit_blocked_by_protected_sfx_evidence",
            defect=True,
            visual_review=True,
            candidate_conflict=True,
        )
        _component_auth_set_state(
            record,
            AUTH_PROTECT_SFX_DECORATIVE,
            reason="ogkalu_speech_unit_blocked_by_protected_sfx_evidence",
        )
        record.explicit_cleanup_authority = False
        record.explicit_protected_authority = True
        if not record.protection_owner_ids:
            record.protection_owner_ids = list(record.protection_container_ids or [])


def _component_auth_cleanup_group_keys(record: TextAreaComponentAuthorizationRecord) -> List[str]:
    keys: List[str] = []
    for source in (
        record.owning_container_ids,
        record.cleanup_owner_ids,
        record.candidate_container_ids,
    ):
        for item in source or []:
            value = str(item or "")
            if not value or value.startswith("det_large_sfx_"):
                continue
            if "sfx" in value.lower() or "decorative" in value.lower():
                continue
            if value not in keys:
                keys.append(value)
    return keys


def _component_auth_is_strong_speech_authority_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = _component_auth_record_marker(record)
    if any(
        token in marker
        for token in (
            "kitsumed_mask_primary_geometry",
            "speech_mask_container",
            "speech_bubble_mask_evidence",
            "kitsumed_speech_mask_evidence",
            "current_region_speech_authority",
        )
    ):
        return True
    if "ogkalu_text_bubble_without_kitsumed_mask" in marker:
        return False
    if "ogkalu_bubble_without_kitsumed_mask" in marker:
        return False
    return False


def _component_auth_record_has_protected_conflict(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.protection_container_ids or record.protection_owner_ids:
        return True
    if record.explicit_protected_authority:
        return True
    if any(_component_auth_family(state) == "protected" for state in record.semantic_unit_states or []):
        return True
    if any("protect_" in str(reason) or "sfx" in str(reason).lower() for reason in record.protected_reason_codes or []):
        return True
    return False


def _component_auth_is_weak_background_authority_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
    }:
        return False
    marker = " ".join(
        [
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.semantic_evidence_providers or []),
            " ".join(record.semantic_unit_states or []),
        ]
    )
    return any(token in marker for token in WEAK_BACKGROUND_AUTHORITY_REASON_TOKENS)


def _component_auth_add_contract_diagnostic(
    record: TextAreaComponentAuthorizationRecord,
    reason: str,
    *,
    warning: bool = False,
    defect: bool = False,
    visual_review: bool = False,
    candidate_conflict: bool = False,
) -> None:
    if not reason:
        return
    if warning and reason not in record.authorization_warning_codes:
        record.authorization_warning_codes.append(reason)
    if defect and reason not in record.component_contract_defect_codes:
        record.component_contract_defect_codes.append(reason)
    if visual_review:
        record.requires_visual_review = True
        record.review_required = True
    if candidate_conflict:
        record.candidate_conflict_reason = reason
        if reason not in record.ambiguity_reasons:
            record.ambiguity_reasons.append(reason)


def _component_auth_apply_large_decorative_review_rule(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    for record in records:
        if record.authorization_state in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT, AUTH_AMBIGUOUS_COMPONENT_OWNER}:
            continue
        if len(record.component_bbox) < 4:
            continue
        width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
        height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
        if record.component_pixel_count < 6000 or width < 80 or height < 80:
            continue
        if record.authorization_state in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            if _component_auth_has_decorative_evidence(record):
                _component_auth_add_contract_diagnostic(
                    record,
                    "large_decorative_component_conflicts_with_cleanup_authority",
                    defect=True,
                    visual_review=True,
                    candidate_conflict=True,
                )
            continue
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP and _component_auth_has_decorative_evidence(record):
            _component_auth_add_contract_diagnostic(
                record,
                "large_decorative_component_without_translation_authority_candidate",
                warning=True,
                visual_review=True,
            )


def _component_auth_apply_large_decorative_review_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        and not record.owning_container_ids
        and not record.candidate_cleanup_job_ids
        and record.component_pixel_count >= 400
        and not _component_auth_is_speech_record(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        if len(group) < 2:
            continue
        x0 = min(int(item.component_bbox[0]) for item in group)
        y0 = min(int(item.component_bbox[1]) for item in group)
        x1 = max(int(item.component_bbox[2]) for item in group)
        y1 = max(int(item.component_bbox[3]) for item in group)
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        pixels = sum(int(item.component_pixel_count) for item in group)
        if pixels < 2500 or height < 80:
            continue
        if width < 55 and height < 140:
            continue
        for item in group:
            _component_auth_add_contract_diagnostic(
                item,
                "large_decorative_component_group_without_translation_authority_candidate",
                warning=True,
                visual_review=True,
            )


def _component_auth_apply_unowned_display_neighbor_conflicts(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    unowned = [
        record
        for record in records
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        and not record.candidate_container_ids
        and not record.candidate_cleanup_job_ids
        and not record.owning_container_ids
        and _component_auth_is_large_display_fragment(record)
    ]
    if not unowned:
        return
    for record in records:
        if record.authorization_state not in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            continue
        if not _component_auth_is_large_display_fragment(record):
            continue
        neighbor = next((item for item in unowned if _component_auth_display_fragments_adjacent(record, item)), None)
        if neighbor is None:
            continue
        _component_auth_add_contract_diagnostic(
            neighbor,
            "unowned_display_fragment_protected_near_cleanup_claim_candidate",
            warning=True,
            visual_review=True,
        )
        _component_auth_add_contract_diagnostic(
            record,
            "unowned_display_neighbor_conflicts_with_cleanup_authority",
            defect=True,
            visual_review=True,
            candidate_conflict=True,
        )


def _component_auth_is_large_display_fragment(record: TextAreaComponentAuthorizationRecord) -> bool:
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    return bool(record.component_pixel_count >= 1000 and (height >= 55 or width >= 55))


def _component_auth_display_fragments_adjacent(
    record: TextAreaComponentAuthorizationRecord,
    other: TextAreaComponentAuthorizationRecord,
) -> bool:
    if len(record.component_bbox) < 4 or len(other.component_bbox) < 4:
        return False
    box = [int(item) for item in record.component_bbox[:4]]
    other_box = [int(item) for item in other.component_bbox[:4]]
    if _component_auth_bbox_gap(box, other_box) > 45:
        return False
    x_overlap = max(0, min(box[2], other_box[2]) - max(box[0], other_box[0]))
    y_overlap = max(0, min(box[3], other_box[3]) - max(box[1], other_box[1]))
    min_width = max(1, min(box[2] - box[0], other_box[2] - other_box[0]))
    min_height = max(1, min(box[3] - box[1], other_box[3] - other_box[1]))
    return bool(x_overlap / float(min_width) >= 0.25 or y_overlap / float(min_height) >= 0.25)


def _component_auth_bboxes_adjacent(a: Sequence[int], b: Sequence[int]) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    ax0, ay0, ax1, ay1 = [int(item) for item in a[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in b[:4]]
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0, max(ay0, by0) - min(ay1, by1))
    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
    if x_overlap > 0 and y_gap <= 90:
        return True
    if y_overlap > 0 and x_gap <= 90:
        return True
    return x_gap <= 80 and y_gap <= 80


def _component_auth_xywh_to_xyxy(value: Any, mask_shape: tuple[int, int]) -> List[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return _component_auth_clip_bbox([x, y, x + w, y + h], mask_shape)


def _component_auth_valid_or_xywh_bbox(value: Any, mask_shape: tuple[int, int]) -> List[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        a, b, c, d = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if c > a and d > b:
        return _component_auth_clip_bbox([a, b, c, d], mask_shape)
    if c > 0 and d > 0:
        return _component_auth_clip_bbox([a, b, a + c, b + d], mask_shape)
    return None


def _component_auth_clip_bbox(bbox: Sequence[int], mask_shape: tuple[int, int]) -> List[int] | None:
    height, width = int(mask_shape[0]), int(mask_shape[1])
    x0 = max(0, min(width, int(bbox[0])))
    y0 = max(0, min(height, int(bbox[1])))
    x1 = max(0, min(width, int(bbox[2])))
    y1 = max(0, min(height, int(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _component_auth_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _component_auth_merged_values(candidates: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        for value in _component_auth_list(candidate.get(key) or []):
            if value not in output:
                output.append(value)
    return output


def _component_auth_unique_ids(candidates: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        value = str(candidate.get(key) or "")
        if value and value not in output:
            output.append(value)
    return output


def _component_auth_unique_jobs(candidates: Sequence[Mapping[str, Any]]) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        for value in candidate.get("cleanup_job_ids") or []:
            text = str(value or "")
            if text and text not in output:
                output.append(text)
    return output


def _component_auth_job_binding(
    semantic_authorization_state: str,
    cleanup_candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, List[str]]:
    if _component_auth_family(semantic_authorization_state) != "cleanup":
        if semantic_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
            return "not_applicable_semantic_conflict", "", "", []
        return "not_applicable_non_cleanup", "", "", []
    jobs = _component_auth_unique_jobs(cleanup_candidates)
    if not jobs:
        return "missing_cleanup_job", "cleanup_job_binding_contract_error", "", []
    if len(jobs) > 1:
        return "non_unique_cleanup_job", "cleanup_job_binding_contract_error", "", jobs
    return "bound_unique", "", jobs[0], jobs


def _component_auth_first_job_id(candidates: Sequence[Mapping[str, Any]]) -> str:
    jobs = _component_auth_unique_jobs(candidates)
    return jobs[0] if jobs else ""


def _component_auth_safe_id(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "page"))
    text = "_".join(part for part in text.split("_") if part)
    return text or "page"


def build_text_area_plan(
    page_id: str,
    image_path: str | Path,
    image_size: Tuple[int, int],
    bubble_detection_result: Any,
    current_region_records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    finalize_graph: bool = True,
) -> TextAreaPlan:
    """Build a front-of-pipeline TextAreaPlan from BubbleDetection evidence."""

    started = time.perf_counter()
    result = _result_to_mapping(bubble_detection_result)
    plan = TextAreaPlan(page_id=str(page_id), image_path=str(image_path), image_size=(int(image_size[0]), int(image_size[1])))
    plan.runtime.bubble_detection_generated = bool(result.get("generated"))
    plan.runtime.bubble_detection_cache_hit = result.get("cache_hit")
    plan.runtime.provider_fallback_used = result.get("provider_fallback_used")

    try:
        if not result.get("generated"):
            raise PipelineStageTechnicalError(
                stage=PipelineStage.DETECTION,
                code="bubble_detection_artifact_unavailable",
                message="TextAreaPlan requires a generated BubbleDetection artifact.",
                detail=str(result.get("error") or "bubble detection was not generated"),
                page_id=str(page_id),
                operation="build_text_area_plan",
            )

        luma_image = _load_luma_image(image_path)
        seen: set[str] = set()
        for fused in result.get("fused_containers", []) or []:
            container = _container_from_fused(page_id, fused, image_size, luma_image=luma_image)
            if not container or container.container_id in seen:
                continue
            if _duplicate_semantic_text_area_container(container, plan.containers):
                seen.add(container.container_id)
                continue
            seen.add(container.container_id)
            plan.containers.append(container)

        for evidence in result.get("text_area_model_evidence", []) or []:
            container = _container_from_unlinked_text_area_evidence(page_id, evidence, result, image_size, luma_image=luma_image)
            if not container or container.container_id in seen:
                continue
            seen.add(container.container_id)
            plan.containers.append(container)

        _append_deterministic_top_band_caption_containers(
            plan,
            page_id,
            result,
            image_size,
            luma_image=luma_image,
            seen=seen,
        )
        _append_deterministic_vertical_side_caption_containers(
            plan,
            page_id,
            result,
            image_size,
            luma_image=luma_image,
            seen=seen,
        )
        _append_deterministic_large_sfx_containers(
            plan,
            page_id,
            image_size,
            luma_image=luma_image,
            seen=seen,
        )
        _demote_weak_background_authority_overlapping_protected(plan.containers, image_size)

        if not plan.containers:
            reason = TextAreaFallbackReason(
                reason="bubble_detection_no_usable_text_area_containers_blocked",
                safe_to_ocr=False,
                safe_to_translate=False,
            )
            plan.fallback_reasons.append(reason)
            plan.containers.append(_blocked_full_page_review_container(page_id, image_size, reason.reason))

        for container in plan.containers:
            if container.comic_text_detector_scope_eligible:
                plan.scopes.append(
                    TextAreaScope(
                        scope_id=f"scope_{container.container_id}",
                        page_id=page_id,
                        container_id=container.container_id,
                        bbox=list(container.bbox),
                        container_type=container.container_type,
                        route_intent=container.route_intent,
                        ocr_eligible=container.ocr_eligible,
                        comic_text_detector_scope_eligible=container.comic_text_detector_scope_eligible,
                        fallback_reason=container.fallback_reason,
                        ocr_eligibility_reason=container.ocr_eligibility_reason,
                        text_area_pre_ocr_authority=container.text_area_pre_ocr_authority,
                        text_area_enriched_from_region=container.text_area_enriched_from_region,
                    )
                )
            plan.evidence.append(
                TextAreaPlanEvidence(
                    source="bubble_detection",
                    source_ids=list(container.source_model_ids),
                    confidence=container.confidence,
                    reason_codes=list(container.evidence_reason_codes),
                    conflict_flags=list(container.conflict_flags),
                )
            )

        plan.generated = True
        return _finish_plan(plan, started, finalize_graph=finalize_graph)
    except PipelineStageTechnicalError:
        raise
    except Exception as exc:
        plan.runtime.error = f"{type(exc).__name__}: {exc}"
        raise PipelineStageTechnicalError(
            stage=PipelineStage.DETECTION,
            code="text_area_plan_failed",
            message="TextAreaPlan could not produce a valid page-area artifact.",
            detail=plan.runtime.error,
            page_id=str(page_id),
            operation="build_text_area_plan",
        ) from exc


def enrich_text_area_plan_with_region_records(
    plan: TextAreaPlan | Mapping[str, Any] | None,
    current_region_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a post-region audit view of a TextAreaPlan.

    The returned plan is diagnostic/enrichment data only. It documents where
    deterministic OCR-time routing agreed with or overrode pre-OCR model
    ownership, but it must not be used to claim stronger pre-OCR gating.
    """

    plan_dict = copy.deepcopy(plan.to_dict() if isinstance(plan, TextAreaPlan) else dict(plan or {}))
    if not plan_dict:
        return {}
    plan_dict["stage"] = "post_region_enriched"
    containers = {
        str(container.get("container_id") or ""): container
        for container in plan_dict.get("containers", []) or []
        if str(container.get("container_id") or "")
    }
    enriched_count = 0
    for region in current_region_records or []:
        if not isinstance(region, Mapping):
            continue
        assignment = assign_bbox_to_text_area_plan(plan_dict, region.get("bbox") or [0, 0, 0, 0], detection_source=region.get("text_area_detection_source") or DETECTION_SCOPED)
        container = containers.get(str(assignment.get("text_area_container_id") or ""))
        if not container:
            continue
        update = _region_enrichment_update(region, container)
        if not update:
            continue
        _apply_container_enrichment(container, update)
        enriched_count += 1
    plan_dict["summary"] = _summary_for_container_dicts(plan_dict.get("containers") or [], len(plan_dict.get("fallback_reasons") or []), plan_dict.get("runtime", {}).get("compatibility_mode"))
    plan_dict["summary"]["enriched_from_region_count"] = enriched_count
    return plan_dict


def finalize_text_area_plan_with_ctd_boundary_evidence(
    plan: TextAreaPlan,
    *,
    scoped_detection_candidates: Sequence[Mapping[str, Any]] | None,
    full_page_detection_candidates: Sequence[Mapping[str, Any]] | None,
) -> TextAreaPlan:
    """Finalize graph topology after TextAreaPlan adjudicates CTD geometry."""

    if not isinstance(plan, TextAreaPlan):
        raise TypeError("CTD parent-boundary finalization requires a TextAreaPlan object")
    for container in plan.containers:
        role_evidence = dict(container.semantic_role_evidence or {})
        role_evidence.pop("ctd_parent_boundary_evidence_bboxes", None)
        container.semantic_role_evidence = role_evidence

    workflow_containers = [
        container
        for container in plan.containers
        if _text_area_graph_is_workflow_container(container)
    ]
    root_groups, _demoted = _text_area_graph_normalized_root_groups(
        workflow_containers,
        image_size=plan.image_size,
    )
    containers_by_id = {
        str(container.container_id or ""): container
        for container in plan.containers
        if str(container.container_id or "")
    }
    scoped_by_container = _ctd_parent_boundary_candidates_by_container(
        scoped_detection_candidates or [],
        containers_by_id,
    )
    full_by_container = _ctd_parent_boundary_candidates_by_container(
        full_page_detection_candidates or [],
        containers_by_id,
    )

    for group in root_groups:
        members = [
            member
            for member in group.get("containers") or []
            if isinstance(member, TextAreaContainer)
        ]
        member_ids = {
            str(member.container_id or "")
            for member in members
            if str(member.container_id or "")
        }
        existing: List[Tuple[TextAreaContainer, Dict[str, Any]]] = []
        for member in members:
            for entry in _text_area_graph_parent_boundary_entries(
                member,
                image_size=plan.image_size,
                luma_image=None,
            ):
                if bool(entry.get("is_explicit_parent_boundary_evidence")):
                    existing.append((member, dict(entry)))

        if not existing:
            root_candidates = _ctd_dedupe_parent_boundary_candidates(
                [
                    candidate
                    for container_id in member_ids
                    for candidate in full_by_container.get(container_id, [])
                ]
            )
            for candidate in root_candidates:
                owner = containers_by_id.get(str(candidate.get("container_id") or ""))
                if owner is not None:
                    _attach_ctd_parent_boundary_records(owner, [candidate])
            continue

        if len(existing) != 1:
            continue
        owner, provider_entry = existing[0]
        owner_id = str(owner.container_id or "")
        split_candidates = _ctd_vertical_parent_island_partition(
            _text_area_graph_bbox_from_entry(provider_entry),
            scoped_by_container.get(owner_id, []),
        )
        if split_candidates:
            _attach_ctd_parent_boundary_records(owner, split_candidates)

    plan.root_parent_child_plan = build_text_area_root_parent_child_plan(plan)
    _update_text_area_graph_summary(plan)
    return plan


def _ctd_parent_boundary_candidates_by_container(
    candidates: Sequence[Mapping[str, Any]],
    containers_by_id: Mapping[str, TextAreaContainer],
) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates or []):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("detection_id") or f"ctd_{index:04d}")
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        container_id = str(candidate.get("text_area_container_id") or "")
        container = containers_by_id.get(container_id)
        if container is None:
            continue
        route = str(candidate.get("route_intent") or "")
        auth = str(
            candidate.get("semantic_authorization_state")
            or candidate.get("cleanup_authorization")
            or ""
        )
        if route not in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}:
            continue
        if auth not in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            continue
        if not bool(candidate.get("authorization_explicit")):
            continue
        if bool(candidate.get("must_not_mutate")):
            continue
        bbox = _text_area_graph_bbox_from_any(candidate.get("bbox") or [])
        if not bbox or _inside_ratio_xywh(bbox, container.bbox) < 0.50:
            continue
        output.setdefault(container_id, []).append(
            {
                "candidate_id": candidate_id,
                "container_id": container_id,
                "bbox": bbox,
                "confidence": float(candidate.get("confidence") or 0.0),
            }
        )
    return output


def _ctd_dedupe_parent_boundary_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    ordered = sorted(
        (dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)),
        key=lambda item: (
            -float(item.get("confidence") or 0.0),
            -_bbox_area_xywh(item.get("bbox") or []),
            str(item.get("candidate_id") or ""),
        ),
    )
    output: List[Dict[str, Any]] = []
    for candidate in ordered:
        bbox = candidate.get("bbox") or []
        if any(
            _ctd_bbox_intersection_min_ratio(bbox, existing.get("bbox") or []) >= 0.72
            for existing in output
        ):
            continue
        output.append(candidate)
    return output


def _ctd_vertical_parent_island_partition(
    provider_bbox: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    px, py, pw, ph = _coerce_xywh(provider_bbox)
    if pw <= 0 or ph <= 0:
        return []
    provider_area = float(max(1, pw * ph))
    eligible: List[Dict[str, Any]] = []
    for candidate in _ctd_dedupe_parent_boundary_candidates(candidates):
        bbox = _text_area_graph_bbox_from_any(candidate.get("bbox") or [])
        if not bbox or _bbox_area_xywh(bbox) < provider_area * 0.12:
            continue
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
        if not (px <= center_x <= px + pw and py <= center_y <= py + ph):
            continue
        eligible.append({**candidate, "bbox": bbox})
    if len(eligible) != 2:
        return []
    upper, lower = sorted(eligible, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    upper_bbox = upper["bbox"]
    lower_bbox = lower["bbox"]
    metrics = _text_area_graph_axis_overlap_metrics(upper_bbox, lower_bbox)
    if not metrics or metrics["x_overlap_min_ratio"] < 0.55:
        return []
    minimum_height = float(max(1, min(upper_bbox[3], lower_bbox[3])))
    minimum_gap = max(4.0, minimum_height * 0.03)
    maximum_gap = max(12.0, float(ph) * 0.25)
    if not (minimum_gap <= metrics["y_gap"] <= maximum_gap):
        return []
    coverage = (
        _bbox_area_xywh(upper_bbox) + _bbox_area_xywh(lower_bbox)
    ) / provider_area
    if coverage < 0.32:
        return []
    return _ctd_pad_vertical_parent_island_partition(
        [px, py, pw, ph],
        [upper, lower],
    )


def _ctd_pad_vertical_parent_island_partition(
    provider_bbox: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    px, py, pw, ph = _coerce_xywh(provider_bbox)
    if pw <= 0 or ph <= 0 or len(candidates) != 2:
        return []
    upper, lower = [dict(candidate) for candidate in candidates]
    ux, uy, uw, uh = _coerce_xywh(upper.get("bbox") or [])
    lx, ly, lw, lh = _coerce_xywh(lower.get("bbox") or [])
    separator = int(round(((uy + uh) + ly) / 2.0))
    separator = max(py + 1, min(py + ph - 1, separator))

    def padded(
        candidate: Dict[str, Any],
        bbox: Sequence[Any],
        *,
        top_limit: int,
        bottom_limit: int,
    ) -> Dict[str, Any]:
        x, y, width, height = _coerce_xywh(bbox)
        pad_x = max(6, int(round(width * 0.10)))
        pad_y = max(6, int(round(height * 0.10)))
        x0 = max(px, x - pad_x)
        x1 = min(px + pw, x + width + pad_x)
        y0 = max(top_limit, y - pad_y)
        y1 = min(bottom_limit, y + height + pad_y)
        return {
            **candidate,
            "bbox": [x0, y0, max(1, x1 - x0), max(1, y1 - y0)],
        }

    return [
        padded(upper, [ux, uy, uw, uh], top_limit=py, bottom_limit=separator),
        padded(lower, [lx, ly, lw, lh], top_limit=separator, bottom_limit=py + ph),
    ]


def _ctd_bbox_intersection_min_ratio(
    first: Sequence[Any],
    second: Sequence[Any],
) -> float:
    ax, ay, aw, ah = _coerce_xywh(first)
    bx, by, bw, bh = _coerce_xywh(second)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0,
        min(ay + ah, by + bh) - max(ay, by),
    )
    return float(intersection) / float(max(1, min(aw * ah, bw * bh)))


def _attach_ctd_parent_boundary_records(
    container: TextAreaContainer,
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    role_evidence = dict(container.semantic_role_evidence or {})
    records: List[Dict[str, Any]] = []
    semantic_records = list(role_evidence.get("semantic_evidence_records") or [])
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        bbox = _text_area_graph_bbox_from_any(candidate.get("bbox") or [])
        if not candidate_id or not bbox:
            continue
        evidence_id = f"{PROVIDER_COMIC_TEXT_DETECTOR_PARENT_BOUNDARY}:{candidate_id}"
        records.append(
            {
                "evidence_id": evidence_id,
                "provider": PROVIDER_COMIC_TEXT_DETECTOR_PARENT_BOUNDARY,
                "class_name": "ctd_text_block",
                "bbox": bbox,
                "confidence": float(candidate.get("confidence") or 0.0),
                "is_explicit_parent_boundary_evidence": True,
            }
        )
        semantic_records.append(
            {
                "evidence_id": evidence_id,
                "provider": PROVIDER_COMIC_TEXT_DETECTOR_PARENT_BOUNDARY,
                "provider_version": SEMANTIC_EVIDENCE_PROVIDER_VERSION,
                "semantic_target": _text_area_graph_semantic_role(container),
                "authority_state": str(
                    container.semantic_authorization_state
                    or container.cleanup_authorization
                    or ""
                ),
                "bbox": bbox,
                "source_model_ids": [candidate_id],
                "source_container_ids": [str(container.container_id or "")],
                "basis": "typed_source_text_free_ctd_text_block_geometry",
                "confidence_tier": str(candidate.get("confidence") or ""),
                "reason_codes": ["typed_source_text_free_ctd_parent_boundary"],
                "negative_evidence": [],
                "requires_review": False,
                "created_by": "model",
                "page_id": str(container.page_id or ""),
            }
        )
    if not records:
        return
    role_evidence["ctd_parent_boundary_evidence_bboxes"] = records
    role_evidence["semantic_evidence_records"] = semantic_records
    container.semantic_role_evidence = role_evidence


def assign_bbox_to_text_area_plan(
    plan: TextAreaPlan | Mapping[str, Any] | None,
    bbox: Sequence[Any],
    *,
    min_overlap: float = 0.12,
    detection_source: str = DETECTION_SCOPED,
) -> Dict[str, Any]:
    containers = _plan_containers(plan)
    box = _xywh_to_xyxy(bbox)
    if not box:
        return _fallback_assignment("invalid_detection_bbox")

    best: tuple[float, float, Mapping[str, Any]] | None = None
    for container in containers:
        cbox = _xywh_to_xyxy(container.get("bbox") or [])
        if not cbox:
            continue
        overlap = _intersection_area(box, cbox) / max(1.0, _area(box))
        center = _center_inside(box, cbox)
        score = overlap + (0.35 if center else 0.0) + _container_priority(container) * 0.01
        if best is None or score > best[0]:
            best = (score, overlap, container)

    if not best or (best[1] < min_overlap and not _center_inside(box, _xywh_to_xyxy(best[2].get("bbox") or []))):
        return _fallback_assignment("no_text_area_container_overlap")

    container = best[2]
    source = detection_source or DETECTION_SCOPED
    if not bool(container.get("ocr_eligible", True)):
        source = DETECTION_BLOCKED
    cleanup_authorization = str(container.get("cleanup_authorization") or "")
    protection_reason = str(container.get("protection_reason") or "")
    eligibility = _container_authorization_eligibility(container)
    oriented_frame = oriented_frame_from_speech_container(container)
    return {
        "text_area_container_id": container.get("container_id"),
        "text_area_semantic_unit_id": container.get("semantic_unit_id") or container.get("container_id"),
        "text_area_semantic_kind": container.get("semantic_kind") or _semantic_kind_for_container_dict(container),
        "text_area_container_type": container.get("container_type") or CONTAINER_UNKNOWN,
        "text_area_route_intent": container.get("route_intent") or ROUTE_REVIEW_FALLBACK,
        "text_area_cleanup_authorization": cleanup_authorization,
        "text_area_must_not_mutate": bool(container.get("must_not_mutate", False)),
        "text_area_protection_reason": protection_reason,
        "text_area_authorization_source_stage": container.get("authorization_source_stage") or container.get("source_stage") or "text_area_plan",
        "text_area_authorization_basis": container.get("authorization_basis") or "",
        "text_area_authorization_explicit": bool(container.get("authorization_explicit", False)),
        "text_area_authorization_field_origin": container.get("authorization_field_origin") or "",
        "text_area_semantic_authorization_state": container.get("semantic_authorization_state") or cleanup_authorization,
        "text_area_ctd_scope_eligible": eligibility["ctd_scope_eligible"],
        "text_area_comic_text_detector_scope_eligible": eligibility["ctd_scope_eligible"],
        "text_area_ocr_eligible": eligibility["ocr_eligible"],
        "text_area_translation_eligible": eligibility["translation_eligible"],
        "text_area_render_eligible": eligibility["render_eligible"],
        "text_area_cleanup_executable": eligibility["cleanup_executable"],
        "text_area_detection_source": source,
        "text_area_fallback_reason": container.get("fallback_reason"),
        "text_area_confidence_tier": container.get("confidence_tier") or "low",
        "text_area_container_bbox": list(container.get("bbox") or []),
        "text_area_container_polygon": list(
            oriented_frame.get("polygon") or []
        ),
        "text_area_oriented_frame": oriented_frame,
        "text_area_reason_codes": list(container.get("evidence_reason_codes") or []),
        "text_area_conflict_flags": list(container.get("conflict_flags") or []),
        "text_area_pre_ocr_authority": bool(container.get("text_area_pre_ocr_authority", True)),
        "text_area_enriched_from_region": bool(container.get("text_area_enriched_from_region", False)),
        "text_area_ocr_eligibility_reason": container.get("ocr_eligibility_reason") or "",
        "text_area_overlap_ratio": round(best[1], 6),
    }


def build_scoped_detection_candidates(
    page_id: str,
    detections: Sequence[Tuple[Sequence[Sequence[float]], float]],
    plan: TextAreaPlan | Mapping[str, Any] | None,
    *,
    detection_source: str = DETECTION_SCOPED,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx, (polygon, conf) in enumerate(detections):
        bbox = polygon_to_bbox(polygon)
        assignment = assign_bbox_to_text_area_plan(plan, bbox, detection_source=detection_source)
        candidate = ScopedDetectionCandidate(
            detection_id=f"d{idx:03d}",
            page_id=page_id,
            bbox=bbox,
            polygon=[[float(x), float(y)] for x, y in polygon],
            confidence=float(conf or 0.0),
            text_area_container_id=assignment.get("text_area_container_id"),
            container_type=assignment.get("text_area_container_type") or CONTAINER_UNKNOWN,
            route_intent=assignment.get("text_area_route_intent") or ROUTE_REVIEW_FALLBACK,
            ocr_eligible=bool(assignment.get("text_area_ocr_eligible")),
            detection_source=assignment.get("text_area_detection_source") or DETECTION_COMPATIBILITY_FALLBACK,
            fallback_reason=assignment.get("text_area_fallback_reason"),
            reason_codes=list(assignment.get("text_area_reason_codes") or []),
            conflict_flags=list(assignment.get("text_area_conflict_flags") or []),
            text_area_pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
            text_area_enriched_from_region=bool(assignment.get("text_area_enriched_from_region", False)),
            cleanup_authorization=assignment.get("text_area_cleanup_authorization") or "",
            must_not_mutate=bool(assignment.get("text_area_must_not_mutate", False)),
            protection_reason=assignment.get("text_area_protection_reason") or "",
            pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
            source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
            authorization_source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
            authorization_basis=assignment.get("text_area_authorization_basis") or "",
            authorization_explicit=bool(assignment.get("text_area_authorization_explicit", False)),
            authorization_field_origin=assignment.get("text_area_authorization_field_origin") or "",
            semantic_authorization_state=assignment.get("text_area_semantic_authorization_state") or "",
            ctd_scope_eligible=bool(
                assignment.get("text_area_ctd_scope_eligible", assignment.get("text_area_comic_text_detector_scope_eligible", False))
            ),
            translation_eligible=bool(assignment.get("text_area_translation_eligible", False)),
            render_eligible=bool(assignment.get("text_area_render_eligible", False)),
            cleanup_executable=bool(assignment.get("text_area_cleanup_executable", False)),
        )
        candidates.append(candidate.to_dict())
    return candidates


def build_scoped_ocr_candidate(
    *,
    page_id: str,
    region_id: str,
    bbox: Sequence[Any],
    assignment: Mapping[str, Any],
    ocr_text: str,
    ocr_confidence: float,
    accepted: bool,
) -> Dict[str, Any]:
    ctd_scope_eligible = bool(
        assignment.get("text_area_ctd_scope_eligible", assignment.get("text_area_comic_text_detector_scope_eligible", False))
    )
    return ScopedOcrCandidate(
        ocr_candidate_id=f"ocr_{region_id}",
        page_id=page_id,
        region_id=region_id,
        bbox=[int(v) for v in bbox[:4]],
        text_area_container_id=assignment.get("text_area_container_id"),
        container_type=assignment.get("text_area_container_type") or CONTAINER_UNKNOWN,
        route_intent=assignment.get("text_area_route_intent") or ROUTE_REVIEW_FALLBACK,
        crop_source=assignment.get("text_area_detection_source") or DETECTION_COMPATIBILITY_FALLBACK,
        fallback_reason=assignment.get("text_area_fallback_reason"),
        ocr_eligible=bool(assignment.get("text_area_ocr_eligible", False)),
        ctd_scope_eligible=ctd_scope_eligible,
        ocr_text=ocr_text,
        ocr_confidence=float(ocr_confidence or 0.0),
        accepted=bool(accepted),
        reason_codes=list(assignment.get("text_area_reason_codes") or []),
        text_area_pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
        text_area_enriched_from_region=bool(assignment.get("text_area_enriched_from_region", False)),
        cleanup_authorization=assignment.get("text_area_cleanup_authorization") or "",
        must_not_mutate=bool(assignment.get("text_area_must_not_mutate", False)),
        protection_reason=assignment.get("text_area_protection_reason") or "",
        pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
        source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
        authorization_source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
        authorization_basis=assignment.get("text_area_authorization_basis") or "",
        authorization_explicit=bool(assignment.get("text_area_authorization_explicit", False)),
        authorization_field_origin=assignment.get("text_area_authorization_field_origin") or "",
        semantic_authorization_state=assignment.get("text_area_semantic_authorization_state") or "",
        translation_eligible=bool(assignment.get("text_area_translation_eligible", False)),
        render_eligible=bool(assignment.get("text_area_render_eligible", False)),
        cleanup_executable=bool(assignment.get("text_area_cleanup_executable", False)),
    ).to_dict()


def apply_text_area_assignment_to_region(region: Dict[str, Any], assignment: Mapping[str, Any]) -> None:
    if not isinstance(region, dict):
        return
    for key in (
        "text_area_container_id",
        "text_area_semantic_unit_id",
        "text_area_semantic_kind",
        "text_area_container_type",
        "text_area_route_intent",
        "text_area_ocr_eligible",
        "text_area_detection_source",
        "text_area_fallback_reason",
        "text_area_confidence_tier",
        "text_area_container_bbox",
        "text_area_container_polygon",
        "text_area_oriented_frame",
        "text_area_reason_codes",
        "text_area_conflict_flags",
        "text_area_pre_ocr_authority",
        "text_area_enriched_from_region",
        "text_area_ocr_eligibility_reason",
        "text_area_cleanup_authorization",
        "text_area_must_not_mutate",
        "text_area_protection_reason",
        "text_area_authorization_source_stage",
        "text_area_authorization_basis",
        "text_area_authorization_explicit",
        "text_area_authorization_field_origin",
        "text_area_semantic_authorization_state",
        "text_area_ctd_scope_eligible",
        "text_area_comic_text_detector_scope_eligible",
        "text_area_translation_eligible",
        "text_area_render_eligible",
        "text_area_cleanup_executable",
    ):
        region[key] = assignment.get(key)
    render = region.setdefault("render", {})
    render.update({key: assignment.get(key) for key in region if key.startswith("text_area_")})


def write_text_area_plan_artifacts(
    *,
    page_dir: str | Path,
    image_path: str | Path,
    plan: TextAreaPlan | Mapping[str, Any] | None,
    pre_ocr_plan: TextAreaPlan | Mapping[str, Any] | None = None,
    scoped_detection_candidates: Sequence[Mapping[str, Any]] | None = None,
    scoped_ocr_candidates: Sequence[Mapping[str, Any]] | None = None,
    fallback_decisions: Sequence[Mapping[str, Any]] | None = None,
    blocked_text_area_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    plan_dict = plan.to_dict() if isinstance(plan, TextAreaPlan) else dict(plan or {})
    paths = {
        "text_area_plan": page_dir / "text_area_plan.json",
        "text_area_plan_pre_ocr": page_dir / "text_area_plan_pre_ocr.json",
        "text_area_plan_enriched": page_dir / "text_area_plan_enriched.json",
        "text_area_plan_overlay": page_dir / "text_area_plan_overlay.jpg",
        "text_area_plan_summary": page_dir / "text_area_plan_summary.md",
        "scoped_detection_candidates": page_dir / "scoped_detection_candidates.json",
        "scoped_ocr_candidates": page_dir / "scoped_ocr_candidates.json",
        "fallback_decisions": page_dir / "fallback_decisions.json",
        "blocked_text_area_candidates": page_dir / "blocked_text_area_candidates.json",
        "caption_localization_candidates": page_dir / "caption_localization_candidates.json",
    }
    pre_ocr_dict = pre_ocr_plan.to_dict() if isinstance(pre_ocr_plan, TextAreaPlan) else dict(pre_ocr_plan or plan_dict or {})
    paths["text_area_plan"].write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["text_area_plan_pre_ocr"].write_text(json.dumps(pre_ocr_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["text_area_plan_enriched"].write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["scoped_detection_candidates"].write_text(json.dumps(list(scoped_detection_candidates or []), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["scoped_ocr_candidates"].write_text(json.dumps(list(scoped_ocr_candidates or []), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["fallback_decisions"].write_text(json.dumps(list(fallback_decisions or []), ensure_ascii=False, indent=2), encoding="utf-8")
    blocked = list(blocked_text_area_candidates or []) + _blocked_containers_from_plan(plan_dict)
    paths["blocked_text_area_candidates"].write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8")
    caption_candidates = _caption_localization_candidates_from_plan(plan_dict)
    paths["caption_localization_candidates"].write_text(json.dumps(caption_candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    graph_plan = plan_dict.get("root_parent_child_plan")
    if isinstance(graph_plan, Mapping):
        paths.update(
            {
                key: Path(value)
                for key, value in write_root_parent_child_overlay_artifacts(
                    page_dir=page_dir,
                    image_path=image_path,
                    graph_plan=graph_plan,
                ).items()
            }
        )
    _write_plan_summary(paths["text_area_plan_summary"], plan_dict, scoped_detection_candidates or [], scoped_ocr_candidates or [], fallback_decisions or [], blocked)
    _write_plan_overlay(paths["text_area_plan_overlay"], image_path, plan_dict, scoped_detection_candidates or [])
    return {key: str(path) for key, path in paths.items()}


def write_root_parent_child_overlay_artifacts(
    *,
    page_dir: str | Path,
    image_path: str | Path,
    graph_plan: Mapping[str, Any] | None,
) -> Dict[str, str]:
    """Write visual graph overlays from an already-built TextAreaPlan graph."""

    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    graph = dict(graph_plan or {})
    paths: Dict[str, Path] = {
        "root_parent_child_plan": page_dir / "root_parent_child_plan.json",
    }
    paths["root_parent_child_plan"].write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    if graph:
        paths.update(
            {
                "root_workflow_nodes_overlay": page_dir / "root_workflow_nodes_overlay.jpg",
                "parent_boundary_overlay": page_dir / "parent_boundary_overlay.jpg",
                "child_slot_overlay": page_dir / "child_slot_overlay.jpg",
                "root_parent_overlay": page_dir / "root_parent_overlay.jpg",
            }
        )
        _write_graph_plan_overlay(
            paths["root_workflow_nodes_overlay"],
            image_path,
            graph,
            layer="root",
        )
        _write_graph_plan_overlay(
            paths["parent_boundary_overlay"],
            image_path,
            graph,
            layer="parent",
        )
        _write_graph_plan_overlay(
            paths["child_slot_overlay"],
            image_path,
            graph,
            layer="child",
        )
        _write_graph_plan_overlay(
            paths["root_parent_overlay"],
            image_path,
            graph,
            layer="root_parent",
        )
    return {key: str(path) for key, path in paths.items()}


def polygon_to_bbox(polygon: Sequence[Sequence[Any]]) -> List[int]:
    xs: List[float] = []
    ys: List[float] = []
    for point in polygon or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except Exception:
            continue
    if not xs or not ys:
        return [0, 0, 0, 0]
    x0 = int(math.floor(min(xs)))
    y0 = int(math.floor(min(ys)))
    x1 = int(math.ceil(max(xs)))
    y1 = int(math.ceil(max(ys)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _region_enrichment_update(region: Mapping[str, Any], container: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip()
    flags = region.get("flags") if isinstance(region.get("flags"), Mapping) else {}
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    cleanup = str(region.get("cleanup_mode") or render.get("cleanup_mode") or "").strip()
    reason = str(region.get("classification_reason") or render.get("classification_reason") or "").strip()
    current_type = str(container.get("container_type") or "")
    current_route = str(container.get("route_intent") or "")
    current_is_translatable = current_route in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}

    if (
        reason in DECORATIVE_REGION_REASONS
        or semantic in {"decorative_text", "sfx"}
        or (cleanup == "preserve" and bool(flags.get("ignore")))
    ):
        if current_is_translatable:
            return None
        return {
            "container_type": CONTAINER_SFX,
            "route_intent": ROUTE_PRESERVE_SFX,
            "ocr_eligible": False,
            "comic_text_detector_scope_eligible": False,
            "confidence_tier": "conflict_preserve_wins",
            "fallback_reason": "post_region_deterministic_preserve_evidence",
            "ocr_eligibility_reason": "blocked_deterministic_preserve_evidence",
            "reason_code": f"text_area_plan:post_region_sfx_decorative:{reason or semantic or cleanup}",
        }

    if reason in CAPTION_REGION_REASONS or semantic == "background_text" or bool(flags.get("bg_text")):
        if current_type == CONTAINER_SPEECH:
            return None
        return {
            "container_type": CONTAINER_CAPTION,
            "route_intent": ROUTE_TRANSLATE_CAPTION,
            "ocr_eligible": True,
            "comic_text_detector_scope_eligible": True,
            "fallback_reason": "post_region_deterministic_caption_evidence",
            "ocr_eligibility_reason": "caption_background_container",
            "reason_code": f"text_area_plan:post_region_caption:{reason or semantic}",
        }

    if semantic == "speech_bubble" or reason in SPEECH_REGION_REASONS:
        if current_type != CONTAINER_UNKNOWN:
            return None
        return {
            "container_type": CONTAINER_SPEECH,
            "route_intent": ROUTE_TRANSLATE_SPEECH,
            "ocr_eligible": True,
            "comic_text_detector_scope_eligible": True,
            "fallback_reason": "post_region_deterministic_speech_evidence",
            "ocr_eligibility_reason": "speech_container_after_region_evidence",
            "reason_code": f"text_area_plan:post_region_speech:{reason or semantic}",
        }

    return None


def _apply_container_enrichment(container: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    changed = False
    for key in (
        "container_type",
        "route_intent",
        "ocr_eligible",
        "comic_text_detector_scope_eligible",
        "confidence_tier",
        "fallback_reason",
        "ocr_eligibility_reason",
    ):
        if key in update and container.get(key) != update.get(key):
            container[key] = update.get(key)
            changed = True
    if not changed:
        return
    container["text_area_pre_ocr_authority"] = False
    container["text_area_enriched_from_region"] = True
    reasons = list(container.get("evidence_reason_codes") or [])
    reason_code = str(update.get("reason_code") or "")
    if reason_code and reason_code not in reasons:
        reasons.append(reason_code)
    if "text_area_plan:post_region_enriched_audit_only" not in reasons:
        reasons.append("text_area_plan:post_region_enriched_audit_only")
    container["evidence_reason_codes"] = reasons
    container["human_review_required"] = True


def _summary_for_container_dicts(
    containers: Sequence[Mapping[str, Any]],
    fallback_count: int,
    compatibility_mode: Any,
) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    by_intent: Dict[str, int] = {}
    blocked_by_type: Dict[str, int] = {}
    ocr_eligible = 0
    scope_eligible = 0
    review_only_blocked = 0
    for container in containers:
        ctype = str(container.get("container_type") or CONTAINER_UNKNOWN)
        intent = str(container.get("route_intent") or ROUTE_REVIEW_FALLBACK)
        by_type[ctype] = by_type.get(ctype, 0) + 1
        by_intent[intent] = by_intent.get(intent, 0) + 1
        if bool(container.get("ocr_eligible")):
            ocr_eligible += 1
        if bool(container.get("comic_text_detector_scope_eligible")):
            scope_eligible += 1
        if not bool(container.get("ocr_eligible")) or not bool(container.get("comic_text_detector_scope_eligible")):
            blocked_by_type[ctype] = blocked_by_type.get(ctype, 0) + 1
            if ctype == CONTAINER_UNKNOWN:
                review_only_blocked += 1
    return {
        "container_count": len(containers),
        "scope_count": scope_eligible,
        "by_container_type": by_type,
        "by_route_intent": by_intent,
        "ocr_eligible_containers": ocr_eligible,
        "ctd_scope_eligible_containers": scope_eligible,
        "blocked_containers_by_type": blocked_by_type,
        "blocked_sfx_decorative_containers": blocked_by_type.get(CONTAINER_SFX, 0),
        "blocked_review_only_unknown_containers": review_only_blocked,
        "fallback_count": int(fallback_count or 0),
        "compatibility_mode": compatibility_mode,
    }


def _container_value(container: Mapping[str, Any] | TextAreaContainer, key: str, default: Any = "") -> Any:
    if isinstance(container, TextAreaContainer):
        return getattr(container, key, default)
    return container.get(key, default)


def _container_list_value(container: Mapping[str, Any] | TextAreaContainer, key: str) -> List[str]:
    value = _container_value(container, key, [])
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _container_semantic_role_evidence(container: Mapping[str, Any] | TextAreaContainer) -> Dict[str, Any]:
    value = _container_value(container, "semantic_role_evidence", {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _semantic_role_values(role_evidence: Mapping[str, Any], key: str) -> List[str]:
    value = role_evidence.get(key)
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _semantic_role_state_values(role_evidence: Mapping[str, Any], key: str) -> List[str]:
    return [item for item in _semantic_role_values(role_evidence, key) if item in COMPONENT_AUTHORIZATION_STATES]


def _semantic_role_has_state(role_evidence: Mapping[str, Any], key: str, state: str) -> bool:
    return state in set(_semantic_role_state_values(role_evidence, key))


def _semantic_evidence_records(role_evidence: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = role_evidence.get("semantic_evidence_records") if isinstance(role_evidence, Mapping) else None
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return []
    output: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, Mapping):
            normalized = dict(record)
            provider = str(normalized.get("provider") or "")
            if provider:
                output.append(normalized)
    return output


def _semantic_target_for_authority_state(state: str) -> str:
    if state == AUTH_CLEANUP_TRANSLATE_SPEECH:
        return SEMANTIC_KIND_SPEECH
    if state == AUTH_CLEANUP_TRANSLATE_BACKGROUND:
        return SEMANTIC_KIND_BACKGROUND_NARRATION
    if state == AUTH_CLEANUP_TRANSLATE_CAPTION:
        return SEMANTIC_KIND_CAPTION
    if state == AUTH_PROTECT_ART_OR_NON_TEXT:
        return SEMANTIC_KIND_ART_OR_NON_TEXT
    if state == AUTH_PROTECT_SFX_DECORATIVE:
        return SEMANTIC_TARGET_SFX_DECORATIVE
    return SEMANTIC_TARGET_REVIEW_UNKNOWN


def _semantic_evidence_provider_for_kind(
    evidence_kind: str,
    role_evidence: Mapping[str, Any],
    state: str,
) -> str:
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    text = " ".join(
        [
            evidence_kind,
            str(role_evidence.get("source") or ""),
            " ".join(role_signals),
            " ".join(_semantic_role_values(role_evidence, "ogkalu_class_names")),
            " ".join(_semantic_role_values(role_evidence, "current_region_roles")),
            " ".join(_semantic_role_values(role_evidence, "evidence_source_list")),
        ]
    ).lower()
    if "deterministic_side" in text or "side_narration" in text or "vertical_side" in text:
        return PROVIDER_TEXTAREA_DETERMINISTIC_SIDE_NARRATION
    if "deterministic_top" in text or "top_band" in text or "top_caption" in text:
        return PROVIDER_TEXTAREA_DETERMINISTIC_TOP_BAND
    if "deterministic_large_sfx" in text or "large_sfx" in text:
        return PROVIDER_TEXTAREA_DETERMINISTIC_LARGE_SFX
    if "current_region" in text or "current_" in evidence_kind:
        return PROVIDER_CURRENT_REGION_SEMANTIC
    has_positive_speech_mask_evidence = bool(
        role_signals & {"speech_bubble_mask_evidence", "kitsumed_speech_mask_evidence"}
    )
    if has_positive_speech_mask_evidence or "kitsumed_speech_mask" in text or "typed_model_speech_bubble_mask" in evidence_kind:
        return PROVIDER_KITSUMED_SPEECH_MASK
    if state == AUTH_PROTECT_SFX_DECORATIVE or any(token in text for token in ("sfx", "decorative")):
        return PROVIDER_OGKALU_SFX_DECORATIVE
    if state in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION} or "text_free" in text:
        return PROVIDER_OGKALU_TEXT_FREE_BACKGROUND
    if state == AUTH_CLEANUP_TRANSLATE_SPEECH or any(token in text for token in ("text_bubble", "bubble")):
        return PROVIDER_OGKALU_TEXT_BUBBLE
    return PROVIDER_CURRENT_REGION_SEMANTIC


def _semantic_provider_created_by(provider: str) -> str:
    if provider.startswith("textarea_deterministic"):
        return "deterministic-textarea-provider"
    if provider == PROVIDER_CURRENT_REGION_SEMANTIC:
        return "fused-model"
    return "model"


def _semantic_evidence_id(
    provider: str,
    state: str,
    evidence_kind: str,
    role_evidence: Mapping[str, Any],
) -> str:
    ids = (
        _semantic_role_values(role_evidence, "model_evidence_ids")
        or _semantic_role_values(role_evidence, "source_model_ids")
        or _semantic_role_values(role_evidence, "source_container_ids")
    )
    suffix = ids[0] if ids else evidence_kind or state
    safe_suffix = "_".join(str(suffix).replace("\\", "_").replace("/", "_").split())
    return f"{provider}:{state}:{safe_suffix}"


def _semantic_evidence_record(
    *,
    provider: str,
    state: str,
    evidence_kind: str,
    role_evidence: Mapping[str, Any],
    page_id: str = "",
    bbox: Sequence[Any] | None = None,
    source_container_ids: Sequence[Any] | None = None,
    reason_codes: Sequence[Any] | None = None,
    basis: str = "",
    confidence_tier: str = "",
    requires_review: bool = False,
) -> Dict[str, Any]:
    source_model_ids = (
        _semantic_role_values(role_evidence, "source_model_ids")
        or _semantic_role_values(role_evidence, "model_evidence_ids")
    )
    source_ids = [str(item) for item in source_container_ids or [] if str(item)]
    reasons = [str(item) for item in reason_codes or [] if str(item)]
    reasons.extend(item for item in _semantic_role_values(role_evidence, "typed_authority_reason_codes") if item not in reasons)
    if evidence_kind and evidence_kind not in reasons:
        reasons.append(evidence_kind)
    record = {
        "evidence_id": _semantic_evidence_id(provider, state, evidence_kind, role_evidence),
        "provider": provider,
        "provider_version": SEMANTIC_EVIDENCE_PROVIDER_VERSION,
        "semantic_target": _semantic_target_for_authority_state(state),
        "authority_state": state,
        "bbox": [int(item) for item in bbox] if bbox is not None and len(list(bbox)) >= 4 else [],
        "source_model_ids": source_model_ids,
        "source_container_ids": source_ids,
        "basis": basis or evidence_kind,
        "confidence_tier": confidence_tier or str(role_evidence.get("confidence") or ""),
        "reason_codes": sorted(set(reasons)),
        "negative_evidence": [],
        "requires_review": bool(requires_review),
        "created_by": _semantic_provider_created_by(provider),
        "page_id": str(page_id or ""),
    }
    return record


def _semantic_role_evidence_with_provider_records(
    role_evidence: Mapping[str, Any],
    *,
    page_id: str = "",
    bbox: Sequence[Any] | None = None,
    container_id: str = "",
    reason_codes: Sequence[Any] | None = None,
    basis: str = "",
    confidence_tier: str = "",
    requires_review: bool = False,
) -> Dict[str, Any]:
    evidence = dict(role_evidence or {})
    records = _semantic_evidence_records(evidence)
    seen = {
        (
            str(record.get("provider") or ""),
            str(record.get("authority_state") or ""),
            str(record.get("evidence_id") or ""),
        )
        for record in records
    }
    authority_pairs = [
        ("cleanup_authority_states", state)
        for state in _semantic_role_state_values(evidence, "cleanup_authority_states")
    ] + [
        ("protected_authority_states", state)
        for state in _semantic_role_state_values(evidence, "protected_authority_states")
    ]
    evidence_kind = str(evidence.get("authority_evidence_kind") or "")
    for _key, state in authority_pairs:
        provider = _semantic_evidence_provider_for_kind(evidence_kind, evidence, state)
        record = _semantic_evidence_record(
            provider=provider,
            state=state,
            evidence_kind=evidence_kind,
            role_evidence=evidence,
            page_id=page_id,
            bbox=bbox,
            source_container_ids=[container_id] if container_id else [],
            reason_codes=reason_codes,
            basis=basis,
            confidence_tier=confidence_tier,
            requires_review=requires_review,
        )
        key = (
            str(record.get("provider") or ""),
            str(record.get("authority_state") or ""),
            str(record.get("evidence_id") or ""),
        )
        if key not in seen:
            seen.add(key)
            records.append(record)
    if records:
        evidence["semantic_evidence_records"] = records
    return evidence


def _semantic_role_evidence_with_state(
    role_evidence: Mapping[str, Any],
    key: str,
    state: str,
    *,
    evidence_kind: str,
) -> Dict[str, Any]:
    evidence = dict(role_evidence or {})
    states = set(_semantic_role_state_values(evidence, key))
    states.add(state)
    evidence[key] = sorted(states)
    evidence.setdefault("authority_evidence_kind", evidence_kind)
    typed_reasons = set(_semantic_role_values(evidence, "typed_authority_reason_codes"))
    typed_reasons.add(evidence_kind)
    evidence["typed_authority_reason_codes"] = sorted(typed_reasons)
    return _semantic_role_evidence_with_provider_records(evidence)


def _semantic_role_evidence_without_cleanup_authority(
    role_evidence: Mapping[str, Any],
    state: str,
) -> Dict[str, Any]:
    evidence = dict(role_evidence or {})
    cleanup_states = [
        str(item)
        for item in _semantic_role_state_values(evidence, "cleanup_authority_states")
        if str(item) != state
    ]
    if cleanup_states:
        evidence["cleanup_authority_states"] = cleanup_states
    else:
        evidence.pop("cleanup_authority_states", None)
    candidate_states = _text_area_graph_unique_strings(
        _semantic_role_state_values(evidence, "cleanup_candidate_states") + [state]
    )
    if candidate_states:
        evidence["cleanup_candidate_states"] = candidate_states
    typed_reasons = [
        str(item)
        for item in _semantic_role_values(evidence, "typed_authority_reason_codes")
        if "typed_speech_bubble_mask_authority" not in str(item)
    ]
    if typed_reasons:
        evidence["typed_authority_reason_codes"] = typed_reasons
    else:
        evidence.pop("typed_authority_reason_codes", None)
    records = []
    for record in evidence.get("semantic_evidence_records") or []:
        if not isinstance(record, Mapping):
            continue
        if str(record.get("authority_state") or "") == state:
            continue
        records.append(dict(record))
    if records:
        evidence["semantic_evidence_records"] = records
    else:
        evidence.pop("semantic_evidence_records", None)
    evidence["authority_evidence_kind"] = "review_only_mask_primary_candidate"
    return evidence


def _container_marker_text(
    container: Mapping[str, Any] | TextAreaContainer,
    *,
    route: str,
    ctype: str,
    reasons: Sequence[str],
    conflicts: Sequence[str],
    role_evidence: Mapping[str, Any],
) -> str:
    pieces = [
        route,
        ctype,
        str(_container_value(container, "fallback_reason", "") or ""),
        str(_container_value(container, "confidence_tier", "") or _container_value(container, "text_area_confidence_tier", "") or ""),
        " ".join(reasons),
        " ".join(conflicts),
        " ".join(_semantic_role_values(role_evidence, "role_signals")),
        " ".join(_semantic_role_values(role_evidence, "ogkalu_class_names")),
        " ".join(_semantic_role_values(role_evidence, "current_region_roles")),
        " ".join(_semantic_role_values(role_evidence, "conflict_evidence")),
    ]
    return " ".join(str(item) for item in pieces if str(item)).lower()


def _has_art_or_non_text_authority(marker: str, role_evidence: Mapping[str, Any]) -> bool:
    del marker
    return _semantic_role_has_state(role_evidence, "protected_authority_states", AUTH_PROTECT_ART_OR_NON_TEXT)


def _has_recognized_art_or_non_text_evidence(role_evidence: Mapping[str, Any]) -> bool:
    """Typed upstream art/non-text evidence, excluding marker/geometry inference."""

    protected_candidates = set(_semantic_role_state_values(role_evidence, "protected_candidate_states"))
    protected_authority = set(_semantic_role_state_values(role_evidence, "protected_authority_states"))
    if AUTH_PROTECT_ART_OR_NON_TEXT in protected_authority:
        return True
    if AUTH_PROTECT_ART_OR_NON_TEXT not in protected_candidates:
        return False
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    model_classes = set(_semantic_role_values(role_evidence, "ogkalu_class_names"))
    current_roles = set(_semantic_role_values(role_evidence, "current_region_roles"))
    typed_reasons = set(_semantic_role_values(role_evidence, "typed_authority_reason_codes"))
    source = str(role_evidence.get("source") or "")
    evidence_kind = str(role_evidence.get("authority_evidence_kind") or "")
    recognized_signal = bool(role_signals & {"art_non_text_candidate"})
    recognized_class = any(
        token in model_class
        for model_class in model_classes
        for token in ("art", "non_text", "non-text")
    )
    recognized_current_role = "art_or_non_text_preserve" in current_roles
    typed_source = bool(source or evidence_kind or typed_reasons)
    return typed_source and bool(recognized_signal or recognized_class or recognized_current_role)


def _has_sfx_or_decorative_authority(route: str, ctype: str, marker: str, role_evidence: Mapping[str, Any]) -> bool:
    del route, ctype, marker
    return _semantic_role_has_state(role_evidence, "protected_authority_states", AUTH_PROTECT_SFX_DECORATIVE)


def _has_recognized_sfx_or_decorative_evidence(role_evidence: Mapping[str, Any]) -> bool:
    """Typed upstream SFX/decorative evidence, excluding marker/geometry inference."""

    protected_candidates = set(_semantic_role_state_values(role_evidence, "protected_candidate_states"))
    protected_authority = set(_semantic_role_state_values(role_evidence, "protected_authority_states"))
    if AUTH_PROTECT_SFX_DECORATIVE in protected_authority:
        return True
    if AUTH_PROTECT_SFX_DECORATIVE not in protected_candidates:
        return False
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    model_classes = set(_semantic_role_values(role_evidence, "ogkalu_class_names"))
    current_roles = set(_semantic_role_values(role_evidence, "current_region_roles"))
    typed_reasons = set(_semantic_role_values(role_evidence, "typed_authority_reason_codes"))
    source = str(role_evidence.get("source") or "")
    evidence_kind = str(role_evidence.get("authority_evidence_kind") or "")
    recognized_signal = bool(
        role_signals
        & {
            "sfx_candidate",
            "sfx_decorative_candidate",
            "decorative_candidate",
            "art_non_text_candidate",
        }
    )
    recognized_class = any(
        token in model_class
        for model_class in model_classes
        for token in ("sfx", "decorative", "art", "non_text", "non-text")
    )
    recognized_current_role = any(
        role in {"sfx_decorative_preserve", "art_or_non_text_preserve"}
        for role in current_roles
    )
    typed_source = bool(source or evidence_kind or typed_reasons)
    return typed_source and bool(recognized_signal or recognized_class or recognized_current_role)


def _has_speech_authority(route: str, ctype: str, marker: str, role_evidence: Mapping[str, Any]) -> bool:
    del route, ctype, marker
    return _semantic_role_has_state(role_evidence, "cleanup_authority_states", AUTH_CLEANUP_TRANSLATE_SPEECH)


def _has_unresolved_ogkalu_speech_risk(marker: str, role_evidence: Mapping[str, Any]) -> bool:
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    typed_reasons = set(_semantic_role_values(role_evidence, "typed_authority_reason_codes"))
    text = " ".join([marker, " ".join(role_signals), " ".join(typed_reasons)]).lower()
    return UNRESOLVED_OGKALU_SPEECH_RISK_REASON in text


def _caption_background_authority_reason(marker: str, role_evidence: Mapping[str, Any]) -> str:
    del marker
    cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
    if AUTH_CLEANUP_TRANSLATE_CAPTION in cleanup_states:
        return "typed_caption_authority"
    if AUTH_CLEANUP_TRANSLATE_BACKGROUND in cleanup_states:
        return "typed_background_authority"
    current_roles = set(_semantic_role_values(role_evidence, "current_region_roles"))
    if "caption_background" in current_roles:
        return "current_region_caption_background_authority"
    return ""


def _caption_background_review_reason(marker: str, role_evidence: Mapping[str, Any]) -> str:
    if "deterministic_top_band_far_right_caption_search" in marker:
        return "deterministic_far_right_caption_requires_review"
    if "deterministic_top_band" in marker or "top_caption_background_candidate" in marker:
        return "top_band_caption_candidate_requires_upstream_confirmation"
    if "vertical_side_caption_localized_ink" in marker:
        return "deterministic_side_caption_localized_ink_requires_review"
    if "vertical_side_caption_search" in marker or "side_narration_candidate" in marker:
        return "side_narration_candidate_requires_upstream_confirmation"
    if (
        "text_free_review_only" in marker
        or "ogkalu_text_free_without_kitsumed_mask" in marker
        or "caption_background_model_candidate_review" in marker
        ):
            return "text_free_review_only_requires_upstream_confirmation"
    roles = set(_semantic_role_values(role_evidence, "role_signals"))
    if "caption_background_candidate" in roles:
        return "caption_background_candidate_requires_upstream_confirmation"
    if "side_narration_candidate" in roles:
        return "side_narration_candidate_requires_upstream_confirmation"
    return ""


def _cleanup_authority_is_weak_background_when_protected(role_evidence: Mapping[str, Any]) -> bool:
    """Protected evidence wins when cleanup evidence is deterministic/background-only."""

    cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
    if not cleanup_states or AUTH_CLEANUP_TRANSLATE_SPEECH in cleanup_states:
        return False
    if not cleanup_states.issubset({AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}):
        return False
    typed_reasons = {str(item) for item in _semantic_role_values(role_evidence, "typed_authority_reason_codes")}
    current_roles = set(_semantic_role_values(role_evidence, "current_region_roles"))
    if current_roles & {"speech", "caption_background", "cleanup_translate_background", "cleanup_translate_caption"}:
        return False
    strong_cleanup_tokens = (
        "typed_speech",
        "typed_current_region",
        "typed_bright_ogkalu_bubble_speech_authority",
    )
    if any(any(token in reason for token in strong_cleanup_tokens) for reason in typed_reasons):
        return False
    weak_background_tokens = (
        "typed_deterministic_side_narration_background_authority",
        "typed_deterministic_top_band_background_authority",
        "typed_text_free_background_model_authority",
    )
    return any(any(token in reason for token in weak_background_tokens) for reason in typed_reasons)


def _ogkalu_speech_authority_blocked_by_protected(role_evidence: Mapping[str, Any]) -> bool:
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    conflict_evidence = set(_semantic_role_values(role_evidence, "conflict_evidence"))
    return bool(
        "ogkalu_speech_authority_blocked_by_protected" in role_signals
        or "protected_overlap_blocks_ogkalu_speech_authority" in conflict_evidence
    )


def _adjudicate_text_area_semantic_authorization(
    container: Mapping[str, Any] | TextAreaContainer,
    evidence_context: Mapping[str, Any] | None = None,
) -> TextAreaSemanticAdjudication:
    del evidence_context
    route = str(_container_value(container, "route_intent", ROUTE_REVIEW_FALLBACK) or ROUTE_REVIEW_FALLBACK)
    ctype = str(_container_value(container, "container_type", CONTAINER_UNKNOWN) or CONTAINER_UNKNOWN)
    reasons = _container_list_value(container, "evidence_reason_codes")
    conflicts = _container_list_value(container, "conflict_flags")
    role_evidence = _container_semantic_role_evidence(container)
    marker = _container_marker_text(
        container,
        route=route,
        ctype=ctype,
        reasons=reasons,
        conflicts=conflicts,
        role_evidence=role_evidence,
    )
    adjudication_reason_codes: List[str] = ["text_area_plan:semantic_adjudicator_v1"]
    cleanup_authority_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
    protected_authority_states = set(_semantic_role_state_values(role_evidence, "protected_authority_states"))
    recognized_art_or_non_text = _has_recognized_art_or_non_text_evidence(role_evidence)
    recognized_sfx_decorative = _has_recognized_sfx_or_decorative_evidence(role_evidence)
    protected_dominates_weak_cleanup = (
        bool(cleanup_authority_states)
        and bool(protected_authority_states or recognized_sfx_decorative or recognized_art_or_non_text)
        and _cleanup_authority_is_weak_background_when_protected(role_evidence)
    )
    blocked_ogkalu_speech_by_protected = _ogkalu_speech_authority_blocked_by_protected(role_evidence)

    if cleanup_authority_states and (protected_authority_states or recognized_sfx_decorative or recognized_art_or_non_text) and not protected_dominates_weak_cleanup:
        adjudication_reason_codes.append("text_area_plan:typed_cleanup_protected_conflict")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_AMBIGUOUS_COMPONENT_OWNER,
            semantic_authorization_state=AUTH_AMBIGUOUS_COMPONENT_OWNER,
            semantic_kind=SEMANTIC_KIND_UNKNOWN,
            must_not_mutate=True,
            protection_reason="typed_cleanup_protected_conflict",
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if blocked_ogkalu_speech_by_protected:
        adjudication_reason_codes.append("text_area_plan:ogkalu_speech_authority_blocked_by_protected")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_authorization_state=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_kind=SEMANTIC_KIND_UNKNOWN,
            must_not_mutate=True,
            protection_reason="ogkalu_speech_authority_overlaps_protected",
            authorization_explicit=False,
            reason_codes=adjudication_reason_codes,
        )

    if recognized_art_or_non_text:
        adjudication_reason_codes.append("text_area_plan:recognized_art_non_text_typed_evidence")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_PROTECT_ART_OR_NON_TEXT,
            semantic_authorization_state=AUTH_PROTECT_ART_OR_NON_TEXT,
            semantic_kind=SEMANTIC_KIND_ART_OR_NON_TEXT,
            must_not_mutate=True,
            protection_reason="recognized_art_non_text_typed_evidence",
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if _has_art_or_non_text_authority(marker, role_evidence):
        adjudication_reason_codes.append("text_area_plan:semantic_art_or_non_text_authority")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_PROTECT_ART_OR_NON_TEXT,
            semantic_authorization_state=AUTH_PROTECT_ART_OR_NON_TEXT,
            semantic_kind=SEMANTIC_KIND_ART_OR_NON_TEXT,
            must_not_mutate=True,
            protection_reason="explicit_art_or_non_text_authorization",
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if _has_sfx_or_decorative_authority(route, ctype, marker, role_evidence):
        adjudication_reason_codes.append("text_area_plan:semantic_sfx_decorative_authority")
        kind = "decorative" if "decorative" in marker else "sfx"
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_PROTECT_SFX_DECORATIVE,
            semantic_authorization_state=AUTH_PROTECT_SFX_DECORATIVE,
            semantic_kind=kind,
            must_not_mutate=True,
            protection_reason="explicit_sfx_decorative_authorization",
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if recognized_sfx_decorative:
        adjudication_reason_codes.append("text_area_plan:recognized_sfx_decorative_typed_evidence")
        kind = SEMANTIC_KIND_DECORATIVE if "decorative" in marker else SEMANTIC_KIND_SFX
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_PROTECT_SFX_DECORATIVE,
            semantic_authorization_state=AUTH_PROTECT_SFX_DECORATIVE,
            semantic_kind=kind,
            must_not_mutate=True,
            protection_reason="recognized_sfx_decorative_typed_evidence",
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if _has_unresolved_ogkalu_speech_risk(marker, role_evidence):
        adjudication_reason_codes.append(f"text_area_plan:{UNRESOLVED_OGKALU_SPEECH_RISK_REASON}")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_AMBIGUOUS_COMPONENT_OWNER,
            semantic_authorization_state=AUTH_AMBIGUOUS_COMPONENT_OWNER,
            semantic_kind=SEMANTIC_KIND_SPEECH,
            must_not_mutate=False,
            protection_reason=UNRESOLVED_OGKALU_SPEECH_RISK_REASON,
            authorization_explicit=False,
            reason_codes=adjudication_reason_codes,
        )

    if _has_speech_authority(route, ctype, marker, role_evidence):
        adjudication_reason_codes.append("text_area_plan:semantic_speech_authority")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_CLEANUP_TRANSLATE_SPEECH,
            semantic_authorization_state=AUTH_CLEANUP_TRANSLATE_SPEECH,
            semantic_kind=SEMANTIC_KIND_SPEECH,
            must_not_mutate=False,
            authorization_explicit=True,
            reason_codes=adjudication_reason_codes,
        )

    if route == ROUTE_TRANSLATE_CAPTION or ctype == CONTAINER_CAPTION:
        review_reason = _caption_background_review_reason(marker, role_evidence)
        authority_reason = _caption_background_authority_reason(marker, role_evidence)
        if review_reason and not authority_reason:
            adjudication_reason_codes.append(f"text_area_plan:{review_reason}")
            return TextAreaSemanticAdjudication(
                cleanup_authorization=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                semantic_authorization_state=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                semantic_kind="background_narration" if ctype == CONTAINER_CAPTION else "unknown",
                must_not_mutate=True,
                protection_reason=review_reason,
                authorization_explicit=False,
                reason_codes=adjudication_reason_codes,
            )
        if authority_reason:
            adjudication_reason_codes.append(f"text_area_plan:{authority_reason}")
            cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
            if AUTH_CLEANUP_TRANSLATE_CAPTION in cleanup_states and AUTH_CLEANUP_TRANSLATE_BACKGROUND not in cleanup_states:
                auth = AUTH_CLEANUP_TRANSLATE_CAPTION
                kind = SEMANTIC_KIND_CAPTION
            else:
                auth = AUTH_CLEANUP_TRANSLATE_BACKGROUND
                kind = SEMANTIC_KIND_BACKGROUND_NARRATION
            return TextAreaSemanticAdjudication(
                cleanup_authorization=auth,
                semantic_authorization_state=auth,
                semantic_kind=kind,
                must_not_mutate=False,
                authorization_explicit=True,
                reason_codes=adjudication_reason_codes,
            )
        adjudication_reason_codes.append("text_area_plan:caption_background_requires_explicit_authority")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_authorization_state=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_kind="background_narration" if ctype == CONTAINER_CAPTION else "unknown",
            must_not_mutate=True,
            protection_reason="caption_background_requires_explicit_authority",
            authorization_explicit=False,
            reason_codes=adjudication_reason_codes,
        )

    if route == ROUTE_REVIEW_FALLBACK or ctype == CONTAINER_UNKNOWN:
        adjudication_reason_codes.append("text_area_plan:review_unknown_not_cleanup")
        return TextAreaSemanticAdjudication(
            cleanup_authorization=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_authorization_state=AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
            semantic_kind="unknown",
            must_not_mutate=True,
            protection_reason="review_unknown_not_cleanup",
            authorization_explicit=False,
            reason_codes=adjudication_reason_codes,
        )

    adjudication_reason_codes.append("text_area_plan:outside_cleanup_scope")
    return TextAreaSemanticAdjudication(
        cleanup_authorization=AUTH_OUTSIDE_CLEANUP_SCOPE,
        semantic_authorization_state=AUTH_OUTSIDE_CLEANUP_SCOPE,
        semantic_kind="unknown",
        must_not_mutate=True,
        protection_reason="outside_cleanup_scope",
        authorization_explicit=False,
        reason_codes=adjudication_reason_codes,
    )


def _cleanup_authorization_for_container(container: Mapping[str, Any] | TextAreaContainer) -> tuple[str, str, bool]:
    adjudication = _adjudicate_text_area_semantic_authorization(container)
    return adjudication.cleanup_authorization, adjudication.protection_reason, adjudication.must_not_mutate


def _apply_cleanup_authorization(container: TextAreaContainer) -> None:
    adjudication = _adjudicate_text_area_semantic_authorization(container)
    container.semantic_role_evidence = _semantic_role_evidence_with_provider_records(
        container.semantic_role_evidence,
        page_id=str(container.page_id or ""),
        bbox=container.bbox,
        container_id=str(container.container_id or ""),
        reason_codes=list(container.evidence_reason_codes or []) + list(adjudication.reason_codes or []),
        basis=adjudication.authorization_basis or "",
        confidence_tier=str(container.confidence_tier or ""),
        requires_review=bool(container.human_review_required),
    )
    container.cleanup_authorization = adjudication.cleanup_authorization
    container.must_not_mutate = adjudication.must_not_mutate
    container.protection_reason = adjudication.protection_reason
    container.semantic_kind = adjudication.semantic_kind
    container.pre_ocr_authority = bool(container.text_area_pre_ocr_authority)
    container.source_stage = "text_area_plan_pre_ocr" if container.text_area_pre_ocr_authority else "text_area_plan_region_enriched"
    container.authorization_source_stage = container.source_stage
    for reason_code in adjudication.reason_codes:
        if reason_code and reason_code not in container.evidence_reason_codes:
            container.evidence_reason_codes.append(reason_code)
    container.authorization_basis = _authorization_basis_for_container(
        container,
        adjudication.cleanup_authorization,
        adjudication.protection_reason,
    )
    container.authorization_explicit = bool(adjudication.authorization_explicit)
    container.authorization_field_origin = adjudication.authorization_field_origin
    container.semantic_authorization_state = adjudication.semantic_authorization_state
    _align_container_route_to_cleanup_authority(container)
    container.parent_source_evidence = {
        "source_model_ids": list(container.source_model_ids or []),
        "evidence_reason_codes": list(container.evidence_reason_codes or []),
        "conflict_flags": list(container.conflict_flags or []),
        "semantic_role_evidence": dict(container.semantic_role_evidence or {}),
    }


def _align_container_route_to_cleanup_authority(container: TextAreaContainer) -> None:
    """Keep semantic cleanup authority and downstream route eligibility coherent.

    Semantic adjudication is the upstream authority. Once it explicitly assigns
    a cleanup-translate state and protected/art conflicts have already been
    resolved, the container must not remain review-only with scoped OCR disabled;
    otherwise the cleanup mask can be correct in isolation while the production
    workflow never creates the corresponding cleanup/translation obligation.
    """

    if not bool(container.authorization_explicit):
        return
    auth = str(container.semantic_authorization_state or container.cleanup_authorization or "")
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH:
        target_route = ROUTE_TRANSLATE_SPEECH
        target_type = CONTAINER_SPEECH
        target_reason = "semantic_cleanup_speech_authority_route_alignment"
        ocr_reason = "semantic_cleanup_speech_authority"
    elif auth in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        target_route = ROUTE_TRANSLATE_CAPTION
        target_type = CONTAINER_CAPTION
        target_reason = "semantic_cleanup_background_authority_route_alignment"
        ocr_reason = "semantic_cleanup_background_authority"
    else:
        return

    if container.route_intent != target_route:
        container.route_intent = target_route
    if container.container_type == CONTAINER_UNKNOWN:
        container.container_type = target_type
    container.ocr_eligible = True
    container.comic_text_detector_scope_eligible = True
    if target_reason not in container.evidence_reason_codes:
        container.evidence_reason_codes.append(f"text_area_plan:{target_reason}")
    container.ocr_eligibility_reason = ocr_reason


def _authorization_is_explicit(auth: str) -> bool:
    return auth in {
        AUTH_CLEANUP_TRANSLATE_SPEECH,
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_PROTECT_SFX_DECORATIVE,
        AUTH_PROTECT_ART_OR_NON_TEXT,
    }


def _authorization_is_cleanup_executable(auth: str) -> bool:
    return auth in {
        AUTH_CLEANUP_TRANSLATE_SPEECH,
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
    }


def _container_authorization_eligibility(container: Mapping[str, Any] | TextAreaContainer) -> Dict[str, bool]:
    auth = str(
        _container_value(container, "semantic_authorization_state", "")
        or _container_value(container, "cleanup_authorization", "")
        or ""
    )
    explicit = bool(_container_value(container, "authorization_explicit", False))
    ocr_eligible = bool(_container_value(container, "ocr_eligible", False))
    ctd_scope_eligible = bool(_container_value(container, "comic_text_detector_scope_eligible", False))
    cleanup_executable = bool(explicit and _authorization_is_cleanup_executable(auth))
    translation_eligible = bool(cleanup_executable and ocr_eligible)
    render_eligible = translation_eligible
    return {
        "ctd_scope_eligible": ctd_scope_eligible,
        "ocr_eligible": ocr_eligible,
        "translation_eligible": translation_eligible,
        "render_eligible": render_eligible,
        "cleanup_executable": cleanup_executable,
    }


def _authorization_basis_for_container(container: TextAreaContainer, auth: str, reason: str) -> str:
    basis = [
        f"auth={auth or AUTH_REVIEW_UNKNOWN_NOT_CLEANUP}",
        f"route_intent={container.route_intent or ROUTE_REVIEW_FALLBACK}",
        f"container_type={container.container_type or CONTAINER_UNKNOWN}",
    ]
    if reason:
        basis.append(f"reason={reason}")
    if container.evidence_reason_codes:
        basis.append("evidence_reason_codes=" + ",".join(str(item) for item in container.evidence_reason_codes if str(item)))
    if container.conflict_flags:
        basis.append("conflict_flags=" + ",".join(str(item) for item in container.conflict_flags if str(item)))
    if container.semantic_role_evidence:
        role_values = []
        for key in ("role_signals", "ogkalu_class_names", "current_region_roles"):
            role_values.extend(_semantic_role_values(container.semantic_role_evidence, key))
        if role_values:
            basis.append("semantic_role_evidence=" + ",".join(str(item) for item in role_values if str(item)))
    return ";".join(item for item in basis if item)


def _copy_container_authorization_to_scope(scope: TextAreaScope, container: TextAreaContainer) -> None:
    scope.cleanup_authorization = container.cleanup_authorization
    scope.must_not_mutate = container.must_not_mutate
    scope.protection_reason = container.protection_reason
    scope.pre_ocr_authority = container.pre_ocr_authority
    scope.source_stage = container.source_stage
    scope.authorization_source_stage = container.authorization_source_stage
    scope.authorization_basis = container.authorization_basis
    scope.authorization_explicit = container.authorization_explicit
    scope.authorization_field_origin = container.authorization_field_origin
    scope.semantic_authorization_state = container.semantic_authorization_state


def _semantic_kind_for_container(container: TextAreaContainer) -> str:
    if str(container.semantic_kind or ""):
        return str(container.semantic_kind)
    auth = str(container.cleanup_authorization or container.semantic_authorization_state or "")
    ctype = str(container.container_type or CONTAINER_UNKNOWN)
    marker = " ".join(
        [
            str(container.route_intent or ""),
            ctype,
            str(container.confidence_tier or ""),
            str(container.fallback_reason or ""),
            " ".join(str(item) for item in container.evidence_reason_codes or []),
            " ".join(str(item) for item in container.conflict_flags or []),
        ]
    ).lower()
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH or ctype == CONTAINER_SPEECH:
        return "speech"
    if auth == AUTH_CLEANUP_TRANSLATE_CAPTION:
        return "caption"
    if auth == AUTH_CLEANUP_TRANSLATE_BACKGROUND or ctype == CONTAINER_CAPTION:
        return "background_narration"
    if auth == AUTH_PROTECT_ART_OR_NON_TEXT:
        return "art_or_non_text"
    if auth == AUTH_PROTECT_SFX_DECORATIVE or ctype == CONTAINER_SFX:
        if "art" in marker or "non_text" in marker or "non-text" in marker:
            return "art_or_non_text"
        if "decorative" in marker:
            return "decorative"
        return "sfx"
    return "unknown"


def _semantic_kind_for_container_dict(container: Mapping[str, Any]) -> str:
    if str(container.get("semantic_kind") or ""):
        return str(container.get("semantic_kind"))
    auth = str(container.get("cleanup_authorization") or container.get("semantic_authorization_state") or "")
    ctype = str(container.get("container_type") or CONTAINER_UNKNOWN)
    route = str(container.get("route_intent") or "")
    marker = " ".join(
        [
            route,
            ctype,
            str(container.get("confidence_tier") or ""),
            str(container.get("fallback_reason") or ""),
            " ".join(str(item) for item in container.get("evidence_reason_codes") or []),
            " ".join(str(item) for item in container.get("conflict_flags") or []),
        ]
    ).lower()
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH or ctype == CONTAINER_SPEECH:
        return "speech"
    if auth == AUTH_CLEANUP_TRANSLATE_CAPTION:
        return "caption"
    if auth == AUTH_CLEANUP_TRANSLATE_BACKGROUND or ctype == CONTAINER_CAPTION:
        return "background_narration"
    if auth == AUTH_PROTECT_ART_OR_NON_TEXT:
        return "art_or_non_text"
    if auth == AUTH_PROTECT_SFX_DECORATIVE or ctype == CONTAINER_SFX:
        if "art" in marker or "non_text" in marker or "non-text" in marker:
            return "art_or_non_text"
        if "decorative" in marker:
            return "decorative"
        return "sfx"
    return "unknown"


def _semantic_unit_bbox_from_evidence(entry: Mapping[str, Any]) -> List[int]:
    values = entry.get("bbox")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) < 4:
        return []
    try:
        x0 = float(values[0])
        y0 = float(values[1])
        x1 = float(values[2])
        y1 = float(values[3])
    except Exception:
        return []
    if x1 <= x0 or y1 <= y0:
        return []
    return [
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        max(1, int(round(x1 - x0))),
        max(1, int(round(y1 - y0))),
    ]


def _semantic_unit_projection_bbox_for_container(
    container: TextAreaContainer,
    entry: Mapping[str, Any],
    bbox: Sequence[int],
) -> List[int]:
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return []
    class_name = str(entry.get("class_name") or "")
    auth = str(container.cleanup_authorization or container.semantic_authorization_state or "")
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH and class_name == "text_bubble" and w >= 24 and h >= 24:
        pad_x = min(90, max(0, int(round(w * 0.38))))
        pad_y = min(40, max(0, int(round(h * 0.18))))
    elif auth in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION} and class_name == "text_free" and w >= 24 and h >= 24:
        pad_x = min(32, max(0, int(round(w * 0.12))))
        pad_y = min(32, max(0, int(round(h * 0.12))))
    else:
        pad_x = 0
        pad_y = 0
    if pad_x <= 0 and pad_y <= 0:
        return [x, y, w, h]
    cx, cy, cw, ch = _coerce_xywh(container.bbox)
    nx0 = x - pad_x
    ny0 = y - pad_y
    nx1 = x + w + pad_x
    ny1 = y + h + pad_y
    if cw > 0 and ch > 0:
        nx0 = max(cx, nx0)
        ny0 = max(cy, ny0)
        nx1 = min(cx + cw, nx1)
        ny1 = min(cy + ch, ny1)
    if nx1 <= nx0 or ny1 <= ny0:
        return [x, y, w, h]
    return [int(nx0), int(ny0), int(nx1 - nx0), int(ny1 - ny0)]


def _semantic_unit_evidence_bboxes_for_container(container: TextAreaContainer) -> List[Dict[str, Any]]:
    role_evidence = dict(container.semantic_role_evidence or {})
    entries = role_evidence.get("text_unit_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    auth = str(container.cleanup_authorization or container.semantic_authorization_state or "")
    if auth == AUTH_CLEANUP_TRANSLATE_SPEECH:
        preferred_classes = {"text_bubble"}
    elif auth in {AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        preferred_classes = {"text_free", "text_bubble"}
    elif auth == AUTH_PROTECT_SFX_DECORATIVE:
        preferred_classes = {"sfx", "decorative", "sfx_or_decorative", "sfx_or_decorative_candidate", "text_free", "text_bubble"}
    elif auth == AUTH_PROTECT_ART_OR_NON_TEXT:
        preferred_classes = {"art", "non_text", "non-text", "text_free"}
    else:
        preferred_classes = set()
    if not preferred_classes:
        return []
    bboxes: List[Dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        class_name = str(entry.get("class_name") or "")
        if class_name not in preferred_classes:
            continue
        bbox = _semantic_unit_bbox_from_evidence(entry)
        if not bbox:
            continue
        overlap_ratio = _inside_ratio_xywh(bbox, container.bbox)
        if overlap_ratio < 0.25:
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        bboxes.append(
            {
                "bbox": bbox,
                "evidence_id": str(entry.get("evidence_id") or ""),
                "class_name": class_name,
                "container_overlap_ratio": round(overlap_ratio, 6),
                "is_explicit_parent_boundary_evidence": _text_area_graph_provider_authored_parent_boundary(
                    container,
                    entry,
                ),
            }
        )
    return bboxes


def _linked_text_free_speech_parent_boundary_entries(
    container: TextAreaContainer | Mapping[str, Any],
) -> List[Dict[str, Any]]:
    auth = str(
        _container_value(container, "cleanup_authorization", "")
        or _container_value(container, "semantic_authorization_state", "")
        or ""
    )
    if auth != AUTH_CLEANUP_TRANSLATE_SPEECH:
        return []
    role_evidence = _container_semantic_role_evidence(container)
    entries = role_evidence.get("text_unit_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    output: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or str(entry.get("class_name") or "") != "text_free":
            continue
        if not _text_free_parent_boundary_has_typed_speech_mask_support(entry, role_evidence):
            continue
        bbox = _semantic_unit_bbox_from_evidence(entry)
        if not bbox:
            continue
        output.append(
            {
                "bbox": bbox,
                "evidence_id": str(entry.get("evidence_id") or ""),
                "class_name": "text_free",
                "container_overlap_ratio": round(
                    _inside_ratio_xywh(bbox, _text_area_graph_container_bbox(container)),
                    6,
                ),
                "boundary_source": TEXT_AREA_GRAPH_PARENT_SOURCE_TEXT_UNIT,
                "is_explicit_parent_boundary_evidence": True,
            }
        )
    return output


def _text_free_parent_boundary_has_typed_speech_mask_support(
    entry: Mapping[str, Any],
    role_evidence: Mapping[str, Any],
) -> bool:
    entry_bbox = _semantic_unit_bbox_from_evidence(entry)
    if not entry_bbox:
        return False
    mask_entries = role_evidence.get("speech_mask_polygons") or []
    if not isinstance(mask_entries, Sequence) or isinstance(mask_entries, (str, bytes)):
        return False
    for mask_entry in mask_entries:
        if not isinstance(mask_entry, Mapping):
            continue
        mask_bbox = _semantic_unit_bbox_from_evidence(mask_entry)
        if not mask_bbox:
            continue
        if _inside_ratio_xywh(entry_bbox, mask_bbox) >= 0.50:
            return True
    return False


def _semantic_unit_from_container(
    container: TextAreaContainer,
    *,
    semantic_unit_id: str | None = None,
    bbox: Sequence[int] | None = None,
    polygon: Sequence[Sequence[float]] | None = None,
    source_evidence_ids: Sequence[str] | None = None,
    extra_reason_codes: Sequence[str] | None = None,
) -> TextAreaSemanticAuthorizationRecord:
    semantic_unit_id = str(semantic_unit_id or container.container_id or "")
    auth = str(container.cleanup_authorization or container.semantic_authorization_state or AUTH_REVIEW_UNKNOWN_NOT_CLEANUP)
    source_ids = [str(item) for item in (source_evidence_ids if source_evidence_ids is not None else container.source_model_ids or []) if str(item)]
    reason_codes = [str(item) for item in container.evidence_reason_codes or [] if str(item)]
    for reason in extra_reason_codes or []:
        if str(reason) and str(reason) not in reason_codes:
            reason_codes.append(str(reason))
    role_evidence = _semantic_role_evidence_with_provider_records(
        container.semantic_role_evidence or {},
        page_id=str(container.page_id or ""),
        bbox=list(bbox or container.bbox or []),
        container_id=str(container.container_id or ""),
        reason_codes=reason_codes,
        basis=str(container.authorization_basis or ""),
        confidence_tier=str(container.confidence_tier or "low"),
        requires_review=bool(container.human_review_required),
    )
    semantic_records = _semantic_evidence_records(role_evidence)
    eligibility = _container_authorization_eligibility(container)
    evidence_sources = []
    for key in ("source", "authority_evidence_kind"):
        value = str(role_evidence.get(key) or "")
        if value and value not in evidence_sources:
            evidence_sources.append(value)
    for record in semantic_records:
        provider = str(record.get("provider") or "")
        if provider and provider not in evidence_sources:
            evidence_sources.append(provider)
    semantic_evidence_ids = sorted({str(record.get("evidence_id") or "") for record in semantic_records if str(record.get("evidence_id") or "")})
    semantic_evidence_providers = sorted({str(record.get("provider") or "") for record in semantic_records if str(record.get("provider") or "")})
    return TextAreaSemanticAuthorizationRecord(
        semantic_unit_id=semantic_unit_id,
        page_id=str(container.page_id or ""),
        bbox=list(bbox or container.bbox or []),
        polygon=[list(point) for point in polygon or []],
        semantic_kind=_semantic_kind_for_container(container),
        cleanup_authorization=auth,
        authorization_source_stage=str(container.authorization_source_stage or container.source_stage or "text_area_plan"),
        authorization_basis=str(container.authorization_basis or ""),
        authorization_explicit=bool(container.authorization_explicit),
        authorization_field_origin=str(container.authorization_field_origin or ""),
        source_evidence_ids=source_ids,
        source_model_ids=source_ids,
        evidence_source_list=evidence_sources,
        confidence_tier=str(container.confidence_tier or "low"),
        reason_codes=reason_codes,
        evidence_reason_codes=reason_codes,
        conflict_flags=[str(item) for item in container.conflict_flags or [] if str(item)],
        semantic_role_evidence=role_evidence,
        semantic_evidence_providers=semantic_evidence_providers,
        semantic_evidence_ids=semantic_evidence_ids,
        semantic_evidence_trace=semantic_records,
        must_not_mutate=bool(container.must_not_mutate),
        review_required=bool(container.human_review_required or _component_auth_family(auth) in {"review", "outside", "ambiguous"}),
        ocr_eligible=eligibility["ocr_eligible"],
        comic_text_detector_scope_eligible=eligibility["ctd_scope_eligible"],
        translation_eligible=eligibility["translation_eligible"],
        render_eligible=eligibility["render_eligible"],
        cleanup_executable=eligibility["cleanup_executable"],
        semantic_authorization_state=str(container.semantic_authorization_state or auth),
        container_id=str(container.container_id or ""),
        route_intent=str(container.route_intent or ROUTE_REVIEW_FALLBACK),
        container_type=str(container.container_type or CONTAINER_UNKNOWN),
        protection_reason=str(container.protection_reason or ""),
    )


def _semantic_unit_mask_polygons_for_container(container: TextAreaContainer) -> List[Dict[str, Any]]:
    auth = str(container.cleanup_authorization or container.semantic_authorization_state or "")
    if auth != AUTH_CLEANUP_TRANSLATE_SPEECH:
        return []
    role_evidence = dict(container.semantic_role_evidence or {})
    entries = role_evidence.get("speech_mask_polygons") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    units: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        polygon = entry.get("polygon") or []
        if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)) or len(polygon) < 3:
            continue
        evidence_id = str(entry.get("evidence_id") or "")
        key = evidence_id or repr(polygon)
        if key in seen:
            continue
        seen.add(key)
        bbox = _semantic_unit_bbox_from_evidence(entry) or list(container.bbox or [])
        units.append({"evidence_id": evidence_id, "bbox": bbox, "polygon": polygon})
    return units


def _semantic_units_from_container(container: TextAreaContainer) -> List[TextAreaSemanticAuthorizationRecord]:
    mask_polygons = _semantic_unit_mask_polygons_for_container(container)
    if mask_polygons:
        units: List[TextAreaSemanticAuthorizationRecord] = []
        for index, entry in enumerate(mask_polygons):
            evidence_id = str(entry.get("evidence_id") or "")
            suffix = evidence_id or f"mask_{index:02d}"
            unit_id = f"{container.container_id}__{suffix}" if container.container_id else suffix
            units.append(
                _semantic_unit_from_container(
                    container,
                    semantic_unit_id=unit_id,
                    bbox=entry.get("bbox") or container.bbox,
                    polygon=entry.get("polygon") or [],
                    source_evidence_ids=[evidence_id] if evidence_id else container.source_model_ids,
                    extra_reason_codes=["text_area_plan:speech_mask_polygon_semantic_unit"],
                )
            )
        return units
    evidence_bboxes = _semantic_unit_evidence_bboxes_for_container(container)
    if not evidence_bboxes:
        return [_semantic_unit_from_container(container)]
    units: List[TextAreaSemanticAuthorizationRecord] = []
    for index, entry in enumerate(evidence_bboxes):
        evidence_id = str(entry.get("evidence_id") or "")
        suffix = evidence_id or f"text_{index:02d}"
        unit_id = f"{container.container_id}__{suffix}" if container.container_id else suffix
        reason = f"text_area_plan:text_evidence_semantic_unit_bbox:{entry.get('class_name') or 'unknown'}"
        units.append(
            _semantic_unit_from_container(
                container,
                semantic_unit_id=unit_id,
                bbox=entry.get("bbox") or container.bbox,
                source_evidence_ids=[evidence_id] if evidence_id else container.source_model_ids,
                extra_reason_codes=[reason],
            )
        )
    return units


def _finish_plan(
    plan: TextAreaPlan,
    started: float,
    *,
    finalize_graph: bool = True,
) -> TextAreaPlan:
    plan.runtime.generated = bool(plan.generated)
    plan.runtime.runtime_sec = round(time.perf_counter() - started, 6)
    by_type: Dict[str, int] = {}
    by_intent: Dict[str, int] = {}
    blocked_by_type: Dict[str, int] = {}
    ocr_eligible = 0
    scope_eligible = 0
    review_only_blocked = 0
    _demote_ogkalu_speech_authority_overlapping_protected(plan.containers, plan.image_size)
    for container in plan.containers:
        _apply_cleanup_authorization(container)
    _sync_scopes_to_authorized_containers(plan)
    semantic_units: List[TextAreaSemanticAuthorizationRecord] = []
    for container in plan.containers:
        semantic_units.extend(_semantic_units_from_container(container))
    plan.semantic_units = semantic_units
    containers_by_id = {str(container.container_id or ""): container for container in plan.containers if str(container.container_id or "")}
    for scope in plan.scopes:
        container = containers_by_id.get(str(scope.container_id or ""))
        if container:
            _copy_container_authorization_to_scope(scope, container)
    if finalize_graph and plan.root_parent_child_plan is None:
        plan.root_parent_child_plan = build_text_area_root_parent_child_plan(plan)
    for container in plan.containers:
        by_type[container.container_type] = by_type.get(container.container_type, 0) + 1
        by_intent[container.route_intent] = by_intent.get(container.route_intent, 0) + 1
        if container.ocr_eligible:
            ocr_eligible += 1
        if container.comic_text_detector_scope_eligible:
            scope_eligible += 1
        if not container.ocr_eligible or not container.comic_text_detector_scope_eligible:
            blocked_by_type[container.container_type] = blocked_by_type.get(container.container_type, 0) + 1
            if container.container_type == CONTAINER_UNKNOWN:
                review_only_blocked += 1
    plan.summary = {
        "container_count": len(plan.containers),
        "scope_count": len(plan.scopes),
        "by_container_type": by_type,
        "by_route_intent": by_intent,
        "ocr_eligible_containers": ocr_eligible,
        "ctd_scope_eligible_containers": scope_eligible,
        "blocked_containers_by_type": blocked_by_type,
        "blocked_sfx_decorative_containers": blocked_by_type.get(CONTAINER_SFX, 0),
        "blocked_review_only_unknown_containers": review_only_blocked,
        "fallback_count": len(plan.fallback_reasons),
        "compatibility_mode": plan.runtime.compatibility_mode,
    }
    _update_text_area_graph_summary(plan)
    return plan


def _update_text_area_graph_summary(plan: TextAreaPlan) -> None:
    if plan.root_parent_child_plan is None:
        plan.summary.pop("root_parent_child_plan", None)
        return
    graph_plan = _root_parent_child_plan_to_dict(plan.root_parent_child_plan)
    diagnostics = graph_plan.get("diagnostics") if isinstance(graph_plan.get("diagnostics"), Mapping) else {}
    plan.summary["root_parent_child_plan"] = {
        "root_node_count": len(graph_plan.get("root_nodes") or []),
        "parent_boundary_candidate_count": len(graph_plan.get("parent_boundary_candidates") or []),
        "parent_node_count": len(graph_plan.get("parent_nodes") or []),
        "child_evidence_slot_count": len(graph_plan.get("child_evidence_slots") or []),
        "excluded_inventory_count": len(graph_plan.get("excluded_inventory") or []),
        "graph_blocker_count": len(graph_plan.get("graph_blockers") or []),
        "source_evidence_payload_count": len(graph_plan.get("source_evidence_payloads") or []),
        "parent_boundary_candidate_state_counts": dict(
            diagnostics.get("parent_boundary_candidate_state_counts") or {}
        ),
        "blocker_counts": dict(diagnostics.get("blocker_counts") or {}),
    }


def _sync_scopes_to_authorized_containers(plan: TextAreaPlan) -> None:
    """Keep scoped OCR/CTD inputs aligned with final container authority."""

    scopes_by_container_id = {
        str(scope.container_id or ""): scope
        for scope in plan.scopes
        if str(scope.container_id or "")
    }
    for container in plan.containers:
        container_id = str(container.container_id or "")
        if not container_id or not bool(container.comic_text_detector_scope_eligible):
            continue
        if container_id in scopes_by_container_id:
            continue
        scope = TextAreaScope(
            scope_id=f"scope_{container.container_id}",
            page_id=str(plan.page_id),
            container_id=container.container_id,
            bbox=list(container.bbox),
            container_type=container.container_type,
            route_intent=container.route_intent,
            ocr_eligible=container.ocr_eligible,
            comic_text_detector_scope_eligible=container.comic_text_detector_scope_eligible,
            fallback_reason=container.fallback_reason,
            ocr_eligibility_reason=container.ocr_eligibility_reason,
            text_area_pre_ocr_authority=container.text_area_pre_ocr_authority,
            text_area_enriched_from_region=container.text_area_enriched_from_region,
        )
        _copy_container_authorization_to_scope(scope, container)
        plan.scopes.append(scope)
        scopes_by_container_id[container_id] = scope


def _typed_multicolumn_unlinked_text_free_background_authority(
    fused: Mapping[str, Any],
    semantic_role_evidence: Mapping[str, Any],
    *,
    bbox: Sequence[int],
    image_size: Tuple[int, int],
    luma_image: Any,
    clipped: bool,
) -> bool:
    if str(fused.get("fused_container_type") or "") != "free_text":
        return False
    if clipped or fused.get("conflict_flags"):
        return False
    if fused.get("linked_kitsumed_mask_ids") or not fused.get("linked_ogkalu_detection_ids"):
        return False
    confidence = _semantic_role_model_confidence(semantic_role_evidence)
    if confidence is None or confidence < OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE:
        return False
    if SEMANTIC_KIND_BACKGROUND_NARRATION not in set(
        _semantic_role_values(semantic_role_evidence, "candidate_roles")
    ):
        return False
    if AUTH_CLEANUP_TRANSLATE_BACKGROUND not in set(
        _semantic_role_values(semantic_role_evidence, "cleanup_candidate_states")
    ):
        return False
    if "no_speech_bubble_mask_link" not in set(
        _semantic_role_values(semantic_role_evidence, "role_signals")
    ):
        return False
    protected_states = set(
        _semantic_role_values(semantic_role_evidence, "protected_authority_states")
    )
    protected_states.update(
        _semantic_role_values(semantic_role_evidence, "protected_candidate_states")
    )
    for record in _semantic_evidence_records(semantic_role_evidence):
        state = str(record.get("authority_state") or "")
        if _component_auth_family(state) in {"protected", "outside"}:
            protected_states.add(state)
    if protected_states:
        return False
    text_entries = semantic_role_evidence.get("text_unit_evidence_bboxes") or []
    if not isinstance(text_entries, Sequence) or isinstance(text_entries, (str, bytes)):
        return False
    if not any(
        isinstance(entry, Mapping)
        and str(entry.get("class_name") or "") == "text_free"
        and bool(_semantic_unit_bbox_from_evidence(entry))
        for entry in text_entries
    ):
        return False
    return _luma_bbox_has_multicolumn_vertical_text_components(
        luma_image,
        bbox,
        image_size,
    )


def _container_from_fused(
    page_id: str,
    fused: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
) -> TextAreaContainer | None:
    container_id = str(fused.get("fused_container_id") or "")
    if not container_id:
        return None
    fused_type = str(fused.get("fused_container_type") or "")
    if fused_type == "text_evidence_only":
        return None
    confidence = str(fused.get("confidence") or "low")
    reasons = list(fused.get("reason_codes") or [])
    conflicts = list(fused.get("conflict_flags") or [])
    bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
    source_ids = list(fused.get("linked_kitsumed_mask_ids") or []) + list(fused.get("linked_ogkalu_detection_ids") or [])
    confidence_tier = _confidence_tier_from_fused(fused)
    semantic_role_evidence = _semantic_role_evidence_from_fused(fused)
    visual = _container_visual_stats(luma_image, bbox, image_size)
    clipped = _is_clipped_or_degenerate_bbox(bbox, image_size)

    if conflicts:
        ctype = CONTAINER_SFX if _has_preserve_conflict(conflicts, reasons) else CONTAINER_UNKNOWN
        route = ROUTE_PRESERVE_SFX if ctype == CONTAINER_SFX else ROUTE_REVIEW_FALLBACK
        if ctype == CONTAINER_SFX:
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "protected_authority_states",
                AUTH_PROTECT_SFX_DECORATIVE,
                evidence_kind="typed_current_conflict_sfx_decorative_authority",
            )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=ctype,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="conflict_preserve_wins" if ctype == CONTAINER_SFX else confidence_tier,
            route_intent=route,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="model_conflict_review_required",
            evidence_reason_codes=reasons + ["text_area_plan:conflict_fail_closed"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_conflict_fail_closed",
        )

    if fused_type == "speech_bubble":
        has_kitsumed = bool(fused.get("linked_kitsumed_mask_ids"))
        has_ogkalu = bool(fused.get("linked_ogkalu_detection_ids"))
        large_mask_primary = has_kitsumed and _bbox_area_xywh(bbox) >= 45000
        speech_authority_evidence = semantic_role_evidence
        if has_kitsumed:
            speech_authority_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "cleanup_authority_states",
                AUTH_CLEANUP_TRANSLATE_SPEECH,
                evidence_kind="typed_speech_bubble_mask_authority",
            )
        if (confidence == "high" and has_kitsumed) or (confidence == "medium" and large_mask_primary):
            extra_reasons = ["text_area_plan:medium_mask_primary_large_container"] if confidence == "medium" else []
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                semantic_role_evidence=speech_authority_evidence,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons + extra_reasons + ["text_area_plan:speech_mask_container"],
                ocr_eligibility_reason="speech_container",
            )
        if has_kitsumed and not has_ogkalu and _mask_primary_speech_activation_completeness_allowed(
            confidence=confidence,
            bbox=bbox,
            reasons=reasons,
            visual=visual,
            image_size=image_size,
            clipped=clipped,
        ):
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                semantic_role_evidence=speech_authority_evidence,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons
                + [
                    "text_area_plan:mask_primary_activation_completeness",
                    "text_area_plan:scope_ocr_required_for_visible_speech_root",
                    "text_area_plan:speech_mask_container",
                ],
                ocr_eligibility_reason="speech_activation_completeness_scope_required",
                human_review_required=True,
            )
        if has_kitsumed and not has_ogkalu and _mask_primary_irregular_speech_activation_allowed(
            confidence=confidence,
            bbox=bbox,
            reasons=reasons,
            visual=visual,
            image_size=image_size,
            clipped=clipped,
        ):
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                semantic_role_evidence=speech_authority_evidence,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons
                + [
                    "text_area_plan:mask_primary_irregular_speech_activation",
                    "text_area_plan:scope_ocr_required_for_irregular_speech",
                    "text_area_plan:speech_mask_container",
                ],
                ocr_eligibility_reason="irregular_speech_mask_primary_scope_required",
                human_review_required=True,
            )
        review_reason = "mask_primary_container_requires_review" if has_kitsumed else "low_confidence_speech_mask"
        if has_kitsumed and not has_ogkalu:
            reasons = reasons + ["text_area_plan:mask_primary_without_support_review_only"]
            semantic_role_evidence = _semantic_role_evidence_without_cleanup_authority(
                semantic_role_evidence,
                AUTH_CLEANUP_TRANSLATE_SPEECH,
            )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_UNKNOWN,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier=confidence_tier,
            route_intent=ROUTE_REVIEW_FALLBACK,
            ocr_eligible=not (has_kitsumed and not has_ogkalu),
            comic_text_detector_scope_eligible=not (has_kitsumed and not has_ogkalu),
            fallback_reason=review_reason,
            evidence_reason_codes=reasons + ["text_area_plan:low_confidence_fallback"],
            human_review_required=True,
            ocr_eligibility_reason="blocked_mask_primary_review_only" if has_kitsumed and not has_ogkalu else "speech_mask_review_fallback",
        )

    caption_background_candidate_bbox = _looks_like_caption_background_bbox(
        bbox,
        image_size,
    ) or _looks_like_high_confidence_top_text_free_background_bbox(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        image_size=image_size,
        model_confidence=_semantic_role_model_confidence(semantic_role_evidence),
    )
    if (fused_type == "caption_or_background_candidate" or fused_type == "free_text") and caption_background_candidate_bbox:
        if _text_free_caption_candidate_is_protected_sfx(
            fused_type=fused_type,
            reasons=reasons,
            bbox=bbox,
            visual=visual,
            image_size=image_size,
            luma_image=luma_image,
        ):
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "protected_authority_states",
                AUTH_PROTECT_SFX_DECORATIVE,
                evidence_kind="typed_text_free_edge_sfx_decorative_authority",
            )
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SFX,
                bbox=bbox,
                source_model_ids=source_ids,
                semantic_role_evidence=semantic_role_evidence,
                confidence=confidence,
                confidence_tier="text_free_edge_sfx_preserve",
                route_intent=ROUTE_PRESERVE_SFX,
                ocr_eligible=False,
                comic_text_detector_scope_eligible=False,
                fallback_reason="text_free_edge_sfx_decorative_preserve",
                evidence_reason_codes=reasons
                + _visual_reason_codes(visual)
                + ["text_area_plan:text_free_edge_sfx_decorative_preserve"],
                human_review_required=True,
                ocr_eligibility_reason="blocked_text_free_edge_sfx_decorative",
            )
        authority_reason = (
            "text_area_plan:caption_background_model_authority"
            if _text_free_caption_candidate_has_background_authority(
                fused_type=fused_type,
                reasons=reasons,
                bbox=bbox,
                visual=visual,
                image_size=image_size,
                luma_image=luma_image,
                model_confidence=_semantic_role_model_confidence(semantic_role_evidence),
            )
            else "text_area_plan:caption_background_candidate"
        )
        review_only = authority_reason != "text_area_plan:caption_background_model_authority"
        if not review_only:
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "cleanup_authority_states",
                AUTH_CLEANUP_TRANSLATE_BACKGROUND,
                evidence_kind="typed_text_free_background_model_authority",
            )
            expanded_bbox = _expand_compact_top_text_free_background_bbox(
                luma_image=luma_image,
                bbox=bbox,
                reasons=reasons,
                visual=visual,
                image_size=image_size,
            )
            if list(expanded_bbox) != list(bbox):
                semantic_role_evidence = dict(semantic_role_evidence)
                x0, y0, bw, bh = _coerce_xywh(expanded_bbox)
                semantic_role_evidence["text_unit_evidence_bboxes"] = [
                    {
                        "bbox": [x0, y0, x0 + bw, y0 + bh],
                        "class_name": "text_free",
                        "evidence_id": f"{container_id}_expanded_top_text_free",
                    }
                ]
            bbox = expanded_bbox
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="text_free_caption_background_authority" if not review_only else "text_free_review_only",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason="caption_background_model_candidate_review" if review_only else None,
            evidence_reason_codes=reasons + [authority_reason],
            human_review_required=review_only,
            ocr_eligibility_reason="caption_background_container",
        )

    if fused_type in {"free_text", "caption_or_background_candidate"}:
        side_caption_signal = _looks_like_vertical_side_caption_signal_bbox(
            bbox,
            image_size,
        ) or _looks_like_side_narration_background_bbox(
            fused_type=fused_type,
            reasons=reasons,
            bbox=bbox,
            visual=visual,
            image_size=image_size,
            semantic_role_evidence=semantic_role_evidence,
        )
        if side_caption_signal:
            # Deterministic side-caption localization owns these roots so broad
            # text_free fused evidence cannot preempt them as protected SFX.
            return None

    if fused_type == "sfx_or_decorative_candidate" or _looks_like_pre_ocr_sfx_or_decorative(fused_type, reasons, bbox, visual, image_size):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "protected_authority_states",
            AUTH_PROTECT_SFX_DECORATIVE,
            evidence_kind="typed_sfx_decorative_model_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="conflict_preserve_wins",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="sfx_decorative_model_candidate",
            evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:sfx_decorative_preserve"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_sfx_decorative_art_container",
        )

    if _looks_like_paired_ogkalu_speech_bubble(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
        clipped=clipped,
        luma_image=luma_image,
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "cleanup_authority_states",
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            evidence_kind="typed_ogkalu_bubble_text_pair_speech_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SPEECH,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier=OGKALU_BUBBLE_TEXT_PAIR_AUTHORITY_REASON,
            route_intent=ROUTE_TRANSLATE_SPEECH,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None,
            evidence_reason_codes=reasons
            + _visual_reason_codes(visual)
            + [
                "text_area_plan:paired_ogkalu_root_bbox_from_bubble_support",
                f"text_area_plan:{OGKALU_BUBBLE_TEXT_PAIR_AUTHORITY_REASON}",
                "text_area_plan:source_text_free_text_presence_proven",
            ],
            conflict_flags=conflicts,
            human_review_required=False,
            ocr_eligibility_reason=OGKALU_BUBBLE_TEXT_PAIR_AUTHORITY_REASON,
            cleanup_authorization=AUTH_CLEANUP_TRANSLATE_SPEECH,
            semantic_kind=SEMANTIC_KIND_SPEECH,
            semantic_authorization_state=AUTH_CLEANUP_TRANSLATE_SPEECH,
            authorization_source_stage="text_area_plan",
            authorization_basis=OGKALU_BUBBLE_TEXT_PAIR_AUTHORITY_REASON,
            authorization_explicit=True,
            authorization_field_origin=FRESH_AUTHORIZATION_FIELD_ORIGIN,
        )

    if _looks_like_standalone_ogkalu_speech_bubble(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
        clipped=clipped,
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "cleanup_authority_states",
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            evidence_kind="typed_bright_ogkalu_bubble_speech_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SPEECH,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="bright_ogkalu_bubble_speech_authority",
            route_intent=ROUTE_TRANSLATE_SPEECH,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None,
            evidence_reason_codes=reasons
            + _visual_reason_codes(visual)
            + ["text_area_plan:bright_ogkalu_bubble_speech_authority"],
            conflict_flags=conflicts,
            human_review_required=False,
            ocr_eligibility_reason="bright_ogkalu_bubble_speech_authority",
        )

    if _looks_like_unresolved_ogkalu_speech_risk_candidate(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
        clipped=clipped,
    ):
        semantic_role_evidence = dict(semantic_role_evidence)
        role_signals = set(_semantic_role_values(semantic_role_evidence, "role_signals"))
        role_signals.add(UNRESOLVED_OGKALU_SPEECH_RISK_REASON)
        semantic_role_evidence["role_signals"] = sorted(role_signals)
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_UNKNOWN,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier=UNRESOLVED_OGKALU_SPEECH_RISK_REASON,
            route_intent=ROUTE_REVIEW_FALLBACK,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=UNRESOLVED_OGKALU_SPEECH_RISK_REASON,
            evidence_reason_codes=reasons
            + _visual_reason_codes(visual)
            + [f"text_area_plan:{UNRESOLVED_OGKALU_SPEECH_RISK_REASON}"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason=UNRESOLVED_OGKALU_SPEECH_RISK_REASON,
        )

    if _looks_like_bright_unlinked_text_free_sfx_or_decorative(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "protected_authority_states",
            AUTH_PROTECT_SFX_DECORATIVE,
            evidence_kind="typed_bright_unlinked_text_free_sfx_decorative_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="bright_unlinked_text_free_sfx_decorative",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="bright_unlinked_text_free_sfx_decorative_preserve",
            evidence_reason_codes=reasons
            + _visual_reason_codes(visual)
            + ["text_area_plan:bright_unlinked_text_free_sfx_decorative_preserve"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_bright_unlinked_text_free_sfx_decorative",
        )

    if _looks_like_side_narration_background_bbox(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
    ):
        semantic_role_evidence = dict(semantic_role_evidence)
        role_signals = set(_semantic_role_values(semantic_role_evidence, "role_signals"))
        role_signals.add("side_narration_candidate")
        semantic_role_evidence["role_signals"] = sorted(role_signals)
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "cleanup_authority_states",
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            evidence_kind="typed_side_narration_background_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="side_narration_background_authority",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None,
            evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:side_narration_background_authority"],
            conflict_flags=conflicts,
            human_review_required=False,
            ocr_eligibility_reason="side_narration_background_container",
        )

    if _looks_like_dark_unlinked_ogkalu_sfx_or_decorative(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "protected_authority_states",
            AUTH_PROTECT_SFX_DECORATIVE,
            evidence_kind="typed_dark_unlinked_ogkalu_sfx_decorative_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="dark_unlinked_ogkalu_sfx_decorative",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="dark_unlinked_ogkalu_sfx_decorative_preserve",
            evidence_reason_codes=reasons
            + _visual_reason_codes(visual)
            + ["text_area_plan:dark_unlinked_ogkalu_sfx_decorative_preserve"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_dark_unlinked_ogkalu_sfx_decorative",
        )

    if _typed_multicolumn_unlinked_text_free_background_authority(
        fused,
        semantic_role_evidence,
        bbox=bbox,
        image_size=image_size,
        luma_image=luma_image,
        clipped=clipped,
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "cleanup_authority_states",
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            evidence_kind="typed_multicolumn_text_free_background_authority",
        )
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier="typed_multicolumn_text_free_background_authority",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None,
            evidence_reason_codes=reasons
            + ["text_area_plan:typed_multicolumn_text_free_background_authority"],
            conflict_flags=conflicts,
            human_review_required=False,
            ocr_eligibility_reason="typed_multicolumn_text_free_background_authority",
            cleanup_authorization=AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            semantic_kind=SEMANTIC_KIND_BACKGROUND_NARRATION,
            semantic_authorization_state=AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            authorization_source_stage="text_area_plan",
            authorization_basis="typed_multicolumn_text_free_background_authority",
            authorization_explicit=True,
            authorization_field_origin=FRESH_AUTHORIZATION_FIELD_ORIGIN,
        )

    safe_unknown = _safe_unknown_ocr_fallback(fused_type, reasons, bbox, visual, image_size)
    if safe_unknown:
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_UNKNOWN,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            semantic_role_evidence=semantic_role_evidence,
            confidence=confidence,
            confidence_tier=confidence_tier,
            route_intent=ROUTE_REVIEW_FALLBACK,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason="safe_unknown_ocr_for_text_conservation",
            evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:unknown_fallback_ocr_for_text_conservation"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="unknown_fallback_ocr_for_text_conservation",
        )

    return TextAreaContainer(
        container_id=container_id,
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=bbox,
        source_model_ids=source_ids,
        semantic_role_evidence=semantic_role_evidence,
        confidence=confidence,
        confidence_tier=confidence_tier,
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=(
            "clipped_or_degenerate_model_container_review_only"
            if clipped
            else f"unresolved_fused_container_type:{fused_type or 'unknown'}"
        ),
        evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:unknown_fallback_blocked_review_only"],
        conflict_flags=conflicts,
        human_review_required=True,
        ocr_eligibility_reason="blocked_unknown_review_only",
    )


def _duplicate_semantic_text_area_container(container: TextAreaContainer, existing: Sequence[TextAreaContainer]) -> bool:
    if container.container_type not in {CONTAINER_CAPTION, CONTAINER_SPEECH}:
        return False
    if container.route_intent not in {ROUTE_TRANSLATE_CAPTION, ROUTE_TRANSLATE_SPEECH}:
        return False
    reason_text = " ".join(str(item).lower() for item in container.evidence_reason_codes or [])
    duplicate_sensitive = any(
        token in reason_text
        for token in (
            "side_narration_background_authority",
            "ogkalu_bubble_without_kitsumed_mask",
            "ogkalu_text_bubble_without_kitsumed_mask",
            "caption_background_model_authority",
        )
    )
    if not duplicate_sensitive:
        return False
    for other in existing:
        if other.container_type != container.container_type or other.route_intent != container.route_intent:
            continue
        overlap = _intersection_ratio_xywh(container.bbox, other.bbox)
        reverse = _intersection_ratio_xywh(other.bbox, container.bbox)
        if max(overlap, reverse) >= 0.55:
            return True
    return False


def _container_from_unlinked_text_area_evidence(
    page_id: str,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
) -> TextAreaContainer | None:
    evidence_id = str(evidence.get("model_evidence_id") or evidence.get("evidence_id") or "")
    if not evidence_id:
        return None
    if _evidence_is_linked_to_fused_container(evidence_id, result):
        return None
    class_name = str(evidence.get("class_name") or "")
    bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
    confidence = float(evidence.get("confidence") or 0.0)
    semantic_role_evidence = _semantic_role_evidence_from_text_area_evidence(evidence)
    visual = _container_visual_stats(luma_image, bbox, image_size)
    top_caption = class_name == "text_free" and (
        _looks_like_caption_background_bbox(bbox, image_size)
        or _looks_like_high_confidence_top_text_free_background_bbox(
            fused_type="free_text",
            reasons=["ogkalu_text_free_without_kitsumed_mask"],
            bbox=bbox,
            image_size=image_size,
            model_confidence=confidence,
        )
    )
    if top_caption:
        reasons = ["ogkalu_text_free_without_kitsumed_mask", "text_area_plan:top_caption_background_candidate"]
        has_background_authority = (
            confidence >= OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE
            and not _text_free_caption_candidate_is_protected_sfx(
                fused_type="free_text",
                reasons=reasons,
                bbox=bbox,
                visual=visual,
                image_size=image_size,
                luma_image=luma_image,
            )
        )
        if has_background_authority:
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "cleanup_authority_states",
                AUTH_CLEANUP_TRANSLATE_BACKGROUND,
                evidence_kind="typed_ogkalu_text_free_background_authority",
            )
        return TextAreaContainer(
            container_id=f"ogkalu_{evidence_id}",
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=[evidence_id],
            semantic_role_evidence=semantic_role_evidence,
            confidence=round(confidence, 6),
            confidence_tier="text_free_background_authority" if has_background_authority else "text_free_review_only",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None if has_background_authority else "top_row_text_free_caption_candidate",
            evidence_reason_codes=["ogkalu_text_free"]
            + reasons[1:]
            + (
                ["text_area_plan:high_confidence_ogkalu_text_free_background_authority"]
                if has_background_authority
                else []
            )
            + _visual_reason_codes(visual),
            human_review_required=not has_background_authority,
            ocr_eligibility_reason="caption_background_container",
        )
    if class_name == "text_free":
        side_caption_signal = _looks_like_vertical_side_caption_signal_bbox(
            bbox,
            image_size,
        ) or _looks_like_side_narration_background_bbox(
            fused_type="free_text",
            reasons=[f"ogkalu_class:{class_name}"],
            bbox=bbox,
            visual=visual,
            image_size=image_size,
            semantic_role_evidence=semantic_role_evidence,
        )
        if side_caption_signal:
            # Deterministic side-caption localization owns these roots so broad
            # text_free evidence cannot preempt them as protected SFX.
            return None
    if class_name == "text_free" and _looks_like_pre_ocr_sfx_or_decorative("free_text", [f"ogkalu_class:{class_name}"], bbox, visual, image_size):
        return TextAreaContainer(
            container_id=f"ogkalu_{evidence_id}",
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=[evidence_id],
            semantic_role_evidence=semantic_role_evidence,
            confidence=round(confidence, 6),
            confidence_tier="text_free_review_only",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="ogkalu_text_free_non_caption_art_review",
            evidence_reason_codes=[f"ogkalu_class:{class_name}", "text_area_plan:text_free_non_caption_preserve_review"] + _visual_reason_codes(visual),
            human_review_required=True,
            ocr_eligibility_reason="blocked_text_free_non_caption",
        )
    if class_name in {"bubble", "text_bubble"} and _looks_like_standalone_ogkalu_speech_bubble(
        fused_type="ambiguous",
        reasons=[f"ogkalu_{class_name}_without_kitsumed_mask"],
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        semantic_role_evidence=semantic_role_evidence,
        clipped=_is_clipped_or_degenerate_bbox(bbox, image_size),
    ):
        semantic_role_evidence = _semantic_role_evidence_with_state(
            semantic_role_evidence,
            "cleanup_authority_states",
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            evidence_kind="typed_ogkalu_text_bubble_speech_authority",
        )
        return TextAreaContainer(
            container_id=f"ogkalu_{evidence_id}",
            page_id=page_id,
            container_type=CONTAINER_SPEECH,
            bbox=bbox,
            source_model_ids=[evidence_id],
            semantic_role_evidence=semantic_role_evidence,
            confidence=round(confidence, 6),
            confidence_tier="ogkalu_text_bubble_speech_authority",
            route_intent=ROUTE_TRANSLATE_SPEECH,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason=None,
            evidence_reason_codes=[
                f"ogkalu_class:{class_name}",
                f"text_area_plan:ogkalu_{class_name}_speech_authority",
            ]
            + _visual_reason_codes(visual),
            human_review_required=False,
            ocr_eligibility_reason="ogkalu_text_bubble_speech_container",
        )
    return TextAreaContainer(
        container_id=f"ogkalu_{evidence_id}",
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=bbox,
        source_model_ids=[evidence_id],
        semantic_role_evidence=semantic_role_evidence,
        confidence=round(confidence, 6),
        confidence_tier="text_free_review_only" if class_name == "text_free" else "text_bubble_review_container",
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=f"ogkalu_only_{class_name or 'unknown'}_review",
        evidence_reason_codes=[f"ogkalu_class:{class_name}", "text_area_plan:ogkalu_only_review_blocked"] + _visual_reason_codes(visual),
        human_review_required=True,
        ocr_eligibility_reason="blocked_ogkalu_only_review",
    )


def _append_deterministic_top_band_caption_containers(
    plan: TextAreaPlan,
    page_id: str,
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
    seen: set[str] | None = None,
) -> None:
    if luma_image is None:
        return
    has_model_signal = _has_top_band_caption_search_signal(result, image_size)
    visual_bboxes = _top_band_visual_caption_search_bboxes(luma_image, image_size)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    search_height = max(120, int(height * 0.22))
    candidates = [
        (
            "det_top_caption_day_center",
            [int(width * 0.56), 0, int(width * 0.09), int(search_height * 0.72)],
            "text_area_plan:deterministic_top_band_day_caption_search",
            "static",
        ),
        (
            "det_top_caption_mid_left",
            [int(width * 0.50), 0, int(width * 0.11), int(search_height * 0.82)],
            "text_area_plan:deterministic_top_band_mid_left_caption_search",
            "static",
        ),
        (
            "det_top_caption_far_right",
            [int(width * 0.82), 0, int(width * 0.16), search_height],
            "text_area_plan:deterministic_top_band_far_right_caption_search",
            "static",
        ),
        (
            "det_top_caption_center",
            [int(width * 0.48), 0, int(width * 0.18), search_height],
            TOP_BAND_CAPTION_SEARCH_REASON,
            "static",
        ),
        (
            "det_top_caption_right",
            [int(width * 0.68), 0, int(width * 0.20), search_height],
            TOP_BAND_CAPTION_SEARCH_REASON,
            "static",
        ),
    ]
    if not has_model_signal and not visual_bboxes and not any(
        _top_band_static_caption_candidate_has_ink(luma_image, raw_bbox, image_size)
        for _cid, raw_bbox, _reason, _kind in candidates
    ):
        return
    for index, bbox in enumerate(visual_bboxes):
        candidates.append(
            (
                f"det_top_caption_visual_{index:02d}",
                bbox,
                "text_area_plan:deterministic_top_band_visual_text_search",
                "visual",
            )
        )
    source_ids = _top_band_caption_search_source_ids(result, image_size)
    pending: List[Tuple[str, List[int], str, str, bool]] = []
    for container_id, raw_bbox, search_reason, candidate_kind in candidates:
        if seen is not None and container_id in seen:
            continue
        bbox = _normalize_xywh(raw_bbox, image_size)
        if candidate_kind == "static":
            if container_id in {
                "det_top_caption_day_center",
                "det_top_caption_mid_left",
                "det_top_caption_center",
                "det_top_caption_right",
            }:
                localized_columns = _top_band_character_column_bboxes(luma_image, bbox, image_size)
                if localized_columns:
                    for column_index, column_bbox in enumerate(localized_columns):
                        column_id = f"{container_id}_visual_{column_index:02d}"
                        if seen is not None and column_id in seen:
                            continue
                        authority_allowed = _top_band_caption_background_authority_allowed(
                            container_id=column_id,
                            bbox=column_bbox,
                            search_reason=f"{search_reason}:localized_character_column",
                            candidate_kind="visual_column",
                            image_size=image_size,
                            existing_containers=plan.containers,
                        )
                        if container_id == "det_top_caption_right" and not _top_band_right_column_has_background_context(
                            luma_image,
                            column_bbox,
                            image_size,
                        ):
                            authority_allowed = False
                        if _top_band_caption_candidate_is_duplicate(
                            column_bbox,
                            pending,
                            plan.containers,
                            source_ids,
                        ):
                            continue
                        pending.append(
                            (
                                column_id,
                                column_bbox,
                                f"{search_reason}:localized_character_column",
                                "visual_column",
                                authority_allowed,
                            )
                        )
                    continue
            localized = _localize_top_caption_bbox(luma_image, bbox, image_size)
            if localized is not None:
                bbox = localized
                search_reason = f"{search_reason}:localized_ink"
            elif container_id == "det_top_caption_far_right":
                # The far-right top band commonly contains action/SFX strokes.
                # It can only become semantic background authority when the
                # existing TextAreaPlan localization can isolate caption text.
                continue
            else:
                # Keep broad top-band boxes as scoped OCR search areas only.
                # Controller/hierarchy must later render and clean against the
                # OCR/glyph evidence, and duplicate partial captions are
                # suppressed before translation.
                search_reason = f"{search_reason}:search_scope_only"
        authority_allowed = _top_band_caption_background_authority_allowed(
            container_id=container_id,
            bbox=bbox,
            search_reason=search_reason,
            candidate_kind=candidate_kind,
            image_size=image_size,
            existing_containers=plan.containers,
        )
        if _top_band_caption_candidate_is_duplicate(bbox, pending, plan.containers, source_ids) and not (
            authority_allowed and _top_band_caption_candidate_expands_existing_background_scope(bbox, plan.containers)
        ):
            continue
        pending.append((container_id, bbox, search_reason, candidate_kind, authority_allowed))

    for container_id, bbox, search_reason, _candidate_kind, authority_allowed in pending:
        if _caption_search_overlaps_blocking_container(bbox, plan.containers):
            continue
        visual = _container_visual_stats(luma_image, bbox, image_size)
        role_signals = ["caption_background_candidate"]
        semantic_role_evidence: Dict[str, Any] = {
            "role_signals": role_signals,
            "source": "text_area_plan_deterministic_top_band",
        }
        reason_codes = [
            search_reason,
            "text_area_plan:caption_background_container",
        ] + _visual_reason_codes(visual)
        if authority_allowed:
            role_signals.append("top_band_background_authority")
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "cleanup_authority_states",
                AUTH_CLEANUP_TRANSLATE_BACKGROUND,
                evidence_kind="typed_deterministic_top_band_background_authority",
            )
            reason_codes.append("text_area_plan:deterministic_top_band_background_authority")
        plan.containers.append(
            TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_CAPTION,
                bbox=bbox,
                source_model_ids=source_ids,
                semantic_role_evidence=semantic_role_evidence,
                confidence="deterministic",
                confidence_tier=(
                    "deterministic_top_band_background_authority"
                    if authority_allowed
                    else "deterministic_top_band_caption_search"
                ),
                route_intent=ROUTE_TRANSLATE_CAPTION,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                fallback_reason=None if authority_allowed else "deterministic_top_band_caption_search",
                evidence_reason_codes=reason_codes,
                human_review_required=not authority_allowed,
                ocr_eligibility_reason="caption_background_container_strict_ocr_gate",
            )
        )
        if seen is not None:
            seen.add(container_id)


def _localize_top_caption_bbox(luma_image: Any, search_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int] | None:
    if luma_image is None:
        return None
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return None
    try:
        crop = luma_image.crop((x, y, min(width, x + w), min(height, y + h))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return None
        pixels = crop.load()
        dark_points: List[Tuple[int, int]] = []
        for cy in range(ch):
            for cx in range(cw):
                if pixels[cx, cy] <= 124:
                    dark_points.append((cx, cy))
        if len(dark_points) < max(24, int(cw * ch * 0.004)):
            return None
        xs = [pt[0] for pt in dark_points]
        ys = [pt[1] for pt in dark_points]
        lx0 = max(0, min(xs) - max(6, int(cw * 0.035)))
        lx1 = min(cw, max(xs) + 1 + max(6, int(cw * 0.035)))
        ly0 = max(0, min(ys) - max(8, int(ch * 0.030)))
        ly1 = min(ch, max(ys) + 1 + max(8, int(ch * 0.030)))
        bw = lx1 - lx0
        bh = ly1 - ly0
        if bw < 12 or bh < 32:
            return None
        localized_area = bw * bh
        search_area = max(1, cw * ch)
        if localized_area > search_area * 0.62:
            return None
        localized = _normalize_xywh([x + lx0, y + ly0, bw, bh], image_size)
        if not _looks_like_top_caption_bbox(localized, image_size):
            return None
        return localized
    except Exception:
        return None


def _top_band_caption_background_authority_allowed(
    *,
    container_id: str,
    bbox: Sequence[int],
    search_reason: str,
    candidate_kind: str,
    image_size: Tuple[int, int],
    existing_containers: Sequence[TextAreaContainer] | None = None,
) -> bool:
    if "far_right" in str(container_id):
        return False
    if not _looks_like_top_caption_bbox(bbox, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    if x + w >= width - max(2, int(width * 0.002)):
        return False
    if candidate_kind == "visual_column":
        return False
    if w < width * 0.045 or w > width * 0.24:
        return False
    if h < height * 0.045 or h > height * 0.24:
        return False
    if candidate_kind == "visual":
        return False
    reason = str(search_reason)
    if "localized_ink" in reason:
        # Ink localization narrows a search scope, but it does not establish
        # the semantic identity required to create an executable root.
        return False
    if "search_scope_only" in reason:
        # Broad static top-band scopes are candidate search areas. They can only
        # carry OCR/projection context. Semantic authority must come from
        # localized text-column evidence or an upstream typed model/current-region
        # record, otherwise SFX/action strokes in the same band can turn green.
        return False
    return False


def _top_band_caption_candidate_expands_existing_background_scope(
    bbox: Sequence[int],
    existing_containers: Sequence[TextAreaContainer],
) -> bool:
    candidate_area = _bbox_area_xywh(bbox)
    if candidate_area <= 0:
        return False
    for container in existing_containers:
        if container.container_type != CONTAINER_CAPTION:
            continue
        existing_area = _bbox_area_xywh(container.bbox)
        if existing_area <= 0 or candidate_area <= existing_area * 1.20:
            continue
        if _inside_ratio_xywh(container.bbox, bbox) >= 0.65:
            return True
    return False


def _top_band_character_column_bboxes(
    luma_image: Any,
    search_bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> List[List[int]]:
    """Find localized top-band vertical text columns without authorizing the full search scope."""
    if luma_image is None or np is None or cv2 is None:
        return []
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return []
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return []
    try:
        crop = luma_image.crop((x0, y0, x1, y1)).convert("L")
        arr = np.asarray(crop)
        mask = (arr <= 96).astype("uint8")
        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return []

    char_components: List[Tuple[int, int, int, int, int]] = []
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, area = [int(value) for value in stats[label][:5]]
        if area < 8 or area > 1800:
            continue
        if cw < 2 or ch < 4 or cw > 70 or ch > 80:
            continue
        aspect = ch / float(max(1, cw))
        if aspect < 0.40 or aspect > 10.0:
            continue
        density = area / float(max(1, cw * ch))
        if density < 0.08 or density > 0.75:
            continue
        char_components.append((cx, cy, cw, ch, area))
    if len(char_components) < 4:
        return []

    clusters: List[Dict[str, Any]] = []
    for comp in sorted(char_components, key=lambda item: item[0] + item[2] / 2.0):
        center_x = comp[0] + comp[2] / 2.0
        placed = False
        for cluster in clusters:
            if abs(center_x - float(cluster["center_x"])) <= max(24.0, float(cluster["max_width"]) * 0.70):
                cluster["items"].append(comp)
                centers = [item[0] + item[2] / 2.0 for item in cluster["items"]]
                cluster["center_x"] = sum(centers) / float(len(centers))
                cluster["max_width"] = max(item[2] for item in cluster["items"])
                placed = True
                break
        if not placed:
            clusters.append({"center_x": center_x, "max_width": comp[2], "items": [comp]})

    candidates: List[List[int]] = []
    for cluster in clusters:
        items = list(cluster.get("items") or [])
        if len(items) < 6:
            continue
        xs = [item[0] for item in items]
        ys = [item[1] for item in items]
        x2 = [item[0] + item[2] for item in items]
        y2 = [item[1] + item[3] for item in items]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(x2), max(y2)
        bw = bx1 - bx0
        bh = by1 - by0
        if bw < 8 or bw > 90:
            continue
        if bh < max(72, int(height * 0.045)) or bh > int(height * 0.24):
            continue
        row_bins = {
            int((item[1] - by0) / float(max(1, bh)) * 5.0)
            for item in items
        }
        if len(row_bins) < 3:
            continue
        pixel_area = sum(item[4] for item in items)
        density = pixel_area / float(max(1, bw * bh))
        median_height = sorted(item[3] for item in items)[len(items) // 2]
        if density > 0.22 or median_height > 40:
            continue
        dominant_action_bar = any(
            item[2] >= max(42, int(bw * 0.70))
            and (item[3] / float(max(1, item[2]))) < 0.90
            for item in items
        )
        if dominant_action_bar:
            continue
        pad_x = max(4, int(bw * 0.12))
        pad_y = max(6, int(bh * 0.04))
        candidate = _normalize_xywh(
            [x0 + bx0 - pad_x, y0 + by0 - pad_y, bw + pad_x * 2, bh + pad_y * 2],
            image_size,
        )
        if _looks_like_top_caption_bbox(candidate, image_size):
            candidates.append(candidate)

    candidates.sort(key=lambda item: (item[0], item[1]))
    deduped: List[List[int]] = []
    for candidate in candidates:
        if any(_intersection_ratio_xywh(candidate, existing) >= 0.65 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= 3:
            break
    return deduped


def _top_band_caption_candidate_is_duplicate(
    bbox: Sequence[int],
    pending: Sequence[Tuple[str, Sequence[int], str, str, bool]],
    existing_containers: Sequence[TextAreaContainer],
    source_ids: Sequence[str],
) -> bool:
    for _cid, existing_bbox, _reason, _kind, _authority_allowed in pending:
        if _intersection_ratio_xywh(bbox, existing_bbox) >= 0.70:
            return True
        if _inside_ratio_xywh(bbox, existing_bbox) >= 0.80 or _inside_ratio_xywh(existing_bbox, bbox) >= 0.80:
            return True
    source_id_set = {str(item) for item in source_ids if str(item)}
    for container in existing_containers:
        if container.container_type != CONTAINER_CAPTION:
            continue
        overlap = _intersection_ratio_xywh(bbox, container.bbox)
        if overlap >= 0.82:
            return True
        existing_source_ids = {str(item) for item in (container.source_model_ids or []) if str(item)}
        if source_id_set and existing_source_ids and source_id_set & existing_source_ids and overlap >= 0.50:
            return True
    return False


def _top_band_visual_caption_search_bboxes(luma_image: Any, image_size: Tuple[int, int]) -> List[List[int]]:
    """Find narrow top-band text columns when model evidence missed captions.

    This stays within TextAreaPlan admission: it only creates caption/background
    OCR scopes, and later OCR quality/SFX gates still decide whether text enters
    translation. Wide impact marks and panel borders are rejected here.
    """
    if luma_image is None:
        return []
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    search_height = max(120, int(height * 0.22))
    try:
        crop = luma_image.crop((0, 0, width, min(height, search_height))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return []
        pixels = crop.load()
    except Exception:
        return []

    col_counts: List[int] = []
    for cx in range(cw):
        count = 0
        for cy in range(ch):
            if pixels[cx, cy] <= 112:
                count += 1
        col_counts.append(count)
    threshold = max(5, int(ch * 0.012))
    active_cols = [idx for idx, count in enumerate(col_counts) if count >= threshold]
    if not active_cols:
        return []

    groups: List[Tuple[int, int]] = []
    start = prev = active_cols[0]
    max_gap = max(8, int(width * 0.012))
    for idx in active_cols[1:]:
        if idx - prev <= max_gap:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))

    candidates: List[List[int]] = []
    for gx0, gx1 in groups:
        gw = gx1 - gx0 + 1
        if gw < 10 or gw > max(150, int(width * 0.14)):
            continue
        band_dark = sum(col_counts[gx0 : gx1 + 1])
        if band_dark < max(70, int(gw * ch * 0.014)):
            continue
        pad_x = max(6, int(gw * 0.20))
        lx0 = max(0, gx0 - pad_x)
        lx1 = min(cw, gx1 + 1 + pad_x)
        row_counts: List[int] = []
        for cy in range(ch):
            count = 0
            for cx in range(lx0, lx1):
                if pixels[cx, cy] <= 124:
                    count += 1
            row_counts.append(count)
        row_threshold = max(2, int((lx1 - lx0) * 0.040))
        active_rows = [idx for idx, count in enumerate(row_counts) if count >= row_threshold]
        if not active_rows:
            continue
        ly0 = max(0, min(active_rows) - max(8, int(ch * 0.018)))
        ly1 = min(ch, max(active_rows) + 1 + max(8, int(ch * 0.018)))
        bh = ly1 - ly0
        bw = lx1 - lx0
        if bh < max(48, int(height * 0.025)):
            continue
        if bh > int(height * 0.24) and bw > int(width * 0.08):
            continue
        area = max(1, bw * bh)
        dark_density = band_dark / float(area)
        if dark_density > 0.42:
            continue
        candidate = _normalize_xywh([lx0, ly0, bw, bh], image_size)
        if _looks_like_top_caption_bbox(candidate, image_size):
            candidates.append(candidate)

    # Deduplicate nested visual candidates, keeping the tighter scope.
    deduped: List[List[int]] = []
    for candidate in candidates:
        if any(_intersection_ratio_xywh(candidate, existing) > 0.72 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= 4:
            break
    return deduped


def _top_band_static_caption_candidate_has_ink(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None:
        return False
    try:
        x, y, w, h = _coerce_xywh(bbox)
        width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
        if w <= 0 or h <= 0:
            return False
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h))).convert("L")
        pixels = list(crop.getdata())
        if not pixels:
            return False
        dark_ratio = sum(1 for value in pixels if value <= 116) / float(len(pixels))
        mid_ratio = sum(1 for value in pixels if value <= 170) / float(len(pixels))
        # These static scopes are still gated by TextAreaPlan ownership and OCR
        # quality. The ink check only avoids adding empty top-band scopes.
        return 0.012 <= dark_ratio <= 0.36 and mid_ratio >= 0.025
    except Exception:
        return False


def _append_deterministic_vertical_side_caption_containers(
    plan: TextAreaPlan,
    page_id: str,
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
    seen: set[str] | None = None,
) -> None:
    if luma_image is None:
        return
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    source_items = _side_caption_signal_items(result, image_size)
    if not source_items:
        return
    added = 0
    added_boxes: list[list[int]] = []
    per_side_counts: Dict[str, int] = {}
    for item in source_items:
        seed_bbox = _bbox_xyxy_to_xywh(item.get("mask_bbox") or item.get("bbox_xyxy") or item.get("bbox"), image_size)
        if _side_caption_seed_overlaps_protected(seed_bbox, plan.containers):
            continue
        side = _side_caption_side_for_bbox(seed_bbox, image_size)
        search_bbox = _side_caption_search_bbox(seed_bbox, image_size)
        paired_text_boxes = _side_caption_paired_text_boundary_bboxes(result, item, search_bbox, image_size)
        if not paired_text_boxes:
            # Visual localization can support search and review, but only an
            # upstream typed text boundary may originate executable ownership.
            continue
        localized_boxes = paired_text_boxes
        item_id = str(item.get("model_evidence_id") or item.get("evidence_id") or item.get("fused_container_id") or "")
        authority_by_bbox: Dict[int, str] = {}
        for pre_index, pre_bbox in enumerate(localized_boxes):
            if _caption_search_overlaps_blocking_container(pre_bbox, plan.containers):
                continue
            authority_by_bbox[pre_index] = (
                "text_area_plan:deterministic_side_narration_background_authority:paired_text_boundary"
            )
        if not authority_by_bbox:
            continue
        authorized_indexes = [index for index in range(len(localized_boxes)) if index in authority_by_bbox]
        if not authorized_indexes:
            continue
        caption_candidates: List[Dict[str, Any]] = []
        if len(authorized_indexes) >= 2:
            group_bbox = _text_area_graph_union_xywh([localized_boxes[index] for index in authorized_indexes])
            if not group_bbox:
                continue
            authority_reasons = _text_area_graph_unique_strings(
                [authority_by_bbox.get(index, "") for index in authorized_indexes]
            )
            caption_candidates.append(
                {
                    "bbox_index": authorized_indexes[0],
                    "bbox": _normalize_xywh(group_bbox, image_size),
                    "authority_reason": (
                        "text_area_plan:deterministic_side_narration_background_authority:grouped_columns"
                    ),
                    "authority_reasons": authority_reasons,
                    "source_indexes": list(authorized_indexes),
                    "grouped_columns": True,
                    "paired_text_boundary": True,
                }
            )
        else:
            for bbox_index in authorized_indexes:
                caption_candidates.append(
                    {
                        "bbox_index": bbox_index,
                        "bbox": localized_boxes[bbox_index],
                        "authority_reason": authority_by_bbox.get(bbox_index, ""),
                        "authority_reasons": [authority_by_bbox.get(bbox_index, "")],
                        "source_indexes": [bbox_index],
                        "grouped_columns": False,
                        "paired_text_boundary": True,
                    }
                )
        for candidate in caption_candidates:
            bbox_index = int(candidate.get("bbox_index") or 0)
            bbox = list(candidate.get("bbox") or [])
            duplicate_threshold = 0.62 if list(bbox) != list(search_bbox) else 0.35
            if any(_intersection_ratio_xywh(bbox, existing) >= duplicate_threshold for existing in added_boxes):
                continue
            if _caption_search_overlaps_blocking_container(bbox, plan.containers):
                continue
            side_index = per_side_counts.get(side, 0)
            container_id = f"det_side_caption_{side}_{side_index:02d}"
            if seen is not None and container_id in seen:
                continue
            visual = _container_visual_stats(luma_image, bbox, image_size)
            authority_reason = str(candidate.get("authority_reason") or "")
            reason_codes = [
                SIDE_CAPTION_SEARCH_REASON,
                "text_area_plan:caption_background_container",
                "text_area_plan:vertical_side_caption_search",
                f"text_area_plan:vertical_side_caption_{side}_search",
            ] + _visual_reason_codes(visual)
            reason_codes.extend(str(item) for item in candidate.get("authority_reasons") or [] if str(item))
            reason_codes.append(authority_reason)
            if bool(candidate.get("grouped_columns")):
                reason_codes.append("text_area_plan:vertical_side_caption_authorized_columns_grouped_root")
            reason_codes.append("text_area_plan:vertical_side_caption_paired_text_boundary")
            role_signals = ["side_narration_candidate"]
            role_signals.append("side_narration_background_authority")
            semantic_role_evidence: Dict[str, Any] = {
                "role_signals": sorted(set(role_signals)),
                "source": "text_area_plan_deterministic_vertical_side",
                "side": side,
                "column_index": bbox_index,
                "column_indexes": list(candidate.get("source_indexes") or [bbox_index]),
            }
            semantic_role_evidence["text_unit_evidence_bboxes"] = [
                {
                    "evidence_id": f"{container_id}_side_caption_text_island_00",
                    "class_name": "text_free",
                    "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    "is_explicit_parent_boundary_evidence": True,
                }
            ]
            semantic_role_evidence = _semantic_role_evidence_with_state(
                semantic_role_evidence,
                "cleanup_authority_states",
                AUTH_CLEANUP_TRANSLATE_BACKGROUND,
                evidence_kind="typed_deterministic_side_narration_background_authority",
            )
            plan.containers.append(
                TextAreaContainer(
                    container_id=container_id,
                    page_id=page_id,
                    container_type=CONTAINER_CAPTION,
                    bbox=bbox,
                    source_model_ids=[item_id] if item_id else [],
                    semantic_role_evidence=semantic_role_evidence,
                    confidence="deterministic",
                    confidence_tier="deterministic_side_narration_background_authority",
                    route_intent=ROUTE_TRANSLATE_CAPTION,
                    ocr_eligible=True,
                    comic_text_detector_scope_eligible=True,
                    fallback_reason=None,
                    evidence_reason_codes=reason_codes,
                    human_review_required=False,
                    ocr_eligibility_reason="caption_background_container_strict_ocr_gate",
                )
            )
            if seen is not None:
                seen.add(container_id)
            added_boxes.append(list(bbox))
            per_side_counts[side] = side_index + 1
            added += 1
            if added >= 4:
                break
        if added >= 4:
            break


def _append_deterministic_large_sfx_containers(
    plan: TextAreaPlan,
    page_id: str,
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
    seen: set[str] | None = None,
) -> None:
    for index, bbox in enumerate(_deterministic_large_sfx_component_bboxes(luma_image, image_size, plan.containers)):
        container_id = f"det_large_sfx_{index:03d}"
        if seen is not None and container_id in seen:
            continue
        source_id = f"{page_id}_{container_id}"
        semantic_role_evidence = {
            "role_signals": ["large_stylized_text_foreground", "sfx_decorative_authority"],
            "source": "text_area_plan_deterministic_large_sfx",
            "source_model_ids": [source_id],
            "evidence_source_list": ["text_area_plan_deterministic_visual_evidence"],
            "protected_authority_states": [AUTH_PROTECT_SFX_DECORATIVE],
            "protected_candidate_states": [],
            "cleanup_authority_states": [],
            "cleanup_candidate_states": [],
            "authority_evidence_kind": "typed_deterministic_large_sfx_visual_authority",
            "confidence": "deterministic",
        }
        plan.containers.append(
            TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SFX,
                bbox=bbox,
                source_model_ids=[source_id],
                semantic_role_evidence=semantic_role_evidence,
                confidence="deterministic",
                confidence_tier="deterministic_large_sfx_visual_authority",
                route_intent=ROUTE_PRESERVE_SFX,
                ocr_eligible=False,
                comic_text_detector_scope_eligible=False,
                fallback_reason="deterministic_large_sfx_visual_preserve",
                evidence_reason_codes=[
                    "text_area_plan:deterministic_large_sfx_visual_search",
                    "text_area_plan:deterministic_large_sfx_visual_authority",
                ],
                human_review_required=True,
                ocr_eligibility_reason="blocked_deterministic_large_sfx_visual",
            )
        )
        if seen is not None:
            seen.add(container_id)


def _demote_weak_background_authority_overlapping_protected(
    containers: Sequence[TextAreaContainer],
    image_size: Tuple[int, int],
) -> None:
    protected_boxes: List[List[int]] = []
    for container in containers:
        role_evidence = _container_semantic_role_evidence(container)
        protected_states = set(_semantic_role_state_values(role_evidence, "protected_authority_states"))
        if (
            container.container_type == CONTAINER_SFX
            or container.route_intent == ROUTE_PRESERVE_SFX
            or AUTH_PROTECT_SFX_DECORATIVE in protected_states
            or AUTH_PROTECT_ART_OR_NON_TEXT in protected_states
        ):
            bbox = _normalize_xywh(container.bbox, image_size)
            if bbox:
                protected_boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    if not protected_boxes:
        return

    for container in containers:
        role_evidence = _container_semantic_role_evidence(container)
        cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
        if not cleanup_states or AUTH_CLEANUP_TRANSLATE_SPEECH in cleanup_states:
            continue
        if not cleanup_states.issubset({AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}):
            continue
        typed_reasons = {str(item) for item in _semantic_role_values(role_evidence, "typed_authority_reason_codes")}
        if not any(any(token in reason for token in WEAK_BACKGROUND_AUTHORITY_REASON_TOKENS) for reason in typed_reasons):
            continue
        if _has_explicit_text_free_background_model_boundary(role_evidence, image_size):
            continue
        bbox = _normalize_xywh(container.bbox, image_size)
        if not bbox:
            continue
        bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        inside_protected = _max_inside_ratio_xyxy(bbox_xyxy, protected_boxes)
        protected_coverage = _max_coverage_of_boxes_xyxy(bbox_xyxy, protected_boxes)
        if inside_protected < 0.16 and protected_coverage < 0.25:
            continue

        updated_evidence = dict(role_evidence)
        updated_evidence["cleanup_authority_states"] = []
        updated_evidence["cleanup_candidate_states"] = sorted(
            set(_semantic_role_state_values(role_evidence, "cleanup_candidate_states")) | cleanup_states
        )
        role_signals = set(_semantic_role_values(updated_evidence, "role_signals"))
        role_signals.add("weak_background_authority_blocked_by_protected")
        updated_evidence["role_signals"] = sorted(role_signals)
        conflict_evidence = set(_semantic_role_values(updated_evidence, "conflict_evidence"))
        conflict_evidence.add("protected_overlap_blocks_weak_background_authority")
        updated_evidence["conflict_evidence"] = sorted(conflict_evidence)
        container.semantic_role_evidence = updated_evidence
        container.text_area_pre_ocr_authority = False
        container.pre_ocr_authority = False
        container.human_review_required = True
        container.fallback_reason = "weak_background_authority_overlaps_protected"
        container.confidence_tier = "weak_background_authority_blocked_by_protected"
        if "text_area_plan:weak_background_authority_blocked_by_protected" not in container.evidence_reason_codes:
            container.evidence_reason_codes.append("text_area_plan:weak_background_authority_blocked_by_protected")


def _is_ogkalu_only_speech_authority_container(container: TextAreaContainer) -> bool:
    role_evidence = _container_semantic_role_evidence(container)
    cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
    if AUTH_CLEANUP_TRANSLATE_SPEECH not in cleanup_states:
        return False
    role_classes = set(_semantic_role_values(role_evidence, "ogkalu_class_names"))
    if not (role_classes & {"bubble", "text_bubble"}):
        return False
    role_signals = set(_semantic_role_values(role_evidence, "role_signals"))
    current_roles = set(_semantic_role_values(role_evidence, "current_region_roles"))
    if "speech_bubble_mask_evidence" in role_signals or "speech" in current_roles:
        return False
    typed_reasons = {str(item) for item in _semantic_role_values(role_evidence, "typed_authority_reason_codes")}
    evidence_kind = str(role_evidence.get("authority_evidence_kind") or "")
    reason_text = " ".join(sorted(typed_reasons | {evidence_kind}))
    return any(token in reason_text for token in OGKALU_ONLY_SPEECH_AUTHORITY_REASON_TOKENS)


def _demote_ogkalu_speech_authority_overlapping_protected(
    containers: Sequence[TextAreaContainer],
    image_size: Tuple[int, int],
) -> None:
    protected_boxes: List[List[int]] = []
    for container in containers:
        role_evidence = _container_semantic_role_evidence(container)
        protected_states = set(_semantic_role_state_values(role_evidence, "protected_authority_states"))
        if (
            container.container_type == CONTAINER_SFX
            or container.route_intent == ROUTE_PRESERVE_SFX
            or AUTH_PROTECT_SFX_DECORATIVE in protected_states
            or AUTH_PROTECT_ART_OR_NON_TEXT in protected_states
        ):
            bbox = _normalize_xywh(container.bbox, image_size)
            if bbox:
                protected_boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    if not protected_boxes:
        return

    for container in containers:
        if not _is_ogkalu_only_speech_authority_container(container):
            continue
        speech_evidence_boxes = _ogkalu_speech_text_evidence_boxes_xyxy(container, image_size)
        if not speech_evidence_boxes:
            continue
        text_inside_protected = max(_max_inside_ratio_xyxy(box, protected_boxes) for box in speech_evidence_boxes)
        if text_inside_protected < 0.16:
            continue

        role_evidence = _container_semantic_role_evidence(container)
        cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
        cleanup_states.discard(AUTH_CLEANUP_TRANSLATE_SPEECH)
        updated_evidence = dict(role_evidence)
        updated_evidence["cleanup_authority_states"] = sorted(cleanup_states)
        updated_evidence["cleanup_candidate_states"] = sorted(
            set(_semantic_role_state_values(role_evidence, "cleanup_candidate_states")) | {AUTH_CLEANUP_TRANSLATE_SPEECH}
        )
        role_signals = set(_semantic_role_values(updated_evidence, "role_signals"))
        role_signals.add("ogkalu_speech_authority_blocked_by_protected")
        updated_evidence["role_signals"] = sorted(role_signals)
        conflict_evidence = set(_semantic_role_values(updated_evidence, "conflict_evidence"))
        conflict_evidence.add("protected_overlap_blocks_ogkalu_speech_authority")
        updated_evidence["conflict_evidence"] = sorted(conflict_evidence)
        container.semantic_role_evidence = updated_evidence
        container.text_area_pre_ocr_authority = False
        container.pre_ocr_authority = False
        container.ocr_eligible = False
        container.comic_text_detector_scope_eligible = False
        container.human_review_required = True
        container.fallback_reason = "ogkalu_speech_authority_overlaps_protected"
        container.confidence_tier = "ogkalu_speech_authority_blocked_by_protected"
        container.ocr_eligibility_reason = "blocked_ogkalu_speech_authority_overlaps_protected"
        if "text_area_plan:ogkalu_speech_authority_blocked_by_protected" not in container.evidence_reason_codes:
            container.evidence_reason_codes.append("text_area_plan:ogkalu_speech_authority_blocked_by_protected")


def _ogkalu_speech_text_evidence_boxes_xyxy(
    container: TextAreaContainer,
    image_size: Tuple[int, int],
) -> List[List[int]]:
    role_evidence = _container_semantic_role_evidence(container)
    boxes: List[List[int]] = []
    for entry in role_evidence.get("text_unit_evidence_bboxes") or []:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("class_name") or "") != "text_bubble":
            continue
        xyxy = _semantic_evidence_bbox_xyxy(entry.get("bbox"), image_size)
        if xyxy:
            boxes.append(xyxy)
    if boxes:
        return boxes
    bbox = _normalize_xywh(container.bbox, image_size)
    if not bbox:
        return []
    return [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]]


def _semantic_evidence_bbox_xyxy(value: Any, image_size: Tuple[int, int]) -> Optional[List[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x0, y0, v2, v3 = [float(item) for item in value[:4]]
    except Exception:
        return None
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if v2 > x0 and v3 > y0:
        x1, y1 = v2, v3
    else:
        x1, y1 = x0 + max(0.0, v2), y0 + max(0.0, v3)
    x0_i = max(0, min(width, int(round(x0))))
    y0_i = max(0, min(height, int(round(y0))))
    x1_i = max(0, min(width, int(round(x1))))
    y1_i = max(0, min(height, int(round(y1))))
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return [x0_i, y0_i, x1_i, y1_i]


def _deterministic_large_sfx_component_bboxes(
    luma_image: Any,
    image_size: Tuple[int, int],
    existing_containers: Sequence[TextAreaContainer],
) -> List[List[int]]:
    if luma_image is None or np is None or cv2 is None:
        return []
    cleanup_boxes: List[List[int]] = []
    protected_boxes: List[List[int]] = []
    for container in existing_containers:
        bbox = _normalize_xywh(container.bbox, image_size)
        if not bbox:
            continue
        xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        role_evidence = _container_semantic_role_evidence(container)
        cleanup_authority_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
        has_cleanup_authority = bool(
            cleanup_authority_states
            or (
                container.container_type == CONTAINER_SPEECH
                and container.route_intent == ROUTE_TRANSLATE_SPEECH
            )
        )
        if has_cleanup_authority and not _is_weak_side_background_cleanup_authority_container(container):
            cleanup_boxes.append(xyxy)
        elif container.container_type == CONTAINER_SFX or container.route_intent == ROUTE_PRESERVE_SFX:
            protected_boxes.append(xyxy)
    try:
        arr = np.asarray(luma_image.convert("L"))
        mask = (arr <= 110).astype("uint8")
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return []
    candidates: List[List[int]] = []
    for label in range(1, int(stats.shape[0])):
        x, y, w, h, area = [int(value) for value in stats[label][:5]]
        if area < 800 or area > 42000:
            continue
        if w < 16 or h < 16 or w > 360 or h > 360:
            continue
        density = area / float(max(1, w * h))
        if density < 0.055 or density > 0.72:
            continue
        if (w > 190 or h > 245) and density > 0.48:
            continue
        if (w > 95 and h < 20) or (h > 180 and w < 12):
            continue
        bbox_xyxy = [x, y, x + w, y + h]
        if _max_inside_ratio_xyxy(bbox_xyxy, cleanup_boxes) > 0.35:
            continue
        if _max_coverage_of_boxes_xyxy(bbox_xyxy, cleanup_boxes) > 0.50:
            continue
        if _max_inside_ratio_xyxy(bbox_xyxy, protected_boxes) > 0.92:
            continue
        padded = _normalize_xywh([x - 4, y - 4, w + 8, h + 8], image_size)
        if padded:
            candidates.append(padded)
    candidates.sort(key=lambda item: (item[1], item[0], item[2] * item[3]))
    deduped: List[List[int]] = []
    for candidate in candidates:
        if any(_intersection_ratio_xywh(candidate, existing) >= 0.70 for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _is_weak_side_background_cleanup_authority_container(container: TextAreaContainer) -> bool:
    role_evidence = _container_semantic_role_evidence(container)
    cleanup_states = set(_semantic_role_state_values(role_evidence, "cleanup_authority_states"))
    if not cleanup_states or not cleanup_states.issubset({AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}):
        return False
    evidence_text = " ".join(
        [
            str(container.confidence_tier or ""),
            " ".join(str(item) for item in container.evidence_reason_codes or []),
            " ".join(_semantic_role_values(role_evidence, "typed_authority_reason_codes")),
            " ".join(_semantic_role_values(role_evidence, "role_signals")),
            str(role_evidence.get("authority_evidence_kind") or ""),
        ]
    ).lower()
    return any(token in evidence_text for token in WEAK_SIDE_BACKGROUND_AUTHORITY_REASON_TOKENS)


def _max_inside_ratio_xyxy(bbox: Sequence[int], boxes: Sequence[Sequence[int]]) -> float:
    if not boxes:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    except Exception:
        return 0.0
    area = max(1.0, (x1 - x0) * (y1 - y0))
    best = 0.0
    for box in boxes:
        try:
            bx0, by0, bx1, by1 = [float(value) for value in box[:4]]
        except Exception:
            continue
        iw = max(0.0, min(x1, bx1) - max(x0, bx0))
        ih = max(0.0, min(y1, by1) - max(y0, by0))
        best = max(best, (iw * ih) / area)
    return best


def _max_coverage_of_boxes_xyxy(bbox: Sequence[int], boxes: Sequence[Sequence[int]]) -> float:
    if not boxes:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    except Exception:
        return 0.0
    best = 0.0
    for box in boxes:
        try:
            bx0, by0, bx1, by1 = [float(value) for value in box[:4]]
        except Exception:
            continue
        box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        iw = max(0.0, min(x1, bx1) - max(x0, bx0))
        ih = max(0.0, min(y1, by1) - max(y0, by0))
        best = max(best, (iw * ih) / box_area)
    return best


def _side_caption_signal_items(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for evidence in result.get("text_area_model_evidence", []) or []:
        bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
        if _looks_like_vertical_side_caption_signal_bbox(bbox, image_size) or _looks_like_clipped_right_side_caption_signal_item(evidence, bbox, image_size):
            items.append(evidence)
    for fused in result.get("fused_containers", []) or []:
        bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
        if _looks_like_vertical_side_caption_signal_bbox(bbox, image_size) or _looks_like_clipped_right_side_caption_signal_item(fused, bbox, image_size):
            items.append(fused)
    return items


def _side_caption_scope_authority_reason(
    source_id: str,
    candidate_bbox: Sequence[int],
    containers: Sequence[TextAreaContainer],
) -> str:
    """Allow side-caption scope expansion only from an existing semantic authority."""
    if not source_id:
        return ""
    source_id = str(source_id)
    for container in containers:
        reason_text = " ".join(str(item).lower() for item in container.evidence_reason_codes or [])
        if "side_narration_background_authority" not in reason_text:
            continue
        container_sources = {str(item) for item in container.source_model_ids or [] if str(item)}
        if source_id not in container_sources and source_id != str(container.container_id or ""):
            continue
        contains_existing = _inside_ratio_xywh(container.bbox, candidate_bbox) >= 0.65
        intersects_existing = _intersection_ratio_xywh(container.bbox, candidate_bbox) >= 0.35
        if contains_existing or intersects_existing:
            return "text_area_plan:side_narration_background_authority:scope_extension"
    return ""


def _side_caption_signal_item_has_background_authority(
    *,
    item: Mapping[str, Any],
    bbox: Sequence[int],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    luma_image: Any,
    existing_containers: Sequence[TextAreaContainer],
) -> bool:
    class_name = str(item.get("class_name") or item.get("fused_container_type") or "")
    if class_name not in {"text_free", "free_text", "text_bubble", "bubble", "caption_or_background_candidate"}:
        return False
    if _caption_search_overlaps_blocking_container(bbox, existing_containers):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    side_context = (x <= width * 0.48 and x + w <= width * 0.56) or (x >= width * 0.54 and x + w <= width * 0.99)
    if not side_context:
        return False
    if y < height * 0.15 or y + h > height * 0.84:
        return False
    if w < width * 0.025 or w > width * 0.18:
        return False
    if h < height * 0.10 or h > height * 0.44:
        return False
    area_ratio = float(visual.get("area_ratio") or 0.0)
    if area_ratio <= 0.0 or area_ratio > 0.055:
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    dark_panel_light_text = bool(
        dark is not None
        and dark >= 0.42
        and bright is not None
        and bright >= 0.025
    )
    if dark is not None and dark >= 0.48 and not dark_panel_light_text:
        return False
    if bright is not None and bright < 0.18 and not dark_panel_light_text:
        return False
    return _side_caption_column_has_text_structure(luma_image, bbox, image_size)


def _side_caption_text_structure_mask(
    arr: Any,
    *,
    dark_threshold: int,
    light_threshold: int,
) -> Any:
    dark_mask = arr <= int(dark_threshold)
    try:
        total = float(max(1, arr.size))
        mean_luma = float(arr.mean())
        bright_ratio = float((arr >= int(light_threshold)).sum()) / total
        dark_ratio = float(dark_mask.sum()) / total
    except Exception:
        return dark_mask.astype("uint8")
    if mean_luma <= 155.0 and dark_ratio >= 0.36 and bright_ratio >= 0.015:
        return (arr >= int(light_threshold)).astype("uint8")
    return dark_mask.astype("uint8")


def _side_caption_column_has_text_structure(
    luma_image: Any,
    bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None or np is None or cv2 is None:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    try:
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h))).convert("L")
        arr = np.asarray(crop)
        mask = _side_caption_text_structure_mask(arr, dark_threshold=112, light_threshold=190)
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return False
    char_like: List[Tuple[int, int, int, int, int]] = []
    large_marks = 0
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, area = [int(value) for value in stats[label][:5]]
        if area < 8:
            continue
        density = area / float(max(1, cw * ch))
        if area > 2400 or cw > max(78, int(w * 0.78)) or ch > max(96, int(h * 0.42)):
            large_marks += 1
            continue
        aspect = ch / float(max(1, cw))
        if 2 <= cw <= 76 and 4 <= ch <= 96 and 0.05 <= density <= 0.86 and 0.32 <= aspect <= 12.0:
            char_like.append((cx, cy, cw, ch, area))
    if len(char_like) < 5:
        return False
    ys = [item[1] for item in char_like]
    y2 = [item[1] + item[3] for item in char_like]
    vertical_span = max(y2) - min(ys)
    row_bins = {
        int((item[1] - min(ys)) / float(max(1, vertical_span)) * 5.0)
        for item in char_like
    }
    if len(row_bins) < 3:
        return False
    total_char_area = sum(item[4] for item in char_like)
    char_density = total_char_area / float(max(1, w * h))
    if char_density > 0.34:
        return False
    return large_marks <= max(1, len(char_like) // 4)


def _side_caption_character_column_bboxes(
    luma_image: Any,
    search_bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> List[List[int]]:
    if luma_image is None or np is None or cv2 is None:
        return []
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return []
    try:
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h))).convert("L")
        arr = np.asarray(crop)
        mask = _side_caption_text_structure_mask(arr, dark_threshold=108, light_threshold=190)
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return []
    char_components: List[Tuple[int, int, int, int, int]] = []
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, area = [int(value) for value in stats[label][:5]]
        if area < 8 or area > 2200:
            continue
        if cw < 2 or ch < 4 or cw > 80 or ch > 110:
            continue
        density = area / float(max(1, cw * ch))
        aspect = ch / float(max(1, cw))
        if 0.05 <= density <= 0.86 and 0.30 <= aspect <= 12.0:
            char_components.append((cx, cy, cw, ch, area))
    if len(char_components) < 5:
        return []
    clusters: List[Dict[str, Any]] = []
    for comp in sorted(char_components, key=lambda item: item[0] + item[2] / 2.0):
        center_x = comp[0] + comp[2] / 2.0
        placed = False
        for cluster in clusters:
            if abs(center_x - float(cluster["center_x"])) <= max(24.0, float(cluster["max_width"]) * 0.80):
                cluster["items"].append(comp)
                centers = [item[0] + item[2] / 2.0 for item in cluster["items"]]
                cluster["center_x"] = sum(centers) / float(len(centers))
                cluster["max_width"] = max(item[2] for item in cluster["items"])
                placed = True
                break
        if not placed:
            clusters.append({"center_x": center_x, "max_width": comp[2], "items": [comp]})

    candidates: List[List[int]] = []
    for cluster in clusters:
        items = list(cluster.get("items") or [])
        if len(items) < 5:
            continue
        xs = [item[0] for item in items]
        ys = [item[1] for item in items]
        x2 = [item[0] + item[2] for item in items]
        y2 = [item[1] + item[3] for item in items]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(x2), max(y2)
        bw = bx1 - bx0
        bh = by1 - by0
        if bw < max(10, int(width * 0.008)) or bw > max(120, int(width * 0.11)):
            continue
        if bh < max(150, int(height * 0.075)) or bh > int(height * 0.45):
            continue
        row_bins = {
            int((item[1] - by0) / float(max(1, bh)) * 6.0)
            for item in items
        }
        if len(row_bins) < 3:
            continue
        pixel_area = sum(item[4] for item in items)
        density = pixel_area / float(max(1, bw * bh))
        if density > 0.30:
            continue
        dominant_action_bar = any(
            item[2] >= max(46, int(bw * 0.80))
            and (item[3] / float(max(1, item[2]))) < 0.82
            for item in items
        )
        if dominant_action_bar:
            continue
        pad_x = max(5, int(bw * 0.16))
        pad_y = max(10, int(bh * 0.04))
        candidate = _normalize_xywh(
            [x + bx0 - pad_x, y + by0 - pad_y, bw + pad_x * 2, bh + pad_y * 2],
            image_size,
        )
        candidate = _clip_xywh_to_xywh(candidate, search_bbox)
        if _side_caption_column_has_text_structure(luma_image, candidate, image_size):
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item[0], item[1]))
    deduped: List[List[int]] = []
    for candidate in candidates:
        if any(_intersection_ratio_xywh(candidate, existing) >= 0.62 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= 4:
            break
    return deduped


def _clip_xywh_to_xywh(bbox: Sequence[int], bounds: Sequence[int]) -> List[int]:
    x, y, w, h = _coerce_xywh(bbox)
    bx, by, bw, bh = _coerce_xywh(bounds)
    if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
        return []
    x0 = max(x, bx)
    y0 = max(y, by)
    x1 = min(x + w, bx + bw)
    y1 = min(y + h, by + bh)
    if x1 <= x0 or y1 <= y0:
        return []
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _looks_like_vertical_side_caption_signal_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    left_side = x <= width * 0.42 and x + w <= width * 0.52
    right_side = x >= width * 0.72
    if not (left_side or right_side):
        return False
    if y < height * 0.18 or y > height * 0.74:
        return False
    min_height_ratio = 0.07 if left_side else 0.12
    if h < height * min_height_ratio:
        return False
    if w < 1 or w > width * 0.28:
        return False
    return True


def _looks_like_clipped_right_side_caption_signal_item(
    item: Mapping[str, Any],
    bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    class_name = str(item.get("class_name") or item.get("fused_container_type") or "")
    if class_name not in {"text_free", "free_text", "text_bubble", "caption_or_background_candidate"}:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if x < width * 0.94 or w > max(3, int(width * 0.004)):
        return False
    if y < height * 0.16 or y > height * 0.76:
        return False
    return h >= height * 0.10


def _side_caption_side_for_bbox(seed_bbox: Sequence[int], image_size: Tuple[int, int]) -> str:
    x, _y, w, _h = _coerce_xywh(seed_bbox)
    width = max(1, int(image_size[0]))
    center = x + max(1, w) / 2.0
    return "left" if center < width / 2.0 else "right"


def _side_caption_search_bbox(seed_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int]:
    x, y, w, h = _coerce_xywh(seed_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    side = _side_caption_side_for_bbox(seed_bbox, image_size)
    if side == "left":
        x0 = max(0, x - max(52, int(max(1, w) * 0.20)))
        x1 = min(int(width * 0.50), x + max(1, w) + max(36, int(max(1, w) * 0.10)))
        y0 = max(int(height * 0.20), y - max(220, int(max(1, h) * 1.20)))
        y1 = min(int(height * 0.78), y + max(1, h) + max(92, int(max(1, h) * 0.42)))
    else:
        if x >= int(width * 0.94) and w <= max(3, int(width * 0.004)):
            x0 = int(width * 0.72)
            x1 = min(width - 1, int(width * 0.985))
            y0 = max(int(height * 0.20), y + max(60, int(max(1, h) * 0.18)))
            y1 = min(int(height * 0.78), y + max(1, h) + max(350, int(max(1, h) * 0.85)))
        else:
            x0 = max(int(width * 0.72), x - max(72, int(max(1, w) * 0.55)))
            x1 = min(width - 1, max(x + max(1, w), int(width * 0.97)))
            y0 = max(int(height * 0.20), y - max(180, int(max(1, h) * 1.35)))
            y1 = min(int(height * 0.78), y + max(1, h) + max(90, int(max(1, h) * 0.28)))
    if x1 <= x0:
        x1 = min(width, x0 + max(80, int(width * 0.14)))
    if y1 <= y0:
        y1 = min(height, y0 + max(220, int(height * 0.25)))
    return _normalize_xywh([x0, y0, x1 - x0, y1 - y0], image_size)


def _side_caption_seed_overlaps_protected(seed_bbox: Sequence[int], containers: Sequence[TextAreaContainer]) -> bool:
    for container in containers:
        if container.container_type != CONTAINER_SFX:
            continue
        if max(_intersection_ratio_xywh(seed_bbox, container.bbox), _inside_ratio_xywh(seed_bbox, container.bbox)) >= 0.45:
            return True
    return False


def _trim_side_caption_search_bbox_against_protected(
    search_bbox: Sequence[int],
    containers: Sequence[TextAreaContainer],
    image_size: Tuple[int, int],
) -> List[int] | None:
    x, y, w, h = _coerce_xywh(search_bbox)
    if w <= 0 or h <= 0:
        return None
    y0 = y
    y1 = y + h
    for container in containers:
        if container.container_type != CONTAINER_SFX:
            continue
        bx, by, bw, bh = _coerce_xywh(container.bbox)
        if bw <= 0 or bh <= 0:
            continue
        overlap_x = max(0, min(x + w, bx + bw) - max(x, bx))
        if overlap_x <= max(8, int(min(w, bw) * 0.15)):
            continue
        bottom = by + bh
        if by < y1 and bottom > y0:
            y0 = max(y0, bottom + max(8, int(bh * 0.035)))
    if y1 - y0 < max(120, int(max(1, image_size[1]) * 0.06)):
        return None
    return _normalize_xywh([x, y0, w, y1 - y0], image_size)


def _localize_side_caption_bbox(luma_image: Any, search_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int] | None:
    if luma_image is None:
        return None
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return None
    try:
        crop = luma_image.crop((x, y, min(width, x + w), min(height, y + h))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return None
        pixels = crop.load()
        col_counts: List[int] = []
        for cx in range(cw):
            count = 0
            for cy in range(ch):
                if pixels[cx, cy] <= 105:
                    count += 1
            col_counts.append(count)
        threshold = max(4, int(ch * 0.018))
        active_cols = [idx for idx, count in enumerate(col_counts) if count >= threshold]
        if not active_cols:
            return None
        # Group nearby dark columns; side captions are narrow but may have several
        # vertical text columns. Ignore very wide SFX/art bands.
        groups: List[Tuple[int, int]] = []
        start = prev = active_cols[0]
        max_gap = max(8, int(cw * 0.055))
        for idx in active_cols[1:]:
            if idx - prev <= max_gap:
                prev = idx
                continue
            groups.append((start, prev))
            start = prev = idx
        groups.append((start, prev))
        valid: List[Tuple[int, int, int]] = []
        for gx0, gx1 in groups:
            gw = gx1 - gx0 + 1
            if gw < 12 or gw > max(150, int(cw * 0.72)):
                continue
            band_dark = sum(col_counts[gx0 : gx1 + 1])
            if band_dark < max(80, int(gw * ch * 0.018)):
                continue
            valid.append((band_dark, gx0, gx1))
        if not valid:
            return None
        _score, gx0, gx1 = max(valid, key=lambda item: item[0])
        pad_x = max(10, int((gx1 - gx0 + 1) * 0.22))
        lx0 = max(0, gx0 - pad_x)
        lx1 = min(cw, gx1 + 1 + pad_x)
        row_counts: List[int] = []
        for cy in range(ch):
            count = 0
            for cx in range(lx0, lx1):
                if pixels[cx, cy] <= 118:
                    count += 1
            row_counts.append(count)
        row_threshold = max(2, int((lx1 - lx0) * 0.045))
        active_rows = [idx for idx, count in enumerate(row_counts) if count >= row_threshold]
        if not active_rows:
            return None
        ly0 = max(0, min(active_rows) - max(12, int(ch * 0.03)))
        ly1 = min(ch, max(active_rows) + 1 + max(12, int(ch * 0.03)))
        localized = [x + lx0, y + ly0, max(1, lx1 - lx0), max(1, ly1 - ly0)]
        if localized[2] < 28 or localized[3] < 160:
            return None
        return _normalize_xywh(localized, image_size)
    except Exception:
        return None


def _has_top_band_caption_search_signal(result: Mapping[str, Any], image_size: Tuple[int, int]) -> bool:
    for item in _top_band_model_text_items(result, image_size):
        class_name = str(item.get("class_name") or item.get("fused_container_type") or "")
        if class_name in {"text_free", "free_text", "caption_or_background_candidate"}:
            return True
    return False


def _top_band_caption_search_source_ids(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[str]:
    ids: List[str] = []
    for item in _top_band_model_text_items(result, image_size):
        item_id = str(item.get("model_evidence_id") or item.get("evidence_id") or item.get("fused_container_id") or "")
        if item_id and item_id not in ids:
            ids.append(item_id)
    return ids


def _top_band_model_text_items(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for evidence in result.get("text_area_model_evidence", []) or []:
        bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
        if _looks_like_top_caption_bbox(bbox, image_size):
            items.append(evidence)
    for fused in result.get("fused_containers", []) or []:
        bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
        if _looks_like_top_caption_bbox(bbox, image_size):
            items.append(fused)
    return items


def _caption_search_overlaps_blocking_container(bbox: Sequence[Any], containers: Sequence[TextAreaContainer]) -> bool:
    candidate = _xywh_to_xyxy(bbox)
    if not candidate:
        return True
    candidate_area = max(1.0, _area(candidate))
    for container in containers:
        if container.container_type not in {CONTAINER_SPEECH, CONTAINER_SFX}:
            continue
        cbox = _xywh_to_xyxy(container.bbox)
        if not cbox:
            continue
        if _intersection_area(candidate, cbox) / candidate_area >= 0.18:
            return True
    return False


def _intersection_ratio_xywh(a: Sequence[Any], b: Sequence[Any]) -> float:
    abox = _xywh_to_xyxy(a)
    bbox = _xywh_to_xyxy(b)
    if not abox or not bbox:
        return 0.0
    inter = _intersection_area(abox, bbox)
    if inter <= 0:
        return 0.0
    return inter / max(1.0, min(_area(abox), _area(bbox)))


def _inside_ratio_xywh(inner: Sequence[Any], outer: Sequence[Any]) -> float:
    ibox = _xywh_to_xyxy(inner)
    obox = _xywh_to_xyxy(outer)
    if not ibox or not obox:
        return 0.0
    return _intersection_area(ibox, obox) / max(1.0, _area(ibox))


def _blocked_full_page_review_container(page_id: str, image_size: Tuple[int, int], reason: str) -> TextAreaContainer:
    width, height = int(image_size[0]), int(image_size[1])
    return TextAreaContainer(
        container_id="blocked_full_page_review",
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=[0, 0, max(1, width), max(1, height)],
        confidence="fallback",
        confidence_tier="compatibility_fallback",
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=reason,
        evidence_reason_codes=[
            "text_area_plan:no_usable_text_area_containers",
            "text_area_plan:blocked_no_full_page_ctd",
        ],
        human_review_required=True,
        ocr_eligibility_reason="blocked_no_usable_text_area_containers",
        text_area_pre_ocr_authority=True,
    )


def _fallback_assignment(reason: str) -> Dict[str, Any]:
    return {
        "text_area_container_id": None,
        "text_area_semantic_unit_id": None,
        "text_area_semantic_kind": SEMANTIC_KIND_UNKNOWN,
        "text_area_container_type": CONTAINER_UNKNOWN,
        "text_area_route_intent": ROUTE_REVIEW_FALLBACK,
        "text_area_cleanup_authorization": AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
        "text_area_must_not_mutate": True,
        "text_area_protection_reason": "compatibility_fallback_not_cleanup_authorized",
        "text_area_authorization_source_stage": "compatibility_fallback",
        "text_area_authorization_basis": "compatibility_fallback_not_cleanup_authorized",
        "text_area_authorization_explicit": False,
        "text_area_authorization_field_origin": "compatibility_fallback",
        "text_area_semantic_authorization_state": AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
        "text_area_ctd_scope_eligible": False,
        "text_area_comic_text_detector_scope_eligible": False,
        "text_area_ocr_eligible": True,
        "text_area_translation_eligible": False,
        "text_area_render_eligible": False,
        "text_area_cleanup_executable": False,
        "text_area_detection_source": DETECTION_COMPATIBILITY_FALLBACK,
        "text_area_fallback_reason": reason,
        "text_area_confidence_tier": "compatibility_fallback",
        "text_area_reason_codes": ["text_area_plan:compatibility_full_page_fallback"],
        "text_area_conflict_flags": [],
        "text_area_pre_ocr_authority": False,
        "text_area_enriched_from_region": False,
        "text_area_ocr_eligibility_reason": "compatibility_fallback_requires_legacy_ocr",
        "text_area_overlap_ratio": 0.0,
    }


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict"):
        try:
            return dict(result.to_dict())
        except Exception:
            return {}
    return {}


def _plan_containers(plan: TextAreaPlan | Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    if plan is None:
        return []
    if isinstance(plan, TextAreaPlan):
        return [container.to_dict() for container in plan.containers]
    return list(plan.get("containers") or [])


def _confidence_tier_from_fused(fused: Mapping[str, Any]) -> str:
    confidence = str(fused.get("confidence") or "")
    has_kitsumed = bool(fused.get("linked_kitsumed_mask_ids"))
    has_ogkalu = bool(fused.get("linked_ogkalu_detection_ids"))
    if confidence == "high" and has_kitsumed and has_ogkalu:
        return "strong_model_container"
    if has_kitsumed:
        return "mask_primary_container"
    if has_ogkalu:
        return "text_bubble_review_container"
    return confidence or "low"


def _semantic_role_evidence_from_fused(fused: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = dict(fused.get("semantic_role_evidence") or {})
    role_signals = set(_semantic_role_values(evidence, "role_signals"))
    ogkalu_classes = set(_semantic_role_values(evidence, "ogkalu_class_names"))
    fused_type = str(fused.get("fused_container_type") or "")
    reasons = [str(item) for item in fused.get("reason_codes") or [] if str(item)]
    if fused.get("linked_kitsumed_mask_ids"):
        role_signals.add("speech_bubble_mask_evidence")
    if "speech_bubble" in fused_type:
        role_signals.add("speech_candidate")
    if fused_type in {"caption_or_background_candidate", "free_text"}:
        role_signals.add("caption_background_candidate")
    if fused_type == "sfx_or_decorative_candidate":
        role_signals.add("sfx_decorative_candidate")
    for reason in reasons:
        if reason == OGKALU_BUBBLE_TEXT_PAIR_REASON:
            role_signals.add("ogkalu_bubble_text_pair")
        if "ogkalu_text_free" in reason:
            ogkalu_classes.add("text_free")
        if "ogkalu_text_bubble" in reason:
            ogkalu_classes.add("text_bubble")
        if "ogkalu_bubble" in reason:
            ogkalu_classes.add("bubble")
    evidence["role_signals"] = sorted(role_signals)
    evidence["ogkalu_class_names"] = sorted(ogkalu_classes)
    return evidence


def _semantic_role_evidence_from_text_area_evidence(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    class_name = str(evidence.get("class_name") or "")
    base = dict(evidence.get("semantic_role_evidence") or {}) if isinstance(evidence.get("semantic_role_evidence"), Mapping) else {}
    role_signals: List[str] = list(_semantic_role_values(base, "role_signals"))
    cleanup_candidate_states: List[str] = list(_semantic_role_state_values(base, "cleanup_candidate_states"))
    cleanup_authority_states: List[str] = list(_semantic_role_state_values(base, "cleanup_authority_states"))
    protected_authority_states: List[str] = list(_semantic_role_state_values(base, "protected_authority_states"))
    protected_candidate_states: List[str] = list(_semantic_role_state_values(base, "protected_candidate_states"))
    evidence_id = str(evidence.get("model_evidence_id") or evidence.get("evidence_id") or "")
    confidence = _optional_float(evidence.get("confidence"))
    normalized_kind = str(base.get("normalized_semantic_candidate_kind") or "review_unknown")
    candidate_authority_state = str(base.get("candidate_authority_state") or AUTH_REVIEW_UNKNOWN_NOT_CLEANUP)
    reason_codes = list(_semantic_role_values(base, "reason_codes"))
    if class_name in {"bubble", "text_bubble"}:
        role_signals.append("text_container_candidate")
        cleanup_candidate_states.append(AUTH_CLEANUP_TRANSLATE_SPEECH)
        normalized_kind = "speech_container" if class_name == "bubble" else "speech_text_area"
        candidate_authority_state = AUTH_CLEANUP_TRANSLATE_SPEECH
        reason_codes.append(
            "ogkalu_bubble_speech_container_evidence"
            if class_name == "bubble"
            else "ogkalu_text_bubble_first_class_speech_evidence"
        )
    if class_name == "text_free":
        role_signals.append("caption_background_candidate")
        cleanup_candidate_states.append(AUTH_CLEANUP_TRANSLATE_BACKGROUND)
        normalized_kind = "background_narration"
        candidate_authority_state = AUTH_CLEANUP_TRANSLATE_BACKGROUND
        reason_codes.append("ogkalu_text_free_background_evidence")
    if evidence.get("linked_bubble_mask_ids"):
        role_signals.append("speech_bubble_mask_evidence")
        cleanup_candidate_states = [state for state in cleanup_candidate_states if state != AUTH_CLEANUP_TRANSLATE_SPEECH]
        cleanup_authority_states = [AUTH_CLEANUP_TRANSLATE_SPEECH]
    ogkalu_classes = set(_semantic_role_values(base, "ogkalu_class_names"))
    if class_name:
        ogkalu_classes.add(class_name)
    source_model_ids = set(_semantic_role_values(base, "source_model_ids"))
    model_evidence_ids = set(_semantic_role_values(base, "model_evidence_ids"))
    if evidence_id:
        source_model_ids.add(evidence_id)
        model_evidence_ids.add(evidence_id)
    evidence_strength = str(base.get("evidence_strength") or "")
    if not evidence_strength:
        evidence_strength = (
            "strong_single_model_candidate"
            if confidence is not None
            and confidence >= OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE
            and class_name in {"bubble", "text_bubble", "text_free"}
            else "single_model_candidate"
        )
    return {
        **base,
        "role_signals": sorted(set(role_signals)),
        "ogkalu_class_names": sorted(ogkalu_classes),
        "cleanup_authority_states": sorted(set(cleanup_authority_states)),
        "cleanup_candidate_states": sorted(set(cleanup_candidate_states)),
        "protected_authority_states": sorted(set(protected_authority_states)),
        "protected_candidate_states": sorted(set(protected_candidate_states)),
        "authority_evidence_kind": str(base.get("authority_evidence_kind") or "typed_text_area_model_evidence"),
        "source": "bubble_detection_text_area_evidence",
        "raw_class_name": str(base.get("raw_class_name") or class_name),
        "normalized_semantic_candidate_kind": normalized_kind,
        "candidate_authority_state": candidate_authority_state,
        "evidence_strength": evidence_strength,
        "reason_codes": sorted({str(item) for item in reason_codes if str(item)}),
        "source_evidence_ids": sorted(source_model_ids),
        "source_model_ids": sorted(source_model_ids),
        "model_evidence_ids": sorted(model_evidence_ids),
        "confidence": evidence.get("confidence", base.get("confidence")),
    }


def _has_preserve_conflict(conflicts: Sequence[Any], reasons: Sequence[Any]) -> bool:
    text = " ".join(str(item).lower() for item in list(conflicts) + list(reasons))
    return any(token in text for token in ("sfx", "decorative", "preserve"))


def _evidence_is_linked_to_fused_container(evidence_id: str, result: Mapping[str, Any]) -> bool:
    for fused in result.get("fused_containers", []) or []:
        if evidence_id in [str(item) for item in fused.get("linked_ogkalu_detection_ids", []) or []]:
            return True
    return False


def _looks_like_top_caption_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    _x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    return y <= height * 0.24 and h >= height * 0.06 and w <= width * 0.32


def _looks_like_caption_background_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    top_band = y <= height * 0.24
    narrow_text_column = width * 0.025 <= w <= width * 0.18
    caption_height = height * 0.04 <= h <= height * 0.16
    # Cover/title/logo art can be detected as text_free, but those tall,
    # page-edge columns are not safe caption/background OCR scopes.
    return top_band and narrow_text_column and caption_height and x < width * 0.96


def _looks_like_high_confidence_top_text_free_background_bbox(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
    model_confidence: float | None,
) -> bool:
    if fused_type not in {"free_text", "caption_or_background_candidate"}:
        return False
    if not _looks_like_top_caption_bbox(bbox, image_size):
        return False
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if "ogkalu_text_free" not in reason_text:
        return False
    if model_confidence is None or model_confidence < OGKALU_TEXT_FREE_BACKGROUND_ROOT_CONFIDENCE:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    if x + w >= width - max(2, int(width * 0.002)):
        return False
    # This admits only typed model text_free columns; deterministic visual
    # top-band searches remain non-authoritative root candidates.
    return bool(
        y <= height * 0.12
        and width * 0.045 <= w <= width * 0.12
        and height * 0.12 <= h <= height * 0.24
    )


def _looks_like_side_narration_background_bbox(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "preserve")):
        return False
    if fused_type not in {"ambiguous", "caption_or_background_candidate", "free_text"}:
        return False
    role_classes = set(_semantic_role_values(semantic_role_evidence, "ogkalu_class_names"))
    if not ({"bubble", "text_bubble", "text_free"} & role_classes or "ogkalu_bubble" in reason_text or "ogkalu_text_bubble" in reason_text):
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    left_side = x <= width * 0.42 and (x + w) <= width * 0.48
    right_side = x >= width * 0.56 and x + w <= width * 0.98
    vertical_context = y >= height * 0.28 and h >= height * 0.07
    text_block_size = w >= width * 0.08 and h >= height * 0.06
    not_edge_stroke = not (w <= width * 0.06 and h >= w * 2.0)
    area_ratio = float(visual.get("area_ratio") or 0.0)
    return bool((left_side or right_side) and vertical_context and text_block_size and not_edge_stroke and area_ratio <= 0.08)


def _load_luma_image(image_path: str | Path) -> Any:
    try:
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            return None
        return Image.open(path).convert("L")
    except Exception:
        return None


def _container_visual_stats(luma_image: Any, bbox: Sequence[Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    area = max(0, w) * max(0, h)
    stats: Dict[str, Any] = {
        "area_px": area,
        "area_ratio": area / max(1, width * height),
        "mean_luma": None,
        "bright_ratio": None,
        "dark_ratio": None,
        "clipped_or_degenerate": _is_clipped_or_degenerate_bbox(bbox, image_size),
    }
    if luma_image is None or area <= 0:
        return stats
    try:
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + max(0, w)), min(height, y + max(0, h))))
        pixels = list(crop.getdata())
        if not pixels:
            return stats
        total = float(len(pixels))
        stats["mean_luma"] = round(sum(pixels) / total, 4)
        stats["bright_ratio"] = round(sum(1 for value in pixels if value >= 220) / total, 6)
        stats["dark_ratio"] = round(sum(1 for value in pixels if value <= 90) / total, 6)
    except Exception:
        pass
    return stats


def _looks_like_pre_ocr_sfx_or_decorative(
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "preserve")):
        return True
    if fused_type != "free_text":
        return False
    if _looks_like_caption_background_bbox(bbox, image_size):
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    area_ratio = float(visual.get("area_ratio") or 0.0)
    # Non-caption text_free is not speech evidence. Dark/non-uniform regions are
    # treated as art/SFX review and blocked before normal OCR/translation.
    if dark is not None and dark >= 0.16:
        return True
    if bright is not None and bright < 0.62:
        return True
    if area_ratio < 0.003:
        return True
    return False


def _looks_like_dark_unlinked_ogkalu_sfx_or_decorative(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    if fused_type != "ambiguous":
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if not any(token in reason_text for token in ("ogkalu_bubble_without_kitsumed_mask", "ogkalu_text_bubble_without_kitsumed_mask")):
        return False
    role_signals = set(_semantic_role_values(semantic_role_evidence, "role_signals"))
    if "speech_bubble_mask_evidence" in role_signals:
        return False
    role_classes = set(_semantic_role_values(semantic_role_evidence, "ogkalu_class_names"))
    if not (role_classes & {"bubble", "text_bubble"}):
        return False
    conflict_evidence = set(_semantic_role_values(semantic_role_evidence, "conflict_evidence"))
    has_typed_sfx_conflict = bool(
        _has_recognized_sfx_or_decorative_evidence(semantic_role_evidence)
        or any(
            any(token in str(conflict).lower() for token in ("sfx", "decorative", "art", "non_text", "non-text"))
            for conflict in conflict_evidence
        )
    )
    if not has_typed_sfx_conflict:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    left_side = x <= width * 0.42 and (x + w) <= width * 0.48
    right_side = x >= width * 0.56 and x + w <= width * 0.98
    vertical_context = y >= height * 0.28 and h >= height * 0.07
    text_block_size = w >= width * 0.08 and h >= height * 0.06
    not_edge_stroke = not (w <= width * 0.06 and h >= w * 2.0)
    if (left_side or right_side) and vertical_context and text_block_size and not_edge_stroke:
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    area_ratio = float(visual.get("area_ratio") or 0.0)
    if area_ratio <= 0.0 or area_ratio > 0.045:
        return False
    dark_art_context = dark is not None and dark >= 0.16
    non_bright_context = bright is not None and bright < 0.62
    return bool(dark_art_context or non_bright_context)


def _looks_like_bright_unlinked_text_free_sfx_or_decorative(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    if fused_type != "free_text":
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if not any(token in reason_text for token in ("ogkalu_text_free_without_kitsumed_mask", "text_free")):
        return False
    role_classes = set(_semantic_role_values(semantic_role_evidence, "ogkalu_class_names"))
    if role_classes and role_classes != {"text_free"}:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    # Bright lower-panel unlinked free-text boxes are often stylized SFX/action
    # marks on artwork, not side narration. Keep this upstream and typed so
    # CleanupMask remains a strict consumer.
    lower_action_zone = y >= height * 0.45
    side_or_body_text_scale = w >= width * 0.10 and h >= height * 0.08
    not_page_edge_caption = x + w < width - max(2, int(width * 0.002))
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    bright_art_context = bright is not None and bright >= 0.68 and dark is not None and dark <= 0.12
    return bool(lower_action_zone and side_or_body_text_scale and not_page_edge_caption and bright_art_context)


def _text_free_caption_candidate_is_protected_sfx(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    luma_image: Any = None,
) -> bool:
    if fused_type != "free_text":
        return False
    if not _looks_like_caption_background_bbox(bbox, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, _height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "preserve")):
        return True
    edge_attached = x + w >= width - max(2, int(width * 0.002))
    very_narrow = w <= width * 0.075
    tall_column = h >= w * 2.0
    dark = _optional_float(visual.get("dark_ratio"))
    bright = _optional_float(visual.get("bright_ratio"))
    mixed_art_context = dark is not None and dark >= 0.20 and bright is not None and bright < 0.55
    return bool(edge_attached and very_narrow and tall_column and mixed_art_context)


def _text_free_caption_candidate_has_background_authority(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    luma_image: Any = None,
    model_confidence: float | None = None,
) -> bool:
    if fused_type not in {"free_text", "caption_or_background_candidate"}:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    typed_text_free_model = "ogkalu_text_free" in reason_text
    model_confident_text_free = (
        typed_text_free_model
        and model_confidence is not None
        and model_confidence >= OGKALU_TEXT_FREE_BACKGROUND_ROOT_CONFIDENCE
    )
    high_confidence_top_text_free_background = _looks_like_high_confidence_top_text_free_background_bbox(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        image_size=image_size,
        model_confidence=model_confidence,
    )
    if not (_looks_like_caption_background_bbox(bbox, image_size) or high_confidence_top_text_free_background):
        return False
    if _text_free_caption_candidate_is_protected_sfx(
        fused_type=fused_type,
        reasons=reasons,
        bbox=bbox,
        visual=visual,
        image_size=image_size,
        luma_image=luma_image,
    ):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, _height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    # The upstream model has emitted a top-band text_free/caption box, but the
    # final authority still needs a coherent localized text unit.
    wide_enough_for_caption = w >= width * 0.070
    not_page_edge_art = x + w < width - max(2, int(width * 0.002))
    tall_top_band_model_text = (
        model_confident_text_free
        and not_page_edge_art
        and y <= int(image_size[1]) * 0.12
        and width * 0.045 <= w <= width * 0.12
        and int(image_size[1]) * 0.12 <= h <= int(image_size[1]) * 0.24
    )
    if tall_top_band_model_text:
        if luma_image is not None:
            if (
                _luma_bbox_has_text_like_dark_components(luma_image, bbox, image_size)
                or _luma_bbox_has_sparse_vertical_text_components(luma_image, bbox, image_size)
                or _has_coherent_caption_text_column(luma_image, bbox, image_size)
            ):
                return True
        else:
            dark = _optional_float(visual.get("dark_ratio"))
            bright = _optional_float(visual.get("bright_ratio"))
            if dark is not None and dark >= 0.22 and bright is not None and bright <= 0.62:
                return True
    if (
        luma_image is not None
        and model_confident_text_free
        and not_page_edge_art
        and y <= int(image_size[1]) * 0.12
        and width * 0.025 <= w <= width * 0.11
        and int(image_size[1]) * 0.025 <= h <= int(image_size[1]) * 0.12
        and (
            _luma_bbox_has_text_like_dark_components(luma_image, bbox, image_size)
            or _luma_bbox_has_sparse_vertical_text_components(luma_image, bbox, image_size)
            or _has_compact_top_title_text(luma_image, bbox, image_size)
        )
    ):
        return True
    if (
        luma_image is not None
        and typed_text_free_model
        and not_page_edge_art
        and y <= int(image_size[1]) * 0.24
        and width * 0.030 <= w <= width * 0.095
        and int(image_size[1]) * 0.035 <= h <= int(image_size[1]) * 0.095
        and _has_compact_top_title_text(luma_image, bbox, image_size)
    ):
        return True
    compact_top_text_free = (
        typed_text_free_model
        and not_page_edge_art
        and y <= int(image_size[1]) * 0.08
        and width * 0.030 <= w <= width * 0.095
        and int(image_size[1]) * 0.030 <= h <= int(image_size[1]) * 0.075
    )
    if compact_top_text_free:
        dark = _optional_float(visual.get("dark_ratio"))
        bright = _optional_float(visual.get("bright_ratio"))
        textured_text_column = dark is not None and dark >= 0.32 and bright is not None and bright >= 0.24
        if textured_text_column:
            return True
    if luma_image is None:
        dark = _optional_float(visual.get("dark_ratio"))
        bright = _optional_float(visual.get("bright_ratio"))
        background_contrast = dark is not None and dark >= 0.45 and bright is not None and bright <= 0.35
        return bool(wide_enough_for_caption and not_page_edge_art and typed_text_free_model and background_contrast)
    return bool(
        wide_enough_for_caption
        and not_page_edge_art
        and _has_coherent_caption_text_column(luma_image, bbox, image_size)
        and _top_band_right_column_has_background_context(luma_image, bbox, image_size)
    )


def _has_explicit_text_free_background_model_boundary(
    role_evidence: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    reason_text = " ".join(
        [str(role_evidence.get("authority_evidence_kind") or "")]
        + [str(item) for item in _semantic_role_values(role_evidence, "typed_authority_reason_codes")]
    )
    if "typed_text_free_background_model_authority" not in reason_text:
        return False
    for field_name in ("text_unit_evidence_bboxes", "model_evidence_bboxes"):
        entries = role_evidence.get(field_name) or []
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("class_name") or "") != "text_free":
                continue
            confidence = _optional_float(entry.get("confidence"))
            if confidence is None or confidence < OGKALU_TEXT_FREE_BACKGROUND_ROOT_CONFIDENCE:
                continue
            x, y, w, h = _coerce_xywh(_text_evidence_bbox_xywh(entry, image_size))
            width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
            if w <= 0 or h <= 0:
                continue
            if w > width * 0.18 or h > height * 0.18:
                continue
            return True
    return False


def _has_coherent_caption_text_column(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None:
        return False
    return bool(_top_band_character_column_bboxes(luma_image, bbox, image_size))


def _expand_compact_top_text_free_background_bbox(
    *,
    luma_image: Any,
    bbox: Sequence[Any],
    reasons: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> List[int]:
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    typed_text_free_model = "ogkalu_text_free" in " ".join(str(item).lower() for item in reasons)
    dark = _optional_float(visual.get("dark_ratio"))
    bright = _optional_float(visual.get("bright_ratio"))
    if (
        luma_image is None
        or not typed_text_free_model
        or w <= 0
        or h <= 0
        or y > height * 0.08
        or not (width * 0.030 <= w <= width * 0.095)
        or not (height * 0.030 <= h <= height * 0.075)
        or dark is None
        or dark < 0.32
        or bright is None
        or bright < 0.24
    ):
        return _normalize_xywh(bbox, image_size)
    search_pad_left = max(20, int(width * 0.030))
    search_pad_right = max(150, int(width * 0.105))
    search_pad_y = max(48, int(height * 0.025))
    sx0 = max(0, x - search_pad_left)
    sx1 = min(width, x + w + search_pad_right)
    search = [
        sx0,
        max(0, y - search_pad_y),
        sx1 - sx0,
        min(height, y + h + search_pad_y) - max(0, y - search_pad_y),
    ]
    columns = _top_band_character_column_bboxes(luma_image, search, image_size)
    selected: List[List[int]] = []
    for column in columns:
        cx, cy, cw, ch = _coerce_xywh(column)
        if cw <= 0 or ch <= 0:
            continue
        vertical_overlap = _intersection_ratio_xywh([cx, cy, cw, ch], [0, y - search_pad_y, width, h + 2 * search_pad_y])
        if vertical_overlap <= 0:
            continue
        if cx + cw < x - max(12, int(width * 0.012)) or cx > x + w + search_pad_right:
            continue
        selected.append([cx, cy, cw, ch])
    if not selected:
        return _normalize_xywh(bbox, image_size)
    boxes = selected + [_normalize_xywh(bbox, image_size)]
    x0 = min(item[0] for item in boxes)
    y0 = min(item[1] for item in boxes)
    x1 = max(item[0] + item[2] for item in boxes)
    y1 = max(item[1] + item[3] for item in boxes)
    return _normalize_xywh([x0, y0, x1 - x0, y1 - y0], image_size)


def _has_compact_top_title_text(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None or np is None or cv2 is None:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    if y > height * 0.24 or w > width * 0.095 or h > height * 0.095:
        return False
    try:
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h))).convert("L")
        arr = np.asarray(crop)
        mask = (arr <= 104).astype("uint8")
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return False
    char_like = 0
    large_marks = 0
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, area = [int(value) for value in stats[label][:5]]
        del cx, cy
        if area < 10:
            continue
        if area > 2200 or cw > max(56, w * 0.72) or ch > max(70, h * 0.92):
            large_marks += 1
            continue
        density = area / float(max(1, cw * ch))
        if 0.08 <= density <= 0.82 and 2 <= cw <= 48 and 4 <= ch <= 64:
            char_like += 1
    return char_like >= 4 and large_marks == 0


def _top_band_right_column_has_background_context(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    _x, _y, _w, h = _coerce_xywh(bbox)
    _width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if h < height * 0.16:
        return False
    stats = _container_visual_stats(luma_image, bbox, image_size)
    bright = _optional_float(stats.get("bright_ratio"))
    dark = _optional_float(stats.get("dark_ratio"))
    if bright is None or dark is None:
        return False
    # Right-edge top-band action marks often form coherent-looking columns on
    # bright panel backgrounds. Require darker caption/narration context before
    # turning those right-band columns into cleanup authority.
    return bool(bright <= 0.49 or dark >= 0.36)


def _text_evidence_bbox_xywh(entry: Mapping[str, Any], image_size: Tuple[int, int]) -> List[int]:
    bbox = entry.get("bbox") or []
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [0, 0, 0, 0]
    try:
        x0, y0, third, fourth = [float(value) for value in list(bbox)[:4]]
    except Exception:
        return [0, 0, 0, 0]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if third > x0 and fourth > y0 and third <= width and fourth <= height:
        return _bbox_xyxy_to_xywh([x0, y0, third, fourth], image_size)
    return _normalize_xywh([x0, y0, third, fourth], image_size)


def _paired_text_boundary_bboxes_from_role_evidence(
    role_evidence: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    allowed_classes: Sequence[str] = ("text_bubble",),
) -> List[List[int]]:
    bboxes: List[List[int]] = []
    seen_ids: set[str] = set()
    accepted_classes = {str(value) for value in allowed_classes if str(value)}
    for field_name in ("text_unit_evidence_bboxes", "model_evidence_bboxes"):
        entries = role_evidence.get(field_name) or []
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("class_name") or "") not in accepted_classes:
                continue
            confidence = _optional_float(entry.get("confidence"))
            if confidence is not None and confidence < OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE:
                continue
            bbox = _text_evidence_bbox_xywh(entry, image_size)
            x, y, w, h = _coerce_xywh(bbox)
            if w <= 0 or h <= 0:
                continue
            evidence_id = str(entry.get("evidence_id") or "")
            if evidence_id and evidence_id in seen_ids:
                continue
            if any(_intersection_ratio_xywh(bbox, existing) >= 0.86 for existing in bboxes):
                continue
            if evidence_id:
                seen_ids.add(evidence_id)
            bboxes.append([x, y, w, h])
    return bboxes


def _padded_text_boundary_union_bbox(
    text_bboxes: Sequence[Sequence[int]],
    support_bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> List[int]:
    union = _text_area_graph_union_xywh(text_bboxes)
    if not union:
        return []
    x, y, w, h = _coerce_xywh(union)
    if w <= 0 or h <= 0:
        return []
    pad_x = max(6, int(round(w * 0.10)))
    pad_y = max(6, int(round(h * 0.10)))
    padded = _normalize_xywh([x - pad_x, y - pad_y, w + pad_x * 2, h + pad_y * 2], image_size)
    support = _normalize_xywh(support_bbox, image_size)
    clipped = _clip_xywh_to_xywh(padded, support) if support else []
    return clipped or padded


def _paired_text_boundary_root_bbox(
    role_evidence: Mapping[str, Any],
    support_bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> List[int]:
    text_bboxes = _paired_text_boundary_bboxes_from_role_evidence(role_evidence, image_size)
    if not text_bboxes:
        return []
    support = _normalize_xywh(support_bbox, image_size)
    if support:
        owned = [
            bbox
            for bbox in text_bboxes
            if _inside_ratio_xywh(bbox, support) >= 0.65
            or _intersection_ratio_xywh(bbox, support) >= 0.65
        ]
    else:
        owned = list(text_bboxes)
    if not owned:
        return []
    return _padded_text_boundary_union_bbox(owned, support or _text_area_graph_union_xywh(owned), image_size)


def _side_caption_paired_text_boundary_bboxes(
    result: Mapping[str, Any],
    item: Mapping[str, Any],
    search_bbox: Sequence[int],
    image_size: Tuple[int, int],
) -> List[List[int]]:
    item_id = str(item.get("model_evidence_id") or item.get("evidence_id") or item.get("fused_container_id") or "")
    if not item_id:
        return []
    raw_bboxes: List[List[int]] = []

    def collect(role_evidence: Any) -> None:
        if not isinstance(role_evidence, Mapping):
            return
        for bbox in _paired_text_boundary_bboxes_from_role_evidence(
            role_evidence,
            image_size,
            allowed_classes=("text_free", "text_bubble"),
        ):
            if _inside_ratio_xywh(bbox, search_bbox) < 0.65 and _intersection_ratio_xywh(bbox, search_bbox) < 0.65:
                continue
            if any(_intersection_ratio_xywh(bbox, existing) >= 0.86 for existing in raw_bboxes):
                continue
            raw_bboxes.append(bbox)

    collect(item.get("semantic_role_evidence") or {})
    for fused in result.get("fused_containers", []) or []:
        if not isinstance(fused, Mapping):
            continue
        fused_id = str(fused.get("fused_container_id") or "")
        linked_ids = {str(value) for value in fused.get("linked_ogkalu_detection_ids") or [] if str(value)}
        linked_ids.update(str(value) for value in fused.get("source_model_ids") or [] if str(value))
        if item_id != fused_id and item_id not in linked_ids:
            continue
        collect(fused.get("semantic_role_evidence") or {})

    padded: List[List[int]] = []
    for bbox in raw_bboxes:
        candidate = _padded_text_boundary_union_bbox([bbox], search_bbox, image_size)
        if not candidate:
            continue
        if any(_intersection_ratio_xywh(candidate, existing) >= 0.86 for existing in padded):
            continue
        padded.append(candidate)
    return padded


def _luma_bbox_has_text_like_dark_components(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None:
        return False
    x, y, w, h = _normalize_xywh(bbox, image_size)
    if w < 8 or h < 12:
        return False
    try:
        crop = luma_image.crop((x, y, x + w, y + h)).convert("L")
    except Exception:
        return False
    if np is None:
        pixels = list(crop.getdata())
        if not pixels:
            return False
        dark_count = sum(1 for value in pixels if value <= 120)
        area = max(1, len(pixels))
        return bool(max(16, int(area * 0.003)) <= dark_count <= int(area * 0.35))
    arr = np.asarray(crop)
    if arr.size <= 0:
        return False
    dark_mask = (arr <= 120).astype("uint8")
    dark_count = int(dark_mask.sum())
    area = int(arr.size)
    dark_ratio = dark_count / float(max(1, area))
    if dark_ratio < 0.20 or dark_ratio > 0.55:
        return False
    if cv2 is None:
        return True
    try:
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    except Exception:
        return True
    text_like = 0
    oversized = 0
    largest_component_area = 0
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, component_area = [int(value) for value in stats[label][:5]]
        del cx, cy
        if component_area < 8:
            continue
        largest_component_area = max(largest_component_area, component_area)
        density = component_area / float(max(1, cw * ch))
        if cw > max(72, int(w * 0.82)) or ch > max(120, int(h * 0.95)):
            oversized += 1
            continue
        if 0.06 <= density <= 1.0 and 2 <= cw <= max(72, int(w * 0.72)) and 4 <= ch <= max(120, int(h * 0.92)):
            text_like += 1
    if largest_component_area / float(max(1, dark_count)) > 0.55:
        return False
    return bool(text_like >= 3 and oversized <= 1)


def _luma_bbox_has_sparse_vertical_text_components(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None or np is None or cv2 is None:
        return False
    x, y, w, h = _normalize_xywh(bbox, image_size)
    if w < 16 or h < 40:
        return False
    try:
        crop = luma_image.crop((x, y, x + w, y + h)).convert("L")
    except Exception:
        return False
    arr = np.asarray(crop)
    if arr.size <= 0:
        return False
    dark_mask = (arr <= 120).astype("uint8")
    dark_count = int(dark_mask.sum())
    area = int(arr.size)
    dark_ratio = dark_count / float(max(1, area))
    if dark_ratio < 0.025 or dark_ratio > 0.20:
        return False
    bright_ratio = float((arr >= 220).sum()) / float(max(1, area))
    if bright_ratio < 0.70:
        return False
    try:
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    except Exception:
        return False
    components: List[Tuple[float, float, int]] = []
    oversized = 0
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, component_area = [int(value) for value in stats[label][:5]]
        if component_area < 20:
            continue
        if cw > max(48, int(w * 0.72)) or ch > max(72, int(h * 0.72)):
            oversized += 1
            continue
        density = component_area / float(max(1, cw * ch))
        if 0.10 <= density <= 0.95 and 4 <= cw <= max(42, int(w * 0.65)) and 6 <= ch <= max(48, int(h * 0.45)):
            components.append((cx + cw / 2.0, cy + ch / 2.0, component_area))
    if oversized > 0 or len(components) < 3:
        return False
    x_centers = [item[0] for item in components]
    y_centers = [item[1] for item in components]
    if max(y_centers) - min(y_centers) < max(32.0, h * 0.35):
        return False
    if max(x_centers) - min(x_centers) > max(28.0, w * 0.42):
        return False
    component_area = sum(item[2] for item in components)
    if component_area / float(max(1, dark_count)) < 0.58:
        return False
    return True


def _luma_bbox_has_multicolumn_vertical_text_components(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None or np is None or cv2 is None:
        return False
    x, y, w, h = _normalize_xywh(bbox, image_size)
    if w < 48 or h < 64:
        return False
    try:
        crop = luma_image.crop((x, y, x + w, y + h)).convert("L")
    except Exception:
        return False
    arr = np.asarray(crop)
    if arr.size <= 0:
        return False
    dark_mask = (arr <= 120).astype("uint8")
    dark_count = int(dark_mask.sum())
    area = int(arr.size)
    dark_ratio = dark_count / float(max(1, area))
    if dark_ratio < 0.025 or dark_ratio > 0.20:
        return False
    bright_ratio = float((arr >= 220).sum()) / float(max(1, area))
    if bright_ratio < 0.70:
        return False
    try:
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_mask, 8)
    except Exception:
        return False

    components: List[Tuple[float, float, int, int, int]] = []
    oversized = 0
    for label in range(1, int(stats.shape[0])):
        cx, cy, cw, ch, component_area = [int(value) for value in stats[label][:5]]
        if component_area < 20:
            continue
        if cw > max(48, int(w * 0.72)) or ch > max(72, int(h * 0.72)):
            oversized += 1
            continue
        density = component_area / float(max(1, cw * ch))
        if 0.10 <= density <= 0.95 and 4 <= cw <= max(42, int(w * 0.65)) and 6 <= ch <= max(48, int(h * 0.45)):
            components.append((cx + cw / 2.0, cy + ch / 2.0, cw, ch, component_area))
    if oversized > 0 or len(components) < 6:
        return False

    widths = sorted(component[2] for component in components)
    median_width = float(widths[len(widths) // 2])
    column_gap = max(8.0, median_width * 0.95)
    columns: List[List[Tuple[float, float, int, int, int]]] = []
    for component in sorted(components, key=lambda item: item[0]):
        if not columns or component[0] - columns[-1][-1][0] > column_gap:
            columns.append([component])
        else:
            columns[-1].append(component)

    qualifying_columns: List[List[Tuple[float, float, int, int, int]]] = []
    for column in columns:
        if len(column) < 3:
            continue
        y_centers = [component[1] for component in column]
        if max(y_centers) - min(y_centers) < max(28.0, h * 0.18):
            continue
        qualifying_columns.append(column)
    if len(qualifying_columns) < 2:
        return False

    column_centers = [
        sum(component[0] for component in column) / float(len(column))
        for column in qualifying_columns
    ]
    column_centers.sort()
    if min(
        right - left
        for left, right in zip(column_centers, column_centers[1:])
    ) < max(10.0, median_width * 0.75):
        return False

    accepted_component_area = sum(component[4] for component in components)
    qualifying_component_area = sum(
        component[4]
        for column in qualifying_columns
        for component in column
    )
    if qualifying_component_area / float(max(1, accepted_component_area)) < 0.72:
        return False
    if accepted_component_area / float(max(1, dark_count)) < 0.58:
        return False
    return True


def _paired_ogkalu_text_presence_proven(
    *,
    luma_image: Any,
    semantic_role_evidence: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    entries = [
        entry
        for entry in (semantic_role_evidence.get("text_unit_evidence_bboxes") or [])
        if isinstance(entry, Mapping) and str(entry.get("class_name") or "") == "text_bubble"
    ]
    if not entries:
        return False
    for entry in entries:
        if (
            _optional_float(entry.get("confidence")) is not None
            and float(entry.get("confidence") or 0.0) < OGKALU_RETAINED_PAIRED_TEXT_EVIDENCE_CONFIDENCE
        ):
            continue
        bbox = _text_evidence_bbox_xywh(entry, image_size)
        if _luma_bbox_has_text_like_dark_components(luma_image, bbox, image_size):
            return True
        if _luma_bbox_has_sparse_vertical_text_components(luma_image, bbox, image_size):
            return True
        if _luma_bbox_has_multicolumn_vertical_text_components(luma_image, bbox, image_size):
            return True
    return False


def _semantic_role_has_strong_ogkalu_bubble_evidence(
    semantic_role_evidence: Mapping[str, Any],
) -> bool:
    entries = semantic_role_evidence.get("model_evidence_bboxes") or []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("class_name") or "") != "bubble":
            continue
        confidence = _optional_float(entry.get("confidence"))
        if confidence is not None and confidence >= OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE:
            return True
    return False


def _looks_like_paired_ogkalu_speech_bubble(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
    clipped: bool,
    luma_image: Any,
) -> bool:
    del visual
    if clipped or fused_type not in {"ambiguous", "speech_bubble"}:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    role_classes = set(_semantic_role_values(semantic_role_evidence, "ogkalu_class_names"))
    role_signals = set(_semantic_role_values(semantic_role_evidence, "role_signals"))
    if OGKALU_BUBBLE_TEXT_PAIR_REASON not in reason_text and "ogkalu_bubble_text_pair" not in role_signals:
        return False
    if not {"bubble", "text_bubble"}.issubset(role_classes):
        return False
    if any(token in reason_text for token in ("sfx", "decorative", "preserve", "text_free", "art_only", "non_text", "non-text")):
        return False
    conflicts = set(_semantic_role_values(semantic_role_evidence, "conflict_evidence"))
    if any(any(token in conflict.lower() for token in ("sfx", "decorative", "preserve", "art", "non_text", "non-text")) for conflict in conflicts):
        return False
    if not _semantic_role_has_strong_ogkalu_bubble_evidence(semantic_role_evidence):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    if w < width * 0.08 or h < height * 0.055:
        return False
    if w > width * 0.42 or h > height * 0.34:
        return False
    area_ratio = _bbox_area_xywh([x, y, w, h]) / max(1.0, float(width * height))
    if area_ratio <= 0.0 or area_ratio > 0.10:
        return False
    return _paired_ogkalu_text_presence_proven(
        luma_image=luma_image,
        semantic_role_evidence=semantic_role_evidence,
        image_size=image_size,
    )


def _looks_like_standalone_ogkalu_speech_bubble(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
    clipped: bool,
) -> bool:
    del fused_type, reasons, bbox, visual, image_size, semantic_role_evidence, clipped
    return False


def _looks_like_unresolved_ogkalu_speech_risk_candidate(
    *,
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    semantic_role_evidence: Mapping[str, Any],
    clipped: bool,
) -> bool:
    if clipped or fused_type not in {"ambiguous", "speech_bubble"}:
        return False
    role_classes = set(_semantic_role_values(semantic_role_evidence, "ogkalu_class_names"))
    role_signals = set(_semantic_role_values(semantic_role_evidence, "role_signals"))
    if not (role_classes & {"bubble", "text_bubble"}):
        return False
    if "speech_bubble_mask_evidence" in role_signals:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if not any(
        token in reason_text
        for token in ("ogkalu_bubble_without_kitsumed_mask", "ogkalu_text_bubble_without_kitsumed_mask")
    ):
        return False
    if any(token in reason_text for token in ("sfx", "decorative", "preserve", "text_free", "art_only", "non_text", "non-text")):
        return False
    conflicts = set(_semantic_role_values(semantic_role_evidence, "conflict_evidence"))
    if any(any(token in conflict.lower() for token in ("sfx", "decorative", "preserve", "art", "non_text", "non-text")) for conflict in conflicts):
        return False
    confidence = _semantic_role_model_confidence(semantic_role_evidence)
    if confidence < OGKALU_SINGLE_MODEL_AUTHORITY_CONFIDENCE:
        return False
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return False
    if w < width * 0.08 or h < height * 0.055:
        return False
    if w > width * 0.36 or h > height * 0.30:
        return False
    area_ratio = float(visual.get("area_ratio") or 0.0)
    if area_ratio <= 0.0 or area_ratio > 0.08:
        return False
    page_edge_touching = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
    dark = _optional_float(visual.get("dark_ratio"))
    bright = _optional_float(visual.get("bright_ratio"))
    dark_or_art_context = bool(
        (dark is not None and dark >= 0.15)
        or (bright is not None and bright < 0.62)
    )
    neighbor_ids = _semantic_role_values(semantic_role_evidence, "neighboring_speech_context_ids")
    if not neighbor_ids:
        return False
    return bool(page_edge_touching or dark_or_art_context or "text_bubble" in role_classes)


def _semantic_role_model_confidence(role_evidence: Mapping[str, Any]) -> float:
    confidence = _optional_float(role_evidence.get("model_confidence"))
    if confidence is not None:
        return confidence
    confidence = _optional_float(role_evidence.get("confidence"))
    if confidence is not None:
        return confidence
    values: List[float] = []
    for field_name in ("text_unit_evidence_bboxes", "model_evidence_bboxes"):
        entries = role_evidence.get(field_name) or []
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            value = _optional_float(entry.get("confidence"))
            if value is not None:
                values.append(value)
    return max(values) if values else 0.0


def _safe_unknown_ocr_fallback(
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if "text_free" in reason_text or fused_type == "free_text":
        return False
    if not any(token in reason_text for token in ("ogkalu_bubble_without_kitsumed_mask", "ogkalu_text_bubble_without_kitsumed_mask")):
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    area_ratio = float(visual.get("area_ratio") or 0.0)
    area = float(visual.get("area_px") or 0.0)
    if bright is None or dark is None:
        return False
    return area >= 80000 and area_ratio >= 0.035 and bright >= 0.78 and dark <= 0.14


def _mask_primary_speech_activation_completeness_allowed(
    *,
    confidence: str,
    bbox: Sequence[Any],
    reasons: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    clipped: bool,
) -> bool:
    if clipped or confidence != "medium":
        return False
    if _looks_like_pre_ocr_sfx_or_decorative(CONTAINER_SPEECH, reasons, bbox, visual, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < 36_000:
        return False
    if w < 150 or h < 180:
        return False
    aspect = w / max(1, h)
    if aspect < 0.48 or aspect > 1.15:
        return False
    page_w, page_h = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w > page_w * 0.28 or h > page_h * 0.18:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "art_only", "non_text", "preserve")):
        return False
    return True


def _mask_primary_irregular_speech_activation_allowed(
    *,
    confidence: str,
    bbox: Sequence[Any],
    reasons: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    clipped: bool,
) -> bool:
    if clipped or confidence != "medium":
        return False
    if _looks_like_pre_ocr_sfx_or_decorative(CONTAINER_SPEECH, reasons, bbox, visual, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < 18_000:
        return False
    if w < 80 or h < 120:
        return False
    aspect = w / max(1, h)
    if aspect < 0.35 or aspect > 1.25:
        return False
    page_w, page_h = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w > page_w * 0.24 or h > page_h * 0.20:
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    if dark is not None and dark >= 0.24:
        return False
    if bright is not None and bright < 0.58:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "art_only", "non_text", "preserve")):
        return False
    return True


def _visual_reason_codes(visual: Mapping[str, Any]) -> List[str]:
    codes: List[str] = []
    if visual.get("clipped_or_degenerate"):
        codes.append("text_area_plan:clipped_or_degenerate_container")
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    if bright is not None and bright >= 0.78:
        codes.append("text_area_plan:bright_container_context")
    if dark is not None and dark >= 0.16:
        codes.append("text_area_plan:dark_or_art_context")
    return codes


def _is_clipped_or_degenerate_bbox(bbox: Sequence[Any], image_size: Tuple[int, int]) -> bool:
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 2 or h <= 2:
        return True
    return x <= -2 or y <= -2 or x + w >= width + 2 or y + h >= height + 2


def _coerce_xywh(value: Sequence[Any]) -> Tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return 0, 0, 0, 0
    try:
        x, y, w, h = [int(round(float(v or 0))) for v in value[:4]]
        return x, y, w, h
    except Exception:
        return 0, 0, 0, 0


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _bbox_xyxy_to_xywh(value: Any, image_size: Tuple[int, int]) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return [0, 0, max(1, int(image_size[0])), max(1, int(image_size[1]))]
    x0, y0, x1, y1 = [float(v) for v in value[:4]]
    if x1 < x0 or y1 < y0:
        return [int(round(x0)), int(round(y0)), max(1, int(round(x1))), max(1, int(round(y1)))]
    width, height = int(image_size[0]), int(image_size[1])
    x0 = max(0, min(width - 1, int(math.floor(x0))))
    y0 = max(0, min(height - 1, int(math.floor(y0))))
    x1 = max(x0 + 1, min(width, int(math.ceil(x1))))
    y1 = max(y0 + 1, min(height, int(math.ceil(y1))))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _normalize_xywh(value: Any, image_size: Tuple[int, int]) -> List[int]:
    x, y, w, h = _coerce_xywh(value)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _bbox_area_xywh(bbox: Sequence[Any]) -> int:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0
    try:
        return max(0, int(bbox[2])) * max(0, int(bbox[3]))
    except Exception:
        return 0


def _xywh_to_xyxy(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in value[:4]]
    except Exception:
        return None
    return (x, y, x + max(0.0, w), y + max(0.0, h))


def _area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _center_inside(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float] | None) -> bool:
    if not b:
        return False
    cx = (a[0] + a[2]) / 2.0
    cy = (a[1] + a[3]) / 2.0
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _container_priority(container: Mapping[str, Any]) -> int:
    ctype = container.get("container_type")
    if ctype == CONTAINER_SFX:
        return 4
    if ctype == CONTAINER_SPEECH:
        return 3
    if ctype == CONTAINER_CAPTION:
        return 2
    return 1


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _write_plan_summary(
    path: Path,
    plan: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    ocr_candidates: Sequence[Mapping[str, Any]],
    fallbacks: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# TextAreaPlan Summary",
        "",
        f"- version: `{plan.get('version')}`",
        f"- generated: `{plan.get('generated')}`",
        f"- containers: `{len(plan.get('containers') or [])}`",
        f"- scopes: `{len(plan.get('scopes') or [])}`",
        f"- scoped detection candidates: `{len(detections)}`",
        f"- scoped OCR candidates: `{len(ocr_candidates)}`",
        f"- fallback decisions: `{len(fallbacks)}`",
        f"- blocked text-area candidates: `{len(blocked)}`",
        f"- summary: `{json.dumps(plan.get('summary') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| container_id | type | intent | ocr | tier | fallback | conflicts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for container in plan.get("containers", []) or []:
        lines.append(
            "| {container_id} | {container_type} | {route_intent} | {ocr_eligible} | {confidence_tier} | {fallback_reason} | {conflicts} |".format(
                container_id=container.get("container_id"),
                container_type=container.get("container_type"),
                route_intent=container.get("route_intent"),
                ocr_eligible=container.get("ocr_eligible"),
                confidence_tier=container.get("confidence_tier"),
                fallback_reason=container.get("fallback_reason") or "",
                conflicts=",".join(str(item) for item in container.get("conflict_flags") or []),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocked_containers_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    blocked: List[Dict[str, Any]] = []
    for container in plan.get("containers", []) or []:
        if bool(container.get("ocr_eligible")) and bool(container.get("comic_text_detector_scope_eligible")):
            continue
        blocked.append(
            {
                "candidate_id": f"blocked_{container.get('container_id')}",
                "container_id": container.get("container_id"),
                "container_type": container.get("container_type"),
                "route_intent": container.get("route_intent"),
                "bbox": container.get("bbox"),
                "ocr_eligible": bool(container.get("ocr_eligible")),
                "comic_text_detector_scope_eligible": bool(container.get("comic_text_detector_scope_eligible")),
                "fallback_reason": container.get("fallback_reason"),
                "ocr_eligibility_reason": container.get("ocr_eligibility_reason"),
                "reason_codes": container.get("evidence_reason_codes") or [],
                "conflict_flags": container.get("conflict_flags") or [],
                "would_change_behavior": False,
            }
        )
    return blocked


def _caption_localization_candidates_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for container in plan.get("containers", []) or []:
        if container.get("container_type") != CONTAINER_CAPTION:
            continue
        candidates.append(
            {
                "candidate_id": f"caption_{container.get('container_id')}",
                "container_id": container.get("container_id"),
                "bbox": container.get("bbox"),
                "route_intent": container.get("route_intent"),
                "ocr_eligible": bool(container.get("ocr_eligible")),
                "comic_text_detector_scope_eligible": bool(container.get("comic_text_detector_scope_eligible")),
                "confidence_tier": container.get("confidence_tier"),
                "fallback_reason": container.get("fallback_reason"),
                "source_model_ids": container.get("source_model_ids") or [],
                "reason_codes": container.get("evidence_reason_codes") or [],
                "conflict_flags": container.get("conflict_flags") or [],
                "would_change_behavior": False,
            }
        )
    return candidates


def _component_auth_overlap_for_label(
    *,
    labels: Any,
    component_label: int,
    component_bbox: Sequence[int],
    scope_bbox: Sequence[int],
    scope_mask: Any,
    centroid: Sequence[float],
) -> Tuple[int, bool]:
    """Count component/scope overlap without materializing a full-page mask."""

    if np is None:
        return 0, False
    cx = float(centroid[0])
    cy = float(centroid[1])
    sx0, sy0, sx1, sy1 = [int(item) for item in scope_bbox[:4]]
    cx0, cy0, cx1, cy1 = [int(item) for item in component_bbox[:4]]
    height = int(getattr(labels, "shape", [0, 0])[0] or 0)
    width = int(getattr(labels, "shape", [0, 0])[1] or 0)
    if width <= 0 or height <= 0:
        return 0, False
    sx0 = max(0, min(width, sx0))
    sx1 = max(0, min(width, sx1))
    sy0 = max(0, min(height, sy0))
    sy1 = max(0, min(height, sy1))
    cx0 = max(0, min(width, cx0))
    cx1 = max(0, min(width, cx1))
    cy0 = max(0, min(height, cy0))
    cy1 = max(0, min(height, cy1))
    ix0 = max(sx0, cx0)
    ix1 = min(sx1, cx1)
    iy0 = max(sy0, cy0)
    iy1 = min(sy1, cy1)
    centroid_inside = sx0 <= cx < sx1 and sy0 <= cy < sy1
    if scope_mask is not None:
        try:
            mask_h, mask_w = scope_mask.shape[:2]
            mx = max(0, min(mask_w - 1, int(round(cx))))
            my = max(0, min(mask_h - 1, int(round(cy))))
            centroid_inside = bool(scope_mask[my, mx] > 0)
            ix0 = max(0, min(min(width, mask_w), ix0))
            ix1 = max(0, min(min(width, mask_w), ix1))
            iy0 = max(0, min(min(height, mask_h), iy0))
            iy1 = max(0, min(min(height, mask_h), iy1))
        except Exception:
            scope_mask = None
    if ix1 <= ix0 or iy1 <= iy0:
        return 0, centroid_inside
    label_slice = labels[iy0:iy1, ix0:ix1]
    if scope_mask is not None:
        try:
            mask_slice = scope_mask[iy0:iy1, ix0:ix1] > 0
            overlap = int(np.count_nonzero((label_slice == int(component_label)) & mask_slice))
        except Exception:
            overlap = int(np.count_nonzero(label_slice == int(component_label)))
    else:
        overlap = int(np.count_nonzero(label_slice == int(component_label)))
    return overlap, centroid_inside


def _write_graph_plan_overlay(
    path: Path,
    image_path: str | Path,
    graph_plan: Mapping[str, Any],
    *,
    layer: str,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    image_path = Path(image_path)
    if not image_path.exists():
        return
    try:
        base = Image.open(image_path).convert("RGB")
    except Exception:
        return
    draw = ImageDraw.Draw(base)
    font = _overlay_font(ImageFont)
    colors = {
        "root": (40, 120, 255),
        "parent": (255, 130, 0),
        "child": (160, 0, 220),
        "excluded": (120, 120, 120),
        "blocker": (220, 0, 0),
    }
    title_by_layer = {
        "root": "Root workflow nodes",
        "parent": "Parent boundary nodes",
        "child": "Child evidence slots",
        "root_parent": "Root/parent graph overlay",
    }
    legend = title_by_layer.get(layer, "Root/parent graph overlay")
    draw.rectangle((8, 8, 640, 74), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((16, 14), legend, fill=(0, 0, 0), font=font)
    draw.text(
        (16, 38),
        "blue=root, orange=parent, purple=child, gray=excluded, red=blocker",
        fill=(0, 0, 0),
        font=font,
    )
    if layer in {"root", "root_parent"}:
        _draw_graph_nodes(
            draw,
            font,
            graph_plan.get("root_nodes") or [],
            id_key="root_node_id",
            color=colors["root"],
            prefix="R",
            width=4,
        )
    if layer in {"parent", "root_parent"}:
        _draw_graph_nodes(
            draw,
            font,
            graph_plan.get("parent_nodes") or [],
            id_key="parent_node_id",
            color=colors["parent"],
            prefix="P",
            width=3,
        )
    if layer == "child":
        _draw_graph_nodes(
            draw,
            font,
            graph_plan.get("child_evidence_slots") or [],
            id_key="child_slot_id",
            color=colors["child"],
            prefix="C",
            width=3,
        )
    if layer in {"root", "root_parent"}:
        _draw_graph_nodes(
            draw,
            font,
            graph_plan.get("excluded_inventory") or [],
            id_key="excluded_id",
            color=colors["excluded"],
            prefix="X",
            width=2,
        )
    _draw_graph_nodes(
        draw,
        font,
        graph_plan.get("graph_blockers") or [],
        id_key="blocker_id",
        color=colors["blocker"],
        prefix="B",
        width=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, quality=92)


def _draw_graph_nodes(
    draw: Any,
    font: Any,
    nodes: Sequence[Any],
    *,
    id_key: str,
    color: Tuple[int, int, int],
    prefix: str,
    width: int,
) -> None:
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        box = _xywh_to_xyxy(node.get("bbox") or [])
        if not box:
            continue
        xyxy = tuple(int(round(v)) for v in box)
        draw.rectangle(xyxy, outline=color, width=width)
        node_id = str(node.get(id_key) or node.get("id") or "").strip()
        role = str(
            node.get("semantic_role")
            or node.get("parent_kind")
            or node.get("exclusion_kind")
            or node.get("blocker_kind")
            or ""
        ).strip()
        label = f"{prefix} {node_id}"
        if role:
            label = f"{label} {role}"
        _draw_label(draw, font, (xyxy[0] + 2, max(0, xyxy[1] - 18)), label, color)


def _write_plan_overlay(
    path: Path,
    image_path: str | Path,
    plan: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    image_path = Path(image_path)
    if not image_path.exists():
        return
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = _overlay_font(ImageFont)
    colors = {
        CONTAINER_SPEECH: (40, 120, 255),
        CONTAINER_CAPTION: (0, 170, 120),
        CONTAINER_SFX: (220, 0, 180),
        CONTAINER_UNKNOWN: (245, 150, 0),
    }
    for container in plan.get("containers", []) or []:
        box = _xywh_to_xyxy(container.get("bbox") or [])
        if not box:
            continue
        color = colors.get(container.get("container_type"), (180, 180, 180))
        xyxy = tuple(int(round(v)) for v in box)
        draw.rectangle(xyxy, outline=color, width=4)
        _draw_label(draw, font, (xyxy[0] + 2, max(0, xyxy[1] - 18)), f"{container.get('container_id')} {container.get('container_type')}", color)
    for candidate in detections:
        box = _xywh_to_xyxy(candidate.get("bbox") or [])
        if not box:
            continue
        xyxy = tuple(int(round(v)) for v in box)
        source = candidate.get("detection_source")
        color = (255, 0, 0) if source == DETECTION_BLOCKED else (0, 0, 0)
        draw.rectangle(xyxy, outline=color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, quality=92)


def _overlay_font(image_font_module: Any) -> Any:
    for candidate in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ):
        try:
            if os.path.isfile(candidate):
                return image_font_module.truetype(candidate, 14)
        except Exception:
            pass
    try:
        return image_font_module.load_default()
    except Exception:
        return None


def _draw_label(draw: Any, font: Any, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    x, y = xy
    try:
        box = draw.textbbox((x, y), text, font=font)
    except Exception:
        box = (x, y, x + max(1, len(text)) * 8, y + 16)
    draw.rectangle(box, fill=(255, 255, 255))
    draw.text((x, y), text, fill=fill, font=font)
