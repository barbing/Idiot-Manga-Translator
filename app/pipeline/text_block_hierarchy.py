# -*- coding: utf-8 -*-
"""Root / parent / child text-block hierarchy contracts.

This module normalizes the already-built BubbleDetection/TextAreaPlan and
LogicalTextBlock ownership records into one explicit hierarchy. It does not run
model inference, OCR, translation, cleanup, or rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import os
import re
from typing import Any

from app.pipeline.text_block_root_graph import (
    ROOT_ACCEPTED,
    ROOT_BLOCKED_NON_TEXT_OR_DECORATIVE,
    ROOT_CAPTION_BACKGROUND_REVIEW_ONLY,
    ROOT_PARTIAL_REVIEW,
    ROOT_REVIEW_ONLY_UNRESOLVED,
    apply_strict_root_transaction_contract,
)


TEXT_BLOCK_HIERARCHY_VERSION = "text_block_hierarchy_v1"

_PHASE2E_TARGET_ROOT_IDS: dict[str, str] = {}

ROOT_SPEECH = "speech_bubble"
ROOT_CAPTION = "caption_background"
ROOT_SFX = "sfx_decorative_art"
ROOT_UNKNOWN = "unknown_fallback"
ROOT_REVIEW = "review"

ROUTE_TRANSLATE_SPEECH = "translate_speech"
ROUTE_TRANSLATE_CAPTION = "translate_caption"
ROUTE_PRESERVE = "preserve"
ROUTE_REVIEW = "review"
ROUTE_FALLBACK = "fallback"

ROLE_SPEECH = "speech"
ROLE_CAPTION = "caption"
ROLE_BACKGROUND = "background"
ROLE_REVIEW = "review"
GRAPH_CAPTION_BACKGROUND_KINDS = {"caption", "background", "caption_background", "background_narration"}

EXECUTABLE_CLEANUP_AUTHORIZATIONS = {
    "cleanup_translate_speech",
    "cleanup_translate_background",
    "cleanup_translate_caption",
}
NON_EXECUTABLE_AUTHORIZATIONS = {
    "review_unknown_not_cleanup",
    "outside_cleanup_scope",
    "ambiguous_component_owner",
    "protect_sfx_decorative",
    "protect_art_or_non_text",
}

STATE_PARENT_ANCHOR = "parent_anchor"
STATE_PARENT_CHILD = "parent_child"
STATE_STANDALONE_PARENT = "standalone_parent"
STATE_DUPLICATE_CHILD = "duplicate_child"
STATE_PUNCTUATION_CHILD = "punctuation_child"
STATE_NOISE_REVIEW_ONLY = "noise_review_only"
STATE_BLOCKED_BY_ROOT_POLICY = "blocked_by_root_policy"
STATE_UNRESOLVED_REVIEW_ONLY = "unresolved_review_only"

OCR_STATE_RECOGNIZED_FOR_TRANSLATION = "recognized_for_translation"
OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING = "recognized_low_confidence_warning"
OCR_TRANSLATION_QUEUED_STATES = {
    OCR_STATE_RECOGNIZED_FOR_TRANSLATION,
    OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING,
}
OCR_TRANSACTION_BLOCKER_STATES = {
    "ocr_empty_blocker",
    "ocr_punctuation_only_blocker",
    "ocr_malformed_blocker",
}
ROUTE_OWNED_OCR_BLOCKER_NO_PARENT_RENDER_CLEANUP_REASON = "route_owned_ocr_blocker_no_parent_render_cleanup"
ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON = "route_owned_ocr_fragment_represented_by_meaningful_parent"

ROOT_FINAL_TRANSLATED_CLEAN = "translated_clean"
ROOT_FINAL_TRANSLATED_RENDER_WARNING = "translated_with_render_warning"
ROOT_FINAL_TRANSLATED_CLEANUP_WARNING = "translated_with_cleanup_warning"
ROOT_FINAL_PRESERVED_SFX_DECORATIVE = "preserved_sfx_decorative"
ROOT_FINAL_CAPTION_BACKGROUND_REVIEW_ONLY = "caption_background_review_only"
ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING = "punctuation_only_nonblocking"
ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER = "unresolved_meaningful_blocker"

VISUAL_EVALUATION_STATUS_REQUIRED = "requires_comprehensive_visual_evaluation"
VISUAL_EVALUATION_PASS_SOURCE = "manual_visual_ledger_only"
VISUAL_EVALUATION_BLOCKED_REASON = "comprehensive_visual_evaluation_not_recorded"


@dataclass
class TextAreaRootBlock:
    root_id: str
    page_id: str
    root_type: str
    text_area_container_ids: list[str] = field(default_factory=list)
    physical_bubble_id: str | None = None
    bbox: list[int] = field(default_factory=list)
    mask_source_ids: list[str] = field(default_factory=list)
    route_policy: str = ROUTE_REVIEW
    ocr_eligible: bool = False
    ctd_scope_eligible: bool = False
    parent_unit_ids: list[str] = field(default_factory=list)
    child_segment_ids: list[str] = field(default_factory=list)
    reading_order_index: int = 0
    reason_codes: list[str] = field(default_factory=list)
    confidence_tier: str | None = None
    fallback_reason: str | None = None
    review_reason: str | None = None
    root_parent_count: int = 0
    root_child_count: int = 0
    root_fragmentation_score: float = 0.0
    root_malformed_parent_count: int = 0
    root_duplicate_partial_parent_count: int = 0
    root_requires_reconstruction: bool = False
    root_source_coherence_status: str = "not_evaluated"
    root_source_coherence_failure_reason: str | None = None
    root_reconstruction_required: bool = False
    root_reconstruction_attempted: bool = False
    root_reconstruction_status: str = "not_attempted"
    root_reconstruction_before_sources: list[str] = field(default_factory=list)
    root_reconstruction_after_source: str = ""
    root_reconstruction_rejected_attempts: list[dict[str, Any]] = field(default_factory=list)
    root_unresolved_visible_source_count: int = 0
    root_validation_blocker: bool = False
    root_transaction_status: str = "not_evaluated"
    root_transaction_reason: str = ""
    root_has_accepted_parent: bool = False
    root_accepted_parent_ids: list[str] = field(default_factory=list)
    root_rejected_parent_count: int = 0
    root_low_quality_parent_count: int = 0
    root_unresolved_meaningful_child_count: int = 0
    root_review_child_ids: list[str] = field(default_factory=list)
    root_acceptance_blocker: bool = False
    root_acceptance_blocker_reason: str = ""
    root_final_state: str = "not_evaluated"
    root_final_state_reason: str = ""
    root_has_meaningful_visible_source: bool = False
    root_is_punctuation_only: bool = False
    root_is_sfx_decorative: bool = False
    root_is_caption_background: bool = False
    root_closeout_blocker: bool = False
    root_closeout_warning_reasons: list[str] = field(default_factory=list)
    source_erasure_expected: bool = False
    source_erasure_mask_coverage: float | None = None
    source_erasure_visual_residual_score: float | None = None
    source_erasure_failure_reason: str = ""
    source_glyph_mask_id: str = ""
    cleanup_partition_id: str = ""
    render_text_completeness_pass: bool | None = None
    render_wrapped_lines: list[str] = field(default_factory=list)
    render_missing_characters: str = ""
    render_outside_root_ratio: float = 0.0
    render_density_score: float = 0.0
    render_readability_warning_reason: str = ""
    root_visual_separation_status: str = "not_evaluated"
    root_visual_separation_score: float = 0.0
    root_overmerge_risk: bool = False
    root_overmerge_rejection_reason: str = ""
    cleanup_authorization: str = ""
    semantic_authorization_state: str = ""
    authorization_explicit: bool = False
    must_not_mutate: bool = False
    human_review_required: bool = False
    root_source_erasure_uncovered_component_bboxes: list[list[int]] = field(default_factory=list)
    root_source_erasure_complete: bool = True
    root_source_erasure_blocker_reason: str = ""
    root_committed_source_group_ids: list[str] = field(default_factory=list)
    root_cleanup_covered_group_ids: list[str] = field(default_factory=list)
    root_render_feasibility_status: str = ""
    root_closeout_blocker_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        omitted = {
            "root_source_erasure_uncovered_component_bboxes",
            "root_source_erasure_complete",
            "root_source_erasure_blocker_reason",
            "root_committed_source_group_ids",
            "root_cleanup_covered_group_ids",
            "root_render_feasibility_status",
            "root_closeout_blocker_reasons",
        }
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in omitted
        }


@dataclass
class ParentLogicalTextUnit:
    parent_id: str
    page_id: str
    root_id: str
    role: str
    source_text: str = ""
    source_text_before_reconstruction: str = ""
    source_reconstruction_status: str = "not_attempted"
    source_reconstruction_crop_bbox: list[int] = field(default_factory=list)
    source_reconstruction_confidence: float | None = None
    anchor_child_id: str | None = None
    child_segment_ids: list[str] = field(default_factory=list)
    dependent_child_ids: list[str] = field(default_factory=list)
    duplicate_child_ids: list[str] = field(default_factory=list)
    punctuation_child_ids: list[str] = field(default_factory=list)
    noise_child_ids: list[str] = field(default_factory=list)
    rejected_child_ids: list[str] = field(default_factory=list)
    translation_unit: bool = False
    cleanup_unit: bool = False
    render_unit: bool = False
    cleanup_target_bbox: list[int] = field(default_factory=list)
    render_allowed_area: list[int] = field(default_factory=list)
    source_conservation_status: str = "complete"
    unresolved_reason: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    confidence: float | None = None
    source_contract_owner: str = ""
    source_contract_region_id: str = ""
    source_contract_quality_state: str = ""
    source_contract_quality_action: str = ""
    source_quality_warning_reason_codes: list[str] = field(default_factory=list)
    source_coherence_status: str = "not_evaluated"
    source_coherence_reason_codes: list[str] = field(default_factory=list)
    source_coherence_action: str = "translate"
    duplicate_partial_with_parent_ids: list[str] = field(default_factory=list)
    standalone_parent_rejected: bool = False
    parent_visual_group_id: str = ""
    parent_visual_group_bbox: list[int] = field(default_factory=list)
    parent_visual_group_child_ids: list[str] = field(default_factory=list)
    reconstruction_rejected_for_visual_overmerge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ChildRecognizedTextSegment:
    child_id: str
    page_id: str
    root_id: str
    parent_id: str | None
    source_region_id: str
    bbox: list[int] = field(default_factory=list)
    polygon: list[Any] = field(default_factory=list)
    ocr_text: str = ""
    ocr_confidence: float | None = None
    source_orientation: str | None = None
    detection_source: str | None = None
    final_state: str = STATE_UNRESOLVED_REVIEW_ONLY
    represented_by_parent_id: str | None = None
    translated_independently: bool = False
    cleanup_independently: bool = False
    render_independently: bool = False
    reason_codes: list[str] = field(default_factory=list)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FinalizedTextExecutionParent:
    parent_id: str
    page_id: str
    root_id: str
    state: str
    source_text: str = ""
    role: str = ""
    translation_required: bool = False
    cleanup_required: bool = False
    render_required: bool = False
    represented_child_ids: list[str] = field(default_factory=list)
    source_region_ids: list[str] = field(default_factory=list)
    cleanup_target_bbox: list[int] = field(default_factory=list)
    render_allowed_area: list[int] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    unresolved_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FinalizedTextExecutionUnits:
    page_id: str
    active_translation_parents: list[FinalizedTextExecutionParent] = field(default_factory=list)
    punctuation_parent_obligations: list[FinalizedTextExecutionParent] = field(default_factory=list)
    blocked_or_unresolved_parents: list[FinalizedTextExecutionParent] = field(default_factory=list)
    excluded_nonworkflow_children: list[ChildRecognizedTextSegment] = field(default_factory=list)
    active_parent_ids: list[str] = field(default_factory=list)
    punctuation_parent_ids: list[str] = field(default_factory=list)
    blocked_parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "active_parent_ids": list(self.active_parent_ids),
            "punctuation_parent_ids": list(self.punctuation_parent_ids),
            "blocked_parent_ids": list(self.blocked_parent_ids),
            "active_translation_parents": [
                parent.to_dict() for parent in self.active_translation_parents
            ],
            "punctuation_parent_obligations": [
                parent.to_dict() for parent in self.punctuation_parent_obligations
            ],
            "blocked_or_unresolved_parents": [
                parent.to_dict() for parent in self.blocked_or_unresolved_parents
            ],
            "excluded_nonworkflow_children": [
                child.to_dict() for child in self.excluded_nonworkflow_children
            ],
        }


@dataclass
class TextBlockHierarchyResult:
    page_id: str
    version: str = TEXT_BLOCK_HIERARCHY_VERSION
    generated: bool = True
    roots: list[TextAreaRootBlock] = field(default_factory=list)
    parent_units: list[ParentLogicalTextUnit] = field(default_factory=list)
    child_segments: list[ChildRecognizedTextSegment] = field(default_factory=list)
    unresolved_children: list[ChildRecognizedTextSegment] = field(default_factory=list)
    translation_unit_ids: list[str] = field(default_factory=list)
    cleanup_unit_ids: list[str] = field(default_factory=list)
    render_unit_ids: list[str] = field(default_factory=list)
    error: str | None = None

    def summary_counts(self) -> dict[str, Any]:
        root_types: dict[str, int] = {}
        child_states: dict[str, int] = {}
        parent_roles: dict[str, int] = {}
        raw_child_translation_violations = 0
        root_coherence: dict[str, int] = {}
        root_reconstruction: dict[str, int] = {}
        root_transactions: dict[str, int] = {}
        root_final_states: dict[str, int] = {}
        roots_requiring_reconstruction = 0
        root_validation_blockers = 0
        root_acceptance_blockers = 0
        root_closeout_blockers = 0
        cleanup_warning_roots = 0
        render_warning_roots = 0
        punctuation_only_roots = 0
        malformed_parent_count = 0
        rejected_parent_count = 0
        route_owned_translatable_count = 0
        route_owned_translation_queued_count = 0
        route_owned_ocr_warning_count = 0
        route_owned_ocr_blocker_count = 0
        accepted_parent_without_translation_count = 0
        blocked_route_with_render_unit_count = 0
        blocked_route_with_cleanup_unit_count = 0
        route_owned_root_ids = {
            root.root_id
            for root in self.roots
            if root.route_policy in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}
        }
        blocked_child_parent_ids: set[str] = set()
        for root in self.roots:
            root_types[root.root_type] = root_types.get(root.root_type, 0) + 1
            status = root.root_source_coherence_status or "not_evaluated"
            root_coherence[status] = root_coherence.get(status, 0) + 1
            reconstruction_status = root.root_reconstruction_status or "not_attempted"
            root_reconstruction[reconstruction_status] = root_reconstruction.get(reconstruction_status, 0) + 1
            transaction_status = root.root_transaction_status or "not_evaluated"
            root_transactions[transaction_status] = root_transactions.get(transaction_status, 0) + 1
            final_state = root.root_final_state or "not_evaluated"
            root_final_states[final_state] = root_final_states.get(final_state, 0) + 1
            if root.root_requires_reconstruction:
                roots_requiring_reconstruction += 1
            if root.root_validation_blocker:
                root_validation_blockers += 1
            if root.root_acceptance_blocker:
                root_acceptance_blockers += 1
            if root.root_closeout_blocker:
                root_closeout_blockers += 1
            if root.root_final_state == ROOT_FINAL_TRANSLATED_CLEANUP_WARNING:
                cleanup_warning_roots += 1
            if root.root_final_state == ROOT_FINAL_TRANSLATED_RENDER_WARNING:
                render_warning_roots += 1
            if root.root_final_state == ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING:
                punctuation_only_roots += 1
        for parent in self.parent_units:
            parent_roles[parent.role] = parent_roles.get(parent.role, 0) + 1
            if parent.source_coherence_action in {"repair_required", "block_review_only"}:
                malformed_parent_count += 1
            if parent.standalone_parent_rejected:
                rejected_parent_count += 1
            if parent.root_id in route_owned_root_ids:
                if parent.translation_unit:
                    route_owned_translation_queued_count += 1
                elif parent.cleanup_unit or parent.render_unit:
                    accepted_parent_without_translation_count += 1
        for child in self.child_segments:
            child_states[child.final_state] = child_states.get(child.final_state, 0) + 1
            if child.translated_independently and child.final_state != STATE_STANDALONE_PARENT:
                raw_child_translation_violations += 1
            if child.root_id in route_owned_root_ids:
                route_owned_translatable_count += 1
                state = _ocr_transaction_state_from_child(child)
                if state == OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING:
                    route_owned_ocr_warning_count += 1
                if state in OCR_TRANSACTION_BLOCKER_STATES:
                    route_owned_ocr_blocker_count += 1
                    if child.parent_id and not _route_owned_ocr_fragment_represented_by_parent(child):
                        blocked_child_parent_ids.add(str(child.parent_id))
                    if child.render_independently:
                        blocked_route_with_render_unit_count += 1
                    if child.cleanup_independently:
                        blocked_route_with_cleanup_unit_count += 1
        for parent in self.parent_units:
            if parent.parent_id not in blocked_child_parent_ids:
                continue
            if parent.render_unit:
                blocked_route_with_render_unit_count += 1
            if parent.cleanup_unit:
                blocked_route_with_cleanup_unit_count += 1
        metadata_closeout_candidate = (
            bool(self.roots)
            and root_closeout_blockers == 0
            and cleanup_warning_roots == 0
            and render_warning_roots == 0
        )
        return {
            "root_count": len(self.roots),
            "root_type_counts": root_types,
            "parent_unit_count": len(self.parent_units),
            "parent_role_counts": parent_roles,
            "child_segment_count": len(self.child_segments),
            "child_final_state_counts": child_states,
            "unresolved_child_count": len(self.unresolved_children),
            "translation_unit_count": len(self.translation_unit_ids),
            "cleanup_unit_count": len(self.cleanup_unit_ids),
            "render_unit_count": len(self.render_unit_ids),
            "raw_child_translation_violation_count": raw_child_translation_violations,
            "root_source_coherence_status_counts": root_coherence,
            "root_reconstruction_status_counts": root_reconstruction,
            "root_transaction_status_counts": root_transactions,
            "root_final_state_counts": root_final_states,
            "root_requires_reconstruction_count": roots_requiring_reconstruction,
            "root_validation_blocker_count": root_validation_blockers,
            "root_acceptance_blocker_count": root_acceptance_blockers,
            "root_closeout_blocker_count": root_closeout_blockers,
            "unresolved_meaningful_blocker_count": root_final_states.get(ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER, 0),
            "cleanup_warning_root_count": cleanup_warning_roots,
            "render_warning_root_count": render_warning_roots,
            "punctuation_only_nonblocking_count": punctuation_only_roots,
            "page_metadata_closeout_candidate": metadata_closeout_candidate,
            "page_visual_evaluation_required": True,
            "page_visual_evaluation_status": VISUAL_EVALUATION_STATUS_REQUIRED,
            "page_visual_evaluation_pass_source": VISUAL_EVALUATION_PASS_SOURCE,
            "page_visual_closeout_blocked_reason": VISUAL_EVALUATION_BLOCKED_REASON,
            "page_visual_closeout_pass": False,
            "malformed_source_parent_count": malformed_parent_count,
            "standalone_parent_rejection_count": rejected_parent_count,
            "route_owned_translatable_count": route_owned_translatable_count,
            "route_owned_translation_queued_count": route_owned_translation_queued_count,
            "route_owned_ocr_warning_count": route_owned_ocr_warning_count,
            "route_owned_ocr_blocker_count": route_owned_ocr_blocker_count,
            "accepted_parent_without_translation_count": accepted_parent_without_translation_count,
            "blocked_route_with_render_unit_count": blocked_route_with_render_unit_count,
            "blocked_route_with_cleanup_unit_count": blocked_route_with_cleanup_unit_count,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "text_block_hierarchy_version": self.version,
            "text_block_hierarchy_generated": self.generated,
            "text_block_hierarchy_error": self.error,
            "text_area_root_blocks": [root.to_dict() for root in self.roots],
            "parent_logical_text_units": [parent.to_dict() for parent in self.parent_units],
            "child_recognized_text_segments": [child.to_dict() for child in self.child_segments],
            "unresolved_child_segments": [child.to_dict() for child in self.unresolved_children],
            "translation_unit_ids": list(self.translation_unit_ids),
            "cleanup_unit_ids": list(self.cleanup_unit_ids),
            "render_unit_ids": list(self.render_unit_ids),
            "finalized_execution_units": self.finalized_execution_units().to_dict(),
            "text_block_hierarchy_summary": self.summary_counts(),
        }

    def finalized_execution_units(self) -> FinalizedTextExecutionUnits:
        children_by_parent: dict[str, list[ChildRecognizedTextSegment]] = {}
        for child in self.child_segments:
            if child.parent_id:
                children_by_parent.setdefault(str(child.parent_id), []).append(child)

        units = FinalizedTextExecutionUnits(page_id=self.page_id)
        for parent in self.parent_units:
            children = children_by_parent.get(parent.parent_id, [])
            finalized_parent = _finalized_parent_record(parent, children)
            if _parent_is_punctuation_identity(parent):
                units.punctuation_parent_obligations.append(finalized_parent)
                units.punctuation_parent_ids.append(parent.parent_id)
            elif _parent_is_active_translation_unit(parent):
                units.active_translation_parents.append(finalized_parent)
                units.active_parent_ids.append(parent.parent_id)
            else:
                units.blocked_or_unresolved_parents.append(finalized_parent)
                units.blocked_parent_ids.append(parent.parent_id)

        units.excluded_nonworkflow_children = [
            child for child in self.child_segments
            if child.final_state in {STATE_BLOCKED_BY_ROOT_POLICY, STATE_NOISE_REVIEW_ONLY}
        ]
        return units

    def region_audit_fields(self) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        root_by_id = {root.root_id: root for root in self.roots}
        parent_by_id = {parent.parent_id: parent for parent in self.parent_units}
        children_by_parent: dict[str, list[ChildRecognizedTextSegment]] = {}
        for child in self.child_segments:
            if child.parent_id:
                children_by_parent.setdefault(str(child.parent_id), []).append(child)
        for child in self.child_segments:
            root = root_by_id.get(child.root_id)
            parent = parent_by_id.get(str(child.parent_id or ""))
            parent_children = children_by_parent.get(str(child.parent_id or ""), [])
            parent_active = bool(parent and _parent_is_active_translation_unit(parent))
            graph_ownership_status = _graph_child_ownership_status(child.final_state)
            fields[child.source_region_id] = {
                "text_block_root_id": child.root_id,
                "parent_logical_text_unit_id": child.parent_id,
                "child_recognized_text_segment_id": child.child_id,
                "child_final_state": child.final_state,
                "represented_by_parent_id": child.represented_by_parent_id,
                "logical_text_block_id": child.parent_id,
                "logical_text_ownership_status": graph_ownership_status,
                "logical_text_block_source_text": parent.source_text if parent else "",
                "logical_text_block_bbox": list(parent.cleanup_target_bbox) if parent else [],
                "logical_text_block_allowed_bbox": list(parent.render_allowed_area) if parent else [],
                "parent_logical_text_unit_cleanup_target_bbox": list(parent.cleanup_target_bbox) if parent else [],
                "parent_logical_text_unit_render_allowed_area": list(parent.render_allowed_area) if parent else [],
                "logical_text_block_member_region_ids": [
                    item.source_region_id for item in parent_children if item.source_region_id
                ],
                "logical_text_block_anchor_region_id": (
                    next((item.source_region_id for item in parent_children if item.final_state == STATE_PARENT_ANCHOR), "")
                ),
                "logical_text_block_transferred_region_ids": [
                    item.source_region_id for item in parent_children if item.final_state == STATE_PARENT_CHILD
                ],
                "logical_text_block_duplicate_region_ids": [
                    item.source_region_id for item in parent_children if item.final_state == STATE_DUPLICATE_CHILD
                ],
                "logical_text_block_translation_unit": bool(parent_active and child.final_state == STATE_PARENT_ANCHOR),
                "source_text_represented_by_block_id": child.represented_by_parent_id,
                "translated_independently": child.translated_independently,
                "cleanup_independently": child.cleanup_independently,
                "render_independently": child.render_independently,
                "hierarchy_unresolved_reason": (
                    None if child.final_state != STATE_UNRESOLVED_REVIEW_ONLY else "unresolved_review_only"
                ),
                "hierarchy_reason_codes": list(child.reason_codes),
                "active_translation_unit_id": (
                    child.parent_id if parent_active and child.parent_id and child.final_state in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT} else None
                ),
                "root_parent_count": root.root_parent_count if root else None,
                "root_child_count": root.root_child_count if root else None,
                "root_fragmentation_score": root.root_fragmentation_score if root else None,
                "root_malformed_parent_count": root.root_malformed_parent_count if root else None,
                "root_duplicate_partial_parent_count": root.root_duplicate_partial_parent_count if root else None,
                "root_requires_reconstruction": root.root_requires_reconstruction if root else None,
                "root_source_coherence_status": root.root_source_coherence_status if root else None,
                "root_source_coherence_failure_reason": root.root_source_coherence_failure_reason if root else None,
                "root_reconstruction_required": root.root_reconstruction_required if root else None,
                "root_reconstruction_attempted": root.root_reconstruction_attempted if root else None,
                "root_reconstruction_status": root.root_reconstruction_status if root else None,
                "root_reconstruction_before_sources": list(root.root_reconstruction_before_sources) if root else [],
                "root_reconstruction_after_source": root.root_reconstruction_after_source if root else "",
                "root_unresolved_visible_source_count": root.root_unresolved_visible_source_count if root else None,
                "root_validation_blocker": root.root_validation_blocker if root else None,
                "root_transaction_status": root.root_transaction_status if root else None,
                "root_transaction_reason": root.root_transaction_reason if root else None,
                "root_has_accepted_parent": root.root_has_accepted_parent if root else None,
                "root_accepted_parent_ids": list(root.root_accepted_parent_ids) if root else [],
                "root_rejected_parent_count": root.root_rejected_parent_count if root else None,
                "root_low_quality_parent_count": root.root_low_quality_parent_count if root else None,
                "root_unresolved_meaningful_child_count": root.root_unresolved_meaningful_child_count if root else None,
                "root_review_child_ids": list(root.root_review_child_ids) if root else [],
                "root_acceptance_blocker": root.root_acceptance_blocker if root else None,
                "root_acceptance_blocker_reason": root.root_acceptance_blocker_reason if root else "",
                "root_final_state": root.root_final_state if root else None,
                "root_final_state_reason": root.root_final_state_reason if root else "",
                "root_has_meaningful_visible_source": root.root_has_meaningful_visible_source if root else None,
                "root_is_punctuation_only": root.root_is_punctuation_only if root else None,
                "root_is_sfx_decorative": root.root_is_sfx_decorative if root else None,
                "root_is_caption_background": root.root_is_caption_background if root else None,
                "root_closeout_blocker": root.root_closeout_blocker if root else None,
                "root_closeout_warning_reasons": list(root.root_closeout_warning_reasons) if root else [],
                "source_erasure_expected": root.source_erasure_expected if root else None,
                "source_erasure_mask_coverage": root.source_erasure_mask_coverage if root else None,
                "source_erasure_visual_residual_score": root.source_erasure_visual_residual_score if root else None,
                "source_erasure_failure_reason": root.source_erasure_failure_reason if root else "",
                "source_glyph_mask_id": root.source_glyph_mask_id if root else "",
                "cleanup_partition_id": root.cleanup_partition_id if root else "",
                "render_text_completeness_pass": root.render_text_completeness_pass if root else None,
                "root_render_wrapped_lines": list(root.render_wrapped_lines) if root else [],
                "render_missing_characters": root.render_missing_characters if root else "",
                "render_outside_root_ratio": root.render_outside_root_ratio if root else None,
                "render_density_score": root.render_density_score if root else None,
                "render_readability_warning_reason": root.render_readability_warning_reason if root else "",
                "parent_source_coherence_status": parent.source_coherence_status if parent else None,
                "parent_source_coherence_reason_codes": list(parent.source_coherence_reason_codes) if parent else [],
                "parent_source_coherence_action": parent.source_coherence_action if parent else None,
                "standalone_parent_rejected": parent.standalone_parent_rejected if parent else False,
            }
        return fields


def build_text_block_hierarchy(
    *,
    page_id: str,
    regions: list[dict[str, Any]],
    text_area_plan: Any | None = None,
    logical_block_result: Any | None = None,
    mutate_regions: bool = True,
    root_reconstruction_status: dict[str, Any] | None = None,
) -> TextBlockHierarchyResult:
    """Build and optionally stamp the root / parent / child hierarchy."""
    try:
        plan = _to_dict(text_area_plan)
        blocks = _logical_blocks(logical_block_result)
        physical_groups = _physical_groups(logical_block_result)
        roots_by_key = _build_roots(page_id, plan, physical_groups)
        roots_by_container = _roots_by_container(roots_by_key)
        roots_by_physical = {
            str(root.physical_bubble_id): root
            for root in roots_by_key.values()
            if root.physical_bubble_id
        }
        parent_units: list[ParentLogicalTextUnit] = []
        children: list[ChildRecognizedTextSegment] = []
        child_by_region: dict[str, ChildRecognizedTextSegment] = {}

        region_by_id = {
            str(region.get("region_id") or ""): region
            for region in regions
            if isinstance(region, dict) and str(region.get("region_id") or "")
        }

        graph_parent_nodes = _graph_parent_nodes(plan)
        if graph_parent_nodes:
            graph_children = _materialize_graph_plan_parents(
                page_id=page_id,
                regions=regions,
                parent_nodes=graph_parent_nodes,
                roots_by_key=roots_by_key,
            )
            parent_units.extend(graph_children[0])
            children.extend(graph_children[1])
            child_by_region.update(graph_children[2])
        else:
            for block in blocks:
                parent = _parent_from_logical_block(page_id, block, roots_by_container, roots_by_physical)
                parent_units.append(parent)
                root = roots_by_key.get(parent.root_id)
                if root:
                    _append_unique(root.parent_unit_ids, parent.parent_id)
                member_region_ids = [
                    str(rid)
                    for rid in (block.get("logical_text_block_member_region_ids") or [])
                    if str(rid)
                ]
                for rid in member_region_ids:
                    region = region_by_id.get(rid)
                    if not region:
                        continue
                    child = _child_from_region(page_id, region, parent.root_id, parent.parent_id, block=block)
                    children.append(child)
                    child_by_region[rid] = child
                    if root:
                        _append_unique(root.child_segment_ids, child.child_id)

        for region in regions:
            rid = str(region.get("region_id") or "")
            if not rid or rid in child_by_region:
                continue
            root = _root_for_region(page_id, region, roots_by_container, roots_by_key)
            child = _child_from_region(page_id, region, root.root_id, None, block=None)
            if child.final_state == STATE_STANDALONE_PARENT:
                if graph_parent_nodes:
                    _set_child_final_state(child, STATE_UNRESOLVED_REVIEW_ONLY)
                    child.translated_independently = False
                    child.cleanup_independently = False
                    child.render_independently = False
                    child.represented_by_parent_id = None
                    _append_unique(child.reason_codes, "text_area_graph_plan_unattached_source_evidence")
                else:
                    parent = _standalone_parent_from_child(page_id, root.root_id, child, region)
                    parent_units.append(parent)
                    child.parent_id = parent.parent_id
                    child.represented_by_parent_id = parent.parent_id
                    _append_unique(root.parent_unit_ids, parent.parent_id)
            children.append(child)
            child_by_region[rid] = child
            _append_unique(root.child_segment_ids, child.child_id)

        parent_by_id = {parent.parent_id: parent for parent in parent_units}
        for child in children:
            if child.parent_id and child.parent_id in parent_by_id:
                parent = parent_by_id[child.parent_id]
                _append_unique(parent.child_segment_ids, child.child_id)
                if child.final_state == STATE_PARENT_CHILD:
                    _append_unique(parent.dependent_child_ids, child.child_id)
                elif child.final_state == STATE_DUPLICATE_CHILD:
                    _append_unique(parent.duplicate_child_ids, child.child_id)
                elif child.final_state == STATE_PUNCTUATION_CHILD:
                    _append_unique(parent.punctuation_child_ids, child.child_id)
                elif child.final_state == STATE_NOISE_REVIEW_ONLY:
                    _append_unique(parent.noise_child_ids, child.child_id)
                elif child.final_state in {STATE_BLOCKED_BY_ROOT_POLICY, STATE_UNRESOLVED_REVIEW_ONLY}:
                    _append_unique(parent.rejected_child_ids, child.child_id)

        _enforce_route_owned_ocr_blockers(parent_units, children)
        _evaluate_root_source_coherence(
            roots_by_key.values(),
            parent_units,
            children,
            root_reconstruction_status=root_reconstruction_status,
        )
        _evaluate_pre_render_root_final_states(
            roots_by_key.values(),
            parent_units,
            children,
        )

        unresolved = [child for child in children if child.final_state == STATE_UNRESOLVED_REVIEW_ONLY]
        translation_units = [parent.parent_id for parent in parent_units if _parent_is_active_translation_unit(parent)]
        cleanup_units = [
            parent.parent_id
            for parent in parent_units
            if parent.cleanup_unit and _parent_is_downstream_execution_obligation(parent)
        ]
        render_units = [
            parent.parent_id
            for parent in parent_units
            if parent.render_unit and _parent_is_downstream_execution_obligation(parent)
        ]
        result = TextBlockHierarchyResult(
            page_id=page_id,
            roots=sorted(roots_by_key.values(), key=lambda root: (root.reading_order_index, root.root_id)),
            parent_units=parent_units,
            child_segments=children,
            unresolved_children=unresolved,
            translation_unit_ids=translation_units,
            cleanup_unit_ids=cleanup_units,
            render_unit_ids=render_units,
        )
        if mutate_regions:
            _stamp_regions(regions, result)
        return result
    except Exception as exc:
        return TextBlockHierarchyResult(
            page_id=page_id,
            generated=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _enforce_route_owned_ocr_blockers(
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> None:
    parent_by_id = {parent.parent_id: parent for parent in parents}
    child_by_parent: dict[str, list[ChildRecognizedTextSegment]] = {}
    for child in children:
        if child.parent_id:
            child_by_parent.setdefault(child.parent_id, []).append(child)
        state = _ocr_transaction_state_from_child(child)
        if state not in OCR_TRANSACTION_BLOCKER_STATES:
            continue
        parent = parent_by_id.get(str(child.parent_id or ""))
        if _route_owned_ocr_blocker_can_be_represented_by_parent(child, parent):
            _set_child_final_state(child, _represented_route_owned_ocr_blocker_state(child))
            child.represented_by_parent_id = child.parent_id
            child.translated_independently = False
            child.cleanup_independently = False
            child.render_independently = False
            if parent is not None:
                _remove_value(parent.rejected_child_ids, child.child_id)
                if child.final_state == STATE_PUNCTUATION_CHILD:
                    _append_unique(parent.punctuation_child_ids, child.child_id)
                else:
                    _append_unique(parent.dependent_child_ids, child.child_id)
            _append_unique(child.reason_codes, ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON)
            continue
        _set_child_final_state(child, STATE_UNRESOLVED_REVIEW_ONLY)
        child.represented_by_parent_id = None
        child.translated_independently = False
        child.cleanup_independently = False
        child.render_independently = False
        _append_unique(child.reason_codes, ROUTE_OWNED_OCR_BLOCKER_NO_PARENT_RENDER_CLEANUP_REASON)

    for parent in parents:
        linked_children = child_by_parent.get(parent.parent_id, [])
        if not linked_children:
            continue
        blocker_states = {
            _ocr_transaction_state_from_child(child)
            for child in linked_children
            if _ocr_transaction_state_from_child(child) in OCR_TRANSACTION_BLOCKER_STATES
            and not _route_owned_ocr_fragment_represented_by_parent(child)
        }
        if not blocker_states:
            continue
        _reject_parent_translation_unit(
            parent,
            linked_children,
            [ROUTE_OWNED_OCR_BLOCKER_NO_PARENT_RENDER_CLEANUP_REASON] + sorted(blocker_states),
        )


def _route_owned_ocr_blocker_can_be_represented_by_parent(
    child: ChildRecognizedTextSegment,
    parent: ParentLogicalTextUnit | None,
) -> bool:
    if parent is None:
        return False
    state = _ocr_transaction_state_from_child(child)
    if state not in OCR_TRANSACTION_BLOCKER_STATES:
        return False
    if str(parent.role or "") not in {ROLE_SPEECH, ROLE_CAPTION, ROLE_BACKGROUND}:
        return False
    if not (parent.cleanup_unit and parent.render_unit):
        return False
    if _parent_is_punctuation_identity(parent):
        return state == "ocr_punctuation_only_blocker"
    if not parent.translation_unit:
        return False
    parent_status, _parent_reasons, parent_action = _parent_source_coherence(parent.source_text, role=parent.role)
    if parent_status not in {"coherent", "weak"} or parent_action not in {"translate", "translate_with_root_proof"}:
        return False
    if not _source_body_requires_root_blocker(parent.source_text):
        return False
    if state == "ocr_malformed_blocker":
        return True
    if _source_body_requires_root_blocker(child.ocr_text):
        return False
    return True


def _represented_route_owned_ocr_blocker_state(child: ChildRecognizedTextSegment) -> str:
    state = _ocr_transaction_state_from_child(child)
    if state == "ocr_malformed_blocker":
        return STATE_PARENT_CHILD
    return STATE_PUNCTUATION_CHILD


def _set_child_final_state(child: ChildRecognizedTextSegment, state: str) -> None:
    child.final_state = state
    child.reason_codes = [
        reason
        for reason in (child.reason_codes or [])
        if not str(reason or "").startswith("hierarchy_final_state:")
    ]
    _append_unique(child.reason_codes, f"hierarchy_final_state:{state}")


def _route_owned_ocr_fragment_represented_by_parent(child: ChildRecognizedTextSegment) -> bool:
    return ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON in set(child.reason_codes or [])


def _evaluate_root_source_coherence(
    roots: Any,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
    *,
    root_reconstruction_status: dict[str, Any] | None = None,
) -> None:
    roots_list = list(roots or [])
    parent_by_root: dict[str, list[ParentLogicalTextUnit]] = {}
    child_by_root: dict[str, list[ChildRecognizedTextSegment]] = {}
    child_by_parent: dict[str, list[ChildRecognizedTextSegment]] = {}
    for parent in parents:
        parent_by_root.setdefault(parent.root_id, []).append(parent)
    for child in children:
        child_by_root.setdefault(child.root_id, []).append(child)
        if child.parent_id:
            child_by_parent.setdefault(child.parent_id, []).append(child)

    for parent in parents:
        if _parent_is_punctuation_identity(parent):
            status = "coherent"
            reasons = []
            action = "identity_punctuation"
            parent.source_contract_quality_state = parent.source_contract_quality_state or "punctuation_identity_source"
            parent.source_contract_quality_action = parent.source_contract_quality_action or "identity_punctuation"
        else:
            status, reasons, action = _parent_source_coherence(parent.source_text, role=parent.role)
        if (
            _parent_has_explicit_text_area_graph_boundary(parent)
            and str(parent.source_text or "").strip()
            and (status == "malformed" or action in {"repair_required", "block_review_only"})
        ):
            status = "weak"
            action = "translate_with_review"
            reasons = sorted(
                set(
                    list(reasons or [])
                    + ["explicit_parent_source_quality_diagnostic_only"]
                )
            )
        if _parent_owned_ocr_source_should_translate_with_review(parent, action):
            status = "weak"
            action = "translate_with_review"
            reasons = sorted(
                set(
                    list(reasons or [])
                    + list(parent.source_quality_warning_reason_codes or [])
                    + ["parent_owned_ocr_source_usable_with_warning"]
                )
            )
            parent.source_contract_quality_state = parent.source_contract_quality_state or "usable_source_with_warning"
            parent.source_contract_quality_action = "translate_with_review"
        parent.source_coherence_status = status
        parent.source_coherence_reason_codes = reasons
        parent.source_coherence_action = action
        for reason in reasons:
            _append_unique(parent.reason_codes, reason)

    for root in roots_list:
        reconstruction_record = _root_reconstruction_record(root.root_id, root_reconstruction_status)
        root_parents = sorted(parent_by_root.get(root.root_id, []), key=lambda parent: (parent.render_allowed_area[:2], parent.parent_id))
        root_children = child_by_root.get(root.root_id, [])
        root.root_parent_count = len(root_parents)
        root.root_child_count = len(root_children)
        root.root_reconstruction_before_sources = [
            str(parent.source_text or "")
            for parent in root_parents
            if str(parent.source_text or "").strip()
        ]
        if reconstruction_record:
            root.root_reconstruction_attempted = bool(reconstruction_record.get("attempted"))
            root.root_reconstruction_status = str(reconstruction_record.get("status") or "not_attempted")
            root.root_reconstruction_after_source = str(reconstruction_record.get("after_source") or "")
            root.root_reconstruction_rejected_attempts = list(reconstruction_record.get("rejected_attempts") or [])
            visual = reconstruction_record.get("visual_separation") or {}
            if visual:
                root.root_visual_separation_status = str(visual.get("status") or "not_evaluated")
                root.root_visual_separation_score = float(visual.get("score") or 0.0)
                root.root_overmerge_risk = bool(visual.get("overmerge_risk"))
                root.root_overmerge_rejection_reason = str(visual.get("rejection_reason") or "")
        malformed = [
            parent
            for parent in root_parents
            if parent.source_coherence_action in {"repair_required", "block_review_only"}
        ]
        parent_by_id = {parent.parent_id: parent for parent in root_parents}
        duplicate_pairs = [
            (left_id, right_id)
            for left_id, right_id in _duplicate_partial_parent_pairs(root_parents)
            if not (
                _parent_has_explicit_text_area_graph_boundary(parent_by_id[left_id])
                and _parent_has_explicit_text_area_graph_boundary(parent_by_id[right_id])
            )
        ]
        duplicate_parent_ids = _duplicate_partial_rejected_parent_ids(root_parents, duplicate_pairs)
        for parent in root_parents:
            duplicate_with = {
                other_id
                for left_id, right_id in duplicate_pairs
                for this_id, other_id in ((left_id, right_id), (right_id, left_id))
                if this_id == parent.parent_id
            }
            parent.duplicate_partial_with_parent_ids = sorted(
                pid for pid in duplicate_with if pid != parent.parent_id
            )
            if parent.duplicate_partial_with_parent_ids:
                _append_unique(parent.source_coherence_reason_codes, "duplicate_partial_parent_in_root")
                _append_unique(parent.reason_codes, "duplicate_partial_parent_in_root")

        weak_standalone = [
            parent
            for parent in root_parents
            if _weak_standalone_parent_in_root(parent, root_parents, root)
        ]
        weak_parent_ids = {parent.parent_id for parent in weak_standalone}
        root.root_malformed_parent_count = len(malformed)
        root.root_duplicate_partial_parent_count = len(duplicate_pairs)
        root.root_fragmentation_score = _root_fragmentation_score(root, root_parents, root_children, malformed, duplicate_pairs, weak_standalone)

        failure_reasons: list[str] = []
        if malformed:
            failure_reasons.append("malformed_parent_source")
        if duplicate_pairs:
            failure_reasons.append("duplicate_partial_parent_source")
        if weak_standalone:
            failure_reasons.append("weak_standalone_parent_inside_fragmented_root")
        if root.root_type in {ROOT_SPEECH, ROOT_CAPTION} and root.root_child_count and not root.root_parent_count:
            failure_reasons.append("root_has_children_without_parent_source")
        if (
            root.root_type == ROOT_SPEECH
            and root.route_policy == ROUTE_TRANSLATE_SPEECH
            and not root.root_parent_count
            and not root.root_child_count
            and root.ocr_eligible
            and root.ctd_scope_eligible
        ):
            failure_reasons.append("speech_root_empty_source")
        if (
            root.root_type == ROOT_SPEECH
            and root.root_parent_count >= 3
            and any(
                not _parent_has_explicit_text_area_graph_boundary(parent)
                for parent in root_parents
            )
        ):
            failure_reasons.append("speech_root_many_parent_units")
        if root.root_type == ROOT_CAPTION and _caption_root_looks_incomplete(root_parents, root_children):
            failure_reasons.append("caption_background_root_incomplete_source")
        if root.root_type == ROOT_CAPTION and not root.root_parent_count:
            failure_reasons.append("caption_background_root_empty_source")
        root.root_requires_reconstruction = bool(
            root.root_type in {ROOT_SPEECH, ROOT_CAPTION}
            and failure_reasons
            and root.route_policy in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}
        )
        root.root_reconstruction_required = root.root_requires_reconstruction
        root.root_unresolved_visible_source_count = _root_unresolved_visible_source_count(
            root,
            root_parents,
            root_children,
        )
        if root.root_type == ROOT_SFX or root.route_policy == ROUTE_PRESERVE:
            root.root_source_coherence_status = "blocked_preserve"
            root.root_source_coherence_failure_reason = None
            root.root_validation_blocker = False
            root.root_transaction_status = "root_accepted"
            root.root_transaction_reason = "blocked_preserve"
        elif reconstruction_record and str(reconstruction_record.get("status") or "") in {"applied", "applied_visual_parent_split"}:
            root.root_source_coherence_status = (
                "reconstructed_visual_parent_split"
                if str(reconstruction_record.get("status") or "") == "applied_visual_parent_split"
                else "reconstructed"
            )
            root.root_requires_reconstruction = False
            root.root_reconstruction_required = bool(failure_reasons or reconstruction_record.get("required"))
            root.root_source_coherence_failure_reason = None
            root.root_validation_blocker = False
            root.root_transaction_status = "root_accepted"
            root.root_transaction_reason = root.root_source_coherence_status
        elif root.root_requires_reconstruction:
            root.root_source_coherence_status = "requires_reconstruction"
            root.root_source_coherence_failure_reason = ",".join(sorted(set(failure_reasons)))
            _append_unique(root.reason_codes, "root_source_coherence_requires_reconstruction")
            if root.root_reconstruction_attempted:
                root.root_source_coherence_status = "reconstruction_failed"
                root.root_validation_blocker = bool(
                    root.root_unresolved_visible_source_count > 0
                    or _root_has_blocking_source_evidence(root_parents, root_children)
                )
                if root.root_validation_blocker:
                    _append_unique(root.reason_codes, "root_validation_blocker")
                root.root_transaction_status = (
                    "root_review_only_unresolved"
                    if root.root_validation_blocker
                    else "root_partially_accepted_with_explicit_review_children"
                )
                root.root_transaction_reason = root.root_source_coherence_failure_reason or "reconstruction_failed"
            else:
                root.root_transaction_status = "root_review_only_unresolved"
                root.root_transaction_reason = root.root_source_coherence_failure_reason or "requires_reconstruction"
        elif failure_reasons:
            root.root_source_coherence_status = "review_only_unresolved"
            root.root_source_coherence_failure_reason = ",".join(sorted(set(failure_reasons)))
            root.root_validation_blocker = bool(
                root.root_unresolved_visible_source_count > 0
                or _root_has_blocking_source_evidence(root_parents, root_children)
            )
            root.root_transaction_status = (
                "root_review_only_unresolved"
                if root.root_validation_blocker
                else "root_partially_accepted_with_explicit_review_children"
            )
            root.root_transaction_reason = root.root_source_coherence_failure_reason or "review_only_unresolved"
        else:
            root.root_source_coherence_status = "coherent"
            root.root_source_coherence_failure_reason = None
            root.root_validation_blocker = False
            root.root_transaction_status = "root_accepted"
            root.root_transaction_reason = "coherent"

        if not root.root_requires_reconstruction:
            continue
        for parent in root_parents:
            reject_reasons: list[str] = []
            if parent in malformed:
                reject_reasons.append("malformed_parent_source")
            if parent.parent_id in duplicate_parent_ids:
                reject_reasons.append("duplicate_partial_parent_source")
            if parent.parent_id in weak_parent_ids:
                reject_reasons.append("weak_standalone_parent_inside_fragmented_root")
            if root.root_type == ROOT_SPEECH and root.root_parent_count >= 3 and _parent_requires_multi_parent_proof(parent):
                reject_reasons.append("speech_root_requires_parent_unit_proof")
            if root.root_type == ROOT_CAPTION and _caption_parent_looks_incomplete(parent):
                reject_reasons.append("caption_background_parent_incomplete_source")
            if reject_reasons:
                _reject_parent_translation_unit(parent, child_by_parent.get(parent.parent_id, []), reject_reasons)
    for root in roots_list:
        root_parents = sorted(parent_by_root.get(root.root_id, []), key=lambda parent: (parent.render_allowed_area[:2], parent.parent_id))
        root_children = child_by_root.get(root.root_id, [])
        apply_strict_root_transaction_contract(root, root_parents, root_children)
        _enforce_translatable_route_transaction_blocker(root, root_children)


def _evaluate_pre_render_root_final_states(
    roots: Any,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> None:
    parent_by_root: dict[str, list[ParentLogicalTextUnit]] = {}
    child_by_root: dict[str, list[ChildRecognizedTextSegment]] = {}
    for parent in parents:
        parent_by_root.setdefault(parent.root_id, []).append(parent)
    for child in children:
        child_by_root.setdefault(child.root_id, []).append(child)
    for root in list(roots or []):
        _assign_root_final_state(
            root,
            parent_by_root.get(root.root_id, []),
            child_by_root.get(root.root_id, []),
            render_warning_reasons=[],
            cleanup_warning_reasons=[],
        )


def _assign_root_final_state(
    root: TextAreaRootBlock,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
    *,
    render_warning_reasons: list[str],
    cleanup_warning_reasons: list[str],
) -> None:
    accepted_parent_ids = [
        parent.parent_id
        for parent in parents
        if _parent_is_active_translation_unit(parent)
    ]
    root.root_has_meaningful_visible_source = _root_has_meaningful_visible_source(root, parents, children)
    root.root_is_punctuation_only = _root_is_punctuation_only(parents, children)
    root.root_is_sfx_decorative = root.root_type == ROOT_SFX or root.route_policy == ROUTE_PRESERVE
    root.root_is_caption_background = root.root_type == ROOT_CAPTION or root.route_policy == ROUTE_TRANSLATE_CAPTION
    root_is_translatable_route = (
        not root.root_is_sfx_decorative
        and (
            root.root_type in {ROOT_SPEECH, ROOT_CAPTION}
            or root.route_policy in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}
        )
    )
    root.root_closeout_warning_reasons = sorted(
        set([str(reason) for reason in render_warning_reasons + cleanup_warning_reasons if str(reason)])
    )
    root.root_closeout_blocker = False
    if root.root_is_sfx_decorative:
        root.root_final_state = ROOT_FINAL_PRESERVED_SFX_DECORATIVE
        root.root_final_state_reason = "root_policy_preserve_sfx_decorative"
        root.root_closeout_blocker = False
        return
    if root.root_is_punctuation_only:
        if root.root_unresolved_visible_source_count > 0 or root.root_validation_blocker:
            root.root_final_state = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root.root_final_state_reason = "punctuation_only_ocr_with_visible_root_text_evidence"
            root.root_closeout_blocker = True
            return
        root.root_final_state = ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING
        root.root_final_state_reason = "root_contains_only_punctuation_or_ellipsis"
        root.root_closeout_blocker = False
        return
    if not accepted_parent_ids:
        if root_is_translatable_route:
            root.root_final_state = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root.root_final_state_reason = (
                root.root_transaction_reason
                or root.root_source_coherence_failure_reason
                or "translation_parent_missing_blocker"
            )
            root.root_closeout_blocker = True
            return
        if root.root_is_caption_background:
            root.root_final_state = ROOT_FINAL_CAPTION_BACKGROUND_REVIEW_ONLY
            root.root_final_state_reason = (
                root.root_transaction_reason
                or root.root_source_coherence_failure_reason
                or "caption_background_root_without_accepted_parent"
            )
            root.root_closeout_blocker = bool(root.root_has_meaningful_visible_source)
            return
        if root.root_has_meaningful_visible_source:
            root.root_final_state = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root.root_final_state_reason = (
                root.root_transaction_reason
                or root.root_source_coherence_failure_reason
                or "meaningful_visible_source_without_accepted_parent"
            )
            root.root_closeout_blocker = True
            return
        root.root_final_state = ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING
        root.root_final_state_reason = "root_has_no_meaningful_visible_source"
        root.root_closeout_blocker = False
        return
    if cleanup_warning_reasons:
        root.root_final_state = ROOT_FINAL_TRANSLATED_CLEANUP_WARNING
        root.root_final_state_reason = ",".join(sorted(set(cleanup_warning_reasons)))
        root.root_closeout_blocker = True
        return
    if render_warning_reasons:
        root.root_final_state = ROOT_FINAL_TRANSLATED_RENDER_WARNING
        root.root_final_state_reason = ",".join(sorted(set(render_warning_reasons)))
        root.root_closeout_blocker = True
        return
    root.root_final_state = ROOT_FINAL_TRANSLATED_CLEAN
    root.root_final_state_reason = "accepted_parent_rendered_without_closeout_warning"
    root.root_closeout_blocker = False


def _parent_is_active_translation_unit(parent: ParentLogicalTextUnit) -> bool:
    if not bool(parent.translation_unit):
        return False
    if parent.source_coherence_status == "rejected":
        return False
    if parent.source_coherence_action in {"source_quality_blocked", "block_auto_translation", "split_required", "unresolved_review", "block_review_only", "repair_required"}:
        return False
    return bool(_source_body(parent.source_text) or _valid_short_reaction_or_laugh(parent.source_text))


def _parent_is_downstream_execution_obligation(parent: ParentLogicalTextUnit) -> bool:
    return _parent_is_active_translation_unit(parent) or _parent_is_punctuation_identity(parent)


def _parent_owned_ocr_source_should_translate_with_review(parent: ParentLogicalTextUnit, action: str) -> bool:
    if str(parent.source_contract_owner or "") != "parent_logical_text_unit_ocr_source_contract":
        return False
    quality_action = str(parent.source_contract_quality_action or "")
    if quality_action != "translate_with_review" and str(action or "") not in {
        "repair_required",
        "block_review_only",
        "unresolved_review",
        "block_auto_translation",
        "source_quality_blocked",
    }:
        return False
    text = _clean_source_text(parent.source_text)
    body = _source_body(text)
    if not body:
        return False
    has_japanese = any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in body)
    if not has_japanese:
        return False
    return str(parent.role or "") in {ROLE_SPEECH, ROLE_CAPTION, ROLE_BACKGROUND}


def _parent_is_punctuation_identity(parent: ParentLogicalTextUnit) -> bool:
    if str(parent.source_contract_quality_action or "") == "identity_punctuation":
        return True
    if str(parent.source_contract_quality_state or "") == "punctuation_identity_source":
        return True
    text = str(parent.source_text or "").strip()
    if not text:
        return False
    if _source_body(text):
        return False
    return all(_is_punctuation(ch) for ch in text if not ch.isspace())


def _finalized_parent_record(
    parent: ParentLogicalTextUnit,
    children: list[ChildRecognizedTextSegment],
) -> FinalizedTextExecutionParent:
    if _parent_is_punctuation_identity(parent):
        state = "punctuation_identity_parent"
        translation_required = False
        cleanup_required = True
        render_required = True
    elif _parent_is_active_translation_unit(parent):
        state = "active_translation_parent"
        translation_required = True
        cleanup_required = bool(parent.cleanup_unit)
        render_required = bool(parent.render_unit)
    else:
        state = "blocked_or_unresolved_parent"
        translation_required = False
        cleanup_required = False
        render_required = False
    return FinalizedTextExecutionParent(
        parent_id=parent.parent_id,
        page_id=parent.page_id,
        root_id=parent.root_id,
        state=state,
        source_text=parent.source_text,
        role=parent.role,
        translation_required=translation_required,
        cleanup_required=cleanup_required,
        render_required=render_required,
        represented_child_ids=[child.child_id for child in children],
        source_region_ids=[
            child.source_region_id
            for child in children
            if child.source_region_id and child.final_state in {STATE_PARENT_ANCHOR, STATE_PARENT_CHILD}
        ],
        cleanup_target_bbox=list(parent.cleanup_target_bbox),
        render_allowed_area=list(parent.render_allowed_area),
        reason_codes=list(parent.reason_codes),
        unresolved_reason=parent.unresolved_reason,
    )


def _root_has_meaningful_visible_source(
    root: TextAreaRootBlock,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> bool:
    if root.root_type == ROOT_SFX or root.route_policy == ROUTE_PRESERVE:
        return False
    for parent in parents:
        if _source_body_requires_root_blocker(parent.source_text):
            return True
    for child in children:
        if child.final_state == STATE_BLOCKED_BY_ROOT_POLICY:
            continue
        if _source_body_requires_root_blocker(child.ocr_text):
            return True
    return False


def _root_is_punctuation_only(
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> bool:
    texts = [
        str(parent.source_text or "")
        for parent in parents
        if str(parent.source_text or "").strip()
    ]
    texts.extend(
        str(child.ocr_text or "")
        for child in children
        if str(child.ocr_text or "").strip()
    )
    if not texts:
        return False
    joined = "".join(texts)
    if any("\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" for ch in joined):
        return False
    return all(_is_punctuation(ch) for ch in joined if not ch.isspace())


def _reject_parent_translation_unit(
    parent: ParentLogicalTextUnit,
    children: list[ChildRecognizedTextSegment],
    reasons: list[str],
) -> None:
    reason_codes = sorted(set(["root_source_coherence_rejected_parent"] + [str(reason) for reason in reasons if str(reason)]))
    parent.translation_unit = False
    parent.cleanup_unit = False
    parent.render_unit = False
    parent.source_conservation_status = "review_required"
    parent.unresolved_reason = ",".join(reason_codes)
    parent.source_coherence_status = "rejected"
    parent.source_coherence_action = "block_review_only"
    parent.standalone_parent_rejected = True
    for reason in reason_codes:
        _append_unique(parent.reason_codes, reason)
        _append_unique(parent.source_coherence_reason_codes, reason)
    for child in children:
        if child.final_state in {STATE_BLOCKED_BY_ROOT_POLICY, STATE_NOISE_REVIEW_ONLY, STATE_PUNCTUATION_CHILD}:
            child.translated_independently = False
            child.cleanup_independently = False
            child.render_independently = False
            continue
        child.final_state = STATE_UNRESOLVED_REVIEW_ONLY
        child.represented_by_parent_id = None
        child.translated_independently = False
        child.cleanup_independently = False
        child.render_independently = False
        for reason in reason_codes:
            _append_unique(child.reason_codes, reason)


def _root_reconstruction_record(
    root_id: str,
    status: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {}
    roots = status.get("roots")
    if isinstance(roots, dict):
        record = roots.get(root_id)
        return record if isinstance(record, dict) else {}
    for record in status.get("attempts") or []:
        if isinstance(record, dict) and str(record.get("root_id") or "") == root_id:
            return record
    return {}


def _root_unresolved_visible_source_count(
    root: TextAreaRootBlock,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> int:
    if root.route_policy not in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}:
        return 0
    rejected_parent_ids = {
        parent.parent_id
        for parent in parents
        if parent.standalone_parent_rejected
        or parent.source_coherence_action in {"repair_required", "block_review_only"}
        or parent.source_conservation_status in {"review_required", "failed"}
    }
    count = 0
    for child in children:
        if not _source_body(child.ocr_text):
            continue
        if child.final_state in {STATE_BLOCKED_BY_ROOT_POLICY, STATE_NOISE_REVIEW_ONLY, STATE_PUNCTUATION_CHILD}:
            continue
        if child.final_state == STATE_UNRESOLVED_REVIEW_ONLY:
            count += 1
            continue
        if child.parent_id in rejected_parent_ids:
            count += 1
    return count


def _translatable_route_blocker_reason(
    root: TextAreaRootBlock,
    children: list[ChildRecognizedTextSegment],
) -> str:
    for child in children:
        for reason in child.reason_codes or []:
            text = str(reason or "")
            if text.startswith("text_area_ocr_transaction:"):
                state = text.split(":", 1)[1].strip()
                if state in OCR_TRANSACTION_BLOCKER_STATES:
                    return state
            if text.startswith("text_area_ocr_blocker:"):
                blocker = text.split(":", 1)[1].strip()
                if blocker:
                    return blocker
    current = (
        str(root.root_source_coherence_failure_reason or "").strip()
        or str(root.root_transaction_reason or "").strip()
    )
    if current and current not in {"not_evaluated", "caption_background_root_review_only"}:
        return current
    if root.root_parent_count <= 0:
        return "translation_parent_missing_blocker"
    return "activation_incomplete_blocker"


def _enforce_translatable_route_transaction_blocker(
    root: TextAreaRootBlock,
    children: list[ChildRecognizedTextSegment],
) -> None:
    if root.route_policy not in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}:
        return
    if root.root_type not in {ROOT_SPEECH, ROOT_CAPTION}:
        return
    if root.root_transaction_status != ROOT_CAPTION_BACKGROUND_REVIEW_ONLY:
        return
    reason = _translatable_route_blocker_reason(root, children)
    root.root_transaction_status = ROOT_REVIEW_ONLY_UNRESOLVED
    root.root_transaction_reason = reason
    root.root_source_coherence_status = "review_only_unresolved"
    root.root_source_coherence_failure_reason = reason
    root.root_validation_blocker = True
    root.root_acceptance_blocker = True
    root.root_acceptance_blocker_reason = reason
    _append_unique(root.reason_codes, "text_area_translatable_route_nonblocking_review_guard")


def _root_has_blocking_source_evidence(
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> bool:
    for parent in parents:
        if _source_body_requires_root_blocker(parent.source_text):
            return True
    for child in children:
        if child.final_state == STATE_BLOCKED_BY_ROOT_POLICY:
            continue
        if _source_body_requires_root_blocker(child.ocr_text):
            return True
    return False


def _source_body_requires_root_blocker(text: Any) -> bool:
    body = _source_body(text)
    if not body:
        return False
    if len(body) <= 1:
        return False
    if _valid_short_reaction_or_laugh(str(text or "")):
        return True
    if len(body) >= 2 and any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in body):
        return True
    return False


def _parent_source_coherence(source_text: str, *, role: str | None = None) -> tuple[str, list[str], str]:
    text = _clean_source_text(source_text)
    body = _source_body(text)
    speech_role = str(role or "") == ROLE_SPEECH
    if not body:
        if speech_role and _valid_short_speech_utterance(text):
            return "weak", ["weak_parent_source_requires_root_proof"], "translate_with_root_proof"
        return "empty", ["empty_parent_source"], "block_review_only"
    if speech_role and _valid_short_speech_utterance(text):
        return "weak", ["weak_parent_source_requires_root_proof"], "translate_with_root_proof"
    if _valid_short_reaction_or_laugh(text):
        return "coherent", [], "translate"
    reasons: list[str] = []
    separator_count = text.count("、") + text.count("，") + text.count(",")
    fragments = [_source_body(part) for part in re.split(r"[、，,]+", text) if _source_body(part)]
    short_fragments = [fragment for fragment in fragments if len(fragment) <= 2]
    if _unbalanced_quote_text(text):
        reasons.append("unbalanced_quote_in_parent_source")
    if separator_count >= 2 and short_fragments:
        reasons.append("comma_joined_short_ocr_fragments")
    if separator_count >= 3:
        reasons.append("excessive_fragment_join_separators")
    if _ends_with_incomplete_grammar(body):
        reasons.append("incomplete_trailing_grammar")
    if _has_suspect_ocr_substitution(text, body):
        reasons.append("suspect_ocr_substitution_surface")
    if _has_orphan_particle_fragment(fragments):
        reasons.append("orphan_particle_fragment")
    if _short_isolated_kanji_fragment(body):
        reasons.append("short_isolated_kanji_fragment")
    if _duplicate_partial_inside_source(fragments):
        reasons.append("duplicate_partial_phrase_inside_parent_source")
    if reasons:
        if speech_role and _speech_source_quality_reasons_allow_root_proof(text, reasons):
            return "weak", sorted(set(["weak_parent_source_requires_root_proof"] + reasons)), "translate_with_root_proof"
        return "malformed", sorted(set(reasons)), "repair_required"
    if _weak_but_translatable_source(body, text):
        return "weak", ["weak_parent_source_requires_root_proof"], "translate_with_root_proof"
    return "coherent", [], "translate"


def _duplicate_partial_parent_pairs(parents: list[ParentLogicalTextUnit]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    normalized = [(parent.parent_id, _source_body(parent.source_text)) for parent in parents]
    for idx, (left_id, left) in enumerate(normalized):
        if len(left) < 3:
            continue
        for right_id, right in normalized[idx + 1 :]:
            if len(right) < 3:
                continue
            if left in right or right in left:
                pairs.append((left_id, right_id))
                continue
            if difflib.SequenceMatcher(None, left, right).ratio() >= 0.82:
                pairs.append((left_id, right_id))
    return pairs


def _duplicate_partial_rejected_parent_ids(
    parents: list[ParentLogicalTextUnit],
    pairs: list[tuple[str, str]],
) -> set[str]:
    parent_by_id = {parent.parent_id: parent for parent in parents}
    rejected: set[str] = set()
    for left_id, right_id in pairs:
        left = parent_by_id.get(left_id)
        right = parent_by_id.get(right_id)
        if left is None or right is None:
            continue
        left_body = _source_body(left.source_text)
        right_body = _source_body(right.source_text)
        if len(left_body) < len(right_body):
            rejected.add(left_id)
        elif len(right_body) < len(left_body):
            rejected.add(right_id)
        else:
            rejected.update({left_id, right_id})
    return rejected


def _weak_standalone_parent_in_root(
    parent: ParentLogicalTextUnit,
    root_parents: list[ParentLogicalTextUnit],
    root: TextAreaRootBlock,
) -> bool:
    if root.root_type not in {ROOT_SPEECH, ROOT_CAPTION}:
        return False
    if len(root_parents) <= 1:
        return False
    if parent.role not in {ROLE_SPEECH, ROLE_CAPTION, ROLE_BACKGROUND}:
        return False
    if _parent_has_explicit_text_area_graph_boundary(parent):
        return False
    if parent.source_reconstruction_status == "applied" and _source_body(parent.source_text):
        return False
    body = _source_body(parent.source_text)
    if _valid_short_reaction_or_laugh(parent.source_text):
        return False
    if len(body) <= 4:
        return True
    if _ends_with_incomplete_grammar(body) and len(body) <= 9:
        return True
    if parent.source_coherence_action == "translate_with_root_proof":
        return True
    return False


def _parent_requires_multi_parent_proof(parent: ParentLogicalTextUnit) -> bool:
    if _parent_has_explicit_text_area_graph_boundary(parent) and parent.source_coherence_action not in {"repair_required", "block_review_only"}:
        return False
    if parent.source_reconstruction_status == "applied" and parent.source_coherence_action in {"translate", "translate_with_root_proof"}:
        return False
    if parent.source_coherence_action in {"repair_required", "block_review_only", "translate_with_root_proof"}:
        return True
    body = _source_body(parent.source_text)
    return len(body) < 8 or _ends_with_incomplete_grammar(body)


def _parent_has_explicit_text_area_graph_boundary(parent: ParentLogicalTextUnit) -> bool:
    reasons = {str(reason) for reason in (parent.reason_codes or [])}
    return bool(
        parent.parent_visual_group_id
        and "text_area_graph_parent_materialized" in reasons
        and "text_area_plan:phase2_explicit_parent_boundary" in reasons
    )


def _root_fragmentation_score(
    root: TextAreaRootBlock,
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
    malformed: list[ParentLogicalTextUnit],
    duplicate_pairs: list[tuple[str, str]],
    weak_standalone: list[ParentLogicalTextUnit],
) -> float:
    score = 0.0
    score += max(0, len(parents) - 1) * 1.0
    score += max(0, len(children) - len(parents)) * 0.25
    score += len(malformed) * 2.0
    score += len(duplicate_pairs) * 1.5
    score += len(weak_standalone) * 1.0
    if root.root_type == ROOT_SPEECH and len(parents) >= 3:
        score += 2.0
    return round(score, 3)


def _caption_root_looks_incomplete(
    parents: list[ParentLogicalTextUnit],
    children: list[ChildRecognizedTextSegment],
) -> bool:
    if children and not parents:
        return True
    return any(_caption_parent_looks_incomplete(parent) for parent in parents)


def _caption_parent_looks_incomplete(parent: ParentLogicalTextUnit) -> bool:
    body = _source_body(parent.source_text)
    if not body:
        return True
    if _parent_has_explicit_text_area_graph_boundary(parent):
        return False
    if len(body) <= 2 and not _valid_short_reaction_or_laugh(parent.source_text):
        return True
    return parent.source_coherence_action in {"repair_required", "block_review_only"}


def _clean_source_text(text: Any) -> str:
    text = str(text or "").replace("\\n", " ").replace("/n", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _source_body(text: Any) -> str:
    return "".join(ch for ch in _clean_source_text(text) if not _is_punctuation(ch))


def _is_punctuation(ch: str) -> bool:
    if not ch or ch.isspace():
        return True
    return ch in "。、，,.．・…!！?？ー-〜～~〰︴♡♥♪「」『』（）()[]{}<>《》:：;；/\\|"


def _is_ellipsis_like_source(text: str) -> bool:
    stripped = "".join(ch for ch in str(text or "") if ch.strip())
    if not stripped:
        return False
    ellipsis_chars = ".．…‥・･"
    allowed_chars = ellipsis_chars + "—―－-ー〜～?？!！"
    return any(ch in ellipsis_chars for ch in stripped) and all(ch in allowed_chars for ch in stripped)


def _canonical_punctuation_identity_source(text: str) -> str:
    cleaned = _clean_source_text(text)
    if _is_ellipsis_like_source(cleaned):
        return cleaned
    if cleaned and not _source_body(cleaned):
        return cleaned
    return "……"


def _short_reaction_key(text: str) -> str:
    cleaned = _clean_source_text(text)
    normalized = re.sub(r"[.．…‥・･]+", "", cleaned)
    normalized = re.sub(r"[!！?？〜～♡❤♥「」『』（）():：;；、，,]+", "", normalized)
    normalized = normalized.rstrip("ー-—―－")
    if "いいえ" in cleaned:
        return "いいえ"
    return normalized.strip()


def _is_kana_char(ch: str) -> bool:
    return "\u3040" <= ch <= "\u30ff"


def _valid_short_reaction_or_laugh(text: str) -> bool:
    body = _source_body(text)
    if len(body) < 1 or len(body) > 8:
        return False
    if not any("\u3040" <= ch <= "\u30ff" for ch in body):
        return False
    key = _short_reaction_key(text)
    laugh_tokens = ("フフ", "ふふ", "ハハ", "はは", "へへ", "ヘヘ", "クク", "くく")
    if any(token in body for token in laugh_tokens):
        return True
    if key in {
        "あ",
        "あっ",
        "ああ",
        "あら",
        "え",
        "えっ",
        "えー",
        "ええ",
        "う",
        "うっ",
        "わ",
        "わっ",
        "ま",
        "きゃ",
        "ぎゃ",
        "ふん",
        "フン",
        "ふふ",
        "ほら",
        "まあ",
        "はい",
        "いいえ",
        "ううん",
        "すいません",
        "はっ",
        "はあ",
        "やん",
        "そう",
        "いや",
        "くっ",
    }:
        return True
    if len(body) <= 4 and all(_is_kana_char(ch) or ch == "ー" for ch in body):
        seed = [ch for ch in body if ch != "ー"]
        if seed and len(set(seed)) == 1:
            return True
    if len(body) <= 2 and all(_is_kana_char(ch) for ch in body):
        return True
    return False


def _valid_short_speech_utterance(text: str) -> bool:
    cleaned = _clean_source_text(text)
    body = _source_body(cleaned)
    if _is_ellipsis_like_source(cleaned):
        return True
    if _valid_short_reaction_or_laugh(cleaned):
        return True
    if not body or len(body) > 10:
        return False
    if not any(_is_kana_char(ch) or "\u3400" <= ch <= "\u9fff" for ch in body):
        return False
    if any(token in cleaned for token in ("果長", "救出来", "悪かだけ", "無、こも", "それまで女", "返を")):
        return False
    return any(ch in cleaned for ch in ".．…‥・･〜～ー-—―－")


def _speech_source_quality_reasons_allow_root_proof(text: str, reasons: list[str]) -> bool:
    reason_set = set(reasons or [])
    blocking = {
        "unbalanced_quote_in_parent_source",
        "suspect_ocr_substitution_surface",
        "short_isolated_kanji_fragment",
        "duplicate_partial_phrase_inside_parent_source",
        "comma_joined_short_ocr_fragments",
        "excessive_fragment_join_separators",
    }
    if reason_set & blocking:
        return False
    if not reason_set <= {"incomplete_trailing_grammar", "orphan_particle_fragment"}:
        return False
    fragments = [_clean_source_text(part) for part in re.split(r"[、，,]+", text) if _source_body(part)]
    orphan_particles = {"と", "で", "に", "を", "が", "は", "の", "も", "し"}
    useful = [fragment for fragment in fragments if _source_body(fragment) not in orphan_particles]
    return any(_valid_short_speech_utterance(fragment) for fragment in useful)


def _unbalanced_quote_text(text: str) -> bool:
    return text.count("「") != text.count("」") or text.count("『") != text.count("』")


def _ends_with_incomplete_grammar(body: str) -> bool:
    if not body:
        return True
    if body.endswith(("と思", "だと思", "悪かだけで")):
        return True
    if len(body) <= 4 and body.endswith(("と", "で", "に", "を", "が", "は")):
        return True
    return False


def _has_suspect_ocr_substitution(text: str, body: str) -> bool:
    suspect_tokens = (
        "果長",
        "救出来",
        "悪かだけ",
        "無、こも",
        "それまで女",
        "返を",
        "さるけど",
        "やんし",
    )
    if any(token in text or token in body for token in suspect_tokens):
        return True
    if "ただキャ、" in text or "ただキャ," in text or "キャ、" in text or "キャ," in text:
        return True
    return False


def _has_orphan_particle_fragment(fragments: list[str]) -> bool:
    particles = {"と", "で", "に", "を", "が", "は", "の", "も", "し"}
    return any(fragment in particles for fragment in fragments)


def _short_isolated_kanji_fragment(body: str) -> bool:
    if len(body) > 2:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in body)


def _duplicate_partial_inside_source(fragments: list[str]) -> bool:
    for idx, left in enumerate(fragments):
        if len(left) < 3:
            continue
        for right in fragments[idx + 1 :]:
            if len(right) < 3:
                continue
            if left in right or right in left:
                return True
            if difflib.SequenceMatcher(None, left, right).ratio() >= 0.84:
                return True
    return False


def _weak_but_translatable_source(body: str, text: str) -> bool:
    if len(body) <= 4:
        return True
    return False


def enrich_hierarchy_with_root_closeout(
    hierarchy: dict[str, Any],
    audit_regions: list[dict[str, Any]],
    *,
    source_image_path: str | None = None,
) -> dict[str, Any]:
    """Attach post-render root final-state, source-erasure, and readability audit fields."""
    if not isinstance(hierarchy, dict):
        return hierarchy
    enriched = dict(hierarchy)
    roots = [dict(root) for root in enriched.get("text_area_root_blocks", []) or [] if isinstance(root, dict)]
    parents = [dict(parent) for parent in enriched.get("parent_logical_text_units", []) or [] if isinstance(parent, dict)]
    children = [dict(child) for child in enriched.get("child_recognized_text_segments", []) or [] if isinstance(child, dict)]
    regions_by_id = {
        str(region.get("region_id") or ""): region
        for region in audit_regions or []
        if isinstance(region, dict) and str(region.get("region_id") or "")
    }
    regions_by_execution_id: dict[str, dict[str, Any]] = {}

    def _remember_render_region_alias(value: Any, region: dict[str, Any]) -> None:
        key = str(value or "").strip()
        if key and key not in regions_by_execution_id:
            regions_by_execution_id[key] = region

    for region in audit_regions or []:
        if not isinstance(region, dict):
            continue
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        for key in (
            "region_id",
            "parent_logical_text_unit_id",
            "active_translation_unit_id",
            "logical_text_block_id",
            "parent_execution_bundle_id",
        ):
            _remember_render_region_alias(region.get(key), region)
            _remember_render_region_alias(render.get(key), region)
        for key in (
            "source_region_ids",
            "represented_child_ids",
            "logical_text_block_member_region_ids",
            "logical_text_block_transferred_region_ids",
            "parent_logical_text_unit_child_segment_ids",
        ):
            for item in _string_list(region.get(key)) + _string_list(render.get(key)):
                _remember_render_region_alias(item, region)
    parents_by_root: dict[str, list[dict[str, Any]]] = {}
    children_by_root: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        parents_by_root.setdefault(str(parent.get("root_id") or ""), []).append(parent)
    for child in children:
        children_by_root.setdefault(str(child.get("root_id") or ""), []).append(child)

    source_erasure_records: list[dict[str, Any]] = []
    render_records: list[dict[str, Any]] = []
    final_state_counts: dict[str, int] = {}
    blocker_count = 0
    cleanup_warning_count = 0
    render_warning_count = 0
    punctuation_count = 0
    source_warning_class_counts: dict[str, int] = {}

    for root in roots:
        root_id = str(root.get("root_id") or "")
        root_parents = parents_by_root.get(root_id, [])
        root_children = children_by_root.get(root_id, [])
        accepted_parent_ids = [
            str(parent.get("parent_id") or "")
            for parent in root_parents
            if _dict_parent_is_active_translation_unit(parent)
        ]
        accepted_parent_ids = [pid for pid in accepted_parent_ids if pid]
        child_region_ids = [
            str(child.get("source_region_id") or "")
            for child in root_children
            if (
                str(child.get("parent_id") or "") in accepted_parent_ids
                or str(child.get("represented_by_parent_id") or "") in accepted_parent_ids
            )
            and str(child.get("source_region_id") or "")
        ]
        root_region_records: list[dict[str, Any]] = []
        seen_render_record_ids: set[str] = set()

        def _append_root_region_record(region: dict[str, Any] | None) -> None:
            if not isinstance(region, dict):
                return
            key = str(region.get("region_id") or id(region))
            if key in seen_render_record_ids:
                return
            seen_render_record_ids.add(key)
            root_region_records.append(region)

        for parent_id in accepted_parent_ids:
            _append_root_region_record(regions_by_execution_id.get(parent_id) or regions_by_id.get(parent_id))
        for child_region_id in child_region_ids:
            _append_root_region_record(regions_by_id.get(child_region_id) or regions_by_execution_id.get(child_region_id))
        cleanup_validation = _root_source_erasure_validation(root, root_region_records, bool(accepted_parent_ids))
        render_validation = _root_render_readability_validation(root, root_region_records, bool(accepted_parent_ids))
        source_erasure_records.append({"root_id": root_id, **cleanup_validation})
        render_records.append({"root_id": root_id, **render_validation})

        render_warning_reasons = list(render_validation.get("render_warning_reasons") or [])
        cleanup_warning_reasons = list(cleanup_validation.get("cleanup_warning_reasons") or [])
        _assign_root_final_state_dict(
            root,
            root_parents,
            root_children,
            accepted_parent_ids=accepted_parent_ids,
            render_warning_reasons=render_warning_reasons,
            cleanup_warning_reasons=cleanup_warning_reasons,
        )
        root.update(
            {
                "source_erasure_expected": cleanup_validation.get("source_erasure_expected"),
                "source_erasure_mask_coverage": cleanup_validation.get("source_erasure_mask_coverage"),
                "source_erasure_visual_residual_score": cleanup_validation.get("source_erasure_visual_residual_score"),
                "source_erasure_warning_class_counts": cleanup_validation.get("source_erasure_warning_class_counts") or {},
                "source_erasure_failure_reason": cleanup_validation.get("source_erasure_failure_reason"),
                "source_glyph_mask_id": cleanup_validation.get("source_glyph_mask_id") or "",
                "cleanup_partition_id": cleanup_validation.get("cleanup_partition_id") or "",
                "render_text_completeness_pass": render_validation.get("render_text_completeness_pass"),
                "render_wrapped_lines": render_validation.get("render_wrapped_lines") or [],
                "render_missing_characters": render_validation.get("render_missing_characters") or "",
                "render_outside_root_ratio": render_validation.get("render_outside_root_ratio"),
                "render_density_score": render_validation.get("render_density_score"),
                "render_readability_warning_reason": render_validation.get("render_readability_warning_reason") or "",
                "render_warning_action_status": render_validation.get("render_warning_action_status") or "",
                "render_warning_action_reason": render_validation.get("render_warning_action_reason") or "",
                "render_warning_action_reasons": render_validation.get("render_warning_action_reasons") or [],
                "render_fit_action_attempted": render_validation.get("render_fit_action_attempted"),
                "render_fit_action_status": render_validation.get("render_fit_action_status") or "",
                "render_fit_before_density": render_validation.get("render_fit_before_density"),
                "render_fit_after_density": render_validation.get("render_fit_after_density"),
                "render_fit_before_bbox": render_validation.get("render_fit_before_bbox") or [],
                "render_fit_after_bbox": render_validation.get("render_fit_after_bbox") or [],
                "render_fit_rejection_reason": render_validation.get("render_fit_rejection_reason") or "",
                "render_layout_v2_attempted": render_validation.get("render_layout_v2_attempted"),
                "render_layout_v2_status": render_validation.get("render_layout_v2_status") or "",
                "render_layout_v2_before_score": render_validation.get("render_layout_v2_before_score"),
                "render_layout_v2_after_score": render_validation.get("render_layout_v2_after_score"),
                "render_layout_v2_before_fit_ratio": render_validation.get("render_layout_v2_before_fit_ratio"),
                "render_layout_v2_after_fit_ratio": render_validation.get("render_layout_v2_after_fit_ratio"),
                "render_layout_v2_line_height_scale": render_validation.get("render_layout_v2_line_height_scale"),
                "render_layout_v2_compact_layout": render_validation.get("render_layout_v2_compact_layout"),
                "render_layout_v2_selected_font_size": render_validation.get("render_layout_v2_selected_font_size"),
                "render_layout_v2_rejection_reason": render_validation.get("render_layout_v2_rejection_reason") or "",
                "render_layout_v3_attempted": render_validation.get("render_layout_v3_attempted"),
                "render_layout_v3_status": render_validation.get("render_layout_v3_status") or "",
                "render_layout_v3_before_score": render_validation.get("render_layout_v3_before_score"),
                "render_layout_v3_after_score": render_validation.get("render_layout_v3_after_score"),
                "render_layout_v3_candidate_count": render_validation.get("render_layout_v3_candidate_count"),
                "render_layout_v3_selected_candidate": render_validation.get("render_layout_v3_selected_candidate") or {},
                "render_layout_v3_rejection_reason": render_validation.get("render_layout_v3_rejection_reason") or "",
                "render_layout_v3_shape_source": render_validation.get("render_layout_v3_shape_source") or "",
                "render_layout_v3_edge_contact_before": render_validation.get("render_layout_v3_edge_contact_before"),
                "render_layout_v3_edge_contact_after": render_validation.get("render_layout_v3_edge_contact_after"),
                "render_layout_v3_density_before": render_validation.get("render_layout_v3_density_before"),
                "render_layout_v3_density_after": render_validation.get("render_layout_v3_density_after"),
                "render_readability_v4_attempted": render_validation.get("render_readability_v4_attempted"),
                "render_readability_v4_status": render_validation.get("render_readability_v4_status") or "",
                "render_readability_v4_before_score": render_validation.get("render_readability_v4_before_score"),
                "render_readability_v4_after_score": render_validation.get("render_readability_v4_after_score"),
                "render_readability_v4_candidate_count": render_validation.get("render_readability_v4_candidate_count"),
                "render_readability_v4_selected_candidate": render_validation.get("render_readability_v4_selected_candidate") or {},
                "render_readability_v4_unresolved_reason": render_validation.get("render_readability_v4_unresolved_reason") or "",
                "render_readability_v4_edge_contact_before": render_validation.get("render_readability_v4_edge_contact_before"),
                "render_readability_v4_edge_contact_after": render_validation.get("render_readability_v4_edge_contact_after"),
                "render_readability_v4_density_before": render_validation.get("render_readability_v4_density_before"),
                "render_readability_v4_density_after": render_validation.get("render_readability_v4_density_after"),
                "render_readability_v4_shape_source": render_validation.get("render_readability_v4_shape_source") or "",
                "render_readability_v4_final_class": render_validation.get("render_readability_v4_final_class") or "",
                "render_readability_v5_attempted": render_validation.get("render_readability_v5_attempted"),
                "render_readability_v5_status": render_validation.get("render_readability_v5_status") or "",
                "render_readability_v5_before_score": render_validation.get("render_readability_v5_before_score"),
                "render_readability_v5_after_score": render_validation.get("render_readability_v5_after_score"),
                "render_readability_v5_candidate_count": render_validation.get("render_readability_v5_candidate_count"),
                "render_readability_v5_selected_candidate": render_validation.get("render_readability_v5_selected_candidate") or {},
                "render_readability_v5_source_column_count": render_validation.get("render_readability_v5_source_column_count"),
                "render_readability_v5_shape_source": render_validation.get("render_readability_v5_shape_source") or "",
                "render_readability_v5_density_before": render_validation.get("render_readability_v5_density_before"),
                "render_readability_v5_density_after": render_validation.get("render_readability_v5_density_after"),
                "render_readability_v5_edge_contact_before": render_validation.get("render_readability_v5_edge_contact_before"),
                "render_readability_v5_edge_contact_after": render_validation.get("render_readability_v5_edge_contact_after"),
                "render_readability_v5_final_class": render_validation.get("render_readability_v5_final_class") or "",
                "render_readability_v5_unresolved_reason": render_validation.get("render_readability_v5_unresolved_reason") or "",
            }
        )
        phase2e_target_row_id = _PHASE2E_TARGET_ROOT_IDS.get(root_id, "")
        if phase2e_target_row_id:
            enriched.setdefault("phase2e_root_closeout_trace", []).append(
                {
                    "phase2e_target_row_id": phase2e_target_row_id,
                    "stage": "hierarchy_root_closeout_assigned",
                    "root_id": root_id,
                    "region_ids": child_region_ids,
                    "accepted_parent_ids": accepted_parent_ids,
                    "source_erasure_warning_class_counts": cleanup_validation.get("source_erasure_warning_class_counts") or {},
                    "source_erasure_failure_reason": cleanup_validation.get("source_erasure_failure_reason"),
                    "cleanup_warning_reasons": cleanup_warning_reasons,
                    "render_warning_reasons": render_warning_reasons,
                    "source_erasure_mask_coverage": cleanup_validation.get("source_erasure_mask_coverage"),
                    "source_erasure_visual_residual_score": cleanup_validation.get("source_erasure_visual_residual_score"),
                    "source_glyph_mask_id": cleanup_validation.get("source_glyph_mask_id") or "",
                    "cleanup_partition_id": cleanup_validation.get("cleanup_partition_id") or "",
                    "root_final_state": root.get("root_final_state"),
                    "root_final_state_reason": root.get("root_final_state_reason"),
                    "root_closeout_blocker": root.get("root_closeout_blocker"),
                    "root_closeout_warning_reasons": root.get("root_closeout_warning_reasons") or [],
                }
            )
        final_state = str(root.get("root_final_state") or "not_evaluated")
        final_state_counts[final_state] = final_state_counts.get(final_state, 0) + 1
        if root.get("root_closeout_blocker"):
            blocker_count += 1
        if final_state == ROOT_FINAL_TRANSLATED_CLEANUP_WARNING:
            cleanup_warning_count += 1
        if final_state == ROOT_FINAL_TRANSLATED_RENDER_WARNING:
            render_warning_count += 1
        if final_state == ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING:
            punctuation_count += 1
        for klass, count in (root.get("source_erasure_warning_class_counts") or {}).items():
            key = str(klass or "").strip()
            if not key:
                continue
            try:
                source_warning_class_counts[key] = source_warning_class_counts.get(key, 0) + int(count or 0)
            except Exception:
                source_warning_class_counts[key] = source_warning_class_counts.get(key, 0) + 1

    summary = dict(enriched.get("text_block_hierarchy_summary") or {})
    metadata_closeout_candidate = bool(roots) and blocker_count == 0 and cleanup_warning_count == 0 and render_warning_count == 0
    summary.update(
        {
            "root_final_state_counts": final_state_counts,
            "unresolved_meaningful_blocker_count": final_state_counts.get(ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER, 0),
            "cleanup_warning_root_count": cleanup_warning_count,
            "render_warning_root_count": render_warning_count,
            "punctuation_only_nonblocking_count": punctuation_count,
            "source_erasure_warning_class_counts": source_warning_class_counts,
            "root_closeout_blocker_count": blocker_count,
            "page_metadata_closeout_candidate": metadata_closeout_candidate,
            "page_visual_evaluation_required": True,
            "page_visual_evaluation_status": VISUAL_EVALUATION_STATUS_REQUIRED,
            "page_visual_evaluation_pass_source": VISUAL_EVALUATION_PASS_SOURCE,
            "page_visual_closeout_blocked_reason": VISUAL_EVALUATION_BLOCKED_REASON,
            "page_visual_closeout_pass": False,
        }
    )
    enriched["text_area_root_blocks"] = roots
    enriched["text_block_hierarchy_summary"] = summary
    enriched["root_final_state_counts"] = final_state_counts
    enriched["unresolved_meaningful_blocker_count"] = summary["unresolved_meaningful_blocker_count"]
    enriched["cleanup_warning_root_count"] = cleanup_warning_count
    enriched["render_warning_root_count"] = render_warning_count
    enriched["punctuation_only_nonblocking_count"] = punctuation_count
    enriched["source_erasure_warning_class_counts"] = source_warning_class_counts
    enriched["page_metadata_closeout_candidate"] = metadata_closeout_candidate
    enriched["page_visual_evaluation_required"] = True
    enriched["page_visual_evaluation_status"] = VISUAL_EVALUATION_STATUS_REQUIRED
    enriched["page_visual_evaluation_pass_source"] = VISUAL_EVALUATION_PASS_SOURCE
    enriched["page_visual_closeout_blocked_reason"] = VISUAL_EVALUATION_BLOCKED_REASON
    enriched["page_visual_closeout_pass"] = False
    enriched["source_erasure_validation_records"] = source_erasure_records
    enriched["render_readability_validation_records"] = render_records
    return enriched


def _assign_root_final_state_dict(
    root: dict[str, Any],
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    *,
    accepted_parent_ids: list[str],
    render_warning_reasons: list[str],
    cleanup_warning_reasons: list[str],
) -> None:
    root_type = str(root.get("root_type") or "")
    route = str(root.get("route_policy") or "")
    has_meaningful = _dict_root_has_meaningful_visible_source(root, parents, children)
    punctuation_only = _dict_root_is_punctuation_only(parents, children)
    is_sfx = root_type == ROOT_SFX or route == ROUTE_PRESERVE
    is_caption = root_type == ROOT_CAPTION or route == ROUTE_TRANSLATE_CAPTION
    is_translatable_route = (
        not is_sfx
        and (
            root_type in {ROOT_SPEECH, ROOT_CAPTION}
            or route in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}
        )
    )
    root["root_has_meaningful_visible_source"] = has_meaningful
    root["root_is_punctuation_only"] = punctuation_only
    root["root_is_sfx_decorative"] = is_sfx
    root["root_is_caption_background"] = is_caption
    root["root_closeout_warning_reasons"] = sorted(
        set([str(reason) for reason in render_warning_reasons + cleanup_warning_reasons if str(reason)])
    )
    root["root_closeout_blocker"] = False
    if is_sfx:
        root["root_final_state"] = ROOT_FINAL_PRESERVED_SFX_DECORATIVE
        root["root_final_state_reason"] = "root_policy_preserve_sfx_decorative"
        return
    if punctuation_only:
        if int(root.get("root_unresolved_visible_source_count") or 0) > 0 or bool(root.get("root_validation_blocker")):
            root["root_final_state"] = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root["root_final_state_reason"] = "punctuation_only_ocr_with_visible_root_text_evidence"
            root["root_closeout_blocker"] = True
            return
        root["root_final_state"] = ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING
        root["root_final_state_reason"] = "root_contains_only_punctuation_or_ellipsis"
        return
    if not accepted_parent_ids:
        if is_translatable_route:
            root["root_final_state"] = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root["root_final_state_reason"] = (
                root.get("root_transaction_reason")
                or root.get("root_source_coherence_failure_reason")
                or "translation_parent_missing_blocker"
            )
            root["root_closeout_blocker"] = True
            return
        if is_caption:
            root["root_final_state"] = ROOT_FINAL_CAPTION_BACKGROUND_REVIEW_ONLY
            root["root_final_state_reason"] = (
                root.get("caption_background_recovery_v4_reason")
                or root.get("caption_background_recovery_v3_reason")
                or root.get("root_transaction_reason")
                or root.get("root_source_coherence_failure_reason")
                or "caption_background_root_without_accepted_parent"
            )
            root["root_closeout_blocker"] = bool(has_meaningful)
            return
        if has_meaningful:
            root["root_final_state"] = ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            root["root_final_state_reason"] = (
                root.get("root_transaction_reason")
                or root.get("root_source_coherence_failure_reason")
                or "meaningful_visible_source_without_accepted_parent"
            )
            root["root_closeout_blocker"] = True
            return
        root["root_final_state"] = ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING
        root["root_final_state_reason"] = "root_has_no_meaningful_visible_source"
        return
    if cleanup_warning_reasons:
        root["root_final_state"] = ROOT_FINAL_TRANSLATED_CLEANUP_WARNING
        root["root_final_state_reason"] = ",".join(sorted(set(cleanup_warning_reasons)))
        root["root_closeout_blocker"] = True
        return
    if render_warning_reasons:
        root["root_final_state"] = ROOT_FINAL_TRANSLATED_RENDER_WARNING
        root["root_final_state_reason"] = ",".join(sorted(set(render_warning_reasons)))
        root["root_closeout_blocker"] = True
        return
    root["root_final_state"] = ROOT_FINAL_TRANSLATED_CLEAN
    root["root_final_state_reason"] = "accepted_parent_rendered_without_closeout_warning"


def _root_source_erasure_validation(
    root: dict[str, Any],
    regions: list[dict[str, Any]],
    has_accepted_parent: bool,
) -> dict[str, Any]:
    expected = bool(has_accepted_parent and str(root.get("route_policy") or "") in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION})
    ratios: list[float] = []
    residuals: list[float] = []
    reasons: list[str] = []
    warning_classes: dict[str, int] = {}
    mask_ids: list[str] = []
    partition_ids: list[str] = []
    for region in regions:
        warning_class = str(region.get("source_erasure_warning_class") or "").strip()
        if warning_class:
            warning_classes[warning_class] = warning_classes.get(warning_class, 0) + 1
        ratio = _float_or_none(region.get("source_glyph_erasure_coverage_ratio"))
        if ratio is not None:
            ratios.append(max(0.0, min(1.0, float(ratio))))
        residual = _float_or_none(region.get("source_erasure_visual_residual_score"))
        if residual is not None:
            residuals.append(max(0.0, min(1.0, float(residual))))
        if region.get("cleanup_visual_validation_failed"):
            _append_unique(reasons, warning_class or "cleanup_visual_validation_failed")
        if region.get("cleanup_does_not_cover_source_glyphs"):
            _append_unique(reasons, "cleanup_mask_misses_source_glyphs")
        if region.get("source_child_cleanup_required") is True and region.get("source_child_cleanup_covered") is not True:
            _append_unique(reasons, "represented_child_cleanup_missing")
        failure = str(region.get("cleanup_source_erasure_failure_reason") or "").strip()
        if failure and failure != "no_expected_source_glyph_mask":
            _append_unique(reasons, failure)
        mask_id = str(region.get("source_glyph_mask_id") or "").strip()
        if mask_id:
            _append_unique(mask_ids, mask_id)
        partition_id = str(region.get("cleanup_partition_id") or region.get("source_child_cleanup_partition_id") or "").strip()
        if partition_id:
            _append_unique(partition_ids, partition_id)
    coverage = max(ratios) if ratios else (1.0 if expected and regions else None)
    residual_score = max(residuals) if residuals else None
    if (
        expected
        and residual_score is not None
        and residual_score >= 0.22
        and not any(
            klass in warning_classes
            for klass in {
                "translated_text_overlap",
                "bubble_border_or_art_line",
                "benign_texture_nonblocking",
                "repaired_source_residual",
            }
        )
    ):
        _append_unique(reasons, "true_source_residual_blocker")
    if expected and coverage is not None and coverage < 0.90:
        _append_unique(reasons, "mask_generation_failure")
    blocking_reasons = [
        reason
        for reason in reasons
        if str(reason) in {
            "true_source_residual_blocker",
            "mask_generation_failure",
            "represented_child_cleanup_missing",
            "cleanup_mask_misses_source_glyphs",
            "represented_child_source_not_cleaned",
            "cleanup_backend_failure",
        }
    ]
    return {
        "source_erasure_expected": expected,
        "source_erasure_mask_coverage": round(float(coverage), 3) if coverage is not None else None,
        "source_erasure_visual_residual_score": round(float(residual_score), 3) if residual_score is not None else None,
        "source_erasure_warning_class_counts": warning_classes,
        "source_erasure_failure_reason": ",".join(blocking_reasons),
        "cleanup_warning_reasons": blocking_reasons,
        "source_glyph_mask_id": ",".join(mask_ids),
        "cleanup_partition_id": ",".join(partition_ids),
    }


def _root_render_readability_validation(
    root: dict[str, Any],
    regions: list[dict[str, Any]],
    has_accepted_parent: bool,
) -> dict[str, Any]:
    if not has_accepted_parent:
        return {
            "render_text_completeness_pass": None,
            "render_wrapped_lines": [],
            "render_missing_characters": "",
            "render_outside_root_ratio": 0.0,
            "render_density_score": 0.0,
            "render_readability_warning_reason": "",
            "render_warning_reasons": [],
            "render_warning_action_status": "not_applicable_no_accepted_parent",
            "render_warning_action_reason": "",
            "render_warning_action_reasons": [],
            "render_fit_action_attempted": False,
            "render_fit_action_status": "not_applicable_no_accepted_parent",
            "render_fit_before_density": None,
            "render_fit_after_density": None,
            "render_fit_before_bbox": [],
            "render_fit_after_bbox": [],
            "render_fit_rejection_reason": "",
            "render_layout_v2_attempted": False,
            "render_layout_v2_status": "not_applicable_no_accepted_parent",
            "render_layout_v2_before_score": None,
            "render_layout_v2_after_score": None,
            "render_layout_v2_before_fit_ratio": None,
            "render_layout_v2_after_fit_ratio": None,
            "render_layout_v2_line_height_scale": None,
            "render_layout_v2_compact_layout": False,
            "render_layout_v2_selected_font_size": None,
            "render_layout_v2_rejection_reason": "",
            "render_layout_v3_attempted": False,
            "render_layout_v3_status": "not_applicable_no_accepted_parent",
            "render_layout_v3_before_score": None,
            "render_layout_v3_after_score": None,
            "render_layout_v3_candidate_count": 0,
            "render_layout_v3_selected_candidate": {},
            "render_layout_v3_rejection_reason": "",
            "render_layout_v3_shape_source": "",
            "render_layout_v3_edge_contact_before": None,
            "render_layout_v3_edge_contact_after": None,
            "render_layout_v3_density_before": None,
            "render_layout_v3_density_after": None,
            "render_readability_v4_attempted": False,
            "render_readability_v4_status": "not_applicable_no_accepted_parent",
            "render_readability_v4_before_score": None,
            "render_readability_v4_after_score": None,
            "render_readability_v4_candidate_count": 0,
            "render_readability_v4_selected_candidate": {},
            "render_readability_v4_unresolved_reason": "",
            "render_readability_v4_edge_contact_before": None,
            "render_readability_v4_edge_contact_after": None,
            "render_readability_v4_density_before": None,
            "render_readability_v4_density_after": None,
            "render_readability_v4_shape_source": "",
            "render_readability_v4_final_class": "render_readability_not_applicable",
            "render_readability_v5_attempted": False,
            "render_readability_v5_status": "not_applicable_no_accepted_parent",
            "render_readability_v5_before_score": None,
            "render_readability_v5_after_score": None,
            "render_readability_v5_candidate_count": 0,
            "render_readability_v5_selected_candidate": {},
            "render_readability_v5_source_column_count": 0,
            "render_readability_v5_shape_source": "",
            "render_readability_v5_density_before": None,
            "render_readability_v5_density_after": None,
            "render_readability_v5_edge_contact_before": None,
            "render_readability_v5_edge_contact_after": None,
            "render_readability_v5_final_class": "render_readability_v5_resolved",
            "render_readability_v5_unresolved_reason": "",
        }
    warning_reasons: list[str] = []
    wrapped_lines_all: list[str] = []
    missing: list[str] = []
    outside_ratios: list[float] = []
    density_scores: list[float] = []
    fit_attempted = False
    fit_statuses: list[str] = []
    fit_rejection_reasons: list[str] = []
    fit_before_densities: list[float] = []
    fit_after_densities: list[float] = []
    fit_before_boxes: list[list[int]] = []
    fit_after_boxes: list[list[int]] = []
    layout_v2_attempted = False
    layout_v2_statuses: list[str] = []
    layout_v2_rejection_reasons: list[str] = []
    layout_v2_before_scores: list[float] = []
    layout_v2_after_scores: list[float] = []
    layout_v2_before_fit_ratios: list[float] = []
    layout_v2_after_fit_ratios: list[float] = []
    layout_v2_line_heights: list[float] = []
    layout_v2_font_sizes: list[float] = []
    layout_v2_compact = False
    layout_v3_attempted = False
    layout_v3_statuses: list[str] = []
    layout_v3_rejection_reasons: list[str] = []
    layout_v3_before_scores: list[float] = []
    layout_v3_after_scores: list[float] = []
    layout_v3_candidate_counts: list[float] = []
    layout_v3_edge_before: list[float] = []
    layout_v3_edge_after: list[float] = []
    layout_v3_density_before: list[float] = []
    layout_v3_density_after: list[float] = []
    layout_v3_shape_sources: list[str] = []
    layout_v3_selected_candidates: list[Any] = []
    render_v4_attempted = False
    render_v4_statuses: list[str] = []
    render_v4_unresolved_reasons: list[str] = []
    render_v4_before_scores: list[float] = []
    render_v4_after_scores: list[float] = []
    render_v4_candidate_counts: list[float] = []
    render_v4_edge_before: list[float] = []
    render_v4_edge_after: list[float] = []
    render_v4_density_before: list[float] = []
    render_v4_density_after: list[float] = []
    render_v4_shape_sources: list[str] = []
    render_v4_selected_candidates: list[Any] = []
    render_v4_final_classes: list[str] = []
    render_v5_attempted = False
    render_v5_statuses: list[str] = []
    render_v5_unresolved_reasons: list[str] = []
    render_v5_before_scores: list[float] = []
    render_v5_after_scores: list[float] = []
    render_v5_candidate_counts: list[float] = []
    render_v5_edge_before: list[float] = []
    render_v5_edge_after: list[float] = []
    render_v5_density_before: list[float] = []
    render_v5_density_after: list[float] = []
    render_v5_shape_sources: list[str] = []
    render_v5_selected_candidates: list[Any] = []
    render_v5_final_classes: list[str] = []
    render_v5_source_column_counts: list[float] = []
    text_seen = False
    completeness_pass = True
    root_box = _xyxy_from_xywh(root.get("bbox"))
    for region in regions:
        text = str(region.get("translated_text") or "")
        if not text.strip():
            continue
        if region.get("render_fit_action_attempted"):
            fit_attempted = True
        fit_status = str(region.get("render_fit_action_status") or "").strip()
        if fit_status:
            _append_unique(fit_statuses, fit_status)
        fit_reason = str(region.get("render_fit_rejection_reason") or "").strip()
        if fit_reason:
            _append_unique(fit_rejection_reasons, fit_reason)
        if region.get("render_layout_v2_attempted"):
            layout_v2_attempted = True
        layout_status = str(region.get("render_layout_v2_status") or "").strip()
        if layout_status:
            _append_unique(layout_v2_statuses, layout_status)
        layout_reason = str(region.get("render_layout_v2_rejection_reason") or "").strip()
        if layout_reason:
            _append_unique(layout_v2_rejection_reasons, layout_reason)
        if region.get("render_layout_v2_compact_layout"):
            layout_v2_compact = True
        if region.get("render_layout_v3_attempted"):
            layout_v3_attempted = True
        layout_v3_status = str(region.get("render_layout_v3_status") or "").strip()
        if layout_v3_status:
            _append_unique(layout_v3_statuses, layout_v3_status)
        layout_v3_reason = str(region.get("render_layout_v3_rejection_reason") or "").strip()
        if layout_v3_reason:
            _append_unique(layout_v3_rejection_reasons, layout_v3_reason)
        shape_source = str(region.get("render_layout_v3_shape_source") or "").strip()
        if shape_source:
            _append_unique(layout_v3_shape_sources, shape_source)
        selected_candidate = region.get("render_layout_v3_selected_candidate")
        if isinstance(selected_candidate, dict) and selected_candidate:
            layout_v3_selected_candidates.append(selected_candidate)
        if region.get("render_readability_v4_attempted"):
            render_v4_attempted = True
        render_v4_status = str(region.get("render_readability_v4_status") or "").strip()
        if render_v4_status:
            _append_unique(render_v4_statuses, render_v4_status)
        render_v4_reason = str(region.get("render_readability_v4_unresolved_reason") or "").strip()
        if render_v4_reason:
            _append_unique(render_v4_unresolved_reasons, render_v4_reason)
        render_v4_shape_source = str(region.get("render_readability_v4_shape_source") or "").strip()
        if render_v4_shape_source:
            _append_unique(render_v4_shape_sources, render_v4_shape_source)
        render_v4_selected = region.get("render_readability_v4_selected_candidate")
        if isinstance(render_v4_selected, dict) and render_v4_selected:
            render_v4_selected_candidates.append(render_v4_selected)
        render_v4_class = str(region.get("render_readability_v4_final_class") or "").strip()
        if render_v4_class:
            _append_unique(render_v4_final_classes, render_v4_class)
        if region.get("render_readability_v5_attempted"):
            render_v5_attempted = True
        render_v5_status = str(region.get("render_readability_v5_status") or "").strip()
        if render_v5_status:
            _append_unique(render_v5_statuses, render_v5_status)
        render_v5_reason = str(region.get("render_readability_v5_unresolved_reason") or "").strip()
        if render_v5_reason:
            _append_unique(render_v5_unresolved_reasons, render_v5_reason)
        render_v5_shape_source = str(region.get("render_readability_v5_shape_source") or "").strip()
        if render_v5_shape_source:
            _append_unique(render_v5_shape_sources, render_v5_shape_source)
        render_v5_selected = region.get("render_readability_v5_selected_candidate")
        if isinstance(render_v5_selected, dict) and render_v5_selected:
            render_v5_selected_candidates.append(render_v5_selected)
        render_v5_class = str(region.get("render_readability_v5_final_class") or "").strip()
        if render_v5_class:
            _append_unique(render_v5_final_classes, render_v5_class)
        for value, bucket in (
            (region.get("render_layout_v2_before_score"), layout_v2_before_scores),
            (region.get("render_layout_v2_after_score"), layout_v2_after_scores),
            (region.get("render_layout_v2_before_fit_ratio"), layout_v2_before_fit_ratios),
            (region.get("render_layout_v2_after_fit_ratio"), layout_v2_after_fit_ratios),
            (region.get("render_layout_v2_line_height_scale"), layout_v2_line_heights),
            (region.get("render_layout_v2_selected_font_size"), layout_v2_font_sizes),
            (region.get("render_layout_v3_before_score"), layout_v3_before_scores),
            (region.get("render_layout_v3_after_score"), layout_v3_after_scores),
            (region.get("render_layout_v3_candidate_count"), layout_v3_candidate_counts),
            (region.get("render_layout_v3_edge_contact_before"), layout_v3_edge_before),
            (region.get("render_layout_v3_edge_contact_after"), layout_v3_edge_after),
            (region.get("render_layout_v3_density_before"), layout_v3_density_before),
            (region.get("render_layout_v3_density_after"), layout_v3_density_after),
            (region.get("render_readability_v4_before_score"), render_v4_before_scores),
            (region.get("render_readability_v4_after_score"), render_v4_after_scores),
            (region.get("render_readability_v4_candidate_count"), render_v4_candidate_counts),
            (region.get("render_readability_v4_edge_contact_before"), render_v4_edge_before),
            (region.get("render_readability_v4_edge_contact_after"), render_v4_edge_after),
            (region.get("render_readability_v4_density_before"), render_v4_density_before),
            (region.get("render_readability_v4_density_after"), render_v4_density_after),
            (region.get("render_readability_v5_before_score"), render_v5_before_scores),
            (region.get("render_readability_v5_after_score"), render_v5_after_scores),
            (region.get("render_readability_v5_candidate_count"), render_v5_candidate_counts),
            (region.get("render_readability_v5_edge_contact_before"), render_v5_edge_before),
            (region.get("render_readability_v5_edge_contact_after"), render_v5_edge_after),
            (region.get("render_readability_v5_density_before"), render_v5_density_before),
            (region.get("render_readability_v5_density_after"), render_v5_density_after),
            (region.get("render_readability_v5_source_column_count"), render_v5_source_column_counts),
        ):
            parsed = _float_or_none(value)
            if parsed is not None:
                bucket.append(float(parsed))
        before_density = _float_or_none(region.get("render_fit_before_density"))
        after_density = _float_or_none(region.get("render_fit_after_density"))
        if before_density is not None:
            fit_before_densities.append(float(before_density))
        if after_density is not None:
            fit_after_densities.append(float(after_density))
        before_box = _xyxy(region.get("render_fit_before_bbox"))
        after_box = _xyxy(region.get("render_fit_after_bbox"))
        if before_box:
            fit_before_boxes.append(before_box)
        if after_box:
            fit_after_boxes.append(after_box)
        text_seen = True
        wrapped = _string_list(region.get("wrapped_lines") or region.get("wrapped_lines_after"))
        wrapped_lines_all.extend(wrapped)
        if wrapped:
            if not _wrapped_text_contains_translated_text(text, wrapped):
                completeness_pass = False
                _append_unique(warning_reasons, "translated_text_missing_from_wrapped_lines")
                missing.append(_missing_render_text_note(text, wrapped))
        else:
            completeness_pass = False
            _append_unique(warning_reasons, "translated_text_has_no_wrapped_lines")
            missing.append("wrapped_lines_missing")
        outside = _float_or_none(region.get("final_render_outside_allowed_area_ratio"))
        if outside is None:
            outside = _bbox_outside_ratio(_xyxy(region.get("final_render_bbox")), root_box)
        if outside is not None:
            outside_ratios.append(max(0.0, float(outside)))
            if outside > 0.08:
                _append_unique(warning_reasons, "render_outside_root_or_allowed_area")
        if region.get("render_outside_text_area_container"):
            _append_unique(warning_reasons, "render_outside_text_area_container")
        render_box = _xyxy(region.get("final_render_bbox"))
        if render_box:
            density = len(_meaningful_render_chars(text)) / max(1.0, _box_area(render_box) / 1000.0)
            density_scores.append(density)
            layout_after_score = _float_or_none(region.get("render_layout_v2_after_score"))
            layout_after_fit = _float_or_none(region.get("render_layout_v2_after_fit_ratio"))
            layout_v3_after_score = _float_or_none(region.get("render_layout_v3_after_score"))
            layout_v3_after_edge = _float_or_none(region.get("render_layout_v3_edge_contact_after"))
            render_v4_after_score = _float_or_none(region.get("render_readability_v4_after_score"))
            render_v4_after_edge = _float_or_none(region.get("render_readability_v4_edge_contact_after"))
            render_v4_after_density = _float_or_none(region.get("render_readability_v4_density_after"))
            render_v4_final_class = str(region.get("render_readability_v4_final_class") or "")
            render_v5_after_score = _float_or_none(region.get("render_readability_v5_after_score"))
            render_v5_after_edge = _float_or_none(region.get("render_readability_v5_edge_contact_after"))
            render_v5_after_density = _float_or_none(region.get("render_readability_v5_density_after"))
            render_v5_final_class = str(region.get("render_readability_v5_final_class") or "")
            render_v4_resolved = render_v4_final_class in {
                "render_readability_resolved",
                "render_readability_not_required",
            }
            render_v5_resolved = render_v5_final_class == "render_readability_v5_resolved"
            if render_v5_final_class == "render_readability_v5_accepted_complete_dense_watch":
                _append_unique(warning_reasons, "render_readability_dense_watch_requires_visual_review")
            elif render_v5_final_class.startswith("render_readability_v5_unresolved"):
                _append_unique(warning_reasons, render_v5_final_class.replace("render_readability_v5_", "render_"))
            layout_v4_density_ok = (
                render_v4_after_density is not None
                and render_v4_after_density < 0.34
                and render_v4_after_score is not None
                and render_v4_after_score < 15.0
            )
            layout_v5_density_ok = (
                render_v5_resolved
                or (
                    render_v5_after_density is not None
                    and render_v5_after_density < 0.40
                    and render_v5_after_score is not None
                    and render_v5_after_score < 18.0
                )
            )
            if density > 0.38 and len(_meaningful_render_chars(text)) >= 12 and not layout_v4_density_ok and not layout_v5_density_ok:
                _append_unique(warning_reasons, "render_density_high")
            edge_touch = _render_edge_touch_ratio(render_box, root_box)
            fit_expanded_inside_root = (
                ("expanded_within_root_allowed_area" in fit_status or "layout_v2_candidate_applied" in fit_status)
                and (outside is None or outside <= 0.01)
            )
            layout_v3_edge_ok = layout_v3_after_edge is not None and layout_v3_after_edge < 0.5 and layout_v3_after_score is not None and layout_v3_after_score < 14.0
            layout_v4_edge_ok = (
                render_v4_after_edge is not None
                and render_v4_after_edge < 0.5
                and render_v4_after_score is not None
                and render_v4_after_score < 15.0
            )
            layout_v5_edge_ok = (
                render_v5_resolved
                or (
                    render_v5_after_edge is not None
                    and render_v5_after_edge < 0.5
                    and render_v5_after_score is not None
                    and render_v5_after_score < 18.0
                )
            )
            if edge_touch >= 0.5 and len(_meaningful_render_chars(text)) >= 8 and not fit_expanded_inside_root and not layout_v3_edge_ok and not layout_v4_edge_ok and not layout_v5_edge_ok:
                _append_unique(warning_reasons, "render_near_root_or_container_edges")
            if len(wrapped) >= 5 and len(_meaningful_render_chars(text)) >= 10 and not render_v5_resolved:
                _append_unique(warning_reasons, "excessive_wrapped_line_count_for_root")
            fit_ratio = _float_or_none(region.get("fit_ratio"))
            layout_v3_soft_fit_ok = (
                layout_v3_after_score is not None
                and layout_v3_after_score < 8.0
                and (layout_v3_after_edge is None or layout_v3_after_edge < 0.5)
                and density < 0.28
            )
            layout_v3_short_text_not_required = (
                "not_required_short_text" in str(region.get("render_layout_v3_status") or "")
                and density < 0.25
            )
            layout_v4_soft_fit_ok = (
                render_v4_after_score is not None
                and render_v4_after_score < 8.0
                and (render_v4_after_edge is None or render_v4_after_edge < 0.5)
                and (render_v4_after_density is None or render_v4_after_density < 0.28)
            )
            layout_v5_soft_fit_ok = (
                render_v5_resolved
                or (
                    render_v5_after_score is not None
                    and render_v5_after_score < 10.0
                    and (render_v5_after_edge is None or render_v5_after_edge < 0.5)
                    and (render_v5_after_density is None or render_v5_after_density < 0.34)
                )
            )
            if (
                fit_ratio is not None
                and fit_ratio >= 0.96
                and len(_meaningful_render_chars(text)) >= 10
                and not (layout_after_fit is not None and layout_after_fit < 0.92 and layout_after_score is not None and layout_after_score < 12.0)
                and not layout_v3_soft_fit_ok
                and not layout_v3_short_text_not_required
                and not layout_v4_soft_fit_ok
                and not layout_v5_soft_fit_ok
                and not render_v4_resolved
            ):
                _append_unique(warning_reasons, "render_fit_ratio_tight")
        font_size = _float_or_none(region.get("selected_font_size"))
        region_v5_final_class = str(region.get("render_readability_v5_final_class") or "")
        region_v5_resolved = region_v5_final_class == "render_readability_v5_resolved"
        if font_size is not None and font_size < 32 and len(_meaningful_render_chars(text)) >= 8 and not region_v5_resolved:
            _append_unique(warning_reasons, "render_font_tiny_for_text_length")
    if not text_seen:
        completeness_pass = False
        _append_unique(warning_reasons, "accepted_parent_without_rendered_translation")
    root_density = _root_level_render_density(root_box, regions)
    if root_density > 0:
        density_scores.append(root_density)
        max_region_density = max(density_scores) if density_scores else 0.0
        max_layout_score = max(layout_v2_after_scores) if layout_v2_after_scores else None
        max_layout_v3_score = max(layout_v3_after_scores) if layout_v3_after_scores else None
        max_render_v4_score = max(render_v4_after_scores) if render_v4_after_scores else None
        max_render_v4_density = max(render_v4_density_after) if render_v4_density_after else None
        max_render_v5_score = max(render_v5_after_scores) if render_v5_after_scores else None
        max_render_v5_density = max(render_v5_density_after) if render_v5_density_after else None
        render_v4_root_density_ok = (
            max_render_v4_score is not None
            and max_render_v4_score < 10.0
            and max_render_v4_density is not None
            and max_render_v4_density < 0.28
            and render_v4_final_classes
            and all(str(item) in {"render_readability_resolved", "render_readability_not_required"} for item in render_v4_final_classes)
            and not any(str(item).endswith("text_too_dense") for item in render_v4_final_classes)
        )
        render_v5_root_density_ok = (
            render_v5_final_classes
            and all(
                str(item) == "render_readability_v5_resolved"
                for item in render_v5_final_classes
            )
            and (
                max_render_v5_score is None
                or max_render_v5_score < 12.0
            )
            and (
                max_render_v5_density is None
                or max_render_v5_density < 0.28
            )
        )
        if (
            root_density > 0.22
            or (
                root_density > 0.16
                and (
                    max_region_density > 0.28
                    or (max_layout_score is not None and max_layout_score >= 12.0)
                    or (max_layout_v3_score is not None and max_layout_v3_score >= 14.0)
                    or (max_render_v4_score is not None and max_render_v4_score >= 15.0)
                    or (max_render_v5_score is not None and max_render_v5_score >= 18.0)
                )
            )
        ) and not render_v4_root_density_ok and not render_v5_root_density_ok:
            _append_unique(warning_reasons, "root_level_render_density_high")
    action_status, action_reason, action_reasons = _render_warning_action_metadata(warning_reasons)
    if warning_reasons and fit_attempted and not fit_rejection_reasons:
        _append_unique(fit_rejection_reasons, "warning_remains_after_root_local_fit_action")
    return {
        "render_text_completeness_pass": completeness_pass,
        "render_wrapped_lines": wrapped_lines_all,
        "render_missing_characters": ";".join(item for item in missing if item),
        "render_outside_root_ratio": round(max(outside_ratios) if outside_ratios else 0.0, 3),
        "render_density_score": round(max(density_scores) if density_scores else 0.0, 3),
        "render_readability_warning_reason": ",".join(warning_reasons),
        "render_warning_reasons": warning_reasons,
        "render_warning_action_status": action_status,
        "render_warning_action_reason": action_reason,
        "render_warning_action_reasons": action_reasons,
        "render_fit_action_attempted": fit_attempted,
        "render_fit_action_status": ",".join(fit_statuses) if fit_statuses else "not_recorded",
        "render_fit_before_density": round(max(fit_before_densities), 3) if fit_before_densities else None,
        "render_fit_after_density": round(max(fit_after_densities), 3) if fit_after_densities else None,
        "render_fit_before_bbox": fit_before_boxes[0] if fit_before_boxes else [],
        "render_fit_after_bbox": fit_after_boxes[0] if fit_after_boxes else [],
        "render_fit_rejection_reason": ",".join(fit_rejection_reasons),
        "render_layout_v2_attempted": layout_v2_attempted,
        "render_layout_v2_status": ",".join(layout_v2_statuses) if layout_v2_statuses else "not_recorded",
        "render_layout_v2_before_score": round(max(layout_v2_before_scores), 3) if layout_v2_before_scores else None,
        "render_layout_v2_after_score": round(max(layout_v2_after_scores), 3) if layout_v2_after_scores else None,
        "render_layout_v2_before_fit_ratio": round(max(layout_v2_before_fit_ratios), 3) if layout_v2_before_fit_ratios else None,
        "render_layout_v2_after_fit_ratio": round(max(layout_v2_after_fit_ratios), 3) if layout_v2_after_fit_ratios else None,
        "render_layout_v2_line_height_scale": round(min(layout_v2_line_heights), 3) if layout_v2_line_heights else None,
        "render_layout_v2_compact_layout": layout_v2_compact,
        "render_layout_v2_selected_font_size": round(min(layout_v2_font_sizes), 3) if layout_v2_font_sizes else None,
        "render_layout_v2_rejection_reason": ",".join(layout_v2_rejection_reasons),
        "render_layout_v3_attempted": layout_v3_attempted,
        "render_layout_v3_status": ",".join(layout_v3_statuses) if layout_v3_statuses else "not_recorded",
        "render_layout_v3_before_score": round(max(layout_v3_before_scores), 3) if layout_v3_before_scores else None,
        "render_layout_v3_after_score": round(max(layout_v3_after_scores), 3) if layout_v3_after_scores else None,
        "render_layout_v3_candidate_count": int(max(layout_v3_candidate_counts)) if layout_v3_candidate_counts else 0,
        "render_layout_v3_selected_candidate": layout_v3_selected_candidates[0] if layout_v3_selected_candidates else {},
        "render_layout_v3_rejection_reason": ",".join(layout_v3_rejection_reasons),
        "render_layout_v3_shape_source": ",".join(layout_v3_shape_sources),
        "render_layout_v3_edge_contact_before": round(max(layout_v3_edge_before), 3) if layout_v3_edge_before else None,
        "render_layout_v3_edge_contact_after": round(max(layout_v3_edge_after), 3) if layout_v3_edge_after else None,
        "render_layout_v3_density_before": round(max(layout_v3_density_before), 3) if layout_v3_density_before else None,
        "render_layout_v3_density_after": round(max(layout_v3_density_after), 3) if layout_v3_density_after else None,
        "render_readability_v4_attempted": render_v4_attempted,
        "render_readability_v4_status": ",".join(render_v4_statuses) if render_v4_statuses else "not_recorded",
        "render_readability_v4_before_score": round(max(render_v4_before_scores), 3) if render_v4_before_scores else None,
        "render_readability_v4_after_score": round(max(render_v4_after_scores), 3) if render_v4_after_scores else None,
        "render_readability_v4_candidate_count": int(max(render_v4_candidate_counts)) if render_v4_candidate_counts else 0,
        "render_readability_v4_selected_candidate": render_v4_selected_candidates[0] if render_v4_selected_candidates else {},
        "render_readability_v4_unresolved_reason": ",".join(render_v4_unresolved_reasons),
        "render_readability_v4_edge_contact_before": round(max(render_v4_edge_before), 3) if render_v4_edge_before else None,
        "render_readability_v4_edge_contact_after": round(max(render_v4_edge_after), 3) if render_v4_edge_after else None,
        "render_readability_v4_density_before": round(max(render_v4_density_before), 3) if render_v4_density_before else None,
        "render_readability_v4_density_after": round(max(render_v4_density_after), 3) if render_v4_density_after else None,
        "render_readability_v4_shape_source": ",".join(render_v4_shape_sources),
        "render_readability_v4_final_class": ",".join(render_v4_final_classes),
        "render_readability_v5_attempted": render_v5_attempted,
        "render_readability_v5_status": ",".join(render_v5_statuses) if render_v5_statuses else "not_recorded",
        "render_readability_v5_before_score": round(max(render_v5_before_scores), 3) if render_v5_before_scores else None,
        "render_readability_v5_after_score": round(max(render_v5_after_scores), 3) if render_v5_after_scores else None,
        "render_readability_v5_candidate_count": int(max(render_v5_candidate_counts)) if render_v5_candidate_counts else 0,
        "render_readability_v5_selected_candidate": render_v5_selected_candidates[0] if render_v5_selected_candidates else {},
        "render_readability_v5_source_column_count": int(max(render_v5_source_column_counts)) if render_v5_source_column_counts else 0,
        "render_readability_v5_shape_source": ",".join(render_v5_shape_sources),
        "render_readability_v5_density_before": round(max(render_v5_density_before), 3) if render_v5_density_before else None,
        "render_readability_v5_density_after": round(max(render_v5_density_after), 3) if render_v5_density_after else None,
        "render_readability_v5_edge_contact_before": round(max(render_v5_edge_before), 3) if render_v5_edge_before else None,
        "render_readability_v5_edge_contact_after": round(max(render_v5_edge_after), 3) if render_v5_edge_after else None,
        "render_readability_v5_final_class": ",".join(render_v5_final_classes),
        "render_readability_v5_unresolved_reason": ",".join(render_v5_unresolved_reasons),
    }


def _render_warning_action_metadata(warning_reasons: list[str]) -> tuple[str, str, list[str]]:
    if not warning_reasons:
        return "not_required", "", []
    reasons: list[str] = []
    mutation_required = False
    if any(reason in warning_reasons for reason in (
        "translated_text_missing_from_wrapped_lines",
        "translated_text_has_no_wrapped_lines",
        "accepted_parent_without_rendered_translation",
    )):
        _append_unique(reasons, "render_output_incomplete_requires_renderer_fix")
        mutation_required = True
    if any(reason in warning_reasons for reason in (
        "render_density_high",
        "root_level_render_density_high",
        "excessive_wrapped_line_count_for_root",
        "render_font_tiny_for_text_length",
    )):
        _append_unique(reasons, "density_requires_layout_fit_pass")
        mutation_required = True
    if any(reason in warning_reasons for reason in (
        "render_fit_ratio_tight",
        "render_near_root_or_container_edges",
    )):
        _append_unique(reasons, "tight_fit_recorded_no_safe_root_local_mutation")
    if "render_outside_root_or_allowed_area" in warning_reasons or "render_outside_text_area_container" in warning_reasons:
        _append_unique(reasons, "render_outside_container_requires_constraint_fix")
        mutation_required = True
    status = (
        "warning_recorded_renderer_mutation_required"
        if mutation_required
        else "warning_recorded_no_safe_renderer_mutation"
    )
    return status, ",".join(reasons), reasons


def _dict_parent_is_active_translation_unit(parent: dict[str, Any]) -> bool:
    if not bool(parent.get("translation_unit")):
        return False
    if str(parent.get("source_coherence_status") or "") == "rejected":
        return False
    if str(parent.get("source_coherence_action") or "") in {
        "source_quality_blocked",
        "block_auto_translation",
        "split_required",
        "unresolved_review",
        "block_review_only",
    }:
        return False
    return bool(_source_body(parent.get("source_text")) or _valid_short_reaction_or_laugh(str(parent.get("source_text") or "")))


def _dict_root_has_meaningful_visible_source(
    root: dict[str, Any],
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
) -> bool:
    if str(root.get("root_type") or "") == ROOT_SFX or str(root.get("route_policy") or "") == ROUTE_PRESERVE:
        return False
    for parent in parents:
        if _source_body_requires_root_blocker(parent.get("source_text")):
            return True
    for child in children:
        if str(child.get("final_state") or "") == STATE_BLOCKED_BY_ROOT_POLICY:
            continue
        if _source_body_requires_root_blocker(child.get("ocr_text")):
            return True
    return False


def _dict_root_is_punctuation_only(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
) -> bool:
    texts = [
        str(parent.get("source_text") or "")
        for parent in parents
        if str(parent.get("source_text") or "").strip()
    ]
    texts.extend(
        str(child.get("ocr_text") or "")
        for child in children
        if str(child.get("ocr_text") or "").strip()
    )
    if not texts:
        return False
    joined = "".join(texts)
    if any("\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" for ch in joined):
        return False
    return all(_is_punctuation(ch) for ch in joined if not ch.isspace())


def _wrapped_text_contains_translated_text(text: str, wrapped_lines: list[str]) -> bool:
    expected = _meaningful_render_chars(text)
    if not expected:
        return True
    rendered = _meaningful_render_chars("".join(wrapped_lines))
    return expected in rendered or rendered in expected


def _meaningful_render_chars(text: Any) -> str:
    return "".join(ch for ch in str(text or "") if not ch.isspace() and ch not in "，,。.．、…·・!！?？:：;；「」『』“”\"'‘’（）()[]【】")


def _missing_render_text_note(text: str, wrapped_lines: list[str]) -> str:
    expected = _meaningful_render_chars(text)
    rendered = _meaningful_render_chars("".join(wrapped_lines))
    if not expected or expected in rendered:
        return ""
    return f"expected_not_wrapped:{_short(expected, 24)}"


def _root_level_render_density(root_box: list[int] | None, regions: list[dict[str, Any]]) -> float:
    if not root_box:
        return 0.0
    text_len = 0
    for region in regions:
        text_len += len(_meaningful_render_chars(region.get("translated_text")))
    if text_len <= 0:
        return 0.0
    return text_len / max(1.0, _box_area(root_box) / 1000.0)


def _render_edge_touch_ratio(
    render_box: list[int] | tuple[int, int, int, int] | None,
    root_box: list[int] | tuple[int, int, int, int] | None,
) -> float:
    if not render_box or not root_box:
        return 0.0
    rx0, ry0, rx1, ry1 = [int(v) for v in render_box[:4]]
    ox0, oy0, ox1, oy1 = [int(v) for v in root_box[:4]]
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    tolerance = max(6, min(18, int(min(ox1 - ox0, oy1 - oy0) * 0.04)))
    touches = 0
    if abs(rx0 - ox0) <= tolerance:
        touches += 1
    if abs(ry0 - oy0) <= tolerance:
        touches += 1
    if abs(rx1 - ox1) <= tolerance:
        touches += 1
    if abs(ry1 - oy1) <= tolerance:
        touches += 1
    return touches / 4.0


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _xyxy(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None


def _xyxy_from_xywh(value: Any) -> list[int] | None:
    box = _xyxy(value)
    if not box:
        return None
    x, y, w, h = box
    return [x, y, x + max(0, w), y + max(0, h)]


def _box_area(box: list[int] | tuple[int, int, int, int] | None) -> float:
    if not box or len(box) < 4:
        return 0.0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _bbox_outside_ratio(inner: list[int] | None, outer: list[int] | None) -> float | None:
    if not inner or not outer:
        return None
    area = _box_area(inner)
    if area <= 0:
        return 0.0
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    cx0, cy0 = max(ix0, ox0), max(iy0, oy0)
    cx1, cy1 = min(ix1, ox1), min(iy1, oy1)
    inside = _box_area([cx0, cy0, cx1, cy1])
    return max(0.0, min(1.0, (area - inside) / area))


def write_text_block_hierarchy_artifacts(
    *,
    page_dir: str,
    hierarchy: dict[str, Any],
    source_glyph_masks: dict[str, Any] | None = None,
) -> dict[str, str]:
    os.makedirs(page_dir, exist_ok=True)
    paths: dict[str, str] = {}
    json_path = os.path.join(page_dir, "text_block_hierarchy.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(hierarchy, handle, ensure_ascii=False, indent=2)
    paths["text_block_hierarchy_json"] = json_path

    paths["root_block_table"] = _write_table(
        os.path.join(page_dir, "text_block_root_blocks.md"),
        ["root_id", "type", "containers", "physical", "route", "parents", "children", "coherence", "reconstruction", "blocker", "frag", "malformed", "dupes", "tier", "reasons"],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                ",".join(root.get("text_area_container_ids") or []),
                root.get("physical_bubble_id"),
                root.get("route_policy"),
                len(root.get("parent_unit_ids") or []),
                len(root.get("child_segment_ids") or []),
                root.get("root_source_coherence_status"),
                root.get("root_reconstruction_status"),
                root.get("root_validation_blocker"),
                root.get("root_fragmentation_score"),
                root.get("root_malformed_parent_count"),
                root.get("root_duplicate_partial_parent_count"),
                root.get("confidence_tier"),
                ",".join(root.get("reason_codes") or []),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
        ],
    )
    paths["parent_logical_unit_table"] = _write_table(
        os.path.join(page_dir, "parent_logical_text_units.md"),
        ["parent_id", "root", "role", "translation", "anchor", "children", "source", "reocr", "coherence", "action", "conservation"],
        [
            [
                parent.get("parent_id"),
                parent.get("root_id"),
                parent.get("role"),
                parent.get("translation_unit"),
                parent.get("anchor_child_id"),
                ",".join(parent.get("child_segment_ids") or []),
                _short(parent.get("source_text")),
                parent.get("source_reconstruction_status"),
                parent.get("source_coherence_status"),
                parent.get("source_coherence_action"),
                parent.get("source_conservation_status"),
            ]
            for parent in hierarchy.get("parent_logical_text_units", []) or []
        ],
    )
    paths["child_segment_table"] = _write_table(
        os.path.join(page_dir, "child_recognized_text_segments.md"),
        ["child_id", "region", "root", "parent", "state", "independent", "ocr", "reasons"],
        [
            [
                child.get("child_id"),
                child.get("source_region_id"),
                child.get("root_id"),
                child.get("parent_id"),
                child.get("final_state"),
                child.get("translated_independently"),
                _short(child.get("ocr_text")),
                ",".join(child.get("reason_codes") or []),
            ]
            for child in hierarchy.get("child_recognized_text_segments", []) or []
        ],
    )
    paths["unresolved_child_table"] = _write_table(
        os.path.join(page_dir, "unresolved_child_segments.md"),
        ["child_id", "region", "root", "ocr", "reasons"],
        [
            [
                child.get("child_id"),
                child.get("source_region_id"),
                child.get("root_id"),
                _short(child.get("ocr_text")),
                ",".join(child.get("reason_codes") or []),
            ]
            for child in hierarchy.get("unresolved_child_segments", []) or []
        ],
    )
    paths["translation_unit_table"] = _write_table(
        os.path.join(page_dir, "translation_unit_table.md"),
        ["parent_id", "root", "role", "source", "cleanup", "render"],
        [
            [
                parent.get("parent_id"),
                parent.get("root_id"),
                parent.get("role"),
                _short(parent.get("source_text")),
                parent.get("cleanup_unit"),
                parent.get("render_unit"),
            ]
            for parent in hierarchy.get("parent_logical_text_units", []) or []
            if parent.get("translation_unit")
        ],
    )
    paths["source_glyph_parent_anchor_table"] = _write_source_glyph_parent_table(page_dir, hierarchy, source_glyph_masks or {})
    paths["root_source_coherence_table"] = _write_root_source_coherence_table(page_dir, hierarchy)
    paths["standalone_parent_rejection_table"] = _write_standalone_parent_rejection_table(page_dir, hierarchy)
    paths["malformed_source_parent_table"] = _write_malformed_source_parent_table(page_dir, hierarchy)
    paths["duplicate_partial_parent_table"] = _write_duplicate_partial_parent_table(page_dir, hierarchy)
    paths["caption_background_root_coverage_table"] = _write_caption_background_root_coverage_table(page_dir, hierarchy)
    paths["root_reconstruction_table"] = _write_root_reconstruction_table(page_dir, hierarchy)
    paths["root_final_state_table"] = _write_root_final_state_table(page_dir, hierarchy)
    paths["page_final_state_summary"] = _write_page_final_state_summary(page_dir, hierarchy)
    paths["route_owned_transaction_contract_table"] = _write_route_owned_transaction_contract_table(page_dir, hierarchy)
    paths["visual_evaluation_gate"] = _write_visual_evaluation_gate(page_dir, hierarchy)
    paths["unresolved_meaningful_blocker_table"] = _write_unresolved_meaningful_blocker_table(page_dir, hierarchy)
    paths["punctuation_only_nonblocking_table"] = _write_punctuation_only_nonblocking_table(page_dir, hierarchy)
    paths["caption_background_recovery_table"] = _write_caption_background_recovery_table(page_dir, hierarchy)
    paths["caption_sfx_rejection_table"] = _write_caption_sfx_rejection_table(page_dir, hierarchy)
    paths["caption_component_split_table"] = _write_caption_component_split_table(page_dir, hierarchy)
    paths["caption_background_failed_reason_table"] = _write_caption_background_failed_reason_table(page_dir, hierarchy)
    paths["source_erasure_validation_table"] = _write_source_erasure_validation_table(page_dir, hierarchy)
    paths["render_readability_validation_table"] = _write_render_readability_validation_table(page_dir, hierarchy)
    return paths


def _graph_plan(plan: dict[str, Any]) -> dict[str, Any]:
    graph = plan.get("root_parent_child_plan")
    return graph if isinstance(graph, dict) else {}


def _graph_root_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    graph = _graph_plan(plan)
    return [node for node in graph.get("root_nodes") or [] if isinstance(node, dict)]


def _graph_parent_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    graph = _graph_plan(plan)
    return [node for node in graph.get("parent_nodes") or [] if isinstance(node, dict)]


def _graph_root_type(node: dict[str, Any]) -> str:
    role = str(node.get("semantic_role") or "").strip()
    if role == "speech":
        return ROOT_SPEECH
    if role in GRAPH_CAPTION_BACKGROUND_KINDS:
        return ROOT_CAPTION
    if role in {"sfx", "decorative", "art", "sfx_decorative_art"}:
        return ROOT_SFX
    return ROOT_REVIEW


def _graph_route_policy(node: dict[str, Any]) -> str:
    root_type = _graph_root_type(node)
    if root_type == ROOT_SPEECH:
        return ROUTE_TRANSLATE_SPEECH
    if root_type == ROOT_CAPTION:
        return ROUTE_TRANSLATE_CAPTION
    if root_type == ROOT_SFX:
        return ROUTE_PRESERVE
    return ROUTE_REVIEW


def _graph_parent_role(node: dict[str, Any]) -> str:
    kind = str(node.get("parent_kind") or "").strip()
    if kind == "speech":
        return ROLE_SPEECH
    if kind in GRAPH_CAPTION_BACKGROUND_KINDS:
        return ROLE_CAPTION
    return ROLE_REVIEW


def _materialize_graph_plan_parents(
    *,
    page_id: str,
    regions: list[dict[str, Any]],
    parent_nodes: list[dict[str, Any]],
    roots_by_key: dict[str, TextAreaRootBlock],
) -> tuple[list[ParentLogicalTextUnit], list[ChildRecognizedTextSegment], dict[str, ChildRecognizedTextSegment]]:
    parent_units: list[ParentLogicalTextUnit] = []
    children: list[ChildRecognizedTextSegment] = []
    child_by_region: dict[str, ChildRecognizedTextSegment] = {}

    for parent_node in parent_nodes:
        parent = _parent_from_graph_node(page_id, parent_node)
        root = roots_by_key.get(parent.root_id)
        if root is not None:
            _append_unique(root.parent_unit_ids, parent.parent_id)

        attached_regions = _regions_for_graph_parent(parent_node, regions, root_node=root)
        source_text, source_region_ids, duplicate_region_ids = _graph_parent_source_from_regions(
            attached_regions,
            parent_node=parent_node,
            sibling_parent_nodes=parent_nodes,
        )
        parent.source_text = source_text
        parent.source_text_before_reconstruction = source_text
        _attach_graph_parent_source_contract(parent, attached_regions, source_region_ids)
        _attach_graph_parent_source_reconstruction(parent, attached_regions, source_region_ids)
        parent.anchor_child_id = _child_id(page_id, source_region_ids[0]) if source_region_ids else None
        parent.source_conservation_status = "complete" if source_text else "unresolved"
        parent.confidence = _graph_parent_confidence(attached_regions)
        if not source_text:
            parent.translation_unit = False
            parent.cleanup_unit = False
            parent.render_unit = False
            parent.unresolved_reason = "text_area_graph_parent_without_attached_source_evidence"
            _append_unique(parent.reason_codes, parent.unresolved_reason)
        elif _parent_is_punctuation_identity(parent):
            parent.translation_unit = False
            parent.cleanup_unit = True
            parent.render_unit = True
            _append_unique(parent.reason_codes, "text_area_graph_punctuation_identity_parent")
        else:
            parent.translation_unit = True
            parent.cleanup_unit = True
            parent.render_unit = True
        parent_units.append(parent)

        first_source_region = source_region_ids[0] if source_region_ids else ""
        for region in attached_regions:
            rid = str(region.get("region_id") or "")
            if not rid or rid in child_by_region:
                continue
            if rid == first_source_region:
                final_state = STATE_PARENT_ANCHOR
            elif rid in duplicate_region_ids:
                final_state = STATE_DUPLICATE_CHILD
            elif rid in source_region_ids:
                final_state = STATE_PARENT_CHILD
            elif source_region_ids:
                final_state = _graph_attached_non_authoring_child_state(region)
            else:
                final_state = STATE_UNRESOLVED_REVIEW_ONLY
            child = _child_from_graph_region(
                page_id=page_id,
                region=region,
                root_id=parent.root_id,
                parent_id=parent.parent_id,
                final_state=final_state,
                source_authoring=rid in set(source_region_ids),
            )
            children.append(child)
            child_by_region[rid] = child
            if root is not None:
                _append_unique(root.child_segment_ids, child.child_id)

    return parent_units, children, child_by_region


def _parent_from_graph_node(page_id: str, node: dict[str, Any]) -> ParentLogicalTextUnit:
    parent_id = str(node.get("parent_node_id") or "")
    root_id = str(node.get("root_node_id") or "")
    bbox = _bbox(node.get("bbox"))
    return ParentLogicalTextUnit(
        parent_id=parent_id or _root_id(page_id, f"graph_parent_{node.get('container_id') or 'unknown'}"),
        page_id=page_id,
        root_id=root_id or _root_id(page_id, str(node.get("container_id") or parent_id or "graph_root")),
        role=_graph_parent_role(node),
        cleanup_target_bbox=list(bbox),
        render_allowed_area=list(bbox),
        reason_codes=_string_list(node.get("reason_codes")) + ["text_area_graph_parent_materialized"],
        parent_visual_group_id=str(node.get("parent_boundary_id") or ""),
        parent_visual_group_bbox=list(bbox),
        parent_visual_group_child_ids=_string_list(node.get("support_geometry_ids")),
    )


def _attach_graph_parent_source_reconstruction(
    parent: ParentLogicalTextUnit,
    regions: list[dict[str, Any]],
    source_region_ids: list[str],
) -> None:
    source_id_set = {str(rid) for rid in source_region_ids if str(rid)}
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or rid not in source_id_set:
            continue
        status = str(region.get("logical_text_source_reconstruction_status") or "").strip()
        if status != "applied":
            continue
        after_text = _clean_source_text(region.get("logical_text_source_reconstruction_after_text"))
        if after_text and after_text != _clean_source_text(parent.source_text):
            continue
        parent.source_reconstruction_status = status
        before_text = _clean_source_text(region.get("logical_text_source_reconstruction_before_text"))
        if before_text:
            parent.source_text_before_reconstruction = before_text
        parent.source_reconstruction_crop_bbox = _bbox(region.get("logical_text_source_reconstruction_crop_bbox"))
        parent.source_reconstruction_confidence = _float_or_none(region.get("logical_text_source_reconstruction_ocr_confidence"))
        _append_unique(parent.reason_codes, "graph_parent_source_reconstruction_applied")
        return


def _attach_graph_parent_source_contract(
    parent: ParentLogicalTextUnit,
    regions: list[dict[str, Any]],
    source_region_ids: list[str],
) -> None:
    source_id_set = {str(rid) for rid in source_region_ids if str(rid)}
    if not source_id_set:
        return
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or rid not in source_id_set:
            continue
        if not bool(region.get("parent_boundary_ocr_source_contract")):
            continue
        parent.source_contract_owner = str(
            region.get("source_contract_owner")
            or region.get("parent_ocr_source_contract_owner")
            or "parent_logical_text_unit_ocr_source_contract"
        )
        parent.source_contract_region_id = rid
        parent.source_contract_quality_state = str(region.get("parent_ocr_source_quality_state") or "")
        parent.source_contract_quality_action = str(region.get("parent_ocr_source_quality_action") or "")
        parent.source_quality_warning_reason_codes = _string_list(region.get("parent_ocr_source_quality_reason_codes"))
        _append_unique(parent.reason_codes, "parent_boundary_ocr_source_contract_owner")
        return


def _regions_for_graph_parent(
    parent_node: dict[str, Any],
    regions: list[dict[str, Any]],
    *,
    root_node: TextAreaRootBlock | None = None,
) -> list[dict[str, Any]]:
    parent_box = _bbox(parent_node.get("bbox"))
    container_id = str(parent_node.get("container_id") or "")
    scored: list[tuple[float, tuple[Any, ...], dict[str, Any]]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        if str(region.get("text_area_container_id") or "") != container_id and not _graph_region_can_attach_to_normalized_root_parent(
            region,
            parent_node=parent_node,
            root_node=root_node,
        ):
            continue
        score = _graph_parent_attachment_score(parent_box, _bbox(region.get("bbox")))
        if score <= 0.0:
            continue
        scored.append((score, _graph_region_reading_key(region), region))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [region for _score, _key, region in scored]


def _graph_region_can_attach_to_normalized_root_parent(
    region: dict[str, Any],
    *,
    parent_node: dict[str, Any],
    root_node: TextAreaRootBlock | None,
) -> bool:
    if root_node is None:
        return False
    region_box = _bbox(region.get("bbox"))
    parent_box = _bbox(parent_node.get("bbox"))
    root_box = list(root_node.bbox or [])
    if not region_box or not parent_box or not root_box:
        return False
    if _xywh_intersection_area(root_box, region_box) / max(1.0, _xywh_area(region_box)) < 0.80:
        return False
    route_intent = str(
        region.get("text_area_route_intent")
        or (region.get("render") or {}).get("text_area_route_intent")
        or ""
    ).strip()
    parent_kind = str(parent_node.get("parent_kind") or "").strip()
    if parent_kind == "speech" and route_intent not in {"", "translate_speech"}:
        return False
    if parent_kind in GRAPH_CAPTION_BACKGROUND_KINDS and route_intent not in {
        "",
        "translate_caption",
        "translate_caption_background",
    }:
        return False
    intersection = _xywh_intersection_area(parent_box, region_box)
    if intersection / max(1.0, _xywh_area(region_box)) >= 0.55:
        return True
    return _xywh_center_inside(region_box, parent_box)


def _graph_parent_attachment_score(parent_box: list[int], region_box: list[int]) -> float:
    if not parent_box or not region_box:
        return 0.0
    if _xywh_center_inside(region_box, parent_box):
        return 1.0
    intersection = _xywh_intersection_area(parent_box, region_box)
    region_area = max(1.0, _xywh_area(region_box))
    parent_area = max(1.0, _xywh_area(parent_box))
    region_overlap = intersection / region_area
    parent_overlap = intersection / parent_area
    if region_overlap >= 0.45 or (region_overlap >= 0.20 and parent_overlap >= 0.03):
        return max(region_overlap, parent_overlap)
    return 0.0


def _graph_parent_source_from_regions(
    regions: list[dict[str, Any]],
    *,
    parent_node: dict[str, Any] | None = None,
    sibling_parent_nodes: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str], set[str]]:
    parent_owned = _graph_parent_owned_ocr_source_from_regions(
        regions,
        parent_node=parent_node,
    )
    if parent_owned is not None:
        return parent_owned
    reconstructed = _graph_parent_reconstructed_source_from_regions(
        regions,
        parent_node=parent_node,
        sibling_parent_nodes=sibling_parent_nodes,
    )
    if reconstructed is not None:
        return reconstructed
    selected, duplicate_region_ids = _graph_parent_source_from_regions_pass(
        regions,
        parent_node=parent_node,
        sibling_parent_nodes=sibling_parent_nodes,
        allow_ignored_duplicate=False,
    )
    dominant = _graph_parent_dominant_boundary_source(
        selected,
        regions,
        parent_node=parent_node,
    )
    if dominant is not None:
        return dominant
    if selected:
        supplemental, supplemental_duplicate_region_ids = _graph_parent_source_from_regions_pass(
            regions,
            parent_node=parent_node,
            sibling_parent_nodes=sibling_parent_nodes,
            allow_ignored_duplicate=True,
        )
        selected_region_ids = {str(item["region"].get("region_id") or "") for item in selected}
        for item in supplemental:
            rid = str(item["region"].get("region_id") or "")
            if rid and rid not in selected_region_ids:
                selected.append(item)
                selected_region_ids.add(rid)
        duplicate_region_ids.update(supplemental_duplicate_region_ids)
    else:
        selected, duplicate_region_ids = _graph_parent_source_from_regions_pass(
            regions,
            parent_node=parent_node,
            sibling_parent_nodes=sibling_parent_nodes,
            allow_ignored_duplicate=True,
        )
    selected.sort(key=lambda item: _graph_region_reading_key(item["region"]))
    source_region_ids = [str(item["region"].get("region_id") or "") for item in selected]
    source_text = "".join(str(item["text"]) for item in selected)
    return source_text, source_region_ids, duplicate_region_ids


def _graph_parent_owned_ocr_source_from_regions(
    regions: list[dict[str, Any]],
    *,
    parent_node: dict[str, Any] | None,
) -> tuple[str, list[str], set[str]] | None:
    if not parent_node:
        return None
    parent_id = str(parent_node.get("parent_node_id") or "")
    parent_box = _bbox(parent_node.get("bbox"))
    if not parent_id or not parent_box:
        return None
    candidates: list[tuple[int, tuple[Any, ...], dict[str, Any], str]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        if not bool(region.get("parent_boundary_ocr_source_contract")):
            continue
        owner_parent_id = str(region.get("parent_ocr_source_parent_id") or "")
        if owner_parent_id and owner_parent_id != parent_id:
            continue
        rid = str(region.get("region_id") or "")
        text = _clean_source_text(region.get("ocr_text"))
        if not rid or not text:
            continue
        region_box = _bbox(region.get("bbox"))
        if not _graph_region_covers_parent_boundary(region_box, parent_box):
            continue
        if (
            str(region.get("parent_ocr_source_quality_action") or "") == "identity_punctuation"
            or str(region.get("parent_ocr_source_quality_state") or "") == "punctuation_identity_source"
        ):
            text = _canonical_punctuation_identity_source(text)
        candidates.append((len(_source_body(text) or text), _graph_region_reading_key(region), region, text))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _length, _key, dominant_region, source_text = candidates[0]
    dominant_region_id = str(dominant_region.get("region_id") or "")
    dominant_box = _bbox(dominant_region.get("bbox"))
    duplicate_region_ids: set[str] = set()
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid or rid == dominant_region_id:
            continue
        if _graph_region_is_structurally_represented_by_box(region, dominant_box):
            duplicate_region_ids.add(rid)
    return source_text, [dominant_region_id], duplicate_region_ids


def _graph_parent_dominant_boundary_source(
    selected: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    *,
    parent_node: dict[str, Any] | None,
) -> tuple[str, list[str], set[str]] | None:
    if not selected or not parent_node:
        return None
    parent_box = _bbox(parent_node.get("bbox"))
    if not parent_box:
        return None
    candidates: list[tuple[int, tuple[Any, ...], dict[str, Any]]] = []
    for item in selected:
        region = item.get("region") if isinstance(item, dict) else None
        text = str(item.get("text") or "") if isinstance(item, dict) else ""
        if not isinstance(region, dict) or not text:
            continue
        region_box = _bbox(region.get("bbox"))
        if not _graph_region_covers_parent_boundary(region_box, parent_box):
            continue
        status_name, _reasons, action = _parent_source_coherence(
            text,
            role=_graph_parent_role(parent_node),
        )
        if status_name == "malformed" or action in {"repair_required", "block_review_only"}:
            continue
        candidates.append((len(_source_body(text) or text), _graph_region_reading_key(region), item))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _length, _key, dominant = candidates[0]
    dominant_region = dominant["region"]
    dominant_region_id = str(dominant_region.get("region_id") or "")
    dominant_box = _bbox(dominant_region.get("bbox"))
    if not dominant_region_id or not dominant_box:
        return None
    for item in selected:
        region = item.get("region") if isinstance(item, dict) else None
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id") or "")
        if not rid or rid == dominant_region_id:
            continue
        if not _graph_region_is_structurally_represented_by_box(region, dominant_box):
            return None
    duplicate_region_ids: set[str] = set()
    for item in selected:
        region = item.get("region") if isinstance(item, dict) else None
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id") or "")
        if not rid or rid == dominant_region_id:
            continue
        if _graph_region_is_structurally_represented_by_box(region, dominant_box):
            duplicate_region_ids.add(rid)
    return str(dominant["text"]), [dominant_region_id], duplicate_region_ids


def _graph_region_covers_parent_boundary(region_box: list[int], parent_box: list[int]) -> bool:
    if not region_box or not parent_box:
        return False
    intersection = _xywh_intersection_area(region_box, parent_box)
    parent_area = max(1.0, _xywh_area(parent_box))
    region_area = max(1.0, _xywh_area(region_box))
    return intersection / parent_area >= 0.80 and intersection / region_area >= 0.80


def _graph_region_is_structurally_represented_by_box(region: dict[str, Any], source_box: list[int]) -> bool:
    region_box = _bbox(region.get("bbox"))
    if not region_box or not source_box:
        return False
    intersection = _xywh_intersection_area(region_box, source_box)
    region_area = max(1.0, _xywh_area(region_box))
    return intersection / region_area >= 0.80 or _xywh_center_inside(region_box, source_box)


def _graph_parent_reconstructed_source_from_regions(
    regions: list[dict[str, Any]],
    *,
    parent_node: dict[str, Any] | None,
    sibling_parent_nodes: list[dict[str, Any]] | None,
) -> tuple[str, list[str], set[str]] | None:
    candidates: list[tuple[int, tuple[Any, ...], dict[str, Any], str]] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        status = str(region.get("logical_text_source_reconstruction_status") or "").strip()
        if status != "applied":
            continue
        source_text = _clean_source_text(region.get("logical_text_source_reconstruction_after_text"))
        if not source_text:
            continue
        if _graph_region_is_overbroad_for_parent_boundary(
            region,
            parent_node=parent_node,
            sibling_parent_nodes=sibling_parent_nodes,
        ):
            continue
        route_intent = str(
            region.get("text_area_route_intent")
            or (region.get("render") or {}).get("text_area_route_intent")
            or ""
        ).strip()
        if route_intent not in {"", "translate_speech", "translate_caption", "translate_caption_background"}:
            continue
        status_name, _reasons, action = _parent_source_coherence(source_text, role=_graph_parent_role(parent_node or {}))
        if status_name == "malformed" or action in {"repair_required", "block_review_only"}:
            continue
        candidates.append((len(_source_body(source_text)), _graph_region_reading_key(region), region, source_text))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _length, _key, region, source_text = candidates[0]
    return source_text, [str(region.get("region_id") or "")], set()


def _graph_parent_source_from_regions_pass(
    regions: list[dict[str, Any]],
    *,
    parent_node: dict[str, Any] | None,
    sibling_parent_nodes: list[dict[str, Any]] | None,
    allow_ignored_duplicate: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    selected: list[dict[str, Any]] = []
    duplicate_region_ids: set[str] = set()
    for region in sorted(regions, key=_graph_region_reading_key):
        rid = str(region.get("region_id") or "")
        text = _clean_source_text(region.get("ocr_text"))
        if not rid or not text:
            continue
        if not _graph_region_can_author_parent_source(
            region,
            parent_node=parent_node,
            sibling_parent_nodes=sibling_parent_nodes,
            allow_ignored_duplicate=allow_ignored_duplicate,
        ):
            continue
        rank = _graph_region_source_rank(region)
        absorbed = False
        next_selected: list[dict[str, Any]] = []
        for item in selected:
            existing_text = str(item["text"])
            if _source_text_contains(existing_text, text):
                if _source_text_contains(text, existing_text) and rank > item["rank"]:
                    duplicate_region_ids.add(str(item["region"].get("region_id") or ""))
                    continue
                duplicate_region_ids.add(rid)
                absorbed = True
                next_selected.append(item)
                continue
            if _source_text_contains(text, existing_text):
                duplicate_region_ids.add(str(item["region"].get("region_id") or ""))
                continue
            next_selected.append(item)
        if absorbed:
            selected = next_selected
            continue
        next_selected.append({"region": region, "text": text, "rank": rank})
        selected = next_selected
    return selected, duplicate_region_ids


def _graph_region_can_author_parent_source(
    region: dict[str, Any],
    *,
    parent_node: dict[str, Any] | None,
    sibling_parent_nodes: list[dict[str, Any]] | None,
    allow_ignored_duplicate: bool,
) -> bool:
    if _graph_region_is_overbroad_for_parent_boundary(
        region,
        parent_node=parent_node,
        sibling_parent_nodes=sibling_parent_nodes,
    ):
        return False
    flags = region.get("flags") if isinstance(region.get("flags"), dict) else {}
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    child_state = str(region.get("child_final_state") or render.get("child_final_state") or "").strip()
    ownership_status = str(
        region.get("logical_text_ownership_status")
        or render.get("logical_text_ownership_status")
        or ""
    ).strip()
    if bool(flags.get("ignore")) and not (
        allow_ignored_duplicate
        and _graph_region_is_ignored_source_fragment(region, child_state, ownership_status)
    ):
        return False
    if child_state in {
        STATE_DUPLICATE_CHILD,
        STATE_NOISE_REVIEW_ONLY,
        STATE_BLOCKED_BY_ROOT_POLICY,
        STATE_UNRESOLVED_REVIEW_ONLY,
    } and not (
        (allow_ignored_duplicate and child_state == STATE_DUPLICATE_CHILD)
        or (
            allow_ignored_duplicate
            and child_state == STATE_UNRESOLVED_REVIEW_ONLY
            and _graph_region_has_route_owned_source_evidence(region)
        )
        or (
            allow_ignored_duplicate
            and child_state == STATE_NOISE_REVIEW_ONLY
            and _graph_region_has_stale_route_owned_blocker_label(region)
            and _graph_region_has_route_owned_source_evidence(region)
        )
    ):
        return False
    if ownership_status in {
        "noise_review_only",
        "blocked_by_root_policy",
        "unresolved_review_only",
        "duplicate_child",
    } and not (
        allow_ignored_duplicate
        and (
            (
                child_state == STATE_DUPLICATE_CHILD
                and ownership_status in {"duplicate_child", "noise_review_only"}
            )
            or (
                ownership_status in {"unresolved_review_only", "duplicate_child"}
                and _graph_region_has_route_owned_source_evidence(region)
            )
            or (
                ownership_status == "noise_review_only"
                and _graph_region_has_stale_route_owned_blocker_label(region)
                and _graph_region_has_route_owned_source_evidence(region)
            )
        )
    ):
        return False
    source_action = str(
        region.get("logical_text_source_quality_action")
        or render.get("logical_text_source_quality_action")
        or ""
    ).strip()
    return source_action not in {
        "source_quality_blocked",
        "block_auto_translation",
        "split_required",
        "unresolved_review",
        "block_review_only",
    }


def _graph_region_is_ignored_source_fragment(
    region: dict[str, Any],
    child_state: str,
    ownership_status: str,
) -> bool:
    if not _graph_region_has_route_owned_source_evidence(region):
        return False
    if child_state in {STATE_PARENT_ANCHOR, STATE_PARENT_CHILD, STATE_DUPLICATE_CHILD}:
        return ownership_status not in {"noise_review_only", "blocked_by_root_policy"}
    if (
        ownership_status == "noise_review_only"
        and child_state in {"", STATE_NOISE_REVIEW_ONLY, STATE_UNRESOLVED_REVIEW_ONLY}
        and _graph_region_has_stale_route_owned_blocker_label(region)
    ):
        return True
    return ownership_status in {"transferred_child", "duplicate_child", "block_anchor"}


def _graph_region_has_stale_route_owned_blocker_label(region: dict[str, Any]) -> bool:
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    flags = region.get("flags") if isinstance(region.get("flags"), dict) else {}
    cleanup_mode = str(render.get("cleanup_mode") or region.get("cleanup_mode") or "").strip()
    classification = str(region.get("classification_reason") or render.get("classification_reason") or "").strip()
    return bool(
        region.get("translation_blocked_by_ocr_transaction")
        or render.get("translation_blocked_by_ocr_transaction")
        or flags.get("hard_fail")
        or cleanup_mode == "ocr_blocked_no_cleanup"
        or classification == "text_area_route_authority_root_blocker"
    )


def _graph_region_has_route_owned_source_evidence(region: dict[str, Any]) -> bool:
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    if not _clean_source_text(region.get("ocr_text")):
        return False
    ocr_state = str(
        region.get("text_area_ocr_transaction_state")
        or render.get("text_area_ocr_transaction_state")
        or ""
    ).strip()
    if ocr_state and ocr_state not in OCR_TRANSLATION_QUEUED_STATES:
        return False
    route_intent = str(
        region.get("text_area_route_intent")
        or render.get("text_area_route_intent")
        or ""
    ).strip()
    return route_intent in {"", "translate_speech", "translate_caption", "translate_caption_background"}


def _graph_region_is_overbroad_for_parent_boundary(
    region: dict[str, Any],
    *,
    parent_node: dict[str, Any] | None,
    sibling_parent_nodes: list[dict[str, Any]] | None,
) -> bool:
    if not parent_node:
        return False
    parent_box = _bbox(parent_node.get("bbox"))
    region_box = _bbox(region.get("bbox"))
    if not parent_box or not region_box:
        return False
    parent_area = max(1.0, _xywh_area(parent_box))
    region_area = max(1.0, _xywh_area(region_box))
    if region_area <= parent_area * 1.35:
        return False
    intersection = _xywh_intersection_area(parent_box, region_box)
    if intersection / parent_area < 0.80:
        return False
    if intersection / region_area >= 0.78:
        return False
    parent_id = str(parent_node.get("parent_node_id") or "")
    container_id = str(parent_node.get("container_id") or "")
    for sibling in sibling_parent_nodes or []:
        if not isinstance(sibling, dict):
            continue
        if str(sibling.get("parent_node_id") or "") == parent_id:
            continue
        if container_id and str(sibling.get("container_id") or "") != container_id:
            continue
        sibling_box = _bbox(sibling.get("bbox"))
        if sibling_box and _xywh_intersection_area(sibling_box, region_box) > 0:
            return True
    return False


def _graph_attached_non_authoring_child_state(region: dict[str, Any]) -> str:
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    child_state = str(region.get("child_final_state") or render.get("child_final_state") or "").strip()
    if child_state == STATE_DUPLICATE_CHILD:
        return STATE_DUPLICATE_CHILD
    if child_state in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT, STATE_PARENT_CHILD}:
        return STATE_DUPLICATE_CHILD
    if child_state in {STATE_NOISE_REVIEW_ONLY, STATE_BLOCKED_BY_ROOT_POLICY}:
        return child_state
    ownership_status = str(
        region.get("logical_text_ownership_status")
        or render.get("logical_text_ownership_status")
        or ""
    ).strip()
    if ownership_status == "duplicate_child":
        return STATE_DUPLICATE_CHILD
    if ownership_status in {"noise_review_only", "blocked_by_root_policy"}:
        return ownership_status
    flags = region.get("flags") if isinstance(region.get("flags"), dict) else {}
    if bool(flags.get("ignore")):
        return STATE_NOISE_REVIEW_ONLY
    if _clean_source_text(region.get("ocr_text")):
        return STATE_DUPLICATE_CHILD
    return STATE_UNRESOLVED_REVIEW_ONLY


def _child_from_graph_region(
    *,
    page_id: str,
    region: dict[str, Any],
    root_id: str,
    parent_id: str,
    final_state: str,
    source_authoring: bool,
) -> ChildRecognizedTextSegment:
    rid = str(region.get("region_id") or "")
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    represented_by = parent_id if final_state in {
        STATE_PARENT_ANCHOR,
        STATE_PARENT_CHILD,
        STATE_DUPLICATE_CHILD,
        STATE_PUNCTUATION_CHILD,
    } else None
    reasons = _child_reason_codes(region, str(region.get("logical_text_ownership_status") or ""), final_state)
    _append_unique(reasons, "text_area_graph_child_attachment")
    if source_authoring and _graph_region_has_stale_route_owned_blocker_label(region):
        _append_unique(reasons, ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON)
    child = ChildRecognizedTextSegment(
        child_id=_child_id(page_id, rid),
        page_id=page_id,
        root_id=root_id,
        parent_id=parent_id,
        source_region_id=rid,
        bbox=_bbox(region.get("bbox")),
        polygon=list(region.get("polygon") or []),
        ocr_text=str(region.get("ocr_text") or ""),
        ocr_confidence=_ocr_confidence(region),
        source_orientation=render.get("source_orientation"),
        detection_source=region.get("text_area_detection_source"),
        final_state=final_state,
        represented_by_parent_id=represented_by,
        translated_independently=False,
        cleanup_independently=final_state == STATE_PARENT_ANCHOR,
        render_independently=final_state == STATE_PARENT_ANCHOR,
        reason_codes=reasons,
        confidence=_ocr_confidence(region),
    )
    if final_state == STATE_PARENT_ANCHOR:
        region["active_translation_unit_id"] = parent_id
    return child


def _graph_parent_confidence(regions: list[dict[str, Any]]) -> float | None:
    values = [_ocr_confidence(region) for region in regions]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _graph_region_source_rank(region: dict[str, Any]) -> tuple[int, int, float]:
    reconstructed = int(
        str(region.get("logical_text_source_reconstruction_status") or "").startswith("root_reconstruction")
        or str(region.get("text_area_detection_source") or "") == "root_reconstruction_reocr"
    )
    text = _clean_source_text(region.get("ocr_text"))
    return (reconstructed, len(_source_body(text) or text), float(_ocr_confidence(region) or 0.0))


def _graph_region_reading_key(region: dict[str, Any]) -> tuple[Any, ...]:
    box = _bbox(region.get("bbox"))
    if not box:
        return (0, 0, str(region.get("region_id") or ""))
    x, y, w, _h = box
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    orientation = str(region.get("source_orientation") or render.get("source_orientation") or "").lower()
    if orientation.startswith("horizontal"):
        return (y, x, str(region.get("region_id") or ""))
    center_x = x + (w / 2.0)
    column = int(round(center_x / 24.0))
    return (-column, y, -x, str(region.get("region_id") or ""))


def _source_text_contains(container: str, contained: str) -> bool:
    outer = _clean_source_text(container)
    inner = _clean_source_text(contained)
    if not outer or not inner:
        return False
    outer_body = _source_body(outer)
    inner_body = _source_body(inner)
    if outer_body and inner_body:
        return inner_body in outer_body
    return inner in outer


def _xywh_area(box: list[int]) -> float:
    if not box or len(box) < 4:
        return 0.0
    return float(max(0, int(box[2])) * max(0, int(box[3])))


def _xywh_intersection_area(a: list[int], b: list[int]) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + max(0, aw), ay0 + max(0, ah)
    bx1, by1 = bx0 + max(0, bw), by0 + max(0, bh)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    return float(max(0, ix1 - ix0) * max(0, iy1 - iy0))


def _xywh_center_inside(inner: list[int], outer: list[int]) -> bool:
    if not inner or not outer or len(inner) < 4 or len(outer) < 4:
        return False
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx = ix + iw / 2.0
    cy = iy + ih / 2.0
    return ox <= cx <= ox + max(0, ow) and oy <= cy <= oy + max(0, oh)


def _container_authorization_state(container: dict[str, Any]) -> str:
    return str(
        container.get("semantic_authorization_state")
        or container.get("cleanup_authorization")
        or ""
    ).strip()


def _container_allows_legacy_root_materialization(container: dict[str, Any]) -> bool:
    if bool(container.get("must_not_mutate")):
        return False

    auth = _container_authorization_state(container)
    explicit = bool(container.get("authorization_explicit"))
    if auth in EXECUTABLE_CLEANUP_AUTHORIZATIONS:
        return explicit
    if auth in NON_EXECUTABLE_AUTHORIZATIONS:
        return False
    if auth:
        return False

    route = str(container.get("route_intent") or "")
    container_type = str(container.get("container_type") or "")
    if route not in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}:
        return False
    if container_type not in {ROOT_SPEECH, ROOT_CAPTION, "speech", "caption", "caption_background"}:
        return False
    return not bool(container.get("human_review_required"))


def _region_has_text_area_authority_fields(region: dict[str, Any]) -> bool:
    return any(str(key).startswith("text_area_") for key in region.keys())


def _region_authorization_state(region: dict[str, Any]) -> str:
    return str(
        region.get("text_area_semantic_authorization_state")
        or region.get("semantic_authorization_state")
        or region.get("text_area_cleanup_authorization")
        or region.get("cleanup_authorization")
        or ""
    ).strip()


def _region_allows_standalone_root_materialization(region: dict[str, Any]) -> bool:
    if not _region_has_text_area_authority_fields(region):
        return True
    if bool(region.get("text_area_must_not_mutate") or region.get("must_not_mutate")):
        return False
    auth = _region_authorization_state(region)
    explicit = bool(region.get("text_area_authorization_explicit") or region.get("authorization_explicit"))
    if auth in EXECUTABLE_CLEANUP_AUTHORIZATIONS:
        return explicit
    if auth in NON_EXECUTABLE_AUTHORIZATIONS:
        return False
    if auth:
        return False
    route = str(region.get("text_area_route_intent") or "").strip()
    if route in {"translate_speech", "translate_caption_background", "translate_caption"}:
        return False
    return True


def _build_roots(
    page_id: str,
    plan: dict[str, Any],
    physical_groups: list[dict[str, Any]],
) -> dict[str, TextAreaRootBlock]:
    roots: dict[str, TextAreaRootBlock] = {}
    graph_roots = _graph_root_nodes(plan)
    if graph_roots:
        containers_by_root: dict[str, list[str]] = {}
        for parent in _graph_parent_nodes(plan):
            root_id = str(parent.get("root_node_id") or "")
            container_id = str(parent.get("container_id") or "")
            if root_id and container_id:
                containers_by_root.setdefault(root_id, [])
                _append_unique(containers_by_root[root_id], container_id)
        for index, node in enumerate(graph_roots):
            root_id = str(node.get("root_node_id") or "")
            container_id = str(node.get("container_id") or "")
            if not root_id:
                continue
            container_ids = list(containers_by_root.get(root_id) or [])
            if container_id:
                _append_unique(container_ids, container_id)
            roots[root_id] = TextAreaRootBlock(
                root_id=root_id,
                page_id=page_id,
                root_type=_graph_root_type(node),
                text_area_container_ids=container_ids,
                bbox=_bbox(node.get("bbox")),
                mask_source_ids=_string_list(node.get("source_evidence_ids")),
                route_policy=_graph_route_policy(node),
                ocr_eligible=_graph_route_policy(node) in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION},
                ctd_scope_eligible=True,
                reading_order_index=index,
                reason_codes=_string_list(node.get("reason_codes")) + ["text_area_graph_root_materialized"],
            )
        return roots

    container_to_physical: dict[str, dict[str, Any]] = {}
    for group in physical_groups:
        physical_id = str(group.get("logical_text_physical_bubble_id") or group.get("physical_bubble_graph_id") or "")
        for cid in group.get("logical_text_physical_bubble_member_container_ids") or []:
            container_to_physical[str(cid)] = group
        if physical_id:
            root_id = _root_id(page_id, physical_id)
            roots[root_id] = TextAreaRootBlock(
                root_id=root_id,
                page_id=page_id,
                root_type=ROOT_SPEECH,
                text_area_container_ids=[str(cid) for cid in group.get("logical_text_physical_bubble_member_container_ids") or []],
                physical_bubble_id=physical_id,
                bbox=_bbox(group.get("logical_text_physical_bubble_bbox")),
                route_policy=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                ctd_scope_eligible=True,
                reason_codes=list(group.get("logical_text_physical_bubble_reason_codes") or []),
            )
    index = 0
    for container in plan.get("containers") or []:
        if not isinstance(container, dict):
            continue
        cid = str(container.get("container_id") or "")
        if not cid:
            continue
        physical = container_to_physical.get(cid)
        if physical:
            root = roots.get(_root_id(page_id, str(physical.get("logical_text_physical_bubble_id") or "")))
            if root:
                root.ocr_eligible = root.ocr_eligible or bool(container.get("ocr_eligible"))
                root.ctd_scope_eligible = root.ctd_scope_eligible or bool(container.get("comic_text_detector_scope_eligible"))
                root.confidence_tier = root.confidence_tier or container.get("confidence_tier")
                for reason in container.get("evidence_reason_codes") or []:
                    _append_unique(root.reason_codes, str(reason))
                continue
        if not _container_allows_legacy_root_materialization(container):
            continue
        root_type = _root_type(container)
        root = TextAreaRootBlock(
            root_id=_root_id(page_id, cid),
            page_id=page_id,
            root_type=root_type,
            text_area_container_ids=[cid],
            bbox=_bbox(container.get("bbox")),
            mask_source_ids=[str(value) for value in (container.get("source_model_ids") or []) if str(value)],
            route_policy=_route_policy(container),
            ocr_eligible=bool(container.get("ocr_eligible")),
            ctd_scope_eligible=bool(container.get("comic_text_detector_scope_eligible")),
            reading_order_index=index,
            reason_codes=[str(value) for value in (container.get("evidence_reason_codes") or []) if str(value)],
            confidence_tier=container.get("confidence_tier"),
            fallback_reason=container.get("fallback_reason"),
            review_reason=container.get("fallback_reason") if root_type in {ROOT_UNKNOWN, ROOT_REVIEW} else None,
            cleanup_authorization=str(container.get("cleanup_authorization") or ""),
            semantic_authorization_state=str(container.get("semantic_authorization_state") or ""),
            authorization_explicit=bool(container.get("authorization_explicit")),
            must_not_mutate=bool(container.get("must_not_mutate")),
            human_review_required=bool(container.get("human_review_required")),
        )
        roots[root.root_id] = root
        index += 1
    return roots


def _parent_from_logical_block(
    page_id: str,
    block: dict[str, Any],
    roots_by_container: dict[str, TextAreaRootBlock],
    roots_by_physical: dict[str, TextAreaRootBlock],
) -> ParentLogicalTextUnit:
    block_id = str(block.get("logical_text_block_id") or "")
    physical_id = str(block.get("logical_text_physical_bubble_id") or block.get("physical_bubble_graph_id") or "")
    container_id = str(block.get("logical_text_block_container_id") or "")
    root = roots_by_physical.get(physical_id) or roots_by_container.get(container_id)
    root_id = root.root_id if root else _root_id(page_id, physical_id or container_id or block_id)
    translation_unit = bool(block.get("logical_text_block_translation_unit", True))
    anchor_region_id = str(block.get("logical_text_block_anchor_region_id") or "")
    anchor_child_id = _child_id(page_id, anchor_region_id) if anchor_region_id else None
    member_region_ids = [str(rid) for rid in (block.get("logical_text_block_member_region_ids") or []) if str(rid)]
    block_role = str(block.get("logical_text_block_role") or "")
    if block_role == "speech_bubble":
        role = ROLE_SPEECH
    elif block_role == "caption_background":
        role = ROLE_CAPTION
    else:
        role = ROLE_REVIEW
    return ParentLogicalTextUnit(
        parent_id=block_id,
        page_id=page_id,
        root_id=root_id,
        role=role,
        source_text=str(block.get("logical_text_block_source_text") or ""),
        source_text_before_reconstruction=str(block.get("logical_text_source_reconstruction_before_text") or ""),
        source_reconstruction_status=str(block.get("logical_text_source_reconstruction_status") or "not_attempted"),
        source_reconstruction_crop_bbox=_bbox(block.get("logical_text_source_reconstruction_crop_bbox")),
        source_reconstruction_confidence=_float_or_none(block.get("logical_text_source_reconstruction_ocr_confidence")),
        anchor_child_id=anchor_child_id,
        child_segment_ids=[_child_id(page_id, rid) for rid in member_region_ids],
        dependent_child_ids=[_child_id(page_id, rid) for rid in (block.get("logical_text_block_transferred_region_ids") or [])],
        duplicate_child_ids=[_child_id(page_id, rid) for rid in (block.get("logical_text_block_duplicate_region_ids") or [])],
        punctuation_child_ids=[_child_id(page_id, rid) for rid in (block.get("logical_text_block_punctuation_child_ids") or [])],
        noise_child_ids=[_child_id(page_id, rid) for rid in (block.get("logical_text_block_noise_child_ids") or [])],
        rejected_child_ids=[_child_id(page_id, rid) for rid in (block.get("logical_text_source_reconstruction_rejected_child_region_ids") or [])],
        translation_unit=translation_unit,
        cleanup_unit=translation_unit,
        render_unit=translation_unit,
        cleanup_target_bbox=_bbox(block.get("logical_text_block_bbox") or block.get("logical_text_block_allowed_bbox")),
        render_allowed_area=_bbox(block.get("logical_text_block_allowed_bbox") or block.get("logical_text_block_bbox")),
        source_conservation_status=str(block.get("source_conservation_status") or block.get("logical_text_block_text_conservation_status") or "complete"),
        unresolved_reason=block.get("logical_text_block_unresolved_reason") or block.get("source_conservation_failure_reason"),
        reason_codes=[str(value) for value in (block.get("logical_text_block_reason_codes") or []) if str(value)],
        confidence=_float_or_none(block.get("logical_text_block_confidence")),
        parent_visual_group_id=str(block.get("logical_text_parent_visual_group_id") or block.get("parent_visual_group_id") or ""),
        parent_visual_group_bbox=_bbox(block.get("logical_text_parent_visual_group_bbox") or block.get("parent_visual_group_bbox")),
        parent_visual_group_child_ids=[str(value) for value in (block.get("logical_text_parent_visual_group_child_ids") or block.get("parent_visual_group_child_ids") or []) if str(value)],
        reconstruction_rejected_for_visual_overmerge=bool(block.get("logical_text_reconstruction_rejected_for_visual_overmerge") or block.get("reconstruction_rejected_for_visual_overmerge")),
    )


def _child_from_region(
    page_id: str,
    region: dict[str, Any],
    root_id: str,
    parent_id: str | None,
    *,
    block: dict[str, Any] | None,
) -> ChildRecognizedTextSegment:
    rid = str(region.get("region_id") or "")
    render = region.get("render") if isinstance(region.get("render"), dict) else {}
    status = str(region.get("logical_text_ownership_status") or render.get("logical_text_ownership_status") or "")
    final_state = _child_final_state(region, status, bool(block), bool(parent_id))
    represented_by = parent_id if final_state in {
        STATE_PARENT_ANCHOR,
        STATE_PARENT_CHILD,
        STATE_STANDALONE_PARENT,
        STATE_DUPLICATE_CHILD,
        STATE_PUNCTUATION_CHILD,
    } else None
    translated_independently = final_state == STATE_STANDALONE_PARENT
    active_parent = parent_id if final_state in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT} else None
    child = ChildRecognizedTextSegment(
        child_id=_child_id(page_id, rid),
        page_id=page_id,
        root_id=root_id,
        parent_id=parent_id,
        source_region_id=rid,
        bbox=_bbox(region.get("bbox")),
        polygon=list(region.get("polygon") or []),
        ocr_text=str(region.get("ocr_text") or ""),
        ocr_confidence=_ocr_confidence(region),
        source_orientation=render.get("source_orientation"),
        detection_source=region.get("text_area_detection_source"),
        final_state=final_state,
        represented_by_parent_id=represented_by,
        translated_independently=translated_independently,
        cleanup_independently=final_state in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT},
        render_independently=final_state in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT},
        reason_codes=_child_reason_codes(region, status, final_state),
        confidence=_ocr_confidence(region),
    )
    if active_parent:
        region["active_translation_unit_id"] = active_parent
    return child


def _standalone_parent_from_child(
    page_id: str,
    root_id: str,
    child: ChildRecognizedTextSegment,
    region: dict[str, Any],
) -> ParentLogicalTextUnit:
    parent_id = f"ptu_{page_id}_{child.source_region_id}"
    parent = ParentLogicalTextUnit(
        parent_id=parent_id,
        page_id=page_id,
        root_id=root_id,
        role=_role_for_region(region),
        source_text=child.ocr_text,
        anchor_child_id=child.child_id,
        child_segment_ids=[child.child_id],
        translation_unit=True,
        cleanup_unit=True,
        render_unit=True,
        cleanup_target_bbox=list(child.bbox),
        render_allowed_area=_bbox(region.get("text_area_container_bbox") or region.get("bbox")),
        source_conservation_status="complete",
        reason_codes=["standalone_parent_from_active_region"],
        confidence=child.confidence,
    )
    return parent


def _stamp_regions(regions: list[dict[str, Any]], result: TextBlockHierarchyResult) -> None:
    fields_by_region = result.region_audit_fields()
    parent_by_id = {parent.parent_id: parent for parent in result.parent_units}
    for region in regions:
        rid = str(region.get("region_id") or "")
        fields = fields_by_region.get(rid)
        if not fields:
            continue
        parent = parent_by_id.get(str(fields.get("parent_logical_text_unit_id") or ""))
        if parent:
            fields["parent_logical_text_unit_child_segment_ids"] = list(parent.child_segment_ids)
            fields["parent_logical_text_unit_anchor_child_id"] = parent.anchor_child_id
        render = region.setdefault("render", {})
        for key, value in fields.items():
            region[key] = value
            render[key] = value
        route_intent = str(
            region.get("text_area_route_intent")
            or render.get("text_area_route_intent")
            or ""
        ).strip()
        if route_intent in {"translate_speech", "translate_caption_background"}:
            state = str(
                region.get("text_area_ocr_transaction_state")
                or render.get("text_area_ocr_transaction_state")
                or ""
            ).strip()
            represented_blocker = _route_owned_ocr_fragment_represented_by_region(region)
            blocked = _route_owned_ocr_blocker(region) and not represented_blocker
            warning = state == OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING
            parent_active = bool(parent and _parent_is_active_translation_unit(parent))
            active_source_anchor = bool(
                parent_active
                and fields.get("child_final_state") in {STATE_PARENT_ANCHOR, STATE_STANDALONE_PARENT}
            )
            region["route_owned_translatable"] = True
            region["route_owned_ocr_blocked"] = blocked
            region["route_owned_ocr_fragment_represented_by_parent"] = represented_blocker
            region["route_owned_ocr_warning"] = warning
            region["route_owned_translation_queued"] = bool(parent_active and not blocked)
            region["parent_activation_state"] = (
                "blocked_before_parent_activation"
                if blocked
                else (
                    "represented_by_active_translation_parent"
                    if represented_blocker and parent_active
                    else ("active_translation_parent" if parent_active else "inactive_no_translation_parent")
                )
            )
            region["render_activation_state"] = (
                "blocked_before_translation"
                if blocked
                else (
                    "represented_by_parent_render_unit"
                    if represented_blocker and bool(parent and parent.render_unit)
                    else ("active_render_unit" if bool(parent and parent.render_unit) else "inactive_no_render_unit")
                )
            )
            region["cleanup_activation_state"] = (
                "blocked_before_translation"
                if blocked
                else (
                    "represented_by_parent_cleanup_unit"
                    if represented_blocker and bool(parent and parent.cleanup_unit)
                    else ("active_cleanup_unit" if bool(parent and parent.cleanup_unit) else "inactive_no_cleanup_unit")
                )
            )
            for key in (
                "route_owned_translatable",
                "route_owned_ocr_blocked",
                "route_owned_ocr_fragment_represented_by_parent",
                "route_owned_ocr_warning",
                "route_owned_translation_queued",
                "parent_activation_state",
                "render_activation_state",
                "cleanup_activation_state",
            ):
                render[key] = region[key]
            if active_source_anchor:
                flags = region.setdefault("flags", {})
                flags["ignore"] = False
                flags.pop("hard_fail", None)
                region.pop("skip_reason", None)
                region.pop("translation_blocked_by_ocr_transaction", None)
                render.pop("translation_blocked_by_ocr_transaction", None)
                if route_intent == "translate_speech":
                    render["cleanup_mode"] = "bubble"
                elif route_intent == "translate_caption_background":
                    render["cleanup_mode"] = "background"
            if represented_blocker:
                flags = region.setdefault("flags", {})
                flags["ignore"] = False
                flags.pop("hard_fail", None)
                region.pop("translation_blocked_by_ocr_transaction", None)
                render.pop("translation_blocked_by_ocr_transaction", None)
                render["classification_reason"] = "route_owned_ocr_fragment_represented_by_parent"
                if route_intent == "translate_speech":
                    render["cleanup_mode"] = "bubble"
                elif route_intent == "translate_caption_background":
                    render["cleanup_mode"] = "background"
            if blocked:
                flags = region.setdefault("flags", {})
                flags["ignore"] = False
                flags["needs_review"] = True
                flags["hard_fail"] = True
                region["translation"] = ""
                region["translated_text"] = ""
                region["translation_blocked_by_ocr_transaction"] = True
                region["logical_text_block_translation_unit"] = False
                region["active_translation_unit_id"] = None
                region["source_text_represented_by_block_id"] = None
                render["translation_blocked_by_ocr_transaction"] = True
                render["logical_text_block_translation_unit"] = False
                render["active_translation_unit_id"] = None
                render["source_text_represented_by_block_id"] = None
                render["cleanup_mode"] = "ocr_blocked_no_cleanup"
                render.pop("final_render_bbox", None)
                render.pop("wrapped_lines", None)
                region.pop("final_render_bbox", None)
                region.pop("wrapped_lines", None)
        if fields.get("standalone_parent_rejected") or (
            fields.get("root_requires_reconstruction")
            and fields.get("child_final_state") == STATE_UNRESOLVED_REVIEW_ONLY
            and fields.get("parent_source_coherence_action") in {"repair_required", "block_review_only"}
        ) or (
            fields.get("root_validation_blocker")
            and fields.get("root_transaction_status") == "root_review_only_unresolved"
            and fields.get("child_final_state") not in {STATE_BLOCKED_BY_ROOT_POLICY, STATE_PUNCTUATION_CHILD}
        ):
            reason = str(fields.get("root_source_coherence_failure_reason") or "root_source_coherence_review_only")
            route_intent = str(
                region.get("text_area_route_intent")
                or (region.get("render") or {}).get("text_area_route_intent")
                or ""
            ).strip()
            if route_intent in {"translate_speech", "translate_caption_background"}:
                region["text_area_ocr_transaction_state"] = region.get("text_area_ocr_transaction_state") or "ocr_malformed_blocker"
                region["text_area_ocr_blocker_reason"] = region.get("text_area_ocr_blocker_reason") or reason
                flags = region.setdefault("flags", {})
                flags["ignore"] = False
                flags["needs_review"] = True
                flags["hard_fail"] = True
                flags["bg_text"] = route_intent == "translate_caption_background"
                region["translation"] = ""
                region["translated_text"] = ""
                region["translation_blocked_by_ocr_transaction"] = True
                region["logical_text_block_translation_unit"] = False
                region["active_translation_unit_id"] = None
                region["route_owned_translation_queued"] = False
                region["parent_activation_state"] = "blocked_before_parent_activation"
                region["render_activation_state"] = "blocked_before_translation"
                region["cleanup_activation_state"] = "blocked_before_translation"
                render["classification_reason"] = "text_area_route_authority_root_blocker"
                render["cleanup_mode"] = "ocr_blocked_no_cleanup"
                render["translation_blocked_by_ocr_transaction"] = True
                render["text_area_ocr_transaction_state"] = region["text_area_ocr_transaction_state"]
                render["text_area_ocr_blocker_reason"] = region["text_area_ocr_blocker_reason"]
                render["route_owned_translation_queued"] = False
                render["parent_activation_state"] = region["parent_activation_state"]
                render["render_activation_state"] = region["render_activation_state"]
                render["cleanup_activation_state"] = region["cleanup_activation_state"]
                render.pop("final_render_bbox", None)
                render.pop("wrapped_lines", None)
                region.pop("final_render_bbox", None)
                region.pop("wrapped_lines", None)
                continue
            region["root_source_coherence_review_only"] = True
            region["root_transaction_cleared_unsafe_fragment"] = True
            region["skip_reason"] = "root_source_coherence_review_only"
            flags = region.setdefault("flags", {})
            flags["ignore"] = True
            flags["bg_text"] = False
            flags["needs_review"] = True
            region["translation"] = ""
            region["translated_text"] = ""
            region["logical_text_block_translation_unit"] = False
            region["active_translation_unit_id"] = None
            render["root_source_coherence_review_only"] = True
            render["root_transaction_cleared_unsafe_fragment"] = True
            render["classification_reason"] = "root_source_coherence_review_only"
            render["cleanup_mode"] = "preserve"
            render["logical_text_render_eligibility_repaired"] = True
            render["logical_text_render_eligibility_reason"] = reason
            render.pop("final_render_bbox", None)
            render.pop("wrapped_lines", None)
            region.pop("final_render_bbox", None)
            region.pop("wrapped_lines", None)


def _root_for_region(
    page_id: str,
    region: dict[str, Any],
    roots_by_container: dict[str, TextAreaRootBlock],
    roots_by_key: dict[str, TextAreaRootBlock],
) -> TextAreaRootBlock:
    container_id = str(region.get("text_area_container_id") or "")
    root = roots_by_container.get(container_id)
    if root:
        return root
    root_id = _root_id(page_id, f"region_{region.get('region_id') or len(roots_by_key)}")
    allowed = _region_allows_standalone_root_materialization(region)
    root_type = _root_type_for_region(region) if allowed else ROOT_REVIEW
    root = TextAreaRootBlock(
        root_id=root_id,
        page_id=page_id,
        root_type=root_type,
        text_area_container_ids=[container_id] if container_id else [],
        bbox=_bbox(region.get("text_area_container_bbox") or region.get("bbox")),
        route_policy=_route_policy_for_region(region) if allowed else ROUTE_REVIEW,
        ocr_eligible=allowed and not bool((region.get("flags") or {}).get("ignore")),
        ctd_scope_eligible=False,
        fallback_reason=region.get("text_area_fallback_reason"),
        review_reason=region.get("skip_reason"),
        reason_codes=[str(value) for value in (region.get("text_area_reason_codes") or []) if str(value)],
        cleanup_authorization=str(region.get("text_area_cleanup_authorization") or region.get("cleanup_authorization") or ""),
        semantic_authorization_state=str(
            region.get("text_area_semantic_authorization_state")
            or region.get("semantic_authorization_state")
            or ""
        ),
        authorization_explicit=bool(region.get("text_area_authorization_explicit") or region.get("authorization_explicit")),
        must_not_mutate=bool(region.get("text_area_must_not_mutate") or region.get("must_not_mutate")),
        human_review_required=not allowed,
    )
    roots_by_key[root_id] = root
    return root


def _child_final_state(region: dict[str, Any], status: str, has_block: bool, has_parent: bool) -> str:
    flags = region.get("flags") or {}
    route = str(region.get("text_area_route_intent") or "")
    container_type = str(region.get("text_area_container_type") or "")
    if not _region_allows_standalone_root_materialization(region):
        return STATE_UNRESOLVED_REVIEW_ONLY
    if container_type == ROOT_SFX or route == "preserve_sfx_decorative":
        return STATE_BLOCKED_BY_ROOT_POLICY
    if _route_owned_ocr_blocker(region):
        return STATE_UNRESOLVED_REVIEW_ONLY
    if status == "block_anchor":
        return STATE_PARENT_ANCHOR
    if status == "standalone_block" or status == "standalone_utterance":
        return STATE_STANDALONE_PARENT if has_parent else STATE_UNRESOLVED_REVIEW_ONLY
    if status == "transferred_child":
        return STATE_PARENT_CHILD
    if status == "dependent_child":
        return STATE_PARENT_CHILD
    if status == "duplicate_child":
        return STATE_DUPLICATE_CHILD
    if status == "punctuation_child":
        return STATE_PUNCTUATION_CHILD
    if status == "noise_review_only":
        return STATE_NOISE_REVIEW_ONLY
    if flags.get("ignore"):
        return STATE_BLOCKED_BY_ROOT_POLICY if container_type in {ROOT_SFX, ROOT_UNKNOWN} else STATE_UNRESOLVED_REVIEW_ONLY
    if str(region.get("translation") or "").strip() or (not has_block and str(region.get("ocr_text") or "").strip()):
        return STATE_STANDALONE_PARENT
    return STATE_UNRESOLVED_REVIEW_ONLY


def _graph_child_ownership_status(final_state: str) -> str:
    if final_state == STATE_PARENT_ANCHOR:
        return "block_anchor"
    if final_state == STATE_PARENT_CHILD:
        return "transferred_child"
    if final_state == STATE_DUPLICATE_CHILD:
        return "duplicate_child"
    if final_state == STATE_PUNCTUATION_CHILD:
        return "punctuation_child"
    if final_state == STATE_NOISE_REVIEW_ONLY:
        return "noise_review_only"
    if final_state == STATE_STANDALONE_PARENT:
        return "standalone_block"
    return "unresolved_review_only"


def _route_owned_ocr_blocker(region: dict[str, Any] | None) -> bool:
    if not isinstance(region, dict):
        return False
    route = str(region.get("text_area_route_intent") or (region.get("render") or {}).get("text_area_route_intent") or "").strip()
    if route not in {"translate_speech", "translate_caption_background"}:
        return False
    state = str(
        region.get("text_area_ocr_transaction_state")
        or (region.get("render") or {}).get("text_area_ocr_transaction_state")
        or ""
    ).strip()
    return state in OCR_TRANSACTION_BLOCKER_STATES or bool(region.get("translation_blocked_by_ocr_transaction"))


def _route_owned_ocr_fragment_represented_by_region(region: dict[str, Any] | None) -> bool:
    if not isinstance(region, dict):
        return False
    reasons = region.get("hierarchy_reason_codes")
    if not isinstance(reasons, list):
        render = region.get("render") if isinstance(region.get("render"), dict) else {}
        reasons = render.get("hierarchy_reason_codes") if isinstance(render.get("hierarchy_reason_codes"), list) else []
    return ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON in {str(reason) for reason in reasons or []}


def _route_owned_translation_warning(region: dict[str, Any] | None) -> bool:
    if not isinstance(region, dict):
        return False
    route = str(region.get("text_area_route_intent") or (region.get("render") or {}).get("text_area_route_intent") or "").strip()
    if route not in {"translate_speech", "translate_caption_background"}:
        return False
    state = str(
        region.get("text_area_ocr_transaction_state")
        or (region.get("render") or {}).get("text_area_ocr_transaction_state")
        or ""
    ).strip()
    return state == OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING


def _child_reason_codes(region: dict[str, Any], status: str, final_state: str) -> list[str]:
    reasons = [str(value) for value in (region.get("logical_text_block_reason_codes") or []) if str(value)]
    if status:
        _append_unique(reasons, f"logical_text_status:{status}")
    _append_unique(reasons, f"hierarchy_final_state:{final_state}")
    if region.get("text_area_container_type"):
        _append_unique(reasons, f"text_area_container:{region.get('text_area_container_type')}")
    if region.get("text_area_ocr_transaction_state"):
        _append_unique(reasons, f"text_area_ocr_transaction:{region.get('text_area_ocr_transaction_state')}")
    if region.get("text_area_ocr_blocker_reason"):
        _append_unique(reasons, f"text_area_ocr_blocker:{region.get('text_area_ocr_blocker_reason')}")
    if region.get("text_area_ocr_warning_reason"):
        _append_unique(reasons, f"text_area_ocr_warning:{region.get('text_area_ocr_warning_reason')}")
    if region.get("skip_reason"):
        _append_unique(reasons, f"skip_reason:{region.get('skip_reason')}")
    return reasons


def _ocr_transaction_state_from_child(child: ChildRecognizedTextSegment) -> str:
    for reason in child.reason_codes or []:
        text = str(reason or "")
        if text.startswith("text_area_ocr_transaction:"):
            return text.split(":", 1)[1].strip()
    return ""


def _roots_by_container(roots_by_key: dict[str, TextAreaRootBlock]) -> dict[str, TextAreaRootBlock]:
    result: dict[str, TextAreaRootBlock] = {}
    for root in roots_by_key.values():
        for cid in root.text_area_container_ids:
            result[str(cid)] = root
    return result



def _logical_blocks(logical_block_result: Any | None) -> list[dict[str, Any]]:
    if logical_block_result is None:
        return []
    blocks = getattr(logical_block_result, "blocks", None)
    if blocks is None and isinstance(logical_block_result, dict):
        blocks = logical_block_result.get("logical_text_blocks") or []
    result: list[dict[str, Any]] = []
    for block in blocks or []:
        result.append(_to_dict(block))
    return result


def _physical_groups(logical_block_result: Any | None) -> list[dict[str, Any]]:
    if logical_block_result is None:
        return []
    groups = getattr(logical_block_result, "physical_bubble_groups", None)
    if groups is None and isinstance(logical_block_result, dict):
        groups = logical_block_result.get("logical_text_physical_bubble_groups") or []
    return [_to_dict(group) for group in groups or []]


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def _root_type(container: dict[str, Any]) -> str:
    value = str(container.get("container_type") or "").strip()
    if value in {ROOT_SPEECH, ROOT_CAPTION, ROOT_SFX, ROOT_UNKNOWN}:
        return value
    return ROOT_REVIEW


def _root_type_for_region(region: dict[str, Any]) -> str:
    value = str(region.get("text_area_container_type") or "").strip()
    if value in {ROOT_SPEECH, ROOT_CAPTION, ROOT_SFX, ROOT_UNKNOWN}:
        return value
    if str(region.get("type") or "") == "background_text":
        return ROOT_CAPTION
    if str(region.get("type") or "") == "decorative_text":
        return ROOT_SFX
    return ROOT_REVIEW


def _route_policy(container: dict[str, Any]) -> str:
    route = str(container.get("route_intent") or "").strip()
    if route == "translate_speech":
        return ROUTE_TRANSLATE_SPEECH
    if route == "translate_caption_background":
        return ROUTE_TRANSLATE_CAPTION
    if route == "preserve_sfx_decorative":
        return ROUTE_PRESERVE
    if route == "review_or_fallback":
        return ROUTE_FALLBACK
    return ROUTE_REVIEW


def _route_policy_for_region(region: dict[str, Any]) -> str:
    route = str(region.get("text_area_route_intent") or "").strip()
    if route == "translate_speech":
        return ROUTE_TRANSLATE_SPEECH
    if route == "translate_caption_background":
        return ROUTE_TRANSLATE_CAPTION
    if route == "preserve_sfx_decorative":
        return ROUTE_PRESERVE
    if route == "review_or_fallback":
        return ROUTE_FALLBACK
    if str(region.get("type") or "") == "speech_bubble":
        return ROUTE_TRANSLATE_SPEECH
    if str(region.get("type") or "") == "background_text":
        return ROUTE_TRANSLATE_CAPTION
    if str(region.get("type") or "") == "decorative_text":
        return ROUTE_PRESERVE
    return ROUTE_REVIEW


def _role_for_region(region: dict[str, Any]) -> str:
    route = str(region.get("text_area_route_intent") or "")
    kind = str(region.get("text_area_container_type") or region.get("type") or "")
    if route == "translate_speech" or kind == "speech_bubble":
        return ROLE_SPEECH
    if route == "translate_caption_background":
        return ROLE_CAPTION
    if kind == "background_text":
        return ROLE_BACKGROUND
    return ROLE_REVIEW


def _root_id(page_id: str, raw: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(raw or "root"))
    return f"tbr_{page_id}_{token}"


def _child_id(page_id: str, region_id: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(region_id or "region"))
    return f"cts_{page_id}_{token}"


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [int(round(float(v))) for v in value[:4]]
    except Exception:
        return []


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _ocr_confidence(region: dict[str, Any]) -> float | None:
    confidence = region.get("confidence")
    if isinstance(confidence, dict):
        return _float_or_none(confidence.get("ocr"))
    return _float_or_none(confidence)


def _append_unique(items: list[Any], value: Any) -> None:
    if value is None:
        return
    if value not in items:
        items.append(value)


def _remove_value(items: list[Any], value: Any) -> None:
    while value in items:
        items.remove(value)


def _short(value: Any, limit: int = 48) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_table(path: str, headers: list[str], rows: list[list[Any]]) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        if rows:
            for row in rows:
                handle.write("| " + " | ".join(_md(value) for value in row) + " |\n")
        else:
            handle.write("| " + " | ".join("-" for _ in headers) + " |\n")
    return path


def _write_source_glyph_parent_table(page_dir: str, hierarchy: dict[str, Any], source_glyph_masks: dict[str, Any]) -> str:
    child_by_region = {
        str(child.get("source_region_id") or ""): child
        for child in hierarchy.get("child_recognized_text_segments", []) or []
    }
    rows: list[list[Any]] = []
    for record in source_glyph_masks.get("source_glyph_masks", []) or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("region_id") or "")
        child = child_by_region.get(rid) or {}
        rows.append(
            [
                rid,
                record.get("source_glyph_mask_id"),
                record.get("source_glyph_mask_consumed_by_renderer"),
                child.get("root_id"),
                child.get("parent_id"),
                child.get("child_id"),
                record.get("generation_method"),
            ]
        )
    return _write_table(
        os.path.join(page_dir, "source_glyph_mask_parent_anchor_table.md"),
        ["region", "mask", "consumed", "root", "parent", "child", "method"],
        rows,
    )



def _write_root_source_coherence_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "root_source_coherence_table.md"),
        ["root", "type", "parents", "children", "fragmentation", "malformed", "duplicate_partial", "requires_reconstruction", "reconstruction_status", "unresolved_visible", "blocker", "status", "failure"],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                root.get("root_parent_count"),
                root.get("root_child_count"),
                root.get("root_fragmentation_score"),
                root.get("root_malformed_parent_count"),
                root.get("root_duplicate_partial_parent_count"),
                root.get("root_requires_reconstruction"),
                root.get("root_reconstruction_status"),
                root.get("root_unresolved_visible_source_count"),
                root.get("root_validation_blocker"),
                root.get("root_source_coherence_status"),
                root.get("root_source_coherence_failure_reason"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
        ],
    )


def _write_standalone_parent_rejection_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "standalone_parent_rejection_table.md"),
        ["parent", "root", "source", "translation_unit", "coherence", "action", "reason"],
        [
            [
                parent.get("parent_id"),
                parent.get("root_id"),
                _short(parent.get("source_text")),
                parent.get("translation_unit"),
                parent.get("source_coherence_status"),
                parent.get("source_coherence_action"),
                parent.get("unresolved_reason"),
            ]
            for parent in hierarchy.get("parent_logical_text_units", []) or []
            if parent.get("standalone_parent_rejected")
        ],
    )


def _write_route_owned_transaction_contract_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    root_by_id = {
        str(root.get("root_id") or ""): root
        for root in hierarchy.get("text_area_root_blocks", []) or []
        if isinstance(root, dict)
    }
    parent_by_id = {
        str(parent.get("parent_id") or ""): parent
        for parent in hierarchy.get("parent_logical_text_units", []) or []
        if isinstance(parent, dict)
    }
    rows: list[list[Any]] = []
    for child in hierarchy.get("child_recognized_text_segments", []) or []:
        if not isinstance(child, dict):
            continue
        root = root_by_id.get(str(child.get("root_id") or "")) or {}
        if str(root.get("route_policy") or "") not in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}:
            continue
        parent = parent_by_id.get(str(child.get("parent_id") or "")) or {}
        state = _ocr_transaction_state_from_reason_codes(child.get("reason_codes") or [])
        blocked = state in OCR_TRANSACTION_BLOCKER_STATES
        warning = state == OCR_STATE_RECOGNIZED_LOW_CONFIDENCE_WARNING
        represented_fragment = ROUTE_OWNED_OCR_FRAGMENT_REPRESENTED_REASON in {
            str(reason) for reason in (child.get("reason_codes") or [])
        }
        rows.append(
            [
                child.get("source_region_id"),
                child.get("root_id"),
                root.get("route_policy"),
                state,
                warning,
                blocked,
                represented_fragment,
                bool(parent.get("translation_unit")),
                bool(parent.get("render_unit")),
                bool(parent.get("cleanup_unit")),
                child.get("final_state"),
                child.get("parent_id"),
                root.get("root_source_coherence_status"),
                root.get("root_source_coherence_failure_reason"),
                root.get("root_final_state"),
                root.get("root_final_state_reason"),
                parent.get("source_coherence_status"),
                parent.get("source_coherence_action"),
                parent.get("unresolved_reason"),
                ",".join(child.get("reason_codes") or []),
            ]
        )
    return _write_table(
        os.path.join(page_dir, "route_owned_transaction_contract_table.md"),
        [
            "region",
            "root",
            "route",
            "ocr_state",
            "warning",
            "blocker",
            "represented_fragment",
            "translation_queued",
            "render_unit",
            "cleanup_unit",
            "child_state",
            "parent",
            "root_coherence",
            "root_failure",
            "root_final",
            "root_final_reason",
            "parent_coherence",
            "parent_action",
            "parent_reject_reason",
            "reasons",
        ],
        rows,
    )


def _ocr_transaction_state_from_reason_codes(reasons: list[Any]) -> str:
    for reason in reasons or []:
        text = str(reason or "")
        if text.startswith("text_area_ocr_transaction:"):
            return text.split(":", 1)[1].strip()
    return ""


def _write_malformed_source_parent_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "malformed_source_parent_table.md"),
        ["parent", "root", "source", "reocr", "translation_unit", "coherence", "reasons"],
        [
            [
                parent.get("parent_id"),
                parent.get("root_id"),
                _short(parent.get("source_text")),
                parent.get("source_reconstruction_status"),
                parent.get("translation_unit"),
                parent.get("source_coherence_status"),
                ",".join(parent.get("source_coherence_reason_codes") or []),
            ]
            for parent in hierarchy.get("parent_logical_text_units", []) or []
            if parent.get("source_coherence_action") in {"repair_required", "block_review_only"}
        ],
    )


def _write_duplicate_partial_parent_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "duplicate_partial_parent_table.md"),
        ["parent", "root", "source", "duplicate_with"],
        [
            [
                parent.get("parent_id"),
                parent.get("root_id"),
                _short(parent.get("source_text")),
                ",".join(parent.get("duplicate_partial_with_parent_ids") or []),
            ]
            for parent in hierarchy.get("parent_logical_text_units", []) or []
            if parent.get("duplicate_partial_with_parent_ids")
        ],
    )


def _write_caption_background_root_coverage_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    roots = hierarchy.get("text_area_root_blocks", []) or []
    parents = hierarchy.get("parent_logical_text_units", []) or []
    parents_by_root: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        parents_by_root.setdefault(str(parent.get("root_id") or ""), []).append(parent)
    rows: list[list[Any]] = []
    for root in roots:
        if root.get("root_type") != ROOT_CAPTION and root.get("route_policy") != ROUTE_TRANSLATE_CAPTION:
            continue
        root_parents = parents_by_root.get(str(root.get("root_id") or ""), [])
        rows.append(
            [
                root.get("root_id"),
                ",".join(root.get("text_area_container_ids") or []),
                root.get("root_child_count"),
                len(root_parents),
                " / ".join(_short(parent.get("source_text"), limit=28) for parent in root_parents),
                root.get("root_reconstruction_status"),
                root.get("root_source_coherence_status"),
                root.get("root_source_coherence_failure_reason"),
            ]
        )
    return _write_table(
        os.path.join(page_dir, "caption_background_root_coverage_table.md"),
        ["root", "containers", "children", "parents", "source", "reconstruction", "status", "failure"],
        rows,
    )


def _write_root_reconstruction_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "root_reconstruction_table.md"),
        [
            "root",
            "type",
            "required",
            "attempted",
            "status",
            "before_sources",
            "after_source",
            "rejected_attempts",
            "blocker",
        ],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                root.get("root_reconstruction_required"),
                root.get("root_reconstruction_attempted"),
                root.get("root_reconstruction_status"),
                " / ".join(_short(source, limit=28) for source in (root.get("root_reconstruction_before_sources") or [])),
                _short(root.get("root_reconstruction_after_source"), limit=36),
                len(root.get("root_reconstruction_rejected_attempts") or []),
                root.get("root_validation_blocker"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
            if root.get("root_reconstruction_required")
            or root.get("root_reconstruction_attempted")
            or root.get("root_reconstruction_status") not in {None, "", "not_attempted"}
        ],
    )


def _write_root_final_state_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "root_final_state_table.md"),
        [
            "root",
            "type",
            "route",
            "final_state",
            "reason",
            "meaningful_source",
            "punctuation_only",
            "blocker",
            "warnings",
            "cleanup_coverage",
            "residual_score",
            "render_complete",
            "render_density",
        ],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                root.get("route_policy"),
                root.get("root_final_state"),
                root.get("root_final_state_reason"),
                root.get("root_has_meaningful_visible_source"),
                root.get("root_is_punctuation_only"),
                root.get("root_closeout_blocker"),
                ",".join(root.get("root_closeout_warning_reasons") or []),
                root.get("source_erasure_mask_coverage"),
                root.get("source_erasure_visual_residual_score"),
                root.get("render_text_completeness_pass"),
                root.get("render_density_score"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
        ],
    )


def _write_page_final_state_summary(page_dir: str, hierarchy: dict[str, Any]) -> str:
    summary = hierarchy.get("text_block_hierarchy_summary") or {}
    counts = summary.get("root_final_state_counts") or hierarchy.get("root_final_state_counts") or {}
    rows = [[state, count] for state, count in sorted(counts.items())]
    rows.extend(
        [
            ["unresolved_meaningful_blocker_count", summary.get("unresolved_meaningful_blocker_count")],
            ["cleanup_warning_root_count", summary.get("cleanup_warning_root_count")],
            ["render_warning_root_count", summary.get("render_warning_root_count")],
            ["punctuation_only_nonblocking_count", summary.get("punctuation_only_nonblocking_count")],
            ["root_closeout_blocker_count", summary.get("root_closeout_blocker_count")],
            ["route_owned_translatable_count", summary.get("route_owned_translatable_count")],
            ["route_owned_translation_queued_count", summary.get("route_owned_translation_queued_count")],
            ["route_owned_ocr_warning_count", summary.get("route_owned_ocr_warning_count")],
            ["route_owned_ocr_blocker_count", summary.get("route_owned_ocr_blocker_count")],
            ["accepted_parent_without_translation_count", summary.get("accepted_parent_without_translation_count")],
            ["blocked_route_with_render_unit_count", summary.get("blocked_route_with_render_unit_count")],
            ["blocked_route_with_cleanup_unit_count", summary.get("blocked_route_with_cleanup_unit_count")],
            ["page_metadata_closeout_candidate", summary.get("page_metadata_closeout_candidate")],
            ["page_visual_evaluation_required", summary.get("page_visual_evaluation_required")],
            ["page_visual_evaluation_status", summary.get("page_visual_evaluation_status")],
            ["page_visual_evaluation_pass_source", summary.get("page_visual_evaluation_pass_source")],
            ["page_visual_closeout_blocked_reason", summary.get("page_visual_closeout_blocked_reason")],
            ["page_visual_closeout_pass", summary.get("page_visual_closeout_pass")],
        ]
    )
    return _write_table(
        os.path.join(page_dir, "page_final_state_summary.md"),
        ["metric", "value"],
        rows,
    )


def _write_visual_evaluation_gate(page_dir: str, hierarchy: dict[str, Any]) -> str:
    summary = hierarchy.get("text_block_hierarchy_summary") or {}
    rows = [
        ["page_visual_closeout_pass", summary.get("page_visual_closeout_pass")],
        ["page_metadata_closeout_candidate", summary.get("page_metadata_closeout_candidate")],
        ["page_visual_evaluation_required", summary.get("page_visual_evaluation_required")],
        ["page_visual_evaluation_status", summary.get("page_visual_evaluation_status")],
        ["page_visual_evaluation_pass_source", summary.get("page_visual_evaluation_pass_source")],
        ["page_visual_closeout_blocked_reason", summary.get("page_visual_closeout_blocked_reason")],
        ["required_view_1", "original page"],
        ["required_view_2", "translated output page"],
        ["required_view_3", "overlay/debug page"],
        ["required_view_4", "source glyph mask overlay when available"],
        ["required_view_5", "region audit JSON and hierarchy tables"],
        ["blocking_rule_1", "visible untranslated normal speech/caption/background text"],
        ["blocking_rule_2", "source text left under a translated root"],
        ["blocking_rule_3", "over-merged root or parent causing chaotic layout"],
        ["blocking_rule_4", "unreadable dense/tiny/overlapping translated text"],
        ["blocking_rule_5", "translated text outside the intended root/container"],
        ["blocking_rule_6", "automated reviewer clean result without direct visual ledger"],
    ]
    return _write_table(
        os.path.join(page_dir, "visual_evaluation_gate.md"),
        ["metric", "value"],
        rows,
    )


def _write_unresolved_meaningful_blocker_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "unresolved_meaningful_blocker_table.md"),
        ["root", "type", "route", "reason", "parents", "children", "transaction", "coherence"],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                root.get("route_policy"),
                root.get("root_final_state_reason"),
                root.get("root_parent_count"),
                root.get("root_child_count"),
                root.get("root_transaction_status"),
                root.get("root_source_coherence_status"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
            if root.get("root_final_state") == ROOT_FINAL_UNRESOLVED_MEANINGFUL_BLOCKER
            or root.get("root_closeout_blocker")
        ],
    )


def _write_punctuation_only_nonblocking_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "punctuation_only_nonblocking_table.md"),
        ["root", "type", "route", "reason", "children", "parents"],
        [
            [
                root.get("root_id"),
                root.get("root_type"),
                root.get("route_policy"),
                root.get("root_final_state_reason"),
                root.get("root_child_count"),
                root.get("root_parent_count"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
            if root.get("root_final_state") == ROOT_FINAL_PUNCTUATION_ONLY_NONBLOCKING
        ],
    )


def _write_caption_background_recovery_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    roots = hierarchy.get("text_area_root_blocks", []) or []
    parents = hierarchy.get("parent_logical_text_units", []) or []
    parents_by_root: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        parents_by_root.setdefault(str(parent.get("root_id") or ""), []).append(parent)
    rows: list[list[Any]] = []
    for root in roots:
        if root.get("root_type") != ROOT_CAPTION and root.get("route_policy") != ROUTE_TRANSLATE_CAPTION:
            continue
        root_parents = parents_by_root.get(str(root.get("root_id") or ""), [])
        rows.append(
            [
                root.get("root_id"),
                ",".join(root.get("text_area_container_ids") or []),
                root.get("root_final_state"),
                root.get("root_transaction_status"),
                root.get("root_reconstruction_status"),
                root.get("caption_background_recovery_v3_status"),
                root.get("caption_background_recovery_v4_status"),
                root.get("caption_background_sfx_rejection_status"),
                len(root_parents),
                " / ".join(_short(parent.get("source_text"), 32) for parent in root_parents),
                root.get("root_final_state_reason"),
            ]
        )
    return _write_table(
        os.path.join(page_dir, "caption_background_recovery_table.md"),
        ["root", "containers", "final_state", "transaction", "reconstruction", "recovery_v3", "recovery_v4", "sfx_rejection", "parents", "source", "reason"],
        rows,
    )


def _write_caption_background_failed_reason_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "caption_background_failed_reason_table.md"),
        ["root", "containers", "final_state", "meaningful_source", "blocker", "recovery_v3", "recovery_v4", "sfx_rejection", "sfx_reason", "reason", "coherence_failure"],
        [
            [
                root.get("root_id"),
                ",".join(root.get("text_area_container_ids") or []),
                root.get("root_final_state"),
                root.get("root_has_meaningful_visible_source"),
                root.get("root_closeout_blocker"),
                root.get("caption_background_recovery_v3_status"),
                root.get("caption_background_recovery_v4_status"),
                root.get("caption_background_sfx_rejection_status"),
                root.get("caption_background_sfx_rejection_reason"),
                root.get("root_final_state_reason"),
                root.get("root_source_coherence_failure_reason"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
            if (root.get("root_type") == ROOT_CAPTION or root.get("route_policy") == ROUTE_TRANSLATE_CAPTION)
            and root.get("root_final_state") != ROOT_FINAL_TRANSLATED_CLEAN
        ],
    )


def _write_caption_sfx_rejection_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    return _write_table(
        os.path.join(page_dir, "caption_sfx_rejection_table.md"),
        [
            "root",
            "containers",
            "final_state",
            "visibility",
            "sfx_status",
            "sfx_score",
            "sfx_reason",
            "rejected_ocr",
            "meaningful_rejected_ocr",
        ],
        [
            [
                root.get("root_id"),
                ",".join(root.get("text_area_container_ids") or []),
                root.get("root_final_state"),
                root.get("caption_background_visibility_status"),
                root.get("caption_background_sfx_rejection_status"),
                root.get("caption_background_sfx_artlike_score"),
                root.get("caption_background_sfx_rejection_reason"),
                " / ".join(_short(text, 24) for text in (root.get("caption_background_rejected_ocr_texts") or [])),
                " / ".join(_short(text, 24) for text in (root.get("caption_background_meaningful_rejected_ocr_texts") or [])),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
            if (root.get("root_type") == ROOT_CAPTION or root.get("route_policy") == ROUTE_TRANSLATE_CAPTION)
            and (
                root.get("caption_background_sfx_rejection_status")
                or root.get("caption_background_rejected_ocr_texts")
            )
        ],
    )


def _write_caption_component_split_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    records = [
        record for record in (hierarchy.get("caption_component_recovery_records") or [])
        if isinstance(record, dict)
    ]
    return _write_table(
        os.path.join(page_dir, "caption_component_split_table.md"),
        ["root", "container", "component", "bbox", "role", "status", "v4_id", "v4_axis", "v4_score", "ocr_text", "ocr_conf", "reason"],
        [
            [
                record.get("parent_root_id"),
                record.get("text_area_container_id"),
                record.get("component_id") or record.get("caption_component_id"),
                record.get("component_bbox") or record.get("bbox"),
                record.get("component_role") or record.get("caption_component_role"),
                record.get("status"),
                record.get("caption_component_v4_candidate_id"),
                record.get("caption_component_v4_axis"),
                record.get("caption_component_v4_score"),
                _short(record.get("ocr_text"), 32),
                record.get("ocr_confidence"),
                record.get("rejection_reason") or record.get("component_reason") or record.get("reason"),
            ]
            for record in records
        ],
    )



def _write_source_erasure_validation_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    records = hierarchy.get("source_erasure_validation_records") or []
    if records:
        rows = [
            [
                record.get("root_id"),
                record.get("source_erasure_expected"),
                record.get("source_erasure_mask_coverage"),
                record.get("source_erasure_visual_residual_score"),
                record.get("source_erasure_warning_class_counts"),
                record.get("source_erasure_failure_reason"),
                record.get("source_glyph_mask_id"),
                record.get("cleanup_partition_id"),
            ]
            for record in records
        ]
    else:
        rows = [
            [
                root.get("root_id"),
                root.get("source_erasure_expected"),
                root.get("source_erasure_mask_coverage"),
                root.get("source_erasure_visual_residual_score"),
                root.get("source_erasure_warning_class_counts"),
                root.get("source_erasure_failure_reason"),
                root.get("source_glyph_mask_id"),
                root.get("cleanup_partition_id"),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
        ]
    return _write_table(
        os.path.join(page_dir, "source_erasure_validation_table.md"),
        ["root", "expected", "mask_coverage", "visual_residual_score", "warning_class_counts", "failure_reason", "source_glyph_mask_id", "cleanup_partition_id"],
        rows,
    )


def _write_render_readability_validation_table(page_dir: str, hierarchy: dict[str, Any]) -> str:
    records = hierarchy.get("render_readability_validation_records") or []
    if records:
        rows = [
            [
                record.get("root_id"),
                record.get("render_text_completeness_pass"),
                record.get("render_missing_characters"),
                record.get("render_outside_root_ratio"),
                record.get("render_density_score"),
                record.get("render_readability_warning_reason"),
                record.get("render_warning_action_status"),
                record.get("render_warning_action_reason"),
                record.get("render_fit_action_status"),
                record.get("render_fit_before_density"),
                record.get("render_fit_after_density"),
                record.get("render_fit_rejection_reason"),
                record.get("render_layout_v2_status"),
                record.get("render_layout_v2_before_score"),
                record.get("render_layout_v2_after_score"),
                record.get("render_layout_v2_rejection_reason"),
                record.get("render_layout_v3_status"),
                record.get("render_layout_v3_before_score"),
                record.get("render_layout_v3_after_score"),
                record.get("render_layout_v3_candidate_count"),
                record.get("render_layout_v3_rejection_reason"),
                record.get("render_readability_v4_status"),
                record.get("render_readability_v4_before_score"),
                record.get("render_readability_v4_after_score"),
                record.get("render_readability_v4_candidate_count"),
                record.get("render_readability_v4_final_class"),
                record.get("render_readability_v4_unresolved_reason"),
                record.get("render_readability_v5_status"),
                record.get("render_readability_v5_before_score"),
                record.get("render_readability_v5_after_score"),
                record.get("render_readability_v5_candidate_count"),
                record.get("render_readability_v5_source_column_count"),
                record.get("render_readability_v5_final_class"),
                record.get("render_readability_v5_unresolved_reason"),
                " / ".join(_short(line, 24) for line in (record.get("render_wrapped_lines") or [])),
            ]
            for record in records
        ]
    else:
        rows = [
            [
                root.get("root_id"),
                root.get("render_text_completeness_pass"),
                root.get("render_missing_characters"),
                root.get("render_outside_root_ratio"),
                root.get("render_density_score"),
                root.get("render_readability_warning_reason"),
                root.get("render_warning_action_status"),
                root.get("render_warning_action_reason"),
                root.get("render_fit_action_status"),
                root.get("render_fit_before_density"),
                root.get("render_fit_after_density"),
                root.get("render_fit_rejection_reason"),
                root.get("render_layout_v2_status"),
                root.get("render_layout_v2_before_score"),
                root.get("render_layout_v2_after_score"),
                root.get("render_layout_v2_rejection_reason"),
                root.get("render_layout_v3_status"),
                root.get("render_layout_v3_before_score"),
                root.get("render_layout_v3_after_score"),
                root.get("render_layout_v3_candidate_count"),
                root.get("render_layout_v3_rejection_reason"),
                root.get("render_readability_v4_status"),
                root.get("render_readability_v4_before_score"),
                root.get("render_readability_v4_after_score"),
                root.get("render_readability_v4_candidate_count"),
                root.get("render_readability_v4_final_class"),
                root.get("render_readability_v4_unresolved_reason"),
                root.get("render_readability_v5_status"),
                root.get("render_readability_v5_before_score"),
                root.get("render_readability_v5_after_score"),
                root.get("render_readability_v5_candidate_count"),
                root.get("render_readability_v5_source_column_count"),
                root.get("render_readability_v5_final_class"),
                root.get("render_readability_v5_unresolved_reason"),
                " / ".join(_short(line, 24) for line in (root.get("render_wrapped_lines") or [])),
            ]
            for root in hierarchy.get("text_area_root_blocks", []) or []
        ]
    return _write_table(
        os.path.join(page_dir, "render_readability_validation_table.md"),
        [
            "root",
            "text_complete",
            "missing",
            "outside_ratio",
            "density",
            "warning_reason",
            "action_status",
            "action_reason",
            "fit_action_status",
            "fit_before_density",
            "fit_after_density",
            "fit_rejection_reason",
            "layout_v2_status",
            "layout_v2_before_score",
            "layout_v2_after_score",
            "layout_v2_rejection_reason",
            "layout_v3_status",
            "layout_v3_before_score",
            "layout_v3_after_score",
            "layout_v3_candidate_count",
            "layout_v3_rejection_reason",
            "readability_v4_status",
            "readability_v4_before_score",
            "readability_v4_after_score",
            "readability_v4_candidate_count",
            "readability_v4_final_class",
            "readability_v4_unresolved_reason",
            "readability_v5_status",
            "readability_v5_before_score",
            "readability_v5_after_score",
            "readability_v5_candidate_count",
            "readability_v5_source_columns",
            "readability_v5_final_class",
            "readability_v5_unresolved_reason",
            "wrapped_lines",
        ],
        rows,
    )


def _md(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("|", "\\|").replace("\n", " ")
