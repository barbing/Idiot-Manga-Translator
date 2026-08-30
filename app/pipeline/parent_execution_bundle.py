# -*- coding: utf-8 -*-
"""Parent-keyed execution contract for post-hierarchy pipeline stages.

This module does not build root/parent topology. It converts the finalized
hierarchy view into the downstream execution unit consumed by translation,
cleanup, render eligibility, and rendering.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.pipeline.target_presentation import (
    TargetPresentationPolicy,
    target_presentation_policy as target_presentation_policy_for_language,
)

PARENT_EXECUTION_BUNDLE_VERSION = "parent_execution_bundle_v2"
PARENT_RENDER_STYLE_VERSION = "parent_render_style_v4"
PARENT_STYLE_ARBITRATOR_SOURCE = "parent_authorized_style_evidence"
PARENT_STYLE_ARBITRATOR_PROVIDER = "ParentStyleArbitrator"
PARENT_STYLE_RESOLUTION_STATUSES = {"complete"}
PARENT_STYLE_DEFAULT_FALLBACK_FONT_CHAIN_KEY = "cjk-sc"
PARENT_STYLE_UNRESOLVED_FONT_SIZE = 24
PARENT_STYLE_UNRESOLVED_FONT_SIZE_MIN = 17
PARENT_STYLE_UNRESOLVED_FONT_SIZE_MAX = 24
PARENT_STYLE_UNRESOLVED_FONT_SIZE_AUTHORITY = (
    "parent_style_arbitrator_unresolved_scale_fallback"
)
PARENT_STYLE_UNRESOLVED_FONT_SIZE_POLICY = (
    "arbitrator_owned_unresolved_scale_fallback"
)
SOURCE_SIDE_ANCHOR_POLICY_VERSION = "source_side_anchor_v1"
SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_RATIO = 0.02
SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_MIN_PX = 4.0

_PARENT_STYLE_WRITING_MODES = {"vertical", "horizontal"}
_PARENT_STYLE_ALIGNMENTS = {"center", "left", "right", "start", "end"}
_PARENT_STYLE_FAMILY_ROLES = {"sans", "serif"}
_PARENT_STYLE_WEIGHT_TIERS = {"slender", "base", "emphasis", "heavy"}
_PARENT_STYLE_FONT_ROLE_MATRIX = {
    ("sans", "slender"): ("sans_regular", "registered_role"),
    ("sans", "base"): ("sans_medium", "registered_role"),
    ("sans", "emphasis"): ("sans_bold", "registered_role"),
    ("sans", "heavy"): ("sans_black", "registered_role"),
    ("serif", "slender"): ("serif_regular", "registered_role"),
    ("serif", "base"): ("serif_semibold", "registered_role"),
    ("serif", "emphasis"): ("serif_bold", "registered_role"),
    ("serif", "heavy"): ("serif_bold", "degraded_registered_role"),
}
_PARENT_STYLE_ROLE_STATUSES = {
    "registered_role",
    "degraded_registered_role",
    "fallback_registered_role",
}
_PARENT_STYLE_SOURCE_CELL_STATUSES = {
    "direct",
    "peer",
    "fallback",
    "unavailable",
}
_PARENT_STYLE_AXIS_NAMES = (
    "family",
    "weight",
    "source_scale",
    "fill",
    "outline",
    "orientation",
    "rotation",
    "shadow",
)
_PARENT_STYLE_AXIS_STATUSES = {"direct", "peer", "fallback", "unavailable"}
_HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

_LEGACY_RENDER_STYLE_FLAT_FIELDS = {
    "font_family": "font",
    "font_size": "font_size",
    "font_size_hint": "source_size_hint",
    "font_size_min": "source_size_min",
    "font_size_max": "source_size_max",
    "font_size_locked": "font_size_locked",
    "font_size_policy": "font_size_policy",
    "font_size_fallback_policy": "font_size_fallback_policy",
    "source_orientation": "source_orientation",
    "wrap_mode": "wrap_mode",
    "line_height": "line_height",
    "align": "align",
    "fill_color": "color",
    "stroke_color": "stroke",
    "stroke_width": "stroke_width",
    "style_class": "font_style",
    "font_weight": "font_weight",
    "spacing_profile": "spacing_profile",
}
_PARENT_STYLE_REQUIRED_FIELDS = {
    "render_style_version",
    "render_style_owner",
    "render_style_source",
    "render_style_provider",
    "style_resolution_status",
    "source_evidence_status",
    "render_style_confidence",
    "font_family_role",
    "font_weight_tier",
    "primary_font_role",
    "primary_font_role_status",
    "fallback_font_chain_key",
    "target_presentation_policy",
    "target_language",
    "target_script",
    "shaping_locale",
    "source_visual_cell",
    "source_writing_mode",
    "target_optical_reference_em_px",
    "target_fit_start_em_px",
    "target_preferred_em_px",
    "target_preferred_em_interval_px",
    "target_face_profile_id",
    "target_em_conversion_audit",
    "target_size_preference",
    "fill",
    "outline",
    "writing_mode",
    "line_height",
    "align",
    "axis_authority",
    "fallback_status",
}
_PARENT_STYLE_OPTIONAL_FIELDS = {
    "parent_layer_effects",
    "diagnostic_uncertainty",
    "readability_diagnostic",
}
_PARENT_STYLE_FORBIDDEN_LEGACY_FIELDS = (
    set(_LEGACY_RENDER_STYLE_FLAT_FIELDS) - {"line_height", "align"}
) | {
    "font_weight",
    "font_size_authority",
    "font_size_source",
    "target_font_request",
    "target_font_mapping_source",
    "target_font_mapping_family_role",
    "target_font_mapping_weight",
    "source_scale_px",
    "style_axis_decisions",
    "typographic_style_class",
    "base_style_id",
}


@dataclass(frozen=True)
class ResolvedRenderStyleValidation:
    """Central acceptance result for an executable parent render style."""

    status: str
    style: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and bool(self.style)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class ParentExecutionBundle:
    page_id: str
    bundle_id: str
    root_id: str
    parent_id: str
    graph_parent_id: str
    state: str
    role: str
    source_text: str = ""
    source_quality_state: str = "accepted_for_translation"
    source_quality_action: str = "translate"
    source_contract_owner: str = ""
    source_contract_region_id: str = ""
    source_contract_bbox: list[int] = field(default_factory=list)
    source_contract_scope: str = ""
    source_contract_stage: str = ""
    source_contract_ocr_confidence: float | None = None
    ocr_backend: str = ""
    ocr_model_path: str = ""
    ocr_mmproj_path: str = ""
    ocr_endpoint: str = ""
    ocr_prompt_version: str = ""
    source_quality_reason_codes: list[str] = field(default_factory=list)
    translation_required: bool = False
    cleanup_required: bool = False
    render_required: bool = False
    parent_bbox: list[int] = field(default_factory=list)
    cleanup_target_bbox: list[int] = field(default_factory=list)
    render_allowed_area: list[int] = field(default_factory=list)
    root_bbox: list[int] = field(default_factory=list)
    source_region_ids: list[str] = field(default_factory=list)
    represented_child_ids: list[str] = field(default_factory=list)
    source_candidates: list[dict[str, Any]] = field(default_factory=list)
    semantic_class: str = ""
    route_intent: str = ""
    cleanup_mode: str = ""
    text_area_container_id: str = ""
    text_area_container_type: str = ""
    text_area_container_bbox: list[int] = field(default_factory=list)
    text_area_container_polygon: list[list[float]] = field(default_factory=list)
    text_area_oriented_frame: dict[str, Any] = field(default_factory=dict)
    target_language: str = ""
    target_presentation_policy: dict[str, Any] = field(default_factory=dict)
    render_layout_domain: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    unresolved_reason: str | None = None
    translated_text: str = ""
    source_glyph_mask_ids: list[str] = field(default_factory=list)
    cleanup_job_ids: list[str] = field(default_factory=list)
    cleanup_mask_ids: list[str] = field(default_factory=list)
    render_decision_id: str = ""
    renderer_audit_id: str = ""
    style_evidence_summary: dict[str, Any] = field(default_factory=dict)
    source_punctuation_geometry: dict[str, Any] = field(default_factory=dict)
    render_layout_summary: dict[str, Any] = field(default_factory=dict)
    render_style: dict[str, Any] = field(default_factory=dict)
    execution_region: dict[str, Any] = field(default_factory=dict)
    reading_order_index: int = 0

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "root_id": self.root_id,
            "parent_id": self.parent_id,
            "graph_parent_id": self.graph_parent_id,
            "state": self.state,
            "role": self.role,
            "source_text": self.source_text,
            "source_quality_state": self.source_quality_state,
            "source_quality_action": self.source_quality_action,
            "source_contract_owner": self.source_contract_owner,
            "source_contract_region_id": self.source_contract_region_id,
            "source_contract_bbox": list(self.source_contract_bbox),
            "source_contract_scope": self.source_contract_scope,
            "source_contract_stage": self.source_contract_stage,
            "source_contract_ocr_confidence": self.source_contract_ocr_confidence,
            "ocr_backend": self.ocr_backend,
            "ocr_model_path": self.ocr_model_path,
            "ocr_mmproj_path": self.ocr_mmproj_path,
            "ocr_endpoint": self.ocr_endpoint,
            "ocr_prompt_version": self.ocr_prompt_version,
            "source_quality_reason_codes": list(self.source_quality_reason_codes),
            "translation_required": self.translation_required,
            "cleanup_required": self.cleanup_required,
            "render_required": self.render_required,
            "parent_bbox": list(self.parent_bbox),
            "cleanup_target_bbox": list(self.cleanup_target_bbox),
            "render_allowed_area": list(self.render_allowed_area),
            "root_bbox": list(self.root_bbox),
            "source_region_ids": list(self.source_region_ids),
            "represented_child_ids": list(self.represented_child_ids),
            "source_candidates": [dict(item) for item in self.source_candidates],
            "semantic_class": self.semantic_class,
            "route_intent": self.route_intent,
            "cleanup_mode": self.cleanup_mode,
            "text_area_container_id": self.text_area_container_id,
            "text_area_container_type": self.text_area_container_type,
            "text_area_container_bbox": list(self.text_area_container_bbox),
            "text_area_container_polygon": _copy_jsonish(
                self.text_area_container_polygon
            ),
            "text_area_oriented_frame": _copy_jsonish(
                self.text_area_oriented_frame
            ),
            "target_language": self.target_language,
            "target_presentation_policy": _copy_jsonish(
                self.target_presentation_policy
            ),
            "render_layout_domain": _copy_jsonish(self.render_layout_domain),
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "unresolved_reason": self.unresolved_reason,
            "translated_text": self.translated_text,
            "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
            "cleanup_job_ids": list(self.cleanup_job_ids),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "render_decision_id": self.render_decision_id,
            "renderer_audit_id": self.renderer_audit_id,
            "style_evidence_summary": _copy_jsonish(self.style_evidence_summary),
            "source_punctuation_geometry": _copy_jsonish(
                self.source_punctuation_geometry
            ),
            "render_layout_summary": _copy_jsonish(self.render_layout_summary),
            "render_style": _copy_jsonish(
                resolved_render_style_contract(self.render_style)
            ),
            "execution_region": self.to_region_record(),
            "reading_order_index": int(self.reading_order_index),
        }

    def to_region_record(self) -> dict[str, Any]:
        """Return this bundle's parent-owned execution region.

        The record identity is the finalized parent id. Represented child/source
        regions remain evidence only and must not become separate execution
        units after this handoff.
        """

        if self.execution_region:
            record = _copy_region_record(self.execution_region)
            _sync_execution_region_from_bundle(self, record)
            self.execution_region = _copy_region_record(record)
            return _copy_region_record(record)

        bbox = _best_bbox(self.parent_bbox, self.render_allowed_area, self.cleanup_target_bbox, self.root_bbox)
        render_allowed = _best_bbox(self.render_allowed_area, bbox)
        cleanup_target = _best_bbox(self.cleanup_target_bbox, bbox)
        root_bbox = _best_bbox(self.root_bbox, render_allowed)
        semantic_class = self.semantic_class or _semantic_class_for_role(self.role)
        route_intent = self.route_intent or _route_intent_for_role(self.role)
        cleanup_mode = self.cleanup_mode or _cleanup_mode_for_role(self.role)
        container_type = self.text_area_container_type or _container_type_for_role(self.role)
        semantic_kind = _semantic_kind_for_role(self.role)
        cleanup_authorization = _cleanup_authorization_for_role(self.role)
        render_style = resolved_render_style_contract(self.render_style)
        record = {
            "region_id": self.bundle_id,
            "page_id": self.page_id,
            "type": semantic_class,
            "semantic_class": semantic_class,
            "semantic_kind": semantic_kind,
            **_render_style_record_fields(render_style),
            "cleanup_authorization": cleanup_authorization,
            "text_area_cleanup_authorization": cleanup_authorization,
            "semantic_authorization_state": cleanup_authorization,
            "text_area_semantic_authorization_state": cleanup_authorization,
            "authorization_explicit": True,
            "text_area_authorization_explicit": True,
            "authorization_field_origin": "parent_execution_bundle",
            "text_area_authorization_field_origin": "parent_execution_bundle",
            "authorization_basis": "finalized_parent_execution_bundle",
            "source_stage": "parent_execution_bundle",
            "target_language": self.target_language,
            "target_presentation_policy": _copy_jsonish(
                self.target_presentation_policy
            ),
            "render_layout_domain": _copy_jsonish(self.render_layout_domain),
            "execution_region_authority": "parent_execution_bundle",
            "execution_region_role": "parent_execution",
            "legacy_region_execution_authority": False,
            "source_region_evidence_only": True,
            "text_area_authorization_source_stage": "parent_execution_bundle",
            "route_intent": route_intent,
            "text_area_route_intent": route_intent,
            "container_type": container_type,
            "text_area_container_type": container_type,
            "text_area_container_id": self.text_area_container_id,
            "text_area_container_bbox": list(self.text_area_container_bbox),
            "text_area_container_polygon": _copy_jsonish(
                self.text_area_container_polygon
            ),
            "text_area_oriented_frame": _copy_jsonish(
                self.text_area_oriented_frame
            ),
            "cleanup_mode": cleanup_mode,
            "ocr_text": self.source_text,
            "source_text": self.source_text,
            "translation": self.translated_text,
            "translated_text": self.translated_text,
            "translation_required": bool(self.translation_required),
            "cleanup_required": bool(self.cleanup_required),
            "render_required": bool(self.render_required),
            "bbox": list(bbox),
            "polygon": _polygon_from_bbox(bbox),
            "order_index": int(self.reading_order_index),
            "reading_order_index": int(self.reading_order_index),
            "flags": {
                "ignore": not self.translation_required and self.state != "punctuation_identity_parent",
                "bg_text": self.role in {"caption", "background", "caption_background", "background_narration"},
                "needs_review": bool(self.unresolved_reason),
            },
            "parent_execution_bundle_id": self.bundle_id,
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "parent_execution_state": self.state,
            "parent_execution_authoritative": True,
            "text_block_root_id": self.root_id,
            "parent_logical_text_unit_id": self.parent_id,
            "active_translation_unit_id": self.parent_id if self.translation_required else "",
            "logical_text_block_id": self.parent_id,
            "logical_text_ownership_status": "parent_execution_bundle",
            "logical_text_block_source_text": self.source_text,
            "parent_logical_text_unit_source_text": self.source_text,
            "logical_text_block_bbox": list(cleanup_target),
            "parent_logical_text_unit_cleanup_target_bbox": list(cleanup_target),
            "parent_logical_text_unit_render_allowed_area": list(render_allowed),
            "logical_text_block_allowed_bbox": list(render_allowed),
            "logical_text_block_member_region_ids": list(self.source_region_ids),
            "logical_text_block_transferred_region_ids": list(self.source_region_ids),
            "logical_text_block_translation_unit": bool(self.translation_required),
            "child_final_state": "parent_anchor",
            "represented_child_ids": list(self.represented_child_ids),
            "source_region_ids": list(self.source_region_ids),
            "parent_source_coherence_action": self.source_quality_action,
            "logical_text_source_quality_action": self.source_quality_action,
            "source_conservation_status": self.source_quality_state,
            "source_contract_owner": self.source_contract_owner,
            "source_contract_region_id": self.source_contract_region_id,
            "source_contract_bbox": list(self.source_contract_bbox),
            "source_contract_scope": self.source_contract_scope,
            "source_contract_stage": self.source_contract_stage,
            "source_contract_ocr_confidence": self.source_contract_ocr_confidence,
            "ocr_backend": self.ocr_backend,
            "ocr_model_path": self.ocr_model_path,
            "ocr_mmproj_path": self.ocr_mmproj_path,
            "ocr_endpoint": self.ocr_endpoint,
            "ocr_prompt_version": self.ocr_prompt_version,
            "source_quality_reason_codes": list(self.source_quality_reason_codes),
            "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
            "cleanup_job_ids": list(self.cleanup_job_ids),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "render_decision_id": self.render_decision_id,
            "renderer_audit_id": self.renderer_audit_id,
            "source_punctuation_geometry": _copy_jsonish(
                self.source_punctuation_geometry
            ),
            "render_layout_summary": _copy_jsonish(self.render_layout_summary),
            "render": {
                "parent_execution_bundle_id": self.bundle_id,
                "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
                **_render_style_record_fields(render_style),
                **_render_style_flattened_fields(render_style),
                "text_block_root_id": self.root_id,
                "parent_logical_text_unit_id": self.parent_id,
                "active_translation_unit_id": self.parent_id if self.translation_required else "",
                "logical_text_block_source_text": self.source_text,
                "parent_logical_text_unit_source_text": self.source_text,
                "source_text": self.source_text,
                "translation": self.translated_text,
                "translated_text": self.translated_text,
                "translation_required": bool(self.translation_required),
                "cleanup_required": bool(self.cleanup_required),
                "render_required": bool(self.render_required),
                "child_final_state": "parent_anchor",
                "cleanup_mode": cleanup_mode,
                "semantic_class": semantic_class,
                "semantic_kind": semantic_kind,
                "cleanup_authorization": cleanup_authorization,
                "text_area_cleanup_authorization": cleanup_authorization,
                "semantic_authorization_state": cleanup_authorization,
                "text_area_semantic_authorization_state": cleanup_authorization,
                "authorization_explicit": True,
                "text_area_authorization_explicit": True,
                "authorization_field_origin": "parent_execution_bundle",
                "text_area_authorization_field_origin": "parent_execution_bundle",
                "authorization_basis": "finalized_parent_execution_bundle",
                "source_stage": "parent_execution_bundle",
                "execution_region_authority": "parent_execution_bundle",
                "execution_region_role": "parent_execution",
                "legacy_region_execution_authority": False,
                "source_region_evidence_only": True,
                "parent_execution_authoritative": True,
                "text_area_authorization_source_stage": "parent_execution_bundle",
                "text_area_route_intent": route_intent,
                "route_intent": route_intent,
                "container_type": container_type,
                "text_area_container_type": container_type,
                "text_area_container_id": self.text_area_container_id,
                "text_area_container_bbox": list(
                    self.text_area_container_bbox
                ),
                "text_area_container_polygon": _copy_jsonish(
                    self.text_area_container_polygon
                ),
                "text_area_oriented_frame": _copy_jsonish(
                    self.text_area_oriented_frame
                ),
                "target_language": self.target_language,
                "target_presentation_policy": _copy_jsonish(
                    self.target_presentation_policy
                ),
                "render_layout_domain": _copy_jsonish(
                    self.render_layout_domain
                ),
                "cleanup_allowed_area": list(render_allowed),
                "allowed_cleanup_area": list(render_allowed),
                "render_allowed_area": list(render_allowed),
                "logical_text_block_bbox": list(cleanup_target),
                "parent_logical_text_unit_cleanup_target_bbox": list(cleanup_target),
                "parent_logical_text_unit_render_allowed_area": list(render_allowed),
                "source_region_ids": list(self.source_region_ids),
                "represented_child_ids": list(self.represented_child_ids),
                "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
                "cleanup_job_ids": list(self.cleanup_job_ids),
                "cleanup_mask_ids": list(self.cleanup_mask_ids),
                "render_decision_id": self.render_decision_id,
                "renderer_audit_id": self.renderer_audit_id,
                "source_punctuation_geometry": _copy_jsonish(
                    self.source_punctuation_geometry
                ),
                "render_layout_summary": _copy_jsonish(self.render_layout_summary),
                "order_index": int(self.reading_order_index),
                "reading_order_index": int(self.reading_order_index),
                "parent_source_coherence_action": self.source_quality_action,
                "logical_text_source_quality_action": self.source_quality_action,
                "source_contract_owner": self.source_contract_owner,
                "source_contract_region_id": self.source_contract_region_id,
                "source_contract_bbox": list(self.source_contract_bbox),
                "source_contract_scope": self.source_contract_scope,
                "source_contract_stage": self.source_contract_stage,
                "source_contract_ocr_confidence": self.source_contract_ocr_confidence,
                "ocr_backend": self.ocr_backend,
                "ocr_model_path": self.ocr_model_path,
                "ocr_mmproj_path": self.ocr_mmproj_path,
                "ocr_endpoint": self.ocr_endpoint,
                "ocr_prompt_version": self.ocr_prompt_version,
                "source_quality_reason_codes": list(self.source_quality_reason_codes),
            },
        }
        _sync_execution_region_from_bundle(self, record)
        self.execution_region = _copy_region_record(record)
        return _copy_region_record(record)


@dataclass
class ParentExecutionBundleResult:
    page_id: str
    bundles: list[ParentExecutionBundle] = field(default_factory=list)
    blocked_bundles: list[ParentExecutionBundle] = field(default_factory=list)
    excluded_nonworkflow_children: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def executable_bundles(self) -> list[ParentExecutionBundle]:
        return [
            bundle for bundle in self.bundles
            if bundle.state in {"active_translation_parent", "punctuation_identity_parent"}
        ]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "page_id": self.page_id,
            "bundle_count": len(self.bundles),
            "blocked_bundle_count": len(self.blocked_bundles),
            "executable_bundle_ids": [bundle.bundle_id for bundle in self.executable_bundles()],
            "bundles": [bundle.to_audit_dict() for bundle in self.bundles],
            "blocked_bundles": [bundle.to_audit_dict() for bundle in self.blocked_bundles],
            "excluded_nonworkflow_children": list(self.excluded_nonworkflow_children),
            "errors": list(self.errors),
        }


def build_parent_execution_bundles(
    *,
    page_id: str,
    hierarchy_result: Any,
    regions: Sequence[Mapping[str, Any]],
    target_presentation_policy: TargetPresentationPolicy | None = None,
) -> ParentExecutionBundleResult:
    presentation_policy = (
        target_presentation_policy
        if isinstance(target_presentation_policy, TargetPresentationPolicy)
        else target_presentation_policy_for_language("Simplified Chinese")
    )
    finalized = hierarchy_result.finalized_execution_units()
    root_by_id = {
        str(getattr(root, "root_id", "") or ""): root
        for root in getattr(hierarchy_result, "roots", []) or []
        if str(getattr(root, "root_id", "") or "")
    }
    parent_by_id = {
        str(getattr(parent, "parent_id", "") or ""): parent
        for parent in getattr(hierarchy_result, "parent_units", []) or []
        if str(getattr(parent, "parent_id", "") or "")
    }
    region_by_id = {
        str(region.get("region_id") or ""): dict(region)
        for region in regions or []
        if isinstance(region, Mapping) and str(region.get("region_id") or "")
    }

    result = ParentExecutionBundleResult(
        page_id=str(page_id or getattr(finalized, "page_id", "") or ""),
        excluded_nonworkflow_children=[
            child.to_dict() if hasattr(child, "to_dict") else dict(child)
            for child in getattr(finalized, "excluded_nonworkflow_children", []) or []
        ],
    )
    seen_parent_ids: set[str] = set()
    finalized_parents = (
        list(getattr(finalized, "active_translation_parents", []) or [])
        + list(getattr(finalized, "punctuation_parent_obligations", []) or [])
    )
    for parent in finalized_parents:
        parent_id = str(getattr(parent, "parent_id", "") or "")
        if not parent_id:
            result.errors.append("finalized_parent_missing_parent_id")
            continue
        if parent_id in seen_parent_ids:
            result.errors.append(f"duplicate_finalized_parent_id:{parent_id}")
            continue
        seen_parent_ids.add(parent_id)
        bundle = _bundle_from_finalized_parent(
            page_id=result.page_id,
            parent=parent,
            root_by_id=root_by_id,
            parent_by_id=parent_by_id,
            region_by_id=region_by_id,
            target_presentation_policy=presentation_policy,
        )
        result.bundles.append(bundle)

    for parent in getattr(finalized, "blocked_or_unresolved_parents", []) or []:
        bundle = _bundle_from_finalized_parent(
            page_id=result.page_id,
            parent=parent,
            root_by_id=root_by_id,
            parent_by_id=parent_by_id,
            region_by_id=region_by_id,
            target_presentation_policy=presentation_policy,
        )
        result.blocked_bundles.append(bundle)

    _assign_parent_execution_reading_order(result.bundles)
    _assign_parent_execution_reading_order(result.blocked_bundles)
    _validate_bundle_result(result)
    return result


def parent_execution_region_records(
    bundles: Sequence[ParentExecutionBundle],
) -> list[dict[str, Any]]:
    return [bundle.to_region_record() for bundle in bundles or []]


def sync_bundles_from_region_records(
    bundles: Sequence[ParentExecutionBundle],
    region_records: Sequence[Mapping[str, Any]],
) -> None:
    records_by_id = {
        str(record.get("region_id") or ""): record
        for record in region_records or []
        if isinstance(record, Mapping) and str(record.get("region_id") or "")
    }
    for bundle in bundles or []:
        record = records_by_id.get(bundle.bundle_id)
        if not record:
            continue
        bundle.execution_region = _copy_region_record(record)
        bundle.translated_text = str(record.get("translation") or record.get("translated_text") or "")
        bundle.source_glyph_mask_ids = _list_strings(record.get("source_glyph_mask_ids"))
        bundle.cleanup_job_ids = _list_strings(record.get("cleanup_job_ids"))
        bundle.cleanup_mask_ids = _list_strings(record.get("cleanup_mask_ids"))
        bundle.render_decision_id = str(record.get("render_decision_id") or "")
        bundle.renderer_audit_id = str(record.get("renderer_audit_id") or "")
        if "style_evidence_summary" in record:
            bundle.style_evidence_summary = _copy_mapping(record.get("style_evidence_summary"))
        render_record = record.get("render") if isinstance(record.get("render"), Mapping) else {}
        if "source_punctuation_geometry" in record or "source_punctuation_geometry" in render_record:
            bundle.source_punctuation_geometry = _copy_mapping(
                record.get("source_punctuation_geometry")
                or render_record.get("source_punctuation_geometry")
            )
        bundle.render_layout_summary = _copy_mapping(
            record.get("render_layout_summary")
            or render_record.get("render_layout_summary")
        )
        resolved_style = _resolved_render_style_from_region(record)
        if resolved_style:
            bundle.render_style = resolved_style
        elif not resolved_render_style_contract(bundle.render_style):
            bundle.render_style = {}
        bundle.source_contract_owner = str(record.get("source_contract_owner") or bundle.source_contract_owner or "")
        bundle.source_contract_region_id = str(record.get("source_contract_region_id") or bundle.source_contract_region_id or "")
        bundle.source_contract_bbox = _best_bbox(record.get("source_contract_bbox"), bundle.source_contract_bbox)
        bundle.source_contract_scope = str(record.get("source_contract_scope") or bundle.source_contract_scope or "")
        bundle.source_contract_stage = str(record.get("source_contract_stage") or bundle.source_contract_stage or "")
        bundle.source_contract_ocr_confidence = _float_or_none(
            record.get("source_contract_ocr_confidence")
            if record.get("source_contract_ocr_confidence") is not None
            else bundle.source_contract_ocr_confidence
        )
        if record.get("source_quality_reason_codes"):
            bundle.source_quality_reason_codes = _list_strings(record.get("source_quality_reason_codes"))
        bundle.to_region_record()


def parent_execution_bundles_from_audit_records(
    records: Sequence[Mapping[str, Any]],
) -> list[ParentExecutionBundle]:
    """Rehydrate saved parent execution bundle audit records for UI consumers."""

    bundles: list[ParentExecutionBundle] = []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        bundle = ParentExecutionBundle(
            page_id=str(record.get("page_id") or ""),
            bundle_id=str(record.get("bundle_id") or record.get("parent_id") or ""),
            root_id=str(record.get("root_id") or ""),
            parent_id=str(record.get("parent_id") or record.get("bundle_id") or ""),
            graph_parent_id=str(record.get("graph_parent_id") or record.get("parent_id") or ""),
            state=str(record.get("state") or ""),
            role=str(record.get("role") or ""),
            source_text=str(record.get("source_text") or ""),
            source_quality_state=str(record.get("source_quality_state") or "accepted_for_translation"),
            source_quality_action=str(record.get("source_quality_action") or "translate"),
            source_contract_owner=str(record.get("source_contract_owner") or ""),
            source_contract_region_id=str(record.get("source_contract_region_id") or ""),
            source_contract_bbox=_best_bbox(record.get("source_contract_bbox")),
            source_contract_scope=str(record.get("source_contract_scope") or ""),
            source_contract_stage=str(record.get("source_contract_stage") or ""),
            source_contract_ocr_confidence=_float_or_none(record.get("source_contract_ocr_confidence")),
            source_quality_reason_codes=_list_strings(record.get("source_quality_reason_codes")),
            translation_required=bool(record.get("translation_required")),
            cleanup_required=bool(record.get("cleanup_required")),
            render_required=bool(record.get("render_required")),
            parent_bbox=_bbox(record.get("parent_bbox")),
            cleanup_target_bbox=_bbox(record.get("cleanup_target_bbox")),
            render_allowed_area=_bbox(record.get("render_allowed_area")),
            root_bbox=_bbox(record.get("root_bbox")),
            source_region_ids=_list_strings(record.get("source_region_ids")),
            represented_child_ids=_list_strings(record.get("represented_child_ids")),
            source_candidates=[
                dict(item)
                for item in (record.get("source_candidates") or [])
                if isinstance(item, Mapping)
            ],
            semantic_class=str(record.get("semantic_class") or ""),
            route_intent=str(record.get("route_intent") or ""),
            cleanup_mode=str(record.get("cleanup_mode") or ""),
            text_area_container_id=str(record.get("text_area_container_id") or ""),
            text_area_container_type=str(record.get("text_area_container_type") or ""),
            text_area_container_bbox=_bbox(
                record.get("text_area_container_bbox")
            ),
            text_area_container_polygon=_polygon(
                record.get("text_area_container_polygon")
            ),
            text_area_oriented_frame=_copy_mapping(
                record.get("text_area_oriented_frame")
            ),
            target_language=str(record.get("target_language") or ""),
            target_presentation_policy=_copy_mapping(
                record.get("target_presentation_policy")
            ),
            render_layout_domain=_copy_mapping(
                record.get("render_layout_domain")
            ),
            confidence=_float_or_none(record.get("confidence")),
            reason_codes=_list_strings(record.get("reason_codes")),
            unresolved_reason=record.get("unresolved_reason"),
            translated_text=str(record.get("translated_text") or ""),
            source_glyph_mask_ids=_list_strings(record.get("source_glyph_mask_ids")),
            cleanup_job_ids=_list_strings(record.get("cleanup_job_ids")),
            cleanup_mask_ids=_list_strings(record.get("cleanup_mask_ids")),
            render_decision_id=str(record.get("render_decision_id") or ""),
            renderer_audit_id=str(record.get("renderer_audit_id") or ""),
            style_evidence_summary=_copy_mapping(record.get("style_evidence_summary")),
            source_punctuation_geometry=_copy_mapping(
                record.get("source_punctuation_geometry")
                or (
                    (record.get("execution_region") or {}).get(
                        "source_punctuation_geometry"
                    )
                    if isinstance(record.get("execution_region"), Mapping)
                    else {}
                )
            ),
            render_layout_summary=_copy_mapping(
                record.get("render_layout_summary")
                or (
                    (record.get("execution_region") or {}).get("render", {}).get("render_layout_summary")
                    if isinstance(record.get("execution_region"), Mapping)
                    and isinstance((record.get("execution_region") or {}).get("render"), Mapping)
                    else {}
                )
            ),
            render_style=resolved_render_style_contract(record.get("render_style")),
            execution_region=_copy_region_record(record.get("execution_region") or {}),
            reading_order_index=int(record.get("reading_order_index") or record.get("order_index") or 0),
        )
        if not bundle.execution_region:
            bundle.to_region_record()
        bundles.append(bundle)
    return bundles


def _assign_parent_execution_reading_order(bundles: list[ParentExecutionBundle]) -> None:
    if not bundles:
        return
    bundles[:] = _sort_parent_execution_bundles_for_reading_order(bundles)
    for index, bundle in enumerate(bundles):
        bundle.reading_order_index = index
        if bundle.execution_region:
            bundle.to_region_record()


def _sort_parent_execution_bundles_for_reading_order(
    bundles: Sequence[ParentExecutionBundle],
) -> list[ParentExecutionBundle]:
    """Return bundles in Japanese manga page order without splitting roots.

    Page-level order is root-first: upper bands before lower bands, then
    right-to-left within a band. Parent order inside a root follows vertical
    Japanese text flow: right-side columns before left-side columns, and
    top-to-bottom within a column.
    """

    root_groups: dict[str, list[ParentExecutionBundle]] = {}
    for bundle in bundles or []:
        root_key = str(bundle.root_id or bundle.bundle_id or bundle.parent_id or "")
        root_groups.setdefault(root_key, []).append(bundle)
    if not root_groups:
        return []

    root_records: list[tuple[str, list[int], list[ParentExecutionBundle]]] = []
    for root_id, root_bundles in root_groups.items():
        root_bbox = _best_bbox(
            root_bundles[0].root_bbox if root_bundles else [],
            _union_bboxes([_bundle_reading_bbox(bundle) for bundle in root_bundles]),
        )
        root_records.append((root_id, root_bbox, root_bundles))

    root_heights = [box[3] for _root_id, box, _items in root_records if _valid_bbox(box)]
    root_band = max(128.0, _median(root_heights) * 0.45) if root_heights else 128.0

    ordered: list[ParentExecutionBundle] = []
    for _root_id, _root_bbox, root_bundles in _sort_root_records_for_page_reading(root_records, root_band):
        ordered.extend(_sort_root_parent_bundles(root_bundles))
    return ordered


def _sort_root_records_for_page_reading(
    root_records: Sequence[tuple[str, list[int], list[ParentExecutionBundle]]],
    row_threshold: float,
) -> list[tuple[str, list[int], list[ParentExecutionBundle]]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(root_records, key=lambda item: (_root_top(item[1]), -_root_right(item[1]), item[0])):
        y = _root_top(record[1])
        target = None
        for row in rows:
            if abs(y - float(row["anchor_y"])) <= row_threshold:
                target = row
                break
        if target is None:
            rows.append({"anchor_y": y, "records": [record]})
        else:
            target["records"].append(record)
            target["anchor_y"] = min(float(target["anchor_y"]), y)

    ordered: list[tuple[str, list[int], list[ParentExecutionBundle]]] = []
    for row in sorted(rows, key=lambda item: float(item["anchor_y"])):
        ordered.extend(
            sorted(
                row["records"],
                key=lambda item: _root_row_reading_key(item[1], item[0]),
            )
        )
    return ordered


def _root_row_reading_key(root_bbox: Sequence[int], root_id: str) -> tuple[float, float, float, str]:
    box = _bbox(root_bbox)
    if not _valid_bbox(box):
        return (0.0, 0.0, 0.0, str(root_id or ""))
    x, y, w, _h = [float(value) for value in box]
    return (-(x + w), y, x, str(root_id or ""))


def _root_top(root_bbox: Sequence[int]) -> float:
    box = _bbox(root_bbox)
    return float(box[1]) if _valid_bbox(box) else 0.0


def _root_right(root_bbox: Sequence[int]) -> float:
    box = _bbox(root_bbox)
    return float(box[0] + box[2]) if _valid_bbox(box) else 0.0


def _sort_root_parent_bundles(
    bundles: Sequence[ParentExecutionBundle],
) -> list[ParentExecutionBundle]:
    entries: list[tuple[ParentExecutionBundle, list[int], float]] = []
    for bundle in bundles or []:
        box = _bundle_reading_bbox(bundle)
        if not _valid_bbox(box):
            entries.append((bundle, box, 0.0))
            continue
        x, _y, w, _h = box
        entries.append((bundle, box, float(x) + float(w) / 2.0))
    if len(entries) <= 1:
        return [entry[0] for entry in entries]

    widths = [box[2] for _bundle, box, _center_x in entries if _valid_bbox(box)]
    column_threshold = max(32.0, _median(widths) * 0.45) if widths else 32.0
    columns: list[dict[str, Any]] = []
    column_by_bundle_id: dict[int, int] = {}
    for bundle, box, center_x in sorted(entries, key=lambda item: (-item[2], _bbox(item[1])[1], str(item[0].parent_id))):
        assigned = None
        for index, column in enumerate(columns):
            if abs(center_x - float(column["center_x"])) <= column_threshold:
                assigned = index
                values = list(column["centers"])
                values.append(center_x)
                column["centers"] = values
                column["center_x"] = sum(values) / len(values)
                break
        if assigned is None:
            assigned = len(columns)
            columns.append({"center_x": center_x, "centers": [center_x]})
        column_by_bundle_id[id(bundle)] = assigned

    def parent_key(entry: tuple[ParentExecutionBundle, list[int], float]) -> tuple[Any, ...]:
        bundle, box, _center_x = entry
        if not _valid_bbox(box):
            return (9999, 0, 0, str(bundle.parent_id or bundle.bundle_id or ""))
        x, y, _w, _h = box
        if _bundle_source_orientation(bundle).startswith("horizontal"):
            return (0, y, x, str(bundle.parent_id or bundle.bundle_id or ""))
        return (
            column_by_bundle_id.get(id(bundle), 9999),
            y,
            -x,
            str(bundle.parent_id or bundle.bundle_id or ""),
        )

    return [entry[0] for entry in sorted(entries, key=parent_key)]


def _bundle_source_orientation(bundle: ParentExecutionBundle) -> str:
    style = bundle.render_style if isinstance(bundle.render_style, Mapping) else {}
    if style.get("source_orientation"):
        return str(style.get("source_orientation") or "").strip().lower()
    region = bundle.execution_region if isinstance(bundle.execution_region, Mapping) else {}
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    return str(region.get("source_orientation") or render.get("source_orientation") or "").strip().lower()


def _bundle_reading_bbox(bundle: ParentExecutionBundle) -> list[int]:
    return _best_bbox(bundle.parent_bbox, bundle.render_allowed_area, bundle.cleanup_target_bbox, bundle.root_bbox)


def _union_bboxes(boxes: Sequence[Any]) -> list[int]:
    valid = [_bbox(box) for box in boxes or []]
    valid = [box for box in valid if _valid_bbox(box)]
    if not valid:
        return []
    x1 = min(box[0] for box in valid)
    y1 = min(box[1] for box in valid)
    x2 = max(box[0] + box[2] for box in valid)
    y2 = max(box[1] + box[3] for box in valid)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _median(values: Sequence[int | float]) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def _bundle_from_finalized_parent(
    *,
    page_id: str,
    parent: Any,
    root_by_id: Mapping[str, Any],
    parent_by_id: Mapping[str, Any],
    region_by_id: Mapping[str, Mapping[str, Any]],
    target_presentation_policy: TargetPresentationPolicy,
) -> ParentExecutionBundle:
    parent_id = str(getattr(parent, "parent_id", "") or "")
    root_id = str(getattr(parent, "root_id", "") or "")
    parent_unit = parent_by_id.get(parent_id)
    root = root_by_id.get(root_id)
    source_region_ids = _list_strings(getattr(parent, "source_region_ids", []))
    source_records = [
        _source_candidate_from_region(region_by_id.get(region_id, {}), region_id)
        for region_id in source_region_ids
    ]
    source_records = [record for record in source_records if record]
    role = str(getattr(parent, "role", "") or getattr(parent_unit, "role", "") or "")
    parent_bbox = _best_bbox(
        getattr(parent, "render_allowed_area", []),
        getattr(parent, "cleanup_target_bbox", []),
        getattr(parent_unit, "parent_visual_group_bbox", []),
        _union_region_bboxes([region_by_id.get(region_id, {}) for region_id in source_region_ids]),
    )
    cleanup_target = _best_bbox(getattr(parent, "cleanup_target_bbox", []), parent_bbox)
    render_allowed = _best_bbox(getattr(parent, "render_allowed_area", []), parent_bbox)
    root_bbox = _best_bbox(getattr(root, "bbox", []), render_allowed)
    primary_region = region_by_id.get(source_region_ids[0], {}) if source_region_ids else {}
    primary_render = primary_region.get("render") if isinstance(primary_region, Mapping) else {}
    if not isinstance(primary_render, Mapping):
        primary_render = {}
    source_action = str(
        getattr(parent_unit, "source_coherence_action", "")
        or primary_region.get("logical_text_source_quality_action")
        or primary_render.get("logical_text_source_quality_action")
        or "translate"
    )
    source_state = str(
        getattr(parent_unit, "source_contract_quality_state", "")
        or getattr(parent_unit, "source_conservation_status", "")
        or primary_region.get("source_conservation_status")
        or "accepted_for_translation"
    )
    source_contract_owner = str(
        getattr(parent_unit, "source_contract_owner", "")
        or primary_region.get("source_contract_owner")
        or primary_render.get("source_contract_owner")
        or ""
    )
    source_contract_region_id = str(
        getattr(parent_unit, "source_contract_region_id", "")
        or primary_region.get("source_contract_region_id")
        or primary_render.get("source_contract_region_id")
        or ""
    )
    if not source_contract_region_id and source_contract_owner and (
        primary_region.get("parent_boundary_ocr_source_contract")
        or primary_render.get("parent_boundary_ocr_source_contract")
    ):
        source_contract_region_id = str(primary_region.get("region_id") or "")
    source_contract_bbox = _best_bbox(
        primary_region.get("source_contract_bbox"),
        primary_render.get("source_contract_bbox"),
        primary_region.get("bbox") if (
            primary_region.get("parent_boundary_ocr_source_contract")
            or primary_render.get("parent_boundary_ocr_source_contract")
        ) else [],
    )
    if not source_contract_bbox and source_contract_owner:
        source_contract_bbox = list(parent_bbox)
    source_contract_scope = str(
        primary_region.get("source_contract_scope")
        or primary_render.get("source_contract_scope")
        or primary_region.get("parent_source_candidate_scope")
        or primary_render.get("parent_source_candidate_scope")
        or ""
    )
    if not source_contract_scope and source_contract_owner:
        source_contract_scope = "parent_execution_region"
    source_contract_stage = str(
        primary_region.get("source_contract_stage")
        or primary_render.get("source_contract_stage")
        or primary_region.get("parent_source_candidate_stage")
        or primary_render.get("parent_source_candidate_stage")
        or ""
    )
    if not source_contract_stage and source_contract_owner:
        source_contract_stage = (
            "text_block_hierarchy_punctuation_identity"
            if source_action == "identity_punctuation"
            else "parent_execution_bundle_source_contract_fallback"
        )
    source_contract_ocr_confidence = _float_or_none(
        primary_region.get("source_contract_ocr_confidence")
        if primary_region.get("source_contract_ocr_confidence") is not None
        else primary_render.get("source_contract_ocr_confidence")
    )
    if source_contract_ocr_confidence is None:
        confidence = primary_region.get("confidence") if isinstance(primary_region.get("confidence"), Mapping) else {}
        source_contract_ocr_confidence = _float_or_none(confidence.get("ocr"))
    source_reason_codes = _list_strings(getattr(parent_unit, "source_quality_warning_reason_codes", []))
    if not source_reason_codes:
        source_reason_codes = _list_strings(
            primary_region.get("parent_ocr_source_quality_reason_codes")
            or primary_render.get("parent_ocr_source_quality_reason_codes")
        )
    ocr_backend = str(primary_region.get("ocr_backend") or primary_render.get("ocr_backend") or "")
    ocr_model_path = str(primary_region.get("ocr_model_path") or primary_render.get("ocr_model_path") or "")
    ocr_mmproj_path = str(primary_region.get("ocr_mmproj_path") or primary_render.get("ocr_mmproj_path") or "")
    ocr_endpoint = str(primary_region.get("ocr_endpoint") or primary_render.get("ocr_endpoint") or "")
    ocr_prompt_version = str(primary_region.get("ocr_prompt_version") or primary_render.get("ocr_prompt_version") or "")
    state = str(getattr(parent, "state", "") or "")
    source_text = str(getattr(parent, "source_text", "") or "")
    translation_required = bool(getattr(parent, "translation_required", False))
    translated_text = source_text if _is_punctuation_identity_parent(state, source_action, source_state) else ""
    container_id = str(
        primary_region.get("text_area_container_id")
        or primary_render.get("text_area_container_id")
        or ""
    )
    container_type = str(
        primary_region.get("text_area_container_type")
        or primary_render.get("text_area_container_type")
        or _container_type_for_role(role)
    )
    container_bbox = _best_bbox(
        primary_region.get("text_area_container_bbox"),
        primary_render.get("text_area_container_bbox"),
    )
    container_polygon = _polygon(
        primary_region.get("text_area_container_polygon")
        or primary_render.get("text_area_container_polygon")
    )
    oriented_frame = _copy_mapping(
        primary_region.get("text_area_oriented_frame")
        or primary_render.get("text_area_oriented_frame")
    )
    render_layout_domain = _parent_render_layout_domain(
        source_bounds=render_allowed,
        container_id=container_id,
        container_type=container_type,
        container_bbox=container_bbox,
        container_polygon=container_polygon,
        oriented_frame=oriented_frame,
        region=primary_region,
        render=primary_render,
        policy=target_presentation_policy,
    )
    return ParentExecutionBundle(
        page_id=page_id,
        bundle_id=parent_id,
        root_id=root_id,
        parent_id=parent_id,
        graph_parent_id=parent_id,
        state=state,
        role=role,
        source_text=source_text,
        source_quality_state=source_state,
        source_quality_action=source_action,
        source_contract_owner=source_contract_owner,
        source_contract_region_id=source_contract_region_id,
        source_contract_bbox=source_contract_bbox,
        source_contract_scope=source_contract_scope,
        source_contract_stage=source_contract_stage,
        source_contract_ocr_confidence=source_contract_ocr_confidence,
        ocr_backend=ocr_backend,
        ocr_model_path=ocr_model_path,
        ocr_mmproj_path=ocr_mmproj_path,
        ocr_endpoint=ocr_endpoint,
        ocr_prompt_version=ocr_prompt_version,
        source_quality_reason_codes=source_reason_codes,
        translation_required=translation_required,
        cleanup_required=bool(getattr(parent, "cleanup_required", False)),
        render_required=bool(getattr(parent, "render_required", False)),
        parent_bbox=list(parent_bbox),
        cleanup_target_bbox=list(cleanup_target),
        render_allowed_area=list(render_allowed),
        root_bbox=list(root_bbox),
        source_region_ids=source_region_ids,
        represented_child_ids=_list_strings(getattr(parent, "represented_child_ids", [])),
        source_candidates=source_records,
        semantic_class=_semantic_class_for_role(role),
        route_intent=_route_intent_for_role(role),
        cleanup_mode=_cleanup_mode_for_role(role),
        text_area_container_id=container_id,
        text_area_container_type=container_type,
        text_area_container_bbox=container_bbox,
        text_area_container_polygon=container_polygon,
        text_area_oriented_frame=oriented_frame,
        target_language=target_presentation_policy.target_language,
        target_presentation_policy=(
            target_presentation_policy.to_contract_dict()
        ),
        render_layout_domain=render_layout_domain,
        confidence=_float_or_none(getattr(parent_unit, "confidence", None)),
        reason_codes=_list_strings(getattr(parent, "reason_codes", [])),
        unresolved_reason=getattr(parent, "unresolved_reason", None),
        translated_text=translated_text,
        render_style={},
    )


def _parent_render_layout_domain(
    *,
    source_bounds: Sequence[Any],
    container_id: str,
    container_type: str,
    container_bbox: Sequence[Any],
    container_polygon: Sequence[Any],
    oriented_frame: Mapping[str, Any],
    region: Mapping[str, Any],
    render: Mapping[str, Any],
    policy: TargetPresentationPolicy,
) -> dict[str, Any]:
    source = _bbox(source_bounds)
    container = _bbox(container_bbox)
    polygon = _polygon(container_polygon)
    frame = _copy_mapping(oriented_frame)
    conflicts = _list_strings(
        region.get("text_area_conflict_flags")
        or render.get("text_area_conflict_flags")
    )
    explicit = bool(
        region.get("text_area_authorization_explicit")
        or render.get("text_area_authorization_explicit")
    )
    protected = bool(
        region.get("text_area_must_not_mutate")
        or render.get("text_area_must_not_mutate")
    )
    authorization = str(
        region.get("text_area_semantic_authorization_state")
        or render.get("text_area_semantic_authorization_state")
        or ""
    )
    source_inside_container = _xywh_contains(container, source)
    shape_safe_latin_policy = policy.automatic_domain_policy == (
        "shape_safe_speech_container_with_source_alignment_prior_or_source"
    )
    supported = bool(
        source
        and container
        and source_inside_container
        and container_id
        and container_type == "speech_bubble"
        and explicit
        and not protected
        and not conflicts
        and authorization == "cleanup_translate_speech"
        and (not shape_safe_latin_policy or bool(polygon))
    )
    if supported:
        source_side_anchor = {
            "policy_version": SOURCE_SIDE_ANCHOR_POLICY_VERSION,
            "status": "not_applicable",
            "reason": "automatic_domain_policy_does_not_request_anchor",
        }
        if policy.automatic_domain_policy == (
            "source_side_anchored_speech_container_or_source"
        ):
            automatic, source_side_anchor = _source_side_anchored_container_bounds(
                source,
                container,
            )
            status = "authorized_source_side_anchored_speech_container"
            reasons = [
                "exact_text_area_plan_speech_container",
                "source_side_anchor_preserved",
            ]
        elif shape_safe_latin_policy:
            alignment_prior_bounds, source_side_anchor = (
                _source_side_anchored_container_bounds(
                    source,
                    container,
                )
            )
            source_side_anchor = {
                **source_side_anchor,
                "capacity_role": "alignment_prior_only",
                "alignment_prior_bounds": list(alignment_prior_bounds),
                "automatic_bounds": list(container),
                "editable_bounds": list(container),
            }
            automatic = list(container)
            status = "authorized_shape_safe_speech_container"
            reasons = [
                "exact_text_area_plan_speech_container",
                "exact_text_area_plan_speech_polygon",
                "source_side_anchor_retained_as_alignment_prior",
            ]
        elif policy.automatic_domain_policy == (
            "authorized_speech_container_or_source"
        ):
            automatic = list(container)
            status = "authorized_speech_container"
            reasons = ["exact_text_area_plan_speech_container"]
        else:
            automatic = list(source)
            status = "authorized_speech_container"
            reasons = ["exact_text_area_plan_speech_container"]
        editable = list(container)
    else:
        automatic = list(source)
        editable = list(source)
        status = "conservative_source_bounds"
        reasons = [
            reason
            for reason, present in (
                ("source_bounds_missing", not source),
                ("container_bbox_missing", not container),
                ("source_outside_container", not source_inside_container),
                ("container_id_missing", not container_id),
                ("container_not_speech", container_type != "speech_bubble"),
                ("container_authorization_not_explicit", not explicit),
                ("container_protected", protected),
                ("container_conflicted", bool(conflicts)),
                (
                    "container_not_cleanup_translate_speech",
                    authorization != "cleanup_translate_speech",
                ),
                (
                    "container_polygon_missing",
                    shape_safe_latin_policy and not bool(polygon),
                ),
            )
            if present
        ]
        source_side_anchor = {
            "policy_version": SOURCE_SIDE_ANCHOR_POLICY_VERSION,
            "status": "not_applied",
            "reason": "container_domain_not_supported",
        }
    return {
        "contract_version": "parent_render_domain_v1",
        "policy_id": policy.policy_id,
        "status": status,
        "source_bounds": list(source),
        "automatic_bounds": automatic,
        "editable_bounds": editable,
        "container_id": str(container_id or ""),
        "container_type": str(container_type or ""),
        "container_bbox": list(container),
        "container_polygon": _copy_jsonish(polygon),
        "oriented_frame": _copy_jsonish(frame),
        "container_authorization_state": authorization,
        "container_conflict_flags": conflicts,
        "provenance": "TextAreaPlan",
        "reason_codes": reasons,
        "source_side_anchor": source_side_anchor,
    }


def _xywh_contains(outer: Sequence[Any], inner: Sequence[Any]) -> bool:
    outer_box = _bbox(outer)
    inner_box = _bbox(inner)
    if not outer_box or not inner_box:
        return False
    outer_x, outer_y, outer_w, outer_h = outer_box
    inner_x, inner_y, inner_w, inner_h = inner_box
    return bool(
        outer_x <= inner_x
        and outer_y <= inner_y
        and inner_x + inner_w <= outer_x + outer_w
        and inner_y + inner_h <= outer_y + outer_h
    )


def _source_side_anchored_container_bounds(
    source: Sequence[Any],
    container: Sequence[Any],
) -> tuple[list[int], dict[str, Any]]:
    """Expand vertically while preserving the source text's horizontal side.

    Manga bubbles often contain tails, icons, or decorative art opposite a
    source text column.  English may use the authorized container, but it must
    not automatically cross that evidence-backed source-side anchor merely to
    obtain a wider rectangle.  A small deterministic center tolerance avoids
    treating detection noise as a meaningful side preference.
    """

    source_box = _bbox(source)
    container_box = _bbox(container)
    if not _xywh_contains(container_box, source_box):
        return list(source_box), {
            "policy_version": SOURCE_SIDE_ANCHOR_POLICY_VERSION,
            "status": "not_applied",
            "reason": "source_outside_container",
        }
    source_x, _source_y, source_w, _source_h = source_box
    container_x, container_y, container_w, container_h = container_box
    source_center = source_x + source_w / 2.0
    container_center = container_x + container_w / 2.0
    center_tolerance = max(
        SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_MIN_PX,
        container_w * SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_RATIO,
    )
    relation = "centered"
    if source_center > container_center + center_tolerance:
        relation = "right"
        bounds = [
            source_x,
            container_y,
            container_x + container_w - source_x,
            container_h,
        ]
    elif source_center < container_center - center_tolerance:
        relation = "left"
        bounds = [
            container_x,
            container_y,
            source_x + source_w - container_x,
            container_h,
        ]
    else:
        bounds = list(container_box)
    return bounds, {
        "policy_version": SOURCE_SIDE_ANCHOR_POLICY_VERSION,
        "status": "applied",
        "source_horizontal_relation": relation,
        "source_center_x": round(float(source_center), 6),
        "container_center_x": round(float(container_center), 6),
        "center_delta_x": round(float(source_center - container_center), 6),
        "center_tolerance_px": round(float(center_tolerance), 6),
        "center_tolerance_ratio": (
            SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_RATIO
        ),
        "center_tolerance_min_px": (
            SOURCE_SIDE_ANCHOR_CENTER_TOLERANCE_MIN_PX
        ),
        "automatic_bounds": list(bounds),
        "editable_bounds": list(container_box),
        "translation_content_consulted": False,
        "render_output_consulted": False,
    }


def _is_punctuation_identity_parent(state: str, source_action: str, source_state: str) -> bool:
    return (
        str(state or "") == "punctuation_identity_parent"
        or str(source_action or "") == "identity_punctuation"
        or str(source_state or "") == "punctuation_identity_source"
    )


def _sync_execution_region_from_bundle(
    bundle: ParentExecutionBundle,
    record: dict[str, Any],
) -> None:
    render = record.setdefault("render", {})
    if not isinstance(render, dict):
        render = {}
        record["render"] = render
    record["region_id"] = bundle.bundle_id
    record["parent_execution_bundle_id"] = bundle.bundle_id
    record["parent_execution_bundle_version"] = PARENT_EXECUTION_BUNDLE_VERSION
    record["parent_execution_state"] = bundle.state
    record["parent_execution_authoritative"] = True
    record["text_block_root_id"] = bundle.root_id
    record["parent_logical_text_unit_id"] = bundle.parent_id
    record["active_translation_unit_id"] = bundle.parent_id if bundle.translation_required else ""
    record["logical_text_block_id"] = bundle.parent_id
    record["ocr_text"] = bundle.source_text
    record["source_text"] = bundle.source_text
    record["logical_text_block_source_text"] = bundle.source_text
    record["parent_logical_text_unit_source_text"] = bundle.source_text
    record["source_contract_owner"] = bundle.source_contract_owner
    record["source_contract_region_id"] = bundle.source_contract_region_id
    record["source_contract_bbox"] = list(bundle.source_contract_bbox)
    record["source_contract_scope"] = bundle.source_contract_scope
    record["source_contract_stage"] = bundle.source_contract_stage
    record["source_contract_ocr_confidence"] = bundle.source_contract_ocr_confidence
    record["ocr_backend"] = bundle.ocr_backend
    record["ocr_model_path"] = bundle.ocr_model_path
    record["ocr_mmproj_path"] = bundle.ocr_mmproj_path
    record["ocr_endpoint"] = bundle.ocr_endpoint
    record["ocr_prompt_version"] = bundle.ocr_prompt_version
    record["source_quality_reason_codes"] = list(bundle.source_quality_reason_codes)
    record["translation"] = bundle.translated_text
    record["translated_text"] = bundle.translated_text
    record["translation_required"] = bool(bundle.translation_required)
    record["cleanup_required"] = bool(bundle.cleanup_required)
    record["render_required"] = bool(bundle.render_required)
    record["order_index"] = int(bundle.reading_order_index)
    record["reading_order_index"] = int(bundle.reading_order_index)
    record["source_region_ids"] = list(bundle.source_region_ids)
    record["represented_child_ids"] = list(bundle.represented_child_ids)
    record["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    record["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    record["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    record["render_decision_id"] = bundle.render_decision_id
    record["renderer_audit_id"] = bundle.renderer_audit_id
    record["style_evidence_summary"] = _copy_jsonish(bundle.style_evidence_summary)
    record["source_punctuation_geometry"] = _copy_jsonish(
        bundle.source_punctuation_geometry
    )
    record["render_layout_summary"] = _copy_jsonish(bundle.render_layout_summary)
    record["execution_region_authority"] = "parent_execution_bundle"
    record["execution_region_role"] = "parent_execution"
    record["legacy_region_execution_authority"] = False
    record["source_region_evidence_only"] = True
    record["text_area_container_id"] = bundle.text_area_container_id
    record["text_area_container_type"] = bundle.text_area_container_type
    record["text_area_container_bbox"] = list(bundle.text_area_container_bbox)
    record["text_area_container_polygon"] = _copy_jsonish(
        bundle.text_area_container_polygon
    )
    record["text_area_oriented_frame"] = _copy_jsonish(
        bundle.text_area_oriented_frame
    )
    record["target_language"] = bundle.target_language
    record["target_presentation_policy"] = _copy_jsonish(
        bundle.target_presentation_policy
    )
    record["render_layout_domain"] = _copy_jsonish(bundle.render_layout_domain)
    _clear_executable_style_fields(record)
    _clear_executable_style_fields(render)
    render_style = resolved_render_style_contract(bundle.render_style)
    bundle.render_style = _copy_jsonish(render_style)
    if render_style:
        record.update(_render_style_record_fields(render_style))
    render["parent_execution_bundle_id"] = bundle.bundle_id
    render["parent_execution_bundle_version"] = PARENT_EXECUTION_BUNDLE_VERSION
    if render_style:
        render.update(_render_style_record_fields(render_style))
        render.update(_render_style_flattened_fields(render_style))
    render["parent_execution_authoritative"] = True
    render["text_block_root_id"] = bundle.root_id
    render["parent_logical_text_unit_id"] = bundle.parent_id
    render["active_translation_unit_id"] = bundle.parent_id if bundle.translation_required else ""
    render["logical_text_block_source_text"] = bundle.source_text
    render["parent_logical_text_unit_source_text"] = bundle.source_text
    render["source_text"] = bundle.source_text
    render["source_contract_owner"] = bundle.source_contract_owner
    render["source_contract_region_id"] = bundle.source_contract_region_id
    render["source_contract_bbox"] = list(bundle.source_contract_bbox)
    render["source_contract_scope"] = bundle.source_contract_scope
    render["source_contract_stage"] = bundle.source_contract_stage
    render["source_contract_ocr_confidence"] = bundle.source_contract_ocr_confidence
    render["ocr_backend"] = bundle.ocr_backend
    render["ocr_model_path"] = bundle.ocr_model_path
    render["ocr_mmproj_path"] = bundle.ocr_mmproj_path
    render["ocr_endpoint"] = bundle.ocr_endpoint
    render["ocr_prompt_version"] = bundle.ocr_prompt_version
    render["source_quality_reason_codes"] = list(bundle.source_quality_reason_codes)
    render["translation"] = bundle.translated_text
    render["translated_text"] = bundle.translated_text
    render["translation_required"] = bool(bundle.translation_required)
    render["cleanup_required"] = bool(bundle.cleanup_required)
    render["render_required"] = bool(bundle.render_required)
    render["order_index"] = int(bundle.reading_order_index)
    render["reading_order_index"] = int(bundle.reading_order_index)
    render["source_region_ids"] = list(bundle.source_region_ids)
    render["represented_child_ids"] = list(bundle.represented_child_ids)
    render["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    render["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    render["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    render["render_decision_id"] = bundle.render_decision_id
    render["renderer_audit_id"] = bundle.renderer_audit_id
    render["source_punctuation_geometry"] = _copy_jsonish(
        bundle.source_punctuation_geometry
    )
    render["render_layout_summary"] = _copy_jsonish(bundle.render_layout_summary)
    render["execution_region_authority"] = "parent_execution_bundle"
    render["execution_region_role"] = "parent_execution"
    render["legacy_region_execution_authority"] = False
    render["source_region_evidence_only"] = True
    render["text_area_container_id"] = bundle.text_area_container_id
    render["text_area_container_type"] = bundle.text_area_container_type
    render["text_area_container_bbox"] = list(bundle.text_area_container_bbox)
    render["text_area_container_polygon"] = _copy_jsonish(
        bundle.text_area_container_polygon
    )
    render["text_area_oriented_frame"] = _copy_jsonish(
        bundle.text_area_oriented_frame
    )
    render["target_language"] = bundle.target_language
    render["target_presentation_policy"] = _copy_jsonish(
        bundle.target_presentation_policy
    )
    render["render_layout_domain"] = _copy_jsonish(bundle.render_layout_domain)


def _validate_bundle_result(result: ParentExecutionBundleResult) -> None:
    seen: set[str] = set()
    for bundle in result.bundles:
        if bundle.parent_id in seen:
            result.errors.append(f"duplicate_bundle_parent_id:{bundle.parent_id}")
        seen.add(bundle.parent_id)
        if bundle.state == "active_translation_parent" and not bundle.source_text.strip():
            result.errors.append(f"active_parent_missing_source_text:{bundle.parent_id}")
        if bundle.state == "active_translation_parent" and not _valid_bbox(bundle.parent_bbox):
            result.errors.append(f"active_parent_missing_parent_bbox:{bundle.parent_id}")
        if not bundle.root_id:
            result.errors.append(f"bundle_missing_root_id:{bundle.parent_id}")


def _source_candidate_from_region(region: Mapping[str, Any], region_id: str) -> dict[str, Any]:
    if not isinstance(region, Mapping):
        return {}
    render = region.get("render") or {}
    if not isinstance(render, Mapping):
        render = {}
    return {
        "region_id": region_id,
        "ocr_text": str(region.get("ocr_text") or render.get("ocr_text") or ""),
        "bbox": _bbox(region.get("bbox")),
        "polygon": list(region.get("polygon") or []),
        "child_id": str(region.get("child_recognized_text_segment_id") or render.get("child_recognized_text_segment_id") or ""),
        "confidence": region.get("confidence"),
        "detection_source": str(region.get("detection_source") or render.get("detection_source") or ""),
        "parent_boundary_ocr_source_contract": bool(
            region.get("parent_boundary_ocr_source_contract")
            or render.get("parent_boundary_ocr_source_contract")
        ),
        "source_contract_owner": str(region.get("source_contract_owner") or render.get("source_contract_owner") or ""),
        "source_contract_region_id": str(region.get("source_contract_region_id") or render.get("source_contract_region_id") or ""),
        "source_contract_bbox": _best_bbox(region.get("source_contract_bbox"), render.get("source_contract_bbox")),
        "source_contract_scope": str(region.get("source_contract_scope") or render.get("source_contract_scope") or ""),
        "source_contract_stage": str(region.get("source_contract_stage") or render.get("source_contract_stage") or ""),
        "source_contract_ocr_confidence": _float_or_none(
            region.get("source_contract_ocr_confidence")
            if region.get("source_contract_ocr_confidence") is not None
            else render.get("source_contract_ocr_confidence")
        ),
        "ocr_backend": str(region.get("ocr_backend") or render.get("ocr_backend") or ""),
        "ocr_model_path": str(region.get("ocr_model_path") or render.get("ocr_model_path") or ""),
        "ocr_mmproj_path": str(region.get("ocr_mmproj_path") or render.get("ocr_mmproj_path") or ""),
        "ocr_endpoint": str(region.get("ocr_endpoint") or render.get("ocr_endpoint") or ""),
        "ocr_prompt_version": str(region.get("ocr_prompt_version") or render.get("ocr_prompt_version") or ""),
        "parent_ocr_source_quality_state": str(
            region.get("parent_ocr_source_quality_state")
            or render.get("parent_ocr_source_quality_state")
            or ""
        ),
        "parent_ocr_source_quality_action": str(
            region.get("parent_ocr_source_quality_action")
            or render.get("parent_ocr_source_quality_action")
            or ""
        ),
        "parent_ocr_source_quality_reason_codes": _list_strings(
            region.get("parent_ocr_source_quality_reason_codes")
            or render.get("parent_ocr_source_quality_reason_codes")
        ),
    }


def _resolved_render_style_from_region(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    render = record.get("render")
    for value in (
        record.get("render_style"),
        render.get("render_style") if isinstance(render, Mapping) else None,
    ):
        style = resolved_render_style_contract(value)
        if style:
            return style
    return {}


def validate_resolved_render_style(value: Any) -> ResolvedRenderStyleValidation:
    """Validate and isolate the sole executable ``parent_render_style_v4``.

    Stored v3 records receive only the conservative pixel-preserving migration
    defined below; the gate accepts no v2 aliases or flattened projection.
    Style evidence and target realization belong to ``ParentStyleArbitrator``;
    this function only enforces the complete handoff consumed by rendering.
    """

    if not isinstance(value, Mapping):
        return ResolvedRenderStyleValidation(
            status="rejected",
            reason_codes=("resolved_render_style_not_mapping",),
        )
    style = _copy_jsonish(value)
    if not isinstance(style, dict):
        return ResolvedRenderStyleValidation(
            status="rejected",
            reason_codes=("resolved_render_style_not_json_mapping",),
        )
    if style.get("render_style_version") == "parent_render_style_v3":
        style = _upgrade_legacy_render_style_v3(style)

    reasons: list[str] = []
    keys = {key for key in style if isinstance(key, str)}
    for field_name in sorted(_PARENT_STYLE_REQUIRED_FIELDS - keys):
        reasons.append(f"required_field_missing:{field_name}")
    for field_name in sorted(
        keys - _PARENT_STYLE_REQUIRED_FIELDS - _PARENT_STYLE_OPTIONAL_FIELDS
    ):
        reasons.append(f"unknown_field:{field_name}")
    for field_name in sorted(_PARENT_STYLE_FORBIDDEN_LEGACY_FIELDS & keys):
        reasons.append(f"legacy_field_forbidden:{field_name}")

    expected_stamps = {
        "render_style_version": PARENT_RENDER_STYLE_VERSION,
        "render_style_owner": "ParentStyleArbitrator",
        "render_style_source": PARENT_STYLE_ARBITRATOR_SOURCE,
        "render_style_provider": PARENT_STYLE_ARBITRATOR_PROVIDER,
        "style_resolution_status": "complete",
    }
    for field_name, expected in expected_stamps.items():
        if style.get(field_name) != expected:
            reasons.append(f"{field_name}_invalid")
    if style.get("source_evidence_status") not in {"observed", "unavailable"}:
        reasons.append("source_evidence_status_invalid")
    confidence = _finite_number(style.get("render_style_confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        reasons.append("render_style_confidence_invalid")

    family = style.get("font_family_role")
    weight = style.get("font_weight_tier")
    if family not in _PARENT_STYLE_FAMILY_ROLES:
        reasons.append("font_family_role_invalid")
    if weight not in _PARENT_STYLE_WEIGHT_TIERS:
        reasons.append("font_weight_tier_invalid")
    expected_role, expected_role_status = _PARENT_STYLE_FONT_ROLE_MATRIX.get(
        (family, weight),
        (None, None),
    )
    if expected_role is not None and style.get("primary_font_role") != expected_role:
        reasons.append("primary_font_role_matrix_mismatch")
    role_status = style.get("primary_font_role_status")
    if role_status not in _PARENT_STYLE_ROLE_STATUSES:
        reasons.append("primary_font_role_status_invalid")
    if (
        role_status not in {None, "fallback_registered_role"}
        and expected_role_status is not None
        and role_status != expected_role_status
    ):
        reasons.append("primary_font_role_status_matrix_mismatch")
    if style.get("fallback_font_chain_key") != PARENT_STYLE_DEFAULT_FALLBACK_FONT_CHAIN_KEY:
        reasons.append("fallback_font_chain_key_invalid")

    presentation = style.get("target_presentation_policy")
    if not isinstance(presentation, Mapping):
        reasons.append("target_presentation_policy_invalid")
        presentation = {}
    elif presentation.get("contract_version") != "target_presentation_policy_v1":
        reasons.append("target_presentation_policy_invalid")
    target_language = style.get("target_language")
    target_script = style.get("target_script")
    shaping_locale = style.get("shaping_locale")
    if target_language not in {"zh-Hans", "en"}:
        reasons.append("target_language_invalid")
    if target_script not in {"Hani", "Latn"}:
        reasons.append("target_script_invalid")
    if not isinstance(shaping_locale, str) or not shaping_locale:
        reasons.append("shaping_locale_invalid")
    if presentation:
        for key, actual in (
            ("target_language", target_language),
            ("target_script", target_script),
            ("shaping_locale", shaping_locale),
        ):
            if presentation.get(key) != actual:
                reasons.append(f"target_presentation_{key}_mismatch")
    if style.get("source_writing_mode") not in _PARENT_STYLE_WRITING_MODES:
        reasons.append("source_writing_mode_invalid")

    source_cell = style.get("source_visual_cell")
    if not isinstance(source_cell, Mapping):
        reasons.append("source_visual_cell_invalid")
        source_cell = {}
    source_status = source_cell.get("status")
    if source_status not in _PARENT_STYLE_SOURCE_CELL_STATUSES:
        reasons.append("source_visual_cell_status_invalid")
    if source_cell.get("writing_mode") not in _PARENT_STYLE_WRITING_MODES:
        reasons.append("source_visual_cell_writing_mode_invalid")
    cell_confidence = _finite_number(source_cell.get("confidence"))
    if cell_confidence is None or not 0.0 <= cell_confidence <= 1.0:
        reasons.append("source_visual_cell_confidence_invalid")
    if not str(source_cell.get("provenance") or ""):
        reasons.append("source_visual_cell_provenance_missing")
    cell_interval = tuple(
        _finite_number(source_cell.get(name))
        for name in ("p20_px", "median_px", "p80_px")
    )
    if source_status in {"direct", "peer", "fallback"}:
        if any(value is None or value <= 0.0 for value in cell_interval):
            reasons.append("source_visual_cell_measurement_missing")
        elif not cell_interval[0] <= cell_interval[1] <= cell_interval[2]:
            reasons.append("source_visual_cell_interval_invalid")
        if source_cell.get("authority") != source_status:
            reasons.append("source_visual_cell_authority_invalid")
    elif source_status == "unavailable":
        if any(value is not None for value in cell_interval):
            reasons.append("source_visual_cell_unavailable_has_measurement")
        if source_cell.get("authority") != "fallback":
            reasons.append("source_visual_cell_authority_invalid")

    optical_reference_em = _finite_number(
        style.get("target_optical_reference_em_px")
    )


    fit_start_em = _finite_number(style.get("target_fit_start_em_px"))
    preferred_em = _finite_number(style.get("target_preferred_em_px"))
    if optical_reference_em is None or optical_reference_em <= 0.0:
        reasons.append("target_optical_reference_em_invalid")
    if fit_start_em is None or fit_start_em <= 0.0:
        reasons.append("target_fit_start_em_invalid")
    if preferred_em is None or preferred_em <= 0.0:
        reasons.append("target_preferred_em_invalid")
    if (
        fit_start_em is not None
        and preferred_em is not None
        and abs(fit_start_em - preferred_em) > 1e-6
    ):
        reasons.append("target_fit_start_preferred_alias_mismatch")
    interval = style.get("target_preferred_em_interval_px")
    if not isinstance(interval, list) or len(interval) != 2:
        reasons.append("target_preferred_em_interval_invalid")
    else:
        low, high = (_finite_number(interval[0]), _finite_number(interval[1]))
        if (
            low is None
            or high is None
            or low <= 0.0
            or high < low
            or preferred_em is None
            or not low <= preferred_em <= high
            or optical_reference_em is None
            or not low <= optical_reference_em <= high
        ):
            reasons.append("target_preferred_em_interval_invalid")
    if not str(style.get("target_face_profile_id") or ""):
        reasons.append("target_face_profile_id_missing")
    if not isinstance(style.get("target_em_conversion_audit"), Mapping):
        reasons.append("target_em_conversion_audit_invalid")
    size_preference = style.get("target_size_preference")
    if not isinstance(size_preference, Mapping):
        reasons.append("target_size_preference_invalid")
    else:
        if size_preference.get("never_decrease") is not True:
            reasons.append("target_size_preference_invalid")
        if size_preference.get("render_admission") is not False:
            reasons.append("target_size_preference_invalid")

    fill = style.get("fill")
    if not isinstance(fill, Mapping):
        reasons.append("fill_invalid")
    elif (
        not _HEX_COLOR_PATTERN.fullmatch(str(fill.get("color") or ""))
        or fill.get("polarity") not in {"dark", "light"}
    ):
        reasons.append("fill_invalid")
    outline = style.get("outline")
    if not isinstance(outline, Mapping):
        reasons.append("outline_invalid")
    else:
        if not isinstance(outline.get("present"), bool):
            reasons.append("outline_present_invalid")
        if not _HEX_COLOR_PATTERN.fullmatch(str(outline.get("color") or "")):
            reasons.append("outline_color_invalid")
        ratio = _finite_number(outline.get("source_width_to_cell_ratio"))
        width = _finite_number(outline.get("target_width_px"))
        if ratio is None or ratio < 0.0 or width is None or width < 0.0:
            reasons.append("outline_geometry_invalid")
    if style.get("writing_mode") not in _PARENT_STYLE_WRITING_MODES:
        reasons.append("writing_mode_invalid")
    line_height = _finite_number(style.get("line_height"))
    if line_height is None or line_height <= 0.0:
        reasons.append("line_height_invalid")
    if style.get("align") not in _PARENT_STYLE_ALIGNMENTS:
        reasons.append("align_invalid")

    authority = style.get("axis_authority")
    fallback_axes: list[str] = []
    if not isinstance(authority, Mapping) or set(authority) != set(_PARENT_STYLE_AXIS_NAMES):
        reasons.append("axis_authority_inventory_invalid")
    else:
        for axis in _PARENT_STYLE_AXIS_NAMES:
            record = authority.get(axis)
            if not isinstance(record, Mapping):
                reasons.append(f"axis_authority_invalid:{axis}")
                continue
            status = record.get("status")
            if status not in _PARENT_STYLE_AXIS_STATUSES:
                reasons.append(f"axis_authority_status_invalid:{axis}")
            if status == "fallback":
                fallback_axes.append(axis)
            axis_confidence = _finite_number(record.get("confidence"))
            if axis_confidence is None or not 0.0 <= axis_confidence <= 1.0:
                reasons.append(f"axis_authority_confidence_invalid:{axis}")
            if not str(record.get("provenance") or ""):
                reasons.append(f"axis_authority_provenance_missing:{axis}")
            if not isinstance(record.get("reason_codes"), list):
                reasons.append(f"axis_authority_reason_codes_invalid:{axis}")

    fallback_status = style.get("fallback_status")
    if not isinstance(fallback_status, Mapping):
        reasons.append("fallback_status_invalid")
    else:
        expected_fallback_axes = [
            axis for axis in _PARENT_STYLE_AXIS_NAMES if axis in fallback_axes
        ]
        if list(fallback_status.get("axes") or []) != expected_fallback_axes:
            reasons.append("fallback_axis_accounting_mismatch")
        if bool(fallback_status.get("used")) != bool(expected_fallback_axes):
            reasons.append("fallback_used_status_mismatch")
        if not isinstance(fallback_status.get("reason_codes"), list):
            reasons.append("fallback_reason_codes_invalid")
        if role_status == "fallback_registered_role" and not any(
            axis in expected_fallback_axes for axis in ("family", "weight")
        ):
            reasons.append("primary_font_role_fallback_status_unjustified")

    for field_name in ("diagnostic_uncertainty", "readability_diagnostic"):
        diagnostic = style.get(field_name)
        if diagnostic is not None and (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("render_admission") is not False
        ):
            reasons.append(f"{field_name}_invalid")
    effects = style.get("parent_layer_effects")
    if effects is not None and not isinstance(effects, Mapping):
        reasons.append("parent_layer_effects_invalid")

    if reasons:
        return ResolvedRenderStyleValidation(
            status="rejected",
            reason_codes=tuple(dict.fromkeys(reasons)),
        )
    return ResolvedRenderStyleValidation(
        status="accepted",
        style=style,
        reason_codes=("resolved_render_style_contract_accepted",),
    )


def resolved_render_style_contract(value: Any) -> dict[str, Any]:
    """Return an isolated executable style only when the central gate accepts it."""

    validation = validate_resolved_render_style(value)
    return validation.style if validation.accepted else {}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _upgrade_legacy_render_style_v3(
    style: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one stored v3 style into a pixel-preserving CJK v4 view."""

    upgraded = _copy_jsonish(style)
    if not isinstance(upgraded, dict):
        return {}
    preferred = _finite_number(upgraded.get("target_preferred_em_px"))
    if preferred is None or preferred <= 0.0:
        return upgraded
    source_cell = upgraded.get("source_visual_cell")
    source_mode = (
        str(source_cell.get("writing_mode") or "")
        if isinstance(source_cell, Mapping)
        else ""
    ) or str(upgraded.get("writing_mode") or "vertical")
    presentation = {
        "contract_version": "target_presentation_policy_v1",
        "policy_id": "target-presentation:zh-Hans:v1",
        "target_language": "zh-Hans",
        "target_script": "Hani",
        "shaping_locale": "zh-Hans",
        "block_mode_policy": "preserve_source",
        "optical_profile_key": "cjk",
        "measured_fallback_size_policy": (
            "upper_supported_non_decreasing"
        ),
        "automatic_domain_policy": "source_parent",
        "editable_domain_policy": (
            "authorized_speech_container_or_source"
        ),
    }
    source_status = (
        str(source_cell.get("status") or "unavailable")
        if isinstance(source_cell, Mapping)
        else "unavailable"
    )
    preference = {
        "contract_version": "target_size_preference_v1",
        "policy_id": "legacy_v3_preserved",
        "source_scale_status": source_status,
        "central_optical_reference_em_px": preferred,
        "upper_supported_em_px": None,
        "fit_start_em_px": preferred,
        "never_decrease": True,
        "translation_content_consulted": False,
        "fit_output_consulted": False,
        "geometry_consulted": False,
        "render_admission": False,
    }
    audit = upgraded.get("target_em_conversion_audit")
    audit = _copy_jsonish(audit) if isinstance(audit, Mapping) else {}
    audit["target_presentation_policy"] = presentation
    audit["target_size_preference"] = preference
    audit["legacy_contract_migration"] = {
        "status": "pixel_preserving_cjk_projection",
        "source_version": "parent_render_style_v3",
        "target_version": "parent_render_style_v4",
    }
    upgraded.update(
        {
            "render_style_version": "parent_render_style_v4",
            "target_presentation_policy": presentation,
            "target_language": "zh-Hans",
            "target_script": "Hani",
            "shaping_locale": "zh-Hans",
            "source_writing_mode": source_mode,
            "target_optical_reference_em_px": preferred,
            "target_fit_start_em_px": preferred,
            "target_preferred_em_px": preferred,
            "target_em_conversion_audit": audit,
            "target_size_preference": preference,
        }
    )
    return upgraded


def _clear_executable_style_fields(record: dict[str, Any]) -> None:
    for key in (
        "render_style",
        "render_style_owner",
        "render_style_version",
        "render_style_source",
        "render_style_provider",
        "render_style_provider_model",
        "render_style_confidence",
        "style_resolution_status",
        *tuple(_LEGACY_RENDER_STYLE_FLAT_FIELDS.values()),
    ):
        record.pop(key, None)


def _render_style_record_fields(render_style: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(render_style, Mapping) or not render_style:
        return {}
    return {
        "render_style": _copy_jsonish(render_style),
        "render_style_owner": render_style.get("render_style_owner"),
        "render_style_version": render_style.get("render_style_version"),
        "render_style_source": render_style.get("render_style_source"),
        "render_style_provider": render_style.get("render_style_provider"),
        "render_style_provider_model": render_style.get("render_style_provider_model"),
        "render_style_confidence": render_style.get("render_style_confidence"),
    }


def _render_style_flattened_fields(render_style: Mapping[str, Any]) -> dict[str, Any]:
    """Do not project the v3 style back into executable v2 aliases."""

    return {}


def _style_value_present(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _semantic_class_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech_bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "caption_background"
    if lowered == "review":
        return "review"
    return "speech_bubble" if not lowered else lowered


def _route_intent_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "translate_speech"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "translate_caption"
    return "translate"


def _cleanup_mode_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "background_box"
    return "bubble"


def _cleanup_authorization_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "cleanup_translate_speech"
    if lowered in {"caption", "caption_background"}:
        return "cleanup_translate_caption"
    if lowered in {"background", "background_narration"}:
        return "cleanup_translate_background"
    return "cleanup_translate_speech"


def _semantic_kind_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech"
    if lowered in {"caption", "caption_background"}:
        return "caption"
    if lowered in {"background", "background_narration"}:
        return "background_narration"
    if lowered == "review":
        return "unknown"
    return lowered or "speech"


def _container_type_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech_bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "caption_background"
    return "text_area"


def _best_bbox(*candidates: Any) -> list[int]:
    for candidate in candidates:
        bbox = _bbox(candidate)
        if _valid_bbox(bbox):
            return bbox
    return []


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [int(round(float(value[0]))), int(round(float(value[1]))), int(round(float(value[2]))), int(round(float(value[3])))]
    except Exception:
        return []


def _polygon(value: Any) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        return []
    points: list[list[float]] = []
    for point in value:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes, bytearray))
            or len(point) < 2
        ):
            return []
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return []
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        points.append([x, y])
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points if len(points) >= 3 else []


def _valid_bbox(value: Any) -> bool:
    bbox = _bbox(value)
    return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0


def _polygon_from_bbox(bbox: Sequence[int]) -> list[list[int]]:
    box = _bbox(bbox)
    if not _valid_bbox(box):
        return []
    x, y, w, h = box
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _union_region_bboxes(regions: Sequence[Mapping[str, Any]]) -> list[int]:
    boxes = [_bbox(region.get("bbox")) for region in regions or [] if isinstance(region, Mapping)]
    boxes = [box for box in boxes if _valid_bbox(box)]
    if not boxes:
        return []
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _copy_region_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    copied = dict(record)
    render = copied.get("render")
    if isinstance(render, Mapping):
        copied["render"] = dict(render)
    flags = copied.get("flags")
    if isinstance(flags, Mapping):
        copied["flags"] = dict(flags)
    return copied


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_jsonish(item) for item in value]
    return value


def _copy_mapping(value: Any) -> dict[str, Any]:
    copied = _copy_jsonish(value)
    return copied if isinstance(copied, dict) else {}


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
