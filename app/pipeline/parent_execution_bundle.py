# -*- coding: utf-8 -*-
"""Parent-keyed execution contract for post-hierarchy pipeline stages.

This module does not build root/parent topology. It converts the finalized
hierarchy view into the downstream execution unit consumed by translation,
cleanup, render eligibility, and rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


PARENT_EXECUTION_BUNDLE_VERSION = "parent_execution_bundle_v1"
PARENT_RENDER_STYLE_VERSION = "parent_render_style_v1"

_RENDER_STYLE_FLAT_FIELDS = {
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
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    unresolved_reason: str | None = None
    translated_text: str = ""
    source_glyph_mask_ids: list[str] = field(default_factory=list)
    cleanup_job_ids: list[str] = field(default_factory=list)
    cleanup_mask_ids: list[str] = field(default_factory=list)
    render_decision_id: str = ""
    renderer_audit_id: str = ""
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
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "unresolved_reason": self.unresolved_reason,
            "translated_text": self.translated_text,
            "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
            "cleanup_job_ids": list(self.cleanup_job_ids),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "render_decision_id": self.render_decision_id,
            "renderer_audit_id": self.renderer_audit_id,
            "render_style": _copy_jsonish(self.render_style),
            "execution_region": _copy_region_record(self.execution_region) if self.execution_region else self.to_region_record(),
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
        render_style = _render_style_contract_for_bundle(
            self,
            semantic_class=semantic_class,
            semantic_kind=semantic_kind,
            route_intent=route_intent,
            render_allowed_area=render_allowed,
        )
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
            "execution_region_authority": "parent_execution_bundle",
            "execution_region_role": "parent_execution",
            "legacy_region_execution_authority": False,
            "source_region_evidence_only": True,
            "text_area_authorization_source_stage": "parent_execution_bundle",
            "route_intent": route_intent,
            "text_area_route_intent": route_intent,
            "container_type": container_type,
            "text_area_container_type": container_type,
            "cleanup_mode": cleanup_mode,
            "ocr_text": self.source_text,
            "source_text": self.source_text,
            "translation": self.translated_text,
            "translated_text": self.translated_text,
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
                "text_area_container_bbox": list(root_bbox),
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
                "wrap_mode": "vertical",
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
) -> ParentExecutionBundleResult:
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
        )
        result.bundles.append(bundle)

    for parent in getattr(finalized, "blocked_or_unresolved_parents", []) or []:
        bundle = _bundle_from_finalized_parent(
            page_id=result.page_id,
            parent=parent,
            root_by_id=root_by_id,
            parent_by_id=parent_by_id,
            region_by_id=region_by_id,
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
        bundle.render_style = _render_style_contract_from_region(
            bundle.role,
            record,
            source_region_ids=bundle.source_region_ids,
        )
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
            confidence=_float_or_none(record.get("confidence")),
            reason_codes=_list_strings(record.get("reason_codes")),
            unresolved_reason=record.get("unresolved_reason"),
            translated_text=str(record.get("translated_text") or ""),
            source_glyph_mask_ids=_list_strings(record.get("source_glyph_mask_ids")),
            cleanup_job_ids=_list_strings(record.get("cleanup_job_ids")),
            cleanup_mask_ids=_list_strings(record.get("cleanup_mask_ids")),
            render_decision_id=str(record.get("render_decision_id") or ""),
            renderer_audit_id=str(record.get("renderer_audit_id") or ""),
            render_style=_render_style_contract_from_audit_record(record),
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
        text_area_container_id=str(
            primary_region.get("text_area_container_id")
            or primary_render.get("text_area_container_id")
            or ""
        ),
        text_area_container_type=str(
            primary_region.get("text_area_container_type")
            or primary_render.get("text_area_container_type")
            or _container_type_for_role(role)
        ),
        confidence=_float_or_none(getattr(parent_unit, "confidence", None)),
        reason_codes=_list_strings(getattr(parent, "reason_codes", [])),
        unresolved_reason=getattr(parent, "unresolved_reason", None),
        translated_text=translated_text,
        render_style=_render_style_contract_from_source_regions(
            role,
            source_region_ids,
            region_by_id,
            semantic_class=_semantic_class_for_role(role),
            render_allowed_area=render_allowed,
        ),
    )


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
    record["order_index"] = int(bundle.reading_order_index)
    record["reading_order_index"] = int(bundle.reading_order_index)
    record["source_region_ids"] = list(bundle.source_region_ids)
    record["represented_child_ids"] = list(bundle.represented_child_ids)
    record["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    record["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    record["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    record["render_decision_id"] = bundle.render_decision_id
    record["renderer_audit_id"] = bundle.renderer_audit_id
    record["execution_region_authority"] = "parent_execution_bundle"
    record["execution_region_role"] = "parent_execution"
    record["legacy_region_execution_authority"] = False
    record["source_region_evidence_only"] = True
    render_style = _render_style_contract_for_bundle(
        bundle,
        semantic_class=str(record.get("semantic_class") or record.get("type") or ""),
        semantic_kind=str(record.get("semantic_kind") or ""),
        route_intent=str(record.get("route_intent") or record.get("text_area_route_intent") or ""),
        render_allowed_area=_bbox(record.get("render_allowed_area") or record.get("bbox")),
        record=record,
    )
    bundle.render_style = _copy_jsonish(render_style)
    record.update(_render_style_record_fields(render_style))
    render["parent_execution_bundle_id"] = bundle.bundle_id
    render["parent_execution_bundle_version"] = PARENT_EXECUTION_BUNDLE_VERSION
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
    render["order_index"] = int(bundle.reading_order_index)
    render["reading_order_index"] = int(bundle.reading_order_index)
    render["source_region_ids"] = list(bundle.source_region_ids)
    render["represented_child_ids"] = list(bundle.represented_child_ids)
    render["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    render["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    render["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    render["render_decision_id"] = bundle.render_decision_id
    render["renderer_audit_id"] = bundle.renderer_audit_id
    render["execution_region_authority"] = "parent_execution_bundle"
    render["execution_region_role"] = "parent_execution"
    render["legacy_region_execution_authority"] = False
    render["source_region_evidence_only"] = True


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


def _render_style_contract_from_audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    execution_region = record.get("execution_region")
    if isinstance(execution_region, Mapping):
        nested = _render_style_contract_from_region(
            str(record.get("role") or execution_region.get("role") or ""),
            execution_region,
            source_region_ids=_list_strings(record.get("source_region_ids")),
        )
        if nested:
            return nested
    return _render_style_contract_from_region(
        str(record.get("role") or ""),
        record,
        source_region_ids=_list_strings(record.get("source_region_ids")),
    )


def _render_style_contract_from_source_regions(
    role: str,
    source_region_ids: Sequence[str],
    region_by_id: Mapping[str, Mapping[str, Any]],
    *,
    semantic_class: str = "",
    render_allowed_area: Sequence[int] | None = None,
) -> dict[str, Any]:
    for region_id in source_region_ids or []:
        region = region_by_id.get(str(region_id), {})
        style = _render_style_contract_from_region(
            role,
            region,
            source_region_ids=[str(region_id)],
            semantic_class=semantic_class,
            render_allowed_area=render_allowed_area,
        )
        if style and style.get("render_style_source") != "parent_execution_role_default":
            return style
    return _default_render_style_contract(
        role,
        semantic_class=semantic_class,
        render_allowed_area=render_allowed_area,
        source_region_ids=source_region_ids,
    )


def _render_style_contract_for_bundle(
    bundle: ParentExecutionBundle,
    *,
    semantic_class: str,
    semantic_kind: str,
    route_intent: str,
    render_allowed_area: Sequence[int] | None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    existing = _normalize_render_style_contract(
        bundle.render_style,
        role=bundle.role,
        semantic_class=semantic_class,
        semantic_kind=semantic_kind,
        route_intent=route_intent,
        render_allowed_area=render_allowed_area,
        source_region_ids=bundle.source_region_ids,
        source="parent_execution_bundle",
    )
    if existing:
        return existing
    if isinstance(record, Mapping):
        from_record = _render_style_contract_from_region(
            bundle.role,
            record,
            source_region_ids=bundle.source_region_ids,
            semantic_class=semantic_class,
            semantic_kind=semantic_kind,
            route_intent=route_intent,
            render_allowed_area=render_allowed_area,
        )
        if from_record:
            return from_record
    return _default_render_style_contract(
        bundle.role,
        semantic_class=semantic_class,
        semantic_kind=semantic_kind,
        route_intent=route_intent,
        render_allowed_area=render_allowed_area,
        source_region_ids=bundle.source_region_ids,
    )


def _render_style_contract_from_region(
    role: str,
    region: Mapping[str, Any],
    *,
    source_region_ids: Sequence[str] | None = None,
    semantic_class: str = "",
    semantic_kind: str = "",
    route_intent: str = "",
    render_allowed_area: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(region, Mapping):
        return {}
    render = region.get("render")
    if not isinstance(render, Mapping):
        render = {}
    for source in (region.get("render_style"), render.get("render_style")):
        style = _normalize_render_style_contract(
            source,
            role=role,
            semantic_class=semantic_class or str(region.get("semantic_class") or region.get("type") or ""),
            semantic_kind=semantic_kind or str(region.get("semantic_kind") or ""),
            route_intent=route_intent or str(region.get("route_intent") or render.get("route_intent") or ""),
            render_allowed_area=render_allowed_area or _bbox(region.get("render_allowed_area") or region.get("bbox")),
            source_region_ids=source_region_ids or _list_strings(region.get("source_region_ids")),
            source="render_style_contract",
        )
        if style:
            return style

    collected: dict[str, Any] = {}
    reverse = {flat: nested for nested, flat in _RENDER_STYLE_FLAT_FIELDS.items()}
    for source in (render, region):
        for flat_key, nested_key in reverse.items():
            if nested_key in collected:
                continue
            value = source.get(flat_key) if isinstance(source, Mapping) else None
            if _style_value_present(value):
                collected[nested_key] = value
    if "style_class" not in collected:
        font_style = render.get("font_style") or region.get("font_style")
        if _style_value_present(font_style):
            collected["style_class"] = font_style
    if not collected:
        return {}
    collected["render_style_source"] = str(
        render.get("render_style_source")
        or region.get("render_style_source")
        or "legacy_region_render_hints"
    )
    collected["render_style_provider"] = str(
        render.get("render_style_provider")
        or region.get("render_style_provider")
        or collected.get("render_style_provider")
        or "legacy_region_render_hints"
    )
    return _normalize_render_style_contract(
        collected,
        role=role,
        semantic_class=semantic_class or str(region.get("semantic_class") or region.get("type") or ""),
        semantic_kind=semantic_kind or str(region.get("semantic_kind") or ""),
        route_intent=route_intent or str(region.get("route_intent") or render.get("route_intent") or ""),
        render_allowed_area=render_allowed_area or _bbox(region.get("render_allowed_area") or region.get("bbox")),
        source_region_ids=source_region_ids or _list_strings(region.get("source_region_ids") or region.get("region_id")),
        source="legacy_region_render_hints",
    )


def _default_render_style_contract(
    role: str,
    *,
    semantic_class: str = "",
    semantic_kind: str = "",
    route_intent: str = "",
    render_allowed_area: Sequence[int] | None = None,
    source_region_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    lowered = str(role or semantic_kind or semantic_class or route_intent or "").strip().lower()
    caption_like = any(token in lowered for token in ("caption", "background", "narration", "sign"))
    style = {
        "render_style_source": "parent_execution_role_default",
        "render_style_provider": "parent_execution_bundle",
        "style_class": "caption" if caption_like else "dialogue",
        "fill_color": "#000000",
        "stroke_color": "#FFFFFF",
        "stroke_width": 2 if caption_like else 1,
        "source_orientation": "vertical",
        "wrap_mode": "vertical",
        "line_height": 1.1 if caption_like else 1.0,
        "align": "center",
    }
    return _normalize_render_style_contract(
        style,
        role=role,
        semantic_class=semantic_class,
        semantic_kind=semantic_kind,
        route_intent=route_intent,
        render_allowed_area=render_allowed_area,
        source_region_ids=source_region_ids,
        source="parent_execution_role_default",
    )


def _normalize_render_style_contract(
    value: Any,
    *,
    role: str = "",
    semantic_class: str = "",
    semantic_kind: str = "",
    route_intent: str = "",
    render_allowed_area: Sequence[int] | None = None,
    source_region_ids: Sequence[str] | None = None,
    source: str = "",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    style = _copy_jsonish(value)
    if not isinstance(style, dict):
        return {}
    style.setdefault("render_style_version", PARENT_RENDER_STYLE_VERSION)
    style.setdefault("render_style_owner", "parent_execution_bundle")
    if source and not style.get("render_style_source"):
        style["render_style_source"] = source
    provider = (
        style.get("render_style_provider")
        or style.get("provider")
        or style.get("provider_name")
        or style.get("model_provider")
    )
    if provider:
        style["render_style_provider"] = str(provider)
    model = (
        style.get("render_style_provider_model")
        or style.get("provider_model")
        or style.get("model")
        or style.get("model_name")
        or style.get("model_id")
    )
    if model:
        style["render_style_provider_model"] = str(model)
    confidence = style.get("render_style_confidence")
    if confidence is None:
        confidence = style.get("confidence")
    if confidence is not None:
        coerced = _float_or_none(confidence)
        if coerced is not None:
            style["render_style_confidence"] = coerced
    if role and not style.get("source_role"):
        style["source_role"] = str(role)
    if semantic_class and not style.get("semantic_class"):
        style["semantic_class"] = str(semantic_class)
    if semantic_kind and not style.get("semantic_kind"):
        style["semantic_kind"] = str(semantic_kind)
    if route_intent and not style.get("route_intent"):
        style["route_intent"] = str(route_intent)
    lowered = str(role or semantic_kind or semantic_class or route_intent or "").strip().lower()
    caption_like = any(token in lowered for token in ("caption", "background", "narration", "sign"))
    style.setdefault("style_class", "caption" if caption_like else "dialogue")
    style.setdefault("fill_color", "#000000")
    style.setdefault("stroke_color", "#FFFFFF")
    style.setdefault("stroke_width", 2 if caption_like else 1)
    style.setdefault("source_orientation", "vertical")
    style.setdefault("wrap_mode", "vertical")
    style.setdefault("line_height", 1.1 if caption_like else 1.0)
    style.setdefault("align", "center")
    bbox = _bbox(render_allowed_area)
    if bbox and not style.get("render_allowed_area"):
        style["render_allowed_area"] = bbox
    ids = _list_strings(source_region_ids)
    if ids and not style.get("source_region_ids"):
        style["source_region_ids"] = ids
    if style.get("stroke_width") is not None:
        try:
            style["stroke_width"] = max(0, int(style.get("stroke_width") or 0))
        except Exception:
            style.pop("stroke_width", None)
    for key in ("font_size", "font_size_hint", "font_size_min", "font_size_max"):
        if style.get(key) is not None:
            try:
                style[key] = max(0, int(style.get(key) or 0))
            except Exception:
                style.pop(key, None)
    if "font_size_locked" in style:
        style["font_size_locked"] = bool(style.get("font_size_locked"))
    return style


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
    if not isinstance(render_style, Mapping):
        return {}
    fields: dict[str, Any] = {}
    for nested_key, flat_key in _RENDER_STYLE_FLAT_FIELDS.items():
        value = render_style.get(nested_key)
        if _style_value_present(value):
            fields[flat_key] = value
    return fields


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
