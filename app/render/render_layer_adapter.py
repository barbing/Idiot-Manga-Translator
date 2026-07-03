# -*- coding: utf-8 -*-
"""Adapters from parent execution bundles to render-layer plans."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.pipeline.parent_execution_bundle import PARENT_EXECUTION_BUNDLE_VERSION
from app.render.typesetting_contracts import (
    RENDER_LAYER_PLAN_VERSION,
    RenderLayerPlan,
    bbox_from_value,
    copy_jsonish,
    render_layer_plans_to_audit_dict,
)


RENDER_LAYER_ADAPTER_VERSION = "render_layer_adapter_v1"


class RenderLayerContractError(ValueError):
    """Raised when a render-required parent cannot form a layer identity."""


def build_render_layer_plans_from_parent_bundles(
    *,
    page_id: str,
    parent_execution_bundles: Sequence[Any],
    cleaned_page_base: Mapping[str, Any] | None = None,
) -> list[RenderLayerPlan]:
    """Create one render-layer plan for each render-required parent bundle.

    Non-render-required bundles are not render layers. Render-required bundles
    are never merged, split, or suppressed by this adapter.
    """

    plans: list[RenderLayerPlan] = []
    for index, bundle in enumerate(parent_execution_bundles or []):
        if not _bool_field(bundle, "render_required"):
            continue
        plans.append(
            render_layer_plan_from_parent_bundle(
                bundle,
                page_id=page_id,
                cleaned_page_base=cleaned_page_base,
                draw_order=index,
            )
        )
    return plans


def render_layer_plan_from_parent_bundle(
    bundle: Any,
    *,
    page_id: str = "",
    cleaned_page_base: Mapping[str, Any] | None = None,
    draw_order: int | None = None,
) -> RenderLayerPlan:
    page_id_value = str(page_id or _field(bundle, "page_id") or "")
    bundle_id = _identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id")
    parent_id = _identity_field(
        bundle,
        "parent_id",
        "parent_logical_text_unit_id",
        "logical_text_block_id",
    )
    root_id = _identity_field(bundle, "root_id", "text_block_root_id")
    missing = [
        name
        for name, value in (
            ("page_id", page_id_value),
            ("bundle_id", bundle_id),
            ("parent_id", parent_id),
            ("root_id", root_id),
        )
        if not value
    ]
    if missing:
        raise RenderLayerContractError(
            "render_required_parent_missing_identity:" + ",".join(missing)
        )

    execution_region = _mapping_field(bundle, "execution_region")
    render_record = _nested_mapping(execution_region, "render")
    render_style = _style_from_bundle(bundle, execution_region, render_record)
    target_box = _target_box_from_bundle(bundle, execution_region, render_record)
    hard_bounds = _hard_bounds_from_bundle(bundle, execution_region, render_record, target_box)
    translated_text = _translated_text_from_bundle(bundle, execution_region, render_record)
    reading_order_index = _int_field(bundle, "reading_order_index", None)
    if reading_order_index is None:
        reading_order_index = _int_field(bundle, "order_index", None)
    plan_draw_order = int(draw_order if draw_order is not None else (reading_order_index or 0))
    contract_issues: list[str] = []
    if not target_box:
        contract_issues.append("missing_target_box")
    if not translated_text:
        contract_issues.append("missing_translated_text")

    return RenderLayerPlan(
        page_id=page_id_value,
        layer_id=_layer_id(page_id_value, bundle_id),
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        translated_text=translated_text,
        source_text_summary=_source_text_summary(str(_field(bundle, "source_text") or "")),
        source_provenance_ref=_source_provenance_ref(bundle),
        target_box=target_box,
        hard_bounds=hard_bounds,
        clipping_region_ref=_clipping_region_ref(bundle, execution_region, render_record),
        resolved_render_style=render_style,
        writing_mode=_writing_mode(render_style, execution_region, render_record),
        draw_order=plan_draw_order,
        editable=True,
        editability_flags=["text", "resolved_render_style"],
        cleaned_page_base_ref=_cleaned_page_base_ref(cleaned_page_base),
        parent_execution_bundle_ref=_parent_execution_bundle_ref(bundle),
        legacy_region_ref=_legacy_region_ref(execution_region, render_record),
        role=str(_field(bundle, "role") or execution_region.get("type") or ""),
        state=str(_field(bundle, "state") or execution_region.get("parent_execution_state") or ""),
        render_required=True,
        metadata={
            "render_layer_adapter_version": RENDER_LAYER_ADAPTER_VERSION,
            "render_layer_plan_version": RENDER_LAYER_PLAN_VERSION,
            "contract_issues": contract_issues,
        },
    )


def render_layer_plan_audit(plans: Sequence[RenderLayerPlan]) -> dict[str, Any]:
    records = render_layer_plans_to_audit_dict(plans)
    return {
        "render_layer_adapter_version": RENDER_LAYER_ADAPTER_VERSION,
        "render_layer_plan_version": RENDER_LAYER_PLAN_VERSION,
        "layer_count": len(records),
        "layer_ids": [str(record.get("layer_id") or "") for record in records],
        "plans": records,
    }


def _field(source: Any, key: str, default: Any = "") -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _bool_field(source: Any, key: str) -> bool:
    return bool(_field(source, key, False))


def _int_field(source: Any, key: str, default: int | None = 0) -> int | None:
    value = _field(source, key, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _identity_field(source: Any, *keys: str) -> str:
    for key in keys:
        value = str(_field(source, key) or "").strip()
        if value:
            return value
    return ""


def _mapping_field(source: Any, key: str) -> dict[str, Any]:
    value = _field(source, key, {})
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {}


def _nested_mapping(source: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key) if isinstance(source, Mapping) else {}
    if isinstance(value, Mapping):
        return copy_jsonish(value)
    return {}


def _layer_id(page_id: str, bundle_id: str) -> str:
    return f"rlayer_{_safe_id(page_id)}_{_safe_id(bundle_id)}"


def _safe_id(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def _style_from_bundle(
    bundle: Any,
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> dict[str, Any]:
    for value in (
        _field(bundle, "render_style", {}),
        execution_region.get("render_style"),
        render_record.get("render_style"),
    ):
        if isinstance(value, Mapping) and value:
            return copy_jsonish(value)
    return {}


def _target_box_from_bundle(
    bundle: Any,
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> list[int]:
    for value in (
        _field(bundle, "render_allowed_area", []),
        render_record.get("parent_logical_text_unit_render_allowed_area"),
        render_record.get("render_allowed_area"),
        execution_region.get("parent_logical_text_unit_render_allowed_area"),
        execution_region.get("logical_text_block_allowed_bbox"),
        execution_region.get("bbox"),
        _field(bundle, "parent_bbox", []),
    ):
        bbox = bbox_from_value(value)
        if bbox:
            return bbox
    return []


def _hard_bounds_from_bundle(
    bundle: Any,
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
    target_box: Sequence[int],
) -> list[int]:
    for value in (
        _field(bundle, "render_allowed_area", []),
        render_record.get("render_allowed_area"),
        execution_region.get("logical_text_block_allowed_bbox"),
        target_box,
    ):
        bbox = bbox_from_value(value)
        if bbox:
            return bbox
    return []


def _translated_text_from_bundle(
    bundle: Any,
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> str:
    for value in (
        _field(bundle, "translated_text", ""),
        render_record.get("translated_text"),
        render_record.get("translation"),
        execution_region.get("translated_text"),
        execution_region.get("translation"),
    ):
        text = str(value or "")
        if text:
            return text
    return ""


def _writing_mode(
    render_style: Mapping[str, Any],
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> str:
    for value in (
        render_style.get("writing_mode"),
        render_style.get("source_orientation"),
        render_style.get("wrap_mode"),
        render_record.get("writing_mode"),
        render_record.get("source_orientation"),
        render_record.get("wrap_mode"),
        execution_region.get("writing_mode"),
        execution_region.get("source_orientation"),
        execution_region.get("wrap_mode"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "auto"


def _source_text_summary(source_text: str) -> str:
    text = " ".join(str(source_text or "").split())
    if len(text) <= 80:
        return text
    return text[:77] + "..."


def _source_provenance_ref(bundle: Any) -> dict[str, Any]:
    return {
        "source_contract_owner": str(_field(bundle, "source_contract_owner") or ""),
        "source_contract_region_id": str(_field(bundle, "source_contract_region_id") or ""),
        "source_contract_bbox": bbox_from_value(_field(bundle, "source_contract_bbox", [])),
        "source_contract_scope": str(_field(bundle, "source_contract_scope") or ""),
        "source_contract_stage": str(_field(bundle, "source_contract_stage") or ""),
        "source_contract_ocr_confidence": _field(bundle, "source_contract_ocr_confidence", None),
        "source_quality_state": str(_field(bundle, "source_quality_state") or ""),
        "source_quality_action": str(_field(bundle, "source_quality_action") or ""),
        "source_quality_reason_codes": copy_jsonish(_field(bundle, "source_quality_reason_codes", [])),
        "ocr_backend": str(_field(bundle, "ocr_backend") or ""),
        "ocr_prompt_version": str(_field(bundle, "ocr_prompt_version") or ""),
        "source_region_ids": copy_jsonish(_field(bundle, "source_region_ids", [])),
        "represented_child_ids": copy_jsonish(_field(bundle, "represented_child_ids", [])),
    }


def _clipping_region_ref(
    bundle: Any,
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cleanup_target_bbox": bbox_from_value(_field(bundle, "cleanup_target_bbox", []))
        or bbox_from_value(render_record.get("parent_logical_text_unit_cleanup_target_bbox"))
        or bbox_from_value(execution_region.get("parent_logical_text_unit_cleanup_target_bbox")),
        "root_bbox": bbox_from_value(_field(bundle, "root_bbox", []))
        or bbox_from_value(execution_region.get("text_area_container_bbox")),
        "render_allowed_area": bbox_from_value(_field(bundle, "render_allowed_area", []))
        or bbox_from_value(render_record.get("render_allowed_area"))
        or bbox_from_value(execution_region.get("logical_text_block_allowed_bbox")),
    }


def _cleaned_page_base_ref(cleaned_page_base: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cleaned_page_base, Mapping) or not cleaned_page_base:
        return {}
    keys = (
        "cleaned_page_base_version",
        "page_id",
        "state",
        "valid",
        "image_path",
        "cache_path",
        "source_image_path",
        "source_sha256",
        "cleaned_page_base_sha256",
        "parent_execution_signature",
        "cleanup_identity_signature",
    )
    return {key: copy_jsonish(cleaned_page_base.get(key)) for key in keys if key in cleaned_page_base}


def _parent_execution_bundle_ref(bundle: Any) -> dict[str, Any]:
    return {
        "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
        "bundle_id": _identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id"),
        "parent_id": _identity_field(bundle, "parent_id", "parent_logical_text_unit_id"),
        "root_id": _identity_field(bundle, "root_id", "text_block_root_id"),
        "graph_parent_id": str(_field(bundle, "graph_parent_id") or ""),
        "reading_order_index": _int_field(bundle, "reading_order_index", 0),
        "state": str(_field(bundle, "state") or ""),
        "role": str(_field(bundle, "role") or ""),
        "render_required": bool(_field(bundle, "render_required", False)),
        "source_quality_state": str(_field(bundle, "source_quality_state") or ""),
        "source_quality_action": str(_field(bundle, "source_quality_action") or ""),
        "render_decision_id": str(_field(bundle, "render_decision_id") or ""),
        "renderer_audit_id": str(_field(bundle, "renderer_audit_id") or ""),
    }


def _legacy_region_ref(
    execution_region: Mapping[str, Any],
    render_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(execution_region, Mapping) or not execution_region:
        return {}
    keys = (
        "region_id",
        "bbox",
        "parent_execution_bundle_id",
        "execution_region_authority",
        "execution_region_role",
        "legacy_region_execution_authority",
        "source_region_evidence_only",
        "semantic_class",
        "semantic_kind",
        "route_intent",
        "cleanup_mode",
    )
    ref = {key: copy_jsonish(execution_region.get(key)) for key in keys if key in execution_region}
    if render_record:
        ref["render"] = {
            key: copy_jsonish(render_record.get(key))
            for key in (
                "parent_execution_bundle_id",
                "render_allowed_area",
                "parent_logical_text_unit_render_allowed_area",
                "execution_region_authority",
                "legacy_region_execution_authority",
                "source_region_evidence_only",
            )
            if key in render_record
        }
    return ref
