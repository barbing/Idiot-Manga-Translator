# -*- coding: utf-8 -*-
"""Adapters from parent execution bundles to render-layer plans."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from app.pipeline.parent_execution_bundle import (
    PARENT_EXECUTION_BUNDLE_VERSION,
    validate_resolved_render_style,
)
from app.pipeline.oriented_layout import ORIENTED_LAYOUT_MAX_POLYGON_VERTICES
from app.pipeline.parent_style_evidence import (
    AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
    SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY,
    SOURCE_TEXT_FOOTPRINT_VERSION,
)
from app.render.source_punctuation_hints import (
    SOURCE_PUNCTUATION_CELL_CALIBRATION_VERSION,
    SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION,
    SOURCE_PUNCTUATION_GEOMETRY_OBSERVER_VERSION,
    SOURCE_PUNCTUATION_GEOMETRY_SUPPORTED_KINDS,
    SOURCE_PUNCTUATION_GEOMETRY_SUPPORT_VERSION,
    SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE,
    SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED,
    source_punctuation_cell_calibration_sha256,
    source_punctuation_geometry_fact_set_id,
)
from app.render.typesetting_contracts import (
    RENDER_LAYER_PLAN_VERSION,
    RenderLayerPlan,
    bbox_from_value,
    copy_jsonish,
    render_layer_plans_to_audit_dict,
)


RENDER_LAYER_ADAPTER_VERSION = "render_layer_adapter_v2"


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

    bundles = list(parent_execution_bundles or [])
    plans: list[RenderLayerPlan] = []
    for bundle in bundles:
        if not _bool_field(bundle, "render_required"):
            continue
        plans.append(
            render_layer_plan_from_parent_bundle(
                bundle,
                page_id=page_id,
                cleaned_page_base=cleaned_page_base,
            )
        )
    return plans


def render_layer_plan_from_parent_bundle(
    bundle: Any,
    *,
    page_id: str = "",
    cleaned_page_base: Mapping[str, Any] | None = None,
) -> RenderLayerPlan:
    supplied_page_id = str(page_id or "").strip()
    bundle_page_id = str(_field(bundle, "page_id") or "").strip()
    if supplied_page_id and supplied_page_id != bundle_page_id:
        raise RenderLayerContractError(
            "render_required_parent_page_id_mismatch:"
            f"supplied={supplied_page_id},bundle={bundle_page_id or '<missing>'}"
        )
    page_id_value = bundle_page_id
    bundle_id = str(_field(bundle, "bundle_id") or "").strip()
    parent_id = str(_field(bundle, "parent_id") or "").strip()
    root_id = str(_field(bundle, "root_id") or "").strip()
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
    _validate_oriented_geometry_budget(bundle)

    style_validation = validate_resolved_render_style(
        _field(bundle, "render_style", {})
    )
    if not style_validation.accepted:
        raise RenderLayerContractError(
            "render_required_parent_missing_arbitrator_resolved_style:"
            f"{bundle_id}:"
            + ",".join(style_validation.reason_codes)
        )
    render_style = style_validation.style
    target_box, target_box_source = _target_box_from_bundle(bundle)
    hard_bounds = _hard_bounds_from_bundle(bundle, target_box)
    render_domain = (
        copy_jsonish(_field(bundle, "render_layout_domain", {}))
        if isinstance(_field(bundle, "render_layout_domain", {}), Mapping)
        else {}
    )
    oriented_layout_frame = (
        copy_jsonish(_field(bundle, "text_area_oriented_frame", {}))
        if isinstance(_field(bundle, "text_area_oriented_frame", {}), Mapping)
        else {}
    )
    editable_bounds = bbox_from_value(
        render_domain.get("editable_bounds")
        if isinstance(render_domain, Mapping)
        else []
    )
    translated_text = str(_field(bundle, "translated_text") or "")
    reading_order_index = _int_field(bundle, "reading_order_index", None)
    plan_draw_order = int(reading_order_index or 0)
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
        clipping_region_ref=_clipping_region_ref(bundle),
        resolved_render_style=render_style,
        writing_mode=_writing_mode(render_style),
        draw_order=plan_draw_order,
        editable=True,
        editability_flags=["text", "resolved_render_style"],
        cleaned_page_base_ref=_cleaned_page_base_ref(cleaned_page_base),
        parent_execution_bundle_ref=_parent_execution_bundle_ref(bundle),
        role=str(_field(bundle, "role") or ""),
        state=str(_field(bundle, "state") or ""),
        render_required=True,
        metadata={
            "render_layer_adapter_version": RENDER_LAYER_ADAPTER_VERSION,
            "render_layer_plan_version": RENDER_LAYER_PLAN_VERSION,
            "target_box_source": target_box_source,
            "target_presentation_policy_id": str(
                render_domain.get("policy_id") or ""
            ),
            "parent_render_domain": render_domain,
            "oriented_layout_frame": oriented_layout_frame,
            "editable_hard_bounds": editable_bounds or list(hard_bounds),
            "contract_issues": contract_issues,
            "resolved_render_style_validation": (
                style_validation.to_audit_dict()
            ),
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


def _validate_oriented_geometry_budget(bundle: Any) -> None:
    domain = _field(bundle, "render_layout_domain", {})
    domain = domain if isinstance(domain, Mapping) else {}
    domain_frame = domain.get("oriented_frame")
    domain_frame = domain_frame if isinstance(domain_frame, Mapping) else {}
    frame = _field(bundle, "text_area_oriented_frame", {})
    frame = frame if isinstance(frame, Mapping) else {}
    for polygon in (
        _field(bundle, "text_area_container_polygon", []),
        domain.get("container_polygon"),
        frame.get("polygon"),
        domain_frame.get("polygon"),
    ):
        if not isinstance(polygon, Sequence) or isinstance(
            polygon,
            (str, bytes, bytearray),
        ):
            continue
        if len(polygon) > ORIENTED_LAYOUT_MAX_POLYGON_VERTICES:
            raise RenderLayerContractError(
                "oriented_polygon_vertex_budget_exceeded"
            )
        for point in polygon:
            if (
                isinstance(point, Sequence)
                and not isinstance(point, (str, bytes, bytearray))
                and len(point) > 2
            ):
                raise RenderLayerContractError(
                    "oriented_polygon_coordinate_budget_exceeded"
                )


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


def _layer_id(page_id: str, bundle_id: str) -> str:
    return f"rlayer_{_safe_id(page_id)}_{_safe_id(bundle_id)}"


def _safe_id(value: str) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def _target_box_from_bundle(
    bundle: Any,
) -> tuple[list[int], str]:
    candidates = (
        ("bundle.render_allowed_area", _field(bundle, "render_allowed_area", [])),
        ("bundle.parent_bbox", _field(bundle, "parent_bbox", [])),
    )
    for source, value in candidates:
        bbox = bbox_from_value(value)
        if bbox:
            return bbox, source
    return [], "missing_parent_contract_box"


def _hard_bounds_from_bundle(
    bundle: Any,
    target_box: Sequence[int],
) -> list[int]:
    domain = _field(bundle, "render_layout_domain", {})
    if isinstance(domain, Mapping):
        automatic = bbox_from_value(domain.get("automatic_bounds"))
        if automatic:
            return automatic
    for value in (
        _field(bundle, "render_allowed_area", []),
        _field(bundle, "parent_bbox", []),
        target_box,
    ):
        bbox = bbox_from_value(value)
        if bbox:
            return bbox
    return []


def _writing_mode(render_style: Mapping[str, Any]) -> str:
    for value in (
        render_style.get("writing_mode"),
        render_style.get("source_orientation"),
        render_style.get("wrap_mode"),
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
    source_bbox = bbox_from_value(_field(bundle, "source_contract_bbox", []))
    record = {
        "source_contract_owner": str(_field(bundle, "source_contract_owner") or ""),
        "source_contract_region_id": str(_field(bundle, "source_contract_region_id") or ""),
        "source_contract_bbox": source_bbox,
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
    footprint, validation = _validated_source_text_footprint(bundle)
    record["source_text_footprint_validation"] = validation
    if footprint:
        record["source_text_footprint"] = footprint
    punctuation, punctuation_validation = (
        _validated_source_punctuation_geometry(bundle)
    )
    record["source_punctuation_geometry_validation"] = (
        punctuation_validation
    )
    if punctuation:
        record["source_punctuation_geometry"] = punctuation
    return record


def _validated_source_punctuation_geometry(
    bundle: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the only executable source-punctuation geometry carrier."""

    raw = _field(bundle, "source_punctuation_geometry", {})
    if raw in ({}, None):
        return {}, _punctuation_geometry_validation_record(
            "unavailable", ("source_punctuation_geometry_missing",)
        )
    if not isinstance(raw, Mapping):
        return {}, _punctuation_geometry_validation_record(
            "rejected", ("source_punctuation_geometry_not_mapping",)
        )
    evidence = copy_jsonish(raw)
    reasons = _source_punctuation_geometry_rejection_reasons(
        bundle=bundle,
        evidence=evidence,
    )
    if reasons:
        return {}, _punctuation_geometry_validation_record(
            "rejected", reasons
        )
    fact_set_id = str(evidence.get("fact_set_id") or "")
    if str(evidence.get("status") or "") != "observed":
        return {}, _punctuation_geometry_validation_record(
            "unavailable",
            (
                "source_punctuation_geometry_observer_abstained",
                *tuple(str(value) for value in evidence.get("reason_codes") or []),
            ),
            fact_set_id=fact_set_id,
        )
    return evidence, _punctuation_geometry_validation_record(
        "accepted",
        ("source_punctuation_geometry_identity_and_fact_set_validated",),
        fact_set_id=fact_set_id,
    )


def _punctuation_geometry_validation_record(
    status: str,
    reason_codes: Sequence[str],
    *,
    fact_set_id: str = "",
) -> dict[str, Any]:
    return {
        "status": str(status or "unavailable"),
        "contract_version": SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION,
        "reason_codes": [str(value) for value in reason_codes if str(value)],
        "fact_set_id": str(fact_set_id or ""),
    }


def _source_punctuation_geometry_rejection_reasons(
    *,
    bundle: Any,
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    _validate_exact_fields(
        evidence,
        {
            "contract_version",
            "observer_version",
            "source_identity",
            "status",
            "view_id",
            "support_identity",
            "occurrences",
            "abstention_reason",
            "reason_codes",
            "text_identity_authority",
            "geometry_authority",
            "render_admission_authority",
            "fact_set_id",
        },
        reasons,
        "source_punctuation_geometry",
    )
    if str(evidence.get("contract_version") or "") != (
        SOURCE_PUNCTUATION_GEOMETRY_EVIDENCE_VERSION
    ):
        reasons.append("source_punctuation_geometry_version_invalid")
    if str(evidence.get("observer_version") or "") != (
        SOURCE_PUNCTUATION_GEOMETRY_OBSERVER_VERSION
    ):
        reasons.append("source_punctuation_geometry_observer_version_invalid")
    status = str(evidence.get("status") or "")
    if status not in {"observed", "abstained", "unavailable"}:
        reasons.append("source_punctuation_geometry_status_invalid")
    if str(evidence.get("text_identity_authority") or "") != (
        "translated_lossless_tokens"
    ):
        reasons.append("source_punctuation_geometry_text_authority_invalid")
    if str(evidence.get("geometry_authority") or "") != (
        "parent_authorized_source_pixels"
    ):
        reasons.append("source_punctuation_geometry_pixel_authority_invalid")
    if evidence.get("render_admission_authority") is not False:
        reasons.append("source_punctuation_geometry_render_admission_invalid")
    reason_codes = evidence.get("reason_codes")
    if not isinstance(reason_codes, list) or any(
        not isinstance(value, str) for value in reason_codes
    ):
        reasons.append("source_punctuation_geometry_reason_codes_invalid")

    identity = evidence.get("source_identity")
    if not isinstance(identity, Mapping):
        reasons.append("source_punctuation_geometry_identity_not_mapping")
        identity = {}
    else:
        _validate_exact_fields(
            identity,
            {"page_id", "bundle_id", "parent_id", "root_id"},
            reasons,
            "source_punctuation_geometry_identity",
        )
    expected_identity = {
        "page_id": str(_field(bundle, "page_id") or ""),
        "bundle_id": str(_field(bundle, "bundle_id") or ""),
        "parent_id": str(_field(bundle, "parent_id") or ""),
        "root_id": str(_field(bundle, "root_id") or ""),
    }
    for key, expected in expected_identity.items():
        if str(identity.get(key) or "") != expected:
            reasons.append(f"source_punctuation_geometry_identity_mismatch:{key}")

    support = evidence.get("support_identity")
    if not isinstance(support, Mapping):
        reasons.append("source_punctuation_geometry_support_not_mapping")
        support = {}
    required_support = {
        "contract_version",
        "authorized_source_style_view_version",
        "page_id",
        "bundle_id",
        "parent_id",
        "root_id",
        "view_id",
        "cleanup_mask_ids",
        "owned_component_ids",
        "pixel_projection_owner",
        "geometry_observer_owner",
        "source_cell_calibration",
    }
    observed_support = {
        "analysis_bbox_page_xywh",
        "authorized_foreground_mask_sha256",
        "source_pixel_crop_sha256",
        "contrast_ink_sha256",
    }
    expected_support = required_support | (observed_support if status in {"observed", "abstained"} else set())
    if isinstance(support, Mapping):
        _validate_exact_fields(
            support,
            expected_support,
            reasons,
            "source_punctuation_geometry_support",
        )
    if str(support.get("contract_version") or "") != (
        SOURCE_PUNCTUATION_GEOMETRY_SUPPORT_VERSION
    ):
        reasons.append("source_punctuation_geometry_support_version_invalid")
    if str(support.get("authorized_source_style_view_version") or "") != (
        AUTHORIZED_SOURCE_STYLE_VIEW_VERSION
    ):
        reasons.append("source_punctuation_geometry_style_view_version_invalid")
    for key, expected in expected_identity.items():
        if str(support.get(key) or "") != expected:
            reasons.append(f"source_punctuation_geometry_support_identity_mismatch:{key}")
    if str(support.get("view_id") or "") != str(evidence.get("view_id") or ""):
        reasons.append("source_punctuation_geometry_view_identity_mismatch")
    if str(support.get("pixel_projection_owner") or "") != "CleanupMask":
        reasons.append("source_punctuation_geometry_pixel_owner_invalid")
    if str(support.get("geometry_observer_owner") or "") != (
        "SourcePunctuationGeometryEvidence"
    ):
        reasons.append("source_punctuation_geometry_observer_owner_invalid")
    for key in ("cleanup_mask_ids", "owned_component_ids"):
        values = support.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            reasons.append(f"source_punctuation_geometry_support_{key}_invalid")
    calibration_cells = _validate_source_punctuation_cell_calibration(
        support.get("source_cell_calibration"),
        expected_identity=expected_identity,
        support=support,
        reasons=reasons,
    )
    analysis_bbox = (
        _strict_bbox(support.get("analysis_bbox_page_xywh"))
        if status in {"observed", "abstained"}
        else []
    )
    if status in {"observed", "abstained"} and not analysis_bbox:
        reasons.append("source_punctuation_geometry_analysis_bbox_invalid")
    for key in (
        "authorized_foreground_mask_sha256",
        "source_pixel_crop_sha256",
        "contrast_ink_sha256",
    ):
        if status in {"observed", "abstained"} and not _is_sha256(support.get(key)):
            reasons.append(f"source_punctuation_geometry_support_{key}_invalid")

    occurrences = evidence.get("occurrences")
    if not isinstance(occurrences, list):
        reasons.append("source_punctuation_geometry_occurrences_not_list")
        occurrences = []
    if status == "observed" and not occurrences:
        reasons.append("source_punctuation_geometry_observed_without_occurrences")
    if status != "observed" and occurrences:
        reasons.append("source_punctuation_geometry_nonobserved_with_occurrences")
    kind_ordinals: dict[str, list[int]] = {}
    visual_ordinals: list[int] = []
    for index, occurrence in enumerate(occurrences):
        if not isinstance(occurrence, Mapping):
            reasons.append(f"source_punctuation_geometry_occurrence_not_mapping:{index}")
            continue
        _validate_source_punctuation_occurrence(
            occurrence,
            index=index,
            analysis_bbox=analysis_bbox,
            reasons=reasons,
        )
        inline_axis = str(occurrence.get("inline_axis") or "")
        measurement_basis = str(
            occurrence.get("measurement_basis") or ""
        )
        occurrence_cell = _strict_finite_number(
            occurrence.get("source_cell_px")
        )
        calibration_cell = calibration_cells.get(inline_axis)
        if measurement_basis == SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED:
            calibration_matches = (
                occurrence_cell is not None
                and calibration_cell is not None
                and abs(occurrence_cell - calibration_cell) <= 1.0e-6
            )
        elif measurement_basis == (
            SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE
        ):
            calibration_matches = (
                occurrence_cell == 0.0 and calibration_cell is None
            )
        else:
            calibration_matches = False
        if not calibration_matches:
            reasons.append(
                f"source_punctuation_geometry_occurrence_calibration_mismatch:{index}"
            )
        kind = str(occurrence.get("kind") or "")
        kind_ordinal = _strict_nonnegative_int(occurrence.get("kind_ordinal"))
        visual_ordinal = _strict_nonnegative_int(
            occurrence.get("visual_reading_order_ordinal")
        )
        if kind_ordinal is not None:
            kind_ordinals.setdefault(kind, []).append(kind_ordinal)
        if visual_ordinal is not None:
            visual_ordinals.append(visual_ordinal)
    if visual_ordinals and sorted(visual_ordinals) != list(range(len(visual_ordinals))):
        reasons.append("source_punctuation_geometry_visual_ordinals_invalid")
    for kind, ordinals in kind_ordinals.items():
        if sorted(ordinals) != list(range(len(ordinals))):
            reasons.append(f"source_punctuation_geometry_kind_ordinals_invalid:{kind}")

    fact_set_id = str(evidence.get("fact_set_id") or "")
    expected_fact_set_id = source_punctuation_geometry_fact_set_id(evidence)
    if fact_set_id != expected_fact_set_id:
        reasons.append("source_punctuation_geometry_fact_set_mismatch")
    return tuple(dict.fromkeys(str(value) for value in reasons if str(value)))


def _validate_source_punctuation_cell_calibration(
    value: Any,
    *,
    expected_identity: Mapping[str, str],
    support: Mapping[str, Any],
    reasons: list[str],
) -> dict[str, float]:
    prefix = "source_punctuation_geometry_cell_calibration"
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix}_not_mapping")
        return {}
    _validate_exact_fields(
        value,
        {
            "contract_version",
            "status",
            "style_evidence_identity",
            "source_scale_axis_sha256",
            "axes",
            "reason_codes",
            "calibration_sha256",
        },
        reasons,
        prefix,
    )
    if str(value.get("contract_version") or "") != (
        SOURCE_PUNCTUATION_CELL_CALIBRATION_VERSION
    ):
        reasons.append(f"{prefix}_version_invalid")
    status = str(value.get("status") or "")
    if status not in {"supported", "unavailable"}:
        reasons.append(f"{prefix}_status_invalid")
    reason_codes = value.get("reason_codes")
    if not isinstance(reason_codes, list) or any(
        not isinstance(item, str) for item in reason_codes
    ):
        reasons.append(f"{prefix}_reason_codes_invalid")

    style_identity = value.get("style_evidence_identity")
    if not isinstance(style_identity, Mapping):
        reasons.append(f"{prefix}_style_identity_not_mapping")
        style_identity = {}
    else:
        _validate_exact_fields(
            style_identity,
            {
                "page_id",
                "bundle_id",
                "parent_id",
                "root_id",
                "view_id",
                "cleanup_mask_ids",
                "owned_component_ids",
                "content_bbox_xywh",
                "analysis_bbox_xywh",
                "style_evidence_status",
            },
            reasons,
            f"{prefix}_style_identity",
        )
    if status == "supported":
        for key, expected in expected_identity.items():
            if str(style_identity.get(key) or "") != expected:
                reasons.append(f"{prefix}_style_identity_mismatch:{key}")
        if str(style_identity.get("view_id") or "") != str(
            support.get("view_id") or ""
        ):
            reasons.append(f"{prefix}_style_view_identity_mismatch")
        if str(style_identity.get("style_evidence_status") or "") != "observed":
            reasons.append(f"{prefix}_style_status_invalid")
        for key in ("cleanup_mask_ids", "owned_component_ids"):
            if style_identity.get(key) != support.get(key):
                reasons.append(f"{prefix}_style_identity_mismatch:{key}")
        if not _strict_bbox(style_identity.get("content_bbox_xywh")):
            reasons.append(f"{prefix}_content_bbox_invalid")
        if _strict_bbox(style_identity.get("analysis_bbox_xywh")) != (
            _strict_bbox(support.get("analysis_bbox_page_xywh"))
        ):
            reasons.append(f"{prefix}_analysis_bbox_mismatch")
        if not _is_sha256(value.get("source_scale_axis_sha256")):
            reasons.append(f"{prefix}_scale_axis_hash_invalid")

    axes = value.get("axes")
    cells: dict[str, float] = {}
    if not isinstance(axes, Mapping):
        reasons.append(f"{prefix}_axes_not_mapping")
        axes = {}
    else:
        _validate_exact_fields(
            axes,
            {"ttb", "ltr"},
            reasons,
            f"{prefix}_axes",
        )
    for inline_axis, direction in (("ttb", "vertical"), ("ltr", "horizontal")):
        axis = axes.get(inline_axis)
        axis_prefix = f"{prefix}_axis:{inline_axis}"
        if not isinstance(axis, Mapping):
            reasons.append(f"{axis_prefix}_not_mapping")
            continue
        _validate_exact_fields(
            axis,
            {
                "source_axis",
                "source_direction",
                "status",
                "source_cell_px",
                "confidence",
                "support_status",
                "reason",
            },
            reasons,
            axis_prefix,
        )
        if str(axis.get("source_axis") or "") != "scale":
            reasons.append(f"{axis_prefix}_source_axis_invalid")
        if str(axis.get("source_direction") or "") != direction:
            reasons.append(f"{axis_prefix}_source_direction_invalid")
        axis_status = str(axis.get("status") or "")
        cell = _strict_finite_number(axis.get("source_cell_px"))
        confidence = _strict_finite_number(axis.get("confidence"))
        if axis_status == "supported":
            if cell is None or cell <= 0.0:
                reasons.append(f"{axis_prefix}_cell_invalid")
            else:
                cells[inline_axis] = cell
            if confidence is None or confidence <= 0.0 or confidence > 1.0:
                reasons.append(f"{axis_prefix}_confidence_invalid")
            if not str(axis.get("support_status") or "").startswith("supported_"):
                reasons.append(f"{axis_prefix}_support_status_invalid")
            if str(axis.get("reason") or ""):
                reasons.append(f"{axis_prefix}_supported_with_reason")
        elif axis_status == "unavailable":
            if cell != 0.0 or confidence != 0.0:
                reasons.append(f"{axis_prefix}_unavailable_measurement_invalid")
            if str(axis.get("support_status") or ""):
                reasons.append(f"{axis_prefix}_unavailable_support_invalid")
            if not str(axis.get("reason") or ""):
                reasons.append(f"{axis_prefix}_unavailable_reason_missing")
        else:
            reasons.append(f"{axis_prefix}_status_invalid")
    if status == "supported" and not cells:
        reasons.append(f"{prefix}_supported_without_axis")
    if status == "unavailable" and cells:
        reasons.append(f"{prefix}_unavailable_with_supported_axis")
    if str(value.get("calibration_sha256") or "") != (
        source_punctuation_cell_calibration_sha256(value)
    ):
        reasons.append(f"{prefix}_hash_mismatch")
    return cells


def _validate_source_punctuation_occurrence(
    occurrence: Mapping[str, Any],
    *,
    index: int,
    analysis_bbox: Sequence[int],
    reasons: list[str],
) -> None:
    prefix = f"source_punctuation_geometry_occurrence:{index}"
    _validate_exact_fields(
        occurrence,
        {
            "occurrence_id",
            "kind",
            "visual_reading_order_ordinal",
            "kind_ordinal",
            "inline_axis",
            "component_bboxes_local_xywh",
            "component_bboxes_page_xywh",
            "group_bbox_local_xywh",
            "group_bbox_page_xywh",
            "span_px",
            "pitch_px",
            "measurement_basis",
            "source_cell_px",
            "normalized_span",
            "normalized_pitch",
            "confidence",
            "reason_codes",
        },
        reasons,
        prefix,
    )
    kind = str(occurrence.get("kind") or "")
    if kind not in SOURCE_PUNCTUATION_GEOMETRY_SUPPORTED_KINDS:
        reasons.append(f"{prefix}_kind_invalid")
    if str(occurrence.get("inline_axis") or "") not in {"ttb", "ltr"}:
        reasons.append(f"{prefix}_inline_axis_invalid")
    measurement_basis = str(occurrence.get("measurement_basis") or "")
    if measurement_basis not in {
        SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED,
        SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE,
    }:
        reasons.append(f"{prefix}_measurement_basis_invalid")
    if (
        measurement_basis
        == SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE
        and kind != "dash"
    ):
        reasons.append(f"{prefix}_absolute_basis_kind_invalid")
    if not str(occurrence.get("occurrence_id") or ""):
        reasons.append(f"{prefix}_identity_missing")
    if _strict_nonnegative_int(occurrence.get("visual_reading_order_ordinal")) is None:
        reasons.append(f"{prefix}_visual_ordinal_invalid")
    if _strict_nonnegative_int(occurrence.get("kind_ordinal")) is None:
        reasons.append(f"{prefix}_kind_ordinal_invalid")
    local_group = _strict_bbox(occurrence.get("group_bbox_local_xywh"))
    page_group = _strict_bbox(occurrence.get("group_bbox_page_xywh"))
    if not local_group or not page_group:
        reasons.append(f"{prefix}_group_bbox_invalid")
    elif analysis_bbox:
        expected_page = [
            int(analysis_bbox[0]) + local_group[0],
            int(analysis_bbox[1]) + local_group[1],
            local_group[2],
            local_group[3],
        ]
        if page_group != expected_page:
            reasons.append(f"{prefix}_page_local_group_mismatch")
        local_container = [0, 0, int(analysis_bbox[2]), int(analysis_bbox[3])]
        if not _bbox_inside(local_group, local_container):
            reasons.append(f"{prefix}_group_outside_analysis_bbox")
    local_components = occurrence.get("component_bboxes_local_xywh")
    page_components = occurrence.get("component_bboxes_page_xywh")
    if (
        not isinstance(local_components, list)
        or not isinstance(page_components, list)
        or not local_components
        or len(local_components) != len(page_components)
    ):
        reasons.append(f"{prefix}_component_bboxes_invalid")
    else:
        for component_index, (local, page) in enumerate(
            zip(local_components, page_components)
        ):
            local_box = _strict_bbox(local)
            page_box = _strict_bbox(page)
            if not local_box or not page_box:
                reasons.append(
                    f"{prefix}_component_bbox_invalid:{component_index}"
                )
                continue
            if local_group and not _bbox_inside(local_box, local_group):
                reasons.append(
                    f"{prefix}_component_outside_group:{component_index}"
                )
            if analysis_bbox:
                expected_page = [
                    int(analysis_bbox[0]) + local_box[0],
                    int(analysis_bbox[1]) + local_box[1],
                    local_box[2],
                    local_box[3],
                ]
                if page_box != expected_page:
                    reasons.append(
                        f"{prefix}_page_local_component_mismatch:{component_index}"
                    )
    span = _strict_finite_number(occurrence.get("span_px"))
    pitch = _strict_finite_number(occurrence.get("pitch_px"))
    cell = _strict_finite_number(occurrence.get("source_cell_px"))
    normalized_span = _strict_finite_number(occurrence.get("normalized_span"))
    normalized_pitch = _strict_finite_number(occurrence.get("normalized_pitch"))
    confidence = _strict_finite_number(occurrence.get("confidence"))
    if span is None or span <= 0.0:
        reasons.append(f"{prefix}_span_invalid")
    if pitch is None or pitch <= 0.0:
        reasons.append(f"{prefix}_pitch_invalid")
    if measurement_basis == SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED:
        if cell is None or cell <= 0.0:
            reasons.append(f"{prefix}_source_cell_invalid")
        if normalized_span is None or normalized_span <= 0.0:
            reasons.append(f"{prefix}_normalized_span_invalid")
        if normalized_pitch is None or normalized_pitch <= 0.0:
            reasons.append(f"{prefix}_normalized_pitch_invalid")
    elif measurement_basis == (
        SOURCE_PUNCTUATION_MEASUREMENT_BASIS_ABSOLUTE_STROKE
    ):
        if cell != 0.0:
            reasons.append(f"{prefix}_absolute_source_cell_invalid")
        if normalized_span != 0.0:
            reasons.append(f"{prefix}_absolute_normalized_span_invalid")
        if normalized_pitch != 0.0:
            reasons.append(f"{prefix}_absolute_normalized_pitch_invalid")
    if confidence is None or confidence < 0.0 or confidence > 1.0:
        reasons.append(f"{prefix}_confidence_invalid")
    if (
        measurement_basis == SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED
        and span
        and cell
        and normalized_span is not None
        and abs(normalized_span - span / cell) > 1e-5
    ):
        reasons.append(f"{prefix}_normalized_span_mismatch")
    if (
        measurement_basis == SOURCE_PUNCTUATION_MEASUREMENT_BASIS_NORMALIZED
        and pitch
        and cell
        and normalized_pitch is not None
        and abs(normalized_pitch - pitch / cell) > 1e-5
    ):
        reasons.append(f"{prefix}_normalized_pitch_mismatch")
    inline_axis = str(occurrence.get("inline_axis") or "")
    if local_group and span:
        group_span = float(local_group[3] if inline_axis == "ttb" else local_group[2])
        if abs(group_span - span) > 1e-5:
            reasons.append(f"{prefix}_group_span_mismatch")
    occurrence_reasons = occurrence.get("reason_codes")
    if not isinstance(occurrence_reasons, list) or any(
        not isinstance(value, str) for value in occurrence_reasons
    ):
        reasons.append(f"{prefix}_reason_codes_invalid")


def _validated_source_text_footprint(
    bundle: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the sole executable source-footprint carrier.

    The footprint is optional layout evidence. Invalid or missing evidence is
    omitted without blocking rendering; rejection reasons remain auditable.
    Nested observation/style/legacy records are intentionally not consulted.
    """

    evidence = _field(bundle, "style_evidence_summary", {})
    if not isinstance(evidence, Mapping):
        return {}, _footprint_validation_record(
            "unavailable", ("source_text_footprint_style_evidence_not_mapping",)
        )
    raw = evidence.get("source_text_footprint")
    if raw is None:
        return {}, _footprint_validation_record(
            "unavailable", ("source_text_footprint_missing",)
        )
    if not isinstance(raw, Mapping):
        return {}, _footprint_validation_record(
            "rejected", ("source_text_footprint_not_mapping",)
        )

    footprint = copy_jsonish(raw)
    reasons = _source_text_footprint_rejection_reasons(
        bundle=bundle,
        evidence=evidence,
        footprint=footprint,
    )
    if reasons:
        return {}, _footprint_validation_record("rejected", reasons)
    return footprint, _footprint_validation_record(
        "accepted",
        ("source_text_footprint_identity_and_fact_set_validated",),
        fact_set_id=str(footprint.get("fact_set_id") or ""),
    )


def _footprint_validation_record(
    status: str,
    reason_codes: Sequence[str],
    *,
    fact_set_id: str = "",
) -> dict[str, Any]:
    return {
        "status": str(status or "unavailable"),
        "contract_version": SOURCE_TEXT_FOOTPRINT_VERSION,
        "reason_codes": [str(value) for value in reason_codes if str(value)],
        "fact_set_id": str(fact_set_id or ""),
    }


def _source_text_footprint_rejection_reasons(
    *,
    bundle: Any,
    evidence: Mapping[str, Any],
    footprint: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    footprint_fields = {
        "contract_version",
        "source_identity",
        "fact_set_id",
        "coordinate_space",
        "ink_authority",
        "union_bbox_local_xywh",
        "union_bbox_page_xywh",
        "x_occupied_bands",
        "y_occupied_bands",
        "writing_direction_evidence",
        "axis_profiles",
    }
    _validate_exact_fields(
        footprint,
        footprint_fields,
        reasons,
        "source_text_footprint",
    )
    if footprint.get("contract_version") != SOURCE_TEXT_FOOTPRINT_VERSION:
        reasons.append("source_text_footprint_contract_version_invalid")
    if footprint.get("coordinate_space") != "authorized_analysis_crop":
        reasons.append("source_text_footprint_coordinate_space_invalid")
    if footprint.get("ink_authority") != "independent_glyph_geometry":
        reasons.append("source_text_footprint_ink_authority_invalid")

    identity = footprint.get("source_identity")
    identity_fields = {
        "authorized_source_style_view_version",
        "page_id",
        "view_id",
        "bundle_id",
        "parent_id",
        "root_id",
        "cleanup_mask_ids",
        "owned_component_ids",
        "content_bbox_xywh",
        "analysis_bbox_xywh",
        "analysis_crop_shape_hw",
        "detector_input_sha256",
        "authorized_mask_sha256",
        "authorized_pixel_sha256",
        "resolved_ink_mask_sha256",
        "authorized_source_view_sha256",
    }
    if not isinstance(identity, Mapping):
        reasons.append("source_text_footprint_source_identity_not_mapping")
        identity = {}
    else:
        _validate_exact_fields(
            identity,
            identity_fields,
            reasons,
            "source_text_footprint_source_identity",
        )

    expected_identity = {
        "page_id": str(_field(bundle, "page_id") or ""),
        "bundle_id": str(_field(bundle, "bundle_id") or ""),
        "parent_id": str(_field(bundle, "parent_id") or ""),
        "root_id": str(_field(bundle, "root_id") or ""),
    }
    for key, expected in expected_identity.items():
        if str(evidence.get(key) or "") != expected:
            reasons.append(f"source_text_footprint_outer_{key}_mismatch")
        if str(identity.get(key) or "") != expected:
            reasons.append(f"source_text_footprint_{key}_mismatch")
    if evidence.get("style_evidence_version") != "parent_style_evidence_v2":
        reasons.append("source_text_footprint_style_evidence_version_invalid")
    if identity.get("authorized_source_style_view_version") != (
        AUTHORIZED_SOURCE_STYLE_VIEW_VERSION
    ):
        reasons.append("source_text_footprint_source_view_version_invalid")

    for outer_key, inner_key in (
        ("view_id", "view_id"),
        ("cleanup_mask_ids", "cleanup_mask_ids"),
        ("owned_component_ids", "owned_component_ids"),
        ("content_bbox", "content_bbox_xywh"),
        ("analysis_bbox", "analysis_bbox_xywh"),
        ("detector_input_sha256", "detector_input_sha256"),
    ):
        if copy_jsonish(evidence.get(outer_key)) != copy_jsonish(identity.get(inner_key)):
            reasons.append(
                f"source_text_footprint_outer_{outer_key}_identity_mismatch"
            )

    sha_fields = (
        "detector_input_sha256",
        "authorized_mask_sha256",
        "authorized_pixel_sha256",
        "resolved_ink_mask_sha256",
        "authorized_source_view_sha256",
    )
    for key in sha_fields:
        if not _is_sha256(identity.get(key)):
            reasons.append(f"source_text_footprint_{key}_invalid")

    identity_hash_input = dict(identity)
    declared_source_view_sha256 = str(
        identity_hash_input.pop("authorized_source_view_sha256", "") or ""
    )
    computed_source_view_sha256 = _canonical_sha256(identity_hash_input)
    if (
        not computed_source_view_sha256
        or declared_source_view_sha256 != computed_source_view_sha256
    ):
        reasons.append("source_text_footprint_source_view_sha256_mismatch")

    content_bbox = _strict_bbox(identity.get("content_bbox_xywh"))
    analysis_bbox = _strict_bbox(identity.get("analysis_bbox_xywh"))
    crop_shape = _strict_positive_int_pair(identity.get("analysis_crop_shape_hw"))
    if not content_bbox:
        reasons.append("source_text_footprint_content_bbox_invalid")
    if not analysis_bbox:
        reasons.append("source_text_footprint_analysis_bbox_invalid")
    if not crop_shape:
        reasons.append("source_text_footprint_analysis_crop_shape_invalid")
    if content_bbox and analysis_bbox and not _bbox_inside(content_bbox, analysis_bbox):
        reasons.append("source_text_footprint_content_bbox_outside_analysis_bbox")

    local_union_raw = footprint.get("union_bbox_local_xywh")
    page_union_raw = footprint.get("union_bbox_page_xywh")
    local_union = _strict_bbox(local_union_raw, allow_empty=True)
    page_union = _strict_bbox(page_union_raw, allow_empty=True)
    if local_union_raw != [] and not local_union:
        reasons.append("source_text_footprint_local_union_bbox_invalid")
    if page_union_raw != [] and not page_union:
        reasons.append("source_text_footprint_page_union_bbox_invalid")
    if bool(local_union) != bool(page_union):
        reasons.append("source_text_footprint_union_bbox_presence_mismatch")
    if local_union and crop_shape:
        crop_height, crop_width = crop_shape
        if not _bbox_inside(local_union, [0, 0, crop_width, crop_height]):
            reasons.append("source_text_footprint_local_union_outside_crop")
    if local_union and analysis_bbox:
        expected_page_union = [
            analysis_bbox[0] + local_union[0],
            analysis_bbox[1] + local_union[1],
            local_union[2],
            local_union[3],
        ]
        if page_union != expected_page_union:
            reasons.append("source_text_footprint_page_union_identity_mismatch")

    crop_height = crop_shape[0] if crop_shape else 0
    crop_width = crop_shape[1] if crop_shape else 0
    _validate_occupied_bands(
        footprint.get("x_occupied_bands"),
        limit=crop_width,
        axis="x",
        reasons=reasons,
    )
    _validate_occupied_bands(
        footprint.get("y_occupied_bands"),
        limit=crop_height,
        axis="y",
        reasons=reasons,
    )

    direction_evidence = footprint.get("writing_direction_evidence")
    direction_fields = {
        "status",
        "available_directions",
        "selection_authority",
    }
    if not isinstance(direction_evidence, Mapping):
        reasons.append(
            "source_text_footprint_writing_direction_evidence_not_mapping"
        )
        direction_evidence = {}
    else:
        _validate_exact_fields(
            direction_evidence,
            direction_fields,
            reasons,
            "source_text_footprint_writing_direction_evidence",
        )
    if direction_evidence.get("status") != "direction_neutral_axis_profiles":
        reasons.append("source_text_footprint_direction_status_invalid")
    if direction_evidence.get("selection_authority") != (
        SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY
    ):
        reasons.append("source_text_footprint_profile_selection_authority_invalid")

    available_directions = direction_evidence.get("available_directions")
    if (
        not isinstance(available_directions, list)
        or any(not isinstance(value, str) for value in available_directions)
        or len(available_directions) != len(set(available_directions))
        or set(available_directions) != {"ttb", "ltr"}
    ):
        reasons.append("source_text_footprint_available_directions_invalid")

    axis_profiles = footprint.get("axis_profiles")
    if not isinstance(axis_profiles, Mapping):
        reasons.append("source_text_footprint_axis_profiles_not_mapping")
        axis_profiles = {}
    else:
        _validate_exact_fields(
            axis_profiles,
            {"ttb", "ltr"},
            reasons,
            "source_text_footprint_axis_profiles",
        )
    for direction, cross_axis_limit, inline_limit in (
        ("ttb", crop_width, crop_height),
        ("ltr", crop_height, crop_width),
    ):
        _validate_source_text_axis_profile(
            axis_profiles.get(direction),
            direction=direction,
            cross_axis_limit=cross_axis_limit,
            inline_limit=inline_limit,
            union_available=bool(local_union),
            reasons=reasons,
        )

    fact_set_id = str(footprint.get("fact_set_id") or "")
    fact_payload = dict(footprint)
    fact_payload.pop("fact_set_id", None)
    computed_fact_hash = _canonical_sha256(fact_payload)
    expected_fact_set_id = (
        f"{SOURCE_TEXT_FOOTPRINT_VERSION}:{computed_fact_hash}"
        if computed_fact_hash
        else ""
    )
    if fact_set_id != expected_fact_set_id:
        reasons.append("source_text_footprint_fact_set_id_mismatch")
    return tuple(dict.fromkeys(reasons))


def _validate_source_text_axis_profile(
    value: Any,
    *,
    direction: str,
    cross_axis_limit: int,
    inline_limit: int,
    union_available: bool,
    reasons: list[str],
) -> None:
    prefix = f"source_text_footprint_axis_profile_{direction}"
    fields = {
        "writing_direction",
        "cross_axis_group_count",
        "cross_axis_group_count_reliable",
        "cross_axis_group_centers_px",
        "cross_axis_group_spans_px",
        "inline_capacity",
        "inline_capacity_reliable",
        "inline_capacity_provenance",
        "confidence",
        "reason",
    }
    if not isinstance(value, Mapping):
        reasons.append(f"{prefix}_not_mapping")
        return
    _validate_exact_fields(value, fields, reasons, prefix)
    if value.get("writing_direction") != direction:
        reasons.append(f"{prefix}_writing_direction_mismatch")

    group_count = _strict_nonnegative_int(
        value.get("cross_axis_group_count")
    )
    group_reliable = value.get("cross_axis_group_count_reliable")
    centers = _strict_finite_number_list(
        value.get("cross_axis_group_centers_px")
    )
    spans = _strict_finite_number_list(value.get("cross_axis_group_spans_px"))
    if group_count is None:
        reasons.append(f"{prefix}_group_count_invalid")
        group_count = 0
    if not isinstance(group_reliable, bool):
        reasons.append(f"{prefix}_group_reliability_invalid")
        group_reliable = False
    if centers is None or spans is None:
        reasons.append(f"{prefix}_group_geometry_invalid")
        centers, spans = [], []
    if group_reliable:
        if (
            group_count <= 0
            or len(centers) != group_count
            or len(spans) != group_count
        ):
            reasons.append(f"{prefix}_group_cardinality_invalid")
        if not union_available:
            reasons.append(f"{prefix}_reliable_group_without_union")
        if any(
            value < 0.0 or value > float(cross_axis_limit)
            for value in centers
        ):
            reasons.append(f"{prefix}_group_center_out_of_bounds")
        if any(
            value <= 0.0 or value > float(cross_axis_limit)
            for value in spans
        ):
            reasons.append(f"{prefix}_group_span_out_of_bounds")
        if centers != sorted(centers):
            reasons.append(f"{prefix}_group_centers_unsorted")
    elif group_count or centers or spans:
        reasons.append(f"{prefix}_unreliable_group_geometry_present")

    inline_capacity = _strict_nonnegative_int(value.get("inline_capacity"))
    inline_reliable = value.get("inline_capacity_reliable")
    inline_provenance = value.get("inline_capacity_provenance")
    if inline_capacity is None:
        reasons.append(f"{prefix}_inline_capacity_invalid")
        inline_capacity = 0
    if not isinstance(inline_reliable, bool):
        reasons.append(f"{prefix}_inline_reliability_invalid")
        inline_reliable = False
    if not isinstance(inline_provenance, str) or not inline_provenance:
        reasons.append(f"{prefix}_inline_provenance_invalid")
    if inline_reliable:
        if inline_capacity <= 0 or inline_capacity > max(0, inline_limit):
            reasons.append(f"{prefix}_reliable_inline_capacity_invalid")
        if not union_available:
            reasons.append(f"{prefix}_reliable_inline_without_union")
    elif inline_capacity:
        reasons.append(f"{prefix}_unreliable_inline_capacity_present")

    confidence = _strict_finite_number(value.get("confidence"))
    if confidence is None or confidence < 0.0 or confidence > 1.0:
        reasons.append(f"{prefix}_confidence_invalid")
    if not isinstance(value.get("reason"), str) or not value.get("reason"):
        reasons.append(f"{prefix}_reason_invalid")


def _validate_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    reasons: list[str],
    prefix: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        reasons.append(f"{prefix}_non_string_field")
    keys = {key for key in value if isinstance(key, str)}
    for key in sorted(expected - keys):
        reasons.append(f"{prefix}_missing_field:{key}")
    for key in sorted(keys - expected):
        reasons.append(f"{prefix}_unknown_field:{key}")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError, OverflowError):
        return ""
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _strict_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _strict_finite_number_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    output: list[float] = []
    for item in value:
        number = _strict_finite_number(item)
        if number is None:
            return None
        output.append(number)
    return output


def _strict_bbox(value: Any, *, allow_empty: bool = False) -> list[int]:
    if allow_empty and value == []:
        return []
    if not isinstance(value, list) or len(value) != 4:
        return []
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return []
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return []
    return [x, y, width, height]


def _strict_positive_int_pair(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        return []
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        return []
    return list(value)


def _bbox_inside(box: Sequence[int], container: Sequence[int]) -> bool:
    if len(box) != 4 or len(container) != 4:
        return False
    return (
        int(box[0]) >= int(container[0])
        and int(box[1]) >= int(container[1])
        and int(box[0]) + int(box[2]) <= int(container[0]) + int(container[2])
        and int(box[1]) + int(box[3]) <= int(container[1]) + int(container[3])
    )


def _validate_occupied_bands(
    value: Any,
    *,
    limit: int,
    axis: str,
    reasons: list[str],
) -> None:
    if not isinstance(value, list):
        reasons.append(f"source_text_footprint_{axis}_occupied_bands_not_list")
        return
    previous_end = 0
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            reasons.append(f"source_text_footprint_{axis}_band_not_mapping:{index}")
            continue
        _validate_exact_fields(
            item,
            {"start_px", "end_px", "span_px", "center_px"},
            reasons,
            f"source_text_footprint_{axis}_band:{index}",
        )
        start = _strict_nonnegative_int(item.get("start_px"))
        end = _strict_nonnegative_int(item.get("end_px"))
        span = _strict_finite_number(item.get("span_px"))
        center = _strict_finite_number(item.get("center_px"))
        if (
            start is None
            or end is None
            or end <= start
            or start < previous_end
            or end > limit
            or span is None
            or abs(span - float(end - start)) > 1e-6
            or center is None
            or abs(center - (float(start + end) * 0.5)) > 1e-6
        ):
            reasons.append(f"source_text_footprint_{axis}_band_geometry_invalid:{index}")
        previous_end = max(previous_end, int(end or 0))


def _clipping_region_ref(
    bundle: Any,
) -> dict[str, Any]:
    domain = _field(bundle, "render_layout_domain", {})
    return {
        "cleanup_target_bbox": bbox_from_value(_field(bundle, "cleanup_target_bbox", [])),
        "root_bbox": bbox_from_value(_field(bundle, "root_bbox", [])),
        "render_allowed_area": bbox_from_value(_field(bundle, "render_allowed_area", [])),
        "text_area_container_bbox": bbox_from_value(
            _field(bundle, "text_area_container_bbox", [])
        ),
        "text_area_container_polygon": copy_jsonish(
            _field(bundle, "text_area_container_polygon", [])
        ),
        "parent_render_domain": (
            copy_jsonish(domain) if isinstance(domain, Mapping) else {}
        ),
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
        "bundle_id": str(_field(bundle, "bundle_id") or ""),
        "parent_id": str(_field(bundle, "parent_id") or ""),
        "root_id": str(_field(bundle, "root_id") or ""),
        "graph_parent_id": str(_field(bundle, "graph_parent_id") or ""),
        "reading_order_index": _int_field(bundle, "reading_order_index", 0),
        "state": str(_field(bundle, "state") or ""),
        "role": str(_field(bundle, "role") or ""),
        "render_required": bool(_field(bundle, "render_required", False)),
        "source_quality_state": str(_field(bundle, "source_quality_state") or ""),
        "source_quality_action": str(_field(bundle, "source_quality_action") or ""),
        "render_decision_id": str(_field(bundle, "render_decision_id") or ""),
        "renderer_audit_id": str(_field(bundle, "renderer_audit_id") or ""),
        "target_language": str(_field(bundle, "target_language") or ""),
        "target_presentation_policy": copy_jsonish(
            _field(bundle, "target_presentation_policy", {})
        ),
        "render_layout_domain": copy_jsonish(
            _field(bundle, "render_layout_domain", {})
        ),
        "text_area_container_polygon": copy_jsonish(
            _field(bundle, "text_area_container_polygon", [])
        ),
        "text_area_oriented_frame": copy_jsonish(
            _field(bundle, "text_area_oriented_frame", {})
        ),
    }
