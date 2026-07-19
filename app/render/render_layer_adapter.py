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
from app.pipeline.parent_style_evidence import (
    AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
    SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY,
    SOURCE_TEXT_FOOTPRINT_VERSION,
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
    return record


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
    return {
        "cleanup_target_bbox": bbox_from_value(_field(bundle, "cleanup_target_bbox", [])),
        "root_bbox": bbox_from_value(_field(bundle, "root_bbox", [])),
        "render_allowed_area": bbox_from_value(_field(bundle, "render_allowed_area", [])),
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
    }
