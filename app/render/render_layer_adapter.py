# -*- coding: utf-8 -*-
"""Adapters from parent execution bundles to render-layer plans."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.pipeline.parent_execution_bundle import PARENT_EXECUTION_BUNDLE_VERSION
from app.render.source_punctuation_hints import source_visual_punctuation_hints
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

    bundles = list(parent_execution_bundles or [])
    render_slots = _render_layout_slots_for_bundles(bundles)
    plans: list[RenderLayerPlan] = []
    for index, bundle in enumerate(bundles):
        if not _bool_field(bundle, "render_required"):
            continue
        plans.append(
            render_layer_plan_from_parent_bundle(
                bundle,
                page_id=page_id,
                cleaned_page_base=cleaned_page_base,
                draw_order=index,
                render_layout_slot=render_slots.get(_identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id")),
            )
        )
    return plans


def render_layer_plan_from_parent_bundle(
    bundle: Any,
    *,
    page_id: str = "",
    cleaned_page_base: Mapping[str, Any] | None = None,
    draw_order: int | None = None,
    render_layout_slot: Mapping[str, Any] | None = None,
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
    slot = dict(render_layout_slot or _render_layout_slot_for_bundle(bundle))
    target_box = bbox_from_value(slot.get("box")) or _target_box_from_bundle(bundle, execution_region, render_record)
    hard_bounds = bbox_from_value(slot.get("hard_bounds")) or _hard_bounds_from_bundle(bundle, execution_region, render_record, target_box)
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
        source_provenance_ref=_source_provenance_ref(bundle, cleaned_page_base),
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
            "parent_render_slot": copy_jsonish(slot),
            "target_box_source": str(slot.get("source") or "legacy_target_box"),
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


def _render_layout_slots_for_bundles(bundles: Sequence[Any]) -> dict[str, dict[str, Any]]:
    renderable = [bundle for bundle in bundles or [] if _bool_field(bundle, "render_required")]
    groups: dict[str, list[Any]] = {}
    for bundle in renderable:
        root_id = _identity_field(bundle, "root_id", "text_block_root_id")
        groups.setdefault(root_id or _identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id"), []).append(bundle)

    slots: dict[str, dict[str, Any]] = {}
    for _root_id, group in groups.items():
        if len(group) == 1:
            bundle = group[0]
            slots[_identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id")] = _render_layout_slot_for_bundle(bundle)
            continue
        for bundle, slot in _render_layout_slots_for_sibling_group(group).items():
            slots[bundle] = slot
    return slots


def _render_layout_slots_for_sibling_group(group: Sequence[Any]) -> dict[str, dict[str, Any]]:
    bundles = list(group or [])
    if not bundles:
        return {}
    root_box = _common_root_box(bundles)
    if not root_box:
        return {
            _identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id"): _render_layout_slot_for_bundle(bundle)
            for bundle in bundles
        }
    container = _inner_render_box(root_box, semantic_class=_group_semantic_class(bundles))
    if not container:
        container = root_box
    source_boxes = [_bundle_source_box(bundle) for bundle in bundles]
    centers = [_center(box) for box in source_boxes]
    x_range = max((center[0] for center in centers), default=0.0) - min((center[0] for center in centers), default=0.0)
    y_range = max((center[1] for center in centers), default=0.0) - min((center[1] for center in centers), default=0.0)
    axis = "vertical" if y_range >= x_range else "horizontal"
    ordered = sorted(
        zip(bundles, source_boxes, centers),
        key=(lambda item: item[2][1]) if axis == "vertical" else (lambda item: item[2][0]),
    )
    partitions = _partition_container(container, [item[2] for item in ordered], axis=axis)
    output: dict[str, dict[str, Any]] = {}
    for index, ((bundle, source_box, center), part) in enumerate(zip(ordered, partitions)):
        bundle_id = _identity_field(bundle, "bundle_id", "parent_execution_bundle_id", "region_id")
        slot_box = _best_slot_box(part, source_box, semantic_class=_semantic_class(bundle))
        output[bundle_id] = _slot_record(
            bundle=bundle,
            box=slot_box,
            hard_bounds=part,
            source="parent_render_slot_root_partition",
            source_box=source_box,
            container_box=container,
            root_box=root_box,
            sibling_count=len(bundles),
            sibling_axis=axis,
            sibling_index=index,
        )
    return output


def _render_layout_slot_for_bundle(bundle: Any) -> dict[str, Any]:
    explicit = _explicit_render_layout_box(bundle)
    source_box = _bundle_source_box(bundle)
    root_box = bbox_from_value(_field(bundle, "root_bbox", []))
    semantic = _semantic_class(bundle)
    if explicit:
        return _slot_record(
            bundle=bundle,
            box=explicit,
            hard_bounds=explicit,
            source="explicit_parent_render_slot",
            source_box=source_box,
            container_box=explicit,
            root_box=root_box,
            sibling_count=1,
        )
    if _should_use_root_container(semantic, root_box, source_box):
        container = _inner_render_box(root_box, semantic_class=semantic) or root_box
        slot_box = _best_slot_box(container, source_box, semantic_class=semantic)
        return _slot_record(
            bundle=bundle,
            box=slot_box,
            hard_bounds=container,
            source="parent_render_slot_root_container",
            source_box=source_box,
            container_box=container,
            root_box=root_box,
            sibling_count=1,
        )
    fallback = _target_box_from_bundle(
        bundle,
        _mapping_field(bundle, "execution_region"),
        _nested_mapping(_mapping_field(bundle, "execution_region"), "render"),
    )
    return _slot_record(
        bundle=bundle,
        box=fallback,
        hard_bounds=fallback,
        source="parent_render_slot_existing_parent_box",
        source_box=source_box,
        container_box=fallback,
        root_box=root_box,
        sibling_count=1,
    )


def _explicit_render_layout_box(bundle: Any) -> list[int]:
    execution_region = _mapping_field(bundle, "execution_region")
    render_record = _nested_mapping(execution_region, "render")
    for value in (
        _field(bundle, "render_layout_box", []),
        _field(bundle, "parent_render_slot", []),
        render_record.get("render_layout_box"),
        render_record.get("parent_render_slot_box"),
        execution_region.get("render_layout_box"),
        execution_region.get("parent_render_slot_box"),
    ):
        bbox = bbox_from_value(value)
        if bbox:
            return bbox
    return []


def _bundle_source_box(bundle: Any) -> list[int]:
    execution_region = _mapping_field(bundle, "execution_region")
    render_record = _nested_mapping(execution_region, "render")
    for value in (
        _field(bundle, "source_contract_bbox", []),
        render_record.get("source_contract_bbox"),
        execution_region.get("source_contract_bbox"),
        _field(bundle, "parent_bbox", []),
        _field(bundle, "render_allowed_area", []),
        _field(bundle, "cleanup_target_bbox", []),
    ):
        bbox = bbox_from_value(value)
        if bbox:
            return bbox
    return []


def _common_root_box(bundles: Sequence[Any]) -> list[int]:
    boxes = [bbox_from_value(_field(bundle, "root_bbox", [])) for bundle in bundles or []]
    boxes = [box for box in boxes if box]
    if not boxes:
        return []
    first = boxes[0]
    if all(_same_box(first, box) for box in boxes[1:]):
        return first
    return _union_boxes(boxes)


def _semantic_class(bundle: Any) -> str:
    execution_region = _mapping_field(bundle, "execution_region")
    role = str(_field(bundle, "role") or execution_region.get("role") or "").lower()
    semantic = str(_field(bundle, "semantic_class") or execution_region.get("semantic_class") or execution_region.get("type") or "").lower()
    return semantic or role


def _group_semantic_class(bundles: Sequence[Any]) -> str:
    values = [_semantic_class(bundle) for bundle in bundles or []]
    if any("speech" in value for value in values):
        return "speech_bubble"
    if values:
        return values[0]
    return ""


def _should_use_root_container(semantic: str, root_box: Sequence[int], source_box: Sequence[int]) -> bool:
    root = bbox_from_value(root_box)
    source = bbox_from_value(source_box)
    if not root or not source:
        return False
    if "caption" in semantic or "background" in semantic:
        return _area(root) > _area(source) * 1.8 and _box_inside(source, root)
    if "speech" in semantic or "bubble" in semantic or semantic in {"speech"}:
        return _area(root) > _area(source) * 1.15 and _box_inside(source, root)
    return _area(root) > _area(source) * 2.0 and _box_inside(source, root)


def _inner_render_box(box: Sequence[int], *, semantic_class: str = "") -> list[int]:
    bbox = bbox_from_value(box)
    if not bbox:
        return []
    x, y, w, h = bbox
    if "caption" in semantic_class or "background" in semantic_class:
        pad_x = max(2, min(10, int(round(w * 0.03))))
        pad_y = max(2, min(10, int(round(h * 0.03))))
    else:
        pad_x = max(5, min(22, int(round(w * 0.08))))
        pad_y = max(5, min(22, int(round(h * 0.08))))
    if w - pad_x * 2 < 8 or h - pad_y * 2 < 8:
        return bbox
    return [x + pad_x, y + pad_y, w - pad_x * 2, h - pad_y * 2]


def _best_slot_box(container: Sequence[int], source_box: Sequence[int], *, semantic_class: str = "") -> list[int]:
    c = bbox_from_value(container)
    s = bbox_from_value(source_box)
    if not c:
        return s
    if not s or "caption" in semantic_class or "background" in semantic_class:
        return c
    cx, cy = _center(s)
    # Speech text should use the bubble/container as layout capacity, while the
    # source footprint remains an alignment hint for later layout.
    return c if _point_inside((cx, cy), c) else c


def _partition_container(container: Sequence[int], centers: Sequence[tuple[float, float]], *, axis: str) -> list[list[int]]:
    c = bbox_from_value(container)
    if not c:
        return []
    if len(centers) <= 1:
        return [c]
    x, y, w, h = c
    gutter = 4
    partitions: list[list[int]] = []
    if axis == "vertical":
        mids = [
            int(round((centers[index][1] + centers[index + 1][1]) / 2.0))
            for index in range(len(centers) - 1)
        ]
        starts = [y] + mids
        ends = mids + [y + h]
        for start, end in zip(starts, ends):
            py = max(y, start + (gutter if start != y else 0))
            ph = max(1, min(y + h, end - (gutter if end != y + h else 0)) - py)
            partitions.append([x, py, w, ph])
    else:
        mids = [
            int(round((centers[index][0] + centers[index + 1][0]) / 2.0))
            for index in range(len(centers) - 1)
        ]
        starts = [x] + mids
        ends = mids + [x + w]
        for start, end in zip(starts, ends):
            px = max(x, start + (gutter if start != x else 0))
            pw = max(1, min(x + w, end - (gutter if end != x + w else 0)) - px)
            partitions.append([px, y, pw, h])
    return partitions


def _slot_record(
    *,
    bundle: Any,
    box: Sequence[int],
    hard_bounds: Sequence[int],
    source: str,
    source_box: Sequence[int],
    container_box: Sequence[int],
    root_box: Sequence[int],
    sibling_count: int,
    sibling_axis: str = "",
    sibling_index: int | None = None,
) -> dict[str, Any]:
    slot_box = bbox_from_value(box)
    hard_box = bbox_from_value(hard_bounds) or slot_box
    source_bbox = bbox_from_value(source_box)
    record = {
        "render_layout_slot_version": "parent_render_slot_v1",
        "box": slot_box,
        "hard_bounds": hard_box,
        "source": source,
        "source_contract_bbox": source_bbox,
        "source_anchor_center": list(_center(source_bbox)) if source_bbox else [],
        "container_bbox": bbox_from_value(container_box),
        "root_bbox": bbox_from_value(root_box),
        "sibling_count": int(sibling_count),
        "sibling_axis": sibling_axis,
        "sibling_index": sibling_index,
        "parent_id": _identity_field(bundle, "parent_id", "parent_logical_text_unit_id"),
        "root_id": _identity_field(bundle, "root_id", "text_block_root_id"),
    }
    return record


def _center(box: Sequence[int]) -> tuple[float, float]:
    bbox = bbox_from_value(box)
    if not bbox:
        return (0.0, 0.0)
    return (float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0)


def _point_inside(point: tuple[float, float], box: Sequence[int]) -> bool:
    bbox = bbox_from_value(box)
    if not bbox:
        return False
    x, y, w, h = bbox
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def _box_inside(box: Sequence[int], container: Sequence[int]) -> bool:
    bbox = bbox_from_value(box)
    c = bbox_from_value(container)
    if not bbox or not c:
        return False
    return bbox[0] >= c[0] and bbox[1] >= c[1] and bbox[0] + bbox[2] <= c[0] + c[2] and bbox[1] + bbox[3] <= c[1] + c[3]


def _same_box(first: Sequence[int], second: Sequence[int]) -> bool:
    a = bbox_from_value(first)
    b = bbox_from_value(second)
    return bool(a and b and a == b)


def _area(box: Sequence[int]) -> int:
    bbox = bbox_from_value(box)
    if not bbox:
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _union_boxes(boxes: Sequence[Sequence[int]]) -> list[int]:
    normalized = [bbox_from_value(box) for box in boxes or []]
    normalized = [box for box in normalized if box]
    if not normalized:
        return []
    x0 = min(box[0] for box in normalized)
    y0 = min(box[1] for box in normalized)
    x1 = max(box[0] + box[2] for box in normalized)
    y1 = max(box[1] + box[3] for box in normalized)
    return [int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))]


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


def _source_provenance_ref(bundle: Any, cleaned_page_base: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    source_image_path = ""
    if isinstance(cleaned_page_base, Mapping):
        source_image_path = str(cleaned_page_base.get("source_image_path") or "")
    hints = source_visual_punctuation_hints(
        source_text=str(_field(bundle, "source_text") or ""),
        source_contract_bbox=source_bbox,
        source_image_path=source_image_path,
    )
    if hints:
        record["source_visual_punctuation_hints"] = hints
    return record


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
