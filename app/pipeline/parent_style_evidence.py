# -*- coding: utf-8 -*-
"""Parent-authorized source-pixel views for style observation.

This module is an adapter, not a style or cleanup decision owner. It exposes a
read-only runtime view over original-page coordinates and the foreground that
TextAreaPlan-authorized CTD components contributed to a parent CleanupMask.
Raw mask arrays remain runtime-only and are omitted from audit serialization.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import os
from typing import Any

import numpy as np


AUTHORIZED_SOURCE_STYLE_VIEW_VERSION = "authorized_source_style_view_v1"
_COMPONENT_PROJECTION_AUTHORITY = "text_area_component_authorization_map"
_READY_PROJECTION_STATE = "projection_ready"
_READY_MASK_STATE = "mask_ready"
_READY_OWNERSHIP_STATE = "ownership_binding_ready"
_READY_CLEAN_MASK_STATES = {
    "cleanup_mask_ready_from_owned_segmentation_components",
    "cleanup_mask_ready_with_component_exclusions",
}
_AUTHORIZED_SEMANTIC_STATES = {
    "cleanup_translate_speech",
    "cleanup_translate_background",
    "cleanup_translate_caption",
}


@dataclass(frozen=True)
class AuthorizedSourceStyleView:
    """Runtime-only parent-bound foreground available to style observers."""

    page_id: str
    view_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    status: str
    content_bbox: tuple[int, int, int, int] = ()
    analysis_bbox: tuple[int, int, int, int] = ()
    cleanup_mask_ids: tuple[str, ...] = ()
    owned_component_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    foreground_mask: Any = field(default=None, repr=False, compare=False)

    @property
    def available(self) -> bool:
        return self.status == "ready" and self.foreground_mask is not None

    def to_audit_dict(self) -> dict[str, Any]:
        pixels = 0
        shape: list[int] = []
        if self.foreground_mask is not None:
            try:
                pixels = int(np.count_nonzero(np.asarray(self.foreground_mask) > 0))
                shape = [int(value) for value in np.asarray(self.foreground_mask).shape[:2]]
            except Exception:
                pixels = 0
                shape = []
        return {
            "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
            "page_id": self.page_id,
            "view_id": self.view_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "status": self.status,
            "content_bbox": list(self.content_bbox),
            "analysis_bbox": list(self.analysis_bbox),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "owned_component_ids": list(self.owned_component_ids),
            "reason_codes": list(self.reason_codes),
            "foreground_mask_pixels": pixels,
            "foreground_mask_shape": shape,
            "foreground_mask_runtime_only": True,
            "style_decision_owner": "ParentStyleArbitrator",
            "pixel_projection_owner": "CleanupMask",
        }


@dataclass(frozen=True)
class AuthorizedSourceStyleViewBuildResult:
    page_id: str
    views: tuple[AuthorizedSourceStyleView, ...] = ()
    rejected_records: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def views_by_bundle_id(self) -> dict[str, AuthorizedSourceStyleView]:
        return {view.bundle_id: view for view in self.views if view.bundle_id}

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
            "page_id": self.page_id,
            "views": [view.to_audit_dict() for view in self.views],
            "rejected_records": [dict(record) for record in self.rejected_records],
            "errors": list(self.errors),
            "summary": {
                "view_count": len(self.views),
                "ready_count": sum(1 for view in self.views if view.available),
                "unavailable_count": sum(1 for view in self.views if not view.available),
                "rejected_record_count": len(self.rejected_records),
                "error_count": len(self.errors),
            },
        }


@dataclass(frozen=True)
class AuthorizedStyleObservationInputs:
    """Authorized detector presentations plus independently measured axes."""

    primary_input: Any = field(default=None, repr=False, compare=False)
    neutral_input: Any = field(default=None, repr=False, compare=False)
    primary_matte: int = 127
    fill_polarity: str = ""
    fill_color: str = ""
    support_color: str = ""
    source_cell_size_vertical_px: float = 0.0
    source_cell_size_horizontal_px: float = 0.0
    source_stroke_width_px: float = 0.0
    source_ink_stroke_width_px: float = 0.0
    ink_weight_class: str = ""
    ink_weight_confidence: float = 0.0
    scale_confidence: float = 0.0
    paint_confidence: float = 0.0
    stroke_confidence: float = 0.0
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.primary_input is not None and self.neutral_input is not None

    def source_cell_size_for_direction(self, direction: str) -> float:
        return (
            float(self.source_cell_size_horizontal_px)
            if str(direction or "").strip().lower() == "ltr"
            else float(self.source_cell_size_vertical_px)
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "observation_input_version": "authorized_style_observation_inputs_v2",
            "primary_matte": int(self.primary_matte),
            "fill_polarity": self.fill_polarity,
            "fill_color": self.fill_color,
            "support_color": self.support_color,
            "source_cell_size_vertical_px": round(
                float(self.source_cell_size_vertical_px), 6
            ),
            "source_cell_size_horizontal_px": round(
                float(self.source_cell_size_horizontal_px), 6
            ),
            "source_stroke_width_px": round(float(self.source_stroke_width_px), 6),
            "source_ink_stroke_width_px": round(
                float(self.source_ink_stroke_width_px), 6
            ),
            "ink_weight_class": self.ink_weight_class,
            "axis_confidence": {
                "scale": round(float(self.scale_confidence), 8),
                "paint": round(float(self.paint_confidence), 8),
                "stroke": round(float(self.stroke_confidence), 8),
                "ink_weight": round(float(self.ink_weight_confidence), 8),
            },
            "reason_codes": list(self.reason_codes),
            "metrics": _json_safe_mapping(self.metrics),
            "authorized_pixels_only": True,
        }


def build_authorized_source_style_views(
    *,
    page_id: str,
    parent_execution_bundles: Sequence[Any],
    cleanup_masks: Any,
    image_size: tuple[int, int] | Sequence[int] | None = None,
) -> AuthorizedSourceStyleViewBuildResult:
    """Project accepted CleanupMask foreground into immutable parent views.

    One view record is emitted for every render-required parent. Invalid or
    unavailable foreground produces an explicit unavailable view rather than a
    bbox, SourceGlyph, page, or render-slot fallback.
    """

    bundles = list(parent_execution_bundles or [])
    masks = _mask_records(cleanup_masks)
    width, height = _image_size(image_size)
    cleanup_page_id = str(_value(cleanup_masks, "page_id") or "")
    global_reasons: list[str] = []
    if not str(page_id or ""):
        global_reasons.append("style_view_page_identity_missing")
    if not cleanup_page_id:
        global_reasons.append("cleanup_mask_result_page_identity_missing")
    elif cleanup_page_id != str(page_id or ""):
        global_reasons.append("cleanup_mask_result_page_identity_mismatch")
    if width <= 0 or height <= 0:
        global_reasons.append("style_source_image_size_missing_or_invalid")
    grouped: dict[str, list[Any]] = {}
    rejected_records: list[dict[str, Any]] = []

    bundle_ids = [
        str(getattr(bundle, "bundle_id", "") or "") for bundle in bundles
    ]
    known_bundle_ids = {bundle_id for bundle_id in bundle_ids if bundle_id}
    duplicate_bundle_ids = {
        bundle_id
        for bundle_id in bundle_ids
        if bundle_id and bundle_ids.count(bundle_id) > 1
    }
    all_mask_ids = [str(_value(mask, "cleanup_mask_id") or "") for mask in masks]
    duplicate_mask_ids = {
        mask_id
        for mask_id in all_mask_ids
        if mask_id and all_mask_ids.count(mask_id) > 1
    }
    for mask in masks:
        owner = str(_value(mask, "parent_execution_bundle_id") or "")
        if not owner or owner not in known_bundle_ids:
            rejected_records.append(
                {
                    "cleanup_mask_id": str(_value(mask, "cleanup_mask_id") or ""),
                    "parent_execution_bundle_id": owner,
                    "status": "rejected",
                    "reason_codes": ["cleanup_mask_parent_binding_missing_or_unknown"],
                }
            )
            continue
        grouped.setdefault(owner, []).append(mask)

    views: list[AuthorizedSourceStyleView] = []
    errors: list[str] = []
    for bundle in bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            errors.append("parent_execution_bundle_missing_bundle_id")
            continue
        if not bool(getattr(bundle, "render_required", False)):
            continue
        parent_id = str(getattr(bundle, "parent_id", "") or "")
        root_id = str(getattr(bundle, "root_id", "") or "")
        bundle_page_id = str(getattr(bundle, "page_id", "") or "")
        candidates = grouped.get(bundle_id, [])
        if not candidates:
            views.append(
                _unavailable_view(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reasons=("authorized_cleanup_foreground_missing",),
                )
            )
            continue

        arrays: list[np.ndarray] = []
        mask_ids: list[str] = []
        component_ids: list[str] = []
        reasons: list[str] = list(global_reasons)
        if bundle_id in duplicate_bundle_ids:
            reasons.append("duplicate_parent_execution_bundle_identity")
        if bundle_page_id != str(page_id or ""):
            reasons.append("parent_execution_bundle_page_identity_mismatch")
        if not parent_id:
            reasons.append("parent_execution_bundle_parent_identity_missing")
        if not root_id:
            reasons.append("parent_execution_bundle_root_identity_missing")
        expected_shape: tuple[int, int] | None = (height, width) if width > 0 and height > 0 else None
        accepted_notes: list[str] = []
        for mask in candidates:
            reasons.extend(
                _mask_rejection_reasons(
                    mask,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    expected_shape=expected_shape,
                )
            )
            mask_id = str(_value(mask, "cleanup_mask_id") or "")
            if not mask_id:
                reasons.append("cleanup_mask_identity_missing")
            elif mask_id in duplicate_mask_ids:
                reasons.append("duplicate_cleanup_mask_identity")
            if _strings(_value(mask, "protected_component_ids")):
                accepted_notes.append("protected_components_excluded_upstream")
            if _strings(_value(mask, "ambiguous_component_ids")):
                accepted_notes.append("ambiguous_components_excluded_upstream")
            if _strings(_value(mask, "unowned_component_ids")):
                accepted_notes.append("unowned_components_excluded_upstream")
            raw = _value(mask, "foreground_mask")
            array = _foreground_array(raw)
            if array is not None:
                arrays.append(array)
                if expected_shape is None:
                    expected_shape = tuple(int(value) for value in array.shape[:2])
            if mask_id:
                mask_ids.append(mask_id)
            component_ids.extend(_strings(_value(mask, "owned_component_ids")))

        reasons = _unique(reasons)
        if reasons or not arrays:
            unavailable = _unavailable_view(
                page_id=page_id,
                bundle_id=bundle_id,
                parent_id=parent_id,
                root_id=root_id,
                reasons=tuple(reasons or ["authorized_cleanup_foreground_invalid"]),
                cleanup_mask_ids=tuple(_unique(mask_ids)),
                owned_component_ids=tuple(_unique(component_ids)),
            )
            views.append(unavailable)
            rejected_records.append(unavailable.to_audit_dict())
            continue

        union = np.zeros_like(arrays[0], dtype=np.uint8)
        for array in arrays:
            if array.shape != union.shape:
                reasons.append("cleanup_foreground_shape_mismatch")
                continue
            union[array > 0] = 255
        if reasons or int(np.count_nonzero(union)) <= 0:
            unavailable = _unavailable_view(
                page_id=page_id,
                bundle_id=bundle_id,
                parent_id=parent_id,
                root_id=root_id,
                reasons=tuple(_unique(reasons or ["authorized_cleanup_foreground_empty"])),
                cleanup_mask_ids=tuple(_unique(mask_ids)),
                owned_component_ids=tuple(_unique(component_ids)),
            )
            views.append(unavailable)
            rejected_records.append(unavailable.to_audit_dict())
            continue

        content_bbox = _mask_bbox_xywh(union)
        analysis_bbox = _analysis_bbox_from_mask(content_bbox, union.shape)
        frozen = np.array(union, dtype=np.uint8, copy=True)
        frozen.setflags(write=False)
        views.append(
            AuthorizedSourceStyleView(
                page_id=str(page_id or ""),
                view_id=f"styleview_{_safe_id(page_id)}_{_safe_id(bundle_id)}",
                bundle_id=bundle_id,
                parent_id=parent_id,
                root_id=root_id,
                status="ready",
                content_bbox=content_bbox,
                analysis_bbox=analysis_bbox,
                cleanup_mask_ids=tuple(_unique(mask_ids)),
                owned_component_ids=tuple(_unique(component_ids)),
                reason_codes=tuple(
                    _unique(["component_authorized_parent_foreground", *accepted_notes])
                ),
                foreground_mask=frozen,
            )
        )

    return AuthorizedSourceStyleViewBuildResult(
        page_id=str(page_id or ""),
        views=tuple(views),
        rejected_records=tuple(rejected_records),
        errors=tuple(_unique(errors)),
    )


def build_authorized_style_detector_input(
    image: Any,
    view: AuthorizedSourceStyleView,
) -> Any | None:
    """Return the fill-contrast presentation available to a style observer."""

    observation = build_authorized_style_observation_inputs(image, view)
    return observation.primary_input if observation.available else None


def build_authorized_style_observation_inputs(
    image: Any,
    view: AuthorizedSourceStyleView,
) -> AuthorizedStyleObservationInputs:
    """Build source-faithful presentations and direct authorized measurements.

    Pixels outside the accepted foreground are never exposed. The primary matte
    contrasts with the measured glyph fill; the neutral matte retains both dark
    and light selected pixels so model disagreement can be recorded per axis.
    """

    if image is None or not view.available or len(view.analysis_bbox) != 4:
        return AuthorizedStyleObservationInputs(
            reason_codes=("authorized_observation_input_unavailable",)
        )
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = _foreground_array(view.foreground_mask)
    if mask is None or tuple(mask.shape[:2]) != tuple(source.shape[:2]):
        return AuthorizedStyleObservationInputs(
            reason_codes=("authorized_observation_mask_invalid",)
        )
    x, y, width, height = [int(value) for value in view.analysis_bbox]
    if width <= 0 or height <= 0:
        return AuthorizedStyleObservationInputs(
            reason_codes=("authorized_observation_bbox_invalid",)
        )
    x0, y0 = max(0, x), max(0, y)
    x1 = min(source.shape[1], x + width)
    y1 = min(source.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return AuthorizedStyleObservationInputs(
            reason_codes=("authorized_observation_crop_empty",)
        )
    source_crop = source[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1] > 0
    if not np.any(mask_crop):
        return AuthorizedStyleObservationInputs(
            reason_codes=("authorized_observation_foreground_empty",)
        )

    metrics = _measure_authorized_style_crop(source_crop, mask_crop)
    fill_polarity = str(metrics.get("fill_polarity") or "")
    primary_matte = 255 if fill_polarity == "dark" else 0
    primary = np.full_like(source_crop, primary_matte, dtype=np.uint8)
    primary[mask_crop] = source_crop[mask_crop]
    neutral = np.full_like(source_crop, 127, dtype=np.uint8)
    neutral[mask_crop] = source_crop[mask_crop]
    from PIL import Image

    return AuthorizedStyleObservationInputs(
        primary_input=Image.fromarray(primary, mode="RGB"),
        neutral_input=Image.fromarray(neutral, mode="RGB"),
        primary_matte=primary_matte,
        fill_polarity=fill_polarity,
        fill_color=str(metrics.get("fill_color") or ""),
        support_color=str(metrics.get("support_color") or ""),
        source_cell_size_vertical_px=float(
            metrics.get("source_cell_size_vertical_px") or 0.0
        ),
        source_cell_size_horizontal_px=float(
            metrics.get("source_cell_size_horizontal_px") or 0.0
        ),
        source_stroke_width_px=float(metrics.get("source_stroke_width_px") or 0.0),
        source_ink_stroke_width_px=float(
            metrics.get("source_ink_stroke_width_px") or 0.0
        ),
        ink_weight_class=str(metrics.get("ink_weight_class") or ""),
        ink_weight_confidence=float(metrics.get("ink_weight_confidence") or 0.0),
        scale_confidence=float(metrics.get("scale_confidence") or 0.0),
        paint_confidence=float(metrics.get("paint_confidence") or 0.0),
        stroke_confidence=float(metrics.get("stroke_confidence") or 0.0),
        reason_codes=tuple(metrics.get("reason_codes") or ()),
        metrics=metrics,
    )


def _measure_authorized_style_crop(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    reasons: list[str] = ["authorized_pixels_only"]
    mask_binary = np.asarray(mask_crop, dtype=np.uint8)
    luma_image = (
        source_crop[:, :, 0].astype(np.float32) * 0.2126
        + source_crop[:, :, 1].astype(np.float32) * 0.7152
        + source_crop[:, :, 2].astype(np.float32) * 0.0722
    )
    try:
        import cv2

        distance = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    except Exception:
        distance = mask_binary.astype(np.float32)
        reasons.append("distance_transform_unavailable")
    edge = mask_crop & (distance <= 1.4)
    edge_luma = luma_image[edge]
    if edge_luma.size <= 0:
        edge_luma = luma_image[mask_crop]
        reasons.append("support_ring_fallback_all_authorized_pixels")
    support_median = float(np.median(edge_luma)) if edge_luma.size else 127.0
    support_iqr = (
        float(np.percentile(edge_luma, 75) - np.percentile(edge_luma, 25))
        if edge_luma.size
        else 255.0
    )
    fill_polarity = "dark" if support_median >= 128.0 else "light"
    contrast_threshold = 48.0
    fill = mask_crop & (
        (luma_image <= support_median - contrast_threshold)
        if fill_polarity == "dark"
        else (luma_image >= support_median + contrast_threshold)
    )
    minimum_fill_pixels = max(8, int(round(np.count_nonzero(mask_crop) * 0.01)))
    fill_cluster_resolved = True
    if int(np.count_nonzero(fill)) < minimum_fill_pixels:
        contrast_threshold = 32.0
        fill = mask_crop & (
            (luma_image <= support_median - contrast_threshold)
            if fill_polarity == "dark"
            else (luma_image >= support_median + contrast_threshold)
        )
        reasons.append("fill_contrast_threshold_relaxed")
    fill_count = int(np.count_nonzero(fill))
    if fill_count <= 0:
        fill = mask_crop.copy()
        fill_count = int(np.count_nonzero(fill))
        fill_cluster_resolved = False
        reasons.append("fill_cluster_unresolved_all_authorized_pixels_used")

    fill_pixels = source_crop[fill]
    support_pixels = source_crop[edge]
    if support_pixels.size <= 0:
        support_pixels = source_crop[mask_crop]
    fill_luma = luma_image[fill]
    fill_median = float(np.median(fill_luma)) if fill_luma.size else support_median
    contrast = abs(support_median - fill_median)
    fill_color = _polarized_hex_color(
        fill_pixels,
        fill_luma,
        polarity=fill_polarity,
    )
    support_color = _polarized_hex_color(
        support_pixels,
        edge_luma,
        polarity="light" if fill_polarity == "dark" else "dark",
        fraction=10.0,
    )

    x_spans = _projection_spans(fill, axis=0)
    y_spans = _projection_spans(fill, axis=1)
    component_size, component_count, component_mad = _character_component_size(mask_binary)
    vertical_size, vertical_confidence = _source_cell_size_from_geometry(
        x_spans,
        component_size=component_size,
        component_count=component_count,
        component_mad=component_mad,
    )
    horizontal_size, horizontal_confidence = _source_cell_size_from_geometry(
        y_spans,
        component_size=component_size,
        component_count=component_count,
        component_mad=component_mad,
    )
    if vertical_size > 0 or horizontal_size > 0:
        reasons.append("source_cell_scale_measured_from_authorized_geometry")
    else:
        reasons.append("source_cell_scale_unavailable")

    fill_distances = distance[fill]
    fill_distance_p25 = (
        float(np.percentile(fill_distances, 25)) if fill_distances.size else 0.0
    )
    try:
        ink_distance = cv2.distanceTransform(
            np.asarray(fill, dtype=np.uint8), cv2.DIST_L2, 5
        )
    except Exception:
        ink_distance = np.zeros_like(distance, dtype=np.float32)
        reasons.append("ink_distance_transform_unavailable")
    ink_distance_values = ink_distance[fill]
    ink_distance_p75 = (
        float(np.percentile(ink_distance_values, 75))
        if ink_distance_values.size
        else 0.0
    )
    source_ink_stroke_width = max(0.0, ink_distance_p75 * 2.0)
    ink_weight_class = ""
    ink_weight_confidence = 0.0
    if fill_cluster_resolved and fill_count >= 24 and ink_distance_p75 > 0.0:
        if ink_distance_p75 <= 1.5:
            ink_weight_class = "regular"
            ink_weight_confidence = min(0.92, 0.72 + fill_count / 4096.0)
            reasons.append("source_ink_weight_regular_measured")
        elif ink_distance_p75 >= 1.9:
            ink_weight_class = "bold"
            ink_weight_confidence = min(
                0.94,
                0.72 + min(0.16, (ink_distance_p75 - 1.9) * 0.12) + fill_count / 4096.0,
            )
            reasons.append("source_ink_weight_bold_measured")
        else:
            reasons.append("source_ink_weight_transition_unresolved")
    else:
        reasons.append("source_ink_weight_unavailable")

    scale_for_stroke = max(vertical_size, horizontal_size, 1.0)
    stroke_width = 0.0
    stroke_confidence = 0.0
    if fill_polarity == "dark":
        support_opposite_fraction = (
            float(np.mean(edge_luma >= 192.0)) if edge_luma.size else 0.0
        )
    else:
        support_opposite_fraction = (
            float(np.mean(edge_luma <= 63.0)) if edge_luma.size else 0.0
        )
    visible_support_transition = bool(
        contrast >= 64.0
        and support_iqr >= 48.0
        and 0.08 <= support_opposite_fraction <= 0.90
    )
    uniform_support_backing = bool(
        contrast >= 48.0
        and support_iqr <= 44.0
        and support_opposite_fraction >= 0.88
    )
    if visible_support_transition:
        stroke_width = min(
            max(1.0, float(round(scale_for_stroke * 0.055))),
            max(1.0, scale_for_stroke * 0.15),
        )
        stroke_confidence = min(
            0.95,
            0.52
            + min(0.25, contrast / 512.0)
            + min(0.18, support_iqr / 255.0),
        )
        reasons.append("source_visible_support_stroke_measured")
        if ink_weight_class:
            reasons = [
                reason
                for reason in reasons
                if not reason.startswith("source_ink_weight_")
            ]
            ink_weight_class = ""
            ink_weight_confidence = 0.0
            reasons.append("source_ink_weight_deferred_visible_stroke")
    elif uniform_support_backing:
        stroke_confidence = min(
            0.92,
            0.72 + min(0.15, contrast / 1024.0) + min(0.05, support_opposite_fraction / 20.0),
        )
        reasons.append("source_visible_stroke_absent_uniform_support")
    else:
        reasons.append("source_stroke_not_independently_supported")

    paint_confidence = min(
        0.98,
        max(0.0, contrast / 128.0) * min(1.0, fill_count / 64.0),
    )
    if paint_confidence > 0:
        reasons.append("source_fill_measured_from_authorized_contrast")
    scale_confidence = max(vertical_confidence, horizontal_confidence)
    return {
        "fill_polarity": fill_polarity,
        "fill_color": fill_color,
        "support_color": support_color,
        "source_cell_size_vertical_px": round(vertical_size, 6),
        "source_cell_size_horizontal_px": round(horizontal_size, 6),
        "source_stroke_width_px": round(stroke_width, 6),
        "source_ink_stroke_width_px": round(source_ink_stroke_width, 6),
        "ink_weight_class": ink_weight_class,
        "ink_weight_confidence": round(ink_weight_confidence, 8),
        "scale_confidence": round(scale_confidence, 8),
        "paint_confidence": round(paint_confidence, 8),
        "stroke_confidence": round(stroke_confidence, 8),
        "reason_codes": _unique(reasons),
        "support_luma_median": round(support_median, 6),
        "support_luma_iqr": round(support_iqr, 6),
        "fill_luma_median": round(fill_median, 6),
        "fill_support_contrast": round(contrast, 6),
        "authorized_pixel_count": int(np.count_nonzero(mask_crop)),
        "fill_pixel_count": fill_count,
        "fill_distance_p25": round(fill_distance_p25, 6),
        "ink_distance_p75": round(ink_distance_p75, 6),
        "support_opposite_fraction": round(support_opposite_fraction, 8),
        "visible_support_transition": visible_support_transition,
        "uniform_support_backing": uniform_support_backing,
        "fill_x_spans": [round(float(value), 6) for value in x_spans],
        "fill_y_spans": [round(float(value), 6) for value in y_spans],
        "character_component_size_p70": round(component_size, 6),
        "character_component_count": int(component_count),
        "character_component_mad": round(component_mad, 6),
    }


def _projection_spans(binary: np.ndarray, *, axis: int) -> list[float]:
    projected = np.any(np.asarray(binary, dtype=bool), axis=axis).astype(np.uint8)
    if projected.size <= 0:
        return []
    try:
        import cv2

        projected = cv2.morphologyEx(
            projected.reshape(1, -1),
            cv2.MORPH_CLOSE,
            np.ones((1, 3), dtype=np.uint8),
        ).reshape(-1)
    except Exception:
        pass
    padded = np.pad(projected, (1, 1), constant_values=0)
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [float(end - start) for start, end in zip(starts, ends) if end - start >= 2]


def _character_component_size(mask_binary: np.ndarray) -> tuple[float, int, float]:
    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(mask_binary, dtype=np.uint8), connectivity=8
        )
    except Exception:
        return 0.0, 0, 0.0
    sizes: list[float] = []
    for index in range(1, count):
        width = int(stats[index, 2])
        height = int(stats[index, 3])
        area = int(stats[index, 4])
        short = min(width, height)
        long = max(width, height)
        if area < 6 or short <= 0 or long / short > 2.2:
            continue
        sizes.append(float(long))
    if not sizes:
        return 0.0, 0, 0.0
    median = float(np.median(sizes))
    mad = float(np.median(np.abs(np.asarray(sizes, dtype=np.float32) - median)))
    return float(np.percentile(sizes, 70)), len(sizes), mad


def _source_cell_size_from_geometry(
    spans: Sequence[float],
    *,
    component_size: float,
    component_count: int,
    component_mad: float,
) -> tuple[float, float]:
    clean_spans = [float(value) for value in spans if float(value) >= 3.0]
    span_median = float(np.median(clean_spans)) if clean_spans else 0.0
    component_size = max(0.0, float(component_size))
    if component_size > 0 and span_median > 0:
        if (
            len(clean_spans) <= 2
            and component_count <= 6
            and component_size < span_median * 0.65
            and span_median <= component_size * 3.0
        ):
            return span_median, 0.72
        if component_count <= 2 and component_size > span_median * 1.5:
            return span_median, 0.74
        compatible = [
            value
            for value in clean_spans
            if component_size * 0.5 <= value <= component_size * 1.85
        ]
        if compatible:
            span_value = float(np.median(compatible))
            size = float(np.median([component_size, span_value]))
            variability = component_mad / max(1.0, component_size)
            confidence = max(0.62, min(0.94, 0.92 - variability * 0.35))
            return size, confidence
        return component_size, max(0.58, min(0.86, 0.62 + component_count * 0.015))
    if span_median > 0:
        return span_median, 0.72 if len(clean_spans) > 1 else 0.64
    if component_size > 0:
        return component_size, max(0.55, min(0.82, 0.58 + component_count * 0.012))
    return 0.0, 0.0


def _median_hex_color(pixels: np.ndarray) -> str:
    values = np.asarray(pixels, dtype=np.uint8)
    if values.size <= 0:
        return ""
    values = values.reshape(-1, 3)
    median = np.median(values.astype(np.float32), axis=0)
    if float(np.max(median) - np.min(median)) <= 18.0:
        gray = int(round(float(np.mean(median))))
        channels = (gray, gray, gray)
    else:
        channels = tuple(int(round(float(value))) for value in median)
    return "#" + "".join(f"{max(0, min(255, value)):02X}" for value in channels)


def _polarized_hex_color(
    pixels: np.ndarray,
    luma: np.ndarray,
    *,
    polarity: str,
    fraction: float = 35.0,
) -> str:
    values = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    weights = np.asarray(luma, dtype=np.float32).reshape(-1)
    if values.size <= 0 or weights.size != values.shape[0]:
        return _median_hex_color(values)
    fraction = max(5.0, min(50.0, float(fraction)))
    percentile = fraction if polarity == "dark" else 100.0 - fraction
    threshold = float(np.percentile(weights, percentile))
    selected = values[weights <= threshold] if polarity == "dark" else values[weights >= threshold]
    if selected.size <= 0:
        selected = values
    return _median_hex_color(selected)


def _json_safe_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in dict(value or {}).items():
        if isinstance(item, Mapping):
            output[str(key)] = _json_safe_mapping(item)
        elif isinstance(item, (list, tuple)):
            output[str(key)] = [
                _json_safe_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        elif isinstance(item, np.generic):
            output[str(key)] = item.item()
        else:
            output[str(key)] = item
    return output


def write_authorized_source_style_view_debug_artifacts(
    context: Mapping[str, Any] | None,
    *,
    image_path: str,
    result: AuthorizedSourceStyleViewBuildResult,
) -> dict[str, Any]:
    """Persist debug-only mask/overlay evidence without serializing arrays."""

    if not context or bool(context.get("perf_telemetry_only")):
        return {}
    try:
        from PIL import Image, ImageDraw

        from app.pipeline.debug_artifacts import (
            debug_stage_artifact_dir,
            write_debug_image_file,
        )

        artifact_dir = debug_stage_artifact_dir(
            dict(context),
            "parent_style_evidence",
            "authorized_source_style_views",
        )
        if not artifact_dir:
            return {}
        source_image_path = os.path.abspath(str(image_path or ""))
        with Image.open(source_image_path) as source_image:
            source = source_image.convert("RGB")
        source_array = np.asarray(source, dtype=np.uint8)
        page_height, page_width = source_array.shape[:2]
        source_sha256 = _file_sha256(source_image_path)
        overlay_array = np.array(source_array, copy=True)
        union = np.zeros((page_height, page_width), dtype=np.uint8)
        colors = (
            (0, 200, 90),
            (0, 140, 255),
            (255, 120, 0),
            (170, 80, 255),
            (255, 60, 140),
        )
        parent_artifacts: list[dict[str, Any]] = []
        for index, view in enumerate(result.views):
            if not view.available:
                continue
            mask = _foreground_array(view.foreground_mask)
            if mask is None or mask.shape != union.shape:
                continue
            selected = mask > 0
            union[selected] = 255
            color = np.asarray(colors[index % len(colors)], dtype=np.float32)
            overlay_array[selected] = np.clip(
                (overlay_array[selected].astype(np.float32) * 0.42) + (color * 0.58),
                0,
                255,
            ).astype(np.uint8)
            visualization_crop_ref = ""
            detector_input_ref = ""
            detector_input_sha256 = ""
            neutral_input_ref = ""
            neutral_input_sha256 = ""
            mask_ref = write_debug_image_file(
                os.path.join(
                    artifact_dir,
                    f"{_safe_id(result.page_id)}_{_safe_id(view.bundle_id)}_authorized_foreground.png",
                ),
                Image.fromarray(mask, mode="L"),
            )
            mask_ref = os.path.abspath(mask_ref) if mask_ref else ""
            observation_inputs = build_authorized_style_observation_inputs(source, view)
            detector_input = observation_inputs.primary_input
            if detector_input is not None:
                detector_input_ref = write_debug_image_file(
                    os.path.join(
                        artifact_dir,
                        f"{_safe_id(result.page_id)}_{_safe_id(view.bundle_id)}_detector_input.png",
                    ),
                    detector_input,
                )
                detector_input_ref = (
                    os.path.abspath(detector_input_ref) if detector_input_ref else ""
                )
                detector_input_sha256 = hashlib.sha256(
                    np.asarray(detector_input.convert("RGB"), dtype=np.uint8).tobytes()
                ).hexdigest()
            neutral_input = observation_inputs.neutral_input
            if neutral_input is not None:
                neutral_input_ref = write_debug_image_file(
                    os.path.join(
                        artifact_dir,
                        f"{_safe_id(result.page_id)}_{_safe_id(view.bundle_id)}_detector_input_neutral.png",
                    ),
                    neutral_input,
                )
                neutral_input_ref = (
                    os.path.abspath(neutral_input_ref) if neutral_input_ref else ""
                )
                neutral_input_sha256 = hashlib.sha256(
                    np.asarray(neutral_input.convert("RGB"), dtype=np.uint8).tobytes()
                ).hexdigest()
            if len(view.analysis_bbox) == 4:
                x, y, width, height = [int(value) for value in view.analysis_bbox]
                x1, y1 = min(page_width, x + width), min(page_height, y + height)
                if x1 > x and y1 > y:
                    crop_source = source_array[y:y1, x:x1]
                    crop_mask = selected[y:y1, x:x1]
                    masked = np.full_like(crop_source, 224)
                    masked[crop_mask] = crop_source[crop_mask]
                    visualization_crop_ref = write_debug_image_file(
                        os.path.join(
                            artifact_dir,
                            f"{_safe_id(result.page_id)}_{_safe_id(view.bundle_id)}_authorized_pixels.png",
                        ),
                        Image.fromarray(masked, mode="RGB"),
                    )
                    visualization_crop_ref = (
                        os.path.abspath(visualization_crop_ref)
                        if visualization_crop_ref
                        else ""
                    )
            parent_artifacts.append(
                {
                    "page_id": result.page_id,
                    "bundle_id": view.bundle_id,
                    "parent_id": view.parent_id,
                    "root_id": view.root_id,
                    "view_id": view.view_id,
                    "authorized_foreground_mask": mask_ref,
                    "authorized_foreground_mask_sha256": _array_sha256(mask),
                    "authorized_foreground_mask_shape": [
                        int(value) for value in mask.shape[:2]
                    ],
                    "authorized_foreground_mask_pixels": int(np.count_nonzero(mask)),
                    "authorized_pixel_crop": visualization_crop_ref,
                    "authorized_pixel_crop_role": "visualization_only_not_detector_input",
                    "authorized_detector_input": detector_input_ref,
                    "authorized_detector_input_sha256": detector_input_sha256,
                    "authorized_neutral_detector_input": neutral_input_ref,
                    "authorized_neutral_detector_input_sha256": neutral_input_sha256,
                    "authorized_style_observation": observation_inputs.to_audit_dict(),
                    "content_bbox": list(view.content_bbox),
                    "analysis_bbox": list(view.analysis_bbox),
                    "cleanup_mask_ids": list(view.cleanup_mask_ids),
                    "owned_component_ids": list(view.owned_component_ids),
                }
            )

        overlay = Image.fromarray(overlay_array, mode="RGB")
        draw = ImageDraw.Draw(overlay)
        for index, view in enumerate(result.views):
            if not view.available or len(view.analysis_bbox) != 4:
                continue
            x, y, width, height = [int(value) for value in view.analysis_bbox]
            color = colors[index % len(colors)]
            draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
            draw.text((x + 2, max(0, y - 12)), view.bundle_id, fill=color)

        page_token = _safe_id(result.page_id)
        overlay_ref = write_debug_image_file(
            os.path.join(artifact_dir, f"{page_token}_authorized_style_view_overlay.jpg"),
            overlay,
            quality=92,
        )
        union_ref = write_debug_image_file(
            os.path.join(artifact_dir, f"{page_token}_authorized_style_view_union.png"),
            Image.fromarray(union, mode="L"),
        )
        overlay_ref = os.path.abspath(overlay_ref) if overlay_ref else ""
        union_ref = os.path.abspath(union_ref) if union_ref else ""
        return {
            "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
            "page_id": result.page_id,
            "artifact_path_basis": "absolute",
            "source_image_path": source_image_path,
            "source_image_sha256": source_sha256,
            "source_image_size": [page_width, page_height],
            "overlay": overlay_ref,
            "authorized_foreground_union": union_ref,
            "parents": parent_artifacts,
        }
    except Exception as exc:
        return {
            "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
            "page_id": result.page_id,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


def _mask_rejection_reasons(
    mask: Any,
    *,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    expected_shape: tuple[int, int] | None,
) -> list[str]:
    reasons: list[str] = []
    if str(_value(mask, "parent_execution_bundle_id") or "") != bundle_id:
        reasons.append("cleanup_mask_parent_binding_mismatch")
    mask_parent = str(_value(mask, "parent_logical_text_unit_id") or "")
    if not mask_parent:
        reasons.append("cleanup_mask_parent_identity_missing")
    elif mask_parent != parent_id:
        reasons.append("cleanup_mask_parent_identity_mismatch")
    mask_root = str(_value(mask, "text_block_root_id") or "")
    if not mask_root:
        reasons.append("cleanup_mask_root_identity_missing")
    elif not root_id or mask_root != root_id:
        reasons.append("cleanup_mask_root_identity_mismatch")
    if bool(_value(mask, "protected")):
        reasons.append("cleanup_mask_protected")
    if str(_value(mask, "component_projection_method") or "") != _COMPONENT_PROJECTION_AUTHORITY:
        reasons.append("component_projection_authority_invalid")
    if str(_value(mask, "clean_mask_authority") or "") != _COMPONENT_PROJECTION_AUTHORITY:
        reasons.append("clean_mask_authority_invalid")
    if str(_value(mask, "ownership_binding_status") or "") != _READY_OWNERSHIP_STATE:
        reasons.append("cleanup_mask_parent_binding_not_ready")
    if str(_value(mask, "projection_quality_state") or "") != _READY_PROJECTION_STATE:
        reasons.append("component_projection_not_ready")
    if str(_value(mask, "mask_readiness_state") or "") != _READY_MASK_STATE:
        reasons.append("cleanup_mask_not_ready")
    clean_state = str(_value(mask, "clean_mask_state") or "")
    if clean_state not in _READY_CLEAN_MASK_STATES:
        reasons.append("clean_mask_state_not_ready")
    semantic_states = _authorization_states(
        _value(mask, "semantic_authorization_state")
    )
    if (
        not semantic_states
        or any(state not in _AUTHORIZED_SEMANTIC_STATES for state in semantic_states)
    ):
        reasons.append("semantic_authorization_not_executable")
    for key, reason in (
        ("non_segmentation_or_local_fallback_used", "non_segmentation_fallback_detected"),
        ("bbox_executable_foreground_detected", "bbox_foreground_influence_detected"),
        ("page_level_executable_foreground_detected", "page_foreground_influence_detected"),
        ("sourceglyph_executable_influence_detected", "sourceglyph_influence_detected"),
        ("segmentation_contract_override_detected", "segmentation_contract_override_detected"),
    ):
        if bool(_value(mask, key)):
            reasons.append(reason)
    array = _foreground_array(_value(mask, "foreground_mask"))
    if array is None:
        reasons.append("authorized_cleanup_foreground_missing_or_invalid")
    else:
        if expected_shape is not None and tuple(array.shape[:2]) != tuple(expected_shape):
            reasons.append("cleanup_foreground_page_shape_mismatch")
        if int(np.count_nonzero(array)) <= 0:
            reasons.append("authorized_cleanup_foreground_empty")
    if not _strings(_value(mask, "owned_component_ids")):
        reasons.append("owned_component_identity_missing")
    return _unique(reasons)


def _unavailable_view(
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    reasons: tuple[str, ...],
    cleanup_mask_ids: tuple[str, ...] = (),
    owned_component_ids: tuple[str, ...] = (),
) -> AuthorizedSourceStyleView:
    return AuthorizedSourceStyleView(
        page_id=str(page_id or ""),
        view_id=f"styleview_{_safe_id(page_id)}_{_safe_id(bundle_id)}",
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        status="unavailable",
        cleanup_mask_ids=cleanup_mask_ids,
        owned_component_ids=owned_component_ids,
        reason_codes=tuple(_unique(reasons)),
        foreground_mask=None,
    )


def _mask_records(cleanup_masks: Any) -> list[Any]:
    if cleanup_masks is None:
        return []
    if isinstance(cleanup_masks, Mapping):
        values = cleanup_masks.get("masks") or []
        return list(values) if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []
    values = getattr(cleanup_masks, "masks", cleanup_masks)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return []


def _foreground_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim == 3:
        array = np.max(array, axis=2)
    if array.ndim != 2 or array.size <= 0:
        return None
    return (array > 0).astype(np.uint8) * 255


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.uint8).tobytes()).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return ()
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _analysis_bbox_from_mask(
    content_bbox: tuple[int, int, int, int],
    shape: Sequence[int],
) -> tuple[int, int, int, int]:
    if len(content_bbox) != 4 or len(shape) < 2:
        return ()
    x, y, width, height = content_bbox
    page_height, page_width = int(shape[0]), int(shape[1])
    padding = max(2, int(round(max(width, height) * 0.08)))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(page_width, x + width + padding)
    y1 = min(page_height, y + height + padding)
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _image_size(value: tuple[int, int] | Sequence[int] | None) -> tuple[int, int]:
    if value is None or len(value) < 2:
        return (0, 0)
    try:
        return (max(0, int(value[0])), max(0, int(value[1])))
    except Exception:
        return (0, 0)


def _value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


def _authorization_states(value: Any) -> list[str]:
    states: list[str] = []
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    for item in values:
        for token in str(item or "").split(","):
            state = token.strip()
            if state and state not in states:
                states.append(state)
    return states


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in text) or "unknown"
