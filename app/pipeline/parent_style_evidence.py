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
import json
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
_PERCEPTUAL_STYLE_AXES_VERSION = "authorized_perceptual_style_axes_v2"
_PERCEPTUAL_STYLE_PROVENANCE = "cleanup_mask_authorized_source_style_view_v1"
_PERCEPTUAL_STYLE_FACT_SET_PREFIX = "authorized_perceptual_fact_set_v1:"
_PERCEPTUAL_STYLE_AXES = ("fill", "outline", "shadow", "rotation")
_ADDITIVE_FILL_MIN_CHROMA = 48.0
_ADDITIVE_FILL_MIN_CLUSTER_PIXELS = 24
_ADDITIVE_FILL_MIN_CLUSTER_MASK_FRACTION = 0.08
_ADDITIVE_FILL_MIN_CORE_CHROMATIC_FRACTION = 0.38
_ADDITIVE_FILL_MAX_COLOR_DISPERSION = 24.0
_ADDITIVE_FILL_MAX_GLYPH_BBOX_OCCUPANCY = 0.82
_ADDITIVE_FILL_MIN_CORE_DEPTH_RATIO = 0.58
_ADDITIVE_FILL_MIN_CORE_BORDER_MARGIN_PX = 1
_ADDITIVE_FILL_MIN_CORE_DEPTH_SEPARATION_PX = 0.75
_ADDITIVE_FILL_MIN_CORE_DEPTH_SEPARATION_RATIO = 1.20
_ADDITIVE_OUTLINE_MIN_CLUSTER_PIXELS = 24
_ADDITIVE_OUTLINE_MIN_CLUSTER_MASK_FRACTION = 0.08
_ADDITIVE_OUTLINE_MAX_COLOR_DISPERSION = 24.0
_ADDITIVE_OUTLINE_MAX_COHORTS = 8
_ADDITIVE_OUTLINE_MIN_CORE_COMPONENTS = 1
_ADDITIVE_OUTLINE_MAX_CORE_BBOX_OCCUPANCY = 0.82
_ADDITIVE_OUTLINE_MIN_BORDER_MARGIN_PX = 1
_ADDITIVE_OUTLINE_MIN_RADIAL_WIDTH_PX = 1.5
_ADDITIVE_OUTLINE_MAX_RADIAL_WIDTH_PX = 16.0
_ADDITIVE_OUTLINE_MIN_CORE_SHELL_DEPTH_DELTA_PX = 1.5
_ADDITIVE_OUTLINE_MIN_PAIR_MASK_FRACTION = 0.90
_ADDITIVE_OUTLINE_MIN_SHELL_RING_RECALL = 0.88
_ADDITIVE_OUTLINE_MIN_RING_SHELL_PRECISION = 0.80
_ADDITIVE_OUTLINE_MIN_COLOR_DISTANCE = 32.0
_ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS = 24
_ADDITIVE_ROTATION_MIN_CLUSTER_MASK_FRACTION = 0.08
_ADDITIVE_ROTATION_MAX_COLOR_DISPERSION = 24.0
_ADDITIVE_ROTATION_MAX_COHORTS = 8
_ADDITIVE_ROTATION_MIN_COMPONENTS = 2
_ADDITIVE_ROTATION_MIN_COMPONENT_PIXELS = 8
_ADDITIVE_ROTATION_MIN_BORDER_MARGIN_PX = 1
_ADDITIVE_ROTATION_MIN_ASPECT_RATIO = 1.60
_ADDITIVE_ROTATION_MIN_BBOX_OCCUPANCY = 0.12
_ADDITIVE_ROTATION_MAX_BBOX_OCCUPANCY = 0.78
_ADDITIVE_ROTATION_MIN_ABS_DEGREES = 12.0
_ADDITIVE_ROTATION_MAX_ABS_DEGREES = 40.0
_ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES = 3.0
_ADDITIVE_ROTATION_MAX_CANDIDATE_SPREAD_DEGREES = 3.0
_ADDITIVE_SHADOW_MIN_EFFECT_PIXELS = 24
_ADDITIVE_SHADOW_MIN_EFFECT_MASK_FRACTION = 0.05
_ADDITIVE_SHADOW_CORE_COLOR_DISTANCE = 24.0
_ADDITIVE_SHADOW_MIN_EFFECT_BORDER_MARGIN_PX = 1
_ADDITIVE_SHADOW_UNIFORM_LUMA_IQR = 8.0
_ADDITIVE_SHADOW_CENTRAL_LUMA_PERCENTILE = 35.0
_ADDITIVE_SHADOW_MIN_CORE_EFFECT_LUMA_DELTA = 16.0
_ADDITIVE_SHADOW_MIN_OFFSET_PX = 5.0
_ADDITIVE_SHADOW_MAX_OFFSET_PX = 32.0
_ADDITIVE_SHADOW_MIN_CENTRAL_EXPLAINED_FRACTION = 0.88
_ADDITIVE_SHADOW_COMPETING_PEAK_DISTANCE_PX = 8.0
_ADDITIVE_SHADOW_COMPETING_PEAK_RATIO = 0.90
_ADDITIVE_SHADOW_MIN_SPATIAL_RECALL = 0.90
_ADDITIVE_SHADOW_MIN_SPATIAL_PRECISION = 0.85
_ADDITIVE_SHADOW_MAX_SPREAD_RADIUS_PX = 16
_ADDITIVE_SHADOW_BLUR_SPREAD_DIVISOR = 1.4


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
    source_cell_confidence_vertical: float = 0.0
    source_cell_confidence_horizontal: float = 0.0
    source_cell_support_vertical: str = ""
    source_cell_support_horizontal: str = ""
    source_stroke_width_px: float = 0.0
    source_ink_stroke_width_px: float = 0.0
    ink_weight_class: str = ""
    ink_weight_confidence: float = 0.0
    ink_weight_class_vertical: str = ""
    ink_weight_confidence_vertical: float = 0.0
    ink_weight_support_vertical: str = ""
    ink_weight_class_horizontal: str = ""
    ink_weight_confidence_horizontal: float = 0.0
    ink_weight_support_horizontal: str = ""
    scale_confidence: float = 0.0
    paint_confidence: float = 0.0
    stroke_confidence: float = 0.0
    detector_input_sha256: str = ""
    authorized_perceptual_source_identity: Mapping[str, Any] = field(
        default_factory=dict
    )
    perceptual_axis_evidence: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.primary_input is not None and self.neutral_input is not None

    def source_cell_size_for_direction(self, direction: str) -> float:
        return self.source_cell_measurement_for_direction(direction)[0]

    def source_cell_measurement_for_direction(
        self,
        direction: str,
    ) -> tuple[float, float, str]:
        normalized = str(direction or "").strip().lower()
        if normalized == "ltr":
            return (
                float(self.source_cell_size_horizontal_px),
                float(self.source_cell_confidence_horizontal),
                "horizontal",
            )
        if normalized == "ttb":
            return (
                float(self.source_cell_size_vertical_px),
                float(self.source_cell_confidence_vertical),
                "vertical",
            )
        return 0.0, 0.0, ""

    def source_cell_support_for_direction(self, direction: str) -> str:
        normalized = str(direction or "").strip().lower()
        if normalized == "ltr":
            return str(self.source_cell_support_horizontal or "")
        if normalized == "ttb":
            return str(self.source_cell_support_vertical or "")
        return ""

    def ink_weight_measurement_for_direction(
        self,
        direction: str,
    ) -> tuple[str, float, str]:
        normalized = str(direction or "").strip().lower()
        if normalized == "ttb" and str(self.ink_weight_support_vertical).startswith(
            "supported_"
        ):
            return (
                str(self.ink_weight_class_vertical or ""),
                float(self.ink_weight_confidence_vertical),
                str(self.ink_weight_support_vertical),
            )
        if normalized == "ltr" and str(self.ink_weight_support_horizontal).startswith(
            "supported_"
        ):
            return (
                str(self.ink_weight_class_horizontal or ""),
                float(self.ink_weight_confidence_horizontal),
                str(self.ink_weight_support_horizontal),
            )
        if self.ink_weight_class in {"regular", "bold"}:
            return (
                str(self.ink_weight_class),
                float(self.ink_weight_confidence),
                "supported_direct_ink_geometry",
            )
        return "", 0.0, ""

    def to_audit_dict(self) -> dict[str, Any]:
        result = {
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
            "source_cell_confidence_vertical": round(
                float(self.source_cell_confidence_vertical), 8
            ),
            "source_cell_confidence_horizontal": round(
                float(self.source_cell_confidence_horizontal), 8
            ),
            "source_cell_support_vertical": self.source_cell_support_vertical,
            "source_cell_support_horizontal": self.source_cell_support_horizontal,
            "source_stroke_width_px": round(float(self.source_stroke_width_px), 6),
            "source_ink_stroke_width_px": round(
                float(self.source_ink_stroke_width_px), 6
            ),
            "ink_weight_class": self.ink_weight_class,
            "ink_weight_class_vertical": self.ink_weight_class_vertical,
            "ink_weight_confidence_vertical": round(
                float(self.ink_weight_confidence_vertical), 8
            ),
            "ink_weight_support_vertical": self.ink_weight_support_vertical,
            "ink_weight_class_horizontal": self.ink_weight_class_horizontal,
            "ink_weight_confidence_horizontal": round(
                float(self.ink_weight_confidence_horizontal), 8
            ),
            "ink_weight_support_horizontal": self.ink_weight_support_horizontal,
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
        if self.perceptual_axis_evidence:
            result["authorized_perceptual_source_identity"] = _json_safe_mapping(
                self.authorized_perceptual_source_identity
            )
            result["perceptual_axis_evidence"] = _json_safe_mapping(
                self.perceptual_axis_evidence
            )
        return result


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
    detector_input_sha256 = _array_sha256(np.ascontiguousarray(primary))
    perceptual_source_identity = _authorized_perceptual_source_identity(
        view=view,
        source_crop=source_crop,
        mask_crop=mask_crop,
        detector_input_sha256=detector_input_sha256,
    )
    try:
        perceptual_axis_evidence = _build_additive_perceptual_carrier(
            source_crop=source_crop,
            mask_crop=mask_crop,
            source_identity=perceptual_source_identity,
        )
    except Exception:
        # The additive carrier is optional.  It must never make the accepted
        # Task A detector inputs or direct measurements unavailable.
        perceptual_axis_evidence = {}
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
        source_cell_confidence_vertical=float(
            metrics.get("source_cell_confidence_vertical") or 0.0
        ),
        source_cell_confidence_horizontal=float(
            metrics.get("source_cell_confidence_horizontal") or 0.0
        ),
        source_cell_support_vertical=str(
            metrics.get("source_cell_support_vertical") or ""
        ),
        source_cell_support_horizontal=str(
            metrics.get("source_cell_support_horizontal") or ""
        ),
        source_stroke_width_px=float(metrics.get("source_stroke_width_px") or 0.0),
        source_ink_stroke_width_px=float(
            metrics.get("source_ink_stroke_width_px") or 0.0
        ),
        ink_weight_class=str(metrics.get("ink_weight_class") or ""),
        ink_weight_confidence=float(metrics.get("ink_weight_confidence") or 0.0),
        ink_weight_class_vertical=str(
            metrics.get("ink_weight_class_vertical") or ""
        ),
        ink_weight_confidence_vertical=float(
            metrics.get("ink_weight_confidence_vertical") or 0.0
        ),
        ink_weight_support_vertical=str(
            metrics.get("ink_weight_support_vertical") or ""
        ),
        ink_weight_class_horizontal=str(
            metrics.get("ink_weight_class_horizontal") or ""
        ),
        ink_weight_confidence_horizontal=float(
            metrics.get("ink_weight_confidence_horizontal") or 0.0
        ),
        ink_weight_support_horizontal=str(
            metrics.get("ink_weight_support_horizontal") or ""
        ),
        scale_confidence=float(metrics.get("scale_confidence") or 0.0),
        paint_confidence=float(metrics.get("paint_confidence") or 0.0),
        stroke_confidence=float(metrics.get("stroke_confidence") or 0.0),
        detector_input_sha256=detector_input_sha256,
        authorized_perceptual_source_identity=perceptual_source_identity,
        perceptual_axis_evidence=perceptual_axis_evidence,
        reason_codes=tuple(metrics.get("reason_codes") or ()),
        metrics=metrics,
    )


def _authorized_perceptual_source_identity(
    *,
    view: AuthorizedSourceStyleView,
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    detector_input_sha256: str,
) -> dict[str, Any]:
    return {
        "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
        "page_id": view.page_id,
        "view_id": view.view_id,
        "bundle_id": view.bundle_id,
        "parent_id": view.parent_id,
        "root_id": view.root_id,
        "content_bbox": list(view.content_bbox),
        "analysis_bbox": list(view.analysis_bbox),
        "cleanup_mask_ids": list(view.cleanup_mask_ids),
        "owned_component_ids": list(view.owned_component_ids),
        "crop_shape": [int(source_crop.shape[0]), int(source_crop.shape[1])],
        "authorized_mask_sha256": _array_sha256(
            np.ascontiguousarray(mask_crop, dtype=np.uint8)
        ),
        "authorized_pixel_sha256": _array_sha256(
            np.ascontiguousarray(source_crop[mask_crop], dtype=np.uint8)
        ),
        "detector_input_sha256": detector_input_sha256,
    }


def _perceptual_fact_set_id(source_identity: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(source_identity),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (
        TypeError,
        ValueError,
        UnicodeEncodeError,
        RecursionError,
        OverflowError,
    ):
        return ""
    return (
        f"{_PERCEPTUAL_STYLE_FACT_SET_PREFIX}"
        f"{hashlib.sha256(encoded).hexdigest()}"
    )


def _perceptual_axis_record(
    *,
    axis: str,
    fact_set_id: str,
    support_status: str,
    confidence: float,
    reason_codes: Sequence[str],
    support: Mapping[str, Any],
    uncertainty: Mapping[str, Any] | None = None,
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = support_status if support_status in {"supported", "ambiguous"} else "unavailable"
    record: dict[str, Any] = {
        "support_status": status,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 8),
        "provenance": _PERCEPTUAL_STYLE_PROVENANCE,
        "fact_set_id": fact_set_id,
        "reason_codes": _unique([str(item) for item in reason_codes if str(item)]),
        "support": _json_safe_mapping(support),
        "conflict": {
            "status": (
                "clear"
                if status == "supported"
                else "ambiguous"
                if status == "ambiguous"
                else "unavailable"
            )
        },
        "uncertainty": _json_safe_mapping(uncertainty),
    }
    if status == "supported" and value:
        record["value"] = _json_safe_mapping(value)
    return record


def _build_additive_perceptual_carrier(
    *,
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build positive-only sibling paint/effect facts without touching Task A.

    Spatial support comes only from the authorized mask.  Color is considered
    after a glyph-interior cohort is established, so a chromatic outline or
    backing cannot manufacture a fill core.  Outline is assessed independently
    from immutable source pixels and must form a stable multi-pixel concentric
    shell. Rotation is a separately isolated whole-parent geometry fact.
    Non-supported outcomes return an empty production carrier and preserve the
    accepted audit/style path.
    """

    try:
        fill_observed = _observe_additive_chromatic_fill(source_crop, mask_crop)
    except Exception:
        fill_observed = {
            "support_status": "unavailable",
            "confidence": 0.0,
            "reason_codes": ["perceptual_fill_observer_failed_closed"],
            "support": {},
            "uncertainty": {},
        }
    try:
        outline_observed = _observe_additive_outline(source_crop, mask_crop)
    except Exception:
        outline_observed = {
            "support_status": "unavailable",
            "confidence": 0.0,
            "reason_codes": ["perceptual_outline_observer_failed_closed"],
            "support": {},
            "uncertainty": {},
        }
    try:
        rotation_observed = _observe_additive_rotation(source_crop, mask_crop)
    except Exception:
        rotation_observed = {
            "support_status": "unavailable",
            "confidence": 0.0,
            "reason_codes": ["perceptual_rotation_observer_failed_closed"],
            "support": {},
            "uncertainty": {},
        }
    try:
        shadow_observed = _observe_additive_shadow(source_crop, mask_crop)
    except Exception:
        shadow_observed = {
            "support_status": "unavailable",
            "confidence": 0.0,
            "reason_codes": ["perceptual_shadow_observer_failed_closed"],
            "support": {},
            "uncertainty": {},
        }
    observed_by_axis: dict[str, Mapping[str, Any]] = {
        "fill": fill_observed,
        "outline": outline_observed,
        "rotation": rotation_observed,
        "shadow": shadow_observed,
    }
    if not any(
        observed.get("support_status") == "supported"
        for observed in observed_by_axis.values()
    ):
        return {}
    fact_set_id = _perceptual_fact_set_id(source_identity)
    if not fact_set_id:
        return {}
    records: dict[str, Any] = {}
    for axis in _PERCEPTUAL_STYLE_AXES:
        observed = observed_by_axis.get(axis)
        if observed is None:
            observed = {
                "support_status": "unavailable",
                "confidence": 0.0,
                "reason_codes": (f"perceptual_{axis}_producer_not_enabled",),
                "support": {},
                "uncertainty": {},
            }
        records[axis] = _perceptual_axis_record(
            axis=axis,
            fact_set_id=fact_set_id,
            support_status=str(observed.get("support_status") or "unavailable"),
            confidence=float(observed.get("confidence") or 0.0),
            reason_codes=tuple(observed.get("reason_codes") or ()),
            support=(
                observed.get("support")
                if isinstance(observed.get("support"), Mapping)
                else {}
            ),
            uncertainty=(
                observed.get("uncertainty")
                if isinstance(observed.get("uncertainty"), Mapping)
                else {}
            ),
            value=(
                observed.get("value")
                if isinstance(observed.get("value"), Mapping)
                else {}
            ),
        )
    return {
        "contract_version": _PERCEPTUAL_STYLE_AXES_VERSION,
        "source_identity": _json_safe_mapping(source_identity),
        "fact_set_id": fact_set_id,
        **records,
    }


def _observe_additive_chromatic_fill(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {
            "authorized_pixel_count": int(np.count_nonzero(mask)),
        },
        "uncertainty": {},
    }
    if source.ndim != 3 or source.shape[2] != 3 or mask.shape != source.shape[:2]:
        unavailable["reason_codes"].append("perceptual_fill_input_invalid")
        return unavailable
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count < _ADDITIVE_FILL_MIN_CLUSTER_PIXELS:
        unavailable["reason_codes"].append("perceptual_fill_authorized_support_too_small")
        return unavailable

    selected = source[mask].astype(np.float32)
    chroma_values = selected.max(axis=1) - selected.min(axis=1)
    chromatic = chroma_values >= _ADDITIVE_FILL_MIN_CHROMA
    chromatic_count = int(np.count_nonzero(chromatic))
    unavailable["support"].update(
        {
            "chromatic_pixel_count": chromatic_count,
            "chromatic_fraction": round(chromatic_count / pixel_count, 8),
        }
    )
    if chromatic_count < _ADDITIVE_FILL_MIN_CLUSTER_PIXELS:
        unavailable["reason_codes"].append("perceptual_fill_no_non_neutral_paint")
        return unavailable

    try:
        import cv2
    except Exception:
        unavailable["reason_codes"].append("perceptual_fill_spatial_backend_unavailable")
        return unavailable

    chromatic_pixels = selected[chromatic]
    all_quantized = np.clip(
        np.floor((selected + 8.0) / 16.0), 0, 15
    ).astype(np.uint8)
    _, _, all_paint_counts = np.unique(
        all_quantized, axis=0, return_inverse=True, return_counts=True
    )
    largest_paint_count = int(np.max(all_paint_counts))
    largest_paint_tie_count = int(
        np.count_nonzero(all_paint_counts == largest_paint_count)
    )
    unavailable["support"].update(
        {
            "authorized_paint_cluster_count": int(len(all_paint_counts)),
            "largest_authorized_paint_cohort_pixels": largest_paint_count,
            "largest_authorized_paint_cohort_fraction": round(
                largest_paint_count / pixel_count, 8
            ),
            "largest_authorized_paint_cohort_tie_count": largest_paint_tie_count,
        }
    )
    quantized = np.clip(
        np.floor((chromatic_pixels + 8.0) / 16.0), 0, 15
    ).astype(np.uint8)
    keys, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    order = sorted(
        range(len(counts)),
        key=lambda index: (
            -int(counts[index]),
            tuple(int(value) for value in keys[index]),
        ),
    )
    chromatic_flat_indices = np.flatnonzero(mask)[chromatic]
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    overall_depth = float(np.percentile(distance[mask], 75))
    cluster_facts: list[dict[str, Any]] = []
    for cluster_index in order:
        cluster_pixels = chromatic_pixels[inverse == cluster_index]
        count = int(cluster_pixels.shape[0])
        mask_fraction = count / max(1, pixel_count)
        chroma_fraction = count / max(1, chromatic_count)
        if (
            count < _ADDITIVE_FILL_MIN_CLUSTER_PIXELS
            or mask_fraction < _ADDITIVE_FILL_MIN_CLUSTER_MASK_FRACTION
            or chroma_fraction < _ADDITIVE_FILL_MIN_CORE_CHROMATIC_FRACTION
        ):
            continue
        median = np.median(cluster_pixels, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(cluster_pixels - median[None, :], axis=1))
        )
        cohort = np.zeros(mask.size, dtype=bool)
        cohort[chromatic_flat_indices[inverse == cluster_index]] = True
        cohort = cohort.reshape(mask.shape)
        yy, xx = np.where(cohort)
        width = int(xx.max() - xx.min() + 1)
        height = int(yy.max() - yy.min() + 1)
        occupancy = count / max(1, width * height)
        border_margin = _mask_border_margin(cohort)
        cohort_depth = float(np.percentile(distance[cohort], 75))
        depth_ratio = cohort_depth / max(overall_depth, 1e-6)
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            cohort.astype(np.uint8), connectivity=8
        )
        significant = int(
            sum(1 for row in stats[1:] if int(row[cv2.CC_STAT_AREA]) >= 8)
        )
        core_like = bool(
            dispersion <= _ADDITIVE_FILL_MAX_COLOR_DISPERSION
            and occupancy <= _ADDITIVE_FILL_MAX_GLYPH_BBOX_OCCUPANCY
            and border_margin >= _ADDITIVE_FILL_MIN_CORE_BORDER_MARGIN_PX
            and depth_ratio >= _ADDITIVE_FILL_MIN_CORE_DEPTH_RATIO
            and significant >= 2
        )
        cluster_facts.append(
            {
                "color": _rgb_hex(median),
                "pixel_count": count,
                "mask_fraction": round(mask_fraction, 8),
                "chromatic_fraction": round(chroma_fraction, 8),
                "color_dispersion_rgb": round(dispersion, 8),
                "bbox_occupancy": round(occupancy, 8),
                "border_margin_px": border_margin,
                "depth_p75_px": round(cohort_depth, 8),
                "depth_ratio": round(depth_ratio, 8),
                "significant_component_count": significant,
                "core_like": core_like,
            }
        )

    candidates = [item for item in cluster_facts if bool(item.get("core_like"))]
    support = {
        **dict(unavailable["support"]),
        "authorized_depth_p75_px": round(overall_depth, 8),
        "chromatic_cluster_count": int(len(counts)),
        "spatially_analyzed_chromatic_cluster_count": len(cluster_facts),
        "cluster_facts": cluster_facts,
    }
    if not candidates:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_fill_no_spatially_supported_chromatic_core"
        )
        return unavailable
    candidates.sort(
        key=lambda item: (
            -float(item.get("depth_p75_px") or 0.0),
            -int(item.get("significant_component_count") or 0),
            -int(item.get("pixel_count") or 0),
            str(item.get("color") or ""),
        )
    )
    selected_core = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        depth_delta = float(selected_core["depth_p75_px"]) - float(
            runner_up["depth_p75_px"]
        )
        depth_separation = float(selected_core["depth_p75_px"]) / max(
            float(runner_up["depth_p75_px"]), 1e-6
        )
        support.update(
            {
                "core_runner_up_depth_delta_px": round(depth_delta, 8),
                "core_runner_up_depth_ratio": round(depth_separation, 8),
            }
        )
        if (
            depth_delta < _ADDITIVE_FILL_MIN_CORE_DEPTH_SEPARATION_PX
            and depth_separation < _ADDITIVE_FILL_MIN_CORE_DEPTH_SEPARATION_RATIO
        ):
            return {
                "support_status": "ambiguous",
                "confidence": 0.0,
                "reason_codes": [
                    "perceptual_fill_competing_spatially_core_like_paint_roles"
                ],
                "support": support,
                "uncertainty": {
                    "core_depth_delta_px": round(depth_delta, 8),
                    "core_depth_ratio": round(depth_separation, 8),
                },
            }

    selected_paint_count = int(selected_core.get("pixel_count") or 0)
    support["selected_core_authorized_paint_fraction"] = round(
        selected_paint_count / max(1, pixel_count), 8
    )
    if (
        selected_paint_count < largest_paint_count
        or (
            selected_paint_count == largest_paint_count
            and largest_paint_tie_count > 1
        )
    ):
        return {
            "support_status": "ambiguous",
            "confidence": 0.0,
            "reason_codes": [
                "perceptual_fill_chromatic_role_not_unique_dominant_authorized_paint"
            ],
            "support": support,
            "uncertainty": {
                "selected_core_pixel_count": selected_paint_count,
                "largest_authorized_paint_cohort_pixels": largest_paint_count,
                "largest_authorized_paint_cohort_tie_count": (
                    largest_paint_tie_count
                ),
            },
        }

    selected_fraction = float(selected_core.get("chromatic_fraction") or 0.0)
    selected_depth_ratio = float(selected_core.get("depth_ratio") or 0.0)
    selected_components = int(selected_core.get("significant_component_count") or 0)
    support["selected_core"] = dict(selected_core)
    return {
        "support_status": "supported",
        "value": {"color": str(selected_core.get("color") or "")},
        "confidence": round(
            min(
                0.98,
                0.50
                + 0.18 * selected_fraction
                + 0.12 * min(1.0, selected_depth_ratio)
                + 0.10 * min(1.0, selected_components / 4.0),
            ),
            8,
        ),
        "reason_codes": ["perceptual_fill_unique_chromatic_character_core"],
        "support": support,
        "uncertainty": {
            "color_dispersion_rgb": float(
                selected_core.get("color_dispersion_rgb") or 0.0
            )
        },
    }


def _observe_additive_rotation(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    """Return one pronounced, stable whole-parent rotation fact.

    Rotation is estimated from stable authorized paint cohorts rather than the
    full union mask. This lets a complete character core vote even when an
    optional shadow reaches the crop edge, while clipped core paint, symmetric
    shapes, upright or merely italic glyphs, and conflicting geometry fail
    closed. The observer never decides or changes writing mode.
    """

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    pixel_count = int(np.count_nonzero(mask))
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {"authorized_pixel_count": pixel_count},
        "uncertainty": {},
    }
    if source.ndim != 3 or source.shape[2] != 3 or mask.shape != source.shape[:2]:
        unavailable["reason_codes"].append("perceptual_rotation_input_invalid")
        return unavailable
    if pixel_count < _ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS:
        unavailable["reason_codes"].append(
            "perceptual_rotation_authorized_support_too_small"
        )
        return unavailable

    try:
        import cv2
    except Exception:
        unavailable["reason_codes"].append(
            "perceptual_rotation_spatial_backend_unavailable"
        )
        return unavailable

    selected = source[mask].astype(np.float32)
    quantized = np.clip(np.floor((selected + 8.0) / 16.0), 0, 15).astype(
        np.uint8
    )
    keys, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    eligible = [
        index
        for index in sorted(
            range(len(counts)),
            key=lambda item: (
                -int(counts[item]),
                tuple(int(value) for value in keys[item]),
            ),
        )
        if int(counts[index]) >= _ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS
        and int(counts[index]) / max(1, pixel_count)
        >= _ADDITIVE_ROTATION_MIN_CLUSTER_MASK_FRACTION
    ][:_ADDITIVE_ROTATION_MAX_COHORTS]
    unavailable["support"].update(
        {
            "authorized_paint_cluster_count": int(len(counts)),
            "eligible_paint_cluster_count": len(eligible),
        }
    )
    if not eligible:
        unavailable["reason_codes"].append(
            "perceptual_rotation_stable_character_paint_unavailable"
        )
        return unavailable

    flat_indices = np.flatnonzero(mask)
    candidate_facts: list[dict[str, Any]] = []
    rejected_facts: list[dict[str, Any]] = []
    for cluster_index in eligible:
        members = inverse == cluster_index
        pixels = selected[members]
        median = np.median(pixels, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(pixels - median[None, :], axis=1))
        )
        cohort = np.zeros(mask.size, dtype=bool)
        cohort[flat_indices[members]] = True
        cohort = cohort.reshape(mask.shape)
        fact: dict[str, Any] = {
            "color": _rgb_hex(median),
            "pixel_count": int(np.count_nonzero(cohort)),
            "mask_fraction": round(
                int(np.count_nonzero(cohort)) / max(1, pixel_count), 8
            ),
            "color_dispersion_rgb": round(dispersion, 8),
            "border_margin_px": _mask_border_margin(cohort),
        }
        if dispersion > _ADDITIVE_ROTATION_MAX_COLOR_DISPERSION:
            fact["rejection"] = "unstable_paint"
            rejected_facts.append(fact)
            continue

        _, _, stats, _ = cv2.connectedComponentsWithStats(
            cohort.astype(np.uint8), connectivity=8
        )
        significant_components = int(
            sum(
                1
                for row in stats[1:]
                if int(row[cv2.CC_STAT_AREA])
                >= _ADDITIVE_ROTATION_MIN_COMPONENT_PIXELS
            )
        )
        fact["significant_component_count"] = significant_components
        if significant_components < _ADDITIVE_ROTATION_MIN_COMPONENTS:
            fact["rejection"] = "insufficient_character_components"
            rejected_facts.append(fact)
            continue
        if int(fact["border_margin_px"]) < _ADDITIVE_ROTATION_MIN_BORDER_MARGIN_PX:
            fact["rejection"] = "character_core_truncated"
            rejected_facts.append(fact)
            continue

        yy, xx = np.where(cohort)
        points = np.column_stack((xx, yy)).astype(np.float32)
        (_, _), (rect_width, rect_height), rect_angle = cv2.minAreaRect(points)
        major = max(float(rect_width), float(rect_height))
        minor = min(float(rect_width), float(rect_height))
        degrees = (
            float(rect_angle)
            if float(rect_width) >= float(rect_height)
            else float(rect_angle) - 90.0
        )
        aspect_ratio = major / max(minor, 1e-6)
        bbox_occupancy = int(np.count_nonzero(cohort)) / max(
            1.0, float(rect_width) * float(rect_height)
        )
        fact.update(
            {
                "degrees_clockwise": round(degrees, 8),
                "oriented_aspect_ratio": round(aspect_ratio, 8),
                "oriented_bbox_occupancy": round(bbox_occupancy, 8),
            }
        )
        if aspect_ratio < _ADDITIVE_ROTATION_MIN_ASPECT_RATIO:
            fact["rejection"] = "symmetric_or_non_directional_geometry"
            rejected_facts.append(fact)
            continue
        if not (
            _ADDITIVE_ROTATION_MIN_BBOX_OCCUPANCY
            <= bbox_occupancy
            <= _ADDITIVE_ROTATION_MAX_BBOX_OCCUPANCY
        ):
            fact["rejection"] = "non_character_bbox_occupancy"
            rejected_facts.append(fact)
            continue
        if abs(degrees) < _ADDITIVE_ROTATION_MIN_ABS_DEGREES:
            fact["rejection"] = "upright_or_italic_only_geometry"
            rejected_facts.append(fact)
            continue
        if abs(degrees) > _ADDITIVE_ROTATION_MAX_ABS_DEGREES:
            fact["rejection"] = "base_axis_or_rotation_ambiguous"
            rejected_facts.append(fact)
            continue

        eroded = cv2.erode(
            cohort.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool)
        if int(np.count_nonzero(eroded)) < _ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS:
            fact["rejection"] = "eroded_character_geometry_unavailable"
            rejected_facts.append(fact)
            continue
        eroded_y, eroded_x = np.where(eroded)
        eroded_points = np.column_stack((eroded_x, eroded_y)).astype(np.float32)
        (_, _), (eroded_width, eroded_height), eroded_angle = cv2.minAreaRect(
            eroded_points
        )
        eroded_degrees = (
            float(eroded_angle)
            if float(eroded_width) >= float(eroded_height)
            else float(eroded_angle) - 90.0
        )
        raw_delta = abs(degrees - eroded_degrees)
        erosion_delta = min(raw_delta, abs(raw_delta - 90.0))
        fact.update(
            {
                "eroded_degrees_clockwise": round(eroded_degrees, 8),
                "erosion_angle_delta_degrees": round(erosion_delta, 8),
            }
        )
        if erosion_delta > _ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES:
            fact["rejection"] = "rotation_not_stable_under_core_erosion"
            rejected_facts.append(fact)
            continue
        candidate_facts.append(fact)

    support = dict(unavailable["support"])
    support.update(
        {
            "candidate_count": len(candidate_facts),
            "candidate_facts": candidate_facts,
            "rejected_cohort_facts": rejected_facts,
        }
    )
    if not candidate_facts:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_no_unambiguous_character_core_geometry"
        )
        return unavailable

    candidate_degrees = [
        float(item["degrees_clockwise"]) for item in candidate_facts
    ]
    spread = max(candidate_degrees) - min(candidate_degrees)
    support["candidate_spread_degrees"] = round(spread, 8)
    if spread > _ADDITIVE_ROTATION_MAX_CANDIDATE_SPREAD_DEGREES:
        return {
            "support_status": "ambiguous",
            "confidence": 0.0,
            "reason_codes": ["perceptual_rotation_conflicting_character_axes"],
            "support": support,
            "uncertainty": {"candidate_spread_degrees": round(spread, 8)},
        }

    weights = np.asarray(
        [max(1, int(item["pixel_count"])) for item in candidate_facts],
        dtype=np.float64,
    )
    resolved_degrees = float(
        np.average(np.asarray(candidate_degrees, dtype=np.float64), weights=weights)
    )
    minimum_aspect = min(
        float(item["oriented_aspect_ratio"]) for item in candidate_facts
    )
    maximum_erosion_delta = max(
        float(item["erosion_angle_delta_degrees"]) for item in candidate_facts
    )
    confidence = min(
        0.98,
        0.66
        + 0.12 * min(1.0, (minimum_aspect - 1.0) / 2.0)
        + 0.12
        * max(
            0.0,
            1.0
            - maximum_erosion_delta
            / max(_ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES, 1e-6),
        )
        + 0.06 * min(1.0, abs(resolved_degrees) / 24.0),
    )
    return {
        "support_status": "supported",
        "confidence": round(confidence, 8),
        "reason_codes": [
            "perceptual_rotation_stable_whole_parent_character_axis"
        ],
        "support": support,
        "uncertainty": {
            "candidate_spread_degrees": round(spread, 8),
            "maximum_erosion_delta_degrees": round(maximum_erosion_delta, 8),
        },
        "value": {
            "degrees_clockwise": round(resolved_degrees, 8),
            "pivot": "visual_center",
        },
    }


def _observe_additive_outline(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    """Return one independently supported concentric source outline.

    The observer is intentionally conservative. It only accepts two stable
    authorized paint cohorts when one is a multi-component glyph interior and
    the other accounts for nearly all of a multi-pixel morphological ring.
    Color alone never establishes either role.
    """

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    pixel_count = int(np.count_nonzero(mask))
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {"authorized_pixel_count": pixel_count},
        "uncertainty": {},
    }
    if source.ndim != 3 or source.shape[2] != 3 or mask.shape != source.shape[:2]:
        unavailable["reason_codes"].append("perceptual_outline_input_invalid")
        return unavailable
    if pixel_count < _ADDITIVE_OUTLINE_MIN_CLUSTER_PIXELS * 2:
        unavailable["reason_codes"].append(
            "perceptual_outline_authorized_support_too_small"
        )
        return unavailable
    border_margin = _mask_border_margin(mask)
    unavailable["support"]["authorized_border_margin_px"] = border_margin
    if border_margin < _ADDITIVE_OUTLINE_MIN_BORDER_MARGIN_PX:
        unavailable["reason_codes"].append(
            "perceptual_outline_authorized_support_truncated"
        )
        return unavailable

    try:
        import cv2
    except Exception:
        unavailable["reason_codes"].append(
            "perceptual_outline_spatial_backend_unavailable"
        )
        return unavailable

    selected = source[mask].astype(np.float32)
    quantized = np.clip(np.floor((selected + 8.0) / 16.0), 0, 15).astype(
        np.uint8
    )
    keys, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    eligible = [
        index
        for index in sorted(
            range(len(counts)),
            key=lambda item: (
                -int(counts[item]),
                tuple(int(value) for value in keys[item]),
            ),
        )
        if int(counts[index]) >= _ADDITIVE_OUTLINE_MIN_CLUSTER_PIXELS
        and int(counts[index]) / max(1, pixel_count)
        >= _ADDITIVE_OUTLINE_MIN_CLUSTER_MASK_FRACTION
    ][:_ADDITIVE_OUTLINE_MAX_COHORTS]
    unavailable["support"].update(
        {
            "authorized_paint_cluster_count": int(len(counts)),
            "eligible_paint_cluster_count": len(eligible),
        }
    )
    if len(eligible) < 2:
        unavailable["reason_codes"].append(
            "perceptual_outline_stable_paint_pair_unavailable"
        )
        return unavailable

    authorized_distance = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    flat_indices = np.flatnonzero(mask)
    cohorts: list[dict[str, Any]] = []
    for cluster_index in eligible:
        members = inverse == cluster_index
        pixels = selected[members]
        median = np.median(pixels, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(pixels - median[None, :], axis=1))
        )
        if dispersion > _ADDITIVE_OUTLINE_MAX_COLOR_DISPERSION:
            continue
        cohort = np.zeros(mask.size, dtype=bool)
        cohort[flat_indices[members]] = True
        cohort = cohort.reshape(mask.shape)
        yy, xx = np.where(cohort)
        width = int(xx.max() - xx.min() + 1)
        height = int(yy.max() - yy.min() + 1)
        occupancy = int(np.count_nonzero(cohort)) / max(1, width * height)
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            cohort.astype(np.uint8), connectivity=8
        )
        significant_components = int(
            sum(1 for row in stats[1:] if int(row[cv2.CC_STAT_AREA]) >= 8)
        )
        cohorts.append(
            {
                "mask": cohort,
                "median": median,
                "color": _rgb_hex(median),
                "pixel_count": int(np.count_nonzero(cohort)),
                "mask_fraction": round(
                    int(np.count_nonzero(cohort)) / max(1, pixel_count), 8
                ),
                "color_dispersion_rgb": round(dispersion, 8),
                "bbox_occupancy": round(occupancy, 8),
                "border_margin_px": _mask_border_margin(cohort),
                "depth_p50_px": round(
                    float(np.percentile(authorized_distance[cohort], 50)), 8
                ),
                "significant_component_count": significant_components,
            }
        )
    unavailable["support"]["stable_paint_cohort_count"] = len(cohorts)
    unavailable["support"]["cohort_facts"] = [
        {key: value for key, value in cohort.items() if key not in {"mask", "median"}}
        for cohort in cohorts
    ]
    if len(cohorts) < 2:
        unavailable["reason_codes"].append(
            "perceptual_outline_stable_paint_pair_unavailable"
        )
        return unavailable

    candidates: list[dict[str, Any]] = []
    thin_shell_seen = False
    for core in cohorts:
        if (
            int(core["significant_component_count"])
            < _ADDITIVE_OUTLINE_MIN_CORE_COMPONENTS
            or float(core["bbox_occupancy"])
            > _ADDITIVE_OUTLINE_MAX_CORE_BBOX_OCCUPANCY
            or int(core["border_margin_px"])
            < _ADDITIVE_OUTLINE_MIN_BORDER_MARGIN_PX
        ):
            continue
        core_mask = np.asarray(core["mask"], dtype=bool)
        distance_to_core = cv2.distanceTransform(
            (~core_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
        for shell in cohorts:
            if shell is core:
                continue
            color_distance = float(
                np.linalg.norm(
                    np.asarray(core["median"], dtype=np.float32)
                    - np.asarray(shell["median"], dtype=np.float32)
                )
            )
            if color_distance < _ADDITIVE_OUTLINE_MIN_COLOR_DISTANCE:
                continue
            shell_mask = np.asarray(shell["mask"], dtype=bool)
            pair_fraction = int(np.count_nonzero(core_mask | shell_mask)) / max(
                1, pixel_count
            )
            if pair_fraction < _ADDITIVE_OUTLINE_MIN_PAIR_MASK_FRACTION:
                continue
            depth_delta = float(core["depth_p50_px"]) - float(
                shell["depth_p50_px"]
            )
            if depth_delta < _ADDITIVE_OUTLINE_MIN_CORE_SHELL_DEPTH_DELTA_PX:
                continue
            radial_values = distance_to_core[shell_mask]
            if radial_values.size <= 0:
                continue
            radial_p90 = float(np.percentile(radial_values, 90))
            if radial_p90 <= _ADDITIVE_OUTLINE_MIN_RADIAL_WIDTH_PX:
                thin_shell_seen = True
                continue
            if radial_p90 > _ADDITIVE_OUTLINE_MAX_RADIAL_WIDTH_PX:
                continue
            radius = max(2, int(round(radial_p90)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            )
            predicted_ring = (
                cv2.dilate(core_mask.astype(np.uint8), kernel).astype(bool)
                & ~core_mask
            )
            intersection = int(np.count_nonzero(predicted_ring & shell_mask))
            shell_recall = intersection / max(1, int(np.count_nonzero(shell_mask)))
            ring_precision = intersection / max(
                1, int(np.count_nonzero(predicted_ring))
            )
            if (
                shell_recall < _ADDITIVE_OUTLINE_MIN_SHELL_RING_RECALL
                or ring_precision < _ADDITIVE_OUTLINE_MIN_RING_SHELL_PRECISION
            ):
                continue
            candidates.append(
                {
                    "color": str(shell["color"]),
                    "width_px": radial_p90,
                    "core_color": str(core["color"]),
                    "pair_mask_fraction": pair_fraction,
                    "core_shell_depth_delta_px": depth_delta,
                    "radial_distance_p90_px": radial_p90,
                    "shell_ring_recall": shell_recall,
                    "ring_shell_precision": ring_precision,
                    "color_distance_rgb": color_distance,
                }
            )

    support = dict(unavailable["support"])
    support["candidate_count"] = len(candidates)
    support["thin_shell_seen"] = thin_shell_seen
    if not candidates:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_outline_native_antialias_band_rejected"
            if thin_shell_seen
            else "perceptual_outline_no_independently_supported_concentric_shell"
        )
        return unavailable
    if len(candidates) != 1:
        support["candidate_facts"] = [
            {
                key: round(float(value), 8) if isinstance(value, float) else value
                for key, value in candidate.items()
            }
            for candidate in candidates
        ]
        return {
            "support_status": "ambiguous",
            "confidence": 0.0,
            "reason_codes": ["perceptual_outline_competing_concentric_shells"],
            "support": support,
            "uncertainty": {"candidate_count": len(candidates)},
        }

    candidate = candidates[0]
    support.update(
        {
            "radial_distance_p90_px": round(
                float(candidate["radial_distance_p90_px"]), 8
            ),
            "pair_mask_fraction": round(float(candidate["pair_mask_fraction"]), 8),
            "core_shell_depth_delta_px": round(
                float(candidate["core_shell_depth_delta_px"]), 8
            ),
            "shell_ring_recall": round(float(candidate["shell_ring_recall"]), 8),
            "ring_shell_precision": round(
                float(candidate["ring_shell_precision"]), 8
            ),
            "core_color": str(candidate["core_color"]),
        }
    )
    geometric_support = min(
        float(candidate["pair_mask_fraction"]),
        float(candidate["shell_ring_recall"]),
        float(candidate["ring_shell_precision"]),
    )
    return {
        "support_status": "supported",
        "confidence": round(min(0.98, 0.62 + 0.34 * geometric_support), 8),
        "reason_codes": ["perceptual_outline_stable_concentric_shell"],
        "support": support,
        "uncertainty": {
            "color_dispersion_rgb": next(
                (
                    float(item["color_dispersion_rgb"])
                    for item in cohorts
                    if str(item["color"]) == str(candidate["color"])
                ),
                0.0,
            )
        },
        "value": {
            "color": str(candidate["color"]),
            "width_px": round(float(candidate["width_px"]), 8),
        },
    }


def _observe_additive_shadow(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    """Return one complete displaced glyph-correlated shadow.

    A previously supported chromatic character core supplies only the runtime
    shape used for correlation. One darker authorized effect must be explained
    by one displaced copy of that shape. Blur is then estimated from the
    spatial support extending beyond the displaced core, not from RGB
    dispersion. Concentric, centered, repeated, clipped, or ambiguous support
    remains unavailable.
    """

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    pixel_count = int(np.count_nonzero(mask))
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {"authorized_pixel_count": pixel_count},
        "uncertainty": {},
    }
    if source.ndim != 3 or source.shape[2] != 3 or mask.shape != source.shape[:2]:
        unavailable["reason_codes"].append("perceptual_shadow_input_invalid")
        return unavailable
    if pixel_count < _ADDITIVE_SHADOW_MIN_EFFECT_PIXELS * 2:
        unavailable["reason_codes"].append(
            "perceptual_shadow_authorized_support_too_small"
        )
        return unavailable

    try:
        import cv2
    except Exception:
        unavailable["reason_codes"].append(
            "perceptual_shadow_spatial_backend_unavailable"
        )
        return unavailable

    fill = _observe_additive_chromatic_fill(source, mask)
    if fill.get("support_status") != "supported":
        unavailable["reason_codes"].append(
            "perceptual_shadow_character_core_unavailable"
        )
        unavailable["support"]["character_core_status"] = str(
            fill.get("support_status") or "unavailable"
        )
        return unavailable
    fill_value = fill.get("value") if isinstance(fill.get("value"), Mapping) else {}
    fill_color = str(fill_value.get("color") or "")
    if len(fill_color) != 7 or not fill_color.startswith("#"):
        unavailable["reason_codes"].append(
            "perceptual_shadow_character_core_color_invalid"
        )
        return unavailable
    try:
        fill_rgb = np.asarray(
            [int(fill_color[index : index + 2], 16) for index in (1, 3, 5)],
            dtype=np.float32,
        )
    except (TypeError, ValueError):
        unavailable["reason_codes"].append(
            "perceptual_shadow_character_core_color_invalid"
        )
        return unavailable

    color_distance = np.linalg.norm(
        source.astype(np.float32) - fill_rgb[None, None, :], axis=2
    )
    core = mask & (color_distance <= _ADDITIVE_SHADOW_CORE_COLOR_DISTANCE)
    core_count = int(np.count_nonzero(core))
    effect = mask & ~core
    effect_count = int(np.count_nonzero(effect))
    effect_fraction = effect_count / max(1, pixel_count)
    effect_border_margin = _mask_border_margin(effect)
    unavailable["support"].update(
        {
            "character_core_color": fill_color,
            "character_core_pixel_count": core_count,
            "effect_pixel_count": effect_count,
            "effect_mask_fraction": round(effect_fraction, 8),
            "effect_border_margin_px": effect_border_margin,
        }
    )
    if core_count < _ADDITIVE_SHADOW_MIN_EFFECT_PIXELS:
        unavailable["reason_codes"].append(
            "perceptual_shadow_character_core_support_too_small"
        )
        return unavailable
    if (
        effect_count < _ADDITIVE_SHADOW_MIN_EFFECT_PIXELS
        or effect_fraction < _ADDITIVE_SHADOW_MIN_EFFECT_MASK_FRACTION
    ):
        unavailable["reason_codes"].append(
            "perceptual_shadow_displaced_effect_unavailable"
        )
        return unavailable
    if effect_border_margin < _ADDITIVE_SHADOW_MIN_EFFECT_BORDER_MARGIN_PX:
        unavailable["reason_codes"].append(
            "perceptual_shadow_effect_support_truncated"
        )
        return unavailable

    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    effect_luma = luma[effect]
    effect_luma_iqr = float(
        np.percentile(effect_luma, 75) - np.percentile(effect_luma, 25)
    )
    if effect_luma_iqr <= _ADDITIVE_SHADOW_UNIFORM_LUMA_IQR:
        central = effect.copy()
        central_percentile = 100.0
    else:
        central_threshold = float(
            np.percentile(effect_luma, _ADDITIVE_SHADOW_CENTRAL_LUMA_PERCENTILE)
        )
        central = effect & (luma <= central_threshold + 1e-6)
        central_percentile = _ADDITIVE_SHADOW_CENTRAL_LUMA_PERCENTILE
    central_count = int(np.count_nonzero(central))
    core_luma_median = float(np.median(luma[core]))
    central_luma_median = (
        float(np.median(luma[central])) if central_count else core_luma_median
    )
    unavailable["support"].update(
        {
            "effect_luma_iqr": round(effect_luma_iqr, 8),
            "central_effect_percentile": central_percentile,
            "central_effect_pixel_count": central_count,
            "character_core_luma_median": round(core_luma_median, 8),
            "central_effect_luma_median": round(central_luma_median, 8),
        }
    )
    if central_count < _ADDITIVE_SHADOW_MIN_EFFECT_PIXELS:
        unavailable["reason_codes"].append(
            "perceptual_shadow_central_effect_support_too_small"
        )
        return unavailable
    if (
        core_luma_median - central_luma_median
        < _ADDITIVE_SHADOW_MIN_CORE_EFFECT_LUMA_DELTA
    ):
        unavailable["reason_codes"].append(
            "perceptual_shadow_effect_role_not_darker_than_character_core"
        )
        return unavailable

    core_y, core_x = np.where(core)
    x0 = int(core_x.min())
    x1 = int(core_x.max()) + 1
    y0 = int(core_y.min())
    y1 = int(core_y.max()) + 1
    template = core[y0:y1, x0:x1].astype(np.float32)
    correlation = cv2.matchTemplate(
        central.astype(np.float32), template, cv2.TM_CCORR
    )
    peaks: list[dict[str, Any]] = []
    minimum_offset_sq = _ADDITIVE_SHADOW_MIN_OFFSET_PX**2
    maximum_offset_sq = _ADDITIVE_SHADOW_MAX_OFFSET_PX**2
    separation_sq = _ADDITIVE_SHADOW_COMPETING_PEAK_DISTANCE_PX**2
    for flat_index in np.argsort(correlation.ravel())[::-1]:
        match_y, match_x = np.unravel_index(int(flat_index), correlation.shape)
        dx = int(match_x) - x0
        dy = int(match_y) - y0
        offset_sq = float(dx * dx + dy * dy)
        if offset_sq < minimum_offset_sq or offset_sq > maximum_offset_sq:
            continue
        overlap = float(correlation[match_y, match_x])
        if overlap <= 0:
            break
        if any(
            (dx - int(item["dx"])) ** 2 + (dy - int(item["dy"])) ** 2
            < separation_sq
            for item in peaks
        ):
            continue
        peaks.append({"dx": dx, "dy": dy, "overlap_pixels": int(round(overlap))})
        if len(peaks) >= 6:
            break
    unavailable["support"]["correlation_peaks"] = peaks
    if not peaks:
        unavailable["reason_codes"].append(
            "perceptual_shadow_displaced_correlation_unavailable"
        )
        return unavailable

    best = peaks[0]
    best_overlap = int(best["overlap_pixels"])
    central_explained = best_overlap / max(1, central_count)
    competing_ratio = (
        float(peaks[1]["overlap_pixels"]) / max(1, best_overlap)
        if len(peaks) > 1
        else 0.0
    )
    unavailable["support"].update(
        {
            "central_effect_explained_fraction": round(central_explained, 8),
            "competing_peak_ratio": round(competing_ratio, 8),
        }
    )
    if central_explained < _ADDITIVE_SHADOW_MIN_CENTRAL_EXPLAINED_FRACTION:
        unavailable["reason_codes"].append(
            "perceptual_shadow_not_explained_by_one_displaced_character_copy"
        )
        return unavailable
    if competing_ratio >= _ADDITIVE_SHADOW_COMPETING_PEAK_RATIO:
        return {
            "support_status": "ambiguous",
            "confidence": 0.0,
            "reason_codes": ["perceptual_shadow_competing_displaced_offsets"],
            "support": dict(unavailable["support"]),
            "uncertainty": {"competing_peak_ratio": round(competing_ratio, 8)},
        }

    dx = int(best["dx"])
    dy = int(best["dy"])
    shifted_core = np.zeros_like(core, dtype=bool)
    source_y0 = max(0, -dy)
    source_y1 = min(core.shape[0], core.shape[0] - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(core.shape[1], core.shape[1] - dx)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        unavailable["reason_codes"].append(
            "perceptual_shadow_displaced_core_outside_crop"
        )
        return unavailable
    shifted_core[
        source_y0 + dy : source_y1 + dy,
        source_x0 + dx : source_x1 + dx,
    ] = core[source_y0:source_y1, source_x0:source_x1]

    distance_from_shifted = cv2.distanceTransform(
        (~shifted_core).astype(np.uint8), cv2.DIST_L2, 5
    )
    outside_values = distance_from_shifted[effect & ~shifted_core]
    spread_p90 = (
        float(np.percentile(outside_values, 90)) if outside_values.size else 0.0
    )
    spread_p95 = (
        float(np.percentile(outside_values, 95)) if outside_values.size else 0.0
    )
    spread_radius = int(min(
        _ADDITIVE_SHADOW_MAX_SPREAD_RADIUS_PX,
        max(0, int(np.ceil(spread_p95))),
    ))
    if spread_radius > 0:
        predicted_support = cv2.dilate(
            shifted_core.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (spread_radius * 2 + 1, spread_radius * 2 + 1),
            ),
        ).astype(bool)
    else:
        predicted_support = shifted_core
    predicted_visible = predicted_support & ~core
    spatial_intersection = int(np.count_nonzero(predicted_visible & effect))
    spatial_recall = spatial_intersection / max(1, effect_count)
    spatial_precision = spatial_intersection / max(
        1, int(np.count_nonzero(predicted_visible))
    )
    unavailable["support"].update(
        {
            "offset_px": [dx, dy],
            "spatial_spread_p90_px": round(spread_p90, 8),
            "spatial_spread_p95_px": round(spread_p95, 8),
            "predicted_spread_radius_px": spread_radius,
            "spatial_effect_recall": round(spatial_recall, 8),
            "spatial_effect_precision": round(spatial_precision, 8),
        }
    )
    if (
        spatial_recall < _ADDITIVE_SHADOW_MIN_SPATIAL_RECALL
        or spatial_precision < _ADDITIVE_SHADOW_MIN_SPATIAL_PRECISION
    ):
        unavailable["reason_codes"].append(
            "perceptual_shadow_spatial_spread_not_glyph_correlated"
        )
        return unavailable

    if outside_values.size == 0 or spread_p90 <= 0.75:
        blur_radius = 0.0
    else:
        blur_radius = min(
            float(_ADDITIVE_SHADOW_MAX_SPREAD_RADIUS_PX),
            spread_p90 / _ADDITIVE_SHADOW_BLUR_SPREAD_DIVISOR,
        )
    darkest_threshold = float(np.percentile(luma[central], 10))
    darkest = central & (luma <= darkest_threshold + 1e-6)
    shadow_pixels = source[darkest] if np.any(darkest) else source[central]
    shadow_color = _rgb_hex(np.median(shadow_pixels.astype(np.float32), axis=0))
    geometric_support = min(
        central_explained,
        spatial_recall,
        spatial_precision,
    )
    return {
        "support_status": "supported",
        "confidence": round(min(0.98, 0.62 + 0.34 * geometric_support), 8),
        "reason_codes": [
            "perceptual_shadow_single_displaced_glyph_correlated_effect"
        ],
        "support": dict(unavailable["support"]),
        "uncertainty": {
            "competing_peak_ratio": round(competing_ratio, 8),
            "spatial_blur_support_p90_px": round(spread_p90, 8),
        },
        "value": {
            "color": shadow_color,
            "offset_px": [float(dx), float(dy)],
            "blur_radius_px": round(float(blur_radius), 8),
        },
    }


def _mask_border_margin(mask: np.ndarray) -> int:
    yy, xx = np.where(mask)
    if xx.size <= 0:
        return 0
    return int(
        min(
            int(xx.min()),
            int(yy.min()),
            int(mask.shape[1] - 1 - xx.max()),
            int(mask.shape[0] - 1 - yy.max()),
        )
    )


def _rgb_hex(value: Any) -> str:
    rgb = np.clip(np.rint(np.asarray(value, dtype=np.float32)), 0, 255).astype(
        np.uint8
    )
    if rgb.size != 3:
        return ""
    return "#" + "".join(f"{int(channel):02X}" for channel in rgb.tolist())


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
    legacy_vertical_size, legacy_vertical_confidence = _source_cell_size_from_geometry(
        x_spans,
        component_size=component_size,
        component_count=component_count,
        component_mad=component_mad,
    )
    legacy_horizontal_size, legacy_horizontal_confidence = _source_cell_size_from_geometry(
        y_spans,
        component_size=component_size,
        component_count=component_count,
        component_mad=component_mad,
    )
    fill_component_sizes = _fill_component_cell_sizes(fill)
    (
        vertical_size,
        vertical_confidence,
        vertical_support,
        vertical_qualification,
    ) = _qualify_source_cell_measurement(
        fill,
        axis=0,
        spans=x_spans,
        legacy_size=legacy_vertical_size,
        legacy_confidence=legacy_vertical_confidence,
        fill_component_sizes=fill_component_sizes,
    )
    (
        horizontal_size,
        horizontal_confidence,
        horizontal_support,
        horizontal_qualification,
    ) = _qualify_source_cell_measurement(
        fill,
        axis=1,
        spans=y_spans,
        legacy_size=legacy_horizontal_size,
        legacy_confidence=legacy_horizontal_confidence,
        fill_component_sizes=fill_component_sizes,
    )
    if vertical_size > 0 or horizontal_size > 0:
        reasons.append("source_cell_scale_measured_from_authorized_geometry")
    else:
        reasons.append("source_cell_scale_unavailable")
    reasons.extend(
        reason
        for reason in (
            vertical_support,
            horizontal_support,
        )
        if reason
    )

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
    outlined_weight_qualification: dict[str, Any] = {
        "status": "not_applicable_no_visible_support_transition",
        "directions": {},
    }
    ink_weight_class_vertical = ""
    ink_weight_confidence_vertical = 0.0
    ink_weight_support_vertical = ""
    ink_weight_class_horizontal = ""
    ink_weight_confidence_horizontal = 0.0
    ink_weight_support_horizontal = ""
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
        _, _, outlined_weight_qualification = _qualify_outlined_ink_weight(
            fill,
            source_cell_size_vertical_px=vertical_size,
            source_cell_size_horizontal_px=horizontal_size,
        )
        outlined_directions = dict(
            outlined_weight_qualification.get("directions") or {}
        )
        vertical_weight = dict(outlined_directions.get("vertical") or {})
        horizontal_weight = dict(outlined_directions.get("horizontal") or {})
        if str(vertical_weight.get("status") or "").startswith("supported_"):
            ink_weight_class_vertical = str(
                vertical_weight.get("weight_class") or ""
            )
            ink_weight_confidence_vertical = float(
                vertical_weight.get("confidence") or 0.0
            )
            ink_weight_support_vertical = str(vertical_weight.get("status") or "")
        if str(horizontal_weight.get("status") or "").startswith("supported_"):
            ink_weight_class_horizontal = str(
                horizontal_weight.get("weight_class") or ""
            )
            ink_weight_confidence_horizontal = float(
                horizontal_weight.get("confidence") or 0.0
            )
            ink_weight_support_horizontal = str(
                horizontal_weight.get("status") or ""
            )
        if ink_weight_class_vertical or ink_weight_class_horizontal:
            reasons = [
                reason
                for reason in reasons
                if not reason.startswith("source_ink_weight_")
            ]
            ink_weight_class = ""
            ink_weight_confidence = 0.0
            if ink_weight_class_vertical:
                reasons.append(
                    "source_ink_weight_"
                    f"{ink_weight_class_vertical}_supported_outlined_vertical_cell_cohort"
                )
            if ink_weight_class_horizontal:
                reasons.append(
                    "source_ink_weight_"
                    f"{ink_weight_class_horizontal}_supported_outlined_horizontal_cell_cohort"
                )
        elif ink_weight_class:
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
        "source_cell_confidence_vertical": round(vertical_confidence, 8),
        "source_cell_confidence_horizontal": round(horizontal_confidence, 8),
        "source_cell_support_vertical": vertical_support,
        "source_cell_support_horizontal": horizontal_support,
        "source_stroke_width_px": round(stroke_width, 6),
        "source_ink_stroke_width_px": round(source_ink_stroke_width, 6),
        "ink_weight_class": ink_weight_class,
        "ink_weight_confidence": round(ink_weight_confidence, 8),
        "ink_weight_class_vertical": ink_weight_class_vertical,
        "ink_weight_confidence_vertical": round(
            ink_weight_confidence_vertical, 8
        ),
        "ink_weight_support_vertical": ink_weight_support_vertical,
        "ink_weight_class_horizontal": ink_weight_class_horizontal,
        "ink_weight_confidence_horizontal": round(
            ink_weight_confidence_horizontal, 8
        ),
        "ink_weight_support_horizontal": ink_weight_support_horizontal,
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
        "outlined_weight_qualification": outlined_weight_qualification,
        "fill_x_spans": [round(float(value), 6) for value in x_spans],
        "fill_y_spans": [round(float(value), 6) for value in y_spans],
        "fill_component_cell_sizes": [
            round(float(value), 6) for value in fill_component_sizes
        ],
        "density_decomposition_vertical_spans": list(
            vertical_qualification.get("density_spans") or []
        ),
        "density_decomposition_horizontal_spans": list(
            horizontal_qualification.get("density_spans") or []
        ),
        "source_cell_qualification_vertical": vertical_qualification,
        "source_cell_qualification_horizontal": horizontal_qualification,
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


def _projection_spans_at_min_occupancy(
    binary: np.ndarray,
    *,
    axis: int,
    minimum_occupancy: int,
) -> list[float]:
    """Return occupied-axis runs without reconnecting adjacent text columns."""

    return [
        float(end - start)
        for start, end in _projection_runs_at_min_occupancy(
            binary,
            axis=axis,
            minimum_occupancy=minimum_occupancy,
        )
    ]


def _projection_runs_at_min_occupancy(
    binary: np.ndarray,
    *,
    axis: int,
    minimum_occupancy: int,
) -> list[tuple[int, int]]:
    counts = np.count_nonzero(np.asarray(binary, dtype=bool), axis=axis)
    projected = (counts >= max(1, int(minimum_occupancy))).astype(np.uint8)
    if projected.size <= 0:
        return []
    padded = np.pad(projected, (1, 1), constant_values=0)
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= 2
    ]


def _qualify_outlined_ink_weight(
    fill: np.ndarray,
    *,
    source_cell_size_vertical_px: float,
    source_cell_size_horizontal_px: float,
) -> tuple[str, float, dict[str, Any]]:
    """Qualify weight only when an outline covers repeated source-cell bands.

    A parent-wide distance statistic is unsafe when only punctuation or one
    fragment carries visible support. Repeated full-cell bands make the ink
    tier a text-wide observation while leaving mixed/local support unresolved.
    """

    binary = np.asarray(fill, dtype=bool)
    directions: dict[str, dict[str, Any]] = {}
    supported: list[tuple[str, str, float, int, float]] = []
    for direction, axis, cell_size in (
        ("vertical", 0, float(source_cell_size_vertical_px)),
        ("horizontal", 1, float(source_cell_size_horizontal_px)),
    ):
        record: dict[str, Any] = {
            "cell_size_px": round(max(0.0, cell_size), 6),
            "status": "unavailable_source_cell_scale",
            "band_spans_px": [],
            "band_ink_widths_px": [],
        }
        directions[direction] = record
        if cell_size <= 0.0:
            continue
        orthogonal_extent = int(binary.shape[0] if axis == 0 else binary.shape[1])
        runs = _projection_runs_at_min_occupancy(
            binary,
            axis=axis,
            minimum_occupancy=max(2, int(round(orthogonal_extent * 0.10))),
        )
        full_cell_runs = [
            (start, end)
            for start, end in runs
            if cell_size * 0.60 <= float(end - start) <= cell_size * 1.45
        ]
        record["band_spans_px"] = [
            float(end - start) for start, end in full_cell_runs
        ]
        if len(full_cell_runs) < 3:
            record["status"] = "unavailable_insufficient_full_cell_bands"
            continue
        ink_widths: list[float] = []
        try:
            import cv2

            for start, end in full_cell_runs:
                band = (
                    binary[:, start:end]
                    if axis == 0
                    else binary[start:end, :]
                )
                distance = cv2.distanceTransform(
                    np.asarray(band, dtype=np.uint8), cv2.DIST_L2, 5
                )
                values = distance[band]
                if values.size:
                    ink_widths.append(float(np.percentile(values, 75)) * 2.0)
        except Exception:
            record["status"] = "unavailable_distance_transform"
            continue
        record["band_ink_widths_px"] = [
            round(float(value), 6) for value in ink_widths
        ]
        if len(ink_widths) < 3:
            record["status"] = "unavailable_insufficient_ink_bands"
            continue
        median_width = float(np.median(np.asarray(ink_widths, dtype=np.float32)))
        relative_mad = float(
            np.median(np.abs(np.asarray(ink_widths) - median_width))
            / max(1.0, median_width)
        )
        record["median_ink_width_px"] = round(median_width, 6)
        record["relative_mad"] = round(relative_mad, 8)
        if relative_mad > 0.20:
            record["status"] = "unavailable_mixed_ink_tiers"
            continue
        if median_width <= 3.0:
            weight_class = "regular"
        elif median_width >= 3.8:
            weight_class = "bold"
        else:
            record["status"] = "unavailable_transition_ink_tier"
            continue
        confidence = min(
            0.94,
            0.76
            + min(0.10, (len(ink_widths) - 3) * 0.04)
            + min(0.08, max(0.0, 0.20 - relative_mad) * 0.40),
        )
        record["status"] = "supported_outlined_cell_cohort"
        record["weight_class"] = weight_class
        record["confidence"] = round(confidence, 8)
        supported.append(
            (direction, weight_class, confidence, len(ink_widths), relative_mad)
        )

    if not supported:
        return "", 0.0, {"status": "unavailable", "directions": directions}
    classes = {item[1] for item in supported}
    if len(classes) != 1:
        return "", 0.0, {
            "status": "unavailable_directional_weight_disagreement",
            "directions": directions,
        }
    selected = max(supported, key=lambda item: (item[3], item[2], -item[4]))
    return selected[1], float(selected[2]), {
        "status": "supported_outlined_cell_cohort",
        "selected_direction": selected[0],
        "directions": directions,
    }


def _stable_upper_cell_cohort(
    values: Sequence[float],
    *,
    minimum_count: int,
) -> tuple[float, int, float]:
    clean = np.asarray(
        [float(value) for value in values if float(value) >= 3.0],
        dtype=np.float32,
    )
    if clean.size < minimum_count:
        return 0.0, int(clean.size), 0.0
    upper_reference = float(np.percentile(clean, 75))
    cohort = clean[clean >= max(3.0, upper_reference * 0.60)]
    if cohort.size < minimum_count:
        return 0.0, int(cohort.size), 0.0
    median = float(np.median(cohort))
    relative_mad = float(
        np.median(np.abs(cohort - median)) / max(1.0, median)
    )
    if relative_mad > 0.20:
        return 0.0, int(cohort.size), relative_mad
    return median, int(cohort.size), relative_mad


def _fill_component_cell_sizes(fill: np.ndarray) -> list[float]:
    """Return glyph-like component spans from contrast-resolved fill only."""

    try:
        import cv2

        count, _, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(fill, dtype=np.uint8), connectivity=8
        )
    except Exception:
        return []
    sizes: list[float] = []
    for index in range(1, count):
        width = int(stats[index, 2])
        height = int(stats[index, 3])
        area = int(stats[index, 4])
        short = min(width, height)
        long = max(width, height)
        if area < 6 or short < 3 or long / max(1, short) > 2.2:
            continue
        sizes.append(float(long))
    return sizes


def _qualify_source_cell_measurement(
    fill: np.ndarray,
    *,
    axis: int,
    spans: Sequence[float],
    legacy_size: float,
    legacy_confidence: float,
    fill_component_sizes: Sequence[float],
) -> tuple[float, float, str, dict[str, Any]]:
    """Qualify a cell scale independently of halo/backing components.

    The prior value is retained only when authorized glyph-fill projection or
    glyph-like fill components corroborate it. Repeated fill evidence can repair
    halo fragmentation. A parent-sized island must decompose into stable
    occupancy bands; otherwise the scale fails closed.
    """

    binary = np.asarray(fill, dtype=bool)
    legacy_size = max(0.0, float(legacy_size))
    legacy_confidence = max(0.0, float(legacy_confidence))
    raw_candidate, raw_count, raw_relative_mad = _stable_upper_cell_cohort(
        spans,
        minimum_count=1,
    )
    fill_candidate, fill_count, fill_relative_mad = _stable_upper_cell_cohort(
        fill_component_sizes,
        minimum_count=3,
    )
    axis_extent = int(binary.shape[1] if axis == 0 else binary.shape[0])
    orthogonal_extent = int(binary.shape[0] if axis == 0 else binary.shape[1])
    coordinates = np.where(binary)
    orthogonal_coordinates = coordinates[0] if axis == 0 else coordinates[1]
    filled_orthogonal_extent = (
        int(np.ptp(orthogonal_coordinates)) + 1
        if orthogonal_coordinates.size
        else 0
    )
    raw_max = max((float(value) for value in spans), default=0.0)
    parent_sized_island = bool(
        axis_extent > 0
        and orthogonal_extent > 0
        and raw_max >= axis_extent * 0.78
        and filled_orthogonal_extent >= orthogonal_extent * 0.78
    )
    density_spans: list[float] = []
    density_candidate = 0.0
    density_count = 0
    density_relative_mad = 0.0
    if parent_sized_island:
        density_spans = _projection_spans_at_min_occupancy(
            binary,
            axis=axis,
            minimum_occupancy=max(2, int(round(orthogonal_extent * 0.10))),
        )
        (
            density_candidate,
            density_count,
            density_relative_mad,
        ) = _stable_upper_cell_cohort(
            density_spans,
            minimum_count=3,
        )

    audit = {
        "legacy_size": round(legacy_size, 6),
        "legacy_confidence": round(legacy_confidence, 8),
        "raw_projection_candidate": round(raw_candidate, 6),
        "raw_projection_candidate_count": int(raw_count),
        "raw_projection_relative_mad": round(raw_relative_mad, 8),
        "fill_component_candidate": round(fill_candidate, 6),
        "fill_component_candidate_count": int(fill_count),
        "fill_component_relative_mad": round(fill_relative_mad, 8),
        "axis_extent": axis_extent,
        "orthogonal_extent": orthogonal_extent,
        "filled_orthogonal_extent": filled_orthogonal_extent,
        "parent_sized_island": parent_sized_island,
        "density_minimum_occupancy": (
            max(2, int(round(orthogonal_extent * 0.10)))
            if parent_sized_island
            else 0
        ),
        "density_spans": [round(float(value), 6) for value in density_spans],
        "density_candidate": round(density_candidate, 6),
        "density_candidate_count": int(density_count),
        "density_relative_mad": round(density_relative_mad, 8),
    }

    if parent_sized_island:
        if density_candidate > 0.0 and raw_max >= density_candidate * 2.2:
            confidence = max(
                0.72,
                min(0.90, 0.86 - density_relative_mad * 0.40),
            )
            return (
                density_candidate,
                confidence,
                "supported_density_decomposition",
                audit,
            )
        return 0.0, 0.0, "unavailable_parent_sized_island", audit

    fill_projection_agree = bool(
        raw_candidate > 0.0
        and fill_candidate > 0.0
        and 0.70 <= raw_candidate / fill_candidate <= 1.35
    )
    if fill_projection_agree and (
        legacy_size < min(raw_candidate, fill_candidate) * 0.65
        or legacy_size > max(raw_candidate, fill_candidate) * 1.55
    ):
        repaired = float(np.median([raw_candidate, fill_candidate]))
        confidence = max(
            0.72,
            min(
                0.92,
                0.84
                - max(raw_relative_mad, fill_relative_mad) * 0.35
                + min(0.06, (raw_count + fill_count) * 0.003),
            ),
        )
        return (
            repaired,
            confidence,
            "supported_fill_projection_override",
            audit,
        )

    raw_matches = bool(
        raw_candidate > 0.0
        and 0.55 <= legacy_size / raw_candidate <= 1.65
    )
    fill_matches = bool(
        fill_candidate > 0.0
        and 0.60 <= legacy_size / fill_candidate <= 1.60
    )
    if legacy_size > 0.0 and legacy_confidence > 0.0 and (raw_matches or fill_matches):
        return (
            legacy_size,
            legacy_confidence,
            "supported_independent_corroboration",
            audit,
        )
    if fill_projection_agree:
        inferred = float(np.median([raw_candidate, fill_candidate]))
        return (
            inferred,
            0.72,
            "supported_fill_projection_inference",
            audit,
        )
    return 0.0, 0.0, "unavailable_unqualified_geometry", audit


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
