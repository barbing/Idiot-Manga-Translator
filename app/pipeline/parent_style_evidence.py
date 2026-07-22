# -*- coding: utf-8 -*-
"""Parent-authorized source-pixel views for style observation.

This module is an adapter, not a style or cleanup decision owner. It exposes a
read-only runtime view over original-page coordinates and the foreground that
TextAreaPlan-authorized CTD components contributed to a parent CleanupMask.
Raw mask arrays remain runtime-only and are omitted from audit serialization.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from types import MappingProxyType
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
_ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS = 24
_ADDITIVE_ROTATION_MIN_COMPONENTS = 2
_ADDITIVE_ROTATION_MIN_COMPONENT_PIXELS = 8
_ADDITIVE_ROTATION_MIN_BORDER_MARGIN_PX = 1
_ADDITIVE_ROTATION_MIN_ASPECT_RATIO = 1.60
_ADDITIVE_ROTATION_MIN_BBOX_OCCUPANCY = 0.12
_ADDITIVE_ROTATION_MAX_BBOX_OCCUPANCY = 0.78
_ADDITIVE_ROTATION_MIN_ABS_DEGREES = 12.0
_ADDITIVE_ROTATION_MAX_ABS_DEGREES = 40.0
_ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES = 3.0
_ADDITIVE_SHADOW_MIN_EFFECT_PIXELS = 24
_ADDITIVE_SHADOW_MIN_EFFECT_MASK_FRACTION = 0.05
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
EXTERNAL_SOURCE_SURFACE_RING_VERSION = (
    "authorized_external_source_surface_ring_v1"
)
_OUTLINE_SURFACE_CONTINUITY_MAX_RGB_DISTANCE = 40.0
_OUTLINE_SURFACE_CONTINUITY_MAX_LUMA_QUANTILE_DELTA = 24.0
_OUTLINE_BACKING_MIN_RGB_DISTANCE = 64.0
_OUTLINE_BACKING_MIN_LUMA_QUANTILE_DELTA = 36.0
_GRAYSCALE_OUTLINE_MIN_CORE_SUPPORT_LUMA_DISTANCE = 48.0
_GRAYSCALE_OUTLINE_MODERATE_SURFACE_RGB_DISTANCE = 14.0
_GRAYSCALE_OUTLINE_MODERATE_SURFACE_LUMA_DISTANCE = 8.0
_GRAYSCALE_OUTLINE_NARROW_SUPPORT_RATIO = 0.10
_GRAYSCALE_OUTLINE_MAX_SUPPORT_RATIO = 0.35
_GRAYSCALE_OUTLINE_DECISIVE_SURFACE_RATIO = 0.25
_GRAYSCALE_PAINT_GEOMETRY_SCHEMA = "grayscale_core_support_exterior_v2"
_OUTLINE_WIDTH_MEASUREMENT_VERSION = "radial_support_distance_to_core_v1"
_GRAYSCALE_FILL_SCHEMA = "grayscale_core_polarity_v1"
_GRAYSCALE_OUTLINE_SCHEMA = "grayscale_outline_geometry_v1"
_PAINT_CORE_HYPOTHESIS_VERSION = "paint_core_component_topology_v1"
SOURCE_TEXT_FOOTPRINT_VERSION = "authorized_source_text_footprint_v2"
SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY = (
    "parent_style_arbitrator_resolved_writing_direction"
)
SOURCE_STYLE_AXIS_EVIDENCE_VERSION = "source_style_axis_evidence_v1"
SOURCE_STYLE_AXES = (
    "family",
    "weight",
    "scale",
    "fill",
    "outline",
    "orientation",
    "rotation",
    "shadow",
)


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


def _freeze_axis_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_axis_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_axis_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_axis_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True)
class SourceStyleAxisEvidence:
    """One independently qualified source-style axis.

    The record contains no final style decision. Unsupported axes remain
    explicit records so one failure cannot erase observations on other axes.
    """

    axis: str
    status: str
    value: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: str = ""
    support_identity: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    support: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", str(self.axis or "").strip().lower())
        object.__setattr__(
            self,
            "status",
            str(self.status or "unavailable").strip().lower(),
        )
        object.__setattr__(
            self,
            "confidence",
            max(0.0, min(1.0, float(self.confidence or 0.0))),
        )
        object.__setattr__(self, "provenance", str(self.provenance or ""))
        object.__setattr__(self, "value", _freeze_axis_value(self.value or {}))
        object.__setattr__(
            self,
            "support_identity",
            _freeze_axis_value(self.support_identity or {}),
        )
        object.__setattr__(
            self,
            "support",
            _freeze_axis_value(self.support or {}),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_unique([str(value) for value in self.reason_codes if value])),
        )

    @property
    def supported(self) -> bool:
        return self.status == "supported" and self.confidence > 0.0

    @classmethod
    def unavailable(
        cls,
        axis: str,
        *,
        provenance: str,
        support_identity: Mapping[str, Any],
        reason_codes: Sequence[str],
        support: Mapping[str, Any] | None = None,
        status: str = "unavailable",
    ) -> "SourceStyleAxisEvidence":
        return cls(
            axis=axis,
            status=status,
            confidence=0.0,
            provenance=provenance,
            support_identity=support_identity,
            reason_codes=tuple(reason_codes),
            support=support or {},
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "source_style_axis_evidence_version": (
                SOURCE_STYLE_AXIS_EVIDENCE_VERSION
            ),
            "axis": self.axis,
            "status": self.status,
            "value": _json_safe_mapping(self.value),
            "confidence": round(float(self.confidence), 8),
            "provenance": self.provenance,
            "support_identity": _json_safe_mapping(self.support_identity),
            "reason_codes": list(self.reason_codes),
            "support": _json_safe_mapping(self.support),
        }


@dataclass(frozen=True)
class SourceTextAxisProfile:
    """Direction-specific layout geometry without choosing writing mode."""

    writing_direction: str
    cross_axis_group_count: int = 0
    cross_axis_group_count_reliable: bool = False
    cross_axis_group_centers_px: tuple[float, ...] = ()
    cross_axis_group_spans_px: tuple[float, ...] = ()
    inline_capacity: int = 0
    inline_capacity_reliable: bool = False
    inline_capacity_provenance: str = ""
    confidence: float = 0.0
    reason: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "writing_direction": self.writing_direction,
            "cross_axis_group_count": int(self.cross_axis_group_count),
            "cross_axis_group_count_reliable": bool(
                self.cross_axis_group_count_reliable
            ),
            "cross_axis_group_centers_px": [
                float(value) for value in self.cross_axis_group_centers_px
            ],
            "cross_axis_group_spans_px": [
                float(value) for value in self.cross_axis_group_spans_px
            ],
            "inline_capacity": int(self.inline_capacity),
            "inline_capacity_reliable": bool(
                self.inline_capacity_reliable
            ),
            "inline_capacity_provenance": self.inline_capacity_provenance,
            "confidence": round(float(self.confidence), 8),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceTextFootprint:
    """Immutable source-ink geometry bound to one authorized source view."""

    contract_version: str
    page_id: str
    view_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    cleanup_mask_ids: tuple[str, ...]
    owned_component_ids: tuple[str, ...]
    content_bbox_xywh: tuple[int, int, int, int]
    analysis_bbox_xywh: tuple[int, int, int, int]
    analysis_crop_shape_hw: tuple[int, int]
    detector_input_sha256: str
    authorized_mask_sha256: str
    authorized_pixel_sha256: str
    resolved_ink_mask_sha256: str
    authorized_source_view_sha256: str
    fact_set_id: str
    union_bbox_local_xywh: tuple[int, int, int, int] = ()
    union_bbox_page_xywh: tuple[int, int, int, int] = ()
    x_occupied_bands: tuple[tuple[int, int, float, float], ...] = ()
    y_occupied_bands: tuple[tuple[int, int, float, float], ...] = ()
    profile_selection_authority: str = (
        SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY
    )
    axis_profiles: tuple[SourceTextAxisProfile, ...] = ()

    def profile_for_direction(
        self,
        writing_direction: str,
    ) -> SourceTextAxisProfile | None:
        direction = str(writing_direction or "").strip().lower()
        return next(
            (
                profile
                for profile in self.axis_profiles
                if profile.writing_direction == direction
            ),
            None,
        )

    def _audit_payload_without_fact_set(self) -> dict[str, Any]:
        def bands(
            values: tuple[tuple[int, int, float, float], ...],
        ) -> list[dict[str, Any]]:
            return [
                {
                    "start_px": int(start),
                    "end_px": int(end),
                    "span_px": float(span),
                    "center_px": float(center),
                }
                for start, end, span, center in values
            ]

        source_identity = {
            "authorized_source_style_view_version": (
                AUTHORIZED_SOURCE_STYLE_VIEW_VERSION
            ),
            "page_id": self.page_id,
            "view_id": self.view_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "owned_component_ids": list(self.owned_component_ids),
            "content_bbox_xywh": list(self.content_bbox_xywh),
            "analysis_bbox_xywh": list(self.analysis_bbox_xywh),
            "analysis_crop_shape_hw": list(self.analysis_crop_shape_hw),
            "detector_input_sha256": self.detector_input_sha256,
            "authorized_mask_sha256": self.authorized_mask_sha256,
            "authorized_pixel_sha256": self.authorized_pixel_sha256,
            "resolved_ink_mask_sha256": self.resolved_ink_mask_sha256,
            "authorized_source_view_sha256": (
                self.authorized_source_view_sha256
            ),
        }
        return {
            "contract_version": self.contract_version,
            "source_identity": source_identity,
            "coordinate_space": "authorized_analysis_crop",
            "ink_authority": "independent_glyph_geometry",
            "union_bbox_local_xywh": list(self.union_bbox_local_xywh),
            "union_bbox_page_xywh": list(self.union_bbox_page_xywh),
            "x_occupied_bands": bands(self.x_occupied_bands),
            "y_occupied_bands": bands(self.y_occupied_bands),
            "writing_direction_evidence": {
                "status": "direction_neutral_axis_profiles",
                "available_directions": [
                    profile.writing_direction
                    for profile in self.axis_profiles
                ],
                "selection_authority": self.profile_selection_authority,
            },
            "axis_profiles": {
                profile.writing_direction: profile.to_audit_dict()
                for profile in self.axis_profiles
            },
        }

    def to_audit_dict(self) -> dict[str, Any]:
        result = self._audit_payload_without_fact_set()
        result["fact_set_id"] = self.fact_set_id
        return result














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
    spatial_fact_set_id: str = ""
    authorized_perceptual_source_identity: Mapping[str, Any] = field(
        default_factory=dict
    )
    perceptual_axis_evidence: Mapping[str, Any] = field(default_factory=dict)
    axis_evidence: tuple[SourceStyleAxisEvidence, ...] = ()
    source_text_footprint: SourceTextFootprint | None = None
    reason_codes: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.primary_input is not None and self.neutral_input is not None

    @property
    def axis_evidence_by_name(self) -> dict[str, SourceStyleAxisEvidence]:
        return {record.axis: record for record in self.axis_evidence}

    def axis(self, name: str) -> SourceStyleAxisEvidence | None:
        return self.axis_evidence_by_name.get(str(name or "").strip().lower())

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
        selected = (
            (
                str(self.ink_weight_class_vertical or ""),
                float(self.ink_weight_confidence_vertical),
                str(self.ink_weight_support_vertical or ""),
                "vertical",
            )
            if normalized == "ttb"
            else (
                str(self.ink_weight_class_horizontal or ""),
                float(self.ink_weight_confidence_horizontal),
                str(self.ink_weight_support_horizontal or ""),
                "horizontal",
            )
            if normalized == "ltr"
            else ("", 0.0, "", "")
        )
        opposite = (
            (
                str(self.ink_weight_class_horizontal or ""),
                float(self.ink_weight_confidence_horizontal),
                str(self.ink_weight_support_horizontal or ""),
                "horizontal",
            )
            if normalized == "ttb"
            else (
                str(self.ink_weight_class_vertical or ""),
                float(self.ink_weight_confidence_vertical),
                str(self.ink_weight_support_vertical or ""),
                "vertical",
            )
            if normalized == "ltr"
            else ("", 0.0, "", "")
        )
        if selected[2].startswith("supported_"):
            return selected[0], selected[1], selected[2]
        unsafe_unavailable = {
            "unavailable_mixed_ink_tiers",
            "unavailable_transition_ink_tier",
            "unavailable_directional_weight_disagreement",
        }
        if selected[2] in unsafe_unavailable:
            return "", 0.0, selected[2]
        if opposite[2].startswith("supported_"):
            return (
                opposite[0],
                opposite[1],
                "supported_cross_axis_fallback_from_"
                f"{opposite[3]}_cell_cohort",
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
            "axis_evidence": [
                record.to_audit_dict() for record in self.axis_evidence
            ],
            "metrics": _json_safe_mapping(self.metrics),
            "spatial_fact_set_id": self.spatial_fact_set_id,
            "detector_and_ink_axes_authorized_pixels_only": True,
            "carrier_external_surface_context_bounded": True,
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


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result














@dataclass(frozen=True)
class _IndependentGlyphGeometry:
    source: Any = field(repr=False, compare=False)
    authorized_mask: Any = field(repr=False, compare=False)
    glyph_mask: Any = field(repr=False, compare=False)
    support_mask: Any = field(repr=False, compare=False)
    fill_polarity: str = ""
    fill_color: str = ""
    support_color: str = ""
    fill_cluster_resolved: bool = False
    support_luma_median: float = 127.0
    support_luma_iqr: float = 255.0
    fill_luma_median: float = 127.0
    contrast: float = 0.0
    fill_count: int = 0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NativeAuthorizedGlyphGeometry:
    """One native-pixel glyph hypothesis shared by scale and weight only."""

    status: str
    source: Any = field(repr=False, compare=False)
    authorized_mask: Any = field(repr=False, compare=False)
    glyph_mask: Any = field(repr=False, compare=False)
    luma_strength: Any = field(repr=False, compare=False)
    polarity: str = ""
    component_facts: tuple[Mapping[str, Any], ...] = ()
    reason_codes: tuple[str, ...] = ()
    support: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "supported" and bool(self.component_facts)


@dataclass(frozen=True)
class _IndependentScaleMeasurement:
    axis_evidence: SourceStyleAxisEvidence
    glyph_mask: Any = field(repr=False, compare=False)
    vertical_size_px: float = 0.0
    horizontal_size_px: float = 0.0
    vertical_confidence: float = 0.0
    horizontal_confidence: float = 0.0
    vertical_support: str = ""
    horizontal_support: str = ""
    vertical_qualification: Mapping[str, Any] = field(default_factory=dict)
    horizontal_qualification: Mapping[str, Any] = field(default_factory=dict)
    native_geometry: _NativeAuthorizedGlyphGeometry | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _AxisLocalSpatialFacts:
    source_rgb: Any = field(repr=False, compare=False)
    authorized_mask: Any = field(repr=False, compare=False)
    character_core_mask: Any = field(repr=False, compare=False)
    concentric_shell_mask: Any = field(repr=False, compare=False)
    displaced_effect_mask: Any = field(repr=False, compare=False)
    core_color: str
    core_role_status: str
    core_resolution: str


def _axis_support_identity(
    *,
    view: AuthorizedSourceStyleView,
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
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
    }


def _independent_glyph_geometry(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> _IndependentGlyphGeometry:
    """Resolve a private glyph hypothesis for one axis.

    Callers do not share the selected mask. A failed fill observer therefore
    cannot invalidate scale, detector presentation, weight, or effects.
    """

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    reasons: list[str] = ["independent_axis_authorized_pixels_only"]
    mask_binary = np.asarray(mask, dtype=np.uint8)
    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    try:
        import cv2

        distance = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    except Exception:
        distance = mask_binary.astype(np.float32)
        reasons.append("independent_axis_distance_transform_unavailable")
    support_mask = mask & (distance <= 1.4)
    support_luma = luma[support_mask]
    if support_luma.size <= 0:
        support_mask = mask.copy()
        support_luma = luma[mask]
        reasons.append("independent_axis_support_ring_fallback")
    support_median = (
        float(np.median(support_luma)) if support_luma.size else 127.0
    )
    support_iqr = (
        float(np.percentile(support_luma, 75) - np.percentile(support_luma, 25))
        if support_luma.size
        else 255.0
    )
    polarity = "dark" if support_median >= 128.0 else "light"
    contrast_threshold = 48.0
    glyph = mask & (
        (luma <= support_median - contrast_threshold)
        if polarity == "dark"
        else (luma >= support_median + contrast_threshold)
    )
    minimum_pixels = max(8, int(round(np.count_nonzero(mask) * 0.01)))
    if int(np.count_nonzero(glyph)) < minimum_pixels:
        contrast_threshold = 32.0
        glyph = mask & (
            (luma <= support_median - contrast_threshold)
            if polarity == "dark"
            else (luma >= support_median + contrast_threshold)
        )
        reasons.append("independent_axis_contrast_threshold_relaxed")
    fill_count = int(np.count_nonzero(glyph))
    resolved = fill_count > 0
    if not resolved:
        glyph = mask.copy()
        fill_count = int(np.count_nonzero(glyph))
        reasons.append("independent_axis_glyph_geometry_unresolved")
    fill_pixels = source[glyph]
    fill_luma = luma[glyph]
    fill_median = (
        float(np.median(fill_luma)) if fill_luma.size else support_median
    )
    contrast = abs(support_median - fill_median)
    support_pixels = source[support_mask]
    fill_color = _polarized_hex_color(
        fill_pixels,
        fill_luma,
        polarity=polarity,
    )
    support_color = _polarized_hex_color(
        support_pixels,
        support_luma,
        polarity="light" if polarity == "dark" else "dark",
        fraction=10.0,
    )
    return _IndependentGlyphGeometry(
        source=_readonly_array(source, dtype=np.uint8),
        authorized_mask=_readonly_array(mask, dtype=bool),
        glyph_mask=_readonly_array(glyph, dtype=bool),
        support_mask=_readonly_array(support_mask, dtype=bool),
        fill_polarity=polarity,
        fill_color=fill_color,
        support_color=support_color,
        fill_cluster_resolved=resolved,
        support_luma_median=support_median,
        support_luma_iqr=support_iqr,
        fill_luma_median=fill_median,
        contrast=contrast,
        fill_count=fill_count,
        reason_codes=tuple(reasons),
    )


def _native_authorized_candidate(
    authorized_mask: np.ndarray,
    luma: np.ndarray,
    *,
    polarity: str,
    low_luma: float,
    high_luma: float,
) -> dict[str, Any]:
    """Measure one contrast polarity without making a paint decision."""

    mask = np.asarray(authorized_mask, dtype=bool)
    contrast_span = max(0.0, float(high_luma) - float(low_luma))
    reasons: list[str] = []
    if polarity == "dark":
        if contrast_span >= 20.0:
            threshold = min(205.0, float(high_luma) - 20.0)
            glyph = mask & (luma <= threshold)
            strength = np.clip(
                (float(high_luma) - luma) / max(20.0, contrast_span),
                0.0,
                1.0,
            )
        elif float(high_luma) <= 205.0:
            threshold = min(205.0, float(high_luma) + 1.0)
            glyph = mask.copy()
            strength = np.clip((255.0 - luma) / 255.0, 0.0, 1.0)
            reasons.append("uniform_authorized_dark_candidate")
        else:
            threshold = min(205.0, float(high_luma) - 20.0)
            glyph = np.zeros_like(mask, dtype=bool)
            strength = np.zeros_like(luma, dtype=np.float32)
    else:
        if contrast_span >= 20.0:
            threshold = max(50.0, float(low_luma) + 20.0)
            glyph = mask & (luma >= threshold)
            strength = np.clip(
                (luma - float(low_luma)) / max(20.0, contrast_span),
                0.0,
                1.0,
            )
        elif float(low_luma) >= 50.0:
            threshold = max(50.0, float(low_luma) - 1.0)
            glyph = mask.copy()
            strength = np.clip(luma / 255.0, 0.0, 1.0)
            reasons.append("uniform_authorized_light_candidate")
        else:
            threshold = max(50.0, float(low_luma) + 20.0)
            glyph = np.zeros_like(mask, dtype=bool)
            strength = np.zeros_like(luma, dtype=np.float32)

    component_facts: list[dict[str, Any]] = []
    rejected_fragment_count = 0
    try:
        import cv2

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.asarray(glyph, dtype=np.uint8), connectivity=8
        )
        for index in range(1, count):
            x0 = int(stats[index, 0])
            y0 = int(stats[index, 1])
            width = int(stats[index, 2])
            height = int(stats[index, 3])
            area = int(stats[index, 4])
            short = min(width, height)
            long = max(width, height)
            if (
                area < 6
                or short < 2
                or long / max(1, short) > 2.2
            ):
                rejected_fragment_count += 1
                continue
            component = np.asarray(
                labels[y0 : y0 + height, x0 : x0 + width] == index,
                dtype=np.uint8,
            )
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            contour = max(contours, key=cv2.contourArea) if contours else None
            contour_area = (
                float(cv2.contourArea(contour))
                if contour is not None
                else 0.0
            )
            perimeter = (
                float(cv2.arcLength(contour, True))
                if contour is not None
                else 0.0
            )
            hull = cv2.convexHull(contour) if contour is not None else None
            hull_area = (
                float(cv2.contourArea(hull)) if hull is not None else 0.0
            )
            occupancy = float(area) / float(max(1, width * height))
            circularity = (
                float(
                    4.0
                    * np.pi
                    * contour_area
                    / max(perimeter * perimeter, 1e-6)
                )
                if contour_area > 0.0 and perimeter > 0.0
                else 0.0
            )
            solidity = (
                contour_area / max(hull_area, 1e-6)
                if hull_area > 0.0
                else 0.0
            )
            normalized_compact_mark = bool(
                long / max(1, short) <= 1.60
                and occupancy >= 0.50
                and circularity >= 0.50
                and solidity >= 0.88
            )
            punctuation_fragment = bool(
                normalized_compact_mark
                and long / max(1, short) <= 1.35
            )
            component_pixels = (
                labels[y0 : y0 + height, x0 : x0 + width] == index
            )
            weighted_ink_area = float(
                np.sum(
                    np.asarray(strength, dtype=np.float32)[
                        y0 : y0 + height,
                        x0 : x0 + width,
                    ][component_pixels]
                )
            )
            component_facts.append(
                {
                    "component_index": int(index),
                    "bbox_xywh": [x0, y0, width, height],
                    "center_xy": [
                        round(float(centroids[index, 0]), 6),
                        round(float(centroids[index, 1]), 6),
                    ],
                    "area_px": area,
                    "weighted_ink_area_px": round(weighted_ink_area, 8),
                    "width_px": float(width),
                    "height_px": float(height),
                    "long_span_px": float(long),
                    "bbox_area_px": float(width * height),
                    "bbox_occupancy": round(occupancy, 8),
                    "contour_perimeter_px": round(perimeter, 8),
                    "contour_circularity": round(circularity, 8),
                    "contour_solidity": round(solidity, 8),
                    "normalized_compact_mark": normalized_compact_mark,
                    "punctuation_fragment": punctuation_fragment,
                }
            )
    except Exception:
        labels = np.zeros_like(glyph, dtype=np.int32)
        reasons.append("native_component_geometry_unavailable")

    compact_marks = [
        fact
        for fact in component_facts
        if bool(fact.get("normalized_compact_mark"))
    ]
    if len(compact_marks) >= 3:
        compact_tier = float(
            np.median(
                [
                    float(fact.get("long_span_px") or 0.0)
                    for fact in compact_marks
                ]
            )
        )
        for fact in compact_marks:
            span = float(fact.get("long_span_px") or 0.0)
            if compact_tier > 0.0 and 0.65 <= span / compact_tier <= 1.35:
                fact["punctuation_fragment"] = True

    yy, xx = np.where(mask)
    authorized_width = int(np.ptp(xx)) + 1 if xx.size else 0
    authorized_height = int(np.ptp(yy)) + 1 if yy.size else 0
    parent_sized_components = [
        fact
        for fact in component_facts
        if authorized_width > 0
        and authorized_height > 0
        and float(fact.get("width_px") or 0.0) >= authorized_width * 0.78
        and float(fact.get("height_px") or 0.0) >= authorized_height * 0.78
    ]
    body_facts = [
        fact
        for fact in component_facts
        if not bool(fact.get("punctuation_fragment"))
    ]
    if parent_sized_components:
        status = "unavailable_merged_island_geometry"
    elif not body_facts and compact_marks:
        status = "unavailable_punctuation_only_geometry"
    elif len(body_facts) < 2:
        status = (
            "unavailable_fragmented_geometry"
            if rejected_fragment_count > 0
            else "unavailable_insufficient_glyph_support"
        )
    else:
        status = "supported"
    return {
        "status": status,
        "polarity": polarity,
        "glyph_mask": np.asarray(glyph, dtype=bool),
        "luma_strength": np.asarray(strength, dtype=np.float32),
        "component_facts": component_facts,
        "body_component_count": len(body_facts),
        "punctuation_component_count": sum(
            bool(fact.get("punctuation_fragment"))
            for fact in component_facts
        ),
        "rejected_fragment_count": rejected_fragment_count,
        "threshold_luma": round(float(threshold), 6),
        "reason_codes": [
            *reasons,
            (
                "native_authorized_glyph_geometry_supported"
                if status == "supported"
                else status
            ),
        ],
    }


def _native_authorized_glyph_geometry(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> _NativeAuthorizedGlyphGeometry:
    """Select native glyph geometry once for source-cell and weight axes."""

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    authorized_luma = luma[mask]
    if authorized_luma.size < 6:
        empty = np.zeros_like(mask, dtype=bool)
        return _NativeAuthorizedGlyphGeometry(
            status="unavailable_insufficient_authorized_pixels",
            source=_readonly_array(source, dtype=np.uint8),
            authorized_mask=_readonly_array(mask, dtype=bool),
            glyph_mask=_readonly_array(empty, dtype=bool),
            luma_strength=_readonly_array(
                np.zeros_like(luma, dtype=np.float32), dtype=np.float32
            ),
            reason_codes=("unavailable_insufficient_authorized_pixels",),
        )
    low_luma, high_luma = [
        float(value) for value in np.percentile(authorized_luma, [10, 90])
    ]
    candidates = [
        _native_authorized_candidate(
            mask,
            luma,
            polarity=polarity,
            low_luma=low_luma,
            high_luma=high_luma,
        )
        for polarity in ("dark", "light")
    ]
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate["status"] == "supported"
        ),
        max(
            candidates,
            key=lambda candidate: (
                int(candidate.get("body_component_count") or 0),
                int(candidate.get("punctuation_component_count") or 0),
                candidate.get("polarity") == "dark",
            ),
        ),
    )
    facts = tuple(
        MappingProxyType(dict(fact))
        for fact in selected.get("component_facts") or ()
    )
    support = {
        "native_authorized_pixel_count": int(np.count_nonzero(mask)),
        "native_luma_p10": round(low_luma, 6),
        "native_luma_p90": round(high_luma, 6),
        "selected_polarity": str(selected.get("polarity") or ""),
        "selected_threshold_luma": float(
            selected.get("threshold_luma") or 0.0
        ),
        "component_count": len(facts),
        "body_component_count": int(
            selected.get("body_component_count") or 0
        ),
        "punctuation_component_count": int(
            selected.get("punctuation_component_count") or 0
        ),
        "rejected_fragment_count": int(
            selected.get("rejected_fragment_count") or 0
        ),
        "candidate_statuses": {
            str(candidate.get("polarity") or ""): str(
                candidate.get("status") or "unavailable"
            )
            for candidate in candidates
        },
    }
    return _NativeAuthorizedGlyphGeometry(
        status=str(selected.get("status") or "unavailable"),
        source=_readonly_array(source, dtype=np.uint8),
        authorized_mask=_readonly_array(mask, dtype=bool),
        glyph_mask=_readonly_array(
            selected.get("glyph_mask"), dtype=bool
        ),
        luma_strength=_readonly_array(
            selected.get("luma_strength"), dtype=np.float32
        ),
        polarity=str(selected.get("polarity") or ""),
        component_facts=facts,
        reason_codes=tuple(selected.get("reason_codes") or ()),
        support=MappingProxyType(support),
    )
















def _native_directional_cell_record(
    geometry: _NativeAuthorizedGlyphGeometry,
    *,
    direction: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dimension_key = "height_px" if direction == "ttb" else "width_px"
    base_record: dict[str, Any] = {
        "status": str(geometry.status or "unavailable"),
        "cell_p20_px": 0.0,
        "cell_median_px": 0.0,
        "cell_p80_px": 0.0,
        "confidence": 0.0,
        "uncertainty": {
            "relative_cell_spread": 0.0,
            "qualified_component_count": 0,
        },
    }
    qualification: dict[str, Any] = {
        "direction": direction,
        "dimension_key": dimension_key,
        "density_spans": [],
        "qualified_component_indices": [],
        "qualified_component_spans_px": [],
        "body_component_count": int(
            geometry.support.get("body_component_count") or 0
        ),
        "punctuation_component_count": int(
            geometry.support.get("punctuation_component_count") or 0
        ),
        "rejected_fragment_count": int(
            geometry.support.get("rejected_fragment_count") or 0
        ),
    }
    if not geometry.available:
        return base_record, qualification

    body_facts = [
        fact
        for fact in geometry.component_facts
        if not bool(fact.get("punctuation_fragment"))
    ]
    long_spans = np.asarray(
        [float(fact.get("long_span_px") or 0.0) for fact in body_facts],
        dtype=np.float32,
    )
    if long_spans.size < 2:
        base_record["status"] = "unavailable_insufficient_glyph_support"
        return base_record, qualification
    reference = float(np.percentile(long_spans, 75))
    lower = max(2.0, reference * 0.50)
    upper = reference * 1.55
    qualified = [
        fact
        for fact in body_facts
        if lower
        <= float(fact.get("long_span_px") or 0.0)
        <= upper
    ]
    qualification.update(
        {
            "reference_long_span_p75_px": round(reference, 6),
            "qualification_band_px": [round(lower, 6), round(upper, 6)],
            "qualified_component_indices": [
                int(fact.get("component_index") or 0) for fact in qualified
            ],
            "qualified_component_spans_px": [
                round(float(fact.get(dimension_key) or 0.0), 6)
                for fact in qualified
            ],
        }
    )
    if len(qualified) < 2:
        base_record["status"] = "unavailable_fragmented_geometry"
        return base_record, qualification

    lower_fragments = [
        fact
        for fact in body_facts
        if float(fact.get("long_span_px") or 0.0) < lower
    ]
    if len(lower_fragments) >= max(3, len(qualified)):
        lower_median = float(
            np.median(
                [
                    float(fact.get("long_span_px") or 0.0)
                    for fact in lower_fragments
                ]
            )
        )
        if lower_median < reference * 0.45:
            qualification["competing_lower_tier_count"] = len(
                lower_fragments
            )
            qualification["competing_lower_tier_median_px"] = round(
                lower_median, 6
            )
            base_record["status"] = "unavailable_competing_glyph_tiers"
            return base_record, qualification

    # Glyph widths and heights naturally vary within one source style. Judge
    # competing tiers on direction-neutral long spans; keep directional
    # width/height dispersion below as uncertainty instead of a style veto.
    qualified_long_spans = np.asarray(
        [float(fact.get("long_span_px") or 0.0) for fact in qualified],
        dtype=np.float32,
    )
    long_p20, long_median, long_p80 = [
        float(value)
        for value in np.percentile(qualified_long_spans, [20, 50, 80])
    ]
    qualification_relative_spread = (
        long_p80 - long_p20
    ) / max(1.0, long_median)
    qualification["qualification_relative_long_span_spread"] = round(
        qualification_relative_spread, 8
    )
    if qualification_relative_spread > 0.45:
        base_record["status"] = "unavailable_competing_glyph_tiers"
        return base_record, qualification

    values = np.asarray(
        [float(fact.get(dimension_key) or 0.0) for fact in qualified],
        dtype=np.float32,
    )
    p20, median, p80 = [
        float(value) for value in np.percentile(values, [20, 50, 80])
    ]
    relative_spread = (p80 - p20) / max(1.0, median)
    qualification["relative_cell_spread"] = round(relative_spread, 8)
    confidence = max(
        0.58,
        min(
            0.96,
            0.70
            + min(0.16, (len(qualified) - 2) * 0.02)
            + min(0.10, max(0.0, 0.45 - relative_spread) * 0.22),
        ),
    )
    base_record.update(
        {
            "status": "supported",
            "cell_p20_px": round(p20, 6),
            "cell_median_px": round(median, 6),
            "cell_p80_px": round(p80, 6),
            "confidence": round(confidence, 8),
            "uncertainty": {
                "relative_cell_spread": round(relative_spread, 8),
                "qualified_component_count": len(qualified),
                "sample_uncertainty": round(
                    1.0 / float(np.sqrt(len(qualified))), 8
                ),
            },
        }
    )
    return base_record, qualification


def _measure_independent_source_scale(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
    geometry: _NativeAuthorizedGlyphGeometry | None = None,
) -> _IndependentScaleMeasurement:
    """Publish robust native source-cell distributions without recovery."""

    native = geometry or _native_authorized_glyph_geometry(
        source_crop, mask_crop
    )
    vertical, vertical_qualification = _native_directional_cell_record(
        native, direction="ttb"
    )
    horizontal, horizontal_qualification = _native_directional_cell_record(
        native, direction="ltr"
    )
    vertical_supported = vertical.get("status") == "supported"
    horizontal_supported = horizontal.get("status") == "supported"
    vertical_confidence = (
        float(vertical.get("confidence") or 0.0)
        if vertical_supported
        else 0.0
    )
    horizontal_confidence = (
        float(horizontal.get("confidence") or 0.0)
        if horizontal_supported
        else 0.0
    )
    vertical_size = (
        float(vertical.get("cell_median_px") or 0.0)
        if vertical_supported
        else 0.0
    )
    horizontal_size = (
        float(horizontal.get("cell_median_px") or 0.0)
        if horizontal_supported
        else 0.0
    )
    vertical_support = (
        "supported_native_authorized_glyph_distribution"
        if vertical_supported
        else str(vertical.get("status") or "unavailable")
    )
    horizontal_support = (
        "supported_native_authorized_glyph_distribution"
        if horizontal_supported
        else str(horizontal.get("status") or "unavailable")
    )
    value = {
        "schema_version": "native_source_cell_distribution_v1",
        "directions": {"ttb": vertical, "ltr": horizontal},
        # Transitional scalar aliases carry only the measured median. They
        # remain non-authoritative until the Stage 2-3 contract cutover.
        "vertical_px": round(vertical_size, 6),
        "horizontal_px": round(horizontal_size, 6),
        "vertical_confidence": round(vertical_confidence, 8),
        "horizontal_confidence": round(horizontal_confidence, 8),
        "vertical_support": vertical_support,
        "horizontal_support": horizontal_support,
    }
    support = {
        **dict(native.support),
        "glyph_geometry_mask_sha256": _array_sha256(
            np.ascontiguousarray(native.glyph_mask, dtype=np.uint8)
        ),
        "glyph_pixel_count": int(np.count_nonzero(native.glyph_mask)),
        "vertical_qualification": vertical_qualification,
        "horizontal_qualification": horizontal_qualification,
        "shared_scale_weight_geometry": True,
    }
    confidence = max(vertical_confidence, horizontal_confidence)
    supported = bool(vertical_supported or horizontal_supported)
    reasons = tuple(
        _unique(
            [
                *native.reason_codes,
                vertical_support,
                horizontal_support,
                "source_cell_distribution_measured_from_native_authorized_glyphs"
                if supported
                else "source_scale_axis_unavailable",
            ]
        )
    )
    axis_evidence = SourceStyleAxisEvidence(
        axis="scale",
        status="supported" if supported else "unavailable",
        value=value,
        confidence=confidence,
        provenance=(
            "authorized_source_style_view:native_authorized_glyph_scale"
        ),
        support_identity=support_identity,
        reason_codes=reasons,
        support=support,
    )
    return _IndependentScaleMeasurement(
        axis_evidence=axis_evidence,
        glyph_mask=_readonly_array(native.glyph_mask, dtype=bool),
        vertical_size_px=vertical_size,
        horizontal_size_px=horizontal_size,
        vertical_confidence=vertical_confidence,
        horizontal_confidence=horizontal_confidence,
        vertical_support=vertical_support,
        horizontal_support=horizontal_support,
        vertical_qualification=MappingProxyType(vertical_qualification),
        horizontal_qualification=MappingProxyType(horizontal_qualification),
        native_geometry=native,
    )




@dataclass(frozen=True)
class _GrayscalePaintGeometry:
    """One core/support/exterior measurement shared by fill and outline."""

    status: str
    source: Any = field(repr=False, compare=False)
    authorized_mask: Any = field(repr=False, compare=False)
    core_mask: Any = field(repr=False, compare=False)
    adjacent_support_mask: Any = field(repr=False, compare=False)
    external_surface_mask: Any = field(repr=False, compare=False)
    core_polarity: str = ""
    core_color: str = ""
    support_color: str = ""
    source_cell_median_px: float = 0.0
    core_luma: Mapping[str, Any] = field(default_factory=dict)
    adjacent_support_luma: Mapping[str, Any] = field(default_factory=dict)
    external_surface_luma: Mapping[str, Any] = field(default_factory=dict)
    outline_width_px: Mapping[str, Any] = field(default_factory=dict)
    outline_to_cell_ratio: Mapping[str, Any] = field(default_factory=dict)
    facts: Mapping[str, Any] = field(default_factory=dict)
    external_ring_facts: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == "supported" and bool(self.core_polarity)


def _p20_median_p80(values: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(values, dtype=np.float32).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 0:
        return {
            "p20": 0.0,
            "median": 0.0,
            "p80": 0.0,
            "pixel_count": 0,
            "available": False,
        }
    p20, median, p80 = [
        float(value) for value in np.percentile(clean, [20, 50, 80])
    ]
    return {
        "p20": round(p20, 6),
        "median": round(median, 6),
        "p80": round(p80, 6),
        "pixel_count": int(clean.size),
        "available": True,
    }


def _normalize_distribution_to_cell(
    distribution: Mapping[str, Any],
    source_cell_median_px: float,
) -> dict[str, Any]:
    cell = max(0.0, float(source_cell_median_px))
    available = bool(cell > 0.0 and distribution.get("available"))
    normalized = {
        key: (
            round(float(distribution[key]) / cell, 8)
            if available
            else 0.0
        )
        for key in ("p20", "median", "p80")
    }
    normalized.update(
        {
            "pixel_count": int(distribution.get("pixel_count") or 0),
            "available": available,
            "source_cell_median_px": round(cell, 6),
        }
    )
    return normalized


def _distribution_distance(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    if not first.get("available") or not second.get("available"):
        return 0.0
    return float(
        np.median(
            [
                abs(float(first[key]) - float(second[key]))
                for key in ("p20", "median", "p80")
            ]
        )
    )


def _median_rgb(source: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    pixels = np.asarray(source, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.size <= 0:
        return ()
    return tuple(
        round(float(value), 6)
        for value in np.median(pixels.reshape(-1, 3), axis=0)
    )


def _rgb_median_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != 3 or len(second) != 3:
        return 0.0
    return float(
        np.linalg.norm(
            np.asarray(first, dtype=np.float32)
            - np.asarray(second, dtype=np.float32)
        )
    )


def _source_cell_reference(
    scale_measurement: _IndependentScaleMeasurement,
) -> float:
    supported = [
        float(value)
        for value, status in (
            (
                scale_measurement.vertical_size_px,
                scale_measurement.vertical_support,
            ),
            (
                scale_measurement.horizontal_size_px,
                scale_measurement.horizontal_support,
            ),
        )
        if value > 0.0 and str(status).startswith("supported")
    ]
    return max(supported, default=0.0)


def _paint_core_candidate_topology(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    glyph_pixel_count = int(
        np.count_nonzero(np.asarray(candidate.get("glyph_mask"), dtype=bool))
    )
    qualified_component_pixel_count = int(
        sum(
            int(fact.get("area_px") or 0)
            for fact in candidate.get("component_facts") or ()
        )
    )
    has_majority = bool(
        glyph_pixel_count > 0
        and qualified_component_pixel_count * 2 >= glyph_pixel_count
    )
    return {
        "status": str(candidate.get("status") or "unavailable"),
        "glyph_pixel_count": glyph_pixel_count,
        "qualified_component_pixel_count": qualified_component_pixel_count,
        "qualified_component_coverage": round(
            qualified_component_pixel_count / max(1, glyph_pixel_count), 8
        ),
        "has_majority_qualified_component_coverage": has_majority,
        "body_component_count": int(
            candidate.get("body_component_count") or 0
        ),
        "punctuation_component_count": int(
            candidate.get("punctuation_component_count") or 0
        ),
    }


def _qualified_paint_core_geometry(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> tuple[_IndependentGlyphGeometry, Mapping[str, Any]]:
    """Validate the paint core without changing other observer axes."""

    initial = _independent_glyph_geometry(source_crop, mask_crop)
    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    authorized = np.ascontiguousarray(mask_crop, dtype=bool)
    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    authorized_luma = luma[authorized]
    if authorized_luma.size < 6:
        return initial, MappingProxyType(
            {
                "paint_core_hypothesis_version": (
                    _PAINT_CORE_HYPOTHESIS_VERSION
                ),
                "paint_core_initial_polarity": initial.fill_polarity,
                "paint_core_selected_polarity": initial.fill_polarity,
                "paint_core_candidate_topology": {},
                "paint_core_selection_reason": (
                    "paint_core_candidate_topology_unavailable"
                ),
            }
        )

    low_luma, high_luma = [
        float(value) for value in np.percentile(authorized_luma, [10, 90])
    ]
    candidates = {
        polarity: _native_authorized_candidate(
            authorized,
            luma,
            polarity=polarity,
            low_luma=low_luma,
            high_luma=high_luma,
        )
        for polarity in ("dark", "light")
    }
    topology = {
        polarity: _paint_core_candidate_topology(candidate)
        for polarity, candidate in candidates.items()
    }
    initial_polarity = initial.fill_polarity
    selected_polarity = initial_polarity
    selection_reason = "paint_core_support_ring_hypothesis_retained"
    if initial_polarity in candidates:
        alternate_polarity = (
            "light" if initial_polarity == "dark" else "dark"
        )
        initial_candidate = candidates[initial_polarity]
        alternate_candidate = candidates[alternate_polarity]
        initial_topology = topology[initial_polarity]
        alternate_topology = topology[alternate_polarity]
        if (
            initial_candidate.get("status") == "supported"
            and alternate_candidate.get("status") == "supported"
            and not bool(
                initial_topology[
                    "has_majority_qualified_component_coverage"
                ]
            )
            and bool(
                alternate_topology[
                    "has_majority_qualified_component_coverage"
                ]
            )
        ):
            selected_polarity = alternate_polarity
            selection_reason = (
                "paint_core_rejected_unqualified_support_shell"
            )

    selected = candidates.get(selected_polarity)
    resolved = initial
    if selected is not None and selected_polarity != initial_polarity:
        glyph = np.ascontiguousarray(selected.get("glyph_mask"), dtype=bool)
        support = authorized & ~glyph
        fill_luma = luma[glyph]
        support_luma = luma[support]
        support_median = (
            float(np.median(support_luma))
            if support_luma.size
            else initial.support_luma_median
        )
        support_iqr = (
            float(
                np.percentile(support_luma, 75)
                - np.percentile(support_luma, 25)
            )
            if support_luma.size
            else initial.support_luma_iqr
        )
        fill_median = (
            float(np.median(fill_luma))
            if fill_luma.size
            else initial.fill_luma_median
        )
        resolved = replace(
            initial,
            glyph_mask=_readonly_array(glyph, dtype=bool),
            support_mask=_readonly_array(support, dtype=bool),
            fill_polarity=selected_polarity,
            fill_color=_polarized_hex_color(
                source[glyph], fill_luma, polarity=selected_polarity
            ),
            support_color=_polarized_hex_color(
                source[support],
                support_luma,
                polarity=(
                    "light" if selected_polarity == "dark" else "dark"
                ),
                fraction=10.0,
            ),
            fill_cluster_resolved=bool(np.any(glyph)),
            support_luma_median=support_median,
            support_luma_iqr=support_iqr,
            fill_luma_median=fill_median,
            contrast=abs(support_median - fill_median),
            fill_count=int(np.count_nonzero(glyph)),
            reason_codes=tuple(
                _unique(
                    [
                        *initial.reason_codes,
                        selection_reason,
                    ]
                )
            ),
        )

    return resolved, MappingProxyType(
        {
            "paint_core_hypothesis_version": _PAINT_CORE_HYPOTHESIS_VERSION,
            "paint_core_initial_polarity": initial_polarity,
            "paint_core_selected_polarity": selected_polarity,
            "paint_core_candidate_topology": MappingProxyType(
                {
                    polarity: MappingProxyType(dict(facts))
                    for polarity, facts in topology.items()
                }
            ),
            "paint_core_selection_reason": selection_reason,
        }
    )


def _grayscale_paint_geometry(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    source_cell_median_px: float,
) -> _GrayscalePaintGeometry:
    """Measure paint geometry without feeding it back into source scale."""

    resolved, paint_core_support = _qualified_paint_core_geometry(
        source_crop, mask_crop
    )
    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    authorized = np.ascontiguousarray(mask_crop, dtype=bool)
    core = np.ascontiguousarray(resolved.glyph_mask, dtype=bool)
    empty = np.zeros(authorized.shape, dtype=bool)
    reasons = [
        "grayscale_core_support_exterior_single_transaction",
        *resolved.reason_codes,
    ]
    if not resolved.fill_cluster_resolved or not np.any(core):
        return _GrayscalePaintGeometry(
            status="unavailable",
            source=_readonly_array(source, dtype=np.uint8),
            authorized_mask=_readonly_array(authorized, dtype=bool),
            core_mask=_readonly_array(core, dtype=bool),
            adjacent_support_mask=_readonly_array(empty, dtype=bool),
            external_surface_mask=_readonly_array(empty, dtype=bool),
            reason_codes=tuple(
                [*reasons, "grayscale_core_hypothesis_unavailable"]
            ),
        )

    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    cell = max(0.0, float(source_cell_median_px))
    width_values = np.asarray([], dtype=np.float32)
    authorized_extent_values = np.asarray([], dtype=np.float32)
    outline_width_support = np.zeros(authorized.shape, dtype=bool)
    distance_to_core = np.zeros(authorized.shape, dtype=np.float32)
    try:
        import cv2

        authorized_distance = cv2.distanceTransform(
            authorized.astype(np.uint8), cv2.DIST_L2, 5
        )
        core_distance = cv2.distanceTransform(
            core.astype(np.uint8), cv2.DIST_L2, 5
        )
        distance_to_core = cv2.distanceTransform(
            (~core).astype(np.uint8), cv2.DIST_L2, 5
        )
        extension = np.maximum(authorized_distance - core_distance, 0.0)
        authorized_extent_values = extension[core & (extension > 0.05)]
        support_search_limit = max(
            1.5,
            cell * _GRAYSCALE_OUTLINE_MAX_SUPPORT_RATIO,
        )
        support_candidates = (
            authorized
            & ~core
            & (distance_to_core > 0.0)
            & (distance_to_core <= support_search_limit)
        )
        support_luma_distance = np.abs(
            luma - float(resolved.support_luma_median)
        )
        core_luma_distance = np.abs(
            luma - float(resolved.fill_luma_median)
        )
        outline_width_support = (
            support_candidates
            & (support_luma_distance < core_luma_distance)
        )
        width_values = distance_to_core[outline_width_support]
        reasons.append(
            "grayscale_outline_width_from_radial_support_distance"
        )
    except Exception:
        reasons.append("grayscale_support_distance_transform_unavailable")

    width_distribution = _p20_median_p80(width_values)
    ratio_distribution = _normalize_distribution_to_cell(
        width_distribution,
        cell,
    )
    authorized_extent_distribution = _p20_median_p80(
        authorized_extent_values
    )
    authorized_extent_ratio = _normalize_distribution_to_cell(
        authorized_extent_distribution,
        cell,
    )

    support_limit = 1.5
    if authorized_extent_distribution["available"]:
        support_limit = max(
            1.5,
            float(authorized_extent_distribution["p80"]) * 1.25,
        )
    if cell > 0.0:
        support_limit = min(
            support_limit,
            max(1.5, cell * _GRAYSCALE_OUTLINE_MAX_SUPPORT_RATIO),
        )
    adjacent_support = (
        authorized
        & ~core
        & (distance_to_core > 0.0)
        & (distance_to_core <= support_limit)
    )
    if not np.any(adjacent_support):
        adjacent_support = authorized & ~core
        reasons.append("grayscale_adjacent_support_full_remainder_fallback")

    external_surface, external_ring_facts = _external_source_surface_ring(
        authorized
    )
    core_luma = _p20_median_p80(luma[core])
    support_luma = _p20_median_p80(luma[adjacent_support])
    external_luma = _p20_median_p80(luma[external_surface])
    core_rgb = _median_rgb(source, core)
    support_rgb = _median_rgb(source, adjacent_support)
    external_rgb = _median_rgb(source, external_surface)
    core_support_luma_distance = _distribution_distance(
        core_luma, support_luma
    )
    support_surface_luma_distance = _distribution_distance(
        support_luma, external_luma
    )
    core_support_rgb_distance = _rgb_median_distance(core_rgb, support_rgb)
    support_surface_rgb_distance = _rgb_median_distance(
        support_rgb, external_rgb
    )
    support_values = luma[adjacent_support]
    if resolved.fill_polarity == "dark":
        opposite_fraction = (
            float(np.mean(support_values >= 192.0))
            if support_values.size
            else 0.0
        )
    else:
        opposite_fraction = (
            float(np.mean(support_values <= 63.0))
            if support_values.size
            else 0.0
        )
    support_luma_iqr = (
        float(support_luma["p80"]) - float(support_luma["p20"])
        if support_luma["available"]
        else 0.0
    )
    ratio_median = float(ratio_distribution["median"])
    ratio_p80 = float(ratio_distribution["p80"])
    extent_ratio_p80 = float(authorized_extent_ratio["p80"])
    spatially_plausible = bool(
        ratio_distribution["available"]
        and 0.0 < ratio_median <= _GRAYSCALE_OUTLINE_MAX_SUPPORT_RATIO
    )
    visible_transition = bool(
        core_support_luma_distance >= 64.0
        and support_luma_iqr >= 48.0
        and 0.08 <= opposite_fraction <= 0.90
        and spatially_plausible
    )
    uniform_opposite_support = bool(
        core_support_luma_distance
        >= _GRAYSCALE_OUTLINE_MIN_CORE_SUPPORT_LUMA_DISTANCE
        and support_luma_iqr <= 44.0
        and opposite_fraction >= 0.88
    )
    external_available = bool(
        support_luma["available"]
        and external_luma["available"]
        and int(external_ring_facts.get("pixel_count") or 0) >= 24
    )
    external_continuity = bool(
        external_available
        and support_surface_rgb_distance
        <= _OUTLINE_SURFACE_CONTINUITY_MAX_RGB_DISTANCE
        and support_surface_luma_distance
        <= _OUTLINE_SURFACE_CONTINUITY_MAX_LUMA_QUANTILE_DELTA
    )
    external_discontinuity = bool(
        external_available
        and support_surface_rgb_distance >= _OUTLINE_BACKING_MIN_RGB_DISTANCE
        and support_surface_luma_distance
        >= _OUTLINE_BACKING_MIN_LUMA_QUANTILE_DELTA
    )
    moderate_narrow_separation = bool(
        external_available
        and spatially_plausible
        and ratio_p80 <= _GRAYSCALE_OUTLINE_NARROW_SUPPORT_RATIO
        and support_surface_rgb_distance
        >= _GRAYSCALE_OUTLINE_MODERATE_SURFACE_RGB_DISTANCE
        and support_surface_luma_distance
        >= _GRAYSCALE_OUTLINE_MODERATE_SURFACE_LUMA_DISTANCE
    )
    decisive_absence = bool(
        uniform_opposite_support
        and external_continuity
        and (
            not authorized_extent_ratio["available"]
            or extent_ratio_p80
            > _GRAYSCALE_OUTLINE_DECISIVE_SURFACE_RATIO
        )
    )
    facts = MappingProxyType(
        {
            "core_support_luma_distance": round(
                core_support_luma_distance, 6
            ),
            "core_support_rgb_distance": round(
                core_support_rgb_distance, 6
            ),
            "support_surface_luma_distance": round(
                support_surface_luma_distance, 6
            ),
            "support_surface_rgb_distance": round(
                support_surface_rgb_distance, 6
            ),
            "support_luma_iqr": round(support_luma_iqr, 6),
            "support_opposite_fraction": round(opposite_fraction, 8),
            "spatially_plausible_support": spatially_plausible,
            "visible_support_transition": visible_transition,
            "uniform_opposite_support": uniform_opposite_support,
            "external_surface_available": external_available,
            "external_surface_continuity": external_continuity,
            "external_surface_discontinuity": external_discontinuity,
            "moderate_narrow_surface_separation": (
                moderate_narrow_separation
            ),
            "decisive_outline_absence": decisive_absence,
            "core_rgb_median": core_rgb,
            "adjacent_support_rgb_median": support_rgb,
            "external_surface_rgb_median": external_rgb,
            "adjacent_support_limit_px": round(support_limit, 6),
            "outline_width_measurement_version": (
                _OUTLINE_WIDTH_MEASUREMENT_VERSION
            ),
            "outline_width_support_mask_sha256": _array_sha256(
                np.ascontiguousarray(outline_width_support, dtype=np.uint8)
            ),
            "outline_width_support_pixel_count": int(
                np.count_nonzero(outline_width_support)
            ),
            "authorized_support_extent_px": dict(
                authorized_extent_distribution
            ),
            "authorized_support_extent_to_cell_ratio": dict(
                authorized_extent_ratio
            ),
            "outline_absence_extent_ratio_p80": round(
                extent_ratio_p80,
                8,
            ),
            **dict(paint_core_support),
        }
    )
    return _GrayscalePaintGeometry(
        status="supported",
        source=_readonly_array(source, dtype=np.uint8),
        authorized_mask=_readonly_array(authorized, dtype=bool),
        core_mask=_readonly_array(core, dtype=bool),
        adjacent_support_mask=_readonly_array(adjacent_support, dtype=bool),
        external_surface_mask=_readonly_array(external_surface, dtype=bool),
        core_polarity=resolved.fill_polarity,
        core_color=_rgb_hex(core_rgb),
        support_color=_rgb_hex(support_rgb),
        source_cell_median_px=cell,
        core_luma=MappingProxyType(core_luma),
        adjacent_support_luma=MappingProxyType(support_luma),
        external_surface_luma=MappingProxyType(external_luma),
        outline_width_px=MappingProxyType(width_distribution),
        outline_to_cell_ratio=MappingProxyType(ratio_distribution),
        facts=facts,
        external_ring_facts=MappingProxyType(dict(external_ring_facts)),
        reason_codes=tuple(reasons),
    )


def _grayscale_axis_support(
    geometry: _GrayscalePaintGeometry,
) -> dict[str, Any]:
    return {
        "schema_version": _GRAYSCALE_PAINT_GEOMETRY_SCHEMA,
        "core_mask_sha256": _array_sha256(
            np.ascontiguousarray(geometry.core_mask, dtype=np.uint8)
        ),
        "adjacent_support_mask_sha256": _array_sha256(
            np.ascontiguousarray(
                geometry.adjacent_support_mask, dtype=np.uint8
            )
        ),
        "external_surface_mask_sha256": _array_sha256(
            np.ascontiguousarray(
                geometry.external_surface_mask, dtype=np.uint8
            )
        ),
        "core_pixel_count": int(np.count_nonzero(geometry.core_mask)),
        "adjacent_support_pixel_count": int(
            np.count_nonzero(geometry.adjacent_support_mask)
        ),
        "external_surface_pixel_count": int(
            np.count_nonzero(geometry.external_surface_mask)
        ),
        "source_cell_median_px": round(
            geometry.source_cell_median_px, 6
        ),
        "outline_width_px": dict(geometry.outline_width_px),
        "outline_to_cell_ratio": dict(geometry.outline_to_cell_ratio),
        **dict(geometry.facts),
    }


def _grayscale_support_identity(
    support_identity: Mapping[str, Any],
    geometry: _GrayscalePaintGeometry,
) -> dict[str, Any]:
    external = np.asarray(geometry.external_surface_mask, dtype=bool)
    facts = dict(geometry.external_ring_facts)
    identity = dict(support_identity)
    identity.update(
        {
            "external_surface_ring_version": str(
                facts.get("version") or EXTERNAL_SOURCE_SURFACE_RING_VERSION
            ),
            "external_surface_ring_inner_radius_px": float(
                facts.get("inner_radius_px") or 0.0
            ),
            "external_surface_ring_outer_radius_px": float(
                facts.get("outer_radius_px") or 0.0
            ),
            "external_surface_ring_pixel_count": int(
                facts.get("pixel_count") or 0
            ),
            "external_surface_ring_fallback_used": bool(
                facts.get("fallback_used")
            ),
            "external_surface_ring_mask_sha256": _array_sha256(
                np.ascontiguousarray(external, dtype=np.uint8)
            ),
            "external_surface_ring_pixel_sha256": _array_sha256(
                np.ascontiguousarray(
                    np.asarray(geometry.source, dtype=np.uint8)[external],
                    dtype=np.uint8,
                )
            ),
        }
    )
    return identity


def _observe_fill_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
    geometry: _GrayscalePaintGeometry | None = None,
    source_cell_median_px: float = 0.0,
) -> SourceStyleAxisEvidence:
    measured = geometry or _grayscale_paint_geometry(
        source_crop,
        mask_crop,
        source_cell_median_px=source_cell_median_px,
    )
    support = _grayscale_axis_support(measured)
    identity = _grayscale_support_identity(support_identity, measured)
    core_contrast = float(
        measured.facts.get("core_support_luma_distance") or 0.0
    )
    core_count = int(np.count_nonzero(measured.core_mask))
    confidence = (
        min(
            0.98,
            0.55
            + min(0.28, core_contrast / 512.0)
            + min(0.15, core_count / 512.0),
        )
        if measured.available and core_contrast > 0.0
        else 0.0
    )
    if not measured.available or confidence <= 0.0 or not measured.core_color:
        return SourceStyleAxisEvidence.unavailable(
            "fill",
            provenance=(
                "authorized_source_style_view:grayscale_fill_axis_v1"
            ),
            support_identity=identity,
            reason_codes=(
                *measured.reason_codes,
                "source_grayscale_fill_axis_unavailable",
            ),
            support=support,
        )
    return SourceStyleAxisEvidence(
        axis="fill",
        status="supported",
        value={
            "schema_version": _GRAYSCALE_FILL_SCHEMA,
            "color": measured.core_color,
            "support_color": measured.support_color,
            "polarity": measured.core_polarity,
            "core_polarity": measured.core_polarity,
            "core_luma": dict(measured.core_luma),
            "adjacent_support_luma": dict(measured.adjacent_support_luma),
            "external_surface_luma": dict(measured.external_surface_luma),
        },
        confidence=confidence,
        provenance="authorized_source_style_view:grayscale_fill_axis_v1",
        support_identity=identity,
        reason_codes=(
            *measured.reason_codes,
            "source_fill_measured_as_perceptual_core_polarity",
        ),
        support=support,
    )


def _observe_outline_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
    geometry: _GrayscalePaintGeometry | None = None,
    source_cell_median_px: float = 0.0,
) -> SourceStyleAxisEvidence:
    measured = geometry or _grayscale_paint_geometry(
        source_crop,
        mask_crop,
        source_cell_median_px=source_cell_median_px,
    )
    support = _grayscale_axis_support(measured)
    identity = _grayscale_support_identity(support_identity, measured)
    provenance = "authorized_source_style_view:grayscale_outline_axis_v1"
    if not measured.available:
        return SourceStyleAxisEvidence.unavailable(
            "outline",
            provenance=provenance,
            support_identity=identity,
            reason_codes=(
                *measured.reason_codes,
                "source_grayscale_outline_axis_unavailable",
            ),
            support=support,
        )

    facts = measured.facts
    width = dict(measured.outline_width_px)
    ratio = dict(measured.outline_to_cell_ratio)
    contrast = float(facts.get("core_support_luma_distance") or 0.0)
    present = bool(
        contrast >= _GRAYSCALE_OUTLINE_MIN_CORE_SUPPORT_LUMA_DISTANCE
        and bool(facts.get("spatially_plausible_support"))
        and (
            bool(facts.get("visible_support_transition"))
            or bool(facts.get("external_surface_discontinuity"))
            or bool(facts.get("moderate_narrow_surface_separation"))
        )
    )
    if present:
        confidence = min(
            0.95,
            0.68
            + min(0.12, contrast / 1024.0)
            + min(
                0.08,
                float(facts.get("support_surface_luma_distance") or 0.0)
                / 512.0,
            )
            + min(
                0.07,
                float(facts.get("support_surface_rgb_distance") or 0.0)
                / 1024.0,
            ),
        )
        return SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "schema_version": _GRAYSCALE_OUTLINE_SCHEMA,
                "present": True,
                "kind": "outline",
                "color": measured.support_color,
                "width_px": round(float(width.get("median") or 0.0), 6),
                "outline_width_px": width,
                "outline_to_cell_ratio": ratio,
                "core_polarity": measured.core_polarity,
            },
            confidence=confidence,
            provenance=provenance,
            support_identity=identity,
            reason_codes=(
                "source_outline_promoted_from_spatial_perceptual_support",
            ),
            support=support,
        )

    if bool(facts.get("decisive_outline_absence")):
        zero_ratio = {
            "p20": 0.0,
            "median": 0.0,
            "p80": 0.0,
            "pixel_count": int(ratio.get("pixel_count") or 0),
            "available": True,
            "source_cell_median_px": round(
                measured.source_cell_median_px, 6
            ),
        }
        confidence = min(
            0.92,
            0.70
            + min(0.12, contrast / 1024.0)
            + min(
                0.08,
                max(
                    0.0,
                    _OUTLINE_SURFACE_CONTINUITY_MAX_LUMA_QUANTILE_DELTA
                    - float(
                        facts.get("support_surface_luma_distance") or 0.0
                    ),
                )
                / 256.0,
            ),
        )
        return SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "schema_version": _GRAYSCALE_OUTLINE_SCHEMA,
                "present": False,
                "kind": "surface",
                "color": measured.support_color,
                "width_px": 0.0,
                "outline_width_px": {
                    "p20": 0.0,
                    "median": 0.0,
                    "p80": 0.0,
                    "pixel_count": int(width.get("pixel_count") or 0),
                    "available": True,
                },
                "outline_to_cell_ratio": zero_ratio,
                "core_polarity": measured.core_polarity,
            },
            confidence=confidence,
            provenance=provenance,
            support_identity=identity,
            reason_codes=(
                "source_outline_absence_supported_by_broad_continuous_surface",
            ),
            support=support,
        )

    return SourceStyleAxisEvidence(
        axis="outline",
        status="ambiguous",
        value={
            "schema_version": _GRAYSCALE_OUTLINE_SCHEMA,
            "present": None,
            "kind": "ambiguous",
            "color": measured.support_color,
            "width_px": round(float(width.get("median") or 0.0), 6),
            "outline_width_px": width,
            "outline_to_cell_ratio": ratio,
            "core_polarity": measured.core_polarity,
        },
        confidence=min(0.49, max(0.10, contrast / 512.0)),
        provenance=provenance,
        support_identity=identity,
        reason_codes=(
            "source_outline_core_support_surface_relation_ambiguous",
        ),
        support=support,
    )


def _observe_weight_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
    geometry: _NativeAuthorizedGlyphGeometry | None = None,
    scale_measurement: _IndependentScaleMeasurement | None = None,
) -> SourceStyleAxisEvidence:
    native = geometry or _native_authorized_glyph_geometry(
        source_crop, mask_crop
    )
    scale = scale_measurement or _measure_independent_source_scale(
        source_crop,
        mask_crop,
        support_identity=support_identity,
        geometry=native,
    )
    scale_directions = dict(scale.axis_evidence.value).get("directions", {})
    direction_records: dict[str, dict[str, Any]] = {}
    confidences: list[float] = []
    for direction in ("ttb", "ltr"):
        scale_record = dict(scale_directions.get(direction) or {})
        cell_size = float(scale_record.get("cell_median_px") or 0.0)
        qualification = (
            scale.vertical_qualification
            if direction == "ttb"
            else scale.horizontal_qualification
        )
        qualified_indices = {
            int(value)
            for value in qualification.get("qualified_component_indices")
            or ()
        }
        qualified = [
            fact
            for fact in native.component_facts
            if int(fact.get("component_index") or 0) in qualified_indices
        ]
        record: dict[str, Any] = {
            "status": str(scale_record.get("status") or native.status),
            "cell_median_px": round(cell_size, 6),
            "stem_to_cell_ratio": {},
            "ink_occupancy_ratio": {},
            "confidence": 0.0,
            "uncertainty": dict(scale_record.get("uncertainty") or {}),
        }
        direction_records[direction] = record
        if (
            scale_record.get("status") != "supported"
            or cell_size <= 0.0
            or len(qualified) < 2
        ):
            continue

        stem_values: list[float] = []
        occupancy_values: list[float] = []
        total_weighted_area = 0.0
        total_perimeter = 0.0
        total_bbox_area = 0.0
        for fact in qualified:
            weighted_area = float(fact.get("weighted_ink_area_px") or 0.0)
            perimeter = float(fact.get("contour_perimeter_px") or 0.0)
            bbox_area = float(fact.get("bbox_area_px") or 0.0)
            if weighted_area <= 0.0 or perimeter <= 0.0 or bbox_area <= 0.0:
                continue
            stem_values.append(
                min(1.0, (2.0 * weighted_area / perimeter) / cell_size)
            )
            occupancy_values.append(min(1.0, weighted_area / bbox_area))
            total_weighted_area += weighted_area
            total_perimeter += perimeter
            total_bbox_area += bbox_area
        if (
            len(stem_values) < 2
            or len(occupancy_values) < 2
            or total_perimeter <= 0.0
            or total_bbox_area <= 0.0
        ):
            record["status"] = "unavailable_insufficient_weight_support"
            continue

        def distribution(
            values: Sequence[float],
            aggregate: float,
        ) -> dict[str, float]:
            p20, _, p80 = [
                float(value)
                for value in np.percentile(
                    np.asarray(values, dtype=np.float32), [20, 50, 80]
                )
            ]
            median = max(1e-8, min(1.0, float(aggregate)))
            return {
                "p20": round(max(1e-8, min(p20, median)), 8),
                "median": round(median, 8),
                "p80": round(min(1.0, max(p80, median)), 8),
            }

        aggregate_stem = (
            2.0 * total_weighted_area / total_perimeter / cell_size
        )
        aggregate_occupancy = total_weighted_area / total_bbox_area
        stem_distribution = distribution(stem_values, aggregate_stem)
        occupancy_distribution = distribution(
            occupancy_values, aggregate_occupancy
        )
        uncertainty = dict(record["uncertainty"])
        uncertainty.update(
            {
                "weight_component_count": len(stem_values),
                "stem_relative_spread": round(
                    (
                        stem_distribution["p80"]
                        - stem_distribution["p20"]
                    )
                    / max(1e-8, stem_distribution["median"]),
                    8,
                ),
                "ink_occupancy_relative_spread": round(
                    (
                        occupancy_distribution["p80"]
                        - occupancy_distribution["p20"]
                    )
                    / max(1e-8, occupancy_distribution["median"]),
                    8,
                ),
            }
        )
        confidence = max(
            0.55,
            min(
                0.95,
                float(scale_record.get("confidence") or 0.0)
                - min(
                    0.18,
                    max(
                        float(uncertainty["stem_relative_spread"]),
                        float(
                            uncertainty["ink_occupancy_relative_spread"]
                        ),
                    )
                    * 0.08,
                ),
            ),
        )
        record.update(
            {
                "status": "supported",
                "stem_to_cell_ratio": stem_distribution,
                "ink_occupancy_ratio": occupancy_distribution,
                "confidence": round(confidence, 8),
                "uncertainty": uncertainty,
            }
        )
        confidences.append(confidence)

    supported = bool(confidences)
    value = {
        "schema_version": "native_normalized_weight_evidence_v1",
        "directions": direction_records,
    }
    support = {
        **dict(native.support),
        "glyph_geometry_mask_sha256": _array_sha256(
            np.ascontiguousarray(native.glyph_mask, dtype=np.uint8)
        ),
        "glyph_pixel_count": int(np.count_nonzero(native.glyph_mask)),
        "shared_scale_weight_geometry": True,
    }
    return SourceStyleAxisEvidence(
        axis="weight",
        status="supported" if supported else "unavailable",
        value=value,
        confidence=max(confidences, default=0.0),
        provenance=(
            "authorized_source_style_view:native_normalized_weight_axis"
        ),
        support_identity=support_identity,
        reason_codes=tuple(
            _unique(
                [
                    *native.reason_codes,
                    "source_weight_continuous_native_geometry_measured"
                    if supported
                    else "source_ink_weight_axis_unavailable",
                ]
            )
        ),
        support=support,
    )


def _axis_local_effect_facts(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> _AxisLocalSpatialFacts:
    geometry = _independent_glyph_geometry(source_crop, mask_crop)
    core = np.asarray(geometry.glyph_mask, dtype=bool)
    mask = np.asarray(geometry.authorized_mask, dtype=bool)
    try:
        import cv2

        dilated = cv2.dilate(
            core.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)
    except Exception:
        dilated = core.copy()
    shell = mask & dilated & ~core
    effect = mask & ~dilated
    return _AxisLocalSpatialFacts(
        source_rgb=geometry.source,
        authorized_mask=geometry.authorized_mask,
        character_core_mask=_readonly_array(core, dtype=bool),
        concentric_shell_mask=_readonly_array(shell, dtype=bool),
        displaced_effect_mask=_readonly_array(effect, dtype=bool),
        core_color=geometry.fill_color,
        core_role_status=(
            "supported" if geometry.fill_cluster_resolved else "unavailable"
        ),
        core_resolution="independent_axis_private_glyph_geometry",
    )


def _axis_record_from_effect_observation(
    axis: str,
    observed: Mapping[str, Any],
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    status = str(observed.get("support_status") or "unavailable")
    confidence = float(observed.get("confidence") or 0.0)
    reasons = tuple(observed.get("reason_codes") or ())
    support = (
        dict(observed.get("support") or {})
        if isinstance(observed.get("support"), Mapping)
        else {}
    )
    if status != "supported" or confidence <= 0.0:
        return SourceStyleAxisEvidence.unavailable(
            axis,
            provenance=f"authorized_source_style_view:independent_{axis}_axis",
            support_identity=support_identity,
            reason_codes=reasons or (f"source_{axis}_axis_unavailable",),
            support=support,
            status="ambiguous" if status == "ambiguous" else "unavailable",
        )
    value = (
        dict(observed.get("value") or {})
        if isinstance(observed.get("value"), Mapping)
        else {}
    )
    return SourceStyleAxisEvidence(
        axis=axis,
        status="supported",
        value=value,
        confidence=confidence,
        provenance=f"authorized_source_style_view:independent_{axis}_axis",
        support_identity=support_identity,
        reason_codes=reasons,
        support=support,
    )


def _observe_rotation_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    try:
        observed = _observe_additive_rotation(
            spatial_facts=_axis_local_effect_facts(source_crop, mask_crop),
        )
    except Exception:
        observed = {
            "support_status": "unavailable",
            "reason_codes": ["source_rotation_axis_observer_failed_closed"],
        }
    return _axis_record_from_effect_observation(
        "rotation", observed, support_identity=support_identity
    )


def _observe_shadow_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    try:
        observed = _observe_additive_shadow(
            spatial_facts=_axis_local_effect_facts(source_crop, mask_crop),
        )
    except Exception:
        observed = {
            "support_status": "unavailable",
            "reason_codes": ["source_shadow_axis_observer_failed_closed"],
        }
    return _axis_record_from_effect_observation(
        "shadow", observed, support_identity=support_identity
    )


def _unobserved_detector_axis(
    axis: str,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    return SourceStyleAxisEvidence.unavailable(
        axis,
        provenance=f"authorized_source_style_view:independent_{axis}_axis",
        support_identity=support_identity,
        reason_codes=(f"source_{axis}_axis_requires_detector_observation",),
    )


def build_authorized_style_observation_inputs(
    image: Any,
    view: AuthorizedSourceStyleView,
) -> AuthorizedStyleObservationInputs:
    """Build one transaction of independent source-style observations.

    Detector inputs and ink-derived axes use accepted foreground pixels. Fill
    and outline share one grayscale core/support/exterior measurement; its
    bounded exterior annulus is descriptive evidence only and cannot feed
    detector, source-scale, weight, orientation, rotation, or shadow paths.
    Raw exterior pixels are never published.
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

    detector_geometry = _independent_glyph_geometry(source_crop, mask_crop)
    primary_matte = 255 if detector_geometry.fill_polarity == "dark" else 0
    primary = np.full_like(source_crop, primary_matte, dtype=np.uint8)
    primary[mask_crop] = source_crop[mask_crop]
    neutral = np.full_like(source_crop, 127, dtype=np.uint8)
    neutral[mask_crop] = source_crop[mask_crop]
    detector_input_sha256 = _array_sha256(np.ascontiguousarray(primary))
    support_identity = _axis_support_identity(
        view=view,
        source_crop=source_crop,
        mask_crop=mask_crop,
    )
    support_identity["detector_input_sha256"] = detector_input_sha256

    native_geometry = _native_authorized_glyph_geometry(
        source_crop, mask_crop
    )
    scale_measurement = _measure_independent_source_scale(
        source_crop,
        mask_crop,
        support_identity=support_identity,
        geometry=native_geometry,
    )
    source_cell_median_px = _source_cell_reference(scale_measurement)
    grayscale_geometry = _grayscale_paint_geometry(
        source_crop,
        mask_crop,
        source_cell_median_px=source_cell_median_px,
    )
    fill_axis = _observe_fill_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
        geometry=grayscale_geometry,
    )
    outline_axis = _observe_outline_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
        geometry=grayscale_geometry,
    )
    weight_axis = _observe_weight_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
        geometry=native_geometry,
        scale_measurement=scale_measurement,
    )
    rotation_axis = _observe_rotation_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    )
    shadow_axis = _observe_shadow_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    )
    axis_evidence = (
        _unobserved_detector_axis(
            "family", support_identity=support_identity
        ),
        weight_axis,
        scale_measurement.axis_evidence,
        fill_axis,
        outline_axis,
        _unobserved_detector_axis(
            "orientation", support_identity=support_identity
        ),
        rotation_axis,
        shadow_axis,
    )

    scale_value = dict(scale_measurement.axis_evidence.value)
    fill_value = dict(fill_axis.value) if fill_axis.supported else {}
    outline_value = dict(outline_axis.value) if outline_axis.supported else {}
    weight_value = dict(weight_axis.value) if weight_axis.supported else {}
    glyph_mask = np.asarray(scale_measurement.glyph_mask, dtype=bool)
    component_facts = _fill_component_facts(glyph_mask)
    raw_footprint = _summarize_source_text_footprint(
        glyph_mask,
        component_facts=component_facts,
        vertical_cell_size_px=scale_measurement.vertical_size_px,
        vertical_scale_confidence=scale_measurement.vertical_confidence,
        vertical_scale_support=scale_measurement.vertical_support,
        vertical_scale_qualification=scale_measurement.vertical_qualification,
        horizontal_cell_size_px=scale_measurement.horizontal_size_px,
        horizontal_scale_confidence=scale_measurement.horizontal_confidence,
        horizontal_scale_support=scale_measurement.horizontal_support,
        horizontal_scale_qualification=scale_measurement.horizontal_qualification,
    )
    source_text_footprint = _bind_source_text_footprint(
        raw_footprint,
        view=view,
        support_identity=support_identity,
        source_crop=source_crop,
    )
    axis_reasons = _unique(
        [
            "independent_source_style_axis_transaction",
            *detector_geometry.reason_codes,
            *[
                reason
                for record in axis_evidence
                for reason in record.reason_codes
            ],
        ]
    )
    metrics = {
        "fill_polarity": str(fill_value.get("polarity") or ""),
        "fill_color": str(fill_value.get("color") or ""),
        "support_color": str(
            fill_value.get("support_color")
            or outline_value.get("color")
            or ""
        ),
        "source_cell_size_vertical_px": round(
            scale_measurement.vertical_size_px, 6
        ),
        "source_cell_size_horizontal_px": round(
            scale_measurement.horizontal_size_px, 6
        ),
        "source_cell_confidence_vertical": round(
            scale_measurement.vertical_confidence, 8
        ),
        "source_cell_confidence_horizontal": round(
            scale_measurement.horizontal_confidence, 8
        ),
        "source_cell_support_vertical": scale_measurement.vertical_support,
        "source_cell_support_horizontal": scale_measurement.horizontal_support,
        "source_stroke_width_px": round(
            float(outline_value.get("width_px") or 0.0), 6
        ),
        "source_ink_stroke_width_px": round(
            float(weight_value.get("source_ink_stroke_width_px") or 0.0), 6
        ),
        "ink_weight_class": str(weight_value.get("class") or ""),
        "ink_weight_confidence": round(
            float(weight_value.get("confidence") or 0.0), 8
        ),
        "ink_weight_class_vertical": str(
            weight_value.get("vertical_class") or ""
        ),
        "ink_weight_confidence_vertical": round(
            float(weight_value.get("vertical_confidence") or 0.0), 8
        ),
        "ink_weight_support_vertical": str(
            weight_value.get("vertical_support") or ""
        ),
        "ink_weight_class_horizontal": str(
            weight_value.get("horizontal_class") or ""
        ),
        "ink_weight_confidence_horizontal": round(
            float(weight_value.get("horizontal_confidence") or 0.0), 8
        ),
        "ink_weight_support_horizontal": str(
            weight_value.get("horizontal_support") or ""
        ),
        "scale_confidence": round(
            float(scale_measurement.axis_evidence.confidence), 8
        ),
        "paint_confidence": round(float(fill_axis.confidence), 8),
        "stroke_confidence": round(float(outline_axis.confidence), 8),
        "reason_codes": axis_reasons,
        "density_decomposition_vertical_spans": list(
            scale_measurement.vertical_qualification.get("density_spans") or ()
        ),
        "density_decomposition_horizontal_spans": list(
            scale_measurement.horizontal_qualification.get("density_spans") or ()
        ),
        "source_cell_qualification_vertical": _json_safe_mapping(
            scale_measurement.vertical_qualification
        ),
        "source_cell_qualification_horizontal": _json_safe_mapping(
            scale_measurement.horizontal_qualification
        ),
        "axis_evidence": [record.to_audit_dict() for record in axis_evidence],
        "observation_owner": "independent_source_style_axis_transaction",
    }
    from PIL import Image

    return AuthorizedStyleObservationInputs(
        primary_input=Image.fromarray(primary, mode="RGB"),
        neutral_input=Image.fromarray(neutral, mode="RGB"),
        primary_matte=primary_matte,
        fill_polarity=str(fill_value.get("polarity") or ""),
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
        spatial_fact_set_id="",
        authorized_perceptual_source_identity={},
        perceptual_axis_evidence={},
        axis_evidence=axis_evidence,
        source_text_footprint=source_text_footprint,
        reason_codes=tuple(metrics.get("reason_codes") or ()),
        metrics=metrics,
    )


def _bind_source_text_footprint(
    raw: Mapping[str, Any],
    *,
    view: AuthorizedSourceStyleView,
    support_identity: Mapping[str, Any],
    source_crop: np.ndarray,
) -> SourceTextFootprint:
    """Bind local ink geometry to one immutable authorized-view identity."""

    local_union = tuple(int(value) for value in raw.get("union_bbox_xywh") or ())
    if len(local_union) != 4:
        local_union = ()
    analysis_bbox = tuple(int(value) for value in view.analysis_bbox)
    if len(analysis_bbox) != 4:
        analysis_bbox = ()
    content_bbox = tuple(int(value) for value in view.content_bbox)
    if len(content_bbox) != 4:
        content_bbox = ()
    page_union: tuple[int, int, int, int] = ()
    if local_union and analysis_bbox:
        page_union = (
            analysis_bbox[0] + local_union[0],
            analysis_bbox[1] + local_union[1],
            local_union[2],
            local_union[3],
        )
    identity = support_identity
    source_crop = np.asarray(source_crop, dtype=np.uint8)
    detector_input_sha256 = str(identity.get("detector_input_sha256") or "")
    authorized_mask_sha256 = str(identity.get("authorized_mask_sha256") or "")
    authorized_pixel_sha256 = str(identity.get("authorized_pixel_sha256") or "")
    resolved_ink_mask_sha256 = str(
        raw.get("resolved_ink_mask_sha256") or ""
    )
    source_identity = {
        "authorized_source_style_view_version": (
            AUTHORIZED_SOURCE_STYLE_VIEW_VERSION
        ),
        "page_id": view.page_id,
        "view_id": view.view_id,
        "bundle_id": view.bundle_id,
        "parent_id": view.parent_id,
        "root_id": view.root_id,
        "cleanup_mask_ids": list(view.cleanup_mask_ids),
        "owned_component_ids": list(view.owned_component_ids),
        "content_bbox_xywh": list(content_bbox),
        "analysis_bbox_xywh": list(analysis_bbox),
        "analysis_crop_shape_hw": [
            int(source_crop.shape[0]),
            int(source_crop.shape[1]),
        ],
        "detector_input_sha256": detector_input_sha256,
        "authorized_mask_sha256": authorized_mask_sha256,
        "authorized_pixel_sha256": authorized_pixel_sha256,
        "resolved_ink_mask_sha256": resolved_ink_mask_sha256,
    }
    encoded_identity = json.dumps(
        source_identity,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    source_view_sha256 = hashlib.sha256(encoded_identity).hexdigest()

    def band_tuples(key: str) -> tuple[tuple[int, int, float, float], ...]:
        records = raw.get(key)
        if not isinstance(records, Sequence):
            return ()
        output: list[tuple[int, int, float, float]] = []
        for item in records:
            if not isinstance(item, Mapping):
                continue
            output.append(
                (
                    int(item.get("start_px") or 0),
                    int(item.get("end_px") or 0),
                    float(item.get("span_px") or 0.0),
                    float(item.get("center_px") or 0.0),
                )
            )
        return tuple(output)

    raw_profiles = raw.get("axis_profiles")
    axis_profiles: list[SourceTextAxisProfile] = []
    if isinstance(raw_profiles, Mapping):
        for direction in ("ttb", "ltr"):
            record = raw_profiles.get(direction)
            if not isinstance(record, Mapping):
                continue
            axis_profiles.append(
                SourceTextAxisProfile(
                    writing_direction=direction,
                    cross_axis_group_count=int(
                        record.get("cross_axis_group_count") or 0
                    ),
                    cross_axis_group_count_reliable=bool(
                        record.get("cross_axis_group_count_reliable")
                    ),
                    cross_axis_group_centers_px=tuple(
                        float(value)
                        for value in record.get("cross_axis_group_centers_px")
                        or ()
                    ),
                    cross_axis_group_spans_px=tuple(
                        float(value)
                        for value in record.get("cross_axis_group_spans_px")
                        or ()
                    ),
                    inline_capacity=int(record.get("inline_capacity") or 0),
                    inline_capacity_reliable=bool(
                        record.get("inline_capacity_reliable")
                    ),
                    inline_capacity_provenance=str(
                        record.get("inline_capacity_provenance") or ""
                    ),
                    confidence=float(record.get("confidence") or 0.0),
                    reason=str(record.get("reason") or ""),
                )
            )

    candidate = SourceTextFootprint(
        contract_version=SOURCE_TEXT_FOOTPRINT_VERSION,
        page_id=view.page_id,
        view_id=view.view_id,
        bundle_id=view.bundle_id,
        parent_id=view.parent_id,
        root_id=view.root_id,
        cleanup_mask_ids=tuple(view.cleanup_mask_ids),
        owned_component_ids=tuple(view.owned_component_ids),
        content_bbox_xywh=content_bbox,
        analysis_bbox_xywh=analysis_bbox,
        analysis_crop_shape_hw=(
            int(source_crop.shape[0]),
            int(source_crop.shape[1]),
        ),
        detector_input_sha256=detector_input_sha256,
        authorized_mask_sha256=authorized_mask_sha256,
        authorized_pixel_sha256=authorized_pixel_sha256,
        resolved_ink_mask_sha256=resolved_ink_mask_sha256,
        authorized_source_view_sha256=source_view_sha256,
        fact_set_id="",
        union_bbox_local_xywh=local_union,
        union_bbox_page_xywh=page_union,
        x_occupied_bands=band_tuples("x_occupied_bands"),
        y_occupied_bands=band_tuples("y_occupied_bands"),
        profile_selection_authority=(
            SOURCE_TEXT_FOOTPRINT_PROFILE_SELECTION_AUTHORITY
        ),
        axis_profiles=tuple(axis_profiles),
    )
    encoded_fact_set = json.dumps(
        candidate._audit_payload_without_fact_set(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return replace(
        candidate,
        fact_set_id=(
            f"{SOURCE_TEXT_FOOTPRINT_VERSION}:"
            f"{hashlib.sha256(encoded_fact_set).hexdigest()}"
        ),
    )








def _observe_additive_rotation(
    *,
    spatial_facts: _AxisLocalSpatialFacts,
) -> dict[str, Any]:
    """Return one pronounced rotation measured from the canonical core."""

    facts = spatial_facts
    source = np.asarray(facts.source_rgb, dtype=np.uint8)
    mask = np.asarray(facts.authorized_mask, dtype=bool)
    core = np.asarray(facts.character_core_mask, dtype=bool)
    core_count = int(np.count_nonzero(core))
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {
            "authorized_pixel_count": int(np.count_nonzero(mask)),
            "canonical_core_pixel_count": core_count,
            "canonical_core_role_status": facts.core_role_status,
            "canonical_core_role_reason": facts.core_role_reason,
        },
        "uncertainty": {},
    }
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or mask.shape != source.shape[:2]
        or core.shape != mask.shape
    ):
        unavailable["reason_codes"].append(
            "perceptual_rotation_input_invalid"
        )
        return unavailable
    if (
        facts.core_role_status != "supported"
        or core_count < _ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS
    ):
        unavailable["reason_codes"].append(
            "perceptual_rotation_canonical_core_unavailable"
        )
        return unavailable

    try:
        import cv2
    except Exception:
        unavailable["reason_codes"].append(
            "perceptual_rotation_spatial_backend_unavailable"
        )
        return unavailable

    _, _, stats, _ = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), connectivity=8
    )
    significant_components = int(
        sum(
            1
            for row in stats[1:]
            if int(row[cv2.CC_STAT_AREA])
            >= _ADDITIVE_ROTATION_MIN_COMPONENT_PIXELS
        )
    )
    support = dict(unavailable["support"])
    support.update(
        {
            "significant_component_count": significant_components,
            "core_border_margin_px": _mask_border_margin(core),
        }
    )
    if significant_components < _ADDITIVE_ROTATION_MIN_COMPONENTS:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_insufficient_core_components"
        )
        return unavailable
    if _mask_border_margin(core) < _ADDITIVE_ROTATION_MIN_BORDER_MARGIN_PX:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_character_core_truncated"
        )
        return unavailable

    yy, xx = np.where(core)
    points = np.column_stack((xx, yy)).astype(np.float32)
    (_, _), (rect_width, rect_height), rect_angle = cv2.minAreaRect(
        points
    )
    major = max(float(rect_width), float(rect_height))
    minor = min(float(rect_width), float(rect_height))
    degrees = (
        float(rect_angle)
        if float(rect_width) >= float(rect_height)
        else float(rect_angle) - 90.0
    )
    while degrees <= -90.0:
        degrees += 180.0
    while degrees > 90.0:
        degrees -= 180.0
    aspect_ratio = major / max(minor, 1e-6)
    bbox_occupancy = core_count / max(
        1.0, float(rect_width) * float(rect_height)
    )
    support.update(
        {
            "degrees_clockwise": round(degrees, 8),
            "oriented_aspect_ratio": round(aspect_ratio, 8),
            "oriented_bbox_occupancy": round(bbox_occupancy, 8),
        }
    )
    if aspect_ratio < _ADDITIVE_ROTATION_MIN_ASPECT_RATIO:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_symmetric_or_nondirectional_core"
        )
        return unavailable
    if not (
        _ADDITIVE_ROTATION_MIN_BBOX_OCCUPANCY
        <= bbox_occupancy
        <= _ADDITIVE_ROTATION_MAX_BBOX_OCCUPANCY
    ):
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_noncharacter_core_occupancy"
        )
        return unavailable
    if abs(degrees) < _ADDITIVE_ROTATION_MIN_ABS_DEGREES:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_upright_or_italic_only_geometry"
        )
        return unavailable
    if abs(degrees) > _ADDITIVE_ROTATION_MAX_ABS_DEGREES:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_base_axis_or_rotation_ambiguous"
        )
        return unavailable

    eroded = cv2.erode(
        core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    if (
        int(np.count_nonzero(eroded))
        < _ADDITIVE_ROTATION_MIN_CLUSTER_PIXELS
    ):
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_eroded_core_unavailable"
        )
        return unavailable
    eroded_y, eroded_x = np.where(eroded)
    (_, _), (eroded_width, eroded_height), eroded_angle = (
        cv2.minAreaRect(
            np.column_stack((eroded_x, eroded_y)).astype(np.float32)
        )
    )
    eroded_degrees = (
        float(eroded_angle)
        if float(eroded_width) >= float(eroded_height)
        else float(eroded_angle) - 90.0
    )
    while eroded_degrees <= -90.0:
        eroded_degrees += 180.0
    while eroded_degrees > 90.0:
        eroded_degrees -= 180.0
    raw_delta = abs(degrees - eroded_degrees)
    erosion_delta = min(raw_delta, abs(raw_delta - 90.0))
    support.update(
        {
            "eroded_degrees_clockwise": round(eroded_degrees, 8),
            "erosion_angle_delta_degrees": round(erosion_delta, 8),
        }
    )
    if erosion_delta > _ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES:
        unavailable["support"] = support
        unavailable["reason_codes"].append(
            "perceptual_rotation_not_stable_under_core_erosion"
        )
        return unavailable

    confidence = min(
        0.98,
        0.66
        + 0.12 * min(1.0, (aspect_ratio - 1.0) / 2.0)
        + 0.12
        * max(
            0.0,
            1.0
            - erosion_delta
            / max(_ADDITIVE_ROTATION_MAX_EROSION_DELTA_DEGREES, 1e-6),
        )
        + 0.06 * min(1.0, abs(degrees) / 24.0),
    )
    return {
        "support_status": "supported",
        "confidence": round(confidence, 8),
        "reason_codes": [
            "perceptual_rotation_canonical_character_core_axis"
        ],
        "support": support,
        "uncertainty": {
            "erosion_angle_delta_degrees": round(erosion_delta, 8)
        },
        "value": {
            "degrees_clockwise": round(degrees, 8),
            "pivot": "visual_center",
        },
    }


def _observe_additive_shadow(
    *,
    spatial_facts: _AxisLocalSpatialFacts,
) -> dict[str, Any]:
    """Return one complete displaced glyph-correlated shadow.

    A previously supported chromatic character core supplies only the runtime
    shape used for correlation. One darker authorized effect must be explained
    by one displaced copy of that shape. Blur is then estimated from the
    spatial support extending beyond the displaced core, not from RGB
    dispersion. Concentric, centered, repeated, clipped, or ambiguous support
    remains unavailable.
    """

    facts = spatial_facts
    source = np.asarray(facts.source_rgb, dtype=np.uint8)
    mask = np.asarray(facts.authorized_mask, dtype=bool)
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

    core = np.asarray(facts.character_core_mask, dtype=bool)
    fill_color = str(facts.core_color or "")
    if (
        facts.core_role_status != "supported"
        or int(np.count_nonzero(core)) < _ADDITIVE_SHADOW_MIN_EFFECT_PIXELS
    ):
        unavailable["reason_codes"].append(
            "perceptual_shadow_character_core_unavailable"
        )
        unavailable["support"]["character_core_status"] = facts.core_resolution
        return unavailable
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

    core_count = int(np.count_nonzero(core))
    effect = np.asarray(facts.displaced_effect_mask, dtype=bool)
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
    predicted_visible = (
        predicted_support
        & ~core
        & ~np.asarray(facts.concentric_shell_mask, dtype=bool)
    )
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


def _external_source_surface_ring(
    mask_crop: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one bounded annulus outside authorized foreground.

    Only this annulus may contribute non-authorized pixels to carrier-vs-page
    surface classification.  The annulus itself remains runtime-only; callers
    serialize only its geometry, counts, and digests.
    """

    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    empty = np.zeros(mask.shape, dtype=bool)
    facts: dict[str, Any] = {
        "version": EXTERNAL_SOURCE_SURFACE_RING_VERSION,
        "inner_radius_px": 0.0,
        "outer_radius_px": 0.0,
        "pixel_count": 0,
        "fallback_used": False,
    }
    if mask.ndim != 2 or not np.any(mask) or np.all(mask):
        return empty, facts
    try:
        import cv2

        distance = cv2.distanceTransform(
            (~mask).astype(np.uint8), cv2.DIST_L2, 5
        )
    except Exception:
        return empty, facts
    outside_distances = distance[~mask]
    positive = outside_distances[outside_distances > 0.0]
    if positive.size <= 0:
        return empty, facts
    outer_radius = min(10.0, float(np.percentile(positive, 90)))
    if outer_radius <= 0.0:
        return empty, facts
    inner_radius = max(1.0, outer_radius * 0.55)
    ring = (~mask) & (distance > inner_radius) & (distance <= outer_radius)
    if int(np.count_nonzero(ring)) < 24:
        inner_radius = 0.0
        outer_radius = min(3.0, float(np.max(positive)))
        ring = (~mask) & (distance > inner_radius) & (distance <= outer_radius)
        facts["fallback_used"] = True
    facts.update(
        {
            "inner_radius_px": round(float(inner_radius), 6),
            "outer_radius_px": round(float(outer_radius), 6),
            "pixel_count": int(np.count_nonzero(ring)),
        }
    )
    return np.ascontiguousarray(ring, dtype=bool), facts


















def _fill_component_facts(fill: np.ndarray) -> list[dict[str, Any]]:
    """Return JSON-safe geometry facts from contrast-resolved ink only.

    Compact, highly occupied components at punctuation scale are recorded as
    punctuation fragments.  They remain available for footprint accounting
    but cannot vote for character-cell scale.
    """

    try:
        import cv2

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.asarray(fill, dtype=np.uint8), connectivity=8
        )
    except Exception:
        return []
    facts: list[dict[str, Any]] = []
    for index in range(1, count):
        width = int(stats[index, 2])
        height = int(stats[index, 3])
        area = int(stats[index, 4])
        short = min(width, height)
        long = max(width, height)
        if area < 6 or short < 3 or long / max(1, short) > 2.2:
            continue
        occupancy = float(area) / float(max(1, width * height))
        x0 = int(stats[index, 0])
        y0 = int(stats[index, 1])
        component = np.asarray(
            labels[y0 : y0 + height, x0 : x0 + width] == index,
            dtype=np.uint8,
        )
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea) if contours else None
        contour_area = float(cv2.contourArea(contour)) if contour is not None else 0.0
        perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
        circularity = (
            float(4.0 * np.pi * contour_area / max(perimeter * perimeter, 1e-6))
            if contour_area > 0.0 and perimeter > 0.0
            else 0.0
        )
        hull = cv2.convexHull(contour) if contour is not None else None
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = contour_area / max(hull_area, 1e-6) if hull_area > 0.0 else 0.0
        # Dot punctuation is a normalized shape class, not an absolute pixel
        # tier.  The previous ``long <= 10`` gate allowed the same six-dot run
        # to become body text merely by scaling the source crop.
        normalized_compact_mark = bool(
            long / max(1, short) <= 1.60
            and occupancy >= 0.50
            and circularity >= 0.50
            and solidity >= 0.88
        )
        punctuation_fragment = bool(
            normalized_compact_mark
            and long / max(1, short) <= 1.35
        )
        facts.append(
            {
                "component_index": int(index),
                "bbox_xywh": [
                    int(stats[index, 0]),
                    int(stats[index, 1]),
                    width,
                    height,
                ],
                "center_xy": [
                    round(float(centroids[index, 0]), 6),
                    round(float(centroids[index, 1]), 6),
                ],
                "area_px": area,
                "width_px": float(width),
                "height_px": float(height),
                "long_span_px": float(long),
                "bbox_occupancy": round(occupancy, 8),
                "contour_circularity": round(circularity, 8),
                "contour_solidity": round(solidity, 8),
                "normalized_compact_mark": normalized_compact_mark,
                "punctuation_fragment": punctuation_fragment,
            }
        )
    compact_marks = [
        fact for fact in facts if bool(fact.get("normalized_compact_mark"))
    ]
    if len(compact_marks) >= 3:
        compact_tier = float(
            np.median(
                [float(fact.get("long_span_px") or 0.0) for fact in compact_marks]
            )
        )
        for fact in compact_marks:
            span = float(fact.get("long_span_px") or 0.0)
            if compact_tier > 0.0 and 0.65 <= span / compact_tier <= 1.35:
                fact["punctuation_fragment"] = True
    return facts












def _occupied_band_records(binary: np.ndarray, *, axis: int) -> list[dict[str, Any]]:
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
    return [
        {
            "start_px": int(start),
            "end_px": int(end),
            "span_px": float(end - start),
            "center_px": round((float(start) + float(end)) * 0.5, 6),
        }
        for start, end in zip(starts, ends)
        if end - start >= 2
    ]


def _summarize_source_text_footprint(
    fill: np.ndarray,
    *,
    component_facts: Sequence[Mapping[str, Any]],
    vertical_cell_size_px: float,
    vertical_scale_confidence: float,
    vertical_scale_support: str,
    vertical_scale_qualification: Mapping[str, Any],
    horizontal_cell_size_px: float,
    horizontal_scale_confidence: float,
    horizontal_scale_support: str,
    horizontal_scale_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish direction-neutral geometry without selecting writing mode."""

    binary = np.asarray(fill, dtype=bool)
    yy, xx = np.where(binary)
    union_bbox = []
    if xx.size and yy.size:
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        union_bbox = [x0, y0, x1 - x0, y1 - y0]
    summary: dict[str, Any] = {
        "resolved_ink_mask_sha256": _array_sha256(
            np.ascontiguousarray(binary, dtype=np.uint8)
        ),
        "union_bbox_xywh": union_bbox,
        "x_occupied_bands": _occupied_band_records(binary, axis=0),
        "y_occupied_bands": _occupied_band_records(binary, axis=1),
        "axis_profiles": {},
    }

    def axis_profile(
        *,
        direction: str,
        cross_axis: int,
        center_index: int,
        bbox_origin_index: int,
        bbox_span_index: int,
        inline_union_index: int,
        cell_size_px: float,
        scale_confidence: float,
        scale_support: str,
        scale_qualification: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile: dict[str, Any] = {
            "writing_direction": direction,
            "cross_axis_group_count": 0,
            "cross_axis_group_count_reliable": False,
            "cross_axis_group_centers_px": [],
            "cross_axis_group_spans_px": [],
            "inline_capacity": 0,
            "inline_capacity_reliable": False,
            "inline_capacity_provenance": "unavailable_source_scale",
            "confidence": 0.0,
            "reason": "unavailable_no_body_component_groups",
        }
        has_body_component = any(
            not bool(fact.get("punctuation_fragment"))
            for fact in component_facts
        )
        if not has_body_component and scale_support != "supported_density_decomposition":
            return profile
        cell_size = max(0.0, float(cell_size_px))
        if not (
            cell_size > 0.0
            and str(scale_support or "").startswith("supported_")
        ):
            profile["reason"] = "unavailable_source_scale_not_supported"
            return profile
        if union_bbox:
            inline_extent = float(union_bbox[inline_union_index])
            profile.update(
                {
                    "inline_capacity": max(
                        1,
                        int(round(inline_extent / max(1.0, cell_size))),
                    ),
                    "inline_capacity_reliable": True,
                    "inline_capacity_provenance": (
                        "qualified_source_union_inline_extent_over_cell_pitch"
                    ),
                }
            )

        density_spans = [
            float(value)
            for value in scale_qualification.get("density_spans") or ()
            if float(value) > 0.0
        ]
        if scale_support == "supported_density_decomposition":
            minimum_occupancy = int(
                scale_qualification.get("density_minimum_occupancy") or 0
            )
            density_runs = (
                _projection_runs_at_min_occupancy(
                    binary,
                    axis=cross_axis,
                    minimum_occupancy=minimum_occupancy,
                )
                if minimum_occupancy > 0
                else []
            )
            if not (
                len(density_runs) == len(density_spans)
                and len(density_runs) >= 2
                and all(
                    abs(float(end - start) - span) <= 1.0
                    for (start, end), span in zip(density_runs, density_spans)
                )
            ):
                profile["reason"] = (
                    "unavailable_density_projection_identity_mismatch"
                )
                return profile
            profile.update(
                {
                    "cross_axis_group_count": len(density_runs),
                    "cross_axis_group_count_reliable": True,
                    "cross_axis_group_centers_px": [
                        round((float(start) + float(end)) * 0.5, 6)
                        for start, end in density_runs
                    ],
                    "cross_axis_group_spans_px": [
                        float(end - start) for start, end in density_runs
                    ],
                    "confidence": round(
                        min(0.94, max(0.72, float(scale_confidence))), 8
                    ),
                    "reason": "supported_qualified_density_projection_groups",
                }
            )
            return profile

        pitch_tracks = [
            dict(record)
            for record in (
                scale_qualification.get("component_pitch_tracks") or ()
            )
            if isinstance(record, Mapping)
        ]
        pitch_record_count = int(
            scale_qualification.get("center_pitch_candidate_count") or 0
        )
        if scale_support == "supported_repeated_center_pitch":
            reliable_pitch_tracks = [
                record
                for record in pitch_tracks
                if int(record.get("qualified_pitch_delta_count") or 0) > 0
            ]
            profile["cell_pitch_evidence"] = {
                "candidate_count": pitch_record_count,
                "component_track_count": len(reliable_pitch_tracks),
                "independent_sources": bool(
                    scale_qualification.get(
                        "center_pitch_independent_sources"
                    )
                ),
            }
            if pitch_record_count < 3 and not bool(
                scale_qualification.get(
                    "center_pitch_independent_sources"
                )
            ):
                profile["reason"] = (
                    "unavailable_repeated_pitch_records_incomplete"
                )
                return profile

        size_key = "width_px" if cross_axis == 0 else "height_px"
        minimum_body_ratio = (
            0.45
            if scale_support == "supported_repeated_center_pitch"
            else 0.65
        )
        body_facts = [
            fact
            for fact in component_facts
            if not bool(fact.get("punctuation_fragment"))
            and cell_size * minimum_body_ratio
            <= float(fact.get(size_key) or 0.0)
            <= cell_size * 1.35
        ]
        if len(body_facts) < 2:
            return profile
        clusters: list[list[Mapping[str, Any]]] = []
        center_tolerance = max(3.0, cell_size * 0.55)
        for fact in sorted(
            body_facts,
            key=lambda item: float(
                (item.get("center_xy") or [0.0, 0.0])[center_index]
            ),
        ):
            center = float(
                (fact.get("center_xy") or [0.0, 0.0])[center_index]
            )
            if not clusters:
                clusters.append([fact])
                continue
            previous_centers = [
                float(
                    (item.get("center_xy") or [0.0, 0.0])[center_index]
                )
                for item in clusters[-1]
            ]
            if abs(center - float(np.median(previous_centers))) <= center_tolerance:
                clusters[-1].append(fact)
            else:
                clusters.append([fact])

        group_records: list[dict[str, Any]] = []
        for cluster in clusters:
            boxes = [list(item.get("bbox_xywh") or []) for item in cluster]
            if any(len(box) != 4 for box in boxes):
                continue
            centers = [
                float((item.get("center_xy") or [0.0, 0.0])[center_index])
                for item in cluster
            ]
            start = min(int(box[bbox_origin_index]) for box in boxes)
            end = max(
                int(box[bbox_origin_index]) + int(box[bbox_span_index])
                for box in boxes
            )
            group_records.append(
                {
                    "center_px": round(float(np.median(centers)), 6),
                    "span_px": float(end - start),
                }
            )
        if not group_records:
            profile["reason"] = "unavailable_body_cross_axis_groups"
            return profile
        profile.update(
            {
                "cross_axis_group_count": len(group_records),
                "cross_axis_group_count_reliable": True,
                "cross_axis_group_centers_px": [
                    float(record["center_px"]) for record in group_records
                ],
                "cross_axis_group_spans_px": [
                    float(record["span_px"]) for record in group_records
                ],
                "confidence": round(
                    min(
                        0.96,
                        max(0.72, float(scale_confidence))
                        + min(0.08, (len(body_facts) - 2) * 0.02),
                    ),
                    8,
                ),
                "reason": (
                    "supported_repeated_pitch_body_cross_axis_groups"
                    if scale_support == "supported_repeated_center_pitch"
                    else "supported_qualified_body_cross_axis_groups"
                ),
            }
        )
        return profile

    summary["axis_profiles"] = {
        "ttb": axis_profile(
            direction="ttb",
            cross_axis=0,
            center_index=0,
            bbox_origin_index=0,
            bbox_span_index=2,
            inline_union_index=3,
            cell_size_px=vertical_cell_size_px,
            scale_confidence=vertical_scale_confidence,
            scale_support=vertical_scale_support,
            scale_qualification=vertical_scale_qualification,
        ),
        "ltr": axis_profile(
            direction="ltr",
            cross_axis=1,
            center_index=1,
            bbox_origin_index=1,
            bbox_span_index=3,
            inline_union_index=2,
            cell_size_px=horizontal_cell_size_px,
            scale_confidence=horizontal_scale_confidence,
            scale_support=horizontal_scale_support,
            scale_qualification=horizontal_scale_qualification,
        ),
    }
    return summary


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
