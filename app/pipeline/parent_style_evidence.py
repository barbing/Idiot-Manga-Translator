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
_PERCEPTUAL_STYLE_AXES_VERSION = "authorized_perceptual_style_axes_v2"
_PERCEPTUAL_STYLE_PROVENANCE = "cleanup_mask_authorized_source_style_view_v1"
_PERCEPTUAL_STYLE_FACT_SET_PREFIX = "authorized_perceptual_fact_set_v1:"
_PERCEPTUAL_STYLE_AXES = ("fill", "outline", "shadow", "rotation")
_ADDITIVE_FILL_MIN_CHROMA = 48.0
_ADDITIVE_FILL_MIN_CLUSTER_PIXELS = 24
_ADDITIVE_FILL_MIN_CORE_CHROMATIC_FRACTION = 0.38
_ADDITIVE_FILL_MAX_COLOR_DISPERSION = 24.0
_CANONICAL_ROLE_MIN_CLUSTER_PIXELS = 24
_CANONICAL_ROLE_MAX_COLOR_DISPERSION = 24.0
_CANONICAL_ROLE_MODE_MERGE_DISTANCE_RGB = 144.0
_CANONICAL_ROLE_MIN_MODE_DISTANCE_RGB = 160.0
_CANONICAL_ROLE_MAX_ASSIGNMENT_DISTANCE_RGB = 96.0
_CANONICAL_ROLE_MIN_ASSIGNMENT_MARGIN_RGB = 12.0
_CANONICAL_ROLE_MIN_MEDIAL_DEPTH_PX = 1.4
_CANONICAL_ROLE_TARGET_MEDIAL_DEPTH_PX = 3.2
_CANONICAL_ROLE_MIN_MEDIAL_PIXELS = 8
_CANONICAL_ROLE_MIN_MEDIAL_SCORE = 0.20
_CANONICAL_ROLE_MIN_MEDIAL_SCORE_MARGIN = 0.06
_CANONICAL_ROLE_MIN_MEDIAL_SCORE_RATIO = 1.20
_CANONICAL_ROLE_MIN_RADIAL_WIDTH_PX = 1.5
_CANONICAL_ROLE_MAX_RADIAL_WIDTH_PX = 16.0
_CANONICAL_ROLE_MIN_PAIR_MASK_FRACTION = 0.70
_CANONICAL_ROLE_MIN_SHELL_RING_RECALL = 0.85
_CANONICAL_ROLE_MIN_RING_SHELL_PRECISION = 0.38
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
AUTHORIZED_STYLE_SPATIAL_FACT_SET_VERSION = (
    "authorized_style_spatial_fact_set_v1"
)
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
class _SpatialPaintCohort:
    """One immutable paint cohort shared by every source-style observer."""

    cohort_index: int
    quantized_key: tuple[int, int, int]
    color: str
    median_rgb: tuple[float, float, float]
    pixel_count: int
    mask_fraction: float
    color_dispersion_rgb: float
    chroma_median: float
    chromatic_pixel_count: int
    chromatic_fraction: float
    bbox_xywh: tuple[int, int, int, int]
    bbox_occupancy: float
    border_margin_px: int
    depth_p50_px: float
    depth_p75_px: float
    depth_ratio: float
    significant_component_count: int
    mask: Any = field(default=None, repr=False, compare=False)
    chromatic_mask: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "cohort_index": int(self.cohort_index),
            "quantized_key": list(self.quantized_key),
            "color": self.color,
            "pixel_count": int(self.pixel_count),
            "mask_fraction": round(float(self.mask_fraction), 8),
            "color_dispersion_rgb": round(
                float(self.color_dispersion_rgb), 8
            ),
            "chroma_median": round(float(self.chroma_median), 8),
            "chromatic_pixel_count": int(self.chromatic_pixel_count),
            "chromatic_fraction": round(float(self.chromatic_fraction), 8),
            "bbox_xywh": list(self.bbox_xywh),
            "bbox_occupancy": round(float(self.bbox_occupancy), 8),
            "border_margin_px": int(self.border_margin_px),
            "depth_p50_px": round(float(self.depth_p50_px), 8),
            "depth_p75_px": round(float(self.depth_p75_px), 8),
            "depth_ratio": round(float(self.depth_ratio), 8),
            "significant_component_count": int(
                self.significant_component_count
            ),
        }


@dataclass(frozen=True)
class _CanonicalPaintMode:
    """One stable paint mode assembled from nearby authorized cohorts."""

    representative_cohort_index: int
    color: str
    median_rgb: tuple[float, float, float]
    pixel_count: int
    medial_pixel_count: int
    medial_fraction: float
    medial_depth_p75_px: float
    medial_score: float
    cohort_indices: tuple[int, ...] = ()
    mask: Any = field(default=None, repr=False, compare=False)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "representative_cohort_index": int(
                self.representative_cohort_index
            ),
            "cohort_indices": [int(value) for value in self.cohort_indices],
            "color": self.color,
            "pixel_count": int(self.pixel_count),
            "medial_pixel_count": int(self.medial_pixel_count),
            "medial_fraction": round(float(self.medial_fraction), 8),
            "medial_depth_p75_px": round(
                float(self.medial_depth_p75_px), 8
            ),
            "medial_score": round(float(self.medial_score), 8),
        }


@dataclass(frozen=True)
class _SpatialOutlineRole:
    """One uniquely qualified core/shell relation from shared paint facts."""

    core_cohort_index: int
    shell_cohort_index: int
    core_color: str
    shell_color: str
    width_px: float
    pair_mask_fraction: float
    core_shell_depth_delta_px: float
    shell_ring_recall: float
    ring_shell_precision: float
    color_distance_rgb: float
    confidence: float

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "core_cohort_index": int(self.core_cohort_index),
            "shell_cohort_index": int(self.shell_cohort_index),
            "core_color": self.core_color,
            "shell_color": self.shell_color,
            "width_px": round(float(self.width_px), 8),
            "pair_mask_fraction": round(float(self.pair_mask_fraction), 8),
            "core_shell_depth_delta_px": round(
                float(self.core_shell_depth_delta_px), 8
            ),
            "shell_ring_recall": round(float(self.shell_ring_recall), 8),
            "ring_shell_precision": round(
                float(self.ring_shell_precision), 8
            ),
            "color_distance_rgb": round(float(self.color_distance_rgb), 8),
            "confidence": round(float(self.confidence), 8),
        }


@dataclass(frozen=True)
class _CanonicalSpatialRoles:
    """The only core/shell/effect role decision for one authorized mask."""

    core_role_status: str
    core_role_reason: str
    core_resolution: str
    core_color: str
    support_color: str
    core_mask: Any = field(default=None, repr=False, compare=False)
    shell_mask: Any = field(default=None, repr=False, compare=False)
    effect_mask: Any = field(default=None, repr=False, compare=False)
    outline_role: _SpatialOutlineRole | None = None
    outline_role_status: str = "unavailable"
    outline_role_reason: str = ""
    medial_threshold_px: float = 0.0
    paint_modes: tuple[_CanonicalPaintMode, ...] = ()
    role_support: Mapping[str, Any] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "core_role_status": self.core_role_status,
            "core_role_reason": self.core_role_reason,
            "core_resolution": self.core_resolution,
            "medial_threshold_px": round(
                float(self.medial_threshold_px), 8
            ),
            "role_support": _json_safe_mapping(self.role_support),
            "paint_modes": [mode.to_audit_dict() for mode in self.paint_modes],
        }


@dataclass(frozen=True)
class AuthorizedStyleSpatialFactSet:
    """Runtime-only, identity-bound spatial roles for one authorized parent.

    Arrays are immutable and intentionally never serialized. All source-style
    axes consume this same fact set; no observer may rebuild its own competing
    paint cohorts or external carrier ring.
    """

    contract_version: str
    fact_set_id: str
    source_identity: Mapping[str, Any]
    core_resolution: str
    core_role_status: str
    core_role_reason: str
    core_color: str
    support_color: str
    fill_polarity: str
    paint_cohorts: tuple[_SpatialPaintCohort, ...] = ()
    outline_role: _SpatialOutlineRole | None = None
    outline_role_status: str = "unavailable"
    outline_role_reason: str = ""
    core_role_support: Mapping[str, Any] = field(default_factory=dict)
    source_rgb: Any = field(default=None, repr=False, compare=False)
    authorized_mask: Any = field(default=None, repr=False, compare=False)
    character_core_mask: Any = field(default=None, repr=False, compare=False)
    concentric_shell_mask: Any = field(default=None, repr=False, compare=False)
    displaced_effect_mask: Any = field(default=None, repr=False, compare=False)
    external_surface_ring_mask: Any = field(
        default=None, repr=False, compare=False
    )
    detector_primary_rgb: Any = field(default=None, repr=False, compare=False)
    detector_neutral_rgb: Any = field(default=None, repr=False, compare=False)

    def cohort(self, index: int) -> _SpatialPaintCohort | None:
        return next(
            (
                cohort
                for cohort in self.paint_cohorts
                if cohort.cohort_index == int(index)
            ),
            None,
        )

    def audit_summary(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "fact_set_id": self.fact_set_id,
            "core_resolution": self.core_resolution,
            "core_role_status": self.core_role_status,
            "core_role_reason": self.core_role_reason,
            "core_role_support": _json_safe_mapping(self.core_role_support),
            "core_color": self.core_color,
            "support_color": self.support_color,
            "fill_polarity": self.fill_polarity,
            "paint_cohort_count": len(self.paint_cohorts),
            "character_core_pixel_count": int(
                np.count_nonzero(self.character_core_mask)
            ),
            "concentric_shell_pixel_count": int(
                np.count_nonzero(self.concentric_shell_mask)
            ),
            "displaced_effect_pixel_count": int(
                np.count_nonzero(self.displaced_effect_mask)
            ),
            "paint_cohorts": [
                cohort.to_audit_dict() for cohort in self.paint_cohorts
            ],
            "outline_role_status": self.outline_role_status,
            "outline_role_reason": self.outline_role_reason,
            "outline_role": (
                self.outline_role.to_audit_dict()
                if self.outline_role is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _AuthorizedStyleCropMeasurement:
    metrics: Mapping[str, Any]
    source_text_footprint: Mapping[str, Any]


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


def _spatial_paint_cohorts(
    source: np.ndarray,
    mask: np.ndarray,
) -> tuple[_SpatialPaintCohort, ...]:
    """Build the only paint-cohort inventory used by source-style axes."""

    try:
        import cv2
    except Exception:
        return ()
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count <= 0:
        return ()
    selected = source[mask].astype(np.float32)
    quantized = np.clip(np.floor((selected + 8.0) / 16.0), 0, 15).astype(
        np.uint8
    )
    keys, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )
    order = sorted(
        range(len(counts)),
        key=lambda index: (
            -int(counts[index]),
            tuple(int(value) for value in keys[index]),
        ),
    )[:16]
    authorized_distance = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    overall_depth = float(np.percentile(authorized_distance[mask], 75))
    flat_indices = np.flatnonzero(mask)
    selected_chroma = selected.max(axis=1) - selected.min(axis=1)
    total_chromatic = int(
        np.count_nonzero(selected_chroma >= _ADDITIVE_FILL_MIN_CHROMA)
    )
    cohorts: list[_SpatialPaintCohort] = []
    for cohort_index in order:
        members = inverse == cohort_index
        count = int(np.count_nonzero(members))
        if count < 8:
            continue
        pixels = selected[members]
        median = np.median(pixels, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(pixels - median[None, :], axis=1))
        )
        cohort_mask = np.zeros(mask.size, dtype=bool)
        cohort_mask[flat_indices[members]] = True
        cohort_mask = cohort_mask.reshape(mask.shape)
        chromatic_members = members & (
            selected_chroma >= _ADDITIVE_FILL_MIN_CHROMA
        )
        chromatic_mask = np.zeros(mask.size, dtype=bool)
        chromatic_mask[flat_indices[chromatic_members]] = True
        chromatic_mask = chromatic_mask.reshape(mask.shape)
        yy, xx = np.where(cohort_mask)
        if xx.size <= 0 or yy.size <= 0:
            continue
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        occupancy = count / max(1, (x1 - x0) * (y1 - y0))
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            cohort_mask.astype(np.uint8), connectivity=8
        )
        significant = int(
            sum(1 for row in stats[1:] if int(row[cv2.CC_STAT_AREA]) >= 8)
        )
        chromatic_count = int(np.count_nonzero(chromatic_mask))
        cohorts.append(
            _SpatialPaintCohort(
                cohort_index=int(cohort_index),
                quantized_key=tuple(int(value) for value in keys[cohort_index]),
                color=_rgb_hex(median),
                median_rgb=tuple(float(value) for value in median),
                pixel_count=count,
                mask_fraction=count / max(1, pixel_count),
                color_dispersion_rgb=dispersion,
                chroma_median=float(np.max(median) - np.min(median)),
                chromatic_pixel_count=chromatic_count,
                chromatic_fraction=(
                    chromatic_count / max(1, total_chromatic)
                ),
                bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
                bbox_occupancy=occupancy,
                border_margin_px=_mask_border_margin(cohort_mask),
                depth_p50_px=float(
                    np.percentile(authorized_distance[cohort_mask], 50)
                ),
                depth_p75_px=float(
                    np.percentile(authorized_distance[cohort_mask], 75)
                ),
                depth_ratio=float(
                    np.percentile(authorized_distance[cohort_mask], 75)
                    / max(overall_depth, 1e-6)
                ),
                significant_component_count=significant,
                mask=_readonly_array(cohort_mask, dtype=bool),
                chromatic_mask=_readonly_array(chromatic_mask, dtype=bool),
            )
        )
    return tuple(cohorts)


def _canonical_paint_modes(
    source: np.ndarray,
    mask: np.ndarray,
    cohorts: Sequence[_SpatialPaintCohort],
) -> tuple[tuple[_CanonicalPaintMode, ...], float]:
    """Group stable paint cohorts and score their medial topology."""

    try:
        import cv2
    except Exception:
        return (), 0.0
    authorized_count = int(np.count_nonzero(mask))
    if authorized_count <= 0:
        return (), 0.0
    authorized_distance = cv2.distanceTransform(
        np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 5
    )
    authorized_depths = authorized_distance[mask]
    if authorized_depths.size <= 0:
        return (), 0.0
    medial_threshold = max(
        _CANONICAL_ROLE_MIN_MEDIAL_DEPTH_PX,
        min(
            _CANONICAL_ROLE_TARGET_MEDIAL_DEPTH_PX,
            float(np.percentile(authorized_depths, 90)),
        ),
    )
    stable = [
        cohort
        for cohort in cohorts
        if cohort.pixel_count >= _CANONICAL_ROLE_MIN_CLUSTER_PIXELS
        and cohort.color_dispersion_rgb
        <= _CANONICAL_ROLE_MAX_COLOR_DISPERSION
        and cohort.significant_component_count >= 1
    ]
    stable.sort(
        key=lambda item: (
            -item.pixel_count,
            item.quantized_key,
        )
    )
    groups: list[list[_SpatialPaintCohort]] = []
    for cohort in stable:
        cohort_rgb = np.asarray(cohort.median_rgb, dtype=np.float32)
        eligible_groups: list[tuple[float, int]] = []
        for group_index, group in enumerate(groups):
            member_distances = [
                float(
                    np.linalg.norm(
                        cohort_rgb
                        - np.asarray(item.median_rgb, dtype=np.float32)
                    )
                )
                for item in group
            ]
            if (
                not member_distances
                or max(member_distances)
                > _CANONICAL_ROLE_MODE_MERGE_DISTANCE_RGB
            ):
                continue
            weights = np.asarray(
                [item.pixel_count for item in group], dtype=np.float32
            )
            colors = np.asarray(
                [item.median_rgb for item in group], dtype=np.float32
            )
            group_rgb = np.average(colors, axis=0, weights=weights)
            distance = float(np.linalg.norm(cohort_rgb - group_rgb))
            eligible_groups.append((distance, group_index))
        eligible_groups.sort()
        if (
            len(eligible_groups) == 1
            or (
                len(eligible_groups) > 1
                and eligible_groups[1][0] - eligible_groups[0][0]
                >= _CANONICAL_ROLE_MIN_ASSIGNMENT_MARGIN_RGB
            )
        ):
            groups[eligible_groups[0][1]].append(cohort)
        else:
            groups.append([cohort])

    maximum_depth = max(
        _CANONICAL_ROLE_MIN_MEDIAL_DEPTH_PX,
        float(np.percentile(authorized_depths, 95)),
    )
    authorized_medial_count = max(
        1,
        int(
            np.count_nonzero(
                mask & (authorized_distance >= medial_threshold)
            )
        ),
    )
    modes: list[_CanonicalPaintMode] = []
    for group in groups:
        group_mask = np.zeros(mask.shape, dtype=bool)
        for cohort in group:
            group_mask |= np.asarray(cohort.mask, dtype=bool)
        group_mask &= mask
        count = int(np.count_nonzero(group_mask))
        if count < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS:
            continue
        pixels = source[group_mask].astype(np.float32)
        median_rgb = np.median(pixels, axis=0)
        medial = group_mask & (authorized_distance >= medial_threshold)
        medial_count = int(np.count_nonzero(medial))
        medial_fraction = medial_count / max(1, count)
        medial_depth = (
            float(np.percentile(authorized_distance[medial], 75))
            if medial_count
            else 0.0
        )
        medial_score = (
            0.65 * medial_fraction
            + 0.25 * min(1.0, medial_depth / max(maximum_depth, 1e-6))
            + 0.10 * min(
                1.0, medial_count / authorized_medial_count
            )
        )
        representative = max(
            group,
            key=lambda item: (
                item.pixel_count,
                item.depth_p75_px,
                -item.cohort_index,
            ),
        )
        modes.append(
            _CanonicalPaintMode(
                representative_cohort_index=representative.cohort_index,
                color=_rgb_hex(median_rgb),
                median_rgb=tuple(float(value) for value in median_rgb),
                pixel_count=count,
                medial_pixel_count=medial_count,
                medial_fraction=medial_fraction,
                medial_depth_p75_px=medial_depth,
                medial_score=medial_score,
                cohort_indices=tuple(
                    sorted(item.cohort_index for item in group)
                ),
                mask=_readonly_array(group_mask, dtype=bool),
            )
        )
    modes.sort(
        key=lambda item: (
            -item.medial_score,
            -item.medial_pixel_count,
            -item.pixel_count,
            item.color,
        )
    )
    return tuple(modes), float(medial_threshold)


def _canonical_mode_distance(
    first: _CanonicalPaintMode,
    second: _CanonicalPaintMode,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first.median_rgb, dtype=np.float32)
            - np.asarray(second.median_rgb, dtype=np.float32)
        )
    )


def _canonical_mode_assignments(
    source: np.ndarray,
    mask: np.ndarray,
    modes: Sequence[_CanonicalPaintMode],
) -> tuple[np.ndarray, ...]:
    """Expand raw mode support without consuming ambiguous/effect pixels."""

    if not modes:
        return ()
    pixels = source.astype(np.float32)
    distances = np.stack(
        [
            np.linalg.norm(
                pixels
                - np.asarray(mode.median_rgb, dtype=np.float32)[None, None, :],
                axis=2,
            )
            for mode in modes
        ],
        axis=0,
    )
    nearest = np.argmin(distances, axis=0)
    nearest_distance = np.min(distances, axis=0)
    if len(modes) > 1:
        second_distance = np.partition(distances, 1, axis=0)[1]
        assignment_margin = second_distance - nearest_distance
    else:
        assignment_margin = np.full(mask.shape, np.inf, dtype=np.float32)
    assignments = [
        np.asarray(mode.mask, dtype=bool).copy() & mask for mode in modes
    ]
    already_claimed = np.logical_or.reduce(assignments)
    expandable = (
        mask
        & ~already_claimed
        & (
            nearest_distance
            <= _CANONICAL_ROLE_MAX_ASSIGNMENT_DISTANCE_RGB
        )
        & (
            assignment_margin
            >= _CANONICAL_ROLE_MIN_ASSIGNMENT_MARGIN_RGB
        )
    )
    for index, assignment in enumerate(assignments):
        assignment |= expandable & (nearest == index)
    return tuple(
        np.ascontiguousarray(assignment, dtype=bool)
        for assignment in assignments
    )


def _canonical_shell_candidate(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    core_mode: _CanonicalPaintMode,
    shell_modes: Sequence[_CanonicalPaintMode],
    core_mask: np.ndarray,
    shell_mask: np.ndarray,
) -> tuple[_SpatialOutlineRole | None, dict[str, Any]]:
    """Qualify one aggregate shell using authorized radial topology."""

    representative = max(
        shell_modes,
        key=lambda mode: (
            mode.pixel_count,
            mode.medial_pixel_count,
            -mode.representative_cohort_index,
        ),
    )
    audit: dict[str, Any] = {
        "shell_mode_colors": [mode.color for mode in shell_modes],
        "shell_mode_cohort_indices": [
            int(mode.representative_cohort_index)
            for mode in shell_modes
        ],
        "status": "unavailable",
    }
    try:
        import cv2
    except Exception:
        audit["reason"] = "canonical_shell_backend_unavailable"
        return None, audit
    core_count = int(np.count_nonzero(core_mask))
    shell_count = int(np.count_nonzero(shell_mask))
    if (
        core_count < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS
        or shell_count < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS
    ):
        audit["reason"] = "canonical_shell_insufficient_paint_support"
        return None, audit
    shell_rgb = np.median(
        source[shell_mask].astype(np.float32), axis=0
    )
    color_distance = float(
        np.linalg.norm(
            np.asarray(core_mode.median_rgb, dtype=np.float32)
            - shell_rgb
        )
    )
    if color_distance < _CANONICAL_ROLE_MIN_MODE_DISTANCE_RGB:
        audit["reason"] = "canonical_shell_insufficient_color_separation"
        return None, audit
    distance_to_core = cv2.distanceTransform(
        (~np.asarray(core_mask, dtype=bool)).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    radial_values = distance_to_core[shell_mask]
    if radial_values.size <= 0:
        audit["reason"] = "canonical_shell_radial_support_unavailable"
        return None, audit
    radial_p90 = float(np.percentile(radial_values, 90))
    audit["radial_distance_p90_px"] = round(radial_p90, 6)
    if radial_p90 <= _CANONICAL_ROLE_MIN_RADIAL_WIDTH_PX:
        audit["reason"] = "canonical_native_antialias_shell_rejected"
        return None, audit
    if radial_p90 > _CANONICAL_ROLE_MAX_RADIAL_WIDTH_PX:
        audit["reason"] = "canonical_shell_radial_width_excessive"
        return None, audit
    radius = max(1, int(np.ceil(radial_p90)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    predicted_ring = (
        cv2.dilate(core_mask.astype(np.uint8), kernel).astype(bool)
        & mask
        & ~core_mask
    )
    intersection = int(np.count_nonzero(predicted_ring & shell_mask))
    shell_recall = intersection / max(1, shell_count)
    ring_precision = intersection / max(
        1, int(np.count_nonzero(predicted_ring))
    )
    pair_fraction = int(
        np.count_nonzero(core_mask | shell_mask)
    ) / max(1, int(np.count_nonzero(mask)))
    audit.update(
        {
            "pair_mask_fraction": round(pair_fraction, 8),
            "shell_ring_recall": round(shell_recall, 8),
            "ring_shell_precision": round(ring_precision, 8),
            "color_distance_rgb": round(color_distance, 6),
        }
    )
    if (
        pair_fraction < _CANONICAL_ROLE_MIN_PAIR_MASK_FRACTION
        or shell_recall < _CANONICAL_ROLE_MIN_SHELL_RING_RECALL
        or ring_precision < _CANONICAL_ROLE_MIN_RING_SHELL_PRECISION
    ):
        audit["reason"] = "canonical_shell_topology_unqualified"
        return None, audit
    shell_depth = float(
        np.average(
            [mode.medial_depth_p75_px for mode in shell_modes],
            weights=[max(1, mode.pixel_count) for mode in shell_modes],
        )
    )
    depth_delta = core_mode.medial_depth_p75_px - shell_depth
    geometric_support = min(pair_fraction, shell_recall, ring_precision)
    role = _SpatialOutlineRole(
        core_cohort_index=core_mode.representative_cohort_index,
        shell_cohort_index=representative.representative_cohort_index,
        core_color=core_mode.color,
        shell_color=_rgb_hex(shell_rgb),
        width_px=radial_p90,
        pair_mask_fraction=pair_fraction,
        core_shell_depth_delta_px=depth_delta,
        shell_ring_recall=shell_recall,
        ring_shell_precision=ring_precision,
        color_distance_rgb=color_distance,
        confidence=min(0.98, 0.62 + 0.34 * geometric_support),
    )
    audit["status"] = "supported"
    audit["reason"] = "canonical_unique_concentric_shell"
    return role, audit
def _resolve_canonical_spatial_roles(
    source: np.ndarray,
    mask: np.ndarray,
    cohorts: Sequence[_SpatialPaintCohort],
) -> _CanonicalSpatialRoles:
    """Resolve one topology-first core/shell/effect fact set."""

    empty = np.zeros(mask.shape, dtype=bool)
    modes, medial_threshold = _canonical_paint_modes(source, mask, cohorts)
    supported_modes = [
        mode
        for mode in modes
        if mode.medial_pixel_count >= _CANONICAL_ROLE_MIN_MEDIAL_PIXELS
        and mode.medial_score >= _CANONICAL_ROLE_MIN_MEDIAL_SCORE
    ]
    base_support: dict[str, Any] = {
        "authorized_pixel_count": int(np.count_nonzero(mask)),
        "medial_threshold_px": round(medial_threshold, 6),
        "supported_mode_count": len(supported_modes),
    }
    if not supported_modes:
        return _CanonicalSpatialRoles(
            core_role_status="unavailable",
            core_role_reason="canonical_medial_paint_core_unavailable",
            core_resolution="unresolved_canonical_medial_topology",
            core_color="",
            support_color="",
            core_mask=_readonly_array(empty, dtype=bool),
            shell_mask=_readonly_array(empty, dtype=bool),
            effect_mask=_readonly_array(empty, dtype=bool),
            medial_threshold_px=medial_threshold,
            paint_modes=modes,
            role_support=MappingProxyType(base_support),
        )

    core_mode = supported_modes[0]
    competing_modes = [
        mode
        for mode in supported_modes[1:]
        if _canonical_mode_distance(core_mode, mode)
        >= _CANONICAL_ROLE_MIN_MODE_DISTANCE_RGB
    ]
    score_margin = (
        core_mode.medial_score - competing_modes[0].medial_score
        if competing_modes
        else 1.0
    )
    score_ratio = (
        core_mode.medial_score
        / max(competing_modes[0].medial_score, 1e-6)
        if competing_modes
        else float("inf")
    )
    base_support.update(
        {
            "selected_core_color": core_mode.color,
            "selected_core_cohort_index": int(
                core_mode.representative_cohort_index
            ),
            "selected_core_medial_score": round(
                core_mode.medial_score, 8
            ),
            "competing_core_color": (
                competing_modes[0].color if competing_modes else ""
            ),
            "core_medial_score_margin": round(score_margin, 8),
            "core_medial_score_ratio": (
                round(score_ratio, 8)
                if np.isfinite(score_ratio)
                else "infinite"
            ),
        }
    )
    if (
        competing_modes
        and (
            score_margin < _CANONICAL_ROLE_MIN_MEDIAL_SCORE_MARGIN
            or score_ratio < _CANONICAL_ROLE_MIN_MEDIAL_SCORE_RATIO
        )
    ):
        return _CanonicalSpatialRoles(
            core_role_status="ambiguous",
            core_role_reason="canonical_medial_paint_core_competing_modes",
            core_resolution="unresolved_canonical_medial_topology",
            core_color="",
            support_color="",
            core_mask=_readonly_array(empty, dtype=bool),
            shell_mask=_readonly_array(empty, dtype=bool),
            effect_mask=_readonly_array(empty, dtype=bool),
            medial_threshold_px=medial_threshold,
            paint_modes=modes,
            role_support=MappingProxyType(base_support),
        )

    assignments = _canonical_mode_assignments(source, mask, modes)
    core_index = modes.index(core_mode)
    core_mask = np.asarray(assignments[core_index], dtype=bool).copy()
    core_mode_indices = {core_index}
    for mode_index, mode in enumerate(modes):
        if (
            mode_index != core_index
            and _canonical_mode_distance(core_mode, mode)
            < _CANONICAL_ROLE_MIN_MODE_DISTANCE_RGB
        ):
            core_mask |= np.asarray(assignments[mode_index], dtype=bool)
            core_mode_indices.add(mode_index)
    if int(np.count_nonzero(core_mask)) < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS:
        return _CanonicalSpatialRoles(
            core_role_status="unavailable",
            core_role_reason="canonical_medial_paint_core_insufficient_support",
            core_resolution="unresolved_canonical_medial_topology",
            core_color="",
            support_color="",
            core_mask=_readonly_array(empty, dtype=bool),
            shell_mask=_readonly_array(empty, dtype=bool),
            effect_mask=_readonly_array(empty, dtype=bool),
            medial_threshold_px=medial_threshold,
            paint_modes=modes,
            role_support=MappingProxyType(base_support),
        )

    shell_candidates: list[
        tuple[_SpatialOutlineRole, tuple[int, ...], np.ndarray, dict[str, Any]]
    ] = []
    shell_audits: list[dict[str, Any]] = []
    radial_groups: dict[str, list[int]] = {
        "darker": [],
        "lighter": [],
    }
    radial_mode_audits: list[dict[str, Any]] = []
    try:
        import cv2

        distance_to_core = cv2.distanceTransform(
            (~core_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
        core_luma = float(
            np.dot(
                np.asarray(core_mode.median_rgb, dtype=np.float32),
                np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
            )
        )
        for mode_index, mode in enumerate(modes):
            if mode_index in core_mode_indices:
                continue
            mode_mask = np.asarray(assignments[mode_index], dtype=bool)
            mode_count = int(np.count_nonzero(mode_mask))
            if mode_count < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS:
                continue
            radial_values = distance_to_core[mode_mask]
            if radial_values.size <= 0:
                continue
            radial_p90 = float(np.percentile(radial_values, 90))
            radius = max(1, int(np.ceil(radial_p90)))
            predicted_ring = (
                cv2.dilate(
                    core_mask.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (radius * 2 + 1, radius * 2 + 1),
                    ),
                ).astype(bool)
                & mask
                & ~core_mask
            )
            radial_recall = int(
                np.count_nonzero(predicted_ring & mode_mask)
            ) / max(1, mode_count)
            mode_luma = float(
                np.dot(
                    np.asarray(mode.median_rgb, dtype=np.float32),
                    np.asarray(
                        (0.2126, 0.7152, 0.0722),
                        dtype=np.float32,
                    ),
                )
            )
            polarity = "lighter" if mode_luma >= core_luma else "darker"
            qualified = bool(
                radial_p90 <= _CANONICAL_ROLE_MAX_RADIAL_WIDTH_PX
                and radial_recall >= _CANONICAL_ROLE_MIN_SHELL_RING_RECALL
            )
            radial_mode_audits.append(
                {
                    "mode_color": mode.color,
                    "mode_cohort_index": int(
                        mode.representative_cohort_index
                    ),
                    "radial_polarity": polarity,
                    "radial_distance_p90_px": round(radial_p90, 6),
                    "radial_recall": round(radial_recall, 8),
                    "aggregate_shell_eligible": qualified,
                }
            )
            if qualified:
                radial_groups[polarity].append(mode_index)
    except Exception:
        radial_mode_audits.append(
            {
                "aggregate_shell_eligible": False,
                "reason": "canonical_shell_grouping_backend_unavailable",
            }
        )
    base_support["radial_shell_modes"] = radial_mode_audits

    for polarity, shell_indices in radial_groups.items():
        if not shell_indices:
            continue
        aggregate_shell = np.logical_or.reduce(
            [
                np.asarray(assignments[index], dtype=bool)
                for index in shell_indices
            ]
        )
        role, audit = _canonical_shell_candidate(
            source,
            mask,
            core_mode=core_mode,
            shell_modes=[modes[index] for index in shell_indices],
            core_mask=core_mask,
            shell_mask=aggregate_shell,
        )
        audit["radial_polarity"] = polarity
        shell_audits.append(audit)
        if role is not None:
            shell_candidates.append(
                (
                    role,
                    tuple(shell_indices),
                    np.asarray(aggregate_shell, dtype=bool),
                    audit,
                )
            )
    base_support["shell_candidates"] = shell_audits

    outline_role: _SpatialOutlineRole | None = None
    outline_status = "unavailable"
    outline_reason = "canonical_shell_unavailable"
    shell_mask = empty
    support_color = ""
    if len(shell_candidates) == 1:
        outline_role, _, selected_shell, _ = shell_candidates[0]
        shell_mask = np.asarray(selected_shell, dtype=bool)
        support_color = outline_role.shell_color
        outline_status = "supported"
        outline_reason = "canonical_unique_concentric_shell"
    elif len(shell_candidates) > 1:
        outline_status = "ambiguous"
        outline_reason = "canonical_competing_concentric_shells"
    elif any(
        str(item.get("reason") or "")
        == "canonical_native_antialias_shell_rejected"
        for item in shell_audits
    ):
        outline_reason = "canonical_native_antialias_shell_rejected"
    elif shell_audits:
        outline_reason = "canonical_no_concentric_shell"

    effect_mask = (
        np.zeros(mask.shape, dtype=bool)
        if outline_status == "ambiguous"
        else mask & ~core_mask & ~shell_mask
    )
    return _CanonicalSpatialRoles(
        core_role_status="supported",
        core_role_reason="canonical_medial_paint_core_supported",
        core_resolution="canonical_medial_paint_core",
        core_color=core_mode.color,
        support_color=support_color,
        core_mask=_readonly_array(core_mask, dtype=bool),
        shell_mask=_readonly_array(shell_mask, dtype=bool),
        effect_mask=_readonly_array(effect_mask, dtype=bool),
        outline_role=outline_role,
        outline_role_status=outline_status,
        outline_role_reason=outline_reason,
        medial_threshold_px=medial_threshold,
        paint_modes=modes,
        role_support=MappingProxyType(base_support),
    )


def build_authorized_style_spatial_fact_set(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    view: AuthorizedSourceStyleView | None = None,
) -> AuthorizedStyleSpatialFactSet:
    """Build the single runtime fact set consumed by all source-style axes."""

    source = np.ascontiguousarray(source_crop, dtype=np.uint8)
    mask = np.ascontiguousarray(mask_crop, dtype=bool)
    external_ring, ring_facts = _external_source_surface_ring(mask)
    cohorts = _spatial_paint_cohorts(source, mask)
    roles = _resolve_canonical_spatial_roles(source, mask, cohorts)
    core = np.ascontiguousarray(roles.core_mask, dtype=bool)
    concentric_shell = np.ascontiguousarray(roles.shell_mask, dtype=bool)
    displaced_effect = np.ascontiguousarray(roles.effect_mask, dtype=bool)

    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    core_supported = (
        roles.core_role_status == "supported"
        and int(np.count_nonzero(core)) >= _CANONICAL_ROLE_MIN_CLUSTER_PIXELS
    )
    core_luma = float(np.median(luma[core])) if core_supported else 127.0
    # Detector matte polarity belongs only to a resolved canonical core.
    # External pixels never vote on direct paint, scale, or weight axes.
    fill_polarity = (
        ("dark" if core_luma < 128.0 else "light")
        if core_supported
        else ""
    )
    primary_matte = (
        255 if fill_polarity == "dark"
        else 0 if fill_polarity == "light"
        else 127
    )
    primary = np.full_like(source, primary_matte, dtype=np.uint8)
    if core_supported:
        primary[core] = source[core]
    neutral = np.full_like(source, 127, dtype=np.uint8)
    if core_supported:
        neutral[core] = source[core]
    detector_sha256 = _array_sha256(primary) if core_supported else ""
    identity_payload = {
        "authorized_source_style_view_version": AUTHORIZED_SOURCE_STYLE_VIEW_VERSION,
        "page_id": view.page_id if view is not None else "",
        "view_id": view.view_id if view is not None else "",
        "bundle_id": view.bundle_id if view is not None else "",
        "parent_id": view.parent_id if view is not None else "",
        "root_id": view.root_id if view is not None else "",
        "content_bbox": list(view.content_bbox) if view is not None else [],
        "analysis_bbox": list(view.analysis_bbox) if view is not None else [],
        "cleanup_mask_ids": (
            list(view.cleanup_mask_ids) if view is not None else []
        ),
        "owned_component_ids": (
            list(view.owned_component_ids) if view is not None else []
        ),
        "crop_shape": [int(source.shape[0]), int(source.shape[1])],
        "authorized_mask_sha256": _array_sha256(mask),
        "authorized_pixel_sha256": _array_sha256(source[mask]),
        "external_surface_ring_version": EXTERNAL_SOURCE_SURFACE_RING_VERSION,
        "external_surface_ring_inner_radius_px": float(
            ring_facts.get("inner_radius_px") or 0.0
        ),
        "external_surface_ring_outer_radius_px": float(
            ring_facts.get("outer_radius_px") or 0.0
        ),
        "external_surface_ring_pixel_count": int(
            ring_facts.get("pixel_count") or 0
        ),
        "external_surface_ring_mask_sha256": _array_sha256(external_ring),
        "external_surface_ring_pixel_sha256": _array_sha256(
            source[external_ring]
        ),
        "detector_input_sha256": detector_sha256,
    }
    immutable_identity = MappingProxyType(
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in identity_payload.items()
        }
    )
    return AuthorizedStyleSpatialFactSet(
        contract_version=AUTHORIZED_STYLE_SPATIAL_FACT_SET_VERSION,
        fact_set_id=_perceptual_fact_set_id(identity_payload),
        source_identity=immutable_identity,
        core_resolution=roles.core_resolution,
        core_role_status=roles.core_role_status,
        core_role_reason=roles.core_role_reason,
        core_role_support=MappingProxyType(roles.audit_dict()),
        core_color=roles.core_color,
        support_color=roles.support_color,
        fill_polarity=fill_polarity,
        paint_cohorts=cohorts,
        outline_role=roles.outline_role,
        outline_role_status=roles.outline_role_status,
        outline_role_reason=roles.outline_role_reason,
        source_rgb=_readonly_array(source, dtype=np.uint8),
        authorized_mask=_readonly_array(mask, dtype=bool),
        character_core_mask=_readonly_array(core, dtype=bool),
        concentric_shell_mask=_readonly_array(concentric_shell, dtype=bool),
        displaced_effect_mask=_readonly_array(displaced_effect, dtype=bool),
        external_surface_ring_mask=_readonly_array(external_ring, dtype=bool),
        detector_primary_rgb=_readonly_array(primary, dtype=np.uint8),
        detector_neutral_rgb=_readonly_array(neutral, dtype=np.uint8),
    )


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


def _reference_character_component_size(
    mask_binary: np.ndarray,
) -> tuple[float, int, float]:
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
    mad = float(
        np.median(np.abs(np.asarray(sizes, dtype=np.float32) - median))
    )
    return float(np.percentile(sizes, 70)), len(sizes), mad


def _reference_source_cell_size_from_geometry(
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
        return component_size, max(
            0.58, min(0.86, 0.62 + component_count * 0.015)
        )
    if span_median > 0:
        return span_median, 0.72 if len(clean_spans) > 1 else 0.64
    if component_size > 0:
        return component_size, max(
            0.55, min(0.82, 0.58 + component_count * 0.012)
        )
    return 0.0, 0.0


def _reference_stable_upper_cell_cohort(
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


def _reference_fill_component_cell_sizes(fill: np.ndarray) -> list[float]:
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


def _reference_qualify_source_cell_measurement(
    fill: np.ndarray,
    *,
    axis: int,
    spans: Sequence[float],
    legacy_size: float,
    legacy_confidence: float,
    fill_component_sizes: Sequence[float],
) -> tuple[float, float, str, dict[str, Any]]:
    binary = np.asarray(fill, dtype=bool)
    legacy_size = max(0.0, float(legacy_size))
    legacy_confidence = max(0.0, float(legacy_confidence))
    raw_candidate, raw_count, raw_relative_mad = (
        _reference_stable_upper_cell_cohort(spans, minimum_count=1)
    )
    fill_candidate, fill_count, fill_relative_mad = (
        _reference_stable_upper_cell_cohort(
            fill_component_sizes, minimum_count=3
        )
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
    density_minimum_occupancy = 0
    if parent_sized_island:
        density_minimum_occupancy = max(
            2, int(round(orthogonal_extent * 0.10))
        )
        density_spans = _projection_spans_at_min_occupancy(
            binary,
            axis=axis,
            minimum_occupancy=density_minimum_occupancy,
        )
        density_candidate, density_count, density_relative_mad = (
            _reference_stable_upper_cell_cohort(
                density_spans, minimum_count=3
            )
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
        "density_minimum_occupancy": density_minimum_occupancy,
        "density_spans": [round(float(value), 6) for value in density_spans],
        "density_candidate": round(density_candidate, 6),
        "density_candidate_count": int(density_count),
        "density_relative_mad": round(density_relative_mad, 8),
    }
    if parent_sized_island:
        if density_candidate > 0.0 and raw_max >= density_candidate * 2.2:
            confidence = max(
                0.72, min(0.90, 0.86 - density_relative_mad * 0.40)
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
        return repaired, confidence, "supported_fill_projection_override", audit
    raw_matches = bool(
        raw_candidate > 0.0
        and 0.55 <= legacy_size / raw_candidate <= 1.65
    )
    fill_matches = bool(
        fill_candidate > 0.0
        and 0.60 <= legacy_size / fill_candidate <= 1.60
    )
    if legacy_size > 0.0 and legacy_confidence > 0.0 and (
        raw_matches or fill_matches
    ):
        return (
            legacy_size,
            legacy_confidence,
            "supported_independent_corroboration",
            audit,
        )
    if fill_projection_agree:
        inferred = float(np.median([raw_candidate, fill_candidate]))
        return inferred, 0.72, "supported_fill_projection_inference", audit
    return 0.0, 0.0, "unavailable_unqualified_geometry", audit


def _qualify_punctuation_heavy_body_tier(
    fill: np.ndarray,
    *,
    axis: int,
    reference_size: float,
    reference_confidence: float,
    reference_support: str,
    reference_audit: Mapping[str, Any],
) -> tuple[float, float, str, dict[str, Any]]:
    """Recover a corroborated glyph-body tier from punctuation-heavy ink.

    Compact punctuation remains part of the authorized glyph mask and the
    footprint, but it cannot set em scale when a separate body component tier
    and projection agree.  An otherwise unavailable measurement is recovered
    only for a parent-sized island; sparse unavailable axes remain abstentions.
    """

    binary = np.asarray(fill, dtype=bool)
    component_facts = _fill_component_facts(binary)
    axis_key = "width_px" if axis == 0 else "height_px"
    punctuation_count = sum(
        1
        for fact in component_facts
        if bool(fact.get("punctuation_fragment"))
    )
    body_values = [
        float(fact.get(axis_key) or 0.0)
        for fact in component_facts
        if not bool(fact.get("punctuation_fragment"))
        and float(fact.get(axis_key) or 0.0) >= 3.0
    ]
    body_candidate, body_tier_count, body_relative_mad = (
        _stable_numeric_tier(body_values, minimum_count=1)
    )
    projection_spans = _projection_spans(binary, axis=axis)
    axis_extent = int(binary.shape[1] if axis == 0 else binary.shape[0])
    orthogonal_extent = int(
        binary.shape[0] if axis == 0 else binary.shape[1]
    )
    coordinates = np.where(binary)
    orthogonal_coordinates = (
        coordinates[0] if axis == 0 else coordinates[1]
    )
    filled_orthogonal_extent = (
        int(np.ptp(orthogonal_coordinates)) + 1
        if orthogonal_coordinates.size
        else 0
    )
    raw_max = max((float(value) for value in projection_spans), default=0.0)
    parent_sized_island = bool(
        axis_extent > 0
        and orthogonal_extent > 0
        and raw_max >= axis_extent * 0.78
        and filled_orthogonal_extent >= orthogonal_extent * 0.78
    )
    density_minimum_occupancy = (
        max(2, int(round(orthogonal_extent * 0.10)))
        if parent_sized_island
        else 0
    )
    density_spans = (
        _projection_spans_at_min_occupancy(
            binary,
            axis=axis,
            minimum_occupancy=density_minimum_occupancy,
        )
        if density_minimum_occupancy > 0
        else []
    )
    projection_matches = sorted(
        float(value)
        for value in [*projection_spans, *density_spans]
        if body_candidate > 0.0
        and float(value) >= 3.0
        and 0.70 <= float(value) / body_candidate <= 1.35
    )
    reference_size = max(0.0, float(reference_size))
    reference_confidence = max(0.0, float(reference_confidence))
    punctuation_dominated = bool(
        reference_size > 0.0
        and body_candidate > 0.0
        and reference_size < body_candidate * 0.65
    )
    recoverable_unavailable_island = bool(
        reference_size <= 0.0 and parent_sized_island
    )
    recovery_supported = bool(
        punctuation_count >= 3
        and body_candidate > 0.0
        and projection_matches
        and (punctuation_dominated or recoverable_unavailable_island)
    )
    audit = {
        **dict(reference_audit),
        "selected_body_tier_px": round(body_candidate, 6),
        "selected_body_tier_count": int(body_tier_count),
        "selected_body_tier_relative_mad": round(body_relative_mad, 8),
        "body_component_count": len(body_values),
        "punctuation_component_count": int(punctuation_count),
        "body_projection_matches_px": [
            round(value, 6) for value in projection_matches
        ],
        "body_tier_axis_extent": axis_extent,
        "body_tier_orthogonal_extent": orthogonal_extent,
        "body_tier_filled_orthogonal_extent": filled_orthogonal_extent,
        "body_tier_parent_sized_island": parent_sized_island,
        "density_minimum_occupancy": density_minimum_occupancy,
        "density_spans": [round(float(value), 6) for value in density_spans],
        "body_tier_reference_size_px": round(reference_size, 6),
        "body_tier_punctuation_dominated": punctuation_dominated,
        "body_tier_recoverable_unavailable_island": (
            recoverable_unavailable_island
        ),
        "body_tier_recovery_applied": recovery_supported,
    }
    if punctuation_count > 0 and not body_values:
        audit["body_tier_scale_abstention_reason"] = (
            "punctuation_only_geometry_cannot_define_source_em_scale"
        )
        return (
            0.0,
            0.0,
            "unavailable_punctuation_only_geometry",
            audit,
        )
    if not recovery_supported:
        return (
            reference_size,
            reference_confidence,
            reference_support,
            audit,
        )

    if len(body_values) >= 3:
        confidence = max(
            0.76,
            min(
                0.90,
                0.86
                - body_relative_mad * 0.40
                + min(0.04, (len(body_values) - 3) * 0.01),
            ),
        )
        audit["body_tier_recovery_reason"] = (
            "repeated_body_components_with_projection_corroboration"
        )
        return (
            body_candidate,
            confidence,
            "supported_repeated_body_component_tier",
            audit,
        )

    projection_candidate = min(
        projection_matches,
        key=lambda value: abs(value - body_candidate),
    )
    relative_delta = abs(projection_candidate - body_candidate) / max(
        1.0, body_candidate
    )
    confidence = max(
        0.70,
        min(
            0.84,
            0.80
            - relative_delta * 0.24
            + max(0, len(body_values) - 1) * 0.02,
        ),
    )
    audit["body_tier_selected_projection_px"] = round(
        projection_candidate, 6
    )
    audit["body_tier_recovery_reason"] = (
        "sparse_body_component_with_projection_corroboration"
    )
    return (
        projection_candidate,
        confidence,
        "supported_body_projection_component_tier",
        audit,
    )


def _measure_independent_source_scale(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> _IndependentScaleMeasurement:
    geometry = _independent_glyph_geometry(source_crop, mask_crop)
    glyph = np.asarray(geometry.glyph_mask, dtype=bool)
    x_spans = _projection_spans(glyph, axis=0)
    y_spans = _projection_spans(glyph, axis=1)
    component_size, component_count, component_mad = (
        _reference_character_component_size(
            np.asarray(geometry.authorized_mask, dtype=np.uint8)
        )
    )
    legacy_vertical_size, legacy_vertical_confidence = (
        _reference_source_cell_size_from_geometry(
            x_spans,
            component_size=component_size,
            component_count=component_count,
            component_mad=component_mad,
        )
    )
    legacy_horizontal_size, legacy_horizontal_confidence = (
        _reference_source_cell_size_from_geometry(
            y_spans,
            component_size=component_size,
            component_count=component_count,
            component_mad=component_mad,
        )
    )
    component_sizes = _reference_fill_component_cell_sizes(glyph)
    vertical_size, vertical_confidence, vertical_support, vertical_audit = (
        _reference_qualify_source_cell_measurement(
            glyph,
            axis=0,
            spans=x_spans,
            legacy_size=legacy_vertical_size,
            legacy_confidence=legacy_vertical_confidence,
            fill_component_sizes=component_sizes,
        )
    )
    horizontal_size, horizontal_confidence, horizontal_support, horizontal_audit = (
        _reference_qualify_source_cell_measurement(
            glyph,
            axis=1,
            spans=y_spans,
            legacy_size=legacy_horizontal_size,
            legacy_confidence=legacy_horizontal_confidence,
            fill_component_sizes=component_sizes,
        )
    )
    (
        vertical_size,
        vertical_confidence,
        vertical_support,
        vertical_audit,
    ) = _qualify_punctuation_heavy_body_tier(
        glyph,
        axis=0,
        reference_size=vertical_size,
        reference_confidence=vertical_confidence,
        reference_support=vertical_support,
        reference_audit=vertical_audit,
    )
    (
        horizontal_size,
        horizontal_confidence,
        horizontal_support,
        horizontal_audit,
    ) = _qualify_punctuation_heavy_body_tier(
        glyph,
        axis=1,
        reference_size=horizontal_size,
        reference_confidence=horizontal_confidence,
        reference_support=horizontal_support,
        reference_audit=horizontal_audit,
    )
    confidence = max(vertical_confidence, horizontal_confidence)
    value = {
        "vertical_px": round(vertical_size, 6),
        "horizontal_px": round(horizontal_size, 6),
        "vertical_confidence": round(vertical_confidence, 8),
        "horizontal_confidence": round(horizontal_confidence, 8),
        "vertical_support": vertical_support,
        "horizontal_support": horizontal_support,
    }
    support = {
        "glyph_geometry_mask_sha256": _array_sha256(
            np.ascontiguousarray(glyph, dtype=np.uint8)
        ),
        "glyph_pixel_count": int(np.count_nonzero(glyph)),
        "x_projection_spans_px": [round(float(value), 6) for value in x_spans],
        "y_projection_spans_px": [round(float(value), 6) for value in y_spans],
        "component_size_p70_px": round(component_size, 6),
        "component_count": int(component_count),
        "component_mad_px": round(component_mad, 6),
        "vertical_qualification": vertical_audit,
        "horizontal_qualification": horizontal_audit,
    }
    reasons = [
        *geometry.reason_codes,
        vertical_support,
        horizontal_support,
    ]
    axis_evidence = (
        SourceStyleAxisEvidence(
            axis="scale",
            status="supported",
            value=value,
            confidence=confidence,
            provenance=(
                "authorized_source_style_view:independent_glyph_geometry_scale"
            ),
            support_identity=support_identity,
            reason_codes=tuple(reason for reason in reasons if reason),
            support=support,
        )
        if confidence > 0.0 and (vertical_size > 0.0 or horizontal_size > 0.0)
        else SourceStyleAxisEvidence.unavailable(
            "scale",
            provenance=(
                "authorized_source_style_view:independent_glyph_geometry_scale"
            ),
            support_identity=support_identity,
            reason_codes=tuple(
                reason
                for reason in (
                    *reasons,
                    "source_scale_axis_unavailable",
                )
                if reason
            ),
            support=support,
        )
    )
    return _IndependentScaleMeasurement(
        axis_evidence=axis_evidence,
        glyph_mask=_readonly_array(glyph, dtype=bool),
        vertical_size_px=vertical_size,
        horizontal_size_px=horizontal_size,
        vertical_confidence=vertical_confidence,
        horizontal_confidence=horizontal_confidence,
        vertical_support=vertical_support,
        horizontal_support=horizontal_support,
        vertical_qualification=MappingProxyType(vertical_audit),
        horizontal_qualification=MappingProxyType(horizontal_audit),
    )


def _observe_source_scale_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    return _measure_independent_source_scale(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    ).axis_evidence


def _observe_fill_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    geometry = _independent_glyph_geometry(source_crop, mask_crop)
    confidence = (
        min(
            0.98,
            max(0.0, geometry.contrast / 128.0)
            * min(1.0, geometry.fill_count / 64.0),
        )
        if geometry.fill_cluster_resolved
        else 0.0
    )
    support = {
        "glyph_geometry_mask_sha256": _array_sha256(
            np.ascontiguousarray(geometry.glyph_mask, dtype=np.uint8)
        ),
        "glyph_pixel_count": geometry.fill_count,
        "support_luma_median": round(geometry.support_luma_median, 6),
        "support_luma_iqr": round(geometry.support_luma_iqr, 6),
        "fill_luma_median": round(geometry.fill_luma_median, 6),
        "fill_support_contrast": round(geometry.contrast, 6),
    }
    if not (
        geometry.fill_cluster_resolved
        and confidence > 0.0
        and geometry.fill_color
    ):
        return SourceStyleAxisEvidence.unavailable(
            "fill",
            provenance="authorized_source_style_view:independent_fill_axis",
            support_identity=support_identity,
            reason_codes=(
                *geometry.reason_codes,
                "source_fill_axis_unavailable",
            ),
            support=support,
        )
    return SourceStyleAxisEvidence(
        axis="fill",
        status="supported",
        value={
            "color": geometry.fill_color,
            "support_color": geometry.support_color,
            "polarity": geometry.fill_polarity,
        },
        confidence=confidence,
        provenance="authorized_source_style_view:independent_fill_axis",
        support_identity=support_identity,
        reason_codes=(
            *geometry.reason_codes,
            "source_fill_measured_from_independent_authorized_contrast",
        ),
        support=support,
    )


def _outline_support_facts(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> dict[str, Any]:
    geometry = _independent_glyph_geometry(source_crop, mask_crop)
    source = np.asarray(geometry.source, dtype=np.uint8)
    support_mask = np.asarray(geometry.support_mask, dtype=bool)
    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    support_luma = luma[support_mask]
    if geometry.fill_polarity == "dark":
        opposite_fraction = (
            float(np.mean(support_luma >= 192.0)) if support_luma.size else 0.0
        )
    else:
        opposite_fraction = (
            float(np.mean(support_luma <= 63.0)) if support_luma.size else 0.0
        )
    visible_transition = bool(
        geometry.contrast >= 64.0
        and geometry.support_luma_iqr >= 48.0
        and 0.08 <= opposite_fraction <= 0.90
    )
    uniform_backing = bool(
        geometry.contrast >= 48.0
        and geometry.support_luma_iqr <= 44.0
        and opposite_fraction >= 0.88
    )
    external_ring, external_ring_facts = _external_source_surface_ring(
        mask_crop
    )
    internal_support_pixels = source[support_mask]
    external_surface_pixels = source[external_ring]
    external_surface_available = bool(
        internal_support_pixels.size > 0
        and external_surface_pixels.size > 0
        and int(external_ring_facts.get("pixel_count") or 0) >= 24
    )
    internal_rgb_median: list[float] = []
    external_rgb_median: list[float] = []
    internal_luma_quantiles: list[float] = []
    external_luma_quantiles: list[float] = []
    external_rgb_distance = 0.0
    external_luma_quantile_delta = 0.0
    if external_surface_available:
        internal_rgb_median = [
            round(float(value), 6)
            for value in np.median(
                internal_support_pixels.reshape(-1, 3), axis=0
            )
        ]
        external_rgb_median = [
            round(float(value), 6)
            for value in np.median(
                external_surface_pixels.reshape(-1, 3), axis=0
            )
        ]
        external_rgb_distance = float(
            np.linalg.norm(
                np.asarray(internal_rgb_median, dtype=np.float32)
                - np.asarray(external_rgb_median, dtype=np.float32)
            )
        )
        internal_luma_quantiles = _luma_quantiles(
            luma[support_mask]
        )
        external_luma_quantiles = _luma_quantiles(
            luma[external_ring]
        )
        if internal_luma_quantiles and external_luma_quantiles:
            external_luma_quantile_delta = float(
                np.median(
                    np.abs(
                        np.asarray(
                            internal_luma_quantiles, dtype=np.float32
                        )
                        - np.asarray(
                            external_luma_quantiles, dtype=np.float32
                        )
                    )
                )
            )
    external_surface_continuity = bool(
        external_surface_available
        and external_rgb_distance
        <= _OUTLINE_SURFACE_CONTINUITY_MAX_RGB_DISTANCE
        and external_luma_quantile_delta
        <= _OUTLINE_SURFACE_CONTINUITY_MAX_LUMA_QUANTILE_DELTA
    )
    external_surface_discontinuity = bool(
        external_surface_available
        and external_rgb_distance >= _OUTLINE_BACKING_MIN_RGB_DISTANCE
        and external_luma_quantile_delta
        >= _OUTLINE_BACKING_MIN_LUMA_QUANTILE_DELTA
    )
    return {
        "geometry": geometry,
        "support_opposite_fraction": opposite_fraction,
        "visible_support_transition": visible_transition,
        "uniform_support_backing": uniform_backing,
        "external_surface_ring": external_ring,
        "external_surface_ring_facts": external_ring_facts,
        "external_surface_available": external_surface_available,
        "external_surface_continuity": external_surface_continuity,
        "external_surface_discontinuity": external_surface_discontinuity,
        "external_surface_rgb_distance": external_rgb_distance,
        "external_surface_luma_quantile_delta": (
            external_luma_quantile_delta
        ),
        "internal_support_rgb_median": internal_rgb_median,
        "external_surface_rgb_median": external_rgb_median,
        "internal_support_luma_quantiles": internal_luma_quantiles,
        "external_surface_luma_quantiles": external_luma_quantiles,
    }


def _observe_outline_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    facts = _outline_support_facts(source_crop, mask_crop)
    geometry: _IndependentGlyphGeometry = facts["geometry"]
    external_ring = np.asarray(
        facts["external_surface_ring"], dtype=bool
    )
    external_ring_facts = dict(facts["external_surface_ring_facts"])
    outline_support_identity = dict(support_identity)
    outline_support_identity.update(
        {
            "external_surface_ring_version": str(
                external_ring_facts.get("version")
                or EXTERNAL_SOURCE_SURFACE_RING_VERSION
            ),
            "external_surface_ring_inner_radius_px": float(
                external_ring_facts.get("inner_radius_px") or 0.0
            ),
            "external_surface_ring_outer_radius_px": float(
                external_ring_facts.get("outer_radius_px") or 0.0
            ),
            "external_surface_ring_pixel_count": int(
                external_ring_facts.get("pixel_count") or 0
            ),
            "external_surface_ring_fallback_used": bool(
                external_ring_facts.get("fallback_used")
            ),
            "external_surface_ring_mask_sha256": _array_sha256(
                np.ascontiguousarray(external_ring, dtype=np.uint8)
            ),
            "external_surface_ring_pixel_sha256": _array_sha256(
                np.ascontiguousarray(
                    np.asarray(source_crop, dtype=np.uint8)[external_ring],
                    dtype=np.uint8,
                )
            ),
        }
    )
    scale = _measure_independent_source_scale(
        source_crop, mask_crop, support_identity=support_identity
    )
    scale_for_stroke = max(
        scale.vertical_size_px,
        scale.horizontal_size_px,
        1.0,
    )
    support = {
        "support_luma_median": round(geometry.support_luma_median, 6),
        "support_luma_iqr": round(geometry.support_luma_iqr, 6),
        "fill_support_contrast": round(geometry.contrast, 6),
        "support_opposite_fraction": round(
            float(facts["support_opposite_fraction"]), 8
        ),
        "visible_support_transition": bool(
            facts["visible_support_transition"]
        ),
        "uniform_support_backing": bool(facts["uniform_support_backing"]),
        "external_surface_available": bool(
            facts["external_surface_available"]
        ),
        "external_surface_continuity": bool(
            facts["external_surface_continuity"]
        ),
        "external_surface_discontinuity": bool(
            facts["external_surface_discontinuity"]
        ),
        "external_surface_rgb_distance": round(
            float(facts["external_surface_rgb_distance"]), 6
        ),
        "external_surface_luma_quantile_delta": round(
            float(facts["external_surface_luma_quantile_delta"]), 6
        ),
        "internal_support_rgb_median": list(
            facts["internal_support_rgb_median"]
        ),
        "external_surface_rgb_median": list(
            facts["external_surface_rgb_median"]
        ),
        "internal_support_luma_quantiles": list(
            facts["internal_support_luma_quantiles"]
        ),
        "external_surface_luma_quantiles": list(
            facts["external_surface_luma_quantiles"]
        ),
        "surface_continuity_max_rgb_distance": (
            _OUTLINE_SURFACE_CONTINUITY_MAX_RGB_DISTANCE
        ),
        "surface_continuity_max_luma_quantile_delta": (
            _OUTLINE_SURFACE_CONTINUITY_MAX_LUMA_QUANTILE_DELTA
        ),
        "backing_min_rgb_distance": _OUTLINE_BACKING_MIN_RGB_DISTANCE,
        "backing_min_luma_quantile_delta": (
            _OUTLINE_BACKING_MIN_LUMA_QUANTILE_DELTA
        ),
    }
    if facts["visible_support_transition"]:
        width = min(
            max(1.0, float(round(scale_for_stroke * 0.055))),
            max(1.0, scale_for_stroke * 0.15),
        )
        confidence = min(
            0.95,
            0.52
            + min(0.25, geometry.contrast / 512.0)
            + min(0.18, geometry.support_luma_iqr / 255.0),
        )
        return SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "present": True,
                "kind": "outline",
                "color": geometry.support_color,
                "width_px": round(width, 6),
            },
            confidence=confidence,
            provenance="authorized_source_style_view:independent_outline_axis",
            support_identity=outline_support_identity,
            reason_codes=("source_visible_support_stroke_measured",),
            support=support,
        )
    if (
        facts["uniform_support_backing"]
        and facts["external_surface_discontinuity"]
    ):
        width = min(
            max(1.0, float(round(scale_for_stroke * 0.055))),
            max(1.0, scale_for_stroke * 0.15),
        )
        confidence = min(
            0.94,
            0.68
            + min(
                0.14,
                float(facts["external_surface_rgb_distance"]) / 768.0,
            )
            + min(
                0.12,
                float(facts["external_surface_luma_quantile_delta"])
                / 512.0,
            ),
        )
        return SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "present": True,
                "kind": "backing",
                "color": geometry.support_color,
                "width_px": round(width, 6),
            },
            confidence=confidence,
            provenance="authorized_source_style_view:independent_outline_axis",
            support_identity=outline_support_identity,
            reason_codes=(
                "source_uniform_support_backing_measured_against_surface",
            ),
            support=support,
        )
    if (
        facts["uniform_support_backing"]
        and facts["external_surface_continuity"]
    ):
        confidence = min(
            0.92,
            0.72
            + min(0.15, geometry.contrast / 1024.0)
            + min(
                0.05,
                float(facts["support_opposite_fraction"]) / 20.0,
            ),
        )
        return SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "present": False,
                "kind": "surface",
                "color": geometry.support_color,
                "width_px": 0.0,
            },
            confidence=confidence,
            provenance="authorized_source_style_view:independent_outline_axis",
            support_identity=outline_support_identity,
            reason_codes=(
                "source_visible_stroke_absent_continuous_surface",
            ),
            support=support,
        )
    return SourceStyleAxisEvidence.unavailable(
        "outline",
        provenance="authorized_source_style_view:independent_outline_axis",
        support_identity=outline_support_identity,
        reason_codes=(
            "source_outline_axis_not_independently_supported",
            *(
                ("source_uniform_support_surface_context_ambiguous",)
                if facts["uniform_support_backing"]
                else ()
            ),
        ),
        support=support,
    )


def _observe_weight_axis(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    support_identity: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    facts = _outline_support_facts(source_crop, mask_crop)
    geometry: _IndependentGlyphGeometry = facts["geometry"]
    glyph = np.asarray(geometry.glyph_mask, dtype=bool)
    try:
        import cv2

        distance = cv2.distanceTransform(
            np.asarray(glyph, dtype=np.uint8), cv2.DIST_L2, 5
        )
    except Exception:
        distance = np.zeros_like(glyph, dtype=np.float32)
    distance_values = distance[glyph]
    distance_p75 = (
        float(np.percentile(distance_values, 75))
        if distance_values.size
        else 0.0
    )
    ink_width = max(0.0, distance_p75 * 2.0)
    generic_class = ""
    generic_confidence = 0.0
    if (
        geometry.fill_cluster_resolved
        and geometry.fill_count >= 24
        and distance_p75 > 0.0
    ):
        if distance_p75 <= 1.5:
            generic_class = "regular"
            generic_confidence = min(
                0.92, 0.72 + geometry.fill_count / 4096.0
            )
        elif distance_p75 >= 1.9:
            generic_class = "bold"
            generic_confidence = min(
                0.94,
                0.72
                + min(0.16, (distance_p75 - 1.9) * 0.12)
                + geometry.fill_count / 4096.0,
            )
    scale = _measure_independent_source_scale(
        source_crop, mask_crop, support_identity=support_identity
    )
    directional = {
        "status": "not_applicable_no_visible_support_transition",
        "directions": {},
    }
    vertical_class = ""
    vertical_confidence = 0.0
    vertical_support = ""
    horizontal_class = ""
    horizontal_confidence = 0.0
    horizontal_support = ""
    if facts["visible_support_transition"]:
        _, _, directional = _qualify_outlined_ink_weight(
            glyph,
            source_cell_size_vertical_px=scale.vertical_size_px,
            source_cell_size_horizontal_px=scale.horizontal_size_px,
        )
        directions = dict(directional.get("directions") or {})
        vertical = dict(directions.get("vertical") or {})
        horizontal = dict(directions.get("horizontal") or {})
        vertical_support = str(vertical.get("status") or "")
        horizontal_support = str(horizontal.get("status") or "")
        if vertical_support.startswith("supported_"):
            vertical_class = str(vertical.get("weight_class") or "")
            vertical_confidence = float(vertical.get("confidence") or 0.0)
        if horizontal_support.startswith("supported_"):
            horizontal_class = str(horizontal.get("weight_class") or "")
            horizontal_confidence = float(horizontal.get("confidence") or 0.0)
        if vertical_class or horizontal_class:
            generic_class = ""
            generic_confidence = 0.0
        elif generic_class:
            generic_class = ""
            generic_confidence = 0.0
    confidence = max(
        generic_confidence,
        vertical_confidence,
        horizontal_confidence,
    )
    support = {
        "glyph_geometry_mask_sha256": _array_sha256(
            np.ascontiguousarray(glyph, dtype=np.uint8)
        ),
        "glyph_pixel_count": geometry.fill_count,
        "ink_distance_p75": round(distance_p75, 6),
        "source_ink_stroke_width_px": round(ink_width, 6),
        "outlined_weight_qualification": directional,
    }
    value = {
        "class": generic_class,
        "confidence": round(generic_confidence, 8),
        "vertical_class": vertical_class,
        "vertical_confidence": round(vertical_confidence, 8),
        "vertical_support": vertical_support,
        "horizontal_class": horizontal_class,
        "horizontal_confidence": round(horizontal_confidence, 8),
        "horizontal_support": horizontal_support,
        "source_ink_stroke_width_px": round(ink_width, 6),
    }
    if confidence <= 0.0:
        return SourceStyleAxisEvidence.unavailable(
            "weight",
            provenance="authorized_source_style_view:independent_weight_axis",
            support_identity=support_identity,
            reason_codes=("source_ink_weight_axis_unavailable",),
            support=support,
        )
    return SourceStyleAxisEvidence(
        axis="weight",
        status="supported",
        value=value,
        confidence=confidence,
        provenance="authorized_source_style_view:independent_weight_axis",
        support_identity=support_identity,
        reason_codes=(
            "source_ink_weight_directional_geometry_measured"
            if vertical_class or horizontal_class
            else "source_ink_weight_direct_geometry_measured",
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
            source_crop,
            mask_crop,
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
            source_crop,
            mask_crop,
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

    Detector inputs and ink-derived axes use accepted foreground pixels. Only
    the outline axis may compare its internal support with a bounded exterior
    annulus, whose identity is attached to that axis alone. Raw exterior
    pixels are never published or reused by scale, fill, weight, orientation,
    rotation, shadow, or detector presentation.
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

    scale_measurement = _measure_independent_source_scale(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    )
    fill_axis = _observe_fill_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    )
    outline_axis = _observe_outline_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
    )
    weight_axis = _observe_weight_axis(
        source_crop,
        mask_crop,
        support_identity=support_identity,
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
    source_identity: Mapping[str, Any],
    observed_by_axis: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize axis observations already derived from one spatial fact set."""
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
    *,
    spatial_facts: AuthorizedStyleSpatialFactSet | None = None,
) -> dict[str, Any]:
    """Observe chromatic fill only from the canonical character core."""

    facts = spatial_facts or build_authorized_style_spatial_fact_set(
        source_crop, mask_crop
    )
    source = np.asarray(facts.source_rgb, dtype=np.uint8)
    mask = np.asarray(facts.authorized_mask, dtype=bool)
    core = np.asarray(facts.character_core_mask, dtype=bool)
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {
            "authorized_pixel_count": int(np.count_nonzero(mask)),
            "canonical_core_pixel_count": int(np.count_nonzero(core)),
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
        unavailable["reason_codes"].append("perceptual_fill_input_invalid")
        return unavailable
    core_count = int(np.count_nonzero(core))
    if (
        facts.core_role_status != "supported"
        or core_count < _ADDITIVE_FILL_MIN_CLUSTER_PIXELS
    ):
        unavailable["reason_codes"].append(
            "perceptual_fill_canonical_core_unavailable"
        )
        return unavailable

    pixels = source[core].astype(np.float32)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    chromatic_count = int(
        np.count_nonzero(chroma >= _ADDITIVE_FILL_MIN_CHROMA)
    )
    chromatic_fraction = chromatic_count / max(1, core_count)
    median_rgb = np.median(pixels, axis=0)
    dispersion = float(
        np.median(
            np.linalg.norm(pixels - median_rgb[None, :], axis=1)
        )
    )
    unavailable["support"].update(
        {
            "canonical_core_color": facts.core_color,
            "chromatic_pixel_count": chromatic_count,
            "chromatic_fraction": round(chromatic_fraction, 8),
            "color_dispersion_rgb": round(dispersion, 8),
        }
    )
    if (
        chromatic_count < _ADDITIVE_FILL_MIN_CLUSTER_PIXELS
        or chromatic_fraction
        < _ADDITIVE_FILL_MIN_CORE_CHROMATIC_FRACTION
    ):
        unavailable["reason_codes"].append(
            "perceptual_fill_canonical_core_not_chromatic"
        )
        return unavailable
    if dispersion > _ADDITIVE_FILL_MAX_COLOR_DISPERSION:
        unavailable["reason_codes"].append(
            "perceptual_fill_canonical_core_paint_incoherent"
        )
        return unavailable

    confidence = min(
        0.98,
        0.64
        + 0.20 * min(1.0, chromatic_fraction)
        + 0.10 * max(
            0.0,
            1.0
            - dispersion
            / max(_ADDITIVE_FILL_MAX_COLOR_DISPERSION, 1e-6),
        ),
    )
    return {
        "support_status": "supported",
        "value": {"color": facts.core_color or _rgb_hex(median_rgb)},
        "confidence": round(confidence, 8),
        "reason_codes": [
            "perceptual_fill_canonical_chromatic_character_core"
        ],
        "support": dict(unavailable["support"]),
        "uncertainty": {
            "color_dispersion_rgb": round(dispersion, 8)
        },
    }
def _observe_additive_rotation(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    spatial_facts: AuthorizedStyleSpatialFactSet | None = None,
) -> dict[str, Any]:
    """Return one pronounced rotation measured from the canonical core."""

    facts = spatial_facts or build_authorized_style_spatial_fact_set(
        source_crop, mask_crop
    )
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
def _observe_additive_outline(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    canonical_carrier_observation: Mapping[str, Any] | None = None,
    spatial_facts: AuthorizedStyleSpatialFactSet | None = None,
) -> dict[str, Any]:
    """Project the canonical carrier decision onto the additive outline axis."""

    canonical = (
        dict(canonical_carrier_observation)
        if isinstance(canonical_carrier_observation, Mapping)
        else {}
    )
    carrier_kind = str(canonical.get("carrier_kind") or "unavailable")
    support_status = str(canonical.get("support_status") or "unavailable")
    support = (
        dict(canonical.get("support") or {})
        if isinstance(canonical.get("support"), Mapping)
        else {}
    )
    if support_status == "supported" and carrier_kind in {"outline", "backing"}:
        value = (
            dict(canonical.get("value") or {})
            if isinstance(canonical.get("value"), Mapping)
            else {}
        )
        return {
            "support_status": "supported",
            "confidence": float(canonical.get("confidence") or 0.0),
            "reason_codes": [
                f"perceptual_outline_canonical_source_{carrier_kind}"
            ],
            "support": support,
            "uncertainty": {},
            "value": {
                "color": str(value.get("color") or ""),
                "width_px": float(value.get("width_px") or 0.0),
            },
        }
    if canonical:
        return {
            "support_status": "unavailable",
            "confidence": 0.0,
            "reason_codes": [
                "perceptual_outline_external_surface_continuity"
                if support_status == "supported" and carrier_kind == "surface"
                else "perceptual_outline_canonical_source_carrier_absent"
                if support_status == "supported" and carrier_kind == "none"
                else "perceptual_outline_canonical_source_carrier_unavailable"
            ],
            "support": support,
            "uncertainty": {},
        }
    return {
        "support_status": "unavailable",
        "confidence": 0.0,
        "reason_codes": ["perceptual_outline_canonical_carrier_missing"],
        "support": {
            "authorized_pixel_count": int(np.count_nonzero(mask_crop))
        },
        "uncertainty": {},
    }


def _observe_additive_shadow(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    spatial_facts: AuthorizedStyleSpatialFactSet | None = None,
) -> dict[str, Any]:
    """Return one complete displaced glyph-correlated shadow.

    A previously supported chromatic character core supplies only the runtime
    shape used for correlation. One darker authorized effect must be explained
    by one displaced copy of that shape. Blur is then estimated from the
    spatial support extending beyond the displaced core, not from RGB
    dispersion. Concentric, centered, repeated, clipped, or ambiguous support
    remains unavailable.
    """

    facts = spatial_facts or build_authorized_style_spatial_fact_set(
        source_crop, mask_crop
    )
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


def _luma_quantiles(values: np.ndarray) -> list[float]:
    clean = np.asarray(values, dtype=np.float32).reshape(-1)
    if clean.size <= 0:
        return []
    return [
        round(float(np.percentile(clean, percentile)), 6)
        for percentile in (10, 25, 50, 75, 90)
    ]


def _observe_source_carrier(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    spatial_facts: AuthorizedStyleSpatialFactSet,
    shadow_observation: Mapping[str, Any] | None,
    source_cell_size_px: float,
) -> dict[str, Any]:
    """Classify only the canonical shell against the external page surface."""

    _ = (source_crop, mask_crop, shadow_observation, source_cell_size_px)
    source = np.asarray(spatial_facts.source_rgb, dtype=np.uint8)
    mask = np.asarray(spatial_facts.authorized_mask, dtype=bool)
    core = np.asarray(spatial_facts.character_core_mask, dtype=bool)
    shell = np.asarray(spatial_facts.concentric_shell_mask, dtype=bool)
    external_ring = np.asarray(
        spatial_facts.external_surface_ring_mask, dtype=bool
    )
    source_identity = spatial_facts.source_identity
    unavailable: dict[str, Any] = {
        "support_status": "unavailable",
        "carrier_kind": "unavailable",
        "confidence": 0.0,
        "reason_codes": [],
        "support": {
            "authorized_pixel_count": int(np.count_nonzero(mask)),
            "core_pixel_count": int(np.count_nonzero(core)),
            "canonical_shell_pixel_count": int(np.count_nonzero(shell)),
            "canonical_outline_role_status": (
                spatial_facts.outline_role_status
            ),
            "canonical_outline_role_reason": (
                spatial_facts.outline_role_reason
            ),
            "external_surface_ring_version": EXTERNAL_SOURCE_SURFACE_RING_VERSION,
            "external_surface_ring_inner_radius_px": float(
                source_identity.get("external_surface_ring_inner_radius_px")
                or 0.0
            ),
            "external_surface_ring_outer_radius_px": float(
                source_identity.get("external_surface_ring_outer_radius_px")
                or 0.0
            ),
            "external_surface_ring_pixel_count": int(
                source_identity.get("external_surface_ring_pixel_count") or 0
            ),
        },
        "uncertainty": {},
    }
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or mask.shape != source.shape[:2]
        or core.shape != mask.shape
        or shell.shape != mask.shape
    ):
        unavailable["reason_codes"].append("source_carrier_input_invalid")
        return unavailable
    if (
        spatial_facts.core_role_status != "supported"
        or int(np.count_nonzero(core))
        < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS
    ):
        unavailable["reason_codes"].append(
            "source_carrier_canonical_core_unavailable"
        )
        return unavailable
    if _mask_border_margin(mask) < 1:
        unavailable["reason_codes"].append(
            "source_carrier_authorized_support_truncated"
        )
        return unavailable

    role = spatial_facts.outline_role
    if role is None:
        if spatial_facts.outline_role_status == "ambiguous":
            unavailable["reason_codes"].append(
                "source_carrier_canonical_shell_ambiguous"
            )
            return unavailable
        return {
            **unavailable,
            "support_status": "supported",
            "carrier_kind": "none",
            "confidence": 0.84,
            "reason_codes": ["source_carrier_canonical_shell_absent"],
            "value": {"color": "", "width_px": 0.0},
        }

    shell_count = int(np.count_nonzero(shell))
    if shell_count < _CANONICAL_ROLE_MIN_CLUSTER_PIXELS:
        unavailable["reason_codes"].append(
            "source_carrier_canonical_shell_support_unavailable"
        )
        return unavailable
    external_count = int(np.count_nonzero(external_ring))
    if external_count < 24:
        unavailable["reason_codes"].append(
            "source_carrier_external_surface_ring_unavailable"
        )
        return unavailable

    external_rgb = np.median(
        source[external_ring].astype(np.float32), axis=0
    )
    shell_rgb = np.median(
        source[shell].astype(np.float32), axis=0
    )
    color_distance = float(np.linalg.norm(shell_rgb - external_rgb))
    luma = (
        source[:, :, 0].astype(np.float32) * 0.2126
        + source[:, :, 1].astype(np.float32) * 0.7152
        + source[:, :, 2].astype(np.float32) * 0.0722
    )
    shell_luma = luma[shell]
    external_luma = luma[external_ring]
    shell_quantiles = _luma_quantiles(shell_luma)
    external_quantiles = _luma_quantiles(external_luma)
    quantile_delta = float(
        np.median(
            np.abs(
                np.asarray(shell_quantiles, dtype=np.float32)
                - np.asarray(external_quantiles, dtype=np.float32)
            )
        )
    )
    shell_iqr = float(
        np.percentile(shell_luma, 75)
        - np.percentile(shell_luma, 25)
    )
    external_iqr = float(
        np.percentile(external_luma, 75)
        - np.percentile(external_luma, 25)
    )
    continuity_votes = (
        color_distance <= 40.0,
        quantile_delta <= 24.0,
        abs(shell_iqr - external_iqr) <= 24.0,
    )
    surface_continuity = all(continuity_votes)
    support = dict(unavailable["support"])
    support.update(
        {
            "core_color": spatial_facts.core_color,
            "internal_support_color": _rgb_hex(shell_rgb),
            "carrier_shell_pixel_count": shell_count,
            "external_surface_color": _rgb_hex(external_rgb),
            "internal_external_color_distance_rgb": round(
                color_distance, 6
            ),
            "internal_external_luma_quantile_delta": round(
                quantile_delta, 6
            ),
            "internal_luma_iqr": round(shell_iqr, 6),
            "external_luma_iqr": round(external_iqr, 6),
            "radial_distance_p90_px": round(role.width_px, 6),
            "shell_ring_recall": round(role.shell_ring_recall, 8),
            "ring_shell_precision": round(
                role.ring_shell_precision, 8
            ),
            "spatial_outline_role": role.to_audit_dict(),
        }
    )
    if surface_continuity:
        return {
            **unavailable,
            "support_status": "supported",
            "carrier_kind": "surface",
            "confidence": round(role.confidence, 8),
            "reason_codes": [
                "source_carrier_external_surface_continuity"
            ],
            "support": support,
            "value": {"color": _rgb_hex(shell_rgb), "width_px": 0.0},
        }
    if not continuity_votes[0] and not continuity_votes[1]:
        return {
            **unavailable,
            "support_status": "supported",
            "carrier_kind": "outline",
            "confidence": round(role.confidence, 8),
            "reason_codes": [
                "source_outline_supported_canonical_spatial_role"
            ],
            "support": support,
            "value": {
                "color": _rgb_hex(shell_rgb),
                "width_px": round(role.width_px, 6),
            },
        }
    return {
        **unavailable,
        "reason_codes": [
            "source_carrier_external_surface_evidence_conflicting"
        ],
        "support": support,
    }


def _measure_authorized_style_crop(
    source_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    spatial_facts: AuthorizedStyleSpatialFactSet | None = None,
    shadow_observation: Mapping[str, Any] | None = None,
) -> _AuthorizedStyleCropMeasurement:
    facts = spatial_facts or build_authorized_style_spatial_fact_set(
        source_crop, mask_crop
    )
    source_crop = np.asarray(facts.source_rgb, dtype=np.uint8)
    mask_crop = np.asarray(facts.authorized_mask, dtype=bool)
    reasons: list[str] = ["detector_and_ink_axes_authorized_pixels_only"]
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
    fill = np.asarray(facts.character_core_mask, dtype=bool)
    fill_cluster_resolved = facts.core_role_status == "supported"
    support_mask = np.asarray(facts.concentric_shell_mask, dtype=bool)
    if int(np.count_nonzero(support_mask)) < 8:
        # Direct diagnostics may compare the core with itself when no
        # canonical shell exists, but support color remains unavailable.
        support_mask = fill.copy()
        reasons.append("direct_paint_canonical_shell_unavailable")
    edge = support_mask
    edge_luma = luma_image[support_mask]
    support_median = float(np.median(edge_luma)) if edge_luma.size else 127.0
    support_iqr = (
        float(np.percentile(edge_luma, 75) - np.percentile(edge_luma, 25))
        if edge_luma.size
        else 255.0
    )
    fill_polarity = facts.fill_polarity
    fill_count = int(np.count_nonzero(fill))
    if fill_count <= 0:
        fill_cluster_resolved = False
        reasons.append("canonical_character_core_unresolved")

    fill_pixels = source_crop[fill]
    support_pixels = source_crop[edge]
    fill_luma = luma_image[fill]
    fill_median = float(np.median(fill_luma)) if fill_luma.size else support_median
    contrast = abs(support_median - fill_median)
    fill_color = facts.core_color if fill_cluster_resolved else ""
    support_color = facts.support_color

    x_spans = _projection_spans(fill, axis=0)
    y_spans = _projection_spans(fill, axis=1)
    fill_component_facts = _fill_component_facts(fill)
    (
        vertical_size,
        vertical_confidence,
        vertical_support,
        vertical_qualification,
    ) = _qualify_source_cell_measurement(
        fill,
        axis=0,
        spans=x_spans,
        component_facts=fill_component_facts,
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
        component_facts=fill_component_facts,
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
    source_text_footprint = _summarize_source_text_footprint(
        fill,
        component_facts=fill_component_facts,
        vertical_cell_size_px=vertical_size,
        vertical_scale_confidence=vertical_confidence,
        vertical_scale_support=vertical_support,
        vertical_scale_qualification=vertical_qualification,
        horizontal_cell_size_px=horizontal_size,
        horizontal_scale_confidence=horizontal_confidence,
        horizontal_scale_support=horizontal_support,
        horizontal_scale_qualification=horizontal_qualification,
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
    source_carrier_observation = _observe_source_carrier(
        source_crop,
        mask_crop,
        spatial_facts=facts,
        shadow_observation=shadow_observation,
        source_cell_size_px=scale_for_stroke,
    )
    carrier_status = str(
        source_carrier_observation.get("support_status") or "unavailable"
    )
    carrier_kind = str(
        source_carrier_observation.get("carrier_kind") or "unavailable"
    )
    carrier_value = (
        dict(source_carrier_observation.get("value") or {})
        if isinstance(source_carrier_observation.get("value"), Mapping)
        else {}
    )
    outlined_weight_qualification: dict[str, Any] = {
        "status": "not_applicable_source_carrier_unavailable",
        "directions": {},
    }
    ink_weight_class_vertical = ""
    ink_weight_confidence_vertical = 0.0
    ink_weight_support_vertical = ""
    ink_weight_class_horizontal = ""
    ink_weight_confidence_horizontal = 0.0
    ink_weight_support_horizontal = ""
    if carrier_status == "supported" and carrier_kind in {"outline", "backing"}:
        stroke_width = max(0.0, float(carrier_value.get("width_px") or 0.0))
        stroke_confidence = max(
            0.0, float(source_carrier_observation.get("confidence") or 0.0)
        )
        reasons.append(f"source_{carrier_kind}_supported_external_surface_ring")
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
        ink_weight_support_vertical = str(
            vertical_weight.get("status") or ""
        )
        ink_weight_support_horizontal = str(
            horizontal_weight.get("status") or ""
        )
        if str(vertical_weight.get("status") or "").startswith("supported_"):
            ink_weight_class_vertical = str(
                vertical_weight.get("weight_class") or ""
            )
            ink_weight_confidence_vertical = float(
                vertical_weight.get("confidence") or 0.0
            )
        if str(horizontal_weight.get("status") or "").startswith("supported_"):
            ink_weight_class_horizontal = str(
                horizontal_weight.get("weight_class") or ""
            )
            ink_weight_confidence_horizontal = float(
                horizontal_weight.get("confidence") or 0.0
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
            # ``fill`` is the contrast-resolved glyph interior, not the full
            # authorized carrier. A supported external carrier therefore does
            # not invalidate the independently measured core weight when a
            # repeated outlined-cell cohort is unavailable.
            reasons.append(
                "source_ink_weight_retained_contrast_resolved_core"
            )
    elif carrier_status == "supported" and carrier_kind in {"surface", "none"}:
        stroke_confidence = max(
            0.0, float(source_carrier_observation.get("confidence") or 0.0)
        )
        outlined_weight_qualification["status"] = (
            "not_applicable_source_surface_continuity"
            if carrier_kind == "surface"
            else "not_applicable_source_carrier_absent"
        )
        reasons.append(
            "source_carrier_absent_external_surface_continuity"
            if carrier_kind == "surface"
            else "source_carrier_absent_no_internal_support"
        )
    else:
        reasons.append("source_carrier_not_independently_supported")

    fill_rgb = fill_pixels.astype(np.float32)
    fill_rgb_median = (
        np.median(fill_rgb, axis=0)
        if fill_rgb.size
        else np.asarray((127.0, 127.0, 127.0), dtype=np.float32)
    )
    fill_color_distances = (
        np.linalg.norm(fill_rgb - fill_rgb_median[None, :], axis=1)
        if fill_rgb.size
        else np.asarray([], dtype=np.float32)
    )
    fill_color_dispersion_p75 = (
        float(np.percentile(fill_color_distances, 75))
        if fill_color_distances.size
        else 255.0
    )
    fill_color_coherence = max(
        0.0,
        1.0 - min(1.0, fill_color_dispersion_p75 / 64.0),
    )
    paint_confidence = (
        min(
            0.98,
            0.98
            * min(1.0, fill_count / 64.0)
            * fill_color_coherence,
        )
        if fill_cluster_resolved
        else 0.0
    )
    if paint_confidence > 0:
        reasons.append("source_fill_measured_from_authorized_core_paint")
    scale_confidence = max(vertical_confidence, horizontal_confidence)
    metrics = {
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
        "fill_color_dispersion_rgb_p75": round(
            fill_color_dispersion_p75, 6
        ),
        "fill_color_coherence": round(fill_color_coherence, 8),
        "authorized_pixel_count": int(np.count_nonzero(mask_crop)),
        "fill_pixel_count": fill_count,
        "fill_distance_p25": round(fill_distance_p25, 6),
        "ink_distance_p75": round(ink_distance_p75, 6),
        "source_carrier_kind": carrier_kind,
        "source_carrier_observation": source_carrier_observation,
        "outlined_weight_qualification": outlined_weight_qualification,
        "fill_x_spans": [round(float(value), 6) for value in x_spans],
        "fill_y_spans": [round(float(value), 6) for value in y_spans],
        "fill_component_facts": [
            _json_safe_mapping(fact) for fact in fill_component_facts
        ],
        "density_decomposition_vertical_spans": list(
            vertical_qualification.get("density_spans") or []
        ),
        "density_decomposition_horizontal_spans": list(
            horizontal_qualification.get("density_spans") or []
        ),
        "source_cell_qualification_vertical": vertical_qualification,
        "source_cell_qualification_horizontal": horizontal_qualification,
        "authorized_style_spatial_facts": facts.audit_summary(),
    }
    return _AuthorizedStyleCropMeasurement(
        metrics=metrics,
        source_text_footprint=source_text_footprint,
    )


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


def _stable_numeric_tier(
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
    upper_reference = float(np.max(clean))
    cohort = clean[clean >= max(3.0, upper_reference * 0.78)]
    if cohort.size < minimum_count:
        return 0.0, int(cohort.size), 0.0
    median = float(np.median(cohort))
    relative_mad = float(
        np.median(np.abs(cohort - median)) / max(1.0, median)
    )
    if relative_mad > 0.20:
        return 0.0, int(cohort.size), relative_mad
    return median, int(cohort.size), relative_mad


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


def _densest_center_pitch_tier(
    values: Sequence[float],
) -> tuple[float, int, float, bool]:
    """Return one non-harmonic repeated pitch tier or fail ambiguous."""

    clean = sorted(float(value) for value in values if float(value) >= 3.0)
    if len(clean) < 2:
        return 0.0, len(clean), 0.0, False
    candidates: list[tuple[int, float, float, list[float]]] = []
    for seed in clean:
        tolerance = max(1.5, seed * 0.12)
        cohort = [value for value in clean if abs(value - seed) <= tolerance]
        median = float(np.median(cohort))
        relative_mad = float(
            np.median(np.abs(np.asarray(cohort) - median))
            / max(1.0, median)
        )
        candidates.append(
            (len(cohort), relative_mad, median, cohort)
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best = candidates[0]
    if best[1] > 0.12:
        return 0.0, int(best[0]), float(best[1]), False
    competing = [
        item
        for item in candidates[1:]
        if item[0] == best[0]
        and abs(item[2] - best[2]) / max(1.0, best[2]) > 0.18
    ]
    if competing:
        return 0.0, int(best[0]), float(best[1]), True
    return float(best[2]), int(best[0]), float(best[1]), False


def _component_center_pitch_candidates(
    component_facts: Sequence[Mapping[str, Any]],
    *,
    axis: int,
    reference_px: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Collect adjacent glyph-center deltas without bridging punctuation."""

    if reference_px <= 0.0:
        return [], []
    cross_index = 0 if axis == 0 else 1
    inline_index = 1 if axis == 0 else 0
    records: list[dict[str, Any]] = []
    for fact in component_facts:
        center = list(fact.get("center_xy") or ())
        if len(center) != 2:
            continue
        records.append(
            {
                "cross": float(center[cross_index]),
                "inline": float(center[inline_index]),
                "punctuation": bool(
                    fact.get("punctuation_fragment")
                ),
                "area": max(1.0, float(fact.get("area_px") or 1.0)),
            }
        )
    if len(records) < 3:
        return [], []
    track_tolerance = max(3.0, reference_px * 0.65)
    tracks: list[list[dict[str, Any]]] = []
    for record in sorted(records, key=lambda item: item["cross"]):
        candidates = [
            (
                abs(
                    record["cross"]
                    - float(
                        np.average(
                            [item["cross"] for item in track],
                            weights=[item["area"] for item in track],
                        )
                    )
                ),
                index,
            )
            for index, track in enumerate(tracks)
        ]
        candidates = [
            item for item in candidates if item[0] <= track_tolerance
        ]
        if candidates:
            tracks[min(candidates)[1]].append(record)
        else:
            tracks.append([record])

    inline_tolerance = max(
        2.0, min(14.0, reference_px * 0.65)
    )
    deltas: list[float] = []
    track_audit: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks):
        glyph_groups: list[list[dict[str, Any]]] = []
        for record in sorted(track, key=lambda item: item["inline"]):
            if (
                glyph_groups
                and abs(
                    record["inline"]
                    - float(
                        np.average(
                            [item["inline"] for item in glyph_groups[-1]],
                            weights=[item["area"] for item in glyph_groups[-1]],
                        )
                    )
                )
                <= inline_tolerance
            ):
                glyph_groups[-1].append(record)
            else:
                glyph_groups.append([record])
        glyph_records: list[dict[str, Any]] = []
        for group in glyph_groups:
            center = float(
                np.average(
                    [item["inline"] for item in group],
                    weights=[item["area"] for item in group],
                )
            )
            glyph_records.append(
                {
                    "center_px": center,
                    "body": any(
                        not bool(item["punctuation"]) for item in group
                    ),
                    "punctuation": all(
                        bool(item["punctuation"]) for item in group
                    ),
                }
            )
        track_deltas: list[float] = []
        for first, second in zip(glyph_records, glyph_records[1:]):
            delta = float(second["center_px"] - first["center_px"])
            if (
                first["body"]
                and second["body"]
                and reference_px * 0.70
                <= delta
                <= reference_px * 2.40
            ):
                track_deltas.append(delta)
        track_tier = _densest_center_pitch_tier(track_deltas)
        track_qualified = bool(
            track_tier[0] > 0.0
            and track_tier[1] >= 3
            and not track_tier[3]
        )
        if track_qualified:
            tolerance = max(1.5, track_tier[0] * 0.12)
            deltas.extend(
                value
                for value in track_deltas
                if abs(value - track_tier[0]) <= tolerance
            )
        track_audit.append(
            {
                "track_index": track_index,
                "cross_center_px": round(
                    float(
                        np.average(
                            [item["cross"] for item in track],
                            weights=[item["area"] for item in track],
                        )
                    ),
                    6,
                ),
                "glyph_centers_px": [
                    round(float(item["center_px"]), 6)
                    for item in glyph_records
                ],
                "glyph_body_flags": [
                    bool(item["body"]) for item in glyph_records
                ],
                "pitch_candidates_px": [
                    round(value, 6) for value in track_deltas
                ],
                "pitch_tier_px": round(track_tier[0], 6),
                "pitch_tier_count": int(track_tier[1]),
                "pitch_relative_mad": round(track_tier[2], 8),
                "pitch_ambiguous": bool(track_tier[3]),
                "pitch_track_qualified": track_qualified,
                "qualified_pitch_delta_count": (
                    int(track_tier[1]) if track_qualified else 0
                ),
            }
        )
    return deltas, track_audit


def _band_center_pitch_candidates(
    fill: np.ndarray,
    *,
    axis: int,
    reference_px: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Collect adjacent occupied-band deltas without skipping small bands."""

    if reference_px <= 0.0:
        return [], []
    records = _occupied_band_records(fill, axis=1 - axis)
    body_flags = [
        max(3.0, reference_px * 0.35)
        <= float(record.get("span_px") or 0.0)
        <= reference_px * 2.20
        for record in records
    ]
    deltas: list[float] = []
    for index, (first, second) in enumerate(zip(records, records[1:])):
        delta = float(second["center_px"]) - float(first["center_px"])
        if (
            body_flags[index]
            and body_flags[index + 1]
            and reference_px * 0.70
            <= delta
            <= reference_px * 2.40
        ):
            deltas.append(delta)
    audit = [
        {
            **dict(record),
            "body_like_for_pitch": bool(body_flags[index]),
        }
        for index, record in enumerate(records)
    ]
    return deltas, audit


def _repeated_center_pitch(
    fill: np.ndarray,
    *,
    axis: int,
    raw_candidate: float,
    body_candidate: float,
    component_facts: Sequence[Mapping[str, Any]],
) -> tuple[float, int, float, dict[str, Any]]:
    """Resolve cell pitch from repeated centers before body dimensions."""

    reference = body_candidate if body_candidate > 0.0 else raw_candidate
    component_values, component_tracks = (
        _component_center_pitch_candidates(
            component_facts,
            axis=axis,
            reference_px=reference,
        )
    )
    band_values, band_records = _band_center_pitch_candidates(
        fill,
        axis=axis,
        reference_px=reference,
    )
    component_tier = _densest_center_pitch_tier(component_values)
    band_tier = _densest_center_pitch_tier(band_values)
    audit: dict[str, Any] = {
        "center_pitch_reference_px": round(reference, 6),
        "component_center_pitch_candidates_px": [
            round(value, 6) for value in component_values
        ],
        "component_center_pitch_tier_px": round(component_tier[0], 6),
        "component_center_pitch_tier_count": int(component_tier[1]),
        "component_center_pitch_relative_mad": round(
            component_tier[2], 8
        ),
        "component_center_pitch_ambiguous": bool(component_tier[3]),
        "component_pitch_tracks": component_tracks,
        "band_center_pitch_candidates_px": [
            round(value, 6) for value in band_values
        ],
        "band_center_pitch_tier_px": round(band_tier[0], 6),
        "band_center_pitch_tier_count": int(band_tier[1]),
        "band_center_pitch_relative_mad": round(band_tier[2], 8),
        "band_center_pitch_ambiguous": bool(band_tier[3]),
        "band_pitch_records": band_records,
        "center_pitch_independent_sources": False,
        "center_pitch_status": "unavailable",
    }
    if component_tier[3] or band_tier[3]:
        audit["center_pitch_status"] = "ambiguous_competing_pitch_tiers"
        return 0.0, 0, 0.0, audit
    component_ready = component_tier[0] > 0.0 and component_tier[1] >= 2
    band_ready = band_tier[0] > 0.0 and band_tier[1] >= 2
    if component_ready and band_ready:
        disagreement = abs(component_tier[0] - band_tier[0]) / max(
            1.0, min(component_tier[0], band_tier[0])
        )
        audit["center_pitch_source_disagreement"] = round(
            disagreement, 8
        )
        if disagreement > 0.10:
            lower = min(component_tier[0], band_tier[0])
            upper = max(component_tier[0], band_tier[0])
            harmonic_ratio = upper / max(1.0, lower)
            upper_tier = (
                component_tier
                if component_tier[0] >= band_tier[0]
                else band_tier
            )
            if 1.80 <= harmonic_ratio <= 2.20 and upper_tier[1] >= 3:
                audit["center_pitch_harmonic_rejected_px"] = round(
                    lower, 6
                )
                audit["center_pitch_status"] = (
                    "supported_higher_nonfragment_harmonic"
                )
                return (
                    float(upper_tier[0]),
                    int(upper_tier[1]),
                    float(upper_tier[2]),
                    audit,
                )
            audit["center_pitch_status"] = (
                "ambiguous_pitch_sources_disagree"
            )
            return 0.0, 0, 0.0, audit
        candidate = float(
            np.median([component_tier[0], band_tier[0]])
        )
        count = int(component_tier[1] + band_tier[1])
        relative_mad = max(component_tier[2], band_tier[2])
        audit["center_pitch_independent_sources"] = True
        audit["center_pitch_status"] = "supported"
        return candidate, count, relative_mad, audit
    selected = component_tier if component_ready else band_tier
    if selected[0] > 0.0 and selected[1] >= 3:
        audit["center_pitch_status"] = "supported"
        return float(selected[0]), int(selected[1]), float(selected[2]), audit
    return 0.0, 0, 0.0, audit


def _qualify_source_cell_measurement(
    fill: np.ndarray,
    *,
    axis: int,
    spans: Sequence[float],
    component_facts: Sequence[Mapping[str, Any]],
) -> tuple[float, float, str, dict[str, Any]]:
    """Resolve source cell scale from repeated pitch, then corroboration."""

    binary = np.asarray(fill, dtype=bool)
    raw_candidate, raw_count, raw_relative_mad = _stable_numeric_tier(
        spans,
        minimum_count=1,
    )
    axis_key = "width_px" if axis == 0 else "height_px"
    punctuation_count = sum(
        1
        for fact in component_facts
        if bool(fact.get("punctuation_fragment"))
    )
    body_values = [
        float(fact.get(axis_key) or 0.0)
        for fact in component_facts
        if not bool(fact.get("punctuation_fragment"))
        and float(fact.get(axis_key) or 0.0) >= 3.0
    ]
    body_candidate, body_count, body_relative_mad = (
        _stable_numeric_tier(body_values, minimum_count=1)
    )
    (
        pitch_candidate,
        pitch_count,
        pitch_relative_mad,
        pitch_audit,
    ) = _repeated_center_pitch(
        binary,
        axis=axis,
        raw_candidate=raw_candidate,
        body_candidate=body_candidate,
        component_facts=component_facts,
    )

    axis_extent = int(binary.shape[1] if axis == 0 else binary.shape[0])
    orthogonal_extent = int(
        binary.shape[0] if axis == 0 else binary.shape[1]
    )
    coordinates = np.where(binary)
    orthogonal_coordinates = (
        coordinates[0] if axis == 0 else coordinates[1]
    )
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
            minimum_occupancy=max(
                2, int(round(orthogonal_extent * 0.10))
            ),
        )
        (
            density_candidate,
            density_count,
            density_relative_mad,
        ) = _stable_numeric_tier(
            density_spans,
            minimum_count=3,
        )

    projection_matches = sorted(
        float(value)
        for value in [*spans, *density_spans]
        if body_candidate > 0.0
        and float(value) >= 3.0
        and 0.70 <= float(value) / body_candidate <= 1.35
    )
    body_pitch_ratio = (
        body_candidate / pitch_candidate
        if pitch_candidate > 0.0
        else 0.0
    )
    projection_pitch_ratio = (
        raw_candidate / pitch_candidate
        if pitch_candidate > 0.0 and raw_candidate > 0.0
        else 0.0
    )
    audit = {
        "raw_projection_candidate": round(raw_candidate, 6),
        "raw_projection_candidate_count": int(raw_count),
        "raw_projection_relative_mad": round(raw_relative_mad, 8),
        "selected_body_tier_px": round(body_candidate, 6),
        "selected_body_tier_count": int(body_count),
        "selected_body_tier_relative_mad": round(
            body_relative_mad, 8
        ),
        "body_component_count": len(body_values),
        "punctuation_component_count": int(punctuation_count),
        "axis_extent": axis_extent,
        "orthogonal_extent": orthogonal_extent,
        "filled_orthogonal_extent": filled_orthogonal_extent,
        "parent_sized_island": parent_sized_island,
        "density_minimum_occupancy": (
            max(2, int(round(orthogonal_extent * 0.10)))
            if parent_sized_island
            else 0
        ),
        "density_spans": [
            round(float(value), 6) for value in density_spans
        ],
        "density_candidate": round(density_candidate, 6),
        "density_candidate_count": int(density_count),
        "density_relative_mad": round(
            density_relative_mad, 8
        ),
        "body_projection_matches_px": [
            round(value, 6) for value in projection_matches
        ],
        "center_pitch_candidate_px": round(pitch_candidate, 6),
        "center_pitch_candidate_count": int(pitch_count),
        "center_pitch_relative_mad": round(
            pitch_relative_mad, 8
        ),
        "body_to_center_pitch_ratio": round(body_pitch_ratio, 8),
        "projection_to_center_pitch_ratio": round(
            projection_pitch_ratio, 8
        ),
        **pitch_audit,
    }

    if pitch_candidate > 0.0:
        body_corroborated = (
            body_candidate > 0.0
            and 0.45 <= body_pitch_ratio <= 1.15
        )
        projection_corroborated = (
            raw_candidate > 0.0
            and 0.65 <= projection_pitch_ratio <= 1.35
        )
        independent_sources = bool(
            pitch_audit.get("center_pitch_independent_sources")
        )
        audit.update(
            {
                "center_pitch_body_corroborated": body_corroborated,
                "center_pitch_projection_corroborated": (
                    projection_corroborated
                ),
            }
        )
        if body_corroborated and pitch_count >= 3:
            confidence = max(
                0.76,
                min(
                    0.94,
                    0.86
                    - pitch_relative_mad * 0.50
                    + (0.04 if independent_sources else 0.0),
                ),
            )
            return (
                pitch_candidate,
                confidence,
                "supported_repeated_center_pitch",
                audit,
            )
        return (
            0.0,
            0.0,
            "unavailable_center_pitch_not_corroborated",
            audit,
        )

    if str(pitch_audit.get("center_pitch_status") or "").startswith(
        "ambiguous_"
    ):
        return (
            0.0,
            0.0,
            "unavailable_competing_center_pitch",
            audit,
        )

    if parent_sized_island:
        if (
            density_candidate > 0.0
            and raw_max >= density_candidate * 2.2
        ):
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

    if (
        body_candidate > 0.0
        and body_count >= 2
        and projection_matches
    ):
        projection_candidate = min(
            projection_matches,
            key=lambda value: abs(value - body_candidate),
        )
        confidence = max(
            0.72,
            min(
                0.88,
                0.82
                - body_relative_mad * 0.30
                - abs(projection_candidate - body_candidate)
                / max(1.0, body_candidate)
                * 0.18,
            ),
        )
        audit["selected_projection_match_px"] = round(
            projection_candidate, 6
        )
        return (
            projection_candidate,
            confidence,
            "supported_projection_span_body_corroborated",
            audit,
        )

    if not body_values and punctuation_count > 0:
        return (
            0.0,
            0.0,
            "unavailable_punctuation_only_geometry",
            audit,
        )
    return (
        0.0,
        0.0,
        "unavailable_unqualified_body_geometry",
        audit,
    )


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
