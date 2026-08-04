# -*- coding: utf-8 -*-
"""Parent-authorized font observation and pure style arbitration.

The observer in this module accepts only AuthorizedSourceStyleView records.
Unmasked parent bboxes, SourceGlyph diagnostics, page pixels, and render slots
are not executable style evidence. ParentStyleArbitrator is the only owner that
turns typed observations into a complete resolved render style.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from app.models.resolution import (
    resolve_noto_cjk_sc_font_file,
    resolve_yuzumarker_font_labels_file,
    resolve_yuzumarker_font_onnx_file,
)
from app.pipeline.parent_execution_bundle import (
    PARENT_RENDER_STYLE_VERSION,
    PARENT_STYLE_DEFAULT_FALLBACK_FONT_CHAIN_KEY,
    PARENT_STYLE_ARBITRATOR_PROVIDER,
    PARENT_STYLE_ARBITRATOR_SOURCE,
    PARENT_STYLE_UNRESOLVED_FONT_SIZE,
    PARENT_STYLE_UNRESOLVED_FONT_SIZE_AUTHORITY,
    PARENT_STYLE_UNRESOLVED_FONT_SIZE_MAX,
    PARENT_STYLE_UNRESOLVED_FONT_SIZE_MIN,
    PARENT_STYLE_UNRESOLVED_FONT_SIZE_POLICY,
    validate_resolved_render_style,
)
from app.pipeline.parent_style_evidence import (
    AuthorizedSourceStyleView,
    EXTERNAL_SOURCE_SURFACE_RING_VERSION,
    SOURCE_STYLE_AXES,
    SOURCE_ADVANCE_GRID_VERSION,
    SourceAdvanceGridEvidence,
    SourceTextFootprint,
    SourceStyleAxisEvidence,
    build_authorized_style_observation_inputs,
)
from app.pipeline.source_style_contracts import (
    SOURCE_FONT_SUPPORT_FLOOR_MET,
    SOURCE_FONT_SUPPORT_TRUNCATED,
    TARGET_FONT_AFFINITY_OBSERVATION_KEY,
    TARGET_FONT_AFFINITY_ROLE_IDS,
    SourceFontCandidate,
    SourceFontObservationV3,
    SourceFontScoreSupportV1,
    SourceStyleEvidenceBindingV1,
    TargetFontAffinityObservationV1,
    source_font_overlap_bounds,
)
from app.render.font_manager import (
    TARGET_OPTICAL_PROFILE_POLICY_ID,
    FontManager,
)


FONT_COUNT = 6150
SOURCE_FONT_RETAINED_MASS_FLOOR = 0.999
SOURCE_FONT_CANDIDATE_CEILING = 1024
SOURCE_FONT_SUPPORT_POLICY_VERSION = (
    "yuzumarker_adaptive_identity_support_mass_0_999_ceiling_1024_v1"
)
SOURCE_FONT_SCORE_SUPPORT_KEY = "source_font_score_support_v1"
TARGET_FONT_AFFINITY_DESCRIPTOR_POLICY_VERSION = (
    "hellinger_t2__primary__top3_mean"
)
TARGET_FONT_AFFINITY_PROBE_POLICY_VERSION = (
    "fixed_disjoint_cjk_probe_bank_v1"
)
TARGET_FONT_AFFINITY_TEMPERATURE = 2.0
TARGET_FONT_AFFINITY_TOP_PROBE_COUNT = 3
TARGET_FONT_AFFINITY_ERROR_KEY = "target_font_affinity_error"
_V3_TARGET_FONT_NEIGHBOR_COUNT = 2
_V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT = {
    "sans_regular": 400,
    "sans_medium": 500,
    "sans_bold": 700,
    "sans_black": 900,
    "serif_regular": 400,
    "serif_semibold": 600,
    "serif_bold": 700,
}
_V3_TARGET_FONT_ROLE_FAMILY = {
    role_id: ("sans" if role_id.startswith("sans_") else "serif")
    for role_id in TARGET_FONT_AFFINITY_ROLE_IDS
}
_V3_TARGET_FONT_FAMILY_ROLES = {
    family: tuple(
        role_id
        for role_id in TARGET_FONT_AFFINITY_ROLE_IDS
        if _V3_TARGET_FONT_ROLE_FAMILY[role_id] == family
    )
    for family in ("sans", "serif")
}
_V3_TARGET_FONT_WEIGHT_COMPATIBILITY_ALIAS = {
    "sans_regular": "slender",
    "sans_medium": "base",
    "sans_bold": "emphasis",
    "sans_black": "heavy",
    "serif_regular": "slender",
    "serif_semibold": "base",
    "serif_bold": "emphasis",
}
TARGET_FONT_AFFINITY_PROBE_SPECS = (
    ("cjk_cosmos_single", "天地玄黄宇宙洪荒", 1, 52, 56),
    ("cjk_seasons_single", "春夏秋冬東西南北", 1, 52, 56),
    ("cjk_nature_single", "海山川空月星光雨", 1, 52, 56),
    ("hiragana_single", "あいうえおかきくけこ", 1, 44, 46),
    ("katakana_single", "アイウエオカキクケコ", 1, 44, 46),
    ("cjk_life_two_column", "永語愛無人生活仕事", 2, 62, 68),
    ("cjk_story_two_column", "漢字仮名読書物語心", 2, 62, 68),
    ("cjk_motion_two_column", "時風道夢声旅力世界", 2, 62, 68),
)
FAMILY_POSTERIOR_VERSION = "yuzumarker_complete_family_posterior_v1"
SOURCE_STYLE_EVIDENCE_V2 = "source_style_evidence_v2"
NORMALIZED_STROKE_PROFILE_V2 = "normalized_stroke_profile_v2"
PARENT_STYLE_OPTICAL_REALIZATION_BRIDGE_VERSION = (
    "parent_style_optical_realization_bridge_v1"
)
PARENT_STYLE_OPTICAL_ESTIMATOR_POLICY = (
    "median_supported_nonduplicate_sans_target_to_source_fixed_probe_ratio_v1"
)
PARENT_STYLE_OPTICAL_ESTIMATOR_CONVERSION = 0.8680860635000001
PARENT_STYLE_TARGET_OUTLINE_REALIZATION_VERSION = (
    "fixed_target_outline_carrier_v1"
)
FACTORIZED_ATTRIBUTE_POSTERIOR_VERSION = (
    "yuzumarker_factorized_attribute_posterior_v1"
)
FACTORIZED_ATTRIBUTE_TAXONOMY_VERSION = (
    "yuzumarker_factorized_attribute_taxonomy_v1"
)
FACTORIZED_ATTRIBUTE_VARIANTS_VERSION = (
    "yuzumarker_factorized_attribute_variants_v1"
)
FAMILY_CALIBRATION_VERSION = "stage1a_family_posterior_calibration_v2"
FAMILY_CALIBRATION_RELIABILITY_METHOD = (
    "wilson_score_lower_bound_95_two_sided"
)
FAMILY_CALIBRATION_Z_SCORE = 1.959963984540054
FAMILY_KNOWN_MASS_MINIMUM = 0.80
FAMILY_MARGIN_MINIMUM = 1.0
FAMILY_NORMALIZED_ENTROPY_MAXIMUM = 1e-12
FAMILY_REQUIRE_VARIANT_AGREEMENT = True
FAMILY_CALIBRATION_PROMOTED = 1
FAMILY_CALIBRATION_CORRECT = 1
FAMILY_CALIBRATION_FALSE_HIGH_CONFIDENCE = 0
FAMILY_CALIBRATION_RELIABILITY_MINIMUM = 0.95
YUZUMARKER_PROVIDER = "YuzuMarker.FontDetection"
YUZUMARKER_PROVIDER_MODEL = "ogkalu/yuzumarker-font-detection-onnx:font-detector.onnx"
YUZUMARKER_STYLE_SOURCE = "authorized_source_style_view_yuzumarker"
HEURISTIC_PROVIDER = "ParentFontHeuristic"
HEURISTIC_STYLE_SOURCE = "authorized_source_style_view_heuristic"
STYLE_ARBITRATOR_PROVIDER = PARENT_STYLE_ARBITRATOR_PROVIDER
STYLE_ARBITRATOR_SOURCE = PARENT_STYLE_ARBITRATOR_SOURCE
MIN_STYLE_EVIDENCE_CONFIDENCE = 0.05
PERCEPTUAL_STYLE_AXES_VERSION = "authorized_perceptual_style_axes_v2"
PERCEPTUAL_STYLE_RESOLUTION_VERSION = "parent_style_perceptual_axis_resolution_v2"
PERCEPTUAL_STYLE_PROVENANCE = "cleanup_mask_authorized_source_style_view_v1"
PERCEPTUAL_STYLE_FACT_SET_PREFIX = "authorized_perceptual_fact_set_v1:"
PERCEPTUAL_STYLE_AXES = ("fill", "outline", "shadow", "rotation")
CORE_STYLE_AXES = ("family", "weight", "scale", "fill", "outline", "orientation")
PEER_ASSIST_AXES = ("family", "weight", "orientation", "scale")
DIRECT_AXIS_MIN_CONFIDENCE = 0.20
DIRECT_PAINT_MIN_CONFIDENCE = 0.20
DIRECT_OUTLINE_MIN_CONFIDENCE = 0.20
PEER_DONOR_MIN_CONFIDENCE = 0.65
PEER_TARGET_RELIABLE_CONFIDENCE = 0.65
ORIENTATION_VOTE_MIN_CONFIDENCE = 0.60
PEER_SCALE_MAXIMUM_RELATIVE_SPREAD = 0.18
PEER_COMPATIBLE_SCALE_MAXIMUM_RELATIVE_SPREAD = 0.25
PEER_MINIMUM_DONOR_COUNT = 2
MAX_STYLE_CARRIER_DEPTH = 64
MAX_STYLE_CARRIER_NODES = 10000
PARENT_STYLE_DECISION_LEDGER_VERSION = "parent_style_decision_ledger_v3"
PARENT_STYLE_DECISION_AXES_V3 = (
    "family",
    "weight",
    "source_scale",
    "fill",
    "outline",
    "orientation",
    "rotation",
    "shadow",
)
PARENT_STYLE_PEER_AXES_V3 = ("family", "weight", "source_scale")
PARENT_RENDER_STYLE_V3_VERSION = "parent_render_style_v3"
PARENT_RENDER_STYLE_LEDGER_V3_VERSION = "parent_render_style_ledger_v3"
PARENT_RENDER_STYLE_V3_TARGET_FALLBACK_EM_PX = 24.0
PARENT_RENDER_STYLE_V3_FONT_ROLE_MATRIX = {
    "sans": {
        "slender": ("sans_regular", "registered_role"),
        "base": ("sans_medium", "registered_role"),
        "emphasis": ("sans_bold", "registered_role"),
        "heavy": ("sans_black", "registered_role"),
    },
    "serif": {
        "slender": ("serif_regular", "registered_role"),
        "base": ("serif_semibold", "registered_role"),
        "emphasis": ("serif_bold", "registered_role"),
        "heavy": ("serif_bold", "degraded_registered_role"),
    },
}
V3_DIRECT_FAMILY_MIN_CONFIDENCE = 0.65
V3_DIRECT_WEIGHT_MIN_CONFIDENCE = 0.55
V3_DIRECT_SCALE_MIN_CONFIDENCE = 0.55
V3_PEER_DONOR_MIN_CONFIDENCE = 0.65
V3_DIRECT_FILL_MIN_CONFIDENCE = 0.50
V3_DIRECT_OUTLINE_MIN_CONFIDENCE = 0.65
V3_DIRECT_ORIENTATION_MIN_CONFIDENCE = 0.60
V3_POSTERIOR_KNOWN_MASS_MINIMUM = 0.80
V3_POSTERIOR_LEADING_PROBABILITY_MINIMUM = 0.85
V3_POSTERIOR_MARGIN_MINIMUM = 0.70
V3_WEIGHT_SLENDER_SCORE_RANGE = (0.1395, 0.1630)
V3_WEIGHT_BASE_SCORE_RANGE = (0.1800, 0.2500)
V3_WEIGHT_HEAVY_SCORE_RANGE = (0.2600, 0.3400)
V3_SCALE_PEER_MAXIMUM_MAD_RATIO = 0.10
TARGET_FONT_REQUEST_VERSION = "target_font_request_v1"
TARGET_FONT_REQUEST_PROVENANCE = "parent_style_arbitrator_source_label_taxonomy_v1"
OPTIONAL_TARGET_FONT_LABEL_TAXONOMY: dict[str, dict[str, str]] = {
    "STXINGKA.TTF": {
        "catalog_face_id": "stxingkai_regular",
        "style_class": "calligraphic",
        "weight": "regular",
    },
}


@dataclass(frozen=True)
class _InvalidFrozenJsonValue:
    """Non-serializable marker retained so malformed axes fail locally."""

    reason: str


class _FrozenJsonDict(dict[Any, Any]):
    """JSON-serializable mapping that cannot be mutated through evidence aliases."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("frozen JSON snapshot")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> dict[str, Any]:
        return _plain_json_mapping_snapshot(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        snapshot = _plain_json_mapping_snapshot(self)
        memo[id(self)] = snapshot
        return snapshot


@dataclass(frozen=True)
class _FactorizedAttributeTaxonomy:
    """Source-label attribute partitions used only to transport posterior mass."""

    label_count: int
    generic_family_codes: np.ndarray
    face_character_codes: np.ndarray
    weight_strength_codes: np.ndarray


@dataclass(frozen=True)
class FontFamilyPosterior:
    """Complete serif/sans/unknown mass from all exact-font outputs."""

    label_count: int
    known_label_count: int
    unknown_label_count: int
    sans_mass: float
    serif_mass: float
    unknown_mass: float
    known_mass: float
    conditional_sans_probability: float
    conditional_serif_probability: float
    leading_family: str
    margin: float
    normalized_entropy: float
    version: str = FAMILY_POSTERIOR_VERSION

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "label_count": int(self.label_count),
            "known_label_count": int(self.known_label_count),
            "unknown_label_count": int(self.unknown_label_count),
            "sans_mass": float(self.sans_mass),
            "serif_mass": float(self.serif_mass),
            "unknown_mass": float(self.unknown_mass),
            "known_mass": float(self.known_mass),
            "conditional_sans_probability": float(
                self.conditional_sans_probability
            ),
            "conditional_serif_probability": float(
                self.conditional_serif_probability
            ),
            "leading_family": self.leading_family,
            "margin": float(self.margin),
            "normalized_entropy": float(self.normalized_entropy),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "FontFamilyPosterior | None":
        if not isinstance(value, Mapping):
            return None
        if str(value.get("version") or "") != FAMILY_POSTERIOR_VERSION:
            return None
        numeric_keys = (
            "sans_mass",
            "serif_mass",
            "unknown_mass",
            "known_mass",
            "conditional_sans_probability",
            "conditional_serif_probability",
            "margin",
            "normalized_entropy",
        )
        parsed: dict[str, float] = {}
        for key in numeric_keys:
            try:
                number = float(value.get(key))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number) or number < 0.0 or number > 1.0 + 1e-6:
                return None
            parsed[key] = min(1.0, number)
        if abs(
            parsed["sans_mass"]
            + parsed["serif_mass"]
            + parsed["unknown_mass"]
            - 1.0
        ) > 1e-5:
            return None
        if abs(
            parsed["sans_mass"]
            + parsed["serif_mass"]
            - parsed["known_mass"]
        ) > 1e-5:
            return None
        leading_family = str(value.get("leading_family") or "")
        if leading_family not in {"", "sans", "serif"}:
            return None
        known_mass = parsed["known_mass"]
        expected_conditional_sans = (
            parsed["sans_mass"] / known_mass if known_mass > 0.0 else 0.0
        )
        expected_conditional_serif = (
            parsed["serif_mass"] / known_mass if known_mass > 0.0 else 0.0
        )
        expected_leading_family = (
            "sans"
            if expected_conditional_sans > expected_conditional_serif
            else "serif"
            if expected_conditional_serif > expected_conditional_sans
            else ""
        )
        expected_margin = abs(
            expected_conditional_sans - expected_conditional_serif
        )
        expected_entropy = _binary_normalized_entropy(
            expected_conditional_sans,
            expected_conditional_serif,
        )
        derived_values = (
            (
                parsed["conditional_sans_probability"],
                expected_conditional_sans,
            ),
            (
                parsed["conditional_serif_probability"],
                expected_conditional_serif,
            ),
            (parsed["margin"], expected_margin),
            (parsed["normalized_entropy"], expected_entropy),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in derived_values
        ) or leading_family != expected_leading_family:
            return None
        try:
            label_count = int(value.get("label_count") or 0)
            known_label_count = int(value.get("known_label_count") or 0)
            unknown_label_count = int(value.get("unknown_label_count") or 0)
        except (TypeError, ValueError):
            return None
        if min(label_count, known_label_count, unknown_label_count) < 0:
            return None
        if known_label_count + unknown_label_count != label_count:
            return None
        return cls(
            label_count=label_count,
            known_label_count=known_label_count,
            unknown_label_count=unknown_label_count,
            sans_mass=parsed["sans_mass"],
            serif_mass=parsed["serif_mass"],
            unknown_mass=parsed["unknown_mass"],
            known_mass=parsed["known_mass"],
            conditional_sans_probability=parsed[
                "conditional_sans_probability"
            ],
            conditional_serif_probability=parsed[
                "conditional_serif_probability"
            ],
            leading_family=leading_family,
            margin=parsed["margin"],
            normalized_entropy=parsed["normalized_entropy"],
        )


@dataclass(frozen=True)
class FamilyAxisObservation:
    posterior: FontFamilyPosterior
    promoted: bool
    family_role: str = ""
    font_serif: bool | None = None
    confidence: float = 0.0
    calibration_reliability: float = 0.0
    reason: str = ""
    variant_agreement: bool = False


@dataclass(frozen=True)
class StyleEvidence:
    """JSON-safe style observation summary for one parent."""

    page_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    status: str
    vote_eligible: bool = False
    reason_codes: tuple[str, ...] = ()
    view_id: str = ""
    cleanup_mask_ids: tuple[str, ...] = ()
    owned_component_ids: tuple[str, ...] = ()
    content_bbox: tuple[int, int, int, int] = ()
    analysis_bbox: tuple[int, int, int, int] = ()
    detector_input_sha256: str = ""
    source_text_footprint: SourceTextFootprint | None = None
    source_advance_grid: SourceAdvanceGridEvidence | None = None
    source_font_observation: SourceFontObservationV3 | None = None
    source_font_style_evidence: SourceStyleAxisEvidence | None = None
    authorized_perceptual_source_identity: Mapping[str, Any] = field(
        default_factory=dict
    )
    evidence_provider: str = ""
    evidence_source: str = ""
    evidence_model: str = ""
    confidence: float = 0.0
    font_label: str = ""
    font_weight: str = ""
    font_language: str = ""
    font_serif: bool | None = None
    family_posterior: FontFamilyPosterior | None = None
    top_candidates: tuple[dict[str, Any], ...] = ()
    direction: str = ""
    direction_confidence: float = 0.0
    text_color: str = ""
    stroke_color: str = ""
    text_size_ratio: float = 0.0
    source_size_px: float = 0.0
    source_size_vertical_px: float = 0.0
    source_size_horizontal_px: float = 0.0
    source_size_confidence_vertical: float = 0.0
    source_size_confidence_horizontal: float = 0.0
    source_size_support_vertical: str = ""
    source_size_support_horizontal: str = ""
    source_scale_support_status: str = ""
    source_stroke_width_px: float = 0.0
    source_ink_stroke_width_px: float = 0.0
    stroke_width_ratio: float = 0.0
    line_spacing_ratio: float = 0.0
    angle_degrees: float = 0.0
    axis_confidence: Mapping[str, float] = field(default_factory=dict)
    axis_provenance: Mapping[str, str] = field(default_factory=dict)
    observation_summary: Mapping[str, Any] = field(default_factory=dict)
    detector_variant_summary: Mapping[str, Any] = field(default_factory=dict)
    perceptual_axis_evidence: Mapping[str, Any] = field(default_factory=dict)
    axis_evidence: tuple[SourceStyleAxisEvidence, ...] = ()

    def __post_init__(self) -> None:
        # StyleEvidence is the raw observation contract.  Snapshot and freeze
        # every nested JSON carrier at construction so producer inputs,
        # arbitration audit records, and bundle transport can never alias back
        # into that evidence.  SourceTextFootprint is already a frozen typed
        # contract and the remaining fields are scalars or immutable tuples.
        for field_name in (
            "authorized_perceptual_source_identity",
            "axis_confidence",
            "axis_provenance",
            "observation_summary",
            "detector_variant_summary",
            "perceptual_axis_evidence",
        ):
            object.__setattr__(
                self,
                field_name,
                _frozen_json_mapping_snapshot(getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "top_candidates",
            _frozen_json_sequence_snapshot(self.top_candidates),
        )
        if self.family_posterior is not None and not isinstance(
            self.family_posterior,
            FontFamilyPosterior,
        ):
            object.__setattr__(
                self,
                "family_posterior",
                FontFamilyPosterior.from_mapping(self.family_posterior),
            )
        if self.source_font_observation is not None and not isinstance(
            self.source_font_observation,
            SourceFontObservationV3,
        ):
            raise TypeError(
                "source_font_observation must be SourceFontObservationV3"
            )
        if self.source_advance_grid is not None and not isinstance(
            self.source_advance_grid,
            SourceAdvanceGridEvidence,
        ):
            raise TypeError(
                "source_advance_grid must be SourceAdvanceGridEvidence"
            )
        if self.source_font_style_evidence is not None and not isinstance(
            self.source_font_style_evidence,
            SourceStyleAxisEvidence,
        ):
            raise TypeError(
                "source_font_style_evidence must be SourceStyleAxisEvidence"
            )
        object.__setattr__(self, "axis_evidence", tuple(self.axis_evidence or ()))

    @classmethod
    def unavailable(
        cls,
        *,
        page_id: str,
        bundle_id: str,
        parent_id: str,
        root_id: str,
        reason_codes: Sequence[str],
        view: AuthorizedSourceStyleView | None = None,
        detector_input_sha256: str = "",
        source_text_footprint: SourceTextFootprint | None = None,
        authorized_perceptual_source_identity: Mapping[str, Any] | None = None,
        perceptual_axis_evidence: Mapping[str, Any] | None = None,
    ) -> "StyleEvidence":
        return cls(
            page_id=str(page_id or ""),
            bundle_id=str(bundle_id or ""),
            parent_id=str(parent_id or bundle_id or ""),
            root_id=str(root_id or ""),
            status="unavailable",
            vote_eligible=False,
            reason_codes=tuple(_unique_strings(reason_codes)),
            view_id=str(getattr(view, "view_id", "") or ""),
            cleanup_mask_ids=tuple(getattr(view, "cleanup_mask_ids", ()) or ()),
            owned_component_ids=tuple(getattr(view, "owned_component_ids", ()) or ()),
            content_bbox=tuple(getattr(view, "content_bbox", ()) or ()),
            analysis_bbox=tuple(getattr(view, "analysis_bbox", ()) or ()),
            detector_input_sha256=str(detector_input_sha256 or ""),
            source_text_footprint=source_text_footprint,
            authorized_perceptual_source_identity=dict(
                authorized_perceptual_source_identity or {}
            ),
            perceptual_axis_evidence=dict(perceptual_axis_evidence or {}),
        )

    @classmethod
    def observed_for_test(
        cls,
        *,
        page_id: str,
        bundle_id: str,
        parent_id: str,
        root_id: str,
        font_serif: bool,
        font_label: str,
        confidence: float,
        source_size_px: float,
    ) -> "StyleEvidence":
        support_identity = {
            "page_id": page_id,
            "view_id": f"styleview_{page_id}_{bundle_id}",
            "bundle_id": bundle_id,
            "parent_id": parent_id,
            "root_id": root_id,
            "cleanup_mask_ids": [f"cmask_{bundle_id}"],
            "authorized_mask_sha256": "test",
            "authorized_pixel_sha256": "test",
            "detector_input_sha256": "test",
        }
        confidence = float(confidence)
        source_size_px = float(source_size_px)
        axis_evidence = (
            SourceStyleAxisEvidence(
                axis="family",
                status="supported",
                value={
                    "font_label": font_label,
                    "font_serif": bool(font_serif),
                    "font_language": "CJK",
                },
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence(
                axis="weight",
                status="supported",
                value={"class": _font_weight_from_label(font_label) or "regular"},
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence(
                axis="scale",
                status="supported",
                value={
                    "vertical_px": source_size_px,
                    "vertical_confidence": confidence,
                    "vertical_support": "supported_test_evidence",
                    "horizontal_px": source_size_px,
                    "horizontal_confidence": confidence,
                    "horizontal_support": "supported_test_evidence",
                },
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence(
                axis="fill",
                status="supported",
                value={"color": "#111111", "support_color": "#EEEEEE"},
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence(
                axis="outline",
                status="supported",
                value={
                    "present": True,
                    "kind": "outline",
                    "color": "#EEEEEE",
                    "width_px": max(0.0, source_size_px * 0.02),
                },
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence(
                axis="orientation",
                status="supported",
                value={"direction": "ttb"},
                confidence=confidence,
                provenance="test_authorized_evidence",
                support_identity=support_identity,
            ),
            SourceStyleAxisEvidence.unavailable(
                "rotation",
                provenance="test_authorized_evidence",
                support_identity=support_identity,
                reason_codes=("test_rotation_unavailable",),
            ),
            SourceStyleAxisEvidence.unavailable(
                "shadow",
                provenance="test_authorized_evidence",
                support_identity=support_identity,
                reason_codes=("test_shadow_unavailable",),
            ),
        )
        return cls(
            page_id=page_id,
            bundle_id=bundle_id,
            parent_id=parent_id,
            root_id=root_id,
            status="observed",
            vote_eligible=True,
            reason_codes=("authorized_source_style_view_observed",),
            view_id=f"styleview_{page_id}_{bundle_id}",
            cleanup_mask_ids=(f"cmask_{bundle_id}",),
            owned_component_ids=("component-test",),
            content_bbox=(0, 0, 32, 64),
            analysis_bbox=(0, 0, 36, 68),
            detector_input_sha256="test",
            evidence_provider=YUZUMARKER_PROVIDER,
            evidence_source=YUZUMARKER_STYLE_SOURCE,
            evidence_model=YUZUMARKER_PROVIDER_MODEL,
            confidence=confidence,
            font_label=font_label,
            font_weight=_font_weight_from_label(font_label) or "",
            font_language="CJK",
            font_serif=bool(font_serif),
            direction="ttb",
            direction_confidence=confidence,
            text_color="#111111",
            stroke_color="#EEEEEE",
            text_size_ratio=source_size_px / 36.0,
            source_size_px=source_size_px,
            source_size_vertical_px=source_size_px,
            source_size_horizontal_px=source_size_px,
            source_size_confidence_vertical=confidence,
            source_size_confidence_horizontal=confidence,
            source_size_support_vertical="supported_test_evidence",
            source_size_support_horizontal="supported_test_evidence",
            source_scale_support_status="supported_test_evidence",
            source_stroke_width_px=max(0.0, source_size_px * 0.02),
            source_ink_stroke_width_px=max(0.0, source_size_px * 0.08),
            stroke_width_ratio=0.02,
            line_spacing_ratio=0.05,
            axis_confidence={
                "family": confidence,
                "weight": confidence,
                "scale": confidence,
                "fill": confidence,
                "outline": confidence,
                "orientation": confidence,
            },
            axis_provenance={
                "family": "test_authorized_evidence",
                "weight": "test_authorized_evidence",
                "scale": "test_authorized_evidence",
                "fill": "test_authorized_evidence",
                "outline": "test_authorized_evidence",
                "orientation": "test_authorized_evidence",
            },
            axis_evidence=axis_evidence,
        )

    def source_axes(self) -> dict[str, Any]:
        if self.status != "observed" or not self.vote_eligible:
            return {}
        return {
            "font_label": self.font_label,
            "font_weight": self.font_weight,
            "font_serif": self.font_serif,
            "family_posterior": (
                self.family_posterior.to_audit_dict()
                if self.family_posterior is not None
                else None
            ),
            "font_language": self.font_language,
            "direction": self.direction,
            "direction_confidence": round(float(self.direction_confidence), 8),
            "text_color": self.text_color,
            "stroke_color": self.stroke_color,
            "text_size_ratio": round(float(self.text_size_ratio), 8),
            "source_size_px": round(float(self.source_size_px), 8),
            "source_size_vertical_px": round(
                float(self.source_size_vertical_px), 8
            ),
            "source_size_horizontal_px": round(
                float(self.source_size_horizontal_px), 8
            ),
            "source_size_confidence_vertical": round(
                float(self.source_size_confidence_vertical), 8
            ),
            "source_size_confidence_horizontal": round(
                float(self.source_size_confidence_horizontal), 8
            ),
            "source_size_support_vertical": self.source_size_support_vertical,
            "source_size_support_horizontal": self.source_size_support_horizontal,
            "source_scale_support_status": self.source_scale_support_status,
            "source_stroke_width_px": round(float(self.source_stroke_width_px), 8),
            "source_ink_stroke_width_px": round(
                float(self.source_ink_stroke_width_px), 8
            ),
            "stroke_width_ratio": round(float(self.stroke_width_ratio), 8),
            "line_spacing_ratio": round(float(self.line_spacing_ratio), 8),
            "angle_degrees": round(float(self.angle_degrees), 8),
            "axis_confidence": _plain_json_mapping_snapshot(
                self.axis_confidence
            ),
            "axis_provenance": _plain_json_mapping_snapshot(
                self.axis_provenance
            ),
            "observation_summary": _plain_json_mapping_snapshot(
                self.observation_summary
            ),
            "detector_variant_summary": _plain_json_mapping_snapshot(
                self.detector_variant_summary
            ),
            "axis_evidence": [
                record.to_audit_dict() for record in self.axis_evidence
            ],
        }

    def to_audit_dict(self) -> dict[str, Any]:
        result = {
            "style_evidence_version": "parent_style_evidence_v2",
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "status": self.status,
            "vote_eligible": bool(self.vote_eligible),
            "reason_codes": list(self.reason_codes),
            "view_id": self.view_id,
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "owned_component_ids": list(self.owned_component_ids),
            "content_bbox": list(self.content_bbox),
            "analysis_bbox": list(self.analysis_bbox),
            "detector_input_sha256": self.detector_input_sha256,
            "evidence_provider": self.evidence_provider,
            "evidence_source": self.evidence_source,
            "evidence_model": self.evidence_model,
            "confidence": float(self.confidence),
            "source_axes": self.source_axes(),
            "axis_evidence": [
                record.to_audit_dict() for record in self.axis_evidence
            ],
        }
        if self.source_text_footprint is not None:
            result["source_text_footprint"] = (
                self.source_text_footprint.to_audit_dict()
            )
        if self.source_advance_grid is not None:
            result["source_advance_grid"] = (
                self.source_advance_grid.to_audit_dict()
            )
        if self.source_font_observation is not None:
            result["source_font_observation"] = (
                self.source_font_observation.to_audit_dict()
            )
        if self.source_font_style_evidence is not None:
            result["source_font_style_evidence"] = (
                self.source_font_style_evidence.to_audit_dict()
            )
        if self.perceptual_axis_evidence:
            result["authorized_perceptual_source_identity"] = (
                _json_safe_audit_mapping(
                    self.authorized_perceptual_source_identity
                )
            )
            result["perceptual_axis_evidence"] = _json_safe_audit_mapping(
                self.perceptual_axis_evidence
            )
        return result


@dataclass
class ParentStyleEvidenceRunResult:
    page_id: str
    mode: str
    enabled: bool = False
    evidence: list[StyleEvidence] = field(default_factory=list)
    model_path: str = ""
    labels_path: str = ""
    gpu_requested: bool = False
    requested_execution_provider: str = ""
    available_execution_providers: list[str] = field(default_factory=list)
    active_execution_providers: list[str] = field(default_factory=list)
    primary_execution_provider: str = ""
    provider_fallback_reason: str = ""
    provider_preload_error: str = ""
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_style_evidence_run_version": "parent_style_evidence_run_v1",
            "page_id": self.page_id,
            "mode": self.mode,
            "enabled": bool(self.enabled),
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "gpu_requested": bool(self.gpu_requested),
            "requested_execution_provider": self.requested_execution_provider,
            "available_execution_providers": list(self.available_execution_providers),
            "active_execution_providers": list(self.active_execution_providers),
            "primary_execution_provider": self.primary_execution_provider,
            "provider_fallback_reason": self.provider_fallback_reason,
            "provider_preload_error": self.provider_preload_error,
            "errors": list(self.errors),
            "evidence": [item.to_audit_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class ParentStyleArbitrationResult:
    resolved_styles: dict[str, dict[str, Any]]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _AxisCandidate:
    axis: str
    value: Any
    confidence: float
    provenance: str
    source: str
    support_status: str = "supported"
    reason_codes: tuple[str, ...] = ()
    peer_support: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AxisDecision:
    axis: str
    value: Any
    status: str
    confidence: float
    authority: str
    provenance: str
    source: str
    support_status: str = ""
    reason_codes: tuple[str, ...] = ()
    peer_support: Mapping[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, Mapping):
            value = dict(value)
        return {
            "status": self.status,
            "value": value,
            "confidence": round(float(self.confidence), 8),
            "authority": self.authority,
            "provenance": self.provenance,
            "source": self.source,
            "support_status": self.support_status,
            "reason_codes": list(self.reason_codes),
            "peer_support": dict(self.peer_support),
        }


@dataclass(frozen=True)
class _ParentAxisCandidates:
    direct: Mapping[str, _AxisCandidate] = field(default_factory=dict)
    directional_weight: Mapping[str, _AxisCandidate] = field(default_factory=dict)
    directional_scale: Mapping[str, _AxisCandidate] = field(default_factory=dict)


@dataclass(frozen=True)
class _ParentAxisDecisionSet:
    decisions: Mapping[str, _AxisDecision]
    peer_assisted_axes: tuple[str, ...] = ()
    peer_support: Mapping[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            axis: self.decisions[axis].to_audit_dict()
            for axis in (
                "family",
                "weight",
                "orientation",
                "scale",
                "fill",
                "outline",
                "rotation",
                "shadow",
            )
            if axis in self.decisions
        }


@dataclass(frozen=True)
class ParentStyleAxisDecisionV3:
    """One immutable Stage 2C style-axis decision.

    This is deliberately a non-runtime policy record.  Stage 3 performs the
    single executable contract cutover; constructing this record never mutates
    a ParentExecutionBundle or projects a v3 decision back into v2 fields.
    """

    axis: str
    value: Any
    status: str
    confidence: float
    provenance: str
    reason_codes: tuple[str, ...] = ()
    peer_support: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axis = str(self.axis or "").strip().lower()
        status = str(self.status or "unavailable").strip().lower()
        if axis not in PARENT_STYLE_DECISION_AXES_V3:
            raise ValueError(f"unsupported Stage 2C axis: {axis}")
        if status not in {"direct", "peer", "fallback", "unavailable"}:
            raise ValueError(f"unsupported Stage 2C decision status: {status}")
        confidence = _unit_interval(self.confidence)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "confidence", float(confidence or 0.0))
        object.__setattr__(self, "provenance", str(self.provenance or ""))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_unique_strings(self.reason_codes)),
        )
        if isinstance(self.value, Mapping):
            frozen_value: Any = _frozen_json_mapping_snapshot(self.value)
        elif _is_plain_sequence(self.value):
            frozen_value = _frozen_json_sequence_snapshot(self.value)
        else:
            frozen_value = self.value
        object.__setattr__(self, "value", frozen_value)
        object.__setattr__(
            self,
            "peer_support",
            _frozen_json_mapping_snapshot(self.peer_support),
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "status": self.status,
            "authority": self.status,
            "value": _v3_plain_json_value(self.value),
            "confidence": round(float(self.confidence), 8),
            "provenance": self.provenance,
            "reason_codes": list(self.reason_codes),
            "peer_support": _v3_plain_json_value(self.peer_support),
        }


@dataclass(frozen=True)
class ParentStyleParentDecisionV3:
    """Complete non-runtime Stage 2C decision record for one parent."""

    page_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    semantic_role_class: str
    writing_mode: str
    source_evidence_status: str
    axes: tuple[ParentStyleAxisDecisionV3, ...]
    peer_assisted_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(
                tuple(self.axes or ()),
                key=lambda item: PARENT_STYLE_DECISION_AXES_V3.index(item.axis),
            )
        )
        if tuple(item.axis for item in ordered) != PARENT_STYLE_DECISION_AXES_V3:
            raise ValueError("Stage 2C parent decision must contain every axis once")
        object.__setattr__(self, "page_id", str(self.page_id or ""))
        object.__setattr__(self, "bundle_id", str(self.bundle_id or ""))
        object.__setattr__(self, "parent_id", str(self.parent_id or ""))
        object.__setattr__(self, "root_id", str(self.root_id or ""))
        object.__setattr__(
            self,
            "semantic_role_class",
            str(self.semantic_role_class or "unknown"),
        )
        object.__setattr__(self, "writing_mode", str(self.writing_mode or "vertical"))
        object.__setattr__(
            self,
            "source_evidence_status",
            str(self.source_evidence_status or "unavailable"),
        )
        object.__setattr__(self, "axes", ordered)
        object.__setattr__(
            self,
            "peer_assisted_axes",
            tuple(
                axis
                for axis in PARENT_STYLE_PEER_AXES_V3
                if axis in set(self.peer_assisted_axes)
            ),
        )

    def axis(self, name: str) -> ParentStyleAxisDecisionV3:
        normalized = str(name or "").strip().lower()
        for decision in self.axes:
            if decision.axis == normalized:
                return decision
        raise KeyError(normalized)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "semantic_role_class": self.semantic_role_class,
            "writing_mode": self.writing_mode,
            "source_evidence_status": self.source_evidence_status,
            "peer_assisted_axes": list(self.peer_assisted_axes),
            "axes": {
                decision.axis: decision.to_audit_dict()
                for decision in self.axes
            },
        }


@dataclass(frozen=True)
class ParentStyleDecisionLedgerV3:
    """Order-independent immutable output of the Stage 2C policy core."""

    decisions: tuple[ParentStyleParentDecisionV3, ...]
    version: str = PARENT_STYLE_DECISION_LEDGER_VERSION

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(tuple(self.decisions or ()), key=lambda item: item.bundle_id)
        )
        bundle_ids = tuple(item.bundle_id for item in ordered)
        if not all(bundle_ids) or len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("Stage 2C ledger requires unique non-empty bundle ids")
        object.__setattr__(self, "decisions", ordered)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_style_decision_ledger_version": self.version,
            "decisions": [item.to_audit_dict() for item in self.decisions],
        }


@dataclass(frozen=True)
class ResolvedParentRenderStyleV3:
    """Immutable non-runtime Stage-3B realization for one parent."""

    page_id: str
    bundle_id: str
    parent_id: str
    root_id: str
    source_evidence_status: str
    render_style_confidence: float
    font_family_role: str
    font_weight_tier: str
    primary_font_role: str
    primary_font_role_status: str
    fallback_font_chain_key: str
    source_visual_cell: Mapping[str, Any]
    target_preferred_em_px: float
    target_preferred_em_interval_px: tuple[float, float]
    target_face_profile_id: str
    target_em_conversion_audit: Mapping[str, Any]
    fill: Mapping[str, Any]
    outline: Mapping[str, Any]
    writing_mode: str
    line_height: float
    align: str
    axes: tuple[ParentStyleAxisDecisionV3, ...]
    diagnostic_uncertainty: Mapping[str, Any]
    readability_diagnostic: Mapping[str, Any]
    parent_layer_effects: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered_axes = tuple(
            sorted(
                tuple(self.axes or ()),
                key=lambda item: PARENT_STYLE_DECISION_AXES_V3.index(item.axis),
            )
        )
        if tuple(item.axis for item in ordered_axes) != PARENT_STYLE_DECISION_AXES_V3:
            raise ValueError("Stage 3B style must retain every Stage 2 axis once")
        low, high = tuple(float(item) for item in self.target_preferred_em_interval_px)
        preferred = float(self.target_preferred_em_px)
        if not (
            math.isfinite(low)
            and math.isfinite(high)
            and math.isfinite(preferred)
            and 0.0 < low <= preferred <= high
        ):
            raise ValueError("Stage 3B target em values are invalid")
        object.__setattr__(self, "page_id", str(self.page_id or ""))
        object.__setattr__(self, "bundle_id", str(self.bundle_id or ""))
        object.__setattr__(self, "parent_id", str(self.parent_id or ""))
        object.__setattr__(self, "root_id", str(self.root_id or ""))
        object.__setattr__(
            self,
            "source_evidence_status",
            str(self.source_evidence_status or "unavailable"),
        )
        object.__setattr__(
            self,
            "render_style_confidence",
            float(_unit_interval(self.render_style_confidence) or 0.0),
        )
        object.__setattr__(self, "target_preferred_em_px", preferred)
        object.__setattr__(self, "target_preferred_em_interval_px", (low, high))
        object.__setattr__(self, "axes", ordered_axes)
        for field_name in (
            "source_visual_cell",
            "target_em_conversion_audit",
            "fill",
            "outline",
            "diagnostic_uncertainty",
            "readability_diagnostic",
            "parent_layer_effects",
        ):
            object.__setattr__(
                self,
                field_name,
                _frozen_json_mapping_snapshot(getattr(self, field_name)),
            )

    @property
    def fallback_axes(self) -> tuple[str, ...]:
        return tuple(item.axis for item in self.axes if item.status == "fallback")

    def axis(self, name: str) -> ParentStyleAxisDecisionV3:
        normalized = str(name or "").strip().lower()
        for decision in self.axes:
            if decision.axis == normalized:
                return decision
        raise KeyError(normalized)

    def to_contract_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "render_style_version": PARENT_RENDER_STYLE_V3_VERSION,
            "render_style_owner": "ParentStyleArbitrator",
            "render_style_source": "parent_authorized_style_evidence",
            "render_style_provider": "ParentStyleArbitrator",
            "style_resolution_status": "complete",
            "source_evidence_status": self.source_evidence_status,
            "render_style_confidence": round(
                float(self.render_style_confidence), 8
            ),
            "font_family_role": self.font_family_role,
            "font_weight_tier": self.font_weight_tier,
            "primary_font_role": self.primary_font_role,
            "primary_font_role_status": self.primary_font_role_status,
            "fallback_font_chain_key": self.fallback_font_chain_key,
            "source_visual_cell": _v3_plain_json_value(self.source_visual_cell),
            "target_preferred_em_px": round(
                float(self.target_preferred_em_px), 6
            ),
            "target_preferred_em_interval_px": [
                round(float(item), 6)
                for item in self.target_preferred_em_interval_px
            ],
            "target_face_profile_id": self.target_face_profile_id,
            "target_em_conversion_audit": _v3_plain_json_value(
                self.target_em_conversion_audit
            ),
            "fill": _v3_plain_json_value(self.fill),
            "outline": _v3_plain_json_value(self.outline),
            "writing_mode": self.writing_mode,
            "line_height": float(self.line_height),
            "align": self.align,
            "axis_authority": {
                item.axis: {
                    "status": item.status,
                    "confidence": round(float(item.confidence), 8),
                    "provenance": item.provenance,
                    "reason_codes": list(item.reason_codes),
                }
                for item in self.axes
            },
            "fallback_status": {
                "used": bool(self.fallback_axes),
                "axes": list(self.fallback_axes),
                "reason_codes": (
                    ["deterministic_axis_fallback_applied"]
                    if self.fallback_axes
                    else []
                ),
            },
            "diagnostic_uncertainty": _v3_plain_json_value(
                self.diagnostic_uncertainty
            ),
            "readability_diagnostic": _v3_plain_json_value(
                self.readability_diagnostic
            ),
        }
        if self.parent_layer_effects:
            result["parent_layer_effects"] = _v3_plain_json_value(
                self.parent_layer_effects
            )
        return result

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "render_style": self.to_contract_dict(),
        }


@dataclass(frozen=True)
class ParentRenderStyleLedgerV3:
    """Order-independent non-runtime output of the Stage-3B target realizer."""

    styles: tuple[ResolvedParentRenderStyleV3, ...]
    version: str = PARENT_RENDER_STYLE_LEDGER_V3_VERSION

    def __post_init__(self) -> None:
        ordered = tuple(sorted(tuple(self.styles or ()), key=lambda item: item.bundle_id))
        bundle_ids = tuple(item.bundle_id for item in ordered)
        if not all(bundle_ids) or len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("Stage 3B style ledger requires unique non-empty bundle ids")
        object.__setattr__(self, "styles", ordered)

    def style(self, bundle_id: str) -> ResolvedParentRenderStyleV3:
        normalized = str(bundle_id or "")
        for style in self.styles:
            if style.bundle_id == normalized:
                return style
        raise KeyError(normalized)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_render_style_ledger_version": self.version,
            "styles": [item.to_audit_dict() for item in self.styles],
        }


def realize_parent_render_styles_v3(
    *,
    parent_execution_bundles: Sequence[Any],
    decision_ledger: ParentStyleDecisionLedgerV3,
    font_manager: FontManager,
) -> ParentRenderStyleLedgerV3:
    """Realize v3 styles without mutating bundles or activating runtime v3."""

    if not isinstance(decision_ledger, ParentStyleDecisionLedgerV3):
        raise TypeError("Stage 3B requires a ParentStyleDecisionLedgerV3")
    if not isinstance(font_manager, FontManager):
        raise TypeError("Stage 3B requires FontManager")
    bundles_by_id: dict[str, Any] = {}
    for bundle in tuple(parent_execution_bundles or ()):
        if not bool(getattr(bundle, "render_required", False)):
            continue
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id or bundle_id in bundles_by_id:
            raise ValueError("Stage 3B requires unique non-empty render bundle ids")
        bundles_by_id[bundle_id] = bundle
    decisions_by_id = {item.bundle_id: item for item in decision_ledger.decisions}
    if set(bundles_by_id) != set(decisions_by_id):
        raise ValueError("Stage 3B bundle and decision identities differ")

    inventory = {item.role_id: item for item in font_manager.required_role_inventory()}
    styles: list[ResolvedParentRenderStyleV3] = []
    for bundle_id in sorted(bundles_by_id):
        bundle = bundles_by_id[bundle_id]
        decision = decisions_by_id[bundle_id]
        _v3_require_matching_parent_identity(bundle=bundle, decision=decision)
        styles.append(
            _realize_parent_render_style_v3(
                bundle=bundle,
                decision=decision,
                font_manager=font_manager,
                role_inventory=inventory,
            )
        )
    return ParentRenderStyleLedgerV3(
        styles=_v3_standardize_target_outline_realization(
            styles=tuple(styles),
            decisions_by_id=decisions_by_id,
        )
    )


def _v3_require_matching_parent_identity(
    *,
    bundle: Any,
    decision: ParentStyleParentDecisionV3,
) -> None:
    for field_name in ("page_id", "bundle_id", "parent_id", "root_id"):
        if str(getattr(bundle, field_name, "") or "") != str(
            getattr(decision, field_name, "") or ""
        ):
            raise ValueError(f"Stage 3B parent identity mismatch: {field_name}")


def _realize_parent_render_style_v3(
    *,
    bundle: Any,
    decision: ParentStyleParentDecisionV3,
    font_manager: FontManager,
    role_inventory: Mapping[str, Any],
) -> ResolvedParentRenderStyleV3:
    family_axis = decision.axis("family")
    weight_axis = decision.axis("weight")
    scale_axis = decision.axis("source_scale")
    fill_axis = decision.axis("fill")
    outline_axis = decision.axis("outline")
    orientation_axis = decision.axis("orientation")

    family = str(family_axis.value or "")
    weight = str(weight_axis.value or "")
    weight_support = (
        weight_axis.peer_support
        if isinstance(weight_axis.peer_support, Mapping)
        else {}
    )
    concrete_role_value = weight_support.get("target_font_role")
    if concrete_role_value is not None:
        primary_font_role = str(concrete_role_value or "").strip()
        if primary_font_role not in TARGET_FONT_AFFINITY_ROLE_IDS:
            raise ValueError("Stage 3B target font role is not registered")
        expected_family = _V3_TARGET_FONT_ROLE_FAMILY[primary_font_role]
        expected_weight = _V3_TARGET_FONT_WEIGHT_COMPATIBILITY_ALIAS[
            primary_font_role
        ]
        if family != expected_family or weight != expected_weight:
            raise ValueError(
                "Stage 3B target font role conflicts with compatibility axes"
            )
        numeric_weight = weight_support.get("target_font_numeric_weight")
        if (
            isinstance(numeric_weight, bool)
            or not isinstance(numeric_weight, (int, float))
            or not math.isfinite(float(numeric_weight))
            or int(numeric_weight)
            != _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[primary_font_role]
        ):
            raise ValueError(
                "Stage 3B target font numeric weight conflicts with the "
                "registered role"
            )
        matrix_role_status = "registered_role"
    else:
        matrix_entry = PARENT_RENDER_STYLE_V3_FONT_ROLE_MATRIX.get(
            family, {}
        ).get(weight)
        if matrix_entry is None:
            raise ValueError(
                "Stage 3B family/weight decision is outside the role matrix"
            )
        primary_font_role, matrix_role_status = matrix_entry
    role_status = role_inventory.get(primary_font_role)
    if (
        role_status is None
        or not bool(getattr(role_status, "native_asset_available", False))
        or not str(getattr(role_status, "selected_face_id", "") or "")
    ):
        raise ValueError(f"Stage 3B registered target role is unavailable: {primary_font_role}")
    face = font_manager.face(role_status.selected_face_id)
    if face is None:
        raise ValueError(f"Stage 3B registered target face is unavailable: {primary_font_role}")

    writing_mode = str(orientation_axis.value or decision.writing_mode).strip().lower()
    if writing_mode not in {"vertical", "horizontal"}:
        raise ValueError("Stage 3B writing mode is invalid")
    profile_resolution = font_manager.target_optical_profile(
        face,
        writing_mode,
    )
    profile = profile_resolution.profile
    visible_ratio = float(profile.visible_ink_height_ratio)
    if not math.isfinite(visible_ratio) or visible_ratio <= 0.0:
        raise ValueError("Stage 3B target visible-ink ratio is invalid")

    source_cell, preferred_em, preferred_interval, conversion_status = (
        _v3_target_em_from_source_scale(
            scale_axis=scale_axis,
            writing_mode=writing_mode,
            target_visible_ink_height_ratio=visible_ratio,
        )
    )
    source_optical_support = weight_support.get(
        "source_optical_realization"
    )
    (
        preferred_em,
        preferred_interval,
        source_optical_bridge_audit,
    ) = _v3_apply_source_optical_realization_bridge(
        source_cell=source_cell,
        height_preferred_em=preferred_em,
        height_interval=preferred_interval,
        source_optical_support=(
            source_optical_support
            if isinstance(source_optical_support, Mapping)
            else None
        ),
        target_visible_ink_height_ratio=visible_ratio,
        target_stem_to_ink_ratio=float(profile.stem_to_ink_ratio),
        target_profile_policy_id=str(profile.profile_policy_id),
    )
    fill_value = dict(fill_axis.value) if isinstance(fill_axis.value, Mapping) else {}
    outline_value = (
        dict(outline_axis.value) if isinstance(outline_axis.value, Mapping) else {}
    )
    fill = {
        "color": _hex_color(fill_value.get("color")) or "#000000",
        "polarity": str(fill_value.get("polarity") or "dark"),
    }
    outline_present = outline_value.get("present") is True
    outline_ratio = _bounded_float(
        outline_value.get("source_width_to_cell_ratio"),
        minimum=0.0,
        maximum=1.0,
    )
    if outline_ratio is None:
        outline_ratio = 0.0
    if not outline_present:
        outline_ratio = 0.0
    outline_reference = _v3_target_outline_reference(
        family=family,
        writing_mode=writing_mode,
        font_manager=font_manager,
        role_inventory=role_inventory,
    )
    outline_reference_ratio = float(
        outline_reference["target_outline_reference_ratio"]
    )
    source_requested_outline_width = (
        float(outline_ratio) * preferred_em if outline_present else 0.0
    )
    target_reference_outline_width = (
        outline_reference_ratio * preferred_em if outline_present else 0.0
    )
    direct_wider_source_outline = bool(
        outline_present
        and outline_axis.status == "direct"
        and float(outline_ratio) > outline_reference_ratio + 1e-12
    )
    continuous_target_outline_width = (
        source_requested_outline_width
        if direct_wider_source_outline
        else target_reference_outline_width
    )
    outline = {
        "present": outline_present,
        "color": _hex_color(outline_value.get("color")) or "#FFFFFF",
        "source_width_to_cell_ratio": float(outline_ratio),
        "target_width_px": round(continuous_target_outline_width, 6),
    }

    profile_audit = profile.to_audit_dict()
    selection_audit = profile_resolution.selection.to_audit_dict()
    optical_bridge_applied = (
        source_optical_bridge_audit.get("status") == "applied"
    )
    conversion_audit: dict[str, Any] = {
        "status": conversion_status,
        "formula": (
            "height_if_sufficient_else_sqrt_height_times_stem"
            if optical_bridge_applied
            else (
                "source_visual_cell_px/target_visible_ink_height_ratio"
                if conversion_status
                in {
                    "source_to_target_optical_conversion",
                    "low_confidence_source_to_target_optical_conversion",
                }
                else "deterministic_target_fallback_em"
            )
        ),
        "source_visual_cell_p20_px": source_cell.get("p20_px"),
        "source_visual_cell_median_px": source_cell.get("median_px"),
        "source_visual_cell_p80_px": source_cell.get("p80_px"),
        "target_visible_ink_height_ratio": visible_ratio,
        "target_stem_to_ink_ratio": float(profile.stem_to_ink_ratio),
        "target_face_profile_id": profile.profile_id,
        "target_profile_selection": selection_audit,
        "target_profile_metrics": profile_audit,
        "source_optical_realization": source_optical_bridge_audit,
        "target_outline_realization": {
            "policy_version": (
                PARENT_STYLE_TARGET_OUTLINE_REALIZATION_VERSION
            ),
            "status": (
                "pending_local_thick_source_outline"
                if direct_wider_source_outline
                else (
                    "pending_current_page_standardization"
                    if outline_present
                    else "outline_absent"
                )
            ),
            "reference_role": outline_reference["reference_role"],
            "reference_profile_id": outline_reference[
                "reference_profile_id"
            ],
            "reference_profile_policy_id": outline_reference[
                "reference_profile_policy_id"
            ],
            "target_outline_reference_ratio": outline_reference_ratio,
            "source_requested_width_px": source_requested_outline_width,
            "target_reference_width_px": target_reference_outline_width,
            "continuous_target_width_px": (
                continuous_target_outline_width
            ),
            "eligible_for_current_page_standardization": bool(
                outline_present and not direct_wider_source_outline
            ),
            "source_authority_status": str(outline_axis.status),
            "direct_wider_source_outline": direct_wider_source_outline,
            "cohort_member_bundle_ids": (),
            "quantization": "deterministic_half_up_integer_px",
            "target_width_px": 0.0,
            "paint_authority_changed": False,
            "translation_content_consulted": False,
            "render_admission": False,
        },
        "diagnostic_only": True,
        "render_admission": False,
    }
    if conversion_status == "deterministic_target_fallback":
        conversion_audit["reason"] = "source_visual_cell_unavailable"
    elif conversion_status == (
        "low_confidence_source_to_target_optical_conversion"
    ):
        conversion_audit["reason"] = (
            "local_source_scale_below_direct_confidence_after_peer_reconciliation"
        )

    target_visible_ink_height = preferred_em * visible_ratio
    readability = {
        "status": "diagnostic_only",
        "render_admission": False,
        "target_visible_ink_height_px": round(target_visible_ink_height, 6),
        "target_stem_width_px": round(
            target_visible_ink_height * float(profile.stem_to_ink_ratio), 6
        ),
        "target_advance_px": round(
            preferred_em * float(profile.advance_to_cell_ratio), 6
        ),
        "comparison_owner": "stage3c_visual_quality_audit",
    }
    uncertainty = {
        "status": "diagnostic_only",
        "render_admission": False,
        "source_status": source_cell["status"],
        "target_preferred_em_interval_px": [
            round(preferred_interval[0], 6),
            round(preferred_interval[1], 6),
        ],
    }
    effects = {
        axis_name: _v3_plain_json_value(effect.value)
        for axis_name in ("rotation", "shadow")
        if (effect := decision.axis(axis_name)).value is not None
    }
    confidence_axes = tuple(
        decision.axis(axis)
        for axis in (
            "family",
            "weight",
            "source_scale",
            "fill",
            "outline",
            "orientation",
        )
    )
    confidence = min(float(item.confidence) for item in confidence_axes)
    primary_font_role_status = (
        "fallback_registered_role"
        if family_axis.status == "fallback" or weight_axis.status == "fallback"
        else matrix_role_status
    )
    return ResolvedParentRenderStyleV3(
        page_id=decision.page_id,
        bundle_id=decision.bundle_id,
        parent_id=decision.parent_id,
        root_id=decision.root_id,
        source_evidence_status=decision.source_evidence_status,
        render_style_confidence=confidence,
        font_family_role=family,
        font_weight_tier=weight,
        primary_font_role=primary_font_role,
        primary_font_role_status=primary_font_role_status,
        fallback_font_chain_key=PARENT_STYLE_DEFAULT_FALLBACK_FONT_CHAIN_KEY,
        source_visual_cell=source_cell,
        target_preferred_em_px=preferred_em,
        target_preferred_em_interval_px=preferred_interval,
        target_face_profile_id=profile.profile_id,
        target_em_conversion_audit=conversion_audit,
        fill=fill,
        outline=outline,
        writing_mode=writing_mode,
        line_height=1.0,
        align="center",
        axes=decision.axes,
        diagnostic_uncertainty=uncertainty,
        readability_diagnostic=readability,
        parent_layer_effects=effects,
    )


def _v3_target_outline_reference(
    *,
    family: str,
    writing_mode: str,
    font_manager: FontManager,
    role_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the fixed regular-face carrier reference for one target family."""

    if family not in {"sans", "serif"}:
        raise ValueError("Stage 3B outline reference family is invalid")
    reference_role = f"{family}_regular"
    role_status = role_inventory.get(reference_role)
    if (
        role_status is None
        or not bool(getattr(role_status, "native_asset_available", False))
        or not str(getattr(role_status, "selected_face_id", "") or "")
    ):
        raise ValueError(
            "Stage 3B target outline reference role is unavailable: "
            f"{reference_role}"
        )
    face = font_manager.face(role_status.selected_face_id)
    if face is None:
        raise ValueError(
            "Stage 3B target outline reference face is unavailable: "
            f"{reference_role}"
        )
    profile = font_manager.target_optical_profile(
        face,
        writing_mode,
    ).profile
    reference_ratio = (
        float(profile.visible_ink_height_ratio)
        * float(profile.stem_to_ink_ratio)
    )
    if not math.isfinite(reference_ratio) or reference_ratio <= 0.0:
        raise ValueError("Stage 3B target outline reference ratio is invalid")
    return {
        "reference_role": reference_role,
        "reference_profile_id": str(profile.profile_id),
        "reference_profile_policy_id": str(profile.profile_policy_id),
        "target_outline_reference_ratio": reference_ratio,
    }


def _v3_standardize_target_outline_realization(
    *,
    styles: Sequence[ResolvedParentRenderStyleV3],
    decisions_by_id: Mapping[str, ParentStyleParentDecisionV3],
) -> tuple[ResolvedParentRenderStyleV3, ...]:
    """Quantize compatible already-authorized target outlines exactly once."""

    ordered = tuple(sorted(tuple(styles or ()), key=lambda item: item.bundle_id))
    by_id = {style.bundle_id: style for style in ordered}
    standard_groups: dict[tuple[Any, ...], list[str]] = {}
    thick_local: list[str] = []

    for style in ordered:
        if not bool(style.outline.get("present")):
            continue
        outline_audit = style.target_em_conversion_audit.get(
            "target_outline_realization"
        )
        if not isinstance(outline_audit, Mapping):
            raise ValueError(
                "Stage 3B target outline realization audit is unavailable"
            )
        if not bool(
            outline_audit.get(
                "eligible_for_current_page_standardization"
            )
        ):
            thick_local.append(style.bundle_id)
            continue
        decision = decisions_by_id[style.bundle_id]
        weight_support = decision.axis("weight").peer_support
        component_id = (
            str(weight_support.get("target_font_component_id") or "")
            if isinstance(weight_support, Mapping)
            else ""
        )
        if not component_id:
            component_id = f"parent-local:{style.bundle_id}"
        key = (
            style.page_id,
            component_id,
            style.primary_font_role,
            style.writing_mode,
            str(style.fill.get("color") or ""),
            str(style.fill.get("polarity") or ""),
            str(style.outline.get("color") or ""),
            str(outline_audit.get("reference_profile_id") or ""),
        )
        standard_groups.setdefault(key, []).append(style.bundle_id)

    replacements: dict[str, ResolvedParentRenderStyleV3] = {}
    for member_ids in standard_groups.values():
        for cohort_ids in _v3_outline_scale_cohorts(
            member_ids=tuple(sorted(member_ids)),
            styles_by_id=by_id,
        ):
            continuous_width = max(
                float(
                    by_id[bundle_id]
                    .target_em_conversion_audit[
                        "target_outline_realization"
                    ]["continuous_target_width_px"]
                )
                for bundle_id in cohort_ids
            )
            target_width = float(
                max(1, math.floor(continuous_width + 0.5))
            )
            for bundle_id in cohort_ids:
                replacements[bundle_id] = (
                    _v3_with_final_target_outline_width(
                        style=by_id[bundle_id],
                        target_width=target_width,
                        status=(
                            "standardized_current_page_outline_cohort"
                            if len(cohort_ids) > 1
                            else "local_fixed_target_outline_reference"
                        ),
                        cohort_member_bundle_ids=cohort_ids,
                    )
                )

    for bundle_id in sorted(thick_local):
        style = by_id[bundle_id]
        continuous_width = float(
            style.target_em_conversion_audit[
                "target_outline_realization"
            ]["continuous_target_width_px"]
        )
        replacements[bundle_id] = _v3_with_final_target_outline_width(
            style=style,
            target_width=float(
                max(1, math.floor(continuous_width + 0.5))
            ),
            status="local_direct_thick_source_outline",
            cohort_member_bundle_ids=(bundle_id,),
        )

    for style in ordered:
        if bool(style.outline.get("present")):
            continue
        replacements[style.bundle_id] = (
            _v3_with_final_target_outline_width(
                style=style,
                target_width=0.0,
                status="outline_absent",
                cohort_member_bundle_ids=(),
            )
        )

    return tuple(replacements.get(style.bundle_id, style) for style in ordered)


def _v3_outline_scale_cohorts(
    *,
    member_ids: Sequence[str],
    styles_by_id: Mapping[str, ResolvedParentRenderStyleV3],
) -> tuple[tuple[str, ...], ...]:
    """Partition one paint/component group by a shared realized-em range."""

    ordered = sorted(
        (str(item) for item in member_ids),
        key=lambda bundle_id: (
            float(styles_by_id[bundle_id].target_preferred_em_px),
            bundle_id,
        ),
    )
    cohorts: list[tuple[str, ...]] = []
    active: list[str] = []
    shared_low = 0.0
    shared_high = 0.0
    for bundle_id in ordered:
        interval = styles_by_id[
            bundle_id
        ].target_preferred_em_interval_px
        interval_low, interval_high = (
            float(interval[0]),
            float(interval[1]),
        )
        if not active:
            active = [bundle_id]
            shared_low, shared_high = interval_low, interval_high
            continue
        next_low = max(shared_low, interval_low)
        next_high = min(shared_high, interval_high)
        if next_low <= next_high + 1e-12:
            active.append(bundle_id)
            shared_low, shared_high = next_low, next_high
            continue
        cohorts.append(tuple(sorted(active)))
        active = [bundle_id]
        shared_low, shared_high = interval_low, interval_high
    if active:
        cohorts.append(tuple(sorted(active)))
    return tuple(cohorts)


def _v3_with_final_target_outline_width(
    *,
    style: ResolvedParentRenderStyleV3,
    target_width: float,
    status: str,
    cohort_member_bundle_ids: Sequence[str],
) -> ResolvedParentRenderStyleV3:
    outline = dict(style.outline)
    outline["target_width_px"] = float(target_width)
    conversion_audit = dict(style.target_em_conversion_audit)
    outline_audit = dict(
        conversion_audit.get("target_outline_realization") or {}
    )
    outline_audit.update(
        {
            "status": str(status),
            "cohort_member_bundle_ids": tuple(
                sorted(str(item) for item in cohort_member_bundle_ids)
            ),
            "target_width_px": float(target_width),
        }
    )
    conversion_audit["target_outline_realization"] = outline_audit
    return replace(
        style,
        outline=outline,
        target_em_conversion_audit=conversion_audit,
    )


def _v3_apply_source_optical_realization_bridge(
    *,
    source_cell: Mapping[str, Any],
    height_preferred_em: float,
    height_interval: tuple[float, float],
    source_optical_support: Mapping[str, Any] | None,
    target_visible_ink_height_ratio: float,
    target_stem_to_ink_ratio: float,
    target_profile_policy_id: str,
) -> tuple[float, tuple[float, float], dict[str, Any]]:
    """Apply the one-sided current-page source-to-target optical bridge."""

    preferred = float(height_preferred_em)
    interval = (
        float(height_interval[0]),
        float(height_interval[1]),
    )
    support = (
        dict(source_optical_support)
        if isinstance(source_optical_support, Mapping)
        else {}
    )
    audit: dict[str, Any] = {
        "bridge_version": PARENT_STYLE_OPTICAL_REALIZATION_BRIDGE_VERSION,
        "status": "not_applied",
        "reason": str(
            support.get("reason")
            or "current_page_source_optical_fact_unavailable"
        ),
        "render_admission": False,
        "cache_contributor_count": int(
            support.get("cache_contributor_count") or 0
        ),
        "qualification": str(support.get("qualification") or ""),
        "component_member_bundle_ids": tuple(
            support.get("member_bundle_ids") or ()
        ),
        "source_contributor_bundle_ids": tuple(
            support.get("contributor_bundle_ids") or ()
        ),
        "estimator_conversion_policy": (
            PARENT_STYLE_OPTICAL_ESTIMATOR_POLICY
        ),
        "estimator_conversion": (
            PARENT_STYLE_OPTICAL_ESTIMATOR_CONVERSION
        ),
        "target_profile_policy_id": str(
            target_profile_policy_id or ""
        ),
        "height_preferred_em_px": preferred,
        "height_interval_px": interval,
    }
    if support.get("status") != "qualified":
        return preferred, interval, audit
    if target_profile_policy_id != TARGET_OPTICAL_PROFILE_POLICY_ID:
        audit["reason"] = "target_optical_profile_policy_mismatch"
        return preferred, interval, audit
    try:
        source_cells = tuple(
            float(source_cell[key])
            for key in ("p20_px", "median_px", "p80_px")
        )
        source_stem_ratios = tuple(
            float(support[key]) for key in ("p20", "median", "p80")
        )
        visible_ratio = float(target_visible_ink_height_ratio)
        stem_to_ink = float(target_stem_to_ink_ratio)
    except (KeyError, TypeError, ValueError):
        audit["reason"] = "source_or_target_optical_value_invalid"
        return preferred, interval, audit
    if (
        not all(
            math.isfinite(value)
            for value in (
                *source_cells,
                *source_stem_ratios,
                visible_ratio,
                stem_to_ink,
                preferred,
                *interval,
            )
        )
        or not (
            0.0 < source_cells[0]
            <= source_cells[1]
            <= source_cells[2]
        )
        or not (
            0.0 < source_stem_ratios[0]
            <= source_stem_ratios[1]
            <= source_stem_ratios[2]
            <= 1.0
        )
        or visible_ratio <= 0.0
        or stem_to_ink <= 0.0
        or not (0.0 < interval[0] <= preferred <= interval[1])
    ):
        audit["reason"] = "source_or_target_optical_value_invalid"
        return preferred, interval, audit

    target_stem_per_em = visible_ratio * stem_to_ink
    height_estimates = (interval[0], preferred, interval[1])
    stem_estimates = tuple(
        source_cell_px
        * source_stem_ratio
        * PARENT_STYLE_OPTICAL_ESTIMATOR_CONVERSION
        / target_stem_per_em
        for source_cell_px, source_stem_ratio in zip(
            source_cells,
            source_stem_ratios,
        )
    )
    realized = tuple(
        height_em
        if stem_em <= height_em
        else math.sqrt(height_em * stem_em)
        for height_em, stem_em in zip(
            height_estimates,
            stem_estimates,
        )
    )
    if not (
        0.0 < realized[0] <= realized[1] <= realized[2]
        and all(math.isfinite(value) for value in realized)
    ):
        audit["reason"] = "source_optical_realization_interval_invalid"
        return preferred, interval, audit

    final_preferred = float(realized[1])
    final_interval = (float(realized[0]), float(realized[2]))
    audit.update(
        {
            "status": (
                "applied"
                if final_preferred > preferred
                else "not_applied"
            ),
            "reason": (
                "source_stem_requires_optical_size_raise"
                if final_preferred > preferred
                else "height_baseline_already_sufficient"
            ),
            "source_stem_ratio_p20": source_stem_ratios[0],
            "source_stem_ratio_median": source_stem_ratios[1],
            "source_stem_ratio_p80": source_stem_ratios[2],
            "target_visible_ink_height_ratio": visible_ratio,
            "target_stem_to_ink_ratio": stem_to_ink,
            "target_stem_per_em": target_stem_per_em,
            "stem_estimate_p20_px": stem_estimates[0],
            "stem_estimate_median_px": stem_estimates[1],
            "stem_estimate_p80_px": stem_estimates[2],
            "final_preferred_em_px": final_preferred,
            "final_interval_px": final_interval,
            "formula": (
                "height_if_sufficient_else_sqrt_height_times_stem"
            ),
        }
    )
    return final_preferred, final_interval, audit


def _v3_target_em_from_source_scale(
    *,
    scale_axis: ParentStyleAxisDecisionV3,
    writing_mode: str,
    target_visible_ink_height_ratio: float,
) -> tuple[dict[str, Any], float, tuple[float, float], str]:
    value = dict(scale_axis.value) if isinstance(scale_axis.value, Mapping) else {}
    if scale_axis.status in {"direct", "peer"} or (
        scale_axis.status == "fallback" and value
    ):
        interval = _v3_numeric_interval(
            value,
            keys=("p20_px", "median_px", "p80_px"),
            minimum=1e-8,
            maximum=None,
        )
        if interval is None:
            raise ValueError("Stage 3B resolved source-scale interval is invalid")
        p20, median, p80 = interval
        preferred = float(median) / target_visible_ink_height_ratio
        target_interval = (
            float(p20) / target_visible_ink_height_ratio,
            float(p80) / target_visible_ink_height_ratio,
        )
        source_cell = {
            "status": scale_axis.status,
            "writing_mode": writing_mode,
            "p20_px": float(p20),
            "median_px": float(median),
            "p80_px": float(p80),
            "confidence": float(scale_axis.confidence),
            "authority": scale_axis.status,
            "provenance": scale_axis.provenance,
        }
        return (
            source_cell,
            preferred,
            target_interval,
            (
                "low_confidence_source_to_target_optical_conversion"
                if scale_axis.status == "fallback"
                else "source_to_target_optical_conversion"
            ),
        )
    if scale_axis.status != "fallback" or scale_axis.value is not None:
        raise ValueError("Stage 3B unresolved source scale lacks explicit fallback")
    fallback = float(PARENT_RENDER_STYLE_V3_TARGET_FALLBACK_EM_PX)
    return (
        {
            "status": "unavailable",
            "writing_mode": writing_mode,
            "p20_px": None,
            "median_px": None,
            "p80_px": None,
            "confidence": 0.0,
            "authority": "fallback",
            "provenance": scale_axis.provenance,
        },
        fallback,
        (fallback, fallback),
        "deterministic_target_fallback",
    )


def resolve_parent_style_decision_ledger_v3(
    *,
    parent_execution_bundles: Sequence[Any],
    evidence: Sequence[StyleEvidence],
    style_context_snapshot: Any | None = None,
) -> ParentStyleDecisionLedgerV3:
    """Resolve the immutable non-runtime Stage 2C parent-style ledger.

    Resolution order is binding: calibrated parent-local evidence, compatible
    current-page cohorts, optional qualified prior-page evidence, then
    deterministic axis-local fallback.  Cross-page evidence is never part of
    a current-page cohort.
    """

    contexts = _v3_identity_bound_parent_contexts(
        parent_execution_bundles=parent_execution_bundles,
        evidence=evidence,
    )
    facts_by_bundle = {
        bundle_id: _v3_collect_parent_facts(context)
        for bundle_id, context in contexts.items()
    }
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]] = {
        bundle_id: dict(facts["direct_decisions"])
        for bundle_id, facts in facts_by_bundle.items()
    }

    _v3_apply_current_page_target_font_components(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
        style_context_snapshot=style_context_snapshot,
    )
    _v3_apply_family_axis(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
        style_context_snapshot=style_context_snapshot,
    )
    _v3_apply_weight_axis(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
        style_context_snapshot=style_context_snapshot,
    )
    _v3_apply_source_scale_peer_axis(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )
    _v3_apply_low_confidence_local_source_scale(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )

    parent_decisions: list[ParentStyleParentDecisionV3] = []
    for bundle_id in sorted(facts_by_bundle):
        facts = facts_by_bundle[bundle_id]
        decisions = decisions_by_bundle[bundle_id]
        _v3_apply_axis_fallbacks(facts=facts, decisions=decisions)
        axes = tuple(decisions[axis] for axis in PARENT_STYLE_DECISION_AXES_V3)
        peer_axes = tuple(
            axis
            for axis in PARENT_STYLE_PEER_AXES_V3
            if decisions[axis].status == "peer"
        )
        parent_decisions.append(
            ParentStyleParentDecisionV3(
                page_id=facts["page_id"],
                bundle_id=bundle_id,
                parent_id=facts["parent_id"],
                root_id=facts["root_id"],
                semantic_role_class=facts["semantic_role_class"],
                writing_mode=facts["writing_mode"],
                source_evidence_status=facts["source_evidence_status"],
                axes=axes,
                peer_assisted_axes=peer_axes,
            )
        )
    return ParentStyleDecisionLedgerV3(decisions=tuple(parent_decisions))


def _v3_identity_bound_parent_contexts(
    *,
    parent_execution_bundles: Sequence[Any],
    evidence: Sequence[StyleEvidence],
) -> dict[str, dict[str, Any]]:
    bundles_by_id: dict[str, list[Any]] = {}
    for bundle in tuple(parent_execution_bundles or ()):
        if not bool(getattr(bundle, "render_required", False)):
            continue
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if bundle_id:
            bundles_by_id.setdefault(bundle_id, []).append(bundle)
    evidence_by_id: dict[str, list[StyleEvidence]] = {}
    for item in tuple(evidence or ()):
        if isinstance(item, StyleEvidence) and item.bundle_id:
            evidence_by_id.setdefault(item.bundle_id, []).append(item)

    contexts: dict[str, dict[str, Any]] = {}
    for bundle_id in sorted(bundles_by_id):
        bundles = sorted(bundles_by_id[bundle_id], key=_v3_bundle_identity_sort_key)
        bundle = bundles[0]
        page_id = str(getattr(bundle, "page_id", "") or "")
        parent_id = str(getattr(bundle, "parent_id", "") or "")
        root_id = str(getattr(bundle, "root_id", "") or "")
        reasons: list[str] = []
        if len(bundles) != 1:
            reasons.append("duplicate_parent_execution_bundle_identity")
        candidates = evidence_by_id.get(bundle_id, [])
        if len(candidates) != 1:
            reasons.append(
                "style_evidence_missing"
                if not candidates
                else "duplicate_style_evidence_for_bundle"
            )
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate is not None:
            if candidate.page_id != page_id:
                reasons.append("style_evidence_page_identity_mismatch")
            if candidate.parent_id != parent_id:
                reasons.append("style_evidence_parent_identity_mismatch")
            if candidate.root_id != root_id:
                reasons.append("style_evidence_root_identity_mismatch")
        contexts[bundle_id] = {
            "bundle": bundle,
            "evidence": candidate if not reasons else None,
            "identity_reason_codes": tuple(_unique_strings(reasons)),
        }
    return contexts


def _v3_bundle_identity_sort_key(bundle: Any) -> tuple[str, ...]:
    return tuple(
        str(getattr(bundle, name, "") or "")
        for name in (
            "page_id",
            "parent_id",
            "root_id",
            "role",
            "semantic_class",
            "bundle_id",
        )
    )


def _v3_collect_parent_facts(context: Mapping[str, Any]) -> dict[str, Any]:
    bundle = context["bundle"]
    item = context.get("evidence")
    evidence_item = item if isinstance(item, StyleEvidence) else None
    records = _v3_axis_record_map(bundle=bundle, evidence=evidence_item)
    role_class = _v3_semantic_role_class(bundle)
    identity_reasons = tuple(context.get("identity_reason_codes") or ())

    orientation = _v3_direct_orientation(records.get("orientation"))
    direction_key = (
        "ltr"
        if orientation is not None and orientation.value == "horizontal"
        else "ttb"
    )
    fill = _v3_direct_fill(records.get("fill"))
    outline = _v3_direct_outline(records.get("outline"))
    paint_signature = _v3_paint_signature(fill=fill, outline=outline)
    posterior = _v3_family_posterior_fact(
        records.get("family"), evidence_item
    )
    target_font_affinity = _v3_target_font_affinity_fact(evidence_item)
    scale_fact = _v3_source_scale_fact(
        records.get("scale"), direction=direction_key
    )
    source_advance_grid_fact = _v3_source_advance_grid_fact(
        evidence_item,
        direction=direction_key,
    )
    source_optical_fact = _v3_source_optical_fact(
        records.get("weight"),
        direction=direction_key,
        bundle_id=str(getattr(bundle, "bundle_id", "") or ""),
    )
    weight_fact = _v3_weight_fact(
        records.get("weight"), direction=direction_key
    )
    punctuation_only = _v3_source_text_is_punctuation_only(
        str(getattr(bundle, "source_text", "") or "")
    )

    direct: dict[str, ParentStyleAxisDecisionV3] = {}
    family = (
        None
        if target_font_affinity is not None
        else _v3_direct_family(records.get("family"))
    )
    if family is not None:
        direct["family"] = family
    if orientation is not None:
        direct["orientation"] = orientation
    if fill is not None:
        direct["fill"] = fill
    if outline is not None:
        direct["outline"] = outline
    scale = _v3_direct_source_scale(scale_fact)
    if scale is not None:
        direct["source_scale"] = scale
    weight_interval_ambiguous_heavy_candidate = (
        _v3_weight_is_interval_ambiguous_heavy_candidate(
            weight_fact=weight_fact,
            punctuation_only=punctuation_only,
        )
    )
    weight = (
        None
        if target_font_affinity is not None
        else _v3_direct_weight(
            weight_fact=weight_fact,
            punctuation_only=punctuation_only,
        )
    )
    if weight is not None:
        direct["weight"] = weight
    for axis in ("rotation", "shadow"):
        effect = _v3_direct_effect(axis, records.get(axis))
        if effect is not None:
            direct[axis] = effect

    weight_measurement_reliable = bool(
        weight_fact is not None
        and float(weight_fact["confidence"]) >= V3_DIRECT_WEIGHT_MIN_CONFIDENCE
        and not punctuation_only
    )
    writing_mode = (
        str(orientation.value)
        if orientation is not None
        else "vertical"
    )
    return {
        "page_id": str(getattr(bundle, "page_id", "") or ""),
        "bundle_id": str(getattr(bundle, "bundle_id", "") or ""),
        "parent_id": str(getattr(bundle, "parent_id", "") or ""),
        "root_id": str(getattr(bundle, "root_id", "") or ""),
        "semantic_role_class": role_class,
        "writing_mode": writing_mode,
        "writing_mode_reliable": orientation is not None,
        "source_evidence_status": (
            "observed"
            if evidence_item is not None
            and evidence_item.status == "observed"
            and evidence_item.vote_eligible
            else "unavailable"
        ),
        "identity_reason_codes": identity_reasons,
        "direct_decisions": direct,
        "target_font_affinity": target_font_affinity,
        "family_posterior": posterior,
        "weight_fact": weight_fact,
        "weight_interval": (
            tuple(weight_fact["score_interval"])
            if weight_fact is not None and not punctuation_only
            else None
        ),
        "weight_reliable_unclassified": bool(
            weight_measurement_reliable
            and weight is None
            and not weight_interval_ambiguous_heavy_candidate
        ),
        "weight_interval_ambiguous_heavy_candidate": (
            weight_interval_ambiguous_heavy_candidate
        ),
        "source_scale_fact": scale_fact,
        "source_advance_grid_fact": source_advance_grid_fact,
        "source_optical_fact": source_optical_fact,
        "scale_interval": (
            tuple(scale_fact["interval"])
            if scale_fact is not None
            else None
        ),
        "paint_signature": paint_signature,
        "family_resolution_reasons": (),
    }


def _v3_axis_record_map(
    *,
    bundle: Any,
    evidence: StyleEvidence | None,
) -> dict[str, SourceStyleAxisEvidence]:
    if (
        evidence is None
        or evidence.status != "observed"
        or not evidence.vote_eligible
    ):
        return {}
    grouped: dict[str, list[SourceStyleAxisEvidence]] = {}
    for record in tuple(evidence.axis_evidence or ()):
        if isinstance(record, SourceStyleAxisEvidence):
            grouped.setdefault(record.axis, []).append(record)
    result: dict[str, SourceStyleAxisEvidence] = {}
    for axis in SOURCE_STYLE_AXES:
        records = grouped.get(axis, [])
        if len(records) != 1:
            continue
        record = records[0]
        if not record.provenance:
            continue
        if not _axis_support_identity_matches(
            bundle=bundle,
            evidence=evidence,
            record=record,
        ):
            continue
        result[axis] = record
    return result


def _v3_semantic_role_class(bundle: Any) -> str:
    role = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(bundle, "role", "") or "").strip().lower(),
    ).strip("_")
    semantic = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(bundle, "semantic_class", "") or "").strip().lower(),
    ).strip("_")
    joined = f"{role}_{semantic}"
    if "background" in joined:
        return "background"
    if "narrat" in joined:
        return "narration"
    if "caption" in joined:
        return "caption"
    if any(token in joined for token in ("speech", "bubble", "dialog")):
        return "speech"
    if "sign" in joined:
        return "sign"
    return role or semantic or "unknown"


def _v3_direct_family(
    record: SourceStyleAxisEvidence | None,
) -> ParentStyleAxisDecisionV3 | None:
    if (
        record is None
        or not record.supported
        or record.confidence < V3_DIRECT_FAMILY_MIN_CONFIDENCE
    ):
        return None
    value = dict(record.value)
    font_serif = value.get("font_serif")
    if not isinstance(font_serif, bool):
        return None
    family = "serif" if font_serif else "sans"
    family_role = str(value.get("family_role") or family).strip().lower()
    if family_role not in {"", family}:
        return None
    return _v3_axis_decision(
        axis="family",
        value=family,
        status="direct",
        confidence=record.confidence,
        provenance=record.provenance,
        reason_codes=(*record.reason_codes, "calibrated_local_family_decision"),
    )


def _v3_family_posterior_fact(
    record: SourceStyleAxisEvidence | None,
    evidence: StyleEvidence | None,
) -> dict[str, Any] | None:
    posterior: FontFamilyPosterior | None = None
    variant_agreement = False
    if record is not None:
        value = dict(record.value)
        support = dict(record.support)
        posterior = FontFamilyPosterior.from_mapping(
            value.get("family_posterior")
        ) or FontFamilyPosterior.from_mapping(support.get("family_posterior"))
        variant_agreement = support.get("variant_agreement") is True
    if posterior is None and evidence is not None:
        posterior = evidence.family_posterior
        summary = evidence.detector_variant_summary
        if isinstance(summary, Mapping):
            primary = summary.get("primary")
            neutral = summary.get("neutral")
            if isinstance(primary, Mapping) and isinstance(neutral, Mapping):
                variant_agreement = bool(
                    primary.get("font_serif") is not None
                    and primary.get("font_serif") == neutral.get("font_serif")
                )
    if posterior is None:
        return None
    leading_probability = max(
        posterior.conditional_sans_probability,
        posterior.conditional_serif_probability,
    )
    reliable = bool(
        posterior.leading_family in {"sans", "serif"}
        and posterior.known_mass >= V3_POSTERIOR_KNOWN_MASS_MINIMUM
        and leading_probability >= V3_POSTERIOR_LEADING_PROBABILITY_MINIMUM
        and posterior.margin >= V3_POSTERIOR_MARGIN_MINIMUM
        and variant_agreement
    )
    return {
        "leading_family": posterior.leading_family,
        "leading_probability": float(leading_probability),
        "sans_probability": float(
            posterior.conditional_sans_probability
        ),
        "serif_probability": float(
            posterior.conditional_serif_probability
        ),
        "known_mass": float(posterior.known_mass),
        "margin": float(posterior.margin),
        "variant_agreement": bool(variant_agreement),
        "reliable": reliable,
    }


def _v3_direct_orientation(
    record: SourceStyleAxisEvidence | None,
) -> ParentStyleAxisDecisionV3 | None:
    if (
        record is None
        or not record.supported
        or record.confidence < V3_DIRECT_ORIENTATION_MIN_CONFIDENCE
    ):
        return None
    direction = str(dict(record.value).get("direction") or "").strip().lower()
    writing_mode = {"ttb": "vertical", "ltr": "horizontal"}.get(direction)
    if writing_mode is None:
        return None
    return _v3_axis_decision(
        axis="orientation",
        value=writing_mode,
        status="direct",
        confidence=record.confidence,
        provenance=record.provenance,
        reason_codes=(*record.reason_codes, "parent_local_orientation_decision"),
    )


def _v3_direct_fill(
    record: SourceStyleAxisEvidence | None,
) -> ParentStyleAxisDecisionV3 | None:
    if (
        record is None
        or not record.supported
        or record.confidence < V3_DIRECT_FILL_MIN_CONFIDENCE
    ):
        return None
    value = dict(record.value)
    if str(value.get("schema_version") or "") != "grayscale_core_polarity_v1":
        return None
    observed_color = _hex_color(value.get("color"))
    polarity = str(
        value.get("core_polarity") or value.get("polarity") or ""
    ).strip().lower()
    if not observed_color or polarity not in {"dark", "light"}:
        return None
    target_color = "#000000" if polarity == "dark" else "#FFFFFF"
    return _v3_axis_decision(
        axis="fill",
        value={"color": target_color, "polarity": polarity},
        status="direct",
        confidence=record.confidence,
        provenance=record.provenance,
        reason_codes=(
            *record.reason_codes,
            "grayscale_core_polarity_target_anchor",
            "parent_local_fill_decision",
        ),
    )


def _v3_direct_outline(
    record: SourceStyleAxisEvidence | None,
) -> ParentStyleAxisDecisionV3 | None:
    if (
        record is None
        or not record.supported
        or record.confidence < V3_DIRECT_OUTLINE_MIN_CONFIDENCE
    ):
        return None
    value = dict(record.value)
    if str(value.get("schema_version") or "") != "grayscale_outline_geometry_v1":
        return None
    present = value.get("present")
    observed_color = _hex_color(value.get("color"))
    core_polarity = str(value.get("core_polarity") or "").strip().lower()
    ratio_value = value.get("outline_to_cell_ratio")
    ratio = (
        _bounded_float(
            dict(ratio_value).get("median"), minimum=0.0, maximum=1.0
        )
        if isinstance(ratio_value, Mapping)
        else None
    )
    if not isinstance(present, bool) or not observed_color:
        return None
    if core_polarity not in {"dark", "light"}:
        return None
    if present and ratio is None:
        return None
    if not present:
        ratio = 0.0
    target_color = "#FFFFFF" if core_polarity == "dark" else "#000000"
    return _v3_axis_decision(
        axis="outline",
        value={
            "present": present,
            "color": target_color,
            "source_width_to_cell_ratio": float(ratio or 0.0),
            "core_polarity": core_polarity,
        },
        status="direct",
        confidence=record.confidence,
        provenance=record.provenance,
        reason_codes=(
            *record.reason_codes,
            "grayscale_outline_polarity_target_anchor",
            "parent_local_outline_decision",
        ),
    )


def _v3_paint_signature(
    *,
    fill: ParentStyleAxisDecisionV3 | None,
    outline: ParentStyleAxisDecisionV3 | None,
) -> tuple[str, bool, str] | None:
    if fill is None or outline is None:
        return None
    if not isinstance(fill.value, Mapping) or not isinstance(outline.value, Mapping):
        return None
    polarity = str(fill.value.get("polarity") or "")
    present = outline.value.get("present")
    core_polarity = str(outline.value.get("core_polarity") or "")
    if polarity not in {"dark", "light"} or not isinstance(present, bool):
        return None
    if core_polarity != polarity:
        return None
    return polarity, present, core_polarity


def _v3_source_scale_fact(
    record: SourceStyleAxisEvidence | None,
    *,
    direction: str,
) -> dict[str, Any] | None:
    if record is None or not record.supported:
        return None
    value = dict(record.value)
    if str(value.get("schema_version") or "") != "native_source_cell_distribution_v1":
        return None
    directions = value.get("directions")
    if not isinstance(directions, Mapping):
        return None
    selected = directions.get(direction)
    if not isinstance(selected, Mapping) or str(selected.get("status") or "") != "supported":
        return None
    interval = _v3_numeric_interval(
        selected,
        keys=("cell_p20_px", "cell_median_px", "cell_p80_px"),
        minimum=1e-8,
        maximum=None,
    )
    confidence = _unit_interval(selected.get("confidence"))
    if interval is None or confidence is None:
        return None
    return {
        "direction": direction,
        "interval": interval,
        "confidence": float(confidence),
        "provenance": record.provenance,
        "reason_codes": tuple(record.reason_codes),
    }


def _v3_source_advance_grid_fact(
    evidence: StyleEvidence | None,
    *,
    direction: str,
) -> dict[str, Any] | None:
    """Validate one non-executable cadence fact at the arbitration boundary."""

    if (
        evidence is None
        or evidence.status != "observed"
        or not evidence.vote_eligible
        or not isinstance(evidence.source_advance_grid, SourceAdvanceGridEvidence)
    ):
        return None
    carrier = evidence.source_advance_grid
    if (
        carrier.contract_version != SOURCE_ADVANCE_GRID_VERSION
        or carrier.status != "observed"
    ):
        return None
    identity = carrier.source_identity
    if not isinstance(identity, Mapping):
        return None
    expected = {
        "page_id": evidence.page_id,
        "view_id": evidence.view_id,
        "bundle_id": evidence.bundle_id,
        "parent_id": evidence.parent_id,
        "root_id": evidence.root_id,
        "detector_input_sha256": evidence.detector_input_sha256,
    }
    if any(
        not expected_value
        or str(identity.get(key) or "") != str(expected_value)
        for key, expected_value in expected.items()
    ):
        return None
    if not str(identity.get("authorized_mask_sha256") or ""):
        return None
    cleanup_mask_ids = identity.get("cleanup_mask_ids")
    if cleanup_mask_ids is not None and tuple(cleanup_mask_ids) != tuple(
        evidence.cleanup_mask_ids
    ):
        return None

    record = carrier.direction_record(direction)
    if (
        not isinstance(record, Mapping)
        or str(record.get("status") or "") != "supported"
        or bool(record.get("harmonic_ambiguous"))
        or str(record.get("writing_direction") or "") != direction
        or bool(record.get("executable_source_scale"))
    ):
        return None
    interval = _v3_numeric_interval(
        record,
        keys=("advance_p20_px", "advance_median_px", "advance_p80_px"),
        minimum=1e-8,
        maximum=None,
    )
    confidence = _unit_interval(record.get("confidence"))
    try:
        body_landmark_count = int(record.get("body_landmark_count") or 0)
        qualified_gap_count = int(
            record.get("qualified_adjacent_gap_count") or 0
        )
    except (TypeError, ValueError):
        return None
    spans_raw = record.get("visible_ink_spans_px")
    if not _is_plain_sequence(spans_raw):
        return None
    try:
        spans = tuple(
            float(value)
            for value in spans_raw
            if math.isfinite(float(value)) and float(value) > 0.0
        )
    except (TypeError, ValueError):
        return None
    if (
        interval is None
        or confidence is None
        or body_landmark_count < 3
        or qualified_gap_count < 2
        or not spans
    ):
        return None
    return {
        "direction": direction,
        "interval": interval,
        "confidence": float(confidence),
        "visible_ink_spans_px": spans,
        "body_landmark_count": body_landmark_count,
        "qualified_adjacent_gap_count": qualified_gap_count,
        "harmonic_ambiguous": False,
        "provenance": (
            "authorized_source_style_view:source_advance_grid_relation"
        ),
        "reason_codes": tuple(carrier.reason_codes),
    }


def _v3_direct_source_scale(
    fact: Mapping[str, Any] | None,
) -> ParentStyleAxisDecisionV3 | None:
    if fact is None or float(fact["confidence"]) < V3_DIRECT_SCALE_MIN_CONFIDENCE:
        return None
    p20, median, p80 = [float(value) for value in fact["interval"]]
    return _v3_axis_decision(
        axis="source_scale",
        value={"p20_px": p20, "median_px": median, "p80_px": p80},
        status="direct",
        confidence=float(fact["confidence"]),
        provenance=str(fact["provenance"]),
        reason_codes=(*fact["reason_codes"], "native_source_cell_distribution_resolved"),
    )


def _v3_weight_fact(
    record: SourceStyleAxisEvidence | None,
    *,
    direction: str,
) -> dict[str, Any] | None:
    if record is None or not record.supported:
        return None
    value = dict(record.value)
    if str(value.get("schema_version") or "") != "native_normalized_weight_evidence_v1":
        return None
    directions = value.get("directions")
    if not isinstance(directions, Mapping):
        return None
    selected = directions.get(direction)
    if not isinstance(selected, Mapping) or str(selected.get("status") or "") != "supported":
        return None
    stem = _v3_numeric_interval(
        selected.get("stem_to_cell_ratio"),
        keys=("p20", "median", "p80"),
        minimum=1e-8,
        maximum=1.0,
    )
    ink = _v3_numeric_interval(
        selected.get("ink_occupancy_ratio"),
        keys=("p20", "median", "p80"),
        minimum=1e-8,
        maximum=1.0,
    )
    confidence = _unit_interval(selected.get("confidence"))
    source_cell = _bounded_float(
        selected.get("cell_median_px"), minimum=1e-8, maximum=1e6
    )
    if stem is None or ink is None or confidence is None or source_cell is None:
        return None
    score_interval = tuple(
        math.sqrt(float(stem[index]) * float(ink[index]))
        for index in range(3)
    )
    if not score_interval[0] <= score_interval[1] <= score_interval[2]:
        return None
    result = {
        "direction": direction,
        "score_interval": score_interval,
        "score": float(score_interval[1]),
        "source_cell_median_px": float(source_cell),
        "confidence": float(confidence),
        "provenance": record.provenance,
        "reason_codes": tuple(record.reason_codes),
    }

    alternate_direction = {"ttb": "ltr", "ltr": "ttb"}.get(direction)
    alternate = directions.get(alternate_direction) if alternate_direction else None
    if isinstance(alternate, Mapping) and str(alternate.get("status") or "") == "supported":
        alternate_stem = _v3_numeric_interval(
            alternate.get("stem_to_cell_ratio"),
            keys=("p20", "median", "p80"),
            minimum=1e-8,
            maximum=1.0,
        )
        alternate_ink = _v3_numeric_interval(
            alternate.get("ink_occupancy_ratio"),
            keys=("p20", "median", "p80"),
            minimum=1e-8,
            maximum=1.0,
        )
        alternate_confidence = _unit_interval(alternate.get("confidence"))
        if (
            alternate_stem is not None
            and alternate_ink is not None
            and alternate_confidence is not None
        ):
            alternate_score_interval = tuple(
                math.sqrt(
                    float(alternate_stem[index]) * float(alternate_ink[index])
                )
                for index in range(3)
            )
            direction_neutral_interval = tuple(
                math.sqrt(
                    float(score_interval[index])
                    * float(alternate_score_interval[index])
                )
                for index in range(3)
            )
            if (
                direction_neutral_interval[0]
                <= direction_neutral_interval[1]
                <= direction_neutral_interval[2]
            ):
                result.update(
                    {
                        "direction_neutral_score_interval": (
                            direction_neutral_interval
                        ),
                        "direction_neutral_score": float(
                            direction_neutral_interval[1]
                        ),
                        "direction_neutral_confidence": min(
                            float(confidence), float(alternate_confidence)
                        ),
                    }
                )
    return result


def _v3_weight_tier_for_score(
    *,
    score: float,
) -> str:
    if (
        V3_WEIGHT_SLENDER_SCORE_RANGE[0]
        <= score
        <= V3_WEIGHT_SLENDER_SCORE_RANGE[1]
    ):
        return "slender"
    if V3_WEIGHT_BASE_SCORE_RANGE[0] <= score <= V3_WEIGHT_BASE_SCORE_RANGE[1]:
        return "base"
    if (
        V3_WEIGHT_HEAVY_SCORE_RANGE[0]
        <= score
        <= V3_WEIGHT_HEAVY_SCORE_RANGE[1]
    ):
        return "heavy"
    return ""


def _v3_weight_score_is_transition_gap(score: float) -> bool:
    return bool(
        V3_WEIGHT_SLENDER_SCORE_RANGE[1]
        < score
        < V3_WEIGHT_BASE_SCORE_RANGE[0]
        or V3_WEIGHT_BASE_SCORE_RANGE[1]
        < score
        < V3_WEIGHT_HEAVY_SCORE_RANGE[0]
    )


def _v3_direct_weight_candidate(
    *,
    weight_fact: Mapping[str, Any] | None,
    punctuation_only: bool,
) -> tuple[str, float, tuple[str, ...], tuple[float, float, float]] | None:
    if (
        weight_fact is None
        or float(weight_fact["confidence"]) < V3_DIRECT_WEIGHT_MIN_CONFIDENCE
        or punctuation_only
    ):
        return None
    score = float(weight_fact["score"])
    tier = _v3_weight_tier_for_score(score=score)
    decision_confidence = float(weight_fact["confidence"])
    decision_reason_codes = list(weight_fact["reason_codes"])
    score_interval = tuple(
        float(value) for value in tuple(weight_fact["score_interval"])
    )
    if not tier and _v3_weight_score_is_transition_gap(score):
        direction_neutral_score = _bounded_float(
            weight_fact.get("direction_neutral_score"),
            minimum=1e-8,
            maximum=1.0,
        )
        direction_neutral_confidence = _unit_interval(
            weight_fact.get("direction_neutral_confidence")
        )
        if (
            direction_neutral_score is not None
            and direction_neutral_confidence is not None
            and direction_neutral_confidence >= V3_DIRECT_WEIGHT_MIN_CONFIDENCE
        ):
            tier = _v3_weight_tier_for_score(
                score=float(direction_neutral_score),
            )
            if tier:
                decision_confidence = min(
                    decision_confidence, float(direction_neutral_confidence)
                )
                decision_reason_codes.append(
                    "direction_neutral_transition_gap_resolved"
                )
                direction_neutral_interval = tuple(
                    float(value)
                    for value in tuple(
                        weight_fact.get("direction_neutral_score_interval")
                        or ()
                    )
                )
                if len(direction_neutral_interval) == 3:
                    score_interval = direction_neutral_interval
    if not tier or len(score_interval) != 3:
        return None
    return (
        tier,
        decision_confidence,
        tuple(decision_reason_codes),
        score_interval,
    )


def _v3_weight_has_robust_heavy_support(
    score_interval: Sequence[float],
    *,
    reason_codes: Sequence[str] = (),
) -> bool:
    values = tuple(float(value) for value in score_interval)
    return bool(
        "direction_neutral_transition_gap_resolved" in tuple(reason_codes)
        or (
            len(values) == 3
            and values[0] > V3_WEIGHT_BASE_SCORE_RANGE[1]
        )
    )


def _v3_weight_is_interval_ambiguous_heavy_candidate(
    *,
    weight_fact: Mapping[str, Any] | None,
    punctuation_only: bool,
) -> bool:
    candidate = _v3_direct_weight_candidate(
        weight_fact=weight_fact,
        punctuation_only=punctuation_only,
    )
    return bool(
        candidate is not None
        and candidate[0] == "heavy"
        and not _v3_weight_has_robust_heavy_support(
            candidate[3],
            reason_codes=candidate[2],
        )
    )


def _v3_direct_weight(
    *,
    weight_fact: Mapping[str, Any] | None,
    punctuation_only: bool,
) -> ParentStyleAxisDecisionV3 | None:
    candidate = _v3_direct_weight_candidate(
        weight_fact=weight_fact,
        punctuation_only=punctuation_only,
    )
    if candidate is None or weight_fact is None:
        return None
    tier, decision_confidence, reason_codes, score_interval = candidate
    decision_reason_codes = list(reason_codes)
    if tier == "heavy":
        if not _v3_weight_has_robust_heavy_support(
            score_interval,
            reason_codes=reason_codes,
        ):
            return None
        decision_reason_codes.append("robust_heavy_lower_bound_supported")
    return _v3_axis_decision(
        axis="weight",
        value=tier,
        status="direct",
        confidence=decision_confidence,
        provenance=str(weight_fact["provenance"]),
        reason_codes=(
            *decision_reason_codes,
            "weight_resolution_tier:direct",
            f"calibrated_normalized_weight_region:{tier}",
        ),
    )


def _v3_direct_effect(
    axis: str,
    record: SourceStyleAxisEvidence | None,
) -> ParentStyleAxisDecisionV3 | None:
    if record is None or not record.supported or record.confidence < DIRECT_AXIS_MIN_CONFIDENCE:
        return None
    value, reasons = _validated_perceptual_axis_value(axis, dict(record.value))
    if value is None or reasons:
        return None
    return _v3_axis_decision(
        axis=axis,
        value=value,
        status="direct",
        confidence=record.confidence,
        provenance=record.provenance,
        reason_codes=(*record.reason_codes, f"parent_local_{axis}_decision"),
    )


def _v3_numeric_interval(
    value: Any,
    *,
    keys: tuple[str, str, str],
    minimum: float,
    maximum: float | None,
) -> tuple[float, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    values: list[float] = []
    for key in keys:
        number = _bounded_float(
            value.get(key),
            minimum=minimum,
            maximum=maximum if maximum is not None else 1e12,
        )
        if number is None:
            return None
        values.append(float(number))
    if not values[0] <= values[1] <= values[2]:
        return None
    return values[0], values[1], values[2]


def _v3_source_text_is_punctuation_only(text: str) -> bool:
    visible = [character for character in str(text or "") if not character.isspace()]
    return bool(visible) and not any(
        unicodedata.category(character).startswith(("L", "N"))
        for character in visible
    )


def _v3_target_font_affinity_fact(
    evidence: StyleEvidence | None,
) -> dict[str, Any] | None:
    if (
        evidence is None
        or evidence.status != "observed"
        or not evidence.vote_eligible
    ):
        return None
    observation = evidence.source_font_observation
    if not isinstance(observation, SourceFontObservationV3):
        return None
    affinity = observation.target_font_affinity
    if not isinstance(affinity, TargetFontAffinityObservationV1):
        return None
    if (
        not evidence.detector_input_sha256
        or evidence.detector_input_sha256 != affinity.source_input_sha256
    ):
        return None
    scores: dict[str, float] = {}
    for role_id in TARGET_FONT_AFFINITY_ROLE_IDS:
        try:
            score = float(affinity.role_scores[role_id])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            return None
        scores[role_id] = score
    return {
        "catalog_identity_sha256": affinity.catalog_identity_sha256,
        "descriptor_policy_version": affinity.descriptor_policy_version,
        "model_identity": affinity.model_identity,
        "label_catalog_version": affinity.label_catalog_version,
        "role_scores": scores,
    }


def _v3_target_font_family_probabilities(
    posterior: Any,
) -> tuple[float, float] | None:
    if not isinstance(posterior, Mapping):
        return None
    for sans_key, serif_key in (
        ("sans_probability", "serif_probability"),
        (
            "conditional_sans_probability",
            "conditional_serif_probability",
        ),
        ("sans", "serif"),
        ("sans_mass", "serif_mass"),
    ):
        if sans_key not in posterior or serif_key not in posterior:
            continue
        try:
            sans = float(posterior[sans_key])
            serif = float(posterior[serif_key])
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(sans)
            and math.isfinite(serif)
            and sans >= 0.0
            and serif >= 0.0
        ):
            return sans, serif
    return None


def _v3_target_font_prior_records(
    style_context_snapshot: Any | None,
) -> tuple[dict[str, Any], ...]:
    """Decode only committed, identity-bound target-affinity cache records.

    The cache is deliberately not another component-building input.  It
    transports prior observations that may stabilize the family choice only
    after the current page has formed its own components and numeric weights.
    """

    prefix_page_ids, records = _v3_qualified_prior_snapshot_records(
        style_context_snapshot
    )
    if not prefix_page_ids or not records:
        return ()
    prefix_order = {
        page_id: index for index, page_id in enumerate(prefix_page_ids)
    }
    role_ids = tuple(TARGET_FONT_AFFINITY_ROLE_IDS)
    decoded: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys: set[tuple[str, str]] = set()
    for record in records:
        page_id = str(getattr(record, "page_id", "") or "")
        bundle_id = str(getattr(record, "bundle_id", "") or "")
        key = (page_id, bundle_id)
        if (
            not page_id
            or page_id not in prefix_order
            or not bundle_id
        ):
            continue
        if key in decoded:
            duplicate_keys.add(key)
            continue
        affinity = getattr(record, "target_font_affinity", None)
        if not isinstance(affinity, Mapping):
            continue
        catalog_identity = (
            str(affinity.get("catalog_identity_sha256") or ""),
            str(affinity.get("descriptor_policy_version") or ""),
            str(affinity.get("model_identity") or ""),
            str(affinity.get("label_catalog_version") or ""),
        )
        raw_scores = affinity.get("role_scores")
        if (
            not all(catalog_identity)
            or not isinstance(raw_scores, Mapping)
            or set(raw_scores) != set(role_ids)
        ):
            continue
        try:
            scores = np.asarray(
                [float(raw_scores[role_id]) for role_id in role_ids],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            scores.shape != (len(role_ids),)
            or not np.all(np.isfinite(scores))
            or np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            continue
        leading_role = role_ids[int(np.argmax(scores))]
        family_probabilities = _v3_target_font_family_probabilities(
            getattr(record, "family_posterior", None)
        )
        if (
            family_probabilities is None
            or sum(family_probabilities) <= 0.0
        ):
            leading_family = _V3_TARGET_FONT_ROLE_FAMILY[leading_role]
            family_probabilities = (
                (1.0, 0.0)
                if leading_family == "sans"
                else (0.0, 1.0)
            )
        else:
            family_total = float(sum(family_probabilities))
            family_probabilities = (
                float(family_probabilities[0]) / family_total,
                float(family_probabilities[1]) / family_total,
            )
        decoded[key] = {
            "page_id": page_id,
            "bundle_id": bundle_id,
            "catalog_identity": catalog_identity,
            "role_scores": scores,
            "leading_role": leading_role,
            "numeric_weight": _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[
                leading_role
            ],
            "family_probabilities": family_probabilities,
        }
    for key in duplicate_keys:
        decoded.pop(key, None)
    return tuple(
        decoded[key]
        for key in sorted(
            decoded,
            key=lambda item: (prefix_order[item[0]], item[1]),
        )
    )


def _v3_component_source_optical_fact(
    member_bundle_ids: Sequence[str],
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one current-page optical fact without changing components."""

    member_ids = tuple(
        sorted(str(bundle_id or "") for bundle_id in member_bundle_ids)
    )
    contributors: list[dict[str, Any]] = []
    for bundle_id in member_ids:
        facts = facts_by_bundle.get(bundle_id)
        fact = (
            facts.get("source_optical_fact")
            if isinstance(facts, Mapping)
            else None
        )
        if not isinstance(fact, Mapping):
            continue
        if (
            str(fact.get("bundle_id") or "") != bundle_id
            or str(fact.get("direction") or "") not in {"ttb", "ltr"}
        ):
            continue
        try:
            p20 = float(fact.get("p20"))
            median = float(fact.get("median"))
            p80 = float(fact.get("p80"))
            expanded = tuple(
                float(value)
                for value in fact.get("expanded_interval") or ()
            )
            confidence = float(fact.get("measurement_confidence"))
        except (TypeError, ValueError):
            continue
        if (
            len(expanded) != 2
            or not all(
                math.isfinite(value)
                for value in (
                    p20,
                    median,
                    p80,
                    expanded[0],
                    expanded[1],
                    confidence,
                )
            )
            or not (0.0 < p20 <= median <= p80 <= 1.0)
            or not (0.0 < expanded[0] <= expanded[1] <= 1.0)
            or not (0.0 <= confidence <= 1.0)
        ):
            continue
        contributors.append(dict(fact))

    base = {
        "bridge_version": PARENT_STYLE_OPTICAL_REALIZATION_BRIDGE_VERSION,
        "status": "unavailable",
        "render_admission": False,
        "member_bundle_ids": member_ids,
        "contributor_bundle_ids": tuple(
            str(fact["bundle_id"]) for fact in contributors
        ),
        "contributor_count": len(contributors),
        "cache_contributor_count": 0,
    }
    if not contributors:
        return {
            **base,
            "reason": "current_page_source_optical_fact_unavailable",
        }

    directions = {
        str(fact.get("direction") or "") for fact in contributors
    }
    if len(directions) != 1:
        return {
            **base,
            "reason": "current_page_source_optical_direction_mismatch",
        }

    if len(contributors) == 1:
        if contributors[0].get("singleton_authoritative") is not True:
            return {
                **base,
                "reason": (
                    "singleton_source_optical_fact_not_authoritative"
                ),
            }
        qualification = "supported_corroborating_singleton"
        consensus_interval = tuple(
            float(value)
            for value in contributors[0]["expanded_interval"]
        )
    else:
        consensus_interval = (
            max(
                float(fact["expanded_interval"][0])
                for fact in contributors
            ),
            min(
                float(fact["expanded_interval"][1])
                for fact in contributors
            ),
        )
        if consensus_interval[0] > consensus_interval[1]:
            return {
                **base,
                "reason": (
                    "current_page_source_optical_intervals_disjoint"
                ),
                "consensus_interval": consensus_interval,
            }
        qualification = "current_page_component_interval_consensus"

    p20 = float(np.median([fact["p20"] for fact in contributors]))
    median = float(
        np.median([fact["median"] for fact in contributors])
    )
    p80 = float(np.median([fact["p80"] for fact in contributors]))
    return {
        **base,
        "status": "qualified",
        "reason": "current_page_source_optical_fact_qualified",
        "qualification": qualification,
        "direction": next(iter(directions)),
        "p20": p20,
        "median": median,
        "p80": p80,
        "interval": (p20, p80),
        "consensus_interval": consensus_interval,
        "measurement_states": tuple(
            str(fact.get("measurement_state") or "")
            for fact in contributors
        ),
        "estimator_agreements": tuple(
            str(fact.get("estimator_agreement") or "")
            for fact in contributors
        ),
        "source_identity_sha256": tuple(
            str(fact.get("source_identity_sha256") or "")
            for fact in contributors
        ),
        "measurement_confidence": min(
            float(fact["measurement_confidence"])
            for fact in contributors
        ),
    }


def _v3_target_font_components(
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    *,
    style_context_snapshot: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Resolve target roles from only current-page Yuzu affinity geometry.

    Component membership deliberately excludes semantic role, text, position,
    paint, source scale, fit state, and cache state.  Identity-incompatible
    affinity catalogs are resolved independently.
    """

    role_ids = tuple(TARGET_FONT_AFFINITY_ROLE_IDS)
    grouped: dict[tuple[str, str, str, str, str], list[str]] = {}
    valid_scores: dict[str, np.ndarray] = {}
    for bundle_id in sorted(facts_by_bundle):
        facts = facts_by_bundle[bundle_id]
        affinity = facts.get("target_font_affinity")
        if not isinstance(affinity, Mapping):
            continue
        raw_scores = affinity.get("role_scores")
        if not isinstance(raw_scores, Mapping):
            continue
        try:
            scores = np.asarray(
                [float(raw_scores[role_id]) for role_id in role_ids],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            scores.shape != (len(role_ids),)
            or not np.all(np.isfinite(scores))
            or np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            continue
        identity = (
            str(facts.get("page_id") or ""),
            str(affinity.get("catalog_identity_sha256") or ""),
            str(affinity.get("descriptor_policy_version") or ""),
            str(affinity.get("model_identity") or ""),
            str(affinity.get("label_catalog_version") or ""),
        )
        if not all(identity):
            continue
        grouped.setdefault(identity, []).append(bundle_id)
        valid_scores[bundle_id] = scores

    prior_records = _v3_target_font_prior_records(style_context_snapshot)
    resolved: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        member_ids = tuple(sorted(grouped[identity]))
        affinities = np.stack(
            [valid_scores[bundle_id] for bundle_id in member_ids],
            axis=0,
        )
        centered = affinities - affinities.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        descriptors = centered / np.maximum(norms, 1e-12)
        leading_roles = {
            bundle_id: role_ids[int(np.argmax(affinities[index]))]
            for index, bundle_id in enumerate(member_ids)
        }
        index_by_bundle = {
            bundle_id: index
            for index, bundle_id in enumerate(member_ids)
        }
        nearest: dict[str, set[str]] = {}
        for bundle_id in member_ids:
            left_index = index_by_bundle[bundle_id]
            ranked = sorted(
                (
                    (
                        float(
                            np.dot(
                                descriptors[left_index],
                                descriptors[index_by_bundle[other_id]],
                            )
                        ),
                        other_id,
                    )
                    for other_id in member_ids
                    if other_id != bundle_id
                ),
                key=lambda item: (-item[0], item[1]),
            )
            nearest[bundle_id] = {
                other_id
                for similarity, other_id in ranked[
                    :_V3_TARGET_FONT_NEIGHBOR_COUNT
                ]
                if similarity > 0.0
            }

        parent = {bundle_id: bundle_id for bundle_id in member_ids}

        def find(bundle_id: str) -> str:
            while parent[bundle_id] != bundle_id:
                parent[bundle_id] = parent[parent[bundle_id]]
                bundle_id = parent[bundle_id]
            return bundle_id

        def union(left_id: str, right_id: str) -> None:
            left_root = find(left_id)
            right_root = find(right_id)
            if left_root != right_root:
                parent[right_root] = left_root

        for offset, left_id in enumerate(member_ids):
            for right_id in member_ids[offset + 1 :]:
                same_numeric_weight = (
                    _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[
                        leading_roles[left_id]
                    ]
                    == _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[
                        leading_roles[right_id]
                    ]
                )
                reciprocal_neighbor = (
                    right_id in nearest[left_id]
                    and left_id in nearest[right_id]
                )
                if same_numeric_weight or reciprocal_neighbor:
                    union(left_id, right_id)

        components_by_root: dict[str, list[str]] = {}
        for bundle_id in member_ids:
            components_by_root.setdefault(find(bundle_id), []).append(
                bundle_id
            )
        components = sorted(
            (
                tuple(sorted(component_ids))
                for component_ids in components_by_root.values()
            ),
            key=lambda component_ids: component_ids[0],
        )
        for component_ids in components:
            component_indices = [
                index_by_bundle[bundle_id]
                for bundle_id in component_ids
            ]
            current_pooled = affinities[component_indices].mean(axis=0)
            current_family_scores = {"sans": 0.0, "serif": 0.0}
            for bundle_id in component_ids:
                family_probabilities = (
                    _v3_target_font_family_probabilities(
                        facts_by_bundle[bundle_id].get(
                            "family_posterior"
                        )
                    )
                )
                if family_probabilities is None:
                    current_family_scores[
                        _V3_TARGET_FONT_ROLE_FAMILY[
                            leading_roles[bundle_id]
                        ]
                    ] += 1.0
                else:
                    current_family_scores["sans"] += family_probabilities[0]
                    current_family_scores["serif"] += family_probabilities[1]
            current_family = max(
                ("sans", "serif"),
                key=lambda item: (current_family_scores[item], item),
            )
            current_role = max(
                _V3_TARGET_FONT_FAMILY_ROLES[current_family],
                key=lambda item: (
                    float(current_pooled[role_ids.index(item)]),
                    -_V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[item],
                    item,
                ),
            )
            numeric_weight = _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[
                current_role
            ]
            compatible_prior = tuple(
                record
                for record in prior_records
                if (
                    tuple(record["catalog_identity"])
                    == identity[1:]
                    and int(record["numeric_weight"]) == numeric_weight
                )
            )
            prior_effective_weight = min(
                len(compatible_prior),
                len(component_ids),
            )
            pooled = current_pooled
            family_scores = dict(current_family_scores)
            family = current_family
            role = current_role
            if prior_effective_weight > 0:
                prior_role_mean = np.stack(
                    [
                        record["role_scores"]
                        for record in compatible_prior
                    ],
                    axis=0,
                ).mean(axis=0)
                current_weight = float(len(component_ids))
                prior_weight = float(prior_effective_weight)
                pooled = (
                    current_pooled * current_weight
                    + prior_role_mean * prior_weight
                ) / (current_weight + prior_weight)
                prior_family_mean = np.asarray(
                    [
                        (
                            float(record["family_probabilities"][0]),
                            float(record["family_probabilities"][1]),
                        )
                        for record in compatible_prior
                    ],
                    dtype=np.float64,
                ).mean(axis=0)
                family_scores = {
                    "sans": (
                        float(current_family_scores["sans"])
                        + float(prior_family_mean[0]) * prior_weight
                    ),
                    "serif": (
                        float(current_family_scores["serif"])
                        + float(prior_family_mean[1]) * prior_weight
                    ),
                }
                candidate_family = max(
                    ("sans", "serif"),
                    key=lambda item: (family_scores[item], item),
                )
                same_weight_roles = tuple(
                    role_id
                    for role_id in _V3_TARGET_FONT_FAMILY_ROLES[
                        candidate_family
                    ]
                    if _V3_TARGET_FONT_ROLE_NUMERIC_WEIGHT[role_id]
                    == numeric_weight
                )
                if same_weight_roles:
                    family = candidate_family
                    role = max(
                        same_weight_roles,
                        key=lambda item: (
                            float(pooled[role_ids.index(item)]),
                            item,
                        ),
                    )
            component_identity = {
                "page_id": identity[0],
                "catalog_identity_sha256": identity[1],
                "descriptor_policy_version": identity[2],
                "model_identity": identity[3],
                "label_catalog_version": identity[4],
                "member_bundle_ids": component_ids,
            }
            component_digest = hashlib.sha256(
                json.dumps(
                    component_identity,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            source_optical_realization = (
                _v3_component_source_optical_fact(
                    component_ids,
                    facts_by_bundle,
                )
            )
            resolved.append(
                {
                    "page_id": identity[0],
                    "member_bundle_ids": component_ids,
                    "target_font_component_id": (
                        f"target-font-component-v1:{component_digest}"
                    ),
                    "target_font_family": family,
                    "target_font_role": role,
                    "target_font_numeric_weight": numeric_weight,
                    "current_page_target_font_family": current_family,
                    "current_page_target_font_role": current_role,
                    "current_page_pooled_role_affinities": {
                        role_id: float(current_pooled[index])
                        for index, role_id in enumerate(role_ids)
                    },
                    "current_page_pooled_family_scores": (
                        current_family_scores
                    ),
                    "pooled_role_affinities": {
                        role_id: float(pooled[index])
                        for index, role_id in enumerate(role_ids)
                    },
                    "pooled_family_scores": family_scores,
                    "prior_cache_compatible_record_count": len(
                        compatible_prior
                    ),
                    "prior_cache_effective_weight": prior_effective_weight,
                    "prior_cache_donor_bundle_ids": tuple(
                        str(record["bundle_id"])
                        for record in compatible_prior
                    ),
                    "prior_cache_donor_page_ids": tuple(
                        sorted(
                            {
                                str(record["page_id"])
                                for record in compatible_prior
                            }
                        )
                    ),
                    "prior_cache_snapshot_id": (
                        str(
                            getattr(
                                style_context_snapshot,
                                "snapshot_id",
                                "",
                            )
                            or ""
                        )
                        if prior_effective_weight > 0
                        else ""
                    ),
                    "source_optical_realization": (
                        source_optical_realization
                    ),
                }
            )
    return tuple(resolved)


def _v3_apply_current_page_target_font_components(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
    style_context_snapshot: Any | None = None,
) -> None:
    for component in _v3_target_font_components(
        facts_by_bundle,
        style_context_snapshot=style_context_snapshot,
    ):
        member_ids = tuple(component["member_bundle_ids"])
        family = str(component["target_font_family"])
        role = str(component["target_font_role"])
        numeric_weight = int(component["target_font_numeric_weight"])
        pooled_role_affinities = dict(
            component["pooled_role_affinities"]
        )
        pooled_family_scores = dict(component["pooled_family_scores"])
        component_support = {
            "target_font_component_id": str(
                component["target_font_component_id"]
            ),
            "target_font_role": role,
            "target_font_numeric_weight": numeric_weight,
            "member_bundle_ids": member_ids,
            "pooled_role_affinities": pooled_role_affinities,
            "pooled_family_scores": pooled_family_scores,
            "current_page_target_font_family": str(
                component["current_page_target_font_family"]
            ),
            "current_page_target_font_role": str(
                component["current_page_target_font_role"]
            ),
            "current_page_pooled_role_affinities": dict(
                component["current_page_pooled_role_affinities"]
            ),
            "current_page_pooled_family_scores": dict(
                component["current_page_pooled_family_scores"]
            ),
            "prior_cache_compatible_record_count": int(
                component["prior_cache_compatible_record_count"]
            ),
            "prior_cache_effective_weight": int(
                component["prior_cache_effective_weight"]
            ),
            "prior_cache_donor_bundle_ids": tuple(
                component["prior_cache_donor_bundle_ids"]
            ),
            "prior_cache_donor_page_ids": tuple(
                component["prior_cache_donor_page_ids"]
            ),
            "prior_cache_snapshot_id": str(
                component["prior_cache_snapshot_id"]
            ),
            "source_optical_realization": dict(
                component["source_optical_realization"]
            ),
        }
        family_total = float(sum(pooled_family_scores.values()))
        family_confidence = (
            float(pooled_family_scores[family]) / family_total
            if family_total > 0.0
            else 0.0
        )
        role_confidence = float(pooled_role_affinities[role])
        for bundle_id in member_ids:
            decisions = decisions_by_bundle[bundle_id]
            decisions["family"] = _v3_axis_decision(
                axis="family",
                value=family,
                status="peer",
                confidence=family_confidence,
                provenance=(
                    "parent_style_arbitrator_v3:"
                    "current_page_target_font_component"
                ),
                reason_codes=(
                    "typed_yuzu_target_font_affinity",
                    "component_pooled_target_family",
                    *(
                        ("compatible_prior_prefix_target_affinity",)
                        if int(
                            component[
                                "prior_cache_effective_weight"
                            ]
                        )
                        > 0
                        else ()
                    ),
                ),
                peer_support=component_support,
            )
            decisions["weight"] = _v3_axis_decision(
                axis="weight",
                value=_V3_TARGET_FONT_WEIGHT_COMPATIBILITY_ALIAS[role],
                status="peer",
                confidence=role_confidence,
                provenance=(
                    "parent_style_arbitrator_v3:"
                    "current_page_target_font_component"
                ),
                reason_codes=(
                    "typed_yuzu_target_font_affinity",
                    "component_pooled_registered_target_role",
                    f"target_font_role:{role}",
                    *(
                        ("compatible_prior_prefix_target_affinity",)
                        if int(
                            component[
                                "prior_cache_effective_weight"
                            ]
                        )
                        > 0
                        else ()
                    ),
                ),
                peer_support=component_support,
            )


def _v3_apply_family_axis(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
    style_context_snapshot: Any | None,
) -> None:
    """Resolve only family, in the architecture-defined evidence order."""

    _v3_apply_current_page_family_cohorts(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )
    _v3_apply_prior_page_family_cache(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
        style_context_snapshot=style_context_snapshot,
    )


def _v3_apply_current_page_family_cohorts(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
) -> None:
    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for target_id in sorted(facts_by_bundle):
        if "family" in decisions_by_bundle[target_id]:
            continue
        target = facts_by_bundle[target_id]
        target_candidate = _v3_family_candidate_fact(
            facts=target,
            decision=decisions_by_bundle[target_id].get("family"),
        )
        candidates: list[tuple[str, dict[str, Any]]] = []
        for member_id in sorted(facts_by_bundle):
            member = facts_by_bundle[member_id]
            if member["page_id"] != target["page_id"]:
                continue
            candidate = _v3_family_candidate_fact(
                facts=member,
                decision=decisions_by_bundle[member_id].get("family"),
            )
            if candidate is None:
                continue
            if member_id != target_id and not _v3_family_facts_are_compatible(
                target,
                member,
            ):
                continue
            candidates.append((member_id, candidate))

        values = {str(candidate["value"]) for _, candidate in candidates}
        if len(values) > 1:
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "family_candidate_cohort_conflict",
            )
            continue
        if not candidates:
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "family_candidate_cohort_insufficient",
            )
            continue

        cluster_ids = _v3_maximum_pairwise_family_cluster(
            candidate_ids=[member_id for member_id, _ in candidates],
            facts_by_id=facts_by_bundle,
        )
        candidate_by_id = dict(candidates)
        if target_candidate is not None:
            eligible = target_id in cluster_ids and len(cluster_ids) >= 2
        else:
            eligible = bool(
                len(cluster_ids) >= PEER_MINIMUM_DONOR_COUNT
                and all(
                    candidate_by_id[member_id]["source"] == "direct"
                    for member_id in cluster_ids
                )
            )
        if not eligible:
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "family_candidate_cohort_insufficient",
            )
            continue

        selected = [candidate_by_id[member_id] for member_id in cluster_ids]
        value = str(selected[0]["value"])
        if any(str(item["value"]) != value for item in selected):
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "family_candidate_cohort_conflict",
            )
            continue
        updates[target_id] = _v3_axis_decision(
            axis="family",
            value=value,
            status="peer",
            confidence=float(np.mean([item["confidence"] for item in selected])),
            provenance=(
                "parent_style_arbitrator_v3:"
                "current_page_family_candidate_consensus"
            ),
            reason_codes=("current_page_family_candidate_consensus",),
            peer_support={
                "evidence_source": "current_page_cohort",
                "page_id": str(target["page_id"]),
                "member_bundle_ids": cluster_ids,
                "member_count": len(cluster_ids),
                "member_sources": {
                    member_id: str(candidate_by_id[member_id]["source"])
                    for member_id in cluster_ids
                },
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["family"] = decision


def _v3_family_candidate_fact(
    *,
    facts: Mapping[str, Any],
    decision: ParentStyleAxisDecisionV3 | None,
) -> dict[str, Any] | None:
    if (
        decision is not None
        and decision.status == "direct"
        and decision.value in {"sans", "serif"}
        and decision.confidence >= V3_PEER_DONOR_MIN_CONFIDENCE
    ):
        return {
            "value": str(decision.value),
            "confidence": float(decision.confidence),
            "source": "direct",
        }
    posterior = facts.get("family_posterior")
    if not isinstance(posterior, Mapping) or not bool(posterior.get("reliable")):
        return None
    value = str(posterior.get("leading_family") or "")
    confidence = _unit_interval(posterior.get("leading_probability"))
    if value not in {"sans", "serif"} or confidence is None:
        return None
    return {
        "value": value,
        "confidence": float(confidence),
        "source": "unpromoted_candidate",
    }


def _v3_family_facts_are_compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    if (
        first["semantic_role_class"] != second["semantic_role_class"]
        or not bool(first["writing_mode_reliable"])
        or not bool(second["writing_mode_reliable"])
        or first["writing_mode"] != second["writing_mode"]
        or first["paint_signature"] is None
        or second["paint_signature"] is None
        or first["paint_signature"] != second["paint_signature"]
        or _v3_has_fragmented_cell_population_scale_fact(first)
        or _v3_has_fragmented_cell_population_scale_fact(second)
    ):
        return False
    return bool(
        _v3_intervals_are_compatible(
            first.get("weight_interval"),
            second.get("weight_interval"),
        )
        and _v3_intervals_are_compatible(
            first.get("scale_interval"),
            second.get("scale_interval"),
        )
    )


def _v3_maximum_pairwise_family_cluster(
    *,
    candidate_ids: Sequence[str],
    facts_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    ordered = sorted(set(str(item) for item in candidate_ids if item))
    clusters: list[tuple[str, ...]] = []
    for seed in ordered:
        cluster = [seed]
        for candidate in ordered:
            if candidate == seed:
                continue
            if all(
                _v3_family_facts_are_compatible(
                    facts_by_id[member],
                    facts_by_id[candidate],
                )
                for member in cluster
            ):
                cluster.append(candidate)
        clusters.append(tuple(sorted(cluster)))
    if not clusters:
        return []
    maximum = max(len(cluster) for cluster in clusters)
    return list(sorted(set(cluster for cluster in clusters if len(cluster) == maximum))[0])


def _v3_add_family_resolution_reason(
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    bundle_id: str,
    reason: str,
) -> None:
    facts = facts_by_bundle.get(bundle_id)
    if not isinstance(facts, dict):
        return
    facts["family_resolution_reasons"] = tuple(
        _unique_strings(
            [
                *tuple(facts.get("family_resolution_reasons") or ()),
                str(reason or ""),
            ]
        )
    )


def _v3_apply_prior_page_family_cache(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
    style_context_snapshot: Any | None,
) -> None:
    prefix_page_ids = tuple(
        str(item)
        for item in tuple(
            getattr(style_context_snapshot, "prefix_page_ids", ()) or ()
        )
        if str(item)
    )
    records = tuple(getattr(style_context_snapshot, "records", ()) or ())
    if not prefix_page_ids or not records:
        return
    prefix = set(prefix_page_ids)
    cache_facts: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    for record in records:
        page_id = str(getattr(record, "page_id", "") or "")
        bundle_id = str(getattr(record, "bundle_id", "") or "")
        if not page_id or page_id not in prefix or not bundle_id:
            continue
        key = f"prior::{page_id}::{bundle_id}"
        if key in cache_facts:
            duplicate_keys.add(key)
            continue
        facts = _v3_cache_family_facts(record)
        if facts is not None:
            cache_facts[key] = facts
    for key in duplicate_keys:
        cache_facts.pop(key, None)
    if not cache_facts:
        return

    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for target_id in sorted(facts_by_bundle):
        if "family" in decisions_by_bundle[target_id]:
            continue
        target = facts_by_bundle[target_id]
        donor_ids = [
            donor_id
            for donor_id in sorted(cache_facts)
            if _v3_family_facts_are_compatible(target, cache_facts[donor_id])
        ]
        if len(donor_ids) < PEER_MINIMUM_DONOR_COUNT:
            continue
        cluster_ids = _v3_maximum_pairwise_family_cluster(
            candidate_ids=donor_ids,
            facts_by_id=cache_facts,
        )
        if len(cluster_ids) < PEER_MINIMUM_DONOR_COUNT:
            continue
        values = {
            str(cache_facts[donor_id]["family_decision"].value)
            for donor_id in cluster_ids
        }
        if len(values) != 1:
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "prior_page_family_cache_conflict",
            )
            continue
        value = next(iter(values))
        posterior = target.get("family_posterior")
        if (
            isinstance(posterior, Mapping)
            and bool(posterior.get("reliable"))
            and str(posterior.get("leading_family") or "") != value
        ):
            _v3_add_family_resolution_reason(
                facts_by_bundle,
                target_id,
                "prior_page_family_cache_conflicts_with_local_candidate",
            )
            continue
        donor_decisions = [
            cache_facts[donor_id]["family_decision"]
            for donor_id in cluster_ids
        ]
        updates[target_id] = _v3_axis_decision(
            axis="family",
            value=value,
            status="peer",
            confidence=float(
                np.mean([decision.confidence for decision in donor_decisions])
            ),
            provenance=(
                "parent_style_arbitrator_v3:"
                "prior_page_family_cache_consensus"
            ),
            reason_codes=("prior_page_family_cache_consensus",),
            peer_support={
                "evidence_source": "prior_page_cache",
                "donor_bundle_ids": [
                    str(cache_facts[donor_id]["bundle_id"])
                    for donor_id in cluster_ids
                ],
                "donor_page_ids": sorted(
                    {
                        str(cache_facts[donor_id]["page_id"])
                        for donor_id in cluster_ids
                    }
                ),
                "donor_count": len(cluster_ids),
                "snapshot_id": str(
                    getattr(style_context_snapshot, "snapshot_id", "") or ""
                ),
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["family"] = decision


def _v3_cache_family_facts(
    record: Any,
    *,
    require_paint: bool = True,
) -> dict[str, Any] | None:
    axis_items = tuple(getattr(record, "assist_axes", ()) or ()) + tuple(
        getattr(record, "compatibility_axes", ()) or ()
    )
    grouped: dict[str, list[Any]] = {}
    for item in axis_items:
        axis = str(getattr(item, "axis", "") or "").strip().lower()
        if axis:
            grouped.setdefault(axis, []).append(item)
    required_axes = ("family", "weight", "scale", "orientation")
    if require_paint:
        required_axes = (*required_axes, "fill", "outline")
    if any(
        len(grouped.get(axis, ())) != 1
        for axis in required_axes
    ):
        return None

    def evidence(axis: str) -> SourceStyleAxisEvidence:
        item = grouped[axis][0]
        return SourceStyleAxisEvidence(
            axis=axis,
            status="supported",
            value=getattr(item, "value", {}) or {},
            confidence=float(getattr(item, "confidence", 0.0) or 0.0),
            provenance=str(getattr(item, "provenance", "") or ""),
            support_identity={
                "qualified_prior_page_support_identity_sha256": str(
                    getattr(item, "support_identity_sha256", "") or ""
                )
            },
            reason_codes=("qualified_prior_page_cache_axis",),
        )

    orientation = _v3_direct_orientation(evidence("orientation"))
    if orientation is None:
        return None
    direction = "ltr" if orientation.value == "horizontal" else "ttb"
    fill = (
        _v3_direct_fill(evidence("fill"))
        if len(grouped.get("fill", ())) == 1
        else None
    )
    outline = (
        _v3_direct_outline(evidence("outline"))
        if len(grouped.get("outline", ())) == 1
        else None
    )
    family = _v3_direct_family(evidence("family"))
    weight_fact = _v3_weight_fact(evidence("weight"), direction=direction)
    scale_fact = _v3_source_scale_fact(evidence("scale"), direction=direction)
    paint_signature = _v3_paint_signature(fill=fill, outline=outline)
    if (
        family is None
        or weight_fact is None
        or scale_fact is None
        or (require_paint and paint_signature is None)
    ):
        return None
    return {
        "page_id": str(getattr(record, "page_id", "") or ""),
        "bundle_id": str(getattr(record, "bundle_id", "") or ""),
        "parent_id": str(getattr(record, "parent_id", "") or ""),
        "root_id": str(getattr(record, "root_id", "") or ""),
        "semantic_role_class": _v3_semantic_role_class(record),
        "writing_mode": str(orientation.value),
        "writing_mode_reliable": True,
        "family_decision": family,
        "weight_fact": weight_fact,
        "weight_interval": tuple(weight_fact["score_interval"]),
        "source_scale_fact": scale_fact,
        "scale_interval": tuple(scale_fact["interval"]),
        "paint_signature": paint_signature,
    }


def _v3_apply_weight_axis(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
    style_context_snapshot: Any | None,
) -> None:
    """Resolve weight without allowing another axis to own a local fact."""

    _v3_reconcile_current_page_direct_weight_conflicts(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )
    _v3_apply_current_page_weight_cohorts(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )
    _v3_apply_prior_page_weight_cache(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
        style_context_snapshot=style_context_snapshot,
    )


def _v3_reconcile_current_page_direct_weight_conflicts(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
) -> None:
    """Reconcile one proven direct conflict without weakening local evidence.

    A direction-neutral transition-gap decision is the only trigger.  The
    compatible current-page cohort may replace its direct decisions only when
    the intervals that actually supported those decisions have one non-empty
    intersection wholly contained by one existing calibrated tier.
    """

    proposals: dict[
        str,
        list[
            tuple[
                str,
                tuple[str, ...],
                tuple[float, float],
                tuple[str, ...],
            ]
        ],
    ] = {}
    for anchor_id in sorted(facts_by_bundle):
        anchor_decision = decisions_by_bundle[anchor_id].get("weight")
        if not _v3_is_transition_gap_direct_weight(anchor_decision):
            continue
        anchor = facts_by_bundle[anchor_id]
        page_id = str(anchor["page_id"])
        candidate_ids = [
            bundle_id
            for bundle_id in sorted(facts_by_bundle)
            if (
                str(facts_by_bundle[bundle_id]["page_id"]) == page_id
                and (
                    decision := decisions_by_bundle[bundle_id].get("weight")
                )
                is not None
                and decision.status == "direct"
                and decision.confidence >= V3_PEER_DONOR_MIN_CONFIDENCE
                and _v3_weight_facts_are_compatible(
                    anchor,
                    facts_by_bundle[bundle_id],
                )
            )
        ]
        cohort_ids = _v3_unique_maximum_weight_compatibility_cluster(
            candidate_ids=candidate_ids,
            facts_by_bundle=facts_by_bundle,
            required_member_id=anchor_id,
        )
        if len(cohort_ids) < PEER_MINIMUM_DONOR_COUNT:
            continue
        direct_values = tuple(
            sorted(
                {
                    str(decisions_by_bundle[bundle_id]["weight"].value)
                    for bundle_id in cohort_ids
                }
            )
        )
        if len(direct_values) < 2:
            continue
        intervals = [
            _v3_effective_direct_weight_interval(
                facts=facts_by_bundle[bundle_id],
                decision=decisions_by_bundle[bundle_id]["weight"],
            )
            for bundle_id in cohort_ids
        ]
        if any(interval is None for interval in intervals):
            continue
        consensus = _v3_weight_consensus_bounds(
            tuple(interval for interval in intervals if interval is not None)
        )
        tier = _v3_weight_tier_containing_consensus(consensus)
        if not tier or tier not in direct_values or consensus is None:
            continue
        transition_ids = tuple(
            bundle_id
            for bundle_id in cohort_ids
            if _v3_is_transition_gap_direct_weight(
                decisions_by_bundle[bundle_id].get("weight")
            )
        )
        proposal = (
            tier,
            tuple(cohort_ids),
            tuple(float(value) for value in consensus),
            transition_ids,
        )
        for bundle_id in cohort_ids:
            proposals.setdefault(bundle_id, []).append(proposal)

    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for bundle_id in sorted(proposals):
        unique = sorted(set(proposals[bundle_id]))
        if len(unique) != 1:
            continue
        tier, cohort_ids, consensus, transition_ids = unique[0]
        current = decisions_by_bundle[bundle_id].get("weight")
        if current is None or current.status != "direct":
            continue
        cohort_decisions = [
            decisions_by_bundle[member_id]["weight"]
            for member_id in cohort_ids
        ]
        updates[bundle_id] = _v3_axis_decision(
            axis="weight",
            value=tier,
            status="peer",
            confidence=min(
                float(decision.confidence)
                for decision in cohort_decisions
            ),
            provenance=(
                "parent_style_arbitrator_v3:"
                "current_page_direct_weight_interval_consensus"
            ),
            reason_codes=(
                "weight_resolution_tier:current_page_direct_conflict",
                "current_page_direct_weight_interval_consensus",
                f"calibrated_normalized_weight_region:{tier}",
            ),
            peer_support={
                "evidence_source": (
                    "current_page_direct_weight_interval_consensus"
                ),
                "page_id": str(facts_by_bundle[bundle_id]["page_id"]),
                "cohort_bundle_ids": cohort_ids,
                "cohort_size": len(cohort_ids),
                "transition_gap_bundle_ids": transition_ids,
                "source_weight_consensus_interval": {
                    "low": float(consensus[0]),
                    "high": float(consensus[1]),
                },
                "observed_direct_values": sorted(
                    {
                        str(decision.value)
                        for decision in cohort_decisions
                    }
                ),
                "replaced_direct_value": str(current.value),
                "replaced_direct_confidence": float(current.confidence),
                "replaced_direct_provenance": current.provenance,
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["weight"] = decision


def _v3_is_transition_gap_direct_weight(
    decision: ParentStyleAxisDecisionV3 | None,
) -> bool:
    return bool(
        isinstance(decision, ParentStyleAxisDecisionV3)
        and decision.axis == "weight"
        and decision.status == "direct"
        and "direction_neutral_transition_gap_resolved"
        in tuple(decision.reason_codes)
    )


def _v3_effective_direct_weight_interval(
    *,
    facts: Mapping[str, Any],
    decision: ParentStyleAxisDecisionV3,
) -> tuple[float, float, float] | None:
    weight_fact = facts.get("weight_fact")
    if not isinstance(weight_fact, Mapping):
        return None
    interval = (
        weight_fact.get("direction_neutral_score_interval")
        if _v3_is_transition_gap_direct_weight(decision)
        else facts.get("weight_interval")
    )
    bounds = _v3_weight_interval_bounds(interval)
    if bounds is None or not _is_plain_sequence(interval):
        return None
    return tuple(float(value) for value in interval)


def _v3_weight_tier_containing_consensus(
    consensus: tuple[float, float] | None,
) -> str:
    if consensus is None:
        return ""
    low, high = [float(value) for value in consensus]
    matches = [
        tier
        for tier, bounds in (
            ("slender", V3_WEIGHT_SLENDER_SCORE_RANGE),
            ("base", V3_WEIGHT_BASE_SCORE_RANGE),
            ("heavy", V3_WEIGHT_HEAVY_SCORE_RANGE),
        )
        if low >= float(bounds[0]) and high <= float(bounds[1])
    ]
    return matches[0] if len(matches) == 1 else ""


def _v3_unique_maximum_weight_compatibility_cluster(
    *,
    candidate_ids: Sequence[str],
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    required_member_id: str,
) -> list[str]:
    clusters: set[tuple[str, ...]] = set()
    for seed in sorted(candidate_ids):
        cluster = [seed]
        for candidate in sorted(candidate_ids):
            if candidate == seed:
                continue
            if all(
                _v3_weight_facts_are_compatible(
                    facts_by_bundle[member],
                    facts_by_bundle[candidate],
                )
                for member in cluster
            ):
                cluster.append(candidate)
        normalized = tuple(sorted(cluster))
        if required_member_id in normalized:
            clusters.add(normalized)
    if not clusters:
        return []
    maximum = max(len(cluster) for cluster in clusters)
    maxima = sorted(
        cluster
        for cluster in clusters
        if len(cluster) == maximum
    )
    if len(maxima) != 1:
        return []
    return list(maxima[0])


def _v3_apply_current_page_weight_cohorts(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
) -> None:
    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for target_id in sorted(facts_by_bundle):
        target = facts_by_bundle[target_id]
        if (
            "weight" in decisions_by_bundle[target_id]
            or bool(target["weight_reliable_unclassified"])
        ):
            continue
        target_page_id = str(target["page_id"])
        donor_ids: list[str] = []
        for donor_id in sorted(facts_by_bundle):
            if donor_id == target_id:
                continue
            donor = facts_by_bundle[donor_id]
            if str(donor["page_id"]) != target_page_id:
                continue
            donor_decision = decisions_by_bundle[donor_id].get("weight")
            if (
                donor_decision is None
                or donor_decision.status != "direct"
                or donor_decision.confidence < V3_PEER_DONOR_MIN_CONFIDENCE
                or not _v3_weight_facts_are_compatible(target, donor)
            ):
                continue
            donor_ids.append(donor_id)

        selected = _v3_select_weight_cluster(
            donor_ids=donor_ids,
            facts_by_id=facts_by_bundle,
            decision_for_id={
                donor_id: decisions_by_bundle[donor_id]["weight"]
                for donor_id in donor_ids
            },
            target_facts=target,
        )
        if len(selected) < PEER_MINIMUM_DONOR_COUNT:
            continue
        donor_decisions = [
            decisions_by_bundle[donor_id]["weight"]
            for donor_id in selected
        ]
        updates[target_id] = _v3_axis_decision(
            axis="weight",
            value=donor_decisions[0].value,
            status="peer",
            confidence=float(
                np.mean([decision.confidence for decision in donor_decisions])
            ),
            provenance=(
                "parent_style_arbitrator_v3:"
                "current_page_weight_cohort_consensus"
            ),
            reason_codes=(
                *(
                    ("local_heavy_interval_overlaps_base",)
                    if bool(
                        target.get(
                            "weight_interval_ambiguous_heavy_candidate"
                        )
                    )
                    else ()
                ),
                "weight_resolution_tier:current_page_cohort",
                "current_page_weight_cohort_consensus",
            ),
            peer_support={
                "evidence_source": "current_page_cohort",
                "page_id": target_page_id,
                "donor_bundle_ids": selected,
                "donor_count": len(selected),
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["weight"] = decision


def _v3_apply_prior_page_weight_cache(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
    style_context_snapshot: Any | None,
) -> None:
    prefix_page_ids, records = _v3_qualified_prior_snapshot_records(
        style_context_snapshot
    )
    if not prefix_page_ids or not records:
        return
    prefix = set(prefix_page_ids)
    cache_facts: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    for record in records:
        page_id = str(getattr(record, "page_id", "") or "")
        bundle_id = str(getattr(record, "bundle_id", "") or "")
        if not page_id or page_id not in prefix or not bundle_id:
            continue
        key = f"prior::{page_id}::{bundle_id}"
        if key in cache_facts:
            duplicate_keys.add(key)
            continue
        facts = _v3_cache_weight_facts(record)
        if facts is not None:
            cache_facts[key] = facts
    for key in duplicate_keys:
        cache_facts.pop(key, None)
    if not cache_facts:
        return

    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for target_id in sorted(facts_by_bundle):
        target = facts_by_bundle[target_id]
        if (
            "weight" in decisions_by_bundle[target_id]
            or bool(target["weight_reliable_unclassified"])
        ):
            continue
        donor_ids = [
            donor_id
            for donor_id in sorted(cache_facts)
            if str(cache_facts[donor_id]["page_id"]) != str(target["page_id"])
            and _v3_weight_facts_are_compatible(
                target,
                cache_facts[donor_id],
            )
        ]
        selected = _v3_select_weight_cluster(
            donor_ids=donor_ids,
            facts_by_id=cache_facts,
            decision_for_id={
                donor_id: cache_facts[donor_id]["weight_decision"]
                for donor_id in donor_ids
            },
            target_facts=target,
        )
        if len(selected) < PEER_MINIMUM_DONOR_COUNT:
            continue
        donor_decisions = [
            cache_facts[donor_id]["weight_decision"]
            for donor_id in selected
        ]
        updates[target_id] = _v3_axis_decision(
            axis="weight",
            value=donor_decisions[0].value,
            status="peer",
            confidence=float(
                np.mean([decision.confidence for decision in donor_decisions])
            ),
            provenance=(
                "parent_style_arbitrator_v3:"
                "prior_page_weight_cache_consensus"
            ),
            reason_codes=(
                *(
                    ("local_heavy_interval_overlaps_base",)
                    if bool(
                        target.get(
                            "weight_interval_ambiguous_heavy_candidate"
                        )
                    )
                    else ()
                ),
                "weight_resolution_tier:prior_page_cache",
                "prior_page_weight_cache_consensus",
            ),
            peer_support={
                "evidence_source": "prior_page_cache",
                "donor_bundle_ids": [
                    str(cache_facts[donor_id]["bundle_id"])
                    for donor_id in selected
                ],
                "donor_page_ids": sorted(
                    {
                        str(cache_facts[donor_id]["page_id"])
                        for donor_id in selected
                    }
                ),
                "donor_count": len(selected),
                "snapshot_id": str(
                    getattr(style_context_snapshot, "snapshot_id", "") or ""
                ),
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["weight"] = decision


def _v3_qualified_prior_snapshot_records(
    style_context_snapshot: Any | None,
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if style_context_snapshot is None:
        return (), ()
    prefix_page_ids = tuple(
        str(item)
        for item in tuple(
            getattr(style_context_snapshot, "prefix_page_ids", ()) or ()
        )
        if str(item)
    )
    if (
        not prefix_page_ids
        or len(prefix_page_ids) != len(set(prefix_page_ids))
        or int(getattr(style_context_snapshot, "page_index", -1))
        != len(prefix_page_ids)
    ):
        return (), ()
    committed_delta_ids = tuple(
        str(item)
        for item in tuple(
            getattr(style_context_snapshot, "committed_delta_ids", ()) or ()
        )
        if str(item)
    )
    if len(committed_delta_ids) != len(prefix_page_ids):
        return (), ()
    return (
        prefix_page_ids,
        tuple(getattr(style_context_snapshot, "records", ()) or ()),
    )


def _v3_cache_weight_facts(record: Any) -> dict[str, Any] | None:
    facts = _v3_cache_family_facts(record, require_paint=False)
    if facts is None:
        return None
    weight_fact = facts.get("weight_fact")
    weight = _v3_direct_weight(
        weight_fact=weight_fact if isinstance(weight_fact, Mapping) else None,
        punctuation_only=False,
    )
    if (
        weight is None
        or weight.status != "direct"
        or weight.confidence < V3_PEER_DONOR_MIN_CONFIDENCE
    ):
        return None
    result = dict(facts)
    result["weight_decision"] = weight
    result["direct_decisions"] = {
        "family": result["family_decision"],
        "weight": weight,
    }
    return result


def _v3_select_weight_cluster(
    *,
    donor_ids: Sequence[str],
    facts_by_id: Mapping[str, Mapping[str, Any]],
    decision_for_id: Mapping[str, ParentStyleAxisDecisionV3],
    target_facts: Mapping[str, Any] | None = None,
) -> list[str]:
    values = {
        str(decision_for_id[donor_id].value)
        for donor_id in donor_ids
        if donor_id in decision_for_id
    }
    if not values:
        return []
    if len(values) == 1:
        return _v3_maximum_compatible_weight_cluster(
            donor_ids=donor_ids,
            facts_by_id=facts_by_id,
        )

    target_bounds = _v3_weight_interval_bounds(
        target_facts.get("weight_interval")
        if isinstance(target_facts, Mapping)
        else None
    )
    if target_bounds is None:
        return []

    qualifying_clusters: list[list[str]] = []
    for value in sorted(values):
        value_donor_ids = [
            donor_id
            for donor_id in sorted(donor_ids)
            if donor_id in decision_for_id
            and str(decision_for_id[donor_id].value) == value
        ]
        cluster = _v3_maximum_compatible_weight_cluster(
            donor_ids=value_donor_ids,
            facts_by_id=facts_by_id,
        )
        if len(cluster) < PEER_MINIMUM_DONOR_COUNT:
            continue
        consensus_bounds = _v3_weight_consensus_bounds(
            tuple(
                facts_by_id[donor_id].get("weight_interval")
                for donor_id in cluster
            )
        )
        if (
            consensus_bounds is not None
            and max(target_bounds[0], consensus_bounds[0])
            <= min(target_bounds[1], consensus_bounds[1])
        ):
            qualifying_clusters.append(cluster)
    if len(qualifying_clusters) != 1:
        return []
    return qualifying_clusters[0]


def _v3_maximum_compatible_weight_cluster(
    *,
    donor_ids: Sequence[str],
    facts_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    clusters: list[tuple[str, ...]] = []
    for seed in sorted(donor_ids):
        cluster = [seed]
        for candidate in sorted(donor_ids):
            if candidate == seed:
                continue
            if all(
                _v3_weight_facts_are_compatible(
                    facts_by_id[member],
                    facts_by_id[candidate],
                )
                for member in cluster
            ):
                cluster.append(candidate)
        clusters.append(tuple(sorted(cluster)))
    if not clusters:
        return []
    maximum = max(len(cluster) for cluster in clusters)
    if maximum < PEER_MINIMUM_DONOR_COUNT:
        return []
    return list(
        sorted(
            set(cluster for cluster in clusters if len(cluster) == maximum)
        )[0]
    )


def _v3_weight_consensus_bounds(
    intervals: Sequence[Any],
) -> tuple[float, float] | None:
    bounds = [_v3_weight_interval_bounds(interval) for interval in intervals]
    if any(item is None for item in bounds):
        return None
    valid_bounds = [item for item in bounds if item is not None]
    if len(valid_bounds) < PEER_MINIMUM_DONOR_COUNT:
        return None
    low = max(item[0] for item in valid_bounds)
    high = min(item[1] for item in valid_bounds)
    return (low, high) if low <= high else None


def _v3_weight_interval_bounds(
    interval: Any,
) -> tuple[float, float] | None:
    if not _is_plain_sequence(interval) or len(interval) != 3:
        return None
    try:
        low, median, high = [float(value) for value in interval]
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (low, median, high))
        or low > median
        or median > high
    ):
        return None
    return low, high


def _v3_weight_facts_are_compatible(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> bool:
    # Paint axes are independently resolved and cannot veto weight assistance.
    if (
        target["semantic_role_class"] != donor["semantic_role_class"]
        or not bool(target["writing_mode_reliable"])
        or not bool(donor["writing_mode_reliable"])
        or target["writing_mode"] != donor["writing_mode"]
        or _v3_has_fragmented_cell_population_scale_fact(target)
        or _v3_has_fragmented_cell_population_scale_fact(donor)
    ):
        return False
    target_family = _v3_weight_compatibility_family(target)
    donor_family = _v3_weight_compatibility_family(donor)
    if not target_family or target_family != donor_family:
        return False
    return _v3_intervals_are_compatible(
        target.get("scale_interval"),
        donor.get("scale_interval"),
    )


def _v3_weight_compatibility_family(facts: Mapping[str, Any]) -> str:
    family_decision = facts.get("family_decision")
    if not isinstance(family_decision, ParentStyleAxisDecisionV3):
        direct = facts.get("direct_decisions")
        if isinstance(direct, Mapping):
            family_decision = direct.get("family")
    if (
        isinstance(family_decision, ParentStyleAxisDecisionV3)
        and family_decision.value in {"sans", "serif"}
    ):
        return str(family_decision.value)
    posterior = facts.get("family_posterior")
    if isinstance(posterior, Mapping) and bool(posterior.get("reliable")):
        family = str(posterior.get("leading_family") or "")
        if family in {"sans", "serif"}:
            return family
    return ""


def _v3_peer_facts_are_compatible(
    target: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    axis: str,
    decisions_by_bundle: Mapping[str, Mapping[str, ParentStyleAxisDecisionV3]],
    target_id: str,
    donor_id: str,
) -> bool:
    if (
        target["semantic_role_class"] != donor["semantic_role_class"]
        or not bool(target["writing_mode_reliable"])
        or not bool(donor["writing_mode_reliable"])
        or target["writing_mode"] != donor["writing_mode"]
        or target["paint_signature"] is None
        or donor["paint_signature"] is None
        or target["paint_signature"] != donor["paint_signature"]
    ):
        return False
    target_fragmented_scale = (
        _v3_has_fragmented_cell_population_scale_fact(target)
    )
    donor_fragmented_scale = (
        _v3_has_fragmented_cell_population_scale_fact(donor)
    )
    if axis in {"family", "weight"} and (
        target_fragmented_scale or donor_fragmented_scale
    ):
        return False
    if axis == "family":
        return bool(
            _v3_intervals_are_compatible(
                target["weight_interval"], donor["weight_interval"]
            )
            and _v3_intervals_are_compatible(
                target["scale_interval"], donor["scale_interval"]
            )
        )
    if axis == "weight":
        return bool(
            _v3_family_posteriors_are_compatible(
                target["family_posterior"], donor["family_posterior"]
            )
            and _v3_intervals_are_compatible(
                target["scale_interval"], donor["scale_interval"]
            )
        )
    if axis == "source_scale":
        target_scale_fact = target.get("source_scale_fact")
        donor_scale_fact = donor.get("source_scale_fact")
        if (
            not isinstance(target_scale_fact, Mapping)
            or not isinstance(donor_scale_fact, Mapping)
            or not _v3_intervals_are_compatible(
                target_scale_fact.get("interval"),
                donor_scale_fact.get("interval"),
            )
        ):
            return False
        for required_axis in ("family", "weight"):
            if _v3_effective_target_role_for_scale(
                axis=required_axis,
                decisions_by_bundle=decisions_by_bundle,
                bundle_id=target_id,
            ) != _v3_effective_target_role_for_scale(
                axis=required_axis,
                decisions_by_bundle=decisions_by_bundle,
                bundle_id=donor_id,
            ):
                return False
        if not _v3_advisory_family_relation_matches_for_scale(target, donor):
            return False
        donor_weight = donor.get("weight_interval")
        if target_fragmented_scale:
            if donor_weight is None:
                return False
            target_weight = target.get("weight_interval")
            return bool(
                target_weight is None
                or _v3_intervals_are_compatible(target_weight, donor_weight)
            )
        return _v3_intervals_are_compatible(
            target.get("weight_interval"), donor_weight
        )
    return False


def _v3_has_fragmented_cell_population_scale_fact(
    facts: Mapping[str, Any],
) -> bool:
    fact = facts.get("source_scale_fact")
    return bool(
        isinstance(fact, Mapping)
        and "source_scale_fragmented_cell_population_proven"
        in tuple(fact.get("reason_codes") or ())
    )


def _v3_effective_target_role_for_scale(
    *,
    axis: str,
    decisions_by_bundle: Mapping[
        str, Mapping[str, ParentStyleAxisDecisionV3]
    ],
    bundle_id: str,
) -> str:
    decision = decisions_by_bundle[bundle_id].get(axis)
    if decision is not None and decision.status in {"direct", "peer"}:
        return str(decision.value)
    return {"family": "sans", "weight": "base"}[axis]


def _v3_advisory_family_relation_matches_for_scale(
    target: Mapping[str, Any], donor: Mapping[str, Any]
) -> bool:
    """Use the unpromoted family relation only to veto unlike scale peers."""

    target_posterior = target.get("family_posterior")
    donor_posterior = donor.get("family_posterior")
    if not isinstance(target_posterior, Mapping) or not isinstance(
        donor_posterior, Mapping
    ):
        return False
    target_family = str(target_posterior.get("leading_family") or "")
    donor_family = str(donor_posterior.get("leading_family") or "")
    return bool(target_family and target_family == donor_family)


def _v3_intervals_are_compatible(first: Any, second: Any) -> bool:
    if (
        not _is_plain_sequence(first)
        or not _is_plain_sequence(second)
        or len(first) != 3
        or len(second) != 3
    ):
        return False
    first_low, _, first_high = [float(value) for value in first]
    second_low, _, second_high = [float(value) for value in second]
    return max(first_low, second_low) <= min(first_high, second_high)


def _v3_family_posteriors_are_compatible(first: Any, second: Any) -> bool:
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return False
    return bool(
        first.get("reliable")
        and second.get("reliable")
        and first.get("leading_family") == second.get("leading_family")
    )


def _v3_source_advance_complete_link_components(
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    """Return disjoint interval cliques; transitive overlap is insufficient."""

    grouped: dict[tuple[str, str], list[tuple[str, tuple[float, float, float]]]] = {}
    for bundle_id in sorted(facts_by_bundle):
        facts = facts_by_bundle[bundle_id]
        relation = facts.get("source_advance_grid_fact")
        if (
            not isinstance(relation, Mapping)
            or bool(relation.get("harmonic_ambiguous"))
            or int(relation.get("body_landmark_count") or 0) < 3
            or int(relation.get("qualified_adjacent_gap_count") or 0) < 2
            or not bool(facts.get("writing_mode_reliable"))
            or str(facts.get("source_evidence_status") or "") != "observed"
        ):
            continue
        interval = relation.get("interval")
        if not _is_plain_sequence(interval) or len(interval) != 3:
            continue
        try:
            low, median, high = [float(value) for value in interval]
        except (TypeError, ValueError):
            continue
        if (
            not all(math.isfinite(value) for value in (low, median, high))
            or low <= 0.0
            or low > median
            or median > high
        ):
            continue
        key = (
            str(facts.get("page_id") or ""),
            str(facts.get("writing_mode") or ""),
        )
        if not all(key):
            continue
        grouped.setdefault(key, []).append(
            (bundle_id, (low, median, high))
        )

    components: list[tuple[str, ...]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        endpoints = sorted(
            {
                value
                for _, interval in candidates
                for value in (interval[0], interval[2])
            }
        )
        cliques = {
            tuple(
                sorted(
                    bundle_id
                    for bundle_id, interval in candidates
                    if interval[0] <= point <= interval[2]
                )
            )
            for point in endpoints
        }
        cliques = {clique for clique in cliques if len(clique) >= 3}
        maximal = {
            clique
            for clique in cliques
            if not any(set(clique) < set(other) for other in cliques)
        }
        for clique in sorted(maximal):
            if any(
                clique != other and set(clique).intersection(other)
                for other in maximal
            ):
                continue
            components.append(clique)
    return tuple(components)


def _v3_pooled_visible_ink_interval(
    *,
    member_ids: Sequence[str],
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Resolve the supported upper/full-cell tier in the existing ink basis."""

    members = tuple(sorted(str(value) for value in member_ids if value))
    if len(members) < 3:
        return None
    relation_intervals = [
        facts_by_bundle[bundle_id]["source_advance_grid_fact"]["interval"]
        for bundle_id in members
    ]
    grid_upper = max(float(interval[2]) for interval in relation_intervals)
    samples: list[tuple[float, str]] = []
    for bundle_id in members:
        relation = facts_by_bundle[bundle_id].get("source_advance_grid_fact")
        if not isinstance(relation, Mapping):
            return None
        spans = relation.get("visible_ink_spans_px")
        if not _is_plain_sequence(spans):
            return None
        for raw in spans:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 0.0 < value <= grid_upper * 1.10:
                samples.append((value, bundle_id))
    if len(samples) < 3:
        return None

    values = np.asarray([value for value, _ in samples], dtype=np.float32)
    upper_threshold = float(np.percentile(values, 75))
    upper = [
        (value, bundle_id)
        for value, bundle_id in samples
        if value >= upper_threshold
    ]
    if len({bundle_id for _, bundle_id in upper}) < 2:
        return None
    upper_values = np.asarray([value for value, _ in upper], dtype=np.float32)
    p20, median, p80 = [
        float(value) for value in np.percentile(upper_values, [20, 50, 80])
    ]
    if not (0.0 < p20 <= median <= p80):
        return None
    return {
        "interval": (p20, median, p80),
        "raw_visible_ink_sample_count": len(samples),
        "upper_tier_sample_count": len(upper),
        "upper_tier_parent_count": len(
            {bundle_id for _, bundle_id in upper}
        ),
        "upper_tier_threshold_px": upper_threshold,
        "grid_upper_bound_px": grid_upper,
    }


def _v3_source_advance_relation_updates(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: Mapping[
        str, Mapping[str, ParentStyleAxisDecisionV3]
    ],
) -> dict[str, ParentStyleAxisDecisionV3]:
    """Use source cadence only to authorize pooled visible-ink reconciliation."""

    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for component in _v3_source_advance_complete_link_components(
        facts_by_bundle
    ):
        pooled = _v3_pooled_visible_ink_interval(
            member_ids=component,
            facts_by_bundle=facts_by_bundle,
        )
        if pooled is None:
            continue
        pooled_interval = tuple(float(value) for value in pooled["interval"])
        direct_intervals: list[tuple[float, float, float]] = []
        direct_conflict = False
        for bundle_id in component:
            decision = decisions_by_bundle.get(bundle_id, {}).get(
                "source_scale"
            )
            if decision is None or decision.status != "direct":
                continue
            value = decision.value if isinstance(decision.value, Mapping) else {}
            interval = _v3_numeric_interval(
                value,
                keys=("p20_px", "median_px", "p80_px"),
                minimum=1e-8,
                maximum=None,
            )
            if interval is None or not _v3_intervals_are_compatible(
                interval,
                pooled_interval,
            ):
                direct_conflict = True
                break
            direct_intervals.append(interval)
        if direct_conflict or any(
            not _v3_intervals_are_compatible(first, second)
            for index, first in enumerate(direct_intervals)
            for second in direct_intervals[index + 1 :]
        ):
            continue

        confidences = [
            float(
                facts_by_bundle[bundle_id]["source_advance_grid_fact"].get(
                    "confidence"
                )
                or 0.0
            )
            for bundle_id in component
        ]
        confidence = max(0.0, min(1.0, float(np.mean(confidences))))
        group_fingerprint = hashlib.sha256(
            "\n".join(component).encode("utf-8")
        ).hexdigest()[:16]
        p20, median, p80 = pooled_interval
        for bundle_id in component:
            if decisions_by_bundle.get(bundle_id, {}).get("source_scale") is not None:
                continue
            local = facts_by_bundle[bundle_id].get("source_scale_fact")
            local_interval = (
                local.get("interval") if isinstance(local, Mapping) else None
            )
            if _is_plain_sequence(local_interval) and len(local_interval) == 3:
                try:
                    local_median = float(local_interval[1])
                    local_confidence = float(local.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    continue
                if local_confidence >= V3_DIRECT_SCALE_MIN_CONFIDENCE:
                    continue
                # This contract repairs underestimation only. A low-confidence
                # larger observation may be a real scale distinction and is
                # therefore preserved rather than normalized downward.
                if local_median >= p20:
                    continue
            updates[bundle_id] = _v3_axis_decision(
                axis="source_scale",
                value={
                    "p20_px": p20,
                    "median_px": median,
                    "p80_px": p80,
                },
                status="peer",
                confidence=confidence,
                provenance=(
                    "parent_style_arbitrator_v3:"
                    "source_advance_relation_pooled_visible_ink"
                ),
                reason_codes=(
                    "source_advance_grid_complete_link_relation",
                    "pooled_visible_ink_full_cell_tier",
                    "low_confidence_source_scale_underestimate_reconciled",
                ),
                peer_support={
                    "group_id": (
                        "current-page:source-scale:advance-relation:"
                        f"{group_fingerprint}"
                    ),
                    "member_bundle_ids": component,
                    "member_count": len(component),
                    "relation_evidence": "source_advance_grid",
                    "executable_measurement_basis": (
                        "pooled_visible_ink_span"
                    ),
                    **{
                        key: value
                        for key, value in pooled.items()
                        if key != "interval"
                    },
                },
            )
    return updates


def _v3_apply_source_scale_peer_axis(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
) -> None:
    relation_updates = _v3_source_advance_relation_updates(
        facts_by_bundle=facts_by_bundle,
        decisions_by_bundle=decisions_by_bundle,
    )
    for bundle_id, decision in relation_updates.items():
        decisions_by_bundle[bundle_id]["source_scale"] = decision

    updates: dict[str, ParentStyleAxisDecisionV3] = {}
    for target_id in sorted(facts_by_bundle):
        if "source_scale" in decisions_by_bundle[target_id]:
            continue
        if not isinstance(
            facts_by_bundle[target_id].get("source_scale_fact"), Mapping
        ):
            continue
        donors: list[str] = []
        for donor_id in sorted(facts_by_bundle):
            if donor_id == target_id:
                continue
            donor_decision = decisions_by_bundle[donor_id].get("source_scale")
            if (
                donor_decision is None
                or donor_decision.status != "direct"
                or donor_decision.confidence < V3_PEER_DONOR_MIN_CONFIDENCE
                or not _v3_peer_facts_are_compatible(
                    facts_by_bundle[target_id],
                    facts_by_bundle[donor_id],
                    axis="source_scale",
                    decisions_by_bundle=decisions_by_bundle,
                    target_id=target_id,
                    donor_id=donor_id,
                )
            ):
                continue
            donors.append(donor_id)
        if len(donors) < PEER_MINIMUM_DONOR_COUNT:
            continue
        medians = [
            float(decisions_by_bundle[item]["source_scale"].value["median_px"])
            for item in donors
        ]
        median = float(np.median(medians))
        deviations = [abs(value - median) for value in medians]
        mad = float(np.median(deviations))
        mad_ratio = mad / max(1e-8, median)
        if mad_ratio > V3_SCALE_PEER_MAXIMUM_MAD_RATIO:
            continue
        p20 = float(
            np.median(
                [
                    decisions_by_bundle[item]["source_scale"].value["p20_px"]
                    for item in donors
                ]
            )
        )
        p80 = float(
            np.median(
                [
                    decisions_by_bundle[item]["source_scale"].value["p80_px"]
                    for item in donors
                ]
            )
        )
        p20 = min(p20, median)
        p80 = max(p80, median)
        updates[target_id] = _v3_axis_decision(
            axis="source_scale",
            value={"p20_px": p20, "median_px": median, "p80_px": p80},
            status="peer",
            confidence=float(
                np.mean(
                    [
                        decisions_by_bundle[item]["source_scale"].confidence
                        for item in donors
                    ]
                )
            ),
            provenance=(
                "parent_style_arbitrator_v3:"
                "effective_target_role_scale_median_mad"
            ),
            reason_codes=("effective_target_role_source_scale_consensus",),
            peer_support={
                "group_id": (
                    "run-wide:source_scale:effective-target-role:"
                    f"{facts_by_bundle[target_id]['semantic_role_class']}:"
                    f"{facts_by_bundle[target_id]['writing_mode']}"
                ),
                "donor_bundle_ids": donors,
                "donor_count": len(donors),
                "median_absolute_deviation_px": mad,
                "median_absolute_deviation_ratio": mad_ratio,
            },
        )
    for bundle_id, decision in updates.items():
        decisions_by_bundle[bundle_id]["source_scale"] = decision


def _v3_apply_low_confidence_local_source_scale(
    *,
    facts_by_bundle: Mapping[str, Mapping[str, Any]],
    decisions_by_bundle: dict[str, dict[str, ParentStyleAxisDecisionV3]],
) -> None:
    for bundle_id in sorted(facts_by_bundle):
        decisions = decisions_by_bundle[bundle_id]
        if "source_scale" in decisions:
            continue
        fact = facts_by_bundle[bundle_id].get("source_scale_fact")
        if not isinstance(fact, Mapping):
            continue
        p20, median, p80 = [float(value) for value in fact["interval"]]
        decisions["source_scale"] = _v3_axis_decision(
            axis="source_scale",
            value={"p20_px": p20, "median_px": median, "p80_px": p80},
            status="fallback",
            confidence=float(fact["confidence"]),
            provenance=(
                "parent_style_arbitrator_v3:"
                "low_confidence_local_source_scale_fallback"
            ),
            reason_codes=(
                *tuple(fact.get("reason_codes") or ()),
                "peer_unavailable_low_confidence_local_scale_retained",
            ),
        )


def _v3_apply_axis_fallbacks(
    *,
    facts: Mapping[str, Any],
    decisions: dict[str, ParentStyleAxisDecisionV3],
) -> None:
    reasons = tuple(facts.get("identity_reason_codes") or ())
    if "family" not in decisions:
        family_reasons = tuple(facts.get("family_resolution_reasons") or ())
        decisions["family"] = _v3_axis_decision(
            axis="family",
            value=_v3_calibrated_target_family_fallback(facts),
            status="fallback",
            confidence=0.0,
            provenance=(
                "parent_style_arbitrator_v3:"
                "calibrated_target_family_fallback"
            ),
            reason_codes=(
                *reasons,
                *family_reasons,
                "family_calibrated_target_fallback",
            ),
        )
    if "weight" not in decisions:
        decisions["weight"] = _v3_axis_decision(
            axis="weight",
            value="base",
            status="fallback",
            confidence=0.0,
            provenance=(
                "parent_style_arbitrator_v3:"
                "calibrated_target_weight_fallback"
            ),
            reason_codes=(
                *reasons,
                *(
                    ("local_heavy_interval_overlaps_base",)
                    if bool(
                        facts.get(
                            "weight_interval_ambiguous_heavy_candidate"
                        )
                    )
                    else ()
                ),
                "weight_resolution_tier:calibrated_target_fallback",
                "weight_calibrated_target_fallback",
            ),
        )
    fallback_values: dict[str, Any] = {
        "source_scale": None,
        "fill": {"color": "#000000", "polarity": "dark"},
        "outline": (
            {
                "present": True,
                "color": "#FFFFFF",
                "source_width_to_cell_ratio": 0.06,
            }
            if facts["semantic_role_class"] in {"background", "caption", "narration"}
            else {
                "present": False,
                "color": "#FFFFFF",
                "source_width_to_cell_ratio": 0.0,
            }
        ),
        "orientation": "vertical",
    }
    for axis, value in fallback_values.items():
        if axis in decisions:
            continue
        decisions[axis] = _v3_axis_decision(
            axis=axis,
            value=value,
            status="fallback",
            confidence=0.0,
            provenance=f"parent_style_arbitrator_v3:deterministic_{axis}_fallback",
            reason_codes=(*reasons, f"{axis}_evidence_unresolved_after_peer"),
        )
    for axis in ("rotation", "shadow"):
        if axis in decisions:
            continue
        decisions[axis] = _v3_axis_decision(
            axis=axis,
            value=None,
            status="unavailable",
            confidence=0.0,
            provenance=f"parent_style_arbitrator_v3:{axis}_unavailable",
            reason_codes=(*reasons, f"{axis}_evidence_unavailable"),
        )


def _v3_calibrated_target_family_fallback(facts: Mapping[str, Any]) -> str:
    role_class = str(facts.get("semantic_role_class") or "").strip().lower()
    if role_class in {"background", "caption", "narration"}:
        return "serif"
    return "sans"


def _v3_axis_decision(
    *,
    axis: str,
    value: Any,
    status: str,
    confidence: float,
    provenance: str,
    reason_codes: Sequence[str],
    peer_support: Mapping[str, Any] | None = None,
) -> ParentStyleAxisDecisionV3:
    return ParentStyleAxisDecisionV3(
        axis=axis,
        value=value,
        status=status,
        confidence=confidence,
        provenance=provenance,
        reason_codes=tuple(reason_codes),
        peer_support=peer_support or {},
    )


def _v3_plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _v3_plain_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if _is_plain_sequence(value):
        return [_v3_plain_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ParentFontDetectionRunResult:
    page_id: str
    mode: str
    enabled: bool = False
    applied_count: int = 0
    fallback_count: int = 0
    skipped_count: int = 0
    model_path: str = ""
    labels_path: str = ""
    gpu_requested: bool = False
    requested_execution_provider: str = ""
    available_execution_providers: list[str] = field(default_factory=list)
    active_execution_providers: list[str] = field(default_factory=list)
    primary_execution_provider: str = ""
    provider_fallback_reason: str = ""
    provider_preload_error: str = ""
    errors: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_font_detection_version": "parent_font_detection_v2",
            "page_id": self.page_id,
            "mode": self.mode,
            "enabled": self.enabled,
            "applied_count": self.applied_count,
            "fallback_count": self.fallback_count,
            "skipped_count": self.skipped_count,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "gpu_requested": self.gpu_requested,
            "requested_execution_provider": self.requested_execution_provider,
            "available_execution_providers": list(self.available_execution_providers),
            "active_execution_providers": list(self.active_execution_providers),
            "primary_execution_provider": self.primary_execution_provider,
            "provider_fallback_reason": self.provider_fallback_reason,
            "provider_preload_error": self.provider_preload_error,
            "errors": list(self.errors),
            "records": [dict(record) for record in self.records],
        }


def _source_font_candidate_descriptors(
    label: Mapping[str, Any],
) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    path = str(label.get("path") or "")
    language = str(label.get("language") or "")
    if path:
        descriptors["path"] = path
    if language:
        descriptors["language"] = language
    if isinstance(label.get("serif"), bool):
        descriptors["serif"] = bool(label["serif"])
    return descriptors


def _source_font_label_catalog_version(
    labels: Sequence[Mapping[str, Any]],
    *,
    label_count: int | None = None,
) -> str:
    resolved_label_count = (
        int(label_count) if label_count is not None else len(labels)
    )
    if resolved_label_count <= 0:
        raise ValueError("source-font label catalog must be non-empty")
    payload = [
        {
            "class_index": index,
            "descriptors": _source_font_candidate_descriptors(
                _label_at(labels, index)
            ),
        }
        for index in range(resolved_label_count)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "yuzumarker_label_catalog_sha256:" + hashlib.sha256(
        encoded
    ).hexdigest()


def _source_font_label_identity(
    *,
    catalog_version: str,
    class_index: int,
    descriptors: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        {
            "catalog_version": str(catalog_version),
            "class_index": int(class_index),
            "descriptors": dict(descriptors),
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "yuzumarker_label:" + hashlib.sha256(encoded).hexdigest()


def _build_source_font_score_support(
    probabilities: Any,
    labels: Sequence[Mapping[str, Any]],
    *,
    catalog_version: str,
) -> SourceFontScoreSupportV1:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size <= 0:
        raise ValueError("source-font scores must be non-empty")
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("source-font scores must contain positive finite mass")
    values = values / total

    indices = np.arange(values.size, dtype=np.int64)
    ordered = np.lexsort((indices, -values))
    cumulative = np.cumsum(values[ordered], dtype=np.float64)
    required_count = int(
        np.searchsorted(
            cumulative,
            SOURCE_FONT_RETAINED_MASS_FLOOR,
            side="left",
        )
        + 1
    )
    candidate_ceiling = min(
        SOURCE_FONT_CANDIDATE_CEILING,
        int(values.size),
    )
    retained_count = min(required_count, candidate_ceiling)
    retained_indices = ordered[:retained_count]
    candidates = []
    for raw_index in retained_indices:
        class_index = int(raw_index)
        descriptors = _source_font_candidate_descriptors(
            _label_at(labels, class_index)
        )
        candidates.append(
            SourceFontCandidate(
                catalog_version=catalog_version,
                class_index=class_index,
                label_identity=_source_font_label_identity(
                    catalog_version=catalog_version,
                    class_index=class_index,
                    descriptors=descriptors,
                ),
                normalized_model_score=float(values[class_index]),
                descriptors=descriptors,
            )
        )
    retained_mass = math.fsum(
        item.normalized_model_score for item in candidates
    )
    residual_mass = max(0.0, min(1.0, 1.0 - retained_mass))
    positive = values[values > 0.0]
    normalized_entropy = (
        float(
            -np.sum(positive * np.log(positive))
            / math.log(float(values.size))
        )
        if values.size > 1
        else 0.0
    )
    margin = float(
        values[ordered[0]] - values[ordered[1]]
        if values.size > 1
        else values[ordered[0]]
    )
    return SourceFontScoreSupportV1(
        catalog_version=catalog_version,
        label_count=int(values.size),
        retained_mass_floor=SOURCE_FONT_RETAINED_MASS_FLOOR,
        candidate_ceiling=candidate_ceiling,
        candidates=tuple(candidates),
        retained_mass=retained_mass,
        residual_mass=residual_mass,
        status=(
            SOURCE_FONT_SUPPORT_FLOOR_MET
            if required_count <= candidate_ceiling
            else SOURCE_FONT_SUPPORT_TRUNCATED
        ),
        normalized_entropy=max(0.0, min(1.0, normalized_entropy)),
        margin=max(0.0, min(1.0, margin)),
    )


@dataclass(frozen=True)
class _TargetFontAffinityCatalog:
    """Process-local Yuzu descriptors for the installed automatic roles."""

    identity_sha256: str
    role_probe_vectors: np.ndarray
    role_records: tuple[Mapping[str, Any], ...]


_TARGET_FONT_AFFINITY_CATALOG_CACHE: dict[
    str,
    _TargetFontAffinityCatalog,
] = {}
_TARGET_FONT_AFFINITY_CATALOG_ERROR_CACHE: dict[str, str] = {}
_TARGET_FONT_AFFINITY_CATALOG_LOCK = threading.RLock()
_TARGET_FONT_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _target_font_file_sha256(path: str) -> str:
    normalized = os.path.abspath(str(path or ""))
    stat = os.stat(normalized)
    key = (normalized, int(stat.st_size), int(stat.st_mtime_ns))
    with _TARGET_FONT_AFFINITY_CATALOG_LOCK:
        cached = _TARGET_FONT_FILE_SHA256_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with open(normalized, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _TARGET_FONT_AFFINITY_CATALOG_LOCK:
        _TARGET_FONT_FILE_SHA256_CACHE[key] = value
    return value


def _target_font_affinity_descriptor(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != FONT_COUNT or not np.all(np.isfinite(values)):
        raise ValueError("target-font affinity requires complete finite logits")
    scaled = values / TARGET_FONT_AFFINITY_TEMPERATURE
    scaled -= float(scaled.max())
    exponent = np.exp(scaled)
    denominator = float(exponent.sum())
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("target-font affinity softmax is invalid")
    descriptor = np.sqrt(exponent / denominator)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("target-font affinity descriptor is empty")
    return np.asarray(descriptor / norm, dtype=np.float32)


def _draw_target_probe_glyph(
    draw: Any,
    *,
    center_x: float,
    center_y: float,
    glyph: str,
    font: Any,
) -> None:
    bbox = draw.textbbox((0, 0), glyph, font=font)
    x = float(center_x) - (float(bbox[0]) + float(bbox[2])) * 0.5
    y = float(center_y) - (float(bbox[1]) + float(bbox[3])) * 0.5
    draw.text((round(x), round(y)), glyph, font=font, fill=(0, 0, 0))


def _render_target_font_affinity_probe(
    *,
    font_path: str,
    text: str,
    columns: int,
    font_size: int,
    cell_step: int,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), int(font_size))
    glyphs = list(str(text))
    column_count = max(1, int(columns))
    per_column = int(math.ceil(len(glyphs) / column_count))
    if column_count == 1:
        centers = [256.0]
    else:
        column_gap = max(94.0, float(font_size) * 1.65)
        right = 256.0 + column_gap * 0.5
        centers = [
            right - column_gap * index for index in range(column_count)
        ]
    for column_index in range(column_count):
        chunk = glyphs[
            column_index * per_column : (column_index + 1) * per_column
        ]
        if not chunk:
            continue
        total_height = float(cell_step) * max(0, len(chunk) - 1)
        start_y = 256.0 - total_height * 0.5
        for row_index, glyph in enumerate(chunk):
            _draw_target_probe_glyph(
                draw,
                center_x=centers[column_index],
                center_y=start_y + row_index * float(cell_step),
                glyph=glyph,
                font=font,
            )
    return canvas


class YuzuMarkerOnnxFontDetector:
    """ONNX adapter for YuzuMarker.FontDetection."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        labels_path: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.model_path = model_path or resolve_yuzumarker_font_onnx_file() or ""
        self.labels_path = labels_path or resolve_yuzumarker_font_labels_file() or ""
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError("YuzuMarker ONNX model is missing")
        if not self.labels_path or not os.path.isfile(self.labels_path):
            raise FileNotFoundError("YuzuMarker font labels are missing")
        self._labels = _load_font_labels(self.labels_path)
        self._source_font_label_catalog_version = (
            _source_font_label_catalog_version(
                self._labels,
                label_count=FONT_COUNT,
            )
        )
        self._factorized_attribute_taxonomy = (
            _build_factorized_attribute_taxonomy(
                self._labels,
                label_count=FONT_COUNT,
            )
        )
        self._session = _load_onnx_session(self.model_path, use_gpu=use_gpu)
        metadata = _onnx_session_provider_metadata(
            self.model_path,
            use_gpu=use_gpu,
            session=self._session,
        )
        self.gpu_requested = bool(metadata.get("gpu_requested"))
        self.requested_execution_provider = str(metadata.get("requested_execution_provider") or "")
        self.available_execution_providers = list(metadata.get("available_execution_providers") or [])
        self.active_execution_providers = list(metadata.get("active_execution_providers") or [])
        self.primary_execution_provider = str(metadata.get("primary_execution_provider") or "")
        self.provider_fallback_reason = str(metadata.get("provider_fallback_reason") or "")
        self.provider_preload_error = str(metadata.get("provider_preload_error") or "")
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("YuzuMarker ONNX model has no inputs")
        self._input_name = inputs[0].name
        self._target_font_catalog_identity_sha256 = ""
        self._target_font_catalog_role_records: tuple[
            dict[str, Any],
            ...,
        ] = ()

    def _infer_vector(self, image: Any) -> np.ndarray:
        """Run one unchanged 512px Yuzu inference for one image."""

        from PIL import ImageOps

        prepared = ImageOps.exif_transpose(image).convert("RGB").resize((512, 512))
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)[None, ...]
        output = self._session.run(None, {self._input_name: array})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape[0] < FONT_COUNT + 12:
            raise RuntimeError(
                f"Unexpected YuzuMarker output length: {vector.shape[0]}"
            )
        return vector

    def _target_font_catalog_identity(
        self,
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        cached_identity = str(
            getattr(
                self,
                "_target_font_catalog_identity_sha256",
                "",
            )
            or ""
        )
        cached_records = tuple(
            getattr(self, "_target_font_catalog_role_records", ()) or ()
        )
        if cached_identity and cached_records:
            return cached_identity, cached_records

        manager = FontManager()
        inventory = {
            item.role_id: item for item in manager.required_role_inventory()
        }
        role_records: list[dict[str, Any]] = []
        for role_id in TARGET_FONT_AFFINITY_ROLE_IDS:
            status = inventory.get(role_id)
            if status is None or not status.selected_face_id:
                raise RuntimeError(
                    f"installed_target_role_unavailable:{role_id}"
                )
            face = manager.face(status.selected_face_id)
            if face is None or not face.path or not os.path.isfile(face.path):
                raise RuntimeError(
                    f"installed_target_face_unavailable:{role_id}"
                )
            role_records.append(
                {
                    "role_id": role_id,
                    "selected_face_id": face.face_id,
                    "font_path": face.path,
                    "font_sha256": _target_font_file_sha256(face.path),
                }
            )
        identity_payload = {
            "descriptor_policy_version": (
                TARGET_FONT_AFFINITY_DESCRIPTOR_POLICY_VERSION
            ),
            "probe_policy_version": TARGET_FONT_AFFINITY_PROBE_POLICY_VERSION,
            "model_sha256": _target_font_file_sha256(self.model_path),
            "labels_sha256": _target_font_file_sha256(self.labels_path),
            "label_catalog_version": (
                self._source_font_label_catalog_version
            ),
            "roles": [
                {
                    "role_id": record["role_id"],
                    "selected_face_id": record["selected_face_id"],
                    "font_sha256": record["font_sha256"],
                }
                for record in role_records
            ],
            "probes": [
                {
                    "probe_id": probe_id,
                    "text": text,
                    "columns": columns,
                    "font_size": font_size,
                    "cell_step": cell_step,
                }
                for (
                    probe_id,
                    text,
                    columns,
                    font_size,
                    cell_step,
                ) in TARGET_FONT_AFFINITY_PROBE_SPECS
            ],
        }
        encoded = json.dumps(
            identity_payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        identity = hashlib.sha256(encoded).hexdigest()
        records = tuple(role_records)
        self._target_font_catalog_identity_sha256 = identity
        self._target_font_catalog_role_records = records
        return identity, records

    def _target_font_affinity_catalog(
        self,
    ) -> tuple[_TargetFontAffinityCatalog | None, str]:
        try:
            catalog_identity, role_records = (
                self._target_font_catalog_identity()
            )
        except Exception as exc:
            return None, (
                "target_font_catalog_identity_unavailable:"
                f"{type(exc).__name__}"
            )

        with _TARGET_FONT_AFFINITY_CATALOG_LOCK:
            cached = _TARGET_FONT_AFFINITY_CATALOG_CACHE.get(
                catalog_identity
            )
            if cached is not None:
                return cached, ""
            cached_error = _TARGET_FONT_AFFINITY_CATALOG_ERROR_CACHE.get(
                catalog_identity
            )
            if cached_error:
                return None, cached_error
            try:
                role_vectors: list[list[np.ndarray]] = []
                for record in role_records:
                    probe_vectors: list[np.ndarray] = []
                    for (
                        _probe_id,
                        text,
                        columns,
                        font_size,
                        cell_step,
                    ) in TARGET_FONT_AFFINITY_PROBE_SPECS:
                        probe = _render_target_font_affinity_probe(
                            font_path=str(record["font_path"]),
                            text=text,
                            columns=columns,
                            font_size=font_size,
                            cell_step=cell_step,
                        )
                        vector = self._infer_vector(probe)
                        probe_vectors.append(
                            _target_font_affinity_descriptor(
                                vector[:FONT_COUNT]
                            )
                        )
                    role_vectors.append(probe_vectors)
                matrix = np.asarray(role_vectors, dtype=np.float32)
                expected_shape = (
                    len(TARGET_FONT_AFFINITY_ROLE_IDS),
                    len(TARGET_FONT_AFFINITY_PROBE_SPECS),
                    FONT_COUNT,
                )
                if matrix.shape != expected_shape:
                    raise RuntimeError(
                        "target_font_catalog_shape_invalid:"
                        f"{matrix.shape}"
                    )
                matrix.setflags(write=False)
                catalog = _TargetFontAffinityCatalog(
                    identity_sha256=catalog_identity,
                    role_probe_vectors=matrix,
                    role_records=role_records,
                )
                _TARGET_FONT_AFFINITY_CATALOG_CACHE[
                    catalog_identity
                ] = catalog
                return catalog, ""
            except Exception as exc:
                error = (
                    "target_font_catalog_build_unavailable:"
                    f"{type(exc).__name__}"
                )
                _TARGET_FONT_AFFINITY_CATALOG_ERROR_CACHE[
                    catalog_identity
                ] = error
                return None, error

    def _target_font_affinity_observation(
        self,
        *,
        font_logits: Any,
        source_input_sha256: str,
    ) -> tuple[TargetFontAffinityObservationV1 | None, str]:
        catalog, error = self._target_font_affinity_catalog()
        if catalog is None:
            return None, error or "target_font_catalog_unavailable"
        try:
            source = _target_font_affinity_descriptor(font_logits)
            similarities = np.einsum(
                "d,rpd->rp",
                source,
                catalog.role_probe_vectors,
                optimize=True,
            )
            top_count = min(
                TARGET_FONT_AFFINITY_TOP_PROBE_COUNT,
                similarities.shape[1],
            )
            strongest = np.sort(similarities, axis=1)[:, -top_count:]
            scores = np.clip(strongest.mean(axis=1), 0.0, 1.0)
            return (
                TargetFontAffinityObservationV1(
                    catalog_identity_sha256=catalog.identity_sha256,
                    descriptor_policy_version=(
                        TARGET_FONT_AFFINITY_DESCRIPTOR_POLICY_VERSION
                    ),
                    source_input_sha256=str(source_input_sha256 or ""),
                    model_identity=YUZUMARKER_PROVIDER_MODEL,
                    label_catalog_version=(
                        self._source_font_label_catalog_version
                    ),
                    provider_provenance={
                        "gpu_requested": bool(self.gpu_requested),
                        "requested_execution_provider": (
                            self.requested_execution_provider
                        ),
                        "available_execution_providers": list(
                            self.available_execution_providers
                        ),
                        "active_execution_providers": list(
                            self.active_execution_providers
                        ),
                        "primary_execution_provider": (
                            self.primary_execution_provider
                        ),
                        "provider_fallback_reason": (
                            self.provider_fallback_reason
                        ),
                    },
                    role_scores={
                        role_id: float(scores[index])
                        for index, role_id in enumerate(
                            TARGET_FONT_AFFINITY_ROLE_IDS
                        )
                    },
                ),
                "",
            )
        except Exception as exc:
            return None, (
                "target_font_affinity_projection_unavailable:"
                f"{type(exc).__name__}"
            )

    def detect(self, image: Any) -> dict[str, Any]:
        vector = self._infer_vector(image)

        font_prob = _softmax(vector[:FONT_COUNT])
        source_font_label_catalog_version = str(
            getattr(self, "_source_font_label_catalog_version", "") or ""
        )
        if not source_font_label_catalog_version:
            source_font_label_catalog_version = (
                _source_font_label_catalog_version(
                    self._labels,
                    label_count=int(font_prob.size),
                )
            )
            self._source_font_label_catalog_version = (
                source_font_label_catalog_version
            )
        source_font_score_support = _build_source_font_score_support(
            font_prob,
            self._labels,
            catalog_version=source_font_label_catalog_version,
        )
        family_posterior = _font_family_posterior_from_probabilities(
            font_prob,
            self._labels,
        )
        factorized_taxonomy = getattr(
            self,
            "_factorized_attribute_taxonomy",
            None,
        )
        if (
            not isinstance(
                factorized_taxonomy,
                _FactorizedAttributeTaxonomy,
            )
            or factorized_taxonomy.label_count != int(font_prob.size)
        ):
            factorized_taxonomy = _build_factorized_attribute_taxonomy(
                self._labels,
                label_count=int(font_prob.size),
            )
            self._factorized_attribute_taxonomy = factorized_taxonomy
        model_attribute_posterior = (
            _factorized_attribute_posterior_from_probabilities(
                font_prob,
                factorized_taxonomy,
            )
        )
        top_indices = np.argsort(-font_prob)[:5]
        top_candidates: list[dict[str, Any]] = []
        for index in top_indices:
            label = _label_at(self._labels, int(index))
            top_candidates.append(
                {
                    "index": int(index),
                    "confidence": float(font_prob[int(index)]),
                    "path": str(label.get("path") or ""),
                    "language": str(label.get("language") or ""),
                    "serif": (
                        label.get("serif")
                        if isinstance(label.get("serif"), bool)
                        else None
                    ),
                }
            )
        direction_prob = _softmax(vector[FONT_COUNT : FONT_COUNT + 2])
        direction_index = int(direction_prob.argmax())
        direction_confidence = float(direction_prob[direction_index])
        regression = vector[FONT_COUNT + 2 : FONT_COUNT + 12]
        angle_ratio = _unit_interval(regression[9])
        top = top_candidates[0] if top_candidates else {}
        detection = {
            "font_index": int(top_indices[0]) if len(top_indices) else -1,
            "confidence": float(top.get("confidence") or 0.0),
            "font_path": str(top.get("path") or ""),
            "font_language": str(top.get("language") or ""),
            "font_serif": (
                top.get("serif") if isinstance(top.get("serif"), bool) else None
            ),
            "family_posterior": family_posterior.to_audit_dict(),
            SOURCE_FONT_SCORE_SUPPORT_KEY: source_font_score_support,
            "model_attribute_posterior": model_attribute_posterior,
            "top_candidates": top_candidates,
            "direction": (
                ("ltr" if direction_index == 0 else "ttb")
                if direction_confidence > 0.0
                else ""
            ),
            "direction_confidence": direction_confidence,
            "text_color": _rgb_from_unit_values(regression[0:3]),
            "text_size_ratio": _unit_interval(regression[3]),
            "stroke_width_ratio": _unit_interval(regression[4]),
            "stroke_color": _rgb_from_unit_values(regression[5:8]),
            "line_spacing_ratio": _unit_interval(regression[8]),
            "angle_degrees": (
                round((angle_ratio - 0.5) * 180.0, 3)
                if angle_ratio is not None
                else None
            ),
        }
        affinity, affinity_error = self._target_font_affinity_observation(
            font_logits=vector[:FONT_COUNT],
            source_input_sha256=_image_sha256(image),
        )
        if affinity is not None:
            detection[TARGET_FONT_AFFINITY_OBSERVATION_KEY] = affinity
        if affinity_error:
            detection[TARGET_FONT_AFFINITY_ERROR_KEY] = affinity_error
        return detection


_SESSION_CACHE: dict[tuple[str, bool], Any] = {}
_SESSION_PROVIDER_METADATA: dict[tuple[str, bool], dict[str, Any]] = {}


def _scale_support_is_supported(value: str) -> bool:
    return str(value or "").startswith("supported_")


def _observation_axis_records(
    observation: Any,
    *,
    view: AuthorizedSourceStyleView,
) -> tuple[SourceStyleAxisEvidence, ...]:
    records = tuple(getattr(observation, "axis_evidence", ()) or ())
    if records:
        return records
    footprint = getattr(observation, "source_text_footprint", None)
    support_identity = {
        "page_id": view.page_id,
        "view_id": view.view_id,
        "bundle_id": view.bundle_id,
        "parent_id": view.parent_id,
        "root_id": view.root_id,
        "authorized_mask_sha256": str(
            getattr(footprint, "authorized_mask_sha256", "") or ""
        ),
        "authorized_pixel_sha256": str(
            getattr(footprint, "authorized_pixel_sha256", "") or ""
        ),
        "detector_input_sha256": str(
            getattr(observation, "detector_input_sha256", "") or ""
        ),
    }

    def unavailable(axis: str) -> SourceStyleAxisEvidence:
        return SourceStyleAxisEvidence.unavailable(
            axis,
            provenance=f"authorized_source_style_view:legacy_{axis}_projection",
            support_identity=support_identity,
            reason_codes=(f"source_{axis}_axis_unavailable",),
        )

    scale_confidence = max(
        float(getattr(observation, "source_cell_confidence_vertical", 0.0) or 0.0),
        float(getattr(observation, "source_cell_confidence_horizontal", 0.0) or 0.0),
    )
    scale = (
        SourceStyleAxisEvidence(
            axis="scale",
            status="supported",
            value={
                "vertical_px": float(
                    getattr(observation, "source_cell_size_vertical_px", 0.0)
                    or 0.0
                ),
                "horizontal_px": float(
                    getattr(observation, "source_cell_size_horizontal_px", 0.0)
                    or 0.0
                ),
                "vertical_confidence": float(
                    getattr(
                        observation,
                        "source_cell_confidence_vertical",
                        0.0,
                    )
                    or 0.0
                ),
                "horizontal_confidence": float(
                    getattr(
                        observation,
                        "source_cell_confidence_horizontal",
                        0.0,
                    )
                    or 0.0
                ),
                "vertical_support": str(
                    getattr(observation, "source_cell_support_vertical", "")
                    or ""
                ),
                "horizontal_support": str(
                    getattr(observation, "source_cell_support_horizontal", "")
                    or ""
                ),
            },
            confidence=scale_confidence,
            provenance="authorized_source_style_view:legacy_scale_projection",
            support_identity=support_identity,
        )
        if scale_confidence > 0.0
        else unavailable("scale")
    )
    fill_confidence = float(
        getattr(observation, "paint_confidence", 0.0) or 0.0
    )
    fill_color = str(getattr(observation, "fill_color", "") or "")
    fill = (
        SourceStyleAxisEvidence(
            axis="fill",
            status="supported",
            value={
                "color": fill_color,
                "support_color": str(
                    getattr(observation, "support_color", "") or ""
                ),
                "polarity": str(
                    getattr(observation, "fill_polarity", "") or ""
                ),
            },
            confidence=fill_confidence,
            provenance="authorized_source_style_view:legacy_fill_projection",
            support_identity=support_identity,
        )
        if fill_confidence > 0.0 and fill_color
        else unavailable("fill")
    )
    outline_confidence = float(
        getattr(observation, "stroke_confidence", 0.0) or 0.0
    )
    outline = (
        SourceStyleAxisEvidence(
            axis="outline",
            status="supported",
            value={
                "present": bool(
                    float(
                        getattr(observation, "source_stroke_width_px", 0.0)
                        or 0.0
                    )
                    > 0.0
                ),
                "color": str(
                    getattr(observation, "support_color", "") or ""
                ),
                "width_px": float(
                    getattr(observation, "source_stroke_width_px", 0.0)
                    or 0.0
                ),
            },
            confidence=outline_confidence,
            provenance="authorized_source_style_view:legacy_outline_projection",
            support_identity=support_identity,
        )
        if outline_confidence > 0.0
        else unavailable("outline")
    )
    weight_confidence = max(
        float(getattr(observation, "ink_weight_confidence", 0.0) or 0.0),
        float(
            getattr(observation, "ink_weight_confidence_vertical", 0.0) or 0.0
        ),
        float(
            getattr(observation, "ink_weight_confidence_horizontal", 0.0) or 0.0
        ),
    )
    weight = (
        SourceStyleAxisEvidence(
            axis="weight",
            status="supported",
            value={
                "class": str(
                    getattr(observation, "ink_weight_class", "") or ""
                ),
                "confidence": float(
                    getattr(observation, "ink_weight_confidence", 0.0) or 0.0
                ),
                "vertical_class": str(
                    getattr(observation, "ink_weight_class_vertical", "") or ""
                ),
                "vertical_confidence": float(
                    getattr(
                        observation,
                        "ink_weight_confidence_vertical",
                        0.0,
                    )
                    or 0.0
                ),
                "vertical_support": str(
                    getattr(observation, "ink_weight_support_vertical", "") or ""
                ),
                "horizontal_class": str(
                    getattr(observation, "ink_weight_class_horizontal", "") or ""
                ),
                "horizontal_confidence": float(
                    getattr(
                        observation,
                        "ink_weight_confidence_horizontal",
                        0.0,
                    )
                    or 0.0
                ),
                "horizontal_support": str(
                    getattr(observation, "ink_weight_support_horizontal", "") or ""
                ),
                "source_ink_stroke_width_px": float(
                    getattr(observation, "source_ink_stroke_width_px", 0.0)
                    or 0.0
                ),
            },
            confidence=weight_confidence,
            provenance="authorized_source_style_view:legacy_weight_projection",
            support_identity=support_identity,
        )
        if weight_confidence > 0.0
        else unavailable("weight")
    )
    return (
        unavailable("family"),
        weight,
        scale,
        fill,
        outline,
        unavailable("orientation"),
        unavailable("rotation"),
        unavailable("shadow"),
    )


def _replace_axis_records(
    records: Sequence[SourceStyleAxisEvidence],
    replacements: Mapping[str, SourceStyleAxisEvidence],
) -> tuple[SourceStyleAxisEvidence, ...]:
    existing = {record.axis: record for record in records}
    existing.update(replacements)
    return tuple(existing[axis] for axis in SOURCE_STYLE_AXES)


def _validated_factorized_attribute_posterior(
    detection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(detection, Mapping):
        return {}
    raw = detection.get("model_attribute_posterior")
    if not isinstance(raw, Mapping):
        return {}
    try:
        label_count = int(raw.get("label_count") or 0)
    except (TypeError, ValueError):
        return {}
    if (
        str(raw.get("schema_version") or "")
        != FACTORIZED_ATTRIBUTE_POSTERIOR_VERSION
        or str(raw.get("taxonomy_version") or "")
        != FACTORIZED_ATTRIBUTE_TAXONOMY_VERSION
        or label_count != FONT_COUNT
    ):
        return {}
    expected_classes = {
        "generic_family": ("sans", "serif"),
        "face_character": (
            "standard_candidate",
            "slender_candidate",
        ),
        "weight_strength": (
            "normal_candidate",
            "strong_candidate",
        ),
    }
    for axis, required_classes in expected_classes.items():
        record = raw.get(axis)
        if not isinstance(record, Mapping):
            return {}
        raw_classes = record.get("classes")
        if not isinstance(raw_classes, Sequence) or isinstance(
            raw_classes,
            (str, bytes),
        ):
            return {}
        classes = tuple(raw_classes)
        masses = record.get("masses")
        conditional = record.get("conditional_probabilities")
        if (
            str(record.get("schema_version") or "")
            != "yuzumarker_factorized_attribute_axis_v1"
            or classes != required_classes
            or not isinstance(masses, Mapping)
            or not isinstance(conditional, Mapping)
        ):
            return {}
        try:
            axis_label_count = int(record.get("label_count") or 0)
            known_label_count = int(
                record.get("known_label_count") or 0
            )
            unknown_label_count = int(
                record.get("unknown_label_count") or 0
            )
            known_masses = [
                float(masses.get(name) or 0.0)
                for name in required_classes
            ]
            unknown_mass = float(masses.get("unknown") or 0.0)
            known_mass = float(record.get("known_mass") or 0.0)
            recorded_unknown_mass = float(
                record.get("unknown_mass") or 0.0
            )
            conditional_values = [
                float(conditional.get(name) or 0.0)
                for name in required_classes
            ]
            margin = float(record.get("margin") or 0.0)
            normalized_entropy = float(
                record.get("normalized_entropy") or 0.0
            )
        except (TypeError, ValueError):
            return {}
        numeric_values = [
            *known_masses,
            unknown_mass,
            known_mass,
            recorded_unknown_mass,
            *conditional_values,
            margin,
            normalized_entropy,
        ]
        mass_total = sum(known_masses) + unknown_mass
        conditional_total = sum(conditional_values)
        if (
            axis_label_count != FONT_COUNT
            or min(known_label_count, unknown_label_count) < 0
            or known_label_count + unknown_label_count != FONT_COUNT
            or any(
                not math.isfinite(value)
                or value < 0.0
                or value > 1.0 + 1e-6
                for value in numeric_values
            )
            or abs(mass_total - 1.0) > 1e-6
            or abs(sum(known_masses) - known_mass) > 1e-6
            or abs(unknown_mass - recorded_unknown_mass) > 1e-6
            or (
                abs(conditional_total - 1.0) > 1e-6
                if known_mass > 0.0
                else abs(conditional_total) > 1e-6
            )
            or str(record.get("leading_candidate") or "")
            not in {"", *required_classes}
        ):
            return {}
    return _copy_jsonish(raw)


def _validated_normalized_stroke_profile_v2(
    direct_weight_axis: SourceStyleAxisEvidence | None,
) -> dict[str, Any]:
    if direct_weight_axis is None:
        return {}
    raw_value = direct_weight_axis.value
    raw_identity = direct_weight_axis.support_identity
    if not isinstance(raw_value, Mapping) or not isinstance(
        raw_identity,
        Mapping,
    ):
        return {}
    raw_profile = raw_value.get(NORMALIZED_STROKE_PROFILE_V2)
    if not isinstance(raw_profile, Mapping):
        return {}
    if (
        str(raw_profile.get("schema_version") or "")
        != NORMALIZED_STROKE_PROFILE_V2
    ):
        return {}
    authorized_mask_sha256 = str(
        raw_identity.get("authorized_mask_sha256") or ""
    )
    source_core_mask_sha256 = str(
        raw_profile.get("source_core_mask_sha256") or ""
    )
    if (
        not authorized_mask_sha256
        or str(raw_profile.get("authorized_mask_sha256") or "")
        != authorized_mask_sha256
        or not source_core_mask_sha256
    ):
        return {}
    directions = raw_profile.get("directions")
    if not isinstance(directions, Mapping) or any(
        not isinstance(directions.get(direction), Mapping)
        for direction in ("ttb", "ltr")
    ):
        return {}

    identity_payload = {
        "authorized_mask_sha256": authorized_mask_sha256,
        "authorized_pixel_sha256": str(
            raw_identity.get("authorized_pixel_sha256") or ""
        ),
        "crop_shape": list(raw_identity.get("crop_shape") or ()),
        "source_core_mask_sha256": source_core_mask_sha256,
    }
    encoded_identity = json.dumps(
        identity_payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected_identity_sha256 = hashlib.sha256(encoded_identity).hexdigest()
    if (
        str(raw_profile.get("source_identity_sha256") or "")
        != expected_identity_sha256
    ):
        return {}
    return _copy_jsonish(raw_profile)


def _v3_source_optical_fact(
    direct_weight_axis: SourceStyleAxisEvidence | None,
    *,
    direction: str,
    bundle_id: str,
) -> dict[str, Any] | None:
    """Extract one identity-bound, scale-independent source-stroke fact."""

    direction_key = str(direction or "").strip().lower()
    if direction_key not in {"ttb", "ltr"}:
        return None
    profile = _validated_normalized_stroke_profile_v2(
        direct_weight_axis
    )
    directions = profile.get("directions")
    if not isinstance(directions, Mapping):
        return None
    raw = directions.get(direction_key)
    if not isinstance(raw, Mapping):
        return None
    measurement_state = str(
        raw.get("measurement_state") or ""
    ).strip().lower()
    if measurement_state not in {
        "supported",
        "provisional",
        "unclassified",
    }:
        return None
    distribution = raw.get("medial_width_to_cell")
    if not isinstance(distribution, Mapping) or (
        distribution.get("available") is not True
    ):
        return None
    try:
        p20 = float(distribution.get("p20"))
        median = float(distribution.get("median"))
        p80 = float(distribution.get("p80"))
        local_cell = float(raw.get("local_cell_reference_px"))
        confidence = float(raw.get("measurement_confidence"))
    except (TypeError, ValueError):
        return None
    if (
        not all(
            math.isfinite(value)
            for value in (p20, median, p80, local_cell, confidence)
        )
        or not (0.0 < p20 <= median <= p80 <= 1.0)
        or local_cell <= 0.0
        or not (0.0 <= confidence <= 1.0)
    ):
        return None
    agreement_record = raw.get("estimator_agreement")
    estimator_agreement = (
        str(agreement_record.get("status") or "").strip().lower()
        if isinstance(agreement_record, Mapping)
        else ""
    )
    if estimator_agreement not in {
        "corroborating",
        "divergent",
        "unavailable",
    }:
        estimator_agreement = "unavailable"
    tolerance = min(1.0, 0.5 / local_cell)
    source_identity_sha256 = str(
        profile.get("source_identity_sha256") or ""
    )
    if not source_identity_sha256:
        return None
    return {
        "bundle_id": str(bundle_id or ""),
        "direction": direction_key,
        "measurement_state": measurement_state,
        "measurement_confidence": confidence,
        "estimator_agreement": estimator_agreement,
        "local_cell_reference_px": local_cell,
        "resolution_tolerance_ratio": tolerance,
        "p20": p20,
        "median": median,
        "p80": p80,
        "interval": (p20, p80),
        "expanded_interval": (
            max(1e-8, p20 - tolerance),
            min(1.0, p80 + tolerance),
        ),
        "source_identity_sha256": source_identity_sha256,
        "modality_status": str(
            raw.get("modality_status") or ""
        ).strip(),
        "singleton_authoritative": bool(
            measurement_state == "supported"
            and estimator_agreement == "corroborating"
        ),
    }


def _canonical_source_style_sha256(value: Any) -> str:
    encoded = json.dumps(
        _copy_jsonish(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_font_observation_v3(
    *,
    direct_weight_axis: SourceStyleAxisEvidence | None,
    support_identity: Mapping[str, Any],
    primary_detection: Mapping[str, Any],
    neutral_detection: Mapping[str, Any] | None,
    primary_input_sha256: str,
    neutral_input_sha256: str,
    neutral_error: str,
) -> SourceFontObservationV3 | None:
    primary = primary_detection.get(SOURCE_FONT_SCORE_SUPPORT_KEY)
    if not isinstance(primary, SourceFontScoreSupportV1):
        return None
    neutral = (
        neutral_detection.get(SOURCE_FONT_SCORE_SUPPORT_KEY)
        if isinstance(neutral_detection, Mapping)
        else None
    )
    if neutral is not None and not isinstance(
        neutral,
        SourceFontScoreSupportV1,
    ):
        neutral = None
    target_font_affinity = primary_detection.get(
        TARGET_FONT_AFFINITY_OBSERVATION_KEY
    )
    if not isinstance(
        target_font_affinity,
        TargetFontAffinityObservationV1,
    ):
        target_font_affinity = None
    elif (
        target_font_affinity.source_input_sha256
        != str(primary_input_sha256 or "")
        or target_font_affinity.model_identity != YUZUMARKER_PROVIDER_MODEL
        or target_font_affinity.label_catalog_version
        != primary.catalog_version
    ):
        target_font_affinity = None

    authorized_view_sha256 = _canonical_source_style_sha256(
        dict(support_identity or {})
    )
    profile = _validated_normalized_stroke_profile_v2(
        direct_weight_axis
    )
    profile_source_identity = str(
        profile.get("source_identity_sha256") or ""
    )
    source_identity_sha256 = (
        profile_source_identity
        or _canonical_source_style_sha256(
            {
                "authorized_view_sha256": authorized_view_sha256,
                "primary_input_sha256": str(primary_input_sha256 or ""),
            }
        )
    )
    style_binding = (
        SourceStyleEvidenceBindingV1(
            source_identity_sha256=source_identity_sha256,
            evidence_schema_version=NORMALIZED_STROKE_PROFILE_V2,
            evidence_sha256=_canonical_source_style_sha256(profile),
        )
        if profile
        else None
    )
    overlap_bounds = (
        source_font_overlap_bounds(primary, neutral)
        if neutral is not None
        else None
    )
    variant_agreement = (
        "same_leading_identity"
        if (
            neutral is not None
            and primary.leading_candidate.label_identity
            == neutral.leading_candidate.label_identity
        )
        else "different_leading_identity"
        if neutral is not None
        else "neutral_unavailable"
    )
    return SourceFontObservationV3(
        source_identity_sha256=source_identity_sha256,
        authorized_view_sha256=authorized_view_sha256,
        model_identity=YUZUMARKER_PROVIDER_MODEL,
        label_catalog_version=primary.catalog_version,
        support_policy_version=SOURCE_FONT_SUPPORT_POLICY_VERSION,
        primary_input_sha256=str(primary_input_sha256 or ""),
        neutral_input_sha256=(
            str(neutral_input_sha256 or "") if neutral is not None else ""
        ),
        primary=primary,
        neutral=neutral,
        neutral_error=(
            str(neutral_error or "")
            if neutral is not None
            else str(
                neutral_error
                or "neutral_source_font_identity_support_unavailable"
            )
        ),
        variant_agreement=variant_agreement,
        variant_overlap_bounds=overlap_bounds,
        target_font_affinity=target_font_affinity,
        style_evidence_binding=style_binding,
    )


def _factorized_variant_agreement(
    primary: Mapping[str, Any],
    neutral: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        axis: bool(
            str(dict(primary.get(axis) or {}).get("leading_candidate") or "")
            and str(
                dict(primary.get(axis) or {}).get("leading_candidate") or ""
            )
            == str(
                dict(neutral.get(axis) or {}).get("leading_candidate") or ""
            )
        )
        for axis in ("generic_family", "face_character", "weight_strength")
    }


def _source_style_evidence_v2_carrier(
    *,
    direct_weight_axis: SourceStyleAxisEvidence | None,
    primary_detection: Mapping[str, Any],
    neutral_detection: Mapping[str, Any] | None,
    primary_input_sha256: str,
    neutral_input_sha256: str,
    neutral_error: str,
) -> dict[str, Any]:
    profile = _validated_normalized_stroke_profile_v2(
        direct_weight_axis
    )
    primary = _validated_factorized_attribute_posterior(
        primary_detection
    )
    neutral = _validated_factorized_attribute_posterior(
        neutral_detection
    )
    if not profile and not primary and not neutral:
        return {}
    model_variants = {
        "schema_version": FACTORIZED_ATTRIBUTE_VARIANTS_VERSION,
        "taxonomy_version": FACTORIZED_ATTRIBUTE_TAXONOMY_VERSION,
        "primary": primary,
        "neutral": neutral,
        "primary_input_sha256": str(primary_input_sha256 or ""),
        "neutral_input_sha256": str(neutral_input_sha256 or ""),
        "neutral_error": str(neutral_error or ""),
        "variant_agreement": (
            _factorized_variant_agreement(primary, neutral)
            if primary and neutral
            else {}
        ),
    }
    return {
        "schema_version": SOURCE_STYLE_EVIDENCE_V2,
        "source_only": True,
        "resolved_target_style": False,
        "geometry_profile": (
            {NORMALIZED_STROKE_PROFILE_V2: profile}
            if profile
            else {}
        ),
        "model_attribute_posterior": model_variants,
        "source_support_identity": (
            _copy_jsonish(direct_weight_axis.support_identity)
            if profile and direct_weight_axis is not None
            else {}
        ),
    }


def _axis_with_source_style_evidence_v2(
    record: SourceStyleAxisEvidence,
    carrier: Mapping[str, Any],
) -> SourceStyleAxisEvidence:
    if not carrier:
        return record
    value = _copy_jsonish(record.value)
    value[SOURCE_STYLE_EVIDENCE_V2] = _copy_jsonish(carrier)
    return SourceStyleAxisEvidence(
        axis=record.axis,
        status=record.status,
        value=value,
        confidence=record.confidence,
        provenance=record.provenance,
        support_identity=record.support_identity,
        reason_codes=record.reason_codes,
        support=record.support,
    )


def _direct_only_style_evidence(
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    view: AuthorizedSourceStyleView,
    observation: Any,
    detector_reason: str,
    detector_input_sha256: str = "",
) -> StyleEvidence | None:
    records = _observation_axis_records(observation, view=view)
    supported = [record for record in records if record.supported]
    if not supported:
        return None
    by_axis = {record.axis: record for record in records}
    scale = dict(by_axis["scale"].value)
    fill = dict(by_axis["fill"].value)
    outline = dict(by_axis["outline"].value)
    weight = dict(by_axis["weight"].value)
    reasons = _unique_strings(
        [
            "authorized_source_style_view_observed_partial_detector_unavailable",
            detector_reason,
            *list(getattr(observation, "reason_codes", ()) or ()),
        ]
    )
    return StyleEvidence(
        page_id=str(page_id or ""),
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        status="observed",
        vote_eligible=True,
        reason_codes=tuple(reasons),
        view_id=view.view_id,
        cleanup_mask_ids=tuple(view.cleanup_mask_ids),
        owned_component_ids=tuple(view.owned_component_ids),
        content_bbox=tuple(view.content_bbox),
        analysis_bbox=tuple(view.analysis_bbox),
        detector_input_sha256=str(
            detector_input_sha256
            or getattr(observation, "detector_input_sha256", "")
            or ""
        ),
        source_text_footprint=getattr(observation, "source_text_footprint", None),
        source_advance_grid=getattr(observation, "source_advance_grid", None),
        evidence_provider="AuthorizedSourceStyleObserver",
        evidence_source="authorized_source_style_view_independent_axes",
        confidence=max(record.confidence for record in supported),
        font_weight=str(weight.get("class") or ""),
        direction="",
        direction_confidence=0.0,
        text_color=_hex_color(fill.get("color")),
        stroke_color=_hex_color(
            outline.get("color") or fill.get("support_color")
        ),
        source_size_px=0.0,
        source_size_vertical_px=float(scale.get("vertical_px") or 0.0),
        source_size_horizontal_px=float(scale.get("horizontal_px") or 0.0),
        source_size_confidence_vertical=float(
            scale.get("vertical_confidence") or 0.0
        ),
        source_size_confidence_horizontal=float(
            scale.get("horizontal_confidence") or 0.0
        ),
        source_size_support_vertical=str(scale.get("vertical_support") or ""),
        source_size_support_horizontal=str(
            scale.get("horizontal_support") or ""
        ),
        source_stroke_width_px=float(outline.get("width_px") or 0.0),
        source_ink_stroke_width_px=float(
            weight.get("source_ink_stroke_width_px") or 0.0
        ),
        axis_confidence={record.axis: record.confidence for record in records},
        axis_provenance={record.axis: record.provenance for record in records},
        observation_summary=observation.to_audit_dict(),
        detector_variant_summary={
            "status": "unavailable",
            "reason": detector_reason,
        },
        axis_evidence=records,
    )


def _direct_or_unavailable_style_evidence(
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    view: AuthorizedSourceStyleView,
    observation: Any,
    detector_reason: str,
    detector_input_sha256: str = "",
) -> StyleEvidence:
    direct = _direct_only_style_evidence(
        page_id=page_id,
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        view=view,
        observation=observation,
        detector_reason=detector_reason,
        detector_input_sha256=detector_input_sha256,
    )
    if direct is not None:
        return direct
    return StyleEvidence.unavailable(
        page_id=page_id,
        bundle_id=bundle_id,
        parent_id=parent_id,
        root_id=root_id,
        reason_codes=(detector_reason,),
        view=view,
        detector_input_sha256=detector_input_sha256,
        source_text_footprint=getattr(observation, "source_text_footprint", None),
    )


def observe_parent_style_evidence(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: Sequence[Any],
    authorized_style_views: Mapping[str, AuthorizedSourceStyleView] | Any,
    mode: str,
    use_gpu: bool = False,
    models_dir: str | None = None,
    detector: Any | None = None,
) -> ParentStyleEvidenceRunResult:
    """Observe style axes only through authorized parent foreground."""

    normalized_mode = str(mode or "off").strip().lower()
    result = ParentStyleEvidenceRunResult(page_id=str(page_id or ""), mode=normalized_mode)
    bundles = list(parent_execution_bundles or [])
    views = _authorized_view_mapping(authorized_style_views)
    if normalized_mode == "off":
        result.evidence = [
            StyleEvidence.unavailable(
                page_id=page_id,
                bundle_id=str(getattr(bundle, "bundle_id", "") or ""),
                parent_id=str(getattr(bundle, "parent_id", "") or ""),
                root_id=str(getattr(bundle, "root_id", "") or ""),
                reason_codes=("font_detection_disabled",),
                view=views.get(str(getattr(bundle, "bundle_id", "") or "")),
            )
            for bundle in bundles
            if bool(getattr(bundle, "render_required", False))
        ]
        return result
    if normalized_mode not in {"yuzumarker", "heuristic"}:
        result.errors.append(f"unsupported_font_detection_mode:{normalized_mode}")
        result.evidence = [
            StyleEvidence.unavailable(
                page_id=page_id,
                bundle_id=str(getattr(bundle, "bundle_id", "") or ""),
                parent_id=str(getattr(bundle, "parent_id", "") or ""),
                root_id=str(getattr(bundle, "root_id", "") or ""),
                reason_codes=("unsupported_font_detection_mode",),
                view=views.get(str(getattr(bundle, "bundle_id", "") or "")),
            )
            for bundle in bundles
            if bool(getattr(bundle, "render_required", False))
        ]
        return result
    result.enabled = True

    active_detector = detector
    detector_initialization_attempted = active_detector is not None
    if normalized_mode == "yuzumarker" and active_detector is not None:
        result.model_path = str(getattr(active_detector, "model_path", "") or "")
        result.labels_path = str(getattr(active_detector, "labels_path", "") or "")
        _copy_provider_metadata(result, active_detector)

    image = None
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        result.errors.append(f"image_open_failed:{type(exc).__name__}:{exc}")

    for bundle in bundles:
        if not bool(getattr(bundle, "render_required", False)):
            continue
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        parent_id = str(getattr(bundle, "parent_id", "") or bundle_id)
        root_id = str(getattr(bundle, "root_id", "") or "")
        view = views.get(bundle_id)
        invalid_reasons = _authorized_view_rejection_reasons(
            view,
            page_id=page_id,
            bundle_id=bundle_id,
            parent_id=parent_id,
            root_id=root_id,
            image=image,
        )
        if invalid_reasons:
            result.evidence.append(
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=invalid_reasons,
                    view=view,
                )
            )
            continue
        observation_inputs = build_authorized_style_observation_inputs(image, view)
        source_text_footprint = getattr(
            observation_inputs, "source_text_footprint", None
        )
        detector_input = observation_inputs.primary_input
        if detector_input is None or not observation_inputs.available:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="authorized_detector_input_unavailable",
                )
            )
            continue
        actual_detector_input_sha256 = _image_sha256(detector_input)
        declared_detector_input_sha256 = str(
            getattr(observation_inputs, "detector_input_sha256", "") or ""
        )
        if (
            not declared_detector_input_sha256
            or declared_detector_input_sha256 != actual_detector_input_sha256
        ):
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="authorized_detector_input_identity_mismatch",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        if (
            normalized_mode == "yuzumarker"
            and active_detector is None
            and not detector_initialization_attempted
        ):
            detector_initialization_attempted = True
            try:
                active_detector = YuzuMarkerOnnxFontDetector(
                    model_path=resolve_yuzumarker_font_onnx_file(models_dir),
                    labels_path=resolve_yuzumarker_font_labels_file(models_dir),
                    use_gpu=use_gpu,
                )
                result.model_path = str(getattr(active_detector, "model_path", "") or "")
                result.labels_path = str(getattr(active_detector, "labels_path", "") or "")
                _copy_provider_metadata(result, active_detector)
            except Exception as exc:
                result.errors.append(f"yuzumarker_unavailable:{type(exc).__name__}:{exc}")
                active_detector = None
        if normalized_mode == "yuzumarker" and active_detector is None:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="yuzumarker_detector_unavailable",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        neutral_detection: Mapping[str, Any] | None = None
        neutral_error = ""
        try:
            detection = (
                active_detector.detect(detector_input)
                if normalized_mode == "yuzumarker"
                else _heuristic_detection(detector_input)
            )
            if normalized_mode == "yuzumarker":
                try:
                    neutral_value = active_detector.detect(observation_inputs.neutral_input)
                    if isinstance(neutral_value, Mapping):
                        neutral_detection = neutral_value
                    else:
                        neutral_error = "neutral_detector_output_contract_invalid"
                except Exception as exc:
                    neutral_error = f"neutral_style_detector_failed:{type(exc).__name__}"
            else:
                neutral_detection = detection if isinstance(detection, Mapping) else None
        except Exception as exc:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason=f"style_detector_failed:{type(exc).__name__}",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        if not isinstance(detection, Mapping):
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="style_detector_output_contract_invalid",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        family_observation = _family_axis_from_variants(
            detection,
            neutral_detection,
        )
        confidence_value = _unit_interval(detection.get("confidence"))
        exact_font_reason = ""
        if confidence_value is None:
            exact_font_reason = "font_model_confidence_contract_invalid"
        elif confidence_value < MIN_STYLE_EVIDENCE_CONFIDENCE:
            exact_font_reason = "font_model_confidence_below_observation_floor"
        if exact_font_reason and family_observation.posterior.label_count <= 0:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason=exact_font_reason,
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        exact_font_observation_qualified = not exact_font_reason
        confidence = float(confidence_value or 0.0)
        provider = YUZUMARKER_PROVIDER if normalized_mode == "yuzumarker" else HEURISTIC_PROVIDER
        source = YUZUMARKER_STYLE_SOURCE if normalized_mode == "yuzumarker" else HEURISTIC_STYLE_SOURCE
        model = YUZUMARKER_PROVIDER_MODEL if normalized_mode == "yuzumarker" else ""
        analysis_width = int(view.analysis_bbox[2]) if len(view.analysis_bbox) == 4 else int(detector_input.width)
        font_label = str(detection.get("font_path") or "")
        if exact_font_observation_qualified:
            model_weight, model_weight_confidence, model_weight_reason = _weight_axis_from_variants(
                detection,
                neutral_detection,
            )
        else:
            model_weight = None
            model_weight_confidence = 0.0
            model_weight_reason = exact_font_reason
        font_serif = family_observation.font_serif
        family_confidence = family_observation.confidence
        family_reason = family_observation.reason
        if exact_font_observation_qualified:
            direction, direction_confidence, direction_reason = _orientation_axis_from_variants(
                detection,
                neutral_detection,
            )
        else:
            direction = ""
            direction_confidence = 0.0
            direction_reason = exact_font_reason
        (
            direct_weight,
            direct_weight_confidence,
            direct_weight_support,
        ) = observation_inputs.ink_weight_measurement_for_direction(direction)
        direct_weight = str(direct_weight or "").strip().lower()
        if direct_weight in {"regular", "bold"} and direct_weight_confidence > 0.0:
            parsed_weight = direct_weight
            weight_confidence = direct_weight_confidence
            weight_reason = "weight_authorized_ink_geometry_measured"
        else:
            parsed_weight = model_weight
            weight_confidence = model_weight_confidence
            weight_reason = model_weight_reason
        (
            source_size_px,
            source_scale_confidence,
            source_scale_axis,
        ) = observation_inputs.source_cell_measurement_for_direction(direction)
        source_scale_support_status = (
            observation_inputs.source_cell_support_for_direction(direction)
        )
        source_scale_supported = bool(
            source_size_px > 0.0
            and source_scale_confidence > 0.0
            and _scale_support_is_supported(source_scale_support_status)
        )
        text_size_ratio = (
            float(source_size_px) / float(max(1, analysis_width))
            if source_scale_supported
            else 0.0
        )
        source_stroke_width_px = max(
            0.0, float(observation_inputs.source_stroke_width_px)
        )
        source_ink_stroke_width_px = max(
            0.0, float(observation_inputs.source_ink_stroke_width_px)
        )
        stroke_width_ratio = source_stroke_width_px / float(max(1, analysis_width))
        line_spacing_ratio_value = _unit_interval(detection.get("line_spacing_ratio"))
        line_spacing_ratio = (
            float(line_spacing_ratio_value) if line_spacing_ratio_value is not None else 0.0
        )
        angle_value = _bounded_float(detection.get("angle_degrees"), minimum=-180.0, maximum=180.0)
        angle_degrees = float(angle_value) if angle_value is not None else 0.0
        text_color = _hex_color(observation_inputs.fill_color)
        stroke_color = _hex_color(observation_inputs.support_color)
        paint_valid = bool(text_color and observation_inputs.paint_confidence > 0.0)
        shared_axis_confidence = {
            "family": family_confidence,
            "weight": weight_confidence,
            "scale": (
                float(source_scale_confidence)
                if source_scale_supported
                else 0.0
            ),
            "paint": (
                float(observation_inputs.paint_confidence) if paint_valid else 0.0
            ),
            "stroke": (
                float(observation_inputs.stroke_confidence)
                if observation_inputs.stroke_confidence > 0.0
                else 0.0
            ),
            "orientation": direction_confidence,
        }
        axis_provenance = {
            "family": f"{provider}:complete_family_posterior_calibrated",
            "weight": (
                "authorized_source_style_view:fill_ink_stroke_geometry"
                if direct_weight in {"regular", "bold"}
                and direct_weight_confidence > 0.0
                else f"{provider}:fill_contrast_and_neutral_weight_vote"
                if parsed_weight is not None
                else "target_fallback:unresolved_source_weight_label"
            ),
            "scale": (
                "authorized_source_style_view:foreground_geometry_"
                f"qualified_{source_scale_axis}_cell_measurement"
                if source_scale_supported
                else "typesetting_default:source_scale_unavailable"
            ),
            "paint": (
                "authorized_source_style_view:authorized_core_paint_color_coherence"
                if paint_valid
                else "target_fallback:paint_axis_contract_invalid"
            ),
            "stroke": (
                "authorized_source_style_view:canonical_external_surface_carrier"
                if source_stroke_width_px > 0
                else "authorized_source_style_view:canonical_source_carrier_absent"
                if observation_inputs.stroke_confidence > 0.0
                else "target_fallback:stroke_axis_not_independently_supported"
            ),
            "orientation": (
                f"{provider}:fill_contrast_and_neutral_direction_vote"
                if direction in {"ltr", "ttb"}
                else "target_fallback:orientation_axis_contract_invalid"
            ),
        }
        evidence_reasons = [
            "authorized_source_style_view_observed",
            *list(observation_inputs.reason_codes),
        ]
        if exact_font_reason:
            evidence_reasons.append(exact_font_reason)
        if family_reason:
            evidence_reasons.append(family_reason)
        if weight_reason:
            evidence_reasons.append(weight_reason)
        if direction_reason:
            evidence_reasons.append(direction_reason)
        if neutral_error:
            evidence_reasons.append(neutral_error)
        target_affinity_error = str(
            detection.get(TARGET_FONT_AFFINITY_ERROR_KEY) or ""
        )
        if target_affinity_error:
            evidence_reasons.append(target_affinity_error)
        if parsed_weight is None:
            evidence_reasons.append("source_weight_label_unresolved")
        if not source_scale_supported:
            evidence_reasons.append("source_scale_axis_unavailable")
            if source_scale_support_status:
                evidence_reasons.append(source_scale_support_status)
        if not paint_valid:
            evidence_reasons.append("source_paint_axis_contract_invalid")
        if direction not in {"ltr", "ttb"}:
            evidence_reasons.append("source_orientation_axis_contract_invalid")
        neutral_input_sha256 = _image_sha256(
            observation_inputs.neutral_input
        )
        detector_variant_summary = _detector_variant_summary(
            detection,
            neutral_detection,
            primary_sha256=actual_detector_input_sha256,
            neutral_sha256=neutral_input_sha256,
            neutral_error=neutral_error,
        )
        direct_axis_records = _observation_axis_records(
            observation_inputs,
            view=view,
        )
        direct_by_axis = {
            record.axis: record for record in direct_axis_records
        }
        support_identity = dict(
            direct_axis_records[0].support_identity
            if direct_axis_records
            else {}
        )
        family_support = {
            "family_posterior": family_observation.posterior.to_audit_dict(),
            "family_calibration_rule": family_posterior_calibration_rule(),
            "calibration_reliability": family_observation.calibration_reliability,
            "variant_agreement": family_observation.variant_agreement,
            "exact_top_candidates_diagnostic_only": _compact_candidates(
                detection.get("top_candidates")
            ),
            "detector_variant_summary": detector_variant_summary,
        }
        family_axis = (
            SourceStyleAxisEvidence(
                axis="family",
                status="supported",
                value={
                    "family_role": family_observation.family_role,
                    "font_label": font_label,
                    "font_serif": font_serif,
                    "font_language": str(detection.get("font_language") or ""),
                    "family_posterior": (
                        family_observation.posterior.to_audit_dict()
                    ),
                },
                confidence=family_confidence,
                provenance=(
                    f"{provider}:complete_family_posterior_calibrated"
                ),
                support_identity=support_identity,
                reason_codes=(family_reason,) if family_reason else (),
                support=family_support,
            )
            if family_observation.promoted
            else SourceStyleAxisEvidence.unavailable(
                "family",
                provenance=(
                    f"{provider}:complete_family_posterior_calibrated"
                ),
                support_identity=support_identity,
                reason_codes=(
                    family_reason or "source_family_axis_unavailable",
                ),
                support=family_support,
            )
        )
        orientation_axis = (
            SourceStyleAxisEvidence(
                axis="orientation",
                status="supported",
                value={"direction": direction},
                confidence=direction_confidence,
                provenance=(
                    f"{provider}:independent_primary_neutral_orientation_observation"
                ),
                support_identity=support_identity,
                reason_codes=(direction_reason,) if direction_reason else (),
                support={"detector_variant_summary": detector_variant_summary},
            )
            if direction in {"ltr", "ttb"} and direction_confidence > 0.0
            else SourceStyleAxisEvidence.unavailable(
                "orientation",
                provenance=(
                    f"{provider}:independent_primary_neutral_orientation_observation"
                ),
                support_identity=support_identity,
                reason_codes=(
                    direction_reason or "source_orientation_axis_unavailable",
                ),
                support={"detector_variant_summary": detector_variant_summary},
            )
        )
        direct_weight_axis = direct_by_axis.get("weight")
        source_font_observation = _source_font_observation_v3(
            direct_weight_axis=direct_weight_axis,
            support_identity=support_identity,
            primary_detection=detection,
            neutral_detection=neutral_detection,
            primary_input_sha256=actual_detector_input_sha256,
            neutral_input_sha256=neutral_input_sha256,
            neutral_error=neutral_error,
        )
        source_font_style_evidence = (
            direct_weight_axis
            if _validated_normalized_stroke_profile_v2(
                direct_weight_axis
            )
            else None
        )
        direct_weight_value = (
            dict(direct_weight_axis.value)
            if direct_weight_axis is not None
            else {}
        )
        if (
            direct_weight_axis is not None
            and direct_weight_axis.supported
            and str(direct_weight_value.get("schema_version") or "")
            == "native_normalized_weight_evidence_v1"
        ):
            weight_axis = direct_weight_axis
        elif parsed_weight in {"regular", "bold"} and weight_confidence > 0.0:
            weight_axis = SourceStyleAxisEvidence(
                axis="weight",
                status="supported",
                value={
                    "class": parsed_weight,
                    "model_class": parsed_weight,
                },
                confidence=weight_confidence,
                provenance=(
                    f"{provider}:independent_primary_neutral_weight_observation"
                ),
                support_identity=support_identity,
                reason_codes=(weight_reason,) if weight_reason else (),
                support={"detector_variant_summary": detector_variant_summary},
            )
        else:
            weight_axis = SourceStyleAxisEvidence.unavailable(
                "weight",
                provenance=(
                    f"{provider}:independent_primary_neutral_weight_observation"
                ),
                support_identity=support_identity,
                reason_codes=(
                    weight_reason or "source_weight_axis_unavailable",
                ),
                support={"detector_variant_summary": detector_variant_summary},
            )
        axis_evidence = _replace_axis_records(
            direct_axis_records,
            {
                "family": family_axis,
                "weight": weight_axis,
                "orientation": orientation_axis,
            },
        )
        shared_axis_confidence = {
            record.axis: float(record.confidence)
            for record in axis_evidence
        }
        axis_provenance = {
            record.axis: record.provenance for record in axis_evidence
        }
        result.evidence.append(
            StyleEvidence(
                page_id=str(page_id or ""),
                bundle_id=bundle_id,
                parent_id=parent_id,
                root_id=root_id,
                status="observed",
                vote_eligible=True,
                reason_codes=tuple(evidence_reasons),
                view_id=view.view_id,
                cleanup_mask_ids=tuple(view.cleanup_mask_ids),
                owned_component_ids=tuple(view.owned_component_ids),
                content_bbox=tuple(view.content_bbox),
                analysis_bbox=tuple(view.analysis_bbox),
                detector_input_sha256=actual_detector_input_sha256,
                source_text_footprint=source_text_footprint,
                source_advance_grid=observation_inputs.source_advance_grid,
                source_font_observation=source_font_observation,
                source_font_style_evidence=source_font_style_evidence,
                authorized_perceptual_source_identity={},
                evidence_provider=provider,
                evidence_source=source,
                evidence_model=model,
                confidence=max(confidence, family_confidence),
                font_label=font_label,
                font_weight=parsed_weight or "",
                font_language=str(detection.get("font_language") or ""),
                font_serif=font_serif,
                family_posterior=family_observation.posterior,
                top_candidates=tuple(_compact_candidates(detection.get("top_candidates"))),
                direction=direction,
                direction_confidence=direction_confidence,
                text_color=text_color,
                stroke_color=stroke_color,
                text_size_ratio=text_size_ratio,
                source_size_px=source_size_px,
                source_size_vertical_px=float(
                    observation_inputs.source_cell_size_vertical_px
                ),
                source_size_horizontal_px=float(
                    observation_inputs.source_cell_size_horizontal_px
                ),
                source_size_confidence_vertical=float(
                    observation_inputs.source_cell_confidence_vertical
                ),
                source_size_confidence_horizontal=float(
                    observation_inputs.source_cell_confidence_horizontal
                ),
                source_size_support_vertical=str(
                    observation_inputs.source_cell_support_vertical or ""
                ),
                source_size_support_horizontal=str(
                    observation_inputs.source_cell_support_horizontal or ""
                ),
                source_scale_support_status=source_scale_support_status,
                source_stroke_width_px=source_stroke_width_px,
                source_ink_stroke_width_px=source_ink_stroke_width_px,
                stroke_width_ratio=stroke_width_ratio,
                line_spacing_ratio=line_spacing_ratio,
                angle_degrees=angle_degrees,
                axis_confidence=shared_axis_confidence,
                axis_provenance=axis_provenance,
                observation_summary=observation_inputs.to_audit_dict(),
                detector_variant_summary=detector_variant_summary,
                perceptual_axis_evidence={},
                axis_evidence=axis_evidence,
            )
        )

    try:
        if image is not None:
            image.close()
    except Exception:
        pass
    return result


def arbitrate_parent_styles(
    *,
    parent_execution_bundles: Sequence[Any],
    evidence: Sequence[StyleEvidence],
    default_font_name: str = "",
    models_dir: str | None = None,
) -> ParentStyleArbitrationResult:
    """Atomically activate the accepted v3 decision and realization path.

    ``default_font_name`` is intentionally ignored: target faces are selected
    only through the registered family/weight role matrix.  All styles are
    realized and centrally validated before any bundle is mutated.
    """

    _ = default_font_name
    bundles = tuple(
        bundle
        for bundle in tuple(parent_execution_bundles or ())
        if bool(getattr(bundle, "render_required", False))
    )
    evidence_items = tuple(evidence or ())
    decision_ledger = resolve_parent_style_decision_ledger_v3(
        parent_execution_bundles=bundles,
        evidence=evidence_items,
    )
    font_manager = FontManager(base_dir=models_dir)
    style_ledger = realize_parent_render_styles_v3(
        parent_execution_bundles=bundles,
        decision_ledger=decision_ledger,
        font_manager=font_manager,
    )
    return activate_parent_render_style_ledger_v3(
        parent_execution_bundles=bundles,
        evidence=evidence_items,
        style_ledger=style_ledger,
    )


def activate_parent_render_style_ledger_v3(
    *,
    parent_execution_bundles: Sequence[Any],
    evidence: Sequence[StyleEvidence],
    style_ledger: ParentRenderStyleLedgerV3,
) -> ParentStyleArbitrationResult:
    """Publish one fully realized v3 ledger as an all-or-nothing bundle step."""

    if not isinstance(style_ledger, ParentRenderStyleLedgerV3):
        raise TypeError("Stage 3C requires a ParentRenderStyleLedgerV3")
    bundles = tuple(parent_execution_bundles or ())
    evidence_items = tuple(evidence or ())
    evidence_by_id = {
        item.bundle_id: item
        for item in evidence_items
        if isinstance(item, StyleEvidence) and item.bundle_id
    }
    bundles_by_id = {
        str(getattr(bundle, "bundle_id", "") or ""): bundle
        for bundle in bundles
    }
    resolved: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    activation: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    for resolved_style in style_ledger.styles:
        bundle_id = resolved_style.bundle_id
        style = resolved_style.to_contract_dict()
        validation = validate_resolved_render_style(style)
        if not validation.accepted:
            raise ValueError(
                "Stage 3C rejected a realized v3 style: "
                f"{bundle_id}:{','.join(validation.reason_codes)}"
            )
        style_snapshot = _plain_json_mapping_snapshot(validation.style)
        evidence_item = evidence_by_id.get(bundle_id)
        evidence_snapshot = _plain_json_mapping_snapshot(
            evidence_item.to_audit_dict() if evidence_item is not None else {}
        )
        resolved[bundle_id] = style_snapshot
        activation.append((bundles_by_id[bundle_id], style_snapshot, evidence_snapshot))
        record = resolved_style.to_audit_dict()
        fallback_status = style_snapshot.get("fallback_status")
        record.update(
            {
                "status": (
                    "fallback"
                    if isinstance(fallback_status, Mapping)
                    and bool(fallback_status.get("used"))
                    else "applied"
                ),
                "style_evidence_status": resolved_style.source_evidence_status,
                "render_style": style_snapshot,
            }
        )
        records.append(_plain_json_mapping_snapshot(record))

    prior = [
        (
            bundle,
            getattr(bundle, "render_style", {}),
            getattr(bundle, "style_evidence_summary", {}),
            getattr(bundle, "execution_region", {}),
        )
        for bundle, _style, _evidence in activation
    ]
    try:
        for bundle, style_snapshot, evidence_snapshot in activation:
            bundle.render_style = _plain_json_mapping_snapshot(style_snapshot)
            bundle.style_evidence_summary = _plain_json_mapping_snapshot(
                evidence_snapshot
            )
        for bundle, _style_snapshot, _evidence_snapshot in activation:
            bundle.execution_region = bundle.to_region_record()
    except Exception:
        for bundle, old_style, old_evidence, old_region in prior:
            bundle.render_style = old_style
            bundle.style_evidence_summary = old_evidence
            bundle.execution_region = old_region
        raise
    return ParentStyleArbitrationResult(
        resolved_styles=resolved,
        records=tuple(records),
    )


def _collect_parent_axis_candidates(
    bundle: Any,
    evidence: StyleEvidence,
) -> _ParentAxisCandidates:
    """Project typed source observations into parent-local candidates.

    `StyleEvidence.axis_evidence` is the sole observed-style input. The
    flattened fields remain audit/transport projections and the historical
    perceptual carrier is deliberately not consulted here.
    """

    direct: dict[str, _AxisCandidate] = {}
    directional_weight: dict[str, _AxisCandidate] = {}
    directional_scale: dict[str, _AxisCandidate] = {}
    records = _typed_axis_record_map(bundle=bundle, evidence=evidence)
    family = records.get("family")
    if family is not None and family.confidence >= DIRECT_AXIS_MIN_CONFIDENCE:
        value = dict(family.value)
        font_serif = value.get("font_serif")
        if isinstance(font_serif, bool):
            direct["family"] = _AxisCandidate(
                axis="family",
                value="serif" if font_serif else "sans",
                confidence=family.confidence,
                provenance=family.provenance,
                source="direct",
                support_status="supported_typed_axis_evidence",
                reason_codes=family.reason_codes,
            )

    weight = records.get("weight")
    if weight is not None and weight.confidence >= DIRECT_AXIS_MIN_CONFIDENCE:
        value = dict(weight.value)
        if str(value.get("schema_version") or "") == (
            "native_normalized_weight_evidence_v1"
        ):
            weight_fact = _v3_weight_fact(weight, direction="ttb")
            if weight_fact is not None:
                score = float(weight_fact["score"])
                if score < V3_WEIGHT_BASE_SCORE_RANGE[0]:
                    relative_tier = "slender"
                elif score <= V3_WEIGHT_BASE_SCORE_RANGE[1]:
                    relative_tier = "base"
                elif score < V3_WEIGHT_HEAVY_SCORE_RANGE[0]:
                    relative_tier = "emphasis"
                else:
                    relative_tier = "heavy"
                direct["weight"] = _AxisCandidate(
                    axis="weight",
                    value=relative_tier,
                    confidence=float(weight_fact["confidence"]),
                    provenance=weight.provenance,
                    source="direct",
                    support_status="native_normalized_weight_evidence_v1",
                    reason_codes=weight.reason_codes,
                )
        weight_class = str(value.get("class") or "").strip().lower()
        if weight_class in {"regular", "bold", "black"}:
            direct["weight"] = _AxisCandidate(
                axis="weight",
                value=weight_class,
                confidence=weight.confidence,
                provenance=weight.provenance,
                source="direct",
                support_status="supported_typed_axis_evidence",
                reason_codes=weight.reason_codes,
            )
        for direction, prefix in (("ttb", "vertical"), ("ltr", "horizontal")):
            directional_class = str(
                value.get(f"{prefix}_class") or ""
            ).strip().lower()
            directional_confidence = float(
                _unit_interval(value.get(f"{prefix}_confidence"))
                or weight.confidence
            )
            support_status = str(value.get(f"{prefix}_support") or "")
            if (
                directional_class in {"regular", "bold", "black"}
                and directional_confidence >= DIRECT_AXIS_MIN_CONFIDENCE
                and (
                    not support_status
                    or _scale_support_is_supported(support_status)
                )
            ):
                directional_weight[direction] = _AxisCandidate(
                    axis="weight",
                    value=directional_class,
                    confidence=directional_confidence,
                    provenance=weight.provenance,
                    source="direct",
                    support_status=(
                        support_status or "supported_typed_axis_evidence"
                    ),
                    reason_codes=weight.reason_codes,
                )

    orientation = records.get("orientation")
    if (
        orientation is not None
        and orientation.confidence >= DIRECT_AXIS_MIN_CONFIDENCE
    ):
        direction = str(
            dict(orientation.value).get("direction") or ""
        ).strip().lower()
        if direction in {"ltr", "ttb"}:
            direct["orientation"] = _AxisCandidate(
                axis="orientation",
                value=direction,
                confidence=orientation.confidence,
                provenance=orientation.provenance,
                source="direct",
                support_status="supported_typed_axis_evidence",
                reason_codes=orientation.reason_codes,
            )

    fill = records.get("fill")
    if fill is not None and fill.confidence >= DIRECT_PAINT_MIN_CONFIDENCE:
        fill_color = _hex_color(dict(fill.value).get("color"))
        if fill_color:
            direct["fill"] = _AxisCandidate(
                axis="fill",
                value=fill_color,
                confidence=fill.confidence,
                provenance=fill.provenance,
                source="direct",
                support_status="supported_typed_axis_evidence",
                reason_codes=fill.reason_codes,
            )

    outline = records.get("outline")
    if outline is not None and outline.confidence >= DIRECT_OUTLINE_MIN_CONFIDENCE:
        value = dict(outline.value)
        width_px = _float(value.get("width_px"))
        present = value.get("present")
        if isinstance(present, bool) and not present:
            width_px = 0.0
        outline_color = _hex_color(value.get("color"))
        if width_px >= 0.0 and (outline_color or width_px == 0.0):
            direct["outline"] = _AxisCandidate(
                axis="outline",
                value={
                    "color": outline_color or "#FFFFFF",
                    "width_px": max(0.0, width_px),
                },
                confidence=outline.confidence,
                provenance=outline.provenance,
                source="direct",
                support_status="supported_typed_axis_evidence",
                reason_codes=outline.reason_codes,
            )

    scale = records.get("scale")
    if scale is not None and scale.confidence >= DIRECT_AXIS_MIN_CONFIDENCE:
        value = dict(scale.value)
        for direction, prefix in (("ttb", "vertical"), ("ltr", "horizontal")):
            numeric_value = _float(value.get(f"{prefix}_px"))
            numeric_confidence = float(
                _unit_interval(value.get(f"{prefix}_confidence"))
                or scale.confidence
            )
            support_status = str(value.get(f"{prefix}_support") or "")
            if (
                numeric_value > 0.0
                and numeric_confidence >= DIRECT_AXIS_MIN_CONFIDENCE
                and _scale_support_is_supported(support_status)
            ):
                directional_scale[direction] = _AxisCandidate(
                    axis="scale",
                    value=numeric_value,
                    confidence=numeric_confidence,
                    provenance=scale.provenance,
                    source="direct",
                    support_status=support_status,
                    reason_codes=scale.reason_codes,
                )

    for axis in ("rotation", "shadow"):
        record = records.get(axis)
        if record is None or record.confidence < DIRECT_AXIS_MIN_CONFIDENCE:
            continue
        value, reasons = _validated_perceptual_axis_value(
            axis,
            dict(record.value),
        )
        if value is None or reasons:
            continue
        direct[axis] = _AxisCandidate(
            axis=axis,
            value=value,
            confidence=record.confidence,
            provenance=record.provenance,
            source="direct",
            support_status="supported_typed_axis_evidence",
            reason_codes=record.reason_codes,
        )

    return _ParentAxisCandidates(
        direct=direct,
        directional_weight=directional_weight,
        directional_scale=directional_scale,
    )


def _typed_axis_record_map(
    *,
    bundle: Any,
    evidence: StyleEvidence,
) -> dict[str, SourceStyleAxisEvidence]:
    if evidence.status != "observed" or not evidence.vote_eligible:
        return {}
    grouped: dict[str, list[SourceStyleAxisEvidence]] = {}
    for record in tuple(evidence.axis_evidence or ()):
        if not isinstance(record, SourceStyleAxisEvidence):
            continue
        if record.axis not in SOURCE_STYLE_AXES:
            continue
        grouped.setdefault(record.axis, []).append(record)
    result: dict[str, SourceStyleAxisEvidence] = {}
    for axis in SOURCE_STYLE_AXES:
        records = grouped.get(axis, [])
        if len(records) != 1:
            continue
        record = records[0]
        if not record.supported or not record.provenance:
            continue
        if not _axis_support_identity_matches(
            bundle=bundle,
            evidence=evidence,
            record=record,
        ):
            continue
        result[axis] = record
    return result


def _axis_support_identity_matches(
    *,
    bundle: Any,
    evidence: StyleEvidence,
    record: SourceStyleAxisEvidence,
) -> bool:
    identity = record.support_identity
    if not isinstance(identity, Mapping):
        return False
    expected = {
        "page_id": str(getattr(bundle, "page_id", "") or ""),
        "view_id": evidence.view_id,
        "bundle_id": str(getattr(bundle, "bundle_id", "") or ""),
        "parent_id": str(getattr(bundle, "parent_id", "") or ""),
        "root_id": str(getattr(bundle, "root_id", "") or ""),
        "detector_input_sha256": evidence.detector_input_sha256,
    }
    if any(
        not expected_value
        or str(identity.get(key) or "") != expected_value
        for key, expected_value in expected.items()
    ):
        return False
    if not str(identity.get("authorized_mask_sha256") or ""):
        return False
    cleanup_mask_ids = identity.get("cleanup_mask_ids")
    if cleanup_mask_ids is not None and tuple(cleanup_mask_ids) != tuple(
        evidence.cleanup_mask_ids
    ):
        return False
    return True


def _collect_additive_axis_candidates(
    *,
    bundle: Any,
    evidence: StyleEvidence,
) -> tuple[dict[str, _AxisCandidate], dict[str, Any]]:
    raw_carrier = evidence.perceptual_axis_evidence
    if not raw_carrier:
        return {}, {}
    carrier = dict(raw_carrier) if isinstance(raw_carrier, Mapping) else {}
    global_reasons, carrier_fact_set_id = _perceptual_carrier_validation(
        bundle=bundle,
        evidence=evidence,
        raw_carrier=raw_carrier,
        carrier=carrier,
    )
    candidates: dict[str, _AxisCandidate] = {}
    axis_audits: dict[str, dict[str, Any]] = {}
    for axis in PERCEPTUAL_STYLE_AXES:
        value, audit = _resolve_perceptual_axis(
            axis=axis,
            record=carrier.get(axis),
            carrier_fact_set_id=carrier_fact_set_id,
            global_reasons=global_reasons,
        )
        axis_audits[axis] = audit
        if value is None:
            continue
        if axis == "fill":
            candidate_value: Any = str(value["color"])
        else:
            candidate_value = dict(value)
        candidates[axis] = _AxisCandidate(
            axis=axis,
            value=candidate_value,
            confidence=float(audit.get("confidence") or 0.0),
            provenance=str(audit.get("provenance") or ""),
            source="additive",
            support_status="supported",
            reason_codes=tuple(audit.get("reason_codes") or ()),
        )
    resolved_axes = [axis for axis in PERCEPTUAL_STYLE_AXES if axis in candidates]
    return candidates, {
        "contract_version": PERCEPTUAL_STYLE_RESOLUTION_VERSION,
        "source_contract_version": _plain_string(carrier.get("contract_version")),
        "carrier_status": "valid" if not global_reasons else "rejected",
        "resolved_axes": resolved_axes,
        "unavailable_axes": [
            axis for axis in PERCEPTUAL_STYLE_AXES if axis not in candidates
        ],
        **axis_audits,
    }


def _reconcile_parent_axis_decisions(
    *,
    bundles: Sequence[Any],
    evidence_by_bundle: Mapping[str, StyleEvidence],
    candidates_by_bundle: Mapping[str, _ParentAxisCandidates],
    local_decisions_by_bundle: Mapping[str, _ParentAxisDecisionSet],
) -> dict[str, _ParentAxisDecisionSet]:
    """Apply one bounded, non-cascading peer pass after local resolution."""

    working: dict[str, dict[str, _AxisDecision]] = {
        bundle_id: dict(decision_set.decisions)
        for bundle_id, decision_set in local_decisions_by_bundle.items()
    }
    bundle_by_id = {
        str(getattr(bundle, "bundle_id", "") or ""): bundle
        for bundle in bundles
        if str(getattr(bundle, "bundle_id", "") or "")
    }
    groups: dict[tuple[str, str, str], list[str]] = {}
    for bundle_id, evidence in evidence_by_bundle.items():
        bundle = bundle_by_id.get(bundle_id)
        if bundle is None or not evidence.root_id:
            continue
        role_key = str(getattr(bundle, "role", "") or "speech").strip().lower()
        groups.setdefault((evidence.page_id, evidence.root_id, role_key), []).append(
            bundle_id
        )

    peer_support_by_bundle: dict[str, dict[str, Mapping[str, Any]]] = {
        bundle_id: {} for bundle_id in working
    }
    for (page_id, root_id, role_key), member_ids in sorted(groups.items()):
        if len(member_ids) < PEER_MINIMUM_DONOR_COUNT + 1:
            continue
        group_id = f"root-peer:{page_id}:{root_id}:{role_key}"
        for axis in ("orientation", "family", "weight", "scale"):
            updates: dict[str, _AxisDecision] = {}
            for target_id in sorted(member_ids):
                target_evidence = evidence_by_bundle.get(target_id)
                target_candidates = candidates_by_bundle.get(target_id)
                target_decisions = working.get(target_id)
                if (
                    not _peer_target_has_identity_valid_observation(target_evidence)
                    or target_candidates is None
                    or target_decisions is None
                ):
                    continue
                target_decision = target_decisions.get(axis)
                if not _axis_decision_needs_peer(target_decision):
                    continue
                donors: list[tuple[str, _AxisDecision]] = []
                for donor_id in sorted(member_ids):
                    if donor_id == target_id:
                        continue
                    donor_candidates = candidates_by_bundle.get(donor_id)
                    donor_decisions = working.get(donor_id)
                    if donor_candidates is None or donor_decisions is None:
                        continue
                    donor = donor_decisions.get(axis)
                    if (
                        donor is None
                        or donor.status != "resolved"
                        or donor.source != "direct"
                        or donor.confidence < PEER_DONOR_MIN_CONFIDENCE
                        or not _peer_candidates_are_compatible(
                            target_candidates,
                            donor_candidates,
                            excluded_axis=axis,
                        )
                    ):
                        continue
                    if axis == "scale":
                        target_direction = str(
                            target_decisions["orientation"].value or ""
                        )
                        donor_direction = str(
                            donor_decisions["orientation"].value or ""
                        )
                        if target_direction != donor_direction:
                            continue
                    donors.append((donor_id, donor))
                if len(donors) < PEER_MINIMUM_DONOR_COUNT:
                    continue
                if not _peer_donors_are_mutually_compatible(
                    [candidates_by_bundle[donor_id] for donor_id, _ in donors],
                    excluded_axis=axis,
                ):
                    continue
                peer_candidate = _peer_consensus_candidate(
                    axis=axis,
                    donors=donors,
                    group_id=group_id,
                    direction=str(target_decisions["orientation"].value or ""),
                )
                if peer_candidate is None:
                    continue
                updates[target_id] = _decision_from_candidate(peer_candidate)
            for target_id, decision in updates.items():
                working[target_id][axis] = decision
                peer_support_by_bundle[target_id][axis] = dict(
                    decision.peer_support
                )
                if axis == "orientation":
                    _rebind_directional_local_decisions(
                        working[target_id],
                        candidates_by_bundle[target_id],
                    )

    reconciled: dict[str, _ParentAxisDecisionSet] = {}
    for bundle_id, decisions in working.items():
        peer_support = peer_support_by_bundle.get(bundle_id, {})
        peer_axes = tuple(
            axis for axis in PEER_ASSIST_AXES if axis in peer_support
        )
        reconciled[bundle_id] = _ParentAxisDecisionSet(
            decisions=decisions,
            peer_assisted_axes=peer_axes,
            peer_support=peer_support,
        )
    return reconciled


def _axis_decision_needs_peer(decision: _AxisDecision | None) -> bool:
    return bool(
        decision is None
        or decision.status != "resolved"
        or decision.confidence < PEER_TARGET_RELIABLE_CONFIDENCE
    )


def _peer_consensus_candidate(
    *,
    axis: str,
    donors: Sequence[tuple[str, _AxisDecision]],
    group_id: str,
    direction: str,
) -> _AxisCandidate | None:
    donor_ids = sorted(donor_id for donor_id, _ in donors)
    if axis == "scale":
        values = [float(decision.value) for _, decision in donors]
        median = float(np.median(values))
        spread = (max(values) - min(values)) / max(1.0, median)
        if spread > PEER_SCALE_MAXIMUM_RELATIVE_SPREAD:
            return None
        weights = [decision.confidence for _, decision in donors]
        value: Any = float(np.average(values, weights=weights))
        confidence = float(np.average(weights, weights=weights))
        reason = "root_local_same_role_peer_numeric_consensus"
        extra_support = {
            "relative_spread": round(spread, 8),
            "direction": direction,
        }
    else:
        values = {str(decision.value) for _, decision in donors}
        if len(values) != 1:
            return None
        value = donors[0][1].value
        confidence = float(np.mean([decision.confidence for _, decision in donors]))
        reason = "root_local_same_role_peer_consensus"
        extra_support = {}
    return _AxisCandidate(
        axis=axis,
        value=value,
        confidence=confidence,
        provenance="parent_style_arbitrator:root_local_peer_reconciliation",
        source="peer",
        support_status="supported_root_local_peer_reconciliation",
        reason_codes=(reason,),
        peer_support={
            "group_id": group_id,
            "donor_bundle_ids": donor_ids,
            "donor_count": len(donor_ids),
            **extra_support,
        },
    )


def _rebind_directional_local_decisions(
    decisions: dict[str, _AxisDecision],
    candidates: _ParentAxisCandidates,
) -> None:
    direction = str(decisions["orientation"].value or "ttb")
    weight = candidates.direct.get("weight") or candidates.directional_weight.get(
        direction
    )
    scale = candidates.directional_scale.get(direction)
    decisions["weight"] = (
        _decision_from_candidate(weight)
        if weight is not None
        else _fallback_axis_decision("weight", "regular")
    )
    decisions["scale"] = (
        _decision_from_candidate(scale)
        if scale is not None
        else _fallback_axis_decision("scale", 0.0)
    )


def _peer_target_has_identity_valid_observation(
    evidence: StyleEvidence | None,
) -> bool:
    if evidence is None:
        return False
    if evidence.status == "observed":
        return bool(
            evidence.view_id
            and evidence.cleanup_mask_ids
            and evidence.detector_input_sha256
        )
    if evidence.status != "unavailable":
        return False
    # A detector/model failure can occur after the authorized view and its
    # source geometry were bound.  That parent may receive peer help on the
    # four basic axes; identity/view failures do not carry this footprint.
    return bool(
        evidence.view_id
        and evidence.cleanup_mask_ids
        and evidence.detector_input_sha256
        and evidence.source_text_footprint is not None
    )


def _peer_candidates_are_compatible(
    first: _ParentAxisCandidates,
    second: _ParentAxisCandidates,
    *,
    excluded_axis: str,
) -> bool:
    """Compare reliable peer axes without consulting the axis being repaired."""

    for axis in ("family", "weight", "orientation"):
        if axis == excluded_axis:
            continue
        first_candidate = first.direct.get(axis)
        second_candidate = second.direct.get(axis)
        if (
            first_candidate is not None
            and second_candidate is not None
            and first_candidate.value != second_candidate.value
        ):
            return False
    if excluded_axis == "scale":
        return True

    first_orientation = first.direct.get("orientation")
    second_orientation = second.direct.get("orientation")
    if (
        first_orientation is None
        or second_orientation is None
        or first_orientation.value != second_orientation.value
    ):
        return True
    direction = str(first_orientation.value or "")
    first_scale = first.directional_scale.get(direction)
    second_scale = second.directional_scale.get(direction)
    if first_scale is None or second_scale is None:
        return True
    values = [float(first_scale.value), float(second_scale.value)]
    relative_spread = (max(values) - min(values)) / max(
        1.0, float(np.median(values))
    )
    return relative_spread <= PEER_COMPATIBLE_SCALE_MAXIMUM_RELATIVE_SPREAD


def _peer_donors_are_mutually_compatible(
    donors: Sequence[_ParentAxisCandidates],
    *,
    excluded_axis: str,
) -> bool:
    return all(
        _peer_candidates_are_compatible(
            first,
            second,
            excluded_axis=excluded_axis,
        )
        for index, first in enumerate(donors)
        for second in donors[index + 1 :]
    )


def _resolve_parent_local_axis_decisions(
    candidates: _ParentAxisCandidates,
) -> _ParentAxisDecisionSet:
    """Resolve every style axis once without consulting another parent."""

    decisions: dict[str, _AxisDecision] = {}
    for axis, fallback in (("family", "sans"), ("orientation", "ttb")):
        candidate = candidates.direct.get(axis)
        decisions[axis] = (
            _decision_from_candidate(candidate)
            if candidate is not None
            else _fallback_axis_decision(axis, fallback)
        )

    resolved_direction = str(decisions["orientation"].value or "ttb")
    weight_candidate = candidates.direct.get(
        "weight"
    ) or candidates.directional_weight.get(resolved_direction)
    decisions["weight"] = (
        _decision_from_candidate(weight_candidate)
        if weight_candidate is not None
        else _fallback_axis_decision("weight", "regular")
    )
    scale_candidate = candidates.directional_scale.get(resolved_direction)
    decisions["scale"] = (
        _decision_from_candidate(scale_candidate)
        if scale_candidate is not None
        else _fallback_axis_decision("scale", 0.0)
    )

    for axis, fallback in (
        ("fill", "#000000"),
        ("outline", {"color": "#FFFFFF", "width_px": 0.0}),
    ):
        candidate = candidates.direct.get(axis)
        decisions[axis] = (
            _decision_from_candidate(candidate)
            if candidate is not None
            else _fallback_axis_decision(axis, fallback)
        )

    for axis in ("rotation", "shadow"):
        candidate = candidates.direct.get(axis)
        decisions[axis] = (
            _decision_from_candidate(candidate)
            if candidate is not None
            else _AxisDecision(
                axis=axis,
                value=None,
                status="unavailable",
                confidence=0.0,
                authority="none",
                provenance="authorized_source_style_axis_unavailable",
                source="none",
                reason_codes=(f"{axis}_axis_unavailable",),
            )
        )

    return _ParentAxisDecisionSet(decisions=decisions)


def _decision_from_candidate(candidate: _AxisCandidate) -> _AxisDecision:
    authority = {
        "direct": "authorized_source_style_view",
        "peer": "parent_style_arbitrator_root_local_peer",
    }.get(candidate.source, "unknown")
    return _AxisDecision(
        axis=candidate.axis,
        value=candidate.value,
        status="resolved",
        confidence=float(candidate.confidence),
        authority=authority,
        provenance=candidate.provenance,
        source=candidate.source,
        support_status=candidate.support_status,
        reason_codes=tuple(candidate.reason_codes),
        peer_support=dict(candidate.peer_support),
    )


def _fallback_axis_decision(axis: str, value: Any) -> _AxisDecision:
    provenance = {
        "family": "target_fallback:unresolved_source_family",
        "weight": "target_fallback:unresolved_source_weight",
        "orientation": "target_fallback:unresolved_source_orientation",
        "scale": "typesetting_default:source_scale_unavailable",
        "fill": "target_fallback:unresolved_source_fill",
        "outline": "target_fallback:unresolved_source_outline",
    }[axis]
    return _AxisDecision(
        axis=axis,
        value=value,
        status="fallback",
        confidence=0.0,
        authority="target_fallback" if axis != "scale" else "typesetting_default",
        provenance=provenance,
        source="fallback",
        support_status="unavailable",
        reason_codes=(f"{axis}_axis_unresolved",),
    )




def apply_parent_font_detection(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: Sequence[Any],
    mode: str,
    authorized_style_views: Mapping[str, AuthorizedSourceStyleView] | Any = None,
    default_font_name: str = "",
    use_gpu: bool = False,
    models_dir: str | None = None,
    detector: Any | None = None,
) -> ParentFontDetectionRunResult:
    """Observe authorized pixels, then atomically resolve parent styles."""
    observed = observe_parent_style_evidence(
        page_id=page_id,
        image_path=image_path,
        parent_execution_bundles=parent_execution_bundles,
        authorized_style_views=authorized_style_views,
        mode=mode,
        use_gpu=use_gpu,
        models_dir=models_dir,
        detector=detector,
    )
    arbitration = arbitrate_parent_styles(
        parent_execution_bundles=parent_execution_bundles,
        evidence=observed.evidence,
        default_font_name=default_font_name,
        models_dir=models_dir,
    )
    result = ParentFontDetectionRunResult(
        page_id=str(page_id or ""),
        mode=observed.mode,
        enabled=observed.enabled,
        model_path=observed.model_path,
        labels_path=observed.labels_path,
        gpu_requested=observed.gpu_requested,
        requested_execution_provider=observed.requested_execution_provider,
        available_execution_providers=list(observed.available_execution_providers),
        active_execution_providers=list(observed.active_execution_providers),
        primary_execution_provider=observed.primary_execution_provider,
        provider_fallback_reason=observed.provider_fallback_reason,
        provider_preload_error=observed.provider_preload_error,
        errors=list(observed.errors),
        records=[dict(record) for record in arbitration.records],
    )
    for record in result.records:
        status = str(record.get("status") or "")
        if status == "applied":
            result.applied_count += 1
        elif status == "skipped":
            result.skipped_count += 1
        else:
            result.fallback_count += 1
    return result


def resolve_unavailable_parent_styles(
    *,
    page_id: str,
    parent_execution_bundles: Sequence[Any],
    reason_codes: Sequence[str],
    mode: str,
    default_font_name: str = "",
    models_dir: str | None = None,
    errors: Sequence[str] = (),
) -> ParentFontDetectionRunResult:
    """Assign one truthful arbitrator-owned default after a stage failure."""

    evidence = [
        StyleEvidence.unavailable(
            page_id=str(page_id or ""),
            bundle_id=str(getattr(bundle, "bundle_id", "") or ""),
            parent_id=str(getattr(bundle, "parent_id", "") or ""),
            root_id=str(getattr(bundle, "root_id", "") or ""),
            reason_codes=tuple(reason_codes),
        )
        for bundle in list(parent_execution_bundles or [])
        if bool(getattr(bundle, "render_required", False))
    ]
    arbitration = arbitrate_parent_styles(
        parent_execution_bundles=parent_execution_bundles,
        evidence=evidence,
        default_font_name=default_font_name,
        models_dir=models_dir,
    )
    result = ParentFontDetectionRunResult(
        page_id=str(page_id or ""),
        mode=str(mode or ""),
        enabled=False,
        errors=[str(error) for error in errors if str(error)],
        records=[dict(record) for record in arbitration.records],
    )
    for record in result.records:
        if str(record.get("status") or "") == "applied":
            result.applied_count += 1
        else:
            result.fallback_count += 1
    return result




def _build_resolved_style_from_decisions(
    bundle: Any,
    evidence: StyleEvidence,
    decision_set: _ParentAxisDecisionSet,
    *,
    default_font_name: str,
    models_dir: str | None,
) -> dict[str, Any]:
    """Build the executable style once from immutable per-axis decisions."""

    decisions = dict(decision_set.decisions)
    family = decisions["family"]
    weight = decisions["weight"]
    orientation_decision = decisions["orientation"]
    scale = decisions["scale"]
    fill = decisions["fill"]
    outline = decisions["outline"]
    base = _style_contract_base(bundle)
    role = str(getattr(bundle, "role", "") or "")
    semantic_style_class = _semantic_style_class(role)
    observed = evidence.status == "observed" and evidence.vote_eligible

    target_serif = str(family.value or "sans") == "serif"
    family_role = "serif" if target_serif else "sans"
    target_weight = str(weight.value or "regular")
    if target_weight not in {"regular", "bold", "black"}:
        target_weight = "regular"
    resolved_font = resolve_noto_cjk_sc_font_file(
        base_dir=models_dir,
        serif=target_serif,
        weight=target_weight,
    ) or default_font_name or (
        "Noto Serif CJK SC" if target_serif else "Noto Sans CJK SC"
    )
    direction = str(orientation_decision.value or "ttb")
    if direction not in {"ltr", "ttb"}:
        direction = "ttb"
    orientation = "horizontal" if direction == "ltr" else "vertical"
    preferred_size = (
        max(1, int(round(float(scale.value))))
        if scale.status == "resolved" and float(scale.value or 0.0) > 0.0
        else 0
    )
    outline_value = (
        dict(outline.value) if isinstance(outline.value, Mapping) else {}
    )
    outline_color = _hex_color(outline_value.get("color")) or "#FFFFFF"
    raw_stroke_width = max(0.0, float(outline_value.get("width_px") or 0.0))
    stroke_width: int | float = max(0, int(round(raw_stroke_width)))
    if preferred_size > 0:
        stroke_width = min(stroke_width, max(0, int(round(preferred_size * 0.25))))
    elif outline.status != "resolved":
        stroke_width = 0

    peer_axes = list(decision_set.peer_assisted_axes)
    optional_effect_axes = [
        axis
        for axis in ("rotation", "shadow")
        if decisions[axis].status == "resolved"
    ]
    basic_axis_resolved = any(
        decisions[axis].status == "resolved"
        for axis in ("family", "weight", "orientation", "scale")
    )
    resolution_reasons = (
        ["per_parent_authorized_evidence"]
        if observed
        else list(evidence.reason_codes)
    )
    if peer_axes:
        resolution_reasons.append("root_local_same_role_peer_assistance")
    if optional_effect_axes:
        resolution_reasons.append("authorized_optional_effect_axes_resolved")
    fallback_reasons = {
        "family": "source_family_axis_unresolved_target_sans_fallback",
        "weight": "source_weight_axis_unresolved_target_regular_fallback",
        "scale": "source_scale_axis_unresolved_arbitrator_fallback",
        "fill": "source_fill_axis_unresolved_target_black_fallback",
        "outline": "source_outline_axis_unresolved_zero_outline_fallback",
        "orientation": "source_orientation_axis_unresolved_target_vertical_fallback",
    }
    for axis in CORE_STYLE_AXES:
        if decisions[axis].status != "resolved":
            resolution_reasons.append(fallback_reasons[axis])
    resolution_reasons = _unique_strings(resolution_reasons)
    resolved_confidence = float(
        np.mean(
            [
                decisions[axis].confidence
                if decisions[axis].status == "resolved"
                else 0.0
                for axis in CORE_STYLE_AXES
            ]
        )
    )
    peer_group_ids = _unique_strings(
        [
            str(value.get("group_id") or "")
            for value in decision_set.peer_support.values()
            if isinstance(value, Mapping)
        ]
    )
    style_axis_confidence = {
        "family": float(family.confidence),
        "weight": float(weight.confidence),
        "scale": float(scale.confidence),
        "fill": float(fill.confidence),
        "outline": float(outline.confidence),
        "orientation": float(orientation_decision.confidence),
    }
    style_axis_provenance = {
        "family": family.provenance,
        "weight": weight.provenance,
        "scale": scale.provenance,
        "fill": fill.provenance,
        "outline": outline.provenance,
        "orientation": orientation_decision.provenance,
    }

    style: dict[str, Any] = {
        **base,
        "render_style_version": PARENT_RENDER_STYLE_VERSION,
        "render_style_owner": "parent_execution_bundle",
        "render_style_source": STYLE_ARBITRATOR_SOURCE,
        "render_style_provider": STYLE_ARBITRATOR_PROVIDER,
        "render_style_provider_model": evidence.evidence_model,
        "render_style_confidence": resolved_confidence,
        "style_resolution_status": (
            "authorized_evidence_resolved" if observed else "unresolved"
        ),
        "style_resolution_reason_codes": resolution_reasons,
        "style_arbitration_decision": (
            "per_parent_authorized_evidence_with_root_peer_assistance"
            if observed and peer_axes
            else "identity_valid_observation_with_root_peer_assistance"
            if peer_axes
            else "per_parent_authorized_evidence"
            if observed
            else "authorized_evidence_unavailable"
        ),
        "style_arbitration_peer_scope": (
            "root_local_same_role" if peer_axes else "none"
        ),
        "style_arbitration_peer_group_id": (
            peer_group_ids[0] if len(peer_group_ids) == 1 else ""
        ),
        "style_arbitration_peer_group_ids": peer_group_ids,
        "style_arbitration_peer_assisted_axes": peer_axes,
        "style_arbitration_peer_support": {
            key: dict(value) for key, value in decision_set.peer_support.items()
        },
        "style_axis_decisions": decision_set.to_audit_dict(),
        "style_class": semantic_style_class,
        "typographic_style_class": (
            "unresolved"
            if not basic_axis_resolved
            else f"{family_role}_{target_weight}"
            if weight.status == "resolved"
            else f"{family_role}_fallback_regular"
        ),
        "base_style_id": (
            f"base_{family_role}_{target_weight}_{orientation}"
            if basic_axis_resolved
            else "unresolved"
        ),
        "font_family": resolved_font,
        "font_family_role": (
            family_role
            if observed or family.status == "resolved"
            else "fallback_sans"
        ),
        "font_family_authority": _resolved_axis_field_authority(
            family,
            fallback="target_fallback_unresolved_source_family",
        ),
        "font_weight": target_weight,
        "font_weight_authority": _resolved_axis_field_authority(
            weight,
            fallback="target_fallback_unresolved_source_weight",
        ),
        "fallback_font_chain_key": PARENT_STYLE_DEFAULT_FALLBACK_FONT_CHAIN_KEY,
        "target_font_mapping_source": "noto_cjk_sc_role_weight_glyph_coverage_pack",
        "target_font_mapping_family_role": family_role,
        "target_font_mapping_weight": target_weight,
        "fill_color": _hex_color(fill.value) or "#000000",
        "fill_color_authority": _resolved_axis_field_authority(
            fill,
            fallback="target_fallback_unresolved_source_paint",
        ),
        "stroke_color": outline_color,
        "stroke_width": stroke_width,
        "stroke_authority": _resolved_axis_field_authority(
            outline,
            fallback="target_fallback_unresolved_source_stroke_zero",
        ),
        "source_orientation": orientation,
        "wrap_mode": orientation,
        "source_orientation_authority": _resolved_axis_field_authority(
            orientation_decision,
            fallback="target_fallback_unresolved_source_orientation",
        ),
        "font_size_authority": (
            "automated_style_arbitrator"
            if preferred_size > 0
            else PARENT_STYLE_UNRESOLVED_FONT_SIZE_AUTHORITY
        ),
        "font_size_locked": False,
        "font_size_policy": (
            "authorized_source_preferred"
            if preferred_size > 0
            else PARENT_STYLE_UNRESOLVED_FONT_SIZE_POLICY
        ),
        "font_size_fallback_policy": "typesetting_bounded_fit",
        "font_size_source": (
            "root_local_peer_assist"
            if preferred_size > 0 and scale.source == "peer"
            else "authorized_source_style_view"
            if preferred_size > 0
            else "parent_style_arbitrator_unresolved_scale_fallback"
        ),
        "source_typography_observed": observed,
        "source_typography_matched": False,
        "source_typography_match_status": (
            "mapped_to_supported_target_role"
            if observed
            else "partial_root_peer_axes_resolved"
            if peer_axes
            else "unresolved"
        ),
        "style_evidence_status": evidence.status,
        "style_evidence_view_id": evidence.view_id,
        "style_evidence_cleanup_mask_ids": list(evidence.cleanup_mask_ids),
        "style_evidence_owned_component_ids": list(evidence.owned_component_ids),
        "style_evidence_provider": evidence.evidence_provider,
        "style_evidence_source": evidence.evidence_source,
        "style_evidence_model": evidence.evidence_model,
        "style_axis_confidence": style_axis_confidence,
        "style_axis_provenance": style_axis_provenance,
        "detector_input_sha256": evidence.detector_input_sha256,
        "source_scale_px": (
            round(float(scale.value), 6) if preferred_size > 0 else 0.0
        ),
        "source_scale_support_status": scale.support_status,
        "source_scale_conversion_count": 1 if preferred_size > 0 else 0,
        "source_scale_source": (
            "root_local_peer_directional_scale_assist"
            if scale.source == "peer"
            else "authorized_foreground_geometry_cell_measurement"
            if preferred_size > 0
            else "parent_style_arbitrator_unresolved_scale_fallback"
        ),
        "source_ink_stroke_width_px": round(
            _float(evidence.source_ink_stroke_width_px), 6
        ),
    }
    executable_font_size = (
        preferred_size if preferred_size > 0 else PARENT_STYLE_UNRESOLVED_FONT_SIZE
    )
    style.update(
        {
            "font_size": executable_font_size,
            "font_size_hint": executable_font_size,
            "font_size_min": (
                max(1, int(round(preferred_size * 0.72)))
                if preferred_size > 0
                else PARENT_STYLE_UNRESOLVED_FONT_SIZE_MIN
            ),
            "font_size_max": (
                preferred_size
                if preferred_size > 0
                else PARENT_STYLE_UNRESOLVED_FONT_SIZE_MAX
            ),
        }
    )

    rotation = decisions.get("rotation")
    shadow = decisions.get("shadow")
    if (
        rotation is not None
        and rotation.status == "resolved"
        or shadow is not None
        and shadow.status == "resolved"
    ):
        style["parent_layer_effects"] = {
            "contract_version": "parent_layer_effects_v1",
            "rotation": (
                {"availability": "resolved", **dict(rotation.value)}
                if rotation is not None and rotation.status == "resolved"
                else {"availability": "unavailable"}
            ),
            "shadow": (
                {"availability": "resolved", **dict(shadow.value)}
                if shadow is not None and shadow.status == "resolved"
                else {"availability": "unavailable"}
            ),
        }
    if optional_effect_axes:
        style["style_resolution_coverage"] = (
            "authorized_core_plus_optional_effect_resolution"
            if observed
            else "partial_root_peer_plus_optional_effect_resolution"
            if peer_axes
            else "partial_optional_effect_resolution"
        )
    elif peer_axes and not observed:
        style["style_resolution_coverage"] = "partial_root_peer_resolution"
    validation = validate_resolved_render_style(style)
    if not validation.accepted:
        raise ValueError(
            "parent_style_arbitrator_invalid_resolved_style:"
            + ",".join(validation.reason_codes)
        )
    return validation.style


def _resolved_axis_field_authority(
    decision: _AxisDecision,
    *,
    fallback: str,
) -> str:
    if decision.status != "resolved":
        return fallback
    return decision.authority


def _perceptual_carrier_validation(
    *,
    bundle: Any,
    evidence: StyleEvidence,
    raw_carrier: Any,
    carrier: Mapping[str, Any],
) -> tuple[list[str], str]:
    reasons: list[str] = []
    if not isinstance(raw_carrier, Mapping):
        return ["perceptual_carrier_not_mapping"], ""
    if not _is_json_safe(raw_carrier):
        # Axis-local payloads are checked separately so one malformed axis does
        # not suppress a valid sibling. Only a malformed header is global.
        header = {
            key: carrier.get(key)
            for key in ("contract_version", "source_identity", "fact_set_id")
        }
        if not _is_json_safe(header):
            reasons.append("perceptual_carrier_header_not_json_safe")
    allowed_fields = {
        "contract_version",
        "source_identity",
        "fact_set_id",
        *PERCEPTUAL_STYLE_AXES,
    }
    reasons.extend(
        _mapping_key_reasons(
            carrier,
            allowed_fields=allowed_fields,
            reason_prefix="perceptual_carrier",
        )
    )
    if carrier.get("contract_version") != PERCEPTUAL_STYLE_AXES_VERSION:
        reasons.append("perceptual_carrier_contract_version_invalid")

    source_identity = carrier.get("source_identity")
    expected_identity = {
        "authorized_source_style_view_version": "authorized_source_style_view_v1",
        "page_id": str(getattr(bundle, "page_id", "") or ""),
        "view_id": evidence.view_id,
        "bundle_id": str(getattr(bundle, "bundle_id", "") or ""),
        "parent_id": str(getattr(bundle, "parent_id", "") or ""),
        "root_id": str(getattr(bundle, "root_id", "") or ""),
        "content_bbox": list(evidence.content_bbox),
        "analysis_bbox": list(evidence.analysis_bbox),
        "cleanup_mask_ids": list(evidence.cleanup_mask_ids),
        "owned_component_ids": list(evidence.owned_component_ids),
        "detector_input_sha256": evidence.detector_input_sha256,
    }
    required_identity_fields = {
        *expected_identity,
        "crop_shape",
        "authorized_mask_sha256",
        "authorized_pixel_sha256",
        "external_surface_ring_version",
        "external_surface_ring_inner_radius_px",
        "external_surface_ring_outer_radius_px",
        "external_surface_ring_pixel_count",
        "external_surface_ring_mask_sha256",
        "external_surface_ring_pixel_sha256",
    }
    trusted_source_identity = evidence.authorized_perceptual_source_identity
    trusted_identity: dict[str, Any] = {}
    if not isinstance(trusted_source_identity, Mapping):
        reasons.append("perceptual_trusted_source_identity_not_mapping")
    elif not _is_json_safe(trusted_source_identity):
        reasons.append("perceptual_trusted_source_identity_not_json_safe")
    else:
        trusted_identity = _plain_json_mapping_snapshot(
            trusted_source_identity
        )
        trusted_string_keys = {
            key for key in trusted_identity if isinstance(key, str)
        }
        for key in sorted(required_identity_fields - trusted_string_keys):
            reasons.append(f"perceptual_trusted_source_identity_missing_field:{key}")
        reasons.extend(
            _mapping_key_reasons(
                trusted_identity,
                allowed_fields=required_identity_fields,
                reason_prefix="perceptual_trusted_source_identity",
            )
        )
        for key, expected in expected_identity.items():
            if trusted_identity.get(key) != expected:
                reasons.append(
                    f"perceptual_trusted_source_identity_{key}_mismatch"
                )
        trusted_crop_shape = trusted_identity.get("crop_shape")
        if (
            not _is_plain_sequence(trusted_crop_shape)
            or len(list(trusted_crop_shape)) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in list(trusted_crop_shape)
            )
        ):
            reasons.append("perceptual_trusted_source_identity_crop_shape_invalid")
        for key in (
            "authorized_mask_sha256",
            "authorized_pixel_sha256",
            "external_surface_ring_mask_sha256",
            "external_surface_ring_pixel_sha256",
            "detector_input_sha256",
        ):
            if not _is_sha256(trusted_identity.get(key)):
                reasons.append(
                    f"perceptual_trusted_source_identity_{key}_invalid"
                )
        if trusted_identity.get("external_surface_ring_version") != (
            EXTERNAL_SOURCE_SURFACE_RING_VERSION
        ):
            reasons.append(
                "perceptual_trusted_source_identity_external_surface_ring_version_invalid"
            )
        trusted_inner = trusted_identity.get(
            "external_surface_ring_inner_radius_px"
        )
        trusted_outer = trusted_identity.get(
            "external_surface_ring_outer_radius_px"
        )
        trusted_count = trusted_identity.get("external_surface_ring_pixel_count")
        if (
            isinstance(trusted_inner, bool)
            or not isinstance(trusted_inner, (int, float))
            or not math.isfinite(float(trusted_inner))
            or float(trusted_inner) < 0.0
            or isinstance(trusted_outer, bool)
            or not isinstance(trusted_outer, (int, float))
            or not math.isfinite(float(trusted_outer))
            or float(trusted_outer) < float(trusted_inner or 0.0)
            or isinstance(trusted_count, bool)
            or not isinstance(trusted_count, int)
            or trusted_count < 0
        ):
            reasons.append(
                "perceptual_trusted_source_identity_external_surface_ring_geometry_invalid"
            )
    computed_fact_set_id = ""
    if not isinstance(source_identity, Mapping):
        reasons.append("perceptual_source_identity_not_mapping")
    elif not _is_json_safe(source_identity):
        reasons.append("perceptual_source_identity_not_json_safe")
    else:
        identity = _plain_json_mapping_snapshot(source_identity)
        identity_string_keys = {
            key for key in identity if isinstance(key, str)
        }
        for key in sorted(required_identity_fields - identity_string_keys):
            reasons.append(f"perceptual_source_identity_missing_field:{key}")
        reasons.extend(
            _mapping_key_reasons(
                identity,
                allowed_fields=required_identity_fields,
                reason_prefix="perceptual_source_identity",
            )
        )
        for key, expected in expected_identity.items():
            if identity.get(key) != expected:
                reasons.append(f"perceptual_source_identity_{key}_mismatch")
        for key in sorted(required_identity_fields):
            if identity.get(key) != trusted_identity.get(key):
                reasons.append(
                    f"perceptual_source_identity_trusted_{key}_mismatch"
                )
        crop_shape = identity.get("crop_shape")
        if (
            not _is_plain_sequence(crop_shape)
            or len(list(crop_shape)) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in list(crop_shape)
            )
        ):
            reasons.append("perceptual_source_identity_crop_shape_invalid")
        for key in (
            "authorized_mask_sha256",
            "authorized_pixel_sha256",
            "external_surface_ring_mask_sha256",
            "external_surface_ring_pixel_sha256",
            "detector_input_sha256",
        ):
            if not _is_sha256(identity.get(key)):
                reasons.append(f"perceptual_source_identity_{key}_invalid")
        if identity.get("external_surface_ring_version") != (
            EXTERNAL_SOURCE_SURFACE_RING_VERSION
        ):
            reasons.append(
                "perceptual_source_identity_external_surface_ring_version_invalid"
            )
        inner = identity.get("external_surface_ring_inner_radius_px")
        outer = identity.get("external_surface_ring_outer_radius_px")
        ring_count = identity.get("external_surface_ring_pixel_count")
        if (
            isinstance(inner, bool)
            or not isinstance(inner, (int, float))
            or not math.isfinite(float(inner))
            or float(inner) < 0.0
            or isinstance(outer, bool)
            or not isinstance(outer, (int, float))
            or not math.isfinite(float(outer))
            or float(outer) < float(inner or 0.0)
            or isinstance(ring_count, bool)
            or not isinstance(ring_count, int)
            or ring_count < 0
        ):
            reasons.append(
                "perceptual_source_identity_external_surface_ring_geometry_invalid"
            )
        computed_fact_set_id = _perceptual_fact_set_id(identity)

    carrier_fact_set_id = _plain_string(carrier.get("fact_set_id"))
    if not computed_fact_set_id or carrier_fact_set_id != computed_fact_set_id:
        reasons.append("perceptual_carrier_fact_set_identity_mismatch")
    return _unique_strings(reasons), carrier_fact_set_id


def _resolve_perceptual_axis(
    *,
    axis: str,
    record: Any,
    carrier_fact_set_id: str,
    global_reasons: Sequence[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    validation_reasons = list(global_reasons)
    payload = dict(record) if isinstance(record, Mapping) else {}
    support_status = _plain_string(payload.get("support_status")) or "invalid"
    provenance = _plain_string(payload.get("provenance"))
    fact_set_id = _plain_string(payload.get("fact_set_id"))
    confidence = _strict_perceptual_number(payload.get("confidence"))
    audit_confidence = (
        round(float(confidence), 8)
        if confidence is not None and 0.0 <= confidence <= 1.0
        else 0.0
    )

    if not isinstance(record, Mapping):
        validation_reasons.append(f"perceptual_{axis}_record_not_mapping")
    elif not _is_json_safe(record):
        validation_reasons.append(f"perceptual_{axis}_record_not_json_safe")
    allowed_fields = {
        "support_status",
        "value",
        "confidence",
        "provenance",
        "fact_set_id",
        "reason_codes",
        "support",
        "conflict",
        "uncertainty",
    }
    validation_reasons.extend(
        _mapping_key_reasons(
            payload,
            allowed_fields=allowed_fields,
            reason_prefix=f"perceptual_{axis}",
        )
    )
    if provenance != PERCEPTUAL_STYLE_PROVENANCE:
        validation_reasons.append(f"perceptual_{axis}_provenance_invalid")
    if not carrier_fact_set_id or fact_set_id != carrier_fact_set_id:
        validation_reasons.append(f"perceptual_{axis}_fact_set_identity_mismatch")
    if confidence is None or not 0.0 <= confidence <= 1.0:
        validation_reasons.append(f"perceptual_{axis}_confidence_invalid")
    reason_codes = payload.get("reason_codes")
    if not _is_plain_sequence(reason_codes) or any(
        not isinstance(value, str) for value in reason_codes
    ):
        validation_reasons.append(f"perceptual_{axis}_reason_codes_invalid")
        input_reasons: list[str] = []
    else:
        input_reasons = [value for value in reason_codes if value]
    for key in ("support", "conflict", "uncertainty"):
        if not isinstance(payload.get(key), Mapping):
            validation_reasons.append(f"perceptual_{axis}_{key}_invalid")
    conflict = payload.get("conflict")
    conflict_status = (
        _plain_string(conflict.get("status"))
        if isinstance(conflict, Mapping)
        else ""
    )

    resolved_value: dict[str, Any] | None = None
    if support_status == "supported":
        if confidence is None or confidence <= 0.0:
            validation_reasons.append(f"perceptual_{axis}_supported_confidence_invalid")
        if conflict_status != "clear":
            validation_reasons.append(f"perceptual_{axis}_supported_conflict_invalid")
        value, value_reasons = _validated_perceptual_axis_value(
            axis,
            payload.get("value"),
        )
        validation_reasons.extend(value_reasons)
        if not validation_reasons:
            resolved_value = value
    else:
        if support_status not in {"unavailable", "ambiguous"}:
            validation_reasons.append(f"perceptual_{axis}_support_status_rejected")
        expected_conflict = (
            "ambiguous" if support_status == "ambiguous" else "unavailable"
        )
        if conflict_status != expected_conflict:
            validation_reasons.append(f"perceptual_{axis}_conflict_status_invalid")
        if "value" in payload:
            validation_reasons.append(f"perceptual_{axis}_non_supported_value_rejected")
        validation_reasons.append(f"perceptual_{axis}_not_independently_supported")

    availability = "resolved" if resolved_value is not None else "unavailable"
    audit = {
        "availability": availability,
        "decision": (
            "apply_independently_supported_axis"
            if resolved_value is not None
            else "preserve_task_a_axis"
        ),
        "support_status": support_status,
        "confidence": audit_confidence,
        "provenance": provenance,
        "fact_set_id": fact_set_id,
        "reason_codes": _unique_strings([*input_reasons, *validation_reasons]),
    }
    return resolved_value, audit


def _validated_perceptual_axis_value(
    axis: str,
    raw_value: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw_value, Mapping):
        return None, [f"perceptual_{axis}_value_not_mapping"]
    if not _is_json_safe(raw_value):
        return None, [f"perceptual_{axis}_value_not_json_safe"]
    value = dict(raw_value)
    required: dict[str, set[str]] = {
        "fill": {"color"},
        "outline": {"color", "width_px"},
        "rotation": {"degrees_clockwise", "pivot"},
        "shadow": {"color", "offset_px", "blur_radius_px"},
    }
    expected = required[axis]
    if set(value) != expected:
        return None, [f"perceptual_{axis}_value_fields_invalid"]

    if axis == "fill":
        color = _perceptual_color(value.get("color"), allow_alpha=False)
        return (
            ({"color": color}, [])
            if color
            else (None, ["perceptual_fill_value_color_invalid"])
        )
    if axis == "outline":
        color = _perceptual_color(value.get("color"), allow_alpha=False)
        width = _strict_perceptual_number(value.get("width_px"))
        reasons: list[str] = []
        if not color:
            reasons.append("perceptual_outline_value_color_invalid")
        if width is None or not 0.0 < width <= 256.0:
            reasons.append("perceptual_outline_value_width_px_invalid")
        return (
            ({"color": color, "width_px": float(width)}, [])
            if not reasons and width is not None
            else (None, reasons)
        )
    if axis == "rotation":
        degrees = _strict_perceptual_number(value.get("degrees_clockwise"))
        pivot = value.get("pivot")
        reasons = []
        if degrees is None or not -45.0 <= degrees <= 45.0:
            reasons.append("perceptual_rotation_value_degrees_invalid")
        if pivot != "visual_center":
            reasons.append("perceptual_rotation_value_pivot_invalid")
        return (
            (
                {
                    "degrees_clockwise": float(degrees),
                    "pivot": "visual_center",
                },
                [],
            )
            if not reasons and degrees is not None
            else (None, reasons)
        )

    color = _perceptual_color(value.get("color"), allow_alpha=True)
    offset = value.get("offset_px")
    offsets = list(offset) if _is_plain_sequence(offset) else []
    parsed_offsets = [_strict_perceptual_number(item) for item in offsets]
    blur = _strict_perceptual_number(value.get("blur_radius_px"))
    reasons = []
    if not color:
        reasons.append("perceptual_shadow_value_color_invalid")
    if (
        len(parsed_offsets) != 2
        or any(item is None or abs(item) > 256.0 for item in parsed_offsets)
    ):
        reasons.append("perceptual_shadow_value_offset_px_invalid")
    if blur is None or not 0.0 <= blur <= 64.0:
        reasons.append("perceptual_shadow_value_blur_radius_px_invalid")
    return (
        (
            {
                "color": color,
                "offset_px": [float(parsed_offsets[0]), float(parsed_offsets[1])],
                "blur_radius_px": float(blur),
            },
            [],
        )
        if not reasons and blur is not None
        else (None, reasons)
    )


def _style_has_resolved_perceptual_axis(style: Mapping[str, Any]) -> bool:
    resolution = style.get("style_perceptual_axis_resolution")
    if not isinstance(resolution, Mapping):
        return False
    axes = resolution.get("resolved_axes")
    return bool(_is_plain_sequence(axes) and list(axes))


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
    return f"{PERCEPTUAL_STYLE_FACT_SET_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def _json_safe_audit_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"audit_status": "rejected_non_mapping_payload"}
    if not _json_snapshot_shape_is_bounded(value):
        return {"audit_status": "rejected_unbounded_json_payload"}
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, RecursionError, OverflowError):
        return {"audit_status": "rejected_non_json_payload"}
    return decoded if isinstance(decoded, dict) else {"audit_status": "rejected_payload"}


def _plain_json_mapping_snapshot(value: Any) -> dict[str, Any]:
    """Return an alias-free, bounded JSON mapping or an empty fail-closed view."""

    if not isinstance(value, Mapping) or not _json_snapshot_shape_is_bounded(value):
        return {}
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, RecursionError, OverflowError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _frozen_json_mapping_snapshot(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return _FrozenJsonDict()
    frozen = _freeze_json_snapshot(value)
    return frozen if isinstance(frozen, _FrozenJsonDict) else _FrozenJsonDict()


def _frozen_json_sequence_snapshot(value: Any) -> tuple[Any, ...]:
    """Return an alias-free, recursively frozen sequence."""

    if not _is_plain_sequence(value):
        return ()
    frozen = _freeze_json_snapshot(value)
    return frozen if isinstance(frozen, tuple) else ()


def _freeze_json_snapshot(
    value: Any,
    *,
    depth: int = 0,
    active_containers: set[int] | None = None,
    node_budget: list[int] | None = None,
) -> Any:
    """Freeze without aliasing while retaining malformed subtrees as markers.

    A malformed perceptual axis must not erase valid sibling axes.  The marker
    is intentionally not JSON serializable, so the existing axis-local
    validators reject only the affected record.  Depth, node count, and cycles
    fail closed without recursing through hostile payloads.
    """

    active = active_containers if active_containers is not None else set()
    budget = node_budget if node_budget is not None else [0]
    budget[0] += 1
    if depth > MAX_STYLE_CARRIER_DEPTH:
        return _InvalidFrozenJsonValue("maximum_depth_exceeded")
    if budget[0] > MAX_STYLE_CARRIER_NODES:
        return _InvalidFrozenJsonValue("maximum_node_count_exceeded")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return _InvalidFrozenJsonValue("container_cycle_detected")
        active.add(identity)
        try:
            return _FrozenJsonDict(
                {
                    key: _freeze_json_snapshot(
                        item,
                        depth=depth + 1,
                        active_containers=active,
                        node_budget=budget,
                    )
                    for key, item in value.items()
                }
            )
        finally:
            active.discard(identity)
    if _is_plain_sequence(value):
        identity = id(value)
        if identity in active:
            return _InvalidFrozenJsonValue("container_cycle_detected")
        active.add(identity)
        try:
            return tuple(
                _freeze_json_snapshot(
                    item,
                    depth=depth + 1,
                    active_containers=active,
                    node_budget=budget,
                )
                for item in value
            )
        finally:
            active.discard(identity)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return _InvalidFrozenJsonValue(
        f"unsupported_json_value:{type(value).__name__}"
    )


def _is_json_safe(value: Any) -> bool:
    if not _json_snapshot_shape_is_bounded(value):
        return False
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False
    return True


def _json_snapshot_shape_is_bounded(value: Any) -> bool:
    """Bound snapshot traversal while permitting harmless repeated aliases."""

    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active_containers: set[int] = set()
    node_count = 0
    while stack:
        current, depth, exiting = stack.pop()
        is_container = isinstance(current, Mapping) or _is_plain_sequence(current)
        if exiting:
            if is_container:
                active_containers.discard(id(current))
            continue
        node_count += 1
        if depth > MAX_STYLE_CARRIER_DEPTH or node_count > MAX_STYLE_CARRIER_NODES:
            return False
        if not is_container:
            continue
        identity = id(current)
        if identity in active_containers:
            return False
        active_containers.add(identity)
        stack.append((current, depth, True))
        if isinstance(current, Mapping):
            for key, child in current.items():
                stack.append((key, depth + 1, False))
                stack.append((child, depth + 1, False))
        else:
            for child in current:
                stack.append((child, depth + 1, False))
    return True


def _strict_perceptual_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _perceptual_color(value: Any, *, allow_alpha: bool) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().upper()
    lengths = {7, 9} if allow_alpha else {7}
    if len(text) not in lengths or not text.startswith("#"):
        return ""
    try:
        int(text[1:], 16)
    except ValueError:
        return ""
    return text


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9A-Fa-f]{64}", value))


def _is_plain_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _plain_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _mapping_key_reasons(
    value: Mapping[Any, Any],
    *,
    allowed_fields: set[str],
    reason_prefix: str,
) -> list[str]:
    reasons: list[str] = []
    unknown_strings: list[str] = []
    for key in value:
        if not isinstance(key, str):
            reasons.append(f"{reason_prefix}_key_not_string")
        elif key not in allowed_fields:
            unknown_strings.append(key)
    reasons.extend(
        f"{reason_prefix}_unknown_field:{key}"
        for key in sorted(unknown_strings)
    )
    return _unique_strings(reasons)


def _style_contract_base(bundle: Any) -> dict[str, Any]:
    """Build non-observational renderer context without legacy style input."""

    role = str(getattr(bundle, "role", "") or "").strip().lower()
    caption_like = role in {
        "caption",
        "background",
        "caption_background",
        "background_narration",
    }
    semantic_class = str(getattr(bundle, "semantic_class", "") or "")
    if not semantic_class:
        semantic_class = "caption_background" if caption_like else "speech_bubble"
    route_intent = str(getattr(bundle, "route_intent", "") or "")
    if not route_intent:
        route_intent = "translate_caption" if caption_like else "translate_speech"
    if role == "speech" or not role:
        semantic_kind = "speech"
    elif role in {"caption", "caption_background"}:
        semantic_kind = "caption"
    elif caption_like:
        semantic_kind = "background_narration"
    else:
        semantic_kind = role
    return {
        "source_role": role or "speech",
        "semantic_class": semantic_class,
        "semantic_kind": semantic_kind,
        "route_intent": route_intent,
        "render_allowed_area": [
            int(value) for value in (getattr(bundle, "render_allowed_area", []) or [])[:4]
        ],
        "source_region_ids": [
            str(value) for value in (getattr(bundle, "source_region_ids", []) or []) if str(value)
        ],
        "source_orientation": "vertical",
        "wrap_mode": "vertical",
        "line_height": 1.1 if caption_like else 1.0,
        "align": "center",
    }


def _authorized_view_mapping(value: Any) -> dict[str, AuthorizedSourceStyleView]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, AuthorizedSourceStyleView)
        }
    mapping = getattr(value, "views_by_bundle_id", {})
    return _authorized_view_mapping(mapping)


def _authorized_view_rejection_reasons(
    view: AuthorizedSourceStyleView | None,
    *,
    page_id: str,
    bundle_id: str,
    parent_id: str,
    root_id: str,
    image: Any,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if view is None:
        return ("authorized_style_view_missing",)
    if not view.available:
        reasons.extend(view.reason_codes or ("authorized_style_view_unavailable",))
    if view.page_id != str(page_id or ""):
        reasons.append("authorized_style_view_page_mismatch")
    if view.bundle_id != bundle_id:
        reasons.append("authorized_style_view_bundle_mismatch")
    if view.parent_id not in {bundle_id, parent_id}:
        reasons.append("authorized_style_view_parent_mismatch")
    if root_id and view.root_id and view.root_id != root_id:
        reasons.append("authorized_style_view_root_mismatch")
    mask = getattr(view, "foreground_mask", None)
    if mask is None:
        reasons.append("authorized_style_view_foreground_missing")
    else:
        array = np.asarray(mask)
        if array.ndim != 2 or int(np.count_nonzero(array)) <= 0:
            reasons.append("authorized_style_view_foreground_empty_or_invalid")
        if image is not None and array.shape[:2] != (int(image.height), int(image.width)):
            reasons.append("authorized_style_view_image_shape_mismatch")
        if bool(getattr(array, "flags", None).writeable):
            reasons.append("authorized_style_view_foreground_not_read_only")
    if len(view.analysis_bbox) != 4 or view.analysis_bbox[2] <= 0 or view.analysis_bbox[3] <= 0:
        reasons.append("authorized_style_view_analysis_bbox_invalid")
    return tuple(_unique_strings(reasons))


def _family_axis_from_variants(
    primary: Mapping[str, Any],
    neutral: Mapping[str, Any] | None,
    *,
    calibration_rule: Mapping[str, Any] | None = None,
) -> FamilyAxisObservation:
    primary_posterior = FontFamilyPosterior.from_mapping(
        primary.get("family_posterior")
    )
    neutral_posterior = (
        FontFamilyPosterior.from_mapping(neutral.get("family_posterior"))
        if isinstance(neutral, Mapping)
        else None
    )
    if primary_posterior is None:
        return FamilyAxisObservation(
            posterior=_font_family_posterior_from_masses(
                sans_mass=0.0,
                serif_mass=0.0,
                unknown_mass=0.0,
                label_count=0,
                known_label_count=0,
                unknown_label_count=0,
            ),
            promoted=False,
            reason="family_complete_posterior_unavailable",
        )

    variant_agreement = bool(
        neutral_posterior is not None
        and primary_posterior.leading_family
        and primary_posterior.leading_family == neutral_posterior.leading_family
    )
    posterior = (
        _font_family_posterior_from_masses(
            sans_mass=(
                primary_posterior.sans_mass + neutral_posterior.sans_mass
            )
            / 2.0,
            serif_mass=(
                primary_posterior.serif_mass + neutral_posterior.serif_mass
            )
            / 2.0,
            unknown_mass=(
                primary_posterior.unknown_mass + neutral_posterior.unknown_mass
            )
            / 2.0,
            label_count=max(
                primary_posterior.label_count,
                neutral_posterior.label_count,
            ),
            known_label_count=max(
                primary_posterior.known_label_count,
                neutral_posterior.known_label_count,
            ),
            unknown_label_count=max(
                primary_posterior.unknown_label_count,
                neutral_posterior.unknown_label_count,
            ),
        )
        if neutral_posterior is not None
        else primary_posterior
    )
    rule = dict(calibration_rule or family_posterior_calibration_rule())
    reliability = _validated_family_calibration_reliability(rule)
    known_mass_minimum = _unit_interval(rule.get("known_mass_minimum"))
    margin_minimum = _unit_interval(rule.get("margin_minimum"))
    entropy_maximum = _unit_interval(rule.get("normalized_entropy_maximum"))
    if (
        str(rule.get("version") or "") != FAMILY_CALIBRATION_VERSION
        or reliability is None
        or known_mass_minimum is None
        or margin_minimum is None
        or entropy_maximum is None
    ):
        return FamilyAxisObservation(
            posterior=posterior,
            promoted=False,
            calibration_reliability=float(reliability or 0.0),
            reason="family_posterior_calibration_rule_invalid",
            variant_agreement=variant_agreement,
        )
    if posterior.known_mass < known_mass_minimum:
        return FamilyAxisObservation(
            posterior=posterior,
            promoted=False,
            calibration_reliability=reliability,
            reason="family_posterior_known_mass_below_minimum",
            variant_agreement=variant_agreement,
        )
    if reliability < FAMILY_CALIBRATION_RELIABILITY_MINIMUM:
        return FamilyAxisObservation(
            posterior=posterior,
            promoted=False,
            calibration_reliability=reliability,
            reason="family_posterior_calibration_reliability_below_minimum",
            variant_agreement=variant_agreement,
        )
    if bool(rule.get("require_variant_agreement")) and not variant_agreement:
        return FamilyAxisObservation(
            posterior=posterior,
            promoted=False,
            calibration_reliability=reliability,
            reason=(
                "family_posterior_neutral_variant_unavailable"
                if neutral_posterior is None
                else "family_posterior_variant_disagreement"
            ),
            variant_agreement=False,
        )
    if (
        posterior.margin < margin_minimum
        or posterior.normalized_entropy > entropy_maximum
        or posterior.leading_family not in {"sans", "serif"}
    ):
        return FamilyAxisObservation(
            posterior=posterior,
            promoted=False,
            calibration_reliability=reliability,
            reason="family_posterior_calibration_gate_not_met",
            variant_agreement=variant_agreement,
        )
    confidence = min(
        reliability,
        max(
            posterior.conditional_sans_probability,
            posterior.conditional_serif_probability,
        ),
    )
    family_role = posterior.leading_family
    return FamilyAxisObservation(
        posterior=posterior,
        promoted=True,
        family_role=family_role,
        font_serif=family_role == "serif",
        confidence=confidence,
        calibration_reliability=reliability,
        reason="family_complete_posterior_calibrated_promotion",
        variant_agreement=variant_agreement,
    )


def _weight_axis_from_variants(
    primary: Mapping[str, Any],
    neutral: Mapping[str, Any] | None,
) -> tuple[str | None, float, str]:
    primary_weight = _detected_weight(primary)
    primary_confidence = _unit_interval(primary.get("confidence")) or 0.0
    if not isinstance(neutral, Mapping):
        return (
            primary_weight,
            primary_confidence * 0.7 if primary_weight else 0.0,
            "weight_neutral_vote_unavailable",
        )
    neutral_weight = _detected_weight(neutral)
    neutral_confidence = _unit_interval(neutral.get("confidence")) or 0.0
    if primary_weight and neutral_weight and primary_weight == neutral_weight:
        return (
            primary_weight,
            min(1.0, (primary_confidence + neutral_confidence) / 2.0),
            "weight_variant_consensus",
        )
    if primary_weight and (not neutral_weight or primary_confidence >= 0.6):
        return (
            primary_weight,
            primary_confidence * (0.65 if not neutral_weight else 0.5),
            "weight_variant_disagreement_primary_fill_contrast_retained",
        )
    return None, 0.0, "weight_variant_unresolved"


def _orientation_axis_from_variants(
    primary: Mapping[str, Any],
    neutral: Mapping[str, Any] | None,
) -> tuple[str, float, str]:
    primary_direction = str(primary.get("direction") or "").strip().lower()
    primary_confidence = _unit_interval(primary.get("direction_confidence")) or 0.0
    primary_valid = primary_direction in {"ltr", "ttb"}
    primary_reliable = bool(
        primary_valid and primary_confidence >= ORIENTATION_VOTE_MIN_CONFIDENCE
    )
    if not isinstance(neutral, Mapping):
        if not primary_valid:
            return "", 0.0, "orientation_primary_vote_invalid"
        if not primary_reliable:
            return "", 0.0, "orientation_primary_vote_below_confidence_floor"
        return (
            primary_direction,
            primary_confidence * 0.75,
            "orientation_neutral_vote_unavailable",
        )
    neutral_direction = str(neutral.get("direction") or "").strip().lower()
    neutral_confidence = _unit_interval(neutral.get("direction_confidence")) or 0.0
    neutral_reliable = bool(
        neutral_direction in {"ltr", "ttb"}
        and neutral_confidence >= ORIENTATION_VOTE_MIN_CONFIDENCE
    )
    if not primary_reliable and not neutral_reliable:
        return "", 0.0, "orientation_variant_votes_below_confidence_floor"
    if primary_reliable and not neutral_reliable:
        return (
            primary_direction,
            primary_confidence * 0.75,
            "orientation_single_reliable_primary_vote",
        )
    if neutral_reliable and not primary_reliable:
        return (
            neutral_direction,
            neutral_confidence * 0.75,
            "orientation_single_reliable_neutral_vote",
        )
    if neutral_direction == primary_direction:
        return (
            primary_direction,
            min(1.0, (primary_confidence + neutral_confidence) / 2.0),
            "orientation_variant_consensus",
        )
    if primary_confidence >= 0.8:
        return (
            primary_direction,
            primary_confidence * 0.55,
            "orientation_variant_disagreement_primary_fill_contrast_retained",
        )
    return "", 0.0, "orientation_variant_unresolved"


def _detected_weight(detection: Mapping[str, Any]) -> str | None:
    direct = _font_weight_from_label(str(detection.get("font_path") or ""))
    if direct:
        return direct
    scores: dict[str, float] = {}
    for item in detection.get("top_candidates") or []:
        if not isinstance(item, Mapping):
            continue
        weight = _font_weight_from_label(str(item.get("path") or ""))
        confidence = _unit_interval(item.get("confidence")) or 0.0
        if weight and confidence > 0:
            scores[weight] = scores.get(weight, 0.0) + confidence
    if not scores:
        return None
    return max(scores, key=lambda key: (scores[key], key))


def _detector_variant_summary(
    primary: Mapping[str, Any],
    neutral: Mapping[str, Any] | None,
    *,
    primary_sha256: str,
    neutral_sha256: str,
    neutral_error: str,
) -> dict[str, Any]:
    def compact(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        raw_serif = value.get("font_serif")
        family_posterior = FontFamilyPosterior.from_mapping(
            value.get("family_posterior")
        )
        return {
            "font_path": str(value.get("font_path") or ""),
            "font_serif": raw_serif if isinstance(raw_serif, bool) else None,
            "family_posterior": (
                family_posterior.to_audit_dict()
                if family_posterior is not None
                else None
            ),
            "font_weight": _detected_weight(value) or "",
            "confidence": float(_unit_interval(value.get("confidence")) or 0.0),
            "direction": str(value.get("direction") or ""),
            "direction_confidence": float(
                _unit_interval(value.get("direction_confidence")) or 0.0
            ),
            "text_size_ratio_diagnostic_only": _unit_interval(
                value.get("text_size_ratio")
            ),
            "stroke_width_ratio_diagnostic_only": _unit_interval(
                value.get("stroke_width_ratio")
            ),
        }

    return {
        "variant_contract": "fill_contrast_primary_plus_neutral_disagreement_probe",
        "primary": compact(primary),
        "neutral": compact(neutral),
        "primary_input_sha256": primary_sha256,
        "neutral_input_sha256": neutral_sha256,
        "neutral_error": neutral_error,
        "model_scale_and_paint_regressions_diagnostic_only": True,
    }


def _semantic_style_class(role: str) -> str:
    lowered = str(role or "").strip().lower()
    return "caption" if any(token in lowered for token in ("caption", "background", "narration", "sign")) else "dialogue"


def _font_weight_from_label(label: str) -> str | None:
    lowered = unicodedata.normalize("NFKC", str(label or "")).lower()
    basename = os.path.basename(lowered)
    if any(
        token in basename
        for token in ("black", "heavy", "ultra", "super", "w9", "w10", "w11", "w12", "w13", "w14")
    ):
        return "black"
    if any(
        token in basename
        for token in (
            "bold",
            "semibold",
            "demibold",
            "demi-bold",
            "extrabold",
            "extra-bold",
            "-db",
            "_db",
            "-eb",
            "_eb",
            "w6",
            "w7",
            "w8",
        )
    ) or re.search(r"(?:^|[-_. ])(?:b|bd)(?:[-_. ]|$)|b\.(?:ttf|otf|ttc)$", basename):
        return "bold"
    if any(
        token in basename
        for token in (
            "regular",
            "normal",
            "book",
            "light",
            "thin",
            "medium",
            "roman",
            "w1",
            "w2",
            "w3",
            "w4",
            "w5",
        )
    ) or re.search(r"(?:^|[-_. ])(?:r|l|m|el)(?:[-_. ]|$)", basename):
        return "regular"
    return None


def _normalized_factorized_style_name(path_value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(path_value or ""))
    basename = os.path.splitext(os.path.basename(normalized))[0]
    basename = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", basename)
    basename = basename.lower()
    basename = re.sub(r"[_+.,()\[\]{}-]+", " ", basename)
    return re.sub(r"\s+", " ", basename).strip()


def _factorized_label_attribute_codes(
    label: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Return family, face, and strength codes without resolving a style."""

    family_value = label.get("serif")
    family_code = (
        1
        if family_value is True
        else 0
        if family_value is False
        else 2
    )

    name = _normalized_factorized_style_name(
        str(label.get("path") or "")
    )
    tokens = name.split()
    joined = " ".join(tokens)

    def contains_phrase(phrase: str) -> bool:
        return bool(
            re.search(
                rf"(?:^| ){re.escape(phrase)}(?:$| )",
                joined,
            )
        )

    slender = any(
        contains_phrase(phrase)
        for phrase in (
            "hairline",
            "thin",
            "ultra thin",
            "extra thin",
            "ultra light",
            "extra light",
            "demi light",
            "semi light",
            "light",
        )
    )
    ordinary = any(
        contains_phrase(phrase)
        for phrase in (
            "book",
            "text",
            "regular",
            "normal",
            "roman",
            "medium",
        )
    )
    strong = any(
        contains_phrase(phrase)
        for phrase in (
            "semi bold",
            "demi bold",
            "bold",
            "extra bold",
            "ultra bold",
            "super bold",
            "black",
            "heavy",
            "extra black",
            "ultra black",
        )
    )

    for token in tokens:
        w_match = re.fullmatch(r"w([1-9]|1[0-4])", token)
        numeric_match = re.fullmatch(
            r"(100|200|300|400|500|600|700|800|900|950|1000)",
            token,
        )
        number = (
            int(w_match.group(1)) * 100
            if w_match
            else int(numeric_match.group(1))
            if numeric_match
            else None
        )
        if number is None:
            continue
        if number <= 300:
            slender = True
        elif number <= 500:
            ordinary = True
        else:
            strong = True

    matched_groups = sum((slender, ordinary, strong))
    if matched_groups != 1:
        return family_code, 2, 2
    if slender:
        return family_code, 1, 0
    if ordinary:
        return family_code, 0, 0
    return family_code, 0, 1


def _build_factorized_attribute_taxonomy(
    labels: Sequence[Mapping[str, Any]],
    *,
    label_count: int,
) -> _FactorizedAttributeTaxonomy:
    family_codes = np.full(label_count, 2, dtype=np.uint8)
    face_codes = np.full(label_count, 2, dtype=np.uint8)
    strength_codes = np.full(label_count, 2, dtype=np.uint8)
    for index in range(label_count):
        family, face, strength = _factorized_label_attribute_codes(
            _label_at(labels, index)
        )
        family_codes[index] = family
        face_codes[index] = face
        strength_codes[index] = strength
    for codes in (family_codes, face_codes, strength_codes):
        codes.setflags(write=False)
    return _FactorizedAttributeTaxonomy(
        label_count=label_count,
        generic_family_codes=family_codes,
        face_character_codes=face_codes,
        weight_strength_codes=strength_codes,
    )


def _factorized_axis_posterior(
    probabilities: np.ndarray,
    codes: np.ndarray,
    *,
    classes: tuple[str, str],
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total > 0.0:
        values = values / total
        bins = np.bincount(
            np.asarray(codes, dtype=np.int64),
            weights=values,
            minlength=len(classes) + 1,
        )
    else:
        bins = np.zeros(len(classes) + 1, dtype=np.float64)
        bins[-1] = 1.0

    class_masses = [float(bins[index]) for index in range(len(classes))]
    unknown_mass = float(bins[len(classes)])
    known_mass = float(sum(class_masses))
    conditional_values = (
        [mass / known_mass for mass in class_masses]
        if known_mass > 0.0
        else [0.0 for _ in classes]
    )
    ordered = sorted(conditional_values, reverse=True)
    margin = (
        float(ordered[0] - ordered[1])
        if len(ordered) >= 2
        else float(ordered[0])
        if ordered
        else 0.0
    )
    if known_mass <= 0.0:
        leading_candidate = ""
        normalized_entropy = 1.0
    else:
        leading_index = int(np.argmax(conditional_values))
        leading_candidate = (
            classes[leading_index]
            if conditional_values.count(conditional_values[leading_index])
            == 1
            else ""
        )
        entropy = -sum(
            probability * math.log(probability)
            for probability in conditional_values
            if probability > 0.0
        )
        normalized_entropy = (
            entropy / math.log(len(classes))
            if len(classes) > 1
            else 0.0
        )

    masses = {
        name: class_masses[index]
        for index, name in enumerate(classes)
    }
    masses["unknown"] = unknown_mass
    conditional = {
        name: conditional_values[index]
        for index, name in enumerate(classes)
    }
    unknown_code = len(classes)
    return {
        "schema_version": "yuzumarker_factorized_attribute_axis_v1",
        "classes": list(classes),
        "label_count": int(values.size),
        "known_label_count": int(np.count_nonzero(codes != unknown_code)),
        "unknown_label_count": int(np.count_nonzero(codes == unknown_code)),
        "masses": masses,
        "known_mass": known_mass,
        "unknown_mass": unknown_mass,
        "conditional_probabilities": conditional,
        "leading_candidate": leading_candidate,
        "margin": margin,
        "normalized_entropy": float(normalized_entropy),
    }


def _factorized_attribute_posterior_from_probabilities(
    probabilities: Any,
    taxonomy: _FactorizedAttributeTaxonomy,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size != taxonomy.label_count:
        raise ValueError(
            "factorized attribute taxonomy does not match detector output"
        )
    return {
        "schema_version": FACTORIZED_ATTRIBUTE_POSTERIOR_VERSION,
        "taxonomy_version": FACTORIZED_ATTRIBUTE_TAXONOMY_VERSION,
        "label_count": int(values.size),
        "generic_family": _factorized_axis_posterior(
            values,
            taxonomy.generic_family_codes,
            classes=("sans", "serif"),
        ),
        "face_character": _factorized_axis_posterior(
            values,
            taxonomy.face_character_codes,
            classes=("standard_candidate", "slender_candidate"),
        ),
        "weight_strength": _factorized_axis_posterior(
            values,
            taxonomy.weight_strength_codes,
            classes=("normal_candidate", "strong_candidate"),
        ),
    }


def _heuristic_detection(image: Any) -> dict[str, Any]:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    dark_ratio = float((array < 96).mean()) if array.size else 0.0
    light_on_dark = float(array.mean()) < 120.0 if array.size else False
    return {
        "confidence": 1.0,
        "font_path": "heuristic/serif" if dark_ratio < 0.04 else "heuristic/sans",
        "font_language": "CJK",
        "font_serif": bool(dark_ratio < 0.04),
        "top_candidates": [],
        "direction": "ttb" if image.height >= image.width else "ltr",
        "direction_confidence": 1.0,
        "text_color": "#FFFFFF" if light_on_dark else "#000000",
        "stroke_color": "#000000" if light_on_dark else "#FFFFFF",
        "stroke_width_ratio": 0.004 if light_on_dark else 0.002,
        "text_size_ratio": 0.0,
        "line_spacing_ratio": 0.0,
        "angle_degrees": 0.0,
    }


def _load_onnx_session(model_path: str, *, use_gpu: bool) -> Any:
    key = (os.path.abspath(model_path), bool(use_gpu))
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    import onnxruntime as ort

    preload_error = ""
    if use_gpu:
        preload_dlls = getattr(ort, "preload_dlls", None)
        if callable(preload_dlls):
            try:
                preload_dlls()
            except Exception as exc:
                preload_error = f"{type(exc).__name__}:{exc}"
    available = [str(provider) for provider in ort.get_available_providers()]
    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")
    initialization_error = ""
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        if not use_gpu or providers == ["CPUExecutionProvider"]:
            raise
        initialization_error = f"{type(exc).__name__}:{exc}"
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    active = [str(provider) for provider in session.get_providers()]
    requested = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    fallback_reason = ""
    if use_gpu and "CUDAExecutionProvider" not in active:
        fallback_reason = (
            "cuda_execution_provider_not_available"
            if "CUDAExecutionProvider" not in available
            else "cuda_execution_provider_initialization_failed"
        )
    _SESSION_PROVIDER_METADATA[key] = {
        "gpu_requested": bool(use_gpu),
        "requested_execution_provider": requested,
        "available_execution_providers": available,
        "active_execution_providers": active,
        "primary_execution_provider": active[0] if active else "",
        "provider_fallback_reason": fallback_reason,
        "provider_preload_error": preload_error,
        "provider_initialization_error": initialization_error,
    }
    _SESSION_CACHE[key] = session
    return session


def _onnx_session_provider_metadata(model_path: str, *, use_gpu: bool, session: Any) -> dict[str, Any]:
    key = (os.path.abspath(model_path), bool(use_gpu))
    metadata = dict(_SESSION_PROVIDER_METADATA.get(key) or {})
    if metadata:
        return metadata
    active = [str(provider) for provider in session.get_providers()]
    return {
        "gpu_requested": bool(use_gpu),
        "requested_execution_provider": "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider",
        "available_execution_providers": [],
        "active_execution_providers": active,
        "primary_execution_provider": active[0] if active else "",
        "provider_fallback_reason": "cuda_execution_provider_initialization_failed" if use_gpu and "CUDAExecutionProvider" not in active else "",
        "provider_preload_error": "",
    }


def _copy_provider_metadata(result: Any, detector: Any) -> None:
    result.gpu_requested = bool(getattr(detector, "gpu_requested", False))
    result.requested_execution_provider = str(getattr(detector, "requested_execution_provider", "") or "")
    result.available_execution_providers = list(getattr(detector, "available_execution_providers", []) or [])
    result.active_execution_providers = list(getattr(detector, "active_execution_providers", []) or [])
    result.primary_execution_provider = str(getattr(detector, "primary_execution_provider", "") or "")
    result.provider_fallback_reason = str(getattr(detector, "provider_fallback_reason", "") or "")
    result.provider_preload_error = str(getattr(detector, "provider_preload_error", "") or "")


def _load_font_labels(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise RuntimeError("YuzuMarker font labels must be a list")
    return [dict(item) if isinstance(item, Mapping) else {"path": str(item)} for item in labels]


def _label_at(labels: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    return labels[index] if 0 <= index < len(labels) else {}


def _softmax(values: Any) -> Any:
    array = np.asarray(values, dtype=np.float32)
    if not array.size or not np.all(np.isfinite(array)):
        return np.zeros_like(array)
    array = array - float(array.max())
    exp = np.exp(array)
    denominator = float(exp.sum())
    return np.zeros_like(array) if denominator <= 0 else exp / denominator


def _binary_normalized_entropy(left: float, right: float) -> float:
    values = [max(0.0, float(left)), max(0.0, float(right))]
    total = sum(values)
    if total <= 0.0:
        return 1.0
    probabilities = [value / total for value in values]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return entropy / math.log(2.0)


def _wilson_score_lower_bound(correct: int, promoted: int) -> float:
    """Return the two-sided 95% Wilson lower bound for promotion precision."""

    if promoted <= 0 or correct < 0 or correct > promoted:
        return 0.0
    point = correct / promoted
    z_squared = FAMILY_CALIBRATION_Z_SCORE * FAMILY_CALIBRATION_Z_SCORE
    denominator = 1.0 + z_squared / promoted
    center = point + z_squared / (2.0 * promoted)
    adjustment = FAMILY_CALIBRATION_Z_SCORE * math.sqrt(
        (point * (1.0 - point) + z_squared / (4.0 * promoted))
        / promoted
    )
    return max(0.0, (center - adjustment) / denominator)


def _validated_family_calibration_reliability(
    rule: Mapping[str, Any],
) -> float | None:
    """Validate support metadata and return its conservative reliability."""

    if (
        str(rule.get("version") or "") != FAMILY_CALIBRATION_VERSION
        or str(rule.get("reliability_method") or "")
        != FAMILY_CALIBRATION_RELIABILITY_METHOD
    ):
        return None

    integer_values: dict[str, int] = {}
    for key in (
        "calibration_promoted",
        "calibration_correct",
        "calibration_false_high_confidence",
    ):
        value = rule.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        integer_values[key] = value

    promoted = integer_values["calibration_promoted"]
    correct = integer_values["calibration_correct"]
    false_high_confidence = integer_values[
        "calibration_false_high_confidence"
    ]
    if correct > promoted or false_high_confidence != promoted - correct:
        return None

    point_precision = _unit_interval(rule.get("calibration_point_precision"))
    reliability = _unit_interval(rule.get("calibration_reliability"))
    if point_precision is None or reliability is None:
        return None
    expected_point_precision = correct / promoted if promoted else 0.0
    expected_reliability = _wilson_score_lower_bound(correct, promoted)
    if not math.isclose(
        point_precision,
        expected_point_precision,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        reliability,
        expected_reliability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None

    promotion_enabled = rule.get("promotion_enabled")
    if not isinstance(promotion_enabled, bool):
        return None
    expected_enabled = bool(
        expected_reliability >= FAMILY_CALIBRATION_RELIABILITY_MINIMUM
        and false_high_confidence == 0
    )
    if promotion_enabled != expected_enabled:
        return None
    expected_status = (
        "calibrated_promotion_region"
        if expected_enabled
        else "abstain_all_insufficient_statistical_support"
    )
    if str(rule.get("status") or "") != expected_status:
        return None
    return reliability


def _font_family_posterior_from_masses(
    *,
    sans_mass: float,
    serif_mass: float,
    unknown_mass: float,
    label_count: int,
    known_label_count: int,
    unknown_label_count: int,
) -> FontFamilyPosterior:
    masses = [
        max(0.0, float(sans_mass)),
        max(0.0, float(serif_mass)),
        max(0.0, float(unknown_mass)),
    ]
    total = sum(masses)
    if total > 0.0:
        masses = [mass / total for mass in masses]
    else:
        masses = [0.0, 0.0, 0.0]
    sans_mass, serif_mass, unknown_mass = masses
    known_mass = sans_mass + serif_mass
    conditional_sans = sans_mass / known_mass if known_mass > 0.0 else 0.0
    conditional_serif = serif_mass / known_mass if known_mass > 0.0 else 0.0
    leading_family = (
        "sans"
        if conditional_sans > conditional_serif
        else "serif"
        if conditional_serif > conditional_sans
        else ""
    )
    return FontFamilyPosterior(
        label_count=max(0, int(label_count)),
        known_label_count=max(0, int(known_label_count)),
        unknown_label_count=max(0, int(unknown_label_count)),
        sans_mass=sans_mass,
        serif_mass=serif_mass,
        unknown_mass=unknown_mass,
        known_mass=known_mass,
        conditional_sans_probability=conditional_sans,
        conditional_serif_probability=conditional_serif,
        leading_family=leading_family,
        margin=abs(conditional_sans - conditional_serif),
        normalized_entropy=_binary_normalized_entropy(
            conditional_sans,
            conditional_serif,
        ),
    )


def _font_family_posterior_from_probabilities(
    probabilities: Any,
    labels: Sequence[Mapping[str, Any]],
) -> FontFamilyPosterior:
    array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    sans_mass = 0.0
    serif_mass = 0.0
    unknown_mass = 0.0
    known_label_count = 0
    unknown_label_count = 0
    for index, raw_probability in enumerate(array.tolist()):
        probability = max(0.0, float(raw_probability))
        label = _label_at(labels, index)
        family = label.get("serif") if isinstance(label, Mapping) else None
        if isinstance(family, bool):
            known_label_count += 1
            if family:
                serif_mass += probability
            else:
                sans_mass += probability
        else:
            unknown_label_count += 1
            unknown_mass += probability
    return _font_family_posterior_from_masses(
        sans_mass=sans_mass,
        serif_mass=serif_mass,
        unknown_mass=unknown_mass,
        label_count=int(array.size),
        known_label_count=known_label_count,
        unknown_label_count=unknown_label_count,
    )


def family_posterior_calibration_rule() -> dict[str, Any]:
    """Return the calibration-split-frozen Stage 1A promotion rule."""

    point_precision = (
        FAMILY_CALIBRATION_CORRECT / FAMILY_CALIBRATION_PROMOTED
        if FAMILY_CALIBRATION_PROMOTED
        else 0.0
    )
    reliability = _wilson_score_lower_bound(
        FAMILY_CALIBRATION_CORRECT,
        FAMILY_CALIBRATION_PROMOTED,
    )
    promotion_enabled = bool(
        reliability >= FAMILY_CALIBRATION_RELIABILITY_MINIMUM
        and FAMILY_CALIBRATION_FALSE_HIGH_CONFIDENCE == 0
    )
    return {
        "version": FAMILY_CALIBRATION_VERSION,
        "known_mass_minimum": FAMILY_KNOWN_MASS_MINIMUM,
        "margin_minimum": FAMILY_MARGIN_MINIMUM,
        "normalized_entropy_maximum": FAMILY_NORMALIZED_ENTROPY_MAXIMUM,
        "require_variant_agreement": FAMILY_REQUIRE_VARIANT_AGREEMENT,
        "reliability_method": FAMILY_CALIBRATION_RELIABILITY_METHOD,
        "calibration_promoted": FAMILY_CALIBRATION_PROMOTED,
        "calibration_correct": FAMILY_CALIBRATION_CORRECT,
        "calibration_false_high_confidence": (
            FAMILY_CALIBRATION_FALSE_HIGH_CONFIDENCE
        ),
        "calibration_point_precision": point_precision,
        "calibration_reliability": reliability,
        "promotion_enabled": promotion_enabled,
        "status": (
            "calibrated_promotion_region"
            if promotion_enabled
            else "abstain_all_insufficient_statistical_support"
        ),
    }


def _rgb_from_unit_values(values: Any) -> str | None:
    try:
        raw = list(values)
    except Exception:
        return None
    if len(raw) < 3:
        return None
    numbers = [_unit_interval(value) for value in raw[:3]]
    if any(value is None for value in numbers):
        return None
    channels = [int(round(float(value) * 255.0)) for value in numbers]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def _compact_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, Mapping):
            continue
        output.append(
            {
                "index": item.get("index"),
                "confidence": _float(item.get("confidence")),
                "path": str(item.get("path") or ""),
                "language": str(item.get("language") or ""),
                "serif": (
                    item.get("serif")
                    if isinstance(item.get("serif"), bool)
                    else None
                ),
            }
        )
    return output


def _image_sha256(image: Any) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_jsonish(item) for item in value]
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    return value


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _bounded_float(
    value: Any,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number < minimum or number > maximum:
        return None
    return number


def _unit_interval(value: Any) -> float | None:
    return _bounded_float(value, minimum=0.0, maximum=1.0)


def _hex_color(value: Any) -> str:
    text = str(value or "").strip().upper()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
        except ValueError:
            return ""
        return text
    return ""
