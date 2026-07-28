# -*- coding: utf-8 -*-
"""Immutable source-style contracts shared across Block-F owners.

This module owns transport contracts only. It does not run a model, arbitrate
parents, select a target font, inspect layout, or decide whether text renders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


SOURCE_FONT_CANDIDATE_V1 = "source_font_candidate_v1"
SOURCE_FONT_SCORE_SUPPORT_V1 = "source_font_score_support_v1"
SOURCE_FONT_OVERLAP_BOUNDS_V1 = "source_font_overlap_bounds_v1"
SOURCE_STYLE_EVIDENCE_BINDING_V1 = "source_style_evidence_binding_v1"
SOURCE_FONT_OBSERVATION_V3 = "source_font_observation_v3"
TARGET_FONT_AFFINITY_OBSERVATION_V1 = "target_font_affinity_observation_v1"
TARGET_FONT_AFFINITY_OBSERVATION_KEY = TARGET_FONT_AFFINITY_OBSERVATION_V1

TARGET_FONT_AFFINITY_ROLE_IDS = (
    "sans_regular",
    "sans_medium",
    "sans_bold",
    "sans_black",
    "serif_regular",
    "serif_semibold",
    "serif_bold",
)

SOURCE_FONT_SUPPORT_FLOOR_MET = "retained_mass_floor_met"
SOURCE_FONT_SUPPORT_TRUNCATED = "truncated_uncertain"
SOURCE_FONT_SUPPORT_STATUSES = frozenset(
    {
        SOURCE_FONT_SUPPORT_FLOOR_MET,
        SOURCE_FONT_SUPPORT_TRUNCATED,
    }
)

_MASS_TOLERANCE = 1e-9


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source-style JSON values must be finite")
        return value
    raise TypeError(
        "source-style JSON values must contain only mappings, sequences, "
        "strings, booleans, finite numbers, or null"
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _finite_unit(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite unit value") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{field_name} must be a finite unit value")
    return number


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


@dataclass(frozen=True)
class SourceFontCandidate:
    """One model-label identity and its normalized model score."""

    catalog_version: str
    class_index: int
    label_identity: str
    normalized_model_score: float
    descriptors: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SOURCE_FONT_CANDIDATE_V1

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FONT_CANDIDATE_V1:
            raise ValueError("unsupported source-font candidate schema")
        object.__setattr__(
            self,
            "catalog_version",
            _required_text(self.catalog_version, "catalog_version"),
        )
        if isinstance(self.class_index, bool) or int(self.class_index) < 0:
            raise ValueError("class_index must be a non-negative integer")
        object.__setattr__(self, "class_index", int(self.class_index))
        object.__setattr__(
            self,
            "label_identity",
            _required_text(self.label_identity, "label_identity"),
        )
        object.__setattr__(
            self,
            "normalized_model_score",
            _finite_unit(
                self.normalized_model_score,
                "normalized_model_score",
            ),
        )
        object.__setattr__(
            self,
            "descriptors",
            _freeze_json(dict(self.descriptors or {})),
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "class_index": self.class_index,
            "label_identity": self.label_identity,
            "normalized_model_score": self.normalized_model_score,
            "descriptors": _plain_json(self.descriptors),
        }


@dataclass(frozen=True)
class SourceFontScoreSupportV1:
    """Adaptive sparse support plus exact unretained score mass."""

    catalog_version: str
    label_count: int
    retained_mass_floor: float
    candidate_ceiling: int
    candidates: tuple[SourceFontCandidate, ...]
    retained_mass: float
    residual_mass: float
    status: str
    normalized_entropy: float
    margin: float
    schema_version: str = SOURCE_FONT_SCORE_SUPPORT_V1

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FONT_SCORE_SUPPORT_V1:
            raise ValueError("unsupported source-font score-support schema")
        catalog_version = _required_text(
            self.catalog_version,
            "catalog_version",
        )
        object.__setattr__(self, "catalog_version", catalog_version)
        if isinstance(self.label_count, bool) or int(self.label_count) <= 0:
            raise ValueError("label_count must be a positive integer")
        label_count = int(self.label_count)
        object.__setattr__(self, "label_count", label_count)
        retained_mass_floor = _finite_unit(
            self.retained_mass_floor,
            "retained_mass_floor",
        )
        if retained_mass_floor <= 0.0:
            raise ValueError("retained_mass_floor must be greater than zero")
        object.__setattr__(
            self,
            "retained_mass_floor",
            retained_mass_floor,
        )
        if (
            isinstance(self.candidate_ceiling, bool)
            or int(self.candidate_ceiling) <= 0
            or int(self.candidate_ceiling) > label_count
        ):
            raise ValueError(
                "candidate_ceiling must be within the label catalog"
            )
        candidate_ceiling = int(self.candidate_ceiling)
        object.__setattr__(self, "candidate_ceiling", candidate_ceiling)

        candidates = tuple(self.candidates or ())
        if not candidates or len(candidates) > candidate_ceiling:
            raise ValueError(
                "candidate support must be non-empty and within the ceiling"
            )
        if any(not isinstance(item, SourceFontCandidate) for item in candidates):
            raise TypeError("candidates must be SourceFontCandidate records")
        if any(item.catalog_version != catalog_version for item in candidates):
            raise ValueError("candidate catalog versions must match support")
        indices = [item.class_index for item in candidates]
        if len(set(indices)) != len(indices):
            raise ValueError("candidate class indices must be unique")
        if any(index >= label_count for index in indices):
            raise ValueError("candidate class index exceeds label catalog")
        expected_order = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.normalized_model_score,
                    item.class_index,
                ),
            )
        )
        if candidates != expected_order:
            raise ValueError(
                "candidates must use score-descending, class-index tie order"
            )
        object.__setattr__(self, "candidates", candidates)

        retained_mass = _finite_unit(self.retained_mass, "retained_mass")
        residual_mass = _finite_unit(self.residual_mass, "residual_mass")
        candidate_sum = math.fsum(
            item.normalized_model_score for item in candidates
        )
        if abs(candidate_sum - retained_mass) > _MASS_TOLERANCE:
            raise ValueError("retained_mass must equal candidate-score sum")
        if abs(retained_mass + residual_mass - 1.0) > _MASS_TOLERANCE:
            raise ValueError("retained and residual score mass must sum to one")
        object.__setattr__(self, "retained_mass", retained_mass)
        object.__setattr__(self, "residual_mass", residual_mass)

        status = str(self.status or "")
        if status not in SOURCE_FONT_SUPPORT_STATUSES:
            raise ValueError("unsupported source-font support status")
        previous_mass = retained_mass - candidates[-1].normalized_model_score
        if status == SOURCE_FONT_SUPPORT_FLOOR_MET:
            if retained_mass + _MASS_TOLERANCE < retained_mass_floor:
                raise ValueError("floor-met support does not reach its floor")
            if (
                len(candidates) > 1
                and previous_mass >= retained_mass_floor - _MASS_TOLERANCE
            ):
                raise ValueError("support is not the smallest floor prefix")
        elif (
            len(candidates) != candidate_ceiling
            or retained_mass >= retained_mass_floor - _MASS_TOLERANCE
        ):
            raise ValueError(
                "truncated support must exhaust its ceiling below the floor"
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "normalized_entropy",
            _finite_unit(self.normalized_entropy, "normalized_entropy"),
        )
        object.__setattr__(
            self,
            "margin",
            _finite_unit(self.margin, "margin"),
        )

    @property
    def leading_candidate(self) -> SourceFontCandidate:
        return self.candidates[0]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "label_count": self.label_count,
            "retained_mass_floor": self.retained_mass_floor,
            "candidate_ceiling": self.candidate_ceiling,
            "status": self.status,
            "retained_mass": self.retained_mass,
            "residual_mass": self.residual_mass,
            "normalized_entropy": self.normalized_entropy,
            "margin": self.margin,
            "candidates": [
                item.to_audit_dict() for item in self.candidates
            ],
        }


@dataclass(frozen=True)
class SourceFontOverlapBoundsV1:
    """Conservative interval for full-distribution histogram overlap."""

    lower_bound: float
    upper_bound: float
    shared_retained_identity_count: int
    schema_version: str = SOURCE_FONT_OVERLAP_BOUNDS_V1

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FONT_OVERLAP_BOUNDS_V1:
            raise ValueError("unsupported source-font overlap schema")
        lower = _finite_unit(self.lower_bound, "lower_bound")
        upper = _finite_unit(self.upper_bound, "upper_bound")
        if lower > upper + _MASS_TOLERANCE:
            raise ValueError("overlap lower bound exceeds upper bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        if (
            isinstance(self.shared_retained_identity_count, bool)
            or int(self.shared_retained_identity_count) < 0
        ):
            raise ValueError(
                "shared_retained_identity_count must be non-negative"
            )
        object.__setattr__(
            self,
            "shared_retained_identity_count",
            int(self.shared_retained_identity_count),
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "shared_retained_identity_count": (
                self.shared_retained_identity_count
            ),
        }


def source_font_overlap_bounds(
    left: SourceFontScoreSupportV1,
    right: SourceFontScoreSupportV1,
) -> SourceFontOverlapBoundsV1:
    """Bound histogram intersection without inventing residual identities."""

    if not isinstance(left, SourceFontScoreSupportV1) or not isinstance(
        right,
        SourceFontScoreSupportV1,
    ):
        raise TypeError("overlap inputs must be score-support contracts")
    if (
        left.catalog_version != right.catalog_version
        or left.label_count != right.label_count
    ):
        raise ValueError("overlap supports must share one label catalog")
    left_scores = {
        item.label_identity: item.normalized_model_score
        for item in left.candidates
    }
    right_scores = {
        item.label_identity: item.normalized_model_score
        for item in right.candidates
    }
    shared = set(left_scores) & set(right_scores)
    lower = math.fsum(
        min(left_scores[identity], right_scores[identity])
        for identity in shared
    )
    upper = min(1.0, lower + left.residual_mass + right.residual_mass)
    return SourceFontOverlapBoundsV1(
        lower_bound=lower,
        upper_bound=upper,
        shared_retained_identity_count=len(shared),
    )


@dataclass(frozen=True)
class SourceStyleEvidenceBindingV1:
    """Digest-only binding to independently owned source-style evidence."""

    source_identity_sha256: str
    evidence_schema_version: str
    evidence_sha256: str
    schema_version: str = SOURCE_STYLE_EVIDENCE_BINDING_V1

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_STYLE_EVIDENCE_BINDING_V1:
            raise ValueError("unsupported source-style binding schema")
        for field_name in (
            "source_identity_sha256",
            "evidence_schema_version",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_identity_sha256": self.source_identity_sha256,
            "evidence_schema_version": self.evidence_schema_version,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class TargetFontAffinityObservationV1:
    """Yuzu-space affinity to the fixed installed target-role catalog.

    This is an observation only.  It does not select a font role and carries
    no claim about the exact source font's metadata.
    """

    catalog_identity_sha256: str
    descriptor_policy_version: str
    source_input_sha256: str
    model_identity: str
    label_catalog_version: str
    provider_provenance: Mapping[str, Any]
    role_scores: Mapping[str, float]
    schema_version: str = TARGET_FONT_AFFINITY_OBSERVATION_V1

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_FONT_AFFINITY_OBSERVATION_V1:
            raise ValueError("unsupported target-font affinity schema")
        for field_name in (
            "catalog_identity_sha256",
            "descriptor_policy_version",
            "source_input_sha256",
            "model_identity",
            "label_catalog_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )

        provenance = dict(self.provider_provenance or {})
        if not provenance:
            raise ValueError("provider_provenance must be non-empty")
        primary_provider = _required_text(
            provenance.get("primary_execution_provider"),
            "provider_provenance.primary_execution_provider",
        )
        active_providers = tuple(
            str(provider or "").strip()
            for provider in (
                provenance.get("active_execution_providers") or ()
            )
            if str(provider or "").strip()
        )
        if primary_provider not in active_providers:
            raise ValueError(
                "provider provenance must include its primary provider in "
                "the active provider list"
            )
        object.__setattr__(
            self,
            "provider_provenance",
            _freeze_json(provenance),
        )

        raw_scores = dict(self.role_scores or {})
        if set(raw_scores) != set(TARGET_FONT_AFFINITY_ROLE_IDS):
            raise ValueError(
                "role_scores must contain exactly the seven registered "
                "automatic target roles"
            )
        ordered_scores = {
            role_id: _finite_unit(raw_scores[role_id], f"role_scores.{role_id}")
            for role_id in TARGET_FONT_AFFINITY_ROLE_IDS
        }
        object.__setattr__(
            self,
            "role_scores",
            MappingProxyType(ordered_scores),
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_identity_sha256": self.catalog_identity_sha256,
            "descriptor_policy_version": self.descriptor_policy_version,
            "source_input_sha256": self.source_input_sha256,
            "model_identity": self.model_identity,
            "label_catalog_version": self.label_catalog_version,
            "provider_provenance": _plain_json(self.provider_provenance),
            "role_scores": dict(self.role_scores),
        }


@dataclass(frozen=True)
class SourceFontObservationV3:
    """Dedicated source-font identity observation for one authorized view."""

    source_identity_sha256: str
    authorized_view_sha256: str
    model_identity: str
    label_catalog_version: str
    support_policy_version: str
    primary_input_sha256: str
    neutral_input_sha256: str
    primary: SourceFontScoreSupportV1
    neutral: SourceFontScoreSupportV1 | None
    neutral_error: str
    variant_agreement: str
    variant_overlap_bounds: SourceFontOverlapBoundsV1 | None
    target_font_affinity: TargetFontAffinityObservationV1 | None = None
    style_evidence_binding: SourceStyleEvidenceBindingV1 | None = None
    schema_version: str = SOURCE_FONT_OBSERVATION_V3

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FONT_OBSERVATION_V3:
            raise ValueError("unsupported source-font observation schema")
        for field_name in (
            "source_identity_sha256",
            "authorized_view_sha256",
            "model_identity",
            "label_catalog_version",
            "support_policy_version",
            "primary_input_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "neutral_input_sha256",
            str(self.neutral_input_sha256 or ""),
        )
        object.__setattr__(
            self,
            "neutral_error",
            str(self.neutral_error or ""),
        )
        if not isinstance(self.primary, SourceFontScoreSupportV1):
            raise TypeError("primary must be a score-support contract")
        if self.primary.catalog_version != self.label_catalog_version:
            raise ValueError("primary support catalog does not match observation")
        if self.neutral is not None:
            if not isinstance(self.neutral, SourceFontScoreSupportV1):
                raise TypeError("neutral must be a score-support contract")
            if self.neutral.catalog_version != self.label_catalog_version:
                raise ValueError(
                    "neutral support catalog does not match observation"
                )
            if not self.neutral_input_sha256:
                raise ValueError(
                    "neutral input hash is required with neutral support"
                )
            if not isinstance(
                self.variant_overlap_bounds,
                SourceFontOverlapBoundsV1,
            ):
                raise TypeError(
                    "variant overlap bounds are required with neutral support"
                )
        elif self.variant_overlap_bounds is not None:
            raise ValueError(
                "variant overlap bounds require neutral support"
            )
        agreement = str(self.variant_agreement or "")
        if agreement not in {
            "same_leading_identity",
            "different_leading_identity",
            "neutral_unavailable",
        }:
            raise ValueError("unsupported primary/neutral agreement state")
        if self.neutral is None and agreement != "neutral_unavailable":
            raise ValueError(
                "neutral-unavailable observations require matching agreement"
            )
        if self.neutral is not None and agreement == "neutral_unavailable":
            raise ValueError(
                "available neutral support cannot be marked unavailable"
            )
        object.__setattr__(self, "variant_agreement", agreement)
        if self.style_evidence_binding is not None and not isinstance(
            self.style_evidence_binding,
            SourceStyleEvidenceBindingV1,
        ):
            raise TypeError(
                "style_evidence_binding must be a digest binding"
            )
        if self.target_font_affinity is not None:
            if not isinstance(
                self.target_font_affinity,
                TargetFontAffinityObservationV1,
            ):
                raise TypeError(
                    "target_font_affinity must be a typed affinity observation"
                )
            if (
                self.target_font_affinity.source_input_sha256
                != self.primary_input_sha256
            ):
                raise ValueError(
                    "target-font affinity must bind the primary source input"
                )
            if self.target_font_affinity.model_identity != self.model_identity:
                raise ValueError(
                    "target-font affinity model does not match observation"
                )
            if (
                self.target_font_affinity.label_catalog_version
                != self.label_catalog_version
            ):
                raise ValueError(
                    "target-font affinity label catalog does not match "
                    "observation"
                )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_identity_sha256": self.source_identity_sha256,
            "authorized_view_sha256": self.authorized_view_sha256,
            "model_identity": self.model_identity,
            "label_catalog_version": self.label_catalog_version,
            "support_policy_version": self.support_policy_version,
            "primary_input_sha256": self.primary_input_sha256,
            "neutral_input_sha256": self.neutral_input_sha256,
            "neutral_error": self.neutral_error,
            "variant_agreement": self.variant_agreement,
            "variant_overlap_bounds": (
                self.variant_overlap_bounds.to_audit_dict()
                if self.variant_overlap_bounds is not None
                else None
            ),
            "target_font_affinity": (
                self.target_font_affinity.to_audit_dict()
                if self.target_font_affinity is not None
                else None
            ),
            "style_evidence_binding": (
                self.style_evidence_binding.to_audit_dict()
                if self.style_evidence_binding is not None
                else None
            ),
            "primary": self.primary.to_audit_dict(),
            "neutral": (
                self.neutral.to_audit_dict()
                if self.neutral is not None
                else None
            ),
        }
