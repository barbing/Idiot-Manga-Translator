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
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    SourceTextFootprint,
    SourceStyleAxisEvidence,
    build_authorized_style_observation_inputs,
)


FONT_COUNT = 6150
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
    font_serif: bool = False
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
            "font_serif": bool(self.font_serif),
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

    def detect(self, image: Any) -> dict[str, Any]:
        from PIL import ImageOps

        prepared = ImageOps.exif_transpose(image).convert("RGB").resize((512, 512))
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        array = array.transpose(2, 0, 1)[None, ...]
        output = self._session.run(None, {self._input_name: array})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape[0] < FONT_COUNT + 12:
            raise RuntimeError(f"Unexpected YuzuMarker output length: {vector.shape[0]}")

        font_prob = _softmax(vector[:FONT_COUNT])
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
                    "serif": bool(label.get("serif")),
                }
            )
        direction_prob = _softmax(vector[FONT_COUNT : FONT_COUNT + 2])
        direction_index = int(direction_prob.argmax())
        direction_confidence = float(direction_prob[direction_index])
        regression = vector[FONT_COUNT + 2 : FONT_COUNT + 12]
        angle_ratio = _unit_interval(regression[9])
        top = top_candidates[0] if top_candidates else {}
        return {
            "font_index": int(top_indices[0]) if len(top_indices) else -1,
            "confidence": float(top.get("confidence") or 0.0),
            "font_path": str(top.get("path") or ""),
            "font_language": str(top.get("language") or ""),
            "font_serif": bool(top.get("serif")),
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
        confidence = _unit_interval(detection.get("confidence"))
        if confidence is None:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="font_model_confidence_contract_invalid",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        if confidence < MIN_STYLE_EVIDENCE_CONFIDENCE:
            result.evidence.append(
                _direct_or_unavailable_style_evidence(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    view=view,
                    observation=observation_inputs,
                    detector_reason="font_model_confidence_below_observation_floor",
                    detector_input_sha256=actual_detector_input_sha256,
                )
            )
            continue
        provider = YUZUMARKER_PROVIDER if normalized_mode == "yuzumarker" else HEURISTIC_PROVIDER
        source = YUZUMARKER_STYLE_SOURCE if normalized_mode == "yuzumarker" else HEURISTIC_STYLE_SOURCE
        model = YUZUMARKER_PROVIDER_MODEL if normalized_mode == "yuzumarker" else ""
        analysis_width = int(view.analysis_bbox[2]) if len(view.analysis_bbox) == 4 else int(detector_input.width)
        font_label = str(detection.get("font_path") or "")
        model_weight, model_weight_confidence, model_weight_reason = _weight_axis_from_variants(
            detection,
            neutral_detection,
        )
        font_serif, family_confidence, family_reason = _family_axis_from_variants(
            detection,
            neutral_detection,
        )
        direction, direction_confidence, direction_reason = _orientation_axis_from_variants(
            detection,
            neutral_detection,
        )
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
            "family": f"{provider}:fill_contrast_and_neutral_coarse_family_vote",
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
        if family_reason:
            evidence_reasons.append(family_reason)
        if weight_reason:
            evidence_reasons.append(weight_reason)
        if direction_reason:
            evidence_reasons.append(direction_reason)
        if neutral_error:
            evidence_reasons.append(neutral_error)
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
        detector_variant_summary = _detector_variant_summary(
            detection,
            neutral_detection,
            primary_sha256=actual_detector_input_sha256,
            neutral_sha256=_image_sha256(observation_inputs.neutral_input),
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
        family_axis = (
            SourceStyleAxisEvidence(
                axis="family",
                status="supported",
                value={
                    "font_label": font_label,
                    "font_serif": bool(font_serif),
                    "font_language": str(detection.get("font_language") or ""),
                    "top_candidates": _compact_candidates(
                        detection.get("top_candidates")
                    ),
                },
                confidence=family_confidence,
                provenance=(
                    f"{provider}:independent_primary_neutral_family_observation"
                ),
                support_identity=support_identity,
                reason_codes=(family_reason,) if family_reason else (),
                support={"detector_variant_summary": detector_variant_summary},
            )
            if family_confidence > 0.0
            else SourceStyleAxisEvidence.unavailable(
                "family",
                provenance=(
                    f"{provider}:independent_primary_neutral_family_observation"
                ),
                support_identity=support_identity,
                reason_codes=(
                    family_reason or "source_family_axis_unavailable",
                ),
                support={"detector_variant_summary": detector_variant_summary},
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
        if (
            direct_weight_axis is not None
            and direct_weight_axis.supported
            and direct_weight in {"regular", "bold"}
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
                authorized_perceptual_source_identity={},
                evidence_provider=provider,
                evidence_source=source,
                evidence_model=model,
                confidence=confidence,
                font_label=font_label,
                font_weight=parsed_weight or "",
                font_language=str(detection.get("font_language") or ""),
                font_serif=font_serif,
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
    """Resolve one complete style per parent with compatible root-local peers."""

    evidence_by_bundle: dict[str, list[StyleEvidence]] = {}
    for item in evidence or []:
        if isinstance(item, StyleEvidence) and item.bundle_id:
            evidence_by_bundle.setdefault(item.bundle_id, []).append(item)
    bundle_ids = [
        str(getattr(bundle, "bundle_id", "") or "")
        for bundle in list(parent_execution_bundles or [])
        if bool(getattr(bundle, "render_required", False))
    ]
    duplicate_bundle_ids = {
        bundle_id
        for bundle_id in bundle_ids
        if bundle_id and bundle_ids.count(bundle_id) > 1
    }
    bundles = [
        bundle
        for bundle in list(parent_execution_bundles or [])
        if bool(getattr(bundle, "render_required", False))
    ]
    identity_bound: dict[str, StyleEvidence] = {}
    for bundle in bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            continue
        bundle_page_id = str(getattr(bundle, "page_id", "") or "")
        bundle_parent_id = str(getattr(bundle, "parent_id", "") or "")
        bundle_root_id = str(getattr(bundle, "root_id", "") or "")
        candidates = evidence_by_bundle.get(bundle_id, [])
        identity_reasons: list[str] = []
        if bundle_id in duplicate_bundle_ids:
            identity_reasons.append("duplicate_parent_execution_bundle_identity")
        if len(candidates) > 1:
            identity_reasons.append("duplicate_style_evidence_for_bundle")
        if not candidates:
            identity_reasons.append("style_evidence_missing")
        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate is not None:
            if candidate.page_id != bundle_page_id:
                identity_reasons.append("style_evidence_page_identity_mismatch")
            if candidate.parent_id != bundle_parent_id:
                identity_reasons.append("style_evidence_parent_identity_mismatch")
            if candidate.root_id != bundle_root_id:
                identity_reasons.append("style_evidence_root_identity_mismatch")
        item = (
            StyleEvidence.unavailable(
                page_id=bundle_page_id,
                bundle_id=bundle_id,
                parent_id=bundle_parent_id,
                root_id=bundle_root_id,
                reason_codes=tuple(identity_reasons),
            )
            if identity_reasons
            else candidate
        )
        if item is None:
            item = StyleEvidence.unavailable(
                page_id=bundle_page_id,
                bundle_id=bundle_id,
                parent_id=bundle_parent_id,
                root_id=bundle_root_id,
                reason_codes=("style_evidence_missing",),
            )
        identity_bound[bundle_id] = item

    candidates_by_bundle = {
        str(getattr(bundle, "bundle_id", "") or ""): _collect_parent_axis_candidates(
            bundle,
            identity_bound[str(getattr(bundle, "bundle_id", "") or "")],
        )
        for bundle in bundles
        if str(getattr(bundle, "bundle_id", "") or "") in identity_bound
    }
    local_decisions_by_bundle = {
        bundle_id: _resolve_parent_local_axis_decisions(candidates)
        for bundle_id, candidates in candidates_by_bundle.items()
    }
    reconciled_decisions_by_bundle = _reconcile_parent_axis_decisions(
        bundles=bundles,
        evidence_by_bundle=identity_bound,
        candidates_by_bundle=candidates_by_bundle,
        local_decisions_by_bundle=local_decisions_by_bundle,
    )
    resolved: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            continue
        item = identity_bound.get(bundle_id)
        if item is None:
            item = StyleEvidence.unavailable(
                page_id=str(getattr(bundle, "page_id", "") or ""),
                bundle_id=bundle_id,
                parent_id=str(getattr(bundle, "parent_id", "") or ""),
                root_id=str(getattr(bundle, "root_id", "") or ""),
                reason_codes=("style_evidence_missing",),
            )
        candidates = candidates_by_bundle.get(bundle_id) or _collect_parent_axis_candidates(
            bundle,
            item,
        )
        decision_set = reconciled_decisions_by_bundle.get(
            bundle_id,
            local_decisions_by_bundle.get(bundle_id)
            or _resolve_parent_local_axis_decisions(candidates),
        )
        style = _build_resolved_style_from_decisions(
            bundle,
            item,
            decision_set,
            default_font_name=default_font_name,
            models_dir=models_dir,
        )
        raw_evidence_audit = item.to_audit_dict()
        bundle_evidence_snapshot = _plain_json_mapping_snapshot(
            raw_evidence_audit
        )
        bundle.render_style = dict(style)
        if hasattr(bundle, "style_evidence_summary"):
            bundle.style_evidence_summary = bundle_evidence_snapshot
        else:
            setattr(bundle, "style_evidence_summary", bundle_evidence_snapshot)
        try:
            bundle.execution_region = bundle.to_region_record()
        except Exception:
            pass
        resolved[bundle_id] = dict(style)
        record = _plain_json_mapping_snapshot(raw_evidence_audit)
        record.update(
            {
                "style_evidence_status": item.status,
                "status": (
                    "applied"
                    if item.vote_eligible
                    or bool(style.get("style_arbitration_peer_assisted_axes"))
                    else "fallback"
                ),
                "render_style_provider": style.get("render_style_provider"),
                "render_style_source": style.get("render_style_source"),
                "style_resolution_status": style.get("style_resolution_status"),
                "style_perceptual_axis_resolution": style.get(
                    "style_perceptual_axis_resolution"
                ),
                "style_axis_decisions": style.get("style_axis_decisions"),
                "style_arbitration_peer_assisted_axes": style.get(
                    "style_arbitration_peer_assisted_axes"
                ),
                "style_arbitration_peer_support": style.get(
                    "style_arbitration_peer_support"
                ),
                "base_style_id": style.get("base_style_id"),
                "font_family": style.get("font_family"),
                "font_family_role": style.get("font_family_role"),
                "font_family_authority": style.get("font_family_authority"),
                "font_weight": style.get("font_weight"),
                "font_weight_authority": style.get("font_weight_authority"),
                "font_size_hint": style.get("font_size_hint"),
                "fill_color": style.get("fill_color"),
                "stroke_color": style.get("stroke_color"),
                "stroke_width": style.get("stroke_width"),
                "target_font_request": style.get("target_font_request"),
            }
        )
        compact_record = {
            key: value
            for key, value in record.items()
            if value not in (None, "", [])
        }
        records.append(_plain_json_mapping_snapshot(compact_record))
    return ParentStyleArbitrationResult(resolved_styles=resolved, records=tuple(records))


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

    target_font_request = _optional_target_font_request(evidence)
    if target_font_request:
        style["target_font_request"] = target_font_request

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
) -> tuple[bool, float, str]:
    primary_family = bool(primary.get("font_serif"))
    primary_confidence = _unit_interval(primary.get("confidence")) or 0.0
    if not isinstance(neutral, Mapping):
        return primary_family, primary_confidence * 0.75, "family_neutral_vote_unavailable"
    neutral_confidence = _unit_interval(neutral.get("confidence")) or 0.0
    neutral_family = bool(neutral.get("font_serif"))
    if neutral_family == primary_family:
        return (
            primary_family,
            min(1.0, (primary_confidence + neutral_confidence) / 2.0),
            "family_variant_consensus",
        )
    return (
        primary_family,
        primary_confidence * 0.45,
        "family_variant_disagreement_primary_fill_contrast_retained",
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
        return {
            "font_path": str(value.get("font_path") or ""),
            "font_serif": bool(value.get("font_serif")),
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


def _optional_target_font_request(evidence: StyleEvidence) -> dict[str, str]:
    """Return one exact, independently agreed optional target-face request."""

    summary = evidence.detector_variant_summary
    if not isinstance(summary, Mapping):
        return {}
    if (
        str(summary.get("variant_contract") or "")
        != "fill_contrast_primary_plus_neutral_disagreement_probe"
    ):
        return {}
    primary = summary.get("primary")
    neutral = summary.get("neutral")
    if not isinstance(primary, Mapping) or not isinstance(neutral, Mapping):
        return {}
    source_label = str(evidence.font_label or "")
    primary_label = str(primary.get("font_path") or "")
    neutral_label = str(neutral.get("font_path") or "")
    if not source_label or not (
        source_label == primary_label == neutral_label
    ):
        return {}
    taxonomy = OPTIONAL_TARGET_FONT_LABEL_TAXONOMY.get(source_label)
    if not isinstance(taxonomy, Mapping):
        return {}
    weight = str(evidence.font_weight or "").strip().lower()
    primary_weight = str(primary.get("font_weight") or "").strip().lower()
    neutral_weight = str(neutral.get("font_weight") or "").strip().lower()
    if not weight or not (weight == primary_weight == neutral_weight):
        return {}
    if weight != str(taxonomy.get("weight") or ""):
        return {}
    axis_confidence = dict(evidence.axis_confidence or {})
    if float(axis_confidence.get("family") or 0.0) < 0.8:
        return {}
    if float(axis_confidence.get("weight") or 0.0) < 0.8:
        return {}
    if float(primary.get("confidence") or 0.0) < 0.8:
        return {}
    if float(neutral.get("confidence") or 0.0) < 0.8:
        return {}
    return {
        "contract_version": TARGET_FONT_REQUEST_VERSION,
        "catalog_face_id": str(taxonomy.get("catalog_face_id") or ""),
        "style_class": str(taxonomy.get("style_class") or ""),
        "weight": weight,
        "source_label": source_label,
        "provenance": TARGET_FONT_REQUEST_PROVENANCE,
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
                "serif": bool(item.get("serif")),
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
