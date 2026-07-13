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
    PARENT_STYLE_ARBITRATOR_PROVIDER,
    PARENT_STYLE_ARBITRATOR_SOURCE,
)
from app.pipeline.parent_style_evidence import (
    AuthorizedSourceStyleView,
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
    source_stroke_width_px: float = 0.0
    source_ink_stroke_width_px: float = 0.0
    stroke_width_ratio: float = 0.0
    line_spacing_ratio: float = 0.0
    angle_degrees: float = 0.0
    axis_confidence: Mapping[str, float] = field(default_factory=dict)
    axis_provenance: Mapping[str, str] = field(default_factory=dict)
    observation_summary: Mapping[str, Any] = field(default_factory=dict)
    detector_variant_summary: Mapping[str, Any] = field(default_factory=dict)
    peer_group_id: str = ""
    peer_normalization_applied: bool = False
    peer_normalized_axes: tuple[str, ...] = ()
    peer_support_summary: Mapping[str, Any] = field(default_factory=dict)

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
            confidence=float(confidence),
            font_label=font_label,
            font_weight=_font_weight_from_label(font_label) or "",
            font_language="CJK",
            font_serif=bool(font_serif),
            direction="ttb",
            direction_confidence=float(confidence),
            text_color="#111111",
            stroke_color="#EEEEEE",
            text_size_ratio=float(source_size_px) / 36.0,
            source_size_px=float(source_size_px),
            source_stroke_width_px=max(0.0, float(source_size_px) * 0.02),
            source_ink_stroke_width_px=max(0.0, float(source_size_px) * 0.08),
            stroke_width_ratio=0.02,
            line_spacing_ratio=0.05,
            axis_confidence={
                "family": float(confidence),
                "weight": float(confidence),
                "scale": float(confidence),
                "paint": float(confidence),
                "orientation": float(confidence),
            },
            axis_provenance={
                "family": "test_authorized_evidence",
                "weight": "test_authorized_evidence",
                "scale": "test_authorized_evidence",
                "paint": "test_authorized_evidence",
                "orientation": "test_authorized_evidence",
            },
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
            "source_stroke_width_px": round(float(self.source_stroke_width_px), 8),
            "source_ink_stroke_width_px": round(
                float(self.source_ink_stroke_width_px), 8
            ),
            "stroke_width_ratio": round(float(self.stroke_width_ratio), 8),
            "line_spacing_ratio": round(float(self.line_spacing_ratio), 8),
            "angle_degrees": round(float(self.angle_degrees), 8),
            "axis_confidence": dict(self.axis_confidence),
            "axis_provenance": dict(self.axis_provenance),
            "observation_summary": dict(self.observation_summary),
            "detector_variant_summary": dict(self.detector_variant_summary),
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
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
            "peer_group_id": self.peer_group_id,
            "peer_normalization_applied": bool(self.peer_normalization_applied),
            "peer_normalized_axes": list(self.peer_normalized_axes),
            "peer_support_summary": dict(self.peer_support_summary),
        }


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
        detector_input = observation_inputs.primary_input
        if detector_input is None or not observation_inputs.available:
            result.evidence.append(
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=("authorized_detector_input_unavailable",),
                    view=view,
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
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=("yuzumarker_detector_unavailable",),
                    view=view,
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
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=(f"style_detector_failed:{type(exc).__name__}",),
                    view=view,
                )
            )
            continue
        if not isinstance(detection, Mapping):
            result.evidence.append(
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=("style_detector_output_contract_invalid",),
                    view=view,
                )
            )
            continue
        confidence = _unit_interval(detection.get("confidence"))
        if confidence is None:
            result.evidence.append(
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=("font_model_confidence_contract_invalid",),
                    view=view,
                )
            )
            continue
        if confidence < MIN_STYLE_EVIDENCE_CONFIDENCE:
            result.evidence.append(
                StyleEvidence.unavailable(
                    page_id=page_id,
                    bundle_id=bundle_id,
                    parent_id=parent_id,
                    root_id=root_id,
                    reason_codes=("font_model_confidence_below_observation_floor",),
                    view=view,
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
        direct_weight = str(observation_inputs.ink_weight_class or "").strip().lower()
        direct_weight_confidence = float(observation_inputs.ink_weight_confidence)
        if direct_weight in {"regular", "bold"} and direct_weight_confidence > 0.0:
            parsed_weight = direct_weight
            weight_confidence = direct_weight_confidence
            weight_reason = "weight_authorized_ink_geometry_measured"
        else:
            parsed_weight = model_weight
            weight_confidence = model_weight_confidence
            weight_reason = model_weight_reason
        font_serif, family_confidence, family_reason = _family_axis_from_variants(
            detection,
            neutral_detection,
        )
        direction, direction_confidence, direction_reason = _orientation_axis_from_variants(
            detection,
            neutral_detection,
        )
        source_size_px = observation_inputs.source_cell_size_for_direction(direction)
        text_size_ratio = (
            float(source_size_px) / float(max(1, analysis_width))
            if source_size_px > 0
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
                float(observation_inputs.scale_confidence)
                if source_size_px > 0
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
                "authorized_source_style_view:foreground_geometry_cell_measurement"
                if source_size_px > 0
                else "typesetting_default:source_scale_unavailable"
            ),
            "paint": (
                "authorized_source_style_view:fill_support_contrast_measurement"
                if paint_valid
                else "target_fallback:paint_axis_contract_invalid"
            ),
            "stroke": (
                "authorized_source_style_view:visible_support_transition_measurement"
                if source_stroke_width_px > 0
                else "authorized_source_style_view:uniform_support_no_visible_stroke"
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
        if source_size_px <= 0:
            evidence_reasons.append("source_scale_axis_unavailable")
        if not paint_valid:
            evidence_reasons.append("source_paint_axis_contract_invalid")
        if direction not in {"ltr", "ttb"}:
            evidence_reasons.append("source_orientation_axis_contract_invalid")
        detector_variant_summary = _detector_variant_summary(
            detection,
            neutral_detection,
            primary_sha256=_image_sha256(detector_input),
            neutral_sha256=_image_sha256(observation_inputs.neutral_input),
            neutral_error=neutral_error,
        )
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
                detector_input_sha256=_image_sha256(detector_input),
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
                source_stroke_width_px=source_stroke_width_px,
                source_ink_stroke_width_px=source_ink_stroke_width_px,
                stroke_width_ratio=stroke_width_ratio,
                line_spacing_ratio=line_spacing_ratio,
                angle_degrees=angle_degrees,
                axis_confidence=shared_axis_confidence,
                axis_provenance=axis_provenance,
                observation_summary=observation_inputs.to_audit_dict(),
                detector_variant_summary=detector_variant_summary,
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

    reconciled = _reconcile_root_local_peer_evidence(
        bundles=bundles,
        evidence_by_bundle=identity_bound,
    )
    resolved: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            continue
        item = reconciled.get(bundle_id) or identity_bound.get(bundle_id)
        if item is None:
            item = StyleEvidence.unavailable(
                page_id=str(getattr(bundle, "page_id", "") or ""),
                bundle_id=bundle_id,
                parent_id=str(getattr(bundle, "parent_id", "") or ""),
                root_id=str(getattr(bundle, "root_id", "") or ""),
                reason_codes=("style_evidence_missing",),
            )
        style = _resolved_style_for_bundle(
            bundle,
            item,
            default_font_name=default_font_name,
            models_dir=models_dir,
        )
        bundle.render_style = dict(style)
        if hasattr(bundle, "style_evidence_summary"):
            bundle.style_evidence_summary = item.to_audit_dict()
        else:
            setattr(bundle, "style_evidence_summary", item.to_audit_dict())
        try:
            bundle.execution_region = bundle.to_region_record()
        except Exception:
            pass
        resolved[bundle_id] = dict(style)
        record = item.to_audit_dict()
        record.update(
            {
                "style_evidence_status": item.status,
                "status": "applied" if item.vote_eligible else "fallback",
                "render_style_provider": style.get("render_style_provider"),
                "render_style_source": style.get("render_style_source"),
                "style_resolution_status": style.get("style_resolution_status"),
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
            }
        )
        records.append({key: value for key, value in record.items() if value not in (None, "", [])})
    return ParentStyleArbitrationResult(resolved_styles=resolved, records=tuple(records))


def _reconcile_root_local_peer_evidence(
    *,
    bundles: Sequence[Any],
    evidence_by_bundle: Mapping[str, StyleEvidence],
) -> dict[str, StyleEvidence]:
    """Reconcile noisy axes only inside one visually compatible root cohort."""

    output = dict(evidence_by_bundle)
    groups: dict[tuple[str, str], list[StyleEvidence]] = {}
    for bundle_id, item in evidence_by_bundle.items():
        if item.status != "observed" or not item.vote_eligible or not item.root_id:
            continue
        groups.setdefault((item.page_id, item.root_id), []).append(item)

    for (page_id, root_id), members in groups.items():
        if len(members) < 2 or not _peer_members_are_visually_compatible(members):
            continue
        group_id = f"root-peer:{page_id}:{root_id}"
        member_ids = sorted(item.bundle_id for item in members)
        family_value, family_confidence = _peer_categorical_consensus(
            members,
            axis="family",
            value_getter=lambda item: "serif" if item.font_serif else "sans",
        )
        weight_value, weight_confidence = _peer_categorical_consensus(
            members,
            axis="weight",
            value_getter=lambda item: str(item.font_weight or ""),
        )
        direction_value, direction_confidence = _peer_categorical_consensus(
            members,
            axis="orientation",
            value_getter=lambda item: str(item.direction or ""),
        )
        scale_value, scale_confidence = _peer_numeric_consensus(
            members,
            axis="scale",
            value_getter=lambda item: float(item.source_size_px),
            maximum_relative_spread=0.18,
        )
        paint_member = _peer_representative_member(members, axis="paint")
        stroke_member = _peer_representative_member(members, axis="stroke")
        summary = {
            "scope": "root_local_compatible",
            "member_bundle_ids": member_ids,
            "family_consensus": family_value or "",
            "weight_consensus": weight_value or "",
            "orientation_consensus": direction_value or "",
            "source_size_px_consensus": round(float(scale_value or 0.0), 6),
        }
        for item in members:
            changes: dict[str, Any] = {}
            normalized_axes: list[str] = []
            axis_confidence = dict(item.axis_confidence)
            axis_provenance = dict(item.axis_provenance)
            if family_value in {"serif", "sans"}:
                target_serif = family_value == "serif"
                if bool(item.font_serif) != target_serif:
                    changes["font_serif"] = target_serif
                    normalized_axes.append("family")
                axis_confidence["family"] = max(
                    float(axis_confidence.get("family") or 0.0), family_confidence
                )
                axis_provenance["family"] = "parent_style_arbitrator:root_local_peer_family"
            if weight_value in {"regular", "bold", "black"}:
                if str(item.font_weight or "") != weight_value:
                    changes["font_weight"] = weight_value
                    normalized_axes.append("weight")
                axis_confidence["weight"] = max(
                    float(axis_confidence.get("weight") or 0.0), weight_confidence
                )
                axis_provenance["weight"] = "parent_style_arbitrator:root_local_peer_weight"
            if direction_value in {"ltr", "ttb"}:
                if str(item.direction or "") != direction_value:
                    changes["direction"] = direction_value
                    changes["direction_confidence"] = direction_confidence
                    normalized_axes.append("orientation")
                axis_confidence["orientation"] = max(
                    float(axis_confidence.get("orientation") or 0.0),
                    direction_confidence,
                )
                axis_provenance["orientation"] = (
                    "parent_style_arbitrator:root_local_peer_orientation"
                )
            if scale_value and scale_value > 0.0:
                if abs(float(item.source_size_px) - scale_value) > 0.25:
                    changes["source_size_px"] = scale_value
                    analysis_width = int(item.analysis_bbox[2]) if len(item.analysis_bbox) == 4 else 0
                    changes["text_size_ratio"] = (
                        scale_value / float(max(1, analysis_width))
                    )
                    normalized_axes.append("scale")
                axis_confidence["scale"] = max(
                    float(axis_confidence.get("scale") or 0.0), scale_confidence
                )
                axis_provenance["scale"] = "parent_style_arbitrator:root_local_peer_scale"
            if paint_member is not None:
                if item.text_color != paint_member.text_color:
                    changes["text_color"] = paint_member.text_color
                    normalized_axes.append("paint")
                axis_confidence["paint"] = max(
                    float(axis_confidence.get("paint") or 0.0),
                    float(paint_member.axis_confidence.get("paint") or 0.0),
                )
                axis_provenance["paint"] = "parent_style_arbitrator:root_local_peer_paint"
            if stroke_member is not None:
                if (
                    item.source_stroke_width_px != stroke_member.source_stroke_width_px
                    or item.stroke_color != stroke_member.stroke_color
                ):
                    changes["source_stroke_width_px"] = stroke_member.source_stroke_width_px
                    changes["stroke_width_ratio"] = stroke_member.stroke_width_ratio
                    changes["stroke_color"] = stroke_member.stroke_color
                    normalized_axes.append("stroke")
                axis_confidence["stroke"] = max(
                    float(axis_confidence.get("stroke") or 0.0),
                    float(stroke_member.axis_confidence.get("stroke") or 0.0),
                )
                axis_provenance["stroke"] = "parent_style_arbitrator:root_local_peer_stroke"
            changes.update(
                {
                    "axis_confidence": axis_confidence,
                    "axis_provenance": axis_provenance,
                    "peer_group_id": group_id,
                    "peer_normalization_applied": bool(normalized_axes),
                    "peer_normalized_axes": tuple(_unique_strings(normalized_axes)),
                    "peer_support_summary": summary,
                    "reason_codes": tuple(
                        _unique_strings(
                            [
                                *item.reason_codes,
                                "root_local_compatible_peer_reconciliation",
                                *(
                                    ["root_local_peer_axes_normalized"]
                                    if normalized_axes
                                    else []
                                ),
                            ]
                        )
                    ),
                }
            )
            output[item.bundle_id] = replace(item, **changes)
    return output


def _peer_members_are_visually_compatible(members: Sequence[StyleEvidence]) -> bool:
    sizes = [
        float(item.source_size_px)
        for item in members
        if float(item.axis_confidence.get("scale") or 0.0) > 0.0
        and float(item.source_size_px) > 0.0
    ]
    if len(sizes) >= 2 and max(sizes) / max(1.0, min(sizes)) > 1.25:
        return False
    strong_weights = {
        str(item.font_weight or "")
        for item in members
        if str(item.font_weight or "")
        and float(item.axis_confidence.get("weight") or 0.0) >= 0.65
    }
    if len(strong_weights) > 1:
        return False
    strong_directions = {
        str(item.direction or "")
        for item in members
        if str(item.direction or "") in {"ltr", "ttb"}
        and float(item.axis_confidence.get("orientation") or 0.0) >= 0.65
    }
    if len(strong_directions) > 1:
        return False
    colors = [
        item.text_color
        for item in members
        if _hex_color(item.text_color)
        and float(item.axis_confidence.get("paint") or 0.0) > 0.0
    ]
    if colors and any(
        _color_distance(first, second) > 48.0
        for index, first in enumerate(colors)
        for second in colors[index + 1 :]
    ):
        return False
    strokes = [
        float(item.source_stroke_width_px)
        for item in members
        if float(item.axis_confidence.get("stroke") or 0.0) > 0.0
    ]
    if len(strokes) >= 2 and max(strokes) - min(strokes) > 1.0:
        return False
    return True


def _peer_categorical_consensus(
    members: Sequence[StyleEvidence],
    *,
    axis: str,
    value_getter: Any,
) -> tuple[str | None, float]:
    scores: dict[str, float] = {}
    confidences: dict[str, list[float]] = {}
    for item in members:
        value = str(value_getter(item) or "")
        confidence = float(item.axis_confidence.get(axis) or 0.0)
        if not value or confidence <= 0.0:
            continue
        score = confidence * _peer_observation_reliability(item)
        scores[value] = scores.get(value, 0.0) + score
        confidences.setdefault(value, []).append(confidence)
    if not scores:
        return None, 0.0
    ordered = sorted(scores, key=lambda value: (scores[value], value), reverse=True)
    winner = ordered[0]
    total = sum(scores.values())
    dominance = scores[winner] / max(total, 1e-6)
    if len(ordered) > 1 and dominance < 0.62:
        return None, 0.0
    winner_confidence = float(np.mean(confidences[winner])) if confidences[winner] else 0.0
    return winner, min(1.0, winner_confidence * max(0.72, dominance))


def _peer_numeric_consensus(
    members: Sequence[StyleEvidence],
    *,
    axis: str,
    value_getter: Any,
    maximum_relative_spread: float,
) -> tuple[float | None, float]:
    values: list[tuple[float, float, float]] = []
    for item in members:
        value = float(value_getter(item) or 0.0)
        confidence = float(item.axis_confidence.get(axis) or 0.0)
        if value <= 0.0 or confidence <= 0.0:
            continue
        values.append((value, confidence, _peer_observation_reliability(item)))
    if not values:
        return None, 0.0
    raw = [value for value, _, _ in values]
    if len(raw) >= 2 and (max(raw) - min(raw)) / max(1.0, float(np.median(raw))) > maximum_relative_spread:
        return None, 0.0
    weights = [confidence * reliability for _, confidence, reliability in values]
    consensus = float(np.average(raw, weights=weights))
    confidence = float(np.average([item[1] for item in values], weights=weights))
    return consensus, min(1.0, confidence)


def _peer_representative_member(
    members: Sequence[StyleEvidence],
    *,
    axis: str,
) -> StyleEvidence | None:
    valid = [
        item
        for item in members
        if float(item.axis_confidence.get(axis) or 0.0) > 0.0
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (
            float(item.axis_confidence.get(axis) or 0.0)
            * _peer_observation_reliability(item),
            item.bundle_id,
        ),
    )


def _peer_observation_reliability(item: StyleEvidence) -> float:
    component_count = len(tuple(item.owned_component_ids or ()))
    return min(4.0, max(1.0, math.sqrt(float(max(1, component_count)))))


def _color_distance(first: str, second: str) -> float:
    left = _hex_color(first)
    right = _hex_color(second)
    if not left or not right:
        return float("inf")
    left_rgb = np.asarray(
        [int(left[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.float32,
    )
    right_rgb = np.asarray(
        [int(right[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.float32,
    )
    return float(np.linalg.norm(left_rgb - right_rgb))


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


def _resolved_style_for_bundle(
    bundle: Any,
    evidence: StyleEvidence,
    *,
    default_font_name: str,
    models_dir: str | None,
) -> dict[str, Any]:
    base = _style_contract_base(bundle)
    role = str(getattr(bundle, "role", "") or "")
    semantic_style_class = _semantic_style_class(role)
    if evidence.status != "observed" or not evidence.vote_eligible:
        fallback_font = default_font_name or resolve_noto_cjk_sc_font_file(
            base_dir=models_dir,
            serif=False,
            weight="regular",
        ) or "Noto Sans CJK SC"
        return {
            **base,
            "render_style_version": PARENT_RENDER_STYLE_VERSION,
            "render_style_owner": "parent_execution_bundle",
            "render_style_source": STYLE_ARBITRATOR_SOURCE,
            "render_style_provider": STYLE_ARBITRATOR_PROVIDER,
            "render_style_confidence": 0.0,
            "style_resolution_status": "unresolved",
            "style_resolution_reason_codes": list(evidence.reason_codes),
            "style_arbitration_decision": "authorized_evidence_unavailable",
            "style_arbitration_peer_scope": "none",
            "style_class": semantic_style_class,
            "typographic_style_class": "unresolved",
            "base_style_id": "unresolved",
            "font_family": fallback_font,
            "font_family_role": "fallback_sans",
            "font_weight": "regular",
            "fill_color": "#000000",
            "stroke_color": "#FFFFFF",
            "stroke_width": 0,
            "font_size_authority": "typesetting_default",
            "font_size_source": "typesetting_default_unresolved",
            "font_size_locked": False,
            "font_size_policy": "unresolved_evidence_default",
            "source_typography_observed": False,
            "source_typography_matched": False,
            "style_evidence_status": "unavailable",
            "style_evidence_view_id": evidence.view_id,
            "style_evidence_cleanup_mask_ids": list(evidence.cleanup_mask_ids),
            "source_visual_column_count": 0,
            "source_visual_column_reliable": False,
            "source_visual_column_authority": "none",
        }

    axis_confidence = {
        key: float(value or 0.0) for key, value in dict(evidence.axis_confidence).items()
    }
    family_supported = axis_confidence.get("family", 0.0) >= 0.20
    weight_supported = (
        str(evidence.font_weight or "") in {"regular", "bold", "black"}
        and axis_confidence.get("weight", 0.0) >= 0.20
    )
    scale_supported = (
        float(evidence.source_size_px) > 0.0
        and axis_confidence.get("scale", 0.0) > 0.0
    )
    paint_supported = bool(
        _hex_color(evidence.text_color)
        and axis_confidence.get("paint", 0.0) > 0.0
    )
    stroke_supported = axis_confidence.get("stroke", 0.0) > 0.0
    orientation_supported = (
        str(evidence.direction or "").lower() in {"ltr", "ttb"}
        and axis_confidence.get("orientation", 0.0) > 0.0
    )
    target_serif = bool(evidence.font_serif) if family_supported else False
    family_role = "serif" if target_serif else "sans"
    observed_weight = str(evidence.font_weight or "") if weight_supported else ""
    target_weight = observed_weight or "regular"
    resolved_font = resolve_noto_cjk_sc_font_file(
        base_dir=models_dir,
        serif=target_serif,
        weight=target_weight,
    ) or default_font_name or ("Noto Serif CJK SC" if target_serif else "Noto Sans CJK SC")
    direction = str(evidence.direction or "").lower() if orientation_supported else ""
    orientation = "horizontal" if direction == "ltr" else "vertical"
    preferred_size = (
        max(1, int(round(float(evidence.source_size_px))))
        if scale_supported
        else 0
    )
    stroke_width = (
        max(0, int(round(float(evidence.source_stroke_width_px))))
        if stroke_supported
        else 0
    )
    if preferred_size > 0:
        stroke_width = min(stroke_width, max(0, int(round(preferred_size * 0.25))))
    else:
        stroke_width = 0
    resolution_reasons = ["per_parent_authorized_evidence"]
    if evidence.peer_normalization_applied:
        resolution_reasons.append("root_local_compatible_peer_reconciliation")
    if not family_supported:
        resolution_reasons.append("source_family_axis_unresolved_target_sans_fallback")
    if not weight_supported:
        resolution_reasons.append("source_weight_axis_unresolved_target_regular_fallback")
    if not scale_supported:
        resolution_reasons.append("source_scale_axis_unresolved_typesetting_default")
    if not paint_supported:
        resolution_reasons.append("source_paint_axis_unresolved_target_black_fallback")
    if not stroke_supported:
        resolution_reasons.append("source_stroke_axis_unresolved_zero_stroke_fallback")
    if not orientation_supported:
        resolution_reasons.append("source_orientation_axis_unresolved_target_vertical_fallback")
    resolved_confidence = float(
        np.mean(
            [
                axis_confidence.get("family", 0.0) if family_supported else 0.0,
                axis_confidence.get("weight", 0.0) if weight_supported else 0.0,
                axis_confidence.get("scale", 0.0) if scale_supported else 0.0,
                axis_confidence.get("paint", 0.0) if paint_supported else 0.0,
                axis_confidence.get("stroke", 0.0) if stroke_supported else 0.0,
                axis_confidence.get("orientation", 0.0)
                if orientation_supported
                else 0.0,
            ]
        )
    )
    style: dict[str, Any] = {
        **base,
        "render_style_version": PARENT_RENDER_STYLE_VERSION,
        "render_style_owner": "parent_execution_bundle",
        "render_style_source": STYLE_ARBITRATOR_SOURCE,
        "render_style_provider": STYLE_ARBITRATOR_PROVIDER,
        "render_style_provider_model": evidence.evidence_model,
        "render_style_confidence": resolved_confidence,
        "style_resolution_status": "authorized_evidence_resolved",
        "style_resolution_reason_codes": resolution_reasons,
        "style_arbitration_decision": (
            "per_parent_authorized_evidence_with_root_peer_reconciliation"
            if evidence.peer_normalization_applied
            else "per_parent_authorized_evidence"
        ),
        "style_arbitration_peer_scope": (
            "root_local_compatible" if evidence.peer_group_id else "none"
        ),
        "style_arbitration_peer_group_id": evidence.peer_group_id,
        "style_arbitration_peer_normalized_axes": list(evidence.peer_normalized_axes),
        "style_class": semantic_style_class,
        "typographic_style_class": (
            f"{family_role}_{observed_weight}"
            if observed_weight
            else f"{family_role}_fallback_regular"
        ),
        "base_style_id": f"base_{family_role}_{target_weight}_{orientation}",
        "font_family": resolved_font,
        "font_family_role": family_role,
        "font_family_authority": (
            "authorized_source_style_view"
            if family_supported
            else "target_fallback_unresolved_source_family"
        ),
        "font_weight": target_weight,
        "font_weight_authority": (
            "authorized_source_style_view"
            if observed_weight
            else "target_fallback_unresolved_source_weight"
        ),
        "target_font_mapping_source": "noto_cjk_sc_role_weight_glyph_coverage_pack",
        "target_font_mapping_family_role": family_role,
        "target_font_mapping_weight": target_weight,
        "fill_color": evidence.text_color if paint_supported else "#000000",
        "fill_color_authority": (
            "authorized_source_style_view"
            if paint_supported
            else "target_fallback_unresolved_source_paint"
        ),
        "stroke_color": (
            evidence.stroke_color
            if stroke_supported and _hex_color(evidence.stroke_color)
            else "#FFFFFF"
        ),
        "stroke_width": stroke_width,
        "stroke_authority": (
            "authorized_source_style_view"
            if stroke_supported
            else "target_fallback_unresolved_source_stroke_zero"
        ),
        "source_orientation": orientation,
        "wrap_mode": orientation,
        "source_orientation_authority": (
            "authorized_source_style_view"
            if orientation_supported
            else "target_fallback_unresolved_source_orientation"
        ),
        "font_size_authority": "automated_style_arbitrator",
        "font_size_locked": False,
        "font_size_policy": "authorized_source_preferred",
        "font_size_fallback_policy": "typesetting_bounded_fit",
        "source_typography_observed": True,
        "source_typography_matched": False,
        "source_typography_match_status": "mapped_to_supported_target_role",
        "style_evidence_status": "observed",
        "style_evidence_view_id": evidence.view_id,
        "style_evidence_cleanup_mask_ids": list(evidence.cleanup_mask_ids),
        "style_evidence_owned_component_ids": list(evidence.owned_component_ids),
        "style_evidence_provider": evidence.evidence_provider,
        "style_evidence_source": evidence.evidence_source,
        "style_evidence_model": evidence.evidence_model,
        "style_axis_confidence": dict(evidence.axis_confidence),
        "style_axis_provenance": dict(evidence.axis_provenance),
        "detector_input_sha256": evidence.detector_input_sha256,
        "source_scale_px": round(float(evidence.source_size_px), 6),
        "source_scale_conversion_count": 1 if preferred_size > 0 else 0,
        "source_scale_source": "authorized_foreground_geometry_cell_measurement",
        "source_ink_stroke_width_px": round(
            float(evidence.source_ink_stroke_width_px), 6
        ),
        "source_visual_column_count": 0,
        "source_visual_column_reliable": False,
        "source_visual_column_authority": "none",
    }
    if scale_supported and preferred_size > 0:
        style.update(
            {
                "font_size": preferred_size,
                "font_size_hint": preferred_size,
                "font_size_min": max(1, int(round(preferred_size * 0.72))),
                "font_size_max": preferred_size,
                "font_size_source": "authorized_source_style_view",
            }
        )
    else:
        style.update(
            {
                "font_size_authority": "typesetting_default",
                "font_size_source": "authorized_scale_unavailable",
                "font_size_policy": "unresolved_scale_default",
            }
        )
    return style


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
    if primary_direction not in {"ltr", "ttb"}:
        return "", 0.0, "orientation_primary_vote_invalid"
    if not isinstance(neutral, Mapping):
        return (
            primary_direction,
            primary_confidence * 0.75,
            "orientation_neutral_vote_unavailable",
        )
    neutral_direction = str(neutral.get("direction") or "").strip().lower()
    neutral_confidence = _unit_interval(neutral.get("direction_confidence")) or 0.0
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
