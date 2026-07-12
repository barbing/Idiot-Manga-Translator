# -*- coding: utf-8 -*-
"""Parent-owned font and style detection for execution bundles."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.models.resolution import (
    resolve_noto_cjk_sc_font_file,
    resolve_yuzumarker_font_labels_file,
    resolve_yuzumarker_font_onnx_file,
)
from app.pipeline.parent_execution_bundle import PARENT_RENDER_STYLE_VERSION


FONT_COUNT = 6150
YUZUMARKER_PROVIDER = "YuzuMarker.FontDetection"
YUZUMARKER_PROVIDER_MODEL = "ogkalu/yuzumarker-font-detection-onnx:font-detector.onnx"
YUZUMARKER_STYLE_SOURCE = "parent_execution_bundle_yuzumarker_font_detection"
HEURISTIC_PROVIDER = "ParentFontHeuristic"
HEURISTIC_STYLE_SOURCE = "parent_execution_bundle_font_heuristic"
STYLE_ARBITRATOR_PROVIDER = "ParentStyleArbitrator"
STYLE_ARBITRATOR_SOURCE = "parent_execution_bundle_style_arbitrator"
MIN_STYLE_EVIDENCE_CONFIDENCE = 0.05
STYLE_EXCEPTION_CONFIDENCE = 0.55
STRONG_STYLE_EXCEPTION_CONFIDENCE = 0.85
MIN_COHORT_SIZE_NORMALIZATION_MEMBERS = 3
MIN_CAPTION_COHORT_SIZE_NORMALIZATION_MEMBERS = 2
MAX_TWO_MEMBER_COHORT_SIZE_SPREAD_RATIO = 0.12
MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER = 6
MIN_DOMINANT_COMPONENTS_FOR_VISUAL_COLUMNS = 6
MIN_COMPONENTS_PER_VISUAL_COLUMN = 2
VISUAL_COLUMN_CLUSTER_GAP_RATIO = 0.72
MIN_INDIVIDUAL_WEIGHT_CANDIDATES = 3
COMPLETE_INDIVIDUAL_WEIGHT_DOMINANCE = 0.999
MIN_INDIVIDUAL_WEIGHT_STROKE_RATIO = 1.20
CROP_LIGHT_SURFACE_MEDIAN_MIN = 190.0
CROP_LIGHT_PIXEL_RATIO_MIN = 0.55
CROP_DARK_SURFACE_MEDIAN_MAX = 175.0
CROP_DARK_PIXEL_RATIO_MIN = 0.30
MIN_OUTLINE_TO_TEXT_RATIO_FOR_BOLD = 0.05


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
            "parent_font_detection_version": "parent_font_detection_v1",
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
    """Small ONNX adapter for YuzuMarker.FontDetection."""

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
        provider_metadata = _onnx_session_provider_metadata(
            self.model_path,
            use_gpu=use_gpu,
            session=self._session,
        )
        self.gpu_requested = bool(provider_metadata.get("gpu_requested"))
        self.requested_execution_provider = str(
            provider_metadata.get("requested_execution_provider") or ""
        )
        self.available_execution_providers = list(
            provider_metadata.get("available_execution_providers") or []
        )
        self.active_execution_providers = list(
            provider_metadata.get("active_execution_providers") or []
        )
        self.primary_execution_provider = str(
            provider_metadata.get("primary_execution_provider") or ""
        )
        self.provider_fallback_reason = str(
            provider_metadata.get("provider_fallback_reason") or ""
        )
        self.provider_preload_error = str(
            provider_metadata.get("provider_preload_error") or ""
        )
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("YuzuMarker ONNX model has no inputs")
        self._input_name = inputs[0].name

    def detect(self, image: Any) -> dict[str, Any]:
        import numpy as np
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image).convert("RGB").resize((512, 512))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        output = self._session.run(None, {self._input_name: arr})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape[0] < FONT_COUNT + 12:
            raise RuntimeError(f"Unexpected YuzuMarker output length: {vector.shape[0]}")

        font_logits = vector[:FONT_COUNT]
        font_prob = _softmax(font_logits)
        top_indices = np.argsort(-font_prob)[:5]
        top_candidates = []
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

        direction_logits = vector[FONT_COUNT : FONT_COUNT + 2]
        direction_prob = _softmax(direction_logits)
        direction_index = int(direction_prob.argmax())
        regression = vector[FONT_COUNT + 2 : FONT_COUNT + 12]
        top_label = top_candidates[0] if top_candidates else {}
        return {
            "font_index": int(top_indices[0]) if len(top_indices) else -1,
            "confidence": float(top_candidates[0]["confidence"]) if top_candidates else 0.0,
            "font_path": str(top_label.get("path") or ""),
            "font_language": str(top_label.get("language") or ""),
            "font_serif": bool(top_label.get("serif")),
            "top_candidates": top_candidates,
            "direction": "ltr" if direction_index == 0 else "ttb",
            "direction_confidence": float(direction_prob[direction_index]),
            "text_color": _rgb_from_unit_values(regression[0:3]),
            "text_size_ratio": _float(regression[3]),
            "stroke_width_ratio": _float(regression[4]),
            "stroke_color": _rgb_from_unit_values(regression[5:8]),
            "line_spacing_ratio": _float(regression[8]),
            "angle_degrees": round((_float(regression[9]) - 0.5) * 180.0, 3),
        }


_SESSION_CACHE: dict[tuple[str, bool], Any] = {}
_SESSION_PROVIDER_METADATA: dict[tuple[str, bool], dict[str, Any]] = {}


def apply_parent_font_detection(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: Sequence[Any],
    mode: str,
    default_font_name: str = "",
    use_gpu: bool = False,
    models_dir: str | None = None,
    detector: Any | None = None,
    source_glyph_mask_result: Any | None = None,
) -> ParentFontDetectionRunResult:
    """Attach parent-owned font/style evidence to execution bundles."""

    normalized_mode = str(mode or "off").strip().lower()
    result = ParentFontDetectionRunResult(page_id=page_id, mode=normalized_mode)
    if normalized_mode == "off":
        result.skipped_count = len(list(parent_execution_bundles or []))
        return result
    if normalized_mode not in {"yuzumarker", "heuristic"}:
        result.errors.append(f"unsupported_font_detection_mode:{normalized_mode}")
        normalized_mode = "heuristic"
        result.mode = normalized_mode

    bundles = list(parent_execution_bundles or [])
    if not bundles:
        return result
    result.enabled = True

    image = None
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        result.errors.append(f"image_open_failed:{type(exc).__name__}:{exc}")

    active_detector = detector
    if normalized_mode == "yuzumarker" and active_detector is None:
        try:
            active_detector = YuzuMarkerOnnxFontDetector(
                model_path=resolve_yuzumarker_font_onnx_file(models_dir),
                labels_path=resolve_yuzumarker_font_labels_file(models_dir),
                use_gpu=use_gpu,
            )
            result.model_path = getattr(active_detector, "model_path", "") or ""
            result.labels_path = getattr(active_detector, "labels_path", "") or ""
        except Exception as exc:
            result.errors.append(f"yuzumarker_unavailable:{type(exc).__name__}:{exc}")
            active_detector = None
    if normalized_mode == "yuzumarker" and active_detector is not None:
        _copy_provider_metadata_to_result(result, active_detector)

    evidence_records: list[dict[str, Any]] = []
    for bundle in bundles:
        record = _collect_style_evidence_for_bundle(
            bundle,
            image=image,
            mode=normalized_mode,
            detector=active_detector,
            source_glyph_mask_result=source_glyph_mask_result,
        )
        evidence_records.append(record)

    result.records = _arbitrate_parent_styles(
        bundles=bundles,
        evidence_records=evidence_records,
        default_font_name=default_font_name,
        models_dir=models_dir,
    )
    for record in result.records:
        status = str(record.get("status") or "")
        if status == "applied":
            result.applied_count += 1
        elif status == "skipped":
            result.skipped_count += 1
        else:
            result.fallback_count += 1

    try:
        if image is not None:
            image.close()
    except Exception:
        pass
    return result


def _collect_style_evidence_for_bundle(
    bundle: Any,
    *,
    image: Any | None,
    mode: str,
    detector: Any | None,
    source_glyph_mask_result: Any | None = None,
) -> dict[str, Any]:
    bundle_id = str(getattr(bundle, "bundle_id", "") or "")
    parent_id = str(getattr(bundle, "parent_id", "") or "")
    root_id = str(getattr(bundle, "root_id", "") or "")
    if not bool(getattr(bundle, "render_required", False)):
        return {
            "bundle_id": bundle_id,
            "parent_id": parent_id,
            "root_id": root_id,
            "status": "skipped",
            "reason": "render_not_required",
        }

    bbox = _best_style_bbox(bundle)
    crop = _crop_image(image, bbox)
    detection = None
    status = "fallback"
    reason = "detector_unavailable"
    if mode == "yuzumarker" and detector is not None and crop is not None:
        try:
            detection = detector.detect(crop)
            confidence = _float(detection.get("confidence")) if isinstance(detection, Mapping) else 0.0
            if confidence >= MIN_STYLE_EVIDENCE_CONFIDENCE:
                status = "applied"
                reason = ""
            else:
                status = "fallback"
                reason = "low_font_confidence"
        except Exception as exc:
            status = "fallback"
            reason = f"detector_failed:{type(exc).__name__}"
    elif mode == "heuristic" and crop is not None:
        detection = _heuristic_detection(crop)
        status = "applied"
        reason = ""
    elif crop is None:
        reason = "invalid_parent_style_crop"

    raw_detection = detection if isinstance(detection, Mapping) else {}
    raw_label = str(raw_detection.get("font_path") or "")
    raw_serif = bool(raw_detection.get("font_serif")) if raw_detection else False
    color_bucket = _style_color_bucket(raw_detection)
    crop_surface = _crop_surface_evidence(crop)
    surface_bucket, surface_source, surface_reasons = _style_surface_decision(
        color_bucket=color_bucket,
        crop_surface=crop_surface,
    )
    label_weight = _font_weight_from_label(raw_label, surface_bucket=surface_bucket)
    raw_weight, weight_source, outline_to_text_ratio = _font_weight_from_visual_evidence(
        label_weight,
        detection=raw_detection,
        surface_bucket=surface_bucket,
    )
    top_candidates = _compact_candidates(raw_detection.get("top_candidates")) if raw_detection else []
    raw_weight_dominance = _candidate_weight_dominance(
        top_candidates,
        weight=raw_weight,
        surface_bucket=surface_bucket,
    )
    if raw_detection and mode == "heuristic":
        evidence_provider = HEURISTIC_PROVIDER
        evidence_source = HEURISTIC_STYLE_SOURCE
        evidence_model = ""
    elif raw_detection:
        evidence_provider = YUZUMARKER_PROVIDER
        evidence_source = YUZUMARKER_STYLE_SOURCE
        evidence_model = YUZUMARKER_PROVIDER_MODEL
    else:
        evidence_provider = ""
        evidence_source = ""
        evidence_model = ""
    semantic_style_class = _semantic_style_class(bundle)
    source_glyph_style = _source_glyph_style_evidence(
        source_glyph_mask_result,
        bundle_id=bundle_id,
        parent_id=parent_id,
        semantic_style_class=semantic_style_class,
    )
    glyph_metrics = _source_glyph_size_metrics(crop, surface_bucket=surface_bucket)
    measured_glyph_size_reliable = bool(source_glyph_style.get("geometry_reliable"))
    geometry_orientation = _geometry_orientation_for_bundle(bundle)
    source_visual_column_count = 0
    source_visual_column_source = ""
    source_visual_column_reliable = False
    if semantic_style_class == "caption" and geometry_orientation == "vertical":
        source_visual_column_count = int(glyph_metrics.get("visual_column_count") or 0)
        source_visual_column_source = str(glyph_metrics.get("visual_column_source") or "")
        source_visual_column_reliable = bool(source_visual_column_count > 0)
    record = {
        "bundle_id": bundle_id,
        "parent_id": parent_id,
        "root_id": root_id,
        "status": status,
        "fallback_reason": reason,
        "crop_bbox": list(bbox) if bbox else [],
        "detector_crop_width": int(getattr(crop, "width", 0) or 0),
        "detector_crop_height": int(getattr(crop, "height", 0) or 0),
        "surface_bucket": surface_bucket,
        "semantic_style_class": semantic_style_class,
        "style_surface_evidence_source": surface_source,
        "style_surface_reason_codes": surface_reasons,
        "crop_surface_bucket": crop_surface.get("bucket"),
        "crop_luma_median": crop_surface.get("median_luma"),
        "crop_dark_pixel_ratio": crop_surface.get("dark_pixel_ratio"),
        "crop_light_pixel_ratio": crop_surface.get("light_pixel_ratio"),
        "geometry_orientation": geometry_orientation,
        "color_bucket": color_bucket,
        "raw_style_class": _style_class_name(raw_serif, raw_weight),
        "raw_font_weight": raw_weight,
        "raw_label_weight": label_weight,
        "font_weight_evidence_source": weight_source,
        "source_outline_to_text_ratio": outline_to_text_ratio,
        "raw_font_serif": raw_serif,
        "raw_font_label": raw_label,
        "raw_font_language": str(raw_detection.get("font_language") or ""),
        "raw_direction": str(raw_detection.get("direction") or ""),
        "raw_direction_confidence": _float(raw_detection.get("direction_confidence")),
        "raw_style_confidence": _float(raw_detection.get("confidence")) if raw_detection else 0.0,
        "raw_serif_dominance": _candidate_class_dominance(top_candidates, serif=raw_serif),
        "raw_font_weight_dominance": raw_weight_dominance,
        "source_font_candidate_count": len(top_candidates),
        "raw_text_color": raw_detection.get("text_color"),
        "raw_stroke_color": raw_detection.get("stroke_color"),
        "raw_stroke_width_ratio": _float(raw_detection.get("stroke_width_ratio")),
        "raw_text_size_ratio": _float(raw_detection.get("text_size_ratio")),
        "raw_line_spacing_ratio": _float(raw_detection.get("line_spacing_ratio")),
        "measured_glyph_size_px": glyph_metrics.get("glyph_size_px"),
        "measured_glyph_bbox": glyph_metrics.get("glyph_bbox"),
        "measured_glyph_component_count": glyph_metrics.get("component_count"),
        "measured_glyph_dominant_component_count": glyph_metrics.get("dominant_component_count"),
        "measured_glyph_size_source": glyph_metrics.get("source"),
        "source_visual_column_count": source_visual_column_count,
        "source_visual_column_source": source_visual_column_source,
        "source_visual_column_reliable": source_visual_column_reliable,
        "source_visual_column_component_counts": glyph_metrics.get("visual_column_component_counts"),
        "measured_glyph_size_reliable": measured_glyph_size_reliable,
        "source_glyph_style_evidence_status": source_glyph_style.get("status"),
        "source_glyph_style_evidence_reason_codes": source_glyph_style.get("reason_codes"),
        "source_glyph_style_mask_id": source_glyph_style.get("mask_id"),
        "source_glyph_style_quality_status": source_glyph_style.get("quality_status"),
        "source_glyph_style_foreground_status": source_glyph_style.get("foreground_status"),
        "style_evidence_provider": evidence_provider,
        "style_evidence_source": evidence_source,
        "style_evidence_model": evidence_model,
        "source_font_top_candidates": top_candidates,
        "detection": dict(raw_detection) if raw_detection else {},
    }
    return {key: value for key, value in record.items() if value not in (None, "", [])}


def _arbitrate_parent_styles(
    *,
    bundles: Sequence[Any],
    evidence_records: Sequence[Mapping[str, Any]],
    default_font_name: str,
    models_dir: str | None,
) -> list[dict[str, Any]]:
    records_by_bundle = {
        str(record.get("bundle_id") or ""): dict(record)
        for record in evidence_records
        if isinstance(record, Mapping)
    }
    active: list[tuple[Any, dict[str, Any]]] = []
    final_records: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        record = records_by_bundle.get(bundle_id, {})
        if str(record.get("status") or "") == "skipped":
            final_records.append(dict(record))
            continue
        active.append((bundle, record))

    cohort_profiles = _style_cohort_profiles(
        active,
        default_font_name=default_font_name,
        models_dir=models_dir,
    )
    decision_entries: list[dict[str, Any]] = []
    for bundle, record in active:
        cohort = cohort_profiles.get(_style_cohort_key(record), {})
        decision = _style_decision_for_record(record, cohort, active)
        decision = _resolve_individual_weight_after_cohort(
            record,
            cohort,
            active,
            decision=decision,
        )
        decision_entries.append(
            {
                "bundle": bundle,
                "record": record,
                "decision": decision,
            }
        )

    _reconcile_same_root_style_decisions(
        decision_entries,
        active=active,
        cohort_profiles=cohort_profiles,
    )

    render_entries: list[dict[str, Any]] = []
    for entry in decision_entries:
        bundle = entry["bundle"]
        record = entry["record"]
        decision = entry["decision"]
        detection = record.get("detection") if isinstance(record.get("detection"), Mapping) else None
        style = _style_for_bundle(
            bundle,
            detection=detection,
            status=str(record.get("status") or "fallback"),
            fallback_reason=str(record.get("fallback_reason") or ""),
            default_font_name=default_font_name,
            models_dir=models_dir,
            final_serif=bool(decision.get("serif")),
            final_weight=str(decision.get("weight") or "regular"),
            final_style_class=str(decision.get("style_class") or ""),
            final_surface_bucket=str(decision.get("surface_bucket") or record.get("surface_bucket") or ""),
            arbitration=decision,
            evidence_provider=str(record.get("style_evidence_provider") or ""),
            evidence_source=str(record.get("style_evidence_source") or ""),
            evidence_model=str(record.get("style_evidence_model") or ""),
        )
        render_entries.append({**entry, "style": style})

    _reconcile_same_root_font_sizes(render_entries)

    for entry in render_entries:
        bundle = entry["bundle"]
        record = entry["record"]
        decision = entry["decision"]
        style = entry["style"]
        _sync_style_arbitration_metadata(style, decision)
        _merge_render_style(bundle, style)
        if hasattr(bundle, "execution_region"):
            try:
                bundle.execution_region = bundle.to_region_record()
            except Exception:
                pass

        final_record = dict(record)
        final_record.update(
            {
                "status": "applied" if str(record.get("status") or "") == "applied" else "fallback",
                "render_style_source": style.get("render_style_source"),
                "render_style_provider": style.get("render_style_provider"),
                "render_style_provider_model": style.get("render_style_provider_model"),
                "render_style_confidence": style.get("render_style_confidence"),
                "font_family": style.get("font_family"),
                "font_weight": style.get("font_weight"),
                "font_serif": bool(decision.get("serif")),
                "style_class": style.get("style_class"),
                "style_surface_bucket": style.get("style_surface_bucket"),
                "semantic_style_class": style.get("semantic_style_class"),
                "style_surface_evidence_source": style.get("style_surface_evidence_source"),
                "style_surface_reason_codes": style.get("style_surface_reason_codes"),
                "crop_surface_bucket": style.get("crop_surface_bucket"),
                "crop_luma_median": style.get("crop_luma_median"),
                "crop_dark_pixel_ratio": style.get("crop_dark_pixel_ratio"),
                "crop_light_pixel_ratio": style.get("crop_light_pixel_ratio"),
                "raw_label_weight": style.get("raw_label_weight"),
                "font_weight_evidence_source": style.get("font_weight_evidence_source"),
                "source_outline_to_text_ratio": style.get("source_outline_to_text_ratio"),
                "measured_glyph_size_reliable": style.get("measured_glyph_size_reliable"),
                "source_glyph_style_evidence_status": style.get(
                    "source_glyph_style_evidence_status"
                ),
                "source_glyph_style_evidence_reason_codes": style.get(
                    "source_glyph_style_evidence_reason_codes"
                ),
                "source_glyph_style_mask_id": style.get("source_glyph_style_mask_id"),
                "source_glyph_style_quality_status": style.get(
                    "source_glyph_style_quality_status"
                ),
                "source_glyph_style_foreground_status": style.get(
                    "source_glyph_style_foreground_status"
                ),
                "style_arbitration_decision": decision.get("decision"),
                "style_arbitration_reason_codes": decision.get("reason_codes"),
                "style_arbitration_cohort_id": decision.get("cohort_id"),
                "style_family_bucket": decision.get("style_family_bucket"),
                "source_font_weight_dominance": style.get("source_font_weight_dominance"),
                "individual_weight_arbitration_decision": style.get(
                    "individual_weight_arbitration_decision"
                ),
                "individual_weight_arbitration_reason_codes": style.get(
                    "individual_weight_arbitration_reason_codes"
                ),
                "cohort_font_weight": style.get("cohort_font_weight"),
                "source_weight_peer_stroke_median": style.get(
                    "source_weight_peer_stroke_median"
                ),
                "source_weight_stroke_ratio": style.get("source_weight_stroke_ratio"),
                "detector_crop_width": style.get("detector_crop_width"),
                "detector_crop_height": style.get("detector_crop_height"),
                "source_visual_column_count": style.get("source_visual_column_count"),
                "source_visual_column_source": style.get("source_visual_column_source"),
                "source_visual_column_reliable": style.get("source_visual_column_reliable"),
                "source_visual_column_component_counts": style.get(
                    "source_visual_column_component_counts"
                ),
                "font_size_hint": style.get("font_size_hint"),
                "raw_parent_font_size_hint": style.get("raw_parent_font_size_hint"),
                "font_size_normalization": style.get("font_size_normalization"),
                "font_size_source": style.get("font_size_source"),
                "measured_glyph_size_px": style.get("measured_glyph_size_px"),
                "arbitrated_glyph_size_px": style.get("arbitrated_glyph_size_px"),
                "font_size_arbitration_decision": style.get("font_size_arbitration_decision"),
                "font_size_arbitration_reason_codes": style.get("font_size_arbitration_reason_codes"),
                "measured_glyph_dominant_component_count": style.get("measured_glyph_dominant_component_count"),
            }
        )
        final_records.append({key: value for key, value in final_record.items() if value not in (None, "", [])})
    return final_records


def _style_for_bundle(
    bundle: Any,
    *,
    detection: Mapping[str, Any] | None,
    status: str,
    fallback_reason: str,
    default_font_name: str,
    models_dir: str | None,
    final_serif: bool,
    final_weight: str,
    final_style_class: str,
    final_surface_bucket: str,
    arbitration: Mapping[str, Any],
    evidence_provider: str,
    evidence_source: str,
    evidence_model: str,
) -> dict[str, Any]:
    existing = dict(getattr(bundle, "render_style", {}) or {})
    surface_bucket = str(final_surface_bucket or _style_surface_bucket(bundle, color_bucket="dark_on_light"))
    semantic_style_class = str(
        final_style_class
        or arbitration.get("semantic_style_class")
        or existing.get("style_class")
        or _semantic_style_class(bundle)
    )
    contrast_surface = surface_bucket == "light_on_dark"
    confidence = _float(detection.get("confidence")) if detection else 0.0
    trusted_detection = detection if detection and confidence >= MIN_STYLE_EVIDENCE_CONFIDENCE else {}
    weight = str(final_weight or "regular")
    serif = bool(final_serif)
    font_path = resolve_noto_cjk_sc_font_file(base_dir=models_dir, serif=serif, weight=weight)
    if not font_path:
        font_path = default_font_name or existing.get("font_family") or "Microsoft YaHei"

    source_orientation = _style_source_orientation(bundle, trusted_detection or detection or {})
    colors = _render_colors_for_surface(surface_bucket)
    style: dict[str, Any] = {
        "render_style_version": PARENT_RENDER_STYLE_VERSION,
        "render_style_owner": "parent_execution_bundle",
        "render_style_source": STYLE_ARBITRATOR_SOURCE,
        "render_style_provider": STYLE_ARBITRATOR_PROVIDER,
        "render_style_provider_model": evidence_model if detection else "",
        "style_class": semantic_style_class,
        "semantic_style_class": semantic_style_class,
        "font_family": font_path,
        "font_weight": weight,
        "fill_color": colors["fill_color"],
        "stroke_color": colors["stroke_color"],
        "stroke_width": 2 if contrast_surface else 1,
        "source_orientation": existing.get("source_orientation") or source_orientation,
        "wrap_mode": existing.get("wrap_mode") or ("vertical" if (source_orientation != "horizontal") else "horizontal"),
        "line_height": existing.get("line_height") or (1.1 if contrast_surface else 1.0),
        "align": existing.get("align") or "center",
        "fallback_reason": fallback_reason,
        "font_detection_status": status,
        "style_surface_bucket": surface_bucket,
        "style_surface_evidence_source": arbitration.get("style_surface_evidence_source"),
        "style_surface_reason_codes": list(arbitration.get("style_surface_reason_codes") or []),
        "crop_surface_bucket": arbitration.get("crop_surface_bucket"),
        "crop_luma_median": arbitration.get("crop_luma_median"),
        "crop_dark_pixel_ratio": arbitration.get("crop_dark_pixel_ratio"),
        "crop_light_pixel_ratio": arbitration.get("crop_light_pixel_ratio"),
        "raw_label_weight": arbitration.get("raw_label_weight"),
        "font_weight_evidence_source": arbitration.get("font_weight_evidence_source"),
        "source_outline_to_text_ratio": arbitration.get("source_outline_to_text_ratio"),
        "measured_glyph_size_reliable": bool(
            arbitration.get("measured_glyph_size_reliable")
        ),
        "source_glyph_style_evidence_status": arbitration.get(
            "source_glyph_style_evidence_status"
        ),
        "source_glyph_style_evidence_reason_codes": list(
            arbitration.get("source_glyph_style_evidence_reason_codes") or []
        ),
        "source_glyph_style_mask_id": arbitration.get("source_glyph_style_mask_id"),
        "source_glyph_style_quality_status": arbitration.get(
            "source_glyph_style_quality_status"
        ),
        "source_glyph_style_foreground_status": arbitration.get(
            "source_glyph_style_foreground_status"
        ),
        "style_arbitration": dict(arbitration),
        "style_arbitration_decision": arbitration.get("decision"),
        "style_arbitration_reason_codes": list(arbitration.get("reason_codes") or []),
        "style_arbitration_cohort_id": arbitration.get("cohort_id"),
        "style_family_bucket": arbitration.get("style_family_bucket"),
        "style_arbitration_provider": STYLE_ARBITRATOR_PROVIDER,
        "preserved_size_exception": bool(arbitration.get("preserved_size_exception")),
        "source_font_weight_dominance": _float(arbitration.get("raw_font_weight_dominance")),
        "source_font_candidate_count": int(arbitration.get("source_font_candidate_count") or 0),
        "individual_weight_arbitration_decision": arbitration.get(
            "individual_weight_arbitration_decision"
        ),
        "individual_weight_arbitration_reason_codes": list(
            arbitration.get("individual_weight_arbitration_reason_codes") or []
        ),
        "cohort_font_weight": arbitration.get("cohort_font_weight"),
        "source_weight_peer_stroke_median": _float(
            arbitration.get("source_weight_peer_stroke_median")
        ),
        "source_weight_stroke_ratio": _float(arbitration.get("source_weight_stroke_ratio")),
        "detector_crop_width": int(arbitration.get("detector_crop_width") or 0),
        "detector_crop_height": int(arbitration.get("detector_crop_height") or 0),
        "source_visual_column_count": int(
            arbitration.get("source_visual_column_count") or 0
        ),
        "source_visual_column_source": arbitration.get("source_visual_column_source"),
        "source_visual_column_reliable": bool(
            arbitration.get("source_visual_column_reliable")
        ),
        "source_visual_column_component_counts": list(
            arbitration.get("source_visual_column_component_counts") or []
        ),
    }
    layout_hints = _layout_hints_for_bundle(
        bundle,
        detection=trusted_detection if trusted_detection else None,
        surface_bucket=surface_bucket,
        semantic_style_class=semantic_style_class,
        source_orientation=str(style.get("source_orientation") or ""),
        measured_glyph_size_px=_float(arbitration.get("arbitrated_glyph_size_px")),
        measured_glyph_size_reliable=bool(
            arbitration.get("measured_glyph_size_reliable")
        ),
        cohort_glyph_size_px=_float(arbitration.get("cohort_measured_glyph_size_px")),
        font_size_arbitration_decision=str(arbitration.get("font_size_arbitration_decision") or ""),
        measured_glyph_size_source=str(arbitration.get("measured_glyph_size_source") or ""),
        measured_glyph_dominant_component_count=int(
            arbitration.get("measured_glyph_dominant_component_count") or 0
        ),
        detector_crop_width=int(arbitration.get("detector_crop_width") or 0),
        font_family=font_path,
    )
    style.update(_normalize_layout_hints_to_visual_cohort(layout_hints, arbitration=arbitration))
    _lock_resolved_parent_font_size(style)
    if detection:
        style.update(
            {
                "render_style_confidence": confidence,
                "style_evidence_provider": evidence_provider,
                "style_evidence_source": evidence_source,
                "source_font_label": str(detection.get("font_path") or ""),
                "source_font_language": str(detection.get("font_language") or ""),
                "source_font_serif": bool(detection.get("font_serif")),
                "source_font_top_candidates": _compact_candidates(detection.get("top_candidates")),
                "source_text_color": detection.get("text_color"),
                "source_stroke_color": detection.get("stroke_color"),
                "source_stroke_width_ratio": _float(detection.get("stroke_width_ratio")),
                "source_text_size_ratio": _float(detection.get("text_size_ratio")),
                "source_line_spacing_ratio": _float(detection.get("line_spacing_ratio")),
                "source_angle_degrees": _float(detection.get("angle_degrees")),
                "source_direction": str(detection.get("direction") or ""),
                "source_direction_confidence": _float(detection.get("direction_confidence")),
                "measured_glyph_size_px": _float(arbitration.get("measured_glyph_size_px")),
                "arbitrated_glyph_size_px": _float(arbitration.get("arbitrated_glyph_size_px")),
                "font_size_arbitration_decision": arbitration.get("font_size_arbitration_decision"),
                "font_size_arbitration_reason_codes": list(
                    arbitration.get("font_size_arbitration_reason_codes") or []
                ),
                "measured_glyph_bbox": arbitration.get("measured_glyph_bbox"),
                "measured_glyph_component_count": int(arbitration.get("measured_glyph_component_count") or 0),
                "measured_glyph_dominant_component_count": int(
                    arbitration.get("measured_glyph_dominant_component_count") or 0
                ),
                "measured_glyph_size_source": arbitration.get("measured_glyph_size_source"),
            }
        )
    if str(status or "") == "applied":
        style["fallback_reason"] = ""
    return {key: value for key, value in style.items() if value not in (None, "", [])}


def _style_cohort_profiles(
    active: Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    default_font_name: str = "",
    models_dir: str | None = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[Any, Mapping[str, Any]]]] = {}
    for bundle, record in active:
        grouped.setdefault(_style_cohort_key(record), []).append((bundle, record))

    profiles: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, members in grouped.items():
        semantic_style_class, orientation, family_bucket = key
        records = [record for _bundle, record in members]
        usable = [
            record
            for record in records
            if str(record.get("status") or "") == "applied"
            and _float(record.get("raw_style_confidence")) >= MIN_STYLE_EVIDENCE_CONFIDENCE
        ]
        serif_weight = sum(
            _float(record.get("raw_style_confidence"))
            for record in usable
            if bool(record.get("raw_font_serif"))
        )
        sans_weight = sum(
            _float(record.get("raw_style_confidence"))
            for record in usable
            if not bool(record.get("raw_font_serif"))
        )
        total_weight = serif_weight + sans_weight
        serif_ratio = serif_weight / total_weight if total_weight > 0 else 0.0
        canonical_serif = bool(serif_ratio >= 0.70 and len(usable) >= 2)
        family_reason_codes: list[str] = []
        if len(usable) == 1:
            singleton = usable[0]
            if (
                int(singleton.get("source_font_candidate_count") or 0)
                >= MIN_INDIVIDUAL_WEIGHT_CANDIDATES
                and _float(singleton.get("raw_serif_dominance")) >= 0.90
            ):
                canonical_serif = bool(singleton.get("raw_font_serif"))
                family_reason_codes.append("singleton_candidate_family_consensus")
        canonical_weight = _majority_weight(
            usable,
            default="regular",
        )
        if canonical_weight in {"bold", "black"} and _weight_ratio(usable, canonical_weight) < 0.70:
            canonical_weight = "regular"

        style_class = semantic_style_class
        cohort_id = "style_cohort:{}:{}:{}".format(
            semantic_style_class,
            orientation,
            family_bucket,
        )
        canonical_font_family = (
            resolve_noto_cjk_sc_font_file(base_dir=models_dir, serif=canonical_serif, weight=canonical_weight)
            or default_font_name
        )
        font_size_profile = _cohort_font_size_profile(
            members,
            canonical_font_family=canonical_font_family,
        )
        profiles[key] = {
            "cohort_id": cohort_id,
            "member_parent_ids": [str(record.get("parent_id") or "") for record in records if record.get("parent_id")],
            "semantic_style_class": semantic_style_class,
            "orientation": orientation,
            "style_family_bucket": family_bucket,
            "serif": canonical_serif,
            "weight": canonical_weight,
            "style_class": style_class,
            "canonical_style_class": _style_class_name(canonical_serif, canonical_weight),
            "median_text_size_ratio": _median(_float(record.get("raw_text_size_ratio")) for record in usable),
            "median_line_spacing_ratio": _median(_float(record.get("raw_line_spacing_ratio")) for record in usable),
            "median_stroke_width_ratio": _median(_float(record.get("raw_stroke_width_ratio")) for record in usable),
            "median_measured_glyph_size_px": _median(
                _float(record.get("measured_glyph_size_px"))
                for record in usable
                if bool(record.get("measured_glyph_size_reliable"))
            ),
            "usable_evidence_count": len(usable),
            "serif_ratio": round(float(serif_ratio), 4),
            "family_reason_codes": family_reason_codes,
            **font_size_profile,
        }
    return profiles


def _style_decision_for_record(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    active: Sequence[tuple[Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    surface_bucket = str(record.get("surface_bucket") or "dark_on_light")
    raw_serif = bool(record.get("raw_font_serif"))
    raw_weight = str(record.get("raw_font_weight") or "regular")
    raw_class = _style_class_name(raw_serif, raw_weight)
    cohort_serif = bool(cohort.get("serif"))
    cohort_weight = str(cohort.get("weight") or "regular")
    cohort_class = _style_class_name(cohort_serif, cohort_weight)
    status = str(record.get("status") or "")
    reason_codes: list[str] = list(cohort.get("family_reason_codes") or [])
    preserved_size_exception = _should_preserve_size_exception(record, cohort, active)

    if status != "applied":
        reason_codes.append(str(record.get("fallback_reason") or "no_usable_style_evidence"))
        return _style_decision(
            record,
            cohort,
            surface_bucket=surface_bucket,
            serif=cohort_serif,
            weight=cohort_weight,
            style_class=str(
                cohort.get("style_class")
                or record.get("semantic_style_class")
                or "dialogue"
            ),
            decision="fallback_to_parent_style_default",
            reason_codes=reason_codes,
            preserved=False,
            preserved_size=preserved_size_exception,
        )

    if raw_class == cohort_class:
        reason_codes.append("matches_visual_cohort")
        return _style_decision(
            record,
            cohort,
            surface_bucket=surface_bucket,
            serif=cohort_serif,
            weight=cohort_weight,
            style_class=str(
                cohort.get("style_class")
                or record.get("semantic_style_class")
                or "dialogue"
            ),
            decision="accepted_visual_cohort_style",
            reason_codes=reason_codes,
            preserved=False,
            preserved_size=preserved_size_exception,
        )

    if _should_preserve_style_exception(record, cohort, active):
        reason_codes.extend(
            [
                "strong_distinct_parent_style_evidence",
                "not_contradicted_by_same_root_sibling",
            ]
        )
        return _style_decision(
            record,
            cohort,
            surface_bucket=surface_bucket,
            serif=raw_serif,
            weight=raw_weight,
            style_class=str(
                cohort.get("style_class")
                or record.get("semantic_style_class")
                or "dialogue"
            ),
            decision="preserved_distinct_visual_style",
            reason_codes=reason_codes,
            preserved=True,
            preserved_size=preserved_size_exception,
        )

    reason_codes.extend(
        [
            "normalized_to_visual_cohort",
            "model_family_difference_treated_as_visual_noise",
        ]
    )
    return _style_decision(
        record,
        cohort,
        surface_bucket=surface_bucket,
        serif=cohort_serif,
        weight=cohort_weight,
        style_class=str(
            cohort.get("style_class")
            or record.get("semantic_style_class")
            or "dialogue"
        ),
        decision="normalized_to_visual_cohort",
        reason_codes=reason_codes,
        preserved=False,
        preserved_size=preserved_size_exception,
    )


def _resolve_individual_weight_after_cohort(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    active: Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve weight without changing the record's visual-cohort identity.

    Candidate labels are not sufficient on their own.  A heavier individual
    weight is accepted only when the candidate distribution is complete and
    the source crop has reliable stroke evidence that is materially heavier
    than its same-cohort peers.  Family cohort and size decisions are already
    fixed before this function and remain untouched.
    """

    resolved = dict(decision)
    cohort_weight = str(resolved.get("weight") or cohort.get("weight") or "regular")
    raw_weight = str(record.get("raw_font_weight") or "regular")
    reason_codes: list[str] = []
    resolved.update(
        {
            "cohort_font_weight": cohort_weight,
            "individual_weight_arbitration_decision": "kept_cohort_weight",
            "individual_weight_arbitration_reason_codes": reason_codes,
            "source_weight_peer_stroke_median": 0.0,
            "source_weight_stroke_ratio": 0.0,
        }
    )

    if str(record.get("status") or "") != "applied":
        reason_codes.append("no_applied_style_evidence")
        return resolved
    if raw_weight == cohort_weight:
        reason_codes.append("matches_cohort_weight")
        return resolved
    if _font_weight_rank(raw_weight) <= _font_weight_rank(cohort_weight):
        reason_codes.append("individual_weight_not_heavier_than_cohort")
        return resolved

    candidate_count = int(record.get("source_font_candidate_count") or 0)
    if candidate_count < MIN_INDIVIDUAL_WEIGHT_CANDIDATES:
        reason_codes.append("insufficient_candidate_distribution")
        return resolved
    dominance = _float(record.get("raw_font_weight_dominance"))
    if dominance < COMPLETE_INDIVIDUAL_WEIGHT_DOMINANCE:
        reason_codes.append("candidate_weight_consensus_incomplete")
        return resolved

    dominant_components = int(record.get("measured_glyph_dominant_component_count") or 0)
    if dominant_components < MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER:
        reason_codes.append("source_weight_evidence_sparse")
        return resolved
    source_stroke = _float(record.get("raw_stroke_width_ratio"))
    if source_stroke <= 0:
        reason_codes.append("source_stroke_evidence_missing")
        return resolved

    record_bundle_id = str(record.get("bundle_id") or "")
    cohort_key = _style_cohort_key(record)
    peer_strokes = [
        _float(other.get("raw_stroke_width_ratio"))
        for _bundle, other in active
        if str(other.get("bundle_id") or "") != record_bundle_id
        and str(other.get("status") or "") == "applied"
        and _style_cohort_key(other) == cohort_key
        and int(other.get("measured_glyph_dominant_component_count") or 0)
        >= MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER
        and _float(other.get("raw_stroke_width_ratio")) > 0
    ]
    peer_stroke_median = _median(peer_strokes)
    resolved["source_weight_peer_stroke_median"] = round(float(peer_stroke_median), 6)
    if peer_stroke_median <= 0:
        reason_codes.append("cohort_peer_stroke_baseline_missing")
        return resolved

    stroke_ratio = source_stroke / peer_stroke_median
    resolved["source_weight_stroke_ratio"] = round(float(stroke_ratio), 6)
    if stroke_ratio < MIN_INDIVIDUAL_WEIGHT_STROKE_RATIO:
        reason_codes.append("source_stroke_not_distinct_from_cohort")
        return resolved

    reason_codes.extend(
        [
            "candidate_weight_consensus_complete",
            "source_stroke_distinct_from_cohort",
            "cohort_membership_preserved",
        ]
    )
    arbitration_reasons = list(resolved.get("reason_codes") or [])
    arbitration_reasons.extend(reason for reason in reason_codes if reason not in arbitration_reasons)
    resolved.update(
        {
            "decision": "resolved_individual_weight_within_visual_cohort",
            "reason_codes": arbitration_reasons,
            "weight": raw_weight,
            "individual_weight_arbitration_decision": (
                "resolved_from_candidate_consensus_and_source_stroke"
            ),
            "individual_weight_arbitration_reason_codes": reason_codes,
        }
    )
    return resolved


def _font_weight_rank(weight: str) -> int:
    return {"regular": 0, "bold": 1, "black": 2}.get(str(weight or "regular"), 0)


def _style_decision(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    *,
    surface_bucket: str,
    serif: bool,
    weight: str,
    style_class: str,
    decision: str,
    reason_codes: Sequence[str],
    preserved: bool,
    preserved_size: bool = False,
) -> dict[str, Any]:
    semantic_style_class = str(
        record.get("semantic_style_class")
        or cohort.get("semantic_style_class")
        or style_class
        or "dialogue"
    )
    return {
        "decision": decision,
        "reason_codes": [str(reason) for reason in reason_codes if reason],
        "cohort_id": str(cohort.get("cohort_id") or ""),
        "cohort_parent_ids": list(cohort.get("member_parent_ids") or []),
        "cohort_size": len(list(cohort.get("member_parent_ids") or [])),
        "raw_style_class": str(record.get("raw_style_class") or ""),
        "canonical_style_class": str(cohort.get("canonical_style_class") or ""),
        "style_family_bucket": str(cohort.get("style_family_bucket") or ""),
        "surface_bucket": str(surface_bucket or cohort.get("surface_bucket") or "dark_on_light"),
        "serif": bool(serif),
        "weight": str(weight or "regular"),
        "style_class": semantic_style_class,
        "semantic_style_class": semantic_style_class,
        "style_surface_evidence_source": str(
            record.get("style_surface_evidence_source") or ""
        ),
        "style_surface_reason_codes": list(
            record.get("style_surface_reason_codes") or []
        ),
        "crop_surface_bucket": str(record.get("crop_surface_bucket") or ""),
        "crop_luma_median": _float(record.get("crop_luma_median")),
        "crop_dark_pixel_ratio": _float(record.get("crop_dark_pixel_ratio")),
        "crop_light_pixel_ratio": _float(record.get("crop_light_pixel_ratio")),
        "detector_crop_width": int(record.get("detector_crop_width") or 0),
        "detector_crop_height": int(record.get("detector_crop_height") or 0),
        "raw_label_weight": str(record.get("raw_label_weight") or "regular"),
        "font_weight_evidence_source": str(
            record.get("font_weight_evidence_source") or ""
        ),
        "source_outline_to_text_ratio": _float(
            record.get("source_outline_to_text_ratio")
        ),
        "measured_glyph_size_reliable": bool(
            record.get("measured_glyph_size_reliable")
        ),
        "source_glyph_style_evidence_status": str(
            record.get("source_glyph_style_evidence_status") or ""
        ),
        "source_glyph_style_evidence_reason_codes": list(
            record.get("source_glyph_style_evidence_reason_codes") or []
        ),
        "source_glyph_style_mask_id": str(
            record.get("source_glyph_style_mask_id") or ""
        ),
        "source_glyph_style_quality_status": str(
            record.get("source_glyph_style_quality_status") or ""
        ),
        "source_glyph_style_foreground_status": str(
            record.get("source_glyph_style_foreground_status") or ""
        ),
        "preserved_exception": bool(preserved),
        "preserved_size_exception": bool(preserved_size),
        "raw_style_confidence": _float(record.get("raw_style_confidence")),
        "raw_font_weight_dominance": _float(record.get("raw_font_weight_dominance")),
        "source_font_candidate_count": int(record.get("source_font_candidate_count") or 0),
        "measured_glyph_size_px": _float(record.get("measured_glyph_size_px")),
        "measured_glyph_bbox": list(record.get("measured_glyph_bbox") or []),
        "measured_glyph_component_count": int(record.get("measured_glyph_component_count") or 0),
        "measured_glyph_dominant_component_count": int(record.get("measured_glyph_dominant_component_count") or 0),
        "measured_glyph_size_source": str(record.get("measured_glyph_size_source") or ""),
        "source_visual_column_count": int(record.get("source_visual_column_count") or 0),
        "source_visual_column_source": str(record.get("source_visual_column_source") or ""),
        "source_visual_column_reliable": bool(record.get("source_visual_column_reliable")),
        "source_visual_column_component_counts": list(
            record.get("source_visual_column_component_counts") or []
        ),
        "cohort_measured_glyph_size_px": _float(cohort.get("median_measured_glyph_size_px")),
        **_style_size_decision(
            record,
            cohort,
            style_decision=decision,
            preserved_style=preserved,
            preserved_size=preserved_size,
        ),
        "cohort_font_size_hint": int(cohort.get("cohort_font_size_hint") or 0),
        "cohort_font_size_min": int(cohort.get("cohort_font_size_min") or 0),
        "cohort_font_size_max": int(cohort.get("cohort_font_size_max") or 0),
        "cohort_font_size_source": str(cohort.get("cohort_font_size_source") or ""),
        "cohort_raw_font_size_hints": list(cohort.get("cohort_raw_font_size_hints") or []),
    }


def _reconcile_same_root_style_decisions(
    entries: Sequence[dict[str, Any]],
    *,
    active: Sequence[tuple[Any, Mapping[str, Any]]],
    cohort_profiles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    """Reconcile detector-created style splits inside one semantic root.

    Family cohorts remain independent across roots.  Within one root, however,
    conflicting detector families are noise unless the surrounding visual cohort
    supports the split.  The semantic topology is consumed here; it is never
    recreated or changed.
    """

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        record = entry.get("record") or {}
        root_id = str(record.get("root_id") or "")
        if not root_id:
            continue
        key = (
            root_id,
            str(record.get("surface_bucket") or "dark_on_light"),
            str(record.get("geometry_orientation") or "vertical"),
        )
        grouped.setdefault(key, []).append(entry)

    for (root_id, surface_bucket, orientation), members in grouped.items():
        if len(members) < 2:
            continue
        family_buckets = {
            str((entry.get("decision") or {}).get("style_family_bucket") or "general")
            for entry in members
        }
        style_signatures = {
            (
                bool((entry.get("decision") or {}).get("serif")),
                str((entry.get("decision") or {}).get("weight") or "regular"),
            )
            for entry in members
        }
        if len(family_buckets) <= 1 and len(style_signatures) <= 1:
            continue

        profile = _root_visual_style_profile(
            surface_bucket=surface_bucket,
            orientation=orientation,
            semantic_style_class=str(
                ((members[0].get("record") or {}).get("semantic_style_class") or "dialogue")
            ),
            active=active,
            cohort_profiles=cohort_profiles,
        )
        canonical_serif = bool(profile.get("serif"))
        canonical_weight = str(profile.get("weight") or "regular")
        canonical_family_bucket = str(profile.get("style_family_bucket") or "general")
        root_parent_ids = [
            str((entry.get("record") or {}).get("parent_id") or "")
            for entry in members
            if (entry.get("record") or {}).get("parent_id")
        ]
        for entry in members:
            decision = dict(entry.get("decision") or {})
            original_family_bucket = str(decision.get("style_family_bucket") or "general")
            decision["root_original_style_family_bucket"] = original_family_bucket
            decision["root_visual_style_id"] = f"root_visual_style:{root_id}:{surface_bucket}:{orientation}"
            decision["root_visual_parent_ids"] = root_parent_ids
            decision["root_visual_family_bucket"] = canonical_family_bucket
            differs = (
                bool(decision.get("serif")) is not canonical_serif
                or str(decision.get("weight") or "regular") != canonical_weight
                or original_family_bucket != canonical_family_bucket
            )
            if differs:
                reasons = list(decision.get("reason_codes") or [])
                if "same_root_visual_style_reconciliation" not in reasons:
                    reasons.append("same_root_visual_style_reconciliation")
                decision.update(
                    {
                        "decision": "normalized_to_root_visual_style",
                        "reason_codes": reasons,
                        "serif": canonical_serif,
                        "weight": canonical_weight,
                        "style_class": str(
                            profile.get("style_class")
                            or (entry.get("record") or {}).get("semantic_style_class")
                            or "dialogue"
                        ),
                        "canonical_style_class": _style_class_name(
                            canonical_serif,
                            canonical_weight,
                        ),
                        "style_family_bucket": canonical_family_bucket,
                        "preserved_exception": False,
                    }
                )
            entry["decision"] = decision


def _root_visual_style_profile(
    *,
    surface_bucket: str,
    orientation: str,
    semantic_style_class: str,
    active: Sequence[tuple[Any, Mapping[str, Any]]],
    cohort_profiles: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    visual_records = [
        record
        for _bundle, record in active
        if str(record.get("surface_bucket") or "dark_on_light") == surface_bucket
        and str(record.get("geometry_orientation") or "vertical") == orientation
        and str(record.get("semantic_style_class") or "dialogue") == semantic_style_class
        and str(record.get("status") or "") == "applied"
        and _float(record.get("raw_style_confidence")) >= MIN_STYLE_EVIDENCE_CONFIDENCE
    ]
    general = dict(cohort_profiles.get((semantic_style_class, orientation, "general")) or {})
    if int(general.get("usable_evidence_count") or 0) >= 2:
        return general

    family_scores: dict[str, float] = {}
    for record in visual_records:
        bucket = _style_family_cohort_bucket(record)
        family_scores[bucket] = family_scores.get(bucket, 0.0) + max(
            MIN_STYLE_EVIDENCE_CONFIDENCE,
            _float(record.get("raw_style_confidence")),
        )
    anchor_bucket = "general"
    if family_scores:
        anchor_bucket = max(
            family_scores,
            key=lambda bucket: (family_scores[bucket], bucket == "general"),
        )
    selected = dict(cohort_profiles.get((semantic_style_class, orientation, anchor_bucket)) or {})
    if selected:
        return selected

    serif_weight = sum(
        _float(record.get("raw_style_confidence"))
        for record in visual_records
        if bool(record.get("raw_font_serif"))
    )
    sans_weight = sum(
        _float(record.get("raw_style_confidence"))
        for record in visual_records
        if not bool(record.get("raw_font_serif"))
    )
    total_weight = serif_weight + sans_weight
    serif_ratio = serif_weight / total_weight if total_weight > 0 else 0.0
    canonical_serif = bool(serif_ratio >= 0.70 and len(visual_records) >= 2)
    canonical_weight = _majority_weight(
        visual_records,
        default="regular",
    )
    if canonical_weight in {"bold", "black"} and _weight_ratio(visual_records, canonical_weight) < 0.70:
        canonical_weight = "regular"
    return {
        "surface_bucket": surface_bucket,
        "orientation": orientation,
        "style_family_bucket": anchor_bucket,
        "serif": canonical_serif,
        "weight": canonical_weight,
        "style_class": semantic_style_class,
    }


def _reconcile_same_root_font_sizes(entries: Sequence[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        record = entry.get("record") or {}
        root_id = str(record.get("root_id") or "")
        if not root_id:
            continue
        key = (
            root_id,
            str(record.get("surface_bucket") or "dark_on_light"),
            str(record.get("geometry_orientation") or "vertical"),
        )
        grouped.setdefault(key, []).append(entry)

    for members in grouped.values():
        if len(members) < 2:
            continue
        sizes = [
            int((entry.get("style") or {}).get("font_size_hint") or 0)
            for entry in members
            if int((entry.get("style") or {}).get("font_size_hint") or 0) > 0
        ]
        if len(sizes) < 2 or min(sizes) == max(sizes):
            continue
        median_size = max(1, int(round(_median(sizes))))
        relative_range = (max(sizes) - min(sizes)) / max(1.0, float(median_size))
        style_reconciled = any(
            str((entry.get("decision") or {}).get("decision") or "")
            == "normalized_to_root_visual_style"
            for entry in members
        )
        sparse_evidence = any(
            int((entry.get("record") or {}).get("measured_glyph_dominant_component_count") or 0)
            < MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER
            for entry in members
        )
        if not style_reconciled and not sparse_evidence and relative_range > 0.18:
            continue

        anchor_bucket = str(
            (members[0].get("decision") or {}).get("root_visual_family_bucket")
            or (members[0].get("decision") or {}).get("style_family_bucket")
            or "general"
        )
        anchor_sizes = [
            int((entry.get("style") or {}).get("font_size_hint") or 0)
            for entry in members
            if str(
                (entry.get("decision") or {}).get("root_original_style_family_bucket")
                or (entry.get("decision") or {}).get("style_family_bucket")
                or "general"
            )
            == anchor_bucket
            and int((entry.get("style") or {}).get("font_size_hint") or 0) > 0
        ]
        canonical_size = max(1, int(round(_median(anchor_sizes or sizes))))
        for entry in members:
            style = entry.get("style") or {}
            if int(style.get("font_size_hint") or 0) == canonical_size:
                continue
            _set_locked_parent_font_size(
                style,
                canonical_size,
                normalization="root_visual_style",
                normalization_source="same_root_visual_style_reconciliation",
            )
            decision = dict(entry.get("decision") or {})
            reasons = list(decision.get("reason_codes") or [])
            if "same_root_visual_style_reconciliation" not in reasons:
                reasons.append("same_root_visual_style_reconciliation")
            decision.update(
                {
                    "decision": "normalized_to_root_visual_style",
                    "reason_codes": reasons,
                    "font_size_arbitration_decision": "normalized_to_root_visual_size",
                    "font_size_arbitration_reason_codes": [
                        "same_root_visual_style_reconciliation",
                        f"root_font_size_relative_range:{relative_range:.3f}",
                    ],
                    "root_visual_font_size_hint": canonical_size,
                    "preserved_exception": False,
                }
            )
            entry["decision"] = decision


def _set_locked_parent_font_size(
    style: dict[str, Any],
    size: int,
    *,
    normalization: str,
    normalization_source: str,
) -> None:
    previous = int(style.get("font_size_hint") or style.get("font_size") or 0)
    if previous > 0 and previous != size and not style.get("raw_parent_font_size_hint"):
        style["raw_parent_font_size_hint"] = previous
    style["font_size"] = size
    style["font_size_hint"] = size
    style["font_size_min"] = size
    style["font_size_max"] = size
    style["font_size_locked"] = True
    style["font_size_normalization"] = normalization
    style["font_size_normalization_source"] = normalization_source
    style["root_visual_font_size_hint"] = size
    spacing = dict(style.get("spacing_profile") or {})
    if previous > 0 and previous != size and not spacing.get("raw_parent_font_size_hint"):
        spacing["raw_parent_font_size_hint"] = previous
    spacing.update(
        {
            "font_size": size,
            "font_size_hint": size,
            "font_size_min": size,
            "font_size_max": size,
            "font_size_locked": True,
            "font_size_normalization": normalization,
            "font_size_normalization_source": normalization_source,
            "root_visual_font_size_hint": size,
        }
    )
    style["spacing_profile"] = spacing


def _sync_style_arbitration_metadata(style: dict[str, Any], decision: Mapping[str, Any]) -> None:
    style["style_arbitration"] = dict(decision)
    style["style_arbitration_decision"] = decision.get("decision")
    style["style_arbitration_reason_codes"] = list(decision.get("reason_codes") or [])
    style["style_arbitration_cohort_id"] = decision.get("cohort_id")
    style["style_family_bucket"] = decision.get("style_family_bucket")
    style["source_font_weight_dominance"] = _float(decision.get("raw_font_weight_dominance"))
    style["source_font_candidate_count"] = int(decision.get("source_font_candidate_count") or 0)
    style["individual_weight_arbitration_decision"] = decision.get(
        "individual_weight_arbitration_decision"
    )
    style["individual_weight_arbitration_reason_codes"] = list(
        decision.get("individual_weight_arbitration_reason_codes") or []
    )
    style["cohort_font_weight"] = decision.get("cohort_font_weight")
    style["source_weight_peer_stroke_median"] = _float(
        decision.get("source_weight_peer_stroke_median")
    )
    style["source_weight_stroke_ratio"] = _float(decision.get("source_weight_stroke_ratio"))
    for key in (
        "semantic_style_class",
        "style_surface_evidence_source",
        "crop_surface_bucket",
        "raw_label_weight",
        "font_weight_evidence_source",
        "source_glyph_style_evidence_status",
        "source_glyph_style_mask_id",
        "source_glyph_style_quality_status",
        "source_glyph_style_foreground_status",
    ):
        if key in decision:
            style[key] = decision.get(key)
    for key in (
        "style_surface_reason_codes",
        "source_glyph_style_evidence_reason_codes",
    ):
        if key in decision:
            style[key] = list(decision.get(key) or [])
    for key in (
        "crop_luma_median",
        "crop_dark_pixel_ratio",
        "crop_light_pixel_ratio",
        "source_outline_to_text_ratio",
    ):
        if key in decision:
            style[key] = _float(decision.get(key))
    if "measured_glyph_size_reliable" in decision:
        style["measured_glyph_size_reliable"] = bool(
            decision.get("measured_glyph_size_reliable")
        )
    if decision.get("font_size_arbitration_decision"):
        style["font_size_arbitration_decision"] = decision.get("font_size_arbitration_decision")
        style["font_size_arbitration_reason_codes"] = list(
            decision.get("font_size_arbitration_reason_codes") or []
        )


def _cohort_font_size_profile(
    members: Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    canonical_font_family: str = "",
) -> dict[str, Any]:
    raw_hints: list[int] = []
    semantic_style_class = ""
    for bundle, record in members:
        if not semantic_style_class:
            semantic_style_class = str(
                record.get("semantic_style_class") or "dialogue"
            )
        if str(record.get("status") or "") != "applied":
            continue
        if _float(record.get("raw_style_confidence")) < MIN_STYLE_EVIDENCE_CONFIDENCE:
            continue
        detection = record.get("detection") if isinstance(record.get("detection"), Mapping) else {}
        source_orientation = _style_source_orientation(bundle, detection)
        dominant_component_count = int(record.get("measured_glyph_dominant_component_count") or 0)
        hints = _layout_hints_for_bundle(
            bundle,
            detection=detection if detection else None,
            surface_bucket=str(record.get("surface_bucket") or "dark_on_light"),
            semantic_style_class=str(
                record.get("semantic_style_class") or "dialogue"
            ),
            source_orientation=source_orientation,
            measured_glyph_size_px=_float(record.get("measured_glyph_size_px")),
            measured_glyph_size_reliable=bool(
                record.get("measured_glyph_size_reliable")
            ),
            measured_glyph_size_source=str(record.get("measured_glyph_size_source") or ""),
            measured_glyph_dominant_component_count=dominant_component_count,
            detector_crop_width=int(record.get("detector_crop_width") or 0),
            font_family=canonical_font_family,
        )
        raw_hint = int(hints.get("font_size_hint") or 0)
        if raw_hint > 0 and str(hints.get("font_size_source") or "") in {
            "measured_source_glyph_geometry",
            "model_scale_corrected_source_glyph_geometry",
            "model_caption_scale",
        }:
            raw_hints.append(raw_hint)
    minimum_members = (
        MIN_CAPTION_COHORT_SIZE_NORMALIZATION_MEMBERS
        if semantic_style_class == "caption"
        else MIN_COHORT_SIZE_NORMALIZATION_MEMBERS
    )
    if len(raw_hints) < minimum_members:
        return {}
    if semantic_style_class == "caption" and len(raw_hints) == 2:
        low, high = sorted(raw_hints)
        midpoint = max(1.0, (float(low) + float(high)) / 2.0)
        relative_spread = float(high - low) / midpoint
        if relative_spread > MAX_TWO_MEMBER_COHORT_SIZE_SPREAD_RATIO:
            return {}
    canonical = int(round(_median(raw_hints)))
    if canonical <= 0:
        return {}
    minimum = max(12, int(round(canonical * 0.76)))
    maximum = max(canonical, int(round(canonical * 1.10)))
    return {
        "cohort_font_size_hint": canonical,
        "cohort_font_size_min": minimum,
        "cohort_font_size_max": maximum,
        "cohort_font_size_source": "visual_cohort_median_parent_hint",
        "cohort_raw_font_size_hints": sorted(raw_hints),
    }


def _style_size_decision(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    *,
    style_decision: str = "",
    preserved_style: bool = False,
    preserved_size: bool = False,
) -> dict[str, Any]:
    measured = _float(record.get("measured_glyph_size_px"))
    cohort_size = _float(cohort.get("median_measured_glyph_size_px"))
    source = str(record.get("measured_glyph_size_source") or "")
    dominant_components = int(record.get("measured_glyph_dominant_component_count") or 0)
    reasons: list[str] = []
    if measured > 0 and not bool(record.get("measured_glyph_size_reliable")):
        reasons.extend(record.get("source_glyph_style_evidence_reason_codes") or [])
        if not reasons:
            reasons.append("source_glyph_geometry_not_authorized")
        return {
            "arbitrated_glyph_size_px": 0.0,
            "font_size_arbitration_decision": (
                "rejected_unreliable_background_glyph_measurement"
            ),
            "font_size_arbitration_reason_codes": reasons,
        }
    if measured <= 0:
        reasons.append(source or "no_parent_glyph_size")
        if cohort_size > 0:
            return {
                "arbitrated_glyph_size_px": round(float(cohort_size), 3),
                "font_size_arbitration_decision": "cohort_fallback_no_parent_glyph_size",
                "font_size_arbitration_reason_codes": reasons + ["used_visual_cohort_median"],
            }
        return {
            "arbitrated_glyph_size_px": 0.0,
            "font_size_arbitration_decision": "no_usable_visual_size",
            "font_size_arbitration_reason_codes": reasons,
        }
    if cohort_size > 0:
        distance = abs(measured - cohort_size) / max(1.0, cohort_size)
        if preserved_size and measured > cohort_size:
            return {
                "arbitrated_glyph_size_px": round(float(measured), 3),
                "font_size_arbitration_decision": "preserved_distinct_high_side_visual_size",
                "font_size_arbitration_reason_codes": [
                    "strong_high_side_source_glyph_evidence",
                    f"dominant_component_count:{dominant_components}",
                    f"relative_distance:{distance:.3f}",
                ],
            }
        if distance <= 0.12:
            return {
                "arbitrated_glyph_size_px": round(float(cohort_size), 3),
                "font_size_arbitration_decision": "normalized_to_visual_size_cohort",
                "font_size_arbitration_reason_codes": [
                    "parent_size_within_visual_cohort_tolerance",
                    f"relative_distance:{distance:.3f}",
                ],
            }
        if not preserved_style:
            reason = "high_side_glyph_measurement_normalized_to_visual_cohort"
            if measured < cohort_size:
                reason = "low_side_glyph_measurement_less_reliable_than_visual_cohort"
            return {
                "arbitrated_glyph_size_px": round(float(cohort_size), 3),
                "font_size_arbitration_decision": (
                    "normalized_sparse_low_size_to_visual_cohort"
                    if measured < cohort_size and dominant_components < MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER
                    else "normalized_outlier_size_to_visual_cohort"
                ),
                "font_size_arbitration_reason_codes": [
                    reason,
                    f"dominant_component_count:{dominant_components}",
                    f"relative_distance:{distance:.3f}",
                ],
            }
        return {
            "arbitrated_glyph_size_px": round(float(measured), 3),
            "font_size_arbitration_decision": "preserved_distinct_visual_size",
            "font_size_arbitration_reason_codes": [
                "parent_size_outside_visual_cohort_tolerance",
                f"relative_distance:{distance:.3f}",
            ],
        }
    return {
        "arbitrated_glyph_size_px": round(float(measured), 3),
        "font_size_arbitration_decision": "accepted_parent_visual_size",
        "font_size_arbitration_reason_codes": ["no_visual_size_cohort"],
    }


def _normalize_layout_hints_to_visual_cohort(
    layout_hints: Mapping[str, Any],
    *,
    arbitration: Mapping[str, Any],
) -> dict[str, Any]:
    hints = dict(layout_hints or {})
    if not hints:
        return hints
    canonical = int(arbitration.get("cohort_font_size_hint") or 0)
    hint_source = str(hints.get("font_size_source") or "")
    if (
        canonical <= 0
        or bool(arbitration.get("preserved_exception"))
        or bool(arbitration.get("preserved_size_exception"))
        or hint_source not in {
            "measured_source_glyph_geometry",
            "model_scale_corrected_source_glyph_geometry",
            "model_caption_scale",
        }
    ):
        return hints
    raw_hint = int(hints.get("font_size_hint") or 0)
    raw_min = int(hints.get("font_size_min") or 0)
    raw_max = int(hints.get("font_size_max") or 0)
    cohort_min = int(arbitration.get("cohort_font_size_min") or 0)
    cohort_max = int(arbitration.get("cohort_font_size_max") or 0)
    min_size = min(value for value in (raw_min, cohort_min, canonical) if value > 0)
    max_size = max(canonical, raw_max, cohort_max)

    hints["raw_parent_font_size_hint"] = raw_hint
    hints["font_size_hint"] = canonical
    hints["font_size_min"] = min_size
    hints["font_size_max"] = max_size
    hints["font_size_normalization"] = "visual_cohort"
    hints["font_size_normalization_source"] = str(arbitration.get("cohort_font_size_source") or "")

    spacing = dict(hints.get("spacing_profile") or {})
    spacing["raw_parent_font_size_hint"] = raw_hint
    spacing["font_size_hint"] = canonical
    spacing["font_size_min"] = min_size
    spacing["font_size_max"] = max_size
    spacing["font_size_normalization"] = "visual_cohort"
    spacing["font_size_normalization_source"] = str(arbitration.get("cohort_font_size_source") or "")
    hints["spacing_profile"] = spacing
    return hints


def _lock_resolved_parent_font_size(style: dict[str, Any]) -> None:
    """Make the arbitrated parent font size the renderer contract."""

    try:
        resolved = int(style.get("font_size_hint") or style.get("font_size") or 0)
    except Exception:
        resolved = 0
    if resolved <= 0:
        return

    original_min = style.get("font_size_min")
    original_max = style.get("font_size_max")
    style["font_size"] = resolved
    style["font_size_hint"] = resolved
    style["font_size_min"] = resolved
    style["font_size_max"] = resolved
    style["font_size_locked"] = True
    style["font_size_policy"] = "parent_style_authoritative"
    style["font_size_fallback_policy"] = "layout_failure_audit_only"

    spacing = dict(style.get("spacing_profile") or {})
    if original_min not in (None, "", resolved):
        spacing["detected_font_size_min"] = original_min
    if original_max not in (None, "", resolved):
        spacing["detected_font_size_max"] = original_max
    spacing["font_size"] = resolved
    spacing["font_size_hint"] = resolved
    spacing["font_size_min"] = resolved
    spacing["font_size_max"] = resolved
    spacing["font_size_locked"] = True
    spacing["font_size_policy"] = "parent_style_authoritative"
    spacing["font_size_fallback_policy"] = "layout_failure_audit_only"
    style["spacing_profile"] = spacing


def _should_preserve_size_exception(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    active: Sequence[tuple[Any, Mapping[str, Any]]],
) -> bool:
    if str(record.get("status") or "") != "applied":
        return False
    if not bool(record.get("measured_glyph_size_reliable")):
        return False
    measured = _float(record.get("measured_glyph_size_px"))
    cohort_size = _float(cohort.get("median_measured_glyph_size_px"))
    if measured <= 0 or cohort_size <= 0 or measured < cohort_size * 1.22:
        return False
    if _float(record.get("raw_style_confidence")) < STYLE_EXCEPTION_CONFIDENCE:
        return False
    if int(record.get("measured_glyph_dominant_component_count") or 0) < 4:
        return False
    if str(record.get("measured_glyph_size_source") or "") not in {
        "source_glyph_dominant_component_cluster",
        "source_glyph_merged_outline_column_width",
    }:
        return False
    if not _has_preservable_visual_evidence(record):
        return False
    root_id = str(record.get("root_id") or "")
    parent_id = str(record.get("parent_id") or "")
    surface_bucket = str(record.get("surface_bucket") or "")
    if root_id and any(
        str(other.get("root_id") or "") == root_id
        and str(other.get("parent_id") or "") != parent_id
        and str(other.get("surface_bucket") or "") == surface_bucket
        and str(other.get("status") or "") == "applied"
        for _bundle, other in active
    ):
        return False
    return True


def _should_preserve_style_exception(
    record: Mapping[str, Any],
    cohort: Mapping[str, Any],
    active: Sequence[tuple[Any, Mapping[str, Any]]],
) -> bool:
    confidence = _float(record.get("raw_style_confidence"))
    if confidence < STYLE_EXCEPTION_CONFIDENCE:
        return False
    if not _has_preservable_visual_evidence(record):
        return False
    if _has_same_root_style_conflict(record, active):
        return False
    dominance = _float(record.get("raw_serif_dominance"))
    if dominance < 0.72:
        return False
    if _visual_style_distance_from_cohort(record, cohort):
        return True
    return confidence >= STRONG_STYLE_EXCEPTION_CONFIDENCE and str(record.get("color_bucket") or "") != "unknown"


def _has_same_root_style_conflict(
    record: Mapping[str, Any],
    active: Sequence[tuple[Any, Mapping[str, Any]]],
) -> bool:
    root_id = str(record.get("root_id") or "")
    parent_id = str(record.get("parent_id") or "")
    surface_bucket = str(record.get("surface_bucket") or "")
    raw_class = str(record.get("raw_style_class") or "")
    if not root_id or not raw_class:
        return False
    for _bundle, other in active:
        if str(other.get("parent_id") or "") == parent_id:
            continue
        if str(other.get("root_id") or "") != root_id:
            continue
        if str(other.get("surface_bucket") or "") != surface_bucket:
            continue
        if str(other.get("status") or "") != "applied":
            continue
        if _float(other.get("raw_style_confidence")) < 0.30:
            continue
        if str(other.get("raw_style_class") or "") != raw_class:
            return True
    return False


def _visual_style_distance_from_cohort(record: Mapping[str, Any], cohort: Mapping[str, Any]) -> bool:
    size_diff = abs(_float(record.get("raw_text_size_ratio")) - _float(cohort.get("median_text_size_ratio")))
    line_diff = abs(_float(record.get("raw_line_spacing_ratio")) - _float(cohort.get("median_line_spacing_ratio")))
    stroke_diff = abs(_float(record.get("raw_stroke_width_ratio")) - _float(cohort.get("median_stroke_width_ratio")))
    if size_diff >= 0.070 and line_diff >= 0.045:
        return True
    if stroke_diff >= 0.010 and (size_diff >= 0.045 or line_diff >= 0.035):
        return True
    return False


def _has_preservable_visual_evidence(record: Mapping[str, Any]) -> bool:
    bbox = _bbox(record.get("crop_bbox"))
    if not bbox:
        return False
    _x, _y, width, height = bbox
    if min(width, height) < 42 or width * height < 3600:
        return False
    return True


def _style_cohort_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    semantic_style_class = str(
        record.get("semantic_style_class") or "dialogue"
    )
    orientation = str(record.get("geometry_orientation") or "vertical")
    return (
        semantic_style_class,
        orientation,
        _style_family_cohort_bucket(record),
    )


def _style_family_cohort_bucket(record: Mapping[str, Any]) -> str:
    family_bucket = _distinct_font_family_bucket(str(record.get("raw_font_label") or ""))
    confidence = _float(record.get("raw_style_confidence"))
    if family_bucket:
        required_confidence = 0.15 if family_bucket == "decorative_design" else 0.45
        if confidence >= required_confidence:
            return family_bucket
    raw_weight = str(record.get("raw_font_weight") or "regular")
    if confidence >= 0.55 and raw_weight in {"bold", "black"}:
        return f"emphasis_{raw_weight}"
    return "general"


def _distinct_font_family_bucket(label: str) -> str:
    lowered = str(label or "").replace("\\", "/").lower()
    if any(token in lowered for token in ("kyokasho", "kyoukasho")):
        return "textbook"
    if any(token in lowered for token in ("tsukuar", "maru", "rounded")):
        return "rounded_gothic"
    if any(token in lowered for token in ("ryuseki", "koin", "kantei", "popjoy")):
        return "decorative_design"
    return ""


def _crop_surface_evidence(crop: Any | None) -> dict[str, Any]:
    if crop is None:
        return {
            "bucket": "unknown",
            "median_luma": 0.0,
            "dark_pixel_ratio": 0.0,
            "light_pixel_ratio": 0.0,
            "reason_codes": ["crop_unavailable"],
        }
    try:
        import numpy as np

        values = np.asarray(crop.convert("L"), dtype=np.uint8).reshape(-1)
        if values.size <= 0:
            raise ValueError("empty_crop")
        median_luma = float(np.median(values))
        dark_pixel_ratio = float(np.mean(values <= 128))
        light_pixel_ratio = float(np.mean(values >= 192))
    except Exception as exc:
        return {
            "bucket": "unknown",
            "median_luma": 0.0,
            "dark_pixel_ratio": 0.0,
            "light_pixel_ratio": 0.0,
            "reason_codes": [f"crop_luma_failed:{type(exc).__name__}"],
        }

    if (
        median_luma >= CROP_LIGHT_SURFACE_MEDIAN_MIN
        and light_pixel_ratio >= CROP_LIGHT_PIXEL_RATIO_MIN
    ):
        bucket = "dark_on_light"
        reason_codes = ["crop_light_surface_consensus"]
    elif (
        median_luma <= CROP_DARK_SURFACE_MEDIAN_MAX
        and dark_pixel_ratio >= CROP_DARK_PIXEL_RATIO_MIN
    ):
        bucket = "light_on_dark"
        reason_codes = ["crop_dark_surface_consensus"]
    else:
        bucket = "unknown"
        reason_codes = ["crop_surface_ambiguous"]
    return {
        "bucket": bucket,
        "median_luma": round(median_luma, 3),
        "dark_pixel_ratio": round(dark_pixel_ratio, 6),
        "light_pixel_ratio": round(light_pixel_ratio, 6),
        "reason_codes": reason_codes,
    }


def _style_surface_decision(
    *,
    color_bucket: str,
    crop_surface: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    model_bucket = str(color_bucket or "unknown")
    if model_bucket in {"light_on_dark", "dark_on_light"}:
        return model_bucket, "model_contrast", ["unambiguous_model_text_stroke_contrast"]
    crop_bucket = str(crop_surface.get("bucket") or "unknown")
    if crop_bucket in {"light_on_dark", "dark_on_light"}:
        return (
            crop_bucket,
            "crop_luma",
            [str(reason) for reason in crop_surface.get("reason_codes") or []],
        )
    reasons = [str(reason) for reason in crop_surface.get("reason_codes") or []]
    reasons.append("conservative_dark_on_light_default")
    return "dark_on_light", "conservative_default", reasons


def _style_surface_bucket(bundle: Any, *, color_bucket: str) -> str:
    del bundle
    if str(color_bucket or "") == "light_on_dark":
        return "light_on_dark"
    if str(color_bucket or "") == "dark_on_light":
        return "dark_on_light"
    return "dark_on_light"


def _structural_background_surface(bundle: Any) -> bool:
    return _semantic_style_class(bundle) == "caption"


def _semantic_style_class(bundle: Any) -> str:
    values = [
        getattr(bundle, "role", ""),
        getattr(bundle, "semantic_class", ""),
        getattr(bundle, "semantic_kind", ""),
        getattr(bundle, "route_intent", ""),
        getattr(bundle, "cleanup_mode", ""),
    ]
    lowered = " ".join(str(value or "").lower() for value in values)
    if any(token in lowered for token in ("caption", "background", "narration", "sign")):
        return "caption"
    return "dialogue"


def _source_glyph_style_evidence(
    source_glyph_mask_result: Any | None,
    *,
    bundle_id: str,
    parent_id: str,
    semantic_style_class: str,
) -> dict[str, Any]:
    if str(semantic_style_class or "") != "caption":
        return {
            "geometry_reliable": True,
            "status": "not_required_for_dialogue",
            "reason_codes": ["dialogue_crop_geometry_contract"],
        }
    if source_glyph_mask_result is None:
        return {
            "geometry_reliable": False,
            "status": "unavailable",
            "reason_codes": ["source_glyph_mask_result_unavailable"],
        }

    mask = None
    getter = getattr(source_glyph_mask_result, "get", None)
    if callable(getter):
        for identity in (bundle_id, parent_id):
            if not identity:
                continue
            try:
                mask = getter(identity)
            except Exception:
                mask = None
            if mask is not None:
                break
    if mask is None and isinstance(source_glyph_mask_result, Mapping):
        for identity in (bundle_id, parent_id):
            if identity and source_glyph_mask_result.get(identity) is not None:
                mask = source_glyph_mask_result.get(identity)
                break
    if mask is None:
        return {
            "geometry_reliable": False,
            "status": "missing",
            "reason_codes": ["caption_source_glyph_mask_missing"],
        }

    def value(name: str) -> Any:
        if isinstance(mask, Mapping):
            return mask.get(name)
        return getattr(mask, name, None)

    quality_status = str(value("quality_status") or "")
    foreground_status = str(value("route_owned_foreground_contract_status") or "")
    rejection_reason = str(value("erase_mask_rejected_reason") or "")
    mask_id = str(value("mask_id") or value("source_glyph_mask_id") or "")
    foreground_pixels = int(value("foreground_mask_pixels") or 0)
    geometry_reliable = bool(
        quality_status == "usable"
        and foreground_status == "generated"
        and foreground_pixels > 0
        and not rejection_reason
    )
    if geometry_reliable:
        status = "usable_for_style_geometry"
        reason_codes = ["usable_route_owned_caption_foreground_mask"]
    else:
        status = "rejected_for_style_geometry"
        reason_codes = []
        if quality_status and quality_status != "usable":
            reason_codes.append(f"source_glyph_quality:{quality_status}")
        if foreground_status and foreground_status != "generated":
            reason_codes.append(f"foreground_contract:{foreground_status}")
        if rejection_reason:
            reason_codes.append(f"erase_mask_rejected:{rejection_reason}")
        if foreground_pixels <= 0:
            reason_codes.append("foreground_mask_empty")
        if not reason_codes:
            reason_codes.append("caption_source_glyph_contract_not_usable")
    return {
        "geometry_reliable": geometry_reliable,
        "status": status,
        "reason_codes": reason_codes,
        "mask_id": mask_id,
        "quality_status": quality_status,
        "foreground_status": foreground_status,
    }


def _style_class_for_surface(surface_bucket: str) -> str:
    return "caption" if str(surface_bucket or "") == "light_on_dark" else "dialogue"


def _render_colors_for_surface(surface_bucket: str) -> dict[str, str]:
    if str(surface_bucket or "") == "light_on_dark":
        return {"fill_color": "#FFFFFF", "stroke_color": "#000000"}
    return {"fill_color": "#000000", "stroke_color": "#FFFFFF"}


def _geometry_orientation_for_bundle(bundle: Any) -> str:
    bbox = _best_style_bbox(bundle)
    if not bbox:
        return "vertical"
    _x, _y, width, height = bbox
    return "horizontal" if width > height * 1.25 else "vertical"


def _style_source_orientation(bundle: Any, detection: Mapping[str, Any]) -> str:
    geometry = _geometry_orientation_for_bundle(bundle)
    direction = str(detection.get("direction") or "")
    direction_confidence = _float(detection.get("direction_confidence"))
    if direction == "ltr" and geometry == "horizontal" and direction_confidence >= 0.80:
        return "horizontal"
    if direction == "ttb" and direction_confidence >= 0.50:
        return "vertical"
    return geometry or "vertical"


def _style_color_bucket(detection: Mapping[str, Any]) -> str:
    text_luma = _hex_luminance(str(detection.get("text_color") or ""))
    stroke_luma = _hex_luminance(str(detection.get("stroke_color") or ""))
    if text_luma is None or stroke_luma is None:
        return "unknown"
    if abs(text_luma - stroke_luma) < 0.18:
        return "unknown"
    if text_luma >= 0.66 and stroke_luma <= 0.52:
        return "light_on_dark"
    return "dark_on_light"


def _style_class_name(serif: bool, weight: str) -> str:
    family = "serif" if serif else "sans"
    normalized_weight = str(weight or "regular")
    if normalized_weight not in {"regular", "bold", "black"}:
        normalized_weight = "regular"
    return f"{family}_{normalized_weight}"


def _candidate_class_dominance(candidates: Sequence[Mapping[str, Any]], *, serif: bool) -> float:
    if not candidates:
        return 0.0
    total = sum(_float(candidate.get("confidence")) for candidate in candidates)
    if total <= 0:
        return 0.0
    selected = sum(
        _float(candidate.get("confidence"))
        for candidate in candidates
        if bool(candidate.get("serif")) is bool(serif)
    )
    return selected / total


def _candidate_weight_dominance(
    candidates: Sequence[Mapping[str, Any]],
    *,
    weight: str,
    surface_bucket: str,
) -> float:
    if not candidates:
        return 0.0
    total = sum(_float(candidate.get("confidence")) for candidate in candidates)
    if total <= 0:
        return 0.0
    selected = sum(
        _float(candidate.get("confidence"))
        for candidate in candidates
        if _font_weight_from_label(
            str(candidate.get("path") or ""),
            surface_bucket=surface_bucket,
        )
        == str(weight or "regular")
    )
    return selected / total


def _majority_weight(records: Sequence[Mapping[str, Any]], *, default: str) -> str:
    weights = {"regular": 0.0, "bold": 0.0, "black": 0.0}
    for record in records:
        weight = str(record.get("raw_font_weight") or "regular")
        if weight not in weights:
            weight = "regular"
        weights[weight] += max(0.0, _float(record.get("raw_style_confidence")))
    if not any(value > 0 for value in weights.values()):
        return default
    return max(weights, key=lambda key: weights[key])


def _weight_ratio(records: Sequence[Mapping[str, Any]], weight: str) -> float:
    total = sum(max(0.0, _float(record.get("raw_style_confidence"))) for record in records)
    if total <= 0:
        return 0.0
    selected = sum(
        max(0.0, _float(record.get("raw_style_confidence")))
        for record in records
        if str(record.get("raw_font_weight") or "regular") == weight
    )
    return selected / total


def _median(values: Any) -> float:
    numbers = sorted(_float(value) for value in values if _float(value) > 0)
    if not numbers:
        return 0.0
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2.0


def _hex_luminance(value: str) -> float | None:
    raw = str(value or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) != 6:
        return None
    try:
        red = int(raw[0:2], 16) / 255.0
        green = int(raw[2:4], 16) / 255.0
        blue = int(raw[4:6], 16) / 255.0
    except Exception:
        return None
    return red * 0.2126 + green * 0.7152 + blue * 0.0722


def _merge_render_style(bundle: Any, style: Mapping[str, Any]) -> None:
    existing = dict(getattr(bundle, "render_style", {}) or {})
    merged = dict(existing)
    merged.update(dict(style))
    setattr(bundle, "render_style", merged)


def _layout_hints_for_bundle(
    bundle: Any,
    *,
    detection: Mapping[str, Any] | None,
    surface_bucket: str,
    semantic_style_class: str = "dialogue",
    source_orientation: str,
    measured_glyph_size_px: float = 0.0,
    measured_glyph_size_reliable: bool = True,
    cohort_glyph_size_px: float = 0.0,
    font_size_arbitration_decision: str = "",
    measured_glyph_size_source: str = "",
    measured_glyph_dominant_component_count: int = 0,
    detector_crop_width: int = 0,
    font_family: str = "",
) -> dict[str, Any]:
    bbox = _best_style_bbox(bundle)
    if not bbox:
        return {}
    _x, _y, width, height = bbox
    vertical = str(source_orientation or "").strip().lower() != "horizontal"
    contrast_surface = str(surface_bucket or "") == "light_on_dark"
    size_ratio = _clamp_float(
        detection.get("text_size_ratio") if isinstance(detection, Mapping) else 0.0,
        0.0,
        0.80,
    )
    line_spacing_ratio = _clamp_float(
        detection.get("line_spacing_ratio") if isinstance(detection, Mapping) else 0.0,
        0.0,
        0.65,
    )
    model_scale_hint = _model_scale_font_size_hint(
        bbox,
        size_ratio=size_ratio,
        vertical=vertical,
        contrast_surface=contrast_surface,
    )
    caption_model_scale_hint = 0
    if (
        str(semantic_style_class or "") == "caption"
        and not bool(measured_glyph_size_reliable)
    ):
        caption_model_scale_hint = _caption_model_scale_font_size_hint(
            bbox,
            size_ratio=size_ratio,
            detector_crop_width=detector_crop_width,
        )

    glyph_size_basis = (
        _float(measured_glyph_size_px) if bool(measured_glyph_size_reliable) else 0.0
    )
    font_size_source = "measured_source_glyph_geometry"
    if str(font_size_arbitration_decision or "") == "cohort_fallback_no_parent_glyph_size" and glyph_size_basis > 0:
        font_size_source = "cohort_source_glyph_geometry_fallback"
    if glyph_size_basis <= 0 and _float(cohort_glyph_size_px) > 0:
        glyph_size_basis = _float(cohort_glyph_size_px)
        font_size_source = "cohort_source_glyph_geometry_fallback"

    measured_hint = _font_size_hint_from_glyph_pixels(font_family, glyph_size_basis)
    font_size_reason_codes: list[str] = []
    model_scale_correction = ""
    if caption_model_scale_hint > 0:
        hint = caption_model_scale_hint
        readable_min = 18 if contrast_surface else 16
        line_height = max(
            1.10 if contrast_surface else 1.06,
            1.06 + min(line_spacing_ratio, 0.50) * 0.14,
        )
        line_height = _clamp_float(line_height, 1.06, 1.18)
        font_size_source = "model_caption_scale"
        font_size_reason_codes.extend(
            [
                "caption_source_glyph_geometry_not_authorized",
                "yuzumarker_text_size_ratio_reprojected_to_detector_crop_width",
            ]
        )
    elif measured_hint > 0:
        hint = measured_hint
        readable_min = 18 if contrast_surface else 16
        corrected = _model_scale_corrected_font_size_hint(
            measured_hint,
            model_scale_hint,
            measured_glyph_size_source=measured_glyph_size_source,
            measured_glyph_dominant_component_count=measured_glyph_dominant_component_count,
        )
        if corrected:
            hint = int(corrected["font_size_hint"])
            font_size_source = "model_scale_corrected_source_glyph_geometry"
            model_scale_correction = str(corrected["correction"])
            font_size_reason_codes.append(model_scale_correction)
        line_height = max(
            1.10 if contrast_surface else 1.06,
            1.06 + min(line_spacing_ratio, 0.50) * 0.14,
        )
        line_height = _clamp_float(line_height, 1.06, 1.18)
    elif vertical:
        width_factor = 0.40 if not contrast_surface else 0.44
        height_factor = 0.16 if not contrast_surface else 0.18
        geometry_size = min(width * width_factor, height * height_factor)
        if size_ratio > 0:
            ratio_size = min(
                width * (0.30 + min(size_ratio, 0.50) * 0.12),
                height * (0.11 + min(size_ratio, 0.50) * 0.16),
            )
            hint = int(round(geometry_size * 0.78 + ratio_size * 0.22))
            if min(width, height) <= 120 and size_ratio >= 0.25:
                compact_visual_floor = int(round(min(width * 0.36, height * 0.28)))
                hint = max(hint, compact_visual_floor)
        else:
            hint = int(round(geometry_size))
        readable_min = 18 if contrast_surface else 16
        line_height = max(
            1.10 if contrast_surface else 1.06,
            1.06 + min(line_spacing_ratio, 0.50) * 0.14,
        )
        line_height = _clamp_float(line_height, 1.06, 1.18)
        font_size_source = "parent_bbox_yuzumarker_ratio_fallback" if detection else "parent_bbox_geometry_fallback"
    else:
        geometry_size = min(height * 0.72, width * 0.16)
        if size_ratio > 0:
            ratio_size = min(
                height * (0.45 + min(size_ratio, 0.50) * 0.35),
                width * (0.08 + min(size_ratio, 0.45) * 0.16),
            )
            hint = int(round(geometry_size * 0.60 + ratio_size * 0.40))
        else:
            hint = int(round(geometry_size))
        readable_min = 16 if contrast_surface else 15
        line_height = max(
            1.18 if contrast_surface else 1.16,
            1.14 + min(line_spacing_ratio, 0.50) * 0.24,
        )
        line_height = _clamp_float(line_height, 1.14, 1.32)
        font_size_source = "parent_bbox_yuzumarker_ratio_fallback" if detection else "parent_bbox_geometry_fallback"

    if vertical:
        hint_cap = 58 if contrast_surface else 56
    else:
        hint_cap = 64 if contrast_surface else 58
    hint = max(readable_min, min(hint_cap, int(hint or 0)))
    min_size = max(12, min(hint, int(round(hint * 0.86))))
    max_size = max(hint, min(hint_cap, int(round(hint * 1.08))))
    return {
        "font_size_hint": hint,
        "font_size_min": min_size,
        "font_size_max": max_size,
        "font_size_source": font_size_source,
        "font_size_reason_codes": font_size_reason_codes,
        "model_scale_font_size_hint": model_scale_hint,
        "model_caption_scale_font_size_hint": caption_model_scale_hint,
        "detector_crop_width": int(detector_crop_width or width),
        "model_scale_font_size_correction": model_scale_correction,
        "line_height": round(float(line_height), 3),
        "spacing_profile": {
            "source": "yuzumarker" if detection else "parent_geometry_fallback",
            "orientation": "vertical" if vertical else "horizontal",
            "surface_bucket": str(surface_bucket or "dark_on_light"),
            "font_size_hint": hint,
            "font_size_min": min_size,
            "font_size_max": max_size,
            "font_size_source": font_size_source,
            "font_size_reason_codes": list(font_size_reason_codes),
            "model_scale_font_size_hint": model_scale_hint,
            "model_caption_scale_font_size_hint": caption_model_scale_hint,
            "detector_crop_width": int(detector_crop_width or width),
            "model_scale_font_size_correction": model_scale_correction,
            "line_height": round(float(line_height), 3),
            "minimum_readable_font_size": readable_min,
            "source_text_size_ratio": round(float(size_ratio), 4),
            "source_line_spacing_ratio": round(float(line_spacing_ratio), 4),
            "measured_glyph_size_px": round(float(measured_glyph_size_px), 3),
            "cohort_glyph_size_px": round(float(cohort_glyph_size_px), 3),
            "glyph_size_basis_px": round(float(glyph_size_basis), 3),
        },
    }


def _model_scale_font_size_hint(
    bbox: Sequence[int] | None,
    *,
    size_ratio: float,
    vertical: bool,
    contrast_surface: bool,
) -> int:
    bbox_values = _bbox(bbox)
    if not bbox_values or size_ratio <= 0:
        return 0
    _x, _y, width, height = bbox_values
    short_side = float(max(1, min(width, height)))
    ratio = _clamp_float(size_ratio, 0.0, 0.80)
    if vertical:
        short_side_hint = short_side * ratio * (0.88 if not contrast_surface else 0.86)
        height_hint = float(height) * (0.075 + min(ratio, 0.50) * 0.12)
        compact_hint = 0.0
        if short_side <= 120 and ratio >= 0.25:
            compact_hint = short_side * ratio * (0.96 if not contrast_surface else 0.90)
        return int(round(max(short_side_hint, height_hint, compact_hint)))

    short_side_hint = float(height) * ratio * (0.92 if not contrast_surface else 0.88)
    width_hint = float(width) * (0.055 + min(ratio, 0.50) * 0.12)
    return int(round(max(short_side_hint, width_hint)))


def _caption_model_scale_font_size_hint(
    bbox: Sequence[int] | None,
    *,
    size_ratio: float,
    detector_crop_width: int = 0,
) -> int:
    bbox_values = _bbox(bbox)
    if not bbox_values or size_ratio <= 0:
        return 0
    _x, _y, width, _height = bbox_values
    ratio = _clamp_float(size_ratio, 0.0, 0.80)
    # YuzuMarker trains this regression target as ``text_size / image_width``.
    # The detector resizes the parent crop to the model input width, so mapping
    # the prediction back to page pixels uses the original crop width directly.
    source_width = max(1, int(detector_crop_width or width))
    return int(round(float(source_width) * ratio))


def _model_scale_corrected_font_size_hint(
    measured_hint: int,
    model_scale_hint: int,
    *,
    measured_glyph_size_source: str,
    measured_glyph_dominant_component_count: int,
) -> dict[str, Any]:
    measured = int(measured_hint or 0)
    model_hint = int(model_scale_hint or 0)
    if measured <= 0 or model_hint <= 0:
        return {}
    source = str(measured_glyph_size_source or "")
    dominant_count = int(measured_glyph_dominant_component_count or 0)
    if source == "source_glyph_merged_outline_column_width" and model_hint < measured * 0.96:
        return {
            "font_size_hint": max(1, model_hint),
            "correction": "merged_outline_width_capped_by_model_scale",
        }
    if (
        source == "source_glyph_dominant_component_cluster"
        and dominant_count <= 1
        and model_hint < measured * 0.65
        and measured - model_hint >= 24
    ):
        return {
            "font_size_hint": max(1, model_hint),
            "correction": "single_component_measurement_capped_by_model_scale",
        }
    if (
        source == "source_glyph_dominant_component_cluster"
        and 0 < dominant_count < MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER
        and model_hint > measured * 1.12
    ):
        bounded_hint = min(model_hint, max(measured + 1, int(round(measured * 1.15))))
        return {
            "font_size_hint": bounded_hint,
            "correction": "sparse_source_glyph_measurement_bounded_by_model_scale",
        }
    return {}


def _source_glyph_size_metrics(crop: Any | None, *, surface_bucket: str) -> dict[str, Any]:
    if crop is None:
        return {}
    try:
        import cv2
        import numpy as np
        from PIL import ImageOps

        gray = ImageOps.grayscale(crop)
        arr = np.asarray(gray)
        if arr.size == 0:
            return {}
        if str(surface_bucket or "") == "light_on_dark":
            mask = (arr > 175).astype("uint8")
        else:
            mask = (arr < 135).astype("uint8")
        height, width = mask.shape[:2]
        if height <= 0 or width <= 0:
            return {}
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        kept = []
        for index in range(1, count):
            x, y, comp_w, comp_h, area = [int(value) for value in stats[index]]
            if area < 4:
                continue
            touches_edge = x <= 1 or y <= 1 or x + comp_w >= width - 1 or y + comp_h >= height - 1
            if touches_edge:
                continue
            if comp_w > max(12, width * 0.55) or comp_h > max(12, height * 0.55):
                continue
            if comp_w / max(1, comp_h) > 6.0 or comp_h / max(1, comp_w) > 8.0:
                continue
            kept.append((x, y, comp_w, comp_h, area))
        if not kept:
            return {}
        glyph_like = [
            (x, y, comp_w, comp_h, area)
            for x, y, comp_w, comp_h, area in kept
            if max(comp_w, comp_h) >= 12 and max(comp_w, comp_h) / max(1, min(comp_w, comp_h)) <= 2.6
        ]
        elongated = [
            (x, y, comp_w, comp_h, area)
            for x, y, comp_w, comp_h, area in kept
            if max(comp_w, comp_h) >= 12 and max(comp_w, comp_h) / max(1, min(comp_w, comp_h)) > 2.6
        ]
        if str(surface_bucket or "") == "light_on_dark":
            merged_column_widths = sorted(
                min(comp_w, comp_h)
                for _x, _y, comp_w, comp_h, area in elongated
                if min(comp_w, comp_h) >= 18 and area >= 120
            )
            if len(merged_column_widths) >= 2:
                glyph_like_dims = sorted(max(comp_w, comp_h) for _x, _y, comp_w, comp_h, _area in glyph_like)
                merged_size = _median(merged_column_widths)
                glyph_like_size = _median(glyph_like_dims)
                if not glyph_like_dims or merged_size >= glyph_like_size * 1.30:
                    xs = [x for x, _y, _w, _h, _area in elongated]
                    ys = [y for _x, y, _w, _h, _area in elongated]
                    x2s = [x + comp_w for x, _y, comp_w, _h, _area in elongated]
                    y2s = [y + comp_h for _x, y, _w, comp_h, _area in elongated]
                    glyph_bbox = [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]
                    return {
                        "glyph_size_px": round(float(merged_size), 3),
                        "glyph_bbox": [int(value) for value in glyph_bbox],
                        "component_count": len(kept),
                        "dominant_component_count": len(merged_column_widths),
                        "source": "source_glyph_merged_outline_column_width",
                        **_source_visual_column_metrics(
                            elongated,
                            glyph_size_px=merged_size,
                        ),
                    }
        component_dims = sorted(max(comp_w, comp_h) for _x, _y, comp_w, comp_h, _area in glyph_like)
        if not component_dims:
            small_dims = [max(comp_w, comp_h) for _x, _y, comp_w, comp_h, _area in kept]
            if small_dims and max(small_dims) < 12:
                return {
                    "glyph_bbox": [],
                    "component_count": len(kept),
                    "source": "source_glyph_components_too_small_for_font_size",
                }
            return {}
        max_dim = max(component_dims)
        if max_dim < 12:
            return {
                "glyph_bbox": [],
                "component_count": len(kept),
                "source": "source_glyph_components_too_small_for_font_size",
            }
        dominant_floor = max(12.0, float(max_dim) * 0.55)
        dominant = [
            (x, y, comp_w, comp_h, area)
            for x, y, comp_w, comp_h, area in glyph_like
            if max(comp_w, comp_h) >= dominant_floor
        ]
        if not dominant:
            return {
                "glyph_bbox": [],
                "component_count": len(kept),
                "source": "source_glyph_no_dominant_components",
            }
        dominant_dims = sorted(max(comp_w, comp_h) for _x, _y, comp_w, comp_h, _area in dominant)
        if len(dominant_dims) <= 3:
            glyph_size = max(dominant_dims)
        else:
            glyph_size = _percentile(dominant_dims, 0.85)
        if glyph_size <= 0:
            return {}
        xs = [x for x, _y, _w, _h, _area in dominant]
        ys = [y for _x, y, _w, _h, _area in dominant]
        x2s = [x + comp_w for x, _y, comp_w, _h, _area in dominant]
        y2s = [y + comp_h for _x, y, _w, comp_h, _area in dominant]
        glyph_bbox = [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]
        return {
            "glyph_size_px": round(float(glyph_size), 3),
            "glyph_bbox": [int(value) for value in glyph_bbox],
            "component_count": len(kept),
            "dominant_component_count": len(dominant),
            "source": "source_glyph_dominant_component_cluster",
            **_source_visual_column_metrics(
                dominant,
                glyph_size_px=glyph_size,
            ),
        }
    except Exception:
        return {}


def _source_visual_column_metrics(
    components: Sequence[Sequence[int]],
    *,
    glyph_size_px: float,
) -> dict[str, Any]:
    values = [tuple(int(value) for value in component[:5]) for component in components or []]
    if (
        len(values) < MIN_DOMINANT_COMPONENTS_FOR_VISUAL_COLUMNS
        or float(glyph_size_px) <= 0
    ):
        return {}

    centers = sorted(
        float(x) + float(width) / 2.0
        for x, _y, width, _height, _area in values
    )
    gap = max(8.0, float(glyph_size_px) * VISUAL_COLUMN_CLUSTER_GAP_RATIO)
    clusters: list[list[float]] = []
    for center in centers:
        if not clusters:
            clusters.append([center])
            continue
        cluster_center = sum(clusters[-1]) / float(len(clusters[-1]))
        if center - cluster_center > gap:
            clusters.append([center])
        else:
            clusters[-1].append(center)

    supported = [
        cluster
        for cluster in clusters
        if len(cluster) >= MIN_COMPONENTS_PER_VISUAL_COLUMN
    ]
    represented = sum(len(cluster) for cluster in supported)
    minimum_represented = max(
        MIN_DOMINANT_COMPONENTS_FOR_VISUAL_COLUMNS,
        int(math.ceil(len(values) * 0.70)),
    )
    if not supported or represented < minimum_represented:
        return {}

    return {
        "visual_column_count": len(supported),
        "visual_column_source": "source_glyph_component_x_clusters",
        "visual_column_component_counts": [len(cluster) for cluster in supported],
        "visual_column_centers": [
            round(sum(cluster) / float(len(cluster)), 3)
            for cluster in supported
        ],
    }


def _font_size_hint_from_glyph_pixels(font_family: str, glyph_size_px: Any) -> int:
    target = _float(glyph_size_px)
    if target <= 0:
        return 0
    measured = _font_point_size_for_pixel_height(font_family, target)
    if measured > 0:
        return measured
    return int(round(target * 1.28))


def _font_point_size_for_pixel_height(font_family: str, target_px: float) -> int:
    path = str(font_family or "").strip()
    if not path or not os.path.isfile(path):
        return 0
    try:
        from PIL import Image, ImageDraw, ImageFont

        best_size = 0
        best_delta = float("inf")
        for size in range(8, 73):
            font = ImageFont.truetype(path, size=size)
            canvas = Image.new("L", (size * 3, size * 3), 0)
            draw = ImageDraw.Draw(canvas)
            bbox = draw.textbbox((0, 0), "测", font=font)
            glyph_height = max(1, int(bbox[3] - bbox[1]))
            delta = abs(float(glyph_height) - float(target_px))
            if delta < best_delta:
                best_delta = delta
                best_size = size
        return int(best_size)
    except Exception:
        return 0


def _percentile(values: Sequence[float], ratio: float) -> float:
    numbers = sorted(float(value) for value in values if float(value) > 0)
    if not numbers:
        return 0.0
    if len(numbers) == 1:
        return numbers[0]
    index = max(0.0, min(1.0, float(ratio))) * (len(numbers) - 1)
    low = int(index)
    high = min(len(numbers) - 1, low + 1)
    if low == high:
        return numbers[low]
    fraction = index - low
    return numbers[low] * (1.0 - fraction) + numbers[high] * fraction


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(float(low), min(float(high), number))


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
        if "CUDAExecutionProvider" not in available:
            fallback_reason = "cuda_execution_provider_not_available"
        else:
            fallback_reason = "cuda_execution_provider_initialization_failed"

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


def _onnx_session_provider_metadata(
    model_path: str,
    *,
    use_gpu: bool,
    session: Any,
) -> dict[str, Any]:
    key = (os.path.abspath(model_path), bool(use_gpu))
    metadata = dict(_SESSION_PROVIDER_METADATA.get(key) or {})
    if metadata:
        return metadata
    active = [str(provider) for provider in session.get_providers()]
    requested = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    fallback_reason = ""
    if use_gpu and "CUDAExecutionProvider" not in active:
        fallback_reason = "cuda_execution_provider_initialization_failed"
    return {
        "gpu_requested": bool(use_gpu),
        "requested_execution_provider": requested,
        "available_execution_providers": [],
        "active_execution_providers": active,
        "primary_execution_provider": active[0] if active else "",
        "provider_fallback_reason": fallback_reason,
        "provider_preload_error": "",
    }


def _copy_provider_metadata_to_result(
    result: ParentFontDetectionRunResult,
    detector: Any,
) -> None:
    result.gpu_requested = bool(getattr(detector, "gpu_requested", False))
    result.requested_execution_provider = str(
        getattr(detector, "requested_execution_provider", "") or ""
    )
    result.available_execution_providers = list(
        getattr(detector, "available_execution_providers", []) or []
    )
    result.active_execution_providers = list(
        getattr(detector, "active_execution_providers", []) or []
    )
    result.primary_execution_provider = str(
        getattr(detector, "primary_execution_provider", "") or ""
    )
    result.provider_fallback_reason = str(
        getattr(detector, "provider_fallback_reason", "") or ""
    )
    result.provider_preload_error = str(
        getattr(detector, "provider_preload_error", "") or ""
    )


def _load_font_labels(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise RuntimeError("YuzuMarker font labels must be a list")
    return [dict(item) if isinstance(item, Mapping) else {"path": str(item)} for item in labels]


def _label_at(labels: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    if 0 <= index < len(labels):
        return labels[index]
    return {}


def _softmax(values: Any) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(arr.max())
    exp = np.exp(arr)
    denom = float(exp.sum())
    if denom <= 0:
        return np.zeros_like(arr)
    return exp / denom


def _rgb_from_unit_values(values: Any) -> str:
    try:
        raw_values = list(values)
    except Exception:
        raw_values = []
    vals = [_float(value) for value in raw_values[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    channels = [max(0, min(255, int(round(value * 255.0)))) for value in vals]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def _heuristic_detection(image: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(image.convert("L"), dtype=np.float32)
    mean = float(arr.mean()) if arr.size else 255.0
    dark_ratio = float((arr < 96).mean()) if arr.size else 0.0
    light_on_dark = mean < 120.0
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


def _best_style_bbox(bundle: Any) -> list[int]:
    for attr in (
        "source_contract_bbox",
        "parent_bbox",
        "render_allowed_area",
        "cleanup_target_bbox",
        "root_bbox",
    ):
        bbox = _bbox(getattr(bundle, attr, None))
        if bbox:
            return bbox
    execution_region = getattr(bundle, "execution_region", {}) or {}
    if isinstance(execution_region, Mapping):
        for key in ("source_contract_bbox", "bbox", "render_allowed_area"):
            bbox = _bbox(execution_region.get(key))
            if bbox:
                return bbox
    return []


def _crop_image(image: Any | None, bbox: Sequence[int]) -> Any | None:
    if image is None:
        return None
    box = _bbox(bbox)
    if not box:
        return None
    x, y, w, h = box
    pad = max(2, int(round(min(w, h) * 0.04)))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(int(image.width), x + w + pad)
    bottom = min(int(image.height), y + h + pad)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return []
    try:
        x, y, w, h = [int(round(float(value[index]))) for index in range(4)]
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    return [x, y, w, h]


def _font_weight_from_label(label: str, *, surface_bucket: str = "") -> str:
    del surface_bucket
    lowered = str(label or "").lower()
    if any(token in lowered for token in ("black", "heavy", "ultra", "w9", "w10", "w12", "w14")):
        return "black"
    if any(
        token in lowered
        for token in (
            "bold",
            "semibold",
            "demibold",
            "-b.",
            "_b.",
            "-b.otf",
            "-b.ttf",
            "hei",
        )
    ):
        return "bold"
    return "regular"


def _font_weight_from_visual_evidence(
    label_weight: str,
    *,
    detection: Mapping[str, Any],
    surface_bucket: str,
) -> tuple[str, str, float]:
    text_size_ratio = _float(detection.get("text_size_ratio"))
    stroke_width_ratio = _float(detection.get("stroke_width_ratio"))
    outline_to_text_ratio = (
        stroke_width_ratio / text_size_ratio if text_size_ratio > 0 else 0.0
    )
    normalized_label_weight = str(label_weight or "regular")
    if normalized_label_weight in {"bold", "black"}:
        return (
            normalized_label_weight,
            "font_label",
            round(float(outline_to_text_ratio), 6),
        )
    if (
        str(surface_bucket or "") == "light_on_dark"
        and outline_to_text_ratio >= MIN_OUTLINE_TO_TEXT_RATIO_FOR_BOLD
    ):
        return "bold", "source_outline_ratio", round(float(outline_to_text_ratio), 6)
    return "regular", "font_label_regular", round(float(outline_to_text_ratio), 6)


def _compact_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, Mapping):
            continue
        candidates.append(
            {
                "index": item.get("index"),
                "confidence": _float(item.get("confidence")),
                "path": str(item.get("path") or ""),
                "language": str(item.get("language") or ""),
                "serif": bool(item.get("serif")),
            }
        )
    return candidates


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
