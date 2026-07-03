# -*- coding: utf-8 -*-
"""Parent-owned font and style detection for execution bundles."""
from __future__ import annotations

import json
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
MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER = 6


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

    evidence_records: list[dict[str, Any]] = []
    for bundle in bundles:
        record = _collect_style_evidence_for_bundle(
            bundle,
            image=image,
            mode=normalized_mode,
            detector=active_detector,
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
    surface_bucket = _style_surface_bucket(bundle, color_bucket=color_bucket)
    raw_weight = _font_weight_from_label(raw_label, surface_bucket=surface_bucket)
    top_candidates = _compact_candidates(raw_detection.get("top_candidates")) if raw_detection else []
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
    glyph_metrics = _source_glyph_size_metrics(crop, surface_bucket=surface_bucket)
    record = {
        "bundle_id": bundle_id,
        "parent_id": parent_id,
        "root_id": root_id,
        "status": status,
        "fallback_reason": reason,
        "crop_bbox": list(bbox) if bbox else [],
        "surface_bucket": surface_bucket,
        "geometry_orientation": _geometry_orientation_for_bundle(bundle),
        "color_bucket": color_bucket,
        "raw_style_class": _style_class_name(raw_serif, raw_weight),
        "raw_font_weight": raw_weight,
        "raw_font_serif": raw_serif,
        "raw_font_label": raw_label,
        "raw_font_language": str(raw_detection.get("font_language") or ""),
        "raw_direction": str(raw_detection.get("direction") or ""),
        "raw_direction_confidence": _float(raw_detection.get("direction_confidence")),
        "raw_style_confidence": _float(raw_detection.get("confidence")) if raw_detection else 0.0,
        "raw_serif_dominance": _candidate_class_dominance(top_candidates, serif=raw_serif),
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
    for bundle, record in active:
        cohort = cohort_profiles.get(_style_cohort_key(record), {})
        decision = _style_decision_for_record(record, cohort, active)
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
                "style_arbitration_decision": decision.get("decision"),
                "style_arbitration_reason_codes": decision.get("reason_codes"),
                "style_arbitration_cohort_id": decision.get("cohort_id"),
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
        "style_class": final_style_class or existing.get("style_class") or _style_class_for_surface(surface_bucket),
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
        "style_arbitration": dict(arbitration),
        "style_arbitration_decision": arbitration.get("decision"),
        "style_arbitration_reason_codes": list(arbitration.get("reason_codes") or []),
        "style_arbitration_cohort_id": arbitration.get("cohort_id"),
        "style_arbitration_provider": STYLE_ARBITRATOR_PROVIDER,
    }
    layout_hints = _layout_hints_for_bundle(
        bundle,
        detection=trusted_detection if trusted_detection else None,
        surface_bucket=surface_bucket,
        source_orientation=str(style.get("source_orientation") or ""),
        measured_glyph_size_px=_float(
            arbitration.get("arbitrated_glyph_size_px") or arbitration.get("measured_glyph_size_px")
        ),
        cohort_glyph_size_px=_float(arbitration.get("cohort_measured_glyph_size_px")),
        font_size_arbitration_decision=str(arbitration.get("font_size_arbitration_decision") or ""),
        measured_glyph_size_source=str(arbitration.get("measured_glyph_size_source") or ""),
        measured_glyph_dominant_component_count=int(
            arbitration.get("measured_glyph_dominant_component_count") or 0
        ),
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
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[Any, Mapping[str, Any]]]] = {}
    for bundle, record in active:
        grouped.setdefault(_style_cohort_key(record), []).append((bundle, record))

    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for key, members in grouped.items():
        surface_bucket, orientation = key
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
        canonical_weight = _majority_weight(
            usable,
            default="bold" if surface_bucket == "light_on_dark" else "regular",
        )
        if canonical_weight in {"bold", "black"} and _weight_ratio(usable, canonical_weight) < 0.70:
            canonical_weight = "bold" if surface_bucket == "light_on_dark" else "regular"
        if surface_bucket == "light_on_dark" and canonical_weight == "regular":
            canonical_weight = "bold"

        style_class = _style_class_for_surface(surface_bucket)
        cohort_id = "style_cohort:{}:{}".format(surface_bucket, orientation)
        canonical_font_family = (
            resolve_noto_cjk_sc_font_file(base_dir=models_dir, serif=canonical_serif, weight=canonical_weight)
            or default_font_name
        )
        font_size_profile = _cohort_font_size_profile(
            members,
            surface_bucket=surface_bucket,
            canonical_font_family=canonical_font_family,
        )
        profiles[key] = {
            "cohort_id": cohort_id,
            "member_parent_ids": [str(record.get("parent_id") or "") for record in records if record.get("parent_id")],
            "surface_bucket": surface_bucket,
            "orientation": orientation,
            "serif": canonical_serif,
            "weight": canonical_weight,
            "style_class": style_class,
            "canonical_style_class": _style_class_name(canonical_serif, canonical_weight),
            "median_text_size_ratio": _median(_float(record.get("raw_text_size_ratio")) for record in usable),
            "median_line_spacing_ratio": _median(_float(record.get("raw_line_spacing_ratio")) for record in usable),
            "median_stroke_width_ratio": _median(_float(record.get("raw_stroke_width_ratio")) for record in usable),
            "median_measured_glyph_size_px": _median(_float(record.get("measured_glyph_size_px")) for record in usable),
            "usable_evidence_count": len(usable),
            "serif_ratio": round(float(serif_ratio), 4),
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
    reason_codes: list[str] = []

    if status != "applied":
        reason_codes.append(str(record.get("fallback_reason") or "no_usable_style_evidence"))
        return _style_decision(
            record,
            cohort,
            surface_bucket=surface_bucket,
            serif=cohort_serif,
            weight=cohort_weight,
            style_class=str(cohort.get("style_class") or _style_class_for_surface(surface_bucket)),
            decision="fallback_to_parent_style_default",
            reason_codes=reason_codes,
            preserved=False,
        )

    if raw_class == cohort_class:
        reason_codes.append("matches_visual_cohort")
        return _style_decision(
            record,
            cohort,
            surface_bucket=surface_bucket,
            serif=cohort_serif,
            weight=cohort_weight,
            style_class=str(cohort.get("style_class") or _style_class_for_surface(surface_bucket)),
            decision="accepted_visual_cohort_style",
            reason_codes=reason_codes,
            preserved=False,
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
            style_class=str(cohort.get("style_class") or _style_class_for_surface(surface_bucket)),
            decision="preserved_distinct_visual_style",
            reason_codes=reason_codes,
            preserved=True,
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
        style_class=str(cohort.get("style_class") or _style_class_for_surface(surface_bucket)),
        decision="normalized_to_visual_cohort",
        reason_codes=reason_codes,
        preserved=False,
    )


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
) -> dict[str, Any]:
    return {
        "decision": decision,
        "reason_codes": [str(reason) for reason in reason_codes if reason],
        "cohort_id": str(cohort.get("cohort_id") or ""),
        "cohort_parent_ids": list(cohort.get("member_parent_ids") or []),
        "cohort_size": len(list(cohort.get("member_parent_ids") or [])),
        "raw_style_class": str(record.get("raw_style_class") or ""),
        "canonical_style_class": str(cohort.get("canonical_style_class") or ""),
        "surface_bucket": str(surface_bucket or cohort.get("surface_bucket") or "dark_on_light"),
        "serif": bool(serif),
        "weight": str(weight or "regular"),
        "style_class": str(style_class or _style_class_for_surface(surface_bucket)),
        "preserved_exception": bool(preserved),
        "raw_style_confidence": _float(record.get("raw_style_confidence")),
        "measured_glyph_size_px": _float(record.get("measured_glyph_size_px")),
        "measured_glyph_bbox": list(record.get("measured_glyph_bbox") or []),
        "measured_glyph_component_count": int(record.get("measured_glyph_component_count") or 0),
        "measured_glyph_dominant_component_count": int(record.get("measured_glyph_dominant_component_count") or 0),
        "measured_glyph_size_source": str(record.get("measured_glyph_size_source") or ""),
        "cohort_measured_glyph_size_px": _float(cohort.get("median_measured_glyph_size_px")),
        **_style_size_decision(
            record,
            cohort,
            style_decision=decision,
            preserved_style=preserved,
        ),
        "cohort_font_size_hint": int(cohort.get("cohort_font_size_hint") or 0),
        "cohort_font_size_min": int(cohort.get("cohort_font_size_min") or 0),
        "cohort_font_size_max": int(cohort.get("cohort_font_size_max") or 0),
        "cohort_font_size_source": str(cohort.get("cohort_font_size_source") or ""),
        "cohort_raw_font_size_hints": list(cohort.get("cohort_raw_font_size_hints") or []),
    }


def _cohort_font_size_profile(
    members: Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    surface_bucket: str,
    canonical_font_family: str = "",
) -> dict[str, Any]:
    raw_hints: list[int] = []
    for bundle, record in members:
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
            surface_bucket=surface_bucket,
            source_orientation=source_orientation,
            measured_glyph_size_px=_float(record.get("measured_glyph_size_px")),
            measured_glyph_size_source=str(record.get("measured_glyph_size_source") or ""),
            measured_glyph_dominant_component_count=dominant_component_count,
            font_family=canonical_font_family,
        )
        raw_hint = int(hints.get("font_size_hint") or 0)
        if raw_hint > 0 and str(hints.get("font_size_source") or "") in {
            "measured_source_glyph_geometry",
            "model_scale_corrected_source_glyph_geometry",
        }:
            raw_hints.append(raw_hint)
    if len(raw_hints) < MIN_COHORT_SIZE_NORMALIZATION_MEMBERS:
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
) -> dict[str, Any]:
    measured = _float(record.get("measured_glyph_size_px"))
    cohort_size = _float(cohort.get("median_measured_glyph_size_px"))
    source = str(record.get("measured_glyph_size_source") or "")
    dominant_components = int(record.get("measured_glyph_dominant_component_count") or 0)
    reasons: list[str] = []
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
        or hint_source not in {"measured_source_glyph_geometry", "model_scale_corrected_source_glyph_geometry"}
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


def _style_cohort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    surface_bucket = str(record.get("surface_bucket") or "dark_on_light")
    orientation = str(record.get("geometry_orientation") or "vertical")
    return (surface_bucket, orientation)


def _style_surface_bucket(bundle: Any, *, color_bucket: str) -> str:
    if str(color_bucket or "") == "light_on_dark":
        return "light_on_dark"
    if str(color_bucket or "") == "dark_on_light":
        return "dark_on_light"
    if _structural_background_surface(bundle):
        return "light_on_dark"
    return "dark_on_light"


def _structural_background_surface(bundle: Any) -> bool:
    values = [
        getattr(bundle, "role", ""),
        getattr(bundle, "semantic_class", ""),
        getattr(bundle, "semantic_kind", ""),
        getattr(bundle, "route_intent", ""),
        getattr(bundle, "cleanup_mode", ""),
    ]
    lowered = " ".join(str(value or "").lower() for value in values)
    return any(token in lowered for token in ("caption", "background", "narration", "sign"))


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
    source_orientation: str,
    measured_glyph_size_px: float = 0.0,
    cohort_glyph_size_px: float = 0.0,
    font_size_arbitration_decision: str = "",
    measured_glyph_size_source: str = "",
    measured_glyph_dominant_component_count: int = 0,
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

    glyph_size_basis = _float(measured_glyph_size_px)
    font_size_source = "measured_source_glyph_geometry"
    if str(font_size_arbitration_decision or "") == "cohort_fallback_no_parent_glyph_size" and glyph_size_basis > 0:
        font_size_source = "cohort_source_glyph_geometry_fallback"
    if glyph_size_basis <= 0 and _float(cohort_glyph_size_px) > 0:
        glyph_size_basis = _float(cohort_glyph_size_px)
        font_size_source = "cohort_source_glyph_geometry_fallback"

    measured_hint = _font_size_hint_from_glyph_pixels(font_family, glyph_size_basis)
    font_size_reason_codes: list[str] = []
    model_scale_correction = ""
    if measured_hint > 0:
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
        and 0 < dominant_count < MIN_DOMINANT_COMPONENTS_FOR_LOW_SIZE_OUTLIER
        and model_hint > measured * 1.12
    ):
        return {
            "font_size_hint": model_hint,
            "correction": "sparse_source_glyph_measurement_corrected_by_model_scale",
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
        }
    except Exception:
        return {}


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

    providers = ["CPUExecutionProvider"]
    if use_gpu:
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    _SESSION_CACHE[key] = session
    return session


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


def _font_weight_from_label(label: str, *, surface_bucket: str) -> str:
    lowered = str(label or "").lower()
    if any(token in lowered for token in ("black", "heavy", "ultra", "w9", "w10", "w12", "w14")):
        return "black"
    if any(token in lowered for token in ("bold", "semibold", "demibold", "-b.", "_b.", "hei")):
        return "bold"
    return "bold" if str(surface_bucket or "") == "light_on_dark" else "regular"


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
