# -*- coding: utf-8 -*-
"""Pure source/container oriented-layout geometry contracts."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


ORIENTED_LAYOUT_FRAME_VERSION = "oriented_layout_frame_v1"
ORIENTATION_NORMALIZED_ROTATION_VERSION = (
    "orientation_normalized_rotation_v1"
)
ORIENTATION_NORMALIZED_ROTATION_MIN_ABS_DEGREES = 8.0
ORIENTATION_NORMALIZED_ROTATION_MAX_ABS_DEGREES = 40.0
ORIENTATION_NORMALIZED_ROTATION_MAX_EROSION_DELTA_DEGREES = 3.0
ORIENTATION_NORMALIZED_ROTATION_MAX_FRAME_AGREEMENT_DEGREES = 4.0
ORIENTATION_NORMALIZED_ROTATION_MIN_ORIENTATION_CONFIDENCE = 0.90
ORIENTED_LAYOUT_MAX_POLYGON_VERTICES = 4096


def normalize_axis_degrees(value: Any) -> float:
    """Normalize an unoriented line axis to ``(-90, 90]`` degrees."""

    number = _finite_number(value)
    if number is None:
        raise ValueError("oriented axis angle must be finite")
    while number <= -90.0:
        number += 180.0
    while number > 90.0:
        number -= 180.0
    return float(number)


def shortest_axis_delta(first: Any, second: Any) -> float:
    return normalize_axis_degrees(float(first) - float(second))


def source_axis_residual(
    absolute_degrees_clockwise: Any,
    writing_mode: str,
) -> float:
    mode = str(writing_mode or "").strip().lower()
    if mode == "vertical":
        base = 90.0
    elif mode == "horizontal":
        base = 0.0
    else:
        raise ValueError("source writing mode is not resolved")
    return shortest_axis_delta(absolute_degrees_clockwise, base)


def corroborated_rotation_residual(
    *,
    source_axis_support: Any,
    source_reason_codes: Any,
    oriented_frame: Any,
    writing_mode: str,
    orientation_confidence: Any,
) -> dict[str, Any]:
    """Resolve one source-axis residual only with an exact agreeing frame.

    The source observer intentionally publishes an absolute, unoriented line
    axis.  ParentStyleArbitrator supplies the independently resolved source
    writing mode; TextAreaPlan supplies the exact speech-polygon frame.  This
    pure rule normalizes both against the same base axis and never consults
    target text, fit output, or rendered pixels.
    """

    support = _plain_mapping(source_axis_support)
    frame = _plain_mapping(oriented_frame)
    input_reasons = _strings(source_reason_codes)
    result: dict[str, Any] = {
        "contract_version": ORIENTATION_NORMALIZED_ROTATION_VERSION,
        "status": "unavailable",
        "reason_codes": [],
        "value": None,
        "writing_mode": str(writing_mode or "").strip().lower(),
        "orientation_confidence": _finite_number(orientation_confidence),
        "source_absolute_axis_degrees_clockwise": None,
        "source_eroded_absolute_axis_degrees_clockwise": None,
        "source_residual_degrees_clockwise": None,
        "container_absolute_axis_degrees_clockwise": None,
        "container_residual_degrees_clockwise": None,
        "source_container_axis_agreement_degrees": None,
        "erosion_angle_delta_degrees": None,
        "thresholds": {
            "minimum_absolute_residual_degrees": (
                ORIENTATION_NORMALIZED_ROTATION_MIN_ABS_DEGREES
            ),
            "maximum_absolute_residual_degrees": (
                ORIENTATION_NORMALIZED_ROTATION_MAX_ABS_DEGREES
            ),
            "maximum_erosion_delta_degrees": (
                ORIENTATION_NORMALIZED_ROTATION_MAX_EROSION_DELTA_DEGREES
            ),
            "maximum_frame_agreement_degrees": (
                ORIENTATION_NORMALIZED_ROTATION_MAX_FRAME_AGREEMENT_DEGREES
            ),
            "minimum_orientation_confidence": (
                ORIENTATION_NORMALIZED_ROTATION_MIN_ORIENTATION_CONFIDENCE
            ),
        },
        "source_observation_provenance": (
            "authorized_foreground_principal_axis"
        ),
        "container_frame_provenance": str(frame.get("provenance") or ""),
        "translation_content_consulted": False,
        "target_fit_consulted": False,
        "render_output_consulted": False,
    }

    def unavailable(*reasons: str) -> dict[str, Any]:
        result["reason_codes"] = _unique_strings(reasons)
        return result

    if (
        "perceptual_rotation_base_axis_or_rotation_ambiguous"
        not in input_reasons
    ):
        return unavailable("source_axis_not_base_axis_ambiguous")
    confidence = _finite_number(orientation_confidence)
    if (
        confidence is None
        or confidence
        < ORIENTATION_NORMALIZED_ROTATION_MIN_ORIENTATION_CONFIDENCE
    ):
        return unavailable("source_orientation_not_reliably_resolved")
    mode = result["writing_mode"]
    if mode not in {"vertical", "horizontal"}:
        return unavailable("source_writing_mode_unavailable")
    if (
        frame.get("contract_version") != ORIENTED_LAYOUT_FRAME_VERSION
        or frame.get("status") != "supported"
        or frame.get("container_type") != "speech_bubble"
        or len(str(frame.get("polygon_sha256") or "")) != 64
        or len(_polygon(frame.get("polygon"))) < 3
    ):
        return unavailable("exact_speech_oriented_frame_unavailable")

    source_axis = _finite_number(support.get("degrees_clockwise"))
    eroded_axis = _finite_number(
        support.get("eroded_degrees_clockwise")
    )
    reported_erosion_delta = _finite_number(
        support.get("erosion_angle_delta_degrees")
    )
    container_axis = _finite_number(
        frame.get("absolute_major_axis_degrees_clockwise")
    )
    if any(
        value is None
        for value in (
            source_axis,
            eroded_axis,
            reported_erosion_delta,
            container_axis,
        )
    ):
        return unavailable("rotation_geometry_measurement_incomplete")

    assert source_axis is not None
    assert eroded_axis is not None
    assert reported_erosion_delta is not None
    assert container_axis is not None
    source_residual = source_axis_residual(source_axis, mode)
    container_residual = source_axis_residual(container_axis, mode)
    derived_erosion_delta = abs(shortest_axis_delta(source_axis, eroded_axis))
    agreement = abs(
        shortest_axis_delta(source_residual, container_residual)
    )
    result.update(
        {
            "source_absolute_axis_degrees_clockwise": round(source_axis, 8),
            "source_eroded_absolute_axis_degrees_clockwise": round(
                eroded_axis,
                8,
            ),
            "source_residual_degrees_clockwise": round(source_residual, 8),
            "container_absolute_axis_degrees_clockwise": round(
                container_axis,
                8,
            ),
            "container_residual_degrees_clockwise": round(
                container_residual,
                8,
            ),
            "source_container_axis_agreement_degrees": round(agreement, 8),
            "erosion_angle_delta_degrees": round(
                derived_erosion_delta,
                8,
            ),
        }
    )
    if abs(reported_erosion_delta - derived_erosion_delta) > 1e-4:
        return unavailable("rotation_erosion_measurement_inconsistent")
    if (
        derived_erosion_delta
        > ORIENTATION_NORMALIZED_ROTATION_MAX_EROSION_DELTA_DEGREES
    ):
        return unavailable("source_axis_not_stable_under_erosion")
    if (
        abs(source_residual)
        < ORIENTATION_NORMALIZED_ROTATION_MIN_ABS_DEGREES
    ):
        return unavailable("orientation_residual_below_activation_floor")
    if (
        abs(source_residual)
        > ORIENTATION_NORMALIZED_ROTATION_MAX_ABS_DEGREES
    ):
        return unavailable("orientation_residual_exceeds_safe_range")
    if (
        agreement
        > ORIENTATION_NORMALIZED_ROTATION_MAX_FRAME_AGREEMENT_DEGREES
    ):
        return unavailable("source_container_axis_disagreement")

    result.update(
        {
            "status": "supported",
            "reason_codes": [
                "orientation_normalized_rotation",
                "source_container_axis_corroborated",
                "source_axis_stable_under_erosion",
            ],
            "value": {
                "degrees_clockwise": round(source_residual, 8),
                "pivot": "visual_center",
            },
            "confidence": round(
                min(
                    confidence,
                    0.90
                    + 0.04
                    * (
                        1.0
                        - agreement
                        / ORIENTATION_NORMALIZED_ROTATION_MAX_FRAME_AGREEMENT_DEGREES
                    )
                    + 0.04
                    * (
                        1.0
                        - derived_erosion_delta
                        / ORIENTATION_NORMALIZED_ROTATION_MAX_EROSION_DELTA_DEGREES
                    ),
                ),
                8,
            ),
        }
    )
    return result


def oriented_frame_from_speech_container(container: Any) -> dict[str, Any]:
    """Canonicalize one exact TextAreaPlan speech polygon into a frame."""

    record = _plain_mapping(container)
    base = {
        "contract_version": ORIENTED_LAYOUT_FRAME_VERSION,
        "status": "unavailable",
        "page_id": str(record.get("page_id") or ""),
        "container_id": str(record.get("container_id") or ""),
        "container_type": str(record.get("container_type") or ""),
        "coordinate_space": "page",
        "reason_codes": [],
        "polygon": [],
        "polygon_sha256": "",
        "polygon_evidence_id": "",
        "center_page": [],
        "major_extent_px": 0.0,
        "minor_extent_px": 0.0,
        "absolute_major_axis_degrees_clockwise": 0.0,
        "oriented_corners": [],
        "container_bbox": _bbox(record.get("bbox")),
        "provenance": "TextAreaPlan.speech_mask_polygon",
    }
    if base["container_type"] != "speech_bubble":
        return _unavailable(base, "container_not_speech")
    if bool(record.get("must_not_mutate")):
        return _unavailable(base, "container_protected")
    if _strings(record.get("conflict_flags")):
        return _unavailable(base, "container_conflicted")

    evidence = record.get("semantic_role_evidence")
    raw_polygons = (
        evidence.get("speech_mask_polygons")
        if isinstance(evidence, Mapping)
        else None
    )
    polygons = raw_polygons if isinstance(raw_polygons, Sequence) else ()
    if len(polygons) > 1:
        return _unavailable(base, "speech_polygon_ambiguous")
    if not polygons or not isinstance(polygons[0], Mapping):
        return _unavailable(base, "speech_polygon_missing")
    polygon_record = dict(polygons[0])
    raw_polygon = polygon_record.get("polygon")
    if (
        isinstance(raw_polygon, Sequence)
        and not isinstance(raw_polygon, (str, bytes, bytearray))
        and len(raw_polygon) > ORIENTED_LAYOUT_MAX_POLYGON_VERTICES
    ):
        return _unavailable(base, "speech_polygon_vertex_budget_exceeded")
    polygon = _polygon(raw_polygon)
    if len(polygon) < 3:
        return _unavailable(base, "speech_polygon_invalid")

    points = np.asarray(polygon, dtype=np.float32)
    (center_x, center_y), (width, height), raw_angle = cv2.minAreaRect(
        points
    )
    width = float(width)
    height = float(height)
    if not all(
        math.isfinite(item) and item > 0.0 for item in (width, height)
    ):
        return _unavailable(base, "speech_polygon_frame_invalid")
    major = max(width, height)
    minor = min(width, height)
    absolute = normalize_axis_degrees(
        float(raw_angle) if width >= height else float(raw_angle) - 90.0
    )
    corners = _canonical_corners(
        cv2.boxPoints(((center_x, center_y), (width, height), raw_angle))
    )
    encoded_polygon = json.dumps(
        polygon,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    result = dict(base)
    result.update(
        {
            "status": "supported",
            "reason_codes": ["exact_speech_polygon_oriented_frame"],
            "polygon": polygon,
            "polygon_sha256": hashlib.sha256(encoded_polygon).hexdigest(),
            "polygon_evidence_id": str(
                polygon_record.get("evidence_id") or ""
            ),
            "center_page": [
                round(float(center_x), 6),
                round(float(center_y), 6),
            ],
            "major_extent_px": round(major, 6),
            "minor_extent_px": round(minor, 6),
            "absolute_major_axis_degrees_clockwise": round(absolute, 6),
            "oriented_corners": corners,
            "polygon_confidence": _finite_number(
                polygon_record.get("confidence")
            ),
        }
    )
    return result


def _unavailable(base: Mapping[str, Any], reason: str) -> dict[str, Any]:
    result = dict(base)
    result["reason_codes"] = [str(reason)]
    return result


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    method = getattr(value, "to_dict", None)
    if callable(method):
        result = method()
        return dict(result) if isinstance(result, Mapping) else {}
    values = getattr(value, "__dict__", None)
    return dict(values) if isinstance(values, Mapping) else {}


def _polygon(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return []
    if len(value) > ORIENTED_LAYOUT_MAX_POLYGON_VERTICES:
        return []
    result: list[list[float]] = []
    for point in value:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes, bytearray))
            or len(point) < 2
        ):
            return []
        x = _finite_number(point[0])
        y = _finite_number(point[1])
        if x is None or y is None:
            return []
        result.append([float(x), float(y)])
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _canonical_corners(value: Any) -> list[list[float]]:
    points = [[float(item[0]), float(item[1])] for item in value]
    center_x = sum(item[0] for item in points) / len(points)
    center_y = sum(item[1] for item in points) / len(points)
    ordered = sorted(
        points,
        key=lambda point: math.atan2(
            point[1] - center_y,
            point[0] - center_x,
        ),
    )
    first = min(
        range(len(ordered)),
        key=lambda index: (ordered[index][1], ordered[index][0]),
    )
    ordered = [*ordered[first:], *ordered[:first]]
    return [[round(x, 6), round(y, 6)] for x, y in ordered]


def _bbox(value: Any) -> list[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 4
    ):
        return []
    parsed = [_finite_number(item) for item in value[:4]]
    if any(item is None for item in parsed):
        return []
    return [int(round(float(item))) for item in parsed]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return []
    return [str(item) for item in value if str(item)]


def _unique_strings(value: Any) -> list[str]:
    return list(dict.fromkeys(_strings(value)))
