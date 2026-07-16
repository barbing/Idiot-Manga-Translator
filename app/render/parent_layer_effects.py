# -*- coding: utf-8 -*-
"""Strict explicit parent-layer effect contracts and fit geometry.

This module is deliberately source-blind. It parses already-resolved renderer
style, predicts complete-parent rotation/shadow bounds, and shifts intact
typeset geometry. It does not observe page pixels or draw effects.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from app.render.typesetting_contracts import GlyphPlacement


PARENT_LAYER_EFFECTS_VERSION = "parent_layer_effects_v1"
ROTATION_MIN_DEGREES = -45.0
ROTATION_MAX_DEGREES = 45.0
SHADOW_OFFSET_LIMIT_PX = 256.0
SHADOW_BLUR_MAX_PX = 64.0
ROTATION_RESAMPLE_GUARD_PX = 2.0
FRACTIONAL_OFFSET_GUARD_PX = 2.0

_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")


@dataclass(frozen=True)
class RotationEffect:
    availability: str = "unavailable"
    degrees_clockwise: float = 0.0
    pivot: str = "visual_center"

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "degrees_clockwise": float(self.degrees_clockwise),
            "pivot": self.pivot,
        }


@dataclass(frozen=True)
class ShadowEffect:
    availability: str = "unavailable"
    color: str = ""
    offset_px: list[float] = field(default_factory=lambda: [0.0, 0.0])
    blur_radius_px: float = 0.0

    @property
    def rgba(self) -> tuple[int, int, int, int]:
        return shadow_color_rgba(self.color)

    @property
    def visible(self) -> bool:
        return self.availability == "resolved" and self.rgba[3] > 0

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "color": self.color,
            "rgba": list(self.rgba) if self.color else [],
            "offset_px": [float(value) for value in self.offset_px],
            "blur_radius_px": float(self.blur_radius_px),
        }


@dataclass(frozen=True)
class ParentLayerEffectsResolution:
    status: str = "unavailable"
    requested: bool = False
    active: bool = False
    rotation: RotationEffect = field(default_factory=RotationEffect)
    shadow: ShadowEffect = field(default_factory=ShadowEffect)
    issues: list[str] = field(default_factory=list)
    contract_present: bool = False

    @property
    def fail_closed(self) -> bool:
        return self.status == "invalid"

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "contract_version": PARENT_LAYER_EFFECTS_VERSION,
            "status": self.status,
            "requested": bool(self.requested),
            "active": bool(self.active),
            "contract_present": bool(self.contract_present),
            "fail_closed": bool(self.fail_closed),
            "rotation": self.rotation.to_audit_dict(),
            "shadow": self.shadow.to_audit_dict(),
            "issues": list(self.issues),
            "source_observation_used": False,
        }


@dataclass(frozen=True)
class EffectEnvelopeResult:
    status: str
    contained: bool
    translation: list[int]
    base_bounds: list[float]
    guarded_base_bounds: list[float]
    rotated_bounds: list[float]
    shadow_bounds: list[float]
    final_bounds: list[float]
    fit_bounds: list[float]
    raster_guard_px: float
    rotation_resample_guard_px: float
    shadow_blur_guard_px: float
    fractional_offset_guard_px: float
    allowed_shift_x: list[int] = field(default_factory=list)
    allowed_shift_y: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "policy": "complete_parent_effect_envelope_v1",
            "status": self.status,
            "contained": bool(self.contained),
            "translation": list(self.translation),
            "translation_policy": "smallest_integer_delta",
            "base_bounds": _round_box(self.base_bounds),
            "guarded_base_bounds": _round_box(self.guarded_base_bounds),
            "rotated_bounds": _round_box(self.rotated_bounds),
            "shadow_bounds": _round_box(self.shadow_bounds),
            "final_bounds": _round_box(self.final_bounds),
            "fit_bounds": _round_box(self.fit_bounds),
            "raster_guard_px": float(self.raster_guard_px),
            "rotation_resample_guard_px": float(self.rotation_resample_guard_px),
            "shadow_blur_guard_px": float(self.shadow_blur_guard_px),
            "fractional_offset_guard_px": float(self.fractional_offset_guard_px),
            "allowed_shift_x": list(self.allowed_shift_x),
            "allowed_shift_y": list(self.allowed_shift_y),
            "issues": list(self.issues),
        }


def resolve_parent_layer_effects(style: Mapping[str, Any] | None) -> ParentLayerEffectsResolution:
    values = dict(style or {})
    if "parent_layer_effects" not in values:
        return ParentLayerEffectsResolution()
    carrier = values.get("parent_layer_effects")
    if not isinstance(carrier, Mapping):
        return _invalid_resolution(["parent_layer_effect_contract_not_mapping"])
    payload = dict(carrier)
    issues: list[str] = []
    unknown = sorted(set(payload) - {"contract_version", "rotation", "shadow"})
    if unknown:
        issues.extend(f"parent_layer_effect_unknown_field:{key}" for key in unknown)
    if payload.get("contract_version") != PARENT_LAYER_EFFECTS_VERSION:
        issues.append("parent_layer_effect_contract_version_invalid")

    rotation, rotation_requested, rotation_issues = _parse_rotation(payload.get("rotation"))
    shadow, shadow_requested, shadow_issues = _parse_shadow(payload.get("shadow"))
    issues.extend(rotation_issues)
    issues.extend(shadow_issues)
    requested = bool(rotation_requested or shadow_requested or issues)
    if issues:
        return ParentLayerEffectsResolution(
            status="invalid",
            requested=True,
            active=False,
            rotation=rotation,
            shadow=shadow,
            issues=_unique(issues),
            contract_present=True,
        )
    resolved = rotation.availability == "resolved" or shadow.availability == "resolved"
    active = bool(
        (rotation.availability == "resolved" and abs(rotation.degrees_clockwise) >= 1e-9)
        or shadow.visible
    )
    return ParentLayerEffectsResolution(
        status="resolved" if resolved else "unavailable",
        requested=bool(requested),
        active=active,
        rotation=rotation,
        shadow=shadow,
        issues=[],
        contract_present=True,
    )


def fit_effect_envelope(
    base_bounds: Sequence[Any],
    hard_bounds: Sequence[Any],
    effects: ParentLayerEffectsResolution,
    *,
    raster_guard_px: float = 2.0,
) -> EffectEnvelopeResult:
    base = _xywh_float(base_bounds)
    hard = _xywh_float(hard_bounds)
    guard = _nonnegative_finite(raster_guard_px)
    if not base or not hard:
        return EffectEnvelopeResult(
            status="rejected",
            contained=False,
            translation=[0, 0],
            base_bounds=base,
            guarded_base_bounds=base,
            rotated_bounds=base,
            shadow_bounds=[],
            final_bounds=base,
            fit_bounds=base,
            raster_guard_px=guard,
            rotation_resample_guard_px=0.0,
            shadow_blur_guard_px=0.0,
            fractional_offset_guard_px=0.0,
            issues=["parent_layer_effect_envelope_bounds_invalid"],
        )
    if effects.status == "invalid":
        return EffectEnvelopeResult(
            status="rejected",
            contained=False,
            translation=[0, 0],
            base_bounds=base,
            guarded_base_bounds=base,
            rotated_bounds=base,
            shadow_bounds=[],
            final_bounds=base,
            fit_bounds=base,
            raster_guard_px=guard,
            rotation_resample_guard_px=0.0,
            shadow_blur_guard_px=0.0,
            fractional_offset_guard_px=0.0,
            issues=["parent_layer_effect_contract_invalid", *effects.issues],
        )
    if not effects.active:
        contained = _contains_xywh(hard, base)
        return EffectEnvelopeResult(
            status="unavailable" if effects.status == "unavailable" else "no_visible_effect",
            contained=contained,
            translation=[0, 0],
            base_bounds=base,
            guarded_base_bounds=base,
            rotated_bounds=base,
            shadow_bounds=[],
            final_bounds=base,
            fit_bounds=base,
            raster_guard_px=guard,
            rotation_resample_guard_px=0.0,
            shadow_blur_guard_px=0.0,
            fractional_offset_guard_px=0.0,
            issues=[] if contained else ["base_layout_exceeds_hard_bounds"],
        )

    guarded = _expand_xywh(base, guard)
    rotation_active = (
        effects.rotation.availability == "resolved"
        and abs(effects.rotation.degrees_clockwise) >= 1e-9
    )
    rotation_guard = ROTATION_RESAMPLE_GUARD_PX if rotation_active else 0.0
    rotated = _rotated_xywh(
        guarded,
        effects.rotation.degrees_clockwise if rotation_active else 0.0,
        _center(base),
    )
    if rotation_guard:
        rotated = _expand_xywh(rotated, rotation_guard)

    shadow: list[float] = []
    blur_guard = 0.0
    fractional_guard = 0.0
    if effects.shadow.visible:
        offset_x, offset_y = effects.shadow.offset_px
        blur_guard = _gaussian_blur_guard(effects.shadow.blur_radius_px)
        fractional_guard = (
            FRACTIONAL_OFFSET_GUARD_PX
            if not _is_integer(offset_x) or not _is_integer(offset_y)
            else 0.0
        )
        shadow = _shift_xywh(rotated, offset_x, offset_y)
        if blur_guard or fractional_guard:
            shadow = _expand_xywh(shadow, blur_guard + fractional_guard)

    final = _union_xywh([rotated, shadow] if shadow else [rotated])
    # Staging happens before rotation on a page-coordinate scratch surface.
    # Keep the guarded base inside hard bounds as well as the visible result so
    # no source pixels can be clipped before the final transform.
    fit_bounds = _union_xywh([guarded, final])
    hard_xyxy = _xywh_to_xyxy(hard)
    fit_xyxy = _xywh_to_xyxy(fit_bounds)
    min_dx = int(math.ceil(hard_xyxy[0] - fit_xyxy[0]))
    max_dx = int(math.floor(hard_xyxy[2] - fit_xyxy[2]))
    min_dy = int(math.ceil(hard_xyxy[1] - fit_xyxy[1]))
    max_dy = int(math.floor(hard_xyxy[3] - fit_xyxy[3]))
    allowed_x = [min_dx, max_dx]
    allowed_y = [min_dy, max_dy]
    if min_dx > max_dx or min_dy > max_dy:
        return EffectEnvelopeResult(
            status="rejected",
            contained=False,
            translation=[0, 0],
            base_bounds=base,
            guarded_base_bounds=guarded,
            rotated_bounds=rotated,
            shadow_bounds=shadow,
            final_bounds=final,
            fit_bounds=fit_bounds,
            raster_guard_px=guard,
            rotation_resample_guard_px=rotation_guard,
            shadow_blur_guard_px=blur_guard,
            fractional_offset_guard_px=fractional_guard,
            allowed_shift_x=allowed_x,
            allowed_shift_y=allowed_y,
            issues=["parent_layer_effect_envelope_exceeds_hard_bounds"],
        )
    dx = _closest_to_zero(min_dx, max_dx)
    dy = _closest_to_zero(min_dy, max_dy)
    shifted_base = _shift_xywh(base, dx, dy)
    shifted_guarded = _shift_xywh(guarded, dx, dy)
    shifted_rotated = _shift_xywh(rotated, dx, dy)
    shifted_shadow = _shift_xywh(shadow, dx, dy) if shadow else []
    shifted_final = _shift_xywh(final, dx, dy)
    shifted_fit = _shift_xywh(fit_bounds, dx, dy)
    contained = _contains_xywh(hard, shifted_fit)
    return EffectEnvelopeResult(
        status="fits" if contained else "rejected",
        contained=contained,
        translation=[dx, dy],
        base_bounds=shifted_base,
        guarded_base_bounds=shifted_guarded,
        rotated_bounds=shifted_rotated,
        shadow_bounds=shifted_shadow,
        final_bounds=shifted_final,
        fit_bounds=shifted_fit,
        raster_guard_px=guard,
        rotation_resample_guard_px=rotation_guard,
        shadow_blur_guard_px=blur_guard,
        fractional_offset_guard_px=fractional_guard,
        allowed_shift_x=allowed_x,
        allowed_shift_y=allowed_y,
        issues=[] if contained else ["parent_layer_effect_envelope_exceeds_hard_bounds"],
    )


def shift_layout_geometry(
    glyphs: Sequence[Any],
    lines: Sequence[Any],
    columns: Sequence[Any],
    base_measured_bounds: Sequence[Any],
    dx: int,
    dy: int,
) -> tuple[list[Any], list[Any], list[Any], list[int]]:
    return (
        [_shift_glyph(item, dx, dy) for item in glyphs],
        [_shift_coordinate_record(item, dx, dy) for item in lines],
        [_shift_coordinate_record(item, dx, dy) for item in columns],
        _shift_xywh_int(base_measured_bounds, dx, dy),
    )


def outward_int_xywh(value: Sequence[Any]) -> list[int]:
    box = _xywh_float(value)
    if not box:
        return []
    left = int(math.floor(box[0]))
    top = int(math.floor(box[1]))
    right = int(math.ceil(box[0] + box[2]))
    bottom = int(math.ceil(box[1] + box[3]))
    return [left, top, max(0, right - left), max(0, bottom - top)]


def shadow_color_rgba(value: str) -> tuple[int, int, int, int]:
    if not isinstance(value, str) or not _COLOR_PATTERN.fullmatch(value):
        return (0, 0, 0, 0)
    raw = value[1:]
    if len(raw) == 6:
        raw += "FF"
    return tuple(int(raw[index : index + 2], 16) for index in range(0, 8, 2))  # type: ignore[return-value]


def _parse_rotation(value: Any) -> tuple[RotationEffect, bool, list[str]]:
    if value is None:
        return RotationEffect(), False, []
    if not isinstance(value, Mapping):
        return RotationEffect(), True, ["rotation_availability_invalid"]
    payload = dict(value)
    availability = payload.get("availability")
    if availability == "unavailable":
        unknown = sorted(set(payload) - {"availability"})
        return (
            RotationEffect(),
            False,
            [f"rotation_unknown_field:{key}" for key in unknown],
        )
    if availability != "resolved":
        return RotationEffect(), True, ["rotation_availability_invalid"]
    issues: list[str] = []
    unknown = sorted(set(payload) - {"availability", "degrees_clockwise", "pivot"})
    issues.extend(f"rotation_unknown_field:{key}" for key in unknown)
    degrees = _strict_number(payload.get("degrees_clockwise"))
    if degrees is None or not ROTATION_MIN_DEGREES <= degrees <= ROTATION_MAX_DEGREES:
        issues.append("rotation_degrees_clockwise_invalid")
        degrees = 0.0
    pivot = payload.get("pivot")
    if pivot != "visual_center":
        issues.append("rotation_pivot_invalid")
        pivot = "visual_center"
    return RotationEffect("resolved", float(degrees), str(pivot)), True, issues


def _parse_shadow(value: Any) -> tuple[ShadowEffect, bool, list[str]]:
    if value is None:
        return ShadowEffect(), False, []
    if not isinstance(value, Mapping):
        return ShadowEffect(), True, ["shadow_availability_invalid"]
    payload = dict(value)
    availability = payload.get("availability")
    if availability == "unavailable":
        unknown = sorted(set(payload) - {"availability"})
        return ShadowEffect(), False, [f"shadow_unknown_field:{key}" for key in unknown]
    if availability != "resolved":
        return ShadowEffect(), True, ["shadow_availability_invalid"]
    issues: list[str] = []
    unknown = sorted(
        set(payload) - {"availability", "color", "offset_px", "blur_radius_px"}
    )
    issues.extend(f"shadow_unknown_field:{key}" for key in unknown)
    color = payload.get("color")
    if not isinstance(color, str) or not _COLOR_PATTERN.fullmatch(color):
        issues.append("shadow_color_invalid")
        color = ""
    offset = payload.get("offset_px")
    offsets: list[float] = [0.0, 0.0]
    if not _is_sequence(offset) or len(list(offset)) != 2:
        issues.append("shadow_offset_px_invalid")
    else:
        parsed = [_strict_number(item) for item in list(offset)]
        if any(item is None or abs(item) > SHADOW_OFFSET_LIMIT_PX for item in parsed):
            issues.append("shadow_offset_px_invalid")
        else:
            offsets = [float(parsed[0]), float(parsed[1])]  # type: ignore[arg-type]
    blur = _strict_number(payload.get("blur_radius_px"))
    if blur is None or not 0.0 <= blur <= SHADOW_BLUR_MAX_PX:
        issues.append("shadow_blur_radius_px_invalid")
        blur = 0.0
    return ShadowEffect("resolved", str(color), offsets, float(blur)), True, issues


def _invalid_resolution(issues: Sequence[str]) -> ParentLayerEffectsResolution:
    return ParentLayerEffectsResolution(
        status="invalid",
        requested=True,
        active=False,
        issues=_unique(issues),
        contract_present=True,
    )


def _strict_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_finite(value: Any) -> float:
    number = _strict_number(value)
    return max(0.0, number) if number is not None else 0.0


def _gaussian_blur_guard(radius: float) -> float:
    return float(math.ceil(3.0 * max(0.0, float(radius))) + 2) if radius > 0 else 0.0


def _rotated_xywh(box: Sequence[float], degrees_clockwise: float, pivot: Sequence[float]) -> list[float]:
    if abs(degrees_clockwise) < 1e-9:
        return list(box)
    x, y, width, height = box
    cx, cy = pivot
    radians = math.radians(degrees_clockwise)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    points = []
    for px, py in ((x, y), (x + width, y), (x, y + height), (x + width, y + height)):
        dx = px - cx
        dy = py - cy
        points.append((cx + cosine * dx - sine * dy, cy + sine * dx + cosine * dy))
    left = min(item[0] for item in points)
    top = min(item[1] for item in points)
    right = max(item[0] for item in points)
    bottom = max(item[1] for item in points)
    return [left, top, right - left, bottom - top]


def _shift_glyph(value: Any, dx: int, dy: int) -> Any:
    if isinstance(value, GlyphPlacement):
        metadata = deepcopy(dict(value.metadata or {}))
        metadata["parent_layer_effect_fit_shift"] = [dx, dy]
        return replace(
            value,
            bbox=_shift_xywh_int(value.bbox, dx, dy),
            position=_shift_point(value.position, dx, dy),
            metadata=metadata,
        )
    if isinstance(value, Mapping):
        item = deepcopy(dict(value))
        item["bbox"] = _shift_xywh_int(item.get("bbox"), dx, dy)
        item["position"] = _shift_point(item.get("position"), dx, dy)
        metadata = deepcopy(dict(item.get("metadata") or {}))
        metadata["parent_layer_effect_fit_shift"] = [dx, dy]
        item["metadata"] = metadata
        return item
    return deepcopy(value)


def _shift_coordinate_record(value: Any, dx: int, dy: int) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    item = deepcopy(dict(value))
    for key in ("x", "raw_x"):
        if item.get(key) is not None:
            item[key] = _shift_scalar(item[key], dx)
    for key in ("y", "raw_y"):
        if item.get(key) is not None:
            item[key] = _shift_scalar(item[key], dy)
    for key in ("bbox", "box", "display_box", "raw_box", "centered_block_box", "measured_bounds"):
        if key in item:
            item[key] = _shift_xywh_int(item.get(key), dx, dy)
    alignment = item.get("layout_visual_alignment")
    if isinstance(alignment, Mapping):
        adjusted = deepcopy(dict(alignment))
        if "measured_bounds" in adjusted:
            adjusted["measured_bounds"] = _shift_xywh_int(adjusted.get("measured_bounds"), dx, dy)
        adjusted["parent_layer_effect_fit_shift"] = [dx, dy]
        item["layout_visual_alignment"] = adjusted
    item["parent_layer_effect_fit_shift"] = [dx, dy]
    return item


def _shift_point(value: Any, dx: int, dy: int) -> list[float]:
    if not _is_sequence(value) or len(list(value)) < 2:
        return []
    items = list(value)
    try:
        return [float(items[0]) + dx, float(items[1]) + dy]
    except (TypeError, ValueError):
        return []


def _shift_scalar(value: Any, delta: int) -> Any:
    try:
        return value + delta
    except Exception:
        try:
            return float(value) + delta
        except Exception:
            return value


def _xywh_float(value: Sequence[Any] | Any) -> list[float]:
    if not _is_sequence(value):
        return []
    items = list(value)
    if len(items) != 4:
        return []
    parsed = [_strict_number(item) for item in items]
    if any(item is None for item in parsed):
        return []
    x, y, width, height = [float(item) for item in parsed]  # type: ignore[arg-type]
    return [x, y, width, height] if width > 0 and height > 0 else []


def _shift_xywh_int(value: Any, dx: int, dy: int) -> list[int]:
    if not _is_sequence(value):
        return []
    try:
        items = [int(round(float(item))) for item in list(value)]
    except (TypeError, ValueError):
        return []
    if len(items) != 4:
        return []
    return [items[0] + dx, items[1] + dy, items[2], items[3]]


def _expand_xywh(box: Sequence[float], amount: float) -> list[float]:
    return [
        float(box[0]) - amount,
        float(box[1]) - amount,
        float(box[2]) + amount * 2.0,
        float(box[3]) + amount * 2.0,
    ]


def _shift_xywh(box: Sequence[float], dx: float, dy: float) -> list[float]:
    return [float(box[0]) + dx, float(box[1]) + dy, float(box[2]), float(box[3])]


def _center(box: Sequence[float]) -> list[float]:
    return [float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0]


def _union_xywh(boxes: Sequence[Sequence[float]]) -> list[float]:
    valid = [list(box) for box in boxes if len(box) == 4]
    if not valid:
        return []
    left = min(box[0] for box in valid)
    top = min(box[1] for box in valid)
    right = max(box[0] + box[2] for box in valid)
    bottom = max(box[1] + box[3] for box in valid)
    return [left, top, right - left, bottom - top]


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _contains_xywh(outer: Sequence[float], inner: Sequence[float]) -> bool:
    if len(outer) != 4 or len(inner) != 4:
        return False
    out = _xywh_to_xyxy(outer)
    inside = _xywh_to_xyxy(inner)
    return bool(
        inside[0] >= out[0]
        and inside[1] >= out[1]
        and inside[2] <= out[2]
        and inside[3] <= out[3]
    )


def _closest_to_zero(lower: int, upper: int) -> int:
    if lower <= 0 <= upper:
        return 0
    return lower if lower > 0 else upper


def _is_integer(value: float) -> bool:
    return abs(float(value) - round(float(value))) < 1e-9


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _round_box(value: Sequence[float]) -> list[float]:
    return [round(float(item), 6) for item in value]


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
