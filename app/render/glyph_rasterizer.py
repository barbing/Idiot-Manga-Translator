# -*- coding: utf-8 -*-
"""Direct FreeType rasterization of already-shaped HarfBuzz glyph runs.

This module is a Stage 5 consumer. It never reshapes Unicode text, chooses a
font, or changes placement geometry. Its only authority is to rasterize the
glyph IDs and positions supplied by a TypesetLayout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import freetype
except Exception:  # pragma: no cover - exercised by explicit runtime failure
    freetype = None

try:
    from PIL import Image, ImageChops, ImageFilter
except Exception:  # pragma: no cover - Pillow is a core runtime dependency
    Image = None
    ImageChops = None
    ImageFilter = None


GLYPH_RASTERIZER_VERSION = "harfbuzz_freetype_glyph_rasterizer_v1"
GLYPH_RASTER_AUTHORITY = "harfbuzz_freetype_glyph_ids"


@dataclass
class GlyphRasterResult:
    """One target-sized RGBA raster plus auditable glyph consumption."""

    image: Any | None
    audit: dict[str, Any]

    @property
    def drawn(self) -> bool:
        return bool(self.image is not None and self.audit.get("status") == "drawn")


class FreeTypeGlyphRasterizer:
    """Rasterize positioned glyph IDs without invoking a string renderer."""

    version = GLYPH_RASTERIZER_VERSION

    def __init__(self) -> None:
        self._faces: dict[str, Any] = {}

    def rasterize(
        self,
        *,
        shaped_run: Mapping[str, Any],
        requested_glyph_ids: Sequence[int] | None,
        target_size: tuple[int, int],
        fill: tuple[int, int, int, int],
        stroke_fill: tuple[int, int, int, int],
        stroke_width: int = 0,
        position_policy: str = "harfbuzz",
    ) -> GlyphRasterResult:
        width = max(1, int(target_size[0]))
        height = max(1, int(target_size[1]))
        font_face_id = str(shaped_run.get("font_face_id") or "")
        font_path = str(shaped_run.get("font_path") or "")
        font_size = max(1, int(round(_as_float(shaped_run.get("font_size"), 1.0))))
        direction = str(shaped_run.get("direction") or "")
        features = {
            str(key): bool(value)
            for key, value in dict(shaped_run.get("features") or {}).items()
        }
        shaped_glyphs = [
            dict(item)
            for item in list(shaped_run.get("glyphs") or [])
            if isinstance(item, Mapping)
        ]
        shaped_ids = [_as_int(item.get("glyph_id"), -1) for item in shaped_glyphs]
        requested = (
            [_as_int(item, -1) for item in requested_glyph_ids]
            if requested_glyph_ids is not None
            else list(shaped_ids)
        )
        base_audit: dict[str, Any] = {
            "glyph_rasterizer_version": self.version,
            "raster_authority": GLYPH_RASTER_AUTHORITY,
            "status": "failed",
            "font_face_id": font_face_id,
            "font_path": font_path,
            "font_size": font_size,
            "direction": direction,
            "features": features,
            "position_policy": str(position_policy or "harfbuzz"),
            "freetype_load_policy": "hinted_normal_gray",
            "alpha_composition_policy": "coverage_union_then_single_target_copy",
            "harfbuzz_advances_offsets_recorded": True,
            "requested_glyph_ids": requested,
            "drawn_glyph_ids": [],
            "x_advances": [],
            "y_advances": [],
            "x_offsets": [],
            "y_offsets": [],
            "advances_offsets_consumed": False,
            "target_size": [width, height],
            "ink_bounds_before_centering": [],
            "ink_bounds_in_target": [],
            "raster_clipped_to_target": False,
            "source_alpha_sum": 0,
            "target_alpha_sum": 0,
            "composite_offset": [0, 0],
            "natural_raster_size": [],
            "logical_cell_overhang_px": [0, 0, 0, 0],
            "hard_cell_fit": {
                "policy": "natural_ink_overhang_no_cell_clip",
                "applied": False,
                "accepted": True,
                "trim_px": [0, 0, 0, 0],
                "max_edge_trim_px": 0,
                "alpha_before": 0,
                "alpha_after": 0,
                "alpha_loss": 0,
                "alpha_loss_ratio": 0.0,
            },
            "glyph_rasters": [],
            "issues": [],
        }
        if freetype is None:
            return _failure(base_audit, "freetype_unavailable")
        if Image is None or ImageChops is None or ImageFilter is None:
            return _failure(base_audit, "pillow_mask_compositor_unavailable")
        if not font_path:
            return _failure(base_audit, "raster_font_path_missing")
        if not shaped_glyphs:
            return _failure(base_audit, "raster_shaped_glyphs_missing")
        if position_policy not in {
            "harfbuzz",
            "compact_vertical_sequence_preserved",
            "compact_horizontal_sequence_preserved",
        }:
            return _failure(base_audit, f"raster_position_policy_unsupported:{position_policy}")
        selected = _select_glyph_sequence(shaped_glyphs, requested)
        if selected is None:
            return _failure(base_audit, "raster_glyph_sequence_mismatch")

        try:
            face = self._face(font_path)
            face.set_pixel_sizes(0, font_size)
        except Exception as exc:
            return _failure(base_audit, f"raster_font_load_failed:{type(exc).__name__}")

        pen_x = 0.0
        pen_y = 0.0
        bitmaps: list[dict[str, Any]] = []
        issues: list[str] = []
        for glyph in selected:
            glyph_id = _as_int(glyph.get("glyph_id"), -1)
            x_advance = _as_float(glyph.get("x_advance"), 0.0)
            y_advance = _as_float(glyph.get("y_advance"), 0.0)
            x_offset = _as_float(glyph.get("x_offset"), 0.0)
            y_offset = _as_float(glyph.get("y_offset"), 0.0)
            record: dict[str, Any] = {
                "glyph_id": glyph_id,
                "glyph_name": str(glyph.get("glyph_name") or ""),
                "cluster": _as_int(glyph.get("cluster"), 0),
                "x_advance": x_advance,
                "y_advance": y_advance,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "pen_before": [round(pen_x, 4), round(pen_y, 4)],
                "bitmap_box": [],
                "bitmap_size": [0, 0],
                "drawn": False,
            }
            if glyph_id < 0:
                issues.append("raster_invalid_glyph_id")
                base_audit["glyph_rasters"].append(record)
                pen_x += x_advance
                pen_y += y_advance
                continue
            try:
                face.load_glyph(
                    glyph_id,
                    freetype.FT_LOAD_RENDER
                    | freetype.FT_LOAD_TARGET_NORMAL,
                )
                slot = face.glyph
                bitmap_image = _bitmap_to_image(slot.bitmap)
                bitmap_w, bitmap_h = bitmap_image.size if bitmap_image is not None else (0, 0)
                left = pen_x + x_offset + float(slot.bitmap_left)
                top = -(pen_y + y_offset + float(slot.bitmap_top))
                record["bitmap_box"] = [
                    round(left, 4),
                    round(top, 4),
                    int(bitmap_w),
                    int(bitmap_h),
                ]
                record["bitmap_size"] = [int(bitmap_w), int(bitmap_h)]
                if bitmap_image is not None and bitmap_w > 0 and bitmap_h > 0:
                    record["drawn"] = True
                    bitmaps.append(
                        {
                            "glyph_id": glyph_id,
                            "image": bitmap_image,
                            "left": left,
                            "top": top,
                            "record": record,
                        }
                    )
                else:
                    issues.append(f"raster_empty_glyph_bitmap:{glyph_id}")
            except Exception as exc:
                issues.append(f"raster_glyph_load_failed:{glyph_id}:{type(exc).__name__}")
            base_audit["glyph_rasters"].append(record)
            pen_x += x_advance
            pen_y += y_advance

        base_audit["x_advances"] = [_as_float(item.get("x_advance"), 0.0) for item in selected]
        base_audit["y_advances"] = [_as_float(item.get("y_advance"), 0.0) for item in selected]
        base_audit["x_offsets"] = [_as_float(item.get("x_offset"), 0.0) for item in selected]
        base_audit["y_offsets"] = [_as_float(item.get("y_offset"), 0.0) for item in selected]
        base_audit["advances_offsets_consumed"] = position_policy == "harfbuzz"
        if not bitmaps:
            base_audit["issues"] = _unique(issues or ["raster_no_drawable_glyph_bitmap"])
            return GlyphRasterResult(image=None, audit=base_audit)

        if position_policy == "compact_vertical_sequence_preserved":
            max_bitmap_w = max(int(item["image"].size[0]) for item in bitmaps)
            max_bitmap_h = max(int(item["image"].size[1]) for item in bitmaps)
            if len(bitmaps) > 1:
                available_step = max(0.0, float(height - max_bitmap_h)) / float(len(bitmaps) - 1)
                step = max(0.0, min(float(max_bitmap_h) * 0.50, available_step))
            else:
                step = 0.0
            for index, item in enumerate(bitmaps):
                bitmap_w, bitmap_h = item["image"].size
                item["left"] = (float(max_bitmap_w) - float(bitmap_w)) / 2.0
                item["top"] = float(index) * step + (float(max_bitmap_h) - float(bitmap_h)) / 2.0
                item["record"]["bitmap_box"] = [
                    round(float(item["left"]), 4),
                    round(float(item["top"]), 4),
                    int(bitmap_w),
                    int(bitmap_h),
                ]
            base_audit["compact_vertical_step"] = round(step, 4)
        elif position_policy == "compact_horizontal_sequence_preserved":
            max_bitmap_h = max(int(item["image"].size[1]) for item in bitmaps)
            total_bitmap_w = sum(int(item["image"].size[0]) for item in bitmaps)
            if len(bitmaps) > 1:
                available_gap = max(0.0, float(width - total_bitmap_w)) / float(len(bitmaps) - 1)
                preferred_gap = max(1.0, round(float(font_size) * 0.06))
                gap = max(0.0, min(preferred_gap, available_gap))
            else:
                gap = 0.0
            cursor_x = 0.0
            for item in bitmaps:
                bitmap_w, bitmap_h = item["image"].size
                item["left"] = cursor_x
                item["top"] = (float(max_bitmap_h) - float(bitmap_h)) / 2.0
                item["record"]["bitmap_box"] = [
                    round(float(item["left"]), 4),
                    round(float(item["top"]), 4),
                    int(bitmap_w),
                    int(bitmap_h),
                ]
                cursor_x += float(bitmap_w) + gap
            base_audit["compact_horizontal_gap"] = round(gap, 4)
            base_audit["compact_horizontal_symbol_count"] = len(bitmaps)

        min_x = min(float(item["left"]) for item in bitmaps)
        min_y = min(float(item["top"]) for item in bitmaps)
        max_x = max(float(item["left"]) + int(item["image"].size[0]) for item in bitmaps)
        max_y = max(float(item["top"]) + int(item["image"].size[1]) for item in bitmaps)
        ink_w = max(1, int(round(max_x - min_x)))
        ink_h = max(1, int(round(max_y - min_y)))
        ink_mask = Image.new("L", (ink_w, ink_h), 0)
        for item in bitmaps:
            px = int(round(float(item["left"]) - min_x))
            py = int(round(float(item["top"]) - min_y))
            glyph_mask = Image.new("L", (ink_w, ink_h), 0)
            glyph_mask.paste(item["image"], (px, py))
            ink_mask = ImageChops.lighter(ink_mask, glyph_mask)

        dest_x = int(round((float(width) - float(ink_w)) / 2.0))
        dest_y = int(round((float(height) - float(ink_h)) / 2.0))
        source_alpha_sum = int(sum(ink_mask.getdata()))
        safe_stroke = max(0, int(stroke_width))
        natural_w = ink_w + safe_stroke * 2
        natural_h = ink_h + safe_stroke * 2
        natural_fill_mask = Image.new("L", (natural_w, natural_h), 0)
        natural_fill_mask.paste(ink_mask, (safe_stroke, safe_stroke))
        composite_offset = [dest_x - safe_stroke, dest_y - safe_stroke]
        overhang = [
            max(0, -composite_offset[0]),
            max(0, -composite_offset[1]),
            max(0, composite_offset[0] + natural_w - width),
            max(0, composite_offset[1] + natural_h - height),
        ]
        hard_cell_fit = {
            "policy": "natural_ink_overhang_no_cell_clip",
            "applied": any(value > 0 for value in overhang),
            "accepted": True,
            "trim_px": [0, 0, 0, 0],
            "max_edge_trim_px": 0,
            "alpha_before": source_alpha_sum,
            "alpha_after": source_alpha_sum,
            "alpha_loss": 0,
            "alpha_loss_ratio": 0.0,
        }
        base_audit["source_alpha_sum"] = source_alpha_sum
        base_audit["target_alpha_sum"] = source_alpha_sum
        base_audit["hard_cell_fit"] = hard_cell_fit
        base_audit["raster_clipped_to_target"] = False
        base_audit["composite_offset"] = list(composite_offset)
        base_audit["natural_raster_size"] = [natural_w, natural_h]
        base_audit["logical_cell_overhang_px"] = list(overhang)

        layer = Image.new("RGBA", (natural_w, natural_h), (0, 0, 0, 0))
        if safe_stroke > 0:
            kernel = max(3, safe_stroke * 2 + 1)
            if kernel % 2 == 0:
                kernel += 1
            stroke_mask = natural_fill_mask.filter(ImageFilter.MaxFilter(kernel))
            stroke_layer = Image.new("RGBA", (natural_w, natural_h), tuple(stroke_fill))
            stroke_layer.putalpha(stroke_mask)
            layer.alpha_composite(stroke_layer)
        fill_layer = Image.new("RGBA", (natural_w, natural_h), tuple(fill))
        fill_layer.putalpha(natural_fill_mask)
        layer.alpha_composite(fill_layer)

        base_audit.update(
            {
                "status": "drawn",
                "drawn_glyph_ids": [int(item["glyph_id"]) for item in bitmaps],
                "ink_bounds_before_centering": [
                    round(min_x, 4),
                    round(min_y, 4),
                    ink_w,
                    ink_h,
                ],
                "ink_bounds_in_target": [
                    dest_x,
                    dest_y,
                    dest_x + ink_w,
                    dest_y + ink_h,
                ],
                "raster_clipped_to_target": False,
                "source_alpha_sum": source_alpha_sum,
                "target_alpha_sum": source_alpha_sum,
                "composite_offset": list(composite_offset),
                "natural_raster_size": [natural_w, natural_h],
                "logical_cell_overhang_px": list(overhang),
                "hard_cell_fit": hard_cell_fit,
                "issues": _unique(issues),
            }
        )
        return GlyphRasterResult(image=layer, audit=base_audit)

    def _face(self, path: str):
        face = self._faces.get(path)
        if face is None:
            face = freetype.Face(path)
            self._faces[path] = face
        return face


def _select_glyph_sequence(
    shaped_glyphs: Sequence[Mapping[str, Any]],
    requested: Sequence[int],
) -> list[dict[str, Any]] | None:
    values = [dict(item) for item in shaped_glyphs]
    shaped_ids = [_as_int(item.get("glyph_id"), -1) for item in values]
    wanted = [_as_int(item, -1) for item in requested]
    if not wanted:
        return []
    if wanted == shaped_ids:
        return values
    span = len(wanted)
    for start in range(0, len(shaped_ids) - span + 1):
        if shaped_ids[start : start + span] == wanted:
            return values[start : start + span]
    return None


def _bitmap_to_image(bitmap):
    width = int(getattr(bitmap, "width", 0) or 0)
    rows = int(getattr(bitmap, "rows", 0) or 0)
    if width <= 0 or rows <= 0:
        return None
    pitch = int(getattr(bitmap, "pitch", width) or width)
    pixel_mode = int(getattr(bitmap, "pixel_mode", 0) or 0)
    raw = bytes(bitmap.buffer)
    row_stride = abs(pitch)
    rows_out: list[bytes] = []
    gray_mode = int(getattr(freetype, "FT_PIXEL_MODE_GRAY", 2))
    mono_mode = int(getattr(freetype, "FT_PIXEL_MODE_MONO", 1))
    if pixel_mode == gray_mode:
        for row in range(rows):
            source_row = row if pitch >= 0 else rows - 1 - row
            start = source_row * row_stride
            rows_out.append(raw[start : start + width])
    elif pixel_mode == mono_mode:
        for row in range(rows):
            source_row = row if pitch >= 0 else rows - 1 - row
            start = source_row * row_stride
            packed = raw[start : start + row_stride]
            expanded = bytearray(width)
            for column in range(width):
                byte = packed[column // 8] if column // 8 < len(packed) else 0
                expanded[column] = 255 if byte & (0x80 >> (column % 8)) else 0
            rows_out.append(bytes(expanded))
    else:
        raise RuntimeError(f"unsupported_freetype_pixel_mode:{pixel_mode}")
    return Image.frombytes("L", (width, rows), b"".join(rows_out))


def _failure(audit: Mapping[str, Any], issue: str) -> GlyphRasterResult:
    payload = dict(audit)
    payload["status"] = "failed"
    payload["issues"] = _unique([*(payload.get("issues") or []), issue])
    return GlyphRasterResult(image=None, audit=payload)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _unique(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
