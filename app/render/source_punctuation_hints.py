# -*- coding: utf-8 -*-
"""Source visual punctuation evidence for renderer layout hints.

This module observes source-page pixels for style/layout evidence only. It does
not alter OCR text, parent topology, cleanup masks, or translation ownership.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from statistics import median
from typing import Any, Mapping, Sequence

from app.render.typesetting_contracts import bbox_from_value, copy_jsonish
from app.render.typesetting_text import DASH_CHARS, grapheme_clusters


SOURCE_VISUAL_PUNCTUATION_HINTS_VERSION = "source_visual_punctuation_hints_v1"
_DARK_THRESHOLD = 170


def source_visual_punctuation_hints(
    *,
    source_text: str,
    source_contract_bbox: Sequence[int],
    source_image_path: str,
) -> dict[str, Any]:
    """Return renderer-only source punctuation hints for one parent source box."""

    bbox = bbox_from_value(source_contract_bbox)
    text = str(source_text or "")
    if not bbox or not source_image_path or not _has_dash_text(text):
        return {}
    image = _cached_grayscale_image(source_image_path)
    if image is None:
        return {}
    crop = _crop_source_box(image, bbox)
    if crop is None:
        return {}
    components = _connected_components(crop)
    dash_components = _dash_like_components(components, crop.size)
    if not dash_components:
        return {}
    nominal = _nominal_vertical_glyph_height(components, dash_components, text, bbox)
    if nominal <= 0.0:
        return {}
    source_dash_count = _dash_run_count(text)
    hints: list[dict[str, Any]] = []
    for ordinal, component in enumerate(dash_components[:source_dash_count]):
        height = float(component["bbox"][3] - component["bbox"][1])
        visual_units = max(1.0, height / nominal)
        if visual_units < 1.35:
            continue
        visual_units = min(4.5, max(1.5, visual_units))
        local_box = list(component["bbox"])
        hints.append(
            {
                "kind": "dash",
                "source_ordinal": int(ordinal),
                "visual_units": round(float(visual_units), 3),
                "source_component_bbox": [
                    int(bbox[0] + local_box[0]),
                    int(bbox[1] + local_box[1]),
                    int(local_box[2] - local_box[0]),
                    int(local_box[3] - local_box[1]),
                ],
                "source_component_bbox_local": [
                    int(local_box[0]),
                    int(local_box[1]),
                    int(local_box[2] - local_box[0]),
                    int(local_box[3] - local_box[1]),
                ],
                "nominal_source_glyph_height": round(float(nominal), 3),
                "source": "source_connected_component_dash_length",
            }
        )
    if not hints:
        return {}
    return {
        "source_visual_punctuation_hints_version": SOURCE_VISUAL_PUNCTUATION_HINTS_VERSION,
        "hints": copy_jsonish(hints),
    }


def _has_dash_text(text: str) -> bool:
    return any(char in DASH_CHARS for char in str(text or ""))


def _dash_run_count(text: str) -> int:
    count = 0
    in_run = False
    for cluster in grapheme_clusters(text):
        if all(char in DASH_CHARS for char in cluster):
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return max(1, count)


@lru_cache(maxsize=4)
def _cached_grayscale_image(path: str):
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("L")
    except Exception:
        return None


def _crop_source_box(image, bbox: Sequence[int]):
    box = bbox_from_value(bbox)
    if not box:
        return None
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    width, height = image.size
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(x0, min(width, x + w))
    y1 = max(y0, min(height, y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def _connected_components(image) -> list[dict[str, Any]]:
    width, height = image.size
    pixels = image.load()
    ink: set[tuple[int, int]] = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if int(pixels[x, y]) < _DARK_THRESHOLD
    }
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for point in list(ink):
        if point in seen:
            continue
        queue: deque[tuple[int, int]] = deque([point])
        seen.add(point)
        xs: list[int] = []
        ys: list[int] = []
        while queue:
            x, y = queue.popleft()
            xs.append(x)
            ys.append(y)
            for nx in (x - 1, x, x + 1):
                for ny in (y - 1, y, y + 1):
                    if (nx, ny) == (x, y):
                        continue
                    neighbor = (nx, ny)
                    if neighbor in ink and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
        if not xs:
            continue
        bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
        components.append({"bbox": bbox, "area": len(xs)})
    return sorted(components, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _dash_like_components(components: Sequence[Mapping[str, Any]], size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = size
    results: list[dict[str, Any]] = []
    for component in components:
        bbox = list(component.get("bbox") or [])
        if len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        comp_w = max(1, x1 - x0)
        comp_h = max(1, y1 - y0)
        center_x = (x0 + x1) / 2.0
        if center_x < width * 0.15 or center_x > width * 0.85:
            continue
        if y0 <= 1 or y1 >= height - 1:
            continue
        if comp_h < max(32, height * 0.20):
            continue
        if comp_w > max(12, width * 0.22):
            continue
        if comp_h / float(comp_w) < 5.0:
            continue
        results.append(dict(component))
    return sorted(results, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _nominal_vertical_glyph_height(
    components: Sequence[Mapping[str, Any]],
    dash_components: Sequence[Mapping[str, Any]],
    source_text: str,
    source_bbox: Sequence[int],
) -> float:
    dash_boxes = {tuple(item.get("bbox") or []) for item in dash_components}
    heights: list[float] = []
    for component in components:
        bbox = tuple(component.get("bbox") or [])
        if bbox in dash_boxes or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]
        comp_w = max(1, x1 - x0)
        comp_h = max(1, y1 - y0)
        area = int(component.get("area") or 0)
        if area < 20 or comp_h < 10 or comp_w < 4:
            continue
        if comp_h > max(70, bbox_from_value(source_bbox)[3] * 0.35):
            continue
        heights.append(float(comp_h))
    if heights:
        return float(median(heights))
    source_box = bbox_from_value(source_bbox)
    clusters = [cluster for cluster in grapheme_clusters(source_text) if cluster.strip()]
    if source_box and clusters:
        return max(1.0, float(source_box[3]) / float(len(clusters)))
    return 0.0
