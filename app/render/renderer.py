# -*- coding: utf-8 -*-
"""Simple renderer for translated text."""
from __future__ import annotations
import os
import re
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None


def render_translations(
    image_path: str,
    output_path: str,
    regions: List[Dict[str, object]],
    font_name: str,
    inpaint_mode: str = "fast",
    use_gpu: bool = True,
) -> None:
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        working = img
        img_w, img_h = img.size
        render_regions: List[Tuple[str, Tuple[int, int, int, int], str]] = []
        text_mask = None
        bubble_text_mask = None
        other_text_mask = None
        if cv2 is not None and np is not None:
            text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            bubble_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            other_text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        img_np = np.array(img) if cv2 is not None and np is not None else None

        for region in regions:
            if not isinstance(region, dict):
                continue
            raw_text = str(region.get("translation", "")).strip()
            text = _normalize_text(raw_text)
            flags = region.get("flags", {}) or {}
            if not text or flags.get("bg_text") or flags.get("ignore"):
                continue
            bbox = region.get("bbox", [0, 0, 0, 0])
            polygon = region.get("polygon")
            poly_bounds = _polygon_bounds(polygon)
            if poly_bounds and _box_area(poly_bounds) < _box_area(bbox) * 0.85:
                x, y, w, h = [int(v) for v in poly_bounds]
            else:
                x, y, w, h = [int(v) for v in bbox]
            mask_pad = _fill_padding(w, h)
            text_pad = _text_padding(w, h)
            mx0 = max(0, x - mask_pad)
            my0 = max(0, y - mask_pad)
            mx1 = min(img_w, x + w + mask_pad)
            my1 = min(img_h, y + h + mask_pad)
            tx0 = max(0, x - text_pad)
            ty0 = max(0, y - text_pad)
            tx1 = min(img_w, x + w + text_pad)
            ty1 = min(img_h, y + h + text_pad)
            base_box = (x, y, x + w, y + h)
            render_box = (tx0, ty0, tx1, ty1)
            if img_np is not None and text_mask is not None:
                region_mask, bubble_box, bubble_mask = _region_masks(
                    img_np,
                    (mx0, my0, mx1, my1),
                    bbox,
                    polygon,
                )
                if region_mask is not None:
                    if bubble_mask is not None:
                        region_mask = cv2.bitwise_and(region_mask, bubble_mask)
                    text_mask = cv2.bitwise_or(text_mask, region_mask)
                    if bubble_mask is not None and bubble_text_mask is not None:
                        bubble_text_mask = cv2.bitwise_or(bubble_text_mask, region_mask)
                    elif other_text_mask is not None:
                        other_text_mask = cv2.bitwise_or(other_text_mask, region_mask)
                if bubble_box is not None:
                    bubble_area = _box_area([bubble_box[0], bubble_box[1], bubble_box[2] - bubble_box[0], bubble_box[3] - bubble_box[1]])
                    base_area = _box_area([base_box[0], base_box[1], base_box[2] - base_box[0], base_box[3] - base_box[1]])
                    if base_area and bubble_area >= base_area * 0.6:
                        limited = _intersect_box(render_box, bubble_box)
                        render_box = limited or bubble_box
                    else:
                        render_box = base_box
                else:
                    render_box = base_box
            else:
                render_box = base_box
            render_box = _shrink_box(render_box, max(2, int(min(w, h) * 0.03)))
            render = region.get("render") or {}
            if not isinstance(render, dict):
                render = {}
            region_font = render.get("font") or font_name
            if _has_cjk(text) and _is_cjk_unsupported_font(region_font):
                region_font = font_name
            render_regions.append((text, render_box, region_font))

        if render_regions:
            if bubble_text_mask is not None and bubble_text_mask.any():
                if cv2 is not None and np is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    bubble_text_mask = cv2.erode(bubble_text_mask, kernel, iterations=1)
                working = _apply_white_mask(working, bubble_text_mask)
            if other_text_mask is not None and other_text_mask.any():
                working = _apply_text_removal(working, other_text_mask, inpaint_mode, use_gpu)

        draw = ImageDraw.Draw(working)
        median_height = 0
        preferred_size = None
        if render_regions:
            heights = sorted(max(1, box[3] - box[1]) for _, box, _ in render_regions)
            median_height = heights[len(heights) // 2]
            preferred_size = max(12, int(median_height * 0.33))
        for text, box, region_font in render_regions:
            x0, y0, x1, y1 = box
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            local_preferred = None
            if preferred_size and h >= preferred_size * 2:
                local_preferred = min(preferred_size, max(12, int(h * 0.7)))
            base_font = _fit_font(draw, text, w, h, region_font, preferred_size=local_preferred)
            best_lines = _wrap_text(draw, text, base_font, w)
            best_font = base_font
            best_height = sum(_text_height(base_font, line) for line in best_lines)
            max_lines = max(1, min(6, int(h / max(1, _text_height(base_font, "A"))) + 1))
            for lines_count in range(2, max_lines + 1):
                test_lines = _wrap_text(draw, text, base_font, w, max_lines=lines_count)
                test_height = sum(_text_height(base_font, line) for line in test_lines)
                if test_height <= h and len(test_lines) > len(best_lines):
                    best_lines = test_lines
                    best_height = test_height
            if best_height > h and local_preferred:
                for size in range(local_preferred - 1, 9, -1):
                    test_font = _load_font(_find_font_path(region_font), size, _sample_char(text))
                    test_lines = _wrap_text(draw, text, test_font, w, max_lines=max_lines)
                    test_height = sum(_text_height(test_font, line) for line in test_lines)
                    if test_height <= h:
                        best_font = test_font
                        best_lines = test_lines
                        best_height = test_height
                        break
            offset_y = y0 + max(0, (h - best_height) // 2)
            for line in best_lines:
                bbox = best_font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                offset_x = x0 + max(0, (w - line_width) // 2) - bbox[0]
                draw.text((offset_x, offset_y - bbox[1]), line, fill=(0, 0, 0), font=best_font)
                offset_y += line_height
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        working.save(output_path)


def _apply_text_removal(image, text_mask, mode: str, use_gpu: bool):
    if text_mask is None:
        return image
    mode = (mode or "fast").lower()
    if cv2 is not None and np is not None:
        bubble_fill = _white_bubble_fill(image, text_mask)
        if bubble_fill is not None:
            return bubble_fill
    if mode == "ai":
        if cv2 is not None and np is not None:
            total = text_mask.size
            ratio = float((text_mask > 0).sum()) / max(1, total)
            if ratio > 0.03:
                mode = "fast"
            else:
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                masked = gray[text_mask > 0]
                if masked.size > 0:
                    white_ratio = float((masked > 200).mean())
                    if white_ratio > 0.7:
                        mode = "fast"
        if not use_gpu:
            mode = "fast"
        try:
            from app.render.inpaint_ai import ai_inpaint
            return ai_inpaint(image, text_mask, use_gpu=use_gpu)
        except Exception:
            mode = "fast"
    if mode == "fast" and cv2 is not None and np is not None:
        img_np = np.array(image)
        kernel_size = max(3, int(max(text_mask.shape) * 0.0015))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(text_mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
        rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return _apply_white_mask(image, text_mask)


def _apply_white_mask(image, text_mask):
    if cv2 is None or np is None:
        draw = ImageDraw.Draw(image)
        return image
    img_np = np.array(image)
    img_np[text_mask > 0] = (255, 255, 255)
    return Image.fromarray(img_np)


def _region_masks(
    img_np,
    pad_box: Tuple[int, int, int, int],
    bbox: List[int],
    polygon,
):
    if cv2 is None or np is None:
        return None, None
    x0, y0, x1, y1 = pad_box
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)
    if white.mean() < 5:
        _, white = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)

    text_mask = np.zeros((white.shape[0], white.shape[1]), dtype=np.uint8)
    polys = _normalize_polygons(polygon)
    if polys:
        try:
            for poly in polys:
                poly = np.array(poly, dtype=np.int32)
                poly[:, 0] = poly[:, 0] - x0
                poly[:, 1] = poly[:, 1] - y0
                cv2.fillPoly(text_mask, [poly], 255)
        except Exception:
            text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    else:
        text_mask = _rect_mask(text_mask.shape, bbox, x0, y0)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    bubble_box, bubble_mask = _find_bubble_box(white, text_mask)
    full_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = text_mask
    if bubble_box:
        bx0, by0, bx1, by1 = bubble_box
        bubble_box = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
        bubble_box = _ensure_box_contains(bubble_box, pad_box)
    if bubble_mask is not None:
        full_bubble = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        full_bubble[y0:y1, x0:x1] = bubble_mask
        bubble_mask = full_bubble
    return full_mask, bubble_box, bubble_mask


def _ensure_box_contains(outer, inner):
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        min(ox0, ix0),
        min(oy0, iy0),
        max(ox1, ix1),
        max(oy1, iy1),
    )


def _rect_mask(shape, bbox, x0, y0):
    mask = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    rx0 = max(0, x - x0)
    ry0 = max(0, y - y0)
    rx1 = min(shape[1], rx0 + w)
    ry1 = min(shape[0], ry0 + h)
    cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), 255, thickness=-1)
    return mask


def _find_bubble_box(white_mask, text_mask):
    num_labels, labels = cv2.connectedComponents(white_mask)
    best_label = 0
    best_overlap = 0
    if num_labels <= 1:
        return None, None
    for label in range(1, num_labels):
        overlap = np.sum((labels == label) & (text_mask > 0))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_label == 0:
        return None, None
    ys, xs = np.where(labels == best_label)
    if ys.size == 0 or xs.size == 0:
        return None, None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    area = (x1 - x0) * (y1 - y0)
    total_area = white_mask.shape[0] * white_mask.shape[1]
    if total_area > 0 and area / total_area > 0.6:
        return None, None
    bubble_mask = (labels == best_label).astype(np.uint8) * 255
    return (x0, y0, x1, y1), bubble_mask


def _intersect_box(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _limit_box(box, limit):
    bx0, by0, bx1, by1 = box
    lx0, ly0, lx1, ly1 = limit
    limited = _intersect_box(box, limit)
    if not limited:
        return box
    # If the limit is significantly smaller, use it; otherwise keep the original.
    bw = bx1 - bx0
    bh = by1 - by0
    lw = limited[2] - limited[0]
    lh = limited[3] - limited[1]
    if lw * lh < bw * bh * 0.85:
        return limited
    return box


def _normalize_polygons(polygon) -> List[List[List[float]]]:
    if polygon is None:
        return []
    if hasattr(polygon, "tolist"):
        try:
            polygon = polygon.tolist()
        except Exception:
            return []
    if isinstance(polygon, (list, tuple)) and len(polygon) > 0:
        first = polygon[0]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (int, float)):
            return [polygon]
        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            return [p for p in polygon if p]
    return []


def _polygon_bounds(polygon) -> Tuple[int, int, int, int] | None:
    polys = _normalize_polygons(polygon)
    if not polys:
        return None
    xs = []
    ys = []
    for poly in polys:
        for point in poly:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _box_area(box) -> int:
    if not box:
        return 0
    if len(box) == 4:
        return int(max(1, box[2]) * max(1, box[3]))
    return 0


def _shrink_box(box, padding):
    x0, y0, x1, y1 = box
    x0 = x0 + padding
    y0 = y0 + padding
    x1 = x1 - padding
    y1 = y1 - padding
    if x1 <= x0 or y1 <= y0:
        return box
    return (x0, y0, x1, y1)


def _wrap_text(draw, text: str, font, max_width: int, max_lines: int | None = None) -> List[str]:
    words = _tokenize_text(text)
    lines: List[str] = []
    current = ""
    has_space = " " in text
    for idx, ch in enumerate(words):
        candidate = (current + " " + ch).strip() if has_space else current + ch
        if _is_punct_only(ch) and current:
            current = f"{current}{ch}"
            continue
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
            if max_lines is not None and len(lines) >= max_lines - 1:
                remaining = words[idx + 1 :]
                if remaining:
                    current = _join_tokens([current] + remaining, has_space)
                break
    if current:
        lines.append(current)
    lines = _fix_leading_punct(lines)
    return lines


def _join_tokens(tokens: List[str], has_space: bool) -> str:
    if not tokens:
        return ""
    if not has_space:
        return "".join(tokens)
    combined = tokens[0]
    for token in tokens[1:]:
        if _is_punct_only(token):
            combined = f"{combined}{token}"
        else:
            combined = f"{combined} {token}"
    return combined


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = " ".join(part for part in cleaned.split("\n") if part.strip())
    if _has_cjk(cleaned):
        cleaned = re.sub(r"[.．]{2,}", "…", cleaned)
        cleaned = re.sub(r"…{2,}", "…", cleaned)
        cleaned = cleaned.replace("．", "。")
        cleaned = cleaned.replace(":", "：").replace(";", "；")
        cleaned = cleaned.replace("!", "！").replace("?", "？")
        cleaned = cleaned.replace(".", "。")
        cleaned = re.sub(r"\s*([。，、！？：；…])\s*", r"\1", cleaned)
    return cleaned.strip()


def _tokenize_text(text: str) -> List[str]:
    if " " in text:
        parts = [p for p in text.split(" ") if p]
        tokens: List[str] = []
        for part in parts:
            if _is_punct_only(part) and tokens:
                tokens[-1] = f"{tokens[-1]}{part}"
            else:
                tokens.append(part)
        return tokens
    tokens = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch in {".", "．"}:
            end = index
            while end < len(text) and text[end] in {".", "．"}:
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if ch == "…":
            end = index
            while end < len(text) and text[end] == "…":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_punct_char(ch) and tokens:
            tokens[-1] = f"{tokens[-1]}{ch}"
        else:
            tokens.append(ch)
        index += 1
    return tokens


def _is_punct_char(ch: str) -> bool:
    return ch in {
        "。",
        "．",
        "，",
        "、",
        "！",
        "？",
        "：",
        "；",
        "…",
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        "·",
        "—",
        "～",
        "…",
    }


def _is_punct_only(text: str) -> bool:
    if not text:
        return True
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    return all(_is_punct_char(ch) for ch in stripped)


def _fix_leading_punct(lines: List[str]) -> List[str]:
    if len(lines) <= 1:
        return lines
    fixed: List[str] = [lines[0]]
    for line in lines[1:]:
        line = line.lstrip()
        if line and _is_punct_char(line[0]):
            fixed[-1] = f"{fixed[-1]}{line[0]}"
            remainder = line[1:].lstrip()
            if remainder:
                fixed.append(remainder)
        else:
            fixed.append(line)
    return fixed


def _text_height(font, text: str) -> int:
    bbox = font.getbbox(text)
    return max(1, int(bbox[3] - bbox[1]))


def _fill_padding(w: int, h: int) -> int:
    base = max(12, int(min(w, h) * 0.35), int(max(w, h) * 0.1))
    return min(base, 80)


def _text_padding(w: int, h: int) -> int:
    base = max(6, int(min(w, h) * 0.12), int(max(w, h) * 0.03))
    return min(base, 28)


def _fit_font(
    draw,
    text: str,
    max_width: int,
    max_height: int,
    font_name: str,
    preferred_size: int | None = None,
):
    font_path = _find_font_path(font_name)
    sample = _sample_char(text)
    start_size = min(72, max(14, int(max_height * 0.75)))
    min_size = max(10, int(max_height * 0.18))
    if preferred_size is not None:
        target = max(min_size, min(preferred_size, start_size))
        font = _load_font(font_path, target, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    for size in range(start_size, min_size - 1, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    for size in range(min_size - 1, 9, -1):
        font = _load_font(font_path, size, sample)
        lines = _wrap_text(draw, text, font, max_width)
        total_height = sum(font.getbbox(line)[3] for line in lines)
        if total_height <= max_height:
            return font
    return _load_font(font_path, 10, sample)


def _load_font(font_path: str | None, size: int, sample: str):
    if ImageFont is None:
        raise RuntimeError("Pillow is not installed.")
    candidates = []
    if font_path and os.path.exists(font_path):
        candidates.append(font_path)
    candidates.extend(_fallback_font_paths())
    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            font = ImageFont.truetype(path, size=size)
        except Exception:
            continue
        if _font_supports_text(font, sample):
            return font
    return ImageFont.load_default()


def _font_supports_text(font, text: str) -> bool:
    if not text:
        return True
    try:
        mask = font.getmask(text)
        return mask.getbbox() is not None
    except Exception:
        return False


def _sample_char(text: str) -> str:
    for ch in text:
        if _is_cjk(ch):
            return ch
    return "A"


def _find_font_path(font_name: str) -> str | None:
    if not font_name:
        return None
    if os.path.isfile(font_name):
        return font_name
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return None
    lowered = font_name.lower().replace(" ", "")
    fallback_fonts = [
        "Noto Sans CJK",
        "Microsoft YaHei",
        "SimSun",
        "MS Gothic",
        "Yu Gothic",
        "Meiryo",
    ]
    for entry in os.listdir(fonts_dir):
        name, ext = os.path.splitext(entry)
        if ext.lower() not in {".ttf", ".otf", ".ttc"}:
            continue
        if lowered in name.lower().replace(" ", ""):
            return os.path.join(fonts_dir, entry)
    for font in fallback_fonts:
        lowered = font.lower().replace(" ", "")
        for entry in os.listdir(fonts_dir):
            name, ext = os.path.splitext(entry)
            if ext.lower() not in {".ttf", ".otf", ".ttc"}:
                continue
            if lowered in name.lower().replace(" ", ""):
                return os.path.join(fonts_dir, entry)
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            return candidate
    return None


def _fallback_font_paths() -> List[str]:
    fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    if not os.path.isdir(fonts_dir):
        return []
    known_files = [
        "msyh.ttc",
        "msyhbd.ttc",
        "simsun.ttc",
        "simhei.ttf",
        "msgothic.ttc",
        "yugothic.ttf",
        "yugothicui.ttf",
        "meiryo.ttc",
    ]
    paths = []
    for fname in known_files:
        candidate = os.path.join(fonts_dir, fname)
        if os.path.isfile(candidate):
            paths.append(candidate)
    return paths


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(ch) for ch in text)


def _is_cjk_unsupported_font(font_name: str) -> bool:
    if not font_name:
        return False
    name = font_name.lower()
    return "gothic" in name or "meiryo" in name


def _white_bubble_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    ys, xs = np.where(text_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 12
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(text_mask.shape[1] - 1, x1 + pad)
    y1 = min(text_mask.shape[0] - 1, y1 + pad)
    img_np = np.array(image)
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean > 220 and std < 18:
        filled = _apply_white_mask(image, text_mask)
        return filled
    return None
