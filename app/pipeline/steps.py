# -*- coding: utf-8 -*-
"""Pipeline helper steps (placeholders for now)."""
from __future__ import annotations
import os
from typing import Dict, Tuple


try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional
    Image = None


def read_image_size(path: str) -> Tuple[int, int]:
    if Image is None:
        return 0, 0
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return 0, 0


def build_page_record(image_path: str, page_id: str, regions: list, output_path: str = "") -> Dict[str, object]:
    width, height = read_image_size(image_path)
    return {
        "page_id": page_id,
        "image_path": image_path,
        "output_path": output_path,
        "width": width,
        "height": height,
        "regions": regions,
    }


def build_output_path(export_dir: str, filename: str, suffix: str) -> str:
    name, ext = os.path.splitext(filename)
    return os.path.join(export_dir, f"{name}{suffix}{ext}")
