"""Minimal Ogkalu ONNX helpers required by BubbleDetection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_labels(config_path: Path) -> dict[int, str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    labels = config.get("id2label") or {
        0: "bubble",
        1: "text_bubble",
        2: "text_free",
    }
    return {int(key): str(value) for key, value in labels.items()}


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    tensor = np.transpose(array, (2, 0, 1))[None]
    # This ONNX export decodes boxes with target sizes in width/height order.
    original_target_sizes = np.array([[width, height]], dtype=np.int64)
    return tensor, original_target_sizes


def normalize_box(box: Any, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]
