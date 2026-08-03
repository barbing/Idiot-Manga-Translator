"""Production Kitsumed ONNX helpers required by BubbleDetection.

This module preserves the preprocessing, decoding, and hashing functions from
the accepted Phase 4 provider adapter without its diagnostic CLI.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padw = (size - new_width) / 2.0
    padh = (size - new_height) / 2.0
    left = int(round(padw - 0.1))
    top = int(round(padh - 0.1))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], {
        "scale": scale,
        "padw": padw,
        "padh": padh,
        "input_size": size,
        "resized": [new_width, new_height],
        "left": left,
        "top": top,
    }


def parse_names(raw: str) -> dict[int, str]:
    try:
        data = ast.literal_eval(raw)
    except Exception:
        data = {0: "speech bubble"}
    return {int(key): str(value) for key, value in data.items()}


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    order = sorted(
        range(len(detections)),
        key=lambda index: detections[index]["confidence"],
        reverse=True,
    )
    keep: list[int] = []
    while order:
        index = order.pop(0)
        keep.append(index)
        order = [
            other
            for other in order
            if _box_iou(
                detections[index]["input_bbox_xyxy"],
                detections[other]["input_bbox_xyxy"],
            )
            < threshold
        ]
    return [detections[index] for index in keep]


def decode_outputs(
    raw0: np.ndarray,
    raw1: np.ndarray,
    prep: dict[str, Any],
    original_shape: tuple[int, int],
    class_names: dict[int, str],
    conf_threshold: float,
    nms_threshold: float,
    mask_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    array = raw0[0]
    predictions = array.T if array.shape[0] < array.shape[1] else array
    proto = raw1[0]
    channels, mask_h, mask_w = proto.shape
    input_size = int(prep["input_size"])
    original_h, original_w = original_shape
    scale = float(prep["scale"])
    padw = float(prep["padw"])
    padh = float(prep["padh"])
    class_count = max(1, len(class_names))

    candidates: list[dict[str, Any]] = []
    for anchor_index, row in enumerate(predictions):
        cx, cy, width, height = [float(value) for value in row[:4]]
        scores = row[4 : 4 + class_count]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < conf_threshold:
            continue
        x1 = max(0.0, cx - width / 2.0)
        y1 = max(0.0, cy - height / 2.0)
        x2 = min(float(input_size), cx + width / 2.0)
        y2 = min(float(input_size), cy + height / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            {
                "anchor_index": int(anchor_index),
                "input_bbox_xyxy": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "mask_coefficients": row[
                    4 + class_count : 4 + class_count + channels
                ].astype(np.float32),
            }
        )

    kept = _nms(candidates, nms_threshold)
    proto_flat = proto.reshape(channels, -1)
    detections: list[dict[str, Any]] = []
    for index, detection in enumerate(kept):
        coefficients = detection.pop("mask_coefficients")
        mask = sigmoid(coefficients @ proto_flat).reshape(mask_h, mask_w)
        box = detection["input_bbox_xyxy"]
        sx, sy = mask_w / input_size, mask_h / input_size
        px1 = int(max(0, min(mask_w, math.floor(box[0] * sx))))
        py1 = int(max(0, min(mask_h, math.floor(box[1] * sy))))
        px2 = int(max(0, min(mask_w, math.ceil(box[2] * sx))))
        py2 = int(max(0, min(mask_h, math.ceil(box[3] * sy))))
        cropped = np.zeros_like(mask)
        cropped[py1:py2, px1:px2] = mask[py1:py2, px1:px2]
        input_mask = cv2.resize(
            cropped,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        left, top = int(prep["left"]), int(prep["top"])
        resized_w, resized_h = prep["resized"]
        unpadded = input_mask[top : top + resized_h, left : left + resized_w]
        original_mask = cv2.resize(
            unpadded,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
        binary_mask = original_mask > mask_threshold
        ox1 = max(0.0, min(float(original_w), (box[0] - padw) / scale))
        ox2 = max(0.0, min(float(original_w), (box[2] - padw) / scale))
        oy1 = max(0.0, min(float(original_h), (box[1] - padh) / scale))
        oy2 = max(0.0, min(float(original_h), (box[3] - padh) / scale))
        detection["detection_id"] = f"b{index:03d}"
        detection["bbox_xyxy"] = [
            round(ox1, 1),
            round(oy1, 1),
            round(ox2, 1),
            round(oy2, 1),
        ]
        detection["confidence"] = round(float(detection["confidence"]), 6)
        detection["mask_area_px"] = int(binary_mask.sum())
        detection["mask_bbox_xyxy"] = None
        if binary_mask.any():
            ys, xs = np.where(binary_mask)
            detection["mask_bbox_xyxy"] = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]
        detection["mask"] = binary_mask
        detections.append(detection)
    return detections, {
        "candidates_after_confidence": len(candidates),
        "kept_after_nms": len(detections),
    }
