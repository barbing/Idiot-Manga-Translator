# -*- coding: utf-8 -*-
"""PaddleOCR text detection wrapper."""
from __future__ import annotations
import os
from typing import List, Tuple

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class PaddleTextDetector:
    def __init__(self, use_gpu: bool) -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed.")
        try:
            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_onednn": 0})
        except Exception:
            pass
        self._detector = PaddleOCR(
            det=True,
            rec=False,
            cls=False,
            lang="japan",
            use_gpu=use_gpu,
            ir_optim=False,
            use_tensorrt=False,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=1.7,
            det_limit_side_len=1280,
        )

    def detect(self, image_path: str) -> List[Tuple[List[List[float]], float]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for detection.") from exc
        try:
            import numpy as np
        except Exception:
            np = None

        image = cv2.imread(image_path)
        if image is None and np is not None:
            try:
                data = np.fromfile(image_path, dtype=np.uint8)
                if data.size:
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                image = None
        if image is None:
            return []
        dt_boxes, _ = self._detector.text_detector(image)
        if dt_boxes is None:
            return []
        output = []
        for item in dt_boxes:
            if hasattr(item, "tolist"):
                polygon = item.tolist()
            else:
                polygon = item
            output.append((polygon, 1.0))
        return output
