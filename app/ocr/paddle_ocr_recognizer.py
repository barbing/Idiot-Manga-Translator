# -*- coding: utf-8 -*-
"""PaddleOCR recognition wrapper."""
from __future__ import annotations
import os
from typing import Optional

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class PaddleOcrRecognizer:
    def __init__(self, use_gpu: bool) -> None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR is not installed.")
        try:
            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": 0, "FLAGS_enable_onednn": 0})
        except Exception:
            pass
        self._engine = PaddleOCR(
            det=False,
            rec=True,
            cls=False,
            lang="japan",
            use_gpu=use_gpu,
            ir_optim=False,
            use_tensorrt=False,
        )

    def recognize(self, image) -> str:
        try:
            import numpy as np
        except ImportError:
            np = None
        candidates = []
        for angle in (0, 90, 270):
            img = image
            if angle and hasattr(image, "rotate"):
                img = image.rotate(angle, expand=True)
            if np is not None and not isinstance(img, (str, bytes, list)):
                try:
                    img = np.array(img)
                except Exception:
                    pass
            text = _run_rec(self._engine, img)
            score = _text_score(text)
            candidates.append((score, text))
        candidates.sort(reverse=True)
        best = candidates[0][1] if candidates else ""
        return best.strip()


def _run_rec(engine, image) -> str:
    result = engine.ocr(image, det=False, rec=True, cls=False)
    if not result:
        return ""
    first = result[0]
    if isinstance(first, list) and first:
        item = first[0]
        if isinstance(item, (list, tuple)) and item:
            return str(item[0]).strip()
    return ""


def _text_score(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    jp = sum(1 for ch in text if _is_japanese(ch))
    return jp * 2 + total


def _is_japanese(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana/Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified
    )
