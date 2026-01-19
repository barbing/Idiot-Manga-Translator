# -*- coding: utf-8 -*-
"""Font detection helpers."""
from __future__ import annotations
from typing import Optional


class FontDetection:
    def __init__(
        self,
        mode: str = "off",
    ) -> None:
        self._mode = (mode or "off").lower()
        self._impl = _HeuristicFontDetector() if self._mode == "heuristic" else None

    def detect(self, image) -> Optional[str]:
        if not self._impl:
            return None
        return self._impl.detect(image)


class _HeuristicFontDetector:
    def __init__(self) -> None:
        self._fonts = {
            "thin": "SimSun",
            "normal": "Noto Sans CJK",
            "bold": "Microsoft YaHei",
            "vertical": "Microsoft YaHei",
        }

    def detect(self, image) -> Optional[str]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        if image is None:
            return None
        w, h = image.size
        if w <= 0 or h <= 0:
            return None
        if h > w * 1.3:
            return self._fonts["vertical"]
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink_ratio = float((bw == 0).mean())
        if ink_ratio < 0.08:
            return self._fonts["thin"]
        if ink_ratio > 0.22:
            return self._fonts["bold"]
        return self._fonts["normal"]
