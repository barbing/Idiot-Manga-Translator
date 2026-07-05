# -*- coding: utf-8 -*-
"""ComicTextDetector wrapper."""
from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComicTextSegmentationResult:
    """ComicTextDetector boxes plus dense text foreground masks."""

    detections: List[Tuple[List[List[float]], float]]
    raw_mask: Any = field(default=None, repr=False, compare=False)
    refined_mask: Any = field(default=None, repr=False, compare=False)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    image_size: tuple[int, int] | None = None
    provider: str = "ComicTextDetector"
    backend: str = ""
    threshold_used: int = 30
    runtime_ms: float | None = None
    text_pixel_count: int = 0
    connected_component_stats: dict[str, Any] = field(default_factory=dict)
    keep_undetected_mask: bool = True
    confidence: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _third_party_root() -> str:
    return os.path.join(_repo_root(), "app", "third_party", "comic-text-detector")


def _default_model_dir() -> str:
    return os.path.join(_repo_root(), "models", "comic-text-detector")


def _select_model_path(model_dir: str, use_gpu: bool) -> str:
    # This model is user-downloaded and resides in the models/ folder.
    # We do NOT check system paths to avoid conflict.
    
    # 1. Portable Model Check
    portable_model_root = None
    local_model_path = os.path.join(os.getcwd(), "models", "comic-text-detector")
    if os.path.exists(os.path.join(local_model_path, "comictextdetector.pt")):
        portable_model_root = local_model_path

    # Determine effective root (Allow override, then portable, then default arg)
    effective_model_root = portable_model_root or model_dir

    override = os.environ.get("MT_COMICTEXT_MODEL_PATH", "").strip()
    if override:
        return override

    onnx_path = os.path.join(effective_model_root, "comictextdetector.pt.onnx")
    pt_path = os.path.join(effective_model_root, "comictextdetector.pt")

    if use_gpu and os.path.isfile(pt_path):
        logger.info(f"Selected GPU model: {pt_path}")
        return pt_path
    if os.path.isfile(onnx_path):
        logger.info(f"Selected ONNX model: {onnx_path}")
        return onnx_path
    if os.path.isfile(pt_path):
        return pt_path
    return onnx_path


def _bbox_to_polygon(xyxy: list) -> List[List[float]]:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    if x1 < x0 or y1 < y0:
        x1 = x0 + max(1.0, x1)
        y1 = y0 + max(1.0, y1)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _normalize_line(line) -> List[List[float]]:
    if line is None:
        return []
    if len(line) == 8 and not hasattr(line[0], "__len__"):
        return [
            [float(line[0]), float(line[1])],
            [float(line[2]), float(line[3])],
            [float(line[4]), float(line[5])],
            [float(line[6]), float(line[7])],
        ]
    output = []
    for point in line:
        if point is None:
            continue
        if hasattr(point, "__len__") and len(point) >= 2:
            output.append([float(point[0]), float(point[1])])
    return output


class ComicTextDetector:
    def __init__(self, use_gpu: bool, model_dir: str | None = None) -> None:
        repo_root = _third_party_root()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        _ensure_utils_package(repo_root)
        self.merge_mode = "none"

        try:
            import torch
        except Exception:
            torch = None

        model_root = model_dir or os.environ.get("MT_COMICTEXT_MODEL_DIR", "").strip() or _default_model_dir()
        self._model_path = _select_model_path(model_root, use_gpu)
        if not os.path.isfile(self._model_path):
            raise RuntimeError(
                "ComicTextDetector model not found. Download comictextdetector.pt.onnx (CPU) or "
                "comictextdetector.pt (GPU) from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 "
                f"and place it under {model_root}."
            )

        device = "cpu"
        if (
            use_gpu
            and torch is not None
            and torch.cuda.is_available()
            and self._model_path.endswith(".pt")
        ):
            device = "cuda"

        from inference import TextDetector
        from utils.textmask import REFINEMASK_INPAINT

        self._refine_mode = REFINEMASK_INPAINT
        input_size = int(os.environ.get("MT_COMICTEXT_INPUT_SIZE", "640"))
        self._detector = TextDetector(
            model_path=self._model_path,
            input_size=input_size,
            device=device,
            act="leaky",
            conf_thresh=0.5,
            nms_thresh=0.4,
        )
        logger.info(f"ComicTextDetector initialized. GPU={use_gpu}, Device={device}")

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if hasattr(self, "_detector"):
            del self._detector
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        except Exception:
            pass

    def detect(self, image_path: str, input_size: int = 1024) -> List[Tuple[List[List[float]], float]]:
        image = _read_image(image_path)
        if image is None:
            return []
        return self.detect_image(image, input_size)

    def detect_image(self, image, input_size: int = 1024) -> List[Tuple[List[List[float]], float]]:
        if image is None:
            return []
        # Update input size if changed
        if input_size != self._detector.input_size:
             self._detector.input_size = (input_size, input_size)
             
        _, _, blk_list = self._detector(
            image,
            refine_mode=self._refine_mode,
            keep_undetected_mask=False,
        )
        return _detections_from_blocks(blk_list)

    def detect_with_segmentation(
        self,
        image_path: str,
        input_size: int = 1024,
        *,
        keep_undetected_mask: bool = True,
    ) -> ComicTextSegmentationResult:
        image = _read_image(image_path)
        if image is None:
            return ComicTextSegmentationResult(
                detections=[],
                keep_undetected_mask=keep_undetected_mask,
                backend=os.path.basename(str(getattr(self, "_model_path", "") or "")),
                provenance={"image_path": image_path, "status": "image_unavailable"},
            )
        return self.detect_image_with_segmentation(
            image,
            input_size=input_size,
            keep_undetected_mask=keep_undetected_mask,
            provenance={"image_path": image_path},
        )

    def detect_image_with_segmentation(
        self,
        image,
        input_size: int = 1024,
        *,
        keep_undetected_mask: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> ComicTextSegmentationResult:
        """Return CTD detections and raw/refined text masks without changing detect_image."""

        if image is None:
            return ComicTextSegmentationResult(
                detections=[],
                keep_undetected_mask=keep_undetected_mask,
                backend=os.path.basename(str(getattr(self, "_model_path", "") or "")),
                provenance={"status": "image_unavailable", **(provenance or {})},
            )
        if input_size != self._detector.input_size:
            self._detector.input_size = (input_size, input_size)
        started = time.time()
        mask, mask_refined, blk_list = self._detector(
            image,
            refine_mode=self._refine_mode,
            keep_undetected_mask=keep_undetected_mask,
        )
        mask_refined, refinement_recovery = _recover_line_continuation_refinement_gaps(
            image,
            mask,
            mask_refined,
            blk_list,
        )
        detections = _detections_from_blocks(blk_list)
        height, width = _image_hw(image)
        confidence = _confidence_stats(blk_list)
        if _line_continuation_recovery_is_material(refinement_recovery):
            confidence["line_continuation_refinement_recovery"] = refinement_recovery
        result_provenance = {
            "model_path": str(getattr(self, "_model_path", "") or ""),
            **(provenance or {}),
        }
        if _line_continuation_recovery_is_material(refinement_recovery):
            result_provenance["line_continuation_refinement_recovery"] = refinement_recovery
        return ComicTextSegmentationResult(
            detections=detections,
            raw_mask=mask,
            refined_mask=mask_refined,
            blocks=[_block_audit_dict(blk, index) for index, blk in enumerate(blk_list or [])],
            image_size=(width, height) if width > 0 and height > 0 else None,
            backend=os.path.basename(str(getattr(self, "_model_path", "") or "")),
            threshold_used=30,
            runtime_ms=round((time.time() - started) * 1000.0, 3),
            text_pixel_count=_mask_text_pixels(mask_refined),
            connected_component_stats=_mask_component_stats(mask_refined),
            keep_undetected_mask=keep_undetected_mask,
            confidence=confidence,
            provenance=result_provenance,
        )


def _read_image(image_path: str):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    image = cv2.imread(image_path)
    if image is None:
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            if data.size:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
    return image


def _detections_from_blocks(blk_list) -> List[Tuple[List[List[float]], float]]:
    output: List[Tuple[List[List[float]], float]] = []
    for blk in blk_list or []:
        score = getattr(blk, "prob", 1.0)
        line_box = _lines_bounds(getattr(blk, "lines", []) or [])
        if line_box:
            output.append((_bbox_to_polygon(line_box), score))
            continue
        xyxy = getattr(blk, "xyxy", None)
        if xyxy:
            output.append((_bbox_to_polygon(xyxy), score))
    return output


def _image_hw(image) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def _mask_text_pixels(mask) -> int:
    try:
        import numpy as np

        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = np.any(arr > 30, axis=2)
        else:
            arr = arr > 30
        return int(np.count_nonzero(arr))
    except Exception:
        return 0


def _mask_component_stats(mask) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np

        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = np.any(arr > 30, axis=2)
        elif arr.ndim == 2:
            arr = arr > 30
        else:
            return {"component_count": 0, "largest_component_pixels": 0}
        labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            arr.astype("uint8"),
            connectivity=8,
        )
        areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]
        return {
            "component_count": len(areas),
            "largest_component_pixels": max(areas) if areas else 0,
            "total_component_pixels": int(sum(areas)),
        }
    except Exception:
        pixels = _mask_text_pixels(mask)
        return {
            "component_count": 1 if pixels > 0 else 0,
            "largest_component_pixels": pixels,
            "total_component_pixels": pixels,
        }


def _block_audit_dict(blk, index: int) -> dict[str, Any]:
    xyxy = getattr(blk, "xyxy", None)
    line_box = _lines_bounds(getattr(blk, "lines", []) or [])
    return {
        "block_index": index,
        "prob": float(getattr(blk, "prob", 1.0) or 0.0),
        "xyxy": [float(v) for v in xyxy] if xyxy is not None else [],
        "line_bbox": line_box or [],
    }


def _confidence_stats(blk_list) -> dict[str, Any]:
    scores = [float(getattr(blk, "prob", 1.0) or 0.0) for blk in (blk_list or [])]
    if not scores:
        return {"block_count": 0}
    return {
        "block_count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / float(len(scores)),
    }


def _recover_line_continuation_refinement_gaps(image, raw_mask, refined_mask, blocks) -> tuple[Any, dict[str, Any]]:
    """Recover CTD raw-mask text strokes that refinement dropped inside known text blocks.

    ComicTextDetector raw masks sometimes contain long vertical punctuation strokes
    that the vendored refinement step drops because it thresholds a tight line
    window. This repair stays inside the detector contract: admitted pixels remain
    attached to an existing CTD text block and only improve the refined foreground
    mask consumed by TextAreaPlan/CleanupMask.
    """

    audit: dict[str, Any] = {
        "components_considered": 0,
        "components_recovered": 0,
        "pixels_recovered": 0,
    }
    if image is None or raw_mask is None or refined_mask is None:
        return refined_mask, audit
    try:
        import cv2
        import numpy as np
    except Exception:
        audit["status"] = "dependencies_unavailable"
        return refined_mask, audit

    raw_bool = _mask_bool(raw_mask, threshold=0)
    refined_bool = _mask_bool(refined_mask, threshold=30)
    if raw_bool is None or refined_bool is None or raw_bool.shape != refined_bool.shape:
        audit["status"] = "mask_unavailable_or_mismatched"
        return refined_mask, audit

    residual = np.logical_and(raw_bool, np.logical_not(refined_bool))
    if not np.any(residual):
        return refined_mask, audit

    height, width = raw_bool.shape
    gray = _image_gray(image)
    admitted = np.zeros_like(raw_bool, dtype=bool)
    for block in blocks or []:
        block_bbox = _coerce_bbox(getattr(block, "xyxy", None), width, height)
        if block_bbox is None:
            continue
        line_bbox = _coerce_bbox(
            _lines_bounds(getattr(block, "lines", []) or []),
            width,
            height,
        ) or block_bbox
        vertical = _block_is_vertical(block, block_bbox)
        search_bbox = _line_continuation_search_bbox(
            block_bbox,
            line_bbox,
            width,
            height,
            vertical,
        )
        x0, y0, x1, y1 = search_bbox
        roi = residual[y0:y1, x0:x1]
        if not np.any(roi):
            continue
        labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            roi.astype("uint8"),
            connectivity=8,
        )
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 18:
                continue
            comp_x = x0 + int(stats[label, cv2.CC_STAT_LEFT])
            comp_y = y0 + int(stats[label, cv2.CC_STAT_TOP])
            comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            comp_bbox = (comp_x, comp_y, comp_x + comp_w, comp_y + comp_h)
            local_x = int(stats[label, cv2.CC_STAT_LEFT])
            local_y = int(stats[label, cv2.CC_STAT_TOP])
            component = labels[local_y : local_y + comp_h, local_x : local_x + comp_w] == label
            audit["components_considered"] += 1
            if not _is_recoverable_line_continuation_component(
                comp_bbox=comp_bbox,
                area=area,
                component=component,
                search_bbox=search_bbox,
                block_bbox=block_bbox,
                line_bbox=line_bbox,
                vertical=vertical,
                gray=gray,
            ):
                continue
            page_component = admitted[comp_y : comp_y + comp_h, comp_x : comp_x + comp_w]
            page_component[component] = True
            admitted[comp_y : comp_y + comp_h, comp_x : comp_x + comp_w] = page_component
            audit["components_recovered"] += 1
            audit["pixels_recovered"] += area

    if not np.any(admitted):
        return refined_mask, audit

    try:
        repaired = np.asarray(refined_mask).copy()
        if repaired.ndim == 3:
            repaired[admitted, :] = 255
        elif repaired.ndim == 2:
            repaired[admitted] = 255
        else:
            repaired = np.where(np.logical_or(refined_bool, admitted), 255, 0).astype("uint8")
        return repaired, audit
    except Exception as exc:
        audit["status"] = f"repair_failed:{type(exc).__name__}"
        return refined_mask, audit


def _mask_bool(mask, *, threshold: int = 30) -> Any | None:
    try:
        import numpy as np

        arr = np.asarray(mask)
        if arr.ndim == 3:
            return np.any(arr > threshold, axis=2)
        if arr.ndim == 2:
            return arr > threshold
    except Exception:
        return None
    return None


def _image_gray(image) -> Any | None:
    try:
        import cv2
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr.astype("uint8", copy=False)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    return None


def _coerce_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = max(0, min(width, int(x0)))
    iy0 = max(0, min(height, int(y0)))
    ix1 = max(0, min(width, int(x1)))
    iy1 = max(0, min(height, int(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def _block_is_vertical(block, block_bbox: tuple[int, int, int, int]) -> bool:
    value = getattr(block, "vertical", None)
    if value is not None:
        return bool(value)
    x0, y0, x1, y1 = block_bbox
    return (y1 - y0) >= (x1 - x0)


def _line_continuation_search_bbox(
    block_bbox: tuple[int, int, int, int],
    line_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    vertical: bool,
) -> tuple[int, int, int, int]:
    bx0, by0, bx1, by1 = block_bbox
    lx0, ly0, lx1, ly1 = line_bbox
    block_w = max(1, bx1 - bx0)
    block_h = max(1, by1 - by0)
    line_w = max(1, lx1 - lx0)
    line_h = max(1, ly1 - ly0)
    if vertical:
        cross_pad = max(3, min(20, int(max(block_w, line_w) * 0.45)))
        axis_pad = max(10, min(96, int(max(block_h, line_h, line_w * 4) * 1.0)))
    else:
        cross_pad = max(3, min(20, int(max(block_h, line_h) * 0.45)))
        axis_pad = max(10, min(96, int(max(block_w, line_w, line_h * 4) * 1.0)))
    if vertical:
        return (
            max(0, min(bx0, lx0) - cross_pad),
            max(0, min(by0, ly0) - axis_pad),
            min(width, max(bx1, lx1) + cross_pad),
            min(height, max(by1, ly1) + axis_pad),
        )
    return (
        max(0, min(bx0, lx0) - axis_pad),
        max(0, min(by0, ly0) - cross_pad),
        min(width, max(bx1, lx1) + axis_pad),
        min(height, max(by1, ly1) + cross_pad),
    )


def _is_recoverable_line_continuation_component(
    *,
    comp_bbox: tuple[int, int, int, int],
    area: int,
    component,
    search_bbox: tuple[int, int, int, int],
    block_bbox: tuple[int, int, int, int],
    line_bbox: tuple[int, int, int, int],
    vertical: bool,
    gray,
) -> bool:
    cx0, cy0, cx1, cy1 = comp_bbox
    _sx0, _sy0, _sx1, _sy1 = search_bbox
    bx0, by0, bx1, by1 = block_bbox
    lx0, ly0, lx1, ly1 = line_bbox
    comp_w = max(1, cx1 - cx0)
    comp_h = max(1, cy1 - cy0)
    block_w = max(1, bx1 - bx0)
    block_h = max(1, by1 - by0)
    line_w = max(1, lx1 - lx0)
    line_h = max(1, ly1 - ly0)
    block_area = max(1, block_w * block_h)
    if area > max(2200, int(block_area * 0.35)):
        return False
    if vertical:
        axis_len = comp_h
        cross_len = comp_w
        if axis_len < max(12, int(cross_len * 2.2)):
            return False
        if cross_len > max(24, int(max(line_w, 6) * 1.35)):
            return False
        band_pad = max(3, min(14, int(max(line_w, block_w, 6) * 0.45)))
        band0, band1 = lx0 - band_pad, lx1 + band_pad
        overlap = max(0, min(cx1, band1) - max(cx0, band0))
        if overlap <= 0 or overlap / float(cross_len) < 0.45:
            return False
        gap = max(ly0 - cy1, cy0 - ly1, 0)
        if gap > max(10, min(72, int(max(block_h, line_h) * 0.65))):
            return False
        if axis_len > max(96, int(max(block_h, line_h) * 1.8)):
            return False
    else:
        axis_len = comp_w
        cross_len = comp_h
        if axis_len < max(12, int(cross_len * 2.2)):
            return False
        if cross_len > max(24, int(max(line_h, 6) * 1.35)):
            return False
        band_pad = max(3, min(14, int(max(line_h, block_h, 6) * 0.45)))
        band0, band1 = ly0 - band_pad, ly1 + band_pad
        overlap = max(0, min(cy1, band1) - max(cy0, band0))
        if overlap <= 0 or overlap / float(cross_len) < 0.45:
            return False
        gap = max(lx0 - cx1, cx0 - lx1, 0)
        if gap > max(10, min(72, int(max(block_w, line_w) * 0.65))):
            return False
        if axis_len > max(96, int(max(block_w, line_w) * 1.8)):
            return False
    return _component_has_local_contrast(
        component=component,
        comp_bbox=comp_bbox,
        gray=gray,
    )


def _component_has_local_contrast(*, component, comp_bbox: tuple[int, int, int, int], gray) -> bool:
    if gray is None:
        return True
    try:
        import numpy as np

        gx0, gy0, gx1, gy1 = comp_bbox
        px0 = max(0, gx0 - 4)
        py0 = max(0, gy0 - 4)
        px1 = min(gray.shape[1], gx1 + 4)
        py1 = min(gray.shape[0], gy1 + 4)
        crop = gray[py0:py1, px0:px1]
        if crop.size == 0:
            return True
        local_component = np.zeros(crop.shape[:2], dtype=bool)
        cx0 = max(0, gx0 - px0)
        cy0 = max(0, gy0 - py0)
        cx1 = min(local_component.shape[1], cx0 + component.shape[1])
        cy1 = min(local_component.shape[0], cy0 + component.shape[0])
        source_x0 = max(0, px0 - gx0)
        source_y0 = max(0, py0 - gy0)
        source_x1 = source_x0 + max(0, cx1 - cx0)
        source_y1 = source_y0 + max(0, cy1 - cy0)
        if cx1 <= cx0 or cy1 <= cy0 or source_x1 <= source_x0 or source_y1 <= source_y0:
            return True
        local_component[cy0:cy1, cx0:cx1] = component[source_y0:source_y1, source_x0:source_x1]
        component_values = crop[local_component]
        background_values = crop[~local_component]
        if component_values.size == 0 or background_values.size == 0:
            return True
        comp_median = float(np.median(component_values))
        bg_median = float(np.median(background_values))
        comp_low = float(np.percentile(component_values, 20))
        comp_high = float(np.percentile(component_values, 80))
        bg_low = float(np.percentile(background_values, 20))
        bg_high = float(np.percentile(background_values, 80))
        contrast = max(
            abs(comp_median - bg_median),
            abs(comp_low - bg_high),
            abs(comp_high - bg_low),
        )
        return contrast >= 10.0
    except Exception:
        return True


def _line_continuation_recovery_is_material(audit: dict[str, Any] | None) -> bool:
    if not audit:
        return False
    if audit.get("status"):
        return True
    return bool(
        int(audit.get("components_considered", 0) or 0)
        or int(audit.get("components_recovered", 0) or 0)
        or int(audit.get("pixels_recovered", 0) or 0)
    )


def _ensure_utils_package(repo_root: str) -> None:
    utils_dir = os.path.join(repo_root, "utils")
    if not os.path.isdir(utils_dir):
        return
    existing = sys.modules.get("utils")
    if existing is None:
        import types
        pkg = types.ModuleType("utils")
        pkg.__path__ = [utils_dir]
        sys.modules["utils"] = pkg
        return
    current_path = getattr(existing, "__path__", None)
    if current_path is None or utils_dir not in list(current_path):
        import types
        pkg = types.ModuleType("utils")
        pkg.__path__ = [utils_dir]
        sys.modules["utils"] = pkg


def _lines_bounds(lines: list) -> list | None:
    if not lines:
        return None
    xs = []
    ys = []
    for line in lines:
        for point in line:
            if point is None or not hasattr(point, "__len__") or len(point) < 2:
                continue
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return [x0, y0, x1, y1]
