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
        "punctuation_endpoint_components_considered": 0,
        "punctuation_endpoint_components_recovered": 0,
        "punctuation_endpoint_pixels_recovered": 0,
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
        endpoint_mask, endpoint_audit = _recover_punctuation_endpoint_components(
            raw_bool=raw_bool,
            refined_bool=np.logical_or(refined_bool, admitted),
            search_bbox=search_bbox,
            line_bbox=line_bbox,
            block_bbox=block_bbox,
            vertical=vertical,
            gray=gray,
        )
        audit["punctuation_endpoint_components_considered"] += int(
            endpoint_audit.get("components_considered", 0) or 0
        )
        endpoint_components = int(endpoint_audit.get("components_recovered", 0) or 0)
        endpoint_pixels = int(endpoint_audit.get("pixels_recovered", 0) or 0)
        if endpoint_components and endpoint_mask is not None:
            new_pixels = np.logical_and(endpoint_mask, np.logical_not(admitted))
            if np.any(new_pixels):
                admitted = np.logical_or(admitted, new_pixels)
                new_pixel_count = int(np.count_nonzero(new_pixels))
                audit["components_recovered"] += endpoint_components
                audit["pixels_recovered"] += new_pixel_count
                audit["punctuation_endpoint_components_recovered"] += endpoint_components
                audit["punctuation_endpoint_pixels_recovered"] += endpoint_pixels

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


def _recover_punctuation_endpoint_components(
    *,
    raw_bool,
    refined_bool,
    search_bbox: tuple[int, int, int, int],
    line_bbox: tuple[int, int, int, int],
    block_bbox: tuple[int, int, int, int],
    vertical: bool,
    gray,
) -> tuple[Any | None, dict[str, Any]]:
    """Recover small punctuation endpoints dropped by CTD refinement.

    The raw CTD mask can still contain terminal ellipsis/dot punctuation after
    the refined mask clips it away. This stays inside CTD ownership: recovery is
    limited to small foreground components at the next expected position of an
    already-refined punctuation chain inside the same CTD text block.
    """

    audit = {
        "components_considered": 0,
        "components_recovered": 0,
        "pixels_recovered": 0,
    }
    if raw_bool is None or refined_bool is None or gray is None:
        return None, audit
    try:
        import cv2
        import numpy as np
    except Exception:
        return None, audit

    sx0, sy0, sx1, sy1 = search_bbox
    if sx1 <= sx0 or sy1 <= sy0:
        return None, audit
    raw_roi = raw_bool[sy0:sy1, sx0:sx1]
    refined_roi = refined_bool[sy0:sy1, sx0:sx1]
    residual_roi = np.logical_and(raw_roi, np.logical_not(refined_roi))
    if not np.any(residual_roi):
        return None, audit
    anchor_components = _punctuation_endpoint_dot_components(refined_roi, offset=(sx0, sy0))
    if len(anchor_components) < 3:
        return None, audit

    admitted = np.zeros_like(raw_bool, dtype=bool)
    accepted_keys: set[tuple[int, int, int, int]] = set()
    for run in _punctuation_endpoint_runs(anchor_components, vertical=vertical):
        if len(run) < 3:
            continue
        profile = _punctuation_endpoint_profile(run, gray=gray, vertical=vertical)
        if profile is None:
            continue
        run_mask = _recover_punctuation_run_endpoints(
            raw_bool=raw_bool,
            refined_bool=np.logical_or(refined_bool, admitted),
            search_bbox=search_bbox,
            line_bbox=line_bbox,
            block_bbox=block_bbox,
            run=run,
            profile=profile,
            vertical=vertical,
            gray=gray,
            accepted_keys=accepted_keys,
            audit=audit,
        )
        if run_mask is not None and np.any(run_mask):
            admitted = np.logical_or(admitted, run_mask)

    if not np.any(admitted):
        return None, audit
    return admitted, audit


def _punctuation_endpoint_dot_components(mask, *, offset: tuple[int, int]) -> list[dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    arr = np.asarray(mask).astype("uint8")
    if arr.size == 0:
        return []
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    ox, oy = offset
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not _punctuation_endpoint_dot_shape(width=w, height=h, area=area):
            continue
        local = labels[y : y + h, x : x + w] == label
        components.append(
            {
                "bbox": (ox + x, oy + y, ox + x + w, oy + y + h),
                "area": area,
                "center": (ox + float(centroids[label][0]), oy + float(centroids[label][1])),
                "mask": local,
            }
        )
    return components


def _punctuation_endpoint_dot_shape(*, width: int, height: int, area: int) -> bool:
    if area < 4 or width <= 0 or height <= 0:
        return False
    if width > 24 or height > 24:
        return False
    aspect = max(width, height) / float(max(1, min(width, height)))
    if aspect > 3.2:
        return False
    fill = area / float(max(1, width * height))
    return fill >= 0.16


def _punctuation_endpoint_runs(components: list[dict[str, Any]], *, vertical: bool) -> list[list[dict[str, Any]]]:
    if len(components) < 3:
        return []
    runs: list[list[dict[str, Any]]] = []
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    for seed in components:
        seed_cross = float(seed["center"][0] if vertical else seed["center"][1])
        seed_size = max(
            1,
            (seed["bbox"][2] - seed["bbox"][0]) if vertical else (seed["bbox"][3] - seed["bbox"][1]),
        )
        cross_tolerance = max(6.0, min(18.0, seed_size * 2.4))
        aligned = [
            item
            for item in components
            if abs(float(item["center"][0] if vertical else item["center"][1]) - seed_cross) <= cross_tolerance
        ]
        aligned.sort(key=lambda item: float(item["center"][1] if vertical else item["center"][0]))
        current: list[dict[str, Any]] = []
        previous_axis: float | None = None
        for item in aligned:
            axis = float(item["center"][1] if vertical else item["center"][0])
            if previous_axis is None or axis - previous_axis <= 34.0:
                current.append(item)
            else:
                if _punctuation_endpoint_run_is_regular(current, vertical=vertical):
                    key = tuple(tuple(entry["bbox"]) for entry in current)
                    if key not in seen:
                        seen.add(key)
                        runs.append(list(current))
                current = [item]
            previous_axis = axis
        if _punctuation_endpoint_run_is_regular(current, vertical=vertical):
            key = tuple(tuple(entry["bbox"]) for entry in current)
            if key not in seen:
                seen.add(key)
                runs.append(list(current))
    return runs


def _punctuation_endpoint_run_is_regular(run: list[dict[str, Any]], *, vertical: bool) -> bool:
    if len(run) < 3:
        return False
    axes = sorted(float(item["center"][1] if vertical else item["center"][0]) for item in run)
    gaps = [axes[index + 1] - axes[index] for index in range(len(axes) - 1)]
    gaps = [gap for gap in gaps if gap > 0.5]
    if len(gaps) < 2:
        return False
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    if median_gap < 4.0 or median_gap > 30.0:
        return False
    return all(max(3.0, median_gap * 0.45) <= gap <= min(38.0, median_gap * 2.15) for gap in gaps)


def _punctuation_endpoint_profile(run: list[dict[str, Any]], *, gray, vertical: bool) -> dict[str, float] | None:
    try:
        import numpy as np
    except Exception:
        return None
    foreground_values: list[float] = []
    background_values: list[float] = []
    for item in run:
        x0, y0, x1, y1 = item["bbox"]
        crop = gray[y0:y1, x0:x1]
        mask = item["mask"]
        if crop.shape[:2] != mask.shape:
            continue
        foreground_values.extend(float(value) for value in crop[mask].reshape(-1))
        px0 = max(0, x0 - 4)
        py0 = max(0, y0 - 4)
        px1 = min(gray.shape[1], x1 + 4)
        py1 = min(gray.shape[0], y1 + 4)
        halo = gray[py0:py1, px0:px1]
        halo_mask = np.zeros(halo.shape[:2], dtype=bool)
        hx0 = x0 - px0
        hy0 = y0 - py0
        halo_mask[hy0 : hy0 + mask.shape[0], hx0 : hx0 + mask.shape[1]] = mask
        background_values.extend(float(value) for value in halo[~halo_mask].reshape(-1))
    if not foreground_values:
        return None
    fg = float(np.median(np.asarray(foreground_values)))
    bg = float(np.median(np.asarray(background_values))) if background_values else (255.0 - fg)
    axes = sorted(float(item["center"][1] if vertical else item["center"][0]) for item in run)
    gaps = [axes[index + 1] - axes[index] for index in range(len(axes) - 1) if axes[index + 1] - axes[index] > 0.5]
    gaps.sort()
    spacing = float(gaps[len(gaps) // 2]) if gaps else 14.0
    cross_sizes = [
        max(1, (item["bbox"][2] - item["bbox"][0]) if vertical else (item["bbox"][3] - item["bbox"][1]))
        for item in run
    ]
    axis_sizes = [
        max(1, (item["bbox"][3] - item["bbox"][1]) if vertical else (item["bbox"][2] - item["bbox"][0]))
        for item in run
    ]
    areas = [max(1, int(item.get("area", 1))) for item in run]
    return {
        "foreground": fg,
        "background": bg,
        "spacing": spacing,
        "cross": float(np.median(np.asarray(cross_sizes))),
        "axis": float(np.median(np.asarray(axis_sizes))),
        "area": float(np.median(np.asarray(areas))),
        "cross_center": float(np.median(np.asarray([item["center"][0] if vertical else item["center"][1] for item in run]))),
    }


def _recover_punctuation_run_endpoints(
    *,
    raw_bool,
    refined_bool,
    search_bbox: tuple[int, int, int, int],
    line_bbox: tuple[int, int, int, int],
    block_bbox: tuple[int, int, int, int],
    run: list[dict[str, Any]],
    profile: dict[str, float],
    vertical: bool,
    gray,
    accepted_keys: set[tuple[int, int, int, int]],
    audit: dict[str, Any],
):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    sx0, sy0, sx1, sy1 = search_bbox
    bx0, by0, bx1, by1 = block_bbox
    lx0, ly0, lx1, ly1 = line_bbox
    spacing = float(profile.get("spacing") or 14.0)
    if spacing < 4.0 or spacing > 30.0:
        return None
    cross_radius = max(6, int(round(float(profile.get("cross") or 6.0) * 1.85)))
    axis_radius = max(6, int(round(max(float(profile.get("axis") or 6.0) * 1.85, spacing * 0.45))))
    min_area = max(3.0, float(profile.get("area") or 12.0) * 0.30)
    max_area = max(18.0, float(profile.get("area") or 12.0) * 2.80)
    fg = float(profile.get("foreground") or 0.0)
    bg = float(profile.get("background") or 255.0)
    tolerance = max(38.0, min(82.0, abs(fg - bg) * 0.75 + 18.0))
    axes = sorted(float(item["center"][1] if vertical else item["center"][0]) for item in run)
    if not axes:
        return None
    admitted = np.zeros_like(raw_bool, dtype=bool)
    for direction, start_axis in ((1.0, axes[-1]),):
        current_axis = start_axis
        for _ in range(3):
            expected_axis = current_axis + direction * spacing
            if vertical:
                expected_x = float(profile["cross_center"])
                expected_y = expected_axis
                if expected_y < min(sy0, by0, ly0) - spacing * 1.25 or expected_y > max(sy1, by1, ly1) + spacing * 1.25:
                    break
                wx0 = max(sx0, int(round(expected_x - cross_radius)))
                wx1 = min(sx1, int(round(expected_x + cross_radius + 1)))
                wy0 = max(sy0, int(round(expected_y - axis_radius)))
                wy1 = min(sy1, int(round(expected_y + axis_radius + 1)))
            else:
                expected_x = expected_axis
                expected_y = float(profile["cross_center"])
                if expected_x < min(sx0, bx0, lx0) - spacing * 1.25 or expected_x > max(sx1, bx1, lx1) + spacing * 1.25:
                    break
                wx0 = max(sx0, int(round(expected_x - axis_radius)))
                wx1 = min(sx1, int(round(expected_x + axis_radius + 1)))
                wy0 = max(sy0, int(round(expected_y - cross_radius)))
                wy1 = min(sy1, int(round(expected_y + cross_radius + 1)))
            if wx1 <= wx0 or wy1 <= wy0:
                break
            residual = np.logical_and(raw_bool[wy0:wy1, wx0:wx1], np.logical_not(refined_bool[wy0:wy1, wx0:wx1]))
            if not np.any(residual):
                break
            gray_roi = gray[wy0:wy1, wx0:wx1]
            if fg <= bg:
                foreground = gray_roi <= min(230.0, fg + tolerance)
            else:
                foreground = gray_roi >= max(25.0, fg - tolerance)
            seed = np.logical_and(residual, foreground)
            if not np.any(seed):
                break
            labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(seed.astype("uint8"), connectivity=8)
            candidates: list[tuple[float, int, tuple[int, int, int, int], Any]] = []
            for label in range(1, labels_count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                w = int(stats[label, cv2.CC_STAT_WIDTH])
                h = int(stats[label, cv2.CC_STAT_HEIGHT])
                if not _punctuation_endpoint_dot_shape(width=w, height=h, area=area):
                    continue
                bbox = (wx0 + x, wy0 + y, wx0 + x + w, wy0 + y + h)
                if bbox in accepted_keys:
                    continue
                center_x = wx0 + float(centroids[label][0])
                center_y = wy0 + float(centroids[label][1])
                distance = abs(center_x - expected_x) + abs(center_y - expected_y)
                candidates.append((distance, label, bbox, labels[y : y + h, x : x + w] == label))
            audit["components_considered"] += max(0, labels_count - 1)
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            _distance, _label, bbox, local_mask = candidates[0]
            x0, y0, x1, y1 = bbox
            if not _component_has_local_contrast(component=local_mask, comp_bbox=bbox, gray=gray):
                break
            page_slice = admitted[y0:y1, x0:x1]
            page_slice[local_mask] = True
            admitted[y0:y1, x0:x1] = page_slice
            accepted_keys.add(bbox)
            refined_bool[y0:y1, x0:x1] = np.logical_or(refined_bool[y0:y1, x0:x1], local_mask)
            audit["components_recovered"] += 1
            audit["pixels_recovered"] += int(np.count_nonzero(local_mask))
            current_axis = float((y0 + y1) * 0.5 if vertical else (x0 + x1) * 0.5)
    if not np.any(admitted):
        return None
    return admitted


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
