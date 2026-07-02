"""Fixed local cleanup backend runner.

Production cleanup uses one vetted iopaint LaMa model. Historical local
candidate discovery is kept out of the production path so cleanup plans cannot
switch models by arbitrary path or stale candidate metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from app.inpaint.simple_lama_engine import (
    FIXED_CLEANUP_INPAINT_MODEL_NAME,
    FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH,
    run_simple_lama_model_crop,
)

@dataclass(frozen=True)
class LocalCleanupBackendCandidate:
    candidate_id: str
    backend_family: str
    model_path: str = ""
    adapter_path: str = ""
    available: bool = False
    unavailable_reason: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "backend_family": self.backend_family,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class BackendCandidateExecution:
    cleaned_image: Image.Image | None
    cleaned_crop: Image.Image | None
    candidate_id: str
    backend_name: str
    backend_family: str
    model_path: str
    adapter_path: str = ""
    status: str = "unavailable"
    detail: str = ""
    runtime_ms: float = 0.0
    load_time_ms: float = 0.0
    crop_bbox: list[int] | None = None
    crop_area: int = 0
    mask_pixels: int = 0
    mask_ratio: float = 0.0
    errors: list[str] | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "backend_name": self.backend_name,
            "backend_family": self.backend_family,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "status": self.status,
            "detail": self.detail,
            "runtime_ms": self.runtime_ms,
            "load_time_ms": self.load_time_ms,
            "crop_bbox": list(self.crop_bbox or []),
            "crop_area": self.crop_area,
            "mask_pixels": self.mask_pixels,
            "mask_ratio": self.mask_ratio,
            "errors": list(self.errors or []),
        }


def inventory_local_cleanup_backends(
    models_root: str | Path | None = None,
) -> list[LocalCleanupBackendCandidate]:
    root = Path(models_root) if models_root is not None else _repo_root() / "models" / "inpaint"
    fixed_path = root / FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH[-2] / FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH[-1]

    return [
        LocalCleanupBackendCandidate(
            candidate_id="iopaint_anime_manga_big_lama",
            backend_family="torchscript_simple_lama",
            model_path=str(fixed_path),
            available=fixed_path.exists(),
            unavailable_reason="" if fixed_path.exists() else "fixed_iopaint_model_missing",
        ),
    ]


def inventory_to_audit_dict(candidates: list[LocalCleanupBackendCandidate]) -> dict[str, Any]:
    available = [candidate for candidate in candidates if candidate.available]
    return {
        "version": "cleanup_backend_candidates_fixed_local",
        "candidate_count": len(candidates),
        "available_count": len(available),
        "candidates": [candidate.to_audit_dict() for candidate in candidates],
    }


def candidate_by_id(candidate_id: str, candidates: list[LocalCleanupBackendCandidate]) -> LocalCleanupBackendCandidate | None:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def run_cleanup_backend_candidate(
    *,
    image: Image.Image,
    mask: Any,
    candidate: LocalCleanupBackendCandidate,
    use_gpu: bool,
) -> BackendCandidateExecution:
    start = perf_counter()
    crop_bbox, crop_img, crop_mask, mask_pixels, mask_ratio = _crop_inputs(image, mask)
    fixed_candidate = inventory_local_cleanup_backends()[0]
    fixed_path = str(Path(fixed_candidate.model_path).resolve())
    candidate_path = str(Path(candidate.model_path).resolve()) if candidate.model_path else ""
    if (
        candidate.candidate_id != "iopaint_anime_manga_big_lama"
        or candidate.backend_family != "torchscript_simple_lama"
        or candidate_path != fixed_path
    ):
        return BackendCandidateExecution(
            cleaned_image=image,
            cleaned_crop=crop_img,
            candidate_id=candidate.candidate_id,
            backend_name=candidate.candidate_id,
            backend_family=candidate.backend_family,
            model_path=candidate.model_path,
            adapter_path=candidate.adapter_path,
            status="unavailable",
            detail=f"fixed_cleanup_backend_required:{FIXED_CLEANUP_INPAINT_MODEL_NAME}",
            runtime_ms=round((perf_counter() - start) * 1000.0, 3),
            crop_bbox=crop_bbox,
            crop_area=_bbox_area(crop_bbox),
            mask_pixels=mask_pixels,
            mask_ratio=mask_ratio,
            errors=[f"fixed_cleanup_backend_required:{FIXED_CLEANUP_INPAINT_MODEL_NAME}"],
        )
    if not candidate.available:
        return BackendCandidateExecution(
            cleaned_image=image,
            cleaned_crop=crop_img,
            candidate_id=candidate.candidate_id,
            backend_name=candidate.candidate_id,
            backend_family=candidate.backend_family,
            model_path=candidate.model_path,
            adapter_path=candidate.adapter_path,
            status="unavailable",
            detail=candidate.unavailable_reason,
            runtime_ms=round((perf_counter() - start) * 1000.0, 3),
            crop_bbox=crop_bbox,
            crop_area=_bbox_area(crop_bbox),
            mask_pixels=mask_pixels,
            mask_ratio=mask_ratio,
            errors=[candidate.unavailable_reason] if candidate.unavailable_reason else [],
        )
    if crop_bbox is None or crop_img is None or crop_mask is None or mask_pixels <= 0:
        return BackendCandidateExecution(
            cleaned_image=image,
            cleaned_crop=crop_img,
            candidate_id=candidate.candidate_id,
            backend_name=candidate.candidate_id,
            backend_family=candidate.backend_family,
            model_path=candidate.model_path,
            adapter_path=candidate.adapter_path,
            status="noop",
            detail="empty_mask_or_crop",
            runtime_ms=round((perf_counter() - start) * 1000.0, 3),
            crop_bbox=crop_bbox,
            crop_area=_bbox_area(crop_bbox),
            mask_pixels=mask_pixels,
            mask_ratio=mask_ratio,
            errors=[],
        )
    try:
        if candidate.backend_family == "torchscript_simple_lama":
            result_crop, load_time_ms = _run_simple_lama_crop(
                crop_img=crop_img,
                crop_mask=crop_mask,
                model_path=candidate.model_path,
                use_gpu=use_gpu,
            )
        else:
            raise RuntimeError(f"fixed_cleanup_backend_family_required:{candidate.backend_family}")
        if result_crop.size != crop_img.size:
            result_crop = result_crop.resize(crop_img.size, Image.Resampling.LANCZOS)
        out_crop = Image.composite(result_crop.convert("RGB"), crop_img.convert("RGB"), crop_mask.convert("L"))
        out = image.copy()
        x1, y1, _x2, _y2 = crop_bbox
        out.paste(out_crop, (x1, y1))
        status = "completed"
        detail = "candidate_completed"
        errors: list[str] = []
    except Exception as exc:
        out = image
        out_crop = crop_img
        load_time_ms = 0.0
        status = "error"
        detail = f"{type(exc).__name__}: {exc}"
        errors = [detail]
    return BackendCandidateExecution(
        cleaned_image=out,
        cleaned_crop=out_crop,
        candidate_id=candidate.candidate_id,
        backend_name=candidate.candidate_id,
        backend_family=candidate.backend_family,
        model_path=candidate.model_path,
        adapter_path=candidate.adapter_path,
        status=status,
        detail=detail,
        runtime_ms=round((perf_counter() - start) * 1000.0, 3),
        load_time_ms=round(load_time_ms, 3),
        crop_bbox=crop_bbox,
        crop_area=_bbox_area(crop_bbox),
        mask_pixels=mask_pixels,
        mask_ratio=mask_ratio,
        errors=errors,
    )


def _run_simple_lama_crop(
    *,
    crop_img: Image.Image,
    crop_mask: Image.Image,
    model_path: str,
    use_gpu: bool,
) -> tuple[Image.Image, float]:
    return run_simple_lama_model_crop(
        crop_img=crop_img,
        crop_mask=crop_mask,
        model_path=str(Path(model_path).resolve()),
        use_gpu=use_gpu,
    )


def _crop_inputs(
    image: Image.Image,
    mask: Any,
) -> tuple[list[int] | None, Image.Image | None, Image.Image | None, int, float]:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = (mask_arr > 0).astype(np.uint8)
    mask_pixels = int(np.count_nonzero(mask_arr))
    mask_ratio = mask_pixels / float(max(1, mask_arr.size))
    if mask_pixels <= 0:
        return None, None, None, 0, mask_ratio
    ys, xs = np.where(mask_arr > 0)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = max(32, int(max(x2 - x1, y2 - y1) * 0.25))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(image.width, x2 + pad)
    cy2 = min(image.height, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None, None, mask_pixels, mask_ratio
    crop_img = image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    crop_mask = Image.fromarray((mask_arr[cy1:cy2, cx1:cx2] * 255).astype(np.uint8)).convert("L")
    return [cx1, cy1, cx2, cy2], crop_img, crop_mask, mask_pixels, mask_ratio


def _bbox_area(bbox: list[int] | None) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
