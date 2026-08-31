# -*- coding: utf-8 -*-
"""Project-owned cleanup inpainting backend.

This module owns fixed Anime Manga Big LaMA model resolution, loading, warmup,
and inference orchestration. Cleanup modules decide whether a cleanup job may
call the backend; this backend does not own semantic authorization, cleanup
proof, or commit policy.
"""

from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache

from app.config.defaults import CLEANUP_INPAINT_MODEL_FILE, IOPAINT_ANIME_MANGA_BIG_LAMA
from app.inpaint.torchscript_lama_runner import (
    TorchScriptLamaRunner,
    cleanup_fallback,
    resolve_lama_device_name,
)
from app.platform_services.compute import load_torch_runtime
from app.pipeline.debug_runtime import (
    diagnostic_enabled,
    pipeline_diagnostic_checkpoint,
    write_diagnostic_checkpoint,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


FIXED_CLEANUP_INPAINT_MODEL_ID = IOPAINT_ANIME_MANGA_BIG_LAMA
FIXED_CLEANUP_INPAINT_MODEL_NAME = "SimpleLama(iopaint/anime-manga-big-lama)"
FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH = (
    "models",
    "inpaint",
    "iopaint",
    CLEANUP_INPAINT_MODEL_FILE,
)
FIXED_CLEANUP_INPAINT_SELECTION_POLICY = "fixed_cleanup_iopaint_model"
_WARMED_LAMA_MODEL_KEYS: set[tuple[str, str]] = set()
_WARMUP_LOCK = threading.Lock()
_CLEANUP_DEVICE_FALLBACK = None


def _cleanup_perf_contract_diag_enabled() -> bool:
    return diagnostic_enabled("MT_CLEANUP_PERF_CONTRACT_DIAGNOSTIC")


def _cleanup_perf_contract_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cleanup_perf_contract_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_cleanup_perf_contract_json_safe(item) for item in list(value)[:80]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(item) for item in tuple(shape)]}
    return str(value)


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields) -> None:
    if not _cleanup_perf_contract_diag_enabled():
        return
    try:
        write_diagnostic_checkpoint(
            "cleanup_perf_contract_checkpoints.jsonl",
            module="app.inpaint.simple_lama_engine",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
        )
    except Exception:
        return


def _pipeline_runtime_checkpoint(stage: str, event: str, **fields) -> None:
    _cleanup_perf_contract_checkpoint(stage, event, **fields)
    pipeline_diagnostic_checkpoint(
        module="app.inpaint.simple_lama_engine",
        stage=stage,
        event=event,
        fields=fields,
    )


def clear_model_cache() -> None:
    """Clear the cleanup inpainting model cache."""

    global _CLEANUP_DEVICE_FALLBACK
    _load_lama_model.cache_clear()
    with _WARMUP_LOCK:
        _WARMED_LAMA_MODEL_KEYS.clear()
        _CLEANUP_DEVICE_FALLBACK = None


def _effective_cleanup_device(use_gpu: bool) -> tuple[str, str, str]:
    requested = resolve_lama_device_name(use_gpu=use_gpu)
    with _WARMUP_LOCK:
        fallback = _CLEANUP_DEVICE_FALLBACK
    if requested == "mps" and fallback is not None:
        return fallback.device, requested, fallback.fallback_reason
    return requested, requested, ""


def _synchronize_device(device: str) -> None:
    try:
        torch = load_torch_runtime()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        # MPS synchronization is part of the recoverable execution boundary:
        # propagate its failures so the shared controller can retry atomically
        # on CPU. CUDA synchronization historically remained best effort, and
        # preserving that behavior avoids turning post-inference telemetry into
        # a Windows/Linux render failure.
        if device == "mps":
            raise


def _clear_cached_lama_models() -> None:
    clear = getattr(_load_lama_model, "cache_clear", None)
    if callable(clear):
        clear()


def _run_lama_with_runtime_fallback(
    *,
    image,
    mask,
    use_gpu: bool,
    model_path: str,
    collect_runner_timings: bool = False,
) -> tuple[object, dict[str, object]]:
    """Run one fixed-model inference with one atomic MPS-to-CPU retry."""

    global _CLEANUP_DEVICE_FALLBACK
    device, requested_device, fallback_reason = _effective_cleanup_device(use_gpu)
    total_load_ms = 0.0
    total_inference_ms = 0.0

    def attempt(candidate_device: str) -> tuple[object, dict[str, object]]:
        nonlocal total_load_ms, total_inference_ms
        load_started = time.perf_counter()
        lama = _load_lama_model(candidate_device, str(model_path or ""))
        load_ms = _perf_elapsed_ms(load_started)
        total_load_ms += load_ms
        runner_timings: dict[str, Any] | None = (
            {} if collect_runner_timings else None
        )
        inference_started = time.perf_counter()
        if runner_timings is None:
            result = lama(image, mask)
        else:
            result = lama(image, mask, perf_timings=runner_timings)
        _synchronize_device(candidate_device)
        inference_ms = _perf_elapsed_ms(inference_started)
        total_inference_ms += inference_ms
        return result, {
            "device": candidate_device,
            "load_elapsed_ms": load_ms,
            "inference_elapsed_ms": inference_ms,
            "runner_timings": runner_timings or {},
        }

    try:
        result, attempt_meta = attempt(device)
        return result, {
            **attempt_meta,
            "requested_device": requested_device,
            "fallback_reason": fallback_reason,
            "total_load_elapsed_ms": round(total_load_ms, 3),
            "total_inference_elapsed_ms": round(total_inference_ms, 3),
        }
    except Exception as exc:
        if device != "mps":
            raise
        fallback = cleanup_fallback(exc)
        _clear_cached_lama_models()
        result, attempt_meta = attempt(fallback.device)
        with _WARMUP_LOCK:
            _CLEANUP_DEVICE_FALLBACK = fallback
        return result, {
            **attempt_meta,
            "requested_device": requested_device,
            "fallback_reason": fallback.fallback_reason,
            "mps_failure": f"{type(exc).__name__}: {exc}",
            "total_load_elapsed_ms": round(total_load_ms, 3),
            "total_inference_elapsed_ms": round(total_inference_ms, 3),
        }


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_cleanup_inpaint_model(model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID) -> dict[str, str]:
    """Resolve the one authorized local cleanup model.

    ``model_id`` is retained only as requested-model provenance. Cleanup
    inpainting is intentionally fixed to the vetted iopaint TorchScript model
    so UI/config strings, absolute paths, or legacy candidate IDs cannot switch
    the production cleanup backend.
    """

    root = _repo_root()
    requested = str(model_id or "").strip()
    fixed_path = os.path.join(root, *FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH)
    return {
        "requested_model_id": requested,
        "configured_model_id": FIXED_CLEANUP_INPAINT_MODEL_ID,
        "selection_policy": FIXED_CLEANUP_INPAINT_SELECTION_POLICY,
        "actual_model_name": FIXED_CLEANUP_INPAINT_MODEL_NAME,
        "actual_model_path": fixed_path,
        "model_available": os.path.exists(fixed_path),
        "ignored_requested_model": requested not in {"", FIXED_CLEANUP_INPAINT_MODEL_ID},
    }


@lru_cache(maxsize=4)
def _load_lama_model(device: str, model_path: str = ""):
    """Load the LaMa model for cleanup-owned inpainting."""

    effective_device = resolve_lama_device_name(device)
    if not model_path:
        raise RuntimeError("fixed cleanup inpaint model path required")
    if not os.path.exists(model_path):
        raise RuntimeError(f"fixed cleanup inpaint model missing: {model_path}")

    print(f"[Cleanup Inpaint] Loading TorchScript LaMA model on {effective_device}: {model_path}")
    _pipeline_runtime_checkpoint("cleanup_inpaint_model", "load_start", device=effective_device, model_path=model_path)
    runner = TorchScriptLamaRunner(model_path=model_path, device=effective_device)
    print("[Cleanup Inpaint] TorchScript LaMA model loaded successfully")
    _pipeline_runtime_checkpoint("cleanup_inpaint_model", "load_end", device=effective_device, model_path=model_path)
    return runner


def _warmup_disabled() -> bool:
    value = str(os.environ.get("MT_CLEANUP_INPAINT_WARMUP", "") or "").strip().lower()
    return value in {"0", "false", "off", "no", "disabled"}


def warm_cleanup_inpaint_model(
    *,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
) -> dict[str, object]:
    """Load and warm the cleanup-owned LaMa model once per process.

    The warmup is deliberately model-local. It does not alter cleanup masks,
    proof, or commit policy. It validates the selected accelerator on a tiny
    synthetic crop and records one durable MPS-to-CPU fallback when required.
    """

    started = time.time()
    if _warmup_disabled():
        return {
            "status": "disabled",
            "elapsed_ms": 0.0,
        }
    if Image is None:
        return {
            "status": "skipped",
            "reason": "pillow_unavailable",
            "elapsed_ms": 0.0,
        }

    global _CLEANUP_DEVICE_FALLBACK
    device, requested_device, fallback_reason = _effective_cleanup_device(use_gpu)
    model_info = resolve_cleanup_inpaint_model(model_id)
    actual_model_path = str(model_info.get("actual_model_path") or "")
    key = (device, actual_model_path)
    with _WARMUP_LOCK:
        if key in _WARMED_LAMA_MODEL_KEYS:
            return {
                "status": "already_warmed",
                "device": device,
                "requested_device": requested_device,
                "fallback_reason": fallback_reason,
                "elapsed_ms": 0.0,
            }

    def load_and_warm(candidate_device: str) -> tuple[float, float]:
        load_started = time.time()
        lama = _load_lama_model(candidate_device, actual_model_path)
        load_elapsed = round((time.time() - load_started) * 1000.0, 3)
        infer_started = time.time()
        warm_image = Image.new("RGB", (256, 256), (255, 255, 255))
        warm_mask = Image.new("L", (256, 256), 0)
        warm_mask.paste(255, (96, 96, 160, 160))
        _ = lama(warm_image, warm_mask)
        _synchronize_device(candidate_device)
        return load_elapsed, round((time.time() - infer_started) * 1000.0, 3)

    warm_succeeded = False
    try:
        load_elapsed_ms, infer_elapsed_ms = load_and_warm(device)
        warm_succeeded = True
    except Exception as exc:
        original_error = f"{type(exc).__name__}: {exc}"
        if device == "mps":
            fallback = cleanup_fallback(exc)
            _clear_cached_lama_models()
            device = fallback.device
            fallback_reason = fallback.fallback_reason
            try:
                load_elapsed_ms, infer_elapsed_ms = load_and_warm(device)
                warm_succeeded = True
                with _WARMUP_LOCK:
                    _CLEANUP_DEVICE_FALLBACK = fallback
            except Exception as fallback_exc:
                original_error = (
                    f"{original_error}; cpu_retry={type(fallback_exc).__name__}: "
                    f"{fallback_exc}"
                )
        if not warm_succeeded:
            elapsed_ms = round((time.time() - started) * 1000.0, 3)
            _pipeline_runtime_checkpoint(
                "cleanup_inpaint_model_warmup",
                "error",
                requested_device=requested_device,
                device=device,
                fallback_reason=fallback_reason,
                model_id=model_id,
                error=original_error,
                elapsed_ms=elapsed_ms,
            )
            return {
                "status": "error",
                "requested_device": requested_device,
                "device": device,
                "fallback_reason": fallback_reason,
                "error": original_error,
                "elapsed_ms": elapsed_ms,
            }

    elapsed_ms = round((time.time() - started) * 1000.0, 3)
    key = (device, actual_model_path)
    with _WARMUP_LOCK:
        _WARMED_LAMA_MODEL_KEYS.add(key)
    _pipeline_runtime_checkpoint(
        "cleanup_inpaint_model_warmup",
        "end",
        requested_device=requested_device,
        device=device,
        fallback_reason=fallback_reason,
        model_id=model_id,
        load_elapsed_ms=load_elapsed_ms,
        inference_elapsed_ms=infer_elapsed_ms,
        elapsed_ms=elapsed_ms,
    )
    return {
        "status": "warmed",
        "requested_device": requested_device,
        "device": device,
        "fallback_reason": fallback_reason,
        "load_elapsed_ms": load_elapsed_ms,
        "inference_elapsed_ms": infer_elapsed_ms,
        "elapsed_ms": elapsed_ms,
    }


def ai_inpaint_cleanup_crop(
    image,
    mask,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
    mask_prepared: bool = False,
):
    """Run cleanup-owned LaMa on an already-local cleanup crop.

    The caller owns the parent/cleanup crop and proof scope. This backend must
    not recrop to a new page-space unit or materialize a full-page candidate.
    """

    started = time.time()
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None

    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required for AI inpainting")
    if not hasattr(image, "size"):
        raise RuntimeError("crop-local AI inpainting requires a PIL image crop")

    crop_img = image.convert("RGB") if hasattr(image, "convert") else image
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    if mask_arr.shape[:2] != (crop_img.height, crop_img.width):
        mask_img = Image.fromarray((mask_arr > 0).astype(np.uint8) * 255).convert("L")
        mask_img = mask_img.resize(crop_img.size, Image.NEAREST)
        mask_arr = np.asarray(mask_img)
    if mask_prepared:
        prepared_mask = (mask_arr > 0).astype(np.uint8) * 255
    else:
        kernel_size = max(5, int(max(mask_arr.shape) * 0.005))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        prepared_mask = cv2.dilate((mask_arr > 0).astype(np.uint8) * 255, kernel, iterations=2)

    mask_image = Image.fromarray(prepared_mask).convert("L")
    bbox = mask_image.getbbox()
    parent_crop_w, parent_crop_h = crop_img.size
    if not bbox:
        elapsed_ms = round((time.time() - started) * 1000.0, 3)
        return crop_img, {
            "backend": "none",
            "reason": "empty_mask_bbox",
            "crop_width": parent_crop_w,
            "crop_height": parent_crop_h,
            "crop_area": parent_crop_w * parent_crop_h,
            "elapsed_ms": elapsed_ms,
        }

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad = max(32, int(max(w, h) * 0.2))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(parent_crop_w, x1 + pad)
    cy1 = min(parent_crop_h, y1 + pad)
    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    model_input_img = crop_img.crop((cx0, cy0, cx1, cy1))
    model_input_mask = mask_image.crop((cx0, cy0, cx1, cy1))

    model_info = resolve_cleanup_inpaint_model(model_id)
    actual_model_path = model_info.get("actual_model_path", "")
    initial_device, initial_requested_device, initial_fallback_reason = (
        _effective_cleanup_device(use_gpu)
    )

    print(f"[Cleanup Inpaint] Processing local crop: {crop_w}x{crop_h}")
    _pipeline_runtime_checkpoint(
        "cleanup_ai_inpaint_crop_local",
        "start",
        requested_device=initial_requested_device,
        device=initial_device,
        fallback_reason=initial_fallback_reason,
        parent_crop_width=parent_crop_w,
        parent_crop_height=parent_crop_h,
        inner_crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
        mask_bbox=list(bbox),
    )
    result, runtime_meta = _run_lama_with_runtime_fallback(
        image=model_input_img,
        mask=model_input_mask,
        use_gpu=use_gpu,
        model_path=actual_model_path,
    )
    requested_device = str(runtime_meta["requested_device"])
    device = str(runtime_meta["device"])
    fallback_reason = str(runtime_meta["fallback_reason"])
    load_elapsed_ms = float(runtime_meta["total_load_elapsed_ms"])
    model_call_elapsed_ms = float(runtime_meta["total_inference_elapsed_ms"])

    if result.size != (crop_w, crop_h):
        print(f"[Cleanup Inpaint] Resizing crop result from {result.size} to {(crop_w, crop_h)}")
        result = result.resize((crop_w, crop_h), Image.LANCZOS)
    if model_input_mask.size != result.size:
        model_input_mask = model_input_mask.resize(result.size, Image.NEAREST)

    inner_out = Image.composite(result, model_input_img, model_input_mask)
    out_crop = crop_img.copy()
    out_crop.paste(inner_out, (cx0, cy0))
    elapsed_ms = round((time.time() - started) * 1000.0, 3)
    print("[Cleanup Inpaint] Local crop success")
    _pipeline_runtime_checkpoint(
        "cleanup_ai_inpaint_crop_local",
        "end",
        backend="simple_lama",
        requested_device=requested_device,
        device=device,
        fallback_reason=fallback_reason,
        parent_crop_width=parent_crop_w,
        parent_crop_height=parent_crop_h,
        inner_crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
        mask_bbox=list(bbox),
        load_elapsed_ms=load_elapsed_ms,
        model_call_elapsed_ms=model_call_elapsed_ms,
        elapsed_ms=elapsed_ms,
    )
    return out_crop, {
        "backend": "simple_lama",
        "requested_device": requested_device,
        "device": device,
        "fallback_reason": fallback_reason,
        "parent_crop_width": parent_crop_w,
        "parent_crop_height": parent_crop_h,
        "inner_crop_bbox": [cx0, cy0, cx1, cy1],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "crop_area": crop_w * crop_h,
        "mask_bbox": list(bbox),
        "load_elapsed_ms": load_elapsed_ms,
        "model_call_elapsed_ms": model_call_elapsed_ms,
        "mps_failure": str(runtime_meta.get("mps_failure") or ""),
        "elapsed_ms": elapsed_ms,
    }


def run_simple_lama_model_crop(
    *,
    crop_img,
    crop_mask,
    model_path: str,
    use_gpu: bool,
):
    """Run the fixed SimpleLama model on an already prepared crop.

    This is the low-level backend entry point used by backend inventories that
    already own their crop and mask contracts. It shares the same model cache as
    the cleanup convenience wrappers above.
    """

    result, runtime_meta = _run_lama_with_runtime_fallback(
        image=crop_img.convert("RGB"),
        mask=crop_mask.convert("L"),
        use_gpu=use_gpu,
        model_path=str(model_path or ""),
    )
    return result.convert("RGB"), float(runtime_meta["total_load_elapsed_ms"])


def ai_inpaint_cleanup(
    image,
    mask,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
    mask_prepared: bool = False,
    perf_timings: dict | None = None,
):
    """Perform cleanup-owned AI inpainting using LaMa."""

    started = time.time()
    perf_started = time.perf_counter() if perf_timings is not None else 0.0
    _pipeline_runtime_checkpoint(
        "cleanup_ai_inpaint",
        "start",
        use_gpu=use_gpu,
        model_id=model_id,
        image_size=getattr(image, "size", None),
        mask_shape=getattr(mask, "shape", None),
    )
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None

    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required for AI inpainting")

    mask_prepare_started = time.perf_counter() if perf_timings is not None else 0.0
    if mask_prepared:
        dilated_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    else:
        kernel_size = max(5, int(max(mask.shape) * 0.005))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    if perf_timings is not None:
        perf_timings["mask_prepare_ms"] = _perf_elapsed_ms(mask_prepare_started)

    resolve_started = time.perf_counter() if perf_timings is not None else 0.0
    device, requested_device, fallback_reason = _effective_cleanup_device(use_gpu)
    model_info = resolve_cleanup_inpaint_model(model_id)
    actual_model_path = model_info.get("actual_model_path", "")
    if perf_timings is not None:
        perf_timings["device_model_resolve_ms"] = _perf_elapsed_ms(resolve_started)
        perf_timings["device"] = device
        perf_timings["requested_device"] = requested_device
        perf_timings["device_fallback_reason"] = fallback_reason
        perf_timings["requested_use_gpu"] = bool(use_gpu)

    crop_prepare_started = time.perf_counter() if perf_timings is not None else 0.0
    mask_image = Image.fromarray(dilated_mask).convert("L")
    bbox = mask_image.getbbox()
    if not bbox:
        _pipeline_runtime_checkpoint(
            "cleanup_ai_inpaint",
            "end",
            backend="none",
            reason="empty_mask_bbox",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        if perf_timings is not None:
            perf_timings["crop_prepare_ms"] = _perf_elapsed_ms(crop_prepare_started)
            perf_timings["engine_total_ms"] = _perf_elapsed_ms(perf_started)
        return image

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad = max(32, int(max(w, h) * 0.2))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(image.width, x1 + pad)
    cy1 = min(image.height, y1 + pad)

    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    crop_img = image.crop((cx0, cy0, cx1, cy1))
    crop_mask = mask_image.crop((cx0, cy0, cx1, cy1))
    if perf_timings is not None:
        perf_timings["crop_prepare_ms"] = _perf_elapsed_ms(crop_prepare_started)
        perf_timings["crop_bbox"] = [cx0, cy0, cx1, cy1]
        perf_timings["crop_width"] = crop_w
        perf_timings["crop_height"] = crop_h

    print(f"[Cleanup Inpaint] Processing region: {crop_w}x{crop_h}")
    _pipeline_runtime_checkpoint(
        "cleanup_ai_inpaint",
        "crop",
        device=device,
        crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
    )
    runner_started = time.perf_counter() if perf_timings is not None else 0.0
    result, runtime_meta = _run_lama_with_runtime_fallback(
        image=crop_img,
        mask=crop_mask,
        use_gpu=use_gpu,
        model_path=actual_model_path,
        collect_runner_timings=perf_timings is not None,
    )
    requested_device = str(runtime_meta["requested_device"])
    device = str(runtime_meta["device"])
    fallback_reason = str(runtime_meta["fallback_reason"])
    if perf_timings is not None:
        perf_timings["runner_wall_ms"] = _perf_elapsed_ms(runner_started)
        perf_timings["runner"] = dict(runtime_meta.get("runner_timings") or {})
        perf_timings["model_lookup_ms"] = float(
            runtime_meta["total_load_elapsed_ms"]
        )
        perf_timings["device"] = device
        perf_timings["requested_device"] = requested_device
        perf_timings["device_fallback_reason"] = fallback_reason
        perf_timings["mps_failure"] = str(runtime_meta.get("mps_failure") or "")

    composite_started = time.perf_counter() if perf_timings is not None else 0.0
    if result.size != (crop_w, crop_h):
        print(f"[Cleanup Inpaint] Resizing result from {result.size} to {(crop_w, crop_h)}")
        result = result.resize((crop_w, crop_h), Image.LANCZOS)

    if crop_mask.size != result.size:
        crop_mask = crop_mask.resize(result.size, Image.NEAREST)

    out_crop = Image.composite(result, crop_img, crop_mask)
    out = image.copy()
    out.paste(out_crop, (cx0, cy0))
    if perf_timings is not None:
        perf_timings["result_composite_ms"] = _perf_elapsed_ms(composite_started)
        perf_timings["engine_total_ms"] = _perf_elapsed_ms(perf_started)

    print("[Cleanup Inpaint] Success")
    _pipeline_runtime_checkpoint(
        "cleanup_ai_inpaint",
        "end",
        backend="simple_lama",
        device=device,
        crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return out


def _perf_elapsed_ms(started: float) -> float:
    return round(max(0.0, (time.perf_counter() - started) * 1000.0), 3)


def ai_inpaint(
    image,
    mask,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
    mask_prepared: bool = False,
):
    """Compatibility alias for cleanup-owned callers."""

    return ai_inpaint_cleanup(
        image,
        mask,
        use_gpu=use_gpu,
        model_id=model_id,
        mask_prepared=mask_prepared,
    )
