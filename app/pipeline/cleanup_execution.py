"""Cleanup-owned execution primitives.

This module executes selected cleanup plans without choosing routes or text
admission. AI inpainting ownership lives under app.pipeline, not app.render.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None

from app.pipeline.debug_artifacts import mask_stats
from app.pipeline.debug_runtime import diagnostic_enabled, write_diagnostic_checkpoint
from app.pipeline.cleanup_inpainting import (
    FIXED_CLEANUP_INPAINT_MODEL_ID,
    FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH,
    resolve_cleanup_inpaint_model,
)


def _page014_timeout_diag_enabled() -> bool:
    return diagnostic_enabled("MT_PAGE014_TIMEOUT_DIAGNOSTIC")


def _cleanup_perf_contract_diag_enabled() -> bool:
    return diagnostic_enabled("MT_CLEANUP_PERF_CONTRACT_DIAGNOSTIC")


def _cleanup_perf_contract_json_safe(value: Any) -> Any:
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


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields: Any) -> None:
    if not _cleanup_perf_contract_diag_enabled():
        return
    try:
        write_diagnostic_checkpoint(
            "cleanup_perf_contract_checkpoints.jsonl",
            module="app.pipeline.cleanup_execution",
            stage=stage,
            event=event,
            fields=_cleanup_perf_contract_json_safe(fields),
        )
    except Exception:
        return


def _page014_timeout_checkpoint(stage: str, event: str, **fields: Any) -> None:
    _cleanup_perf_contract_checkpoint(stage, event, **fields)
    if not _page014_timeout_diag_enabled():
        return
    try:
        write_diagnostic_checkpoint(
            "page014_timeout_checkpoints.jsonl",
            module="app.pipeline.cleanup_execution",
            stage=stage,
            event=event,
            fields=fields,
            include_monotonic=False,
        )
    except Exception:
        return


@dataclass(frozen=True)
class CleanupExecutionResult:
    cleaned_image: Any
    backend: str | None = None
    backend_detail: Any = None
    backend_kind: str = ""
    backend_method: str = ""
    requested_model_id: str = ""
    actual_model_name: str = ""
    actual_model_path: str = ""
    model_invocation_attempted: bool = False
    model_invocation_succeeded: bool = False
    effective_inpaint_mode: str | None = None
    crop_bbox: list[int] | None = None
    crop_area: int | None = None
    cleaned_crop: Any = field(default=None, repr=False, compare=False)
    mask_ratio: float | None = None
    errors: list[str] = field(default_factory=list)
    fallback_status: str | None = None


def apply_text_removal(
    image: Any,
    text_mask: Any,
    mode: str,
    use_gpu: bool,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
    debug_info: dict | None = None,
) -> CleanupExecutionResult:
    started = time.time()
    model_required = bool((debug_info or {}).get("model_required"))
    if text_mask is None:
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_mask")
        _page014_timeout_checkpoint("cleanup_apply_text_removal", "end", backend="none", reason="empty_mask")
        return _result(image, debug_info)
    mode = (mode or "fast").lower()
    if debug_info is not None:
        debug_info["effective_inpaint_mode"] = mode
    force_ai_first = (
        mode == "ai"
        and str((debug_info or {}).get("cleanup_tag") or "").strip().lower()
        == "authorized_ai_inpaint_full_mask"
    )
    if force_ai_first and debug_info is not None:
        debug_info["local_fill_shortcut_bypassed"] = True

    if cv2 is None or np is None:
        if force_ai_first or model_required:
            reason = "model_required_backend_unavailable:cv2_or_numpy_unavailable"
            _set_cleanup_backend(
                debug_info,
                "model_required_backend_error",
                backend_detail=reason,
                backend_kind="backend_error",
                model_invocation_attempted=False,
                model_invocation_succeeded=False,
                fallback_reason=reason,
                errors=[reason],
            )
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal",
                "end",
                backend="model_required_backend_error",
                reason=reason,
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
            return _result(image, debug_info)
        print("[TextRemoval] No CV2/numpy, using simple white mask")
        _set_cleanup_backend(debug_info, "white_mask", backend_detail="cv2_or_numpy_unavailable")
        _page014_timeout_checkpoint(
            "cleanup_apply_text_removal",
            "end",
            backend="white_mask",
            reason="cv2_or_numpy_unavailable",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(_apply_white_mask(image, text_mask), debug_info)

    mask_pixels = int((text_mask > 0).sum())
    image_area = int(text_mask.shape[0] * text_mask.shape[1])
    mask_ratio = mask_pixels / max(1, image_area)
    stats = mask_stats(text_mask)
    box = _mask_stats_box(stats)
    if debug_info is not None:
        debug_info["mask_pixels"] = mask_pixels
        debug_info["crop_area"] = image_area
        debug_info["mask_ratio"] = round(mask_ratio, 4)
        debug_info["crop_bbox"] = list(box) if box else None
    _page014_timeout_checkpoint(
        "cleanup_apply_text_removal",
        "start",
        mode=mode,
        use_gpu=use_gpu,
        model_id=model_id,
        mask_pixels=mask_pixels,
        image_area=image_area,
        mask_ratio=round(mask_ratio, 6),
        crop_bbox=list(box) if box else None,
    )

    if not force_ai_first:
        # Always try white bubble fill first (cleanest option)
        bubble_fill = _white_bubble_fill(image, text_mask)
        if bubble_fill is not None:
            print("[TextRemoval] Used WHITE BUBBLE FILL")
            _set_cleanup_backend(debug_info, "white_bubble_fill", backend_detail=None)
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal",
                "end",
                backend="white_bubble_fill",
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
            return _result(bubble_fill, debug_info)

        # Try uniform fill for simple backgrounds (handles most cases)
        uniform_fill = _apply_uniform_fill(image, text_mask)
        if uniform_fill is not None:
            print("[TextRemoval] Used UNIFORM FILL")
            _set_cleanup_backend(debug_info, "uniform_fill", backend_detail=None)
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal",
                "end",
                backend="uniform_fill",
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
            return _result(uniform_fill, debug_info)

    # For remaining complex areas: use AI inpainting if enabled, otherwise CV2
    ai_failed = False
    if mode == "ai" and (use_gpu or force_ai_first or model_required):
        try:
            model_info = _cleanup_inpaint_model_info(model_id)
            print(
                "[TextRemoval] Using AI INPAINT with model: "
                f"{model_info['actual_model_name']} ({model_info['actual_model_path'] or model_id})"
            )
            if debug_info is not None:
                debug_info["model_invocation_attempted"] = True
                debug_info["requested_model_id"] = model_id
                debug_info["actual_model_name"] = model_info["actual_model_name"]
                debug_info["actual_model_path"] = model_info["actual_model_path"]
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal_ai",
                "start",
                model_id=model_id,
                mask_pixels=mask_pixels,
                image_area=image_area,
            )
            from app.pipeline.cleanup_inpainting import ai_inpaint_cleanup
            ai_mask = (np.asarray(text_mask) > 0).astype(np.uint8) * 255
            result = ai_inpaint_cleanup(
                image,
                ai_mask,
                use_gpu=use_gpu,
                model_id=model_id,
                mask_prepared=bool((debug_info or {}).get("model_inpaint_mask_prepared")),
            )
            print("[TextRemoval] AI INPAINT Success")
            _set_cleanup_backend(
                debug_info,
                "cleanup_ai_inpaint",
                backend_detail=model_id,
                backend_kind="model_inpaint",
                backend_method="cleanup_owned_simple_lama_inpaint",
                requested_model_id=model_id,
                actual_model_name=model_info["actual_model_name"],
                actual_model_path=model_info["actual_model_path"],
                model_invocation_attempted=True,
                model_invocation_succeeded=True,
            )
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal",
                "end",
                backend="cleanup_ai_inpaint",
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
            return _result(result, debug_info)
        except Exception as e:
            print(f"[TextRemoval] AI INPAINT failed: {e}, falling back to CV2")
            ai_failed = True
            if force_ai_first or model_required:
                reason = f"model_required_backend_failed:{type(e).__name__}: {e}"
                _set_cleanup_backend(
                    debug_info,
                    "model_required_backend_error",
                    backend_detail=reason,
                    backend_kind="backend_error",
                    backend_method="cleanup_owned_simple_lama_inpaint",
                    requested_model_id=model_id,
                    actual_model_name=_cleanup_inpaint_model_info(model_id)["actual_model_name"],
                    actual_model_path=_cleanup_inpaint_model_info(model_id)["actual_model_path"],
                    model_invocation_attempted=True,
                    model_invocation_succeeded=False,
                    fallback_reason=reason,
                    errors=[reason],
                )
                _page014_timeout_checkpoint(
                    "cleanup_apply_text_removal",
                    "end",
                    backend="model_required_backend_error",
                    reason=reason,
                    elapsed_ms=round((time.time() - started) * 1000.0, 3),
                )
                return _result(image, debug_info)

    print("[TextRemoval] Used CV2 INPAINT (fallback)")
    cv2_detail = "after_ai_failure" if ai_failed else "fallback"
    # CV2 inpainting with moderate dilation for complex areas
    img_np = np.array(image)
    kernel_size = max(5, int(max(text_mask.shape) * 0.004))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, dilated, 5, cv2.INPAINT_NS)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    _set_cleanup_backend(debug_info, "cv2_inpaint", backend_detail=cv2_detail)
    _page014_timeout_checkpoint(
        "cleanup_apply_text_removal",
        "end",
        backend="cv2_inpaint",
        backend_detail=cv2_detail,
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return _result(Image.fromarray(rgb), debug_info)


def apply_local_text_removal(
    image: Any,
    local_mask: Any,
    mode: str,
    use_gpu: bool,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
    cleanup_tag: str | None = None,
    foreground_mask: Any | None = None,
    debug_info: dict | None = None,
    return_full_image: bool = True,
) -> CleanupExecutionResult:
    started = time.time()
    cleanup_tag_normalized = str(cleanup_tag or "").strip().lower()
    model_required = bool((debug_info or {}).get("model_required"))
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "start",
        cleanup_tag=cleanup_tag_normalized,
        mode=(mode or "fast").lower(),
        use_gpu=use_gpu,
        model_id=model_id,
    )
    if cv2 is None or np is None or local_mask is None:
        if model_required:
            reason = "model_required_backend_unavailable:cv2_numpy_or_mask_unavailable"
            _set_cleanup_backend(
                debug_info,
                "model_required_backend_error",
                backend_detail=reason,
                backend_kind="backend_error",
                model_invocation_attempted=False,
                model_invocation_succeeded=False,
                fallback_reason=reason,
                errors=[reason],
            )
        else:
            _set_cleanup_backend(debug_info, "none", backend_detail="cv2_numpy_or_mask_unavailable")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="cv2_numpy_or_mask_unavailable",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    ys, xs = np.where(local_mask > 0)
    if ys.size == 0 or xs.size == 0:
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_local_mask")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="empty_local_mask",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    mask_h, mask_w = local_mask.shape[:2]
    recovered_anchor_local = cleanup_tag_normalized == "speech_recovered_anchor_glyph_local"
    side_caption_local = cleanup_tag_normalized == "side_caption_glyph_local"
    speech_glyph_local = cleanup_tag_normalized == "speech_glyph_local"
    background_narration_inpaint_first = cleanup_tag_normalized == "background_narration_inpaint_first"
    background_narration_root_inpaint = cleanup_tag_normalized == "background_narration_root_inpaint"
    executable_residual_tone_fill = cleanup_tag_normalized == "executable_residual_tone_fill"
    glyph_stroke_local = cleanup_tag_normalized in {"glyph_stroke_local", "background_narration_glyph_local"}
    force_strong_local = (
        cleanup_tag_normalized
        in {"caption_strong", "speech_strong", "authorized_ai_inpaint_full_mask"}
        or recovered_anchor_local
    )
    if force_strong_local:
        pad = max(12, int(max(x1 - x0, y1 - y0) * 0.18))
    else:
        pad = max(6, int(max(x1 - x0, y1 - y0) * 0.10))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask_w, x1 + pad)
    y1 = min(mask_h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="invalid_crop_bounds",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    crop_mask = local_mask[y0:y1, x0:x1]
    if crop_mask.size == 0 or not np.any(crop_mask):
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_crop_mask")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="empty_crop_mask",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    crop_foreground_mask = crop_mask
    if foreground_mask is not None:
        try:
            fg_arr = np.asarray(foreground_mask)
            if fg_arr.shape[:2] == local_mask.shape[:2]:
                local_fg = fg_arr[y0:y1, x0:x1]
                if local_fg.size and np.any(local_fg > 0):
                    crop_foreground_mask = local_fg
        except Exception:
            crop_foreground_mask = crop_mask
    crop_img = image.crop((x0, y0, x1, y1))
    crop_h, crop_w = crop_mask.shape[:2]
    crop_area = crop_mask.shape[0] * crop_mask.shape[1]
    mask_pixels = int((crop_mask > 0).sum())
    local_mode = (mode or "fast").lower()
    requested_mode = local_mode
    if not force_strong_local and (crop_area > 220000 or mask_pixels > 18000):
        local_mode = "fast"
    elif not force_strong_local and local_mode == "ai" and (crop_area > 140000 or mask_pixels > 9000):
        local_mode = "fast"
    mask_ratio = mask_pixels / max(1, crop_area)
    if debug_info is not None:
        debug_info["requested_inpaint_mode"] = requested_mode
        debug_info["effective_inpaint_mode"] = local_mode
        debug_info["cleanup_tag"] = cleanup_tag_normalized
        debug_info["crop_bbox"] = [x0, y0, x1, y1]
        debug_info["crop_area"] = int(crop_area)
        debug_info["mask_pixels"] = int(mask_pixels)
        debug_info["mask_ratio"] = round(mask_ratio, 4)
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "crop",
        cleanup_tag=cleanup_tag_normalized,
        requested_mode=requested_mode,
        effective_mode=local_mode,
        crop_bbox=[x0, y0, x1, y1],
        crop_area=int(crop_area),
        mask_pixels=int(mask_pixels),
        mask_ratio=round(mask_ratio, 6),
    )
    directional_local = (
        mask_ratio <= 0.30
        and (
            crop_h >= crop_w * 1.8
            or crop_w >= crop_h * 2.2
        )
    )
    conservative_local = (
        mask_ratio <= 0.22
        and (
            crop_w <= 140
            or crop_h >= crop_w * 1.8
        )
    )
    if cleanup_tag_normalized == "speech_recovered_anchor_glyph_local":
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=False)
        backend = "speech_strong_local_fill"
        repair_attempted = cleaned is None
        repair_backend = None
        repair_status = "not_required_bright_speech_bubble_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
            repair_backend = backend
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
            repair_backend = backend
        if cleaned is None:
            nested_debug: dict[str, object] = {"model_required": model_required}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                local_mode,
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            repair_backend = backend
            _merge_nested_backend_debug(debug_info, nested_debug)
        if repair_attempted:
            repair_status = "applied" if cleaned is not None else "failed"
        if debug_info is not None:
            debug_info["root_cleanup_repair_v2_attempted"] = bool(repair_attempted)
            debug_info["root_cleanup_repair_v2_status"] = repair_status
            debug_info["root_cleanup_repair_v2_backend"] = repair_backend or backend
            debug_info["root_cleanup_repair_v2_attempt_count"] = 1 if repair_attempted else 0
            debug_info["root_cleanup_repair_v2_reason"] = (
                "speech_strong_local_fill_rejected_complex_or_partial_components"
                if repair_attempted
                else "primary_bright_speech_bubble_fill_accepted"
            )
    elif speech_glyph_local:
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=False)
        backend = "speech_glyph_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif glyph_stroke_local:
        cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
        backend = "glyph_stroke_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif background_narration_inpaint_first:
        if _mask_is_light_text_on_dark_context(crop_img, crop_foreground_mask):
            cleaned = _apply_dark_context_local_fill(crop_img, crop_mask)
            backend = "background_narration_dark_context_fill"
        else:
            cleaned = _apply_light_context_mask_fill(crop_img, crop_mask, crop_foreground_mask)
            if cleaned is not None:
                backend = "background_narration_light_context_mask_fill"
        if cleaned is None and _mask_is_outlined_light_text_on_dark_or_tone_context(
            crop_img,
            crop_mask,
            crop_foreground_mask,
        ):
            cleaned = _apply_dark_context_local_fill(crop_img, crop_mask)
            backend = "background_narration_dark_context_fill"
        if cleaned is None and _mask_is_light_text_on_mixed_context(crop_img, crop_foreground_mask):
            cleaned = _apply_directional_local_fill(crop_img, crop_foreground_mask)
            backend = "background_narration_directional_mixed_context_fill"
        if cleaned is None:
            cleaned = _apply_light_label_dark_glyph_fill(crop_img, crop_mask, crop_foreground_mask)
            if cleaned is not None:
                backend = "background_narration_light_label_dark_glyph_fill"
            if cleaned is None:
                cleaned = _apply_light_context_mask_fill(crop_img, crop_mask, crop_foreground_mask)
                backend = "background_narration_light_context_mask_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_foreground_mask)
            backend = "background_narration_conservative_inpaint"
        repaired = None
        if backend not in {
            "background_narration_light_label_dark_glyph_fill",
            "background_narration_light_context_mask_fill",
        }:
            repaired = _apply_executable_residual_local_fill(crop_img, cleaned, crop_foreground_mask)
        if repaired is not None:
            cleaned = repaired
            backend = "background_narration_inpaint_residual_fill"
        if cleaned is None:
            cleaned = _apply_source_residual_local_fill(crop_img, crop_foreground_mask)
            backend = "background_narration_source_residual_fill"
        if cleaned is None:
            cleaned = _apply_dark_context_local_fill(crop_img, crop_foreground_mask)
            backend = "background_narration_dark_context_fill"
    elif background_narration_root_inpaint:
        cleaned = _apply_mask_faithful_cv2_inpaint(crop_img, crop_mask)
        backend = "background_narration_mask_faithful_cv2_inpaint"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "background_narration_conservative_inpaint"
    elif executable_residual_tone_fill:
        if _mask_is_light_text_on_dark_context(crop_img, crop_mask):
            cleaned = _apply_dark_context_local_fill(crop_img, crop_mask)
            backend = "executable_residual_dark_context_fill"
        else:
            cleaned = _apply_executable_residual_tone_fill(crop_img, crop_mask)
            backend = "executable_residual_tone_fill"
        if cleaned is None:
            cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
            backend = "source_residual_local_fill"
    elif cleanup_tag_normalized == "speech_strong":
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask)
        backend = "speech_strong_local_fill"
        if cleaned is None:
            nested_debug = {"model_required": model_required}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                local_mode,
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            _merge_nested_backend_debug(debug_info, nested_debug)
    elif force_strong_local:
        nested_debug = {"cleanup_tag": cleanup_tag_normalized, "model_required": model_required}
        cleaned = apply_text_removal(
            crop_img,
            crop_mask,
            local_mode,
            use_gpu,
            model_id=model_id,
            debug_info=nested_debug,
        ).cleaned_image
        backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
        _merge_nested_backend_debug(debug_info, nested_debug)
    elif side_caption_local:
        cleaned = _apply_light_context_mask_fill(
            crop_img,
            crop_mask,
            crop_foreground_mask,
            max_context_dark_ratio=0.60,
            max_light_context_luma_delta=24.0,
        )
        backend = "side_caption_light_context_mask_fill"
        if cleaned is None:
            cleaned = _apply_light_label_dark_glyph_fill(crop_img, crop_mask, crop_foreground_mask)
            backend = "side_caption_light_label_dark_glyph_fill"
        if cleaned is None:
            cleaned = _apply_mask_faithful_cv2_inpaint(crop_img, crop_mask)
            backend = "side_caption_mask_faithful_cv2_inpaint"
        if cleaned is None:
            cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=True)
            backend = "speech_strong_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
        if cleaned is None:
            nested_debug = {"model_required": model_required}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                "fast",
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            _merge_nested_backend_debug(debug_info, nested_debug)
            if debug_info is not None:
                debug_info["effective_inpaint_mode"] = "fast"
    elif cleanup_tag_normalized == "cleanup_effectiveness_retry":
        if (
            _mask_is_light_text_on_dark_context(crop_img, crop_foreground_mask)
            or _mask_is_outlined_light_text_on_dark_or_tone_context(
                crop_img,
                crop_mask,
                crop_foreground_mask,
            )
        ):
            cleaned = _apply_dark_context_local_fill(crop_img, crop_foreground_mask)
            backend = "dark_context_residual_local_fill"
        elif _mask_is_light_text_on_mixed_context(crop_img, crop_foreground_mask):
            cleaned = _apply_directional_local_fill(crop_img, crop_foreground_mask)
            backend = "mixed_context_directional_local_fill"
        else:
            cleaned = _apply_light_label_dark_glyph_fill(crop_img, crop_mask, crop_foreground_mask)
            if cleaned is not None:
                backend = "light_label_dark_glyph_fill"
        if cleaned is None:
            cleaned = None
            backend = "source_residual_local_fill"
        if cleaned is None:
            cleaned = _apply_source_residual_local_fill(crop_img, crop_foreground_mask)
            backend = "source_residual_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_foreground_mask)
            backend = "conservative_local_inpaint"
        if cleaned is None:
            nested_debug = {"model_required": model_required}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                "fast",
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            _merge_nested_backend_debug(debug_info, nested_debug)
            if debug_info is not None:
                debug_info["effective_inpaint_mode"] = "fast"
    elif cleanup_tag_normalized == "deterministic_local_fill":
        cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
        backend = "source_residual_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif cleanup_tag_normalized == "minimal_erasure":
        minimal_mask = crop_mask
        if cv2 is not None and np is not None:
            minimal_mask = cv2.erode(
                (crop_mask > 0).astype(np.uint8),
                np.ones((2, 2), np.uint8),
                iterations=1,
            )
            if not np.any(minimal_mask > 0):
                minimal_mask = crop_mask
        cleaned = _apply_source_residual_local_fill(crop_img, minimal_mask)
        backend = "minimal_source_residual_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, minimal_mask)
            backend = "minimal_conservative_local_inpaint"
    elif directional_local:
        cleaned = _apply_directional_local_fill(crop_img, crop_mask)
        backend = "directional_local_fill"
    elif conservative_local:
        cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
        backend = "conservative_local_inpaint"
    else:
        nested_debug = {"model_required": model_required}
        cleaned = apply_text_removal(
            crop_img,
            crop_mask,
            local_mode,
            use_gpu,
            model_id=model_id,
            debug_info=nested_debug,
        ).cleaned_image
        backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
        _merge_nested_backend_debug(debug_info, nested_debug)
    if debug_info is not None:
        debug_info["backend"] = debug_info.get("backend") or backend
    if cleaned is None:
        if model_required:
            reason = f"model_required_backend_failed:{backend}_produced_no_output"
            _set_cleanup_backend(
                debug_info,
                "model_required_backend_error",
                backend_detail=reason,
                backend_kind="backend_error",
                model_invocation_succeeded=False,
                fallback_reason=reason,
                errors=[reason],
            )
        else:
            _set_cleanup_backend(debug_info, "none", backend_detail=f"{backend}_produced_no_output")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason=f"{backend}_produced_no_output",
            cleanup_tag=cleanup_tag_normalized,
            crop_bbox=[x0, y0, x1, y1],
            crop_area=int(crop_area),
            mask_pixels=int(mask_pixels),
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    if not _masked_pixels_changed(crop_img, cleaned, crop_mask) and cleanup_tag_normalized != "authorized_ai_inpaint_full_mask":
        repaired = _apply_authorized_mask_median_fill(crop_img, crop_mask)
        if repaired is not None and _masked_pixels_changed(crop_img, repaired, crop_mask):
            cleaned = repaired
            backend = f"{backend}|authorized_mask_median_noop_repair"
            if debug_info is not None:
                debug_info["backend"] = backend
                debug_info["backend_detail"] = "authorized_mask_median_noop_repair"
                debug_info["noop_repair_applied"] = True
    patched = None
    if return_full_image:
        patched = image.copy()
        patched.paste(cleaned, (x0, y0))
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "end",
        backend=backend,
        cleanup_tag=cleanup_tag_normalized,
        crop_bbox=[x0, y0, x1, y1],
        crop_area=int(crop_area),
        mask_pixels=int(mask_pixels),
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return _result(patched, debug_info, cleaned_crop=cleaned)


def apply_bubble_fill(image: Any, bubble_mask: Any, text_mask: Any, reference_np: Any) -> CleanupExecutionResult:
    if bubble_mask is None or text_mask is None:
        return CleanupExecutionResult(cleaned_image=image, backend="none", backend_detail="missing_mask")
    if cv2 is None or np is None:
        return CleanupExecutionResult(
            cleaned_image=_apply_white_mask(image, text_mask),
            backend="white_mask",
            backend_detail="cv2_or_numpy_unavailable",
        )
    result = np.array(image)
    ref = reference_np if reference_np is not None else result
    num_labels, labels = cv2.connectedComponents(bubble_mask)
    if num_labels <= 1:
        return CleanupExecutionResult(
            cleaned_image=_apply_white_mask(image, text_mask),
            backend="white_mask",
            backend_detail="no_bubble_components",
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_binary = (text_mask > 0).astype(np.uint8)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() == 0:
            continue
        text_region = cv2.bitwise_and(component, text_binary)
        if text_region.sum() == 0:
            continue
        ys, xs = np.where(text_region > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = max(4, int(max(tx1 - tx0, ty1 - ty0) * 0.35))
        bx0 = max(0, tx0 - pad)
        by0 = max(0, ty0 - pad)
        bx1 = min(component.shape[1] - 1, tx1 + pad)
        by1 = min(component.shape[0] - 1, ty1 + pad)
        local = np.zeros_like(component)
        local[by0 : by1 + 1, bx0 : bx1 + 1] = component[by0 : by1 + 1, bx0 : bx1 + 1]
        halo = cv2.dilate(text_region, kernel, iterations=1)
        sample_mask = cv2.bitwise_and(local, cv2.bitwise_not(halo))
        samples = ref[sample_mask > 0]
        if samples.size < 12:
            inner = cv2.erode(component, kernel, iterations=2)
            if inner.sum() > 0:
                samples = ref[inner > 0]
        if samples.size < 3:
            color = (255, 255, 255)
        else:
            median = np.median(samples.reshape(-1, 3), axis=0)
            color = tuple(int(v) for v in median.tolist())
        fill_region = cv2.dilate(text_region, kernel, iterations=2)
        fill_region = cv2.bitwise_and(fill_region, component)
        result[fill_region > 0] = color
    return CleanupExecutionResult(cleaned_image=Image.fromarray(result), backend="bubble_fill")


def _result(image: Any, debug_info: dict | None, cleaned_crop: Any = None) -> CleanupExecutionResult:
    debug_info = debug_info or {}
    return CleanupExecutionResult(
        cleaned_image=image,
        backend=debug_info.get("backend"),
        backend_detail=debug_info.get("backend_detail"),
        backend_kind=str(debug_info.get("backend_kind") or _backend_kind_for_name(debug_info.get("backend"))),
        backend_method=str(debug_info.get("backend_method") or debug_info.get("backend") or ""),
        requested_model_id=str(debug_info.get("requested_model_id") or ""),
        actual_model_name=str(debug_info.get("actual_model_name") or ""),
        actual_model_path=str(debug_info.get("actual_model_path") or ""),
        model_invocation_attempted=bool(debug_info.get("model_invocation_attempted", False)),
        model_invocation_succeeded=bool(debug_info.get("model_invocation_succeeded", False)),
        effective_inpaint_mode=debug_info.get("effective_inpaint_mode"),
        crop_bbox=debug_info.get("crop_bbox"),
        crop_area=debug_info.get("crop_area"),
        cleaned_crop=cleaned_crop,
        mask_ratio=debug_info.get("mask_ratio"),
        errors=list(debug_info.get("errors") or []),
        fallback_status=debug_info.get("fallback_status"),
    )


def _set_cleanup_backend(debug_info: dict | None, backend: str, **fields: object) -> None:
    if debug_info is None:
        return
    debug_info["backend"] = backend
    debug_info.setdefault("backend_kind", _backend_kind_for_name(backend))
    debug_info.setdefault("backend_method", backend)
    for key, value in fields.items():
        debug_info[key] = value


def _merge_nested_backend_debug(debug_info: dict | None, nested_debug: Mapping[str, object]) -> None:
    if debug_info is None:
        return
    for key in (
        "backend_detail",
        "backend_kind",
        "backend_method",
        "requested_model_id",
        "actual_model_name",
        "actual_model_path",
        "model_invocation_attempted",
        "model_invocation_succeeded",
        "fallback_reason",
        "errors",
    ):
        if key in nested_debug:
            debug_info[key] = nested_debug.get(key)


def _backend_kind_for_name(backend: object) -> str:
    name = str(backend or "").strip().lower()
    if not name or name == "none":
        return "noop"
    if "backend_error" in name or "error" in name:
        return "backend_error"
    if "cleanup_ai_inpaint" in name or "simple_lama" in name or "model_inpaint" in name:
        return "model_inpaint"
    if "cv2" in name or "opencv" in name or "conservative_local_inpaint" in name:
        return "opencv_inpaint"
    if "residual" in name or "repair" in name:
        return "residual_repair"
    if any(token in name for token in ("fill", "white", "median", "local", "directional", "bubble")):
        return "heuristic_fill"
    return "heuristic_fill"


def _cleanup_inpaint_model_info(model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID) -> dict[str, str]:
    try:
        return resolve_cleanup_inpaint_model(model_id)
    except Exception:
        return {
            "requested_model_id": str(model_id or ""),
            "configured_model_id": FIXED_CLEANUP_INPAINT_MODEL_ID,
            "selection_policy": "fixed_cleanup_iopaint_model_resolution_error",
            "actual_model_name": "SimpleLama(unknown)",
            "actual_model_path": _cleanup_inpaint_model_path(),
        }


def _cleanup_inpaint_model_path() -> str:
    app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(app_root, *FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH)


def _masked_pixels_changed(before, after, mask) -> bool:
    if cv2 is None or np is None or mask is None:
        return False
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
        if before_np.shape != after_np.shape:
            return False
        mask_bool = np.asarray(mask) > 0
        if mask_bool.shape[:2] != before_np.shape[:2] or not np.any(mask_bool):
            return False
        diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
        changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
        return bool(np.any(changed & mask_bool))
    except Exception:
        return False


def _apply_authorized_mask_median_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.shape[:2] != img_np.shape[:2] or not np.any(mask_u8):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sample_ring = cv2.subtract(cv2.dilate(mask_u8, kernel, iterations=3), mask_u8)
    samples = img_np[sample_ring > 0]
    if samples.size < 12:
        return None
    median = np.median(samples.reshape(-1, 3), axis=0).astype(np.uint8)
    result = img_np.copy()
    result[mask_u8 > 0] = median
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _mask_stats_box(stats) -> tuple[int, int, int, int] | None:
    if not isinstance(stats, dict):
        return None
    box = stats.get("bbox")
    if not box:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _apply_conservative_local_inpaint(image, mask):
    if cv2 is None or np is None or mask is None:
        return image
    img_np = np.array(image)
    if img_np.size == 0:
        return image
    if len(mask.shape) != 2:
        return image
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return image
    max_dim = max(mask_u8.shape[:2])
    kernel_size = 3 if max_dim <= 220 else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    repaired_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    repaired_mask = cv2.bitwise_and(repaired_mask, mask_u8)
    if not np.any(repaired_mask > 0):
        repaired_mask = mask_u8
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    radius = 2 if max_dim <= 220 else 3
    inpainted = cv2.inpaint(img_bgr, repaired_mask, radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_mask_faithful_cv2_inpaint(image, mask):
    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    if len(mask.shape) != 2:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask_u8):
        return None
    ys, xs = np.where(mask_u8 > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    max_span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    radius = 13 if max_span >= 420 else 9 if max_span >= 220 else 7
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, mask_u8, radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_executable_residual_tone_fill(image, mask):
    """Fill proof-derived residual glyph pixels with local light paper tone.

    This is only for residual retry masks that were already derived from the
    executable cleanup foreground. It intentionally avoids pure white so proof
    can continue to reserve broad-white failures for destructive fills.
    """

    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8):
        return None
    if mask_u8.shape[:2] != img_np.shape[:2]:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    near = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = cv2.dilate(near, kernel, iterations=5)
    ring = cv2.subtract(ring, near)
    samples = img_np[ring > 0]
    if samples.size < 18:
        samples = img_np[mask_u8 == 0]
    if samples.size < 18:
        return None

    sample_arr = samples.reshape(-1, 3).astype(np.float32)
    sample_luma = (
        0.299 * sample_arr[:, 0]
        + 0.587 * sample_arr[:, 1]
        + 0.114 * sample_arr[:, 2]
    )
    if float(np.median(sample_luma)) < 215.0:
        return None

    blurred = cv2.GaussianBlur(img_np, (9, 9), 0).astype(np.float32)
    blurred_luma = (
        0.299 * blurred[:, :, 0]
        + 0.587 * blurred[:, :, 1]
        + 0.114 * blurred[:, :, 2]
    )
    target_luma = float(np.clip(np.median(sample_luma) - 18.0, 231.0, 234.0))
    scale = target_luma / np.maximum(1.0, blurred_luma)
    toned = np.clip(blurred * scale[:, :, None], 0, 235)
    toned_luma = (
        0.299 * toned[:, :, 0]
        + 0.587 * toned[:, :, 1]
        + 0.114 * toned[:, :, 2]
    )
    low = (toned_luma < 231.0) & (mask_u8 > 0)
    if np.any(low):
        lift = 231.0 / np.maximum(1.0, toned_luma)
        toned = np.where(low[:, :, None], np.clip(toned * lift[:, :, None], 0, 235), toned)

    result = img_np.copy()
    result[mask_u8 > 0] = toned[mask_u8 > 0].astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_directional_local_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return image
    img_np = np.array(image)
    if img_np.size == 0:
        return image
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return image
    filled = img_np.copy()
    h, w = mask_u8.shape[:2]
    vertical = h >= w * 1.5
    mask_bool = mask_u8 > 0

    if vertical:
        row_fallback = img_np[~mask_bool]
        fallback_color = (
            np.median(row_fallback.reshape(-1, 3), axis=0)
            if row_fallback.size
            else np.array([255, 255, 255], dtype=np.float32)
        )
        for y in range(h):
            xs = np.where(mask_bool[y])[0]
            if xs.size == 0:
                continue
            left = xs[0]
            right = xs[-1]
            left_candidates = np.where(~mask_bool[y, :left])[0]
            right_candidates = np.where(~mask_bool[y, right + 1 :])[0]
            if left_candidates.size and right_candidates.size:
                left_idx = int(left_candidates[-1])
                right_idx = int(right + 1 + right_candidates[0])
                left_color = filled[y, left_idx].astype(np.float32)
                right_color = filled[y, right_idx].astype(np.float32)
                span = max(1, right - left)
                for x in range(left, right + 1):
                    t = (x - left) / span if span > 0 else 0.0
                    filled[y, x] = np.clip((1.0 - t) * left_color + t * right_color, 0, 255)
            elif left_candidates.size:
                filled[y, left : right + 1] = filled[y, int(left_candidates[-1])]
            elif right_candidates.size:
                filled[y, left : right + 1] = filled[y, int(right + 1 + right_candidates[0])]
            else:
                filled[y, left : right + 1] = fallback_color
    else:
        col_fallback = img_np[~mask_bool]
        fallback_color = (
            np.median(col_fallback.reshape(-1, 3), axis=0)
            if col_fallback.size
            else np.array([255, 255, 255], dtype=np.float32)
        )
        for x in range(w):
            ys = np.where(mask_bool[:, x])[0]
            if ys.size == 0:
                continue
            top = ys[0]
            bottom = ys[-1]
            top_candidates = np.where(~mask_bool[:top, x])[0]
            bottom_candidates = np.where(~mask_bool[bottom + 1 :, x])[0]
            if top_candidates.size and bottom_candidates.size:
                top_idx = int(top_candidates[-1])
                bottom_idx = int(bottom + 1 + bottom_candidates[0])
                top_color = filled[top_idx, x].astype(np.float32)
                bottom_color = filled[bottom_idx, x].astype(np.float32)
                span = max(1, bottom - top)
                for y in range(top, bottom + 1):
                    t = (y - top) / span if span > 0 else 0.0
                    filled[y, x] = np.clip((1.0 - t) * top_color + t * bottom_color, 0, 255)
            elif top_candidates.size:
                filled[top : bottom + 1, x] = filled[int(top_candidates[-1]), x]
            elif bottom_candidates.size:
                filled[top : bottom + 1, x] = filled[int(bottom + 1 + bottom_candidates[0])]
            else:
                filled[top : bottom + 1, x] = fallback_color

    blur = cv2.GaussianBlur(filled, (3, 3), 0)
    feather = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    feather = feather.astype(np.float32) / 255.0
    feather = feather[..., None]
    blended = img_np.astype(np.float32) * (1.0 - feather) + blur.astype(np.float32) * feather
    result = img_np.copy()
    result[mask_bool] = blended[mask_bool]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_source_residual_local_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask_u8):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fill_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    fill_mask = cv2.bitwise_and(fill_mask, mask_u8)
    if not np.any(fill_mask):
        fill_mask = mask_u8
    sample_seed = cv2.dilate(mask_u8, kernel, iterations=1)
    sample_ring = cv2.dilate(sample_seed, kernel, iterations=3)
    sample_ring = cv2.subtract(sample_ring, sample_seed)
    samples = img_np[sample_ring > 0]
    if samples.size < 18:
        return None
    sample_arr = samples.reshape(-1, 3)
    median = np.median(sample_arr, axis=0)
    std = float(sample_arr.std(axis=0).mean())
    gray_samples = cv2.cvtColor(sample_arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(-1)
    median_luma = float(np.median(gray_samples))
    if std > 72 and median_luma < 188:
        return None
    result = img_np.copy()
    result[fill_mask > 0] = median.astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_executable_residual_local_fill(source_image, cleaned_image, mask):
    if cv2 is None or np is None or mask is None or cleaned_image is None:
        return None
    source_np = np.array(source_image)
    cleaned_np = np.array(cleaned_image)
    if source_np.size == 0 or cleaned_np.size == 0 or source_np.shape != cleaned_np.shape:
        return None
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return None
    source_gray = cv2.cvtColor(source_np, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(cleaned_np, cv2.COLOR_RGB2GRAY)
    residual = ((mask_u8 > 0) & (source_gray < 210) & (cleaned_gray < 230)).astype(np.uint8)
    light_residual = _light_residual_in_dark_context_mask(source_gray, cleaned_gray, mask_u8)
    has_light_residual = light_residual is not None and np.any(light_residual > 0)
    if has_light_residual:
        residual = np.maximum(residual, light_residual.astype(np.uint8))
    residual_pixels = int(np.count_nonzero(residual))
    if residual_pixels <= 0:
        return None
    source_dark_pixels = int(np.count_nonzero((mask_u8 > 0) & (source_gray < 210)))
    residual_limit = max(24, int(source_dark_pixels * 0.08))
    residual_ratio = residual_pixels / max(1, source_dark_pixels)
    if residual_pixels < residual_limit or residual_ratio < 0.12:
        return None
    if has_light_residual:
        dark_context = _apply_dark_context_local_fill(cleaned_image, residual)
        if dark_context is not None:
            return dark_context
    return _apply_source_residual_local_fill(cleaned_image, residual)


def _apply_dark_context_local_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    near = cv2.dilate(mask_u8, kernel, iterations=2)
    far = cv2.dilate(mask_u8, kernel, iterations=8)
    sample_ring = cv2.subtract(far, near)
    samples = img_np[sample_ring > 0]
    if samples.size < 18:
        samples = img_np[(far > 0) & (mask_u8 == 0)]
    if samples.size < 18:
        return None
    sample_arr = samples.reshape(-1, 3)
    sample_luma = (
        0.299 * sample_arr[:, 0].astype(np.float32)
        + 0.587 * sample_arr[:, 1].astype(np.float32)
        + 0.114 * sample_arr[:, 2].astype(np.float32)
    )
    dark_samples = sample_arr[sample_luma < 180]
    if dark_samples.shape[0] >= 12:
        fill = np.median(dark_samples, axis=0)
    else:
        fill = np.median(sample_arr, axis=0)
    fill_luma = float(0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2])
    if fill_luma > 220:
        return None
    result = img_np.copy()
    result[mask_u8 > 0] = fill.astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_light_label_dark_glyph_fill(image, erase_mask, foreground_mask=None):
    if cv2 is None or np is None or erase_mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    erase_u8 = (np.asarray(erase_mask) > 0).astype(np.uint8)
    if erase_u8.shape[:2] != img_np.shape[:2] or not np.any(erase_u8):
        return None
    if foreground_mask is not None:
        fg_u8 = (np.asarray(foreground_mask) > 0).astype(np.uint8)
        if fg_u8.shape[:2] != img_np.shape[:2] or not np.any(fg_u8):
            fg_u8 = erase_u8
    else:
        fg_u8 = erase_u8
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_bool = fg_u8 > 0
    fg_values = gray[fg_bool]
    if fg_values.size <= 0:
        return None
    foreground_dark_ratio = float(np.count_nonzero(fg_values < 170) / max(1, fg_values.size))
    foreground_median = float(np.median(fg_values))
    if foreground_dark_ratio < 0.20 and foreground_median > 190.0:
        return None
    near_foreground = cv2.dilate(fg_u8, kernel, iterations=1) > 0
    glyph = (fg_u8 > 0) | (((erase_u8 > 0) & near_foreground) & (gray < 210))
    glyph_pixels = int(np.count_nonzero(glyph))
    if glyph_pixels < 8:
        return None
    near = cv2.dilate(fg_u8, kernel, iterations=2) > 0
    local_context = near & ~(fg_u8 > 0)
    context_values = gray[local_context]
    if context_values.size < 12:
        return None
    context_light_ratio = float(np.count_nonzero(context_values >= 225) / max(1, context_values.size))
    context_median = float(np.median(context_values))
    if context_median < 225.0 or context_light_ratio < 0.45:
        return None
    erase_light = (erase_u8 > 0) & (gray >= 225)
    context_light = local_context & (gray >= 225)
    light_samples = img_np[erase_light | context_light]
    if light_samples.size < 18:
        return None
    light_arr = light_samples.reshape(-1, 3)
    light_luma = (
        0.299 * light_arr[:, 0].astype(np.float32)
        + 0.587 * light_arr[:, 1].astype(np.float32)
        + 0.114 * light_arr[:, 2].astype(np.float32)
    )
    if float(np.median(light_luma)) < 232:
        return None
    fill = np.median(light_arr, axis=0)
    fill_luma = float(0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2])
    if fill_luma < 232:
        return None
    if fill_luma > 244:
        fill = fill * (240.0 / max(1.0, fill_luma))
    result = img_np.copy()
    result[glyph] = np.clip(fill, 0, 244).astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_light_context_mask_fill(
    image,
    erase_mask,
    foreground_mask=None,
    *,
    max_context_dark_ratio: float = 0.30,
    max_light_context_luma_delta: float = 10.0,
):
    if cv2 is None or np is None or erase_mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    erase_u8 = (np.asarray(erase_mask) > 0).astype(np.uint8)
    if erase_u8.ndim != 2 or erase_u8.shape[:2] != img_np.shape[:2] or not np.any(erase_u8):
        return None
    if foreground_mask is not None:
        fg_u8 = (np.asarray(foreground_mask) > 0).astype(np.uint8)
        if fg_u8.shape[:2] != erase_u8.shape[:2] or not np.any(fg_u8):
            fg_u8 = erase_u8
    else:
        fg_u8 = erase_u8

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    fg_values = gray[fg_u8 > 0]
    if fg_values.size <= 0:
        return None
    foreground_dark_ratio = float(np.count_nonzero(fg_values < 170) / max(1, fg_values.size))
    foreground_light_ratio = float(np.count_nonzero(fg_values >= 225) / max(1, fg_values.size))
    if foreground_dark_ratio < 0.08 and foreground_light_ratio < 0.18:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near = cv2.dilate(erase_u8, kernel, iterations=1) > 0
    far = cv2.dilate(erase_u8, kernel, iterations=5) > 0
    context = far & ~near & ~(erase_u8 > 0)
    if not np.any(context):
        context = far & ~(erase_u8 > 0)
    context_values = gray[context]
    if context_values.size < 18:
        return None
    context_light_ratio = float(np.count_nonzero(context_values >= 225) / max(1, context_values.size))
    context_dark_ratio = float(np.count_nonzero(context_values < 170) / max(1, context_values.size))
    context_median = float(np.median(context_values))
    if (
        context_median < 225.0
        or context_light_ratio < 0.45
        or context_dark_ratio > max_context_dark_ratio
    ):
        return None

    context_rgb = img_np[context]
    if context_rgb.size < 18:
        return None
    light_context_rgb = context_rgb[context_values >= 225]
    if light_context_rgb.size >= 18:
        fill = np.median(light_context_rgb.reshape(-1, 3), axis=0)
    else:
        fill = np.median(context_rgb.reshape(-1, 3), axis=0)
    fill_luma = float(0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2])
    if abs(fill_luma - context_median) > max_light_context_luma_delta:
        return None

    result = img_np.copy()
    result[erase_u8 > 0] = fill.astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _light_residual_in_dark_context_mask(source_gray, cleaned_gray, mask_u8):
    if cv2 is None or np is None:
        return None
    mask_bool = mask_u8 > 0
    if not np.any(mask_bool):
        return None
    diff = np.abs(source_gray.astype(np.int16) - cleaned_gray.astype(np.int16))
    light_unchanged = mask_bool & (source_gray > 190) & (cleaned_gray > 170) & (diff < 45)
    if not np.any(light_unchanged):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    context = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=2) > 0
    context = context & ~mask_bool
    if not np.any(context):
        return None
    context_values = source_gray[context]
    if context_values.size <= 0:
        return None
    dark_ratio = float(np.count_nonzero(context_values < 170) / max(1, context_values.size))
    median_luma = float(np.median(context_values))
    if not (median_luma < 180 or dark_ratio > 0.15):
        return None
    light_pixels = int(np.count_nonzero(light_unchanged))
    source_light_pixels = int(np.count_nonzero(mask_bool & (source_gray > 190)))
    residual_limit = max(4, int(source_light_pixels * 0.08))
    residual_ratio = light_pixels / max(1, source_light_pixels)
    if light_pixels < residual_limit or residual_ratio < 0.12:
        return None
    return light_unchanged.astype(np.uint8)


def _mask_is_light_text_on_dark_context(image, mask) -> bool:
    if cv2 is None or np is None or mask is None:
        return False
    img_np = np.array(image)
    if img_np.size == 0:
        return False
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return False
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mask_values = gray[mask_u8 > 0]
    if mask_values.size <= 0:
        return False
    light_mask_ratio = float(np.count_nonzero(mask_values > 190) / max(1, mask_values.size))
    if light_mask_ratio < 0.35:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near = cv2.dilate(mask_u8, kernel, iterations=2) > 0
    context = near & ~(mask_u8 > 0)
    if not np.any(context):
        return False
    context_values = gray[context]
    if context_values.size <= 0:
        return False
    context_dark_ratio = float(np.count_nonzero(context_values < 170) / max(1, context_values.size))
    context_median = float(np.median(context_values))
    return bool(context_median < 180 and context_dark_ratio > 0.15)


def _mask_is_outlined_light_text_on_dark_or_tone_context(image, erase_mask, foreground_mask=None) -> bool:
    if cv2 is None or np is None or erase_mask is None:
        return False
    img_np = np.array(image)
    if img_np.size == 0:
        return False
    erase_u8 = (np.asarray(erase_mask) > 0).astype(np.uint8)
    if not np.any(erase_u8):
        return False
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    erase_values = gray[erase_u8 > 0]
    if erase_values.size <= 0:
        return False
    erase_light_ratio = float(np.count_nonzero(erase_values > 190) / max(1, erase_values.size))
    erase_dark_ratio = float(np.count_nonzero(erase_values < 170) / max(1, erase_values.size))
    if erase_light_ratio < 0.35 or erase_dark_ratio < 0.15:
        return False
    if foreground_mask is not None:
        fg_u8 = (np.asarray(foreground_mask) > 0).astype(np.uint8)
        if fg_u8.shape[:2] == erase_u8.shape[:2] and np.any(fg_u8):
            fg_values = gray[fg_u8 > 0]
            if fg_values.size > 0:
                fg_light_ratio = float(np.count_nonzero(fg_values > 190) / max(1, fg_values.size))
                fg_dark_ratio = float(np.count_nonzero(fg_values < 170) / max(1, fg_values.size))
                if fg_light_ratio < 0.12 or fg_dark_ratio < 0.20:
                    return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    near = cv2.dilate(erase_u8, kernel, iterations=2) > 0
    context = near & ~(erase_u8 > 0)
    if not np.any(context):
        return False
    context_values = gray[context]
    if context_values.size <= 0:
        return False
    context_dark_ratio = float(np.count_nonzero(context_values < 170) / max(1, context_values.size))
    context_light_ratio = float(np.count_nonzero(context_values >= 225) / max(1, context_values.size))
    context_median = float(np.median(context_values))
    return bool(context_dark_ratio >= 0.08 and (context_median < 245.0 or context_light_ratio < 0.92))


def _mask_is_light_text_on_mixed_context(image, mask) -> bool:
    if cv2 is None or np is None or mask is None:
        return False
    img_np = np.array(image)
    if img_np.size == 0:
        return False
    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return False
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mask_values = gray[mask_u8 > 0]
    if mask_values.size <= 0:
        return False
    light_mask_ratio = float(np.count_nonzero(mask_values > 190) / max(1, mask_values.size))
    if light_mask_ratio < 0.35:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near = cv2.dilate(mask_u8, kernel, iterations=2) > 0
    context = near & ~(mask_u8 > 0)
    if not np.any(context):
        return False
    context_values = gray[context]
    if context_values.size <= 0:
        return False
    context_dark_ratio = float(np.count_nonzero(context_values < 170) / max(1, context_values.size))
    context_light_ratio = float(np.count_nonzero(context_values >= 225) / max(1, context_values.size))
    context_median = float(np.median(context_values))
    return bool(context_median >= 180 and context_dark_ratio > 0.10 and context_light_ratio > 0.35)


def _apply_uniform_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    if text_mask is None:
        return None
    if not np.any(text_mask):
        return None
    img_np = np.array(image)
    num_labels, labels = cv2.connectedComponents(text_mask)
    if num_labels <= 1:
        return None
    result = img_np.copy()
    applied = False
    skipped_complex = False  # Track if we skipped any complex backgrounds
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() < 8:
            continue

        # First, sample border to determine background complexity
        sample_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sample_dilated = cv2.dilate(component, sample_kernel, iterations=2)
        sample_border = cv2.subtract(sample_dilated, component)
        samples = img_np[sample_border > 0]

        if samples.size < 9:
            # Not enough samples - skip, let AI handle it
            skipped_complex = True
            continue

        sample_arr = samples.reshape(-1, 3)
        std = float(sample_arr.std(axis=0).mean())
        median = np.median(sample_arr, axis=0)

        # ONLY handle uniform backgrounds - skip complex ones for AI
        if std <= 15:
            # Very uniform (white bubble): moderate dilation to preserve borders
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(component, kernel, iterations=2)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        elif std <= 30:
            # Moderately uniform: minimal dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(component, kernel, iterations=1)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        else:
            # Complex background (artwork): SKIP - let AI handle it
            skipped_complex = True
            continue

    # Only return if we processed some regions AND didn't skip any complex ones
    # If we skipped complex regions, return None so AI inpainting handles everything
    if applied and not skipped_complex:
        return Image.fromarray(result)
    return None


def _apply_speech_strong_local_fill(image, text_mask, allow_partial: bool = False):
    if cv2 is None or np is None:
        return None
    if text_mask is None or not np.any(text_mask):
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    num_labels, labels = cv2.connectedComponents((text_mask > 0).astype(np.uint8))
    if num_labels <= 1:
        return None
    result = img_np.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    applied_pixels = 0
    skipped_complex = False
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        component_pixels = int(component.sum())
        if component_pixels < 6:
            continue
        halo = cv2.dilate(component, kernel, iterations=2)
        border = cv2.subtract(halo, component)
        samples = img_np[border > 0]
        if samples.size < 9:
            skipped_complex = True
            continue
        sample_arr = samples.reshape(-1, 3)
        median = np.median(sample_arr, axis=0)
        gray = cv2.cvtColor(sample_arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(-1)
        std = float(sample_arr.std(axis=0).mean())
        median_luma = float(np.median(gray))
        # Speech-bubble cleanup should look like local paper, not CV2 smears.
        # Only use this shortcut when the immediate background is plausibly
        # bubble paper or a gentle tone; complex art still falls back.
        if median_luma < 135 or std > 62:
            skipped_complex = True
            continue
        result[component > 0] = median.astype(np.uint8)
        applied_pixels += component_pixels
    if applied_pixels <= 0:
        return None
    if skipped_complex and not allow_partial:
        return None
    return Image.fromarray(result)


def _apply_white_mask(image, text_mask):
    if cv2 is None or np is None:
        ImageDraw.Draw(image)
        return image
    img_np = np.array(image)
    # Aggressively dilate mask to ensure complete text coverage
    kernel_size = max(7, int(max(text_mask.shape) * 0.006))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=4)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_np[dilated > 0] = (255, 255, 255)
    return Image.fromarray(img_np)


def _white_bubble_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    ys, xs = np.where(text_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 12
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(text_mask.shape[1] - 1, x1 + pad)
    y1 = min(text_mask.shape[0] - 1, y1 + pad)
    img_np = np.array(image)
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean > 220 and std < 18:
        filled = _apply_white_mask(image, text_mask)
        return filled
    return None
