# -*- coding: utf-8 -*-
"""AI inpainting using AnimeMangaInpainting when available."""
from __future__ import annotations
from functools import lru_cache

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


@lru_cache(maxsize=2)
def _load_pipe(model_id: str, device: str):
    try:
        from diffusers import AutoPipelineForInpainting
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("diffusers is not installed") from exc
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is not installed") from exc

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)
    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None
    pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def ai_inpaint(image, mask, use_gpu: bool = True, model_id: str = "dreMaz/AnimeMangaInpainting"):
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    device = "cuda" if use_gpu else "cpu"
    pipe = _load_pipe(model_id, device)
    mask_image = Image.fromarray(mask).convert("L")
    bbox = mask_image.getbbox()
    if not bbox:
        return image
    x0, y0, x1, y1 = bbox
    pad = 16
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width, x1 + pad)
    y1 = min(image.height, y1 + pad)
    crop_img = image.crop((x0, y0, x1, y1))
    crop_mask = mask_image.crop((x0, y0, x1, y1))
    prompt = "clean background, remove text, keep edges"
    negative_prompt = "text, letters, watermark, symbols"
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=crop_img,
        mask_image=crop_mask,
        num_inference_steps=18,
        guidance_scale=0.0,
    )
    out = image.copy()
    out.paste(result.images[0], (x0, y0))
    return out
