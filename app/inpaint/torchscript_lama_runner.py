# -*- coding: utf-8 -*-
"""Project-owned TorchScript runner for the fixed LaMA cleanup model.

The runner deliberately owns only model I/O:
- load a local TorchScript file;
- normalize image and mask tensors;
- pad model inputs to the model stride;
- crop padded model output back to the caller's original crop;
- return a PIL RGB image.

Cleanup authorization, mask construction, proof, and commit policy remain in
the pipeline cleanup modules.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_MODULO = 8


@dataclass(frozen=True)
class LamaTensorMeta:
    original_width: int
    original_height: int
    padded_width: int
    padded_height: int
    pad_right: int
    pad_bottom: int
    device: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "padded_width": self.padded_width,
            "padded_height": self.padded_height,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "device": self.device,
        }


def resolve_lama_device_name(requested_device: str | None = None, *, use_gpu: bool | None = None) -> str:
    """Return the effective torch device name for cleanup inference."""

    import torch

    requested = str(requested_device or "").strip().lower()
    wants_cuda = requested.startswith("cuda")
    if use_gpu is not None:
        wants_cuda = bool(use_gpu)
    if wants_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TorchScriptLamaRunner:
    """Thin, deterministic TorchScript model runner for cleanup inpainting."""

    def __init__(self, model_path: str | os.PathLike[str], device: str = "cpu") -> None:
        import torch

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TorchScript LaMA model not found: {path}")

        self.model_path = str(path)
        self.device_name = resolve_lama_device_name(device)
        self.device = torch.device(self.device_name)
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        self._infer_lock = threading.Lock()
        self.last_tensor_meta: LamaTensorMeta | None = None

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray) -> Image.Image:
        return self.inpaint(image, mask)

    def inpaint(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray) -> Image.Image:
        import torch

        image_pil = _as_rgb_image(image)
        mask_pil = _as_mask_image(mask, image_pil.size)
        image_tensor, mask_tensor, meta = _prepare_tensors(image_pil, mask_pil, self.device_name)
        self.last_tensor_meta = meta

        with self._infer_lock:
            with torch.inference_mode():
                output = self.model(image_tensor, mask_tensor)

        return _tensor_to_rgb_image(output, meta)


def _as_uint8_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.max(initial=0.0) <= 1.0:
            finite = finite * 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _as_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = _as_uint8_array(np.asarray(image))
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim != 3:
        raise ValueError(f"image must have 2 or 3 dimensions, got shape={arr.shape!r}")
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    raise ValueError(f"unsupported image channel count: shape={arr.shape!r}")


def _as_mask_image(mask: Image.Image | np.ndarray, size: tuple[int, int]) -> Image.Image:
    if isinstance(mask, Image.Image):
        mask_img = mask.convert("L")
    else:
        arr = _as_uint8_array(np.asarray(mask))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"mask must have 2 dimensions, got shape={arr.shape!r}")
        mask_img = Image.fromarray(arr, mode="L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return mask_img


def _ceil_modulo(value: int, modulo: int) -> int:
    if modulo <= 1 or value % modulo == 0:
        return value
    return (value // modulo + 1) * modulo


def _pad_chw_symmetric(chw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    channels, height, width = chw.shape
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)
    if pad_h == 0 and pad_w == 0:
        return chw
    return np.pad(chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")


def _prepare_tensors(
    image: Image.Image,
    mask: Image.Image,
    device_name: str,
) -> tuple[Any, Any, LamaTensorMeta]:
    import torch

    image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    height, width = image_arr.shape[:2]
    padded_h = _ceil_modulo(height, MODEL_MODULO)
    padded_w = _ceil_modulo(width, MODEL_MODULO)

    image_chw = np.transpose(image_arr, (2, 0, 1)).astype(np.float32) / 255.0
    mask_chw = (mask_arr > 0).astype(np.float32)[None, :, :]
    image_chw = _pad_chw_symmetric(image_chw, padded_h, padded_w)
    mask_chw = _pad_chw_symmetric(mask_chw, padded_h, padded_w)

    device = torch.device(device_name)
    image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_chw).unsqueeze(0).to(device)
    meta = LamaTensorMeta(
        original_width=width,
        original_height=height,
        padded_width=padded_w,
        padded_height=padded_h,
        pad_right=padded_w - width,
        pad_bottom=padded_h - height,
        device=device_name,
    )
    return image_tensor, mask_tensor, meta


def _coerce_model_output(output: Any) -> Any:
    if isinstance(output, dict):
        for key in ("inpainted", "output", "predicted", "result"):
            if key in output:
                return output[key]
        if output:
            return next(iter(output.values()))
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("TorchScript LaMA returned an empty output sequence")
        return output[0]
    return output


def _tensor_to_rgb_image(output: Any, meta: LamaTensorMeta) -> Image.Image:
    tensor = _coerce_model_output(output)
    if not hasattr(tensor, "detach"):
        raise RuntimeError(f"TorchScript LaMA returned unsupported output type: {type(output)!r}")
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError(f"TorchScript LaMA returned unsupported tensor shape: {tuple(tensor.shape)!r}")

    channels, height, width = [int(v) for v in tensor.shape]
    if height < meta.original_height or width < meta.original_width:
        raise RuntimeError(
            "TorchScript LaMA output is smaller than requested crop: "
            f"output={(width, height)} requested={(meta.original_width, meta.original_height)}"
        )
    tensor = tensor[:, : meta.original_height, : meta.original_width]

    arr = tensor.permute(1, 2, 0).detach().float().cpu().numpy()
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif channels >= 3:
        arr = arr[:, :, :3]
    else:
        raise RuntimeError(f"unsupported TorchScript LaMA channel count: {channels}")
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
