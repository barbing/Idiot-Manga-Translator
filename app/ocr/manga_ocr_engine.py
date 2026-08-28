# -*- coding: utf-8 -*-
"""MangaOCR wrapper."""
from __future__ import annotations
import os
import sys
import ctypes
import shutil
import tempfile
import logging
from pathlib import Path
from app.models.resolution import models_root, resolve_manga_ocr_local_dir, resolve_manga_ocr_system_ref
from app.platform_services.compute import (
    TorchDeviceSelection,
    release_torch_memory,
    select_torch_device,
)
from app.platform_services.contracts import ComputeBackend


logger = logging.getLogger(__name__)

_MANGA_OCR_AUX_FILES = (
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def manga_ocr_force_cpu(selection: TorchDeviceSelection) -> bool:
    return selection.backend is ComputeBackend.CPU


def _add_dll_search_paths() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.prefix) / "DLLs",
        Path(sys.prefix),
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
    ]
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = f"{torch_lib};{os.environ.get('PATH','')}"


def _preload_torch_dlls() -> None:
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.exists():
        return
    for name in ("shm.dll", "torch_cpu.dll", "torch_cuda.dll", "torch.dll"):
        path = torch_lib / name
        if path.exists():
            try:
                ctypes.WinDLL(str(path))
            except OSError:
                pass


def ensure_torch_runtime_ready():
    """Prepare DLL paths and import torch before OCR model init."""
    _add_dll_search_paths()
    _preload_torch_dlls()
    import torch
    return torch


def _torch_version_tuple() -> tuple[int, int]:
    torch = ensure_torch_runtime_ready()
    raw = str(getattr(torch, "__version__", "0")).split("+", 1)[0]
    parts = raw.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor_text = parts[1] if len(parts) > 1 else "0"
    digits = "".join(ch for ch in minor_text if ch.isdigit())
    minor = int(digits) if digits else 0
    return major, minor


def _torch_needs_manga_ocr_safetensors() -> bool:
    return _torch_version_tuple() < (2, 6)


def _has_manga_ocr_safetensors(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        return False
    return all(os.path.isfile(os.path.join(model_dir, name)) for name in _MANGA_OCR_AUX_FILES)


def _prepare_manga_ocr_safetensors_dir(model_dir: str) -> str:
    """
    Create a safetensors-compatible local MangaOCR directory for torch<2.6.

    transformers 4.57+ blocks loading `pytorch_model.bin` through `from_pretrained`
    when torch<2.6 because of CVE-2025-32434. We keep the environment unchanged
    and prepare a trusted local safetensors copy once instead.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return model_dir
    if _has_manga_ocr_safetensors(model_dir):
        return model_dir
    if not _torch_needs_manga_ocr_safetensors():
        return model_dir

    source_bin = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.isfile(source_bin):
        return model_dir

    compat_root = os.path.join(models_root(), "manga-ocr-safe")
    compat_dir = os.path.join(compat_root, Path(model_dir).name or "default")
    if _has_manga_ocr_safetensors(compat_dir):
        return compat_dir

    os.makedirs(compat_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f"{Path(model_dir).name}-", dir=compat_root)
    try:
        torch = ensure_torch_runtime_ready()
        from manga_ocr.ocr import MangaOcrModel
        from transformers import VisionEncoderDecoderConfig

        logger.info(
            "Preparing MangaOCR safetensors compatibility copy because torch %s is below 2.6.",
            getattr(torch, "__version__", "unknown"),
        )
        state_dict = torch.load(source_bin, map_location="cpu", weights_only=True)
        config = VisionEncoderDecoderConfig.from_pretrained(model_dir)
        model = MangaOcrModel(config)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"MangaOCR safetensors conversion missing keys: {missing[:5]}")
        if unexpected:
            logger.warning("MangaOCR safetensors conversion ignored unexpected keys: %s", unexpected[:5])
        model.save_pretrained(temp_dir, safe_serialization=True)

        for name in os.listdir(model_dir):
            src = os.path.join(model_dir, name)
            dst = os.path.join(temp_dir, name)
            if not os.path.isfile(src):
                continue
            if name in {"pytorch_model.bin", "model.safetensors"}:
                continue
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        marker_path = os.path.join(temp_dir, ".prepared_from")
        with open(marker_path, "w", encoding="utf-8") as fh:
            fh.write(model_dir)

        if os.path.isdir(compat_dir):
            shutil.rmtree(compat_dir, ignore_errors=True)
        shutil.move(temp_dir, compat_dir)
        temp_dir = ""
        logger.info("Prepared MangaOCR safetensors compatibility copy at %s", compat_dir)
        return compat_dir
    except Exception:
        logger.exception("Failed to prepare MangaOCR safetensors compatibility copy.")
        raise
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def resolve_manga_ocr_model_ref() -> str | None:
    """Resolve the best MangaOCR model reference: system cache, local path, or None."""
    return resolve_manga_ocr_system_ref() or resolve_manga_ocr_local_dir()


def create_manga_ocr_instance(use_gpu: bool):
    """Create MangaOCR instance using shared model resolution logic."""
    try:
        from manga_ocr import MangaOcr
    except Exception as exc:
        raise RuntimeError(f"Failed to import manga-ocr: {exc}") from exc

    selection = select_torch_device(use_gpu)
    force_cpu = manga_ocr_force_cpu(selection)
    model_ref = resolve_manga_ocr_model_ref()
    if model_ref and os.path.isdir(model_ref):
        model_ref = _prepare_manga_ocr_safetensors_dir(model_ref)
    if model_ref:
        return MangaOcr(
            pretrained_model_name_or_path=model_ref,
            force_cpu=force_cpu,
        )
    # Final fallback (library default behavior, may download)
    return MangaOcr(force_cpu=force_cpu)


class MangaOcrEngine:
    @staticmethod
    def _prepare_input_image(image):
        """Normalize OCR crops so tiny page-space slices do not crash MangaOCR."""
        try:
            import numpy as np
            from PIL import Image, ImageOps
            import cv2

            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            if hasattr(image, "convert"):
                image = image.convert("RGB")
                width, height = image.size
                min_side = 8
                if width < min_side or height < min_side:
                    pad_w = max(0, min_side - width)
                    pad_h = max(0, min_side - height)
                    image = ImageOps.expand(
                        image,
                        border=(
                            pad_w // 2,
                            pad_h // 2,
                            pad_w - (pad_w // 2),
                            pad_h - (pad_h // 2),
                        ),
                        fill="white",
                    )
            return image
        except Exception:
            return image

    def recognize_with_confidence(self, image) -> tuple[str, float]:
        """Recognize text and return confidence score."""
        image = self._prepare_input_image(image)

        try:
            import torch
            import numpy as np
            
            # Access internal components
            if not hasattr(self._engine, "model") or not hasattr(self._engine, "processor"):
                return self._engine(image), 1.0

            model = self._engine.model
            processor = self._engine.processor
            tokenizer = getattr(self._engine, "tokenizer", None)
            if tokenizer is None:
                return self._engine(image), 1.0
            
            # Prepare input - Match manga-ocr behavior exactly
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(model.device)
            if model.device.type == "cuda":
                pixel_values = pixel_values.to(model.dtype)

            # Generate with scores
            # Forces greedy decoding (num_beams=1) to match standard behavior
            # AND include standard args (if any) from the engine wrapper
            gen_args = getattr(self._engine, "args", {})
            
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values, 
                    output_scores=True,
                    return_dict_in_generate=True,
                    **gen_args,
                )
            
            sequences = outputs.sequences
            scores = outputs.scores
            
            text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            # CRITICAL: Clean up artifacts (SentencePiece spaces)
            text = text.replace(" ", "")
            
            # Calculate confidence (mean probability)
            gens = sequences[0]
            if hasattr(model.config, "decoder_start_token_id") and gens[0] == model.config.decoder_start_token_id:
                gens = gens[1:]
            
            probs = []
            for i, score_step in enumerate(scores):
                if i >= len(gens): break
                token_id = gens[i]
                if token_id == tokenizer.eos_token_id:
                    break
                    
                step_probs = torch.softmax(score_step, dim=-1)
                prob = step_probs[0, token_id].item()
                probs.append(prob)
            
            if not probs:
                confidence = 0.0
            else:
                confidence = float(np.mean(probs))
                
            return text, confidence
            
        except Exception as e:
            print(f"[MangaOCR] Confidence extraction failed: {e}")
            try:
                return self._engine(image), 1.0
            except Exception as fallback_exc:
                print(f"[MangaOCR] OCR fallback failed: {fallback_exc}")
                return "", 0.0

    def __init__(self, use_gpu: bool) -> None:
        try:
            ensure_torch_runtime_ready()
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"Failed to load torch: {exc}") from exc
        selection = select_torch_device(use_gpu)
        self.selected_backend = selection.backend.value
        self.provider_fallback_reason = selection.fallback_reason
        self._engine = create_manga_ocr_instance(use_gpu)

    def recognize(self, image) -> str:
        image = self._prepare_input_image(image)
        try:
            return self._engine(image)
        except Exception as exc:
            print(f"[MangaOCR] OCR failed: {exc}")
            return ""

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_engine"):
            del self._engine
        
        try:
            import gc

            release_torch_memory(synchronize=True)
            gc.collect()
        except Exception:
            pass
