# -*- coding: utf-8 -*-
"""MangaOCR wrapper."""
from __future__ import annotations
import os
import sys
import ctypes
from pathlib import Path


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


class MangaOcrEngine:
    def __init__(self, use_gpu: bool) -> None:
        _add_dll_search_paths()
        _preload_torch_dlls()
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(f"Failed to load torch: {exc}") from exc
        try:
            from manga_ocr import MangaOcr
        except Exception as exc:
            raise RuntimeError(f"Failed to import manga-ocr: {exc}") from exc
        
        # Check for local model
        model_path = os.path.join(os.getcwd(), "models", "manga-ocr")
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            self._engine = MangaOcr(pretrained_model_name_or_path=model_path, force_cpu=not use_gpu)
        else:
            self._engine = MangaOcr(force_cpu=not use_gpu)

    def recognize(self, image) -> str:
        return self._engine(image)
