# -*- coding: utf-8 -*-
"""Shared model resolution helpers for startup checks and runtime loaders."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from app.config.defaults import (
    CLEANUP_INPAINT_MODEL_FILE,
    MANGA_OCR_FILES,
    KITSUMED_SPEECH_BUBBLE_MODEL_FILE,
    NOTO_CJK_SC_FONT_FILES,
    NOTO_LATIN_FONT_FILE,
    OGKALU_TEXT_BUBBLE_CONFIG_FILE,
    OGKALU_TEXT_BUBBLE_MODEL_FILE,
    PADDLE_OCR_VL_MMPROJ_FILE,
    PADDLE_OCR_VL_MODEL_FILE,
    PADDLE_OCR_VL_REPO_ID,
    YUZUMARKER_FONT_LABELS_FALLBACK_FILE,
    YUZUMARKER_FONT_LABELS_FILE,
    YUZUMARKER_FONT_LABELS_REPO_ID,
    YUZUMARKER_FONT_ONNX_FILE,
    YUZUMARKER_FONT_ONNX_REPO_ID,
)
from app.platform_services.compute import (
    llama_server_is_launchable,
    resolve_llama_server,
)
from app.platform_services.contracts import PlatformIdentity


def _app_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def models_root() -> str:
    return os.path.join(_app_root(), "models")


def _hf_home() -> str:
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )


def _hf_snapshot_dirs(user: str, repo: str) -> list[str]:
    model_dir = os.path.join(_hf_home(), "hub", f"models--{user}--{repo}", "snapshots")
    if not os.path.isdir(model_dir):
        return []
    dirs: list[str] = []
    for entry in os.listdir(model_dir):
        path = os.path.join(model_dir, entry)
        if os.path.isdir(path):
            dirs.append(path)
    return dirs


def _first_dir_with_files(dirs: Iterable[str], required_files: Iterable[str]) -> Optional[str]:
    required = tuple(required_files)
    for directory in dirs:
        if all(os.path.exists(os.path.join(directory, name)) for name in required):
            return directory
    return None

def resolve_manga_ocr_system_ref() -> Optional[str]:
    snapshot = _first_dir_with_files(
        _hf_snapshot_dirs("kha-white", "manga-ocr-base"),
        MANGA_OCR_FILES,
    )
    if snapshot:
        return snapshot
    return None


def resolve_manga_ocr_local_dir(base_dir: Optional[str] = None) -> Optional[str]:
    target = os.path.join(base_dir or models_root(), "manga-ocr")
    if _first_dir_with_files([target], MANGA_OCR_FILES):
        return target
    return None


def resolve_kitsumed_speech_bubble_model(base_dir: Optional[str] = None) -> Optional[str]:
    path = os.path.join(
        base_dir or models_root(),
        "yolov8m_seg-speech-bubble",
        KITSUMED_SPEECH_BUBBLE_MODEL_FILE,
    )
    return path if os.path.isfile(path) else None


def resolve_ogkalu_text_bubble_model(base_dir: Optional[str] = None) -> Optional[str]:
    path = os.path.join(
        base_dir or models_root(),
        "comic-text-and-bubble-detector",
        OGKALU_TEXT_BUBBLE_MODEL_FILE,
    )
    return path if os.path.isfile(path) else None


def resolve_ogkalu_text_bubble_config(base_dir: Optional[str] = None) -> Optional[str]:
    path = os.path.join(
        base_dir or models_root(),
        "comic-text-and-bubble-detector",
        OGKALU_TEXT_BUBBLE_CONFIG_FILE,
    )
    return path if os.path.isfile(path) else None


def has_bubble_detection_runtime(base_dir: Optional[str] = None) -> bool:
    if not (
        resolve_kitsumed_speech_bubble_model(base_dir)
        and resolve_ogkalu_text_bubble_model(base_dir)
        and resolve_ogkalu_text_bubble_config(base_dir)
    ):
        return False
    try:
        from app.pipeline import bubble_detection_runtime as runtime
    except Exception:
        return False
    return bool(
        callable(getattr(runtime.kit, "letterbox", None))
        and callable(getattr(runtime.kit, "decode_outputs", None))
        and callable(getattr(runtime.og, "preprocess", None))
        and callable(getattr(runtime, "link_regions", None))
        and callable(getattr(runtime, "build_fusion", None))
    )


def resolve_cleanup_inpaint_model_file(base_dir: Optional[str] = None) -> Optional[str]:
    path = os.path.join(
        base_dir or models_root(),
        "inpaint",
        "iopaint",
        CLEANUP_INPAINT_MODEL_FILE,
    )
    return path if os.path.isfile(path) else None


def has_cleanup_inpaint_model(base_dir: Optional[str] = None) -> bool:
    return bool(resolve_cleanup_inpaint_model_file(base_dir))


def _paddle_ocr_vl_local_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "paddleocr-vl-1.6-gguf")


def _paddle_ocr_vl_snapshot_dirs() -> list[str]:
    user, repo = PADDLE_OCR_VL_REPO_ID.split("/", 1)
    return _hf_snapshot_dirs(user, repo)


def resolve_paddle_ocr_vl_model_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_MODEL")
    if override and os.path.isfile(override):
        return override
    local_path = os.path.join(_paddle_ocr_vl_local_dir(base_dir), PADDLE_OCR_VL_MODEL_FILE)
    if os.path.isfile(local_path):
        return local_path
    snapshot = _first_dir_with_files(_paddle_ocr_vl_snapshot_dirs(), [PADDLE_OCR_VL_MODEL_FILE])
    if snapshot:
        return os.path.join(snapshot, PADDLE_OCR_VL_MODEL_FILE)
    return None


def resolve_paddle_ocr_vl_mmproj_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_MMPROJ")
    if override and os.path.isfile(override):
        return override
    local_path = os.path.join(_paddle_ocr_vl_local_dir(base_dir), PADDLE_OCR_VL_MMPROJ_FILE)
    if os.path.isfile(local_path):
        return local_path
    snapshot = _first_dir_with_files(_paddle_ocr_vl_snapshot_dirs(), [PADDLE_OCR_VL_MMPROJ_FILE])
    if snapshot:
        return os.path.join(snapshot, PADDLE_OCR_VL_MMPROJ_FILE)
    return None


def resolve_llama_server_executable(
    base_dir: Optional[str] = None,
    *,
    identity: PlatformIdentity | None = None,
) -> Optional[str]:
    override = os.environ.get("MT_PADDLEOCR_VL_LLAMA_SERVER")
    root = os.path.join(base_dir or models_root(), "llama.cpp")
    resolved = resolve_llama_server(
        identity=identity,
        override=override,
        search_roots=(Path(root),),
    )
    return str(resolved) if resolved is not None else None


def has_paddle_ocr_vl_runtime(
    base_dir: Optional[str] = None,
    *,
    identity: PlatformIdentity | None = None,
) -> bool:
    executable = resolve_llama_server_executable(base_dir, identity=identity)
    return bool(
        resolve_paddle_ocr_vl_model_file(base_dir)
        and resolve_paddle_ocr_vl_mmproj_file(base_dir)
        and executable
        and llama_server_is_launchable(executable)
    )


def _yuzumarker_onnx_local_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "YuzuMarker", "onnx")


def _yuzumarker_labels_local_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "YuzuMarker", "safetensors")


def _yuzumarker_onnx_snapshot_dirs() -> list[str]:
    user, repo = YUZUMARKER_FONT_ONNX_REPO_ID.split("/", 1)
    return _hf_snapshot_dirs(user, repo)


def _yuzumarker_labels_snapshot_dirs() -> list[str]:
    user, repo = YUZUMARKER_FONT_LABELS_REPO_ID.split("/", 1)
    return _hf_snapshot_dirs(user, repo)


def resolve_yuzumarker_font_onnx_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_YUZUMARKER_FONT_ONNX")
    if override and os.path.isfile(override):
        return override
    local_path = os.path.join(_yuzumarker_onnx_local_dir(base_dir), YUZUMARKER_FONT_ONNX_FILE)
    if os.path.isfile(local_path):
        return local_path
    snapshot = _first_dir_with_files(_yuzumarker_onnx_snapshot_dirs(), [YUZUMARKER_FONT_ONNX_FILE])
    if snapshot:
        return os.path.join(snapshot, YUZUMARKER_FONT_ONNX_FILE)
    return None


def resolve_yuzumarker_font_labels_file(base_dir: Optional[str] = None) -> Optional[str]:
    override = os.environ.get("MT_YUZUMARKER_FONT_LABELS")
    if override and os.path.isfile(override):
        return override
    local_dir = _yuzumarker_labels_local_dir(base_dir)
    for filename in (YUZUMARKER_FONT_LABELS_FILE, YUZUMARKER_FONT_LABELS_FALLBACK_FILE):
        local_path = os.path.join(local_dir, filename)
        if os.path.isfile(local_path):
            return local_path
    for filename in (YUZUMARKER_FONT_LABELS_FILE, YUZUMARKER_FONT_LABELS_FALLBACK_FILE):
        snapshot = _first_dir_with_files(_yuzumarker_labels_snapshot_dirs(), [filename])
        if snapshot:
            return os.path.join(snapshot, filename)
    return None


def has_yuzumarker_font_detection_runtime(base_dir: Optional[str] = None) -> bool:
    return bool(
        resolve_yuzumarker_font_onnx_file(base_dir)
        and resolve_yuzumarker_font_labels_file(base_dir)
    )


def _noto_cjk_sc_font_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "fonts", "noto-cjk-sc-core")


def noto_cjk_sc_font_dir(base_dir: Optional[str] = None) -> str:
    return _noto_cjk_sc_font_dir(base_dir)


def has_noto_cjk_sc_font_pack(base_dir: Optional[str] = None) -> bool:
    directory = _noto_cjk_sc_font_dir(base_dir)
    required = [os.path.basename(path) for path in NOTO_CJK_SC_FONT_FILES]
    return all(os.path.isfile(os.path.join(directory, filename)) for filename in required)


def noto_latin_font_dir(base_dir: Optional[str] = None) -> str:
    return os.path.join(base_dir or models_root(), "fonts", "noto-latin-core")


def resolve_noto_latin_variable_font_file(
    base_dir: Optional[str] = None,
) -> Optional[str]:
    path = os.path.join(noto_latin_font_dir(base_dir), NOTO_LATIN_FONT_FILE)
    return path if os.path.isfile(path) else None


def has_noto_latin_font_pack(base_dir: Optional[str] = None) -> bool:
    return resolve_noto_latin_variable_font_file(base_dir) is not None


def resolve_noto_cjk_sc_font_file(
    *,
    base_dir: Optional[str] = None,
    serif: bool = False,
    weight: str = "regular",
    monospace: bool = False,
) -> Optional[str]:
    directory = _noto_cjk_sc_font_dir(base_dir)
    normalized_weight = str(weight or "regular").strip().lower()
    if monospace:
        candidates = (
            "NotoSansMonoCJKsc-Regular.otf",
            "NotoSansCJKsc-Regular.otf",
        )
    elif serif:
        candidates = (
            "NotoSerifCJKsc-Bold.otf",
            "NotoSerifCJKsc-SemiBold.otf",
            "NotoSerifCJKsc-Regular.otf",
        ) if normalized_weight in {"bold", "black", "heavy"} else (
            "NotoSerifCJKsc-SemiBold.otf",
            "NotoSerifCJKsc-Regular.otf",
            "NotoSerifCJKsc-Bold.otf",
        ) if normalized_weight in {"semibold", "semi-bold", "600"} else (
            "NotoSerifCJKsc-Regular.otf",
            "NotoSerifCJKsc-SemiBold.otf",
            "NotoSerifCJKsc-Bold.otf",
        )
    elif normalized_weight in {"black", "heavy"}:
        candidates = (
            "NotoSansCJKsc-Black.otf",
            "NotoSansCJKsc-Bold.otf",
            "NotoSansCJKsc-Regular.otf",
        )
    elif normalized_weight == "bold":
        candidates = (
            "NotoSansCJKsc-Bold.otf",
            "NotoSansCJKsc-Black.otf",
            "NotoSansCJKsc-Regular.otf",
        )
    elif normalized_weight in {"medium", "500"}:
        candidates = (
            "NotoSansCJKsc-Medium.otf",
            "NotoSansCJKsc-Regular.otf",
            "NotoSansCJKsc-Bold.otf",
        )
    else:
        candidates = (
            "NotoSansCJKsc-Regular.otf",
            "NotoSansCJKsc-Medium.otf",
            "NotoSansCJKsc-Bold.otf",
            "NotoSansCJKsc-Black.otf",
        )
    for filename in candidates:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return None


def has_font_style_runtime(base_dir: Optional[str] = None) -> bool:
    return bool(
        has_yuzumarker_font_detection_runtime(base_dir)
        and has_noto_cjk_sc_font_pack(base_dir)
    )


def resolve_ner_system_snapshot() -> Optional[str]:
    required = ("config.json", "tokenizer_config.json", "vocab.txt")
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for snapshot in _hf_snapshot_dirs("jurabi", "bert-ner-japanese"):
        if not all(os.path.exists(os.path.join(snapshot, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(snapshot, name)) for name in optional_weights):
            return snapshot
    return None


def resolve_ner_local_dir(model_dir: Optional[str] = None) -> Optional[str]:
    base_dir = model_dir or os.path.join(models_root(), "ner")
    candidates = [base_dir]
    nested = os.path.join(base_dir, "models--jurabi--bert-ner-japanese", "snapshots")
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            path = os.path.join(nested, entry)
            if os.path.isdir(path):
                candidates.append(path)

    required = ("config.json", "tokenizer_config.json", "vocab.txt")
    optional_weights = ("model.safetensors", "pytorch_model.bin")
    for candidate in candidates:
        if not all(os.path.exists(os.path.join(candidate, name)) for name in required):
            continue
        if any(os.path.exists(os.path.join(candidate, name)) for name in optional_weights):
            return candidate
    return None
