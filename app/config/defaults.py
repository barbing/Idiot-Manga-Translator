# -*- coding: utf-8 -*-
"""Default settings."""
from dataclasses import dataclass


@dataclass
class AppDefaults:
    source_language: str = "Japanese"
    target_language: str = "Simplified Chinese"
    output_suffix: str = "_translated"
    theme: str = "dark"
    json_path: str = ""
    import_dir: str = ""
    export_dir: str = ""
    font_name: str = "Microsoft YaHei"
    font_detection: str = "heuristic"
    detector_engine: str = "ComicTextDetector"
    ocr_engine: str = "MangaOCR"
    filter_strength: str = "normal"
    inpaint_mode: str = "ai"
    translator_backend: str = "Ollama"
    gguf_model_path: str = ""
    gguf_prompt_style: str = "sakura"
    gguf_n_ctx: int = 2048
    gguf_n_gpu_layers: int = -1
    gguf_n_threads: int = 8
    gguf_n_batch: int = 256
    fast_mode: bool = False
    auto_glossary: bool = True


def get_defaults() -> AppDefaults:
    return AppDefaults()
