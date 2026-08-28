# -*- coding: utf-8 -*-
"""Default settings."""
from dataclasses import dataclass

# Model URLs
COMIC_TEXT_DETECTOR_GPU = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt"
COMIC_TEXT_DETECTOR_CPU = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx"
SAKURA_GGUF = "https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF/resolve/main/sakura-14b-qwen3-v1.5-q6k.gguf"
QWEN_GGUF = "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q6_K.gguf"
BIG_LAMA = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
IOPAINT_ANIME_MANGA_BIG_LAMA = "iopaint/anime-manga-big-lama"
KITSUMED_SPEECH_BUBBLE_REPO_ID = "kitsumed/yolov8m_seg-speech-bubble"
KITSUMED_SPEECH_BUBBLE_MODEL_FILE = "model_dynamic.onnx"
OGKALU_TEXT_BUBBLE_REPO_ID = "ogkalu/comic-text-and-bubble-detector"
OGKALU_TEXT_BUBBLE_MODEL_FILE = "detector.onnx"
OGKALU_TEXT_BUBBLE_CONFIG_FILE = "config.json"
CLEANUP_INPAINT_REPO_ID = "df1412/anime-big-lama"
CLEANUP_INPAINT_MODEL_FILE = "anime-manga-big-lama.pt"
PADDLE_OCR_VL_REPO_ID = "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
PADDLE_OCR_VL_MODEL_FILE = "PaddleOCR-VL-1.6-GGUF.gguf"
PADDLE_OCR_VL_MMPROJ_FILE = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
YUZUMARKER_FONT_ONNX_REPO_ID = "ogkalu/yuzumarker-font-detection-onnx"
YUZUMARKER_FONT_ONNX_FILE = "font-detector.onnx"
YUZUMARKER_FONT_LABELS_REPO_ID = "fffonion/yuzumarker-font-detection"
YUZUMARKER_FONT_LABELS_FILE = "font-labels-ex.json"
YUZUMARKER_FONT_LABELS_FALLBACK_FILE = "font-labels.json"
NOTO_CJK_SC_FONT_BASE_URL = "https://raw.githubusercontent.com/notofonts/noto-cjk/main"
NOTO_CJK_SC_FONT_FILES = [
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Medium.otf",
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Black.otf",
    "Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf",
    "Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-SemiBold.otf",
    "Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf",
    "Sans/Mono/NotoSansMonoCJKsc-Regular.otf",
]
SIL_OFL_TEXT_URL = "https://openfontlicense.org/documents/OFL.txt"
MANGA_OCR_BASE_URL = "https://huggingface.co/kha-white/manga-ocr-base/resolve/main/"
MANGA_OCR_FILES = [
    "config.json",
    "preprocessor_config.json", 
    "pytorch_model.bin",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer_config.json"
]


@dataclass
class AppDefaults:
    source_language: str = "Japanese"
    target_language: str = "Simplified Chinese"
    output_suffix: str = "_translated"
    theme: str = "dark"
    json_path: str = ""
    import_dir: str = ""
    export_dir: str = ""
    font_name: str = "Noto Sans CJK SC"
    font_detection: str = "yuzumarker"
    detector_input_size: int = 1024
    detector_engine: str = "ComicTextDetector"
    ocr_engine: str = "PaddleOCR-VL"
    filter_strength: str = "normal"
    inpaint_mode: str = "ai"
    # Cleanup inpainting uses one fixed local model. The UI value is preserved
    # for compatibility/provenance only; cleanup does not load arbitrary paths.
    inpaint_model: str = IOPAINT_ANIME_MANGA_BIG_LAMA
    translator_backend: str = "GGUF"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    gguf_model_path: str = ""
    gguf_prompt_style: str = "sakura"
    gguf_n_ctx: int = 4096
    # -1 is the GUI's Automatic policy. Resource admission resolves it to one
    # exact per-run layer count before controller materialization.
    gguf_n_gpu_layers: int = -1
    gguf_n_threads: int = 8
    gguf_n_batch: int = 256
    gguf_cross_page_context: bool = False
    fast_mode: bool = False
    auto_glossary: bool = True
    
    # Generation Options
    ollama_temperature: float = 0.2
    ollama_top_p: float = 0.9
    ollama_context: int = 4096
    ollama_base_url: str = "http://localhost:11434"
    gguf_temperature: float = 0.2
    gguf_top_p: float = 0.95


def get_defaults() -> AppDefaults:
    return AppDefaults()
