"""Platform-aware runtime asset catalog and immutable download targets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from app.config.defaults import (
    PADDLE_OCR_VL_MMPROJ_FILE,
    PADDLE_OCR_VL_MODEL_FILE,
    PADDLE_OCR_VL_REPO_ID,
)

from .contracts import OperatingSystem, PlatformIdentity


SUPPORTED_DESKTOP_PLATFORMS = frozenset(
    {OperatingSystem.WINDOWS, OperatingSystem.MACOS}
)


@dataclass(frozen=True, slots=True)
class RuntimeAssetTarget:
    url: str
    relative_path: str
    label: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.url.strip() or not self.relative_path.strip() or not self.label.strip():
            raise ValueError("runtime asset target fields are required")
        if self.relative_path.startswith(("/", "\\")):
            raise ValueError("runtime asset target path must be relative")


@dataclass(frozen=True, slots=True)
class RuntimeAssetSpec:
    asset_id: str
    name: str
    detail: str
    platforms: frozenset[OperatingSystem]
    checker: str
    preparer: str | None
    remediation: Mapping[OperatingSystem, str]

    def __post_init__(self) -> None:
        if not self.asset_id.strip() or not self.name.strip() or not self.detail.strip():
            raise ValueError("runtime asset identity and copy are required")
        if not self.checker.strip():
            raise ValueError("runtime asset checker is required")
        if not self.platforms:
            raise ValueError("runtime asset must support at least one platform")
        if any(not isinstance(item, OperatingSystem) for item in self.platforms):
            raise TypeError("runtime asset platforms must be OperatingSystem values")
        normalized = {
            OperatingSystem(key): str(value).strip()
            for key, value in self.remediation.items()
            if str(value).strip()
        }
        object.__setattr__(self, "remediation", MappingProxyType(normalized))

    def remediation_for(
        self,
        identity: PlatformIdentity | OperatingSystem,
    ) -> str:
        operating_system = (
            identity.os if isinstance(identity, PlatformIdentity) else OperatingSystem(identity)
        )
        return self.remediation.get(
            operating_system,
            "Install this runtime asset for the current platform and verify again.",
        )


def _download_remediation(name: str) -> dict[OperatingSystem, str]:
    message = f"Use Download to install {name}, then run Verify all again."
    return {
        OperatingSystem.WINDOWS: message,
        OperatingSystem.MACOS: message,
    }


def runtime_asset_catalog(
    identity: PlatformIdentity | None = None,
) -> tuple[RuntimeAssetSpec, ...]:
    selected = identity or PlatformIdentity.detect()
    pyicu_preparer = (
        "prepare_pyicu_runtime"
        if selected.os is OperatingSystem.WINDOWS
        else None
    )
    specs = (
        RuntimeAssetSpec(
            "comic_text_detector",
            "ComicTextDetector",
            "Portable ONNX and CUDA Torch detector files",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_comic_text_detector",
            "prepare_comic_text_detector",
            _download_remediation("the ComicTextDetector models"),
        ),
        RuntimeAssetSpec(
            "bubble_detection",
            "Bubble detection",
            "Speech-bubble and text-area semantic evidence models",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_bubble_detection",
            "prepare_bubble_detection",
            _download_remediation("the bubble-detection models"),
        ),
        RuntimeAssetSpec(
            "paddle_ocr_vl",
            "PaddleOCR-VL",
            "GGUF model, projector, and platform-native llama.cpp runtime",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_paddle_ocr_vl",
            "prepare_paddle_ocr_vl",
            {
                OperatingSystem.WINDOWS: (
                    "Use Download to install the PaddleOCR-VL model and managed "
                    "Windows CUDA runtime, then verify again."
                ),
                OperatingSystem.MACOS: (
                    "From the repository root, run `conda env update -n "
                    "manga-llm -f environments/macos.yml --prune`, restart "
                    "YomiFrame, then use Download for the model files and run "
                    "Verify all again."
                ),
            },
        ),
        RuntimeAssetSpec(
            "manga_ocr",
            "MangaOCR",
            "Japanese manga recognition model",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_manga_ocr",
            "prepare_manga_ocr",
            _download_remediation("the MangaOCR model"),
        ),
        RuntimeAssetSpec(
            "cleanup_inpaint",
            "Cleanup inpainting",
            "TorchScript LaMa cleanup model",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_big_lama",
            "prepare_big_lama",
            _download_remediation("the cleanup inpainting model"),
        ),
        RuntimeAssetSpec(
            "ner",
            "Japanese NER",
            "BERT Japanese entity model",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_ner",
            "prepare_ner",
            _download_remediation("the Japanese NER model"),
        ),
        RuntimeAssetSpec(
            "font_detection",
            "YuzuMarker font detection",
            "ONNX font classifier and label metadata",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_yuzumarker_font_detection",
            "prepare_yuzumarker_font_detection",
            _download_remediation("the YuzuMarker font-detection assets"),
        ),
        RuntimeAssetSpec(
            "font_pack",
            "Noto CJK font pack",
            "Local CJK fallback fonts",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_noto_cjk_sc_font_pack",
            "prepare_noto_cjk_sc_font_pack",
            _download_remediation("the Noto CJK fallback font pack"),
        ),
        RuntimeAssetSpec(
            "pyicu",
            "PyICU line breaking",
            "PyICU 2.16.2 with ICU 78.3 strict line breaking",
            SUPPORTED_DESKTOP_PLATFORMS,
            "check_pyicu_runtime",
            pyicu_preparer,
            {
                OperatingSystem.WINDOWS: (
                    "Use Download to install the pinned managed PyICU runtime, "
                    "then verify again."
                ),
                OperatingSystem.MACOS: (
                    "Install the Conda runtime with `conda install -n manga-llm "
                    "-c conda-forge pyicu=2.16.2 icu=78.3`, restart YomiFrame, "
                    "then run Verify all again."
                ),
            },
        ),
    )
    return tuple(item for item in specs if selected.os in item.platforms)


def runtime_asset_spec(
    asset_id: str,
    identity: PlatformIdentity | None = None,
) -> RuntimeAssetSpec:
    normalized = str(asset_id or "").strip()
    for item in runtime_asset_catalog(identity):
        if item.asset_id == normalized:
            return item
    raise KeyError(f"unsupported runtime asset: {normalized}")


def required_runtime_asset_ids(
    pipeline_values: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the fixed assets required by one compiled run candidate."""

    if not isinstance(pipeline_values, Mapping):
        raise TypeError("pipeline_values must be mapping-like")

    required = {
        "bubble_detection",
        "font_pack",
        "pyicu",
    }
    detector = str(
        pipeline_values.get("detector_engine") or "ComicTextDetector"
    ).strip()
    if detector.casefold() == "comictextdetector":
        required.add("comic_text_detector")

    ocr = str(
        pipeline_values.get("ocr_engine") or "PaddleOCR-VL"
    ).strip().casefold()
    required.add("manga_ocr" if ocr in {"mangaocr", "manga-ocr"} else "paddle_ocr_vl")

    inpaint_mode = str(
        pipeline_values.get("inpaint_mode") or "ai"
    ).strip().casefold()
    if inpaint_mode not in {"off", "none", "disabled"}:
        required.add("cleanup_inpaint")

    font_detection = str(
        pipeline_values.get("font_detection") or "yuzumarker"
    ).strip().casefold()
    if font_detection == "yuzumarker":
        required.add("font_detection")

    if bool(pipeline_values.get("prescan_use_ner", False)):
        required.add("ner")

    catalog_order = (
        "comic_text_detector",
        "bubble_detection",
        "paddle_ocr_vl",
        "manga_ocr",
        "cleanup_inpaint",
        "ner",
        "font_detection",
        "font_pack",
        "pyicu",
    )
    return tuple(asset_id for asset_id in catalog_order if asset_id in required)


def paddle_targets(
    identity: PlatformIdentity | None = None,
) -> tuple[RuntimeAssetTarget, ...]:
    selected = identity or PlatformIdentity.detect()
    repo_url = f"https://huggingface.co/{PADDLE_OCR_VL_REPO_ID}/resolve/main"
    targets = [
        RuntimeAssetTarget(
            url=f"{repo_url}/{PADDLE_OCR_VL_MODEL_FILE}",
            relative_path=f"paddleocr-vl-1.6-gguf/{PADDLE_OCR_VL_MODEL_FILE}",
            label="Downloading PaddleOCR-VL GGUF model...",
        ),
        RuntimeAssetTarget(
            url=f"{repo_url}/{PADDLE_OCR_VL_MMPROJ_FILE}",
            relative_path=f"paddleocr-vl-1.6-gguf/{PADDLE_OCR_VL_MMPROJ_FILE}",
            label="Downloading PaddleOCR-VL multimodal projector...",
        ),
    ]
    if selected.os is OperatingSystem.WINDOWS:
        targets.extend(
            (
                RuntimeAssetTarget(
                    url=(
                        "https://github.com/ggml-org/llama.cpp/releases/download/"
                        "b9842/llama-b9842-bin-win-cuda-12.4-x64.zip"
                    ),
                    relative_path=(
                        "llama.cpp/llama-b9842-bin-win-cuda-12.4-x64.zip"
                    ),
                    label="Downloading llama.cpp Windows CUDA runtime...",
                ),
                RuntimeAssetTarget(
                    url=(
                        "https://github.com/ggml-org/llama.cpp/releases/download/"
                        "b9842/cudart-llama-bin-win-cuda-12.4-x64.zip"
                    ),
                    relative_path=(
                        "llama.cpp/cudart-llama-bin-win-cuda-12.4-x64.zip"
                    ),
                    label="Downloading llama.cpp CUDA support DLLs...",
                ),
            )
        )
    return tuple(targets)


__all__ = [
    "RuntimeAssetSpec",
    "RuntimeAssetTarget",
    "paddle_targets",
    "required_runtime_asset_ids",
    "runtime_asset_catalog",
    "runtime_asset_spec",
]
