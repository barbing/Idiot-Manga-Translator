# -*- coding: utf-8 -*-
"""Pipeline controller placeholder."""
from __future__ import annotations
import os
import time
from datetime import datetime, timezone
import sys
from dataclasses import dataclass
from typing import List
from PySide6 import QtCore
from app.io.project import default_project_dict, save_project
from app.io.style_guide import default_style_guide, load_style_guide
from app.pipeline.steps import build_output_path, build_page_record
from app.models.ollama import list_models
from app.translate.prompts import build_translation_prompt, build_batch_translation_prompt
import re


class PipelineStatus(QtCore.QObject):
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)


@dataclass
class PipelineSettings:
    import_dir: str
    export_dir: str
    json_path: str
    output_suffix: str
    source_lang: str
    target_lang: str
    ollama_model: str
    style_guide_path: str
    font_name: str
    use_gpu: bool
    filter_background: bool
    filter_strength: str
    detector_engine: str
    ocr_engine: str
    inpaint_mode: str
    font_detection: str
    translator_backend: str
    gguf_model_path: str
    gguf_prompt_style: str
    gguf_n_ctx: int
    gguf_n_gpu_layers: int
    gguf_n_threads: int
    gguf_n_batch: int
    fast_mode: bool


class PipelineWorker(QtCore.QThread):
    progress_changed = QtCore.Signal(int)
    eta_changed = QtCore.Signal(str)
    page_changed = QtCore.Signal(int, int)
    message = QtCore.Signal(str)
    queue_reset = QtCore.Signal(list)
    queue_item = QtCore.Signal(int, str)
    total_time_changed = QtCore.Signal(str)
    page_time_changed = QtCore.Signal(str)

    def __init__(self, settings: PipelineSettings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        images = _list_images(self._settings.import_dir)
        total = len(images)
        self.queue_reset.emit(images)
        if total == 0:
            self.message.emit("No images found in import folder.")
            return
        if self._settings.fast_mode:
            self._settings.detector_engine = "PaddleOCR"
            self._settings.inpaint_mode = "fast"
            self._settings.font_detection = "off"
            self._settings.filter_strength = "normal"
            self.message.emit("Fast Mode: detector=PaddleOCR, inpaint=fast, font detection=off.")
        if not os.path.isdir(self._settings.export_dir):
            try:
                os.makedirs(self._settings.export_dir, exist_ok=True)
            except OSError:
                self.message.emit("Failed to create export folder.")
                return

        start_time = time.time()
        from app.detect.paddle_detector import PaddleTextDetector
        from app.ocr.manga_ocr_engine import MangaOcrEngine
        from app.translate.ollama_client import OllamaClient
        from app.render.renderer import render_translations

        ocr_engine = None
        font_detector = None
        try:
            if self._settings.ocr_engine == "MangaOCR":
                try:
                    ocr_engine = MangaOcrEngine(self._settings.use_gpu)
                except Exception as exc:
                    try:
                        from app.ocr.manga_ocr_worker import MangaOcrWorker
                        self.message.emit("MangaOCR in-process failed; using worker process.")
                        ocr_engine = MangaOcrWorker(use_gpu=self._settings.use_gpu)
                    except Exception as inner_exc:
                        self.message.emit(_friendly_model_error(inner_exc))
                        self.message.emit(f"MangaOCR failed: {inner_exc}")
                        return
            else:
                try:
                    from app.ocr.paddle_ocr_recognizer import PaddleOcrRecognizer
                    ocr_engine = PaddleOcrRecognizer(self._settings.use_gpu)
                except Exception as inner_exc:
                    self.message.emit(_friendly_model_error(inner_exc))
                    return

            if self._settings.font_detection != "off":
                try:
                    from app.render.font_detection import FontDetection
                    font_detector = FontDetection(mode=self._settings.font_detection)
                except Exception as exc:
                    self.message.emit(_friendly_model_error(exc))
                    font_detector = None

            try:
                if self._settings.detector_engine == "ComicTextDetector":
                    from app.detect.comic_text_detector import ComicTextDetector
                    detector = ComicTextDetector(self._settings.use_gpu)
                else:
                    detector = PaddleTextDetector(self._settings.use_gpu)
            except Exception as exc:
                self.message.emit(_friendly_model_error(exc))
                return

            try:
                if self._settings.translator_backend == "GGUF":
                    from app.translate.gguf_client import GGUFClient
                    n_gpu_layers = self._settings.gguf_n_gpu_layers if self._settings.use_gpu else 0
                    ollama = GGUFClient(
                        model_path=self._settings.gguf_model_path,
                        prompt_style=self._settings.gguf_prompt_style,
                        n_ctx=self._settings.gguf_n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=self._settings.gguf_n_threads,
                        n_batch=self._settings.gguf_n_batch,
                    )
                    if self._settings.use_gpu and not getattr(ollama, "gpu_offload", True):
                        self.message.emit(
                            "GGUF is running in CPU mode. For speed, install a CUDA-enabled llama-cpp-python "
                            "build or switch to Ollama."
                        )
                else:
                    ollama = OllamaClient()
                    if not ollama.is_available():
                        self.message.emit("Ollama server is not running. Start it with: ollama serve")
                        return
            except Exception as exc:
                self.message.emit(_friendly_model_error(exc))
                return
            resolved_model = _resolve_model(self._settings.ollama_model)
            if self._settings.translator_backend == "Ollama":
                if resolved_model and self._settings.ollama_model != "auto-detect":
                    available = list_models()
                    if available and resolved_model not in available:
                        self.message.emit(f"Ollama model not found: {resolved_model}")
                        return
            elif not self._settings.gguf_model_path:
                self.message.emit("GGUF model path is required for GGUF backend.")
                return
            style_guide = _load_style_guide(self._settings.style_guide_path)
            context_window = []
            translation_cache: dict[str, str] = {}
            project = default_project_dict()
            project["project"]["name"] = os.path.basename(self._settings.import_dir.rstrip("\\/"))
            project["project"]["language"]["source"] = _lang_code(self._settings.source_lang)
            project["project"]["language"]["target"] = _lang_code(self._settings.target_lang)
            project["project"]["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            project["project"]["model"]["detector"] = self._settings.detector_engine
            project["project"]["model"]["ocr"] = self._settings.ocr_engine
            if self._settings.translator_backend == "GGUF":
                project["project"]["model"]["translator"] = f"gguf:{self._settings.gguf_model_path}"
            else:
                project["project"]["model"]["translator"] = f"ollama:{self._settings.ollama_model}"
            project["project"]["style_guide"] = self._settings.style_guide_path or ""
            pages = []
            model_name = (
                self._settings.gguf_model_path
                if self._settings.translator_backend == "GGUF"
                else self._settings.ollama_model
            )
            for index, name in enumerate(images, start=1):
                if self._stop_requested:
                    self.message.emit("Stopped")
                    return

                page_start = time.time()
                self.queue_item.emit(index - 1, "processing")
                self.page_changed.emit(index, total)

                source_path = os.path.join(self._settings.import_dir, name)
                output_path = build_output_path(self._settings.export_dir, name, self._settings.output_suffix)

                try:
                    regions = _process_page(
                        source_path,
                        detector,
                        ocr_engine,
                        ollama,
                        model_name,
                        style_guide,
                        context_window,
                        self._settings.target_lang,
                        self._settings.source_lang,
                        self._settings.font_name,
                        self._settings.filter_background,
                        self._settings.filter_strength,
                        font_detector,
                        translation_cache,
                    )
                except Exception as exc:
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                    self.message.emit(f"Failed to process {name}: {exc}")
                    continue

                try:
                    render_translations(
                        source_path,
                        output_path,
                        regions,
                        self._settings.font_name,
                        inpaint_mode=self._settings.inpaint_mode,
                        use_gpu=self._settings.use_gpu,
                    )
                except Exception as exc:
                    page_elapsed = time.time() - page_start
                    self.queue_item.emit(index - 1, f"error ({_format_seconds(page_elapsed)}): {exc}")
                    self.message.emit(f"Failed to render {name}: {exc}")
                    continue

                page_id = os.path.splitext(name)[0]
                pages.append(build_page_record(source_path, page_id, regions))

                page_elapsed = time.time() - page_start
                self.page_time_changed.emit(f"Page: {_format_seconds(page_elapsed)}")
                self.queue_item.emit(index - 1, f"done ({_format_seconds(page_elapsed)})")
                progress = int(index / total * 100)
                self.progress_changed.emit(progress)

                elapsed = time.time() - start_time
                self.total_time_changed.emit(f"Total: {_format_seconds(elapsed)}")
                avg = elapsed / index
                remaining = avg * (total - index)
                self.eta_changed.emit(_format_eta(remaining))

            project["pages"] = pages
            json_path = self._settings.json_path or os.path.join(self._settings.export_dir, "project.json")
            try:
                save_project(json_path, project)
            except OSError:
                self.message.emit("Failed to write project JSON.")
            total_elapsed = time.time() - start_time
            self.total_time_changed.emit(f"Total: {_format_seconds(total_elapsed)}")
            self.message.emit("Completed")
        finally:
            try:
                if hasattr(ocr_engine, "close"):
                    ocr_engine.close()
            except Exception:
                pass


class PipelineController(QtCore.QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.status = PipelineStatus()
        self._running = False
        self._worker: PipelineWorker | None = None

    def start(self, settings: PipelineSettings) -> None:
        if self._running:
            return
        if not settings.import_dir:
            self.status.message.emit("Import folder is required.")
            return
        if not settings.export_dir:
            self.status.message.emit("Export folder is required.")
            return
        self._running = True
        self._worker = PipelineWorker(settings, self)
        self._worker.progress_changed.connect(self.status.progress_changed.emit)
        self._worker.eta_changed.connect(self.status.eta_changed.emit)
        self._worker.page_changed.connect(self.status.page_changed.emit)
        self._worker.total_time_changed.connect(self.status.total_time_changed.emit)
        self._worker.page_time_changed.connect(self.status.page_time_changed.emit)
        self._worker.message.connect(self.status.message.emit)
        self._worker.queue_reset.connect(self.status.queue_reset.emit)
        self._worker.queue_item.connect(self.status.queue_item.emit)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.status.message.emit("Started")

    def stop(self) -> None:
        if not self._running:
            return
        if self._worker:
            self._worker.request_stop()
        self.status.message.emit("Stopping...")

    def _on_finished(self) -> None:
        self._running = False
        self._worker = None


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = []
    for entry in os.listdir(folder):
        _, ext = os.path.splitext(entry)
        if ext.lower() in allowed:
            names.append(entry)
    names.sort()
    return names


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _lang_code(label: str) -> str:
    mapping = {
        "Japanese": "ja",
        "Simplified Chinese": "zh-Hans",
        "English": "en",
    }
    return mapping.get(label, label)


def _friendly_model_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "paddleocr" in lowered:
        return "PaddleOCR is not installed. Install it with: pip install paddleocr"
    if "export_model.py" in lowered or "jit.save" in lowered:
        return "PaddleOCR export failed. Try unchecking 'Enable GPU when available' and retry."
    if "failed to load torch" in lowered:
        return (
            "Torch failed to load (DLL dependency error). Restart the app after installing conda PyTorch. "
            "If it persists, reboot Windows to refresh DLL search paths."
        )
    if "manga-ocr" in lowered or "manga_ocr" in lowered:
        return f"MangaOCR failed to load: {text}"
    if "comictextdetector" in lowered or "comic-text-detector" in lowered or "utils.general" in lowered:
        return (
            "ComicTextDetector is not ready. Download comictextdetector.pt.onnx (CPU) or "
            "comictextdetector.pt (GPU) from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 "
            "and place it under models/comic-text-detector."
        )
    if "gguf" in lowered or "llama-cpp-python" in lowered or "llama_cpp" in lowered:
        return (
            "GGUF backend failed. Ensure llama-cpp-python is installed and the GGUF model path is valid."
        )
    if "yuzumarker" in lowered or "font detection" in lowered:
        return (
            "Font detection failed to initialize. Ensure the font model checkpoint is set and dependencies are installed."
        )
    if "numpy" in lowered and "abi" in lowered:
        return (
            "NumPy ABI mismatch. Reinstall numpy and the OCR deps. "
            "Suggested: pip install -U numpy==1.26.4 paddleocr manga-ocr"
        )
    if "shm.dll" in lowered or "winerror 127" in lowered:
        return (
            "PyTorch DLL load failed. Reinstall torch in the conda env. "
            "Suggested: pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
    return f"Failed to initialize models: {text}"


def _load_style_guide(path: str):
    if path and os.path.isfile(path):
        try:
            return load_style_guide(path)
        except Exception:
            return default_style_guide()
    return default_style_guide()


def _process_page(
    image_path: str,
    detector,
    ocr_engine,
    ollama,
    model: str,
    style_guide: dict,
    context_window: list,
    target_lang: str,
    source_lang: str,
    font_name: str,
    filter_background: bool,
    filter_strength: str,
    font_detector,
    translation_cache: dict[str, str],
) -> list:

    detections = detector.detect(image_path)
    image_size = _get_image_size(image_path)
    merge = getattr(detector, "merge_mode", "auto") != "none"
    groups = _merge_detections(detections, image_size, merge=merge)
    if not groups:
        groups = [{"bbox": _polygon_to_bbox(p), "polygons": [p], "conf": float(c or 0.0)} for p, c in detections]
    regions = []
    pending_texts: dict[str, list[str]] = {}
    for idx, group in enumerate(groups):
        bbox = group["bbox"]
        polygons = group["polygons"]
        det_conf = group["conf"]
        bg_text, needs_review = _classify_region(
            bbox,
            image_size,
            det_conf,
            filter_background,
            filter_strength,
        )
        if bg_text:
            regions.append(
                _region_record(
                    idx,
                    polygons,
                    bbox,
                    "",
                    "",
                    det_conf,
                    bg_text=True,
                    needs_review=needs_review,
                    ignore=filter_background,
                    font_name=font_name,
                )
            )
            continue
        crop = _crop_image(image_path, bbox)
        if crop is None:
            continue
        ocr_text = _clean_ocr_text(ocr_engine.recognize(crop))
        if not ocr_text:
            continue
        if _should_skip_text(ocr_text, bbox, image_size):
            regions.append(
                _region_record(
                    idx,
                    polygons,
                    bbox,
                    ocr_text,
                    "",
                    det_conf,
                    bg_text=True,
                    needs_review=True,
                    ignore=True,
                    font_name=font_name,
                )
            )
            continue
        detected_font = None
        if font_detector is not None:
            try:
                detected_font = font_detector.detect(crop)
            except Exception:
                detected_font = None
        region = _region_record(
            idx,
            polygons,
            bbox,
            ocr_text,
            "",
            det_conf,
            bg_text=False,
            needs_review=needs_review,
            ignore=False,
            font_name=font_name,
        )
        if detected_font:
            if target_lang == "Simplified Chinese" and not _is_font_allowed_for_cn(detected_font):
                detected_font = None
        if detected_font:
            render = region.get("render", {})
            render["font"] = detected_font
            region["render"] = render
        regions.append(region)
        cached = translation_cache.get(ocr_text)
        if cached is not None:
            region["translation"] = cached
        else:
            pending_texts.setdefault(ocr_text, []).append(region["region_id"])

    if pending_texts:
        items = []
        id_to_text: dict[str, str] = {}
        for idx, text in enumerate(pending_texts.keys()):
            item_id = f"t{idx:03d}"
            items.append({"id": item_id, "text": text})
            id_to_text[item_id] = text
        translations = _batch_translate(
            ollama,
            model,
            source_lang,
            target_lang,
            style_guide,
            items,
        )
        text_to_translation: dict[str, str] = {}
        if translations:
            for item_id, translation in translations.items():
                text = id_to_text.get(item_id)
                if text is not None:
                    text_to_translation[text] = translation
        if not text_to_translation:
            for text in pending_texts.keys():
                text_to_translation[text] = _translate_single(
                    ollama,
                    model,
                    source_lang,
                    target_lang,
                    style_guide,
                    text,
                )
        for text, region_ids in pending_texts.items():
            translation, lang_ok = _ensure_target_language(
                ollama,
                _resolve_model(model),
                source_lang,
                target_lang,
                text,
                text_to_translation.get(text, ""),
            )
            if translation:
                translation_cache[text] = translation
            for region in regions:
                if region["region_id"] in region_ids:
                    region["translation"] = translation
                    if not lang_ok:
                        region["flags"]["needs_review"] = True
    return regions


def _resolve_model(model: str) -> str:
    if model == "auto-detect":
        models = list_models()
        if models:
            preferred = [
                "aya:35b",
                "huihui_ai/qwen3-abliterated:32b",
                "huihui_ai/qwen3-abliterated:14b",
                "qwen3-coder:30b",
                "dolphin3:8b",
            ]
            for name in preferred:
                if name in models:
                    return name
            return models[0]
        return "aya:35b"
    return model


def _polygon_to_bbox(polygon: list) -> list:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _bbox_to_polygon(bbox: list) -> list:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _merge_detections(detections: list, image_size: tuple[int, int], merge: bool = True) -> list:
    if not detections:
        return []
    groups = []
    for polygon, conf in detections:
        try:
            bbox = _polygon_to_bbox(polygon)
        except Exception:
            continue
        groups.append({"bbox": bbox, "polygons": [polygon], "conf": float(conf or 0.0)})
    if not groups or not merge:
        return []
    changed = True
    while changed:
        changed = False
        result = []
        while groups:
            current = groups.pop(0)
            merged = False
            for i, other in enumerate(groups):
                if _should_merge(current["bbox"], other["bbox"], image_size):
                    current["bbox"] = _union_box(current["bbox"], other["bbox"])
                    current["polygons"].extend(other["polygons"])
                    current["conf"] = max(current["conf"], other["conf"])
                    groups.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                groups = result + groups
                result = []
                break
        if not changed:
            groups = result
    return groups


def _crop_image(image_path: str, bbox: list):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            x, y, w, h = [int(v) for v in bbox]
            return img.crop((x, y, x + w, y + h))
    except Exception:
        return None


def _merge_bboxes(bboxes: list, image_size: tuple[int, int]) -> list:
    if not bboxes:
        return []
    boxes = [_expand_box(b, 8, image_size) for b in bboxes]
    changed = True
    while changed:
        changed = False
        result = []
        while boxes:
            current = boxes.pop(0)
            merged = False
            for i, other in enumerate(boxes):
                if _should_merge(current, other, image_size):
                    current = _union_box(current, other)
                    boxes.pop(i)
                    merged = True
                    changed = True
                    break
            result.append(current)
            if merged:
                boxes = result + boxes
                result = []
                break
        if not changed:
            boxes = result
    return boxes


def _should_merge(a: list, b: list, image_size: tuple[int, int]) -> bool:
    if _boxes_overlap(a, b):
        return _overlap_ratio(a, b) >= 0.25
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    x_overlap = not (ax2 < bx or bx2 < ax)
    y_overlap = not (ay2 < by or by2 < ay)
    v_gap = min(abs(by - ay2), abs(ay - by2))
    h_gap = min(abs(bx - ax2), abs(ax - bx2))
    if x_overlap and v_gap <= max(6, min(ah, bh) * 0.25):
        return _union_area_ratio(a, b, image_size) <= 0.03
    if y_overlap and h_gap <= max(6, min(aw, bw) * 0.2):
        return _union_area_ratio(a, b, image_size) <= 0.03
    return False


def _boxes_overlap(a: list, b: list) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _union_box(a: list, b: list) -> list:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [x0, y0, x1 - x0, y1 - y0]


def _expand_box(box: list, padding: int, image_size: tuple[int, int]) -> list:
    img_w, img_h = image_size
    x, y, w, h = box
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img_w, x + w + padding) if img_w else x + w + padding
    y1 = min(img_h, y + h + padding) if img_h else y + h + padding
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _union_area_ratio(a: list, b: list, image_size: tuple[int, int]) -> float:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return 0.0
    area = img_w * img_h
    union = _union_box(a, b)
    return (union[2] * union[3]) / area


def _overlap_ratio(a: list, b: list) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min(aw * ah, bw * bh)
    return inter / max(1, min_area)


def _clean_translation(text: str) -> str:
    cleaned = text.strip()
    lowered = cleaned.lower()
    if lowered.startswith("translation:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if cleaned.startswith("翻译："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if cleaned.startswith("译文："):
        cleaned = cleaned.split("：", 1)[1].strip()
    if "translates to" in lowered:
        parts = cleaned.split("translates to", 1)
        cleaned = parts[1].strip() if len(parts) > 1 else cleaned
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    cleaned = re.sub(r"<\s*e=\d+\s*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\be=\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"e=\d+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("□", "").replace("�", "")
    cleaned = re.sub(r"(?:口|□){2,}", "", cleaned)
    if _placeholder_ratio(cleaned) >= 0.15:
        cleaned = cleaned.replace("口", "")
    if _placeholder_ratio(cleaned) >= 0.25:
        return ""
    lines = [line for line in cleaned.splitlines() if line.strip()]
    filtered = []
    strip_phrases = [
        "文本：",
        "文本:",
        "仅需翻译",
        "只需翻译",
        "只翻译",
        "不要任何标签",
        "不要任何引号",
        "不要任何解释",
        "不要任何说明",
        "不要任何注释",
        "不要任何多余",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要说明",
        "不要注释",
        "不要多余",
        "只输出译文",
        "仅输出译文",
        "输出译文",
        "只输出翻译",
        "译文如下",
        "翻译如下",
    ]
    for line in lines:
        head = line.strip()
        lower = head.lower()
        if (
            lower.startswith("text:")
            or lower.startswith("文本:")
            or lower.startswith("文本：")
            or lower.startswith("context:")
            or lower.startswith("input:")
            or "return only the translation" in lower
            or "output only the translation" in lower
            or "no labels" in lower
            or "no quotes" in lower
            or "no explanations" in lower
            or "<<text>>" in lower
            or "<</text>>" in lower
        ):
            continue
        head = head.replace("文本：", "").replace("文本:", "")
        if _is_punct_only(head):
            continue
        for phrase in strip_phrases:
            head = head.replace(phrase, "")
        if not head.strip():
            continue
        filtered.append(head)
    cleaned = "\n".join(filtered).strip()
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    if "Return only the translation" in cleaned:
        cleaned = cleaned.split("Return only the translation", 1)[0].strip()
    cleaned = cleaned.strip("<> ")
    return cleaned


def _batch_translate(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    items: list,
) -> dict:
    resolved = _resolve_model(model)
    translations: dict = {}
    batch_size = 8
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompt = build_batch_translation_prompt(source_lang, target_lang, style_guide, chunk)
        token_limit = _estimate_num_predict(chunk)
        try:
            raw = ollama.generate(
                resolved,
                prompt,
                timeout=600,
                options={"num_predict": token_limit, "temperature": 0.1, "top_p": 0.9},
            )
        except Exception:
            return {}
        parsed = _parse_json_list(raw)
        if not isinstance(parsed, list):
            return {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            region_id = str(item.get("id", "")).strip()
            translation = _clean_translation(str(item.get("translation", "")).strip())
            if region_id:
                translations[region_id] = translation
    return translations


def _parse_json_list(text: str):
    cleaned = _clean_translation(text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        import json
        return json.loads(snippet)
    except Exception:
        try:
            import ast
            return ast.literal_eval(snippet)
        except Exception:
            return None


def _estimate_num_predict(items: list) -> int:
    if not items:
        return 128
    lengths = [len(str(item.get("text", ""))) for item in items if isinstance(item, dict)]
    avg_len = sum(lengths) / max(1, len(lengths))
    estimate = int(max(64, min(160, avg_len * 5)))
    return estimate


def _translate_single(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    style_guide: dict,
    text: str,
) -> str:
    prompt = build_translation_prompt(source_lang, target_lang, style_guide, [], text)
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=300,
        options={"num_predict": 160, "temperature": 0.1, "top_p": 0.9},
    )
    return _clean_translation(result)


def _ensure_target_language(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    ocr_text: str,
    translation: str,
) -> tuple[str, bool]:
    if _too_long_translation(translation, ocr_text):
        translation = _translate_brief(ollama, model, source_lang, target_lang, ocr_text)
    if _looks_like_prompt_leak(translation):
        translation = _translate_strict(ollama, model, source_lang, target_lang, ocr_text)
    if _language_ok(target_lang, translation) and not _looks_like_prompt_leak(translation):
        return translation, True
    retry_prompt = (
        f"Translate {source_lang} to {target_lang}.\n"
        "No English, no romaji, no explanations.\n"
        f"Text: {ocr_text}\n"
    )
    retry = _clean_translation(
        ollama.generate(
            model,
            retry_prompt,
            timeout=180,
            options={"num_predict": 128, "temperature": 0.1, "top_p": 0.9},
        )
    )
    if _looks_like_prompt_leak(retry):
        retry = _translate_strict(ollama, model, source_lang, target_lang, ocr_text)
    if _language_ok(target_lang, retry) and not _looks_like_prompt_leak(retry):
        return retry, True
    if _looks_like_prompt_leak(retry or translation):
        return "", False
    return retry or translation, False


def _too_long_translation(translation: str, ocr_text: str) -> bool:
    if not translation or not ocr_text:
        return False
    t_len = len(translation)
    o_len = len(ocr_text)
    if o_len <= 4:
        return t_len > max(12, o_len * 3)
    return t_len > o_len * 2.2


def _translate_brief(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
) -> str:
    if target_lang == "Simplified Chinese":
        prompt = f"将以下{source_lang}翻译成简体中文，保持简短：{text}"
    elif target_lang == "English":
        prompt = f"Translate the following {source_lang} into English. Keep it short: {text}"
    else:
        prompt = f"Translate the following {source_lang} into {target_lang}. Keep it short: {text}"
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=180,
        options={"num_predict": 128, "temperature": 0.1, "top_p": 0.9},
    )
    return _clean_translation(result)


def _language_ok(target_lang: str, text: str) -> bool:
    if not text:
        return False
    if target_lang == "Simplified Chinese":
        return _cjk_ratio(text) >= 0.3 and _kana_ratio(text) <= 0.1
    if target_lang == "English":
        return _cjk_ratio(text) < 0.2
    return True


def _looks_like_prompt_leak(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    markers = [
        "return only",
        "output only",
        "output only the translation",
        "no labels",
        "no quotes",
        "no explanations",
        "text to translate",
        "<<text>>",
        "<</text>>",
        "translation:",
    ]
    chinese_markers = [
        "只需翻译",
        "仅需翻译",
        "只翻译",
        "不要任何",
        "不要标签",
        "不要引号",
        "不要解释",
        "不要多余",
        "不要说明",
        "不要注释",
        "上下文",
        "译文",
        "只输出",
        "输出译文",
        "只输出翻译",
        "翻译如下",
    ]
    if any(m in lowered for m in markers):
        return True
    for marker in chinese_markers:
        if marker in text:
            return True
    return False


def _translate_strict(
    ollama,
    model: str,
    source_lang: str,
    target_lang: str,
    text: str,
) -> str:
    if target_lang == "Simplified Chinese":
        prompt = f"将以下{source_lang}翻译成简体中文：{text}"
    elif target_lang == "English":
        prompt = f"Translate the following {source_lang} into English: {text}"
    else:
        prompt = f"Translate the following {source_lang} into {target_lang}: {text}"
    result = ollama.generate(
        _resolve_model(model),
        prompt,
        timeout=180,
        options={"num_predict": 160, "temperature": 0.1, "top_p": 0.9},
    )
    return _clean_translation(result)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if _is_japanese(ch))
    return cjk / max(1, len(text))


def _kana_ratio(text: str) -> float:
    if not text:
        return 0.0
    kana = sum(1 for ch in text if _is_kana(ch))
    return kana / max(1, len(text))


def _should_skip_text(text: str, bbox: list, image_size: tuple[int, int]) -> bool:
    if not text:
        return True
    if _is_punct_only(text):
        return True
    if _placeholder_ratio(text) >= 0.15:
        return True
    x, y, w, h = bbox
    area = w * h
    img_w, img_h = image_size
    page_area = img_w * img_h if img_w and img_h else 1
    ratio = area / page_area
    length = len(text)
    jp_ratio = _japanese_ratio(text)
    if length <= 2 and ratio < 0.003:
        aspect = w / h if h else 0
        if jp_ratio >= 0.6 and 0.3 < aspect < 3.5:
            return False
        return True
    if jp_ratio < 0.3 and length < 6:
        return True
    if jp_ratio < 0.2 and ratio < 0.006:
        return True
    return False


def _japanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    jp = sum(1 for ch in text if _is_japanese(ch))
    return jp / max(1, len(text))


def _placeholder_ratio(text: str) -> float:
    if not text:
        return 0.0
    placeholders = {"□", "口", "�"}
    count = sum(1 for ch in text if ch in placeholders)
    return count / max(1, len(text))


def _is_punct_only(text: str) -> bool:
    stripped = "".join(ch for ch in text if ch.strip())
    if not stripped:
        return True
    letters = sum(1 for ch in stripped if ch.isalnum() or _is_japanese(ch))
    return letters == 0


def _clean_ocr_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("□", "").replace("�", "")
    if _placeholder_ratio(cleaned) >= 0.2:
        cleaned = cleaned.replace("口", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_japanese(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x4E00 <= code <= 0x9FFF
    )


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x30FF


def _is_font_allowed_for_cn(font_name: str) -> bool:
    if not font_name:
        return False
    allowed = {
        "Noto Sans CJK",
        "Microsoft YaHei",
        "SimSun",
        "SimHei",
    }
    name = font_name.strip().lower()
    for item in allowed:
        if item.lower() in name:
            return True
    return False


def _region_record(
    idx: int,
    polygon: list,
    bbox: list,
    ocr_text: str,
    translation: str,
    det_conf: float,
    bg_text: bool,
    needs_review: bool,
    ignore: bool,
    font_name: str,
) -> dict:
    return {
        "region_id": f"r{idx:03d}",
        "bbox": bbox,
        "polygon": polygon,
        "type": "speech_bubble",
        "ocr_text": ocr_text,
        "translation": translation,
        "confidence": {"det": det_conf, "ocr": 1.0, "trans": 1.0},
        "render": {
            "font": font_name,
            "font_size": 0,
            "line_height": 1.2,
            "align": "center",
            "color": "#000000",
            "stroke": "#FFFFFF",
            "stroke_width": 2,
            "wrap_mode": "auto",
        },
        "flags": {"ignore": ignore, "bg_text": bg_text, "needs_review": needs_review},
    }

def _get_image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return (0, 0)
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def _classify_region(
    bbox: list,
    image_size: tuple[int, int],
    det_conf: float,
    filter_background: bool,
    filter_strength: str,
) -> tuple[bool, bool]:
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return False, det_conf < 0.6
    x, y, w, h = bbox
    area = w * h
    page_area = img_w * img_h
    if page_area <= 0:
        return False, det_conf < 0.6
    ratio = area / page_area
    aspect = w / h if h else 0
    margin_x = img_w * 0.02
    margin_y = img_h * 0.02
    near_edge = x < margin_x or y < margin_y or (x + w) > (img_w - margin_x) or (y + h) > (img_h - margin_y)

    aggressive = filter_strength == "aggressive"
    large_ratio = 0.12 if not aggressive else 0.09
    strip_ratio = 0.05 if not aggressive else 0.03
    edge_ratio = 0.03 if not aggressive else 0.02

    bg_text = False
    if ratio > large_ratio and (near_edge or aspect > 4):
        bg_text = True
    elif aspect > 5 and ratio > strip_ratio:
        bg_text = True
    elif near_edge and ratio > edge_ratio:
        bg_text = True

    if not filter_background:
        bg_text = False

    needs_review = det_conf < 0.6 or (bg_text and aggressive)
    return bg_text, needs_review
