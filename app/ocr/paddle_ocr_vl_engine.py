# -*- coding: utf-8 -*-
"""PaddleOCR-VL GGUF recognizer using native llama.cpp server."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from app.models.resolution import (
    resolve_llama_server_executable,
    resolve_paddle_ocr_vl_mmproj_file,
    resolve_paddle_ocr_vl_model_file,
)


DEFAULT_PROMPT = (
    "Recognize all Japanese text in this manga image crop. Return only the exact "
    "Japanese text. Preserve punctuation. Do not translate or explain."
)


class PaddleOcrVlEngine:
    """OCR engine backed by PaddleOCR-VL-1.6 GGUF.

    The engine owns OCR runtime setup only. It does not create, merge, split, or
    suppress text topology; callers provide the crop selected by the current
    parent/source contract.
    """

    backend_name = "PaddleOCR-VL"
    prompt_version = "paddleocr_vl_exact_japanese_v1"

    def __init__(self, use_gpu: bool = True) -> None:
        self.model_path = _required_path(
            resolve_paddle_ocr_vl_model_file(),
            "PaddleOCR-VL GGUF model",
        )
        self.mmproj_path = _required_path(
            resolve_paddle_ocr_vl_mmproj_file(),
            "PaddleOCR-VL multimodal projector",
        )
        self._session: requests.Session | None = requests.Session()
        self.endpoint = _normalize_endpoint(os.environ.get("MT_PADDLEOCR_VL_ENDPOINT"))
        self._process: subprocess.Popen | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self._prompt = os.environ.get("MT_PADDLEOCR_VL_PROMPT", DEFAULT_PROMPT)
        self._timeout = float(os.environ.get("MT_PADDLEOCR_VL_TIMEOUT_SEC", "240") or 240)
        self._last_recognition_metadata: dict[str, Any] = {}
        if self.endpoint:
            self._assert_endpoint_ready(trust_custom_endpoint=True)
            return

        host = os.environ.get("MT_PADDLEOCR_VL_HOST", "127.0.0.1")
        port = int(os.environ.get("MT_PADDLEOCR_VL_PORT", "18080") or 18080)
        self.endpoint = f"http://{host}:{port}/v1"
        if self._endpoint_ready():
            self._assert_endpoint_ready(trust_custom_endpoint=False)
            return
        self._start_server(host, port, use_gpu=use_gpu)

    def recognize(self, image) -> str:
        text, _confidence = self.recognize_with_confidence(image)
        return text

    def recognize_with_confidence(self, image) -> tuple[str, float]:
        pil_image = _to_rgb_image(image)
        if pil_image is None:
            self._last_recognition_metadata = {
                "ocr_response_authoritative": False,
                "ocr_response_complete": False,
                "ocr_response_rejection_reason": "invalid_ocr_image",
            }
            return "", 0.0
        max_tokens = int(os.environ.get("MT_PADDLEOCR_VL_MAX_TOKENS", "192") or 192)
        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(pil_image)},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        response = self._request_session().post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        response_payload = response.json()
        text = _extract_chat_text(response_payload)
        text = _clean_model_response(text)
        self._last_recognition_metadata = _recognition_metadata(
            response_payload,
            max_tokens=max_tokens,
            has_text=bool(text),
        )
        confidence = 0.95 if self._last_recognition_metadata["ocr_response_authoritative"] else 0.0
        return text, confidence

    def last_recognition_metadata(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_recognition_metadata", {}) or {})

    def backend_metadata(self) -> dict[str, Any]:
        return {
            "ocr_backend": self.backend_name,
            "ocr_model_path": self.model_path,
            "ocr_mmproj_path": self.mmproj_path,
            "ocr_endpoint": self.endpoint,
            "ocr_prompt_version": self.prompt_version,
        }

    def close(self) -> None:
        proc = self._process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._process = None
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()
            self._session = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _start_server(self, host: str, port: int, *, use_gpu: bool) -> None:
        llama_server = _required_path(resolve_llama_server_executable(), "llama-server executable")
        log_dir = Path(self.model_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "paddleocr_vl_llama_server_stdout.log"
        stderr_path = log_dir / "paddleocr_vl_llama_server_stderr.log"
        self._stdout_handle = open(stdout_path, "a", encoding="utf-8")
        self._stderr_handle = open(stderr_path, "a", encoding="utf-8")
        gpu_layers = os.environ.get("MT_PADDLEOCR_VL_N_GPU_LAYERS")
        if gpu_layers is None:
            gpu_layers = "99" if use_gpu else "0"
        args = [
            llama_server,
            "-m",
            self.model_path,
            "--mmproj",
            self.mmproj_path,
            "--host",
            host,
            "--port",
            str(port),
            "--temp",
            "0",
            "--ctx-size",
            os.environ.get("MT_PADDLEOCR_VL_CTX_SIZE", "4096"),
            "--n-gpu-layers",
            gpu_layers,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            args,
            cwd=str(Path(llama_server).parent),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            creationflags=creationflags,
        )
        deadline = time.time() + float(os.environ.get("MT_PADDLEOCR_VL_STARTUP_TIMEOUT_SEC", "120") or 120)
        last_error = ""
        while time.time() < deadline:
            if self._process.poll() is not None:
                last_error = _tail_file(stderr_path)
                break
            if self._endpoint_ready():
                self._assert_endpoint_ready(trust_custom_endpoint=False)
                return
            time.sleep(0.5)
        self.close()
        detail = f" Last stderr: {last_error}" if last_error else ""
        raise RuntimeError(f"PaddleOCR-VL llama-server failed to start on {self.endpoint}.{detail}")

    def _endpoint_ready(self) -> bool:
        try:
            base = self.endpoint.rsplit("/v1", 1)[0]
            response = self._request_session().get(f"{base}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _assert_endpoint_ready(self, *, trust_custom_endpoint: bool) -> None:
        if not self._endpoint_ready():
            raise RuntimeError(f"PaddleOCR-VL endpoint is not healthy: {self.endpoint}")
        if trust_custom_endpoint:
            return
        try:
            response = self._request_session().get(f"{self.endpoint}/models", timeout=5)
            response.raise_for_status()
            body = json.dumps(response.json(), ensure_ascii=False).lower()
            if "paddleocr" not in body and "paddleocr-vl" not in body:
                raise RuntimeError(f"endpoint is running a different model: {self.endpoint}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"failed to verify PaddleOCR-VL endpoint model: {exc}") from exc

    def _request_session(self) -> requests.Session:
        session = getattr(self, "_session", None)
        if session is None:
            raise RuntimeError("PaddleOCR-VL HTTP session is closed")
        return session


def _required_path(path: str | None, label: str) -> str:
    if path and os.path.isfile(path):
        return path
    raise RuntimeError(f"{label} is missing. Run the PaddleOCR-VL model download/setup first.")


def _normalize_endpoint(value: str | None) -> str:
    endpoint = str(value or "").strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/v1"):
        return endpoint
    return endpoint + "/v1"


def _to_rgb_image(image):
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        Image = None
        np = None
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if np is not None and isinstance(image, np.ndarray) and Image is not None:
        array = image
        if array.ndim == 3 and array.shape[2] == 3:
            try:
                array = array[:, :, ::-1]
            except Exception:
                pass
        return Image.fromarray(array).convert("RGB")
    return None


def _image_data_url(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    return str(message.get("content") or "").strip() if isinstance(message, dict) else ""


def _recognition_metadata(
    payload: dict[str, Any],
    *,
    max_tokens: int,
    has_text: bool,
) -> dict[str, Any]:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    finish_reason = str(first.get("finish_reason") or "").strip().lower()
    usage = payload.get("usage") if isinstance(payload, dict) and isinstance(payload.get("usage"), dict) else {}
    response_complete = finish_reason != "length"
    authoritative = bool(has_text and response_complete)
    rejection_reason = ""
    if finish_reason == "length":
        rejection_reason = "ocr_response_token_limit_reached"
    elif not has_text:
        rejection_reason = "empty_ocr_response"
    return {
        "ocr_finish_reason": finish_reason,
        "ocr_prompt_tokens": _optional_int(usage.get("prompt_tokens")),
        "ocr_completion_tokens": _optional_int(usage.get("completion_tokens")),
        "ocr_total_tokens": _optional_int(usage.get("total_tokens")),
        "ocr_max_tokens": int(max_tokens),
        "ocr_response_complete": bool(response_complete),
        "ocr_response_authoritative": bool(authoritative),
        "ocr_response_rejection_reason": rejection_reason,
    }


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean_model_response(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"text", "japanese", "ja"}:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _tail_file(path: Path, limit: int = 2000) -> str:
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-limit:]
    except Exception:
        return ""
