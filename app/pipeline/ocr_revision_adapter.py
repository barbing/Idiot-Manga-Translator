# -*- coding: utf-8 -*-
"""Reuse-only adapter for the controller-owned selected OCR policy."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from app.config.run_settings_compiler import (
    materialize_pipeline_settings_snapshot,
)

from .ocr_revision_contracts import (
    CancellationProbe,
    OcrRecognitionReceipt,
    OcrRecognitionRequest,
    OcrRevisionError,
    OcrRevisionErrorCode,
)


class ControllerOcrRevisionAdapter:
    """Expose existing controller OCR behavior without reimplementing it.

    The injectable callables are deterministic test seams.  Production defaults
    are imported lazily and delegate unchanged to the two controller-owned
    helpers named by the GUI architecture contract.
    """

    def __init__(
        self,
        *,
        engine_factory: Callable[..., Any] | None = None,
        recognizer: Callable[..., tuple[str, float]] | None = None,
        settings_materializer: Callable[..., Any] | None = None,
        message_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._recognizer = recognizer
        self._settings_materializer = (
            settings_materializer or materialize_pipeline_settings_snapshot
        )
        self._message_callback = message_callback

    def recognize(
        self,
        request: OcrRecognitionRequest,
        *,
        cancellation_probe: CancellationProbe | None = None,
    ) -> OcrRecognitionReceipt:
        if not isinstance(request, OcrRecognitionRequest):
            raise TypeError("request must be an OcrRecognitionRequest")
        cancelled = cancellation_probe or (lambda: False)
        if cancelled():
            raise OcrRevisionError(
                OcrRevisionErrorCode.CANCELLED,
                "OCR revision was cancelled before engine initialization.",
            )

        settings = self._settings_materializer(
            request.request.run_settings_snapshot
        )
        if str(getattr(settings, "ocr_engine", "") or "") != (
            request.request.selected_ocr_engine
        ):
            raise OcrRevisionError(
                OcrRevisionErrorCode.SETTINGS_MISMATCH,
                "Materialized OCR engine differs from the immutable run snapshot.",
            )

        engine_factory = self._engine_factory
        recognizer = self._recognizer
        if engine_factory is None or recognizer is None:
            from app.pipeline.controller import (
                _create_selected_ocr_engine,
                _recognize_with_fallback,
            )

            engine_factory = engine_factory or _create_selected_ocr_engine
            recognizer = recognizer or _recognize_with_fallback

        engine: Any = None
        backend_metadata: Mapping[str, Any] = {}
        recognition_metadata: Mapping[str, Any] = {}
        text = ""
        confidence = 0.0
        try:
            engine = engine_factory(
                settings,
                message_callback=self._message_callback,
            )
            if cancelled():
                raise OcrRevisionError(
                    OcrRevisionErrorCode.CANCELLED,
                    "OCR revision was cancelled before inference.",
                )
            text, confidence = recognizer(
                engine,
                request.crop,
                settings,
                bbox=list(request.request.sampling_bbox),
                debug_context=None,
                trace_context={
                    "page_id": request.request.page_id,
                    "parent_id": request.request.parent_id,
                    "attempt_kind": "explicit_parent_revision",
                },
            )
            if hasattr(engine, "backend_metadata"):
                try:
                    backend_metadata = dict(engine.backend_metadata() or {})
                except Exception:
                    backend_metadata = {}
            if hasattr(engine, "last_recognition_metadata"):
                try:
                    recognition_metadata = dict(
                        engine.last_recognition_metadata() or {}
                    )
                except Exception:
                    recognition_metadata = {}
        except OcrRevisionError:
            raise
        except Exception as exc:
            raise OcrRevisionError(
                OcrRevisionErrorCode.RECOGNITION_FAILED,
                f"The selected OCR engine failed: {exc}",
            ) from exc
        finally:
            if engine is not None:
                close = getattr(engine, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        # Match the controller lifecycle: teardown must not mask
                        # the authoritative inference result or its exception.
                        pass

        authoritative = recognition_metadata.get(
            "ocr_response_authoritative"
        ) is not False
        backend_name = str(
            backend_metadata.get("ocr_backend")
            or getattr(engine, "backend_name", "")
            or (engine.__class__.__name__ if engine is not None else "")
        )
        return OcrRecognitionReceipt(
            selected_ocr_engine=request.request.selected_ocr_engine,
            text=str(text),
            confidence=float(confidence),
            authoritative=authoritative,
            backend_name=backend_name,
            backend_metadata=backend_metadata,
            recognition_metadata=recognition_metadata,
            crop_sha256=request.crop_sha256,
        )


__all__ = ["ControllerOcrRevisionAdapter"]
