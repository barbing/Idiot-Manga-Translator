# -*- coding: utf-8 -*-
"""Model-free local runtime-asset verification for the GUI-7 Settings page."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6 import QtCore

from app.config.settings_contracts import DownloadState, ProviderHealth, RuntimeStatus
from app.models.downloader import ModelDownloader
from app.models.resolution import has_noto_cjk_sc_font_pack, models_root
from app.ui.runtime_resource_admission import (
    RuntimeResourceAdmissionReport,
    RuntimeResourceAdmissionService,
    RuntimeResourceMonitorService,
)


class RuntimeAssetProbeWorker(QtCore.QObject):
    """Check existing local files and the pinned PyICU runtime off the GUI thread."""

    status_ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        models_directory: str | None = None,
        downloader_factory: Callable[[], ModelDownloader] = ModelDownloader,
    ) -> None:
        super().__init__()
        self._models_directory = str(
            Path(models_directory or models_root()).resolve()
        )
        self._downloader_factory = downloader_factory

    @QtCore.Slot()
    def run(self) -> None:
        try:
            downloader = self._downloader_factory()
            assets: dict[str, dict[str, object]] = {}

            detector_ready = bool(
                downloader.check_comic_text_detector(self._models_directory)
            )
            assets["comic_text_detector"] = {
                "ready": detector_ready,
                "detail": (
                    "CPU and CUDA model files present"
                    if detector_ready
                    else "Required detector model files are missing"
                ),
            }

            ocr_ready = bool(
                downloader.check_paddle_ocr_vl(self._models_directory)
            )
            assets["ocr"] = {
                "ready": ocr_ready,
                "detail": (
                    "PaddleOCR-VL model and local runtime present"
                    if ocr_ready
                    else "PaddleOCR-VL model or local runtime is missing"
                ),
            }

            pyicu_ready = bool(downloader.check_pyicu_runtime())
            assets["pyicu"] = {
                "ready": pyicu_ready,
                "detail": (
                    "Pinned PyICU and ICU runtime verified"
                    if pyicu_ready
                    else downloader.pyicu_runtime_error
                    or "Pinned PyICU runtime is unavailable"
                ),
            }

            font_ready = bool(has_noto_cjk_sc_font_pack(self._models_directory))
            assets["font_pack"] = {
                "ready": font_ready,
                "detail": (
                    "Noto CJK core font pack present"
                    if font_ready
                    else "Optional Noto CJK core font pack is not installed"
                ),
            }

            self.status_ready.emit(
                RuntimeStatus(
                    provider_health=ProviderHealth.UNKNOWN,
                    installed_assets=assets,
                    download_state=(
                        DownloadState.READY
                        if all(bool(value["ready"]) for value in assets.values())
                        else DownloadState.IDLE
                    ),
                    detail=(
                        "All local runtime assets are ready."
                        if all(bool(value["ready"]) for value in assets.values())
                        else "One or more local runtime assets need attention."
                    ),
                )
            )
        except Exception as exc:
            self.failed.emit(
                f"Runtime asset verification failed ({type(exc).__name__})."
            )
        finally:
            self.finished.emit()


class RuntimeAssetDownloadWorker(QtCore.QObject):
    """Run one existing managed asset download without loading a model."""

    progress_changed = QtCore.Signal(int)
    status_changed = QtCore.Signal(str)
    completed = QtCore.Signal(bool, str)
    finished = QtCore.Signal()

    _SUPPORTED_ASSETS = frozenset(
        {"comic_text_detector", "ocr", "pyicu", "font_pack"}
    )

    def __init__(
        self,
        asset_id: str,
        *,
        models_directory: str | None = None,
        downloader_factory: Callable[[], ModelDownloader] = ModelDownloader,
    ) -> None:
        super().__init__()
        normalized = str(asset_id or "").strip()
        if normalized not in self._SUPPORTED_ASSETS:
            raise ValueError("unsupported runtime asset")
        self._asset_id = normalized
        self._models_directory = str(
            Path(models_directory or models_root()).resolve()
        )
        self._downloader_factory = downloader_factory
        self._active_downloader: ModelDownloader | None = None
        self._cancel_requested = False

    def request_cancel(self) -> bool:
        self._cancel_requested = True
        downloader = self._active_downloader
        if downloader is not None:
            downloader.request_cancel()
        return True

    def _prepare(self, downloader: ModelDownloader) -> None:
        if self._asset_id == "comic_text_detector":
            downloader.prepare_comic_text_detector(self._models_directory)
        elif self._asset_id == "ocr":
            downloader.prepare_paddle_ocr_vl(self._models_directory)
        elif self._asset_id == "pyicu":
            downloader.prepare_pyicu_runtime()
        elif self._asset_id == "font_pack":
            downloader.prepare_noto_cjk_sc_font_pack(self._models_directory)
        else:  # pragma: no cover - constructor validates the closed set
            raise ValueError("unsupported runtime asset")

    @QtCore.Slot()
    def run(self) -> None:
        outcomes: list[tuple[bool, str]] = []
        try:
            downloader = self._downloader_factory()
            self._active_downloader = downloader
            downloader.progress_changed.connect(
                self.progress_changed,
                QtCore.Qt.ConnectionType.DirectConnection,
            )
            downloader.status_changed.connect(
                self.status_changed,
                QtCore.Qt.ConnectionType.DirectConnection,
            )
            downloader.finished.connect(
                lambda success, message: outcomes.append(
                    (bool(success), str(message))
                ),
                QtCore.Qt.ConnectionType.DirectConnection,
            )
            self._prepare(downloader)
            if self._cancel_requested:
                downloader.request_cancel()
            downloader.process_queue()
            failure = next((item for item in outcomes if not item[0]), None)
            outcome = failure or (outcomes[-1] if outcomes else None)
            if outcome is None:
                outcome = (False, "Runtime asset download returned no result.")
            self.completed.emit(*outcome)
        except Exception as exc:
            self.completed.emit(
                False,
                f"Runtime asset download failed ({type(exc).__name__}): {exc}",
            )
        finally:
            self._active_downloader = None
            self.finished.emit()


class RuntimeResourceAdmissionWorker(QtCore.QObject):
    """Measure and estimate one immutable run candidate off the GUI thread."""

    report_ready = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        settings_fingerprint: str,
        pipeline_values: dict[str, object],
        service: RuntimeResourceAdmissionService,
    ) -> None:
        super().__init__()
        normalized = str(settings_fingerprint or "").strip()
        if not normalized:
            raise ValueError("settings_fingerprint is required")
        if not isinstance(service, RuntimeResourceAdmissionService):
            raise TypeError("service must be RuntimeResourceAdmissionService")
        self._settings_fingerprint = normalized
        self._pipeline_values = dict(pipeline_values)
        self._service = service

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.report_ready.emit(
                self._service.evaluate(
                    settings_fingerprint=self._settings_fingerprint,
                    pipeline_values=self._pipeline_values,
                )
            )
        finally:
            self.finished.emit()


class RuntimeResourceMonitorWorker(QtCore.QObject):
    """Sample one active run's reserve without mutating execution."""

    report_ready = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        admission: RuntimeResourceAdmissionReport,
        service: RuntimeResourceMonitorService,
    ) -> None:
        super().__init__()
        if not isinstance(admission, RuntimeResourceAdmissionReport):
            raise TypeError("admission must be RuntimeResourceAdmissionReport")
        if not isinstance(service, RuntimeResourceMonitorService):
            raise TypeError("service must be RuntimeResourceMonitorService")
        self._admission = admission
        self._service = service

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.report_ready.emit(self._service.sample(self._admission))
        finally:
            self.finished.emit()


__all__ = [
    "RuntimeAssetDownloadWorker",
    "RuntimeAssetProbeWorker",
    "RuntimeResourceAdmissionWorker",
    "RuntimeResourceMonitorWorker",
]
