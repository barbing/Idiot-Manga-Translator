# -*- coding: utf-8 -*-
"""Model-free local runtime-asset verification for the GUI-7 Settings page."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from PySide6 import QtCore

from app.config.settings_contracts import DownloadState, ProviderHealth, RuntimeStatus
from app.models.downloader import ModelDownloader
from app.models.resolution import models_root
from app.platform_services.contracts import PlatformIdentity
from app.platform_services.compute import invalidate_compute_capability_cache
from app.platform_services.runtime_assets import (
    RuntimeAssetSpec,
    runtime_asset_catalog,
    runtime_asset_spec,
)
from app.ui.runtime_resource_admission import (
    RuntimeResourceAdmissionReport,
    RuntimeResourceAdmissionService,
    RuntimeResourceMonitorService,
)


def runtime_assets_ready(
    status: RuntimeStatus | None,
    identity: PlatformIdentity | None = None,
    *,
    required_asset_ids: Iterable[str] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    selected = identity or PlatformIdentity.detect()
    installed = status.installed_assets if status is not None else {}
    catalog = runtime_asset_catalog(selected)
    catalog_ids = frozenset(spec.asset_id for spec in catalog)
    required = (
        catalog_ids
        if required_asset_ids is None
        else frozenset(str(value).strip() for value in required_asset_ids if str(value).strip())
    )
    unknown = required - catalog_ids
    if unknown:
        raise ValueError(f"unsupported required runtime assets: {sorted(unknown)}")
    missing: list[str] = []
    for spec in catalog:
        if spec.asset_id not in required:
            continue
        raw = installed.get(spec.asset_id)
        ready = (
            bool(raw.get("ready") or raw.get("installed"))
            if isinstance(raw, Mapping)
            else raw is True
            or str(raw).strip().casefold()
            in {"ready", "installed", "available", "valid"}
        )
        if not ready:
            missing.append(spec.asset_id)
    return not missing, tuple(missing)


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
        identity: PlatformIdentity | None = None,
    ) -> None:
        super().__init__()
        self._models_directory = str(
            Path(models_directory or models_root()).resolve()
        )
        self._downloader_factory = downloader_factory
        self._identity = identity or PlatformIdentity.detect()

    def _check_asset(
        self,
        downloader: ModelDownloader,
        spec: RuntimeAssetSpec,
    ) -> bool:
        checker = getattr(downloader, spec.checker, None)
        if not callable(checker):
            raise AttributeError(f"runtime checker is unavailable: {spec.checker}")
        if spec.asset_id == "pyicu":
            return bool(checker())
        return bool(checker(self._models_directory))

    @QtCore.Slot()
    def run(self) -> None:
        try:
            downloader = self._downloader_factory()
            assets: dict[str, dict[str, object]] = {}
            for spec in runtime_asset_catalog(self._identity):
                ready = self._check_asset(downloader, spec)
                error = (
                    str(getattr(downloader, "pyicu_runtime_error", "") or "")
                    if spec.asset_id == "pyicu"
                    else str(
                        getattr(downloader, "paddle_runtime_error", "") or ""
                    )
                    if spec.asset_id == "paddle_ocr_vl"
                    else ""
                )
                assets[spec.asset_id] = {
                    "ready": ready,
                    "detail": (
                        f"{spec.detail} verified."
                        if ready
                        else error or spec.remediation_for(self._identity)
                    ),
                    "managed_download": spec.preparer is not None,
                }
            if bool(assets.get("paddle_ocr_vl", {}).get("ready")):
                invalidate_compute_capability_cache()

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

    def __init__(
        self,
        asset_id: str,
        *,
        models_directory: str | None = None,
        downloader_factory: Callable[[], ModelDownloader] = ModelDownloader,
        identity: PlatformIdentity | None = None,
    ) -> None:
        super().__init__()
        normalized = str(asset_id or "").strip()
        self._identity = identity or PlatformIdentity.detect()
        try:
            spec = runtime_asset_spec(normalized, self._identity)
        except KeyError as exc:
            raise ValueError("unsupported runtime asset") from exc
        if spec.preparer is None:
            raise ValueError("unsupported runtime asset")
        self._asset_id = normalized
        self._spec = spec
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
        preparer = getattr(downloader, str(self._spec.preparer), None)
        if not callable(preparer):
            raise AttributeError(
                f"runtime preparer is unavailable: {self._spec.preparer}"
            )
        if self._asset_id == "pyicu":
            preparer()
        else:
            preparer(self._models_directory)

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
