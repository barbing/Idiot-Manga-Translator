# -*- coding: utf-8 -*-
"""Single production application coordinator for the GUI-7 cutover.

The coordinator connects presentation intent to existing typed application and
pipeline boundaries.  It never imports a legacy review window and never calls a
pipeline implementation helper directly.  Forward runs cross
``PipelineController.start``; page previews cross ``PageRerenderWorker``; edit
operations remain owned by the shell's typed edit workers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PySide6 import QtCore, QtWidgets
import requests

from app.config.credential_store import (
    resolve_legacy_deepseek_credential,
)
from app.config.provider_profiles import (
    ProviderCapability,
    ProviderKind,
    ProviderProfile,
    ProviderTestStatus,
)
from app.config.run_settings_compiler import (
    CompilationResult,
    RunInvocation,
    RuntimeProviderBinding,
    apply_runtime_admission_overrides,
    materialize_pipeline_settings,
)
from app.config.settings_contracts import CredentialReference
from app.models.resolution import models_root
from app.platform_services.credentials import (
    build_credential_resolver,
    build_credential_store,
    credential_store_label,
)
from app.platform_services import PlatformServices, build_platform_services
from app.pipeline.status_contracts import PipelineLifecycleEvent, PipelineRunState
from app.ui.page_rerender_worker import (
    PageRerenderWorker,
    discard_preview_lease,
)
from app.ui.design_system.dialogs import (
    HybridConfirmDialog,
    HybridTextInputDialog,
)
from app.ui.project_hub.new_project_dialog import (
    NewProjectDialog,
    named_project_filename,
)
from app.ui.viewmodels.page_rerender_model import (
    PageRerenderCommand,
    PageRerenderFailure,
    PageRerenderFailureCode,
    PageRerenderPreviewLease,
    PageRerenderViewModel,
    PageRerenderWorkerStage,
)


_TERMINAL_RUN_STATES = frozenset(
    {
        PipelineRunState.STOPPED,
        PipelineRunState.COMPLETED,
        PipelineRunState.FAILED,
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class NewTranslationRequest:
    """One explicit filesystem-backed request prepared by the Project Hub."""

    import_dir: str
    export_dir: str
    project_path: str
    project_name: str = ""

    def __post_init__(self) -> None:
        import_dir = os.path.abspath(str(self.import_dir or "").strip())
        export_dir = os.path.abspath(str(self.export_dir or "").strip())
        project_path = os.path.abspath(str(self.project_path or "").strip())
        project_name = str(self.project_name or "").strip()
        if not import_dir or not os.path.isdir(import_dir):
            raise ValueError("import_dir must identify an existing folder")
        if not export_dir or not os.path.isdir(export_dir):
            raise ValueError("export_dir must identify an existing folder")
        if not project_path or os.path.isdir(project_path):
            raise ValueError("project_path must identify a project JSON file")
        if os.path.exists(project_path):
            raise ValueError("project_path must not replace an existing project")
        if os.path.normcase(os.path.dirname(project_path)) != os.path.normcase(
            export_dir
        ):
            raise ValueError("project_path must be inside the selected output folder")
        if not project_name:
            filename = os.path.basename(project_path)
            suffix = ".yomiframe.json"
            project_name = (
                filename[: -len(suffix)]
                if filename.casefold().endswith(suffix)
                else os.path.basename(import_dir.rstrip("\\/"))
            )
        if not project_name or any(ord(character) < 32 for character in project_name):
            raise ValueError("project_name must contain visible text")
        object.__setattr__(self, "import_dir", import_dir)
        object.__setattr__(self, "export_dir", export_dir)
        object.__setattr__(self, "project_path", project_path)
        object.__setattr__(self, "project_name", project_name)

    @property
    def invocation(self) -> RunInvocation:
        return RunInvocation(
            import_dir=self.import_dir,
            export_dir=self.export_dir,
            json_path=self.project_path,
        )


@dataclass(frozen=True, slots=True)
class ProviderTestReceipt:
    profile_id: str
    status: ProviderTestStatus
    message: str
    tested_at_utc: str
    resolved_local_model_path: str | None = None
    tested_configuration_fingerprint: str = ""


class _ProviderTestWorker(QtCore.QObject):
    """Bounded connection test for one exact public provider profile."""

    receipt = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        profile: ProviderProfile,
        credential: str | None,
        *,
        models_directory: str | None = None,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._credential = credential
        self._models_directory = Path(
            models_directory or models_root()
        ).expanduser()

    def _resolve_gguf_model_path(self, candidate: str) -> Path | None:
        selected = Path(str(candidate or "").strip()).expanduser()
        if selected.is_file():
            return selected.resolve()
        if not selected.name or not self._models_directory.is_dir():
            return None
        if not selected.is_absolute():
            rooted = self._models_directory / selected
            if rooted.is_file():
                return rooted.resolve()
        matches = tuple(
            path.resolve()
            for path in self._models_directory.rglob("*.gguf")
            if path.is_file() and path.name.casefold() == selected.name.casefold()
        )
        return matches[0] if len(matches) == 1 else None

    @QtCore.Slot()
    def run(self) -> None:
        status = ProviderTestStatus.ERROR
        message = "Provider test failed."
        resolved_local_model_path: str | None = None
        try:
            profile = self._profile
            if profile.kind is ProviderKind.GGUF:
                model_path = self._resolve_gguf_model_path(
                    str(profile.local_model_path or "")
                )
                if model_path is None:
                    status = ProviderTestStatus.UNAVAILABLE
                    message = (
                        "The selected GGUF model file could not be resolved uniquely. "
                        "Choose the exact file with Browse."
                    )
                elif model_path.suffix.casefold() != ".gguf":
                    status = ProviderTestStatus.UNAVAILABLE
                    message = "Select a model file with the .gguf extension."
                else:
                    status = ProviderTestStatus.READY
                    resolved_local_model_path = str(model_path)
                    message = (
                        "GGUF model file is readable. The model will load only when "
                        "a translation run starts."
                    )
            elif ProviderCapability.CONNECTION_TEST not in profile.capabilities:
                status = ProviderTestStatus.UNAVAILABLE
                message = "This provider does not expose a connection test."
            elif profile.kind is ProviderKind.OLLAMA:
                endpoint = str(profile.endpoint or "").rstrip("/")
                request = Request(f"{endpoint}/api/tags", method="GET")
                with urlopen(request, timeout=4.0) as response:  # noqa: S310 - validated provider URL
                    payload = json.loads(response.read(1_000_000).decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("models"), list
                ):
                    raise ValueError("Ollama returned an invalid model inventory")
                models = {
                    str(item.get("name") or item.get("model") or "").strip()
                    for item in payload["models"]
                    if isinstance(item, dict)
                }
                configured_model = str(profile.model_id or "").strip()
                if not configured_model or configured_model == "auto-detect":
                    status = ProviderTestStatus.UNAVAILABLE
                    message = (
                        "Select one explicit Ollama model so its resources can be "
                        "verified before Start."
                    )
                elif configured_model not in models and not (
                    ":" not in configured_model
                    and f"{configured_model}:latest" in models
                ):
                    status = ProviderTestStatus.UNAVAILABLE
                    message = (
                        f"Ollama is reachable, but model {configured_model!r} is not installed. "
                        "Choose an installed model or pull it in Ollama."
                    )
                elif not models:
                    status = ProviderTestStatus.UNAVAILABLE
                    message = "Ollama is reachable, but it has no installed models."
                else:
                    status = ProviderTestStatus.READY
                    message = "Ollama is reachable and the configured model is available."
            elif profile.kind is ProviderKind.DEEPSEEK:
                if not self._credential:
                    status = ProviderTestStatus.UNAVAILABLE
                    message = "The linked DeepSeek credential is unavailable."
                else:
                    endpoint = str(profile.endpoint or "").rstrip("/")
                    response = requests.get(
                        f"{endpoint}/models",
                        headers={"Authorization": f"Bearer {self._credential}"},
                        timeout=5.0,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict) or not isinstance(
                        payload.get("data"), list
                    ):
                        raise ValueError("DeepSeek returned an invalid response")
                    models = {
                        str(item.get("id") or "").strip()
                        for item in payload["data"]
                        if isinstance(item, dict)
                    }
                    configured_model = str(profile.model_id or "").strip()
                    if configured_model not in models:
                        status = ProviderTestStatus.UNAVAILABLE
                        message = (
                            f"DeepSeek accepted the credential, but model "
                            f"{configured_model!r} was not returned by the endpoint."
                        )
                    else:
                        status = ProviderTestStatus.READY
                        message = (
                            "DeepSeek accepted the linked credential and the configured "
                            "model is available."
                        )
            else:
                status = ProviderTestStatus.UNAVAILABLE
                message = "This provider has no implemented connection-test adapter."
        except requests.HTTPError as exc:
            status = ProviderTestStatus.UNAVAILABLE
            code = getattr(getattr(exc, "response", None), "status_code", None)
            message = (
                f"Provider returned HTTP {code}. Check the endpoint, model ID, "
                "and linked credential."
                if code is not None
                else "The provider rejected the connection test. Check the endpoint, "
                "model ID, and linked credential."
            )
        except requests.RequestException as exc:
            status = ProviderTestStatus.UNAVAILABLE
            message = (
                f"Could not reach the provider: {exc}. Check the endpoint and "
                "network connection."
            )
        except HTTPError as exc:
            status = ProviderTestStatus.UNAVAILABLE
            message = (
                f"Provider returned HTTP {exc.code}. Check the endpoint, model ID, "
                "and linked credential."
            )
        except URLError as exc:
            status = ProviderTestStatus.UNAVAILABLE
            reason = str(getattr(exc, "reason", "connection failed") or "connection failed")
            message = f"Could not reach the provider: {reason}. Check that the service is running."
        except (OSError, UnicodeError, ValueError) as exc:
            status = ProviderTestStatus.UNAVAILABLE
            message = f"Provider validation failed: {exc}"
        finally:
            self.receipt.emit(
                ProviderTestReceipt(
                    profile_id=self._profile.profile_id,
                    status=status,
                    message=message,
                    tested_at_utc=_utc_now(),
                    resolved_local_model_path=resolved_local_model_path,
                    tested_configuration_fingerprint=(
                        self._profile.public_configuration_fingerprint
                    ),
                )
            )
            self.finished.emit()


class GuiApplicationCoordinator(QtCore.QObject):
    """Own the one production shell/controller/preview coordination path."""

    _preview_settlement_requested = QtCore.Signal()
    _provider_settlement_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        shell: object,
        controller: object,
        new_translation_chooser: Callable[[], NewTranslationRequest | None] | None = None,
        settings_materializer: Callable[[CompilationResult], object] = materialize_pipeline_settings,
        runtime_binding_resolver: Callable[[RuntimeProviderBinding], object | None] | None = None,
        preview_worker_factory: Callable[[PageRerenderCommand], PageRerenderWorker] = PageRerenderWorker,
        provider_worker_factory: Callable[[ProviderProfile, str | None], object] = _ProviderTestWorker,
        credential_prompt: Callable[[ProviderProfile], tuple[str, bool]] | None = None,
        credential_save_prompt: Callable[[ProviderProfile], bool] | None = None,
        credential_store_factory: Callable[[], object] = build_credential_store,
        credential_resolver_factory: Callable[[], object] = build_credential_resolver,
        legacy_credential_resolver: Callable[[ProviderProfile], str | None]
        | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        qt_parent = parent if isinstance(parent, QtCore.QObject) else None
        super().__init__(qt_parent)
        self._shell = shell
        self._controller = controller
        self._new_translation_chooser = (
            new_translation_chooser or self._choose_new_translation
        )
        self._settings_materializer = settings_materializer
        self._runtime_binding_resolver = (
            runtime_binding_resolver or self._resolve_runtime_binding
        )
        self._preview_worker_factory = preview_worker_factory
        self._provider_worker_factory = provider_worker_factory
        self._credential_prompt = credential_prompt or self._prompt_provider_credential
        self._credential_save_prompt = (
            credential_save_prompt or self._confirm_provider_credential_save
        )
        self._credential_store_factory = credential_store_factory
        self._credential_resolver_factory = credential_resolver_factory
        self._legacy_credential_resolver = (
            legacy_credential_resolver or self._resolve_legacy_provider_credential
        )
        self._prepared_request: NewTranslationRequest | None = None
        self._active_run_snapshot: object | None = None
        self._active_run_invocation: RunInvocation | None = None
        self._preview_model = PageRerenderViewModel()
        self._preview_thread: QtCore.QThread | None = None
        self._preview_worker: PageRerenderWorker | None = None
        self._preview_lease: PageRerenderPreviewLease | None = None
        self._preview_candidate_lease: PageRerenderPreviewLease | None = None
        self._preview_failure: PageRerenderFailure | None = None
        self._provider_thread: QtCore.QThread | None = None
        self._provider_worker: _ProviderTestWorker | None = None
        self._provider_receipt: ProviderTestReceipt | None = None
        self._provider_pending_credential: str | None = None
        self._provider_pending_profile_id = ""
        self._preview_settlement_requested.connect(
            self._settle_preview,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._provider_settlement_requested.connect(
            self._settle_provider_test,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        application = QtCore.QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self._discard_active_preview)
        self._bind()

    @property
    def active_run_snapshot(self) -> object | None:
        return self._active_run_snapshot

    def _bind(self) -> None:
        required_signals = (
            "new_project_requested",
            "start_run_requested",
            "stop_after_page_requested",
            "cancel_page_requested",
            "retry_requested",
            "page_preview_requested",
            "provider_test_requested",
            "credential_link_requested",
            "credential_delete_requested",
        )
        if any(not hasattr(self._shell, name) for name in required_signals):
            raise TypeError("shell does not expose the complete GUI-7 intent contract")
        status = getattr(self._controller, "status", None)
        if status is None or not hasattr(status, "lifecycle_changed"):
            raise TypeError("controller does not expose typed lifecycle status")
        self._shell.attach_controller(self._controller)
        self._shell.new_project_requested.connect(self._prepare_new_translation)
        self._shell.start_run_requested.connect(self._start_run)
        self._shell.stop_after_page_requested.connect(self._stop_after_page)
        self._shell.cancel_page_requested.connect(self._cancel_current_page)
        self._shell.retry_requested.connect(self._retry)
        self._shell.page_preview_requested.connect(self._start_preview)
        self._shell.provider_test_requested.connect(self._start_provider_test)
        self._shell.credential_link_requested.connect(self._link_credential)
        self._shell.credential_delete_requested.connect(self._delete_credential)
        status.lifecycle_changed.connect(self._accept_lifecycle)

    def _choose_new_translation(self) -> NewTranslationRequest | None:
        dialog = NewProjectDialog(
            parent=self._shell if isinstance(self._shell, QtWidgets.QWidget) else None,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        project_name = dialog.selected_project_name
        source = QtWidgets.QFileDialog.getExistingDirectory(
            self._shell,  # type: ignore[arg-type]
            "Select source manga pages",
        )
        if not source:
            return None
        output = QtWidgets.QFileDialog.getExistingDirectory(
            self._shell,  # type: ignore[arg-type]
            "Select translation output folder",
        )
        if not output:
            return None
        return NewTranslationRequest(
            import_dir=source,
            export_dir=output,
            project_path=os.path.join(
                output,
                named_project_filename(project_name),
            ),
            project_name=project_name,
        )

    @QtCore.Slot()
    def _prepare_new_translation(self) -> None:
        try:
            request = self._new_translation_chooser()
        except (OSError, TypeError, ValueError) as exc:
            self._notice(
                str(exc) or "The new translation request is invalid.",
                warning=True,
            )
            return
        if request is None:
            return
        if not isinstance(request, NewTranslationRequest):
            raise TypeError("new translation chooser must return NewTranslationRequest or None")
        try:
            self._shell.prepare_new_translation(
                import_dir=request.import_dir,
                export_dir=request.export_dir,
                project_path=request.project_path,
                project_name=request.project_name,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._notice(str(exc) or "The new translation could not be prepared.", warning=True)
            return
        self._prepared_request = request

    @QtCore.Slot()
    def _start_run(self, *, files_whitelist: tuple[str, ...] = ()) -> None:
        if not files_whitelist:
            consume_admitted = getattr(
                self._shell,
                "consume_admitted_run_files",
                None,
            )
            if callable(consume_admitted):
                admitted_files = consume_admitted()
                if admitted_files is not None:
                    files_whitelist = tuple(admitted_files)
        try:
            admission_report = None
            compilation = self._shell.compile_pipeline_run(
                files_whitelist=tuple(files_whitelist)
            )
            if not bool(getattr(compilation, "ready", False)):
                issues = tuple(getattr(compilation, "issues", ()))
                codes = ", ".join(str(getattr(item, "code", "unresolved")) for item in issues)
                raise RuntimeError(f"Run settings need attention: {codes or 'unresolved settings'}")
            admission_gate = getattr(
                self._shell,
                "resource_admission_report_for",
                None,
            )
            if callable(admission_gate):
                admission_report = admission_gate(compilation)
                if not bool(getattr(admission_report, "safe_to_start", False)):
                    raise RuntimeError(
                        "The current memory budget does not admit this run."
                    )
            runtime_overrides = tuple(
                getattr(admission_report, "runtime_overrides", ())
                if admission_report is not None
                else ()
            )
            if runtime_overrides:
                compilation = apply_runtime_admission_overrides(
                    compilation,
                    overrides=runtime_overrides,
                    effective_pipeline_values_fingerprint=str(
                        getattr(
                            admission_report,
                            "effective_pipeline_values_fingerprint",
                            "",
                        )
                    ),
                )
            settings = self._settings_materializer(compilation)
            runtime_binding = self._runtime_binding_resolver(
                compilation.runtime_binding
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._notice(str(exc) or "The run could not be prepared.", warning=True)
            return
        try:
            accepted = self._controller.start(
                settings,
                runtime_binding=runtime_binding,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._notice(str(exc) or "The pipeline controller could not start the run.", warning=True)
            return
        if not accepted:
            self._notice("The pipeline controller did not accept the run.", warning=True)
            return
        self._active_run_snapshot = compilation.snapshot
        self._active_run_invocation = self._invocation_from_snapshot(
            compilation.snapshot,
            fallback=(self._prepared_request.invocation if self._prepared_request else None),
        )
        self._shell.accept_run_started(
            compilation.snapshot,
            self._active_run_invocation,
        )

    @staticmethod
    def _invocation_from_snapshot(
        snapshot: object,
        *,
        fallback: RunInvocation | None,
    ) -> RunInvocation:
        values = getattr(snapshot, "pipeline_values", None)
        if values is not None:
            try:
                whitelist = values.get("files_whitelist") or ()
                return RunInvocation(
                    import_dir=str(values.get("import_dir") or ""),
                    export_dir=str(values.get("export_dir") or ""),
                    json_path=str(values.get("json_path") or ""),
                    files_whitelist=tuple(whitelist),
                )
            except (AttributeError, TypeError, ValueError):
                pass
        if fallback is None:
            return RunInvocation(import_dir="", export_dir="", json_path="")
        return fallback

    def _resolve_runtime_binding(
        self,
        binding: RuntimeProviderBinding,
    ) -> object | None:
        if not isinstance(binding, RuntimeProviderBinding):
            raise TypeError("compiled runtime binding is invalid")
        if binding.provider_kind is None:
            return None
        resolved: str | None = None
        if binding.credential_reference is not None:
            resolver = self._credential_resolver_factory()
            resolved = resolver.resolve(binding.credential_reference)
            if not resolved:
                raise RuntimeError(
                    "The selected provider credential is unavailable. Relink it in Settings."
                )
        if binding.provider_kind is ProviderKind.DEEPSEEK and not resolved:
            raise RuntimeError("DeepSeek requires a linked credential before Start.")
        from app.pipeline.controller import PipelineRuntimeBinding

        return PipelineRuntimeBinding(
            provider_kind=binding.provider_kind,
            resolved_credential=resolved,
        )

    @QtCore.Slot()
    def _stop_after_page(self) -> None:
        self._controller.stop()

    @QtCore.Slot()
    def _cancel_current_page(self) -> None:
        # The frozen controller exposes only safe-boundary stop.  Present that
        # exact semantic rather than claiming immediate cancellation.
        self._controller.stop()
        self._notice(
            "Immediate page cancellation is unavailable; stop was requested at the next safe page boundary.",
            warning=True,
        )

    @QtCore.Slot(str)
    def _retry(self, action: str) -> None:
        normalized = str(action or "").strip()
        if normalized == "retry_run":
            request_start = getattr(self._shell, "request_run_start", None)
            if callable(request_start):
                request_start(files_whitelist=())
            else:
                self._start_run()
            return
        if normalized == "retry_page":
            files = tuple(self._shell.retry_files_for_selected_page())
            if not files:
                self._notice("Select a persisted page before Retry Page.", warning=True)
                return
            request_start = getattr(self._shell, "request_run_start", None)
            if callable(request_start):
                request_start(files_whitelist=files)
            else:
                self._start_run(files_whitelist=files)
            return
        if normalized == "rebuild":
            page_id = str(getattr(self._shell, "selected_page_id", "") or "")
            if page_id:
                self._start_preview(page_id)
            else:
                self._notice("Select a page before rebuilding its preview.", warning=True)
            return
        if normalized in {"relink", "reset_settings"}:
            self._shell.navigate("settings")
            self._notice("Review and apply the highlighted Settings before retrying.")
            return
        self._notice("The selected error has no retry action.", warning=True)

    @QtCore.Slot(object)
    def _accept_lifecycle(self, event: object) -> None:
        if not isinstance(event, PipelineLifecycleEvent):
            return
        if event.state not in _TERMINAL_RUN_STATES:
            return
        snapshot = self._active_run_snapshot
        invocation = self._active_run_invocation
        self._active_run_snapshot = None
        self._active_run_invocation = None
        if snapshot is None or invocation is None:
            return
        self._shell.accept_run_finished(
            snapshot,
            invocation,
            completed=event.state is PipelineRunState.COMPLETED,
        )
        if invocation.json_path and os.path.isfile(invocation.json_path):
            self._shell.open_project(invocation.json_path)
        self._prepared_request = None

    @QtCore.Slot(str)
    def _start_preview(self, page_id: str) -> None:
        identity = str(page_id or "").strip()
        if not identity:
            return
        if self._preview_thread is not None:
            self._notice("Wait for the active page Preview to finish.", warning=True)
            return
        project_path = str(getattr(self._shell, "current_project_path", "") or "")
        try:
            command = PageRerenderCommand(
                project_path=project_path,
                page_id=identity,
            )
            self._preview_model.begin(command)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._notice(str(exc), warning=True)
            return
        thread = QtCore.QThread(self)
        worker = self._preview_worker_factory(command)
        worker.moveToThread(thread)
        self._preview_thread = thread
        self._preview_worker = worker
        self._preview_candidate_lease = None
        self._preview_failure = None
        self._shell.set_page_preview_activity(
            page_id=identity,
            active=True,
            message="Preparing the current effective page preview...",
        )
        thread.started.connect(worker.run)
        worker.preflight.connect(self._accept_preview_preflight)
        worker.progress.connect(self._accept_preview_progress)
        worker.preview_lease.connect(self._accept_preview_lease)
        worker.receipt.connect(self._accept_preview_receipt)
        worker.failure.connect(self._accept_preview_failure)
        worker.finished.connect(
            thread.quit,
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._preview_settlement_requested.emit,
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        thread.start()

    @QtCore.Slot(object)
    def _accept_preview_preflight(self, value: object) -> None:
        try:
            state = self._preview_model.accept_preflight(value)  # type: ignore[arg-type]
        except (RuntimeError, TypeError, ValueError):
            return
        self._shell.set_page_preview_activity(
            page_id=state.command.page_id if state.command else "",
            active=True,
            message=state.message,
        )

    @QtCore.Slot(object)
    def _accept_preview_progress(self, value: object) -> None:
        try:
            state = self._preview_model.accept_progress(value)  # type: ignore[arg-type]
        except (RuntimeError, TypeError, ValueError):
            return
        self._shell.set_page_preview_activity(
            page_id=state.command.page_id if state.command else "",
            active=True,
            message=state.message,
        )

    @QtCore.Slot(object)
    def _accept_preview_lease(self, value: object) -> None:
        if isinstance(value, PageRerenderPreviewLease):
            self._preview_candidate_lease = value

    @QtCore.Slot(object)
    def _accept_preview_receipt(self, value: object) -> None:
        try:
            self._preview_model.accept_receipt(value)  # type: ignore[arg-type]
        except (RuntimeError, TypeError, ValueError) as exc:
            command = self._preview_model.state.command
            if command is not None:
                self._preview_failure = PageRerenderFailure(
                    code=PageRerenderFailureCode.RERENDER_FAILED,
                    stage=PageRerenderWorkerStage.RERENDERING,
                    project_path=command.project_path,
                    page_id=command.page_id,
                    message=str(exc),
                )

    @QtCore.Slot(object)
    def _accept_preview_failure(self, value: object) -> None:
        if not isinstance(value, PageRerenderFailure):
            return
        self._preview_failure = value
        try:
            self._preview_model.accept_failure(value)
        except (RuntimeError, TypeError, ValueError):
            pass

    @QtCore.Slot()
    def _settle_preview(self) -> None:
        thread = self._preview_thread
        worker = self._preview_worker
        if thread is None or worker is None:
            return
        if not thread.wait():
            self._notice("Page Preview did not settle cleanly.", warning=True)
            return
        thread.deleteLater()
        self._preview_thread = None
        self._preview_worker = None
        command = self._preview_model.state.command
        page_id = command.page_id if command is not None else ""
        failure = self._preview_failure
        lease = self._preview_candidate_lease or self._preview_model.state.preview_lease
        self._preview_candidate_lease = None
        self._preview_failure = None
        if failure is not None:
            self._shell.set_page_preview_activity(
                page_id=page_id,
                active=False,
                message=failure.message,
                warning=True,
            )
            return
        if lease is None:
            self._shell.set_page_preview_activity(
                page_id=page_id,
                active=False,
                message="Page Preview ended without an output.",
                warning=True,
            )
            return
        previous = self._preview_lease
        self._preview_lease = lease
        self._shell.present_page_preview(lease)
        self._shell.set_page_preview_activity(
            page_id=page_id,
            active=False,
            message="Page Preview ready. No project artifact was published.",
        )
        if previous is not None and previous.output_path != lease.output_path:
            discard_preview_lease(previous)

    @QtCore.Slot()
    def _discard_active_preview(self) -> None:
        lease = self._preview_lease
        self._preview_lease = None
        if lease is not None:
            discard_preview_lease(lease)

    @QtCore.Slot(str)
    def _start_provider_test(self, profile_id: str) -> None:
        if self._provider_thread is not None:
            self._notice("Wait for the active provider test to finish.", warning=True)
            return
        profile = self._shell.provider_profile(profile_id)
        if not isinstance(profile, ProviderProfile):
            self._notice("The selected provider profile is unavailable.", warning=True)
            return
        credential: str | None = None
        try:
            if profile.credential_ref is not None:
                resolver = self._credential_resolver_factory()
                credential = resolver.resolve(profile.credential_ref)
        except Exception:
            credential = None
        legacy_credential = None
        if not credential and profile.kind is ProviderKind.DEEPSEEK:
            legacy_credential = self._legacy_credential_resolver(profile)
            credential = legacy_credential
        if (
            ProviderCapability.CREDENTIAL_REFERENCE in profile.capabilities
            and not credential
        ):
            self._link_credential(profile_id)
            return
        self._start_provider_test_worker(
            profile,
            credential,
            persist_credential_on_success=bool(legacy_credential),
        )

    @staticmethod
    def _resolve_legacy_provider_credential(
        profile: ProviderProfile,
    ) -> str | None:
        if profile.kind is not ProviderKind.DEEPSEEK:
            return None
        return resolve_legacy_deepseek_credential()

    def _start_provider_test_worker(
        self,
        profile: ProviderProfile,
        credential: str | None,
        *,
        persist_credential_on_success: bool,
    ) -> None:
        if self._provider_thread is not None:
            self._notice("Wait for the active provider test to finish.", warning=True)
            return
        thread = QtCore.QThread(self)
        worker = self._provider_worker_factory(profile, credential)
        if not isinstance(worker, QtCore.QObject):
            self._notice("The provider test adapter is unavailable.", warning=True)
            return
        worker.moveToThread(thread)
        self._provider_thread = thread
        self._provider_worker = worker
        self._provider_receipt = None
        self._provider_pending_credential = (
            credential if persist_credential_on_success else None
        )
        self._provider_pending_profile_id = (
            profile.profile_id if persist_credential_on_success else ""
        )
        self._shell.set_provider_test_activity(
            active=True,
            message=f"Testing {profile.display_name}...",
        )
        thread.started.connect(worker.run)
        worker.receipt.connect(self._accept_provider_receipt)
        worker.finished.connect(
            thread.quit,
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            self._provider_settlement_requested.emit,
            QtCore.Qt.ConnectionType.DirectConnection,
        )
        thread.start()

    @QtCore.Slot(object)
    def _accept_provider_receipt(self, value: object) -> None:
        if isinstance(value, ProviderTestReceipt):
            self._provider_receipt = value

    @QtCore.Slot()
    def _settle_provider_test(self) -> None:
        thread = self._provider_thread
        worker = self._provider_worker
        if thread is None or worker is None:
            return
        if not thread.wait():
            self._notice("Provider test did not settle cleanly.", warning=True)
            return
        thread.deleteLater()
        self._provider_thread = None
        self._provider_worker = None
        receipt = self._provider_receipt
        self._provider_receipt = None
        pending_credential = self._provider_pending_credential
        pending_profile_id = self._provider_pending_profile_id
        self._provider_pending_credential = None
        self._provider_pending_profile_id = ""
        self._shell.set_provider_test_activity(
            active=False,
            message=(
                receipt.message
                if receipt is not None
                else "Provider test ended without a result."
            ),
            warning=bool(
                receipt is None
                or receipt.status is not ProviderTestStatus.READY
            ),
        )
        if receipt is None:
            return
        if (
            receipt.status is ProviderTestStatus.READY
            and pending_credential
            and pending_profile_id == receipt.profile_id
        ):
            profile = self._shell.provider_profile(receipt.profile_id)
            if not isinstance(profile, ProviderProfile):
                self._notice(
                    "The tested provider profile is no longer available.",
                    warning=True,
                )
                return
            if not self._credential_save_prompt(profile):
                self._notice(
                    "Connection succeeded, but the credential was not saved. "
                    "Test again and save it securely before Start.",
                    warning=True,
                )
                return
            try:
                reference = self._credential_store_factory().store(
                    f"providers/{profile.profile_id}",
                    pending_credential,
                )
                self._shell.accept_provider_credential_reference(
                    profile.profile_id,
                    reference,
                )
            except Exception as exc:
                message = f"Credential could not be saved ({type(exc).__name__})."
                present_failure = getattr(
                    self._shell,
                    "accept_provider_save_failure",
                    None,
                )
                if callable(present_failure):
                    present_failure(profile.profile_id, message)
                else:
                    self._notice(message, warning=True)
                return
            self._shell.accept_provider_test_result(receipt)
            self._notice(
                "The credential was linked securely. Test the linked profile once "
                "more, then choose Use for translation and Apply Settings.",
            )
            return
        self._shell.accept_provider_test_result(receipt)

    @QtCore.Slot(str)
    def _link_credential(self, profile_id: str) -> None:
        profile = self._shell.provider_profile(profile_id)
        if not isinstance(profile, ProviderProfile):
            self._notice("The selected provider profile is unavailable.", warning=True)
            return
        if ProviderCapability.CREDENTIAL_REFERENCE not in profile.capabilities:
            self._notice("This provider does not use a stored credential.", warning=True)
            return
        if self._provider_thread is not None:
            self._notice("Wait for the active provider test to finish.", warning=True)
            return
        secret, accepted = self._credential_prompt(profile)
        if not accepted:
            return
        if not secret:
            self._notice("Credential was not changed.", warning=True)
            return
        self._start_provider_test_worker(
            profile,
            secret,
            persist_credential_on_success=True,
        )

    @QtCore.Slot(object)
    def _delete_credential(self, reference: object) -> None:
        if not isinstance(reference, CredentialReference):
            self._notice("The saved credential reference is invalid.", warning=True)
            return
        try:
            deleted = bool(self._credential_store_factory().delete(reference))
        except Exception as exc:
            self._notice(
                f"The saved credential could not be deleted ({type(exc).__name__}).",
                warning=True,
            )
            return
        self._notice(
            "Saved provider credential deleted."
            if deleted
            else "The saved provider credential was already absent."
        )

    def _prompt_provider_credential(
        self,
        profile: ProviderProfile,
    ) -> tuple[str, bool]:
        return HybridTextInputDialog.get_text(
            self._shell if isinstance(self._shell, QtWidgets.QWidget) else None,
            title="Test provider credential",
            prompt=f"API key for {profile.display_name}",
            confirm_text="Test connection",
        )

    def _confirm_provider_credential_save(self, profile: ProviderProfile) -> bool:
        return HybridConfirmDialog.ask(
            self._shell if isinstance(self._shell, QtWidgets.QWidget) else None,
            title="Save tested credential securely?",
            message=(
                f"{profile.display_name} accepted this credential. Save it in "
                f"{credential_store_label()} and link only an opaque reference "
                "to the provider profile?"
            ),
            confirm_text="Save securely",
            cancel_text="Not now",
            icon_name="shield",
        )

    def _notice(self, text: str, *, warning: bool = False) -> None:
        self._shell.accept_application_notice(text, warning=warning)


def create_gui_application_window(
    *,
    platform_services: PlatformServices | None = None,
) -> object:
    """Construct the sole production shell and retain its coordinator."""

    from app.pipeline.controller import PipelineController
    from app.ui.shell.main_window import YomiFrameMainWindow

    services = platform_services or build_platform_services()
    if not isinstance(services, PlatformServices):
        raise TypeError("platform_services must be PlatformServices")
    controller = PipelineController()
    window = YomiFrameMainWindow(platform_services=services)
    coordinator = GuiApplicationCoordinator(
        shell=window,
        controller=controller,
        credential_store_factory=lambda: services.credential_store,
        credential_resolver_factory=lambda: services.credential_resolver,
        parent=window,
    )
    # PySide parent ownership is sufficient at runtime; the explicit Python
    # reference also prevents wrapper collection in embedding/test hosts.
    window._application_coordinator = coordinator  # type: ignore[attr-defined]
    return window


__all__ = [
    "GuiApplicationCoordinator",
    "NewTranslationRequest",
    "ProviderTestReceipt",
    "create_gui_application_window",
]
