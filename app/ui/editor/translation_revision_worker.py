# -*- coding: utf-8 -*-
"""One-shot Qt worker for an explicit selected-parent translation revision."""
from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.pipeline.translation_revision_adapter import (
    ControllerTranslationRevisionAdapter,
)
from app.pipeline.translation_revision_contracts import (
    ExplicitTranslationRevisionReceipt,
    ExplicitTranslationRevisionRequest,
    TranslationExecutionReceipt,
    TranslationExecutionRequest,
    TranslationRevisionError,
    TranslationRevisionErrorCode,
    TranslationRevisionExecutionPort,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.translation_revision_service import (
    ExplicitTranslationRevisionService,
    compile_translation_revision_policy_snapshots,
)
from app.ui.viewmodels.translation_revision_model import (
    TranslationRevisionCancellationMode,
    TranslationRevisionCancellationState,
    TranslationRevisionCancelledReceipt,
    TranslationRevisionFailureCode,
    TranslationRevisionWorkerBusyState,
    TranslationRevisionWorkerCommand,
    TranslationRevisionWorkerFailure,
    TranslationRevisionWorkerReceipt,
    TranslationRevisionWorkerStage,
    translation_revision_selection_from_projection,
)


ClientFactory = Callable[[Any, Any | None], Any]
SettingsMaterializer = Callable[..., Any]
PostServiceHook = Callable[
    [ExplicitTranslationRevisionReceipt, ProjectEditStore, Mapping[str, Any]],
    None,
]


def _open_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
) -> ProjectEditStore:
    metadata = inspect_project_edit_store(project_path)
    if metadata is None:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PERSISTENCE_REJECTED,
            "The project edit journal containing this user parent is unavailable.",
        )
    project_id = project_id_for(project)
    if str(metadata.get("project_id") or "") != project_id:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
            "The project and edit journal identities do not match.",
        )
    return ProjectEditStore(
        project_path=project_path,
        project_id=project_id,
        project_origin_sha256=str(metadata.get("project_origin_sha256") or ""),
        automated_state_sha256=automated_state_fingerprint(project),
        base_ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
        base_artifact_revisions=project["artifact_revisions"],
    )


class _WorkerTranslationPort:
    """Expose the controller-owned non-preemptive inference as worker phases."""

    def __init__(
        self,
        worker: "TranslationRevisionWorker",
        delegate: TranslationRevisionExecutionPort,
    ) -> None:
        self._worker = worker
        self._delegate = delegate

    def translate(
        self,
        request: TranslationExecutionRequest,
        *,
        cancellation_probe: Callable[[], bool] | None = None,
    ) -> TranslationExecutionReceipt:
        self._worker._begin_translation()
        try:
            return self._delegate.translate(
                request,
                cancellation_probe=cancellation_probe,
            )
        finally:
            self._worker._end_translation()


class TranslationRevisionWorker(QtCore.QObject):
    """Run exactly one target-only translation transaction off the GUI thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        command: TranslationRevisionWorkerCommand,
        *,
        client_factory: ClientFactory | None = None,
        settings_materializer: SettingsMaterializer | None = None,
        post_service_hook: PostServiceHook | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(command, TranslationRevisionWorkerCommand):
            raise TypeError("command must be TranslationRevisionWorkerCommand")
        self.command = command
        self._client_factory = client_factory
        self._settings_materializer = settings_materializer
        self._post_service_hook = post_service_hook
        self._run_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._has_run = False
        self._cancel_requested = False
        self._translation_active = False
        self._translation_completed = False
        self._translation_probe_count = 0
        self._post_translation_probe_count = 0
        self._persistence_locked = False
        self._stage = TranslationRevisionWorkerStage.LOADING_PROJECT

    def request_cancel(self) -> bool:
        """Request deferred cancellation without touching Qt-owned state."""

        with self._cancel_lock:
            if self._persistence_locked or self._cancel_requested:
                return False
            self._cancel_requested = True
            return True

    @QtCore.Slot()
    def run(self) -> None:
        if not self._claim_run():
            self.failure.emit(
                self._failure(
                    TranslationRevisionFailureCode.WORKER_REUSED,
                    TranslationRevisionWorkerStage.LOADING_PROJECT,
                    "TranslationRevisionWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: ExplicitTranslationRevisionReceipt | None = None
        terminal_receipt: TranslationRevisionWorkerReceipt | None = None
        terminal_failure: TranslationRevisionWorkerFailure | None = None
        terminal_cancelled: TranslationRevisionCancelledReceipt | None = None
        project: Mapping[str, Any] | None = None
        stage = TranslationRevisionWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            TranslationRevisionCancellationMode.AVAILABLE,
            stage,
            "Translation can be cancelled before persistence.",
        )
        self._emit_busy(stage, "Loading the current project state...")

        try:
            selection = self.command.selection
            identity = self.command.identity
            project = load_project_for_editing(selection.project_path)
            self._cancel_checkpoint(
                "Translation was cancelled before opening the edit journal."
            )

            stage = TranslationRevisionWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(selection.project_path, project)

            stage = TranslationRevisionWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact selected-parent revision...")
            snapshot = store.materialize_project_snapshot(
                project,
                page_id=identity.page_id,
            )
            self._cancel_checkpoint(
                "Translation was cancelled before validating the selection."
            )

            stage = TranslationRevisionWorkerStage.VALIDATING_SELECTION
            self._emit_busy(
                stage,
                "Validating source, provider, glossary, context, and hierarchy...",
            )
            self._validate_selection(snapshot.project, snapshot.ledger)
            self._cancel_checkpoint(
                "Translation was cancelled before preparing its request."
            )

            stage = TranslationRevisionWorkerStage.PREPARING_REQUEST
            self._emit_busy(stage, "Preparing the immutable translation request...")
            request = ExplicitTranslationRevisionRequest(
                command_id=identity.operation_id,
                project_id=identity.project_id,
                page_id=identity.page_id,
                parent_id=identity.parent_id,
                root_id=identity.root_id,
                parent_authored_edit_id=identity.parent_authored_edit_id,
                parent_role=identity.parent_role,
                policy_region_type=identity.policy_region_type,
                bubble_local_nested_speech=identity.bubble_local_nested_speech,
                expected_hierarchy_revision_id=identity.hierarchy_revision_id,
                expected_hierarchy_fingerprint=identity.hierarchy_fingerprint,
                expected_effective_page_fingerprint=(
                    identity.effective_page_fingerprint
                ),
                effective_source_text=selection.effective_source_text,
                effective_source_authority=selection.effective_source_authority,
                effective_source_fingerprint=identity.effective_source_fingerprint,
                source_revision_id=identity.source_revision_id,
                source_selection_edit_id=identity.source_selection_edit_id,
                run_settings_snapshot=selection.run_settings_snapshot,
                run_settings_fingerprint=identity.run_settings_fingerprint,
                provider=identity.provider,
                glossary_snapshot=selection.glossary_snapshot,
                glossary_fingerprint=identity.glossary_fingerprint,
                prior_page_context=selection.prior_page_context,
                context_fingerprint=identity.context_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
            )

            stage = TranslationRevisionWorkerStage.INITIALIZING_OWNER
            self._emit_busy(stage, "Initializing the selected translation owner...")
            delegate = ControllerTranslationRevisionAdapter(
                runtime_binding=selection.runtime_provider_binding,
                client_factory=self._client_factory,
                settings_materializer=self._settings_materializer,
            )
            translation_port = _WorkerTranslationPort(self, delegate)
            service = ExplicitTranslationRevisionService(
                project=project,
                edit_store=store,
                translation_port=translation_port,
                cancellation_probe=self._cancellation_probe,
            )
            core_receipt = service.run_explicit_translation_revision(request)
            if self._post_service_hook is not None:
                self._post_service_hook(core_receipt, store, project)

            stage = TranslationRevisionWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing the committed project revision...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=identity.page_id,
            )
            materialized = post_commit_snapshot.project

            stage = TranslationRevisionWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing target provenance and stage state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                post_commit_snapshot.ledger,
                project_path=selection.project_path,
            )
            policy = compile_translation_revision_policy_snapshots(
                materialized,
                page_id=identity.page_id,
                run_settings_snapshot=selection.run_settings_snapshot,
            )
            after_selection = translation_revision_selection_from_projection(
                projection,
                page_id=identity.page_id,
                parent_id=identity.parent_id,
                run_settings_snapshot=selection.run_settings_snapshot,
                runtime_provider_binding=selection.runtime_provider_binding,
                glossary_snapshot=policy.glossary_snapshot,
                prior_page_context=policy.prior_page_context,
            )
            terminal_receipt = TranslationRevisionWorkerReceipt(
                identity=identity,
                core_receipt=core_receipt,
                project=materialized,
                projection=projection,
                selection=after_selection,
            )
        except TranslationRevisionError as exc:
            if exc.code is TranslationRevisionErrorCode.CANCELLED:
                terminal_cancelled = TranslationRevisionCancelledReceipt(
                    identity=self.command.identity,
                    stage=stage,
                    inference_completed=self._translation_completed,
                    message=str(exc),
                )
            else:
                terminal_failure = self._failure_from_translation_error(
                    stage,
                    exc,
                    core_receipt=core_receipt,
                )
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                TranslationRevisionFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The translation revision state is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Translation revision I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - fail-closed worker guard
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Translation revision failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = TranslationRevisionWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        TranslationRevisionFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation(
            TranslationRevisionCancellationMode.UNAVAILABLE,
            TranslationRevisionWorkerStage.COMPLETE,
            "Translation revision worker finished.",
        )
        self.busy.emit(
            TranslationRevisionWorkerBusyState(
                identity=self.command.identity,
                busy=False,
                stage=TranslationRevisionWorkerStage.COMPLETE,
                cancellation_mode=TranslationRevisionCancellationMode.UNAVAILABLE,
                persistence_started=core_receipt is not None,
                message="Translation revision worker finished.",
            )
        )
        if terminal_receipt is not None:
            self.receipt.emit(terminal_receipt)
        elif terminal_cancelled is not None:
            self.cancelled.emit(terminal_cancelled)
        elif terminal_failure is not None:
            if terminal_failure.stale:
                self.stale.emit(terminal_failure)
            else:
                self.failure.emit(terminal_failure)
        else:  # pragma: no cover
            self.failure.emit(
                self._failure(
                    TranslationRevisionFailureCode.COMMAND_REJECTED,
                    stage,
                    "Translation revision worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _validate_selection(
        self,
        project: Mapping[str, Any],
        ledger: ProjectEditLedger,
    ) -> None:
        identity = self.command.identity
        selection = self.command.selection
        if project_id_for(project) != identity.project_id:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
                "The open project differs from the translation selection.",
            )
        from app.ui.shell.project_projection import project_ui_projection

        projection = project_ui_projection(
            project,
            ledger,
            project_path=selection.project_path,
        )
        policy = compile_translation_revision_policy_snapshots(
            project,
            page_id=identity.page_id,
            run_settings_snapshot=selection.run_settings_snapshot,
        )
        current = translation_revision_selection_from_projection(
            projection,
            page_id=identity.page_id,
            parent_id=identity.parent_id,
            run_settings_snapshot=selection.run_settings_snapshot,
            runtime_provider_binding=selection.runtime_provider_binding,
            glossary_snapshot=policy.glossary_snapshot,
            prior_page_context=policy.prior_page_context,
        )
        if current.glossary_fingerprint != identity.glossary_fingerprint:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.GLOSSARY_MISMATCH,
                "Existing style-guide content changed; reload Retranslate Parent.",
            )
        if current.context_fingerprint != identity.context_fingerprint:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.CONTEXT_MISMATCH,
                "Committed prior-page context changed; reload Retranslate Parent.",
            )
        if current != selection:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.STALE_EFFECTIVE_PAGE,
                "The selected parent, source, settings, or hierarchy changed; reload Retranslate Parent.",
            )

    def _begin_translation(self) -> None:
        with self._cancel_lock:
            self._translation_active = True
            self._translation_probe_count = 0
        self._emit_busy(
            TranslationRevisionWorkerStage.INITIALIZING_OWNER,
            "Initializing the selected translation provider...",
        )

    def _end_translation(self) -> None:
        with self._cancel_lock:
            self._translation_active = False
            self._translation_completed = True
            self._post_translation_probe_count = 0
        self._emit_busy(
            TranslationRevisionWorkerStage.TRANSLATING,
            "Translation inference finished; validating its bound result...",
        )

    def _cancellation_probe(self) -> bool:
        emit_translating = False
        emit_persisting = False
        with self._cancel_lock:
            if self._cancel_requested:
                return True
            if self._translation_active:
                self._translation_probe_count += 1
                if self._translation_probe_count == 2:
                    emit_translating = True
            elif self._translation_completed:
                self._post_translation_probe_count += 1
                if self._post_translation_probe_count >= 2:
                    self._persistence_locked = True
                    emit_persisting = True
        if emit_translating:
            self._emit_busy(
                TranslationRevisionWorkerStage.TRANSLATING,
                "Translating only the exact selected parent...",
            )
        if emit_persisting:
            self._emit_cancellation(
                TranslationRevisionCancellationMode.LOCKED,
                TranslationRevisionWorkerStage.PERSISTING,
                "The translation revision is being persisted and can no longer be cancelled.",
            )
            self._emit_busy(
                TranslationRevisionWorkerStage.PERSISTING,
                "Saving the target artifact and selected revision atomically...",
            )
        return False

    def _cancel_checkpoint(self, message: str) -> None:
        if self._cancellation_probe():
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.CANCELLED,
                message,
            )

    def _emit_busy(
        self,
        stage: TranslationRevisionWorkerStage,
        message: str,
    ) -> None:
        self._stage = stage
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            if persistence_started:
                mode = TranslationRevisionCancellationMode.LOCKED
            elif self._cancel_requested:
                mode = TranslationRevisionCancellationMode.REQUESTED_DEFERRED
            else:
                mode = TranslationRevisionCancellationMode.AVAILABLE
        self.busy.emit(
            TranslationRevisionWorkerBusyState(
                identity=self.command.identity,
                busy=True,
                stage=stage,
                cancellation_mode=mode,
                persistence_started=persistence_started,
                message=str(message or "Translation revision is running."),
            )
        )

    def _emit_cancellation(
        self,
        mode: TranslationRevisionCancellationMode,
        stage: TranslationRevisionWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
        self.cancellation.emit(
            TranslationRevisionCancellationState(
                identity=self.command.identity,
                mode=mode,
                stage=stage,
                persistence_started=persistence_started,
                message=str(message),
            )
        )

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True

    def _failure_from_translation_error(
        self,
        stage: TranslationRevisionWorkerStage,
        exc: TranslationRevisionError,
        *,
        core_receipt: ExplicitTranslationRevisionReceipt | None,
    ) -> TranslationRevisionWorkerFailure:
        code = {
            TranslationRevisionErrorCode.PROJECT_IDENTITY_MISMATCH: TranslationRevisionFailureCode.PROJECT_INVALID,
            TranslationRevisionErrorCode.PAGE_NOT_FOUND: TranslationRevisionFailureCode.PAGE_NOT_FOUND,
            TranslationRevisionErrorCode.PARENT_NOT_FOUND: TranslationRevisionFailureCode.PARENT_NOT_FOUND,
            TranslationRevisionErrorCode.PARENT_LINEAGE_MISMATCH: TranslationRevisionFailureCode.SNAPSHOT_STALE,
            TranslationRevisionErrorCode.STALE_HIERARCHY: TranslationRevisionFailureCode.SNAPSHOT_STALE,
            TranslationRevisionErrorCode.STALE_EFFECTIVE_PAGE: TranslationRevisionFailureCode.SNAPSHOT_STALE,
            TranslationRevisionErrorCode.STALE_PAGE_HEAD: TranslationRevisionFailureCode.SNAPSHOT_STALE,
            TranslationRevisionErrorCode.STALE_GLOBAL_HEAD: TranslationRevisionFailureCode.SNAPSHOT_STALE,
            TranslationRevisionErrorCode.SOURCE_NOT_CURRENT: TranslationRevisionFailureCode.SOURCE_NOT_CURRENT,
            TranslationRevisionErrorCode.SOURCE_MISMATCH: TranslationRevisionFailureCode.SOURCE_MISMATCH,
            TranslationRevisionErrorCode.SETTINGS_MISMATCH: TranslationRevisionFailureCode.SETTINGS_STALE,
            TranslationRevisionErrorCode.GLOSSARY_MISMATCH: TranslationRevisionFailureCode.GLOSSARY_STALE,
            TranslationRevisionErrorCode.CONTEXT_MISMATCH: TranslationRevisionFailureCode.CONTEXT_STALE,
            TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE: TranslationRevisionFailureCode.PROVIDER_UNAVAILABLE,
            TranslationRevisionErrorCode.MODEL_MISSING: TranslationRevisionFailureCode.MODEL_MISSING,
            TranslationRevisionErrorCode.TRANSLATION_FAILED: TranslationRevisionFailureCode.TRANSLATION_FAILED,
            TranslationRevisionErrorCode.EMPTY_RESULT: TranslationRevisionFailureCode.EMPTY_RESULT,
            TranslationRevisionErrorCode.PROJECTION_REJECTED: TranslationRevisionFailureCode.PROJECTION_FAILED,
            TranslationRevisionErrorCode.PERSISTENCE_REJECTED: TranslationRevisionFailureCode.PERSISTENCE_REJECTED,
        }.get(exc.code, TranslationRevisionFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Translation revision was rejected.",
            exc,
            core_receipt=core_receipt,
        )

    def _failure(
        self,
        code: TranslationRevisionFailureCode,
        stage: TranslationRevisionWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        core_receipt: ExplicitTranslationRevisionReceipt | None = None,
    ) -> TranslationRevisionWorkerFailure:
        committed = core_receipt is not None
        if committed and code is not TranslationRevisionFailureCode.EDIT_STORE_FAILED:
            code = TranslationRevisionFailureCode.COMMITTED_STALE
            message = (
                "The translation revision was published, but the latest project "
                "state could not be verified for an atomic UI refresh. Reload "
                "the project to inspect the committed revision."
            )
        return TranslationRevisionWorkerFailure(
            identity=self.command.identity,
            code=code,
            stage=stage,
            message=str(message or "Translation revision failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            persistence_committed=committed,
            core_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: TranslationRevisionWorkerStage,
        committed: bool,
    ) -> TranslationRevisionFailureCode:
        if committed:
            return TranslationRevisionFailureCode.COMMITTED_STALE
        if stage is TranslationRevisionWorkerStage.LOADING_PROJECT:
            return TranslationRevisionFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            TranslationRevisionWorkerStage.OPENING_EDIT_STORE,
            TranslationRevisionWorkerStage.READING_SNAPSHOT,
            TranslationRevisionWorkerStage.CLOSING_EDIT_STORE,
        }:
            return TranslationRevisionFailureCode.EDIT_STORE_FAILED
        if stage in {
            TranslationRevisionWorkerStage.VALIDATING_SELECTION,
            TranslationRevisionWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return TranslationRevisionFailureCode.PROJECTION_FAILED
        if stage in {
            TranslationRevisionWorkerStage.RESOLVING_CREDENTIAL,
            TranslationRevisionWorkerStage.INITIALIZING_OWNER,
            TranslationRevisionWorkerStage.TRANSLATING,
        }:
            return TranslationRevisionFailureCode.TRANSLATION_FAILED
        if stage is TranslationRevisionWorkerStage.PERSISTING:
            return TranslationRevisionFailureCode.PERSISTENCE_REJECTED
        return TranslationRevisionFailureCode.PROJECT_INVALID


__all__ = [
    "ClientFactory",
    "PostServiceHook",
    "SettingsMaterializer",
    "TranslationRevisionWorker",
]
