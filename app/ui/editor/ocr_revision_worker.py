# -*- coding: utf-8 -*-
"""One-shot Qt worker for an explicit selected-parent OCR revision."""
from __future__ import annotations

import threading
from typing import Any, Callable, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.pipeline.ocr_revision_adapter import ControllerOcrRevisionAdapter
from app.pipeline.ocr_revision_contracts import (
    ExplicitOcrRevisionReceipt,
    ExplicitOcrRevisionRequest,
    OcrRecognitionReceipt,
    OcrRecognitionRequest,
    OcrRevisionError,
    OcrRevisionErrorCode,
    OcrRevisionRecognitionPort,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.ocr_revision_service import (
    ExplicitOcrRevisionService,
    resolve_original_page_asset_binding,
)
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.ocr_revision_model import (
    OcrRevisionCancellationMode,
    OcrRevisionCancellationState,
    OcrRevisionCancelledReceipt,
    OcrRevisionFailureCode,
    OcrRevisionWorkerBusyState,
    OcrRevisionWorkerCommand,
    OcrRevisionWorkerFailure,
    OcrRevisionWorkerReceipt,
    OcrRevisionWorkerStage,
    ocr_revision_selection_from_projection,
)


RecognitionPortFactory = Callable[
    [Callable[[str], None]],
    OcrRevisionRecognitionPort,
]
PostServiceHook = Callable[
    [ExplicitOcrRevisionReceipt, ProjectEditStore, Mapping[str, Any]],
    None,
]


def _open_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
) -> ProjectEditStore:
    metadata = inspect_project_edit_store(project_path)
    if metadata is None:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PERSISTENCE_REJECTED,
            "The project edit journal containing this user parent is unavailable.",
        )
    project_id = project_id_for(project)
    if str(metadata.get("project_id") or "") != project_id:
        raise OcrRevisionError(
            OcrRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
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


class _WorkerRecognitionPort:
    """Wrap recognition so worker cancellation can expose truthful phases."""

    def __init__(
        self,
        worker: "OcrRevisionWorker",
        delegate: OcrRevisionRecognitionPort,
    ) -> None:
        self._worker = worker
        self._delegate = delegate

    def recognize(
        self,
        request: OcrRecognitionRequest,
        *,
        cancellation_probe: Callable[[], bool] | None = None,
    ) -> OcrRecognitionReceipt:
        self._worker._begin_recognition()
        try:
            return self._delegate.recognize(
                request,
                cancellation_probe=cancellation_probe,
            )
        finally:
            self._worker._end_recognition()


class OcrRevisionWorker(QtCore.QObject):
    """Run one source-only OCR transaction outside the GUI thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        command: OcrRevisionWorkerCommand,
        *,
        recognition_port_factory: RecognitionPortFactory | None = None,
        post_service_hook: PostServiceHook | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(command, OcrRevisionWorkerCommand):
            raise TypeError("command must be OcrRevisionWorkerCommand")
        self.command = command
        self._recognition_port_factory = recognition_port_factory
        self._post_service_hook = post_service_hook
        self._run_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._has_run = False
        self._cancel_requested = False
        self._recognition_active = False
        self._recognition_completed = False
        self._recognition_probe_count = 0
        self._post_recognition_probe_count = 0
        self._persistence_locked = False
        self._stage = OcrRevisionWorkerStage.LOADING_PROJECT

    def request_cancel(self) -> bool:
        """Request cooperative cancellation without touching Qt-owned state."""

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
                    OcrRevisionFailureCode.WORKER_REUSED,
                    OcrRevisionWorkerStage.LOADING_PROJECT,
                    "OcrRevisionWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: ExplicitOcrRevisionReceipt | None = None
        terminal_receipt: OcrRevisionWorkerReceipt | None = None
        terminal_failure: OcrRevisionWorkerFailure | None = None
        terminal_cancelled: OcrRevisionCancelledReceipt | None = None
        project: Mapping[str, Any] | None = None
        stage = OcrRevisionWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            OcrRevisionCancellationMode.AVAILABLE,
            stage,
            "OCR can be cancelled before persistence.",
        )
        self._emit_busy(stage, "Loading the current project state...")

        try:
            project = load_project_for_editing(self.command.selection.project_path)
            self._cancel_checkpoint(
                "OCR revision was cancelled before opening the edit journal."
            )

            stage = OcrRevisionWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.selection.project_path,
                project,
            )

            stage = OcrRevisionWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact selected-parent revision...")
            snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.identity.page_id,
            )
            self._cancel_checkpoint(
                "OCR revision was cancelled before validating the selection."
            )

            stage = OcrRevisionWorkerStage.VALIDATING_SELECTION
            self._emit_busy(stage, "Validating hierarchy, source, and settings identity...")
            self._validate_selection(snapshot.project, snapshot.ledger)
            binding = resolve_original_page_asset_binding(
                snapshot.project,
                page_id=self.command.identity.page_id,
                project_path=self.command.selection.project_path,
            )
            if binding != self.command.selection.original_page:
                raise OcrRevisionError(
                    OcrRevisionErrorCode.ORIGINAL_ASSET_MISMATCH,
                    "The original page changed after the OCR action became ready.",
                )
            self._cancel_checkpoint(
                "OCR revision was cancelled before preparing its request."
            )

            stage = OcrRevisionWorkerStage.PREPARING_REQUEST
            self._emit_busy(stage, "Preparing the immutable OCR revision request...")
            identity = self.command.identity
            selection = self.command.selection
            request = ExplicitOcrRevisionRequest(
                command_id=identity.operation_id,
                project_id=identity.project_id,
                page_id=identity.page_id,
                parent_id=identity.parent_id,
                root_id=identity.root_id,
                parent_authored_edit_id=identity.parent_authored_edit_id,
                expected_hierarchy_revision_id=identity.hierarchy_revision_id,
                expected_hierarchy_fingerprint=identity.hierarchy_fingerprint,
                expected_effective_page_fingerprint=(
                    identity.effective_page_fingerprint
                ),
                original_page=binding,
                sampling_bbox=selection.sampling_bbox,
                run_settings_snapshot=selection.run_settings_snapshot,
                run_settings_fingerprint=identity.run_settings_fingerprint,
                selected_ocr_engine=identity.selected_ocr_engine,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
            )

            stage = OcrRevisionWorkerStage.INITIALIZING_OWNER
            self._emit_busy(stage, "Initializing the selected OCR owner...")
            delegate = self._make_recognition_port()
            recognition_port = _WorkerRecognitionPort(self, delegate)
            service = ExplicitOcrRevisionService(
                project=project,
                edit_store=store,
                recognition_port=recognition_port,
                cancellation_probe=self._cancellation_probe,
            )
            core_receipt = service.run_explicit_ocr_revision(request)
            if self._post_service_hook is not None:
                self._post_service_hook(core_receipt, store, project)

            stage = OcrRevisionWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing the committed project revision...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=identity.page_id,
            )
            materialized = post_commit_snapshot.project

            stage = OcrRevisionWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing source provenance and stage state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                post_commit_snapshot.ledger,
                project_path=selection.project_path,
            )
            after_selection = ocr_revision_selection_from_projection(
                projection,
                page_id=identity.page_id,
                parent_id=identity.parent_id,
                run_settings_snapshot=selection.run_settings_snapshot,
            )
            terminal_receipt = OcrRevisionWorkerReceipt(
                identity=identity,
                core_receipt=core_receipt,
                project=materialized,
                projection=projection,
                selection=after_selection,
            )
        except OcrRevisionError as exc:
            if exc.code is OcrRevisionErrorCode.CANCELLED:
                terminal_cancelled = OcrRevisionCancelledReceipt(
                    identity=self.command.identity,
                    stage=stage,
                    inference_completed=self._recognition_completed,
                    message=str(exc),
                )
            else:
                terminal_failure = self._failure_from_ocr_error(
                    stage,
                    exc,
                    core_receipt=core_receipt,
                )
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                OcrRevisionFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The OCR revision state is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "OCR revision I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - fail-closed worker guard
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "OCR revision failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = OcrRevisionWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        OcrRevisionFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation(
            OcrRevisionCancellationMode.UNAVAILABLE,
            OcrRevisionWorkerStage.COMPLETE,
            "OCR revision worker finished.",
        )
        self.busy.emit(
            OcrRevisionWorkerBusyState(
                identity=self.command.identity,
                busy=False,
                stage=OcrRevisionWorkerStage.COMPLETE,
                cancellation_mode=OcrRevisionCancellationMode.UNAVAILABLE,
                persistence_started=core_receipt is not None,
                message="OCR revision worker finished.",
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
                    OcrRevisionFailureCode.COMMAND_REJECTED,
                    stage,
                    "OCR revision worker ended without a typed result.",
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
        if project_id_for(project) != identity.project_id:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PROJECT_IDENTITY_MISMATCH,
                "The open project differs from the OCR revision selection.",
            )
        effective = project_effective_page(project, ledger, page_id=identity.page_id)
        if (
            effective.effective_fingerprint != identity.effective_page_fingerprint
            or effective.hierarchy.revision_id != identity.hierarchy_revision_id
            or effective.hierarchy.fingerprint != identity.hierarchy_fingerprint
        ):
            raise OcrRevisionError(
                OcrRevisionErrorCode.STALE_EFFECTIVE_PAGE,
                "The effective page or hierarchy changed; reload Rerun OCR.",
            )
        parents = tuple(
            parent for parent in effective.parents if parent.parent_id == identity.parent_id
        )
        if len(parents) != 1:
            raise OcrRevisionError(
                OcrRevisionErrorCode.PARENT_NOT_FOUND,
                "The selected user parent is unavailable.",
            )
        parent = parents[0]
        lineage = parent.lineage
        if (
            lineage is None
            or parent.root_id != identity.root_id
            or lineage.authored_edit_id != identity.parent_authored_edit_id
            or tuple(lineage.workflow_area_bbox)
            != self.command.selection.sampling_bbox
        ):
            raise OcrRevisionError(
                OcrRevisionErrorCode.PARENT_LINEAGE_MISMATCH,
                "The selected user-parent lineage changed; reload Rerun OCR.",
            )

    def _make_recognition_port(self) -> OcrRevisionRecognitionPort:
        factory = self._recognition_port_factory
        if factory is None:
            return ControllerOcrRevisionAdapter(
                message_callback=self._owner_message,
            )
        value = factory(self._owner_message)
        if not isinstance(value, OcrRevisionRecognitionPort):
            raise TypeError(
                "recognition_port_factory must return OcrRevisionRecognitionPort"
            )
        return value

    def _owner_message(self, message: str) -> None:
        value = str(message or "").strip()
        if value:
            self._emit_busy(OcrRevisionWorkerStage.RECOGNIZING, value)

    def _begin_recognition(self) -> None:
        with self._cancel_lock:
            self._recognition_active = True
            self._recognition_probe_count = 0
        self._emit_busy(
            OcrRevisionWorkerStage.INITIALIZING_OWNER,
            "Initializing the selected OCR engine...",
        )

    def _end_recognition(self) -> None:
        with self._cancel_lock:
            self._recognition_active = False
            self._recognition_completed = True
            self._post_recognition_probe_count = 0
        self._emit_busy(
            OcrRevisionWorkerStage.RECOGNIZING,
            "OCR inference finished; validating its authoritative result...",
        )

    def _cancellation_probe(self) -> bool:
        emit_recognizing = False
        emit_persisting = False
        with self._cancel_lock:
            if self._cancel_requested:
                return True
            if self._recognition_active:
                self._recognition_probe_count += 1
                if self._recognition_probe_count == 2:
                    emit_recognizing = True
            elif self._recognition_completed:
                self._post_recognition_probe_count += 1
                if self._post_recognition_probe_count >= 2:
                    self._persistence_locked = True
                    emit_persisting = True
        if emit_recognizing:
            self._emit_busy(
                OcrRevisionWorkerStage.RECOGNIZING,
                "Running OCR on the exact selected workflow area...",
            )
        if emit_persisting:
            self._emit_cancellation(
                OcrRevisionCancellationMode.LOCKED,
                OcrRevisionWorkerStage.PERSISTING,
                "The OCR revision is being persisted and can no longer be cancelled.",
            )
            self._emit_busy(
                OcrRevisionWorkerStage.PERSISTING,
                "Saving the OCR artifact and selected revision atomically...",
            )
        return False

    def _cancel_checkpoint(self, message: str) -> None:
        if self._cancellation_probe():
            raise OcrRevisionError(OcrRevisionErrorCode.CANCELLED, message)

    def _emit_busy(self, stage: OcrRevisionWorkerStage, message: str) -> None:
        self._stage = stage
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            if persistence_started:
                mode = OcrRevisionCancellationMode.LOCKED
            elif self._cancel_requested:
                mode = OcrRevisionCancellationMode.REQUESTED_DEFERRED
            else:
                mode = OcrRevisionCancellationMode.AVAILABLE
        self.busy.emit(
            OcrRevisionWorkerBusyState(
                identity=self.command.identity,
                busy=True,
                stage=stage,
                cancellation_mode=mode,
                persistence_started=persistence_started,
                message=str(message or "OCR revision is running."),
            )
        )

    def _emit_cancellation(
        self,
        mode: OcrRevisionCancellationMode,
        stage: OcrRevisionWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
        self.cancellation.emit(
            OcrRevisionCancellationState(
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

    def _failure_from_ocr_error(
        self,
        stage: OcrRevisionWorkerStage,
        exc: OcrRevisionError,
        *,
        core_receipt: ExplicitOcrRevisionReceipt | None,
    ) -> OcrRevisionWorkerFailure:
        code = {
            OcrRevisionErrorCode.PROJECT_IDENTITY_MISMATCH: OcrRevisionFailureCode.PROJECT_INVALID,
            OcrRevisionErrorCode.PAGE_NOT_FOUND: OcrRevisionFailureCode.PAGE_NOT_FOUND,
            OcrRevisionErrorCode.PARENT_NOT_FOUND: OcrRevisionFailureCode.PARENT_NOT_FOUND,
            OcrRevisionErrorCode.PARENT_LINEAGE_MISMATCH: OcrRevisionFailureCode.SNAPSHOT_STALE,
            OcrRevisionErrorCode.STALE_HIERARCHY: OcrRevisionFailureCode.SNAPSHOT_STALE,
            OcrRevisionErrorCode.STALE_EFFECTIVE_PAGE: OcrRevisionFailureCode.SNAPSHOT_STALE,
            OcrRevisionErrorCode.STALE_PAGE_HEAD: OcrRevisionFailureCode.SNAPSHOT_STALE,
            OcrRevisionErrorCode.STALE_GLOBAL_HEAD: OcrRevisionFailureCode.SNAPSHOT_STALE,
            OcrRevisionErrorCode.SOURCE_NOT_RUNNABLE: OcrRevisionFailureCode.SOURCE_NOT_RUNNABLE,
            OcrRevisionErrorCode.SETTINGS_MISMATCH: OcrRevisionFailureCode.SETTINGS_STALE,
            OcrRevisionErrorCode.ORIGINAL_ASSET_UNAVAILABLE: OcrRevisionFailureCode.ORIGINAL_ASSET_UNAVAILABLE,
            OcrRevisionErrorCode.ORIGINAL_ASSET_MISMATCH: OcrRevisionFailureCode.ORIGINAL_ASSET_MISMATCH,
            OcrRevisionErrorCode.RECOGNITION_FAILED: OcrRevisionFailureCode.RECOGNITION_FAILED,
            OcrRevisionErrorCode.NON_AUTHORITATIVE_RESULT: OcrRevisionFailureCode.NON_AUTHORITATIVE_RESULT,
            OcrRevisionErrorCode.EMPTY_RESULT: OcrRevisionFailureCode.EMPTY_RESULT,
            OcrRevisionErrorCode.PROJECTION_REJECTED: OcrRevisionFailureCode.PROJECTION_FAILED,
            OcrRevisionErrorCode.PERSISTENCE_REJECTED: OcrRevisionFailureCode.PERSISTENCE_REJECTED,
        }.get(exc.code, OcrRevisionFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "OCR revision was rejected.",
            exc,
            core_receipt=core_receipt,
        )

    def _failure(
        self,
        code: OcrRevisionFailureCode,
        stage: OcrRevisionWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        core_receipt: ExplicitOcrRevisionReceipt | None = None,
    ) -> OcrRevisionWorkerFailure:
        committed = core_receipt is not None
        if committed and code is not OcrRevisionFailureCode.EDIT_STORE_FAILED:
            code = OcrRevisionFailureCode.COMMITTED_STALE
            message = (
                "The OCR revision was published, but the latest project state "
                "could not be verified for an atomic UI refresh. Reload the "
                "project to inspect the committed revision."
            )
        return OcrRevisionWorkerFailure(
            identity=self.command.identity,
            code=code,
            stage=stage,
            message=str(message or "OCR revision failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            persistence_committed=committed,
            core_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: OcrRevisionWorkerStage,
        committed: bool,
    ) -> OcrRevisionFailureCode:
        if committed:
            return OcrRevisionFailureCode.COMMITTED_STALE
        if stage is OcrRevisionWorkerStage.LOADING_PROJECT:
            return OcrRevisionFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            OcrRevisionWorkerStage.OPENING_EDIT_STORE,
            OcrRevisionWorkerStage.READING_SNAPSHOT,
            OcrRevisionWorkerStage.CLOSING_EDIT_STORE,
        }:
            return OcrRevisionFailureCode.EDIT_STORE_FAILED
        if stage in {
            OcrRevisionWorkerStage.VALIDATING_SELECTION,
            OcrRevisionWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return OcrRevisionFailureCode.PROJECTION_FAILED
        if stage in {
            OcrRevisionWorkerStage.INITIALIZING_OWNER,
            OcrRevisionWorkerStage.RECOGNIZING,
        }:
            return OcrRevisionFailureCode.RECOGNITION_FAILED
        if stage is OcrRevisionWorkerStage.PERSISTING:
            return OcrRevisionFailureCode.PERSISTENCE_REJECTED
        return OcrRevisionFailureCode.PROJECT_INVALID


__all__ = ["OcrRevisionWorker", "PostServiceHook", "RecognitionPortFactory"]
