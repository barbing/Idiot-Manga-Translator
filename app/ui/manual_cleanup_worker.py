# -*- coding: utf-8 -*-
"""One-shot Qt workers for GUI-owned manual cleanup context and revisions."""
from __future__ import annotations

import threading
from typing import Any, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import ProjectEditStore, inspect_project_edit_store
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.manual_cleanup import (
    ManualCleanupCancellationToken,
    ManualCleanupFailure,
    ManualCleanupFailureCode,
    ManualCleanupProgress,
    ManualCleanupReceipt,
    ManualCleanupRequest,
    ManualCleanupService,
    ManualCleanupStage,
    ManualCleanupStatus,
)
from app.project_edits.projection import EffectivePageSnapshot, project_effective_page
from app.ui.viewmodels.manual_cleanup_model import (
    ManualCleanupCancellationState,
    ManualCleanupContextCommand,
    ManualCleanupWorkerCommand,
    ManualCleanupWorkerFailure,
    ManualCleanupWorkerFailureCode,
    ManualCleanupWorkerMode,
    ManualCleanupWorkerStage,
    genesis_edit_heads,
    worker_failure_from_preflight,
)


class _ExactPageNotFound(LookupError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise ValueError("project pages must be a list")
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if not matches:
        raise _ExactPageNotFound(page_id)
    if len(matches) != 1:
        raise ValueError(f"project page identity is duplicated: {page_id}")
    return matches[0]


def _load_snapshot(
    project_path: str,
    page_id: str,
) -> tuple[dict[str, Any], EffectivePageSnapshot]:
    project = load_project_for_editing(project_path)
    _exact_page(project, page_id)
    ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
    snapshot = project_effective_page(project, ledger, page_id=page_id)
    if snapshot.page_id != page_id:
        raise ValueError("projected cleanup page identity changed")
    return project, snapshot


def _load_preview_snapshot(
    project_path: str,
    page_id: str,
) -> tuple[
    dict[str, Any],
    ProjectEditLedger,
    EffectivePageSnapshot,
    str,
    str,
]:
    """Capture preview state and CAS heads from one store read snapshot."""

    project = load_project_for_editing(project_path)
    _exact_page(project, page_id)
    store = _open_project_edit_store(project_path, project, create=False)
    if store is None:
        ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
        page_head, global_head = genesis_edit_heads()
    else:
        try:
            read_snapshot = store.materialize_project_snapshot(
                project,
                page_id=page_id,
            )
            project = read_snapshot.project
            ledger = read_snapshot.ledger
            page_head = read_snapshot.page_head_sha256
            global_head = read_snapshot.global_head_sha256
        finally:
            store.close()
    snapshot = project_effective_page(project, ledger, page_id=page_id)
    if snapshot.page_id != page_id:
        raise ValueError("projected cleanup page identity changed")
    return project, ledger, snapshot, page_head, global_head


def _open_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
    *,
    create: bool,
) -> ProjectEditStore | None:
    """Open the exact sidecar; preview never passes ``create=True``."""

    metadata = inspect_project_edit_store(project_path)
    if metadata is None and not create:
        return None
    project_id = project_id_for(project)
    if metadata is not None:
        if str(metadata.get("project_id") or "") != project_id:
            raise ValueError("project and edit-store identities do not match")
        origin_sha256 = str(metadata.get("project_origin_sha256") or "")
    else:
        origin_sha256 = project_origin_fingerprint(project)
    return ProjectEditStore(
        project_path=project_path,
        project_id=project_id,
        project_origin_sha256=origin_sha256,
        automated_state_sha256=automated_state_fingerprint(project),
        base_ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
        base_artifact_revisions=project["artifact_revisions"],
    )


def discard_manual_cleanup_preview(project_path: str, page_id: str) -> None:
    """Discard only the managed temporary preview for one exact project page."""

    project_path = str(project_path or "").strip()
    page_id = str(page_id or "").strip()
    if not project_path:
        raise ValueError("project_path is required")
    if not page_id:
        raise ValueError("page_id is required")
    ManualCleanupService(
        project_path=project_path,
        edit_store=None,
    ).discard_preview(page_id)


class ManualCleanupContextWorker(QtCore.QObject):
    """Resolve comparison/base context without requiring or creating a mask."""

    context_ready = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: ManualCleanupContextCommand) -> None:
        super().__init__()
        if not isinstance(command, ManualCleanupContextCommand):
            raise TypeError("command must be ManualCleanupContextCommand")
        self.command = command
        self._run_lock = threading.Lock()
        self._has_run = False

    @QtCore.Slot()
    def run(self) -> None:
        if not self._claim_run():
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.WORKER_REUSED,
                    ManualCleanupWorkerStage.LOADING_PROJECT,
                    "ManualCleanupContextWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return
        stage = ManualCleanupWorkerStage.LOADING_PROJECT
        try:
            project, snapshot = _load_snapshot(
                self.command.project_path,
                self.command.page_id,
            )
            stage = ManualCleanupWorkerStage.PREFLIGHT
            service = ManualCleanupService(
                project_path=self.command.project_path,
                edit_store=None,
            )
            ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
            rebase_review = None
            context_snapshot = snapshot
            if self.command.coverage_target is None:
                rebase_review = service.rebase_review(project, ledger, snapshot)
                context_snapshot = (
                    service.rebase_snapshot(
                        project,
                        ledger,
                        snapshot,
                        rebase_review,
                    )
                    if rebase_review is not None
                    else snapshot
                )
            context = service.context(
                context_snapshot,
                rebase_review=rebase_review,
                coverage_target=self.command.coverage_target,
            )
            if context.page_id != self.command.page_id:
                raise ValueError("cleanup context belongs to another page")
            self.context_ready.emit(context)
        except _ExactPageNotFound:
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.PAGE_NOT_FOUND,
                    ManualCleanupWorkerStage.FINDING_PAGE,
                    f"Project page is unavailable: {self.command.page_id}",
                )
            )
        except FileNotFoundError as exc:
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.PROJECT_LOAD_FAILED,
                    stage,
                    "The project file is unavailable.",
                    exc,
                )
            )
        except ManualCleanupFailure as exc:
            self.failure.emit(_failure_from_service(self.command, stage, exc))
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.PROJECT_INVALID,
                    stage,
                    str(exc) or "The cleanup context could not be resolved.",
                    exc,
                )
            )
        self.finished.emit()

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True


class ManualCleanupWorker(QtCore.QObject):
    """Run exactly one preview or commit after being moved to a QThread."""

    preflight = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    preview_ready = QtCore.Signal(object)
    committed = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: ManualCleanupWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, ManualCleanupWorkerCommand):
            raise TypeError("command must be ManualCleanupWorkerCommand")
        self.command = command
        self._token = ManualCleanupCancellationToken()
        self._run_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._has_run = False
        self._cancellation_locked = False

    @property
    def cancellation_token(self) -> ManualCleanupCancellationToken:
        return self._token

    def request_cancel(self) -> bool:
        """Direct thread-safe cancellation; false after persistence locks."""

        with self._cancel_lock:
            if self._cancellation_locked:
                return False
            self._token.cancel()
            return True

    @QtCore.Slot()
    def run(self) -> None:
        if not self._claim_run():
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.WORKER_REUSED,
                    ManualCleanupWorkerStage.LOADING_PROJECT,
                    "ManualCleanupWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        self.cancellation.emit(
            ManualCleanupCancellationState(
                page_id=self.command.page_id,
                enabled=True,
                message="Manual cleanup can be cancelled safely.",
            )
        )
        store: ProjectEditStore | None = None
        terminal_receipt: ManualCleanupReceipt | None = None
        terminal_failure: ManualCleanupWorkerFailure | None = None
        stage = ManualCleanupWorkerStage.LOADING_PROJECT
        try:
            if self.command.mode is ManualCleanupWorkerMode.PREVIEW:
                stage = ManualCleanupWorkerStage.READING_EDIT_HEADS
                (
                    project,
                    ledger,
                    snapshot,
                    page_head,
                    global_head,
                ) = _load_preview_snapshot(
                    self.command.project_path,
                    self.command.page_id,
                )
                # The store is closed before this service can invoke the backend.
                service = ManualCleanupService(
                    project_path=self.command.project_path,
                    edit_store=None,
                )
                stage = ManualCleanupWorkerStage.PREFLIGHT
                rebase_review = None
                request_snapshot = snapshot
                if self.command.rebase_review is not None:
                    rebase_review = service.rebase_review(project, ledger, snapshot)
                    if rebase_review != self.command.rebase_review:
                        raise ManualCleanupFailure(
                            ManualCleanupFailureCode.PREVIEW_STALE,
                            "The stale cleanup selection changed; reload the editor.",
                        )
                    request_snapshot = service.rebase_snapshot(
                        project,
                        ledger,
                        snapshot,
                        rebase_review,
                    )
                    preflight = service.preflight_rebase(
                        request_snapshot,
                        rebase_review,
                    )
                else:
                    preflight = service.preflight(
                        snapshot,
                        self.command.erase_mask_png,
                        self.command.protect_mask_png,
                        parameters=self.command.parameters,
                        coverage_target=self.command.coverage_target,
                    )
                if preflight.page_id != self.command.page_id:
                    raise ValueError("manual cleanup preflight belongs to another page")
                self.preflight.emit(preflight)
                if not preflight.ready:
                    terminal_failure = worker_failure_from_preflight(
                        self.command,
                        preflight,
                    )
                else:
                    stage = ManualCleanupWorkerStage.PREVIEWING
                    request = ManualCleanupRequest(
                        snapshot=request_snapshot,
                        erase_mask_png=self.command.erase_mask_png,
                        protect_mask_png=self.command.protect_mask_png,
                        parameters=self.command.parameters,
                        expected_page_head_sha256=page_head,
                        expected_global_head_sha256=global_head,
                        reviewed_stale_selection_edit_ids=(
                            rebase_review.stale_selection_edit_ids
                            if rebase_review is not None
                            else ()
                        ),
                        reviewed_stale_effective_fingerprint=(
                            rebase_review.stale_effective_fingerprint
                            if rebase_review is not None
                            else ""
                        ),
                        operation_id=self.command.operation_id,
                        transaction_id=self.command.transaction_id,
                        coverage_target=self.command.coverage_target,
                    )
                    terminal_receipt = (
                        service.preview_rebase(
                            request,
                            rebase_review,
                            cancellation=self._token,
                            progress=self._forward_progress,
                        )
                        if rebase_review is not None
                        else service.preview(
                            request,
                            cancellation=self._token,
                            progress=self._forward_progress,
                        )
                    )
            else:
                project, snapshot = _load_snapshot(
                    self.command.project_path,
                    self.command.page_id,
                )
                lease = self.command.preview_lease
                if lease is None:  # constructor already prevents this
                    raise ValueError("commit command has no preview lease")
                if self._token.is_cancelled():
                    service = ManualCleanupService(
                        project_path=self.command.project_path,
                        edit_store=None,
                    )
                else:
                    stage = ManualCleanupWorkerStage.OPENING_COMMIT_STORE
                    store = _open_project_edit_store(
                        self.command.project_path,
                        project,
                        create=True,
                    )
                    if store is None:  # pragma: no cover - create is mandatory
                        raise RuntimeError("commit store was not created")
                    service = ManualCleanupService(
                        project_path=self.command.project_path,
                        edit_store=store,
                    )
                stage = ManualCleanupWorkerStage.COMMITTING
                terminal_receipt = service.commit_preview(
                    lease,
                    cancellation=self._token,
                    progress=self._forward_progress,
                    transaction_id=self.command.transaction_id,
                )
        except _ExactPageNotFound:
            terminal_failure = _worker_failure(
                self.command,
                ManualCleanupWorkerFailureCode.PAGE_NOT_FOUND,
                ManualCleanupWorkerStage.FINDING_PAGE,
                f"Project page is unavailable: {self.command.page_id}",
            )
        except FileNotFoundError as exc:
            terminal_failure = _worker_failure(
                self.command,
                ManualCleanupWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
            )
        except ManualCleanupFailure as exc:
            terminal_failure = _failure_from_service(self.command, stage, exc)
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = _worker_failure(
                self.command,
                (
                    ManualCleanupWorkerFailureCode.EDIT_STORE_FAILED
                    if stage
                    in {
                        ManualCleanupWorkerStage.READING_EDIT_HEADS,
                        ManualCleanupWorkerStage.OPENING_COMMIT_STORE,
                        ManualCleanupWorkerStage.CLOSING_EDIT_STORE,
                    }
                    else ManualCleanupWorkerFailureCode.PROJECT_INVALID
                ),
                stage,
                str(exc) or "The project cannot form a manual cleanup request.",
                exc,
            )
        except OSError as exc:
            terminal_failure = _worker_failure(
                self.command,
                (
                    ManualCleanupWorkerFailureCode.EDIT_STORE_FAILED
                    if stage
                    in {
                        ManualCleanupWorkerStage.READING_EDIT_HEADS,
                        ManualCleanupWorkerStage.OPENING_COMMIT_STORE,
                        ManualCleanupWorkerStage.CLOSING_EDIT_STORE,
                    }
                    else ManualCleanupWorkerFailureCode.PROJECT_LOAD_FAILED
                ),
                stage,
                str(exc) or "Manual cleanup I/O failed.",
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            terminal_failure = _worker_failure(
                self.command,
                ManualCleanupWorkerFailureCode.PROJECT_INVALID,
                stage,
                str(exc) or "Manual cleanup failed.",
                exc,
            )
        finally:
            if store is not None:
                try:
                    stage = ManualCleanupWorkerStage.CLOSING_EDIT_STORE
                    store.close()
                except Exception as exc:  # pragma: no cover - defensive close
                    terminal_receipt = None
                    terminal_failure = _worker_failure(
                        self.command,
                        ManualCleanupWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit store could not be closed safely.",
                        exc,
                    )

        self._publish_terminal(terminal_receipt, terminal_failure, stage)
        self.finished.emit()

    def _forward_progress(self, value: ManualCleanupProgress) -> None:
        if not isinstance(value, ManualCleanupProgress):
            raise TypeError("manual cleanup progress must be typed")
        if value.page_id != self.command.page_id:
            raise ValueError("manual cleanup progress belongs to another page")
        if (
            self.command.mode is ManualCleanupWorkerMode.COMMIT
            and value.stage is ManualCleanupStage.PERSISTING
        ):
            with self._cancel_lock:
                self._cancellation_locked = True
            self.cancellation.emit(
                ManualCleanupCancellationState(
                    page_id=self.command.page_id,
                    enabled=False,
                    message="Commit is being written and can no longer be cancelled.",
                )
            )
        self.progress.emit(value)

    def _publish_terminal(
        self,
        receipt: ManualCleanupReceipt | None,
        failure: ManualCleanupWorkerFailure | None,
        stage: ManualCleanupWorkerStage,
    ) -> None:
        if failure is not None:
            self._discard_failed_preview()
            self.failure.emit(failure)
        elif receipt is None:
            self._discard_failed_preview()
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.INVALID_REQUEST,
                    stage,
                    "Manual cleanup ended without a typed receipt.",
                )
            )
        elif receipt.page_id != self.command.page_id:
            self._discard_failed_preview()
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.INVALID_REQUEST,
                    stage,
                    "Manual cleanup receipt belongs to another page.",
                )
            )
        elif receipt.coverage_target != self.command.coverage_target:
            self._discard_failed_preview()
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.INVALID_REQUEST,
                    stage,
                    "Manual cleanup receipt coverage target changed.",
                )
            )
        elif receipt.status is ManualCleanupStatus.CANCELLED:
            self.cancelled.emit(receipt)
        elif (
            self.command.mode is ManualCleanupWorkerMode.PREVIEW
            and receipt.status is ManualCleanupStatus.PREVIEW_READY
            and receipt.preview_lease is not None
            and receipt.preview_lease.coverage_target
            == self.command.coverage_target
        ):
            self.preview_ready.emit(receipt)
        elif (
            self.command.mode is ManualCleanupWorkerMode.COMMIT
            and receipt.status is ManualCleanupStatus.COMMITTED
            and receipt.commit_receipt is not None
        ):
            self.committed.emit(receipt)
        else:
            self._discard_failed_preview()
            self.failure.emit(
                _worker_failure(
                    self.command,
                    ManualCleanupWorkerFailureCode.INVALID_REQUEST,
                    stage,
                    "Manual cleanup receipt does not match the worker command.",
                )
            )
        with self._cancel_lock:
            self._cancellation_locked = True
        self.cancellation.emit(
            ManualCleanupCancellationState(
                page_id=self.command.page_id,
                enabled=False,
                message="Manual cleanup worker finished.",
            )
        )

    def _discard_failed_preview(self) -> None:
        if self.command.mode is not ManualCleanupWorkerMode.PREVIEW:
            return
        try:
            discard_manual_cleanup_preview(
                self.command.project_path,
                self.command.page_id,
            )
        except Exception:  # pragma: no cover - cleanup must not hide root failure
            # Preview disposal can be retried explicitly by the dialog.
            pass

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True


def _failure_from_service(
    command: ManualCleanupContextCommand | ManualCleanupWorkerCommand,
    stage: ManualCleanupWorkerStage,
    exc: ManualCleanupFailure,
) -> ManualCleanupWorkerFailure:
    try:
        code = ManualCleanupWorkerFailureCode(exc.code.value)
    except ValueError:  # pragma: no cover - future typed service value
        code = ManualCleanupWorkerFailureCode.INVALID_REQUEST
    return _worker_failure(
        command,
        code,
        stage,
        exc.message,
        exc,
        service_code=exc.code,
        service_stage=exc.stage,
    )


def _worker_failure(
    command: ManualCleanupContextCommand | ManualCleanupWorkerCommand,
    code: ManualCleanupWorkerFailureCode,
    stage: ManualCleanupWorkerStage,
    message: str,
    exc: BaseException | None = None,
    *,
    service_code: ManualCleanupFailureCode | None = None,
    service_stage: ManualCleanupStage | None = None,
) -> ManualCleanupWorkerFailure:
    return ManualCleanupWorkerFailure(
        code=code,
        stage=stage,
        project_path=command.project_path,
        page_id=command.page_id,
        message=str(message or "Manual cleanup failed."),
        exception_type=type(exc).__name__ if exc is not None else "",
        service_code=service_code,
        service_stage=service_stage,
    )
