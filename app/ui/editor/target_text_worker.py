# -*- coding: utf-8 -*-
"""One-shot Qt worker for selected-parent target-text commands."""
from __future__ import annotations

import threading
from typing import Any, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    GENESIS_SHA256,
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.commands import (
    TargetTextCommand,
    TargetTextCommandError,
    TargetTextCommandErrorCode,
    TargetTextCommandReceipt,
    TargetTextCommandService,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.editor_command_model import (
    TargetTextCancellationState,
    TargetTextCancelledReceipt,
    TargetTextWorkerBusyState,
    TargetTextWorkerCommand,
    TargetTextWorkerFailure,
    TargetTextWorkerFailureCode,
    TargetTextWorkerReceipt,
    TargetTextWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PAGE_NOT_FOUND,
            "Project pages are unavailable.",
        )
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if len(matches) != 1:
        reason = "missing" if not matches else "duplicated"
        raise TargetTextCommandError(
            TargetTextCommandErrorCode.PAGE_NOT_FOUND,
            f"Project page identity is {reason}: {page_id}",
        )
    return matches[0]


def _open_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
    *,
    create: bool,
) -> ProjectEditStore | None:
    """Open the exact sidecar; only the persistence path may create it."""

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


class TargetTextWorker(QtCore.QObject):
    """Commit one target-text command after being moved to a ``QThread``.

    SQLite is opened, read, used, and closed only inside :meth:`run`.
    Cancellation is direct and thread-safe until the persistence boundary.
    Once persistence starts, cancellation is locked and no rollback is claimed.
    """

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: TargetTextWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, TargetTextWorkerCommand):
            raise TypeError("command must be TargetTextWorkerCommand")
        self.command = command
        self._run_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._has_run = False
        self._cancel_requested = False
        self._persistence_locked = False

    def request_cancel(self) -> bool:
        """Request cancellation without waiting for the worker event loop."""

        with self._cancel_lock:
            if self._persistence_locked or self._cancel_requested:
                return False
            self._cancel_requested = True
            return True

    @QtCore.Slot()
    def run(self) -> None:
        if not self._claim_run():
            failure = self._failure(
                TargetTextWorkerFailureCode.WORKER_REUSED,
                TargetTextWorkerStage.LOADING_PROJECT,
                "TargetTextWorker instances are one-shot.",
            )
            self.failure.emit(failure)
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: TargetTextCommandReceipt | None = None
        terminal_receipt: TargetTextWorkerReceipt | None = None
        terminal_failure: TargetTextWorkerFailure | None = None
        terminal_cancelled: TargetTextCancelledReceipt | None = None
        stage = TargetTextWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            "Target-text update can be cancelled before persistence."
        )
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = TargetTextWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            if store is None:
                ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
                page_head_sha256 = GENESIS_SHA256
                global_head_sha256 = GENESIS_SHA256
            else:
                stage = TargetTextWorkerStage.READING_SNAPSHOT
                self._emit_busy(stage, "Reading the exact edit revision...")
                read_snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
                project = read_snapshot.project
                ledger = read_snapshot.ledger
                page_head_sha256 = read_snapshot.page_head_sha256
                global_head_sha256 = read_snapshot.global_head_sha256
            self._cancel_checkpoint(stage)

            stage = TargetTextWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the selected parent...")
            snapshot = project_effective_page(
                project,
                ledger,
                page_id=self.command.page_id,
            )
            if (
                snapshot.effective_fingerprint
                != self.command.expected_effective_page_fingerprint
            ):
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.STALE_EFFECTIVE_PAGE,
                    "Effective page state changed; reload the selected parent.",
                )
            parent_matches = tuple(
                parent
                for parent in snapshot.parents
                if parent.parent_id == self.command.parent_id
            )
            if len(parent_matches) != 1:
                raise TargetTextCommandError(
                    TargetTextCommandErrorCode.PARENT_NOT_FOUND,
                    "The selected parent is no longer available.",
                )
            self._cancel_checkpoint(stage)

            stage = TargetTextWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed target-text edit...")
            command = TargetTextCommand(
                command_id=self.command.command_id,
                project_id=project_id_for(project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                text=self.command.text,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=page_head_sha256,
                expected_global_head_sha256=global_head_sha256,
                revision_base=self.command.revision_base,
                source_evidence_base=self.command.source_evidence_base,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = TargetTextWorkerStage.PERSISTING
            self._emit_cancellation(
                "The edit is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the target-text edit...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    project,
                    create=True,
                )
                if store is None:  # pragma: no cover - create is mandatory
                    raise RuntimeError("project edit store was not created")
            core_receipt = TargetTextCommandService(edit_store=store).execute(
                project=project,
                command=command,
            )

            stage = TargetTextWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project
            materialized_ledger = post_commit_snapshot.ledger

            stage = TargetTextWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            # Import after worker initialization to avoid the shell package's
            # eager MainWindow import forming a startup cycle.
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                materialized_ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = TargetTextWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = TargetTextCancelledReceipt(
                command_id=self.command.command_id,
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                stage=stage,
            )
        except TargetTextCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                TargetTextWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The target-text command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Target-text edit I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Target-text edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = TargetTextWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover - sqlite close defense
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        TargetTextWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Target-text worker finished.")
        self.busy.emit(
            TargetTextWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=TargetTextWorkerStage.COMPLETE,
                message="Target-text worker finished.",
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
        else:  # pragma: no cover - defensive terminal contract
            self.failure.emit(
                self._failure(
                    TargetTextWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Target-text worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: TargetTextWorkerStage) -> None:
        with self._cancel_lock:
            cancelled = self._cancel_requested and not self._persistence_locked
        if cancelled:
            raise _CancelledBeforePersistence(stage.value)

    def _lock_persistence(self) -> bool:
        with self._cancel_lock:
            if self._cancel_requested:
                return False
            self._persistence_locked = True
            return True

    def _emit_busy(self, stage: TargetTextWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            TargetTextWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=True,
                cancellation_enabled=cancellation_enabled,
                persistence_started=persistence_started,
                stage=stage,
                message=message,
            )
        )

    def _emit_cancellation(self, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            enabled = not persistence_started and not self._cancel_requested
        self.cancellation.emit(
            TargetTextCancellationState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                enabled=enabled,
                persistence_started=persistence_started,
                message=message,
            )
        )

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True

    def _failure_from_command(
        self,
        stage: TargetTextWorkerStage,
        exc: TargetTextCommandError,
    ) -> TargetTextWorkerFailure:
        code = {
            TargetTextCommandErrorCode.PAGE_NOT_FOUND: TargetTextWorkerFailureCode.PAGE_NOT_FOUND,
            TargetTextCommandErrorCode.PARENT_NOT_FOUND: TargetTextWorkerFailureCode.PARENT_NOT_FOUND,
            TargetTextCommandErrorCode.STALE_EFFECTIVE_PAGE: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.STALE_PAGE_HEAD: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.STALE_GLOBAL_HEAD: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.REVISION_ID_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.REVISION_SELECTION_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.REVISION_ARTIFACT_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.REVISION_SOURCE_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.REVISION_HIERARCHY_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.PARENT_LINEAGE_MISMATCH: TargetTextWorkerFailureCode.SNAPSHOT_STALE,
            TargetTextCommandErrorCode.TARGET_SLOT_CONFLICT: TargetTextWorkerFailureCode.TARGET_SLOT_CONFLICT,
            TargetTextCommandErrorCode.DUPLICATE_COMMAND: TargetTextWorkerFailureCode.DUPLICATE_COMMAND,
            TargetTextCommandErrorCode.PROJECTION_REJECTED: TargetTextWorkerFailureCode.PROJECTION_FAILED,
            TargetTextCommandErrorCode.PROJECT_IDENTITY_MISMATCH: TargetTextWorkerFailureCode.PROJECT_INVALID,
            TargetTextCommandErrorCode.STORE_IDENTITY_MISMATCH: TargetTextWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(exc.code, TargetTextWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Target-text command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: TargetTextWorkerFailureCode,
        stage: TargetTextWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: TargetTextCommandErrorCode | None = None,
        core_receipt: TargetTextCommandReceipt | None = None,
    ) -> TargetTextWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            TargetTextWorkerFailureCode.EDIT_STORE_FAILED,
            TargetTextWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = TargetTextWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return TargetTextWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            command_id=self.command.command_id,
            message=str(message or "Target-text edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: TargetTextWorkerStage,
        committed: bool,
    ) -> TargetTextWorkerFailureCode:
        if committed:
            return TargetTextWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is TargetTextWorkerStage.LOADING_PROJECT:
            return TargetTextWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            TargetTextWorkerStage.OPENING_EDIT_STORE,
            TargetTextWorkerStage.READING_SNAPSHOT,
            TargetTextWorkerStage.PERSISTING,
            TargetTextWorkerStage.CLOSING_EDIT_STORE,
        }:
            return TargetTextWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            TargetTextWorkerStage.PROJECTING,
            TargetTextWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return TargetTextWorkerFailureCode.PROJECTION_FAILED
        return TargetTextWorkerFailureCode.PROJECT_INVALID


__all__ = ["TargetTextWorker"]
