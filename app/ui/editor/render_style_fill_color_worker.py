# -*- coding: utf-8 -*-
"""One-shot Qt worker for exact selected-parent fill_color commands."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    ProjectEditReadSnapshot,
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.commands import (
    RenderStyleFillColorCommand,
    RenderStyleFillColorCommandError,
    RenderStyleFillColorCommandErrorCode,
    RenderStyleFillColorCommandReceipt,
    RenderStyleFillColorCommandService,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.editor_command_model import (
    RenderStyleFillColorCancellationState,
    RenderStyleFillColorCancelledReceipt,
    RenderStyleFillColorWorkerBusyState,
    RenderStyleFillColorWorkerCommand,
    RenderStyleFillColorWorkerFailure,
    RenderStyleFillColorWorkerFailureCode,
    RenderStyleFillColorWorkerReceipt,
    RenderStyleFillColorWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PAGE_NOT_FOUND,
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
        raise RenderStyleFillColorCommandError(
            RenderStyleFillColorCommandErrorCode.PAGE_NOT_FOUND,
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


class RenderStyleFillColorWorker(QtCore.QObject):
    """Commit one canonical fill_color command in a dedicated thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: RenderStyleFillColorWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, RenderStyleFillColorWorkerCommand):
            raise TypeError(
                "command must be RenderStyleFillColorWorkerCommand"
            )
        self.command = command
        self._run_lock = threading.Lock()
        self._cancel_lock = threading.Lock()
        self._has_run = False
        self._cancel_requested = False
        self._persistence_locked = False

    def request_cancel(self) -> bool:
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
                    RenderStyleFillColorWorkerFailureCode.WORKER_REUSED,
                    RenderStyleFillColorWorkerStage.LOADING_PROJECT,
                    "RenderStyleFillColorWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: RenderStyleFillColorCommandReceipt | None = None
        terminal_receipt: RenderStyleFillColorWorkerReceipt | None = None
        terminal_failure: RenderStyleFillColorWorkerFailure | None = None
        terminal_cancelled: RenderStyleFillColorCancelledReceipt | None = None
        stage = RenderStyleFillColorWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            "Fill color can be cancelled before persistence."
        )
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = RenderStyleFillColorWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            precommit_snapshot: ProjectEditReadSnapshot | None = None
            if store is None:
                validation_project = project
                validation_ledger = ProjectEditLedger.from_dict(
                    project["edit_ledger"]
                )
            else:
                stage = RenderStyleFillColorWorkerStage.READING_SNAPSHOT
                self._emit_busy(stage, "Reading the exact project snapshot...")
                precommit_snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
                validation_project = precommit_snapshot.project
                validation_ledger = precommit_snapshot.ledger
            self._cancel_checkpoint(stage)

            stage = RenderStyleFillColorWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the selected parent...")
            effective_page = project_effective_page(
                validation_project,
                validation_ledger,
                page_id=self.command.page_id,
            )
            if (
                effective_page.effective_fingerprint
                != self.command.expected_effective_page_fingerprint
            ):
                raise RenderStyleFillColorCommandError(
                    RenderStyleFillColorCommandErrorCode.STALE_EFFECTIVE_PAGE,
                    "Effective page state changed; reload the selected parent.",
                )
            parent_matches = tuple(
                parent
                for parent in effective_page.parents
                if parent.parent_id == self.command.parent_id
            )
            if len(parent_matches) != 1:
                raise RenderStyleFillColorCommandError(
                    RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND,
                    "The selected parent is no longer available.",
            )
            self._cancel_checkpoint(stage)

            if not self._lock_persistence():
                raise _CancelledBeforePersistence
            self._emit_cancellation(
                "The fill_color edit is being persisted and can no longer be cancelled."
            )
            if store is None:
                stage = RenderStyleFillColorWorkerStage.OPENING_EDIT_STORE
                self._emit_busy(stage, "Creating the project edit journal...")
                store = _open_project_edit_store(
                    self.command.project_path,
                    project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
                stage = RenderStyleFillColorWorkerStage.READING_SNAPSHOT
                self._emit_busy(stage, "Reading the exact project snapshot...")
                precommit_snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
            if precommit_snapshot is None:  # pragma: no cover - branch invariant
                raise RuntimeError("precommit project snapshot is unavailable")

            stage = RenderStyleFillColorWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed fill_color edit...")
            command = RenderStyleFillColorCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(precommit_snapshot.project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                fill_color=self.command.fill_color,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=(
                    precommit_snapshot.page_head_sha256
                ),
                expected_global_head_sha256=(
                    precommit_snapshot.global_head_sha256
                ),
            )

            stage = RenderStyleFillColorWorkerStage.PERSISTING
            self._emit_busy(stage, "Saving the fill_color edit...")
            core_receipt = RenderStyleFillColorCommandService(
                edit_store=store
            ).execute_materialized(
                snapshot=precommit_snapshot,
                command=command,
            )

            stage = RenderStyleFillColorWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project
            materialized_ledger = post_commit_snapshot.ledger

            stage = RenderStyleFillColorWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                materialized_ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = RenderStyleFillColorWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = RenderStyleFillColorCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                fill_color=self.command.fill_color,
                stage=stage,
            )
        except RenderStyleFillColorCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                RenderStyleFillColorWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The fill_color command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Fill color I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Fill color edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = RenderStyleFillColorWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        RenderStyleFillColorWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Fill color worker finished.")
        self.busy.emit(
            RenderStyleFillColorWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=RenderStyleFillColorWorkerStage.COMPLETE,
                message="Fill color worker finished.",
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
                    RenderStyleFillColorWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Fill color worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(
        self,
        stage: RenderStyleFillColorWorkerStage,
    ) -> None:
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

    def _emit_busy(
        self,
        stage: RenderStyleFillColorWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            RenderStyleFillColorWorkerBusyState(
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
            RenderStyleFillColorCancellationState(
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
        stage: RenderStyleFillColorWorkerStage,
        exc: RenderStyleFillColorCommandError,
    ) -> RenderStyleFillColorWorkerFailure:
        code = {
            RenderStyleFillColorCommandErrorCode.PAGE_NOT_FOUND: RenderStyleFillColorWorkerFailureCode.PAGE_NOT_FOUND,
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_FOUND: RenderStyleFillColorWorkerFailureCode.PARENT_NOT_FOUND,
            RenderStyleFillColorCommandErrorCode.PARENT_EXCLUDED: RenderStyleFillColorWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStyleFillColorCommandErrorCode.PARENT_NOT_RENDER_REQUIRED: RenderStyleFillColorWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStyleFillColorCommandErrorCode.AUTOMATIC_FILL_COLOR_UNAVAILABLE: RenderStyleFillColorWorkerFailureCode.AUTOMATIC_FILL_COLOR_UNAVAILABLE,
            RenderStyleFillColorCommandErrorCode.NO_OP: RenderStyleFillColorWorkerFailureCode.NO_OP,
            RenderStyleFillColorCommandErrorCode.STALE_EFFECTIVE_PAGE: RenderStyleFillColorWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleFillColorCommandErrorCode.STALE_PAGE_HEAD: RenderStyleFillColorWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleFillColorCommandErrorCode.STALE_GLOBAL_HEAD: RenderStyleFillColorWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleFillColorCommandErrorCode.FILL_COLOR_SLOT_CONFLICT: RenderStyleFillColorWorkerFailureCode.FILL_COLOR_SLOT_CONFLICT,
            RenderStyleFillColorCommandErrorCode.DUPLICATE_COMMAND: RenderStyleFillColorWorkerFailureCode.DUPLICATE_COMMAND,
            RenderStyleFillColorCommandErrorCode.PROJECTION_REJECTED: RenderStyleFillColorWorkerFailureCode.PROJECTION_FAILED,
            RenderStyleFillColorCommandErrorCode.PROJECT_IDENTITY_MISMATCH: RenderStyleFillColorWorkerFailureCode.PROJECT_INVALID,
            RenderStyleFillColorCommandErrorCode.STORE_IDENTITY_MISMATCH: RenderStyleFillColorWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(
            exc.code,
            RenderStyleFillColorWorkerFailureCode.COMMAND_REJECTED,
        )
        return self._failure(
            code,
            stage,
            str(exc) or "Fill color command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: RenderStyleFillColorWorkerFailureCode,
        stage: RenderStyleFillColorWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: RenderStyleFillColorCommandErrorCode | None = None,
        core_receipt: RenderStyleFillColorCommandReceipt | None = None,
    ) -> RenderStyleFillColorWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            RenderStyleFillColorWorkerFailureCode.EDIT_STORE_FAILED,
            RenderStyleFillColorWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = (
                RenderStyleFillColorWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        return RenderStyleFillColorWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            operation=self.command.operation,
            fill_color=self.command.fill_color,
            message=str(message or "Fill color edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: RenderStyleFillColorWorkerStage,
        committed: bool,
    ) -> RenderStyleFillColorWorkerFailureCode:
        if committed:
            return (
                RenderStyleFillColorWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        if stage is RenderStyleFillColorWorkerStage.LOADING_PROJECT:
            return RenderStyleFillColorWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            RenderStyleFillColorWorkerStage.OPENING_EDIT_STORE,
            RenderStyleFillColorWorkerStage.READING_SNAPSHOT,
            RenderStyleFillColorWorkerStage.PERSISTING,
            RenderStyleFillColorWorkerStage.CLOSING_EDIT_STORE,
        }:
            return RenderStyleFillColorWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            RenderStyleFillColorWorkerStage.PROJECTING,
            RenderStyleFillColorWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return RenderStyleFillColorWorkerFailureCode.PROJECTION_FAILED
        return RenderStyleFillColorWorkerFailureCode.PROJECT_INVALID


__all__ = ["RenderStyleFillColorWorker"]
