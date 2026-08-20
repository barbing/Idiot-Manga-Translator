# -*- coding: utf-8 -*-
"""One-shot Qt worker for exact selected-parent preferred-size commands."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    GENESIS_SHA256,
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.commands import (
    RenderStylePreferredSizeCommand,
    RenderStylePreferredSizeCommandError,
    RenderStylePreferredSizeCommandErrorCode,
    RenderStylePreferredSizeCommandReceipt,
    RenderStylePreferredSizeCommandService,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.render_style_preferred_size_model import (
    RenderStylePreferredSizeCancellationState,
    RenderStylePreferredSizeCancelledReceipt,
    RenderStylePreferredSizeWorkerBusyState,
    RenderStylePreferredSizeWorkerCommand,
    RenderStylePreferredSizeWorkerFailure,
    RenderStylePreferredSizeWorkerFailureCode,
    RenderStylePreferredSizeWorkerReceipt,
    RenderStylePreferredSizeWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode.PAGE_NOT_FOUND,
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
        raise RenderStylePreferredSizeCommandError(
            RenderStylePreferredSizeCommandErrorCode.PAGE_NOT_FOUND,
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


class RenderStylePreferredSizeWorker(QtCore.QObject):
    """Commit one canonical preferred-size command in a dedicated thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: RenderStylePreferredSizeWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, RenderStylePreferredSizeWorkerCommand):
            raise TypeError(
                "command must be RenderStylePreferredSizeWorkerCommand"
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
                    RenderStylePreferredSizeWorkerFailureCode.WORKER_REUSED,
                    RenderStylePreferredSizeWorkerStage.LOADING_PROJECT,
                    "RenderStylePreferredSizeWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: RenderStylePreferredSizeCommandReceipt | None = None
        terminal_receipt: RenderStylePreferredSizeWorkerReceipt | None = None
        terminal_failure: RenderStylePreferredSizeWorkerFailure | None = None
        terminal_cancelled: RenderStylePreferredSizeCancelledReceipt | None = None
        stage = RenderStylePreferredSizeWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            "Preferred size can be cancelled before persistence."
        )
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = RenderStylePreferredSizeWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            if store is None:
                page_head_sha256 = GENESIS_SHA256
                global_head_sha256 = GENESIS_SHA256
                ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
            else:
                stage = RenderStylePreferredSizeWorkerStage.READING_SNAPSHOT
                self._emit_busy(stage, "Reading the exact edit heads...")
                page_head_sha256 = store.page_head(self.command.page_id)
                global_head_sha256 = store.global_head()
            self._cancel_checkpoint(stage)

            if store is None:
                stage = RenderStylePreferredSizeWorkerStage.PROJECTING
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
                    raise RenderStylePreferredSizeCommandError(
                        RenderStylePreferredSizeCommandErrorCode.STALE_EFFECTIVE_PAGE,
                        "Effective page state changed; reload the selected parent.",
                    )
                parent_matches = tuple(
                    parent
                    for parent in snapshot.parents
                    if parent.parent_id == self.command.parent_id
                )
                if len(parent_matches) != 1:
                    raise RenderStylePreferredSizeCommandError(
                        RenderStylePreferredSizeCommandErrorCode.PARENT_NOT_FOUND,
                        "The selected parent is no longer available.",
                    )
                self._cancel_checkpoint(stage)

            stage = RenderStylePreferredSizeWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed preferred-size edit...")
            command = RenderStylePreferredSizeCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                preferred_size=self.command.preferred_size,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=page_head_sha256,
                expected_global_head_sha256=global_head_sha256,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = RenderStylePreferredSizeWorkerStage.PERSISTING
            self._emit_cancellation(
                "The preferred-size edit is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the preferred-size edit...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = RenderStylePreferredSizeCommandService(
                edit_store=store
            ).execute(
                project=project,
                command=command,
            )

            stage = RenderStylePreferredSizeWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project
            materialized_ledger = post_commit_snapshot.ledger

            stage = RenderStylePreferredSizeWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                materialized_ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = RenderStylePreferredSizeWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = RenderStylePreferredSizeCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                preferred_size=self.command.preferred_size,
                stage=stage,
            )
        except RenderStylePreferredSizeCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                RenderStylePreferredSizeWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The preferred-size command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Preferred-size I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Preferred-size edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = RenderStylePreferredSizeWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        RenderStylePreferredSizeWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Preferred-size worker finished.")
        self.busy.emit(
            RenderStylePreferredSizeWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=RenderStylePreferredSizeWorkerStage.COMPLETE,
                message="Preferred-size worker finished.",
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
                    RenderStylePreferredSizeWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Preferred-size worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(
        self,
        stage: RenderStylePreferredSizeWorkerStage,
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
        stage: RenderStylePreferredSizeWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            RenderStylePreferredSizeWorkerBusyState(
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
            RenderStylePreferredSizeCancellationState(
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
        stage: RenderStylePreferredSizeWorkerStage,
        exc: RenderStylePreferredSizeCommandError,
    ) -> RenderStylePreferredSizeWorkerFailure:
        code = {
            RenderStylePreferredSizeCommandErrorCode.PAGE_NOT_FOUND: RenderStylePreferredSizeWorkerFailureCode.PAGE_NOT_FOUND,
            RenderStylePreferredSizeCommandErrorCode.PARENT_NOT_FOUND: RenderStylePreferredSizeWorkerFailureCode.PARENT_NOT_FOUND,
            RenderStylePreferredSizeCommandErrorCode.PARENT_EXCLUDED: RenderStylePreferredSizeWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStylePreferredSizeCommandErrorCode.PARENT_NOT_RENDER_REQUIRED: RenderStylePreferredSizeWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStylePreferredSizeCommandErrorCode.AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE: RenderStylePreferredSizeWorkerFailureCode.AUTOMATIC_PREFERRED_SIZE_UNAVAILABLE,
            RenderStylePreferredSizeCommandErrorCode.NO_OP: RenderStylePreferredSizeWorkerFailureCode.NO_OP,
            RenderStylePreferredSizeCommandErrorCode.STALE_EFFECTIVE_PAGE: RenderStylePreferredSizeWorkerFailureCode.SNAPSHOT_STALE,
            RenderStylePreferredSizeCommandErrorCode.STALE_PAGE_HEAD: RenderStylePreferredSizeWorkerFailureCode.SNAPSHOT_STALE,
            RenderStylePreferredSizeCommandErrorCode.STALE_GLOBAL_HEAD: RenderStylePreferredSizeWorkerFailureCode.SNAPSHOT_STALE,
            RenderStylePreferredSizeCommandErrorCode.PREFERRED_SIZE_SLOT_CONFLICT: RenderStylePreferredSizeWorkerFailureCode.PREFERRED_SIZE_SLOT_CONFLICT,
            RenderStylePreferredSizeCommandErrorCode.DUPLICATE_COMMAND: RenderStylePreferredSizeWorkerFailureCode.DUPLICATE_COMMAND,
            RenderStylePreferredSizeCommandErrorCode.PROJECTION_REJECTED: RenderStylePreferredSizeWorkerFailureCode.PROJECTION_FAILED,
            RenderStylePreferredSizeCommandErrorCode.PROJECT_IDENTITY_MISMATCH: RenderStylePreferredSizeWorkerFailureCode.PROJECT_INVALID,
            RenderStylePreferredSizeCommandErrorCode.STORE_IDENTITY_MISMATCH: RenderStylePreferredSizeWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(
            exc.code,
            RenderStylePreferredSizeWorkerFailureCode.COMMAND_REJECTED,
        )
        return self._failure(
            code,
            stage,
            str(exc) or "Preferred-size command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: RenderStylePreferredSizeWorkerFailureCode,
        stage: RenderStylePreferredSizeWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: RenderStylePreferredSizeCommandErrorCode | None = None,
        core_receipt: RenderStylePreferredSizeCommandReceipt | None = None,
    ) -> RenderStylePreferredSizeWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            RenderStylePreferredSizeWorkerFailureCode.EDIT_STORE_FAILED,
            RenderStylePreferredSizeWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = (
                RenderStylePreferredSizeWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        return RenderStylePreferredSizeWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            operation=self.command.operation,
            preferred_size=self.command.preferred_size,
            message=str(message or "Preferred-size edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: RenderStylePreferredSizeWorkerStage,
        committed: bool,
    ) -> RenderStylePreferredSizeWorkerFailureCode:
        if committed:
            return (
                RenderStylePreferredSizeWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        if stage is RenderStylePreferredSizeWorkerStage.LOADING_PROJECT:
            return RenderStylePreferredSizeWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            RenderStylePreferredSizeWorkerStage.OPENING_EDIT_STORE,
            RenderStylePreferredSizeWorkerStage.READING_SNAPSHOT,
            RenderStylePreferredSizeWorkerStage.PERSISTING,
            RenderStylePreferredSizeWorkerStage.CLOSING_EDIT_STORE,
        }:
            return RenderStylePreferredSizeWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            RenderStylePreferredSizeWorkerStage.PROJECTING,
            RenderStylePreferredSizeWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return RenderStylePreferredSizeWorkerFailureCode.PROJECTION_FAILED
        return RenderStylePreferredSizeWorkerFailureCode.PROJECT_INVALID


__all__ = ["RenderStylePreferredSizeWorker"]
