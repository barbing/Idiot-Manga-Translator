# -*- coding: utf-8 -*-
"""One-shot Qt worker for topology-only Split Parent commands."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import ProjectEditStore
from app.project_edits.commands import (
    SplitUserParentCommand,
    SplitUserParentCommandError,
    SplitUserParentCommandErrorCode,
    SplitUserParentCommandReceipt,
    SplitUserParentCommandService,
    SplitUserParentOperation,
)
from app.project_edits.fingerprints import project_id_for
from app.project_edits.projection import project_effective_page
from app.ui.editor.user_parent_add_worker import _open_project_edit_store
from app.ui.viewmodels.editor_command_model import (
    SplitUserParentCancellationState,
    SplitUserParentCancelledReceipt,
    SplitUserParentWorkerBusyState,
    SplitUserParentWorkerCommand,
    SplitUserParentWorkerFailure,
    SplitUserParentWorkerFailureCode,
    SplitUserParentWorkerReceipt,
    SplitUserParentWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    matches = (
        tuple(
            page
            for page in pages
            if isinstance(page, Mapping)
            and str(page.get("page_id") or "").strip() == page_id
        )
        if isinstance(pages, list)
        else ()
    )
    if len(matches) != 1:
        raise SplitUserParentCommandError(
            SplitUserParentCommandErrorCode.PAGE_NOT_FOUND,
            "The selected project page is unavailable.",
        )
    return matches[0]


def _validate_snapshot_target(snapshot: Any, command: SplitUserParentWorkerCommand) -> None:
    try:
        effective = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=command.page_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitUserParentCommandError(
            SplitUserParentCommandErrorCode.PROJECTION_REJECTED,
            "The selected page could not be projected before Split Parent.",
        ) from exc
    if (
        effective.effective_fingerprint
        != command.expected_effective_page_fingerprint
        or effective.hierarchy.revision_id
        != command.expected_hierarchy_revision_id
        or effective.hierarchy.fingerprint
        != command.expected_hierarchy_fingerprint
    ):
        raise SplitUserParentCommandError(
            SplitUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
            "The selected page or hierarchy changed; reload Split Parent.",
        )


class SplitUserParentWorker(QtCore.QObject):
    """Commit one Split Parent edit entirely inside a dedicated QThread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: SplitUserParentWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, SplitUserParentWorkerCommand):
            raise TypeError("command must be SplitUserParentWorkerCommand")
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
                    SplitUserParentWorkerFailureCode.WORKER_REUSED,
                    SplitUserParentWorkerStage.LOADING_PROJECT,
                    "SplitUserParentWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: SplitUserParentCommandReceipt | None = None
        terminal_receipt: SplitUserParentWorkerReceipt | None = None
        terminal_failure: SplitUserParentWorkerFailure | None = None
        terminal_cancelled: SplitUserParentCancelledReceipt | None = None
        stage = SplitUserParentWorkerStage.LOADING_PROJECT
        self._emit_cancellation("Split Parent can be cancelled before persistence.")
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = SplitUserParentWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            if store is None:
                raise SplitUserParentCommandError(
                    SplitUserParentCommandErrorCode.SOURCE_PARENT_NOT_FOUND,
                    "The selected Add-created user parent is unavailable.",
                )

            stage = SplitUserParentWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact hierarchy revision...")
            snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            self._cancel_checkpoint(stage)

            stage = SplitUserParentWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the selected user parent...")
            _validate_snapshot_target(snapshot, self.command)
            self._cancel_checkpoint(stage)

            stage = SplitUserParentWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed Split Parent edit...")
            command = SplitUserParentCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(snapshot.project),
                page_id=self.command.page_id,
                source_parent_id=self.command.source_parent_id,
                first_parent_id=self.command.first_parent_id,
                first_root_id=self.command.first_root_id,
                second_parent_id=self.command.second_parent_id,
                second_root_id=self.command.second_root_id,
                orientation=self.command.orientation,
                split_offset=self.command.split_offset,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                operation=SplitUserParentOperation.SPLIT,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = SplitUserParentWorkerStage.PERSISTING
            self._emit_cancellation(
                "Split Parent is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the exact child partition...")
            core_receipt = SplitUserParentCommandService(
                edit_store=store
            ).execute_materialized(snapshot=snapshot, command=command)

            stage = SplitUserParentWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit = store.materialize_project_snapshot(
                snapshot.project,
                page_id=self.command.page_id,
            )

            stage = SplitUserParentWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing the split-parent projection...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                post_commit.project,
                post_commit.ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = SplitUserParentWorkerReceipt(
                command_receipt=core_receipt,
                project=post_commit.project,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = SplitUserParentCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                source_parent_id=self.command.source_parent_id,
                orientation=self.command.orientation,
                split_offset=self.command.split_offset,
                stage=stage,
            )
        except SplitUserParentCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                SplitUserParentWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Split Parent failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - fail-closed guard
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Split Parent failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = SplitUserParentWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        SplitUserParentWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Split Parent worker finished.")
        self.busy.emit(
            SplitUserParentWorkerBusyState(
                page_id=self.command.page_id,
                source_parent_id=self.command.source_parent_id,
                orientation=self.command.orientation,
                split_offset=self.command.split_offset,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=SplitUserParentWorkerStage.COMPLETE,
                message="Split Parent worker finished.",
            )
        )
        if terminal_receipt is not None:
            self.receipt.emit(terminal_receipt)
        elif terminal_cancelled is not None:
            self.cancelled.emit(terminal_cancelled)
        elif terminal_failure is not None:
            (self.stale if terminal_failure.stale else self.failure).emit(
                terminal_failure
            )
        else:  # pragma: no cover
            self.failure.emit(
                self._failure(
                    SplitUserParentWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Split Parent worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: SplitUserParentWorkerStage) -> None:
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

    def _emit_busy(self, stage: SplitUserParentWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = not persistence_started and not self._cancel_requested
        self.busy.emit(
            SplitUserParentWorkerBusyState(
                page_id=self.command.page_id,
                source_parent_id=self.command.source_parent_id,
                orientation=self.command.orientation,
                split_offset=self.command.split_offset,
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
            SplitUserParentCancellationState(
                page_id=self.command.page_id,
                source_parent_id=self.command.source_parent_id,
                orientation=self.command.orientation,
                split_offset=self.command.split_offset,
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
        stage: SplitUserParentWorkerStage,
        exc: SplitUserParentCommandError,
    ) -> SplitUserParentWorkerFailure:
        code = {
            SplitUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH: SplitUserParentWorkerFailureCode.PROJECT_INVALID,
            SplitUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH: SplitUserParentWorkerFailureCode.EDIT_STORE_FAILED,
            SplitUserParentCommandErrorCode.PAGE_NOT_FOUND: SplitUserParentWorkerFailureCode.PAGE_NOT_FOUND,
            SplitUserParentCommandErrorCode.SOURCE_PARENT_NOT_FOUND: SplitUserParentWorkerFailureCode.SOURCE_PARENT_NOT_FOUND,
            SplitUserParentCommandErrorCode.SOURCE_PARENT_NOT_STANDALONE: SplitUserParentWorkerFailureCode.SOURCE_PARENT_NOT_STANDALONE,
            SplitUserParentCommandErrorCode.SOURCE_PARENT_EXCLUDED: SplitUserParentWorkerFailureCode.SOURCE_PARENT_EXCLUDED,
            SplitUserParentCommandErrorCode.CANVAS_UNAVAILABLE: SplitUserParentWorkerFailureCode.CANVAS_UNAVAILABLE,
            SplitUserParentCommandErrorCode.INVALID_SPLIT_OFFSET: SplitUserParentWorkerFailureCode.INVALID_SPLIT_OFFSET,
            SplitUserParentCommandErrorCode.IDENTITY_COLLISION: SplitUserParentWorkerFailureCode.IDENTITY_COLLISION,
            SplitUserParentCommandErrorCode.SPLIT_SLOT_CONFLICT: SplitUserParentWorkerFailureCode.SPLIT_SLOT_CONFLICT,
            SplitUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE: SplitUserParentWorkerFailureCode.SNAPSHOT_STALE,
            SplitUserParentCommandErrorCode.STALE_PAGE_HEAD: SplitUserParentWorkerFailureCode.SNAPSHOT_STALE,
            SplitUserParentCommandErrorCode.STALE_GLOBAL_HEAD: SplitUserParentWorkerFailureCode.SNAPSHOT_STALE,
            SplitUserParentCommandErrorCode.INVALIDATION_UNRESOLVED: SplitUserParentWorkerFailureCode.INVALIDATION_UNRESOLVED,
            SplitUserParentCommandErrorCode.DUPLICATE_COMMAND: SplitUserParentWorkerFailureCode.DUPLICATE_COMMAND,
            SplitUserParentCommandErrorCode.PROJECTION_REJECTED: SplitUserParentWorkerFailureCode.PROJECTION_FAILED,
        }.get(exc.code, SplitUserParentWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Split Parent command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: SplitUserParentWorkerFailureCode,
        stage: SplitUserParentWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: SplitUserParentCommandErrorCode | None = None,
        core_receipt: SplitUserParentCommandReceipt | None = None,
    ) -> SplitUserParentWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            SplitUserParentWorkerFailureCode.EDIT_STORE_FAILED,
            SplitUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = SplitUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return SplitUserParentWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            source_parent_id=self.command.source_parent_id,
            orientation=self.command.orientation,
            split_offset=self.command.split_offset,
            message=str(message or "Split Parent failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: SplitUserParentWorkerStage,
        committed: bool,
    ) -> SplitUserParentWorkerFailureCode:
        if committed:
            return SplitUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is SplitUserParentWorkerStage.LOADING_PROJECT:
            return SplitUserParentWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            SplitUserParentWorkerStage.OPENING_EDIT_STORE,
            SplitUserParentWorkerStage.READING_SNAPSHOT,
            SplitUserParentWorkerStage.PERSISTING,
            SplitUserParentWorkerStage.CLOSING_EDIT_STORE,
        }:
            return SplitUserParentWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            SplitUserParentWorkerStage.PROJECTING,
            SplitUserParentWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return SplitUserParentWorkerFailureCode.PROJECTION_FAILED
        return SplitUserParentWorkerFailureCode.PROJECT_INVALID


__all__ = ["SplitUserParentWorker"]
