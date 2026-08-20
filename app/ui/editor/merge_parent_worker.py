# -*- coding: utf-8 -*-
"""One-shot Qt worker for pipeline-backed Merge Parent commands."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import ProjectEditStore
from app.project_edits.commands import (
    MergePipelineParentsCommand,
    MergePipelineParentsCommandError,
    MergePipelineParentsCommandErrorCode,
    MergePipelineParentsCommandReceipt,
    MergePipelineParentsCommandService,
    MergePipelineParentsOperation,
)
from app.project_edits.fingerprints import project_id_for
from app.project_edits.projection import project_effective_page
from app.ui.editor.user_parent_add_worker import _open_project_edit_store
from app.ui.viewmodels.editor_command_model import (
    MergePipelineParentsCancellationState,
    MergePipelineParentsCancelledReceipt,
    MergePipelineParentsWorkerBusyState,
    MergePipelineParentsWorkerCommand,
    MergePipelineParentsWorkerFailure,
    MergePipelineParentsWorkerFailureCode,
    MergePipelineParentsWorkerReceipt,
    MergePipelineParentsWorkerStage,
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
        raise MergePipelineParentsCommandError(
            MergePipelineParentsCommandErrorCode.PAGE_NOT_FOUND,
            "The selected project page is unavailable.",
        )
    return matches[0]


def _validate_snapshot_target(snapshot: Any, command: MergePipelineParentsWorkerCommand) -> None:
    try:
        effective = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=command.page_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MergePipelineParentsCommandError(
            MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED,
            "The selected page could not be projected before Merge Parent.",
        ) from exc
    if (
        effective.effective_fingerprint != command.expected_effective_page_fingerprint
        or effective.hierarchy.revision_id != command.expected_hierarchy_revision_id
        or effective.hierarchy.fingerprint != command.expected_hierarchy_fingerprint
    ):
        raise MergePipelineParentsCommandError(
            MergePipelineParentsCommandErrorCode.STALE_EFFECTIVE_PAGE,
            "The selected page or hierarchy changed; reload Merge Parent.",
        )


class MergePipelineParentsWorker(QtCore.QObject):
    """Commit one Merge Parent edit entirely inside a dedicated QThread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: MergePipelineParentsWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, MergePipelineParentsWorkerCommand):
            raise TypeError("command must be MergePipelineParentsWorkerCommand")
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
                    MergePipelineParentsWorkerFailureCode.WORKER_REUSED,
                    MergePipelineParentsWorkerStage.LOADING_PROJECT,
                    "MergePipelineParentsWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return
        store: ProjectEditStore | None = None
        core_receipt: MergePipelineParentsCommandReceipt | None = None
        terminal_receipt: MergePipelineParentsWorkerReceipt | None = None
        terminal_failure: MergePipelineParentsWorkerFailure | None = None
        terminal_cancelled: MergePipelineParentsCancelledReceipt | None = None
        stage = MergePipelineParentsWorkerStage.LOADING_PROJECT
        self._emit_cancellation("Merge Parent can be cancelled before persistence.")
        self._emit_busy(stage, "Loading current project state...")
        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)
            stage = MergePipelineParentsWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(self.command.project_path, project, create=False)
            if store is None:
                raise MergePipelineParentsCommandError(
                    MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_FOUND,
                    "The selected pipeline parents are unavailable.",
                )
            stage = MergePipelineParentsWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact hierarchy revision...")
            snapshot = store.materialize_project_snapshot(project, page_id=self.command.page_id)
            self._cancel_checkpoint(stage)
            stage = MergePipelineParentsWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the two pipeline parents...")
            _validate_snapshot_target(snapshot, self.command)
            self._cancel_checkpoint(stage)
            stage = MergePipelineParentsWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed Merge Parent edit...")
            command = MergePipelineParentsCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(snapshot.project),
                page_id=self.command.page_id,
                source_parent_ids=self.command.source_parent_ids,
                merged_parent_id=self.command.merged_parent_id,
                merged_root_id=self.command.merged_root_id,
                expected_effective_page_fingerprint=self.command.expected_effective_page_fingerprint,
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                operation=MergePipelineParentsOperation.MERGE,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence
            stage = MergePipelineParentsWorkerStage.PERSISTING
            self._emit_cancellation("Merge Parent is being persisted and can no longer be cancelled.")
            self._emit_busy(stage, "Saving the merged pipeline parent...")
            core_receipt = MergePipelineParentsCommandService(edit_store=store).execute_materialized(
                snapshot=snapshot,
                command=command,
            )
            stage = MergePipelineParentsWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit = store.materialize_project_snapshot(snapshot.project, page_id=self.command.page_id)
            stage = MergePipelineParentsWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing the merged-parent projection...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                post_commit.project,
                post_commit.ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = MergePipelineParentsWorkerReceipt(
                command_receipt=core_receipt,
                project=post_commit.project,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = MergePipelineParentsCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                source_parent_ids=self.command.source_parent_ids,
                stage=stage,
            )
        except MergePipelineParentsCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                MergePipelineParentsWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - fail-closed guard
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Merge Parent failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = MergePipelineParentsWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        MergePipelineParentsWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )
        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Merge Parent worker finished.")
        self.busy.emit(
            MergePipelineParentsWorkerBusyState(
                page_id=self.command.page_id,
                source_parent_ids=self.command.source_parent_ids,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=MergePipelineParentsWorkerStage.COMPLETE,
                message="Merge Parent worker finished.",
            )
        )
        if terminal_receipt is not None:
            self.receipt.emit(terminal_receipt)
        elif terminal_cancelled is not None:
            self.cancelled.emit(terminal_cancelled)
        elif terminal_failure is not None:
            (self.stale if terminal_failure.stale else self.failure).emit(terminal_failure)
        else:  # pragma: no cover
            self.failure.emit(
                self._failure(
                    MergePipelineParentsWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Merge Parent worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: MergePipelineParentsWorkerStage) -> None:
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

    def _emit_busy(self, stage: MergePipelineParentsWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = not persistence_started and not self._cancel_requested
        self.busy.emit(
            MergePipelineParentsWorkerBusyState(
                page_id=self.command.page_id,
                source_parent_ids=self.command.source_parent_ids,
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
            MergePipelineParentsCancellationState(
                page_id=self.command.page_id,
                source_parent_ids=self.command.source_parent_ids,
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
        stage: MergePipelineParentsWorkerStage,
        exc: MergePipelineParentsCommandError,
    ) -> MergePipelineParentsWorkerFailure:
        code = {
            MergePipelineParentsCommandErrorCode.PROJECT_IDENTITY_MISMATCH: MergePipelineParentsWorkerFailureCode.PROJECT_INVALID,
            MergePipelineParentsCommandErrorCode.STORE_IDENTITY_MISMATCH: MergePipelineParentsWorkerFailureCode.EDIT_STORE_FAILED,
            MergePipelineParentsCommandErrorCode.PAGE_NOT_FOUND: MergePipelineParentsWorkerFailureCode.PAGE_NOT_FOUND,
            MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_FOUND: MergePipelineParentsWorkerFailureCode.SOURCE_PARENT_NOT_FOUND,
            MergePipelineParentsCommandErrorCode.SOURCE_PARENT_NOT_AUTOMATIC: MergePipelineParentsWorkerFailureCode.SOURCE_PARENT_NOT_AUTOMATIC,
            MergePipelineParentsCommandErrorCode.SOURCE_PARENT_EXCLUDED: MergePipelineParentsWorkerFailureCode.SOURCE_PARENT_EXCLUDED,
            MergePipelineParentsCommandErrorCode.SOURCE_PARENT_EDITED: MergePipelineParentsWorkerFailureCode.SOURCE_PARENT_EDITED,
            MergePipelineParentsCommandErrorCode.SOURCE_EVIDENCE_UNAVAILABLE: MergePipelineParentsWorkerFailureCode.SOURCE_EVIDENCE_UNAVAILABLE,
            MergePipelineParentsCommandErrorCode.ROLE_MISMATCH: MergePipelineParentsWorkerFailureCode.ROLE_MISMATCH,
            MergePipelineParentsCommandErrorCode.SOURCES_NOT_CONSECUTIVE: MergePipelineParentsWorkerFailureCode.SOURCES_NOT_CONSECUTIVE,
            MergePipelineParentsCommandErrorCode.CANVAS_UNAVAILABLE: MergePipelineParentsWorkerFailureCode.CANVAS_UNAVAILABLE,
            MergePipelineParentsCommandErrorCode.IDENTITY_COLLISION: MergePipelineParentsWorkerFailureCode.IDENTITY_COLLISION,
            MergePipelineParentsCommandErrorCode.MERGE_SLOT_CONFLICT: MergePipelineParentsWorkerFailureCode.MERGE_SLOT_CONFLICT,
            MergePipelineParentsCommandErrorCode.STALE_EFFECTIVE_PAGE: MergePipelineParentsWorkerFailureCode.SNAPSHOT_STALE,
            MergePipelineParentsCommandErrorCode.STALE_PAGE_HEAD: MergePipelineParentsWorkerFailureCode.SNAPSHOT_STALE,
            MergePipelineParentsCommandErrorCode.STALE_GLOBAL_HEAD: MergePipelineParentsWorkerFailureCode.SNAPSHOT_STALE,
            MergePipelineParentsCommandErrorCode.INVALIDATION_UNRESOLVED: MergePipelineParentsWorkerFailureCode.INVALIDATION_UNRESOLVED,
            MergePipelineParentsCommandErrorCode.DUPLICATE_COMMAND: MergePipelineParentsWorkerFailureCode.DUPLICATE_COMMAND,
            MergePipelineParentsCommandErrorCode.PROJECTION_REJECTED: MergePipelineParentsWorkerFailureCode.PROJECTION_FAILED,
        }.get(exc.code, MergePipelineParentsWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(code, stage, str(exc) or "Merge Parent command was rejected.", exc, command_error_code=exc.code)

    def _failure(
        self,
        code: MergePipelineParentsWorkerFailureCode,
        stage: MergePipelineParentsWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: MergePipelineParentsCommandErrorCode | None = None,
        core_receipt: MergePipelineParentsCommandReceipt | None = None,
    ) -> MergePipelineParentsWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            MergePipelineParentsWorkerFailureCode.EDIT_STORE_FAILED,
            MergePipelineParentsWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = MergePipelineParentsWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return MergePipelineParentsWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            source_parent_ids=self.command.source_parent_ids,
            message=str(message or "Merge Parent failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: MergePipelineParentsWorkerStage,
        committed: bool,
    ) -> MergePipelineParentsWorkerFailureCode:
        if committed:
            return MergePipelineParentsWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is MergePipelineParentsWorkerStage.LOADING_PROJECT:
            return MergePipelineParentsWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            MergePipelineParentsWorkerStage.OPENING_EDIT_STORE,
            MergePipelineParentsWorkerStage.READING_SNAPSHOT,
            MergePipelineParentsWorkerStage.PERSISTING,
            MergePipelineParentsWorkerStage.CLOSING_EDIT_STORE,
        }:
            return MergePipelineParentsWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            MergePipelineParentsWorkerStage.PROJECTING,
            MergePipelineParentsWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return MergePipelineParentsWorkerFailureCode.PROJECTION_FAILED
        return MergePipelineParentsWorkerFailureCode.PROJECT_INVALID


__all__ = ["MergePipelineParentsWorker"]
