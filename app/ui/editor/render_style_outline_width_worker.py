# -*- coding: utf-8 -*-
"""One-shot Qt worker for exact selected-parent outline-width commands."""
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
    RenderStyleOutlineWidthCommand,
    RenderStyleOutlineWidthCommandError,
    RenderStyleOutlineWidthCommandErrorCode,
    RenderStyleOutlineWidthCommandReceipt,
    RenderStyleOutlineWidthCommandService,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.render_style_outline_width_model import (
    RenderStyleOutlineWidthCancellationState,
    RenderStyleOutlineWidthCancelledReceipt,
    RenderStyleOutlineWidthWorkerBusyState,
    RenderStyleOutlineWidthWorkerCommand,
    RenderStyleOutlineWidthWorkerFailure,
    RenderStyleOutlineWidthWorkerFailureCode,
    RenderStyleOutlineWidthWorkerReceipt,
    RenderStyleOutlineWidthWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode.PAGE_NOT_FOUND,
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
        raise RenderStyleOutlineWidthCommandError(
            RenderStyleOutlineWidthCommandErrorCode.PAGE_NOT_FOUND,
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


class RenderStyleOutlineWidthWorker(QtCore.QObject):
    """Commit one canonical outline-width command in a dedicated thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: RenderStyleOutlineWidthWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, RenderStyleOutlineWidthWorkerCommand):
            raise TypeError(
                "command must be RenderStyleOutlineWidthWorkerCommand"
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
                    RenderStyleOutlineWidthWorkerFailureCode.WORKER_REUSED,
                    RenderStyleOutlineWidthWorkerStage.LOADING_PROJECT,
                    "RenderStyleOutlineWidthWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: RenderStyleOutlineWidthCommandReceipt | None = None
        terminal_receipt: RenderStyleOutlineWidthWorkerReceipt | None = None
        terminal_failure: RenderStyleOutlineWidthWorkerFailure | None = None
        terminal_cancelled: RenderStyleOutlineWidthCancelledReceipt | None = None
        stage = RenderStyleOutlineWidthWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            "Outline width can be cancelled before persistence."
        )
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = RenderStyleOutlineWidthWorkerStage.OPENING_EDIT_STORE
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
                stage = RenderStyleOutlineWidthWorkerStage.READING_SNAPSHOT
                self._emit_busy(stage, "Reading the exact edit heads...")
                page_head_sha256 = store.page_head(self.command.page_id)
                global_head_sha256 = store.global_head()
            self._cancel_checkpoint(stage)

            if store is None:
                stage = RenderStyleOutlineWidthWorkerStage.PROJECTING
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
                    raise RenderStyleOutlineWidthCommandError(
                        RenderStyleOutlineWidthCommandErrorCode.STALE_EFFECTIVE_PAGE,
                        "Effective page state changed; reload the selected parent.",
                    )
                parent_matches = tuple(
                    parent
                    for parent in snapshot.parents
                    if parent.parent_id == self.command.parent_id
                )
                if len(parent_matches) != 1:
                    raise RenderStyleOutlineWidthCommandError(
                        RenderStyleOutlineWidthCommandErrorCode.PARENT_NOT_FOUND,
                        "The selected parent is no longer available.",
                    )
                self._cancel_checkpoint(stage)

            stage = RenderStyleOutlineWidthWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed outline-width edit...")
            command = RenderStyleOutlineWidthCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                outline_width=self.command.outline_width,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=page_head_sha256,
                expected_global_head_sha256=global_head_sha256,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = RenderStyleOutlineWidthWorkerStage.PERSISTING
            self._emit_cancellation(
                "The outline-width edit is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the outline-width edit...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = RenderStyleOutlineWidthCommandService(
                edit_store=store
            ).execute(
                project=project,
                command=command,
            )

            stage = RenderStyleOutlineWidthWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project
            materialized_ledger = post_commit_snapshot.ledger

            stage = RenderStyleOutlineWidthWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                materialized_ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = RenderStyleOutlineWidthWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = RenderStyleOutlineWidthCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=self.command.operation,
                outline_width=self.command.outline_width,
                stage=stage,
            )
        except RenderStyleOutlineWidthCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                RenderStyleOutlineWidthWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The outline-width command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Outline-width I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Outline-width edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = RenderStyleOutlineWidthWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        RenderStyleOutlineWidthWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Outline-width worker finished.")
        self.busy.emit(
            RenderStyleOutlineWidthWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=RenderStyleOutlineWidthWorkerStage.COMPLETE,
                message="Outline-width worker finished.",
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
                    RenderStyleOutlineWidthWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Outline-width worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(
        self,
        stage: RenderStyleOutlineWidthWorkerStage,
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
        stage: RenderStyleOutlineWidthWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            RenderStyleOutlineWidthWorkerBusyState(
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
            RenderStyleOutlineWidthCancellationState(
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
        stage: RenderStyleOutlineWidthWorkerStage,
        exc: RenderStyleOutlineWidthCommandError,
    ) -> RenderStyleOutlineWidthWorkerFailure:
        code = {
            RenderStyleOutlineWidthCommandErrorCode.PAGE_NOT_FOUND: RenderStyleOutlineWidthWorkerFailureCode.PAGE_NOT_FOUND,
            RenderStyleOutlineWidthCommandErrorCode.PARENT_NOT_FOUND: RenderStyleOutlineWidthWorkerFailureCode.PARENT_NOT_FOUND,
            RenderStyleOutlineWidthCommandErrorCode.PARENT_EXCLUDED: RenderStyleOutlineWidthWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStyleOutlineWidthCommandErrorCode.PARENT_NOT_RENDER_REQUIRED: RenderStyleOutlineWidthWorkerFailureCode.PARENT_UNAVAILABLE,
            RenderStyleOutlineWidthCommandErrorCode.AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE: RenderStyleOutlineWidthWorkerFailureCode.AUTOMATIC_OUTLINE_WIDTH_UNAVAILABLE,
            RenderStyleOutlineWidthCommandErrorCode.NO_OP: RenderStyleOutlineWidthWorkerFailureCode.NO_OP,
            RenderStyleOutlineWidthCommandErrorCode.STALE_EFFECTIVE_PAGE: RenderStyleOutlineWidthWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleOutlineWidthCommandErrorCode.STALE_PAGE_HEAD: RenderStyleOutlineWidthWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleOutlineWidthCommandErrorCode.STALE_GLOBAL_HEAD: RenderStyleOutlineWidthWorkerFailureCode.SNAPSHOT_STALE,
            RenderStyleOutlineWidthCommandErrorCode.OUTLINE_WIDTH_SLOT_CONFLICT: RenderStyleOutlineWidthWorkerFailureCode.OUTLINE_WIDTH_SLOT_CONFLICT,
            RenderStyleOutlineWidthCommandErrorCode.DUPLICATE_COMMAND: RenderStyleOutlineWidthWorkerFailureCode.DUPLICATE_COMMAND,
            RenderStyleOutlineWidthCommandErrorCode.PROJECTION_REJECTED: RenderStyleOutlineWidthWorkerFailureCode.PROJECTION_FAILED,
            RenderStyleOutlineWidthCommandErrorCode.PROJECT_IDENTITY_MISMATCH: RenderStyleOutlineWidthWorkerFailureCode.PROJECT_INVALID,
            RenderStyleOutlineWidthCommandErrorCode.STORE_IDENTITY_MISMATCH: RenderStyleOutlineWidthWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(
            exc.code,
            RenderStyleOutlineWidthWorkerFailureCode.COMMAND_REJECTED,
        )
        return self._failure(
            code,
            stage,
            str(exc) or "Outline-width command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: RenderStyleOutlineWidthWorkerFailureCode,
        stage: RenderStyleOutlineWidthWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: RenderStyleOutlineWidthCommandErrorCode | None = None,
        core_receipt: RenderStyleOutlineWidthCommandReceipt | None = None,
    ) -> RenderStyleOutlineWidthWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            RenderStyleOutlineWidthWorkerFailureCode.EDIT_STORE_FAILED,
            RenderStyleOutlineWidthWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = (
                RenderStyleOutlineWidthWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        return RenderStyleOutlineWidthWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            operation=self.command.operation,
            outline_width=self.command.outline_width,
            message=str(message or "Outline-width edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: RenderStyleOutlineWidthWorkerStage,
        committed: bool,
    ) -> RenderStyleOutlineWidthWorkerFailureCode:
        if committed:
            return (
                RenderStyleOutlineWidthWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
            )
        if stage is RenderStyleOutlineWidthWorkerStage.LOADING_PROJECT:
            return RenderStyleOutlineWidthWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            RenderStyleOutlineWidthWorkerStage.OPENING_EDIT_STORE,
            RenderStyleOutlineWidthWorkerStage.READING_SNAPSHOT,
            RenderStyleOutlineWidthWorkerStage.PERSISTING,
            RenderStyleOutlineWidthWorkerStage.CLOSING_EDIT_STORE,
        }:
            return RenderStyleOutlineWidthWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            RenderStyleOutlineWidthWorkerStage.PROJECTING,
            RenderStyleOutlineWidthWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return RenderStyleOutlineWidthWorkerFailureCode.PROJECTION_FAILED
        return RenderStyleOutlineWidthWorkerFailureCode.PROJECT_INVALID


__all__ = ["RenderStyleOutlineWidthWorker"]
