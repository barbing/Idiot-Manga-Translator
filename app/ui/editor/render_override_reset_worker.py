"""One-shot worker for atomic reversible render-override resets."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    GENESIS_SHA256,
    ProjectEditMultiPageReadSnapshot,
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    canonical_sha256,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.render_override_reset_commands import (
    RenderOverrideResetCommand,
    RenderOverrideResetCommandError,
    RenderOverrideResetCommandErrorCode,
    RenderOverrideResetCommandReceipt,
    RenderOverrideResetCommandService,
    render_override_reset_slots,
)
from app.ui.viewmodels.render_override_reset_model import (
    RenderOverrideResetCancelledReceipt,
    RenderOverrideResetWorkerBusyState,
    RenderOverrideResetWorkerCommand,
    RenderOverrideResetWorkerFailure,
    RenderOverrideResetWorkerFailureCode,
    RenderOverrideResetWorkerReceipt,
    RenderOverrideResetWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _open_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
    *,
    create: bool,
) -> ProjectEditStore | None:
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


def _embedded_snapshot(project: dict[str, Any]) -> ProjectEditMultiPageReadSnapshot:
    page_ids = tuple(
        sorted(
            str(page.get("page_id") or "").strip()
            for page in project.get("pages") or ()
            if isinstance(page, Mapping)
        )
    )
    if not page_ids or any(not page_id for page_id in page_ids):
        raise ValueError("project pages are unavailable")
    return ProjectEditMultiPageReadSnapshot(
        project=project,
        ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
        page_head_sha256=tuple((page_id, GENESIS_SHA256) for page_id in page_ids),
        global_head_sha256=GENESIS_SHA256,
    )


class RenderOverrideResetWorker(QtCore.QObject):
    busy = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: RenderOverrideResetWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, RenderOverrideResetWorkerCommand):
            raise TypeError("command must be RenderOverrideResetWorkerCommand")
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
                    RenderOverrideResetWorkerFailureCode.WORKER_REUSED,
                    RenderOverrideResetWorkerStage.LOADING_PROJECT,
                    "RenderOverrideResetWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: RenderOverrideResetCommandReceipt | None = None
        terminal: RenderOverrideResetWorkerReceipt | RenderOverrideResetCancelledReceipt | None = None
        terminal_failure: RenderOverrideResetWorkerFailure | None = None
        stage = RenderOverrideResetWorkerStage.LOADING_PROJECT
        self._emit_busy(stage, "Loading current project state...")
        try:
            project = load_project_for_editing(self.command.project_path)
            if project_id_for(project) != self.command.project_id:
                raise ValueError("loaded project identity differs from the reset command")
            self._cancel_checkpoint(stage)

            stage = RenderOverrideResetWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            stage = RenderOverrideResetWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading every page and edit head...")
            snapshot = (
                store.materialize_multi_page_snapshot(project)
                if store is not None
                else _embedded_snapshot(project)
            )
            self._cancel_checkpoint(stage)

            stage = RenderOverrideResetWorkerStage.VALIDATING_INVENTORY
            self._emit_busy(stage, "Validating the exact override inventory...")
            if canonical_sha256(snapshot.project) != self.command.expected_project_fingerprint:
                raise RenderOverrideResetCommandError(
                    RenderOverrideResetCommandErrorCode.STALE_PROJECT,
                    "Project state changed after the reset was prepared.",
                )
            slots = render_override_reset_slots(
                snapshot.project,
                snapshot.ledger,
                scope=self.command.scope,
                field_group=self.command.field_group,
                selected_page_id=self.command.selected_page_id,
                selected_parent_id=self.command.selected_parent_id,
            )
            if slots != self.command.expected_slots:
                raise RenderOverrideResetCommandError(
                    RenderOverrideResetCommandErrorCode.STALE_SLOT_INVENTORY,
                    "Render overrides changed after the reset inventory was prepared.",
                )
            self._cancel_checkpoint(stage)

            prepared = RenderOverrideResetCommand(
                command_id=uuid.uuid4().hex,
                project_id=self.command.project_id,
                scope=self.command.scope,
                field_group=self.command.field_group,
                selected_page_id=self.command.selected_page_id,
                selected_parent_id=self.command.selected_parent_id,
                expected_project_fingerprint=self.command.expected_project_fingerprint,
                expected_slots=self.command.expected_slots,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence
            stage = RenderOverrideResetWorkerStage.PERSISTING
            self._emit_busy(stage, "Saving reversible restore records...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    snapshot.project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
                snapshot = store.materialize_multi_page_snapshot(snapshot.project)
            core_receipt = RenderOverrideResetCommandService(
                edit_store=store
            ).execute_materialized(snapshot=snapshot, command=prepared)

            stage = RenderOverrideResetWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing the committed project...")
            post = store.materialize_multi_page_snapshot(snapshot.project)
            stage = RenderOverrideResetWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing the editor projection...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                post.project,
                post.ledger,
                project_path=self.command.project_path,
            )
            terminal = RenderOverrideResetWorkerReceipt(
                command=self.command,
                command_receipt=core_receipt,
                project=post.project,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal = RenderOverrideResetCancelledReceipt(
                command=self.command,
                stage=stage,
            )
        except RenderOverrideResetCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc, core_receipt)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                RenderOverrideResetWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The render-reset command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Render-reset I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Render-reset action failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = RenderOverrideResetWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal = None
                    terminal_failure = self._failure(
                        RenderOverrideResetWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not close safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_busy(
            RenderOverrideResetWorkerStage.COMPLETE,
            "Render-reset worker finished.",
            busy=False,
        )
        if isinstance(terminal, RenderOverrideResetWorkerReceipt):
            self.receipt.emit(terminal)
        elif isinstance(terminal, RenderOverrideResetCancelledReceipt):
            self.cancelled.emit(terminal)
        elif terminal_failure is not None:
            (self.stale if terminal_failure.stale else self.failure).emit(
                terminal_failure
            )
        else:  # pragma: no cover
            self.failure.emit(
                self._failure(
                    RenderOverrideResetWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Render-reset worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True

    def _cancel_checkpoint(self, stage: RenderOverrideResetWorkerStage) -> None:
        with self._cancel_lock:
            if self._cancel_requested and not self._persistence_locked:
                raise _CancelledBeforePersistence(stage.value)

    def _lock_persistence(self) -> bool:
        with self._cancel_lock:
            if self._cancel_requested:
                return False
            self._persistence_locked = True
            return True

    def _emit_busy(
        self,
        stage: RenderOverrideResetWorkerStage,
        message: str,
        *,
        busy: bool = True,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                busy and not persistence_started and not self._cancel_requested
            )
        self.busy.emit(
            RenderOverrideResetWorkerBusyState(
                command=self.command,
                stage=stage,
                busy=busy,
                cancellation_enabled=cancellation_enabled,
                persistence_started=persistence_started,
                message=message,
            )
        )

    def _failure_from_command(
        self,
        stage: RenderOverrideResetWorkerStage,
        exc: RenderOverrideResetCommandError,
        core_receipt: RenderOverrideResetCommandReceipt | None,
    ) -> RenderOverrideResetWorkerFailure:
        code = {
            RenderOverrideResetCommandErrorCode.PAGE_NOT_FOUND: RenderOverrideResetWorkerFailureCode.PAGE_NOT_FOUND,
            RenderOverrideResetCommandErrorCode.PARENT_NOT_FOUND: RenderOverrideResetWorkerFailureCode.PARENT_NOT_FOUND,
            RenderOverrideResetCommandErrorCode.NO_OP: RenderOverrideResetWorkerFailureCode.NO_OP,
            RenderOverrideResetCommandErrorCode.STALE_PROJECT: RenderOverrideResetWorkerFailureCode.SNAPSHOT_STALE,
            RenderOverrideResetCommandErrorCode.STALE_SLOT_INVENTORY: RenderOverrideResetWorkerFailureCode.SNAPSHOT_STALE,
            RenderOverrideResetCommandErrorCode.STALE_PAGE_HEAD: RenderOverrideResetWorkerFailureCode.SNAPSHOT_STALE,
            RenderOverrideResetCommandErrorCode.STALE_GLOBAL_HEAD: RenderOverrideResetWorkerFailureCode.SNAPSHOT_STALE,
            RenderOverrideResetCommandErrorCode.SLOT_CONFLICT: RenderOverrideResetWorkerFailureCode.SLOT_CONFLICT,
            RenderOverrideResetCommandErrorCode.AUTOMATIC_BASE_UNAVAILABLE: RenderOverrideResetWorkerFailureCode.AUTOMATIC_BASE_UNAVAILABLE,
            RenderOverrideResetCommandErrorCode.PROJECTION_REJECTED: RenderOverrideResetWorkerFailureCode.PROJECTION_FAILED,
            RenderOverrideResetCommandErrorCode.PROJECT_IDENTITY_MISMATCH: RenderOverrideResetWorkerFailureCode.PROJECT_INVALID,
            RenderOverrideResetCommandErrorCode.STORE_IDENTITY_MISMATCH: RenderOverrideResetWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(exc.code, RenderOverrideResetWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Render-reset command was rejected.",
            exc,
            core_receipt=core_receipt,
        )

    def _failure(
        self,
        code: RenderOverrideResetWorkerFailureCode,
        stage: RenderOverrideResetWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        core_receipt: RenderOverrideResetCommandReceipt | None = None,
    ) -> RenderOverrideResetWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            RenderOverrideResetWorkerFailureCode.EDIT_STORE_FAILED,
            RenderOverrideResetWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = RenderOverrideResetWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return RenderOverrideResetWorkerFailure(
            code=code,
            stage=stage,
            command=self.command,
            message=str(message or "Render-reset action failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: RenderOverrideResetWorkerStage,
        committed: bool,
    ) -> RenderOverrideResetWorkerFailureCode:
        if committed:
            return RenderOverrideResetWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is RenderOverrideResetWorkerStage.LOADING_PROJECT:
            return RenderOverrideResetWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            RenderOverrideResetWorkerStage.OPENING_EDIT_STORE,
            RenderOverrideResetWorkerStage.READING_SNAPSHOT,
            RenderOverrideResetWorkerStage.PERSISTING,
            RenderOverrideResetWorkerStage.CLOSING_EDIT_STORE,
        }:
            return RenderOverrideResetWorkerFailureCode.EDIT_STORE_FAILED
        if stage is RenderOverrideResetWorkerStage.BUILDING_UI_PROJECTION:
            return RenderOverrideResetWorkerFailureCode.PROJECTION_FAILED
        return RenderOverrideResetWorkerFailureCode.PROJECT_INVALID


__all__ = ["RenderOverrideResetWorker"]
