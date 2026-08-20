"""One-shot worker for project glossary edits, History, import, and export."""
from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    GENESIS_SHA256,
    ProjectEditReadSnapshot,
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.commands import (
    EditHistoryCommand,
    EditHistoryCommandError,
    EditHistoryCommandErrorCode,
    EditHistoryCommandService,
    EditHistoryOperation,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.glossary_commands import (
    GlossaryCommand,
    GlossaryCommandError,
    GlossaryCommandErrorCode,
    GlossaryCommandService,
    GlossaryOperation,
    project_glossary_snapshot,
)
from app.project_edits.glossary_io import (
    load_glossary_entries,
    save_glossary_entries,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.glossary_model import (
    GlossaryCancelledReceipt,
    GlossaryExportReceipt,
    GlossaryWorkerBusyState,
    GlossaryWorkerCommand,
    GlossaryWorkerFailure,
    GlossaryWorkerFailureCode,
    GlossaryWorkerOperation,
    GlossaryWorkerReceipt,
    GlossaryWorkerStage,
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


def _base_snapshot(project: dict[str, Any], *, page_id: str) -> ProjectEditReadSnapshot:
    page_ids = tuple(
        str(page.get("page_id") or "").strip()
        for page in project.get("pages") or ()
        if isinstance(page, Mapping)
    )
    if not page_ids or page_ids[0] != page_id:
        raise ValueError("the canonical glossary anchor page is unavailable")
    return ProjectEditReadSnapshot(
        project=project,
        ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
        page_head_sha256=GENESIS_SHA256,
        global_head_sha256=GENESIS_SHA256,
    )


class GlossaryWorker(QtCore.QObject):
    busy = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: GlossaryWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, GlossaryWorkerCommand):
            raise TypeError("command must be a GlossaryWorkerCommand")
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
                    GlossaryWorkerFailureCode.WORKER_REUSED,
                    GlossaryWorkerStage.LOADING_PROJECT,
                    "GlossaryWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: object | None = None
        terminal: object | None = None
        terminal_failure: GlossaryWorkerFailure | None = None
        stage = GlossaryWorkerStage.LOADING_PROJECT
        self._emit_busy(stage, "Loading the current project...")
        try:
            project = load_project_for_editing(self.command.project_path)
            if project_id_for(project) != self.command.project_id:
                raise ValueError("loaded project identity differs from the glossary command")
            self._cancel_checkpoint(stage)

            stage = GlossaryWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(self.command.project_path, project, create=False)
            stage = GlossaryWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact project glossary...")
            snapshot = (
                store.materialize_project_snapshot(project, page_id=self.command.anchor_page_id)
                if store is not None
                else _base_snapshot(project, page_id=self.command.anchor_page_id)
            )
            glossary = project_glossary_snapshot(snapshot.project, snapshot.ledger)
            anchor_page = project_effective_page(
                snapshot.project,
                snapshot.ledger,
                page_id=self.command.anchor_page_id,
            )
            if glossary.anchor_page_id != self.command.anchor_page_id:
                raise ValueError("canonical glossary anchor page changed")
            if glossary.fingerprint != self.command.expected_glossary_fingerprint:
                raise GlossaryCommandError(
                    GlossaryCommandErrorCode.STALE_GLOSSARY,
                    "The effective project glossary changed; reload Settings.",
                )
            if anchor_page.effective_fingerprint != self.command.expected_anchor_page_fingerprint:
                raise GlossaryCommandError(
                    GlossaryCommandErrorCode.STALE_GLOSSARY,
                    "The glossary anchor page changed; reload Settings.",
                )
            self._cancel_checkpoint(stage)

            if self.command.operation is GlossaryWorkerOperation.EXPORT_FILE:
                if not self._lock_persistence():
                    raise _CancelledBeforePersistence
                stage = GlossaryWorkerStage.WRITING_FILE
                self._emit_busy(stage, "Exporting the current project glossary...")
                exported = save_glossary_entries(self.command.file_path, glossary.entries)
                terminal = GlossaryExportReceipt(
                    command=self.command,
                    exported_path=exported,
                    entry_count=len(glossary.entries),
                )
            else:
                entries = self.command.entries
                if self.command.operation is GlossaryWorkerOperation.IMPORT_FILE:
                    stage = GlossaryWorkerStage.READING_FILE
                    self._emit_busy(stage, "Reading and validating glossary import...")
                    entries = load_glossary_entries(self.command.file_path)
                    self._cancel_checkpoint(stage)
                stage = GlossaryWorkerStage.PREPARING_COMMAND
                self._emit_busy(stage, "Preparing the typed project glossary action...")
                if self.command.operation in {
                    GlossaryWorkerOperation.HISTORY_REVOKE,
                    GlossaryWorkerOperation.HISTORY_REAPPLY,
                }:
                    if store is None:
                        raise EditHistoryCommandError(
                            EditHistoryCommandErrorCode.TARGET_EDIT_NOT_FOUND,
                            "The selected glossary History edit is unavailable.",
                        )
                    prepared: object = EditHistoryCommand(
                        command_id=uuid.uuid4().hex,
                        project_id=self.command.project_id,
                        page_id=self.command.anchor_page_id,
                        target_edit_id=self.command.history_edit_id,
                        operation=(
                            EditHistoryOperation.REVOKE
                            if self.command.operation is GlossaryWorkerOperation.HISTORY_REVOKE
                            else EditHistoryOperation.REAPPLY
                        ),
                        expected_effective_page_fingerprint=anchor_page.effective_fingerprint,
                        expected_page_head_sha256=snapshot.page_head_sha256,
                        expected_global_head_sha256=snapshot.global_head_sha256,
                    )
                else:
                    operation = {
                        GlossaryWorkerOperation.SET_ENTRY: GlossaryOperation.SET_ENTRY,
                        GlossaryWorkerOperation.REMOVE_ENTRY: GlossaryOperation.REMOVE_ENTRY,
                        GlossaryWorkerOperation.IMPORT_FILE: GlossaryOperation.IMPORT_ENTRIES,
                    }[self.command.operation]
                    prepared = GlossaryCommand(
                        command_id=uuid.uuid4().hex,
                        project_id=self.command.project_id,
                        anchor_page_id=self.command.anchor_page_id,
                        operation=operation,
                        entries=entries,
                        entry_ids=self.command.entry_ids,
                        expected_glossary_fingerprint=glossary.fingerprint,
                        expected_page_head_sha256=snapshot.page_head_sha256,
                        expected_global_head_sha256=snapshot.global_head_sha256,
                    )
                if not self._lock_persistence():
                    raise _CancelledBeforePersistence
                stage = GlossaryWorkerStage.PERSISTING
                self._emit_busy(stage, "Saving the project glossary action...")
                if store is None:
                    store = _open_project_edit_store(self.command.project_path, snapshot.project, create=True)
                    if store is None:  # pragma: no cover
                        raise RuntimeError("project edit store was not created")
                if isinstance(prepared, GlossaryCommand):
                    core_receipt = GlossaryCommandService(edit_store=store).execute_materialized(snapshot=snapshot, command=prepared)
                else:
                    core_receipt = EditHistoryCommandService(edit_store=store).execute_materialized(snapshot=snapshot, command=prepared)

                stage = GlossaryWorkerStage.MATERIALIZING_PROJECT
                self._emit_busy(stage, "Refreshing the committed project...")
                post = store.materialize_project_snapshot(snapshot.project, page_id=self.command.anchor_page_id)
                stage = GlossaryWorkerStage.BUILDING_UI_PROJECTION
                self._emit_busy(stage, "Refreshing the glossary presentation...")
                from app.ui.shell.project_projection import project_ui_projection

                projection = project_ui_projection(post.project, post.ledger, project_path=self.command.project_path)
                terminal = GlossaryWorkerReceipt(
                    command=self.command,
                    command_receipt=core_receipt,
                    project=post.project,
                    projection=projection,
                )
        except _CancelledBeforePersistence:
            terminal = GlossaryCancelledReceipt(command=self.command, stage=stage)
        except GlossaryCommandError as exc:
            terminal_failure = self._failure_from_glossary(stage, exc, core_receipt)
        except EditHistoryCommandError as exc:
            terminal_failure = self._failure(
                GlossaryWorkerFailureCode.HISTORY_REJECTED,
                stage,
                str(exc) or "Glossary History action was rejected.",
                exc,
                core_receipt=core_receipt,
            )
        except FileNotFoundError as exc:
            if stage is GlossaryWorkerStage.READING_FILE:
                file_code = GlossaryWorkerFailureCode.FILE_INVALID
            elif stage is GlossaryWorkerStage.WRITING_FILE:
                file_code = GlossaryWorkerFailureCode.FILE_WRITE_FAILED
            else:
                file_code = GlossaryWorkerFailureCode.PROJECT_LOAD_FAILED
            terminal_failure = self._failure(
                file_code,
                stage,
                "The selected file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                GlossaryWorkerFailureCode.FILE_INVALID if stage is GlossaryWorkerStage.READING_FILE else self._stage_failure_code(stage, core_receipt is not None),
                stage,
                str(exc) or "The project glossary action is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                GlossaryWorkerFailureCode.FILE_WRITE_FAILED if stage is GlossaryWorkerStage.WRITING_FILE else self._stage_failure_code(stage, core_receipt is not None),
                stage,
                str(exc) or "Glossary I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._stage_failure_code(stage, core_receipt is not None),
                stage,
                str(exc) or "Glossary action failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = GlossaryWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal = None
                    terminal_failure = self._failure(
                        GlossaryWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not close safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_busy(GlossaryWorkerStage.COMPLETE, "Glossary worker finished.", busy=False)
        if isinstance(terminal, GlossaryCancelledReceipt):
            self.cancelled.emit(terminal)
        elif isinstance(terminal, (GlossaryWorkerReceipt, GlossaryExportReceipt)):
            self.receipt.emit(terminal)
        elif terminal_failure is not None:
            (self.stale if terminal_failure.stale else self.failure).emit(terminal_failure)
        else:  # pragma: no cover
            self.failure.emit(self._failure(GlossaryWorkerFailureCode.COMMAND_REJECTED, stage, "Glossary worker ended without a typed result."))
        self.finished.emit()

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True

    def _cancel_checkpoint(self, stage: GlossaryWorkerStage) -> None:
        with self._cancel_lock:
            if self._cancel_requested and not self._persistence_locked:
                raise _CancelledBeforePersistence(stage.value)

    def _lock_persistence(self) -> bool:
        with self._cancel_lock:
            if self._cancel_requested:
                return False
            self._persistence_locked = True
            return True

    def _emit_busy(self, stage: GlossaryWorkerStage, message: str, *, busy: bool = True) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = busy and not persistence_started and not self._cancel_requested
        self.busy.emit(
            GlossaryWorkerBusyState(
                stage=stage,
                busy=busy,
                cancellation_enabled=cancellation_enabled,
                persistence_started=persistence_started,
                message=message,
            )
        )

    def _failure_from_glossary(self, stage: GlossaryWorkerStage, exc: GlossaryCommandError, core_receipt: object | None) -> GlossaryWorkerFailure:
        code = {
            GlossaryCommandErrorCode.STALE_GLOSSARY: GlossaryWorkerFailureCode.SNAPSHOT_STALE,
            GlossaryCommandErrorCode.STALE_PAGE_HEAD: GlossaryWorkerFailureCode.SNAPSHOT_STALE,
            GlossaryCommandErrorCode.STALE_GLOBAL_HEAD: GlossaryWorkerFailureCode.SNAPSHOT_STALE,
            GlossaryCommandErrorCode.DUPLICATE_TERM: GlossaryWorkerFailureCode.DUPLICATE_TERM,
            GlossaryCommandErrorCode.STORE_IDENTITY_MISMATCH: GlossaryWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(exc.code, GlossaryWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(code, stage, str(exc), exc, core_receipt=core_receipt)

    def _failure(self, code: GlossaryWorkerFailureCode, stage: GlossaryWorkerStage, message: str, exc: BaseException | None = None, *, core_receipt: object | None = None) -> GlossaryWorkerFailure:
        committed = core_receipt is not None
        if committed:
            code = GlossaryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return GlossaryWorkerFailure(
            code=code,
            stage=stage,
            command=self.command,
            message=str(message or "Glossary action failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _stage_failure_code(stage: GlossaryWorkerStage, committed: bool) -> GlossaryWorkerFailureCode:
        if committed:
            return GlossaryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is GlossaryWorkerStage.LOADING_PROJECT:
            return GlossaryWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {GlossaryWorkerStage.OPENING_EDIT_STORE, GlossaryWorkerStage.READING_SNAPSHOT, GlossaryWorkerStage.PERSISTING, GlossaryWorkerStage.CLOSING_EDIT_STORE}:
            return GlossaryWorkerFailureCode.EDIT_STORE_FAILED
        return GlossaryWorkerFailureCode.PROJECT_INVALID


__all__ = ["GlossaryWorker"]
