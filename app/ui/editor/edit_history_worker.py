# -*- coding: utf-8 -*-
"""One-shot Qt worker for durable selected-page edit-history controls."""
from __future__ import annotations

import threading
from typing import Any, Mapping
import uuid

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
    EditHistoryCommandReceipt,
    EditHistoryCommandService,
    EditHistoryOperation,
)
from app.project_edits.contracts import EditDomain, EditTargetKind
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.project_edits.render_override_reset_commands import (
    RESETTABLE_RENDER_LAYOUT_FIELDS,
    RESETTABLE_RENDER_STYLE_FIELDS,
)
from app.ui.viewmodels.edit_history_model import (
    EditHistoryCancellationState,
    EditHistoryCancelledReceipt,
    EditHistoryWorkerBusyState,
    EditHistoryWorkerCommand,
    EditHistoryWorkerFailure,
    EditHistoryWorkerFailureCode,
    EditHistoryWorkerReceipt,
    EditHistoryWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.PAGE_NOT_FOUND,
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
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.PAGE_NOT_FOUND,
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


def _base_snapshot(
    project: dict[str, Any],
    *,
    page_id: str,
) -> ProjectEditReadSnapshot:
    _exact_page(project, page_id)
    return ProjectEditReadSnapshot(
        project=project,
        ledger=ProjectEditLedger.from_dict(project["edit_ledger"]),
        page_head_sha256=GENESIS_SHA256,
        global_head_sha256=GENESIS_SHA256,
    )


def _validate_snapshot_target(
    snapshot: ProjectEditReadSnapshot,
    command: EditHistoryWorkerCommand,
) -> None:
    """Fail before sidecar creation for an impossible UI history action."""

    page = project_effective_page(
        snapshot.project,
        snapshot.ledger,
        page_id=command.page_id,
    )
    if page.effective_fingerprint != command.expected_effective_page_fingerprint:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.STALE_EFFECTIVE_PAGE,
            "Effective page state changed; reload selected-page history.",
        )
    target = snapshot.ledger.get(command.target_edit_id)
    if target is None:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.TARGET_EDIT_NOT_FOUND,
            "The selected edit is no longer available.",
        )
    if target.page_id != command.page_id:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.TARGET_EDIT_PAGE_MISMATCH,
            "The selected edit belongs to another page.",
        )
    if target.is_control:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.CONTROL_TARGET_FORBIDDEN,
            "History controls are read-only evidence.",
        )
    if target.target.kind in {EditTargetKind.ARTIFACT, EditTargetKind.EDIT}:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.ARTIFACT_TARGET_FORBIDDEN,
            "Artifact history cannot be used as an edit-history action.",
        )
    supported = target.domain in {
        EditDomain.SOURCE_TEXT,
        EditDomain.TARGET_TEXT,
        EditDomain.REVIEW_METADATA,
    }
    if target.domain is EditDomain.STRUCTURAL:
        supported = bool(
            target.target.kind is EditTargetKind.PARENT
            and target.operation
            in {
                "set_geometry",
                "add_user_parent",
                "split_user_parent",
                "merge_pipeline_parents",
            }
        )
    elif target.domain is EditDomain.CLEANUP:
        supported = target.target.kind is EditTargetKind.PAGE
    elif target.domain in {EditDomain.RENDER_STYLE, EditDomain.RENDER_LAYOUT}:
        fields = target.payload.get("fields")
        supported_fields = (
            frozenset(RESETTABLE_RENDER_STYLE_FIELDS)
            if target.domain is EditDomain.RENDER_STYLE
            else frozenset(RESETTABLE_RENDER_LAYOUT_FIELDS)
        )
        supported = bool(
            (
                target.operation == "set_fields"
                and isinstance(fields, Mapping)
                and len(fields) == 1
                and next(iter(fields), None) in supported_fields
            )
            or (
                target.operation == "restore_automatic"
                and isinstance(fields, tuple)
                and len(fields) == 1
                and fields[0] in supported_fields
            )
        )
    if not supported:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.INVALIDATION_UNRESOLVED,
            "This edit has no exact direction-correct history invalidation.",
        )
    active = target.edit_id in snapshot.ledger.state().active_edit_ids
    if command.operation is EditHistoryOperation.REVOKE and not active:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.ALREADY_REVOKED,
            "The selected edit is already revoked.",
        )
    if command.operation is EditHistoryOperation.REAPPLY and active:
        raise EditHistoryCommandError(
            EditHistoryCommandErrorCode.ALREADY_ACTIVE,
            "The selected edit is already active.",
        )


class EditHistoryWorker(QtCore.QObject):
    """Commit one revoke/reapply control in a dedicated worker thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: EditHistoryWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, EditHistoryWorkerCommand):
            raise TypeError("command must be EditHistoryWorkerCommand")
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
                    EditHistoryWorkerFailureCode.WORKER_REUSED,
                    EditHistoryWorkerStage.LOADING_PROJECT,
                    "EditHistoryWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: EditHistoryCommandReceipt | None = None
        terminal_receipt: EditHistoryWorkerReceipt | None = None
        terminal_failure: EditHistoryWorkerFailure | None = None
        terminal_cancelled: EditHistoryCancelledReceipt | None = None
        stage = EditHistoryWorkerStage.LOADING_PROJECT
        self._emit_cancellation("History action can be cancelled before persistence.")
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = EditHistoryWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            stage = EditHistoryWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact edit revision...")
            if store is None:
                snapshot = _base_snapshot(project, page_id=self.command.page_id)
            else:
                snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
            self._cancel_checkpoint(stage)

            stage = EditHistoryWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the selected history edit...")
            _validate_snapshot_target(snapshot, self.command)
            self._cancel_checkpoint(stage)

            stage = EditHistoryWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed history control...")
            command = EditHistoryCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(snapshot.project),
                page_id=self.command.page_id,
                target_edit_id=self.command.target_edit_id,
                operation=self.command.operation,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = EditHistoryWorkerStage.PERSISTING
            self._emit_cancellation(
                "The history control is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the history control...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    snapshot.project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = EditHistoryCommandService(
                edit_store=store
            ).execute_materialized(
                snapshot=snapshot,
                command=command,
            )

            stage = EditHistoryWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                snapshot.project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project

            stage = EditHistoryWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing selected-page history...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                post_commit_snapshot.ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = EditHistoryWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = EditHistoryCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                target_edit_id=self.command.target_edit_id,
                operation=self.command.operation,
                stage=stage,
            )
        except EditHistoryCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                EditHistoryWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The history action is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "History I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "History action failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = EditHistoryWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        EditHistoryWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("History worker finished.")
        self.busy.emit(
            EditHistoryWorkerBusyState(
                page_id=self.command.page_id,
                target_edit_id=self.command.target_edit_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=EditHistoryWorkerStage.COMPLETE,
                message="History worker finished.",
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
                    EditHistoryWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "History worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: EditHistoryWorkerStage) -> None:
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

    def _emit_busy(self, stage: EditHistoryWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            EditHistoryWorkerBusyState(
                page_id=self.command.page_id,
                target_edit_id=self.command.target_edit_id,
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
            EditHistoryCancellationState(
                page_id=self.command.page_id,
                target_edit_id=self.command.target_edit_id,
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
        stage: EditHistoryWorkerStage,
        exc: EditHistoryCommandError,
    ) -> EditHistoryWorkerFailure:
        code = {
            EditHistoryCommandErrorCode.PROJECT_IDENTITY_MISMATCH: EditHistoryWorkerFailureCode.PROJECT_INVALID,
            EditHistoryCommandErrorCode.STORE_IDENTITY_MISMATCH: EditHistoryWorkerFailureCode.EDIT_STORE_FAILED,
            EditHistoryCommandErrorCode.PAGE_NOT_FOUND: EditHistoryWorkerFailureCode.PAGE_NOT_FOUND,
            EditHistoryCommandErrorCode.TARGET_EDIT_NOT_FOUND: EditHistoryWorkerFailureCode.TARGET_EDIT_NOT_FOUND,
            EditHistoryCommandErrorCode.TARGET_EDIT_PAGE_MISMATCH: EditHistoryWorkerFailureCode.TARGET_EDIT_PAGE_MISMATCH,
            EditHistoryCommandErrorCode.CONTROL_TARGET_FORBIDDEN: EditHistoryWorkerFailureCode.TARGET_FORBIDDEN,
            EditHistoryCommandErrorCode.ARTIFACT_TARGET_FORBIDDEN: EditHistoryWorkerFailureCode.TARGET_FORBIDDEN,
            EditHistoryCommandErrorCode.ALREADY_ACTIVE: EditHistoryWorkerFailureCode.ALREADY_ACTIVE,
            EditHistoryCommandErrorCode.ALREADY_REVOKED: EditHistoryWorkerFailureCode.ALREADY_REVOKED,
            EditHistoryCommandErrorCode.ACTIVE_DEPENDENT_EDIT: EditHistoryWorkerFailureCode.ACTIVE_DEPENDENT_EDIT,
            EditHistoryCommandErrorCode.STALE_EFFECTIVE_PAGE: EditHistoryWorkerFailureCode.SNAPSHOT_STALE,
            EditHistoryCommandErrorCode.STALE_PAGE_HEAD: EditHistoryWorkerFailureCode.SNAPSHOT_STALE,
            EditHistoryCommandErrorCode.STALE_GLOBAL_HEAD: EditHistoryWorkerFailureCode.SNAPSHOT_STALE,
            EditHistoryCommandErrorCode.INVALIDATION_UNRESOLVED: EditHistoryWorkerFailureCode.INVALIDATION_UNRESOLVED,
            EditHistoryCommandErrorCode.DUPLICATE_COMMAND: EditHistoryWorkerFailureCode.DUPLICATE_COMMAND,
            EditHistoryCommandErrorCode.PROJECTION_REJECTED: EditHistoryWorkerFailureCode.PROJECTION_FAILED,
        }.get(exc.code, EditHistoryWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "History action was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: EditHistoryWorkerFailureCode,
        stage: EditHistoryWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: EditHistoryCommandErrorCode | None = None,
        core_receipt: EditHistoryCommandReceipt | None = None,
    ) -> EditHistoryWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            EditHistoryWorkerFailureCode.EDIT_STORE_FAILED,
            EditHistoryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = EditHistoryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return EditHistoryWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            target_edit_id=self.command.target_edit_id,
            operation=self.command.operation,
            message=str(message or "History action failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: EditHistoryWorkerStage,
        committed: bool,
    ) -> EditHistoryWorkerFailureCode:
        if committed:
            return EditHistoryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is EditHistoryWorkerStage.LOADING_PROJECT:
            return EditHistoryWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            EditHistoryWorkerStage.OPENING_EDIT_STORE,
            EditHistoryWorkerStage.READING_SNAPSHOT,
            EditHistoryWorkerStage.PERSISTING,
            EditHistoryWorkerStage.CLOSING_EDIT_STORE,
        }:
            return EditHistoryWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            EditHistoryWorkerStage.PROJECTING,
            EditHistoryWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return EditHistoryWorkerFailureCode.PROJECTION_FAILED
        return EditHistoryWorkerFailureCode.PROJECT_INVALID


__all__ = ["EditHistoryWorker"]
