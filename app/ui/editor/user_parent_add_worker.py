# -*- coding: utf-8 -*-
"""One-shot Qt worker for a standalone topology-only user parent."""
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
    AddUserParentCommand,
    AddUserParentCommandError,
    AddUserParentCommandErrorCode,
    AddUserParentCommandReceipt,
    AddUserParentCommandService,
    AddUserParentOperation,
    ParentGeometryCommandError,
    page_canvas_size_for_project_page,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.editor_command_model import (
    AddUserParentCancellationState,
    AddUserParentCancelledReceipt,
    AddUserParentWorkerBusyState,
    AddUserParentWorkerCommand,
    AddUserParentWorkerFailure,
    AddUserParentWorkerFailureCode,
    AddUserParentWorkerReceipt,
    AddUserParentWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.PAGE_NOT_FOUND,
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
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.PAGE_NOT_FOUND,
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
    command: AddUserParentWorkerCommand,
) -> None:
    """Reject stale or invalid GUI intent before a sidecar may be created."""

    page_mapping = _exact_page(snapshot.project, command.page_id)
    try:
        canvas_size = page_canvas_size_for_project_page(
            page_mapping,
            project_path=command.project_path,
        )
    except ParentGeometryCommandError as exc:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.CANVAS_UNAVAILABLE,
            "Page canvas dimensions are unavailable for Add Parent.",
        ) from exc
    x, y, width, height = command.workflow_area_bbox
    if x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.WORKFLOW_AREA_OUT_OF_BOUNDS,
            "The workflow area must remain fully inside the page canvas.",
        )
    try:
        effective = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=command.page_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.PROJECTION_REJECTED,
            "The effective page could not be projected before Add Parent.",
        ) from exc
    if (
        effective.effective_fingerprint
        != command.expected_effective_page_fingerprint
        or effective.hierarchy.revision_id
        != command.expected_hierarchy_revision_id
        or effective.hierarchy.fingerprint
        != command.expected_hierarchy_fingerprint
    ):
        raise AddUserParentCommandError(
            AddUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE,
            "Effective page or hierarchy state changed; reload Add Parent.",
        )


class AddUserParentWorker(QtCore.QObject):
    """Commit one pending user-parent topology edit in a dedicated thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: AddUserParentWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, AddUserParentWorkerCommand):
            raise TypeError("command must be AddUserParentWorkerCommand")
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
                    AddUserParentWorkerFailureCode.WORKER_REUSED,
                    AddUserParentWorkerStage.LOADING_PROJECT,
                    "AddUserParentWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: AddUserParentCommandReceipt | None = None
        terminal_receipt: AddUserParentWorkerReceipt | None = None
        terminal_failure: AddUserParentWorkerFailure | None = None
        terminal_cancelled: AddUserParentCancelledReceipt | None = None
        stage = AddUserParentWorkerStage.LOADING_PROJECT
        self._emit_cancellation("Add Parent can be cancelled before persistence.")
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = AddUserParentWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            stage = AddUserParentWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact hierarchy revision...")
            if store is None:
                snapshot = _base_snapshot(project, page_id=self.command.page_id)
            else:
                snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
            self._cancel_checkpoint(stage)

            stage = AddUserParentWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the pending user-parent draft...")
            _validate_snapshot_target(snapshot, self.command)
            self._cancel_checkpoint(stage)

            stage = AddUserParentWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed Add Parent edit...")
            command = AddUserParentCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(snapshot.project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                root_id=self.command.root_id,
                role=self.command.role,
                workflow_area_bbox=self.command.workflow_area_bbox,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                operation=AddUserParentOperation.ADD,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = AddUserParentWorkerStage.PERSISTING
            self._emit_cancellation(
                "Add Parent is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the pending user parent...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    snapshot.project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = AddUserParentCommandService(
                edit_store=store
            ).execute_materialized(
                snapshot=snapshot,
                command=command,
            )

            stage = AddUserParentWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                snapshot.project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project

            stage = AddUserParentWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing the pending parent projection...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                post_commit_snapshot.ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = AddUserParentWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = AddUserParentCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                root_id=self.command.root_id,
                role=self.command.role,
                workflow_area_bbox=self.command.workflow_area_bbox,
                stage=stage,
            )
        except AddUserParentCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                AddUserParentWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The Add Parent command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Add Parent I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover - fail-closed worker guard
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Add Parent failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = AddUserParentWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        AddUserParentWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Add Parent worker finished.")
        self.busy.emit(
            AddUserParentWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                root_id=self.command.root_id,
                role=self.command.role,
                workflow_area_bbox=self.command.workflow_area_bbox,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=AddUserParentWorkerStage.COMPLETE,
                message="Add Parent worker finished.",
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
                    AddUserParentWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Add Parent worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: AddUserParentWorkerStage) -> None:
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

    def _emit_busy(self, stage: AddUserParentWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            AddUserParentWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                root_id=self.command.root_id,
                role=self.command.role,
                workflow_area_bbox=self.command.workflow_area_bbox,
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
            AddUserParentCancellationState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                root_id=self.command.root_id,
                role=self.command.role,
                workflow_area_bbox=self.command.workflow_area_bbox,
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
        stage: AddUserParentWorkerStage,
        exc: AddUserParentCommandError,
    ) -> AddUserParentWorkerFailure:
        code = {
            AddUserParentCommandErrorCode.PROJECT_IDENTITY_MISMATCH: AddUserParentWorkerFailureCode.PROJECT_INVALID,
            AddUserParentCommandErrorCode.STORE_IDENTITY_MISMATCH: AddUserParentWorkerFailureCode.EDIT_STORE_FAILED,
            AddUserParentCommandErrorCode.PAGE_NOT_FOUND: AddUserParentWorkerFailureCode.PAGE_NOT_FOUND,
            AddUserParentCommandErrorCode.CANVAS_UNAVAILABLE: AddUserParentWorkerFailureCode.CANVAS_UNAVAILABLE,
            AddUserParentCommandErrorCode.INVALID_WORKFLOW_AREA: AddUserParentWorkerFailureCode.INVALID_WORKFLOW_AREA,
            AddUserParentCommandErrorCode.WORKFLOW_AREA_OUT_OF_BOUNDS: AddUserParentWorkerFailureCode.WORKFLOW_AREA_OUT_OF_BOUNDS,
            AddUserParentCommandErrorCode.IDENTITY_COLLISION: AddUserParentWorkerFailureCode.IDENTITY_COLLISION,
            AddUserParentCommandErrorCode.STALE_EFFECTIVE_PAGE: AddUserParentWorkerFailureCode.SNAPSHOT_STALE,
            AddUserParentCommandErrorCode.STALE_PAGE_HEAD: AddUserParentWorkerFailureCode.SNAPSHOT_STALE,
            AddUserParentCommandErrorCode.STALE_GLOBAL_HEAD: AddUserParentWorkerFailureCode.SNAPSHOT_STALE,
            AddUserParentCommandErrorCode.INVALIDATION_UNRESOLVED: AddUserParentWorkerFailureCode.INVALIDATION_UNRESOLVED,
            AddUserParentCommandErrorCode.DUPLICATE_COMMAND: AddUserParentWorkerFailureCode.DUPLICATE_COMMAND,
            AddUserParentCommandErrorCode.PROJECTION_REJECTED: AddUserParentWorkerFailureCode.PROJECTION_FAILED,
        }.get(exc.code, AddUserParentWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Add Parent command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: AddUserParentWorkerFailureCode,
        stage: AddUserParentWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: AddUserParentCommandErrorCode | None = None,
        core_receipt: AddUserParentCommandReceipt | None = None,
    ) -> AddUserParentWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            AddUserParentWorkerFailureCode.EDIT_STORE_FAILED,
            AddUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = AddUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return AddUserParentWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            root_id=self.command.root_id,
            role=self.command.role,
            workflow_area_bbox=self.command.workflow_area_bbox,
            message=str(message or "Add Parent failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: AddUserParentWorkerStage,
        committed: bool,
    ) -> AddUserParentWorkerFailureCode:
        if committed:
            return AddUserParentWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is AddUserParentWorkerStage.LOADING_PROJECT:
            return AddUserParentWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            AddUserParentWorkerStage.OPENING_EDIT_STORE,
            AddUserParentWorkerStage.READING_SNAPSHOT,
            AddUserParentWorkerStage.PERSISTING,
            AddUserParentWorkerStage.CLOSING_EDIT_STORE,
        }:
            return AddUserParentWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            AddUserParentWorkerStage.PROJECTING,
            AddUserParentWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return AddUserParentWorkerFailureCode.PROJECTION_FAILED
        return AddUserParentWorkerFailureCode.PROJECT_INVALID


__all__ = ["AddUserParentWorker"]
