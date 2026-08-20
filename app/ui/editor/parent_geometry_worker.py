# -*- coding: utf-8 -*-
"""One-shot Qt worker for selected-parent geometry commands."""
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
    ParentGeometryCommand,
    ParentGeometryCommandError,
    ParentGeometryCommandErrorCode,
    ParentGeometryCommandReceipt,
    ParentGeometryCommandService,
    ParentGeometryOperation,
    page_canvas_size_for_project_page,
)
from app.project_edits.contracts import thaw_json
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.ui.viewmodels.editor_command_model import (
    ParentGeometryCancellationState,
    ParentGeometryCancelledReceipt,
    ParentGeometryWorkerBusyState,
    ParentGeometryWorkerCommand,
    ParentGeometryWorkerFailure,
    ParentGeometryWorkerFailureCode,
    ParentGeometryWorkerReceipt,
    ParentGeometryWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(
    project: Mapping[str, Any],
    page_id: str,
) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.PAGE_NOT_FOUND,
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
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.PAGE_NOT_FOUND,
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


def _exact_bbox(value: Any, field_name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in value
        )
    ):
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.INVALID_GEOMETRY,
            f"{field_name} must contain four exact integers.",
        )
    bbox = tuple(int(component) for component in value)
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= 0 or bbox[3] <= 0:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.INVALID_GEOMETRY,
            f"{field_name} must have a non-negative origin and positive size.",
        )
    return bbox


def _require_contained(
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> None:
    x, y, width, height = bbox
    if x + width > canvas_size[0] or y + height > canvas_size[1]:
        raise ParentGeometryCommandError(
            ParentGeometryCommandErrorCode.GEOMETRY_OUT_OF_BOUNDS,
            "Parent geometry must remain fully contained by the page canvas.",
        )


class ParentGeometryWorker(QtCore.QObject):
    """Commit one geometry command after being moved to a QThread.

    SQLite is opened, read, used, and closed entirely inside run. Direct
    cancellation is accepted only before the persistence boundary.
    """

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: ParentGeometryWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, ParentGeometryWorkerCommand):
            raise TypeError("command must be ParentGeometryWorkerCommand")
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
                    ParentGeometryWorkerFailureCode.WORKER_REUSED,
                    ParentGeometryWorkerStage.LOADING_PROJECT,
                    "ParentGeometryWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: ParentGeometryCommandReceipt | None = None
        terminal_receipt: ParentGeometryWorkerReceipt | None = None
        terminal_failure: ParentGeometryWorkerFailure | None = None
        terminal_cancelled: ParentGeometryCancelledReceipt | None = None
        stage = ParentGeometryWorkerStage.LOADING_PROJECT
        self._emit_cancellation(
            "Parent geometry can be cancelled before persistence."
        )
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = ParentGeometryWorkerStage.OPENING_EDIT_STORE
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
                stage = ParentGeometryWorkerStage.READING_SNAPSHOT
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

            stage = ParentGeometryWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating parent and page geometry...")
            page = _exact_page(project, self.command.page_id)
            canvas_size = page_canvas_size_for_project_page(
                page,
                project_path=self.command.project_path,
            )
            requested_bbox = _exact_bbox(self.command.bbox, "bbox")
            _require_contained(requested_bbox, canvas_size)
            snapshot = project_effective_page(
                project,
                ledger,
                page_id=self.command.page_id,
            )
            if (
                snapshot.effective_fingerprint
                != self.command.expected_effective_page_fingerprint
            ):
                raise ParentGeometryCommandError(
                    ParentGeometryCommandErrorCode.STALE_EFFECTIVE_PAGE,
                    "Effective page state changed; reload the selected parent.",
                )
            parent_matches = tuple(
                parent
                for parent in snapshot.parents
                if parent.parent_id == self.command.parent_id
            )
            if len(parent_matches) != 1:
                raise ParentGeometryCommandError(
                    ParentGeometryCommandErrorCode.PARENT_NOT_FOUND,
                    "The selected parent is no longer available.",
                )
            current_bbox = _exact_bbox(
                thaw_json(parent_matches[0].geometry),
                "effective parent geometry",
            )
            _require_contained(current_bbox, canvas_size)
            if current_bbox == requested_bbox:
                raise ParentGeometryCommandError(
                    ParentGeometryCommandErrorCode.NO_OP,
                    "The requested parent geometry is already effective.",
                )
            self._cancel_checkpoint(stage)

            stage = ParentGeometryWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed geometry edit...")
            command = ParentGeometryCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(project),
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                operation=ParentGeometryOperation.SET_GEOMETRY,
                bbox=requested_bbox,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=page_head_sha256,
                expected_global_head_sha256=global_head_sha256,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = ParentGeometryWorkerStage.PERSISTING
            self._emit_cancellation(
                "The geometry edit is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the parent geometry edit...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = ParentGeometryCommandService(
                edit_store=store
            ).execute(
                project=project,
                command=command,
            )

            stage = ParentGeometryWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project
            materialized_ledger = post_commit_snapshot.ledger

            stage = ParentGeometryWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                materialized_ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = ParentGeometryWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = ParentGeometryCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                bbox=self.command.bbox,
                stage=stage,
            )
        except ParentGeometryCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                ParentGeometryWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The geometry command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Parent geometry I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Parent geometry edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = ParentGeometryWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        ParentGeometryWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Parent-geometry worker finished.")
        self.busy.emit(
            ParentGeometryWorkerBusyState(
                page_id=self.command.page_id,
                parent_id=self.command.parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=ParentGeometryWorkerStage.COMPLETE,
                message="Parent-geometry worker finished.",
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
                    ParentGeometryWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Parent-geometry worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: ParentGeometryWorkerStage) -> None:
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
        stage: ParentGeometryWorkerStage,
        message: str,
    ) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            ParentGeometryWorkerBusyState(
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
            ParentGeometryCancellationState(
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
        stage: ParentGeometryWorkerStage,
        exc: ParentGeometryCommandError,
    ) -> ParentGeometryWorkerFailure:
        code = {
            ParentGeometryCommandErrorCode.PAGE_NOT_FOUND: ParentGeometryWorkerFailureCode.PAGE_NOT_FOUND,
            ParentGeometryCommandErrorCode.PARENT_NOT_FOUND: ParentGeometryWorkerFailureCode.PARENT_NOT_FOUND,
            ParentGeometryCommandErrorCode.CANVAS_UNAVAILABLE: ParentGeometryWorkerFailureCode.CANVAS_UNAVAILABLE,
            ParentGeometryCommandErrorCode.INVALID_GEOMETRY: ParentGeometryWorkerFailureCode.INVALID_GEOMETRY,
            ParentGeometryCommandErrorCode.GEOMETRY_OUT_OF_BOUNDS: ParentGeometryWorkerFailureCode.GEOMETRY_OUT_OF_BOUNDS,
            ParentGeometryCommandErrorCode.NO_OP: ParentGeometryWorkerFailureCode.NO_OP,
            ParentGeometryCommandErrorCode.STALE_EFFECTIVE_PAGE: ParentGeometryWorkerFailureCode.SNAPSHOT_STALE,
            ParentGeometryCommandErrorCode.STALE_PAGE_HEAD: ParentGeometryWorkerFailureCode.SNAPSHOT_STALE,
            ParentGeometryCommandErrorCode.STALE_GLOBAL_HEAD: ParentGeometryWorkerFailureCode.SNAPSHOT_STALE,
            ParentGeometryCommandErrorCode.GEOMETRY_SLOT_CONFLICT: ParentGeometryWorkerFailureCode.GEOMETRY_SLOT_CONFLICT,
            ParentGeometryCommandErrorCode.INVALIDATION_UNRESOLVED: ParentGeometryWorkerFailureCode.INVALIDATION_UNRESOLVED,
            ParentGeometryCommandErrorCode.DUPLICATE_COMMAND: ParentGeometryWorkerFailureCode.DUPLICATE_COMMAND,
            ParentGeometryCommandErrorCode.PROJECTION_REJECTED: ParentGeometryWorkerFailureCode.PROJECTION_FAILED,
            ParentGeometryCommandErrorCode.PROJECT_IDENTITY_MISMATCH: ParentGeometryWorkerFailureCode.PROJECT_INVALID,
            ParentGeometryCommandErrorCode.STORE_IDENTITY_MISMATCH: ParentGeometryWorkerFailureCode.EDIT_STORE_FAILED,
        }.get(exc.code, ParentGeometryWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Parent-geometry command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: ParentGeometryWorkerFailureCode,
        stage: ParentGeometryWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: ParentGeometryCommandErrorCode | None = None,
        core_receipt: ParentGeometryCommandReceipt | None = None,
    ) -> ParentGeometryWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            ParentGeometryWorkerFailureCode.EDIT_STORE_FAILED,
            ParentGeometryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = ParentGeometryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return ParentGeometryWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            parent_id=self.command.parent_id,
            bbox=self.command.bbox,
            message=str(message or "Parent geometry edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: ParentGeometryWorkerStage,
        committed: bool,
    ) -> ParentGeometryWorkerFailureCode:
        if committed:
            return ParentGeometryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is ParentGeometryWorkerStage.LOADING_PROJECT:
            return ParentGeometryWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            ParentGeometryWorkerStage.OPENING_EDIT_STORE,
            ParentGeometryWorkerStage.READING_SNAPSHOT,
            ParentGeometryWorkerStage.PERSISTING,
            ParentGeometryWorkerStage.CLOSING_EDIT_STORE,
        }:
            return ParentGeometryWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            ParentGeometryWorkerStage.PROJECTING,
            ParentGeometryWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return ParentGeometryWorkerFailureCode.PROJECTION_FAILED
        return ParentGeometryWorkerFailureCode.PROJECT_INVALID


__all__ = ["ParentGeometryWorker"]
