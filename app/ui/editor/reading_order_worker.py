# -*- coding: utf-8 -*-
"""One-shot Qt worker for exact page-wide reading-order commands."""
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
    ReadingOrderCommand,
    ReadingOrderCommandError,
    ReadingOrderCommandErrorCode,
    ReadingOrderCommandReceipt,
    ReadingOrderCommandService,
    ReadingOrderOperation,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    project_id_for,
    project_origin_fingerprint,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import (
    automatic_parent_fingerprint,
    automatic_ordered_parent_ids_for_page,
    project_effective_page,
)
from app.ui.viewmodels.editor_command_model import (
    ReadingOrderCancellationState,
    ReadingOrderCancelledReceipt,
    ReadingOrderWorkerBusyState,
    ReadingOrderWorkerCommand,
    ReadingOrderWorkerFailure,
    ReadingOrderWorkerFailureCode,
    ReadingOrderWorkerReceipt,
    ReadingOrderWorkerStage,
)


class _CancelledBeforePersistence(RuntimeError):
    pass


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.PAGE_NOT_FOUND,
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
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.PAGE_NOT_FOUND,
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


def _validate_snapshot_order(
    snapshot: ProjectEditReadSnapshot,
    command: ReadingOrderWorkerCommand,
) -> None:
    """Fail impossible/stale GUI drafts before sidecar creation."""

    page_mapping = _exact_page(snapshot.project, command.page_id)
    try:
        automatic = automatic_ordered_parent_ids_for_page(page_mapping)
        effective = project_effective_page(
            snapshot.project,
            snapshot.ledger,
            page_id=command.page_id,
        )
    except (TypeError, ValueError) as exc:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.AUTOMATIC_ORDER_UNAVAILABLE,
            "The automatic page order is unavailable or invalid.",
        ) from exc
    if effective.effective_fingerprint != command.expected_effective_page_fingerprint:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.STALE_EFFECTIVE_PAGE,
            "Effective page state changed; reload the selected page.",
        )
    before = tuple(effective.hierarchy.ordered_parent_ids)
    proposed = command.ordered_parent_ids
    automatic_ids = frozenset(automatic)
    effective_ids = frozenset(before)
    if not automatic_ids.issubset(effective_ids):
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
            "Automatic parent identities must remain in the effective page order.",
        )
    parent_by_id = {parent.parent_id: parent for parent in effective.parents}
    if frozenset(parent_by_id) != effective_ids:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
            "Effective parent identities do not match the hierarchy order.",
        )
    automatic_bundles = page_mapping.get("parent_execution_bundles")
    if not isinstance(automatic_bundles, (list, tuple)):
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.AUTOMATIC_ORDER_UNAVAILABLE,
            "Automatic parent evidence is unavailable.",
        )
    automatic_bundle_by_id = {
        str(bundle.get("parent_id") or "").strip(): bundle
        for bundle in automatic_bundles
        if isinstance(bundle, Mapping)
    }
    if frozenset(automatic_bundle_by_id) != automatic_ids:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.AUTOMATIC_ORDER_UNAVAILABLE,
            "Automatic parent evidence does not match the automatic order.",
        )
    for parent_id in automatic_ids:
        parent = parent_by_id[parent_id]
        bundle = automatic_bundle_by_id[parent_id]
        if (
            parent.origin.value != "automatic"
            or parent.bundle_id != str(bundle.get("bundle_id") or "").strip()
            or parent.automatic_fingerprint
            != automatic_parent_fingerprint(bundle)
        ):
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
                "Automatic parent evidence changed in the effective hierarchy.",
            )
    for parent_id in effective_ids - automatic_ids:
        parent = parent_by_id[parent_id]
        if (
            parent.origin.value != "user"
            or parent.bundle_id is not None
            or parent.automatic_fingerprint is not None
            or parent.automatic_geometry is not None
            or parent.automatic_render_style
            or parent.automatic_render_layout
        ):
            raise ReadingOrderCommandError(
                ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
                "Added effective parents must remain typed user parents without automatic evidence.",
            )
    if frozenset(proposed) != effective_ids:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.INVALID_PERMUTATION,
            "Reading order must contain every current parent exactly once.",
        )
    matches = tuple(
        parent
        for parent in effective.parents
        if parent.parent_id == command.selected_parent_id
    )
    if len(matches) != 1:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.PARENT_NOT_FOUND,
            "The selected parent is no longer available.",
        )
    if matches[0].excluded:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.SELECTED_PARENT_EXCLUDED,
            "Excluded parents cannot be moved.",
        )
    excluded = set(effective.hierarchy.excluded_parent_ids)
    if any(
        proposed[index] != parent_id
        for index, parent_id in enumerate(before)
        if parent_id in excluded
    ):
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.EXCLUDED_PARENT_MOVED,
            "Excluded parents must remain in their effective absolute slots.",
        )
    if proposed == before:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.NO_OP,
            "The requested page reading order is already effective.",
        )
    before_without_selected = tuple(
        parent_id
        for parent_id in before
        if parent_id != command.selected_parent_id and parent_id not in excluded
    )
    proposed_without_selected = tuple(
        parent_id
        for parent_id in proposed
        if parent_id != command.selected_parent_id and parent_id not in excluded
    )
    if before_without_selected != proposed_without_selected:
        raise ReadingOrderCommandError(
            ReadingOrderCommandErrorCode.MULTIPLE_PARENTS_MOVED,
            "Only the selected parent may move relative to active peers.",
        )


class ReadingOrderWorker(QtCore.QObject):
    """Commit one page permutation in a dedicated thread."""

    busy = QtCore.Signal(object)
    cancellation = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    stale = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    cancelled = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(self, command: ReadingOrderWorkerCommand) -> None:
        super().__init__()
        if not isinstance(command, ReadingOrderWorkerCommand):
            raise TypeError("command must be ReadingOrderWorkerCommand")
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
                    ReadingOrderWorkerFailureCode.WORKER_REUSED,
                    ReadingOrderWorkerStage.LOADING_PROJECT,
                    "ReadingOrderWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        core_receipt: ReadingOrderCommandReceipt | None = None
        terminal_receipt: ReadingOrderWorkerReceipt | None = None
        terminal_failure: ReadingOrderWorkerFailure | None = None
        terminal_cancelled: ReadingOrderCancelledReceipt | None = None
        stage = ReadingOrderWorkerStage.LOADING_PROJECT
        self._emit_cancellation("Reading order can be cancelled before persistence.")
        self._emit_busy(stage, "Loading current project state...")

        try:
            project = load_project_for_editing(self.command.project_path)
            _exact_page(project, self.command.page_id)
            self._cancel_checkpoint(stage)

            stage = ReadingOrderWorkerStage.OPENING_EDIT_STORE
            self._emit_busy(stage, "Opening the project edit journal...")
            store = _open_project_edit_store(
                self.command.project_path,
                project,
                create=False,
            )
            stage = ReadingOrderWorkerStage.READING_SNAPSHOT
            self._emit_busy(stage, "Reading the exact page revision...")
            if store is None:
                snapshot = _base_snapshot(project, page_id=self.command.page_id)
            else:
                snapshot = store.materialize_project_snapshot(
                    project,
                    page_id=self.command.page_id,
                )
            self._cancel_checkpoint(stage)

            stage = ReadingOrderWorkerStage.PROJECTING
            self._emit_busy(stage, "Validating the complete page order...")
            _validate_snapshot_order(snapshot, self.command)
            self._cancel_checkpoint(stage)

            stage = ReadingOrderWorkerStage.PREPARING_COMMAND
            self._emit_busy(stage, "Preparing the typed reading-order edit...")
            command = ReadingOrderCommand(
                command_id=uuid.uuid4().hex,
                project_id=project_id_for(snapshot.project),
                page_id=self.command.page_id,
                selected_parent_id=self.command.selected_parent_id,
                operation=ReadingOrderOperation.SET,
                ordered_parent_ids=self.command.ordered_parent_ids,
                expected_effective_page_fingerprint=(
                    self.command.expected_effective_page_fingerprint
                ),
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
            )
            if not self._lock_persistence():
                raise _CancelledBeforePersistence

            stage = ReadingOrderWorkerStage.PERSISTING
            self._emit_cancellation(
                "The reading-order edit is being persisted and can no longer be cancelled."
            )
            self._emit_busy(stage, "Saving the page reading order...")
            if store is None:
                store = _open_project_edit_store(
                    self.command.project_path,
                    snapshot.project,
                    create=True,
                )
                if store is None:  # pragma: no cover
                    raise RuntimeError("project edit store was not created")
            core_receipt = ReadingOrderCommandService(
                edit_store=store
            ).execute_materialized(
                snapshot=snapshot,
                command=command,
            )

            stage = ReadingOrderWorkerStage.MATERIALIZING_PROJECT
            self._emit_busy(stage, "Refreshing committed project state...")
            post_commit_snapshot = store.materialize_project_snapshot(
                snapshot.project,
                page_id=self.command.page_id,
            )
            materialized = post_commit_snapshot.project

            stage = ReadingOrderWorkerStage.BUILDING_UI_PROJECTION
            self._emit_busy(stage, "Refreshing editor state...")
            from app.ui.shell.project_projection import project_ui_projection

            projection = project_ui_projection(
                materialized,
                post_commit_snapshot.ledger,
                project_path=self.command.project_path,
            )
            terminal_receipt = ReadingOrderWorkerReceipt(
                command_receipt=core_receipt,
                project=materialized,
                projection=projection,
            )
        except _CancelledBeforePersistence:
            terminal_cancelled = ReadingOrderCancelledReceipt(
                project_path=self.command.project_path,
                page_id=self.command.page_id,
                selected_parent_id=self.command.selected_parent_id,
                ordered_parent_ids=self.command.ordered_parent_ids,
                stage=stage,
            )
        except ReadingOrderCommandError as exc:
            terminal_failure = self._failure_from_command(stage, exc)
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                ReadingOrderWorkerFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
                core_receipt=core_receipt,
            )
        except (KeyError, TypeError, ValueError) as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "The reading-order command is invalid.",
                exc,
                core_receipt=core_receipt,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Reading-order I/O failed.",
                exc,
                core_receipt=core_receipt,
            )
        except Exception as exc:  # pragma: no cover
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage, core_receipt is not None),
                stage,
                str(exc) or "Reading-order edit failed.",
                exc,
                core_receipt=core_receipt,
            )
        finally:
            if store is not None:
                try:
                    stage = ReadingOrderWorkerStage.CLOSING_EDIT_STORE
                    self._emit_busy(stage, "Closing the project edit journal...")
                    store.close()
                except Exception as exc:  # pragma: no cover
                    terminal_receipt = None
                    terminal_cancelled = None
                    terminal_failure = self._failure(
                        ReadingOrderWorkerFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit journal could not be closed safely.",
                        exc,
                        core_receipt=core_receipt,
                    )

        with self._cancel_lock:
            self._persistence_locked = True
        self._emit_cancellation("Reading-order worker finished.")
        self.busy.emit(
            ReadingOrderWorkerBusyState(
                page_id=self.command.page_id,
                selected_parent_id=self.command.selected_parent_id,
                busy=False,
                cancellation_enabled=False,
                persistence_started=core_receipt is not None,
                stage=ReadingOrderWorkerStage.COMPLETE,
                message="Reading-order worker finished.",
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
                    ReadingOrderWorkerFailureCode.COMMAND_REJECTED,
                    stage,
                    "Reading-order worker ended without a typed result.",
                    core_receipt=core_receipt,
                )
            )
        self.finished.emit()

    def _cancel_checkpoint(self, stage: ReadingOrderWorkerStage) -> None:
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

    def _emit_busy(self, stage: ReadingOrderWorkerStage, message: str) -> None:
        with self._cancel_lock:
            persistence_started = self._persistence_locked
            cancellation_enabled = (
                not self._persistence_locked and not self._cancel_requested
            )
        self.busy.emit(
            ReadingOrderWorkerBusyState(
                page_id=self.command.page_id,
                selected_parent_id=self.command.selected_parent_id,
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
            ReadingOrderCancellationState(
                page_id=self.command.page_id,
                selected_parent_id=self.command.selected_parent_id,
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
        stage: ReadingOrderWorkerStage,
        exc: ReadingOrderCommandError,
    ) -> ReadingOrderWorkerFailure:
        code = {
            ReadingOrderCommandErrorCode.PROJECT_IDENTITY_MISMATCH: ReadingOrderWorkerFailureCode.PROJECT_INVALID,
            ReadingOrderCommandErrorCode.STORE_IDENTITY_MISMATCH: ReadingOrderWorkerFailureCode.EDIT_STORE_FAILED,
            ReadingOrderCommandErrorCode.PAGE_NOT_FOUND: ReadingOrderWorkerFailureCode.PAGE_NOT_FOUND,
            ReadingOrderCommandErrorCode.PARENT_NOT_FOUND: ReadingOrderWorkerFailureCode.PARENT_NOT_FOUND,
            ReadingOrderCommandErrorCode.SELECTED_PARENT_EXCLUDED: ReadingOrderWorkerFailureCode.PARENT_EXCLUDED,
            ReadingOrderCommandErrorCode.AUTOMATIC_ORDER_UNAVAILABLE: ReadingOrderWorkerFailureCode.AUTOMATIC_ORDER_UNAVAILABLE,
            ReadingOrderCommandErrorCode.INVALID_PERMUTATION: ReadingOrderWorkerFailureCode.INVALID_ORDER,
            ReadingOrderCommandErrorCode.EXCLUDED_PARENT_MOVED: ReadingOrderWorkerFailureCode.EXCLUDED_SLOT_MOVED,
            ReadingOrderCommandErrorCode.MULTIPLE_PARENTS_MOVED: ReadingOrderWorkerFailureCode.MULTIPLE_PARENTS_MOVED,
            ReadingOrderCommandErrorCode.NO_OP: ReadingOrderWorkerFailureCode.NO_OP,
            ReadingOrderCommandErrorCode.STALE_EFFECTIVE_PAGE: ReadingOrderWorkerFailureCode.SNAPSHOT_STALE,
            ReadingOrderCommandErrorCode.STALE_PAGE_HEAD: ReadingOrderWorkerFailureCode.SNAPSHOT_STALE,
            ReadingOrderCommandErrorCode.STALE_GLOBAL_HEAD: ReadingOrderWorkerFailureCode.SNAPSHOT_STALE,
            ReadingOrderCommandErrorCode.READING_ORDER_SLOT_CONFLICT: ReadingOrderWorkerFailureCode.READING_ORDER_SLOT_CONFLICT,
            ReadingOrderCommandErrorCode.INVALIDATION_UNRESOLVED: ReadingOrderWorkerFailureCode.INVALIDATION_UNRESOLVED,
            ReadingOrderCommandErrorCode.DUPLICATE_COMMAND: ReadingOrderWorkerFailureCode.DUPLICATE_COMMAND,
            ReadingOrderCommandErrorCode.PROJECTION_REJECTED: ReadingOrderWorkerFailureCode.PROJECTION_FAILED,
        }.get(exc.code, ReadingOrderWorkerFailureCode.COMMAND_REJECTED)
        return self._failure(
            code,
            stage,
            str(exc) or "Reading-order command was rejected.",
            exc,
            command_error_code=exc.code,
        )

    def _failure(
        self,
        code: ReadingOrderWorkerFailureCode,
        stage: ReadingOrderWorkerStage,
        message: str,
        exc: BaseException | None = None,
        *,
        command_error_code: ReadingOrderCommandErrorCode | None = None,
        core_receipt: ReadingOrderCommandReceipt | None = None,
    ) -> ReadingOrderWorkerFailure:
        committed = core_receipt is not None
        if committed and code not in {
            ReadingOrderWorkerFailureCode.EDIT_STORE_FAILED,
            ReadingOrderWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED,
        }:
            code = ReadingOrderWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        return ReadingOrderWorkerFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            selected_parent_id=self.command.selected_parent_id,
            ordered_parent_ids=self.command.ordered_parent_ids,
            message=str(message or "Reading-order edit failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
            command_error_code=command_error_code,
            persistence_committed=committed,
            command_receipt=core_receipt,
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: ReadingOrderWorkerStage,
        committed: bool,
    ) -> ReadingOrderWorkerFailureCode:
        if committed:
            return ReadingOrderWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED
        if stage is ReadingOrderWorkerStage.LOADING_PROJECT:
            return ReadingOrderWorkerFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            ReadingOrderWorkerStage.OPENING_EDIT_STORE,
            ReadingOrderWorkerStage.READING_SNAPSHOT,
            ReadingOrderWorkerStage.PERSISTING,
            ReadingOrderWorkerStage.CLOSING_EDIT_STORE,
        }:
            return ReadingOrderWorkerFailureCode.EDIT_STORE_FAILED
        if stage in {
            ReadingOrderWorkerStage.PROJECTING,
            ReadingOrderWorkerStage.BUILDING_UI_PROJECTION,
        }:
            return ReadingOrderWorkerFailureCode.PROJECTION_FAILED
        return ReadingOrderWorkerFailureCode.PROJECT_INVALID


__all__ = ["ReadingOrderWorker"]
