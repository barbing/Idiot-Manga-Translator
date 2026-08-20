# -*- coding: utf-8 -*-
"""Qt worker bridge for the GUI-owned page-local rerender service."""
from __future__ import annotations

import os
import threading
from typing import Any, Mapping

from PySide6 import QtCore

from app.io.project import load_project_for_editing
from app.io.project_edit_store import (
    ProjectEditStore,
    inspect_project_edit_store,
)
from app.project_edits.fingerprints import (
    automated_state_fingerprint,
    canonical_sha256,
    project_id_for,
)
from app.project_edits.ledger import ProjectEditLedger
from app.project_edits.projection import project_effective_page
from app.project_edits.rerender_service import (
    PageRerenderCancellationToken,
    PageRerenderError,
    PageRerenderReceipt,
    PageRerenderRequest,
    PageRerenderService,
    RerenderStatus,
    rerender_artifact_root,
)
from app.ui.viewmodels.page_rerender_model import (
    PageRerenderCommand,
    PageRerenderFailure,
    PageRerenderFailureCode,
    PageRerenderPreviewLease,
    PageRerenderWorkerStage,
    failure_from_preflight,
)


class _ExactPageNotFound(LookupError):
    pass


_PREVIEW_LEASE_LOCK = threading.RLock()


def _exact_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages")
    if not isinstance(pages, list):
        raise ValueError("project pages must be a list")
    matches = [
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    ]
    if not matches:
        raise _ExactPageNotFound(page_id)
    if len(matches) != 1:
        raise ValueError(f"project page identity is duplicated: {page_id}")
    return matches[0]


def _open_existing_project_edit_store(
    project_path: str,
    project: Mapping[str, Any],
) -> ProjectEditStore | None:
    """Open an existing sidecar without creating one during preview-only open."""

    metadata = inspect_project_edit_store(project_path)
    if metadata is None:
        return None
    ledger = ProjectEditLedger.from_dict(project["edit_ledger"])
    return ProjectEditStore(
        project_path=project_path,
        project_id=project_id_for(project),
        project_origin_sha256=str(metadata["project_origin_sha256"]),
        automated_state_sha256=automated_state_fingerprint(project),
        base_ledger=ledger,
        base_artifact_revisions=project["artifact_revisions"],
    )


def _managed_preview_path(
    lease: PageRerenderPreviewLease,
) -> tuple[str, str, str]:
    directory = os.path.abspath(
        os.path.join(rerender_artifact_root(lease.project_path), "previews")
    )
    output_path = os.path.abspath(lease.output_path)
    if os.path.normcase(os.path.dirname(output_path)) != os.path.normcase(directory):
        raise ValueError("preview output is outside the managed preview directory")
    page_fragment = canonical_sha256({"page_id": lease.page_id})[:16]
    prefix = f"page-{page_fragment}-"
    filename = os.path.basename(output_path)
    if not filename.startswith(prefix) or not filename.endswith(".png"):
        raise ValueError("preview output name does not match its page identity")
    return directory, output_path, prefix


def retain_preview_lease(lease: PageRerenderPreviewLease) -> None:
    """Keep one preview per page across one-shot service instances."""

    if not isinstance(lease, PageRerenderPreviewLease):
        raise TypeError("lease must be PageRerenderPreviewLease")
    directory, output_path, prefix = _managed_preview_path(lease)
    with _PREVIEW_LEASE_LOCK:
        if os.path.islink(output_path) or not os.path.isfile(output_path):
            raise ValueError("leased preview output is unavailable")
        if not os.path.isdir(directory):
            return
        for entry in os.scandir(directory):
            if (
                os.path.normcase(os.path.abspath(entry.path))
                == os.path.normcase(output_path)
                or not entry.name.startswith(prefix)
                or not entry.name.endswith(".png")
                or not entry.is_file(follow_symlinks=False)
            ):
                continue
            try:
                os.unlink(entry.path)
            except OSError:
                # The UI still owns an explicit disposal lease for later retry.
                pass


def discard_preview_lease(lease: PageRerenderPreviewLease) -> bool:
    """Delete only the leased page preview; committed artifacts are unreachable."""

    if not isinstance(lease, PageRerenderPreviewLease):
        raise TypeError("lease must be PageRerenderPreviewLease")
    _, output_path, _ = _managed_preview_path(lease)
    with _PREVIEW_LEASE_LOCK:
        if os.path.islink(output_path) or not os.path.isfile(output_path):
            return False
        try:
            os.unlink(output_path)
        except OSError:
            return False
    return True


class PageRerenderWorker(QtCore.QObject):
    """One-shot worker; move it to a QThread before invoking ``run``."""

    preflight = QtCore.Signal(object)
    progress = QtCore.Signal(object)
    receipt = QtCore.Signal(object)
    preview_lease = QtCore.Signal(object)
    failure = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        command: PageRerenderCommand,
    ) -> None:
        # A worker QObject must be parentless so the GUI can move it to QThread.
        super().__init__()
        if not isinstance(command, PageRerenderCommand):
            raise TypeError("command must be PageRerenderCommand")
        self.command = command
        self._cancellation = PageRerenderCancellationToken()
        self._run_lock = threading.Lock()
        self._has_run = False

    @property
    def cancellation_token(self) -> PageRerenderCancellationToken:
        return self._cancellation

    def request_cancel(self) -> None:
        """Cancel from any thread without waiting for the worker event loop."""

        self._cancellation.cancel()

    @QtCore.Slot()
    def run(self) -> None:
        if not self._claim_run():
            self.failure.emit(
                PageRerenderFailure(
                    code=PageRerenderFailureCode.WORKER_REUSED,
                    stage=PageRerenderWorkerStage.LOADING_PROJECT,
                    project_path=self.command.project_path,
                    page_id=self.command.page_id,
                    message="PageRerenderWorker instances are one-shot.",
                )
            )
            self.finished.emit()
            return

        store: ProjectEditStore | None = None
        terminal_receipt: PageRerenderReceipt | None = None
        terminal_failure: PageRerenderFailure | None = None
        stage = PageRerenderWorkerStage.LOADING_PROJECT
        try:
            project = load_project_for_editing(self.command.project_path)

            stage = PageRerenderWorkerStage.OPENING_EDIT_STORE
            store = _open_existing_project_edit_store(
                self.command.project_path,
                project,
            )
            if store is not None:
                project = store.materialize_project(project)
                ledger = store.load_ledger()
            else:
                ledger = ProjectEditLedger.from_dict(project["edit_ledger"])

            stage = PageRerenderWorkerStage.REHYDRATING_PAGE
            page = _exact_page(project, self.command.page_id)
            bundle_records = page.get("parent_execution_bundles") or ()
            if not isinstance(bundle_records, (list, tuple)) or any(
                not isinstance(record, Mapping) for record in bundle_records
            ):
                raise ValueError(
                    "saved ParentExecutionBundle records are invalid"
                )
            # Preserve the exact immutable saved audit records.  Rehydrating
            # them into mutable runtime objects normalizes fields and changes
            # the automatic fingerprint before the GUI adapter can validate
            # it against the effective snapshot.
            bundles = tuple(dict(record) for record in bundle_records)

            stage = PageRerenderWorkerStage.PROJECTING
            snapshot = project_effective_page(
                project,
                ledger,
                page_id=self.command.page_id,
            )
            service = PageRerenderService(
                project_path=self.command.project_path,
                edit_store=store,
            )

            stage = PageRerenderWorkerStage.PREFLIGHT
            preflight = service.preflight(snapshot, bundles)
            if preflight.page_id != self.command.page_id:
                raise PageRerenderError(
                    "rerender preflight belongs to another page"
                )
            self.preflight.emit(preflight)
            if not preflight.ready:
                terminal_failure = failure_from_preflight(
                    self.command,
                    preflight,
                )
            else:
                stage = PageRerenderWorkerStage.RERENDERING
                request = PageRerenderRequest(
                    snapshot=snapshot,
                    automatic_parent_bundles=bundles,
                    mode=self.command.mode,
                    expected_page_head_sha256=(
                        store.page_head(self.command.page_id) if store is not None else ""
                    ),
                )
                terminal_receipt = service.rerender(
                    request,
                    cancellation=self._cancellation,
                    progress=self.progress.emit,
                )
                if (
                    terminal_receipt.page_id != self.command.page_id
                    or terminal_receipt.mode is not self.command.mode
                ):
                    raise PageRerenderError(
                        "rerender receipt does not match the exact command"
                    )
        except _ExactPageNotFound:
            terminal_failure = self._failure(
                PageRerenderFailureCode.PAGE_NOT_FOUND,
                stage,
                f"Project page is unavailable: {self.command.page_id}",
            )
        except FileNotFoundError as exc:
            terminal_failure = self._failure(
                PageRerenderFailureCode.PROJECT_LOAD_FAILED,
                stage,
                "The project file is unavailable.",
                exc,
            )
        except PageRerenderError as exc:
            terminal_failure = self._failure(
                PageRerenderFailureCode.RERENDER_FAILED,
                stage,
                str(exc) or "Page preview failed.",
                exc,
            )
        except (KeyError, TypeError, ValueError) as exc:
            code = (
                PageRerenderFailureCode.EDIT_STORE_FAILED
                if stage
                in {
                    PageRerenderWorkerStage.OPENING_EDIT_STORE,
                    PageRerenderWorkerStage.CLOSING_EDIT_STORE,
                }
                else PageRerenderFailureCode.PROJECT_INVALID
            )
            terminal_failure = self._failure(
                code,
                stage,
                str(exc) or "The project cannot form a page preview.",
                exc,
            )
        except OSError as exc:
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage),
                stage,
                str(exc) or "Page preview I/O failed.",
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            terminal_failure = self._failure(
                self._failure_code_for_stage(stage),
                stage,
                str(exc) or "Page preview failed.",
                exc,
            )
        finally:
            if store is not None:
                try:
                    stage = PageRerenderWorkerStage.CLOSING_EDIT_STORE
                    store.close()
                except Exception as exc:  # pragma: no cover - sqlite close is defensive
                    terminal_receipt = None
                    terminal_failure = self._failure(
                        PageRerenderFailureCode.EDIT_STORE_FAILED,
                        stage,
                        "The project edit store could not be closed safely.",
                        exc,
                    )

        if terminal_failure is not None:
            self.failure.emit(terminal_failure)
        elif terminal_receipt is not None:
            try:
                if terminal_receipt.status is RerenderStatus.COMPLETED:
                    lease = PageRerenderPreviewLease.from_receipt(
                        self.command,
                        terminal_receipt,
                    )
                    retain_preview_lease(lease)
                    self.preview_lease.emit(lease)
                self.receipt.emit(terminal_receipt)
            except Exception as exc:  # defensive lease/publication boundary
                self.failure.emit(
                    self._failure(
                        PageRerenderFailureCode.RERENDER_FAILED,
                        PageRerenderWorkerStage.RERENDERING,
                        str(exc) or "Preview lifecycle validation failed.",
                        exc,
                    )
                )
        else:  # pragma: no cover - defensive terminal contract
            self.failure.emit(
                self._failure(
                    PageRerenderFailureCode.RERENDER_FAILED,
                    stage,
                    "Page preview ended without a receipt.",
                )
            )
        self.finished.emit()

    def _claim_run(self) -> bool:
        with self._run_lock:
            if self._has_run:
                return False
            self._has_run = True
            return True

    def _failure(
        self,
        code: PageRerenderFailureCode,
        stage: PageRerenderWorkerStage,
        message: str,
        exc: BaseException | None = None,
    ) -> PageRerenderFailure:
        return PageRerenderFailure(
            code=code,
            stage=stage,
            project_path=self.command.project_path,
            page_id=self.command.page_id,
            message=str(message or "Page preview failed."),
            exception_type=type(exc).__name__ if exc is not None else "",
        )

    @staticmethod
    def _failure_code_for_stage(
        stage: PageRerenderWorkerStage,
    ) -> PageRerenderFailureCode:
        if stage is PageRerenderWorkerStage.LOADING_PROJECT:
            return PageRerenderFailureCode.PROJECT_LOAD_FAILED
        if stage in {
            PageRerenderWorkerStage.OPENING_EDIT_STORE,
            PageRerenderWorkerStage.CLOSING_EDIT_STORE,
        }:
            return PageRerenderFailureCode.EDIT_STORE_FAILED
        return PageRerenderFailureCode.RERENDER_FAILED
