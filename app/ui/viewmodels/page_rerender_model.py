# -*- coding: utf-8 -*-
"""Framework-neutral state and commands for page-local GUI rerendering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

from app.project_edits.rerender_service import (
    PageRerenderPreflight,
    PageRerenderReceipt,
    RerenderAvailabilityCode,
    RerenderMode,
    RerenderProgress,
    RerenderStage,
    RerenderStatus,
)


class PageRerenderFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    MISSING_BASE = "missing_base"
    CONFLICT = "conflict"
    BLOCKED = "blocked"
    RERENDER_FAILED = "rerender_failed"
    WORKER_REUSED = "worker_reused"


class PageRerenderWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    REHYDRATING_PAGE = "rehydrating_page"
    PROJECTING = "projecting"
    PREFLIGHT = "preflight"
    RERENDERING = "rerendering"
    CLOSING_EDIT_STORE = "closing_edit_store"


class PageRerenderViewPhase(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    MISSING_BASE = "missing_base"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PageRerenderCommand:
    """Exact page-local command accepted by the GUI-3 preview worker."""

    project_path: str
    page_id: str
    mode: RerenderMode = RerenderMode.PREVIEW

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        page_id = str(self.page_id or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        if not page_id:
            raise ValueError("page_id is required")
        if not isinstance(self.mode, RerenderMode):
            raise TypeError("mode must be RerenderMode")
        if self.mode is not RerenderMode.PREVIEW:
            raise ValueError("the GUI-3 worker currently accepts PREVIEW mode only")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        object.__setattr__(self, "page_id", page_id)


@dataclass(frozen=True, slots=True)
class PageRerenderFailure:
    code: PageRerenderFailureCode
    stage: PageRerenderWorkerStage
    project_path: str
    page_id: str
    message: str
    exception_type: str = ""
    preflight: PageRerenderPreflight | None = None


@dataclass(frozen=True, slots=True)
class PageRerenderPreviewLease:
    """One temporary preview that the editor may display and then discard."""

    project_path: str
    page_id: str
    output_path: str
    output_sha256: str

    @classmethod
    def from_receipt(
        cls,
        command: PageRerenderCommand,
        receipt: PageRerenderReceipt,
    ) -> "PageRerenderPreviewLease":
        if receipt.page_id != command.page_id:
            raise ValueError("preview receipt belongs to another page")
        if receipt.mode is not RerenderMode.PREVIEW:
            raise ValueError("preview lease cannot own a committed artifact")
        if receipt.status is not RerenderStatus.COMPLETED:
            raise ValueError("only a completed preview can be leased")
        output_path = str(receipt.output_path or "").strip()
        output_sha256 = str(receipt.output_sha256 or "").lower()
        if not output_path:
            raise ValueError("preview receipt has no output path")
        if len(output_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in output_sha256
        ):
            raise ValueError("preview receipt has an invalid output hash")
        return cls(
            project_path=command.project_path,
            page_id=command.page_id,
            output_path=os.path.abspath(output_path),
            output_sha256=output_sha256,
        )


@dataclass(frozen=True, slots=True)
class PageRerenderViewState:
    phase: PageRerenderViewPhase = PageRerenderViewPhase.IDLE
    command: PageRerenderCommand | None = None
    preview_enabled: bool = True
    cancel_enabled: bool = False
    message: str = ""
    preflight: PageRerenderPreflight | None = None
    progress: RerenderProgress | None = None
    receipt: PageRerenderReceipt | None = None
    preview_lease: PageRerenderPreviewLease | None = None
    failure: PageRerenderFailure | None = None


def failure_from_preflight(
    command: PageRerenderCommand,
    preflight: PageRerenderPreflight,
) -> PageRerenderFailure:
    """Convert service availability into a stable GUI failure contract."""

    code = {
        RerenderAvailabilityCode.MISSING_BASE: PageRerenderFailureCode.MISSING_BASE,
        RerenderAvailabilityCode.CONFLICT: PageRerenderFailureCode.CONFLICT,
        RerenderAvailabilityCode.BLOCKED: PageRerenderFailureCode.BLOCKED,
    }.get(preflight.code, PageRerenderFailureCode.BLOCKED)
    return PageRerenderFailure(
        code=code,
        stage=PageRerenderWorkerStage.PREFLIGHT,
        project_path=command.project_path,
        page_id=command.page_id,
        message=preflight.message,
        preflight=preflight,
    )


class PageRerenderViewModel:
    """Small UI-thread reducer for worker signals; it imports no Qt types."""

    def __init__(self) -> None:
        self._state = PageRerenderViewState()

    @property
    def state(self) -> PageRerenderViewState:
        return self._state

    def begin(self, command: PageRerenderCommand) -> PageRerenderViewState:
        if not isinstance(command, PageRerenderCommand):
            raise TypeError("command must be PageRerenderCommand")
        if self._state.cancel_enabled:
            raise RuntimeError("a page rerender is already active")
        self._state = PageRerenderViewState(
            phase=PageRerenderViewPhase.LOADING,
            command=command,
            preview_enabled=False,
            cancel_enabled=True,
            message="Loading current page state...",
        )
        return self._state

    def accept_preflight(
        self,
        value: PageRerenderPreflight,
    ) -> PageRerenderViewState:
        command = self._require_active_page(value.page_id)
        if value.ready:
            self._state = PageRerenderViewState(
                phase=PageRerenderViewPhase.RUNNING,
                command=command,
                preview_enabled=False,
                cancel_enabled=True,
                message=value.message,
                preflight=value,
            )
        else:
            self.accept_failure(failure_from_preflight(command, value))
        return self._state

    def accept_progress(self, value: RerenderProgress) -> PageRerenderViewState:
        command = self._require_active_page(value.page_id)
        phase = (
            PageRerenderViewPhase.CANCELLED
            if value.stage is RerenderStage.CANCELLED
            else PageRerenderViewPhase.RUNNING
        )
        self._state = PageRerenderViewState(
            phase=phase,
            command=command,
            preview_enabled=False,
            cancel_enabled=phase is PageRerenderViewPhase.RUNNING,
            message=value.message,
            preflight=self._state.preflight,
            progress=value,
        )
        return self._state

    def accept_receipt(self, value: PageRerenderReceipt) -> PageRerenderViewState:
        command = self._require_active_page(value.page_id)
        cancelled = value.status is RerenderStatus.CANCELLED
        self._state = PageRerenderViewState(
            phase=(
                PageRerenderViewPhase.CANCELLED
                if cancelled
                else PageRerenderViewPhase.COMPLETED
            ),
            command=command,
            preview_enabled=True,
            cancel_enabled=False,
            message=("Preview cancelled." if cancelled else "Preview ready."),
            preflight=self._state.preflight,
            progress=self._state.progress,
            receipt=value,
            preview_lease=(
                None
                if cancelled
                else PageRerenderPreviewLease.from_receipt(command, value)
            ),
        )
        return self._state

    def accept_failure(self, value: PageRerenderFailure) -> PageRerenderViewState:
        command = self._require_active_page(value.page_id)
        if value.code is PageRerenderFailureCode.MISSING_BASE:
            phase = PageRerenderViewPhase.MISSING_BASE
        elif value.code in {
            PageRerenderFailureCode.BLOCKED,
            PageRerenderFailureCode.CONFLICT,
        }:
            phase = PageRerenderViewPhase.BLOCKED
        else:
            phase = PageRerenderViewPhase.FAILED
        self._state = PageRerenderViewState(
            phase=phase,
            command=command,
            preview_enabled=(
                value.code is PageRerenderFailureCode.RERENDER_FAILED
            ),
            cancel_enabled=False,
            message=value.message,
            preflight=value.preflight or self._state.preflight,
            progress=self._state.progress,
            failure=value,
        )
        return self._state

    def reset(self) -> PageRerenderViewState:
        if self._state.cancel_enabled:
            raise RuntimeError("cancel the active page rerender before resetting")
        self._state = PageRerenderViewState()
        return self._state

    def _require_active_page(self, page_id: str) -> PageRerenderCommand:
        command = self._state.command
        if command is None:
            raise RuntimeError("no page rerender command is active")
        if str(page_id or "") != command.page_id:
            raise ValueError("worker event belongs to another page")
        return command
