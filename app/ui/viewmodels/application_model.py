# -*- coding: utf-8 -*-
"""Framework-neutral GUI-5 application lifecycle projection.

The reducer consumes typed controller receipts and project-session events.  It
contains no Qt, model, filesystem, or pipeline implementation imports and never
interprets free-form status messages.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from app.pipeline.status_contracts import (
    PipelineErrorReceipt,
    PipelineLifecycleEvent,
    PipelineRunState,
    PipelineStageEvent,
)
from app.ui.ui_contract import NAVIGATION_IDS


class ProjectSessionState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    RECOVERY_AVAILABLE = "recovery_available"
    FAILED = "failed"


def _clean_required(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _clean_optional(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ProjectSessionError:
    code: str
    message: str
    detail: str = ""
    recoverable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _clean_required(self.code, "code"))
        object.__setattr__(
            self,
            "message",
            _clean_required(self.message, "message"),
        )
        object.__setattr__(self, "detail", _clean_optional(self.detail, "detail"))
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be a boolean")


@dataclass(frozen=True, slots=True)
class ProjectSessionSnapshot:
    state: ProjectSessionState = ProjectSessionState.EMPTY
    project_id: str = ""
    project_name: str = ""
    project_path: str = ""
    error: ProjectSessionError | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ProjectSessionState(self.state))
        for field_name in ("project_id", "project_name", "project_path"):
            object.__setattr__(
                self,
                field_name,
                _clean_optional(getattr(self, field_name), field_name),
            )
        if self.error is not None and not isinstance(self.error, ProjectSessionError):
            raise TypeError("error must be ProjectSessionError or None")
        if self.state in {
            ProjectSessionState.READY,
            ProjectSessionState.RECOVERY_AVAILABLE,
        }:
            for field_name in ("project_id", "project_name", "project_path"):
                if not getattr(self, field_name):
                    raise ValueError(f"{field_name} is required for a loaded project")
        if self.state is ProjectSessionState.LOADING and not self.project_path:
            raise ValueError("project_path is required while loading")
        if self.state is ProjectSessionState.FAILED and self.error is None:
            raise ValueError("a failed project session requires an error")
        if self.state is not ProjectSessionState.FAILED and self.error is not None:
            raise ValueError("only a failed project session may retain an error")


_BUSY_RUN_STATES = frozenset(
    {
        PipelineRunState.VALIDATING,
        PipelineRunState.RUNNING,
        PipelineRunState.STOP_REQUESTED,
        PipelineRunState.CANCELLING,
    }
)


@dataclass(frozen=True, slots=True)
class ApplicationViewState:
    navigation_id: str = NAVIGATION_IDS[0]
    project: ProjectSessionSnapshot = ProjectSessionSnapshot()
    lifecycle: PipelineLifecycleEvent | None = None
    stage: PipelineStageEvent | None = None
    active_error: PipelineErrorReceipt | None = None

    def __post_init__(self) -> None:
        if self.navigation_id not in NAVIGATION_IDS:
            raise ValueError(f"unsupported navigation ID: {self.navigation_id!r}")
        if not isinstance(self.project, ProjectSessionSnapshot):
            raise TypeError("project must be ProjectSessionSnapshot")
        if self.lifecycle is not None and not isinstance(
            self.lifecycle, PipelineLifecycleEvent
        ):
            raise TypeError("lifecycle must be PipelineLifecycleEvent or None")
        if self.stage is not None and not isinstance(self.stage, PipelineStageEvent):
            raise TypeError("stage must be PipelineStageEvent or None")
        if self.active_error is not None and not isinstance(
            self.active_error, PipelineErrorReceipt
        ):
            raise TypeError("active_error must be PipelineErrorReceipt or None")

    @property
    def project_busy(self) -> bool:
        return self.project.state is ProjectSessionState.LOADING

    @property
    def pipeline_busy(self) -> bool:
        return bool(self.lifecycle and self.lifecycle.state in _BUSY_RUN_STATES)

    @property
    def close_allowed(self) -> bool:
        """Whether close may proceed without first awaiting active work."""

        return not self.project_busy and not self.pipeline_busy


class ApplicationViewModel:
    """Small UI-thread reducer for navigation, project, and run state."""

    def __init__(self, state: ApplicationViewState | None = None) -> None:
        self._state = state or ApplicationViewState()
        if not isinstance(self._state, ApplicationViewState):
            raise TypeError("state must be ApplicationViewState or None")

    @property
    def state(self) -> ApplicationViewState:
        return self._state

    def navigate(self, navigation_id: str) -> ApplicationViewState:
        if navigation_id not in NAVIGATION_IDS:
            raise ValueError(f"unsupported navigation ID: {navigation_id!r}")
        self._state = replace(self._state, navigation_id=navigation_id)
        return self._state

    def clear_project(self) -> ApplicationViewState:
        if self._state.pipeline_busy:
            raise RuntimeError("cannot clear the project while a run is active")
        self._state = replace(
            self._state,
            navigation_id=NAVIGATION_IDS[0],
            project=ProjectSessionSnapshot(),
            lifecycle=None,
            stage=None,
            active_error=None,
        )
        return self._state

    def begin_project_load(self, project_path: str) -> ApplicationViewState:
        if self._state.pipeline_busy:
            raise RuntimeError("cannot replace the project while a run is active")
        snapshot = ProjectSessionSnapshot(
            state=ProjectSessionState.LOADING,
            project_path=_clean_required(project_path, "project_path"),
        )
        self._state = replace(
            self._state,
            project=snapshot,
            lifecycle=None,
            stage=None,
            active_error=None,
        )
        return self._state

    def complete_project_load(
        self,
        *,
        project_id: str,
        project_name: str,
        project_path: str,
        recovery_available: bool = False,
    ) -> ApplicationViewState:
        if not isinstance(recovery_available, bool):
            raise TypeError("recovery_available must be a boolean")
        snapshot = ProjectSessionSnapshot(
            state=(
                ProjectSessionState.RECOVERY_AVAILABLE
                if recovery_available
                else ProjectSessionState.READY
            ),
            project_id=_clean_required(project_id, "project_id"),
            project_name=_clean_required(project_name, "project_name"),
            project_path=_clean_required(project_path, "project_path"),
        )
        self._state = replace(self._state, project=snapshot, active_error=None)
        return self._state

    def fail_project_load(
        self,
        *,
        project_path: str,
        code: str,
        message: str,
        detail: str = "",
        recoverable: bool = False,
    ) -> ApplicationViewState:
        snapshot = ProjectSessionSnapshot(
            state=ProjectSessionState.FAILED,
            project_path=_clean_required(project_path, "project_path"),
            error=ProjectSessionError(
                code=code,
                message=message,
                detail=detail,
                recoverable=recoverable,
            ),
        )
        self._state = replace(self._state, project=snapshot)
        return self._state

    def apply_lifecycle(
        self,
        event: PipelineLifecycleEvent,
    ) -> ApplicationViewState:
        if not isinstance(event, PipelineLifecycleEvent):
            raise TypeError("event must be PipelineLifecycleEvent")
        current = self._state.lifecycle
        if (
            current is not None
            and current.run_id != event.run_id
            and current.state in _BUSY_RUN_STATES
        ):
            raise ValueError("cannot replace an active run with another run ID")
        new_run = current is None or current.run_id != event.run_id
        self._state = replace(
            self._state,
            lifecycle=event,
            stage=None if new_run else self._state.stage,
            active_error=None if new_run else self._state.active_error,
        )
        return self._state

    def apply_stage(self, event: PipelineStageEvent) -> ApplicationViewState:
        if not isinstance(event, PipelineStageEvent):
            raise TypeError("event must be PipelineStageEvent")
        lifecycle = self._state.lifecycle
        if lifecycle is not None and lifecycle.run_id != event.run_id:
            raise ValueError("stage event belongs to another run")
        self._state = replace(self._state, stage=event)
        return self._state

    def apply_error(self, receipt: PipelineErrorReceipt) -> ApplicationViewState:
        if not isinstance(receipt, PipelineErrorReceipt):
            raise TypeError("receipt must be PipelineErrorReceipt")
        lifecycle = self._state.lifecycle
        if lifecycle is not None and lifecycle.run_id != receipt.run_id:
            raise ValueError("error receipt belongs to another run")
        self._state = replace(self._state, active_error=receipt)
        return self._state

    def clear_error(self, error_id: str | None = None) -> ApplicationViewState:
        active = self._state.active_error
        if active is None:
            return self._state
        if error_id is not None and active.error_id != error_id:
            raise ValueError("error_id does not match the active error")
        self._state = replace(self._state, active_error=None)
        return self._state


__all__ = [
    "ApplicationViewModel",
    "ApplicationViewState",
    "ProjectSessionError",
    "ProjectSessionSnapshot",
    "ProjectSessionState",
]
