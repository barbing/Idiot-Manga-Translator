"""Immutable, GUI-neutral pipeline lifecycle and status contracts.

The production controller emits these records in addition to its historical
Qt signals.  UI code can therefore present lifecycle, owning-stage, progress,
and recovery state without interpreting free-form log messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class PipelineRunState(str, Enum):
    """Controller-owned lifecycle states for one forward pipeline run."""

    IDLE = "idle"
    VALIDATING = "validating"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CANCELLING = "cancelling"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    """Owning workflow stages exposed to GUI presentation models."""

    IDLE = "idle"
    VALIDATION = "validation"
    INITIALIZATION = "initialization"
    PRESCAN = "prescan"
    DETECTION = "detection"
    OCR = "ocr"
    HIERARCHY = "hierarchy"
    TRANSLATION = "translation"
    SOURCE_GLYPH = "source_glyph"
    CLEANUP = "cleanup"
    STYLE = "style"
    RENDERING = "rendering"
    PERSISTENCE = "persistence"
    FINALIZING = "finalizing"


class PipelineRetryAction(str, Enum):
    """User-facing recovery actions a presentation layer may offer."""

    NONE = "none"
    RETRY_RUN = "retry_run"
    RETRY_PAGE = "retry_page"
    REBUILD = "rebuild"
    RELINK = "relink"
    RESET_SETTINGS = "reset_settings"


class PipelineStageOutcomeState(str, Enum):
    """Whether a required stage produced a valid downstream artifact."""

    VALID = "valid"
    VALID_WITH_DIAGNOSTICS = "valid_with_diagnostics"
    TECHNICAL_FAILURE = "technical_failure"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for immutable receipts."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return f"run-{uuid4().hex}"


def new_error_id() -> str:
    return f"error-{uuid4().hex}"


def new_stage_outcome_id() -> str:
    return f"outcome-{uuid4().hex}"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_optional_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")


@dataclass(frozen=True, slots=True)
class PipelineLifecycleEvent:
    run_id: str
    state: PipelineRunState
    message: str = ""
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if not isinstance(self.state, PipelineRunState):
            raise TypeError("state must be a PipelineRunState")
        _require_optional_text(self.message, "message")
        _require_text(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class PipelineStageEvent:
    run_id: str
    stage: PipelineStage
    page_id: str = ""
    parent_id: str = ""
    detail: str = ""
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if not isinstance(self.stage, PipelineStage):
            raise TypeError("stage must be a PipelineStage")
        _require_optional_text(self.page_id, "page_id")
        _require_optional_text(self.parent_id, "parent_id")
        _require_optional_text(self.detail, "detail")
        _require_text(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class RuntimeBackendEvent:
    run_id: str
    module_id: str
    requested_backend: str
    selected_backend: str
    fallback_reason: str = ""
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "module_id",
            "requested_backend",
            "selected_backend",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_optional_text(self.fallback_reason, "fallback_reason")
        _require_text(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class PipelineStageOutcome:
    """Durable owner-stage result for one page barrier.

    Quality limitations belong in ``diagnostics``.  Only
    ``TECHNICAL_FAILURE`` prevents the next required stage from running.
    """

    run_id: str
    page_id: str
    page_index: int
    stage: PipelineStage
    state: PipelineStageOutcomeState
    outcome_id: str = field(default_factory=new_stage_outcome_id)
    page_name: str = ""
    source_path: str = ""
    output_path: str = ""
    parent_ids: tuple[str, ...] = ()
    artifact_kind: str = ""
    artifact_summary: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    error_code: str = ""
    message: str = ""
    detail: str = ""
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.page_id, "page_id")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int):
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index cannot be negative")
        if not isinstance(self.stage, PipelineStage):
            raise TypeError("stage must be a PipelineStage")
        if not isinstance(self.state, PipelineStageOutcomeState):
            raise TypeError("state must be a PipelineStageOutcomeState")
        _require_text(self.outcome_id, "outcome_id")
        for field_name in (
            "page_name",
            "source_path",
            "output_path",
            "artifact_kind",
            "error_code",
            "message",
            "detail",
        ):
            _require_optional_text(getattr(self, field_name), field_name)
        parent_ids = tuple(str(value) for value in self.parent_ids if str(value))
        diagnostics = tuple(str(value) for value in self.diagnostics if str(value))
        object.__setattr__(self, "parent_ids", parent_ids)
        object.__setattr__(self, "diagnostics", diagnostics)
        if not isinstance(self.artifact_summary, Mapping):
            raise TypeError("artifact_summary must be mapping-like")
        if self.state is PipelineStageOutcomeState.TECHNICAL_FAILURE:
            _require_text(self.error_code, "error_code")
            _require_text(self.message, "message")
        _require_text(self.timestamp, "timestamp")

    @property
    def valid_for_next_stage(self) -> bool:
        return self.state in {
            PipelineStageOutcomeState.VALID,
            PipelineStageOutcomeState.VALID_WITH_DIAGNOSTICS,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "run_id": self.run_id,
            "page_id": self.page_id,
            "page_index": self.page_index,
            "page_name": self.page_name,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "stage": self.stage.value,
            "state": self.state.value,
            "valid_for_next_stage": self.valid_for_next_stage,
            "parent_ids": list(self.parent_ids),
            "artifact_kind": self.artifact_kind,
            "artifact_summary": _json_safe(self.artifact_summary),
            "diagnostics": list(self.diagnostics),
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


class PipelineStageTechnicalError(RuntimeError):
    """Raised only when a required stage cannot produce a valid artifact."""

    def __init__(
        self,
        *,
        stage: PipelineStage,
        code: str,
        message: str,
        detail: str = "",
        page_id: str = "",
        parent_id: str = "",
        operation: str = "",
        artifact_summary: Mapping[str, Any] | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(stage, PipelineStage):
            raise TypeError("stage must be a PipelineStage")
        _require_text(code, "code")
        _require_text(message, "message")
        super().__init__(message)
        self.stage = stage
        self.code = str(code)
        self.message = str(message)
        self.detail = str(detail or "")
        self.page_id = str(page_id or "")
        self.parent_id = str(parent_id or "")
        self.operation = str(operation or "")
        self.artifact_summary = dict(artifact_summary or {})
        self.diagnostics = tuple(str(value) for value in diagnostics if str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class PipelineProgressSnapshot:
    run_id: str
    completed_pages: int
    total_pages: int
    percent: int
    stage: PipelineStage
    eta_seconds: float | None = None
    current_page_id: str = ""
    current_parent_id: str = ""
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if isinstance(self.completed_pages, bool) or not isinstance(
            self.completed_pages, int
        ):
            raise TypeError("completed_pages must be an integer")
        if isinstance(self.total_pages, bool) or not isinstance(self.total_pages, int):
            raise TypeError("total_pages must be an integer")
        if self.completed_pages < 0 or self.total_pages < 0:
            raise ValueError("page counts cannot be negative")
        if self.completed_pages > self.total_pages:
            raise ValueError("completed_pages cannot exceed total_pages")
        if isinstance(self.percent, bool) or not isinstance(self.percent, int):
            raise TypeError("percent must be an integer")
        if not 0 <= self.percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        if not isinstance(self.stage, PipelineStage):
            raise TypeError("stage must be a PipelineStage")
        if self.eta_seconds is not None:
            if isinstance(self.eta_seconds, bool) or not isinstance(
                self.eta_seconds, (int, float)
            ):
                raise TypeError("eta_seconds must be a number or None")
            if self.eta_seconds < 0:
                raise ValueError("eta_seconds cannot be negative")
        _require_optional_text(self.current_page_id, "current_page_id")
        _require_optional_text(self.current_parent_id, "current_parent_id")
        _require_text(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class PipelineErrorReceipt:
    error_id: str
    run_id: str
    code: str
    owner_stage: PipelineStage
    message: str
    detail: str = ""
    page_id: str = ""
    parent_id: str = ""
    recoverable: bool = False
    retry_action: PipelineRetryAction = PipelineRetryAction.NONE
    project_id: str = ""
    operation: str = ""
    revision: str = ""
    prior_state_safe: bool = True
    timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        _require_text(self.error_id, "error_id")
        _require_text(self.run_id, "run_id")
        _require_text(self.code, "code")
        if not isinstance(self.owner_stage, PipelineStage):
            raise TypeError("owner_stage must be a PipelineStage")
        _require_text(self.message, "message")
        for field_name in (
            "detail",
            "page_id",
            "parent_id",
            "project_id",
            "operation",
            "revision",
        ):
            _require_optional_text(getattr(self, field_name), field_name)
        if not isinstance(self.recoverable, bool):
            raise TypeError("recoverable must be a boolean")
        if not isinstance(self.retry_action, PipelineRetryAction):
            raise TypeError("retry_action must be a PipelineRetryAction")
        if not isinstance(self.prior_state_safe, bool):
            raise TypeError("prior_state_safe must be a boolean")
        _require_text(self.timestamp, "timestamp")


__all__ = [
    "PipelineErrorReceipt",
    "PipelineLifecycleEvent",
    "PipelineProgressSnapshot",
    "PipelineRetryAction",
    "PipelineRunState",
    "PipelineStage",
    "PipelineStageEvent",
    "PipelineStageOutcome",
    "PipelineStageOutcomeState",
    "PipelineStageTechnicalError",
    "RuntimeBackendEvent",
    "new_error_id",
    "new_run_id",
    "new_stage_outcome_id",
    "utc_timestamp",
]
