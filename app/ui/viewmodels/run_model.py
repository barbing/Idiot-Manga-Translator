# -*- coding: utf-8 -*-
"""Typed GUI-5 run, error, and read-only runtime presentation models."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.config.settings_contracts import (
    ProviderHealth,
    RunSettingsSnapshot,
    RuntimeStatus,
)
from app.pipeline.status_contracts import (
    PipelineErrorReceipt,
    PipelineLifecycleEvent,
    PipelineProgressSnapshot,
    PipelineRunState,
    PipelineStage,
    PipelineStageEvent,
    PipelineStageOutcome,
)
from app.ui.ui_contract import NAVIGATION_IDS, PresentationTone
from app.ui.viewmodels.project_model import TypedListModelBase


_USER_ROLE = 256
SETTINGS_ROUTE_ID = "settings"
if SETTINGS_ROUTE_ID not in NAVIGATION_IDS:  # Defensive contract check.
    raise RuntimeError("the UI contract must retain the Settings route")


def _clean_required(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _clean_optional(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


def _enum_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


_BUSY_RUN_STATES = frozenset(
    {
        PipelineRunState.VALIDATING,
        PipelineRunState.RUNNING,
        PipelineRunState.STOP_REQUESTED,
        PipelineRunState.CANCELLING,
    }
)
_TERMINAL_RUN_STATES = frozenset(
    {
        PipelineRunState.STOPPED,
        PipelineRunState.COMPLETED,
        PipelineRunState.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class RunViewState:
    lifecycle: PipelineLifecycleEvent | None = None
    stage: PipelineStageEvent | None = None
    progress: PipelineProgressSnapshot | None = None
    errors: tuple[PipelineErrorReceipt, ...] = ()
    stage_outcomes: tuple[PipelineStageOutcome, ...] = ()

    def __post_init__(self) -> None:
        for field_name, expected_type in (
            ("lifecycle", PipelineLifecycleEvent),
            ("stage", PipelineStageEvent),
            ("progress", PipelineProgressSnapshot),
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__} or None")
        errors = tuple(self.errors)
        if any(not isinstance(error, PipelineErrorReceipt) for error in errors):
            raise TypeError("errors must contain PipelineErrorReceipt values")
        if len({error.error_id for error in errors}) != len(errors):
            raise ValueError("errors must have unique identities")
        object.__setattr__(self, "errors", errors)
        outcomes = tuple(self.stage_outcomes)
        if any(not isinstance(outcome, PipelineStageOutcome) for outcome in outcomes):
            raise TypeError("stage_outcomes must contain PipelineStageOutcome values")
        if len({outcome.outcome_id for outcome in outcomes}) != len(outcomes):
            raise ValueError("stage outcomes must have unique identities")
        object.__setattr__(self, "stage_outcomes", outcomes)
        run_ids = {
            value.run_id
            for value in (self.lifecycle, self.stage, self.progress, *errors, *outcomes)
            if value is not None
        }
        if len(run_ids) > 1:
            raise ValueError("run state cannot combine records from different runs")

    @property
    def run_id(self) -> str:
        for value in (self.lifecycle, self.stage, self.progress):
            if value is not None:
                return value.run_id
        return self.errors[0].run_id if self.errors else ""

    @property
    def busy(self) -> bool:
        return bool(self.lifecycle and self.lifecycle.state in _BUSY_RUN_STATES)

    @property
    def active_error(self) -> PipelineErrorReceipt | None:
        return self.errors[-1] if self.errors else None


class PipelineRunProgressModel:
    """Framework-neutral reducer for typed controller status signals."""

    def __init__(self, state: RunViewState | None = None) -> None:
        self._state = state or RunViewState()
        if not isinstance(self._state, RunViewState):
            raise TypeError("state must be RunViewState or None")

    @property
    def state(self) -> RunViewState:
        return self._state

    def reset(self) -> RunViewState:
        """Clear lifecycle, stage, progress, and errors for a project boundary."""

        self._state = RunViewState()
        return self._state

    def _require_run(self, run_id: str) -> None:
        if self._state.run_id and self._state.run_id != run_id:
            raise ValueError("event belongs to another run")

    def apply_lifecycle(self, event: PipelineLifecycleEvent) -> RunViewState:
        if not isinstance(event, PipelineLifecycleEvent):
            raise TypeError("event must be PipelineLifecycleEvent")
        current = self._state.lifecycle
        if current is not None and current.run_id != event.run_id:
            if current.state not in _TERMINAL_RUN_STATES:
                raise ValueError("cannot replace an active run with another run ID")
            self._state = RunViewState(lifecycle=event)
        else:
            self._state = replace(self._state, lifecycle=event)
        return self._state

    def apply_stage(self, event: PipelineStageEvent) -> RunViewState:
        if not isinstance(event, PipelineStageEvent):
            raise TypeError("event must be PipelineStageEvent")
        self._require_run(event.run_id)
        self._state = replace(self._state, stage=event)
        return self._state

    def apply_progress(self, snapshot: PipelineProgressSnapshot) -> RunViewState:
        if not isinstance(snapshot, PipelineProgressSnapshot):
            raise TypeError("snapshot must be PipelineProgressSnapshot")
        self._require_run(snapshot.run_id)
        self._state = replace(self._state, progress=snapshot)
        return self._state

    def apply_error(self, receipt: PipelineErrorReceipt) -> RunViewState:
        if not isinstance(receipt, PipelineErrorReceipt):
            raise TypeError("receipt must be PipelineErrorReceipt")
        self._require_run(receipt.run_id)
        if any(error.error_id == receipt.error_id for error in self._state.errors):
            raise ValueError("error receipt identity is duplicated")
        self._state = replace(self._state, errors=(*self._state.errors, receipt))
        return self._state

    def apply_stage_outcome(self, outcome: PipelineStageOutcome) -> RunViewState:
        if not isinstance(outcome, PipelineStageOutcome):
            raise TypeError("outcome must be PipelineStageOutcome")
        self._require_run(outcome.run_id)
        retained = tuple(
            value
            for value in self._state.stage_outcomes
            if not (
                value.page_id == outcome.page_id
                and value.stage is outcome.stage
            )
        )
        self._state = replace(
            self._state,
            stage_outcomes=(*retained, outcome),
        )
        return self._state

    def clear_errors(self) -> RunViewState:
        self._state = replace(self._state, errors=())
        return self._state


_STAGE_LABELS: Mapping[PipelineStage, str] = {
    PipelineStage.IDLE: "Idle",
    PipelineStage.VALIDATION: "Validation",
    PipelineStage.INITIALIZATION: "Initialization",
    PipelineStage.PRESCAN: "Pre-scan",
    PipelineStage.DETECTION: "Text detection",
    PipelineStage.OCR: "OCR",
    PipelineStage.HIERARCHY: "Text hierarchy",
    PipelineStage.TRANSLATION: "Translation",
    PipelineStage.SOURCE_GLYPH: "Source style",
    PipelineStage.CLEANUP: "Cleanup",
    PipelineStage.STYLE: "Style",
    PipelineStage.RENDERING: "Rendering",
    PipelineStage.PERSISTENCE: "Checkpoint",
    PipelineStage.FINALIZING: "Finalizing",
}


def pipeline_stage_label(stage: PipelineStage) -> str:
    if not isinstance(stage, PipelineStage):
        raise TypeError("stage must be PipelineStage")
    return _STAGE_LABELS[stage]


@dataclass(frozen=True, slots=True)
class RuntimeBackendRow:
    module_id: str
    module_name: str
    backend: str
    detail: str
    device: str
    status: str
    tone: PresentationTone
    run_snapshot_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "module_id",
            "module_name",
            "backend",
            "detail",
            "device",
            "status",
            "run_snapshot_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_required(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "tone", PresentationTone(self.tone))

    @property
    def stable_id(self) -> str:
        return self.module_id

    @property
    def accessibility_text(self) -> str:
        return (
            f"{self.module_name}. {self.backend}. {self.detail}. "
            f"{self.device}. {self.status}."
        )


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    run_snapshot_id: str
    settings_fingerprint: str
    captured_at: str
    status: str
    tone: PresentationTone
    rows: tuple[RuntimeBackendRow, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "run_snapshot_id",
            "settings_fingerprint",
            "captured_at",
            "status",
        ):
            object.__setattr__(
                self,
                field_name,
                _clean_required(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "tone", PresentationTone(self.tone))
        rows = tuple(self.rows)
        if any(not isinstance(row, RuntimeBackendRow) for row in rows):
            raise TypeError("rows must contain RuntimeBackendRow values")
        if len({row.module_id for row in rows}) != len(rows):
            raise ValueError("runtime rows must have unique module IDs")
        if any(row.run_snapshot_id != self.run_snapshot_id for row in rows):
            raise ValueError("runtime row belongs to another run snapshot")
        object.__setattr__(self, "rows", rows)


def _runtime_health(
    runtime_status: RuntimeStatus | None,
    *,
    unresolved: bool,
) -> tuple[str, PresentationTone]:
    if unresolved:
        return "Configuration needs attention", PresentationTone.WARNING
    if runtime_status is None or runtime_status.provider_health is ProviderHealth.UNKNOWN:
        return "Backends captured from run settings", PresentationTone.INFO
    if runtime_status.provider_health is ProviderHealth.READY:
        return "All backends ready", PresentationTone.READY
    if runtime_status.provider_health is ProviderHealth.UNRESOLVED:
        return "Provider unresolved", PresentationTone.WARNING
    if runtime_status.provider_health is ProviderHealth.UNAVAILABLE:
        return "Provider unavailable", PresentationTone.ERROR
    return "Runtime error", PresentationTone.ERROR


def _translation_row_values(
    run_settings: RunSettingsSnapshot,
    runtime_status: RuntimeStatus | None,
) -> tuple[str, str, str, str, PresentationTone]:
    provider = run_settings.provider_profile_snapshot.get("translation")
    pipeline = run_settings.pipeline_values
    if not isinstance(provider, Mapping) or provider.get("status") == "unresolved":
        return (
            "Unresolved provider",
            "Relink or select a provider in Settings",
            "Unavailable",
            "Needs settings",
            PresentationTone.WARNING,
        )
    kind = str(provider.get("kind") or pipeline.get("translator_backend") or "Provider")
    backend = str(provider.get("display_name") or kind)
    model = str(
        provider.get("model_id")
        or Path(str(provider.get("local_model_path") or "")).name
        or "Default model"
    )
    kind_key = kind.lower().replace("_", "-")
    device = "Network" if kind_key in {"deepseek", "openai-compatible"} else "Local"
    health = runtime_status.provider_health if runtime_status is not None else ProviderHealth.UNKNOWN
    if health is ProviderHealth.READY:
        status, tone = "Connected", PresentationTone.READY
    elif health in {ProviderHealth.UNAVAILABLE, ProviderHealth.ERROR}:
        status, tone = "Unavailable", PresentationTone.ERROR
    elif health is ProviderHealth.UNRESOLVED:
        status, tone = "Unresolved", PresentationTone.WARNING
    else:
        status, tone = "Captured", PresentationTone.INFO
    return backend, model, device, status, tone


def build_runtime_snapshot(
    run_settings: RunSettingsSnapshot,
    runtime_status: RuntimeStatus | None = None,
) -> RuntimeSnapshot:
    """Capture immutable active-run backends; Settings drafts are not inputs."""

    if not isinstance(run_settings, RunSettingsSnapshot):
        raise TypeError("run_settings must be RunSettingsSnapshot")
    if runtime_status is not None and not isinstance(runtime_status, RuntimeStatus):
        raise TypeError("runtime_status must be RuntimeStatus or None")
    values = run_settings.pipeline_values
    accelerated_device = "CUDA" if bool(values.get("use_gpu", False)) else "CPU"
    translation = _translation_row_values(run_settings, runtime_status)
    status, tone = _runtime_health(
        runtime_status,
        unresolved=bool(run_settings.unresolved_requirements),
    )
    snapshot_id = run_settings.snapshot_id
    rows = (
        RuntimeBackendRow(
            module_id="text-detection",
            module_name="Text detection",
            backend=str(values.get("detector_engine") or "Automatic detector"),
            detail="Text regions and foreground masks",
            device=accelerated_device,
            status="Ready",
            tone=PresentationTone.READY,
            run_snapshot_id=snapshot_id,
        ),
        RuntimeBackendRow(
            module_id="source-style-observation",
            module_name="Source style",
            backend=str(values.get("font_detection") or "Automatic source style"),
            detail="Font and style observation",
            device=accelerated_device,
            status="Ready",
            tone=PresentationTone.READY,
            run_snapshot_id=snapshot_id,
        ),
        RuntimeBackendRow(
            module_id="ocr",
            module_name="OCR",
            backend=str(values.get("ocr_engine") or "Automatic OCR"),
            detail=str(values.get("source_lang") or "Source language"),
            device=accelerated_device,
            status="Ready",
            tone=PresentationTone.READY,
            run_snapshot_id=snapshot_id,
        ),
        RuntimeBackendRow(
            module_id="translation",
            module_name="Translation",
            backend=translation[0],
            detail=translation[1],
            device=translation[2],
            status=translation[3],
            tone=translation[4],
            run_snapshot_id=snapshot_id,
        ),
        RuntimeBackendRow(
            module_id="cleanup",
            module_name="Cleanup",
            backend=str(
                values.get("inpaint_model_id")
                or values.get("inpaint_mode")
                or "Automatic cleanup"
            ),
            detail="Local inpainting",
            device=accelerated_device,
            status="Ready",
            tone=PresentationTone.READY,
            run_snapshot_id=snapshot_id,
        ),
        RuntimeBackendRow(
            module_id="rendering",
            module_name="Rendering",
            backend="YomiFrame renderer",
            detail=str(values.get("font_name") or "Project typography"),
            device="CPU",
            status="Ready",
            tone=PresentationTone.READY,
            run_snapshot_id=snapshot_id,
        ),
    )
    return RuntimeSnapshot(
        run_snapshot_id=snapshot_id,
        settings_fingerprint=run_settings.settings_fingerprint,
        captured_at=run_settings.created_at,
        status=status,
        tone=tone,
        rows=rows,
    )


class RuntimeRole(IntEnum):
    STABLE_ID = _USER_ROLE + 1
    MODULE_NAME = _USER_ROLE + 2
    BACKEND = _USER_ROLE + 3
    DETAIL = _USER_ROLE + 4
    DEVICE = _USER_ROLE + 5
    STATUS = _USER_ROLE + 6
    STATUS_TONE = _USER_ROLE + 7
    RUN_SNAPSHOT_ID = _USER_ROLE + 8
    ACCESSIBILITY_TEXT = _USER_ROLE + 9


class RuntimeStatusModel(TypedListModelBase):
    """Read-only module rows captured from one immutable active run."""

    settings_route_id = SETTINGS_ROUTE_ID
    _row_type = RuntimeBackendRow
    _role_names = {
        RuntimeRole.STABLE_ID: b"stableId",
        RuntimeRole.MODULE_NAME: b"moduleName",
        RuntimeRole.BACKEND: b"backend",
        RuntimeRole.DETAIL: b"detail",
        RuntimeRole.DEVICE: b"device",
        RuntimeRole.STATUS: b"status",
        RuntimeRole.STATUS_TONE: b"statusTone",
        RuntimeRole.RUN_SNAPSHOT_ID: b"runSnapshotId",
        RuntimeRole.ACCESSIBILITY_TEXT: b"accessibilityText",
    }
    _role_accessors = {
        RuntimeRole.STABLE_ID: lambda row: row.module_id,
        RuntimeRole.MODULE_NAME: lambda row: row.module_name,
        RuntimeRole.BACKEND: lambda row: row.backend,
        RuntimeRole.DETAIL: lambda row: row.detail,
        RuntimeRole.DEVICE: lambda row: row.device,
        RuntimeRole.STATUS: lambda row: row.status,
        RuntimeRole.STATUS_TONE: lambda row: row.tone.value,
        RuntimeRole.RUN_SNAPSHOT_ID: lambda row: row.run_snapshot_id,
        RuntimeRole.ACCESSIBILITY_TEXT: lambda row: row.accessibility_text,
    }

    def __init__(self, snapshot: RuntimeSnapshot | None = None, parent: Any = None) -> None:
        self._snapshot: RuntimeSnapshot | None = None
        super().__init__((), parent)
        if snapshot is not None:
            self.replace_snapshot(snapshot)

    @property
    def snapshot(self) -> RuntimeSnapshot | None:
        return self._snapshot

    @staticmethod
    def _display_value(row: RuntimeBackendRow) -> str:
        return row.module_name

    def replace_snapshot(self, snapshot: RuntimeSnapshot | None) -> None:
        if snapshot is not None and not isinstance(snapshot, RuntimeSnapshot):
            raise TypeError("snapshot must be RuntimeSnapshot or None")
        self._snapshot = snapshot
        super().replace_rows(snapshot.rows if snapshot is not None else ())

    def replace_rows(self, rows: Iterable[RuntimeBackendRow]) -> None:
        """Reject display-only mutations outside an immutable run snapshot."""

        normalized = tuple(rows)
        if normalized:
            raise RuntimeError("runtime rows must be replaced through RuntimeSnapshot")
        super().replace_rows(())

    def capture_run(
        self,
        run_settings: RunSettingsSnapshot,
        runtime_status: RuntimeStatus | None = None,
    ) -> RuntimeSnapshot:
        snapshot = build_runtime_snapshot(run_settings, runtime_status)
        self.replace_snapshot(snapshot)
        return snapshot


@dataclass(frozen=True, slots=True)
class StructuredErrorRow:
    receipt: PipelineErrorReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, PipelineErrorReceipt):
            raise TypeError("receipt must be PipelineErrorReceipt")

    @property
    def stable_id(self) -> str:
        return self.receipt.error_id

    @property
    def accessibility_text(self) -> str:
        location = ", ".join(
            value
            for value in (self.receipt.page_id, self.receipt.parent_id)
            if value
        )
        prefix = f"{location}. " if location else ""
        return (
            f"{prefix}{pipeline_stage_label(self.receipt.owner_stage)} error. "
            f"{self.receipt.message}"
        )


class StructuredErrorRole(IntEnum):
    STABLE_ID = _USER_ROLE + 1
    CODE = _USER_ROLE + 2
    OWNER_STAGE = _USER_ROLE + 3
    MESSAGE = _USER_ROLE + 4
    DETAIL = _USER_ROLE + 5
    PAGE_ID = _USER_ROLE + 6
    PARENT_ID = _USER_ROLE + 7
    RECOVERABLE = _USER_ROLE + 8
    RETRY_ACTION = _USER_ROLE + 9
    PRIOR_STATE_SAFE = _USER_ROLE + 10
    ACCESSIBILITY_TEXT = _USER_ROLE + 11


class StructuredErrorModel(TypedListModelBase):
    _row_type = StructuredErrorRow
    _role_names = {
        StructuredErrorRole.STABLE_ID: b"stableId",
        StructuredErrorRole.CODE: b"code",
        StructuredErrorRole.OWNER_STAGE: b"ownerStage",
        StructuredErrorRole.MESSAGE: b"message",
        StructuredErrorRole.DETAIL: b"detail",
        StructuredErrorRole.PAGE_ID: b"pageId",
        StructuredErrorRole.PARENT_ID: b"parentId",
        StructuredErrorRole.RECOVERABLE: b"recoverable",
        StructuredErrorRole.RETRY_ACTION: b"retryAction",
        StructuredErrorRole.PRIOR_STATE_SAFE: b"priorStateSafe",
        StructuredErrorRole.ACCESSIBILITY_TEXT: b"accessibilityText",
    }
    _role_accessors = {
        StructuredErrorRole.STABLE_ID: lambda row: row.receipt.error_id,
        StructuredErrorRole.CODE: lambda row: row.receipt.code,
        StructuredErrorRole.OWNER_STAGE: lambda row: row.receipt.owner_stage.value,
        StructuredErrorRole.MESSAGE: lambda row: row.receipt.message,
        StructuredErrorRole.DETAIL: lambda row: row.receipt.detail,
        StructuredErrorRole.PAGE_ID: lambda row: row.receipt.page_id,
        StructuredErrorRole.PARENT_ID: lambda row: row.receipt.parent_id,
        StructuredErrorRole.RECOVERABLE: lambda row: row.receipt.recoverable,
        StructuredErrorRole.RETRY_ACTION: lambda row: row.receipt.retry_action.value,
        StructuredErrorRole.PRIOR_STATE_SAFE: lambda row: row.receipt.prior_state_safe,
        StructuredErrorRole.ACCESSIBILITY_TEXT: lambda row: row.accessibility_text,
    }

    @staticmethod
    def _display_value(row: StructuredErrorRow) -> str:
        return row.receipt.message

    def replace_receipts(self, receipts: Iterable[PipelineErrorReceipt]) -> None:
        self.replace_rows(StructuredErrorRow(receipt) for receipt in receipts)


__all__ = [
    "PipelineRunProgressModel",
    "RunViewState",
    "RuntimeBackendRow",
    "RuntimeRole",
    "RuntimeSnapshot",
    "RuntimeStatusModel",
    "SETTINGS_ROUTE_ID",
    "StructuredErrorModel",
    "StructuredErrorRole",
    "StructuredErrorRow",
    "build_runtime_snapshot",
    "pipeline_stage_label",
]
