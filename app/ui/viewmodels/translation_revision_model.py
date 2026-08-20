# -*- coding: utf-8 -*-
"""Framework-neutral UI state for one explicit parent translation revision.

The model keeps four target provenance lanes separate:

* Automatic target belongs to immutable automatic-parent evidence.
* A user-triggered translation result is a selected model revision.
* A manual target replacement remains a user edit.
* Effective target is the deterministic projected value.

The explicit translation result must never be relabelled as either Automatic
target or Your edit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import re
from typing import Any, Mapping, TYPE_CHECKING
import uuid

from app.config.run_settings_compiler import RuntimeProviderBinding
from app.config.settings_contracts import (
    RunSettingsSnapshot,
    canonical_fingerprint,
    freeze_json,
    thaw_json,
)
from app.pipeline.hierarchy_revision_contracts import (
    ParentStageRequirement,
    RevisionRequiredAction,
    RevisionStage,
    RevisionStageState,
    validate_user_parent_identity_pair,
)
from app.pipeline.translation_revision_contracts import (
    ExplicitTranslationRevisionReceipt,
    TranslationProviderSelection,
    TranslationRevisionArtifact,
    translation_context_fingerprint,
    translation_glossary_fingerprint,
    translation_policy_region_type,
)

if TYPE_CHECKING:
    from app.ui.shell.project_projection import ProjectUiProjection


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _required_identity(value: Any, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} is required")
    return candidate


def _required_path_safe_identity(value: Any, field_name: str) -> str:
    candidate = _required_identity(value, field_name)
    if _PATH_SAFE_ID.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be path-safe")
    return candidate


def _required_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "").lower()
    if _SHA256.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return candidate


def _stage_requirement(
    requirements: tuple[ParentStageRequirement, ...],
    stage: RevisionStage,
) -> ParentStageRequirement:
    matches = tuple(value for value in requirements if value.stage is stage)
    if len(matches) != 1:
        raise ValueError(f"selection requires one exact {stage.value} requirement")
    return matches[0]


class TranslationRevisionPhase(str, Enum):
    READY = "ready"
    RUNNING = "running"
    CANCEL_DEFERRED = "cancel_deferred"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class TranslationRevisionWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    VALIDATING_SELECTION = "validating_selection"
    PREPARING_REQUEST = "preparing_request"
    RESOLVING_CREDENTIAL = "resolving_credential"
    INITIALIZING_OWNER = "initializing_owner"
    TRANSLATING = "translating"
    DISCARDING_CANCELLED_RESULT = "discarding_cancelled_result"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    CLOSING_OWNER = "closing_owner"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class TranslationRevisionCancellationMode(str, Enum):
    AVAILABLE = "available"
    REQUESTED_DEFERRED = "requested_deferred"
    LOCKED = "locked"
    UNAVAILABLE = "unavailable"


class TranslationRevisionFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    SNAPSHOT_STALE = "snapshot_stale"
    SETTINGS_STALE = "settings_stale"
    SOURCE_NOT_CURRENT = "source_not_current"
    SOURCE_MISMATCH = "source_mismatch"
    GLOSSARY_STALE = "glossary_stale"
    CONTEXT_STALE = "context_stale"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    TRANSLATION_FAILED = "translation_failed"
    EMPTY_RESULT = "empty_result"
    PERSISTENCE_REJECTED = "persistence_rejected"
    PROJECTION_FAILED = "projection_failed"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    COMMITTED_STALE = "committed_stale"
    WORKER_REUSED = "worker_reused"
    COMMAND_REJECTED = "command_rejected"


@dataclass(frozen=True, slots=True)
class TranslationRevisionSelection:
    """Exact selected-parent state from which translation may start."""

    project_path: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    policy_region_type: str
    bubble_local_nested_speech: bool
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    effective_source_text: str
    effective_source_authority: str
    effective_source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    run_settings_snapshot: RunSettingsSnapshot
    runtime_provider_binding: RuntimeProviderBinding
    provider: TranslationProviderSelection
    glossary_snapshot: Mapping[str, Any]
    glossary_fingerprint: str
    prior_page_context: tuple[str, ...]
    context_fingerprint: str
    stage_requirements: tuple[ParentStageRequirement, ...]
    model_translation_revision: TranslationRevisionArtifact | None = None
    user_target_text: str | None = None
    effective_target_text: str | None = None
    effective_target_authority: str = "unavailable"
    available: bool = True
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        project_path = str(self.project_path or "").strip()
        if not project_path:
            raise ValueError("project_path is required")
        object.__setattr__(self, "project_path", os.path.abspath(project_path))
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "hierarchy_revision_id",
            "source_revision_id",
            "source_selection_edit_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        role = str(self.parent_role or "").strip().casefold()
        expected_region = translation_policy_region_type(role)
        if self.policy_region_type != expected_region:
            raise ValueError("policy_region_type does not match parent_role")
        if not isinstance(self.bubble_local_nested_speech, bool):
            raise TypeError("bubble_local_nested_speech must be a bool")
        object.__setattr__(self, "parent_role", role)
        for field_name in (
            "effective_page_fingerprint",
            "hierarchy_fingerprint",
            "effective_source_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if (
            not isinstance(self.effective_source_text, str)
            or not self.effective_source_text.strip()
        ):
            raise ValueError("effective_source_text must be non-empty")
        if self.effective_source_authority != "ocr_revision":
            raise ValueError(
                "explicit translation requires selected model OCR source authority"
            )
        expected_source_fingerprint = canonical_fingerprint(
            {"parent_id": self.parent_id, "text": self.effective_source_text}
        )
        if self.effective_source_fingerprint != expected_source_fingerprint:
            raise ValueError("effective source fingerprint does not match its text")
        if not isinstance(self.run_settings_snapshot, RunSettingsSnapshot):
            raise TypeError("run_settings_snapshot must be RunSettingsSnapshot")
        if self.run_settings_snapshot.project_id != self.project_id:
            raise ValueError("run settings belong to another project")
        if not isinstance(self.runtime_provider_binding, RuntimeProviderBinding):
            raise TypeError(
                "runtime_provider_binding must be RuntimeProviderBinding"
            )
        provider = (
            self.provider
            if isinstance(self.provider, TranslationProviderSelection)
            else TranslationProviderSelection.from_dict(self.provider)
        )
        expected_provider = TranslationProviderSelection.from_run_settings_snapshot(
            self.run_settings_snapshot
        )
        if provider != expected_provider:
            raise ValueError("provider differs from the immutable run snapshot")
        runtime_kind = self.runtime_provider_binding.provider_kind
        runtime_kind_value = (
            str(getattr(runtime_kind, "value", runtime_kind) or "")
            .casefold()
            .replace("_", "-")
        )
        if (
            self.runtime_provider_binding.profile_id != provider.profile_id
            or runtime_kind_value != provider.provider_kind
        ):
            raise ValueError("runtime provider binding differs from public selection")
        object.__setattr__(self, "provider", provider)
        frozen_glossary = freeze_json(
            self.glossary_snapshot,
            field_name="glossary_snapshot",
        )
        if (
            translation_glossary_fingerprint(thaw_json(frozen_glossary))
            != self.glossary_fingerprint
        ):
            raise ValueError("glossary fingerprint does not match its snapshot")
        object.__setattr__(self, "glossary_snapshot", frozen_glossary)
        context = tuple(self.prior_page_context)
        if any(not isinstance(line, str) for line in context):
            raise TypeError("prior_page_context must contain strings")
        if len(context) > 4:
            raise ValueError("prior_page_context cannot exceed four lines")
        if translation_context_fingerprint(context) != self.context_fingerprint:
            raise ValueError("context fingerprint does not match its snapshot")
        object.__setattr__(self, "prior_page_context", context)
        requirements = tuple(self.stage_requirements)
        if any(not isinstance(value, ParentStageRequirement) for value in requirements):
            raise TypeError("stage_requirements must contain ParentStageRequirement")
        if any(value.parent_id != self.parent_id for value in requirements):
            raise ValueError("stage requirements belong to another parent")
        _stage_requirement(requirements, RevisionStage.SOURCE)
        _stage_requirement(requirements, RevisionStage.TRANSLATION)
        object.__setattr__(self, "stage_requirements", requirements)
        artifact = self.model_translation_revision
        if artifact is not None and not isinstance(
            artifact,
            TranslationRevisionArtifact,
        ):
            raise TypeError(
                "model_translation_revision must be TranslationRevisionArtifact or None"
            )
        for field_name in ("user_target_text", "effective_target_text"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
        if self.effective_target_authority not in {
            "translation_revision",
            "user",
            "unavailable",
        }:
            raise ValueError("effective target authority is unsupported")
        if self.effective_target_authority == "unavailable":
            if (
                artifact is not None
                or self.user_target_text is not None
                or self.effective_target_text is not None
            ):
                raise ValueError("unavailable target cannot expose target evidence")
        elif self.effective_target_authority == "translation_revision":
            if (
                artifact is None
                or self.user_target_text is not None
                or self.effective_target_text != artifact.target_text
            ):
                raise ValueError("selected model translation provenance is inconsistent")
        else:
            if (
                self.user_target_text is None
                or self.effective_target_text != self.user_target_text
            ):
                raise ValueError("user target authority requires one exact edit")
        if artifact is not None and (
            artifact.project_id != self.project_id
            or artifact.page_id != self.page_id
            or artifact.parent_id != self.parent_id
            or artifact.root_id != self.root_id
            or artifact.parent_authored_edit_id != self.parent_authored_edit_id
            or artifact.parent_role != self.parent_role
            or artifact.policy_region_type != self.policy_region_type
            or artifact.bubble_local_nested_speech
            != self.bubble_local_nested_speech
            or artifact.source_text != self.effective_source_text
            or artifact.source_authority != self.effective_source_authority
            or artifact.source_fingerprint != self.effective_source_fingerprint
            or artifact.source_revision_id != self.source_revision_id
            or artifact.source_selection_edit_id != self.source_selection_edit_id
            or artifact.run_settings_fingerprint != self.run_settings_fingerprint
            or artifact.provider != self.provider
            or artifact.glossary_fingerprint != self.glossary_fingerprint
            or artifact.context_fingerprint != self.context_fingerprint
            or artifact.hierarchy_revision_id != self.hierarchy_revision_id
            or artifact.hierarchy_fingerprint != self.hierarchy_fingerprint
        ):
            raise ValueError("selected model translation lineage is inconsistent")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be a string")
        reason = self.unavailable_reason.strip()
        if self.available and reason:
            raise ValueError("available translation revision state cannot have a reason")
        if not self.available and not reason:
            raise ValueError("unavailable translation revision state requires a reason")
        object.__setattr__(self, "unavailable_reason", reason)

    @property
    def run_settings_fingerprint(self) -> str:
        return self.run_settings_snapshot.settings_fingerprint

    @property
    def source_requirement(self) -> ParentStageRequirement:
        return _stage_requirement(self.stage_requirements, RevisionStage.SOURCE)

    @property
    def translation_requirement(self) -> ParentStageRequirement:
        return _stage_requirement(
            self.stage_requirements,
            RevisionStage.TRANSLATION,
        )

    @property
    def source_current(self) -> bool:
        value = self.source_requirement
        return bool(
            value.state is RevisionStageState.CURRENT
            and value.required_action is RevisionRequiredAction.NONE
        )

    @property
    def translation_runnable(self) -> bool:
        value = self.translation_requirement
        return bool(
            value.state is RevisionStageState.MISSING
            and value.required_action is RevisionRequiredAction.EXPLICIT_RUN
        )

    @property
    def translation_current(self) -> bool:
        value = self.translation_requirement
        return bool(
            value.state is RevisionStageState.CURRENT
            and value.required_action is RevisionRequiredAction.NONE
        )


@dataclass(frozen=True, slots=True)
class TranslationRevisionCommandIdentity:
    operation_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    policy_region_type: str
    bubble_local_nested_speech: bool
    effective_page_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    effective_source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    run_settings_fingerprint: str
    provider: TranslationProviderSelection
    glossary_fingerprint: str
    context_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_path_safe_identity(self.operation_id, "operation_id"),
        )
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "hierarchy_revision_id",
            "source_revision_id",
            "source_selection_edit_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        role = str(self.parent_role or "").strip().casefold()
        if self.policy_region_type != translation_policy_region_type(role):
            raise ValueError("policy region type does not match parent role")
        if not isinstance(self.bubble_local_nested_speech, bool):
            raise TypeError("bubble_local_nested_speech must be a bool")
        object.__setattr__(self, "parent_role", role)
        for field_name in (
            "effective_page_fingerprint",
            "hierarchy_fingerprint",
            "effective_source_fingerprint",
            "run_settings_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if not isinstance(self.provider, TranslationProviderSelection):
            raise TypeError("provider must be TranslationProviderSelection")

    @classmethod
    def from_selection(
        cls,
        selection: TranslationRevisionSelection,
        *,
        operation_id: str | None = None,
    ) -> "TranslationRevisionCommandIdentity":
        return cls(
            operation_id=operation_id or uuid.uuid4().hex,
            project_id=selection.project_id,
            page_id=selection.page_id,
            parent_id=selection.parent_id,
            root_id=selection.root_id,
            parent_authored_edit_id=selection.parent_authored_edit_id,
            parent_role=selection.parent_role,
            policy_region_type=selection.policy_region_type,
            bubble_local_nested_speech=selection.bubble_local_nested_speech,
            effective_page_fingerprint=selection.effective_page_fingerprint,
            hierarchy_revision_id=selection.hierarchy_revision_id,
            hierarchy_fingerprint=selection.hierarchy_fingerprint,
            effective_source_fingerprint=selection.effective_source_fingerprint,
            source_revision_id=selection.source_revision_id,
            source_selection_edit_id=selection.source_selection_edit_id,
            run_settings_fingerprint=selection.run_settings_fingerprint,
            provider=selection.provider,
            glossary_fingerprint=selection.glossary_fingerprint,
            context_fingerprint=selection.context_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class TranslationRevisionWorkerCommand:
    identity: TranslationRevisionCommandIdentity
    selection: TranslationRevisionSelection

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TranslationRevisionCommandIdentity):
            raise TypeError("identity must be TranslationRevisionCommandIdentity")
        if not isinstance(self.selection, TranslationRevisionSelection):
            raise TypeError("selection must be TranslationRevisionSelection")
        expected = TranslationRevisionCommandIdentity.from_selection(
            self.selection,
            operation_id=self.identity.operation_id,
        )
        if self.identity != expected:
            raise ValueError("worker command identity differs from its selection")


@dataclass(frozen=True, slots=True)
class TranslationRevisionWorkerBusyState:
    identity: TranslationRevisionCommandIdentity
    busy: bool
    stage: TranslationRevisionWorkerStage
    cancellation_mode: TranslationRevisionCancellationMode
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class TranslationRevisionCancellationState:
    identity: TranslationRevisionCommandIdentity
    mode: TranslationRevisionCancellationMode
    stage: TranslationRevisionWorkerStage
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class TranslationRevisionCancelledReceipt:
    identity: TranslationRevisionCommandIdentity
    stage: TranslationRevisionWorkerStage
    inference_completed: bool
    message: str = (
        "Translation revision cancelled; no target revision or selection edit "
        "was published."
    )


@dataclass(frozen=True, slots=True)
class TranslationRevisionWorkerFailure:
    identity: TranslationRevisionCommandIdentity
    code: TranslationRevisionFailureCode
    stage: TranslationRevisionWorkerStage
    message: str
    exception_type: str = ""
    persistence_committed: bool = False
    core_receipt: ExplicitTranslationRevisionReceipt | None = None

    @property
    def stale(self) -> bool:
        return bool(
            self.code
            in {
                TranslationRevisionFailureCode.SNAPSHOT_STALE,
                TranslationRevisionFailureCode.SETTINGS_STALE,
                TranslationRevisionFailureCode.GLOSSARY_STALE,
                TranslationRevisionFailureCode.CONTEXT_STALE,
                TranslationRevisionFailureCode.POST_COMMIT_PROJECTION_FAILED,
                TranslationRevisionFailureCode.COMMITTED_STALE,
            }
            or self.persistence_committed
        )


@dataclass(frozen=True, slots=True)
class TranslationRevisionWorkerReceipt:
    """Identity-bound atomic shell refresh after a committed translation."""

    identity: TranslationRevisionCommandIdentity
    core_receipt: ExplicitTranslationRevisionReceipt
    project: Mapping[str, Any]
    projection: "ProjectUiProjection"
    selection: TranslationRevisionSelection

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TranslationRevisionCommandIdentity):
            raise TypeError("identity must be TranslationRevisionCommandIdentity")
        if not isinstance(self.core_receipt, ExplicitTranslationRevisionReceipt):
            raise TypeError(
                "core_receipt must be ExplicitTranslationRevisionReceipt"
            )
        if not isinstance(self.project, Mapping):
            raise TypeError("project must be a mapping")
        if not isinstance(self.selection, TranslationRevisionSelection):
            raise TypeError("selection must be TranslationRevisionSelection")

        from app.project_edits.fingerprints import canonical_sha256
        from app.ui.shell.project_projection import ProjectUiProjection

        if not isinstance(self.projection, ProjectUiProjection):
            raise TypeError("projection must be ProjectUiProjection")
        if canonical_sha256(self.project) != self.projection.source_project_fingerprint:
            raise ValueError("worker project mapping does not match its projection")
        receipt = self.core_receipt
        identity = self.identity
        if (
            receipt.command_id != identity.operation_id
            or receipt.project_id != identity.project_id
            or receipt.page_id != identity.page_id
            or receipt.parent_id != identity.parent_id
            or receipt.root_id != identity.root_id
            or receipt.parent_authored_edit_id != identity.parent_authored_edit_id
            or receipt.parent_role != identity.parent_role
            or receipt.policy_region_type != identity.policy_region_type
            or receipt.bubble_local_nested_speech
            != identity.bubble_local_nested_speech
            or receipt.before_effective_page_fingerprint
            != identity.effective_page_fingerprint
            or receipt.hierarchy_revision_id != identity.hierarchy_revision_id
            or receipt.hierarchy_fingerprint != identity.hierarchy_fingerprint
            or receipt.source_fingerprint != identity.effective_source_fingerprint
            or receipt.source_revision_id != identity.source_revision_id
            or receipt.source_selection_edit_id != identity.source_selection_edit_id
            or receipt.run_settings_fingerprint != identity.run_settings_fingerprint
            or receipt.provider != identity.provider
            or receipt.glossary_fingerprint != identity.glossary_fingerprint
            or receipt.context_fingerprint != identity.context_fingerprint
        ):
            raise ValueError(
                "translation receipt belongs to another selected-parent command"
            )
        after = self.selection
        artifact = after.model_translation_revision
        if (
            after.project_id != receipt.project_id
            or after.page_id != receipt.page_id
            or after.parent_id != receipt.parent_id
            or after.root_id != receipt.root_id
            or after.parent_authored_edit_id != receipt.parent_authored_edit_id
            or after.effective_page_fingerprint
            != receipt.after_effective_page_fingerprint
            or after.hierarchy_revision_id != receipt.hierarchy_revision_id
            or after.hierarchy_fingerprint != receipt.hierarchy_fingerprint
            or after.effective_source_text != receipt.source_text
            or after.effective_source_fingerprint != receipt.source_fingerprint
            or after.source_revision_id != receipt.source_revision_id
            or after.source_selection_edit_id != receipt.source_selection_edit_id
            or after.run_settings_fingerprint != receipt.run_settings_fingerprint
            or after.provider != receipt.provider
            or after.glossary_fingerprint != receipt.glossary_fingerprint
            or after.context_fingerprint != receipt.context_fingerprint
            or artifact is None
            or artifact.revision_id != receipt.translation_revision_id
            or artifact.selection_edit_id != receipt.selection_edit_id
            or artifact.target_text != receipt.target_text
            or after.effective_target_text != receipt.target_text
            or after.effective_target_authority != "translation_revision"
            or not after.source_current
            or not after.translation_current
        ):
            raise ValueError(
                "worker projection is not the committed translation revision"
            )
        if self.projection.metadata.project_id != receipt.project_id:
            raise ValueError("worker projection belongs to another project")


@dataclass(frozen=True, slots=True)
class TranslationRevisionState:
    selection: TranslationRevisionSelection
    phase: TranslationRevisionPhase
    message: str
    command: TranslationRevisionWorkerCommand | None = None
    busy_state: TranslationRevisionWorkerBusyState | None = None
    cancellation_state: TranslationRevisionCancellationState | None = None
    receipt: TranslationRevisionWorkerReceipt | None = None
    failure: TranslationRevisionWorkerFailure | None = None
    cancelled: TranslationRevisionCancelledReceipt | None = None

    @property
    def busy(self) -> bool:
        return self.phase in {
            TranslationRevisionPhase.RUNNING,
            TranslationRevisionPhase.CANCEL_DEFERRED,
        }

    @property
    def stale(self) -> bool:
        return self.phase is TranslationRevisionPhase.STALE

    @property
    def run_enabled(self) -> bool:
        return bool(
            self.selection.available
            and self.selection.source_current
            and self.selection.translation_runnable
            and not self.busy
            and not self.stale
            and self.command is None
        )

    @property
    def cancel_enabled(self) -> bool:
        return bool(
            self.busy
            and self.cancellation_state is not None
            and self.cancellation_state.mode
            is TranslationRevisionCancellationMode.AVAILABLE
        )

    @property
    def deferred_cancel(self) -> bool:
        return self.phase is TranslationRevisionPhase.CANCEL_DEFERRED

    @property
    def blocks_navigation(self) -> bool:
        return bool(self.busy or self.stale)

    @property
    def blocks_start_preview(self) -> bool:
        # This slice publishes only translation. Cleanup, style, eligibility,
        # layout, and output revisions remain explicit requirements.
        return True

    @property
    def status_tone(self) -> str:
        return {
            TranslationRevisionPhase.READY: "warning",
            TranslationRevisionPhase.RUNNING: "editing",
            TranslationRevisionPhase.CANCEL_DEFERRED: "warning",
            TranslationRevisionPhase.SUCCEEDED: "ready",
            TranslationRevisionPhase.CANCELLED: "muted",
            TranslationRevisionPhase.STALE: "warning",
            TranslationRevisionPhase.FAILED: "error",
            TranslationRevisionPhase.UNAVAILABLE: "warning",
        }[self.phase]


class TranslationRevisionModel:
    """UI-thread reducer for one explicit selected model translation."""

    def __init__(self, selection: TranslationRevisionSelection) -> None:
        if not isinstance(selection, TranslationRevisionSelection):
            raise TypeError("selection must be TranslationRevisionSelection")
        phase = (
            TranslationRevisionPhase.READY
            if selection.available
            else TranslationRevisionPhase.UNAVAILABLE
        )
        self._state = TranslationRevisionState(
            selection=selection,
            phase=phase,
            message=(
                self._ready_message(selection)
                if selection.available
                else selection.unavailable_reason
            ),
        )

    @property
    def state(self) -> TranslationRevisionState:
        return self._state

    def begin(
        self,
        operation_id: str | None = None,
    ) -> TranslationRevisionWorkerCommand:
        if not self._state.run_enabled:
            raise RuntimeError(
                "Retranslate Parent is not available for this selection"
            )
        identity = TranslationRevisionCommandIdentity.from_selection(
            self._state.selection,
            operation_id=operation_id,
        )
        command = TranslationRevisionWorkerCommand(
            identity=identity,
            selection=self._state.selection,
        )
        cancellation = TranslationRevisionCancellationState(
            identity=identity,
            mode=TranslationRevisionCancellationMode.AVAILABLE,
            stage=TranslationRevisionWorkerStage.LOADING_PROJECT,
            persistence_started=False,
            message="Translation can be cancelled before persistence.",
        )
        self._state = replace(
            self._state,
            phase=TranslationRevisionPhase.RUNNING,
            message="Preparing the selected model translation revision...",
            command=command,
            busy_state=None,
            cancellation_state=cancellation,
            receipt=None,
            failure=None,
            cancelled=None,
        )
        return command

    def accept_busy(
        self,
        value: TranslationRevisionWorkerBusyState,
    ) -> TranslationRevisionState:
        if not isinstance(value, TranslationRevisionWorkerBusyState):
            raise TypeError("value must be TranslationRevisionWorkerBusyState")
        self._require_identity(value.identity)
        deferred_cancel = (
            self._state.phase is TranslationRevisionPhase.CANCEL_DEFERRED
        )
        phase = self._state.phase
        if value.busy:
            phase = (
                TranslationRevisionPhase.CANCEL_DEFERRED
                if deferred_cancel
                or value.cancellation_mode
                is TranslationRevisionCancellationMode.REQUESTED_DEFERRED
                else TranslationRevisionPhase.RUNNING
            )
        self._state = replace(
            self._state,
            phase=phase,
            message=(
                self._state.message
                if deferred_cancel
                else value.message or self._state.message
            ),
            busy_state=value,
        )
        return self._state

    def accept_cancellation(
        self,
        value: TranslationRevisionCancellationState,
    ) -> TranslationRevisionState:
        if not isinstance(value, TranslationRevisionCancellationState):
            raise TypeError("value must be TranslationRevisionCancellationState")
        self._require_identity(value.identity)
        deferred_cancel = (
            self._state.phase is TranslationRevisionPhase.CANCEL_DEFERRED
        )
        cancellation_state = value
        if deferred_cancel and self._state.cancellation_state is not None:
            progress = {
                TranslationRevisionCancellationMode.AVAILABLE: 0,
                TranslationRevisionCancellationMode.REQUESTED_DEFERRED: 1,
                TranslationRevisionCancellationMode.LOCKED: 2,
                TranslationRevisionCancellationMode.UNAVAILABLE: 3,
            }
            if progress[value.mode] < progress[self._state.cancellation_state.mode]:
                cancellation_state = self._state.cancellation_state
        phase = (
            TranslationRevisionPhase.CANCEL_DEFERRED
            if deferred_cancel
            or value.mode
            is TranslationRevisionCancellationMode.REQUESTED_DEFERRED
            else self._state.phase
        )
        self._state = replace(
            self._state,
            phase=phase,
            message=(
                self._state.message
                if deferred_cancel
                else value.message or self._state.message
            ),
            cancellation_state=cancellation_state,
        )
        return self._state

    def mark_cancel_requested(self) -> TranslationRevisionState:
        command = self._require_command()
        stage = (
            self._state.busy_state.stage
            if self._state.busy_state is not None
            else TranslationRevisionWorkerStage.LOADING_PROJECT
        )
        cancellation = TranslationRevisionCancellationState(
            identity=command.identity,
            mode=TranslationRevisionCancellationMode.REQUESTED_DEFERRED,
            stage=stage,
            persistence_started=False,
            message=(
                "Cancellation requested. Active translation inference may finish "
                "before its result is discarded; nothing will be published."
            ),
        )
        return self.accept_cancellation(cancellation)

    def accept_receipt(
        self,
        value: TranslationRevisionWorkerReceipt,
    ) -> TranslationRevisionState:
        if not isinstance(value, TranslationRevisionWorkerReceipt):
            raise TypeError("value must be TranslationRevisionWorkerReceipt")
        self._require_identity(value.identity)
        self._state = TranslationRevisionState(
            selection=value.selection,
            phase=TranslationRevisionPhase.SUCCEEDED,
            message=(
                "Selected model translation revision is current. Cleanup, style, "
                "eligibility, layout, and output remain explicit requirements."
            ),
            receipt=value,
        )
        return self._state

    def accept_failure(
        self,
        value: TranslationRevisionWorkerFailure,
    ) -> TranslationRevisionState:
        if not isinstance(value, TranslationRevisionWorkerFailure):
            raise TypeError("value must be TranslationRevisionWorkerFailure")
        self._require_identity(value.identity)
        self._state = replace(
            self._state,
            phase=(
                TranslationRevisionPhase.STALE
                if value.stale
                else TranslationRevisionPhase.FAILED
            ),
            message=value.message,
            command=None,
            busy_state=None,
            cancellation_state=None,
            receipt=None,
            failure=value,
            cancelled=None,
        )
        return self._state

    def accept_cancelled(
        self,
        value: TranslationRevisionCancelledReceipt,
    ) -> TranslationRevisionState:
        if not isinstance(value, TranslationRevisionCancelledReceipt):
            raise TypeError("value must be TranslationRevisionCancelledReceipt")
        self._require_identity(value.identity)
        self._state = replace(
            self._state,
            phase=TranslationRevisionPhase.CANCELLED,
            message=value.message,
            command=None,
            busy_state=None,
            cancellation_state=None,
            receipt=None,
            failure=None,
            cancelled=value,
        )
        return self._state

    def rebind(
        self,
        selection: TranslationRevisionSelection,
    ) -> TranslationRevisionState:
        if not isinstance(selection, TranslationRevisionSelection):
            raise TypeError("selection must be TranslationRevisionSelection")
        if self._state.busy:
            raise RuntimeError(
                "cannot replace the translation selection while it is active"
            )
        phase = (
            TranslationRevisionPhase.READY
            if selection.available
            else TranslationRevisionPhase.UNAVAILABLE
        )
        self._state = TranslationRevisionState(
            selection=selection,
            phase=phase,
            message=(
                self._ready_message(selection)
                if selection.available
                else selection.unavailable_reason
            ),
        )
        return self._state

    def _require_command(self) -> TranslationRevisionWorkerCommand:
        command = self._state.command
        if command is None:
            raise RuntimeError("no translation revision command is active")
        return command

    def _require_identity(
        self,
        identity: TranslationRevisionCommandIdentity,
    ) -> TranslationRevisionWorkerCommand:
        if not isinstance(identity, TranslationRevisionCommandIdentity):
            raise TypeError(
                "event identity must be TranslationRevisionCommandIdentity"
            )
        command = self._require_command()
        if command.identity != identity:
            raise ValueError(
                "worker event belongs to another translation revision command"
            )
        return command

    @staticmethod
    def _ready_message(selection: TranslationRevisionSelection) -> str:
        if selection.translation_current:
            return (
                "A selected model translation revision is current. Later owner "
                "revisions remain explicitly required."
            )
        return (
            "Source is current. Retranslate Parent will run only the selected "
            "translation owner with the frozen provider, glossary, and context."
        )


def translation_revision_selection_from_projection(
    projection: "ProjectUiProjection",
    *,
    page_id: str,
    parent_id: str,
    run_settings_snapshot: RunSettingsSnapshot,
    runtime_provider_binding: RuntimeProviderBinding,
    glossary_snapshot: Mapping[str, Any],
    prior_page_context: tuple[str, ...],
) -> TranslationRevisionSelection:
    """Build one exact translation action from an immutable UI projection."""

    from app.pipeline.hierarchy_revision_contracts import ParentOrigin
    from app.ui.shell.project_projection import ProjectUiProjection

    if not isinstance(projection, ProjectUiProjection):
        raise TypeError("projection must be ProjectUiProjection")
    if not isinstance(run_settings_snapshot, RunSettingsSnapshot):
        raise TypeError("run_settings_snapshot must be RunSettingsSnapshot")
    if not isinstance(runtime_provider_binding, RuntimeProviderBinding):
        raise TypeError("runtime_provider_binding must be RuntimeProviderBinding")
    page = projection.page(page_id)
    parent = page.parent(parent_id)
    effective = parent.effective
    lineage = effective.lineage
    if effective.origin is not ParentOrigin.USER or lineage is None:
        raise ValueError(
            "explicit translation requires one standalone user-authored parent"
        )
    source_artifact = parent.selected_model_source_revision
    if source_artifact is None:
        raise ValueError(
            "Run explicit OCR before translating this user parent."
        )
    if (
        effective.source_authority != "ocr_revision"
        or effective.source_text != source_artifact.source_text
        or effective.source_revision_id != source_artifact.revision_id
    ):
        raise ValueError("the selected OCR revision is not the effective source")
    provider = TranslationProviderSelection.from_run_settings_snapshot(
        run_settings_snapshot
    )
    frozen_glossary = freeze_json(
        glossary_snapshot,
        field_name="glossary_snapshot",
    )
    glossary_fingerprint = translation_glossary_fingerprint(
        thaw_json(frozen_glossary)
    )
    context = tuple(prior_page_context)
    context_fingerprint = translation_context_fingerprint(context)
    artifact = parent.selected_model_translation_revision
    authority = {
        "unavailable": "unavailable",
        "translation_revision": "translation_revision",
        "user": "user",
    }.get(effective.target_authority)
    if authority is None:
        raise ValueError("projected user-parent target authority is unsupported")
    source_requirement = _stage_requirement(
        tuple(effective.stage_requirements),
        RevisionStage.SOURCE,
    )
    translation_requirement = _stage_requirement(
        tuple(effective.stage_requirements),
        RevisionStage.TRANSLATION,
    )
    source_current = bool(
        source_requirement.state is RevisionStageState.CURRENT
        and source_requirement.required_action is RevisionRequiredAction.NONE
    )
    translation_runnable = bool(
        translation_requirement.state is RevisionStageState.MISSING
        and translation_requirement.required_action
        is RevisionRequiredAction.EXPLICIT_RUN
    )
    if run_settings_snapshot.unresolved_requirements:
        available = False
        reason = (
            "Resolve the current run settings before Retranslate Parent: "
            + "; ".join(run_settings_snapshot.unresolved_requirements)
        )
    elif not source_current:
        available = False
        reason = "The selected parent requires a current source revision first."
    elif not translation_runnable:
        available = False
        reason = (
            "Selected model translation revision is current. Later owner "
            "revisions remain required."
            if artifact is not None
            else "The selected parent does not require explicit translation."
        )
    else:
        available = True
        reason = ""
    source_fingerprint = canonical_fingerprint(
        {"parent_id": effective.parent_id, "text": source_artifact.source_text}
    )
    return TranslationRevisionSelection(
        project_path=projection.metadata.project_path,
        project_id=projection.metadata.project_id,
        page_id=page.effective.page_id,
        parent_id=effective.parent_id,
        root_id=effective.root_id,
        parent_authored_edit_id=lineage.authored_edit_id,
        parent_role=effective.role,
        policy_region_type=translation_policy_region_type(effective.role),
        bubble_local_nested_speech=False,
        effective_page_fingerprint=page.effective.effective_fingerprint,
        hierarchy_revision_id=page.effective.hierarchy.revision_id,
        hierarchy_fingerprint=page.effective.hierarchy.fingerprint,
        effective_source_text=source_artifact.source_text,
        effective_source_authority="ocr_revision",
        effective_source_fingerprint=source_fingerprint,
        source_revision_id=source_artifact.revision_id,
        source_selection_edit_id=source_artifact.selection_edit_id,
        run_settings_snapshot=run_settings_snapshot,
        runtime_provider_binding=runtime_provider_binding,
        provider=provider,
        glossary_snapshot=frozen_glossary,
        glossary_fingerprint=glossary_fingerprint,
        prior_page_context=context,
        context_fingerprint=context_fingerprint,
        stage_requirements=tuple(effective.stage_requirements),
        model_translation_revision=artifact,
        user_target_text=parent.user_target_text,
        effective_target_text=effective.target_text,
        effective_target_authority=authority,
        available=available,
        unavailable_reason=reason,
    )


__all__ = [
    "TranslationRevisionCancellationMode",
    "TranslationRevisionCancellationState",
    "TranslationRevisionCancelledReceipt",
    "TranslationRevisionCommandIdentity",
    "TranslationRevisionFailureCode",
    "TranslationRevisionModel",
    "TranslationRevisionPhase",
    "TranslationRevisionSelection",
    "TranslationRevisionState",
    "TranslationRevisionWorkerBusyState",
    "TranslationRevisionWorkerCommand",
    "TranslationRevisionWorkerFailure",
    "TranslationRevisionWorkerReceipt",
    "TranslationRevisionWorkerStage",
    "translation_revision_selection_from_projection",
]
