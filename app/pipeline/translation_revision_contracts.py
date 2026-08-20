# -*- coding: utf-8 -*-
"""Typed contracts for one explicit, parent-scoped translation revision.

The contracts bind a model-produced target to the exact effective source and
configuration that produced it.  They do not implement prompts, provider
transport, retry policy, formatting policy, or any later pipeline stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from app.config.settings_contracts import (
    RunSettingsSnapshot,
    canonical_fingerprint,
    freeze_json,
    run_settings_snapshot_from_dict,
    thaw_json,
)

from .hierarchy_revision_contracts import (
    EFFECTIVE_HIERARCHY_REVISION_PREFIX,
    validate_user_parent_identity_pair,
)
from .ocr_revision_contracts import (
    OCR_SOURCE_REVISION_ID_PREFIX,
    OCR_SOURCE_SELECTION_EDIT_ID_PREFIX,
)


TRANSLATION_REVISION_SCHEMA_VERSION = "translation_revision_v1"
TRANSLATION_REVISION_ID_PREFIX = "translation-revision-v1-"
TRANSLATION_SELECTION_EDIT_ID_PREFIX = "translation-selection-v1-"
TRANSLATION_GLOSSARY_SNAPSHOT_SCHEMA_VERSION = (
    "translation_glossary_snapshot_v1"
)
TRANSLATION_CONTEXT_SNAPSHOT_SCHEMA_VERSION = (
    "translation_prior_context_snapshot_v1"
)
SUPPORTED_TRANSLATION_PROVIDER_KINDS = frozenset(
    {"gguf", "ollama", "deepseek"}
)
SUPPORTED_TRANSLATION_PARENT_ROLES = frozenset({"speech", "caption"})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_identity(value: Any, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} is required")
    return candidate


def _require_path_safe_identity(value: Any, field_name: str) -> str:
    candidate = _require_identity(value, field_name)
    if _PATH_SAFE_ID.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be path-safe")
    return candidate


def _require_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "").lower()
    if _SHA256.fullmatch(candidate) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return candidate


def translation_glossary_fingerprint(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("glossary snapshot must be a mapping")
    frozen = freeze_json(value, field_name="glossary_snapshot")
    return canonical_fingerprint(
        {
            "schema_version": TRANSLATION_GLOSSARY_SNAPSHOT_SCHEMA_VERSION,
            "snapshot": thaw_json(frozen),
        }
    )


def translation_context_fingerprint(lines: tuple[str, ...]) -> str:
    if not isinstance(lines, tuple) or any(
        not isinstance(line, str) for line in lines
    ):
        raise TypeError("prior page context must be a tuple of strings")
    return canonical_fingerprint(
        {
            "schema_version": TRANSLATION_CONTEXT_SNAPSHOT_SCHEMA_VERSION,
            "lines": list(lines),
        }
    )


def translation_policy_region_type(parent_role: str) -> str:
    """Compile the typed editor role into the existing policy-region input."""

    role = str(parent_role or "").strip().casefold()
    if role not in SUPPORTED_TRANSLATION_PARENT_ROLES:
        raise ValueError("translation parent role is unsupported")
    return "speech_bubble" if role == "speech" else "background_text"


@dataclass(frozen=True)
class TranslationPolicySnapshots:
    """Deterministic, prompt-affecting inputs compiled outside the UI."""

    glossary_snapshot: Mapping[str, Any]
    glossary_fingerprint: str
    prior_page_context: tuple[str, ...]
    context_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.glossary_snapshot, Mapping):
            raise TypeError("glossary_snapshot must be a mapping")
        glossary = freeze_json(
            self.glossary_snapshot,
            field_name="glossary_snapshot",
        )
        glossary_fingerprint = _require_sha256(
            self.glossary_fingerprint,
            "glossary_fingerprint",
        )
        if (
            translation_glossary_fingerprint(thaw_json(glossary))
            != glossary_fingerprint
        ):
            raise ValueError("glossary fingerprint does not match its snapshot")
        context = tuple(self.prior_page_context)
        if any(not isinstance(line, str) for line in context) or len(context) > 4:
            raise ValueError("prior page context snapshot is invalid")
        context_fingerprint = _require_sha256(
            self.context_fingerprint,
            "context_fingerprint",
        )
        if translation_context_fingerprint(context) != context_fingerprint:
            raise ValueError("context fingerprint does not match its snapshot")
        object.__setattr__(self, "glossary_snapshot", glossary)
        object.__setattr__(self, "glossary_fingerprint", glossary_fingerprint)
        object.__setattr__(self, "prior_page_context", context)
        object.__setattr__(self, "context_fingerprint", context_fingerprint)

    def to_dict(self) -> dict[str, object]:
        return {
            "glossary_snapshot": thaw_json(self.glossary_snapshot),
            "glossary_fingerprint": self.glossary_fingerprint,
            "prior_page_context": list(self.prior_page_context),
            "context_fingerprint": self.context_fingerprint,
        }


def _selected_model_from_snapshot(snapshot: RunSettingsSnapshot) -> str:
    values = snapshot.pipeline_values
    backend = str(values.get("translator_backend") or "").strip()
    if backend == "GGUF":
        return _require_identity(
            values.get("gguf_model_path"),
            "pipeline_values.gguf_model_path",
        )
    if backend == "Ollama":
        return _require_identity(
            values.get("ollama_model"),
            "pipeline_values.ollama_model",
        )
    if backend == "DeepSeek":
        return _require_identity(
            values.get("deepseek_model"),
            "pipeline_values.deepseek_model",
        )
    raise ValueError("the selected translation provider is unsupported")


@dataclass(frozen=True)
class TranslationProviderSelection:
    profile_id: str
    provider_kind: str
    model_id: str
    public_configuration_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _require_identity(self.profile_id, "provider.profile_id"),
        )
        kind = _require_identity(
            self.provider_kind,
            "provider.provider_kind",
        ).casefold().replace("_", "-")
        if kind not in SUPPORTED_TRANSLATION_PROVIDER_KINDS:
            raise ValueError("translation provider kind is unsupported")
        object.__setattr__(self, "provider_kind", kind)
        object.__setattr__(
            self,
            "model_id",
            _require_identity(self.model_id, "provider.model_id"),
        )
        if self.model_id == "auto-detect":
            raise ValueError(
                "explicit translation requires one resolved model identity"
            )
        object.__setattr__(
            self,
            "public_configuration_fingerprint",
            _require_sha256(
                self.public_configuration_fingerprint,
                "provider.public_configuration_fingerprint",
            ),
        )

    @classmethod
    def from_run_settings_snapshot(
        cls,
        snapshot: RunSettingsSnapshot,
    ) -> "TranslationProviderSelection":
        if not isinstance(snapshot, RunSettingsSnapshot):
            raise TypeError("snapshot must be a RunSettingsSnapshot")
        profiles = thaw_json(snapshot.provider_profile_snapshot)
        translation = profiles.get("translation")
        if not isinstance(translation, Mapping):
            raise ValueError("translation provider snapshot is unavailable")
        if str(translation.get("status") or "") == "unresolved":
            raise ValueError("translation provider snapshot is unresolved")
        kind = str(translation.get("kind") or "")
        normalized_kind = kind.casefold().replace("_", "-")
        locator_fingerprint = translation.get(
            "credential_locator_fingerprint"
        )
        if locator_fingerprint is not None:
            _require_sha256(
                locator_fingerprint,
                "translation.credential_locator_fingerprint",
            )
        if normalized_kind == "deepseek" and locator_fingerprint is None:
            raise ValueError(
                "DeepSeek provider snapshot has no credential locator binding"
            )
        backend = str(snapshot.pipeline_values.get("translator_backend") or "")
        expected_backend = {
            "gguf": "GGUF",
            "ollama": "Ollama",
            "deepseek": "DeepSeek",
        }.get(normalized_kind)
        if expected_backend is None or backend != expected_backend:
            raise ValueError(
                "translation provider snapshot and pipeline backend differ"
            )
        return cls(
            profile_id=translation.get("profile_id"),
            provider_kind=kind,
            model_id=_selected_model_from_snapshot(snapshot),
            public_configuration_fingerprint=canonical_fingerprint(
                translation
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "provider_kind": self.provider_kind,
            "model_id": self.model_id,
            "public_configuration_fingerprint": (
                self.public_configuration_fingerprint
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "TranslationProviderSelection":
        if not isinstance(value, Mapping):
            raise TypeError("provider selection must be a mapping")
        expected = {
            "profile_id",
            "provider_kind",
            "model_id",
            "public_configuration_fingerprint",
        }
        if set(value) != expected:
            raise ValueError("provider selection fields are invalid")
        return cls(**dict(value))


class TranslationRevisionErrorCode(str, Enum):
    CANCELLED = "cancelled"
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    PAGE_NOT_FOUND = "page_not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_LINEAGE_MISMATCH = "parent_lineage_mismatch"
    STALE_HIERARCHY = "stale_hierarchy"
    STALE_EFFECTIVE_PAGE = "stale_effective_page"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    SOURCE_NOT_CURRENT = "source_not_current"
    SOURCE_MISMATCH = "source_mismatch"
    SETTINGS_MISMATCH = "settings_mismatch"
    GLOSSARY_MISMATCH = "glossary_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MODEL_MISSING = "model_missing"
    TRANSLATION_FAILED = "translation_failed"
    EMPTY_RESULT = "empty_result"
    PROJECTION_REJECTED = "projection_rejected"
    PERSISTENCE_REJECTED = "persistence_rejected"


class TranslationRevisionError(RuntimeError):
    """Typed fail-closed error for an explicit translation transaction."""

    def __init__(
        self,
        code: TranslationRevisionErrorCode,
        message: str,
    ) -> None:
        self.code = TranslationRevisionErrorCode(code)
        super().__init__(str(message))


@dataclass(frozen=True)
class ExplicitTranslationRevisionRequest:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    policy_region_type: str
    bubble_local_nested_speech: bool
    expected_hierarchy_revision_id: str
    expected_hierarchy_fingerprint: str
    expected_effective_page_fingerprint: str
    effective_source_text: str
    effective_source_authority: str
    effective_source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    run_settings_snapshot: RunSettingsSnapshot
    run_settings_fingerprint: str
    provider: TranslationProviderSelection
    glossary_snapshot: Mapping[str, Any]
    glossary_fingerprint: str
    prior_page_context: tuple[str, ...]
    context_fingerprint: str
    expected_page_head_sha256: str
    expected_global_head_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _require_path_safe_identity(self.command_id, "command_id"),
        )
        for field_name in (
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        role = str(self.parent_role or "").strip().casefold()
        expected_region_type = translation_policy_region_type(role)
        if self.policy_region_type != expected_region_type:
            raise ValueError(
                "policy_region_type does not match the typed parent role"
            )
        if not isinstance(self.bubble_local_nested_speech, bool):
            raise TypeError("bubble_local_nested_speech must be a bool")
        object.__setattr__(self, "parent_role", role)
        if not str(self.expected_hierarchy_revision_id).startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("expected_hierarchy_revision_id is invalid")
        object.__setattr__(
            self,
            "expected_hierarchy_revision_id",
            str(self.expected_hierarchy_revision_id),
        )
        for field_name in (
            "expected_hierarchy_fingerprint",
            "expected_effective_page_fingerprint",
            "effective_source_fingerprint",
            "run_settings_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        if (
            not isinstance(self.effective_source_text, str)
            or not self.effective_source_text.strip()
        ):
            raise ValueError("effective_source_text must be non-empty")
        if self.effective_source_authority != "ocr_revision":
            raise ValueError(
                "explicit translation currently requires a selected OCR revision"
            )
        if not str(self.source_revision_id).startswith(
            OCR_SOURCE_REVISION_ID_PREFIX
        ):
            raise ValueError("source_revision_id is invalid")
        if not str(self.source_selection_edit_id).startswith(
            OCR_SOURCE_SELECTION_EDIT_ID_PREFIX
        ):
            raise ValueError("source_selection_edit_id is invalid")
        expected_source_fingerprint = canonical_fingerprint(
            {
                "parent_id": self.parent_id,
                "text": self.effective_source_text,
            }
        )
        if self.effective_source_fingerprint != expected_source_fingerprint:
            raise ValueError(
                "effective source fingerprint does not match its exact text"
            )
        if not isinstance(self.run_settings_snapshot, RunSettingsSnapshot):
            raise TypeError(
                "run_settings_snapshot must be a RunSettingsSnapshot"
            )
        snapshot = self.run_settings_snapshot
        if snapshot.project_id != self.project_id:
            raise ValueError(
                "run settings project identity does not match the request"
            )
        if snapshot.unresolved_requirements:
            raise ValueError("run settings snapshot has unresolved requirements")
        if snapshot.settings_fingerprint != self.run_settings_fingerprint:
            raise ValueError(
                "run settings fingerprint does not match the snapshot"
            )
        provider = (
            self.provider
            if isinstance(self.provider, TranslationProviderSelection)
            else TranslationProviderSelection.from_dict(self.provider)
        )
        expected_provider = TranslationProviderSelection.from_run_settings_snapshot(
            snapshot
        )
        if provider != expected_provider:
            raise ValueError(
                "provider selection does not match the immutable run snapshot"
            )
        object.__setattr__(self, "provider", provider)
        if not isinstance(self.glossary_snapshot, Mapping):
            raise TypeError("glossary_snapshot must be a mapping")
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
        if not isinstance(self.prior_page_context, tuple) or any(
            not isinstance(line, str) for line in self.prior_page_context
        ):
            raise TypeError("prior_page_context must be a tuple of strings")
        if len(self.prior_page_context) > 4:
            raise ValueError("prior_page_context cannot exceed four lines")
        if (
            translation_context_fingerprint(self.prior_page_context)
            != self.context_fingerprint
        ):
            raise ValueError("context fingerprint does not match its snapshot")
        context_enabled = bool(
            snapshot.pipeline_values.get("translator_backend") == "GGUF"
            and snapshot.pipeline_values.get("target_lang")
            == "Simplified Chinese"
            and snapshot.pipeline_values.get("gguf_cross_page_context")
        )
        if not context_enabled and self.prior_page_context:
            raise ValueError(
                "prior page context is not enabled by the run snapshot"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "parent_role": self.parent_role,
            "policy_region_type": self.policy_region_type,
            "bubble_local_nested_speech": self.bubble_local_nested_speech,
            "expected_hierarchy_revision_id": (
                self.expected_hierarchy_revision_id
            ),
            "expected_hierarchy_fingerprint": (
                self.expected_hierarchy_fingerprint
            ),
            "expected_effective_page_fingerprint": (
                self.expected_effective_page_fingerprint
            ),
            "effective_source_text": self.effective_source_text,
            "effective_source_authority": self.effective_source_authority,
            "effective_source_fingerprint": self.effective_source_fingerprint,
            "source_revision_id": self.source_revision_id,
            "source_selection_edit_id": self.source_selection_edit_id,
            "run_settings_snapshot": self.run_settings_snapshot.to_dict(),
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "provider": self.provider.to_dict(),
            "glossary_snapshot": thaw_json(self.glossary_snapshot),
            "glossary_fingerprint": self.glossary_fingerprint,
            "prior_page_context": list(self.prior_page_context),
            "context_fingerprint": self.context_fingerprint,
            "expected_page_head_sha256": self.expected_page_head_sha256,
            "expected_global_head_sha256": self.expected_global_head_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ExplicitTranslationRevisionRequest":
        if not isinstance(value, Mapping):
            raise TypeError("translation revision request must be a mapping")
        expected = {
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "parent_role",
            "policy_region_type",
            "bubble_local_nested_speech",
            "expected_hierarchy_revision_id",
            "expected_hierarchy_fingerprint",
            "expected_effective_page_fingerprint",
            "effective_source_text",
            "effective_source_authority",
            "effective_source_fingerprint",
            "source_revision_id",
            "source_selection_edit_id",
            "run_settings_snapshot",
            "run_settings_fingerprint",
            "provider",
            "glossary_snapshot",
            "glossary_fingerprint",
            "prior_page_context",
            "context_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        }
        if set(value) != expected:
            raise ValueError("translation revision request fields are invalid")
        return cls(
            command_id=value["command_id"],
            project_id=value["project_id"],
            page_id=value["page_id"],
            parent_id=value["parent_id"],
            root_id=value["root_id"],
            parent_authored_edit_id=value["parent_authored_edit_id"],
            parent_role=value["parent_role"],
            policy_region_type=value["policy_region_type"],
            bubble_local_nested_speech=value["bubble_local_nested_speech"],
            expected_hierarchy_revision_id=value[
                "expected_hierarchy_revision_id"
            ],
            expected_hierarchy_fingerprint=value[
                "expected_hierarchy_fingerprint"
            ],
            expected_effective_page_fingerprint=value[
                "expected_effective_page_fingerprint"
            ],
            effective_source_text=value["effective_source_text"],
            effective_source_authority=value["effective_source_authority"],
            effective_source_fingerprint=value[
                "effective_source_fingerprint"
            ],
            source_revision_id=value["source_revision_id"],
            source_selection_edit_id=value["source_selection_edit_id"],
            run_settings_snapshot=run_settings_snapshot_from_dict(
                value["run_settings_snapshot"]
            ),
            run_settings_fingerprint=value["run_settings_fingerprint"],
            provider=TranslationProviderSelection.from_dict(value["provider"]),
            glossary_snapshot=value["glossary_snapshot"],
            glossary_fingerprint=value["glossary_fingerprint"],
            prior_page_context=tuple(value["prior_page_context"]),
            context_fingerprint=value["context_fingerprint"],
            expected_page_head_sha256=value["expected_page_head_sha256"],
            expected_global_head_sha256=value["expected_global_head_sha256"],
        )


@dataclass(frozen=True)
class TranslationExecutionRequest:
    request: ExplicitTranslationRevisionRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request, ExplicitTranslationRevisionRequest):
            raise TypeError(
                "request must be an ExplicitTranslationRevisionRequest"
            )


@dataclass(frozen=True)
class TranslationExecutionReceipt:
    target_text: str
    source_fingerprint: str
    run_settings_fingerprint: str
    provider: TranslationProviderSelection
    glossary_fingerprint: str
    context_fingerprint: str
    policy_metadata: Mapping[str, Any]
    quality_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target_text, str):
            raise TypeError("target_text must be a string")
        for field_name in (
            "source_fingerprint",
            "run_settings_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        provider = (
            self.provider
            if isinstance(self.provider, TranslationProviderSelection)
            else TranslationProviderSelection.from_dict(self.provider)
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "policy_metadata",
            freeze_json(self.policy_metadata, field_name="policy_metadata"),
        )
        warnings = tuple(self.quality_warnings)
        if any(not isinstance(item, str) or not item for item in warnings):
            raise ValueError("quality warnings must be non-empty strings")
        object.__setattr__(self, "quality_warnings", warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_text": self.target_text,
            "source_fingerprint": self.source_fingerprint,
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "provider": self.provider.to_dict(),
            "glossary_fingerprint": self.glossary_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "policy_metadata": thaw_json(self.policy_metadata),
            "quality_warnings": list(self.quality_warnings),
        }


@dataclass(frozen=True)
class TranslationRevisionArtifact:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    policy_region_type: str
    bubble_local_nested_speech: bool
    selection_edit_id: str
    target_text: str
    source_text: str
    source_authority: str
    source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    run_settings_snapshot: RunSettingsSnapshot
    run_settings_fingerprint: str
    provider: TranslationProviderSelection
    glossary_snapshot: Mapping[str, Any]
    glossary_fingerprint: str
    prior_page_context: tuple[str, ...]
    context_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    input_effective_page_fingerprint: str
    policy_metadata: Mapping[str, Any]
    quality_warnings: tuple[str, ...]
    revision_id: str = ""
    provenance: str = "translation_model_revision"
    schema_version: str = TRANSLATION_REVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSLATION_REVISION_SCHEMA_VERSION:
            raise ValueError("unsupported translation revision schema")
        if self.provenance != "translation_model_revision":
            raise ValueError("translation revision provenance is invalid")
        for field_name in (
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "selection_edit_id",
            "source_revision_id",
            "source_selection_edit_id",
            "hierarchy_revision_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        validate_user_parent_identity_pair(self.parent_id, self.root_id)
        role = str(self.parent_role or "").strip().casefold()
        if self.policy_region_type != translation_policy_region_type(role):
            raise ValueError("translation policy region type is invalid")
        if not isinstance(self.bubble_local_nested_speech, bool):
            raise TypeError("bubble_local_nested_speech must be a bool")
        object.__setattr__(self, "parent_role", role)
        if not isinstance(self.target_text, str) or not self.target_text.strip():
            raise ValueError("translation revision target text must be non-empty")
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("translation revision source text must be non-empty")
        if self.source_authority != "ocr_revision":
            raise ValueError("translation revision source authority is invalid")
        if not self.source_revision_id.startswith(OCR_SOURCE_REVISION_ID_PREFIX):
            raise ValueError("translation revision source identity is invalid")
        if not self.source_selection_edit_id.startswith(
            OCR_SOURCE_SELECTION_EDIT_ID_PREFIX
        ):
            raise ValueError("translation revision source selection is invalid")
        if not self.selection_edit_id.startswith(
            TRANSLATION_SELECTION_EDIT_ID_PREFIX
        ):
            raise ValueError("translation revision selection identity is invalid")
        if not self.hierarchy_revision_id.startswith(
            EFFECTIVE_HIERARCHY_REVISION_PREFIX
        ):
            raise ValueError("translation hierarchy revision is invalid")
        for field_name in (
            "source_fingerprint",
            "run_settings_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
            "hierarchy_fingerprint",
            "input_effective_page_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        expected_source = canonical_fingerprint(
            {"parent_id": self.parent_id, "text": self.source_text}
        )
        if expected_source != self.source_fingerprint:
            raise ValueError("translation source fingerprint is invalid")
        if not isinstance(self.run_settings_snapshot, RunSettingsSnapshot):
            raise TypeError("run_settings_snapshot must be a RunSettingsSnapshot")
        if (
            self.run_settings_snapshot.project_id != self.project_id
            or self.run_settings_snapshot.settings_fingerprint
            != self.run_settings_fingerprint
            or self.run_settings_snapshot.unresolved_requirements
        ):
            raise ValueError("translation run settings binding is invalid")
        provider = (
            self.provider
            if isinstance(self.provider, TranslationProviderSelection)
            else TranslationProviderSelection.from_dict(self.provider)
        )
        if provider != TranslationProviderSelection.from_run_settings_snapshot(
            self.run_settings_snapshot
        ):
            raise ValueError("translation provider binding is invalid")
        object.__setattr__(self, "provider", provider)
        glossary = freeze_json(
            self.glossary_snapshot,
            field_name="glossary_snapshot",
        )
        if (
            translation_glossary_fingerprint(thaw_json(glossary))
            != self.glossary_fingerprint
        ):
            raise ValueError("translation glossary binding is invalid")
        object.__setattr__(self, "glossary_snapshot", glossary)
        context = tuple(self.prior_page_context)
        if any(not isinstance(line, str) for line in context) or len(context) > 4:
            raise ValueError("translation context snapshot is invalid")
        if translation_context_fingerprint(context) != self.context_fingerprint:
            raise ValueError("translation context binding is invalid")
        object.__setattr__(self, "prior_page_context", context)
        object.__setattr__(
            self,
            "policy_metadata",
            freeze_json(self.policy_metadata, field_name="policy_metadata"),
        )
        warnings = tuple(self.quality_warnings)
        if any(not isinstance(item, str) or not item for item in warnings):
            raise ValueError("translation quality warnings are invalid")
        object.__setattr__(self, "quality_warnings", warnings)
        expected_revision_id = (
            TRANSLATION_REVISION_ID_PREFIX
            + canonical_fingerprint(self._semantic_dict())
        )
        if self.revision_id and self.revision_id != expected_revision_id:
            raise ValueError(
                "translation revision identity does not match its content"
            )
        object.__setattr__(self, "revision_id", expected_revision_id)

    def _semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "command_id": self.command_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "parent_role": self.parent_role,
            "policy_region_type": self.policy_region_type,
            "bubble_local_nested_speech": self.bubble_local_nested_speech,
            "selection_edit_id": self.selection_edit_id,
            "target_text": self.target_text,
            "source_text": self.source_text,
            "source_authority": self.source_authority,
            "source_fingerprint": self.source_fingerprint,
            "source_revision_id": self.source_revision_id,
            "source_selection_edit_id": self.source_selection_edit_id,
            "run_settings_snapshot": self.run_settings_snapshot.to_dict(),
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "provider": self.provider.to_dict(),
            "glossary_snapshot": thaw_json(self.glossary_snapshot),
            "glossary_fingerprint": self.glossary_fingerprint,
            "prior_page_context": list(self.prior_page_context),
            "context_fingerprint": self.context_fingerprint,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
            "input_effective_page_fingerprint": (
                self.input_effective_page_fingerprint
            ),
            "policy_metadata": thaw_json(self.policy_metadata),
            "quality_warnings": list(self.quality_warnings),
        }

    def to_record(self, *, include_catalog: bool = False) -> dict[str, object]:
        result = self._semantic_dict()
        result["revision_id"] = self.revision_id
        if include_catalog:
            result["catalog"] = "translation_revisions"
        return result

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> "TranslationRevisionArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("translation revision artifact must be a mapping")
        allowed = {
            "schema_version",
            "provenance",
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "parent_role",
            "policy_region_type",
            "bubble_local_nested_speech",
            "selection_edit_id",
            "target_text",
            "source_text",
            "source_authority",
            "source_fingerprint",
            "source_revision_id",
            "source_selection_edit_id",
            "run_settings_snapshot",
            "run_settings_fingerprint",
            "provider",
            "glossary_snapshot",
            "glossary_fingerprint",
            "prior_page_context",
            "context_fingerprint",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
            "input_effective_page_fingerprint",
            "policy_metadata",
            "quality_warnings",
            "revision_id",
            "catalog",
        }
        unknown = frozenset(value) - allowed
        if unknown:
            raise ValueError(
                "translation revision artifact has unsupported fields: "
                f"{sorted(unknown)}"
            )
        if (
            "catalog" in value
            and value.get("catalog") != "translation_revisions"
        ):
            raise ValueError("translation revision artifact catalog is invalid")
        return cls(
            schema_version=str(value.get("schema_version") or ""),
            provenance=str(value.get("provenance") or ""),
            command_id=str(value.get("command_id") or ""),
            project_id=str(value.get("project_id") or ""),
            page_id=str(value.get("page_id") or ""),
            parent_id=str(value.get("parent_id") or ""),
            root_id=str(value.get("root_id") or ""),
            parent_authored_edit_id=str(
                value.get("parent_authored_edit_id") or ""
            ),
            parent_role=str(value.get("parent_role") or ""),
            policy_region_type=str(value.get("policy_region_type") or ""),
            bubble_local_nested_speech=value.get(
                "bubble_local_nested_speech"
            ),
            selection_edit_id=str(value.get("selection_edit_id") or ""),
            target_text=value.get("target_text"),
            source_text=value.get("source_text"),
            source_authority=str(value.get("source_authority") or ""),
            source_fingerprint=str(value.get("source_fingerprint") or ""),
            source_revision_id=str(value.get("source_revision_id") or ""),
            source_selection_edit_id=str(
                value.get("source_selection_edit_id") or ""
            ),
            run_settings_snapshot=run_settings_snapshot_from_dict(
                value.get("run_settings_snapshot") or {}
            ),
            run_settings_fingerprint=str(
                value.get("run_settings_fingerprint") or ""
            ),
            provider=TranslationProviderSelection.from_dict(
                value.get("provider") or {}
            ),
            glossary_snapshot=dict(value.get("glossary_snapshot") or {}),
            glossary_fingerprint=str(value.get("glossary_fingerprint") or ""),
            prior_page_context=tuple(value.get("prior_page_context") or ()),
            context_fingerprint=str(value.get("context_fingerprint") or ""),
            hierarchy_revision_id=str(
                value.get("hierarchy_revision_id") or ""
            ),
            hierarchy_fingerprint=str(
                value.get("hierarchy_fingerprint") or ""
            ),
            input_effective_page_fingerprint=str(
                value.get("input_effective_page_fingerprint") or ""
            ),
            policy_metadata=dict(value.get("policy_metadata") or {}),
            quality_warnings=tuple(value.get("quality_warnings") or ()),
            revision_id=str(value.get("revision_id") or ""),
        )


@dataclass(frozen=True)
class ExplicitTranslationRevisionReceipt:
    command_id: str
    project_id: str
    page_id: str
    parent_id: str
    root_id: str
    parent_authored_edit_id: str
    parent_role: str
    policy_region_type: str
    bubble_local_nested_speech: bool
    translation_revision_id: str
    selection_edit_id: str
    target_text: str
    source_text: str
    source_authority: str
    source_fingerprint: str
    source_revision_id: str
    source_selection_edit_id: str
    run_settings_snapshot: RunSettingsSnapshot
    run_settings_fingerprint: str
    provider: TranslationProviderSelection
    glossary_snapshot: Mapping[str, Any]
    glossary_fingerprint: str
    prior_page_context: tuple[str, ...]
    context_fingerprint: str
    hierarchy_revision_id: str
    hierarchy_fingerprint: str
    before_effective_page_fingerprint: str
    after_effective_page_fingerprint: str
    policy_metadata: Mapping[str, Any]
    quality_warnings: tuple[str, ...]
    invalidation: Mapping[str, Any]
    stage_requirements: tuple[Mapping[str, Any], ...]
    commit_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "translation_revision_id",
            "selection_edit_id",
            "source_revision_id",
            "source_selection_edit_id",
            "hierarchy_revision_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_identity(getattr(self, field_name), field_name),
            )
        for field_name in (
            "source_fingerprint",
            "run_settings_fingerprint",
            "glossary_fingerprint",
            "context_fingerprint",
            "hierarchy_fingerprint",
            "before_effective_page_fingerprint",
            "after_effective_page_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), field_name),
            )
        provider = (
            self.provider
            if isinstance(self.provider, TranslationProviderSelection)
            else TranslationProviderSelection.from_dict(self.provider)
        )
        object.__setattr__(self, "provider", provider)
        for field_name in (
            "glossary_snapshot",
            "policy_metadata",
            "invalidation",
            "commit_receipt",
        ):
            object.__setattr__(
                self,
                field_name,
                freeze_json(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "prior_page_context",
            tuple(self.prior_page_context),
        )
        object.__setattr__(
            self,
            "quality_warnings",
            tuple(self.quality_warnings),
        )
        object.__setattr__(
            self,
            "stage_requirements",
            tuple(
                freeze_json(value, field_name="stage_requirements")
                for value in self.stage_requirements
            ),
        )
        TranslationRevisionArtifact(
            command_id=self.command_id,
            project_id=self.project_id,
            page_id=self.page_id,
            parent_id=self.parent_id,
            root_id=self.root_id,
            parent_authored_edit_id=self.parent_authored_edit_id,
            parent_role=self.parent_role,
            policy_region_type=self.policy_region_type,
            bubble_local_nested_speech=self.bubble_local_nested_speech,
            selection_edit_id=self.selection_edit_id,
            target_text=self.target_text,
            source_text=self.source_text,
            source_authority=self.source_authority,
            source_fingerprint=self.source_fingerprint,
            source_revision_id=self.source_revision_id,
            source_selection_edit_id=self.source_selection_edit_id,
            run_settings_snapshot=self.run_settings_snapshot,
            run_settings_fingerprint=self.run_settings_fingerprint,
            provider=self.provider,
            glossary_snapshot=self.glossary_snapshot,
            glossary_fingerprint=self.glossary_fingerprint,
            prior_page_context=self.prior_page_context,
            context_fingerprint=self.context_fingerprint,
            hierarchy_revision_id=self.hierarchy_revision_id,
            hierarchy_fingerprint=self.hierarchy_fingerprint,
            input_effective_page_fingerprint=(
                self.before_effective_page_fingerprint
            ),
            policy_metadata=self.policy_metadata,
            quality_warnings=self.quality_warnings,
            revision_id=self.translation_revision_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "command_id": self.command_id,
            "project_id": self.project_id,
            "page_id": self.page_id,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
            "parent_authored_edit_id": self.parent_authored_edit_id,
            "parent_role": self.parent_role,
            "policy_region_type": self.policy_region_type,
            "bubble_local_nested_speech": self.bubble_local_nested_speech,
            "translation_revision_id": self.translation_revision_id,
            "selection_edit_id": self.selection_edit_id,
            "target_text": self.target_text,
            "source_text": self.source_text,
            "source_authority": self.source_authority,
            "source_fingerprint": self.source_fingerprint,
            "source_revision_id": self.source_revision_id,
            "source_selection_edit_id": self.source_selection_edit_id,
            "run_settings_snapshot": self.run_settings_snapshot.to_dict(),
            "run_settings_fingerprint": self.run_settings_fingerprint,
            "provider": self.provider.to_dict(),
            "glossary_snapshot": thaw_json(self.glossary_snapshot),
            "glossary_fingerprint": self.glossary_fingerprint,
            "prior_page_context": list(self.prior_page_context),
            "context_fingerprint": self.context_fingerprint,
            "hierarchy_revision_id": self.hierarchy_revision_id,
            "hierarchy_fingerprint": self.hierarchy_fingerprint,
            "before_effective_page_fingerprint": (
                self.before_effective_page_fingerprint
            ),
            "after_effective_page_fingerprint": (
                self.after_effective_page_fingerprint
            ),
            "policy_metadata": thaw_json(self.policy_metadata),
            "quality_warnings": list(self.quality_warnings),
            "invalidation": thaw_json(self.invalidation),
            "stage_requirements": [
                thaw_json(value) for value in self.stage_requirements
            ],
            "commit_receipt": thaw_json(self.commit_receipt),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ExplicitTranslationRevisionReceipt":
        if not isinstance(value, Mapping):
            raise TypeError("translation revision receipt must be a mapping")
        expected = {
            "command_id",
            "project_id",
            "page_id",
            "parent_id",
            "root_id",
            "parent_authored_edit_id",
            "parent_role",
            "policy_region_type",
            "bubble_local_nested_speech",
            "translation_revision_id",
            "selection_edit_id",
            "target_text",
            "source_text",
            "source_authority",
            "source_fingerprint",
            "source_revision_id",
            "source_selection_edit_id",
            "run_settings_snapshot",
            "run_settings_fingerprint",
            "provider",
            "glossary_snapshot",
            "glossary_fingerprint",
            "prior_page_context",
            "context_fingerprint",
            "hierarchy_revision_id",
            "hierarchy_fingerprint",
            "before_effective_page_fingerprint",
            "after_effective_page_fingerprint",
            "policy_metadata",
            "quality_warnings",
            "invalidation",
            "stage_requirements",
            "commit_receipt",
        }
        if set(value) != expected:
            raise ValueError("translation revision receipt fields are invalid")
        return cls(
            command_id=value["command_id"],
            project_id=value["project_id"],
            page_id=value["page_id"],
            parent_id=value["parent_id"],
            root_id=value["root_id"],
            parent_authored_edit_id=value["parent_authored_edit_id"],
            parent_role=value["parent_role"],
            policy_region_type=value["policy_region_type"],
            bubble_local_nested_speech=value["bubble_local_nested_speech"],
            translation_revision_id=value["translation_revision_id"],
            selection_edit_id=value["selection_edit_id"],
            target_text=value["target_text"],
            source_text=value["source_text"],
            source_authority=value["source_authority"],
            source_fingerprint=value["source_fingerprint"],
            source_revision_id=value["source_revision_id"],
            source_selection_edit_id=value["source_selection_edit_id"],
            run_settings_snapshot=run_settings_snapshot_from_dict(
                value["run_settings_snapshot"]
            ),
            run_settings_fingerprint=value["run_settings_fingerprint"],
            provider=TranslationProviderSelection.from_dict(value["provider"]),
            glossary_snapshot=value["glossary_snapshot"],
            glossary_fingerprint=value["glossary_fingerprint"],
            prior_page_context=tuple(value["prior_page_context"]),
            context_fingerprint=value["context_fingerprint"],
            hierarchy_revision_id=value["hierarchy_revision_id"],
            hierarchy_fingerprint=value["hierarchy_fingerprint"],
            before_effective_page_fingerprint=value[
                "before_effective_page_fingerprint"
            ],
            after_effective_page_fingerprint=value[
                "after_effective_page_fingerprint"
            ],
            policy_metadata=value["policy_metadata"],
            quality_warnings=tuple(value["quality_warnings"]),
            invalidation=value["invalidation"],
            stage_requirements=tuple(value["stage_requirements"]),
            commit_receipt=value["commit_receipt"],
        )


CancellationProbe = Callable[[], bool]


@runtime_checkable
class TranslationRevisionExecutionPort(Protocol):
    def translate(
        self,
        request: TranslationExecutionRequest,
        *,
        cancellation_probe: CancellationProbe | None = None,
    ) -> TranslationExecutionReceipt:
        ...


@runtime_checkable
class ExplicitTranslationRevisionPort(Protocol):
    def run_explicit_translation_revision(
        self,
        request: ExplicitTranslationRevisionRequest,
    ) -> ExplicitTranslationRevisionReceipt:
        ...


__all__ = [
    "CancellationProbe",
    "ExplicitTranslationRevisionPort",
    "ExplicitTranslationRevisionReceipt",
    "ExplicitTranslationRevisionRequest",
    "SUPPORTED_TRANSLATION_PROVIDER_KINDS",
    "SUPPORTED_TRANSLATION_PARENT_ROLES",
    "TRANSLATION_CONTEXT_SNAPSHOT_SCHEMA_VERSION",
    "TRANSLATION_GLOSSARY_SNAPSHOT_SCHEMA_VERSION",
    "TRANSLATION_REVISION_ID_PREFIX",
    "TRANSLATION_REVISION_SCHEMA_VERSION",
    "TRANSLATION_SELECTION_EDIT_ID_PREFIX",
    "TranslationExecutionReceipt",
    "TranslationExecutionRequest",
    "TranslationPolicySnapshots",
    "TranslationProviderSelection",
    "TranslationRevisionArtifact",
    "TranslationRevisionError",
    "TranslationRevisionErrorCode",
    "TranslationRevisionExecutionPort",
    "translation_context_fingerprint",
    "translation_glossary_fingerprint",
    "translation_policy_region_type",
]
