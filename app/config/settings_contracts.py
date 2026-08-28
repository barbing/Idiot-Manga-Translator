# -*- coding: utf-8 -*-
"""Typed, immutable settings-scope contracts for the GUI architecture.

These values are persistence and presentation contracts.  They deliberately do
not import or mutate ``PipelineSettings``; the run-settings compiler owns that
one-way conversion at workflow start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SETTINGS_CONTRACT_SCHEMA_VERSION = "settings_contracts_v1"
RUN_SETTINGS_SNAPSHOT_SCHEMA_VERSION = "run_settings_snapshot_v1"


class SettingsScope(str, Enum):
    APPLICATION = "application"
    PROJECT = "project"
    MODULE = "module"
    PROVIDER = "provider"
    CREDENTIAL = "credential"
    EDITOR = "editor"
    RUNTIME = "runtime"
    RUN = "run"


class CredentialReferenceKind(str, Enum):
    WINDOWS_CREDENTIAL = "windows_credential"
    ENVIRONMENT_VARIABLE = "environment_variable"
    SYSTEM_KEYRING = "system_keyring"


class ProviderHealth(str, Enum):
    UNKNOWN = "unknown"
    READY = "ready"
    UNRESOLVED = "unresolved"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class DownloadState(str, Enum):
    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


_SECRET_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|bearer|"
    r"authorization|password|passwd|secret|credential[_-]?ref|"
    r"credential[_-]?reference|credential[_-]?value)(?:$|[_-])",
    flags=re.IGNORECASE,
)
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_THEMES = frozenset({"dark", "light"})
_ALLOWED_UI_DENSITIES = frozenset({"comfortable", "compact"})
_ALLOWED_FONT_SCALES = frozenset(range(100, 201, 5))
_ALLOWED_UI_LANGUAGES = frozenset({"system", "English"})
_ALLOWED_SOURCE_LANGUAGES = frozenset({"Japanese"})
_ALLOWED_TARGET_LANGUAGES = frozenset({"Simplified Chinese", "English"})
_ALLOWED_OPEN_LAST_PROJECT_POLICIES = frozenset({"ask", "always", "never"})
_ALLOWED_OUTPUT_CONVENTIONS = frozenset(
    {"sibling_output_folder", "project_exports"}
)
_ALLOWED_COMPLETED_PAGE_POLICIES = frozenset(
    {"open_for_review", "continue_automatically"}
)
DEFAULT_SHORTCUT_BINDINGS = MappingProxyType(
    {
        "select": "V",
        "pan": "H",
        "undo": "Ctrl+Z",
        "redo": "Ctrl+Shift+Z",
        "preview": "Ctrl+Enter",
        "exit_focus": "Esc",
    }
)
MACOS_DEFAULT_SHORTCUT_BINDINGS = MappingProxyType(
    {
        **DEFAULT_SHORTCUT_BINDINGS,
        "undo": "Meta+Z",
        "redo": "Meta+Shift+Z",
        "preview": "Meta+Enter",
    }
)
HISTORICAL_WINDOWS_PROJECT_LOCATION = "D:/Manga Projects"
_ALLOWED_CANVAS_MODES = frozenset(
    {"original", "cleaned", "final", "compare", "difference"}
)
_ALLOWED_ACTIVITY_DOCK_TABS = frozenset({"overview", "history", "warnings"})
_WORKSPACE_LAYOUT_KEYS = frozenset(
    {
        "activity_dock_expanded",
        "activity_dock_height",
        "activity_dock_tab",
        "geometry_present",
        "window_state_present",
    }
)


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    return result


def _require_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty(value, field_name)


def _require_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return result


def _require_utc_timestamp(value: Any, field_name: str) -> str:
    text = _require_non_empty(value, field_name)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must use UTC")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_secret_key(key: str, field_name: str) -> None:
    normalized = str(key).strip()
    if _SECRET_KEY_PATTERN.search(normalized):
        raise ValueError(f"{field_name} contains forbidden secret field {key!r}")


def freeze_json(value: Any, *, field_name: str = "value") -> Any:
    """Return a recursively immutable, canonical JSON-compatible value.

    Mapping keys are sorted so iteration, serialization, and fingerprints are
    deterministic.  Secret-looking fields are rejected at this boundary.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            _reject_secret_key(key, field_name)
            items.append(
                (
                    key,
                    freeze_json(item, field_name=f"{field_name}.{key}"),
                )
            )
        return MappingProxyType(dict(sorted(items, key=lambda pair: pair[0])))
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            freeze_json(item, field_name=f"{field_name}[]") for item in value
        )
    raise TypeError(f"{field_name} is not JSON-compatible")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_fingerprint(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for public JSON data."""

    frozen = freeze_json(value, field_name="fingerprint_value")
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(thaw_json(frozen)):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    contract_name: str,
) -> None:
    missing = required - frozenset(value)
    unknown = frozenset(value) - required - optional
    if missing:
        raise ValueError(f"{contract_name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(
            f"{contract_name} has unsupported fields: {sorted(unknown)}"
        )


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(_require_non_empty(item, f"{field_name}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _freeze_shortcut_bindings(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("shortcut_bindings must be an object")
    expected = frozenset(DEFAULT_SHORTCUT_BINDINGS)
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "shortcut_bindings must contain the exact supported commands "
            f"(missing={missing}, extra={extra})"
        )
    normalized: dict[str, str] = {}
    comparison: set[str] = set()
    for command_id in DEFAULT_SHORTCUT_BINDINGS:
        sequence = _require_non_empty(
            value[command_id],
            f"shortcut_bindings.{command_id}",
        )
        if len(sequence) > 64 or any(ord(character) < 32 for character in sequence):
            raise ValueError(
                f"shortcut_bindings.{command_id} contains an invalid sequence"
            )
        duplicate_key = sequence.casefold()
        if duplicate_key in comparison:
            raise ValueError("shortcut_bindings must not contain duplicate sequences")
        comparison.add(duplicate_key)
        normalized[command_id] = sequence
    return MappingProxyType(normalized)


def _freeze_workspace_layout(value: Any) -> Mapping[str, Any]:
    """Validate the small, public application-layout persistence contract.

    The GUI prototype currently needs only the Activity dock's presentation
    state.  The two ``*_present`` flags retain evidence from the disabled
    legacy Qt restore path without persisting opaque Qt byte arrays.  Keeping
    this vocabulary closed prevents an application preference from becoming
    an arbitrary (and potentially credential-bearing) object store.
    """

    if not isinstance(value, Mapping):
        raise TypeError("workspace_layout must be an object")
    unknown = frozenset(value) - _WORKSPACE_LAYOUT_KEYS
    if unknown:
        raise ValueError(
            f"workspace_layout has unsupported fields: {sorted(unknown)}"
        )

    result: dict[str, Any] = {}
    for field_name in (
        "activity_dock_expanded",
        "geometry_present",
        "window_state_present",
    ):
        if field_name not in value:
            continue
        field_value = value[field_name]
        if not isinstance(field_value, bool):
            raise TypeError(f"workspace_layout.{field_name} must be a boolean")
        result[field_name] = field_value

    if "activity_dock_height" in value:
        height = value["activity_dock_height"]
        if isinstance(height, bool) or not isinstance(height, int):
            raise TypeError("workspace_layout.activity_dock_height must be an integer")
        if not 1 <= height <= 4096:
            raise ValueError(
                "workspace_layout.activity_dock_height must be between 1 and 4096"
            )
        result["activity_dock_height"] = height

    if "activity_dock_tab" in value:
        tab = value["activity_dock_tab"]
        if not isinstance(tab, str):
            raise TypeError("workspace_layout.activity_dock_tab must be a string")
        if tab not in _ALLOWED_ACTIVITY_DOCK_TABS:
            raise ValueError(f"unsupported Activity dock tab: {tab!r}")
        result["activity_dock_tab"] = tab

    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class ApplicationPreferences:
    theme: str = "dark"
    density: str = "comfortable"
    font_scale: int = 100
    reduced_motion: bool = True
    ui_language: str = "system"
    new_project_location: str = HISTORICAL_WINDOWS_PROJECT_LOCATION
    autosave_interval_seconds: int = 30
    open_last_project: str = "ask"
    shortcut_bindings: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SHORTCUT_BINDINGS)
    )
    recent_projects: tuple[str, ...] = ()
    workspace_layout: Mapping[str, Any] = field(default_factory=dict)
    zoom_default: float = 1.0
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        if self.theme not in _ALLOWED_THEMES:
            raise ValueError(f"unsupported theme: {self.theme}")
        if self.density not in _ALLOWED_UI_DENSITIES:
            raise ValueError(f"unsupported UI density: {self.density}")
        if isinstance(self.font_scale, bool) or not isinstance(self.font_scale, int):
            raise TypeError("font_scale must be an integer percentage")
        if self.font_scale not in _ALLOWED_FONT_SCALES:
            raise ValueError(f"unsupported font scale: {self.font_scale}")
        if not isinstance(self.reduced_motion, bool):
            raise TypeError("reduced_motion must be a boolean")
        if self.ui_language not in _ALLOWED_UI_LANGUAGES:
            raise ValueError(f"unsupported UI language: {self.ui_language}")
        object.__setattr__(
            self,
            "new_project_location",
            _require_non_empty(
                self.new_project_location,
                "new_project_location",
            ),
        )
        if (
            isinstance(self.autosave_interval_seconds, bool)
            or not isinstance(self.autosave_interval_seconds, int)
            or not 5 <= self.autosave_interval_seconds <= 3600
        ):
            raise ValueError(
                "autosave_interval_seconds must be an integer from 5 through 3600"
            )
        if self.open_last_project not in _ALLOWED_OPEN_LAST_PROJECT_POLICIES:
            raise ValueError(
                f"unsupported open-last-project policy: {self.open_last_project}"
            )
        object.__setattr__(
            self,
            "shortcut_bindings",
            _freeze_shortcut_bindings(self.shortcut_bindings),
        )
        object.__setattr__(
            self,
            "recent_projects",
            _string_tuple(self.recent_projects, "recent_projects"),
        )
        object.__setattr__(
            self,
            "workspace_layout",
            _freeze_workspace_layout(self.workspace_layout),
        )
        object.__setattr__(
            self,
            "zoom_default",
            _require_number(
                self.zoom_default,
                "zoom_default",
                minimum=0.05,
                maximum=32.0,
            ),
        )

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.APPLICATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "theme": self.theme,
            "density": self.density,
            "font_scale": self.font_scale,
            "reduced_motion": self.reduced_motion,
            "ui_language": self.ui_language,
            "new_project_location": self.new_project_location,
            "autosave_interval_seconds": self.autosave_interval_seconds,
            "open_last_project": self.open_last_project,
            "shortcut_bindings": dict(self.shortcut_bindings),
            "recent_projects": list(self.recent_projects),
            "workspace_layout": thaw_json(self.workspace_layout),
            "zoom_default": self.zoom_default,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    source_language: str = "Japanese"
    target_language: str = "Simplified Chinese"
    output_suffix: str = "_translated"
    output_convention: str = "sibling_output_folder"
    completed_page_policy: str = "open_for_review"
    glossary_reference: str | None = None
    selected_module_policies: Mapping[str, Any] = field(default_factory=dict)
    provider_profile_references: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        if self.source_language not in _ALLOWED_SOURCE_LANGUAGES:
            raise ValueError(f"unsupported source language: {self.source_language}")
        if self.target_language not in _ALLOWED_TARGET_LANGUAGES:
            raise ValueError(f"unsupported target language: {self.target_language}")
        object.__setattr__(
            self, "output_suffix", _require_non_empty(self.output_suffix, "output_suffix")
        )
        if self.output_convention not in _ALLOWED_OUTPUT_CONVENTIONS:
            raise ValueError(
                f"unsupported output convention: {self.output_convention}"
            )
        if self.completed_page_policy not in _ALLOWED_COMPLETED_PAGE_POLICIES:
            raise ValueError(
                f"unsupported completed-page policy: {self.completed_page_policy}"
            )
        object.__setattr__(
            self,
            "glossary_reference",
            _require_optional_string(self.glossary_reference, "glossary_reference"),
        )
        object.__setattr__(
            self,
            "selected_module_policies",
            freeze_json(
                self.selected_module_policies,
                field_name="selected_module_policies",
            ),
        )
        references: dict[str, str] = {}
        for role, profile_id in self.provider_profile_references.items():
            if role not in {"translation", "discovery"}:
                raise ValueError(f"unsupported provider profile role: {role!r}")
            references[role] = _require_non_empty(
                profile_id,
                f"provider_profile_references.{role}",
            )
        object.__setattr__(
            self,
            "provider_profile_references",
            MappingProxyType(dict(sorted(references.items()))),
        )

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.PROJECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "output_suffix": self.output_suffix,
            "output_convention": self.output_convention,
            "completed_page_policy": self.completed_page_policy,
            "glossary_reference": self.glossary_reference,
            "selected_module_policies": thaw_json(self.selected_module_policies),
            "provider_profile_references": dict(
                self.provider_profile_references
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    module_id: str
    module_schema_version: str
    values: Mapping[str, Any] = field(default_factory=dict)
    legacy_values: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        object.__setattr__(self, "module_id", _require_non_empty(self.module_id, "module_id"))
        object.__setattr__(
            self,
            "module_schema_version",
            _require_non_empty(self.module_schema_version, "module_schema_version"),
        )
        frozen_values = freeze_json(self.values, field_name="module.values")
        frozen_legacy = freeze_json(
            self.legacy_values, field_name="module.legacy_values"
        )
        overlap = frozenset(frozen_values) & frozenset(frozen_legacy)
        if overlap:
            raise ValueError(
                f"module values cannot be both current and legacy: {sorted(overlap)}"
            )
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(self, "legacy_values", frozen_legacy)

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.MODULE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "module_id": self.module_id,
            "module_schema_version": self.module_schema_version,
            "values": thaw_json(self.values),
            "legacy_values": thaw_json(self.legacy_values),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class CredentialReference:
    kind: CredentialReferenceKind
    reference: str
    label: str = ""
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        kind = CredentialReferenceKind(self.kind)
        reference = _require_non_empty(self.reference, "credential reference")
        if kind is CredentialReferenceKind.ENVIRONMENT_VARIABLE and not (
            _ENVIRONMENT_NAME_PATTERN.fullmatch(reference)
        ):
            raise ValueError("environment credential reference is not a valid variable name")
        if any(character in reference for character in ("\r", "\n", "\0")):
            raise ValueError("credential reference contains invalid control characters")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reference", reference)
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        object.__setattr__(self, "label", self.label.strip())

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.CREDENTIAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "kind": self.kind.value,
            "reference": self.reference,
            "label": self.label,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class EditorState:
    open_project_id: str | None = None
    open_page_id: str | None = None
    selected_parent_id: str | None = None
    workspace_layout_id: str = "default"
    panel_arrangement: Mapping[str, Any] = field(default_factory=dict)
    canvas_mode: str = "final"
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        for field_name in ("open_project_id", "open_page_id", "selected_parent_id"):
            object.__setattr__(
                self,
                field_name,
                _require_optional_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "workspace_layout_id",
            _require_non_empty(self.workspace_layout_id, "workspace_layout_id"),
        )
        if self.canvas_mode not in _ALLOWED_CANVAS_MODES:
            raise ValueError(f"unsupported canvas mode: {self.canvas_mode}")
        object.__setattr__(
            self,
            "panel_arrangement",
            freeze_json(self.panel_arrangement, field_name="panel_arrangement"),
        )

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.EDITOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "open_project_id": self.open_project_id,
            "open_page_id": self.open_page_id,
            "selected_parent_id": self.selected_parent_id,
            "workspace_layout_id": self.workspace_layout_id,
            "panel_arrangement": thaw_json(self.panel_arrangement),
            "canvas_mode": self.canvas_mode,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    provider_health: ProviderHealth = ProviderHealth.UNKNOWN
    installed_assets: Mapping[str, Any] = field(default_factory=dict)
    primary_device: str | None = None
    fallback_device: str | None = None
    download_state: DownloadState = DownloadState.IDLE
    detail: str = ""
    schema_version: str = SETTINGS_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SETTINGS_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema: {self.schema_version}")
        object.__setattr__(self, "provider_health", ProviderHealth(self.provider_health))
        object.__setattr__(self, "download_state", DownloadState(self.download_state))
        object.__setattr__(
            self,
            "installed_assets",
            freeze_json(self.installed_assets, field_name="installed_assets"),
        )
        for field_name in ("primary_device", "fallback_device"):
            object.__setattr__(
                self,
                field_name,
                _require_optional_string(getattr(self, field_name), field_name),
            )
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.RUNTIME

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.value,
            "provider_health": self.provider_health.value,
            "installed_assets": thaw_json(self.installed_assets),
            "primary_device": self.primary_device,
            "fallback_device": self.fallback_device,
            "download_state": self.download_state.value,
            "detail": self.detail,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class RunSettingsSnapshot:
    """Immutable, public configuration used by one workflow run.

    ``settings_fingerprint`` excludes the creation timestamp and is therefore
    stable for semantically identical runs.  ``snapshot_id`` is derived from
    that semantic fingerprint rather than assigned by a widget.
    """

    project_id: str
    pipeline_values: Mapping[str, Any]
    provider_profile_snapshot: Mapping[str, Any]
    scope_fingerprints: Mapping[str, str]
    unresolved_requirements: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    schema_version: str = RUN_SETTINGS_SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != RUN_SETTINGS_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported run settings schema: {self.schema_version}")
        object.__setattr__(self, "project_id", _require_non_empty(self.project_id, "project_id"))
        object.__setattr__(
            self,
            "pipeline_values",
            freeze_json(self.pipeline_values, field_name="pipeline_values"),
        )
        object.__setattr__(
            self,
            "provider_profile_snapshot",
            freeze_json(
                self.provider_profile_snapshot,
                field_name="provider_profile_snapshot",
            ),
        )
        frozen_scope_fingerprints = freeze_json(
            self.scope_fingerprints, field_name="scope_fingerprints"
        )
        for scope_name, fingerprint in frozen_scope_fingerprints.items():
            try:
                SettingsScope(scope_name)
            except ValueError as exc:
                raise ValueError(f"unknown scope fingerprint: {scope_name}") from exc
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError(
                    f"scope fingerprint for {scope_name} must be lowercase SHA-256"
                )
        object.__setattr__(self, "scope_fingerprints", frozen_scope_fingerprints)
        object.__setattr__(
            self,
            "unresolved_requirements",
            _string_tuple(self.unresolved_requirements, "unresolved_requirements"),
        )
        object.__setattr__(
            self, "created_at", _require_utc_timestamp(self.created_at, "created_at")
        )
        expected_snapshot_id = f"run-settings:{self.settings_fingerprint}"
        if self.snapshot_id and self.snapshot_id != expected_snapshot_id:
            raise ValueError("snapshot_id does not match the semantic settings fingerprint")
        object.__setattr__(self, "snapshot_id", expected_snapshot_id)

    @property
    def scope(self) -> SettingsScope:
        return SettingsScope.RUN

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "pipeline_values": thaw_json(self.pipeline_values),
            "provider_profile_snapshot": thaw_json(self.provider_profile_snapshot),
            "scope_fingerprints": thaw_json(self.scope_fingerprints),
            "unresolved_requirements": list(self.unresolved_requirements),
        }

    @property
    def settings_fingerprint(self) -> str:
        return canonical_fingerprint(self._semantic_dict())

    @property
    def fingerprint(self) -> str:
        return self.settings_fingerprint

    def to_dict(self) -> dict[str, Any]:
        result = self._semantic_dict()
        result.update(
            {
                "scope": self.scope.value,
                "snapshot_id": self.snapshot_id,
                "created_at": self.created_at,
                "settings_fingerprint": self.settings_fingerprint,
            }
        )
        return result


def project_config_from_dict(payload: Mapping[str, Any]) -> ProjectConfig:
    """Strictly decode one public project-settings contract."""

    if not isinstance(payload, Mapping):
        raise TypeError("ProjectConfig must be a mapping")
    _require_exact_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "scope",
                "source_language",
                "target_language",
                "output_suffix",
                "glossary_reference",
                "selected_module_policies",
                "provider_profile_references",
            }
        ),
        optional=frozenset({"output_convention", "completed_page_policy"}),
        contract_name="ProjectConfig",
    )
    if payload["scope"] != SettingsScope.PROJECT.value:
        raise ValueError("ProjectConfig scope is invalid")
    if not isinstance(payload["selected_module_policies"], Mapping):
        raise TypeError("selected_module_policies must be a mapping")
    if not isinstance(payload["provider_profile_references"], Mapping):
        raise TypeError("provider_profile_references must be a mapping")
    return ProjectConfig(
        source_language=payload["source_language"],
        target_language=payload["target_language"],
        output_suffix=payload["output_suffix"],
        output_convention=payload.get(
            "output_convention", "sibling_output_folder"
        ),
        completed_page_policy=payload.get(
            "completed_page_policy", "open_for_review"
        ),
        glossary_reference=payload["glossary_reference"],
        selected_module_policies=payload["selected_module_policies"],
        provider_profile_references=payload["provider_profile_references"],
        schema_version=payload["schema_version"],
    )


def module_config_from_dict(payload: Mapping[str, Any]) -> ModuleConfig:
    """Strictly decode one module-settings contract."""

    if not isinstance(payload, Mapping):
        raise TypeError("ModuleConfig must be a mapping")
    _require_exact_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "scope",
                "module_id",
                "module_schema_version",
                "values",
                "legacy_values",
            }
        ),
        contract_name="ModuleConfig",
    )
    if payload["scope"] != SettingsScope.MODULE.value:
        raise ValueError("ModuleConfig scope is invalid")
    if not isinstance(payload["values"], Mapping):
        raise TypeError("ModuleConfig values must be a mapping")
    if not isinstance(payload["legacy_values"], Mapping):
        raise TypeError("ModuleConfig legacy_values must be a mapping")
    return ModuleConfig(
        module_id=payload["module_id"],
        module_schema_version=payload["module_schema_version"],
        values=payload["values"],
        legacy_values=payload["legacy_values"],
        schema_version=payload["schema_version"],
    )


def run_settings_snapshot_from_dict(
    payload: Mapping[str, Any],
) -> RunSettingsSnapshot:
    """Strictly decode and fingerprint-check a persisted run snapshot."""

    if not isinstance(payload, Mapping):
        raise TypeError("RunSettingsSnapshot must be a mapping")
    _require_exact_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "scope",
                "project_id",
                "pipeline_values",
                "provider_profile_snapshot",
                "scope_fingerprints",
                "unresolved_requirements",
                "snapshot_id",
                "created_at",
                "settings_fingerprint",
            }
        ),
        contract_name="RunSettingsSnapshot",
    )
    if payload["scope"] != SettingsScope.RUN.value:
        raise ValueError("RunSettingsSnapshot scope is invalid")
    for field_name in (
        "pipeline_values",
        "provider_profile_snapshot",
        "scope_fingerprints",
    ):
        if not isinstance(payload[field_name], Mapping):
            raise TypeError(f"RunSettingsSnapshot {field_name} must be a mapping")
    if isinstance(payload["unresolved_requirements"], (str, bytes, bytearray)) or not isinstance(
        payload["unresolved_requirements"], Sequence
    ):
        raise TypeError("RunSettingsSnapshot unresolved_requirements must be a sequence")
    snapshot = RunSettingsSnapshot(
        project_id=payload["project_id"],
        pipeline_values=payload["pipeline_values"],
        provider_profile_snapshot=payload["provider_profile_snapshot"],
        scope_fingerprints=payload["scope_fingerprints"],
        unresolved_requirements=tuple(payload["unresolved_requirements"]),
        created_at=payload["created_at"],
        schema_version=payload["schema_version"],
        snapshot_id=payload["snapshot_id"],
    )
    if payload["settings_fingerprint"] != snapshot.settings_fingerprint:
        raise ValueError("RunSettingsSnapshot settings fingerprint mismatch")
    return snapshot
