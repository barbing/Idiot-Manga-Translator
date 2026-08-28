# -*- coding: utf-8 -*-
"""Atomic public application-settings persistence for GUI-2."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlparse

from app.config.module_registry import (
    DEFAULT_MODULE_REGISTRY,
    ModuleSchemaRegistry,
)
from app.config.settings_contracts import (
    ApplicationPreferences,
    DEFAULT_SHORTCUT_BINDINGS,
    HISTORICAL_WINDOWS_PROJECT_LOCATION,
    ModuleConfig,
    SettingsScope,
    canonical_fingerprint,
    freeze_json,
    thaw_json,
)
from app.config.settings_migration import migrate_platform_defaults
from app.platform_services.contracts import PlatformIdentity
from app.platform_services.paths import PlatformPaths, qt_platform_paths


APPLICATION_SETTINGS_STORE_SCHEMA_VERSION = 2
_SECRET_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|bearer|"
    r"authorization|password|passwd|secret|credential[_-]?ref|"
    r"credential[_-]?reference|credential[_-]?value)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"^\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"^\s*-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"^\s*sk-[A-Za-z0-9_-]{16,}\s*$"),
    re.compile(
        r"(?:^|[?&;\s])(?:api[_-]?key|token|secret|password)\s*=\s*\S+",
        re.IGNORECASE,
    ),
)


class ApplicationSettingsStoreError(RuntimeError):
    """Raised when public application settings cannot be validated or stored."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApplicationSettingsStoreError(
                f"application settings contain duplicate field {key!r}"
            )
        result[key] = value
    return result


def _require_schema_version(value: Any) -> int:
    if type(value) is not int:
        raise ApplicationSettingsStoreError(
            "application settings schema_version must be an integer"
        )
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    contract: str,
) -> None:
    missing = required - frozenset(payload)
    extra = frozenset(payload) - required - optional
    if missing:
        raise ApplicationSettingsStoreError(
            f"{contract} is missing fields: {sorted(missing)}"
        )
    if extra:
        raise ApplicationSettingsStoreError(
            f"{contract} has unsupported fields: {sorted(extra)}"
        )


def _assert_public_json(value: Any, *, path: str = "settings") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ApplicationSettingsStoreError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise ApplicationSettingsStoreError(
                f"{path} appears to contain secret material"
            )
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and (
            parsed.username is not None or parsed.password is not None
        ):
            raise ApplicationSettingsStoreError(
                f"{path} contains embedded URL credentials"
            )
        if parsed.scheme in {"http", "https"} and any(
            _SECRET_KEY_PATTERN.search(key)
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise ApplicationSettingsStoreError(
                f"{path} contains a secret-bearing URL query"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ApplicationSettingsStoreError(f"{path} has a non-text key")
            if _SECRET_KEY_PATTERN.search(key):
                raise ApplicationSettingsStoreError(
                    f"{path} contains forbidden secret field {key!r}"
                )
            _assert_public_json(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_public_json(child, path=f"{path}[{index}]")
        return
    raise ApplicationSettingsStoreError(f"{path} is not public JSON data")


def _application_preferences_from_dict(
    payload: Mapping[str, Any],
) -> ApplicationPreferences:
    _require_exact_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "scope",
                "theme",
                "ui_language",
                "recent_projects",
                "workspace_layout",
                "zoom_default",
            }
        ),
        optional=frozenset(
            {
                "density",
                "font_scale",
                "reduced_motion",
                "new_project_location",
                "autosave_interval_seconds",
                "open_last_project",
                "shortcut_bindings",
            }
        ),
        contract="ApplicationPreferences",
    )
    if payload["scope"] != SettingsScope.APPLICATION.value:
        raise ApplicationSettingsStoreError("ApplicationPreferences scope is invalid")
    recent_projects = payload["recent_projects"]
    if isinstance(recent_projects, (str, bytes, bytearray)) or not isinstance(
        recent_projects, (list, tuple)
    ):
        raise ApplicationSettingsStoreError(
            "ApplicationPreferences recent_projects must be a sequence of strings"
        )
    if not isinstance(payload["workspace_layout"], Mapping):
        raise ApplicationSettingsStoreError(
            "ApplicationPreferences workspace_layout must be an object"
        )
    try:
        return ApplicationPreferences(
            theme=payload["theme"],
            density=payload.get("density", "comfortable"),
            font_scale=payload.get("font_scale", 100),
            reduced_motion=payload.get("reduced_motion", True),
            ui_language=payload["ui_language"],
            new_project_location=payload.get(
                "new_project_location", HISTORICAL_WINDOWS_PROJECT_LOCATION
            ),
            autosave_interval_seconds=payload.get(
                "autosave_interval_seconds", 30
            ),
            open_last_project=payload.get("open_last_project", "ask"),
            shortcut_bindings=payload.get(
                "shortcut_bindings", DEFAULT_SHORTCUT_BINDINGS
            ),
            recent_projects=recent_projects,
            workspace_layout=payload["workspace_layout"],
            zoom_default=payload["zoom_default"],
            schema_version=payload["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ApplicationSettingsStoreError(
            "ApplicationPreferences is invalid"
        ) from exc


def _module_config_from_dict(payload: Mapping[str, Any]) -> ModuleConfig:
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
        contract="ModuleConfig",
    )
    if payload["scope"] != SettingsScope.MODULE.value:
        raise ApplicationSettingsStoreError("ModuleConfig scope is invalid")
    if not isinstance(payload["values"], Mapping):
        raise ApplicationSettingsStoreError("ModuleConfig values must be an object")
    if not isinstance(payload["legacy_values"], Mapping):
        raise ApplicationSettingsStoreError(
            "ModuleConfig legacy_values must be an object"
        )
    try:
        return ModuleConfig(
            module_id=payload["module_id"],
            module_schema_version=payload["module_schema_version"],
            values=payload["values"],
            legacy_values=payload["legacy_values"],
            schema_version=payload["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ApplicationSettingsStoreError("ModuleConfig is invalid") from exc


def _validate_application_module_config(
    config: ModuleConfig,
    registry: ModuleSchemaRegistry,
) -> None:
    """Reject project/provider settings at the application-store boundary."""

    try:
        module = registry.get_module(config.module_id)
        registry.validate_config(config, allow_legacy=True)
    except (TypeError, ValueError) as exc:
        raise ApplicationSettingsStoreError(
            f"application module config {config.module_id!r} is invalid"
        ) from exc
    for collection_name, values in (
        ("values", config.values),
        ("legacy_values", config.legacy_values),
    ):
        for setting_id in values:
            definition = module.definitions.get(setting_id)
            if definition is None:
                raise ApplicationSettingsStoreError(
                    f"{config.module_id}.{setting_id} has no declared application scope"
                )
            if definition.scope is not SettingsScope.APPLICATION:
                raise ApplicationSettingsStoreError(
                    f"{config.module_id}.{setting_id} is {definition.scope.value}-scoped "
                    f"and cannot be stored in application settings {collection_name}"
                )


@dataclass(frozen=True, slots=True)
class LegacyMigrationIssueEvidence:
    """One public, inactive legacy-migration diagnostic."""

    key: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,127}", self.key
        ):
            raise ValueError("legacy migration issue key is invalid")
        if (
            not isinstance(self.reason, str)
            or not self.reason.strip()
            or len(self.reason) > 2048
            or any(ord(character) < 32 for character in self.reason)
        ):
            raise ValueError("legacy migration issue reason is invalid")
        _assert_public_json(self.to_dict(), path="legacy_migration_issue")

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "reason": self.reason}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LegacyMigrationIssueEvidence":
        _require_exact_keys(
            payload,
            required=frozenset({"key", "reason"}),
            contract="LegacyMigrationIssueEvidence",
        )
        try:
            return cls(key=payload["key"], reason=payload["reason"])
        except (TypeError, ValueError) as exc:
            raise ApplicationSettingsStoreError(
                "legacy migration issue evidence is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class InactiveLegacyMigrationEvidence:
    """Sanitized migration evidence that never becomes settings authority."""

    migration_version: int
    source_fingerprint: str
    legacy_values: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[LegacyMigrationIssueEvidence, ...] = ()
    unresolved_provider_profile_references: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if type(self.migration_version) is not int or self.migration_version < 1:
            raise ValueError("legacy migration version must be a positive integer")
        if not isinstance(self.source_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_fingerprint
        ):
            raise ValueError("legacy migration source fingerprint is invalid")
        legacy_values = freeze_json(
            self.legacy_values,
            field_name="inactive_legacy_migration_evidence.legacy_values",
        )
        issues = tuple(self.issues)
        if any(not isinstance(issue, LegacyMigrationIssueEvidence) for issue in issues):
            raise TypeError(
                "legacy migration issues must contain LegacyMigrationIssueEvidence"
            )
        unresolved = freeze_json(
            self.unresolved_provider_profile_references,
            field_name=(
                "inactive_legacy_migration_evidence."
                "unresolved_provider_profile_references"
            ),
        )
        for role, profile_id in unresolved.items():
            if role not in {"translation", "discovery"}:
                raise ValueError("legacy migration provider role is invalid")
            if not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError(
                    "legacy migration provider references require non-empty text"
                )
        object.__setattr__(self, "legacy_values", legacy_values)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(
            self,
            "unresolved_provider_profile_references",
            unresolved,
        )
        _assert_public_json(self.to_dict(), path="legacy_migration_evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_version": self.migration_version,
            "source_fingerprint": self.source_fingerprint,
            "legacy_values": thaw_json(self.legacy_values),
            "issues": [issue.to_dict() for issue in self.issues],
            "unresolved_provider_profile_references": thaw_json(
                self.unresolved_provider_profile_references
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "InactiveLegacyMigrationEvidence":
        _assert_public_json(payload, path="legacy_migration_evidence")
        _require_exact_keys(
            payload,
            required=frozenset(
                {
                    "migration_version",
                    "source_fingerprint",
                    "legacy_values",
                    "issues",
                    "unresolved_provider_profile_references",
                }
            ),
            contract="InactiveLegacyMigrationEvidence",
        )
        if not isinstance(payload["legacy_values"], Mapping):
            raise ApplicationSettingsStoreError(
                "legacy migration legacy_values must be an object"
            )
        raw_issues = payload["issues"]
        if not isinstance(raw_issues, list):
            raise ApplicationSettingsStoreError(
                "legacy migration issues must be a list"
            )
        if not isinstance(
            payload["unresolved_provider_profile_references"], Mapping
        ):
            raise ApplicationSettingsStoreError(
                "legacy migration unresolved provider references must be an object"
            )
        issues: list[LegacyMigrationIssueEvidence] = []
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, Mapping):
                raise ApplicationSettingsStoreError(
                    "legacy migration issue must be an object"
                )
            issues.append(LegacyMigrationIssueEvidence.from_dict(raw_issue))
        try:
            return cls(
                migration_version=payload["migration_version"],
                source_fingerprint=payload["source_fingerprint"],
                legacy_values=payload["legacy_values"],
                issues=tuple(issues),
                unresolved_provider_profile_references=payload[
                    "unresolved_provider_profile_references"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise ApplicationSettingsStoreError(
                "legacy migration evidence is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class ApplicationSettingsDocument:
    """Typed public application state with no project/provider authority."""

    application_preferences: ApplicationPreferences = field(
        default_factory=ApplicationPreferences
    )
    application_module_configs: tuple[ModuleConfig, ...] = ()
    migration_markers: tuple[str, ...] = ()
    legacy_migration_evidence: tuple[InactiveLegacyMigrationEvidence, ...] = ()
    schema_version: int = APPLICATION_SETTINGS_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != APPLICATION_SETTINGS_STORE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported application settings store schema")
        modules = tuple(
            sorted(
                self.application_module_configs,
                key=lambda config: config.module_id,
            )
        )
        if len({config.module_id for config in modules}) != len(modules):
            raise ValueError("application settings contain duplicate module configs")
        for config in modules:
            _validate_application_module_config(config, DEFAULT_MODULE_REGISTRY)
        markers = tuple(sorted(set(self.migration_markers)))
        if any(not isinstance(marker, str) or not marker.strip() for marker in markers):
            raise ValueError("migration markers must be non-empty strings")
        raw_evidence = tuple(self.legacy_migration_evidence)
        if any(
            not isinstance(item, InactiveLegacyMigrationEvidence)
            for item in raw_evidence
        ):
            raise TypeError(
                "legacy_migration_evidence must contain inactive evidence"
            )
        evidence = tuple(
            sorted(
                raw_evidence,
                key=lambda item: (item.migration_version, item.source_fingerprint),
            )
        )
        if len({item.source_fingerprint for item in evidence}) != len(evidence):
            raise ValueError("legacy migration evidence contains duplicate sources")
        object.__setattr__(
            self,
            "application_module_configs",
            modules,
        )
        object.__setattr__(self, "migration_markers", markers)
        object.__setattr__(self, "legacy_migration_evidence", evidence)
        _assert_public_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "application_preferences": self.application_preferences.to_dict(),
            "application_module_configs": [
                config.to_dict() for config in self.application_module_configs
            ],
            "migration_markers": list(self.migration_markers),
            "legacy_migration_evidence": [
                item.to_dict() for item in self.legacy_migration_evidence
            ],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
    ) -> "ApplicationSettingsDocument":
        _assert_public_json(payload)
        schema_version = _require_schema_version(payload.get("schema_version"))
        if schema_version == 1:
            _require_exact_keys(
                payload,
                required=frozenset(
                    {
                        "schema_version",
                        "application_preferences",
                        "application_module_configs",
                        "migration_markers",
                    }
                ),
                contract="ApplicationSettingsDocumentV1",
            )
            raw_evidence: Any = []
        elif schema_version == APPLICATION_SETTINGS_STORE_SCHEMA_VERSION:
            _require_exact_keys(
                payload,
                required=frozenset(
                    {
                        "schema_version",
                        "application_preferences",
                        "application_module_configs",
                        "migration_markers",
                        "legacy_migration_evidence",
                    }
                ),
                contract="ApplicationSettingsDocument",
            )
            raw_evidence = payload["legacy_migration_evidence"]
        else:
            raise ApplicationSettingsStoreError(
                "unsupported application settings store schema"
            )
        raw_preferences = payload["application_preferences"]
        raw_modules = payload["application_module_configs"]
        raw_markers = payload["migration_markers"]
        if not isinstance(raw_preferences, Mapping):
            raise ApplicationSettingsStoreError("application_preferences must be an object")
        if not isinstance(raw_modules, list):
            raise ApplicationSettingsStoreError(
                "application_module_configs must be a list"
            )
        if not isinstance(raw_markers, list):
            raise ApplicationSettingsStoreError("migration_markers must be a list")
        if not isinstance(raw_evidence, list):
            raise ApplicationSettingsStoreError(
                "legacy_migration_evidence must be a list"
            )
        modules: list[ModuleConfig] = []
        for raw_module in raw_modules:
            if not isinstance(raw_module, Mapping):
                raise ApplicationSettingsStoreError("ModuleConfig must be an object")
            config = _module_config_from_dict(raw_module)
            _validate_application_module_config(config, registry)
            modules.append(config)
        evidence: list[InactiveLegacyMigrationEvidence] = []
        for raw_item in raw_evidence:
            if not isinstance(raw_item, Mapping):
                raise ApplicationSettingsStoreError(
                    "legacy migration evidence must be an object"
                )
            evidence.append(InactiveLegacyMigrationEvidence.from_dict(raw_item))
        try:
            return cls(
                application_preferences=_application_preferences_from_dict(
                    raw_preferences
                ),
                application_module_configs=tuple(modules),
                migration_markers=tuple(raw_markers),
                legacy_migration_evidence=tuple(evidence),
                schema_version=APPLICATION_SETTINGS_STORE_SCHEMA_VERSION,
            )
        except (TypeError, ValueError) as exc:
            raise ApplicationSettingsStoreError(
                "application settings document is invalid"
            ) from exc


class ApplicationSettingsStore:
    """Atomic JSON persistence for non-secret application settings."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
        platform_identity: PlatformIdentity | None = None,
        platform_paths: PlatformPaths | None = None,
    ) -> None:
        self.path = Path(path)
        self.registry = registry
        self.platform_identity = platform_identity or PlatformIdentity.detect()
        if not isinstance(self.platform_identity, PlatformIdentity):
            raise TypeError("platform_identity must be PlatformIdentity")
        if platform_paths is not None and not isinstance(platform_paths, PlatformPaths):
            raise TypeError("platform_paths must be PlatformPaths")
        self.platform_paths = platform_paths

    def _with_platform_defaults(
        self,
        document: ApplicationSettingsDocument,
    ) -> ApplicationSettingsDocument:
        paths = self.platform_paths or qt_platform_paths()
        preferences = migrate_platform_defaults(
            document.application_preferences,
            self.platform_identity,
            paths,
        )
        if preferences is document.application_preferences:
            return document
        return replace(document, application_preferences=preferences)

    def default_document(self) -> ApplicationSettingsDocument:
        """Return platform-correct defaults without reading or writing the store."""

        return self._with_platform_defaults(ApplicationSettingsDocument())

    def load(self) -> ApplicationSettingsDocument:
        if not self.path.exists():
            return self.default_document()
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
        except ApplicationSettingsStoreError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ApplicationSettingsStoreError(
                "application settings could not be read"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ApplicationSettingsStoreError(
                "application settings root must be an object"
            )
        return self._with_platform_defaults(
            ApplicationSettingsDocument.from_dict(
                payload,
                registry=self.registry,
            )
        )

    def save(self, document: ApplicationSettingsDocument) -> str:
        if not isinstance(document, ApplicationSettingsDocument):
            raise TypeError("document must be an ApplicationSettingsDocument")
        document = self._with_platform_defaults(document)
        payload = document.to_dict()
        _assert_public_json(payload)
        # Reparse before publication so an invalid in-memory object cannot
        # bypass registry validation.
        validated = ApplicationSettingsDocument.from_dict(
            payload,
            registry=self.registry,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    validated.to_dict(),
                    stream,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return validated.fingerprint
