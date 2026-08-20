# -*- coding: utf-8 -*-
"""Deterministic migration of the legacy ``MangaTranslator/Pro`` QSettings.

This module intentionally depends on a tiny reader protocol instead of
PySide6.  The migration reads only the explicit keys historically owned by
``MainWindow._save_settings``/``_load_saved_settings``.  It never enumerates a
QSettings store, so unrelated or secret-bearing entries cannot accidentally
become part of the new public settings model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlparse

from app.config.defaults import AppDefaults
from app.config.credential_store import environment_credential_reference
from app.config.module_registry import (
    DEFAULT_MODULE_REGISTRY,
    ModuleSchemaRegistry,
    SettingLifecycle,
)
from app.config.provider_profiles import (
    GGUFProviderOptions,
    GenerationSettings,
    ModelGenerationOverride,
    OllamaProviderOptions,
    ProviderKind,
    ProviderProfile,
    ProviderProfileError,
)
from app.config.settings_contracts import (
    ApplicationPreferences,
    ModuleConfig,
    ProjectConfig,
    canonical_fingerprint,
    freeze_json,
    thaw_json,
)


LEGACY_QSETTINGS_ORGANIZATION = "MangaTranslator"
LEGACY_QSETTINGS_APPLICATION = "Pro"
LEGACY_SETTINGS_MIGRATION_VERSION = 1

# Exact keys written or restored by the current legacy main window.  Keep this
# closed: callers must not add discovery/enumeration around this migration.
LEGACY_QSETTINGS_ALLOWLIST = frozenset(
    {
        "geometry",
        "windowState",
        "import_dir",
        "export_dir",
        "json_path",
        "source_lang",
        "target_lang",
        "detector_engine",
        "detector_input_size",
        "ocr_engine",
        "translator_backend",
        "inpaint_mode",
        "filter_strength",
        "use_gpu",
        "fast_mode",
        "auto_glossary",
        "use_ollama_discovery",
        "prescan_enabled",
        "font_name",
        "gguf_model_path",
        "gguf_n_gpu_layers",
        "gen_preset",
        "ollama_temp",
        "ollama_top_p",
        "ollama_ctx",
        "deepseek_model",
        "deepseek_base_url",
        "gguf_temp",
        "gguf_top_p",
        "discovery_backend",
        "discovery_ollama_model",
        "discovery_gguf_path",
        "model_overrides",
    }
)

_OCR_ALIASES = {
    "paddleocr": "PaddleOCR-VL",
    "paddleocrvl": "PaddleOCR-VL",
    "paddleocr-vl": "PaddleOCR-VL",
    "paddleocr-v1.6": "PaddleOCR-VL",
    "paddleocrvl1.6": "PaddleOCR-VL",
    "mangaocr": "MangaOCR",
    "manga-ocr": "MangaOCR",
}
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_OVERRIDE_KEY_PATTERN = re.compile(r"^(ollama|gguf)::(.+)$", re.IGNORECASE)
_MAX_OVERRIDE_JSON_BYTES = 1_048_576
_MAX_OVERRIDE_ENTRIES = 256
_PROFILE_IDS = {
    "GGUF": "legacy-translation-gguf",
    "Ollama": "legacy-translation-ollama",
    "DeepSeek": "legacy-translation-deepseek",
}


@runtime_checkable
class LegacySettingsReader(Protocol):
    """The safe subset of QSettings used by the migration adapter."""

    def contains(self, key: str) -> bool:
        """Return whether an exact allowlisted key exists."""

    def value(self, key: str) -> Any:
        """Return the value for an exact allowlisted key."""


@dataclass(frozen=True, slots=True)
class MigrationIssue:
    key: str
    reason: str

    def __post_init__(self) -> None:
        if self.key not in LEGACY_QSETTINGS_ALLOWLIST:
            raise ValueError("migration issues may name only allowlisted keys")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("migration issue reason is required")

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class LegacyRunInvocationDefaults:
    """Typed, transient defaults for the first post-migration run.

    Legacy source/output/project paths are workflow invocation inputs, not
    application-window layout.  They remain outside persisted application
    preferences and can be offered to the user without silently starting a
    run or changing the single-page lifecycle.
    """

    import_dir: str = ""
    export_dir: str = ""
    json_path: str = ""

    def __post_init__(self) -> None:
        for field_name in ("import_dir", "export_dir", "json_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be text")
            if any(ord(character) < 32 for character in value):
                raise ValueError(f"{field_name} must not contain control characters")
            object.__setattr__(self, field_name, value.strip())

    @property
    def is_empty(self) -> bool:
        return not any((self.import_dir, self.export_dir, self.json_path))

    def to_dict(self) -> dict[str, str]:
        return {
            "import_dir": self.import_dir,
            "export_dir": self.export_dir,
            "json_path": self.json_path,
        }


@dataclass(frozen=True, slots=True)
class LegacySettingsMigrationResult:
    """Public, deterministic output of one legacy settings read."""

    application_preferences: ApplicationPreferences
    project_config: ProjectConfig
    module_configs: tuple[ModuleConfig, ...]
    provider_profiles: tuple[ProviderProfile, ...]
    run_invocation_defaults: LegacyRunInvocationDefaults = field(
        default_factory=LegacyRunInvocationDefaults
    )
    unresolved_provider_profile_references: Mapping[str, str] = field(
        default_factory=dict
    )
    legacy_values: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[MigrationIssue, ...] = ()
    migrated_keys: tuple[str, ...] = ()
    source_fingerprint: str = ""
    migration_marker: str = ""
    migration_version: int = LEGACY_SETTINGS_MIGRATION_VERSION

    def __post_init__(self) -> None:
        if self.migration_version != LEGACY_SETTINGS_MIGRATION_VERSION:
            raise ValueError("unsupported legacy settings migration version")
        modules = tuple(self.module_configs)
        if len({config.module_id for config in modules}) != len(modules):
            raise ValueError("migration result contains duplicate module configs")
        profiles = tuple(self.provider_profiles)
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise ValueError("migration result contains duplicate provider profiles")
        profile_ids = frozenset(profile.profile_id for profile in profiles)
        dangling = frozenset(self.project_config.provider_profile_references.values()) - profile_ids
        if dangling:
            raise ValueError(
                f"project provider profile references are dangling: {sorted(dangling)}"
            )
        unresolved = freeze_json(
            self.unresolved_provider_profile_references,
            field_name="unresolved_provider_profile_references",
        )
        for role, profile_id in unresolved.items():
            if role not in {"translation", "discovery"}:
                raise ValueError(f"unsupported unresolved provider role: {role!r}")
            if not str(role).strip() or not isinstance(profile_id, str) or not profile_id.strip():
                raise ValueError("unresolved provider references require non-empty strings")
            if role in self.project_config.provider_profile_references:
                raise ValueError(
                    f"provider role {role!r} cannot be both resolved and unresolved"
                )
            if profile_id in profile_ids:
                raise ValueError(
                    f"unresolved provider profile {profile_id!r} already exists"
                )
        object.__setattr__(self, "module_configs", modules)
        object.__setattr__(self, "provider_profiles", profiles)
        object.__setattr__(
            self,
            "unresolved_provider_profile_references",
            unresolved,
        )
        object.__setattr__(
            self,
            "legacy_values",
            freeze_json(self.legacy_values, field_name="legacy_values"),
        )
        object.__setattr__(self, "issues", tuple(self.issues))
        migrated = tuple(sorted(set(self.migrated_keys)))
        if any(key not in LEGACY_QSETTINGS_ALLOWLIST for key in migrated):
            raise ValueError("migrated_keys contains a non-allowlisted key")
        object.__setattr__(self, "migrated_keys", migrated)
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_fingerprint):
            raise ValueError("source_fingerprint must be a SHA-256 digest")
        expected_marker = (
            f"legacy-qsettings-v{self.migration_version}:{self.source_fingerprint}"
        )
        if self.migration_marker != expected_marker:
            raise ValueError("migration marker does not match its source fingerprint")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "migration_version": self.migration_version,
            "source_fingerprint": self.source_fingerprint,
            "migration_marker": self.migration_marker,
            "migrated_keys": list(self.migrated_keys),
            "application_preferences": self.application_preferences.to_dict(),
            "project_config": self.project_config.to_dict(),
            "module_configs": [config.to_dict() for config in self.module_configs],
            "provider_profiles": [
                profile.to_store_dict() for profile in self.provider_profiles
            ],
            "run_invocation_defaults": self.run_invocation_defaults.to_dict(),
            "unresolved_provider_profile_references": thaw_json(
                self.unresolved_provider_profile_references
            ),
            "legacy_values": thaw_json(self.legacy_values),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _safe_fingerprint_value(value: Any) -> Any:
    """Describe a legacy value without copying opaque or possibly secret text."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "non_finite_number"}
    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
        kind = "bytes"
    elif isinstance(value, str):
        payload = value.encode("utf-8", errors="replace")
        kind = "str"
    else:
        # Qt may expose opaque byte-array values. Use the buffer protocol when
        # available; otherwise retain only a stable type marker rather than a
        # repr that may contain a process-specific memory address.
        try:
            payload = memoryview(value).tobytes()
        except TypeError:
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}"
            }
        kind = type(value).__name__
    return {
        "type": kind,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_allowlisted_source(
    reader: LegacySettingsReader,
) -> tuple[dict[str, Any], str]:
    values: dict[str, Any] = {}
    fingerprint_payload: dict[str, Any] = {
        "organization": LEGACY_QSETTINGS_ORGANIZATION,
        "application": LEGACY_QSETTINGS_APPLICATION,
        "migration_version": LEGACY_SETTINGS_MIGRATION_VERSION,
        "values": {},
    }
    for key in sorted(LEGACY_QSETTINGS_ALLOWLIST):
        if reader.contains(key):
            value = reader.value(key)
            values[key] = value
            fingerprint_payload["values"][key] = _safe_fingerprint_value(value)
    return values, canonical_fingerprint(fingerprint_payload)


def _public_string(value: Any, *, allow_empty: bool = False) -> str:
    if value is None or isinstance(value, (bytes, bytearray)):
        raise ValueError("must be text")
    text = str(value).strip()
    if not text and not allow_empty:
        raise ValueError("must not be empty")
    if any(ord(character) < 32 for character in text):
        raise ValueError("must not contain control characters")
    return text


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError("must be a canonical boolean")


def _strict_int(
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and _INTEGER_PATTERN.fullmatch(value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError("must be an integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"must be at most {maximum}")
    return parsed


def _strict_float(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError("must be numeric")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str) and _NUMBER_PATTERN.fullmatch(value.strip()):
        parsed = float(value.strip())
    else:
        raise ValueError("must be numeric")
    if not math.isfinite(parsed):
        raise ValueError("must be finite")
    if minimum is not None and (
        parsed <= minimum if exclusive_minimum else parsed < minimum
    ):
        relation = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"must be {relation} {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"must be at most {maximum}")
    return parsed


def _normalized_ocr(value: Any) -> str:
    text = _public_string(value)
    normalized = text.replace("_", "-").replace(" ", "").lower()
    try:
        return _OCR_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError("is not a recognized OCR engine") from exc


def _safe_legacy_value(value: Any) -> Any:
    """Return a bounded public value suitable for explicit legacy evidence."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        text = value[:4096]
        parsed = urlparse(text)
        secret_query = any(
            re.search(
                r"(?:^|[_-])(?:api[_-]?key|token|secret|password)(?:$|[_-])",
                key,
                re.IGNORECASE,
            )
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
        if (
            re.search(r"^\s*Bearer\s+\S+", text, re.IGNORECASE)
            or re.fullmatch(r"\s*sk-[A-Za-z0-9_-]{16,}\s*", text)
            or parsed.username is not None
            or parsed.password is not None
            or secret_query
        ):
            return _safe_fingerprint_value(text)
        return text
    return _safe_fingerprint_value(value)


def _public_endpoint(value: Any) -> str:
    endpoint = _public_string(value)
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("must not contain a query or fragment")
    return endpoint.rstrip("/")


def _parse_model_overrides(
    raw_value: Any,
    issues: list[MigrationIssue],
) -> dict[str, dict[str, Any]]:
    if raw_value is None or raw_value == "":
        return {}
    if isinstance(raw_value, bytes):
        issues.append(MigrationIssue("model_overrides", "must be UTF-8 JSON text"))
        return {}
    text = str(raw_value)
    if len(text.encode("utf-8")) > _MAX_OVERRIDE_JSON_BYTES:
        issues.append(MigrationIssue("model_overrides", "JSON exceeds the migration size limit"))
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        issues.append(MigrationIssue("model_overrides", "contains invalid JSON"))
        return {}
    if not isinstance(payload, dict):
        issues.append(MigrationIssue("model_overrides", "JSON root must be an object"))
        return {}
    if len(payload) > _MAX_OVERRIDE_ENTRIES:
        issues.append(MigrationIssue("model_overrides", "contains too many model entries"))
        return {}

    sanitized: dict[str, dict[str, Any]] = {}
    allowed_values = {
        "ollama": {
            "ollama_temp": ("float", 0.0, 2.0),
            "ollama_top_p": ("float_exclusive", 0.0, 1.0),
            "ollama_ctx": ("int", 512, 32768),
        },
        "gguf": {
            "gguf_temp": ("float", 0.0, 2.0),
            "gguf_top_p": ("float_exclusive", 0.0, 1.0),
            "gguf_n_ctx": ("int", 512, 32768),
            "gguf_n_gpu_layers": ("int", -1, 200),
            "gguf_n_threads": ("int", 1, 128),
            "gguf_n_batch": ("int", 64, 4096),
        },
    }
    for raw_key in sorted(payload, key=lambda item: str(item)):
        if not isinstance(raw_key, str):
            issues.append(MigrationIssue("model_overrides", "contains a non-text model key"))
            continue
        match = _OVERRIDE_KEY_PATTERN.fullmatch(raw_key)
        if not match or len(match.group(2)) > 2048:
            issues.append(MigrationIssue("model_overrides", "contains an unsupported model key"))
            continue
        backend = match.group(1).lower()
        model_id = match.group(2).strip()
        if not model_id or any(ord(character) < 32 for character in model_id):
            issues.append(MigrationIssue("model_overrides", "contains an invalid model identifier"))
            continue
        record = payload[raw_key]
        if not isinstance(record, dict) or set(record) - {"enabled", "values"}:
            issues.append(MigrationIssue("model_overrides", "contains an invalid model record"))
            continue
        try:
            enabled = _strict_bool(record.get("enabled", False))
        except ValueError:
            issues.append(MigrationIssue("model_overrides", "contains an invalid enabled flag"))
            continue
        raw_values = record.get("values", {})
        if not isinstance(raw_values, dict):
            issues.append(MigrationIssue("model_overrides", "contains invalid override values"))
            continue
        unexpected = set(raw_values) - set(allowed_values[backend])
        if unexpected:
            issues.append(MigrationIssue("model_overrides", "contains unsupported override fields"))
            continue
        parsed_values: dict[str, Any] = {}
        valid = True
        for name, raw in raw_values.items():
            kind, minimum, maximum = allowed_values[backend][name]
            try:
                if kind == "int":
                    parsed_values[name] = _strict_int(
                        raw, minimum=int(minimum), maximum=int(maximum)
                    )
                else:
                    parsed_values[name] = _strict_float(
                        raw,
                        minimum=float(minimum),
                        maximum=float(maximum),
                        exclusive_minimum=kind == "float_exclusive",
                    )
            except ValueError:
                issues.append(MigrationIssue("model_overrides", f"has invalid {name}"))
                valid = False
                break
        if valid:
            canonical_key = f"{backend}::{model_id}"
            sanitized[canonical_key] = {
                "enabled": enabled,
                "values": dict(sorted(parsed_values.items())),
            }
    return sanitized


def _profile_overrides(
    backend: str,
    overrides: Mapping[str, Mapping[str, Any]],
) -> tuple[ModelGenerationOverride, ...]:
    result: list[ModelGenerationOverride] = []
    prefix = f"{backend.lower()}::"
    temp_key = "ollama_temp" if backend == "Ollama" else "gguf_temp"
    top_p_key = "ollama_top_p" if backend == "Ollama" else "gguf_top_p"
    for key, record in sorted(overrides.items()):
        if not key.startswith(prefix) or not bool(record.get("enabled")):
            continue
        values = record.get("values")
        if not isinstance(values, Mapping):
            continue
        temperature = values.get(temp_key)
        top_p = values.get(top_p_key)
        provider_values = (
            {
                "ollama_context_tokens": values.get("ollama_ctx"),
            }
            if backend == "Ollama"
            else {
                "gguf_n_ctx": values.get("gguf_n_ctx"),
                "gguf_n_gpu_layers": values.get("gguf_n_gpu_layers"),
                "gguf_n_threads": values.get("gguf_n_threads"),
                "gguf_n_batch": values.get("gguf_n_batch"),
            }
        )
        if temperature is None and top_p is None and all(
            value is None for value in provider_values.values()
        ):
            continue
        result.append(
            ModelGenerationOverride(
                model_id=key[len(prefix) :],
                temperature=float(temperature) if temperature is not None else None,
                top_p=float(top_p) if top_p is not None else None,
                **provider_values,
            )
        )
    return tuple(result)


def migrate_legacy_qsettings(
    reader: LegacySettingsReader,
    *,
    registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
    defaults: AppDefaults | None = None,
) -> LegacySettingsMigrationResult:
    """Migrate one legacy settings source without importing Qt or reading secrets."""

    if not isinstance(reader, LegacySettingsReader):
        raise TypeError("reader must implement contains() and value()")
    app_defaults = defaults or AppDefaults()
    raw, source_fingerprint = _read_allowlisted_source(reader)
    issues: list[MigrationIssue] = []
    top_level_legacy: dict[str, Any] = {}

    paths: dict[str, str] = {}
    for key in ("import_dir", "export_dir", "json_path"):
        if key not in raw:
            continue
        try:
            paths[key] = _public_string(raw[key], allow_empty=True)
        except ValueError as exc:
            issues.append(MigrationIssue(key, str(exc)))
            top_level_legacy[key] = _safe_legacy_value(raw[key])
    run_invocation_defaults = LegacyRunInvocationDefaults(
        import_dir=paths.get("import_dir", ""),
        export_dir=paths.get("export_dir", ""),
        json_path=paths.get("json_path", ""),
    )
    workspace_layout: dict[str, Any] = {}
    for key, name in (("geometry", "geometry_present"), ("windowState", "window_state_present")):
        if key in raw:
            # The old load path was disabled. Preserve presence without copying
            # opaque Qt byte arrays into the public store.
            workspace_layout[name] = True
    application_preferences = ApplicationPreferences(
        theme=app_defaults.theme,
        workspace_layout=workspace_layout,
    )

    project_values = {
        "source_language": app_defaults.source_language,
        "target_language": app_defaults.target_language,
    }
    for key, field_name, allowed in (
        ("source_lang", "source_language", {"Japanese"}),
        ("target_lang", "target_language", {"Simplified Chinese", "English"}),
    ):
        if key not in raw:
            continue
        try:
            value = _public_string(raw[key])
            if value not in allowed:
                raise ValueError("is not a supported language")
            project_values[field_name] = value
        except ValueError as exc:
            issues.append(MigrationIssue(key, str(exc)))
            top_level_legacy[key] = _safe_legacy_value(raw[key])
    project_config = ProjectConfig(
        source_language=project_values["source_language"],
        target_language=project_values["target_language"],
        output_suffix=app_defaults.output_suffix,
    )

    module_values: dict[str, dict[str, Any]] = {}
    module_legacy: dict[str, dict[str, Any]] = {}

    def add_module_value(
        key: str,
        qualified_id: str,
        parser,
    ) -> None:
        if key not in raw:
            return
        module_id, setting_id = qualified_id.split(".", 1)
        definition = registry.get_setting(qualified_id)
        try:
            value = parser(raw[key])
            definition.validate_value(value, allow_legacy=True)
        except (TypeError, ValueError) as exc:
            issues.append(MigrationIssue(key, str(exc)))
            module_legacy.setdefault(module_id, {})[
                f"{setting_id}__unsupported"
            ] = _safe_legacy_value(raw[key])
            return
        is_legacy_value = any(
            type(value) is type(candidate) and value == candidate
            for candidate in definition.legacy_values
        )
        if definition.lifecycle is not SettingLifecycle.SUPPORTED or is_legacy_value:
            module_legacy.setdefault(module_id, {})[setting_id] = value
        else:
            module_values.setdefault(module_id, {})[setting_id] = value

    add_module_value("detector_engine", "detection.engine", _public_string)
    add_module_value(
        "detector_input_size",
        "detection.input_size",
        lambda value: _strict_int(value, minimum=640, maximum=1280),
    )
    add_module_value("filter_strength", "detection.filter_strength", _public_string)
    add_module_value("ocr_engine", "ocr.engine", _normalized_ocr)
    add_module_value("inpaint_mode", "cleanup.inpaint_mode", _public_string)
    add_module_value("use_gpu", "runtime.use_gpu", _strict_bool)
    add_module_value("fast_mode", "runtime.fast_mode", _strict_bool)
    add_module_value("auto_glossary", "translation.auto_glossary", _strict_bool)
    add_module_value(
        "use_ollama_discovery",
        "translation.use_ollama_discovery",
        _strict_bool,
    )
    add_module_value("prescan_enabled", "translation.prescan_enabled", _strict_bool)
    add_module_value("font_name", "renderer.font_name", _public_string)
    add_module_value(
        "discovery_backend", "translation.discovery_backend", _public_string
    )

    module_configs: list[ModuleConfig] = []
    for module_id in sorted(set(module_values) | set(module_legacy)):
        schema = registry.get_module(module_id)
        config = ModuleConfig(
            module_id=module_id,
            module_schema_version=schema.schema_version,
            values=module_values.get(module_id, {}),
            legacy_values=module_legacy.get(module_id, {}),
        )
        registry.validate_config(config, allow_legacy=True)
        module_configs.append(config)

    parsed_overrides = _parse_model_overrides(raw.get("model_overrides"), issues)
    if parsed_overrides:
        # Retain the bounded source representation as migration provenance;
        # active generation and provider-runtime values are also represented
        # by typed exact-model profile overrides.
        top_level_legacy["model_overrides"] = parsed_overrides

    def number_or_default(
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
        exclusive_minimum: bool = False,
    ) -> float:
        if key not in raw:
            return default
        try:
            return _strict_float(
                raw[key],
                minimum=minimum,
                maximum=maximum,
                exclusive_minimum=exclusive_minimum,
            )
        except ValueError as exc:
            issues.append(MigrationIssue(key, str(exc)))
            top_level_legacy[key] = _safe_legacy_value(raw[key])
            return default

    def integer_or_default(
        key: str,
        default: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        if key not in raw:
            return default
        try:
            return _strict_int(raw[key], minimum=minimum, maximum=maximum)
        except ValueError as exc:
            issues.append(MigrationIssue(key, str(exc)))
            top_level_legacy[key] = _safe_legacy_value(raw[key])
            return default

    ollama_generation = GenerationSettings(
        temperature=number_or_default(
            "ollama_temp",
            app_defaults.ollama_temperature,
            minimum=0.0,
            maximum=2.0,
        ),
        top_p=number_or_default(
            "ollama_top_p",
            app_defaults.ollama_top_p,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        ),
    )
    gguf_generation = GenerationSettings(
        temperature=number_or_default(
            "gguf_temp",
            app_defaults.gguf_temperature,
            minimum=0.0,
            maximum=2.0,
        ),
        top_p=number_or_default(
            "gguf_top_p",
            app_defaults.gguf_top_p,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        ),
    )
    ollama_context = integer_or_default(
        "ollama_ctx", app_defaults.ollama_context, minimum=512, maximum=32768
    )
    gguf_gpu_layers = integer_or_default(
        "gguf_n_gpu_layers",
        app_defaults.gguf_n_gpu_layers,
        minimum=-1,
        maximum=200,
    )

    profiles: dict[str, ProviderProfile] = {}
    references: dict[str, str] = {}
    unresolved_references: dict[str, str] = {}

    translator_backend: str | None = None
    if "translator_backend" in raw:
        try:
            candidate = _public_string(raw["translator_backend"])
            if candidate not in _PROFILE_IDS:
                raise ValueError("is not a supported translation backend")
            translator_backend = candidate
        except ValueError as exc:
            issues.append(MigrationIssue("translator_backend", str(exc)))
            top_level_legacy["translator_backend"] = _safe_legacy_value(
                raw["translator_backend"]
            )

    gguf_path: str | None = None
    if "gguf_model_path" in raw:
        try:
            gguf_path = _public_string(raw["gguf_model_path"])
        except ValueError as exc:
            issues.append(MigrationIssue("gguf_model_path", str(exc)))
            top_level_legacy["gguf_model_path"] = _safe_legacy_value(
                raw["gguf_model_path"]
            )
    if translator_backend == "GGUF" or gguf_path:
        profile_id = _PROFILE_IDS["GGUF"]
        try:
            profiles[profile_id] = ProviderProfile(
                profile_id=profile_id,
                display_name="Migrated GGUF",
                kind=ProviderKind.GGUF,
                local_model_path=gguf_path,
                generation_defaults=gguf_generation,
                model_overrides=_profile_overrides("GGUF", parsed_overrides),
                gguf_options=GGUFProviderOptions(
                    prompt_style=app_defaults.gguf_prompt_style,
                    n_ctx=app_defaults.gguf_n_ctx,
                    n_gpu_layers=gguf_gpu_layers,
                    n_threads=app_defaults.gguf_n_threads,
                    n_batch=app_defaults.gguf_n_batch,
                ),
            )
            if translator_backend == "GGUF":
                references["translation"] = profile_id
        except ProviderProfileError as exc:
            if translator_backend == "GGUF":
                unresolved_references["translation"] = profile_id
                issues.append(MigrationIssue("gguf_model_path", str(exc)))

    if translator_backend == "Ollama" or "ollama_ctx" in raw:
        profile_id = _PROFILE_IDS["Ollama"]
        try:
            profiles[profile_id] = ProviderProfile(
                profile_id=profile_id,
                display_name="Migrated Ollama",
                kind=ProviderKind.OLLAMA,
                endpoint="http://localhost:11434",
                model_id="auto-detect",
                generation_defaults=ollama_generation,
                model_overrides=_profile_overrides("Ollama", parsed_overrides),
                ollama_options=OllamaProviderOptions(context_tokens=ollama_context),
            )
            if translator_backend == "Ollama":
                references["translation"] = profile_id
        except ProviderProfileError as exc:
            if translator_backend == "Ollama":
                unresolved_references["translation"] = profile_id
                issues.append(MigrationIssue("translator_backend", str(exc)))

    if translator_backend == "DeepSeek" or any(
        key in raw for key in ("deepseek_model", "deepseek_base_url")
    ):
        try:
            model_id = (
                _public_string(raw["deepseek_model"])
                if "deepseek_model" in raw
                else app_defaults.deepseek_model
            )
            endpoint = (
                _public_endpoint(raw["deepseek_base_url"])
                if "deepseek_base_url" in raw
                else app_defaults.deepseek_base_url
            )
            profile_id = _PROFILE_IDS["DeepSeek"]
            profiles[profile_id] = ProviderProfile(
                profile_id=profile_id,
                display_name="Migrated DeepSeek",
                kind=ProviderKind.DEEPSEEK,
                endpoint=endpoint,
                model_id=model_id,
                # The legacy public store never contained a secret.  Preserve
                # only the established environment-variable locator; the GUI
                # resolves it lazily at Start and never reads it here.
                credential_ref=environment_credential_reference(
                    "DEEPSEEK_API_KEY",
                    label="DeepSeek API key",
                ),
                generation_defaults=ollama_generation,
            )
            if translator_backend == "DeepSeek":
                references["translation"] = profile_id
        except (ValueError, ProviderProfileError) as exc:
            key = (
                "deepseek_base_url"
                if "endpoint" in str(exc).lower()
                else "deepseek_model"
            )
            issues.append(MigrationIssue(key, str(exc)))
            for legacy_key in ("deepseek_model", "deepseek_base_url"):
                if legacy_key in raw:
                    top_level_legacy[legacy_key] = _safe_legacy_value(raw[legacy_key])
            if translator_backend == "DeepSeek":
                unresolved_references["translation"] = _PROFILE_IDS["DeepSeek"]

    discovery_backend: str | None = None
    if "discovery_backend" in raw:
        try:
            discovery_backend = _public_string(raw["discovery_backend"])
            if discovery_backend not in {"Ollama", "GGUF"}:
                raise ValueError("is not a supported discovery backend")
        except ValueError as exc:
            issues.append(MigrationIssue("discovery_backend", str(exc)))
            top_level_legacy["discovery_backend"] = _safe_legacy_value(
                raw["discovery_backend"]
            )
            discovery_backend = None
    if discovery_backend == "Ollama":
        profile_id = "legacy-discovery-ollama"
        try:
            model_id = _public_string(
                raw.get("discovery_ollama_model", "auto-detect")
            )
            profiles[profile_id] = ProviderProfile(
                profile_id=profile_id,
                display_name="Migrated Discovery Ollama",
                kind=ProviderKind.OLLAMA,
                endpoint="http://localhost:11434",
                model_id=model_id,
                generation_defaults=ollama_generation,
                ollama_options=OllamaProviderOptions(context_tokens=ollama_context),
            )
            references["discovery"] = profile_id
        except (ValueError, ProviderProfileError) as exc:
            issues.append(MigrationIssue("discovery_ollama_model", str(exc)))
            unresolved_references["discovery"] = profile_id
    elif discovery_backend == "GGUF":
        profile_id = "legacy-discovery-gguf"
        try:
            path = _public_string(raw.get("discovery_gguf_path"))
            profiles[profile_id] = ProviderProfile(
                profile_id=profile_id,
                display_name="Migrated Discovery GGUF",
                kind=ProviderKind.GGUF,
                local_model_path=path,
                generation_defaults=gguf_generation,
                gguf_options=GGUFProviderOptions(
                    prompt_style=app_defaults.gguf_prompt_style,
                    n_ctx=app_defaults.gguf_n_ctx,
                    n_gpu_layers=gguf_gpu_layers,
                    n_threads=app_defaults.gguf_n_threads,
                    n_batch=app_defaults.gguf_n_batch,
                ),
            )
            references["discovery"] = profile_id
        except (ValueError, ProviderProfileError) as exc:
            issues.append(MigrationIssue("discovery_gguf_path", str(exc)))
            unresolved_references["discovery"] = profile_id

    if "gen_preset" in raw:
        try:
            top_level_legacy["gen_preset"] = _public_string(raw["gen_preset"])
        except ValueError as exc:
            issues.append(MigrationIssue("gen_preset", str(exc)))

    marker = (
        f"legacy-qsettings-v{LEGACY_SETTINGS_MIGRATION_VERSION}:{source_fingerprint}"
    )
    project_config = replace(
        project_config,
        provider_profile_references=references,
    )
    return LegacySettingsMigrationResult(
        application_preferences=application_preferences,
        project_config=project_config,
        module_configs=tuple(module_configs),
        provider_profiles=tuple(profiles[key] for key in sorted(profiles)),
        run_invocation_defaults=run_invocation_defaults,
        unresolved_provider_profile_references=unresolved_references,
        legacy_values=top_level_legacy,
        issues=tuple(issues),
        migrated_keys=tuple(raw),
        source_fingerprint=source_fingerprint,
        migration_marker=marker,
    )


def migrate_legacy_qsettings_once(
    reader: LegacySettingsReader,
    migration_markers: tuple[str, ...],
    *,
    registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
    defaults: AppDefaults | None = None,
) -> LegacySettingsMigrationResult | None:
    """Read the legacy workflow store only before the v1 marker exists."""

    markers = tuple(migration_markers)
    if any(
        isinstance(marker, str)
        and marker.startswith(
            f"legacy-qsettings-v{LEGACY_SETTINGS_MIGRATION_VERSION}:"
        )
        for marker in markers
    ):
        return None
    return migrate_legacy_qsettings(
        reader,
        registry=registry,
        defaults=defaults,
    )


def legacy_project_settings_seed_required(project: Mapping[str, Any]) -> bool:
    """Return whether a stored project has no typed settings authority yet.

    Schema-1 projects necessarily predate the typed GUI settings section.  A
    schema-2 project is considered empty only when both its typed project
    container and duplicated provider-reference index are empty.  Explicit
    schema-2 settings, even when they happen to equal defaults, remain the
    user's durable authority and are never overwritten by QSettings migration.
    """

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    schema_version = str(project.get("schema_version") or "1.0")
    if schema_version == "1.0":
        return True
    if schema_version != "2.0":
        raise ValueError(f"unsupported project schema: {schema_version}")
    settings = project.get("settings")
    if not isinstance(settings, Mapping):
        return True
    project_container = settings.get("project_config") or {}
    provider_references = settings.get("provider_profile_refs") or {}
    if not isinstance(project_container, Mapping):
        raise ValueError("project settings project_config must be a mapping")
    if not isinstance(provider_references, Mapping):
        raise ValueError("provider_profile_refs must be a mapping")
    return not project_container and not provider_references


def publish_legacy_migration_marker_last(
    *,
    publish_project: Callable[[], bool | None] | None,
    publish_provider_profiles: Callable[[], None],
    publish_application_marker: Callable[[], None],
) -> bool:
    """Publish migration state in the only safe retryable order.

    The application document owns the durable one-time marker, so it is
    always written last.  A project or provider-store failure therefore leaves
    migration unmarked and retryable on the next startup.  A project callback
    may return ``False`` when checkpoint ownership requires deferral; provider
    and marker publication are then skipped and the coordinator returns
    ``False``.
    """

    for callback_name, callback in (
        ("publish_provider_profiles", publish_provider_profiles),
        ("publish_application_marker", publish_application_marker),
    ):
        if not callable(callback):
            raise TypeError(f"{callback_name} must be callable")
    if publish_project is not None and not callable(publish_project):
        raise TypeError("publish_project must be callable or None")
    if publish_project is not None:
        project_published = publish_project()
        if project_published is False:
            return False
    publish_provider_profiles()
    publish_application_marker()
    return True
