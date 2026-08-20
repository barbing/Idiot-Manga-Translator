# -*- coding: utf-8 -*-
"""Declarative module-setting schemas for the GUI-2 settings foundation.

The registry describes configuration that current production owners can
actually consume.  It does not execute modules, probe assets, or silently
translate legacy choices into different behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from app.config.defaults import AppDefaults
from app.config.settings_contracts import (
    ModuleConfig,
    SettingsScope,
    canonical_fingerprint,
    freeze_json,
    thaw_json,
)


MODULE_REGISTRY_SCHEMA_VERSION = "module_registry_v1"
MODULE_SCHEMA_VERSION = "1.0"


class SettingValueType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ENUM = "enum"


class SettingVisibility(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    DEVELOPER = "developer"


class SettingLifecycle(str, Enum):
    SUPPORTED = "supported"
    LEGACY = "legacy"
    DEPRECATED = "deprecated"


class InvalidationImpact(str, Enum):
    NONE = "none"
    FUTURE_RUN_ONLY = "future_run_only"


class SettingValidationError(ValueError):
    """Base error for schema-owned setting validation."""


class UnsupportedSettingError(SettingValidationError):
    pass


class LegacySettingError(SettingValidationError):
    pass


class CapabilityUnavailableError(SettingValidationError):
    pass


_KNOWN_STAGES = frozenset(
    {
        "detection",
        "ocr",
        "hierarchy",
        "translation",
        "cleanup",
        "style_observation",
        "style_arbitration",
        "rendering",
        "persistence",
    }
)


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    return result


def _require_unique_strings(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{field_name} must be a sequence")
    result = tuple(_require_non_empty(item, f"{field_name}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _scalar_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _values_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


@dataclass(frozen=True, slots=True)
class SettingDefinition:
    module_id: str
    setting_id: str
    value_type: SettingValueType
    default: Any
    scope: SettingsScope
    visibility: SettingVisibility = SettingVisibility.BASIC
    lifecycle: SettingLifecycle = SettingLifecycle.SUPPORTED
    allowed_values: tuple[Any, ...] = ()
    legacy_values: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    units: str | None = None
    affected_stages: tuple[str, ...] = ()
    invalidation_impact: InvalidationImpact = InvalidationImpact.FUTURE_RUN_ONLY
    required_capabilities: tuple[str, ...] = ()
    value_capabilities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    editable: bool = True
    replacement_setting_id: str | None = None
    description: str = ""
    schema_version: str = MODULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _require_non_empty(self.module_id, "module_id"))
        object.__setattr__(self, "setting_id", _require_non_empty(self.setting_id, "setting_id"))
        object.__setattr__(self, "value_type", SettingValueType(self.value_type))
        object.__setattr__(self, "scope", SettingsScope(self.scope))
        if self.scope in {
            SettingsScope.CREDENTIAL,
            SettingsScope.EDITOR,
            SettingsScope.RUNTIME,
        }:
            raise ValueError(
                f"module setting {self.qualified_id} cannot use {self.scope.value} scope"
            )
        object.__setattr__(self, "visibility", SettingVisibility(self.visibility))
        object.__setattr__(self, "lifecycle", SettingLifecycle(self.lifecycle))
        object.__setattr__(
            self, "invalidation_impact", InvalidationImpact(self.invalidation_impact)
        )
        if self.schema_version != MODULE_SCHEMA_VERSION:
            raise ValueError(f"unsupported module schema: {self.schema_version}")
        if not isinstance(self.editable, bool):
            raise TypeError("editable must be a boolean")

        frozen_default = freeze_json(self.default, field_name=f"{self.qualified_id}.default")
        frozen_allowed = tuple(
            freeze_json(value, field_name=f"{self.qualified_id}.allowed_values[]")
            for value in self.allowed_values
        )
        frozen_legacy = tuple(
            freeze_json(value, field_name=f"{self.qualified_id}.legacy_values[]")
            for value in self.legacy_values
        )
        for values, field_name in (
            (frozen_allowed, "allowed_values"),
            (frozen_legacy, "legacy_values"),
        ):
            keys = [_scalar_key(thaw_json(value)) for value in values]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{self.qualified_id}.{field_name} contains duplicates")
        if frozen_legacy and not frozen_allowed:
            raise ValueError("legacy_values require an explicit allowed_values vocabulary")
        if any(
            not any(_values_equal(legacy, allowed) for allowed in frozen_allowed)
            for legacy in frozen_legacy
        ):
            raise ValueError("legacy_values must be a subset of allowed_values")
        object.__setattr__(self, "default", frozen_default)
        object.__setattr__(self, "allowed_values", frozen_allowed)
        object.__setattr__(self, "legacy_values", frozen_legacy)

        if self.minimum is not None:
            if isinstance(self.minimum, bool) or not isinstance(self.minimum, (int, float)):
                raise TypeError("minimum must be numeric")
            if not math.isfinite(float(self.minimum)):
                raise ValueError("minimum must be finite")
        if self.maximum is not None:
            if isinstance(self.maximum, bool) or not isinstance(self.maximum, (int, float)):
                raise TypeError("maximum must be numeric")
            if not math.isfinite(float(self.maximum)):
                raise ValueError("maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and float(self.minimum) > float(self.maximum)
        ):
            raise ValueError("minimum cannot exceed maximum")
        if self.units is not None:
            object.__setattr__(self, "units", _require_non_empty(self.units, "units"))

        stages = _require_unique_strings(self.affected_stages, "affected_stages")
        unknown_stages = frozenset(stages) - _KNOWN_STAGES
        if unknown_stages:
            raise ValueError(f"unknown affected stages: {sorted(unknown_stages)}")
        object.__setattr__(self, "affected_stages", stages)
        object.__setattr__(
            self,
            "required_capabilities",
            _require_unique_strings(self.required_capabilities, "required_capabilities"),
        )

        if not isinstance(self.value_capabilities, Mapping):
            raise TypeError("value_capabilities must be a mapping")
        normalized_value_capabilities: dict[str, tuple[str, ...]] = {}
        allowed_keys = {_scalar_key(thaw_json(value)) for value in frozen_allowed}
        for value_key, capabilities in self.value_capabilities.items():
            if not isinstance(value_key, str):
                raise TypeError("value_capabilities keys must be canonical strings")
            if allowed_keys and value_key not in allowed_keys:
                raise ValueError(
                    f"value capability key {value_key!r} is not an allowed value"
                )
            normalized_value_capabilities[value_key] = _require_unique_strings(
                capabilities,
                f"value_capabilities[{value_key}]",
            )
        object.__setattr__(
            self,
            "value_capabilities",
            MappingProxyType(dict(sorted(normalized_value_capabilities.items()))),
        )

        if self.replacement_setting_id is not None:
            object.__setattr__(
                self,
                "replacement_setting_id",
                _require_non_empty(
                    self.replacement_setting_id, "replacement_setting_id"
                ),
            )
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        # Schema definitions must validate their own defaults.  Legacy defaults
        # are allowed only as explicit migration evidence.
        self.validate_value(frozen_default, allow_legacy=True)

    @property
    def qualified_id(self) -> str:
        return f"{self.module_id}.{self.setting_id}"

    def _validate_type(self, value: Any) -> None:
        if self.value_type is SettingValueType.BOOLEAN:
            if not isinstance(value, bool):
                raise TypeError(f"{self.qualified_id} must be a boolean")
        elif self.value_type is SettingValueType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{self.qualified_id} must be an integer")
        elif self.value_type is SettingValueType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{self.qualified_id} must be numeric")
            if not math.isfinite(float(value)):
                raise SettingValidationError(f"{self.qualified_id} must be finite")
        elif self.value_type is SettingValueType.STRING:
            if not isinstance(value, str):
                raise TypeError(f"{self.qualified_id} must be a string")
            if not value.strip():
                raise SettingValidationError(f"{self.qualified_id} cannot be empty")
        elif self.value_type is SettingValueType.ENUM:
            if not isinstance(value, str):
                raise TypeError(f"{self.qualified_id} must be an enum string")

    def validate_value(
        self,
        value: Any,
        *,
        available_capabilities: Iterable[str] | None = None,
        allow_legacy: bool = False,
    ) -> Any:
        frozen_value = freeze_json(value, field_name=self.qualified_id)
        self._validate_type(frozen_value)
        if self.allowed_values and not any(
            _values_equal(frozen_value, allowed) for allowed in self.allowed_values
        ):
            raise SettingValidationError(
                f"{self.qualified_id} has unsupported value {value!r}"
            )
        is_legacy_value = any(
            _values_equal(frozen_value, legacy) for legacy in self.legacy_values
        )
        if (self.lifecycle is not SettingLifecycle.SUPPORTED or is_legacy_value) and not allow_legacy:
            raise LegacySettingError(
                f"{self.qualified_id} value {value!r} is legacy-only"
            )
        if self.value_type in {SettingValueType.INTEGER, SettingValueType.NUMBER}:
            numeric = float(frozen_value)
            if self.minimum is not None and numeric < float(self.minimum):
                raise SettingValidationError(
                    f"{self.qualified_id} must be at least {self.minimum}"
                )
            if self.maximum is not None and numeric > float(self.maximum):
                raise SettingValidationError(
                    f"{self.qualified_id} must be at most {self.maximum}"
                )
        if available_capabilities is not None:
            available = frozenset(str(item) for item in available_capabilities)
            required = set(self.required_capabilities)
            required.update(
                self.value_capabilities.get(_scalar_key(thaw_json(frozen_value)), ())
            )
            missing = required - available
            if missing:
                raise CapabilityUnavailableError(
                    f"{self.qualified_id} requires capabilities {sorted(missing)}"
                )
        return frozen_value

    def is_legacy_value(self, value: Any) -> bool:
        """Return whether ``value`` is retained only as inactive evidence.

        The method validates against the complete historical vocabulary first,
        so callers cannot use it to classify malformed or unknown values as
        legacy.  A setting whose entire lifecycle is legacy treats every valid
        value as legacy evidence.
        """

        frozen_value = self.validate_value(value, allow_legacy=True)
        return self.lifecycle is not SettingLifecycle.SUPPORTED or any(
            _values_equal(frozen_value, legacy) for legacy in self.legacy_values
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "setting_id": self.setting_id,
            "qualified_id": self.qualified_id,
            "value_type": self.value_type.value,
            "default": thaw_json(self.default),
            "scope": self.scope.value,
            "visibility": self.visibility.value,
            "lifecycle": self.lifecycle.value,
            "allowed_values": [thaw_json(value) for value in self.allowed_values],
            "legacy_values": [thaw_json(value) for value in self.legacy_values],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "units": self.units,
            "affected_stages": list(self.affected_stages),
            "invalidation_impact": self.invalidation_impact.value,
            "required_capabilities": list(self.required_capabilities),
            "value_capabilities": {
                key: list(value) for key, value in self.value_capabilities.items()
            },
            "editable": self.editable,
            "replacement_setting_id": self.replacement_setting_id,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ModuleSchema:
    module_id: str
    display_name: str
    settings: tuple[SettingDefinition, ...]
    description: str = ""
    schema_version: str = MODULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _require_non_empty(self.module_id, "module_id"))
        object.__setattr__(
            self, "display_name", _require_non_empty(self.display_name, "display_name")
        )
        if self.schema_version != MODULE_SCHEMA_VERSION:
            raise ValueError(f"unsupported module schema: {self.schema_version}")
        settings = tuple(self.settings)
        if not settings:
            raise ValueError(f"module {self.module_id} must publish settings")
        ids: set[str] = set()
        for definition in settings:
            if not isinstance(definition, SettingDefinition):
                raise TypeError("settings must contain SettingDefinition values")
            if definition.module_id != self.module_id:
                raise ValueError(
                    f"{definition.qualified_id} belongs to another module"
                )
            if definition.setting_id in ids:
                raise ValueError(
                    f"duplicate setting {self.module_id}.{definition.setting_id}"
                )
            ids.add(definition.setting_id)
        object.__setattr__(self, "settings", settings)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

    @property
    def definitions(self) -> Mapping[str, SettingDefinition]:
        return MappingProxyType(
            {definition.setting_id: definition for definition in self.settings}
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "display_name": self.display_name,
            "description": self.description,
            "settings": [definition.to_dict() for definition in self.settings],
        }


@dataclass(frozen=True, slots=True)
class ModuleSchemaRegistry:
    modules: tuple[ModuleSchema, ...]
    schema_version: str = MODULE_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODULE_REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema: {self.schema_version}")
        modules = tuple(self.modules)
        module_ids = [module.module_id for module in modules]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("module registry contains duplicate module IDs")
        object.__setattr__(self, "modules", modules)

    @property
    def module_map(self) -> Mapping[str, ModuleSchema]:
        return MappingProxyType({module.module_id: module for module in self.modules})

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "modules": [module.to_dict() for module in self.modules],
        }

    def get_module(self, module_id: str) -> ModuleSchema:
        normalized = _require_non_empty(module_id, "module_id")
        try:
            return self.module_map[normalized]
        except KeyError as exc:
            raise UnsupportedSettingError(f"unknown module: {normalized}") from exc

    def get_setting(self, qualified_id: str) -> SettingDefinition:
        normalized = _require_non_empty(qualified_id, "qualified_id")
        module_id, separator, setting_id = normalized.partition(".")
        if not separator or not module_id or not setting_id:
            raise UnsupportedSettingError(
                "qualified setting IDs must use module.setting"
            )
        module = self.get_module(module_id)
        try:
            return module.definitions[setting_id]
        except KeyError as exc:
            raise UnsupportedSettingError(f"unknown setting: {normalized}") from exc

    def validate_config(
        self,
        config: ModuleConfig,
        *,
        available_capabilities: Iterable[str] | None = None,
        allow_legacy: bool = False,
    ) -> ModuleConfig:
        if not isinstance(config, ModuleConfig):
            raise TypeError("config must be a ModuleConfig")
        module = self.get_module(config.module_id)
        if config.module_schema_version != module.schema_version:
            raise SettingValidationError(
                f"{config.module_id} schema {config.module_schema_version!r} "
                f"does not match {module.schema_version!r}"
            )
        unknown = frozenset(config.values) - frozenset(module.definitions)
        if unknown:
            raise UnsupportedSettingError(
                f"{config.module_id} has unsupported settings: {sorted(unknown)}"
            )
        for setting_id, value in config.values.items():
            module.definitions[setting_id].validate_value(
                value,
                available_capabilities=available_capabilities,
                allow_legacy=allow_legacy,
            )
        # Legacy storage is evidence, never an active compiled value.  Known
        # entries must genuinely be legacy; unknown entries remain explicit
        # migration evidence rather than being guessed or dropped.
        for setting_id, value in config.legacy_values.items():
            definition = module.definitions.get(setting_id)
            if definition is None:
                continue
            definition.validate_value(value, allow_legacy=True)
            is_legacy_value = any(
                _values_equal(value, legacy) for legacy in definition.legacy_values
            )
            if (
                definition.lifecycle is SettingLifecycle.SUPPORTED
                and not is_legacy_value
            ):
                raise SettingValidationError(
                    f"supported {definition.qualified_id} cannot be stored as legacy"
                )
        return config

    def resolve_values(
        self,
        config: ModuleConfig,
        *,
        available_capabilities: Iterable[str] | None = None,
    ) -> Mapping[str, Any]:
        self.validate_config(
            config,
            available_capabilities=available_capabilities,
            allow_legacy=False,
        )
        module = self.get_module(config.module_id)
        resolved: dict[str, Any] = {}
        for definition in module.settings:
            if definition.lifecycle is not SettingLifecycle.SUPPORTED:
                continue
            value = config.values.get(definition.setting_id, definition.default)
            definition.validate_value(
                value,
                available_capabilities=available_capabilities,
                allow_legacy=False,
            )
            resolved[definition.setting_id] = thaw_json(value)
        return MappingProxyType(dict(sorted(resolved.items())))

    def visible_settings(
        self,
        *,
        advanced: bool = False,
        developer: bool = False,
    ) -> tuple[SettingDefinition, ...]:
        permitted = {SettingVisibility.BASIC}
        if advanced:
            permitted.add(SettingVisibility.ADVANCED)
        if developer:
            permitted.add(SettingVisibility.DEVELOPER)
        return tuple(
            definition
            for module in self.modules
            for definition in module.settings
            if definition.visibility in permitted
            and definition.lifecycle is SettingLifecycle.SUPPORTED
        )

    def supported_values(self, qualified_id: str) -> tuple[Any, ...]:
        """Return the current GUI vocabulary for one enumerated setting.

        Historical values remain in ``allowed_values`` so old projects can be
        decoded without data loss.  They must not leak back into editable GUI
        controls, which consume this filtered registry projection instead of
        maintaining their own list.
        """

        definition = self.get_setting(qualified_id)
        if (
            definition.lifecycle is not SettingLifecycle.SUPPORTED
            or not definition.allowed_values
        ):
            return ()
        return tuple(
            thaw_json(value)
            for value in definition.allowed_values
            if not definition.is_legacy_value(value)
        )


def _capabilities_for(**values: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    return {
        _scalar_key(key): tuple(capabilities)
        for key, capabilities in values.items()
    }


def build_default_module_registry() -> ModuleSchemaRegistry:
    defaults = AppDefaults()
    future = InvalidationImpact.FUTURE_RUN_ONLY
    project = SettingsScope.PROJECT
    application = SettingsScope.APPLICATION

    modules = (
        ModuleSchema(
            module_id="detection",
            display_name="Detection",
            description="Source text-area detection for future workflow runs.",
            settings=(
                SettingDefinition(
                    module_id="detection",
                    setting_id="engine",
                    value_type=SettingValueType.ENUM,
                    default=defaults.detector_engine,
                    allowed_values=("ComicTextDetector",),
                    scope=project,
                    affected_stages=("detection",),
                    invalidation_impact=future,
                    value_capabilities=_capabilities_for(
                        ComicTextDetector=("asset:comic_text_detector",)
                    ),
                    description="Current production text detector.",
                ),
                SettingDefinition(
                    module_id="detection",
                    setting_id="input_size",
                    value_type=SettingValueType.INTEGER,
                    # The production GUI explicitly initializes both legacy
                    # detector-input widgets to 640 before QSettings restore.
                    # Preserve that effective no-QSettings workflow default;
                    # AppDefaults.detector_input_size (1024) is not the live
                    # value at this seam.
                    default=640,
                    allowed_values=(640, 1024, 1280),
                    minimum=640,
                    maximum=1280,
                    units="px",
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("detection",),
                    invalidation_impact=future,
                ),
                SettingDefinition(
                    module_id="detection",
                    setting_id="filter_background",
                    value_type=SettingValueType.BOOLEAN,
                    default=True,
                    allowed_values=(False, True),
                    scope=project,
                    visibility=SettingVisibility.DEVELOPER,
                    lifecycle=SettingLifecycle.LEGACY,
                    affected_stages=("detection",),
                    invalidation_impact=future,
                    description="Legacy compatibility value; hidden from normal settings.",
                ),
                SettingDefinition(
                    module_id="detection",
                    setting_id="filter_strength",
                    value_type=SettingValueType.ENUM,
                    default=defaults.filter_strength,
                    allowed_values=("normal", "aggressive"),
                    scope=project,
                    visibility=SettingVisibility.DEVELOPER,
                    lifecycle=SettingLifecycle.LEGACY,
                    affected_stages=("detection",),
                    invalidation_impact=future,
                    description="Legacy compatibility value; hidden from normal settings.",
                ),
            ),
        ),
        ModuleSchema(
            module_id="ocr",
            display_name="OCR",
            settings=(
                SettingDefinition(
                    module_id="ocr",
                    setting_id="engine",
                    value_type=SettingValueType.ENUM,
                    default=defaults.ocr_engine,
                    allowed_values=("PaddleOCR-VL", "MangaOCR"),
                    scope=project,
                    affected_stages=("ocr",),
                    invalidation_impact=future,
                    value_capabilities=_capabilities_for(
                        **{
                            "PaddleOCR-VL": ("asset:paddle_ocr_vl",),
                            "MangaOCR": ("asset:manga_ocr",),
                        }
                    ),
                ),
            ),
        ),
        ModuleSchema(
            module_id="cleanup",
            display_name="Cleanup",
            settings=(
                SettingDefinition(
                    module_id="cleanup",
                    setting_id="inpaint_mode",
                    value_type=SettingValueType.ENUM,
                    default=defaults.inpaint_mode,
                    allowed_values=("ai", "fast", "off"),
                    legacy_values=("fast", "off"),
                    scope=project,
                    affected_stages=("cleanup",),
                    invalidation_impact=future,
                    value_capabilities=_capabilities_for(
                        ai=("asset:cleanup_inpaint",)
                    ),
                    description="AI is the supported policy; old fast/off values are explicit legacy evidence.",
                ),
                SettingDefinition(
                    module_id="cleanup",
                    setting_id="inpaint_model_id",
                    value_type=SettingValueType.ENUM,
                    default=defaults.inpaint_model,
                    allowed_values=(defaults.inpaint_model,),
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("cleanup",),
                    invalidation_impact=future,
                    required_capabilities=("asset:cleanup_inpaint",),
                    editable=False,
                    description="Fixed cleanup model owned by the cleanup module.",
                ),
            ),
        ),
        ModuleSchema(
            module_id="source_style",
            display_name="Source Style",
            settings=(
                SettingDefinition(
                    module_id="source_style",
                    setting_id="font_detection",
                    value_type=SettingValueType.ENUM,
                    default=defaults.font_detection,
                    allowed_values=("yuzumarker", "heuristic", "off"),
                    legacy_values=("heuristic", "off"),
                    scope=project,
                    affected_stages=("style_observation", "style_arbitration"),
                    invalidation_impact=future,
                    value_capabilities=_capabilities_for(
                        yuzumarker=("asset:yuzumarker",)
                    ),
                    description="YuzuMarker is the supported source-style observation path.",
                ),
            ),
        ),
        ModuleSchema(
            module_id="renderer",
            display_name="Output Defaults",
            settings=(
                SettingDefinition(
                    module_id="renderer",
                    setting_id="font_name",
                    value_type=SettingValueType.STRING,
                    default=defaults.font_name,
                    scope=project,
                    affected_stages=("rendering",),
                    invalidation_impact=future,
                    description=(
                        "Fallback output font for future pipeline results. "
                        "Per-parent detected styles and user overrides remain authoritative."
                    ),
                ),
            ),
        ),
        ModuleSchema(
            module_id="translation",
            display_name="Translation",
            settings=(
                SettingDefinition(
                    module_id="translation",
                    setting_id="auto_glossary",
                    value_type=SettingValueType.BOOLEAN,
                    default=defaults.auto_glossary,
                    scope=project,
                    affected_stages=("translation",),
                    invalidation_impact=future,
                ),
                SettingDefinition(
                    module_id="translation",
                    setting_id="prescan_enabled",
                    value_type=SettingValueType.BOOLEAN,
                    default=False,
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("translation",),
                    invalidation_impact=future,
                ),
                SettingDefinition(
                    module_id="translation",
                    setting_id="use_ollama_discovery",
                    value_type=SettingValueType.BOOLEAN,
                    default=False,
                    allowed_values=(False, True),
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("translation",),
                    invalidation_impact=future,
                    value_capabilities={
                        _scalar_key(True): ("provider:ollama",),
                    },
                ),
                SettingDefinition(
                    module_id="translation",
                    setting_id="discovery_backend",
                    value_type=SettingValueType.ENUM,
                    default="Ollama",
                    allowed_values=("Ollama", "GGUF"),
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("translation",),
                    invalidation_impact=future,
                    value_capabilities=_capabilities_for(
                        Ollama=("provider:ollama",),
                        GGUF=("provider:gguf",),
                    ),
                ),
                SettingDefinition(
                    module_id="translation",
                    setting_id="gguf_cross_page_context",
                    value_type=SettingValueType.BOOLEAN,
                    default=defaults.gguf_cross_page_context,
                    scope=project,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=("translation",),
                    invalidation_impact=future,
                ),
            ),
        ),
        ModuleSchema(
            module_id="runtime",
            display_name="Runtime",
            settings=(
                SettingDefinition(
                    module_id="runtime",
                    setting_id="use_gpu",
                    value_type=SettingValueType.BOOLEAN,
                    default=True,
                    scope=application,
                    visibility=SettingVisibility.ADVANCED,
                    affected_stages=(
                        "detection",
                        "ocr",
                        "cleanup",
                        "style_observation",
                    ),
                    invalidation_impact=future,
                    description=(
                        "Controls application-owned detection, OCR, cleanup, and "
                        "style runtimes. GGUF GPU layers and Ollama processor "
                        "placement remain provider-owned."
                    ),
                ),
                SettingDefinition(
                    module_id="runtime",
                    setting_id="fast_mode",
                    value_type=SettingValueType.BOOLEAN,
                    default=False,
                    allowed_values=(False, True),
                    legacy_values=(True,),
                    scope=application,
                    visibility=SettingVisibility.DEVELOPER,
                    affected_stages=(
                        "detection",
                        "cleanup",
                        "style_observation",
                    ),
                    invalidation_impact=future,
                    description="True preserves the legacy hidden-mutation mode only for migration.",
                ),
            ),
        ),
    )
    return ModuleSchemaRegistry(modules=modules)


DEFAULT_MODULE_REGISTRY = build_default_module_registry()
