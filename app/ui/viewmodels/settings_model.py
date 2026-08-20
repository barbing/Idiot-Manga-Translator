# -*- coding: utf-8 -*-
"""Single mutable draft authority for the Settings surface.

The view model is intentionally independent of PySide6.  Qt widgets project
this state and dispatch edits to it; they never become settings authorities.
Apply replaces the immutable baseline, while Cancel restores it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterable, Mapping

from app.config.module_registry import (
    DEFAULT_MODULE_REGISTRY,
    ModuleSchemaRegistry,
    SettingLifecycle,
)
from app.config.credential_store import environment_credential_reference
from app.config.provider_profiles import (
    GGUFProviderOptions,
    GenerationSettings,
    ModelGenerationOverride,
    OllamaProviderOptions,
    ProviderKind,
    ProviderProfile,
    ProviderTestStatus,
)
from app.config.run_settings_compiler import (
    CompilationIssue,
    CompilationResult,
    InternalRunOptions,
    RunInvocation,
    compile_run_settings,
)
from app.config.settings_contracts import (
    ApplicationPreferences,
    ModuleConfig,
    ProjectConfig,
    RunSettingsSnapshot,
    RuntimeStatus,
    canonical_fingerprint,
    thaw_json,
)


_PROVIDER_KIND_LABELS = MappingProxyType(
    {
        ProviderKind.GGUF: "GGUF local",
        ProviderKind.OLLAMA: "Ollama",
        ProviderKind.DEEPSEEK: "DeepSeek API",
        ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible",
    }
)
_PROVIDER_TEST_STATUS_LABELS = MappingProxyType(
    {
        ProviderTestStatus.NOT_TESTED: "Not tested",
        ProviderTestStatus.READY: "Ready",
        ProviderTestStatus.UNAVAILABLE: "Unavailable",
        ProviderTestStatus.ERROR: "Error",
    }
)


def provider_kind_label(kind: ProviderKind) -> str:
    return _PROVIDER_KIND_LABELS[ProviderKind(kind)]


def provider_test_status_label(status: ProviderTestStatus) -> str:
    return _PROVIDER_TEST_STATUS_LABELS[ProviderTestStatus(status)]


@dataclass(frozen=True, slots=True)
class SettingsDraft:
    application: ApplicationPreferences
    project: ProjectConfig
    module_configs: tuple[ModuleConfig, ...]
    provider_profiles: tuple[ProviderProfile, ...]

    def __post_init__(self) -> None:
        modules = tuple(self.module_configs)
        module_ids = [config.module_id for config in modules]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("settings draft has duplicate module configurations")
        profiles = tuple(self.provider_profiles)
        profile_ids = [profile.profile_id for profile in profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("settings draft has duplicate provider profiles")
        object.__setattr__(self, "module_configs", modules)
        object.__setattr__(self, "provider_profiles", profiles)

    @property
    def translation_profile_id(self) -> str | None:
        return self.project.provider_profile_references.get("translation")

    @property
    def discovery_profile_id(self) -> str | None:
        return self.project.provider_profile_references.get("discovery")

    def profile_for_role(self, role: str) -> ProviderProfile | None:
        if role not in {"translation", "discovery"}:
            raise ValueError(f"unsupported provider profile role: {role!r}")
        profile_id = self.project.provider_profile_references.get(role)
        if profile_id is None:
            return None
        for profile in self.provider_profiles:
            if profile.profile_id == profile_id:
                return profile
        return None

    @property
    def translation_profile(self) -> ProviderProfile | None:
        return self.profile_for_role("translation")

    @property
    def discovery_profile(self) -> ProviderProfile | None:
        return self.profile_for_role("discovery")

    @property
    def unresolved_provider_profile_references(self) -> Mapping[str, str]:
        profile_ids = frozenset(profile.profile_id for profile in self.provider_profiles)
        unresolved = {
            role: profile_id
            for role, profile_id in self.project.provider_profile_references.items()
            if profile_id not in profile_ids
        }
        return MappingProxyType(dict(sorted(unresolved.items())))

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "application": self.application.to_dict(),
                "project": self.project.to_dict(),
                "module_configs": [
                    config.to_dict()
                    for config in sorted(
                        self.module_configs, key=lambda item: item.module_id
                    )
                ],
                "provider_profiles": [
                    {
                        "public_profile": profile.to_public_export_dict(),
                        "credential_locator_fingerprint": (
                            profile.credential_ref.fingerprint
                            if profile.credential_ref is not None
                            else None
                        ),
                    }
                    for profile in sorted(
                        self.provider_profiles, key=lambda item: item.profile_id
                    )
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class LegacyShellSettingsProjection:
    """Typed projection of the settings controls in the legacy Qt shell.

    This is a temporary GUI-2 seam.  It lets the still-visible widgets submit
    one complete value to :class:`SettingsViewModel`; widgets are not read by
    the run compiler and no value is written back to workflow ``QSettings``.
    """

    source_language: str
    target_language: str
    output_suffix: str
    glossary_reference: str | None
    detector_engine: str
    detector_input_size: int
    ocr_engine: str
    inpaint_mode: str
    inpaint_model_id: str
    font_detection: str
    font_name: str
    use_gpu: bool
    auto_glossary: bool
    prescan_enabled: bool
    use_ollama_discovery: bool
    discovery_backend: str
    translation_backend: str
    ollama_model: str
    ollama_endpoint: str
    ollama_generation: GenerationSettings
    ollama_options: OllamaProviderOptions
    gguf_model_path: str
    gguf_generation: GenerationSettings
    gguf_options: GGUFProviderOptions
    deepseek_model: str
    deepseek_endpoint: str
    translation_overrides: tuple[ModelGenerationOverride, ...] = ()
    discovery_ollama_model: str = "auto-detect"
    discovery_gguf_model_path: str = ""

    def __post_init__(self) -> None:
        if self.translation_backend not in {"Ollama", "GGUF", "DeepSeek"}:
            raise ValueError("unsupported translation backend")
        if self.discovery_backend not in {"Ollama", "GGUF"}:
            raise ValueError("unsupported discovery backend")
        for field_name in (
            "use_gpu",
            "auto_glossary",
            "prescan_enabled",
            "use_ollama_discovery",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        overrides = tuple(self.translation_overrides)
        if any(not isinstance(item, ModelGenerationOverride) for item in overrides):
            raise TypeError(
                "translation_overrides must contain ModelGenerationOverride values"
            )
        object.__setattr__(self, "translation_overrides", overrides)


@dataclass(frozen=True, slots=True)
class EffectiveRunSummary:
    """Public, read-only preview of the configuration Start will consume."""

    ready: bool
    pending_changes: bool
    language_pair: str
    provider: str
    model: str
    detection_and_ocr: str
    cleanup_and_style: str
    runtime: str
    snapshot_id: str
    issues: tuple[CompilationIssue, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "language_pair",
            "provider",
            "model",
            "detection_and_ocr",
            "cleanup_and_style",
            "runtime",
            "snapshot_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        issues = tuple(self.issues)
        if any(not isinstance(issue, CompilationIssue) for issue in issues):
            raise TypeError("issues must contain CompilationIssue values")
        object.__setattr__(self, "issues", issues)


def _demote_active_legacy_values(
    registry: ModuleSchemaRegistry,
    config: ModuleConfig,
) -> ModuleConfig:
    """Normalize one imported config into current values plus inactive evidence."""

    schema = registry.get_module(config.module_id)
    active_values = dict(config.values)
    legacy_values = dict(config.legacy_values)
    changed = False
    for setting_id, value in tuple(active_values.items()):
        definition = schema.definitions[setting_id]
        if not definition.is_legacy_value(value):
            continue
        del active_values[setting_id]
        if definition.lifecycle is SettingLifecycle.SUPPORTED:
            legacy_values[f"{setting_id}__previous_legacy"] = value
            active_values[setting_id] = thaw_json(definition.default)
        else:
            legacy_values[setting_id] = value
        changed = True
    if not changed:
        return config
    normalized = ModuleConfig(
        module_id=config.module_id,
        module_schema_version=schema.schema_version,
        values=active_values,
        legacy_values=legacy_values,
    )
    registry.validate_config(normalized, allow_legacy=False)
    return normalized


def _merged_module_config(
    registry: ModuleSchemaRegistry,
    existing: Mapping[str, ModuleConfig],
    module_id: str,
    values: Mapping[str, object],
) -> ModuleConfig:
    schema = registry.get_module(module_id)
    previous = existing.get(module_id)
    if previous is not None:
        previous = _demote_active_legacy_values(registry, previous)
    merged_values = dict(previous.values) if previous is not None else {}
    legacy_values = dict(previous.legacy_values) if previous is not None else {}
    incoming_setting_ids = frozenset(values)

    # ModuleConfig forbids one setting ID from being both active and legacy.
    # When the current GUI supplies a supported replacement, retain the former
    # value under an explicit evidence-only key rather than weakening that
    # invariant or dropping the user's imported choice.
    for setting_id in incoming_setting_ids & frozenset(legacy_values):
        legacy_values[f"{setting_id}__previous_legacy"] = legacy_values.pop(
            setting_id
        )

    merged_values.update(values)
    config = ModuleConfig(
        module_id=module_id,
        module_schema_version=schema.schema_version,
        values=merged_values,
        legacy_values=legacy_values,
    )
    registry.validate_config(config, allow_legacy=False)
    return config


def _profile_for_role_and_kind(
    draft: SettingsDraft,
    *,
    role: str,
    kind: ProviderKind,
) -> ProviderProfile | None:
    selected = draft.profile_for_role(role)
    if selected is not None and selected.kind is kind:
        return selected
    stable_id = f"gui-{role}-{kind.value}"
    for profile in draft.provider_profiles:
        if profile.profile_id == stable_id and profile.kind is kind:
            return profile
    return None


def _replace_profile(
    profiles: tuple[ProviderProfile, ...],
    profile: ProviderProfile,
) -> tuple[ProviderProfile, ...]:
    by_id = {item.profile_id: item for item in profiles}
    by_id[profile.profile_id] = profile
    return tuple(by_id[key] for key in sorted(by_id))


def _merge_generation(
    existing: ProviderProfile | None,
    visible: GenerationSettings,
) -> GenerationSettings:
    if existing is None:
        return visible
    return replace(
        existing.generation_defaults,
        temperature=visible.temperature,
        top_p=visible.top_p,
    )


def _merge_model_overrides(
    existing: ProviderProfile | None,
    visible: tuple[ModelGenerationOverride, ...],
) -> tuple[ModelGenerationOverride, ...]:
    if existing is None:
        return visible
    previous_by_id = {item.model_id: item for item in existing.model_overrides}
    merged: dict[str, ModelGenerationOverride] = {}
    for item in visible:
        previous = previous_by_id.get(item.model_id)
        if previous is None:
            merged[item.model_id] = item
            continue
        replacements = {
            field_name: value
            for field_name, value in (
                ("temperature", item.temperature),
                ("top_p", item.top_p),
                ("ollama_context_tokens", item.ollama_context_tokens),
                ("gguf_n_ctx", item.gguf_n_ctx),
                ("gguf_n_gpu_layers", item.gguf_n_gpu_layers),
                ("gguf_n_threads", item.gguf_n_threads),
                ("gguf_n_batch", item.gguf_n_batch),
            )
            if value is not None
        }
        merged[item.model_id] = replace(previous, **replacements)
    return tuple(merged[key] for key in sorted(merged))


def rebind_run_snapshot_project(
    snapshot: RunSettingsSnapshot,
    project_id: str,
) -> RunSettingsSnapshot:
    """Bind a public pre-run snapshot to the project created by that run.

    The compiled values and timestamp remain unchanged.  The contract derives
    a fresh snapshot ID because project identity is part of semantic identity.
    """

    if not isinstance(snapshot, RunSettingsSnapshot):
        raise TypeError("snapshot must be a RunSettingsSnapshot")
    normalized = str(project_id).strip()
    if not normalized:
        raise ValueError("project_id must not be empty")
    if snapshot.project_id == normalized:
        return snapshot
    return replace(
        snapshot,
        project_id=normalized,
        snapshot_id="",
    )


class SettingsViewModel:
    """Own one immutable baseline and one immutable editable draft."""

    def __init__(
        self,
        initial: SettingsDraft,
        *,
        runtime_status: RuntimeStatus | None = None,
        registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
    ) -> None:
        if not isinstance(initial, SettingsDraft):
            raise TypeError("initial must be SettingsDraft")
        self._registry = registry
        normalized = self._normalize(initial)
        self._validate(normalized)
        self._baseline = normalized
        self._draft = normalized
        self._runtime_status = runtime_status

    @property
    def baseline(self) -> SettingsDraft:
        return self._baseline

    @property
    def draft(self) -> SettingsDraft:
        return self._draft

    @property
    def dirty(self) -> bool:
        return self._draft.fingerprint != self._baseline.fingerprint

    @property
    def runtime_status(self) -> RuntimeStatus | None:
        return self._runtime_status

    def set_runtime_status(self, status: RuntimeStatus | None) -> None:
        if status is not None and not isinstance(status, RuntimeStatus):
            raise TypeError("status must be RuntimeStatus or None")
        self._runtime_status = status

    def replace_application(self, value: ApplicationPreferences) -> SettingsDraft:
        return self._set_draft(replace(self._draft, application=value))

    def replace_draft(self, value: SettingsDraft) -> SettingsDraft:
        """Atomically replace the editable draft without applying it."""

        if not isinstance(value, SettingsDraft):
            raise TypeError("value must be SettingsDraft")
        return self._set_draft(value)

    def replace_project(self, value: ProjectConfig) -> SettingsDraft:
        return self._set_draft(replace(self._draft, project=value))

    def replace_module(self, value: ModuleConfig) -> SettingsDraft:
        configs = {
            config.module_id: config for config in self._draft.module_configs
        }
        configs[value.module_id] = value
        return self._set_draft(
            replace(
                self._draft,
                module_configs=tuple(
                    configs[key] for key in sorted(configs)
                ),
            )
        )

    def replace_modules(
        self,
        values: Iterable[ModuleConfig],
    ) -> SettingsDraft:
        configs = tuple(values)
        return self._set_draft(
            replace(
                self._draft,
                module_configs=configs,
            )
        )

    def replace_profiles(
        self,
        profiles: Iterable[ProviderProfile],
    ) -> SettingsDraft:
        stable = tuple(profiles)
        return self._set_draft(
            replace(
                self._draft,
                provider_profiles=stable,
            )
        )

    def select_provider(self, role: str, profile_id: str | None) -> SettingsDraft:
        if role not in {"translation", "discovery"}:
            raise ValueError(f"unsupported provider profile role: {role!r}")
        references = dict(self._draft.project.provider_profile_references)
        if profile_id is None:
            references.pop(role, None)
        else:
            normalized = str(profile_id).strip()
            if normalized not in {
                profile.profile_id for profile in self._draft.provider_profiles
            }:
                raise ValueError("selected provider profile does not exist")
            references[role] = normalized
        return self._set_draft(
            replace(
                self._draft,
                project=replace(
                    self._draft.project,
                    provider_profile_references=references,
                ),
            )
        )

    def select_translation_provider(self, profile_id: str | None) -> SettingsDraft:
        return self.select_provider("translation", profile_id)

    def select_discovery_provider(self, profile_id: str | None) -> SettingsDraft:
        return self.select_provider("discovery", profile_id)

    def replace_from_legacy_shell(
        self,
        projection: LegacyShellSettingsProjection,
    ) -> SettingsDraft:
        """Replace the draft from one complete legacy-shell projection.

        The conversion is atomic: all contracts are constructed and validated
        before the view model publishes the replacement draft.
        """

        if not isinstance(projection, LegacyShellSettingsProjection):
            raise TypeError("projection must be LegacyShellSettingsProjection")

        translation_kind = {
            "Ollama": ProviderKind.OLLAMA,
            "GGUF": ProviderKind.GGUF,
            "DeepSeek": ProviderKind.DEEPSEEK,
        }[projection.translation_backend]
        existing_translation = _profile_for_role_and_kind(
            self._draft,
            role="translation",
            kind=translation_kind,
        )
        translation_id = (
            existing_translation.profile_id
            if existing_translation is not None
            else f"gui-translation-{translation_kind.value}"
        )
        if translation_kind is ProviderKind.OLLAMA:
            profile_values = dict(
                endpoint=projection.ollama_endpoint,
                model_id=projection.ollama_model,
                generation_defaults=_merge_generation(
                    existing_translation, projection.ollama_generation
                ),
                model_overrides=_merge_model_overrides(
                    existing_translation, projection.translation_overrides
                ),
                ollama_options=projection.ollama_options,
            )
            translation_profile = (
                replace(existing_translation, **profile_values)
                if existing_translation is not None
                else ProviderProfile(
                    profile_id=translation_id,
                    display_name="Ollama",
                    kind=translation_kind,
                    **profile_values,
                )
            )
        elif translation_kind is ProviderKind.GGUF:
            profile_values = dict(
                local_model_path=projection.gguf_model_path or None,
                generation_defaults=_merge_generation(
                    existing_translation, projection.gguf_generation
                ),
                model_overrides=_merge_model_overrides(
                    existing_translation, projection.translation_overrides
                ),
                gguf_options=projection.gguf_options,
            )
            translation_profile = (
                replace(existing_translation, **profile_values)
                if existing_translation is not None
                else ProviderProfile(
                    profile_id=translation_id,
                    display_name="Local GGUF",
                    kind=translation_kind,
                    **profile_values,
                )
            )
        else:
            profile_values = dict(
                endpoint=projection.deepseek_endpoint,
                model_id=projection.deepseek_model,
                credential_ref=(
                    existing_translation.credential_ref
                    if existing_translation is not None
                    and existing_translation.credential_ref is not None
                    else environment_credential_reference(
                        "DEEPSEEK_API_KEY",
                        label="DeepSeek API key",
                    )
                ),
                generation_defaults=_merge_generation(
                    existing_translation, projection.ollama_generation
                ),
            )
            translation_profile = (
                replace(existing_translation, **profile_values)
                if existing_translation is not None
                else ProviderProfile(
                    profile_id=translation_id,
                    display_name="DeepSeek",
                    kind=translation_kind,
                    **profile_values,
                )
            )

        profiles = _replace_profile(
            self._draft.provider_profiles,
            translation_profile,
        )
        references = dict(self._draft.project.provider_profile_references)
        references["translation"] = translation_profile.profile_id

        if projection.use_ollama_discovery:
            discovery_kind = {
                "Ollama": ProviderKind.OLLAMA,
                "GGUF": ProviderKind.GGUF,
            }[projection.discovery_backend]
            existing_discovery = _profile_for_role_and_kind(
                replace(self._draft, provider_profiles=profiles),
                role="discovery",
                kind=discovery_kind,
            )
            discovery_id = (
                existing_discovery.profile_id
                if existing_discovery is not None
                else f"gui-discovery-{discovery_kind.value}"
            )
            if discovery_kind is ProviderKind.OLLAMA:
                profile_values = dict(
                    endpoint=projection.ollama_endpoint,
                    model_id=projection.discovery_ollama_model,
                    generation_defaults=_merge_generation(
                        existing_discovery, projection.ollama_generation
                    ),
                    ollama_options=projection.ollama_options,
                )
                discovery_profile = (
                    replace(existing_discovery, **profile_values)
                    if existing_discovery is not None
                    else ProviderProfile(
                        profile_id=discovery_id,
                        display_name="Discovery Ollama",
                        kind=discovery_kind,
                        **profile_values,
                    )
                )
            else:
                profile_values = dict(
                    local_model_path=projection.discovery_gguf_model_path or None,
                    generation_defaults=_merge_generation(
                        existing_discovery, projection.gguf_generation
                    ),
                    gguf_options=projection.gguf_options,
                )
                discovery_profile = (
                    replace(existing_discovery, **profile_values)
                    if existing_discovery is not None
                    else ProviderProfile(
                        profile_id=discovery_id,
                        display_name="Discovery GGUF",
                        kind=discovery_kind,
                        **profile_values,
                    )
                )
            profiles = _replace_profile(profiles, discovery_profile)
            references["discovery"] = discovery_profile.profile_id
        else:
            references.pop("discovery", None)

        project = ProjectConfig(
            source_language=projection.source_language,
            target_language=projection.target_language,
            output_suffix=projection.output_suffix,
            output_convention=self._draft.project.output_convention,
            completed_page_policy=self._draft.project.completed_page_policy,
            glossary_reference=projection.glossary_reference,
            selected_module_policies=self._draft.project.selected_module_policies,
            provider_profile_references=references,
        )
        existing_modules = {
            config.module_id: config for config in self._draft.module_configs
        }
        updated_modules = dict(existing_modules)
        for config in (
            _merged_module_config(
                self._registry,
                existing_modules,
                "cleanup",
                {
                    "inpaint_mode": projection.inpaint_mode,
                    "inpaint_model_id": projection.inpaint_model_id,
                },
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "detection",
                {
                    "engine": projection.detector_engine,
                    "input_size": projection.detector_input_size,
                },
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "ocr",
                {"engine": projection.ocr_engine},
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "renderer",
                {"font_name": projection.font_name},
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "runtime",
                {"use_gpu": projection.use_gpu},
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "source_style",
                {"font_detection": projection.font_detection},
            ),
            _merged_module_config(
                self._registry,
                existing_modules,
                "translation",
                {
                    "auto_glossary": projection.auto_glossary,
                    "prescan_enabled": projection.prescan_enabled,
                    "use_ollama_discovery": projection.use_ollama_discovery,
                    "discovery_backend": projection.discovery_backend,
                },
            ),
        ):
            updated_modules[config.module_id] = config
        return self._set_draft(
            SettingsDraft(
                application=replace(
                    self._draft.application,
                    theme=self._draft.application.theme,
                ),
                project=project,
                module_configs=tuple(
                    updated_modules[key] for key in sorted(updated_modules)
                ),
                provider_profiles=profiles,
            )
        )

    def preview_run(
        self,
        *,
        project_id: str,
        invocation: RunInvocation,
        internal_options: InternalRunOptions = InternalRunOptions(),
    ) -> CompilationResult:
        return compile_run_settings(
            project_id=project_id,
            application=self._draft.application,
            project=self._draft.project,
            module_configs=self._draft.module_configs,
            provider_profile=self._draft.translation_profile,
            discovery_profile=self._draft.discovery_profile,
            invocation=invocation,
            runtime_status=self._runtime_status,
            registry=self._registry,
            internal_options=internal_options,
        )

    def preview_effective_run_summary(
        self,
        *,
        project_id: str,
        invocation: RunInvocation,
        internal_options: InternalRunOptions = InternalRunOptions(),
    ) -> EffectiveRunSummary:
        """Describe the exact typed run candidate without starting any work."""

        result = self.preview_run(
            project_id=project_id,
            invocation=invocation,
            internal_options=internal_options,
        )
        values = result.snapshot.pipeline_values
        profile = self._draft.translation_profile
        if profile is None:
            provider = "Unresolved"
            model = "No model selected"
        else:
            provider = f"{profile.display_name} ({provider_kind_label(profile.kind)})"
            model = profile.model_id or profile.local_model_path or "Default model"
            model = str(model).replace("\\", "/").rsplit("/", 1)[-1]
        runtime = "GPU when available" if values.get("use_gpu") else "CPU"
        return EffectiveRunSummary(
            ready=result.ready,
            pending_changes=self.dirty,
            language_pair=(
                f"{values.get('source_lang', self._draft.project.source_language)}"
                f" → {values.get('target_lang', self._draft.project.target_language)}"
            ),
            provider=provider,
            model=model,
            detection_and_ocr=(
                f"{values.get('detector_engine', 'Unresolved')} / "
                f"{values.get('ocr_engine', 'Unresolved')}"
            ),
            cleanup_and_style=(
                f"{values.get('inpaint_mode', 'Unresolved')} / "
                f"{values.get('font_detection', 'Unresolved')}"
            ),
            runtime=runtime,
            snapshot_id=result.snapshot.snapshot_id,
            issues=result.issues,
        )

    def apply(self) -> SettingsDraft:
        self._validate(self._draft)
        self._baseline = self._draft
        return self._baseline

    def cancel(self) -> SettingsDraft:
        self._draft = self._baseline
        return self._draft

    def _set_draft(self, value: SettingsDraft) -> SettingsDraft:
        normalized = self._normalize(value)
        self._validate(normalized)
        self._draft = normalized
        return normalized

    def _normalize(self, value: SettingsDraft) -> SettingsDraft:
        modules = tuple(
            _demote_active_legacy_values(self._registry, config)
            for config in value.module_configs
        )
        if modules == value.module_configs:
            return value
        return replace(value, module_configs=modules)

    def _validate(self, value: SettingsDraft) -> None:
        for config in value.module_configs:
            self._registry.validate_config(config, allow_legacy=False)
