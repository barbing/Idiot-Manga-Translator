# -*- coding: utf-8 -*-
"""Compile typed GUI settings into one immutable workflow-run snapshot.

The compiler is deliberately one-way.  Widgets and QSettings are not inputs;
callers provide typed settings scopes and a selected public provider profile.
The existing mutable ``PipelineSettings`` value is materialized only as a
disposable compatibility copy at the current controller boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from app.config.defaults import AppDefaults
from app.config.module_registry import (
    DEFAULT_MODULE_REGISTRY,
    ModuleSchemaRegistry,
    SettingLifecycle,
    SettingValidationError,
)
from app.config.provider_profiles import ProviderKind, ProviderProfile
from app.config.settings_contracts import (
    ApplicationPreferences,
    CredentialReference,
    ModuleConfig,
    ProjectConfig,
    RunSettingsSnapshot,
    RuntimeStatus,
    SettingsScope,
    canonical_fingerprint,
    thaw_json,
)


RUN_SETTINGS_COMPILER_VERSION = "gui2_run_settings_compiler_v1"
_CREDENTIAL_LOCATOR_FINGERPRINT_DOMAIN = (
    "yomiframe.provider-credential-locator.v1"
)


class CompilationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CompilationIssue:
    code: str
    severity: CompilationSeverity
    path: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.path.strip() or not self.message.strip():
            raise ValueError("compilation issue fields must not be empty")
        object.__setattr__(self, "severity", CompilationSeverity(self.severity))


@dataclass(frozen=True, slots=True)
class RunInvocation:
    import_dir: str
    export_dir: str
    json_path: str
    files_whitelist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("import_dir", "export_dir", "json_path"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            object.__setattr__(self, name, value.strip())
        whitelist = tuple(str(item).strip() for item in self.files_whitelist)
        if any(not item for item in whitelist):
            raise ValueError("files_whitelist must not contain empty paths")
        if len(whitelist) != len(set(whitelist)):
            raise ValueError("files_whitelist must not contain duplicates")
        object.__setattr__(self, "files_whitelist", whitelist)


@dataclass(frozen=True, slots=True)
class InternalRunOptions:
    """Non-persisted developer/validation controls for one invocation.

    They are not application, project, module, or provider preferences, but
    execution-affecting values are compiled into ``pipeline_values`` so two
    materially different invocations cannot share a run-snapshot identity.
    """

    prescan_use_ner: bool = False
    debug_ocr: bool = False
    prescan_only: bool = False
    debug_artifacts: bool = False
    debug_pages: str = ""
    debug_stages: str = ""
    debug_disabled_stages: str = ""
    debug_dir: str = ""
    private_cleanup_validation_stop_after_cleanup: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "prescan_use_ner",
            "debug_ocr",
            "prescan_only",
            "debug_artifacts",
            "private_cleanup_validation_stop_after_cleanup",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        for field_name in (
            "debug_pages",
            "debug_stages",
            "debug_disabled_stages",
            "debug_dir",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if any(character in value for character in ("\r", "\n", "\0")):
                raise ValueError(f"{field_name} contains invalid control characters")


@dataclass(frozen=True, slots=True)
class RuntimeProviderBinding:
    """Memory-only binding between a run and an opaque credential reference."""

    profile_id: str | None
    provider_kind: ProviderKind | None
    credential_reference: object | None = None


@dataclass(frozen=True, slots=True)
class CompilationResult:
    snapshot: RunSettingsSnapshot
    runtime_binding: RuntimeProviderBinding
    issues: tuple[CompilationIssue, ...]
    internal_options: InternalRunOptions

    @property
    def ready(self) -> bool:
        return not any(issue.severity is CompilationSeverity.ERROR for issue in self.issues)

    def require_ready(self) -> RunSettingsSnapshot:
        if not self.ready:
            codes = ", ".join(issue.code for issue in self.issues if issue.severity is CompilationSeverity.ERROR)
            raise ValueError(f"run settings are unresolved: {codes}")
        return self.snapshot


def _available_capabilities(
    runtime_status: RuntimeStatus | None,
) -> frozenset[str] | None:
    if runtime_status is None:
        return None
    capabilities: set[str] = set()
    for asset_id, status in runtime_status.installed_assets.items():
        if status is True or str(status).strip().lower() in {"ready", "available", "installed", "valid"}:
            capabilities.add(str(asset_id))
    return frozenset(capabilities)


def _module_configs_by_id(configs: Iterable[ModuleConfig]) -> dict[str, ModuleConfig]:
    result: dict[str, ModuleConfig] = {}
    for config in configs:
        if not isinstance(config, ModuleConfig):
            raise TypeError("module_configs must contain ModuleConfig values")
        if config.module_id in result:
            raise ValueError(f"duplicate module config: {config.module_id}")
        result[config.module_id] = config
    return result


def _resolve_module_values(
    registry: ModuleSchemaRegistry,
    configs: Mapping[str, ModuleConfig],
    *,
    capabilities: frozenset[str] | None,
    issues: list[CompilationIssue],
) -> dict[str, dict[str, Any]]:
    unknown_modules = frozenset(configs) - frozenset(registry.module_map)
    for module_id in sorted(unknown_modules):
        issues.append(
            CompilationIssue(
                code="unsupported_module_config",
                severity=CompilationSeverity.ERROR,
                path=f"modules.{module_id}",
                message="The project references a module schema that is not installed.",
            )
        )

    resolved: dict[str, dict[str, Any]] = {}
    for module in registry.modules:
        config = configs.get(
            module.module_id,
            ModuleConfig(
                module_id=module.module_id,
                module_schema_version=module.schema_version,
            ),
        )
        if config.module_schema_version != module.schema_version:
            issues.append(
                CompilationIssue(
                    code="module_schema_mismatch",
                    severity=CompilationSeverity.ERROR,
                    path=f"modules.{module.module_id}",
                    message=(
                        f"Expected schema {module.schema_version}, got "
                        f"{config.module_schema_version}."
                    ),
                )
            )
            continue
        unknown_settings = frozenset(config.values) - frozenset(module.definitions)
        if unknown_settings:
            issues.append(
                CompilationIssue(
                    code="unsupported_module_setting",
                    severity=CompilationSeverity.ERROR,
                    path=f"modules.{module.module_id}",
                    message=f"Unsupported settings: {sorted(unknown_settings)}.",
                )
            )
        values: dict[str, Any] = {}
        for definition in module.settings:
            if definition.lifecycle is not SettingLifecycle.SUPPORTED:
                if definition.setting_id in config.values:
                    issues.append(
                        CompilationIssue(
                            code="legacy_setting_active",
                            severity=CompilationSeverity.ERROR,
                            path=definition.qualified_id,
                            message="Legacy-only settings cannot be active run values.",
                        )
                    )
                continue
            value = config.values.get(definition.setting_id, definition.default)
            try:
                definition.validate_value(
                    value,
                    available_capabilities=(
                        None
                        if module.module_id == "translation"
                        and definition.setting_id
                        in {"use_ollama_discovery", "discovery_backend"}
                        else capabilities
                    ),
                    allow_legacy=False,
                )
            except (SettingValidationError, TypeError, ValueError) as exc:
                issues.append(
                    CompilationIssue(
                        code="invalid_module_value",
                        severity=CompilationSeverity.ERROR,
                        path=definition.qualified_id,
                        message=str(exc),
                    )
                )
                continue
            values[definition.setting_id] = thaw_json(value)
        for setting_id in sorted(config.legacy_values):
            issues.append(
                CompilationIssue(
                    code="inactive_legacy_evidence",
                    severity=CompilationSeverity.WARNING,
                    path=f"modules.{module.module_id}.legacy_values.{setting_id}",
                    message="Preserved legacy evidence is not applied to this run.",
                )
            )
        resolved[module.module_id] = values
    return resolved


def _provider_backend(profile: ProviderProfile) -> str:
    return {
        ProviderKind.GGUF: "GGUF",
        ProviderKind.OLLAMA: "Ollama",
        ProviderKind.DEEPSEEK: "DeepSeek",
        ProviderKind.OPENAI_COMPATIBLE: "OpenAI-compatible",
    }[profile.kind]


def credential_locator_fingerprint(
    credential_reference: CredentialReference,
) -> str:
    """Return the redacted, domain-separated identity of one locator."""

    if not isinstance(credential_reference, CredentialReference):
        raise TypeError("credential_reference must be a CredentialReference")
    return canonical_fingerprint(
        {
            "domain": _CREDENTIAL_LOCATOR_FINGERPRINT_DOMAIN,
            "locator_kind": credential_reference.kind.value,
            "locator_value": credential_reference.reference,
        }
    )


def _public_provider_snapshot(profile: ProviderProfile) -> dict[str, object]:
    """Return the redacted run provenance for one provider role.

    Portable provider exports intentionally omit credential references.  A run
    snapshot still needs to distinguish which opaque locator was selected so a
    later credential relink cannot silently reuse the same semantic run
    identity.  The domain-separated digest records that distinction without
    exposing the locator kind, value, label, or resolved secret.
    """

    payload = profile.to_public_export_dict()
    credential_reference = profile.credential_ref
    if credential_reference is not None:
        payload["credential_locator_fingerprint"] = (
            credential_locator_fingerprint(credential_reference)
        )
    return payload


def compile_run_settings(
    *,
    project_id: str,
    application: ApplicationPreferences,
    project: ProjectConfig,
    module_configs: Sequence[ModuleConfig],
    provider_profile: ProviderProfile | None,
    invocation: RunInvocation,
    runtime_status: RuntimeStatus | None = None,
    discovery_profile: ProviderProfile | None = None,
    registry: ModuleSchemaRegistry = DEFAULT_MODULE_REGISTRY,
    internal_options: InternalRunOptions = InternalRunOptions(),
) -> CompilationResult:
    """Resolve all settings once without executing providers or modules."""

    if not isinstance(application, ApplicationPreferences):
        raise TypeError("application must be ApplicationPreferences")
    if not isinstance(project, ProjectConfig):
        raise TypeError("project must be ProjectConfig")
    if provider_profile is not None and not isinstance(provider_profile, ProviderProfile):
        raise TypeError("provider_profile must be ProviderProfile or None")
    if discovery_profile is not None and not isinstance(discovery_profile, ProviderProfile):
        raise TypeError("discovery_profile must be ProviderProfile or None")
    if not isinstance(invocation, RunInvocation):
        raise TypeError("invocation must be RunInvocation")
    if not isinstance(internal_options, InternalRunOptions):
        raise TypeError("internal_options must be InternalRunOptions")
    if runtime_status is not None and not isinstance(runtime_status, RuntimeStatus):
        raise TypeError("runtime_status must be RuntimeStatus or None")

    issues: list[CompilationIssue] = []
    translation_reference = project.provider_profile_references.get("translation")
    discovery_reference = project.provider_profile_references.get("discovery")
    if translation_reference is None:
        issues.append(
            CompilationIssue(
                code="translation_profile_required",
                severity=CompilationSeverity.ERROR,
                path="providers.translation",
                message="Select a translation provider profile before Start.",
            )
        )
    elif provider_profile is None:
        issues.append(
            CompilationIssue(
                code="translation_profile_unresolved",
                severity=CompilationSeverity.ERROR,
                path="providers.translation",
                message="The selected translation provider profile must be relinked.",
            )
        )
    elif provider_profile.profile_id != translation_reference:
        issues.append(
            CompilationIssue(
                code="translation_profile_reference_mismatch",
                severity=CompilationSeverity.ERROR,
                path="providers.translation",
                message="The supplied translation profile does not match ProjectConfig.",
            )
        )
    configs = _module_configs_by_id(module_configs)
    capabilities = _available_capabilities(runtime_status)
    modules = _resolve_module_values(
        registry,
        configs,
        capabilities=capabilities,
        issues=issues,
    )

    if provider_profile is not None and not provider_profile.transport_available:
        issues.append(
            CompilationIssue(
                code="provider_transport_unavailable",
                severity=CompilationSeverity.ERROR,
                path=f"providers.{provider_profile.profile_id}",
                message=(
                    "This provider profile is valid configuration, but its translation "
                    "transport is not implemented in the current translation owner."
                ),
            )
        )
    if provider_profile is not None:
        for requirement in provider_profile.configuration_issues:
            field_name = {
                "credential_reference_required": "credential_ref",
                "local_model_path_required": "local_model_path",
                "endpoint_required": "endpoint",
                "model_id_required": "model_id",
            }.get(requirement, "configuration")
            issues.append(
                CompilationIssue(
                    code=requirement,
                    severity=CompilationSeverity.ERROR,
                    path=f"providers.{provider_profile.profile_id}.{field_name}",
                    message="The provider profile is incomplete and must be relinked.",
                )
            )

    defaults = AppDefaults()
    detection = modules.get("detection", {})
    ocr = modules.get("ocr", {})
    cleanup = modules.get("cleanup", {})
    source_style = modules.get("source_style", {})
    renderer = modules.get("renderer", {})
    translation = modules.get("translation", {})
    runtime = modules.get("runtime", {})

    provider_kind = provider_profile.kind if provider_profile is not None else None
    generation = (
        provider_profile.generation_for_model() if provider_profile is not None else None
    )
    gguf_options = (
        provider_profile.gguf_options_for_model() if provider_profile is not None else None
    )
    ollama_options = (
        provider_profile.ollama_options_for_model() if provider_profile is not None else None
    )

    discovery_backend = str(translation.get("discovery_backend", "Ollama"))
    # Pre-scan uses the selected translation client.  Only the optional deep
    # discovery path owns a second Ollama/GGUF provider profile.
    discovery_enabled = bool(translation.get("use_ollama_discovery", False))
    expected_discovery_kind = {
        "Ollama": ProviderKind.OLLAMA,
        "GGUF": ProviderKind.GGUF,
    }.get(discovery_backend)
    discovery = discovery_profile if discovery_enabled else None
    if discovery_enabled:
        if discovery_reference is None:
            issues.append(
                CompilationIssue(
                    code="discovery_profile_required",
                    severity=CompilationSeverity.ERROR,
                    path="providers.discovery",
                    message="Select a discovery provider profile for deep discovery.",
                )
            )
            discovery = None
        elif discovery is None:
            if (
                provider_profile is not None
                and expected_discovery_kind is provider_kind
                and provider_profile.profile_id == discovery_reference
            ):
                discovery = provider_profile
            else:
                issues.append(
                    CompilationIssue(
                        code="discovery_profile_unresolved",
                        severity=CompilationSeverity.ERROR,
                        path="providers.discovery",
                        message="The selected discovery provider profile must be relinked.",
                    )
                )
        elif discovery.profile_id != discovery_reference:
            issues.append(
                CompilationIssue(
                    code="discovery_profile_reference_mismatch",
                    severity=CompilationSeverity.ERROR,
                    path="providers.discovery",
                    message="The supplied discovery profile does not match ProjectConfig.",
                )
            )
            discovery = None
        elif discovery.kind is not expected_discovery_kind:
            issues.append(
                CompilationIssue(
                    code="discovery_profile_kind_mismatch",
                    severity=CompilationSeverity.ERROR,
                    path="providers.discovery",
                    message=(
                        f"Discovery backend {discovery_backend} cannot use a "
                        f"{discovery.kind.value} profile."
                    ),
                )
            )
            discovery = None
        if discovery is not None:
            for requirement in discovery.configuration_issues:
                field_name = {
                    "credential_reference_required": "credential_ref",
                    "local_model_path_required": "local_model_path",
                    "endpoint_required": "endpoint",
                    "model_id_required": "model_id",
                }.get(requirement, "configuration")
                issues.append(
                    CompilationIssue(
                        code=f"discovery_{requirement}",
                        severity=CompilationSeverity.ERROR,
                        path=f"providers.discovery.{field_name}",
                        message="The selected discovery profile is incomplete.",
                    )
                )
    discovery_model = (
        (
            discovery.local_model_path
            if discovery.kind is ProviderKind.GGUF
            else discovery.model_id
        )
        if discovery is not None
        else None
    )

    pipeline_values: dict[str, Any] = {
        "import_dir": invocation.import_dir,
        "export_dir": invocation.export_dir,
        "json_path": invocation.json_path,
        "output_suffix": project.output_suffix,
        "source_lang": project.source_language,
        "target_lang": project.target_language,
        "ollama_model": (
            provider_profile.model_id
            if provider_profile is not None and provider_kind is ProviderKind.OLLAMA
            else "auto-detect"
        ),
        "ollama_base_url": (
            provider_profile.endpoint
            if provider_profile is not None and provider_kind is ProviderKind.OLLAMA
            else defaults.ollama_base_url
        ),
        "style_guide_path": project.glossary_reference or "",
        "font_name": renderer.get("font_name", defaults.font_name),
        "use_gpu": runtime.get("use_gpu", True),
        "filter_background": detection.get("filter_background", True),
        "filter_strength": detection.get("filter_strength", defaults.filter_strength),
        "detector_engine": detection.get("engine", defaults.detector_engine),
        "ocr_engine": ocr.get("engine", defaults.ocr_engine),
        "inpaint_mode": cleanup.get("inpaint_mode", defaults.inpaint_mode),
        "font_detection": source_style.get("font_detection", defaults.font_detection),
        "translator_backend": (
            _provider_backend(provider_profile) if provider_profile is not None else ""
        ),
        "deepseek_model": (
            provider_profile.model_id
            if provider_profile is not None and provider_kind is ProviderKind.DEEPSEEK
            else defaults.deepseek_model
        ),
        "deepseek_base_url": (
            provider_profile.endpoint
            if provider_profile is not None and provider_kind is ProviderKind.DEEPSEEK
            else defaults.deepseek_base_url
        ),
        "ollama_temperature": (
            generation.temperature
            if generation is not None and provider_kind is not ProviderKind.GGUF
            else defaults.ollama_temperature
        ),
        "ollama_top_p": (
            generation.top_p
            if generation is not None and provider_kind is not ProviderKind.GGUF
            else defaults.ollama_top_p
        ),
        "ollama_context": (
            ollama_options.context_tokens if ollama_options else defaults.ollama_context
        ),
        "gguf_temperature": (
            generation.temperature
            if generation is not None and provider_kind is ProviderKind.GGUF
            else defaults.gguf_temperature
        ),
        "gguf_top_p": (
            generation.top_p
            if generation is not None and provider_kind is ProviderKind.GGUF
            else defaults.gguf_top_p
        ),
        "gguf_model_path": (
            provider_profile.local_model_path
            if provider_profile is not None and provider_kind is ProviderKind.GGUF
            else ""
        ),
        "gguf_prompt_style": gguf_options.prompt_style if gguf_options else defaults.gguf_prompt_style,
        "gguf_n_ctx": gguf_options.n_ctx if gguf_options else defaults.gguf_n_ctx,
        "gguf_n_gpu_layers": (
            gguf_options.n_gpu_layers if gguf_options else defaults.gguf_n_gpu_layers
        ),
        "gguf_n_threads": gguf_options.n_threads if gguf_options else defaults.gguf_n_threads,
        "gguf_n_batch": gguf_options.n_batch if gguf_options else defaults.gguf_n_batch,
        "fast_mode": runtime.get("fast_mode", defaults.fast_mode),
        "auto_glossary": translation.get("auto_glossary", defaults.auto_glossary),
        "detector_input_size": detection.get("input_size", 640),
        "inpaint_model_id": cleanup.get("inpaint_model_id", defaults.inpaint_model),
        "use_ollama_discovery": translation.get("use_ollama_discovery", False),
        "files_whitelist": tuple(invocation.files_whitelist) or None,
        "discovery_model": discovery_model,
        "discovery_backend": discovery_backend,
        "discovery_base_url": (
            discovery.endpoint
            if discovery is not None and discovery.kind is ProviderKind.OLLAMA
            else defaults.ollama_base_url
        ),
        "discovery_context": (
            discovery.ollama_options.context_tokens
            if discovery is not None
            and discovery.kind is ProviderKind.OLLAMA
            and discovery.ollama_options is not None
            else defaults.ollama_context
        ),
        "prescan_enabled": translation.get("prescan_enabled", False),
        "prescan_use_ner": internal_options.prescan_use_ner,
        "debug_ocr": internal_options.debug_ocr,
        "prescan_only": internal_options.prescan_only,
        "gguf_cross_page_context": translation.get(
            "gguf_cross_page_context", defaults.gguf_cross_page_context
        ),
        "debug_artifacts": internal_options.debug_artifacts,
        "debug_pages": internal_options.debug_pages,
        "debug_stages": internal_options.debug_stages,
        "debug_disabled_stages": internal_options.debug_disabled_stages,
        "debug_dir": internal_options.debug_dir,
        "private_cleanup_validation_stop_after_cleanup": (
            internal_options.private_cleanup_validation_stop_after_cleanup
        ),
    }

    public_provider_snapshots = {
        "translation": (
            _public_provider_snapshot(provider_profile)
            if provider_profile is not None
            else {"status": "unresolved", "profile_id": translation_reference}
        ),
        "discovery": (
            _public_provider_snapshot(discovery) if discovery is not None else None
        ),
    }
    scope_fingerprints = {
        SettingsScope.APPLICATION.value: application.fingerprint,
        SettingsScope.PROJECT.value: project.fingerprint,
        SettingsScope.MODULE.value: canonical_fingerprint(
            [config.to_dict() for config in sorted(module_configs, key=lambda item: item.module_id)]
        ),
        SettingsScope.PROVIDER.value: canonical_fingerprint(
            public_provider_snapshots
        ),
    }
    unresolved = tuple(
        sorted(
            {
                f"{issue.code}:{issue.path}"
                for issue in issues
                if issue.severity is CompilationSeverity.ERROR
            }
        )
    )
    snapshot = RunSettingsSnapshot(
        project_id=project_id,
        pipeline_values=pipeline_values,
        provider_profile_snapshot=public_provider_snapshots,
        scope_fingerprints=scope_fingerprints,
        unresolved_requirements=unresolved,
    )
    binding = RuntimeProviderBinding(
        profile_id=provider_profile.profile_id if provider_profile is not None else None,
        provider_kind=provider_kind,
        credential_reference=(
            provider_profile.credential_ref if provider_profile is not None else None
        ),
    )
    return CompilationResult(
        snapshot=snapshot,
        runtime_binding=binding,
        issues=tuple(issues),
        internal_options=internal_options,
    )


T = TypeVar("T")


def apply_runtime_admission_overrides(
    result: CompilationResult,
    *,
    overrides: Sequence[tuple[str, object]],
    effective_pipeline_values_fingerprint: str,
) -> CompilationResult:
    """Bind an admitted hardware resolution into a new immutable snapshot.

    The only supported resolution is the existing GGUF Automatic sentinel.
    Persisted provider settings and explicit layer counts remain authoritative;
    this helper never probes hardware and cannot introduce another setting.
    """

    if not isinstance(result, CompilationResult):
        raise TypeError("result must be CompilationResult")
    snapshot = result.require_ready()
    values = thaw_json(snapshot.pipeline_values)
    normalized = tuple((str(key).strip(), value) for key, value in overrides)
    if len(normalized) != len({key for key, _value in normalized}):
        raise ValueError("runtime admission overrides must not contain duplicates")
    for key, value in normalized:
        if key != "gguf_n_gpu_layers":
            raise ValueError(f"unsupported runtime admission override: {key}")
        if str(values.get("translator_backend") or "") != "GGUF":
            raise ValueError("GGUF layer resolution requires the GGUF backend")
        if int(values.get("gguf_n_gpu_layers", 0)) != -1:
            raise ValueError("an explicit GGUF layer setting cannot be overridden")
        if type(value) is not int or not 0 <= value <= 200:
            raise ValueError("resolved GGUF GPU layers must be an integer in 0..200")
        values[key] = value
    expected = str(effective_pipeline_values_fingerprint or "").strip()
    if canonical_fingerprint(values) != expected:
        raise ValueError("runtime admission override fingerprint mismatch")
    if not normalized:
        return result
    derived = RunSettingsSnapshot(
        project_id=snapshot.project_id,
        pipeline_values=values,
        provider_profile_snapshot=snapshot.provider_profile_snapshot,
        scope_fingerprints=snapshot.scope_fingerprints,
        unresolved_requirements=snapshot.unresolved_requirements,
        created_at=snapshot.created_at,
    )
    return CompilationResult(
        snapshot=derived,
        runtime_binding=result.runtime_binding,
        issues=result.issues,
        internal_options=result.internal_options,
    )


def materialize_pipeline_settings_snapshot(
    snapshot: RunSettingsSnapshot,
    *,
    factory: Callable[..., T] | None = None,
) -> T:
    """Materialize one already-compiled immutable run snapshot.

    Explicit owner revisions reuse the same compiled settings authority as a
    normal run.  This helper performs no widget/default merge and deliberately
    accepts no provider or project configuration inputs.
    """

    if not isinstance(snapshot, RunSettingsSnapshot):
        raise TypeError("snapshot must be a RunSettingsSnapshot")
    if snapshot.unresolved_requirements:
        raise ValueError("run settings snapshot has unresolved requirements")
    values = thaw_json(snapshot.pipeline_values)
    values.setdefault("ollama_base_url", AppDefaults().ollama_base_url)
    values.setdefault("discovery_base_url", AppDefaults().ollama_base_url)
    values.setdefault("discovery_context", AppDefaults().ollama_context)
    if values.get("files_whitelist") is not None:
        values["files_whitelist"] = list(values["files_whitelist"])
    if factory is None:
        from app.pipeline.controller import PipelineSettings

        factory = PipelineSettings
    return factory(**values)


def materialize_pipeline_settings(
    result: CompilationResult,
    *,
    factory: Callable[..., T] | None = None,
) -> T:
    """Create a fresh mutable compatibility value after validation.

    Execution-affecting developer controls have already been compiled into the
    immutable snapshot; materialization does not merge any second authority.
    """

    snapshot = result.require_ready()
    return materialize_pipeline_settings_snapshot(snapshot, factory=factory)
