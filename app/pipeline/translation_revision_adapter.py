# -*- coding: utf-8 -*-
"""Reuse-only adapter for the controller-owned translation policy."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from app.config.credential_store import (
    CredentialResolver,
)
from app.config.provider_profiles import ProviderKind
from app.config.run_settings_compiler import (
    RuntimeProviderBinding,
    credential_locator_fingerprint,
    materialize_pipeline_settings_snapshot,
)
from app.config.settings_contracts import thaw_json
from app.platform_services.credentials import build_credential_resolver

from .translation_revision_contracts import (
    CancellationProbe,
    TranslationExecutionReceipt,
    TranslationExecutionRequest,
    TranslationRevisionError,
    TranslationRevisionErrorCode,
)


def resolve_translation_runtime_binding(
    binding: RuntimeProviderBinding,
    *,
    credential_resolver: CredentialResolver | None = None,
) -> Any:
    """Resolve one opaque credential reference into a redacted run carrier.

    Call this only inside the short-lived editor worker.  The returned carrier
    never serializes or exposes the resolved value.
    """

    if not isinstance(binding, RuntimeProviderBinding):
        raise TypeError("binding must be a RuntimeProviderBinding")
    if binding.provider_kind is None:
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "The selected translation provider is unresolved.",
        )
    resolved_credential: str | None = None
    if binding.credential_reference is not None:
        resolver = credential_resolver or build_credential_resolver()
        resolved_credential = resolver.resolve(binding.credential_reference)
        if not resolved_credential:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE,
                "The selected provider credential is unavailable.",
            )
    if (
        binding.provider_kind is ProviderKind.DEEPSEEK
        and not resolved_credential
    ):
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE,
            "DeepSeek requires a resolved credential reference.",
        )
    from app.pipeline.controller import PipelineRuntimeBinding

    return PipelineRuntimeBinding(
        provider_kind=binding.provider_kind,
        resolved_credential=resolved_credential,
    )


class ControllerTranslationRevisionAdapter:
    """Compose the frozen controller helpers for one selected parent.

    The adapter compiles one transient assignment and role-derived region input.
    It does not run ``_process_page``, discover glossary terms, mutate context,
    write region/bundle state, or invoke any later owner.
    """

    def __init__(
        self,
        *,
        runtime_binding: Any | None = None,
        client_factory: Callable[[Any, Any | None], Any] | None = None,
        settings_materializer: Callable[..., Any] | None = None,
        model_inventory: Callable[[], Any] | None = None,
    ) -> None:
        self._runtime_binding = runtime_binding
        self._client_factory = client_factory
        self._settings_materializer = (
            settings_materializer or materialize_pipeline_settings_snapshot
        )
        self._model_inventory = model_inventory

    def _create_client(
        self,
        settings: Any,
        operation: Any,
    ) -> Any:
        backend = str(getattr(settings, "translator_backend", "") or "")
        runtime_binding = self._runtime_binding
        self._require_runtime_binding_matches(
            operation=operation,
            runtime_binding=runtime_binding,
        )
        if self._client_factory is not None:
            return self._client_factory(settings, runtime_binding)
        if backend == "GGUF":
            from app.translate.gguf_client import GGUFClient

            prompt_style = str(
                getattr(settings, "gguf_prompt_style", "qwen") or "qwen"
            )
            model_path = str(getattr(settings, "gguf_model_path", "") or "")
            if "sakura" in model_path.lower() and prompt_style == "qwen":
                prompt_style = "sakura"
            return GGUFClient(
                model_path=model_path,
                prompt_style=prompt_style,
                n_ctx=settings.gguf_n_ctx,
                n_gpu_layers=settings.gguf_n_gpu_layers,
                n_threads=settings.gguf_n_threads,
                n_batch=settings.gguf_n_batch,
            )
        if backend == "DeepSeek":
            from app.translate.ollama_client import DeepSeekClient

            if runtime_binding is None:
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE,
                    "DeepSeek requires a redacted runtime credential binding.",
                )
            if isinstance(runtime_binding, RuntimeProviderBinding):
                runtime_binding = resolve_translation_runtime_binding(
                    runtime_binding
                )
            try:
                credential = runtime_binding.credential_for_backend(
                    "DeepSeek"
                )
            except Exception as exc:
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE,
                    "The DeepSeek runtime credential binding is unavailable.",
                ) from exc
            return DeepSeekClient(
                base_url=settings.deepseek_base_url,
                model_name=settings.deepseek_model,
                api_key=credential,
            )
        if backend == "Ollama":
            from app.translate.ollama_client import OllamaClient

            return OllamaClient()
        raise TranslationRevisionError(
            TranslationRevisionErrorCode.SETTINGS_MISMATCH,
            "The selected translation provider is unsupported.",
        )

    @staticmethod
    def _require_runtime_binding_matches(
        *,
        operation: Any,
        runtime_binding: Any | None,
    ) -> None:
        if runtime_binding is None:
            return
        if not isinstance(runtime_binding, RuntimeProviderBinding):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Runtime provider binding lacks immutable profile identity.",
            )
        runtime_kind = getattr(runtime_binding.provider_kind, "value", None)
        if (
            str(runtime_binding.profile_id or "")
            != operation.provider.profile_id
            or str(runtime_kind or "").casefold().replace("_", "-")
            != operation.provider.provider_kind
        ):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Runtime provider binding differs from the immutable request.",
            )
        profiles = thaw_json(
            operation.run_settings_snapshot.provider_profile_snapshot
        )
        translation = profiles.get("translation")
        if not isinstance(translation, Mapping):
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Immutable translation provider provenance is unavailable.",
            )
        expected_locator = str(
            translation.get("credential_locator_fingerprint") or ""
        )
        reference = runtime_binding.credential_reference
        if expected_locator:
            try:
                observed_locator = credential_locator_fingerprint(reference)
            except (TypeError, ValueError):
                observed_locator = ""
            if observed_locator != expected_locator:
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                    "Runtime credential locator differs from the immutable request.",
                )
        elif reference is not None:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Runtime credential locator differs from the immutable request.",
            )

    @staticmethod
    def _require_provider_ready(
        *,
        client: Any,
        settings: Any,
        operation: Any,
        owner: Any,
        runtime_binding: Any | None,
        model_inventory: Callable[[], Any] | None,
    ) -> str:
        """Apply the frozen controller initialization gates to one client."""

        backend = str(getattr(settings, "translator_backend", "") or "")
        if backend in {"DeepSeek", "Ollama"}:
            is_available = getattr(client, "is_available", None)
            if not callable(is_available) or not is_available():
                message = (
                    owner._deepseek_unavailable_message(runtime_binding)
                    if backend == "DeepSeek"
                    else "Ollama server is not running. Start it with: ollama serve"
                )
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.PROVIDER_UNAVAILABLE,
                    message,
                )

        if backend == "GGUF":
            if owner._missing_required_gguf_model_path(settings):
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.MODEL_MISSING,
                    "GGUF model path is required for GGUF backend.",
                )
            return str(getattr(settings, "gguf_model_path", "") or "")
        if backend == "DeepSeek":
            return str(getattr(settings, "deepseek_model", "") or "")

        selected_model = str(getattr(settings, "ollama_model", "") or "")
        resolved_model = owner._resolve_model(selected_model)
        if resolved_model and selected_model != "auto-detect":
            inventory_loader = model_inventory or owner.list_models
            available = list(inventory_loader())
            if available and resolved_model not in available:
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.MODEL_MISSING,
                    f"Ollama model not found: {resolved_model}",
                )
        if resolved_model != operation.provider.model_id:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Resolved translation model differs from the immutable request.",
            )
        return resolved_model

    @staticmethod
    def _policy_metadata(
        *,
        request: TranslationExecutionRequest,
        route: str,
        language_ok: bool,
        record: Mapping[str, Any],
        terminal_symbol_evidence: Mapping[str, Any],
        applied_term_count: int,
        ignored_term_count: int,
    ) -> dict[str, Any]:
        safe_list_keys = (
            "translation_paths",
            "failure_retry_reason",
            "ensure_retry_hard_failure_reasons",
            "ensure_retry_soft_warning_reasons",
            "json_repair_fallback_status",
            "recent_context_ids",
            "glossary_context_ids",
        )
        metadata: dict[str, Any] = {
            "assignment_id": request.request.parent_id,
            "parent_role": request.request.parent_role,
            "policy_region_type": request.request.policy_region_type,
            "bubble_local_nested_speech": (
                request.request.bubble_local_nested_speech
            ),
            "translation_route": route,
            "language_ok": bool(language_ok),
            "applied_glossary_term_count": int(applied_term_count),
            "ignored_glossary_term_count": int(ignored_term_count),
            "terminal_symbol_evidence": dict(terminal_symbol_evidence),
            "translation_retry_count": int(
                record.get("translation_retry_count") or 0
            ),
            "translation_result_contract_status": str(
                record.get("translation_result_contract_status") or ""
            ),
        }
        for key in safe_list_keys:
            values = record.get(key)
            if isinstance(values, (list, tuple)):
                metadata[key] = [str(value) for value in values if str(value)]
        return metadata

    def translate(
        self,
        request: TranslationExecutionRequest,
        *,
        cancellation_probe: CancellationProbe | None = None,
    ) -> TranslationExecutionReceipt:
        if not isinstance(request, TranslationExecutionRequest):
            raise TypeError("request must be a TranslationExecutionRequest")
        operation = request.request
        cancelled = cancellation_probe or (lambda: False)
        if cancelled():
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.CANCELLED,
                "Translation was cancelled before provider initialization.",
            )
        settings = self._settings_materializer(operation.run_settings_snapshot)
        backend = str(getattr(settings, "translator_backend", "") or "")
        expected_backend = {
            "gguf": "GGUF",
            "ollama": "Ollama",
            "deepseek": "DeepSeek",
        }[operation.provider.provider_kind]
        if backend != expected_backend:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.SETTINGS_MISMATCH,
                "Materialized provider differs from the immutable run snapshot.",
            )

        from app.pipeline import controller as owner

        assignment = owner.TranslationAssignment(
            assignment_id=operation.parent_id,
            parent_id=operation.parent_id,
            source_text=operation.effective_source_text,
            cache_key=operation.effective_source_text,
            region_ids=[operation.parent_id],
            source_contract_owner="explicit_translation_revision",
            source_contract_region_id=operation.source_revision_id,
            source_contract_scope="parent",
            source_contract_stage="source_revision",
        )
        source_text = assignment.source_text
        region_id = assignment.region_ids[0]
        region = {
            "region_id": region_id,
            "type": operation.policy_region_type,
            "ocr_text": source_text,
            "translation": "",
            "flags": {
                "ignore": False,
                "needs_review": False,
                "bg_text": operation.parent_role == "caption",
            },
            "render": {},
        }
        regions = [region]
        context_lines = list(operation.prior_page_context)
        style_guide = thaw_json(operation.glossary_snapshot)
        prompt_style_guide = owner._build_page_style_guide(
            style_guide,
            [source_text],
        )
        available_terms = owner._matched_glossary_terms(
            source_text,
            style_guide,
        )
        record: dict[str, Any] = {"translation_unit_id": operation.parent_id}
        owner._translation_perf_set_glossary_context(record, available_terms)

        client: Any = None
        route = ""
        language_ok = False
        terminal_symbol_evidence: Mapping[str, Any] = {}
        applied_terms: list[dict[str, Any]] = []
        ignored_terms: list[dict[str, Any]] = []
        quality_warnings: list[str] = []
        target_text = ""
        try:
            client = self._create_client(settings, operation)
            resolved_model = self._require_provider_ready(
                client=client,
                settings=settings,
                operation=operation,
                owner=owner,
                runtime_binding=self._runtime_binding,
                model_inventory=self._model_inventory,
            )
            if hasattr(client, "translate_glossary"):
                setattr(client, "model_name", resolved_model)
            if cancelled():
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.CANCELLED,
                    "Translation was cancelled before inference.",
                )

            is_single = owner._should_single_translate_text(
                source_text,
                [region_id],
                regions,
            )
            deepseek_context_lane = None
            if is_single:
                deepseek_context_lane = owner._deepseek_short_batch_context_lane(
                    source_text,
                    [region_id],
                    regions,
                    target_lang=settings.target_lang,
                    settings=settings,
                )
            raw_translation = ""
            if not is_single:
                route = "batch"
                batch = owner._batch_translate(
                    client,
                    operation.provider.model_id,
                    settings.source_lang,
                    settings.target_lang,
                    prompt_style_guide,
                    [{"id": "t000", "text": source_text}],
                    context_lines=context_lines,
                    settings=settings,
                    debug_records_by_text={source_text: record},
                )
                raw_translation = str(batch.get("t000") or "")
                if raw_translation:
                    raw_translation = owner._enforce_glossary(
                        raw_translation,
                        source_text,
                        style_guide,
                    )
                    if owner._has_glossary_count_mismatch(
                        source_text,
                        raw_translation,
                        style_guide,
                    ):
                        protected = owner._translate_with_glossary_placeholders(
                            client,
                            operation.provider.model_id,
                            settings.source_lang,
                            settings.target_lang,
                            source_text,
                            available_terms,
                            debug_record=record,
                            debug_phase="batch_glossary_placeholder",
                        )
                        owner._translation_perf_add_path(
                            record,
                            "glossary_placeholder_repair",
                        )
                        record.setdefault(
                            "json_repair_fallback_status",
                            [],
                        ).append(
                            "glossary_placeholder_repair_after_batch"
                        )
                        if protected:
                            raw_translation = owner._enforce_glossary(
                                protected,
                                source_text,
                                style_guide,
                            )
                if owner._translation_reuses_recent_context(
                    raw_translation,
                    source_text,
                    context_lines,
                ):
                    raw_translation = ""

            if is_single or not raw_translation:
                route = (
                    "single"
                    if is_single
                    else "batch_then_single_fallback"
                )
                single_context = (
                    context_lines
                    if owner._should_use_context_for_text(
                        source_text,
                        [region_id],
                        regions,
                    )
                    else []
                )
                # A one-parent DeepSeek short lane remains a single request in
                # the frozen controller; retain the lane fact for audit only.
                if deepseek_context_lane is False:
                    single_context = []
                raw_translation = owner._translate_single(
                    client,
                    operation.provider.model_id,
                    settings.source_lang,
                    settings.target_lang,
                    prompt_style_guide,
                    source_text,
                    context_lines=single_context,
                    settings=settings,
                    debug_record=record,
                )
                if owner._translation_reuses_recent_context(
                    raw_translation,
                    source_text,
                    single_context,
                ):
                    owner._translation_perf_add_path(
                        record,
                        "context_reuse_retry_no_context",
                    )
                    record.setdefault(
                        "failure_retry_reason",
                        [],
                    ).append("translation_reused_recent_context")
                    raw_translation = owner._translate_single(
                        client,
                        operation.provider.model_id,
                        settings.source_lang,
                        settings.target_lang,
                        prompt_style_guide,
                        source_text,
                        context_lines=[],
                        settings=settings,
                        debug_record=record,
                    )
                raw_translation = owner._enforce_glossary(
                    raw_translation,
                    source_text,
                    style_guide,
                )
                if owner._has_glossary_count_mismatch(
                    source_text,
                    raw_translation,
                    style_guide,
                ):
                    owner._translation_perf_add_path(
                        record,
                        "glossary_placeholder_repair",
                    )
                    record.setdefault(
                        "json_repair_fallback_status",
                        [],
                    ).append(
                        "glossary_placeholder_repair_after_single"
                    )
                    protected = owner._translate_with_glossary_placeholders(
                        client,
                        operation.provider.model_id,
                        settings.source_lang,
                        settings.target_lang,
                        source_text,
                        available_terms,
                        debug_record=record,
                        debug_phase="single_glossary_placeholder",
                    )
                    if protected:
                        raw_translation = owner._enforce_glossary(
                            protected,
                            source_text,
                            style_guide,
                        )

            target_text, language_ok = owner._ensure_target_language(
                client,
                operation.provider.model_id,
                settings.source_lang,
                settings.target_lang,
                source_text,
                raw_translation,
                is_bubble=operation.policy_region_type == "speech_bubble",
                debug_record=record,
            )
            if target_text:
                target_text = owner._enforce_glossary(
                    target_text,
                    source_text,
                    style_guide,
                )
                pre_repair = target_text
                if available_terms:
                    owner._translation_perf_add_path(
                        record,
                        "glossary_repair_check",
                    )
                target_text = owner._repair_translation_with_glossary(
                    client,
                    operation.provider.model_id,
                    settings.source_lang,
                    settings.target_lang,
                    source_text,
                    target_text,
                    style_guide,
                    debug_record=record,
                )
                if owner._translation_is_unsafe_for_output(
                    target_text,
                    source_text,
                ):
                    target_text = pre_repair
            if (
                settings.target_lang == "Simplified Chinese"
                and owner._is_short_reaction_source(source_text)
            ):
                deterministic = owner._translate_short_reaction_fallback(
                    source_text,
                    settings.target_lang,
                )
                if deterministic:
                    target_text = deterministic
                    language_ok = True
            if target_text:
                target_text = owner._apply_source_level_semantic_corrections(
                    source_text,
                    target_text,
                )
                target_text = owner._normalize_translation_format_for_record(
                    settings.target_lang,
                    target_text,
                    record,
                    stage="final_translation_assignment",
                )
                owner._translation_perf_set_final(
                    record,
                    translation=target_text,
                )
                if operation.bubble_local_nested_speech:
                    target_text, repair_reasons = (
                        owner._repair_bubble_local_nested_speech_translation(
                            source_text,
                            target_text,
                            settings.target_lang,
                        )
                    )
                    if repair_reasons:
                        record["bubble_local_translation_repair_reasons"] = (
                            list(repair_reasons)
                        )
                        owner._translation_perf_set_final(
                            record,
                            translation=target_text,
                            status="bubble_local_translation_repair",
                        )
                target_text, terminal_symbol_evidence = (
                    owner._preserve_repeated_terminal_emphasis_symbols(
                        source_text,
                        target_text,
                    )
                )
                if terminal_symbol_evidence.get("changed"):
                    owner._translation_perf_record_terminal_symbol_conservation(
                        record,
                        terminal_symbol_evidence,
                    )
                    owner._translation_perf_set_final(
                        record,
                        translation=target_text,
                        status="terminal_symbol_multiplicity_repaired",
                    )

            for item in available_terms:
                source = str(item.get("source") or "").strip()
                target = str(item.get("target") or "").strip()
                if target and target in target_text:
                    applied_terms.append(item)
                else:
                    ignored_terms.append(item)
                    if source and target:
                        quality_warnings.append(
                            f"missing_glossary_target:{source}->{target}"
                        )
            if not language_ok:
                quality_warnings.append("target_language_check_failed")
            if target_text and owner._translation_is_unsafe_for_output(
                target_text,
                source_text,
            ):
                quality_warnings.append("unsafe_output_shape")
            owner._translation_perf_set_glossary_status(
                record,
                applied_terms=applied_terms,
                ignored_terms=ignored_terms,
                warnings=quality_warnings,
            )
            if cancelled():
                raise TranslationRevisionError(
                    TranslationRevisionErrorCode.CANCELLED,
                    "Translation completed after cancellation; no result was accepted.",
                )
        except TranslationRevisionError:
            raise
        except Exception as exc:
            raise TranslationRevisionError(
                TranslationRevisionErrorCode.TRANSLATION_FAILED,
                "The existing translation policy failed for the selected parent.",
            ) from exc
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        # Match the controller lifecycle: teardown must not mask
                        # the policy result or its owning exception.
                        pass

        return TranslationExecutionReceipt(
            target_text=target_text,
            source_fingerprint=operation.effective_source_fingerprint,
            run_settings_fingerprint=operation.run_settings_fingerprint,
            provider=operation.provider,
            glossary_fingerprint=operation.glossary_fingerprint,
            context_fingerprint=operation.context_fingerprint,
            policy_metadata=self._policy_metadata(
                request=request,
                route=route,
                language_ok=language_ok,
                record=record,
                terminal_symbol_evidence=terminal_symbol_evidence,
                applied_term_count=len(applied_terms),
                ignored_term_count=len(ignored_terms),
            ),
            quality_warnings=tuple(dict.fromkeys(quality_warnings)),
        )


__all__ = [
    "ControllerTranslationRevisionAdapter",
    "resolve_translation_runtime_binding",
]
