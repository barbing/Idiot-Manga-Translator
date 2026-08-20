# -*- coding: utf-8 -*-
"""Public provider profile contracts for GUI-2.

This module describes provider configuration only. In particular, a generic
OpenAI-compatible profile does not imply that a translation transport exists;
that transport remains a separately authorized translation-owner concern.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

from app.config.credential_store import credential_reference_from_dict
from app.config.settings_contracts import CredentialReference


class ProviderProfileError(ValueError):
    """Raised when provider profile public configuration is invalid."""


class ProviderKind(str, Enum):
    GGUF = "gguf"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    OPENAI_COMPATIBLE = "openai_compatible"


class ProviderCapability(str, Enum):
    CONFIGURE_TRANSLATION = "configure_translation"
    TRANSLATION_TRANSPORT = "translation_transport"
    CONNECTION_TEST = "connection_test"
    MODEL_LISTING = "model_listing"
    LOCAL_MODEL_PATH = "local_model_path"
    CREDENTIAL_REFERENCE = "credential_reference"
    GENERATION_PARAMETERS = "generation_parameters"


class ProviderTestStatus(str, Enum):
    NOT_TESTED = "not_tested"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


_DEFAULT_CAPABILITIES: dict[ProviderKind, tuple[ProviderCapability, ...]] = {
    ProviderKind.GGUF: (
        ProviderCapability.CONFIGURE_TRANSLATION,
        ProviderCapability.TRANSLATION_TRANSPORT,
        ProviderCapability.LOCAL_MODEL_PATH,
        ProviderCapability.GENERATION_PARAMETERS,
    ),
    ProviderKind.OLLAMA: (
        ProviderCapability.CONFIGURE_TRANSLATION,
        ProviderCapability.TRANSLATION_TRANSPORT,
        ProviderCapability.CONNECTION_TEST,
        ProviderCapability.MODEL_LISTING,
        ProviderCapability.GENERATION_PARAMETERS,
    ),
    ProviderKind.DEEPSEEK: (
        ProviderCapability.CONFIGURE_TRANSLATION,
        ProviderCapability.TRANSLATION_TRANSPORT,
        ProviderCapability.CONNECTION_TEST,
        ProviderCapability.CREDENTIAL_REFERENCE,
        ProviderCapability.GENERATION_PARAMETERS,
    ),
    ProviderKind.OPENAI_COMPATIBLE: (
        ProviderCapability.CONFIGURE_TRANSLATION,
        ProviderCapability.CREDENTIAL_REFERENCE,
        ProviderCapability.GENERATION_PARAMETERS,
    ),
}

_SECRET_FIELD_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|bearer|"
    r"authorization|password|passwd|secret)(?:$|[_-])",
    flags=re.IGNORECASE,
)


def default_capabilities(kind: ProviderKind) -> tuple[ProviderCapability, ...]:
    """Return fixed capabilities for the current application implementation."""

    return _DEFAULT_CAPABILITIES[ProviderKind(kind)]


def _clean_required(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ProviderProfileError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ProviderProfileError(f"{field} must not be empty")
    if any(ord(character) < 32 for character in cleaned):
        raise ProviderProfileError(f"{field} must not contain control characters")
    return cleaned


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderProfileError("optional provider text must be a string")
    cleaned = value.strip()
    if any(ord(character) < 32 for character in cleaned):
        raise ProviderProfileError("optional provider text must not contain control characters")
    return cleaned or None


def _strict_int(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderProfileError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ProviderProfileError(f"{field} must be between {minimum} and {maximum}")
    return value


def _strict_float(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderProfileError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProviderProfileError(f"{field} must be finite")
    below_minimum = result < minimum if minimum_inclusive else result <= minimum
    if below_minimum or result > maximum:
        qualifier = "at least" if minimum_inclusive else "greater than"
        raise ProviderProfileError(
            f"{field} must be {qualifier} {minimum} and at most {maximum}"
        )
    return result


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    required: frozenset[str],
    field: str,
) -> None:
    if any(not isinstance(key, str) for key in payload):
        raise ProviderProfileError(f"{field} keys must be strings")
    actual = frozenset(payload)
    missing = required - actual
    unexpected = actual - required
    if missing:
        raise ProviderProfileError(f"{field} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ProviderProfileError(
            f"{field} has unsupported fields: {sorted(unexpected)}"
        )


def _require_sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise ProviderProfileError(f"{field} must be a list")
    return value


def _require_optional_string(value: object, *, field: str) -> str | None:
    try:
        return _clean_optional(value)  # type: ignore[arg-type]
    except ProviderProfileError as exc:
        raise ProviderProfileError(f"{field} must be a string or null") from exc


def _reject_public_secret_fields(value: object, *, field: str = "provider store") -> None:
    """Reject secret-shaped public fields before decoding or persistence."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ProviderProfileError(f"{field} keys must be strings")
            if _SECRET_FIELD_PATTERN.search(raw_key):
                raise ProviderProfileError(
                    f"{field} contains forbidden secret field {raw_key!r}"
                )
            _reject_public_secret_fields(item, field=f"{field}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_public_secret_fields(item, field=f"{field}[{index}]")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value {value}")


def _configuration_fingerprint(payload: Mapping[str, object]) -> str:
    """Fingerprint one exact, secret-free provider configuration."""

    _reject_public_secret_fields(payload, field="provider configuration")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_endpoint(endpoint: str | None, *, kind: ProviderKind) -> str | None:
    cleaned = _clean_optional(endpoint)
    if cleaned is None:
        return None
    if any(character.isspace() for character in cleaned) or "\\" in cleaned:
        raise ProviderProfileError("endpoint must not contain whitespace or backslashes")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ProviderProfileError("endpoint must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ProviderProfileError("endpoint must not contain embedded credentials")
    if parsed.query or parsed.fragment or parsed.params:
        raise ProviderProfileError("endpoint must not contain query, fragment, or parameters")
    try:
        parsed.port
    except ValueError as exc:
        raise ProviderProfileError("endpoint contains an invalid port") from exc
    if kind in {ProviderKind.DEEPSEEK, ProviderKind.OPENAI_COMPATIBLE}:
        if parsed.scheme != "https":
            raise ProviderProfileError(f"{kind.value} endpoint must use HTTPS")
    elif kind is ProviderKind.OLLAMA:
        loopback = _is_loopback_host(parsed.hostname)
        if (loopback and parsed.scheme != "http") or (
            not loopback and parsed.scheme != "https"
        ):
            raise ProviderProfileError(
                "Ollama endpoint must use HTTP on a loopback host or HTTPS on "
                "a remote host"
            )
    elif kind is ProviderKind.GGUF:
        raise ProviderProfileError("GGUF profile does not accept an endpoint")
    return cleaned.rstrip("/")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        temperature = _strict_float(
            self.temperature,
            field="temperature",
            minimum=0.0,
            maximum=2.0,
        )
        top_p = _strict_float(
            self.top_p,
            field="top_p",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        max_output_tokens = self.max_output_tokens
        if max_output_tokens is not None:
            max_output_tokens = _strict_int(
                max_output_tokens,
                field="max_output_tokens",
                minimum=1,
                maximum=1_000_000,
            )
        raw_stops = _require_sequence(self.stop_sequences, field="stop_sequences")
        if any(not isinstance(item, str) for item in raw_stops):
            raise ProviderProfileError("stop_sequences must contain only strings")
        stops = tuple(raw_stops)
        if any(not item for item in stops):
            raise ProviderProfileError("stop sequences must not be empty")
        if len(set(stops)) != len(stops):
            raise ProviderProfileError("stop sequences must be unique")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "max_output_tokens", max_output_tokens)
        object.__setattr__(self, "stop_sequences", stops)

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "stop_sequences": list(self.stop_sequences),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GenerationSettings":
        _require_exact_keys(
            payload,
            required=frozenset(
                {"temperature", "top_p", "max_output_tokens", "stop_sequences"}
            ),
            field="generation settings",
        )
        raw_stops = _require_sequence(
            payload["stop_sequences"], field="generation settings stop_sequences"
        )
        return cls(
            temperature=payload["temperature"],  # type: ignore[arg-type]
            top_p=payload["top_p"],  # type: ignore[arg-type]
            max_output_tokens=payload["max_output_tokens"],  # type: ignore[arg-type]
            stop_sequences=tuple(raw_stops),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModelGenerationOverride:
    """Generation override that applies to one exact model identifier."""

    model_id: str
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    ollama_context_tokens: int | None = None
    gguf_n_ctx: int | None = None
    gguf_n_gpu_layers: int | None = None
    gguf_n_threads: int | None = None
    gguf_n_batch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _clean_required(self.model_id, field="model_id"))
        if self.temperature is not None:
            object.__setattr__(
                self,
                "temperature",
                _strict_float(
                    self.temperature,
                    field="override temperature",
                    minimum=0.0,
                    maximum=2.0,
                ),
            )
        if self.top_p is not None:
            object.__setattr__(
                self,
                "top_p",
                _strict_float(
                    self.top_p,
                    field="override top_p",
                    minimum=0.0,
                    maximum=1.0,
                    minimum_inclusive=False,
                ),
            )
        if self.max_output_tokens is not None:
            object.__setattr__(
                self,
                "max_output_tokens",
                _strict_int(
                    self.max_output_tokens,
                    field="override max_output_tokens",
                    minimum=1,
                    maximum=1_000_000,
                ),
            )
        if self.stop_sequences is not None:
            raw_stops = _require_sequence(
                self.stop_sequences, field="override stop_sequences"
            )
            if any(not isinstance(item, str) for item in raw_stops):
                raise ProviderProfileError(
                    "override stop_sequences must contain only strings"
                )
            stops = tuple(raw_stops)
            if any(not item for item in stops):
                raise ProviderProfileError("override stop sequences must not be empty")
            if len(set(stops)) != len(stops):
                raise ProviderProfileError("override stop sequences must be unique")
            object.__setattr__(self, "stop_sequences", stops)
        integer_bounds = {
            "ollama_context_tokens": (512, 32768),
            "gguf_n_ctx": (512, 32768),
            "gguf_n_gpu_layers": (-1, 200),
            "gguf_n_threads": (1, 128),
            "gguf_n_batch": (64, 4096),
        }
        for field_name, (minimum, maximum) in integer_bounds.items():
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _strict_int(
                        value,
                        field=f"override {field_name}",
                        minimum=minimum,
                        maximum=maximum,
                    ),
                )
        if all(
            value is None
            for value in (
                self.temperature,
                self.top_p,
                self.max_output_tokens,
                self.stop_sequences,
                self.ollama_context_tokens,
                self.gguf_n_ctx,
                self.gguf_n_gpu_layers,
                self.gguf_n_threads,
                self.gguf_n_batch,
            )
        ):
            raise ProviderProfileError("model generation override must change at least one value")

    def apply(self, base: GenerationSettings) -> GenerationSettings:
        return GenerationSettings(
            temperature=base.temperature if self.temperature is None else self.temperature,
            top_p=base.top_p if self.top_p is None else self.top_p,
            max_output_tokens=(
                base.max_output_tokens
                if self.max_output_tokens is None
                else self.max_output_tokens
            ),
            stop_sequences=(
                base.stop_sequences if self.stop_sequences is None else self.stop_sequences
            ),
        )

    def apply_ollama_options(self, base: "OllamaProviderOptions") -> "OllamaProviderOptions":
        return OllamaProviderOptions(
            context_tokens=(
                base.context_tokens
                if self.ollama_context_tokens is None
                else self.ollama_context_tokens
            )
        )

    def apply_gguf_options(self, base: "GGUFProviderOptions") -> "GGUFProviderOptions":
        return GGUFProviderOptions(
            prompt_style=base.prompt_style,
            n_ctx=base.n_ctx if self.gguf_n_ctx is None else self.gguf_n_ctx,
            n_gpu_layers=(
                base.n_gpu_layers
                if self.gguf_n_gpu_layers is None
                else self.gguf_n_gpu_layers
            ),
            n_threads=(
                base.n_threads if self.gguf_n_threads is None else self.gguf_n_threads
            ),
            n_batch=base.n_batch if self.gguf_n_batch is None else self.gguf_n_batch,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "stop_sequences": (
                list(self.stop_sequences) if self.stop_sequences is not None else None
            ),
            "ollama_context_tokens": self.ollama_context_tokens,
            "gguf_n_ctx": self.gguf_n_ctx,
            "gguf_n_gpu_layers": self.gguf_n_gpu_layers,
            "gguf_n_threads": self.gguf_n_threads,
            "gguf_n_batch": self.gguf_n_batch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelGenerationOverride":
        _require_exact_keys(
            payload,
            required=frozenset(
                {
                    "model_id",
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "stop_sequences",
                    "ollama_context_tokens",
                    "gguf_n_ctx",
                    "gguf_n_gpu_layers",
                    "gguf_n_threads",
                    "gguf_n_batch",
                }
            ),
            field="model generation override",
        )
        raw_stops = payload["stop_sequences"]
        if raw_stops is not None:
            raw_stops = _require_sequence(
                raw_stops, field="model generation override stop_sequences"
            )
        return cls(
            model_id=payload["model_id"],  # type: ignore[arg-type]
            temperature=payload["temperature"],  # type: ignore[arg-type]
            top_p=payload["top_p"],  # type: ignore[arg-type]
            max_output_tokens=payload["max_output_tokens"],  # type: ignore[arg-type]
            stop_sequences=(
                tuple(raw_stops)
                if raw_stops is not None
                else None
            ),
            ollama_context_tokens=payload["ollama_context_tokens"],  # type: ignore[arg-type]
            gguf_n_ctx=payload["gguf_n_ctx"],  # type: ignore[arg-type]
            gguf_n_gpu_layers=payload["gguf_n_gpu_layers"],  # type: ignore[arg-type]
            gguf_n_threads=payload["gguf_n_threads"],  # type: ignore[arg-type]
            gguf_n_batch=payload["gguf_n_batch"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ProviderRequestPolicy:
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 600.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "connect_timeout_seconds",
            _strict_float(
                self.connect_timeout_seconds,
                field="connect timeout",
                minimum=0.0,
                maximum=600.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "request_timeout_seconds",
            _strict_float(
                self.request_timeout_seconds,
                field="request timeout",
                minimum=0.0,
                maximum=86_400.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_retries",
            _strict_int(self.max_retries, field="max_retries", minimum=0, maximum=100),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "connect_timeout_seconds": float(self.connect_timeout_seconds),
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "max_retries": int(self.max_retries),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProviderRequestPolicy":
        _require_exact_keys(
            payload,
            required=frozenset(
                {"connect_timeout_seconds", "request_timeout_seconds", "max_retries"}
            ),
            field="provider request policy",
        )
        return cls(
            connect_timeout_seconds=payload["connect_timeout_seconds"],  # type: ignore[arg-type]
            request_timeout_seconds=payload["request_timeout_seconds"],  # type: ignore[arg-type]
            max_retries=payload["max_retries"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GGUFProviderOptions:
    """Existing public GGUF runtime controls, validated without free-form data."""

    prompt_style: str = "sakura"
    n_ctx: int = 4096
    # Automatic; the profile stays portable while the admitted run snapshot
    # records one exact hardware-fitted layer count.
    n_gpu_layers: int = -1
    n_threads: int = 8
    n_batch: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_style, str):
            raise ProviderProfileError("GGUF prompt_style must be a string")
        if self.prompt_style not in {"sakura", "qwen", "plain"}:
            raise ProviderProfileError("unsupported GGUF prompt_style")
        object.__setattr__(
            self,
            "n_ctx",
            _strict_int(self.n_ctx, field="GGUF n_ctx", minimum=512, maximum=32768),
        )
        object.__setattr__(
            self,
            "n_gpu_layers",
            _strict_int(
                self.n_gpu_layers,
                field="GGUF n_gpu_layers",
                minimum=-1,
                maximum=200,
            ),
        )
        object.__setattr__(
            self,
            "n_threads",
            _strict_int(self.n_threads, field="GGUF n_threads", minimum=1, maximum=128),
        )
        object.__setattr__(
            self,
            "n_batch",
            _strict_int(self.n_batch, field="GGUF n_batch", minimum=64, maximum=4096),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_style": self.prompt_style,
            "n_ctx": int(self.n_ctx),
            "n_gpu_layers": int(self.n_gpu_layers),
            "n_threads": int(self.n_threads),
            "n_batch": int(self.n_batch),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GGUFProviderOptions":
        _require_exact_keys(
            payload,
            required=frozenset(
                {"prompt_style", "n_ctx", "n_gpu_layers", "n_threads", "n_batch"}
            ),
            field="GGUF provider options",
        )
        return cls(
            prompt_style=payload["prompt_style"],  # type: ignore[arg-type]
            n_ctx=payload["n_ctx"],  # type: ignore[arg-type]
            n_gpu_layers=payload["n_gpu_layers"],  # type: ignore[arg-type]
            n_threads=payload["n_threads"],  # type: ignore[arg-type]
            n_batch=payload["n_batch"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class OllamaProviderOptions:
    """Existing public Ollama context control."""

    context_tokens: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_tokens",
            _strict_int(
                self.context_tokens,
                field="Ollama context_tokens",
                minimum=512,
                maximum=32768,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {"context_tokens": int(self.context_tokens)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OllamaProviderOptions":
        _require_exact_keys(
            payload,
            required=frozenset({"context_tokens"}),
            field="Ollama provider options",
        )
        return cls(context_tokens=payload["context_tokens"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    profile_id: str
    display_name: str
    kind: ProviderKind
    endpoint: str | None = None
    model_id: str | None = None
    local_model_path: str | None = None
    credential_ref: CredentialReference | None = None
    capabilities: tuple[ProviderCapability, ...] | None = None
    request_policy: ProviderRequestPolicy = ProviderRequestPolicy()
    generation_defaults: GenerationSettings = GenerationSettings()
    model_overrides: tuple[ModelGenerationOverride, ...] = ()
    gguf_options: GGUFProviderOptions | None = None
    ollama_options: OllamaProviderOptions | None = None
    last_test_status: ProviderTestStatus = ProviderTestStatus.NOT_TESTED
    last_tested_at_utc: str | None = None
    last_tested_configuration_fingerprint: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _clean_required(self.profile_id, field="profile_id"))
        object.__setattr__(self, "display_name", _clean_required(self.display_name, field="display_name"))
        try:
            kind = ProviderKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ProviderProfileError("unsupported provider kind") from exc
        object.__setattr__(self, "kind", kind)

        if not isinstance(self.request_policy, ProviderRequestPolicy):
            raise ProviderProfileError("request_policy must be a ProviderRequestPolicy")
        if not isinstance(self.generation_defaults, GenerationSettings):
            raise ProviderProfileError("generation_defaults must be GenerationSettings")
        if self.credential_ref is not None and not isinstance(
            self.credential_ref, CredentialReference
        ):
            raise ProviderProfileError("credential_ref must be a CredentialReference")
        if self.gguf_options is not None and not isinstance(
            self.gguf_options, GGUFProviderOptions
        ):
            raise ProviderProfileError("gguf_options must be GGUFProviderOptions")
        if self.ollama_options is not None and not isinstance(
            self.ollama_options, OllamaProviderOptions
        ):
            raise ProviderProfileError("ollama_options must be OllamaProviderOptions")

        endpoint = _validate_endpoint(self.endpoint, kind=kind)
        model_id = _clean_optional(self.model_id)
        local_path = _clean_optional(self.local_model_path)
        if kind is ProviderKind.GGUF:
            if self.credential_ref is not None:
                raise ProviderProfileError("GGUF profile does not accept a credential reference")
            object.__setattr__(self, "gguf_options", self.gguf_options or GGUFProviderOptions())
            if self.ollama_options is not None:
                raise ProviderProfileError("GGUF profile does not accept Ollama options")
        else:
            if local_path is not None:
                raise ProviderProfileError(f"{kind.value} profile does not accept a local_model_path")
            if self.gguf_options is not None:
                raise ProviderProfileError(f"{kind.value} profile does not accept GGUF options")
            if kind is ProviderKind.OLLAMA:
                if self.credential_ref is not None:
                    raise ProviderProfileError(
                        "Ollama profile does not accept a credential reference"
                    )
                object.__setattr__(
                    self,
                    "ollama_options",
                    self.ollama_options or OllamaProviderOptions(),
                )
            elif self.ollama_options is not None:
                raise ProviderProfileError(f"{kind.value} profile does not accept Ollama options")

        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "local_model_path", local_path)

        expected = default_capabilities(kind)
        if self.capabilities is None:
            capabilities = expected
        else:
            raw_capabilities = _require_sequence(
                self.capabilities, field="provider capabilities"
            )
            try:
                capabilities = tuple(
                    ProviderCapability(item) for item in raw_capabilities
                )
            except (TypeError, ValueError) as exc:
                raise ProviderProfileError("unsupported provider capability") from exc
        if len(set(capabilities)) != len(capabilities):
            raise ProviderProfileError("provider capabilities must be unique")
        unexpected = sorted(set(capabilities) - set(expected), key=lambda item: item.value)
        if unexpected:
            names = ", ".join(item.value for item in unexpected)
            raise ProviderProfileError(f"unsupported capabilities for {kind.value}: {names}")
        if ProviderCapability.CONFIGURE_TRANSLATION not in capabilities:
            raise ProviderProfileError("provider must retain configure_translation capability")
        if (
            kind is ProviderKind.OPENAI_COMPATIBLE
            and ProviderCapability.TRANSLATION_TRANSPORT in capabilities
        ):
            raise ProviderProfileError(
                "OpenAI-compatible translation transport is not implemented in GUI-2"
            )
        object.__setattr__(self, "capabilities", capabilities)

        raw_overrides = _require_sequence(
            self.model_overrides, field="model_overrides"
        )
        if any(not isinstance(item, ModelGenerationOverride) for item in raw_overrides):
            raise ProviderProfileError(
                "model_overrides must contain ModelGenerationOverride values"
            )
        overrides = tuple(raw_overrides)
        override_ids = [override.model_id for override in overrides]
        if len(set(override_ids)) != len(override_ids):
            raise ProviderProfileError("model override identifiers must be unique")
        for override in overrides:
            has_ollama_options = override.ollama_context_tokens is not None
            has_gguf_options = any(
                value is not None
                for value in (
                    override.gguf_n_ctx,
                    override.gguf_n_gpu_layers,
                    override.gguf_n_threads,
                    override.gguf_n_batch,
                )
            )
            if has_ollama_options and kind is not ProviderKind.OLLAMA:
                raise ProviderProfileError(
                    "Ollama model options require an Ollama provider profile"
                )
            if has_gguf_options and kind is not ProviderKind.GGUF:
                raise ProviderProfileError(
                    "GGUF model options require a GGUF provider profile"
                )
        object.__setattr__(self, "model_overrides", overrides)

        schema_version = _strict_int(
            self.schema_version,
            field="provider profile schema_version",
            minimum=1,
            maximum=1,
        )
        if schema_version != 1:
            raise ProviderProfileError("unsupported provider profile schema version")
        object.__setattr__(self, "schema_version", schema_version)

        try:
            test_status = ProviderTestStatus(self.last_test_status)
        except (TypeError, ValueError) as exc:
            raise ProviderProfileError("unsupported provider test status") from exc
        tested_at = _clean_optional(self.last_tested_at_utc)
        if tested_at is not None:
            try:
                parsed = datetime.fromisoformat(tested_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ProviderProfileError(
                    "last_tested_at_utc must be an ISO-8601 timestamp"
                ) from exc
            if parsed.tzinfo is None:
                raise ProviderProfileError(
                    "last_tested_at_utc must include a timezone"
                )
        tested_fingerprint = _clean_optional(
            self.last_tested_configuration_fingerprint
        )
        if tested_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", tested_fingerprint
        ):
            raise ProviderProfileError(
                "last_tested_configuration_fingerprint must be a lowercase SHA-256 value"
            )

        current_fingerprint = self.public_configuration_fingerprint
        test_binding_is_current = (
            test_status is not ProviderTestStatus.NOT_TESTED
            and tested_at is not None
            and tested_fingerprint == current_fingerprint
            and (
                test_status is not ProviderTestStatus.READY
                or self.is_configured
            )
        )
        if not test_binding_is_current:
            test_status = ProviderTestStatus.NOT_TESTED
            tested_at = None
            tested_fingerprint = None
        object.__setattr__(self, "last_test_status", test_status)
        object.__setattr__(self, "last_tested_at_utc", tested_at)
        object.__setattr__(
            self,
            "last_tested_configuration_fingerprint",
            tested_fingerprint,
        )

    @property
    def transport_available(self) -> bool:
        return ProviderCapability.TRANSLATION_TRANSPORT in (self.capabilities or ())

    @property
    def requires_credential(self) -> bool:
        return self.kind is ProviderKind.DEEPSEEK

    @property
    def configuration_issues(self) -> tuple[str, ...]:
        """Return missing public configuration without probing runtime state."""

        issues: list[str] = []
        if self.kind is ProviderKind.GGUF:
            if self.local_model_path is None:
                issues.append("local_model_path_required")
        else:
            if self.endpoint is None:
                issues.append("endpoint_required")
            if self.model_id is None or (
                self.kind is ProviderKind.OLLAMA
                and self.model_id == "auto-detect"
            ):
                issues.append("model_id_required")
        if self.requires_credential and self.credential_ref is None:
            issues.append("credential_reference_required")
        return tuple(issues)

    @property
    def is_configured(self) -> bool:
        """Whether all required public fields/references are present."""

        return not self.configuration_issues

    @property
    def is_resolved(self) -> bool:
        """Compatibility alias for static configuration completeness.

        This does not resolve credentials, inspect a model path, or contact an
        endpoint. Callers must not interpret it as current runtime readiness.
        """

        return self.is_configured

    @property
    def runtime_ready(self) -> bool:
        """Return only externally recorded, timestamped readiness state."""

        return (
            self.is_configured
            and self.last_test_status is ProviderTestStatus.READY
            and self.last_tested_at_utc is not None
            and self.last_tested_configuration_fingerprint
            == self.public_configuration_fingerprint
        )

    @property
    def public_configuration_fingerprint(self) -> str:
        """Fingerprint exactly the public configuration covered by a test."""

        return _configuration_fingerprint(self._configuration_store_dict())

    def with_test_result(
        self,
        status: ProviderTestStatus,
        *,
        tested_at_utc: str,
    ) -> "ProviderProfile":
        """Bind a provider test result to this exact public configuration."""

        try:
            normalized_status = ProviderTestStatus(status)
        except (TypeError, ValueError) as exc:
            raise ProviderProfileError("unsupported provider test status") from exc
        if normalized_status is ProviderTestStatus.NOT_TESTED:
            return replace(
                self,
                last_test_status=ProviderTestStatus.NOT_TESTED,
                last_tested_at_utc=None,
                last_tested_configuration_fingerprint=None,
            )
        return replace(
            self,
            last_test_status=normalized_status,
            last_tested_at_utc=tested_at_utc,
            last_tested_configuration_fingerprint=(
                self.public_configuration_fingerprint
            ),
        )

    def generation_for_model(self, model_id: str | None = None) -> GenerationSettings:
        """Apply an override only when its model identifier matches exactly."""

        exact_model_id = _clean_optional(model_id) or self.model_id or self.local_model_path
        for override in self.model_overrides:
            if override.model_id == exact_model_id:
                return override.apply(self.generation_defaults)
        return self.generation_defaults

    def ollama_options_for_model(
        self, model_id: str | None = None
    ) -> OllamaProviderOptions | None:
        if self.ollama_options is None:
            return None
        exact_model_id = _clean_optional(model_id) or self.model_id
        for override in self.model_overrides:
            if override.model_id == exact_model_id:
                return override.apply_ollama_options(self.ollama_options)
        return self.ollama_options

    def gguf_options_for_model(
        self, model_id: str | None = None
    ) -> GGUFProviderOptions | None:
        if self.gguf_options is None:
            return None
        exact_model_id = _clean_optional(model_id) or self.local_model_path
        for override in self.model_overrides:
            if override.model_id == exact_model_id:
                return override.apply_gguf_options(self.gguf_options)
        return self.gguf_options

    def _configuration_store_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "kind": self.kind.value,
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "local_model_path": self.local_model_path,
            "credential_ref": self.credential_ref.to_dict() if self.credential_ref else None,
            "capabilities": [item.value for item in self.capabilities or ()],
            "request_policy": self.request_policy.to_dict(),
            "generation_defaults": self.generation_defaults.to_dict(),
            "model_overrides": [override.to_dict() for override in self.model_overrides],
            "gguf_options": self.gguf_options.to_dict() if self.gguf_options else None,
            "ollama_options": (
                self.ollama_options.to_dict() if self.ollama_options else None
            ),
        }

    def to_store_dict(self) -> dict[str, object]:
        """Serialize public profile state, including only an opaque secret ref."""

        payload = self._configuration_store_dict()
        payload.update({
            "last_test_status": self.last_test_status.value,
            "last_tested_at_utc": self.last_tested_at_utc,
            "last_tested_configuration_fingerprint": (
                self.last_tested_configuration_fingerprint
            ),
        })
        return payload

    def to_public_export_dict(self) -> dict[str, object]:
        """Export a portable unresolved profile without a credential reference."""

        payload = self.to_store_dict()
        payload.pop("credential_ref", None)
        payload["credential_resolution"] = "unresolved"
        payload["last_test_status"] = ProviderTestStatus.NOT_TESTED.value
        payload["last_tested_at_utc"] = None
        payload["last_tested_configuration_fingerprint"] = None
        return payload

    def without_credential(self) -> "ProviderProfile":
        return replace(self, credential_ref=None)

    @classmethod
    def from_store_dict(cls, payload: Mapping[str, object]) -> "ProviderProfile":
        _require_exact_keys(
            payload,
            required=frozenset(
                {
                    "schema_version",
                    "profile_id",
                    "display_name",
                    "kind",
                    "endpoint",
                    "model_id",
                    "local_model_path",
                    "credential_ref",
                    "capabilities",
                    "request_policy",
                    "generation_defaults",
                    "model_overrides",
                    "gguf_options",
                    "ollama_options",
                    "last_test_status",
                    "last_tested_at_utc",
                    "last_tested_configuration_fingerprint",
                }
            ),
            field="provider profile",
        )
        raw_reference = payload["credential_ref"]
        if raw_reference is not None and not isinstance(raw_reference, Mapping):
            raise ProviderProfileError(
                "provider profile credential_ref must be an object or null"
            )
        if raw_reference is not None:
            _require_exact_keys(
                raw_reference,
                required=frozenset(
                    {"schema_version", "scope", "kind", "reference", "label"}
                ),
                field="provider profile credential_ref",
            )
        raw_capabilities = _require_sequence(
            payload["capabilities"], field="provider profile capabilities"
        )
        if any(not isinstance(item, str) for item in raw_capabilities):
            raise ProviderProfileError(
                "provider profile capabilities must contain only strings"
            )
        raw_request_policy = payload["request_policy"]
        if not isinstance(raw_request_policy, Mapping):
            raise ProviderProfileError("provider profile request_policy must be an object")
        raw_generation_defaults = payload["generation_defaults"]
        if not isinstance(raw_generation_defaults, Mapping):
            raise ProviderProfileError(
                "provider profile generation_defaults must be an object"
            )
        raw_overrides = _require_sequence(
            payload["model_overrides"], field="provider profile model_overrides"
        )
        if any(not isinstance(item, Mapping) for item in raw_overrides):
            raise ProviderProfileError(
                "provider profile model_overrides must contain only objects"
            )
        raw_gguf_options = payload["gguf_options"]
        if raw_gguf_options is not None and not isinstance(raw_gguf_options, Mapping):
            raise ProviderProfileError(
                "provider profile gguf_options must be an object or null"
            )
        raw_ollama_options = payload["ollama_options"]
        if raw_ollama_options is not None and not isinstance(
            raw_ollama_options, Mapping
        ):
            raise ProviderProfileError(
                "provider profile ollama_options must be an object or null"
            )
        if not isinstance(payload["kind"], str):
            raise ProviderProfileError("provider profile kind must be a string")
        if not isinstance(payload["last_test_status"], str):
            raise ProviderProfileError(
                "provider profile last_test_status must be a string"
            )
        return cls(
            profile_id=payload["profile_id"],  # type: ignore[arg-type]
            display_name=payload["display_name"],  # type: ignore[arg-type]
            kind=payload["kind"],  # type: ignore[arg-type]
            endpoint=_require_optional_string(
                payload["endpoint"], field="provider profile endpoint"
            ),
            model_id=_require_optional_string(
                payload["model_id"], field="provider profile model_id"
            ),
            local_model_path=_require_optional_string(
                payload["local_model_path"],
                field="provider profile local_model_path",
            ),
            credential_ref=(
                credential_reference_from_dict(raw_reference)
                if raw_reference is not None
                else None
            ),
            capabilities=tuple(raw_capabilities),  # type: ignore[arg-type]
            request_policy=ProviderRequestPolicy.from_dict(raw_request_policy),
            generation_defaults=GenerationSettings.from_dict(
                raw_generation_defaults
            ),
            model_overrides=tuple(
                ModelGenerationOverride.from_dict(item)  # type: ignore[arg-type]
                for item in raw_overrides
            ),
            gguf_options=(
                GGUFProviderOptions.from_dict(raw_gguf_options)
                if raw_gguf_options is not None
                else None
            ),
            ollama_options=(
                OllamaProviderOptions.from_dict(raw_ollama_options)
                if raw_ollama_options is not None
                else None
            ),
            last_test_status=payload["last_test_status"],  # type: ignore[arg-type]
            last_tested_at_utc=_require_optional_string(
                payload["last_tested_at_utc"],
                field="provider profile last_tested_at_utc",
            ),
            last_tested_configuration_fingerprint=_require_optional_string(
                payload["last_tested_configuration_fingerprint"],
                field="provider profile last_tested_configuration_fingerprint",
            ),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )


class ProviderProfileStore:
    """Small atomic application-profile store containing no secret values."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> tuple[ProviderProfile, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProviderProfileError("provider profile store could not be read") from exc
        except ValueError as exc:
            raise ProviderProfileError("provider profile store contains invalid JSON") from exc
        return self._decode_payload(payload)

    def _decode_payload(self, payload: object) -> tuple[ProviderProfile, ...]:
        _reject_public_secret_fields(payload)
        if not isinstance(payload, Mapping):
            raise ProviderProfileError("provider profile store root must be an object")
        _require_exact_keys(
            payload,
            required=frozenset({"schema_version", "profiles"}),
            field="provider profile store",
        )
        schema_version = _strict_int(
            payload["schema_version"],
            field="provider profile store schema_version",
            minimum=self.SCHEMA_VERSION,
            maximum=self.SCHEMA_VERSION,
        )
        if schema_version != self.SCHEMA_VERSION:
            raise ProviderProfileError("unsupported provider profile store schema")
        raw_profiles = payload["profiles"]
        if not isinstance(raw_profiles, list):
            raise ProviderProfileError("provider profile store profiles must be a list")
        profiles_list: list[ProviderProfile] = []
        for index, item in enumerate(raw_profiles):
            if not isinstance(item, Mapping):
                raise ProviderProfileError(
                    f"provider profile store profiles[{index}] must be an object"
                )
            try:
                profiles_list.append(ProviderProfile.from_store_dict(item))
            except (TypeError, ValueError) as exc:
                raise ProviderProfileError(
                    f"provider profile store profiles[{index}] is invalid"
                ) from exc
        profiles = tuple(profiles_list)
        self._validate_unique_ids(profiles)
        return profiles

    def save(self, profiles: Iterable[ProviderProfile]) -> None:
        stable_profiles = tuple(profiles)
        if any(type(profile) is not ProviderProfile for profile in stable_profiles):
            raise ProviderProfileError(
                "provider profile store accepts only ProviderProfile values"
            )
        self._validate_unique_ids(stable_profiles)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "profiles": [profile.to_store_dict() for profile in stable_profiles],
        }
        _reject_public_secret_fields(payload)
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n"
            reparsed = json.loads(
                encoded,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
            reparsed_profiles = self._decode_payload(reparsed)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderProfileError(
                "provider profile store failed pre-write round-trip validation"
            ) from exc
        if reparsed_profiles != stable_profiles:
            raise ProviderProfileError(
                "provider profile store failed pre-write round-trip validation"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _validate_unique_ids(profiles: Iterable[ProviderProfile]) -> None:
        identifiers = [profile.profile_id for profile in profiles]
        if len(set(identifiers)) != len(identifiers):
            raise ProviderProfileError("provider profile identifiers must be unique")
