# -*- coding: utf-8 -*-
"""Opaque credential references and credential-store adapters.

Provider profiles persist :class:`CredentialReference` values only. Secret
material is accepted transiently by a credential store or returned to the
caller that explicitly resolves a reference; it is never part of the
serializable reference contract.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from app.config.settings_contracts import (
    CredentialReference,
    CredentialReferenceKind,
    SettingsScope,
)


class CredentialReferenceError(ValueError):
    """Raised when an opaque credential reference is malformed."""


class CredentialStoreError(RuntimeError):
    """Raised when the platform credential store cannot complete an action."""


def _require_reference_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise CredentialReferenceError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise CredentialReferenceError(f"{field} must not be empty")
    if any(ord(character) < 32 for character in normalized):
        raise CredentialReferenceError(f"{field} must not contain control characters")
    return normalized


def windows_credential_reference(target_name: str, *, label: str = "") -> CredentialReference:
    """Create an opaque Windows credential reference."""

    try:
        return CredentialReference(
            kind=CredentialReferenceKind.WINDOWS_CREDENTIAL,
            reference=_require_reference_identifier(target_name, field="target_name"),
            label=label,
        )
    except (TypeError, ValueError) as exc:
        raise CredentialReferenceError("Windows credential reference is invalid") from exc


def environment_credential_reference(
    variable_name: str,
    *,
    label: str = "",
) -> CredentialReference:
    """Create an explicit environment-variable credential reference."""

    try:
        return CredentialReference(
            kind=CredentialReferenceKind.ENVIRONMENT_VARIABLE,
            reference=variable_name,
            label=label,
        )
    except (TypeError, ValueError) as exc:
        raise CredentialReferenceError("environment credential reference is invalid") from exc


def credential_reference_from_dict(payload: Mapping[str, object]) -> CredentialReference:
    """Parse the canonical credential-reference persistence contract."""

    required = {"schema_version", "scope", "kind", "reference", "label"}
    if any(not isinstance(key, str) for key in payload):
        raise CredentialReferenceError("credential reference keys must be strings")
    missing = sorted(required - set(payload))
    unexpected = sorted(str(key) for key in payload.keys() if key not in required)
    if unexpected:
        raise CredentialReferenceError(
            f"unsupported credential reference fields: {', '.join(unexpected)}"
        )
    if missing:
        raise CredentialReferenceError(
            f"credential reference is missing fields: {', '.join(missing)}"
        )
    string_fields = ("schema_version", "scope", "kind", "reference", "label")
    if any(not isinstance(payload[field], str) for field in string_fields):
        raise CredentialReferenceError("credential reference fields must be strings")
    schema_version = payload["schema_version"]
    scope = payload["scope"]
    if scope != SettingsScope.CREDENTIAL.value:
        raise CredentialReferenceError("credential reference has an invalid settings scope")
    try:
        return CredentialReference(
            kind=CredentialReferenceKind(payload["kind"]),
            reference=payload["reference"],
            label=payload["label"],
            schema_version=schema_version,
        )
    except (TypeError, ValueError) as exc:
        raise CredentialReferenceError("credential reference is invalid") from exc


@runtime_checkable
class CredentialResolver(Protocol):
    """Read-only credential resolution seam used by run configuration."""

    def resolve(self, reference: CredentialReference) -> str | None:
        """Resolve ``reference`` or return ``None`` when it is unavailable."""


@runtime_checkable
class CredentialStore(CredentialResolver, Protocol):
    """Mutable store for explicit, user-approved secret persistence."""

    def store(self, target_name: str, secret: str) -> CredentialReference:
        """Persist ``secret`` and return its opaque reference."""

    def delete(self, reference: CredentialReference) -> bool:
        """Delete a stored secret, returning whether it existed."""


class EnvironmentCredentialResolver:
    """Resolve explicit environment-variable references.

    Tests and embedding applications can inject a mapping. When no mapping is
    supplied, the process environment is consulted lazily at resolution time.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ

    def resolve(self, reference: CredentialReference) -> str | None:
        if reference.kind is not CredentialReferenceKind.ENVIRONMENT_VARIABLE:
            return None
        source = os.environ if self._environ is None else self._environ
        value = source.get(reference.reference)
        return value if value else None


_LEGACY_DEEPSEEK_KEY_FIELDS = ("DEEPSEEK_API_KEY", "API_KEY", "api_key", "key")
_LEGACY_DEEPSEEK_PROVIDER_FIELDS = ("Deepseek", "DeepSeek", "deepseek")
_MAX_LEGACY_CREDENTIAL_BYTES = 64 * 1024


def _legacy_credential_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip("\"'").strip()


def _legacy_deepseek_key_from_payload(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in _LEGACY_DEEPSEEK_KEY_FIELDS:
        value = _legacy_credential_text(payload.get(key))
        if value:
            return value
    for provider_key in _LEGACY_DEEPSEEK_PROVIDER_FIELDS:
        provider = payload.get(provider_key)
        if isinstance(provider, Mapping):
            for key in _LEGACY_DEEPSEEK_KEY_FIELDS:
                value = _legacy_credential_text(provider.get(key))
                if value:
                    return value
        else:
            value = _legacy_credential_text(provider)
            if value:
                return value
    return ""


def _parse_legacy_deepseek_key(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return _legacy_deepseek_key_from_payload(parsed)
    for line in clean.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#") or "=" not in candidate:
            continue
        key, raw_value = candidate.split("=", 1)
        if key.strip() in {"DEEPSEEK_API_KEY", "API_KEY"}:
            value = _legacy_credential_text(raw_value)
            if value:
                return value
    return _legacy_credential_text(clean)


def resolve_legacy_deepseek_credential(
    *,
    application_root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve the pre-GUI-7 key only during an explicit migration test.

    The caller must keep the value transient and offer secure persistence only
    after a successful provider test. No file path or secret becomes a
    serializable credential reference.
    """

    root = (
        Path(application_root).expanduser()
        if application_root is not None
        else Path.cwd()
    )
    api_root = (root / "api").resolve()
    for filename in ("API_KEY", "API_KEY.json"):
        candidate = api_root / filename
        try:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(api_root):
                continue
            if not resolved.is_file():
                continue
            if resolved.stat().st_size > _MAX_LEGACY_CREDENTIAL_BYTES:
                continue
            secret = _parse_legacy_deepseek_key(
                resolved.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError):
            continue
        if secret:
            return secret
    return None


class CompositeCredentialResolver:
    """Dispatch references to resolvers without probing unrelated backends."""

    def __init__(
        self,
        *,
        environment: CredentialResolver | None = None,
        windows: CredentialResolver | None = None,
    ) -> None:
        self._environment = environment
        self._windows = windows

    def resolve(self, reference: CredentialReference) -> str | None:
        if reference.kind is CredentialReferenceKind.ENVIRONMENT_VARIABLE:
            return self._environment.resolve(reference) if self._environment else None
        if reference.kind is CredentialReferenceKind.WINDOWS_CREDENTIAL:
            return self._windows.resolve(reference) if self._windows else None
        return None


if os.name == "nt":
    class _CredentialAttributeW(ctypes.Structure):
        _fields_ = [
            ("Keyword", wintypes.LPWSTR),
            ("Flags", wintypes.DWORD),
            ("ValueSize", wintypes.DWORD),
            ("Value", ctypes.POINTER(wintypes.BYTE)),
        ]


    class _CredentialW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(wintypes.BYTE)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.POINTER(_CredentialAttributeW)),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]
else:
    _CredentialW = object  # type: ignore[assignment,misc]


class WindowsCredentialStore:
    """Minimal Windows Credential Manager adapter for generic credentials.

    Constructing this adapter does not read or write the credential store.
    Calls fail explicitly on non-Windows platforms. The adapter never includes
    secret material in its exceptions.
    """

    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2
    _ERROR_NOT_FOUND = 1168
    _YOMIFRAME_TARGET_PREFIX = "YomiFrame/"

    def __init__(self, *, target_prefix: str = "YomiFrame/") -> None:
        self._target_prefix = self._canonical_target_prefix(target_prefix)
        self._advapi32 = None
        if os.name == "nt":
            library = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
            credential_pointer = ctypes.POINTER(_CredentialW)
            library.CredWriteW.argtypes = [credential_pointer, wintypes.DWORD]
            library.CredWriteW.restype = wintypes.BOOL
            library.CredReadW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(credential_pointer),
            ]
            library.CredReadW.restype = wintypes.BOOL
            library.CredDeleteW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            library.CredDeleteW.restype = wintypes.BOOL
            library.CredFree.argtypes = [ctypes.c_void_p]
            library.CredFree.restype = None
            self._advapi32 = library

    @classmethod
    def _canonical_target_prefix(cls, target_prefix: str) -> str:
        prefix = _require_reference_identifier(
            target_prefix, field="target_prefix"
        ).replace("\\", "/")
        parts = prefix.rstrip("/").split("/")
        if (
            not parts
            or parts[0] != cls._YOMIFRAME_TARGET_PREFIX.rstrip("/")
            or any(not part or part in {".", ".."} for part in parts)
        ):
            raise CredentialReferenceError(
                "Windows credential target_prefix must remain in the YomiFrame namespace"
            )
        return "/".join(parts) + "/"

    def _require_windows(self) -> object:
        if self._advapi32 is None:
            raise CredentialStoreError("Windows Credential Manager is unavailable")
        return self._advapi32

    def _target_name(self, name: str) -> str:
        clean_name = _require_reference_identifier(
            name, field="target_name"
        ).replace("\\", "/")
        if clean_name.startswith(self._target_prefix):
            suffix = clean_name[len(self._target_prefix) :]
        elif clean_name.startswith(self._YOMIFRAME_TARGET_PREFIX):
            raise CredentialReferenceError(
                "Windows credential target_name is outside the configured YomiFrame component"
            )
        else:
            suffix = clean_name
        if not suffix or any(part in {"", ".", ".."} for part in suffix.split("/")):
            raise CredentialReferenceError(
                "Windows credential target_name must identify a child component"
            )
        return f"{self._target_prefix}{suffix}"

    def _owned_reference_target(self, reference: CredentialReference) -> str:
        target = _require_reference_identifier(
            reference.reference, field="credential reference"
        ).replace("\\", "/")
        if not target.startswith(self._target_prefix):
            raise CredentialStoreError(
                "credential reference is outside the YomiFrame namespace"
            )
        suffix = target[len(self._target_prefix) :]
        if not suffix or any(part in {"", ".", ".."} for part in suffix.split("/")):
            raise CredentialStoreError(
                "credential reference is outside the YomiFrame namespace"
            )
        return target

    def store(self, target_name: str, secret: str) -> CredentialReference:
        library = self._require_windows()
        target = self._target_name(target_name)
        if not isinstance(secret, str) or not secret:
            raise CredentialStoreError("credential value must not be empty")

        encoded = secret.encode("utf-16-le")
        blob = (wintypes.BYTE * len(encoded)).from_buffer_copy(encoded)
        credential = _CredentialW()
        credential.Type = self._CRED_TYPE_GENERIC
        credential.TargetName = target
        credential.CredentialBlobSize = len(encoded)
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(wintypes.BYTE))
        credential.Persist = self._CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "YomiFrame"
        try:
            if not library.CredWriteW(ctypes.byref(credential), 0):
                code = ctypes.get_last_error()
                raise CredentialStoreError(f"credential store write failed (Windows error {code})")
        finally:
            ctypes.memset(ctypes.addressof(blob), 0, len(encoded))
        return windows_credential_reference(target)

    def resolve(self, reference: CredentialReference) -> str | None:
        if reference.kind is not CredentialReferenceKind.WINDOWS_CREDENTIAL:
            return None
        target = self._owned_reference_target(reference)
        library = self._require_windows()
        credential_pointer = ctypes.POINTER(_CredentialW)()
        if not library.CredReadW(
            target,
            self._CRED_TYPE_GENERIC,
            0,
            ctypes.byref(credential_pointer),
        ):
            code = ctypes.get_last_error()
            if code == self._ERROR_NOT_FOUND:
                return None
            raise CredentialStoreError(f"credential store read failed (Windows error {code})")
        try:
            credential = credential_pointer.contents
            if not credential.CredentialBlob or not credential.CredentialBlobSize:
                return None
            encoded = ctypes.string_at(
                credential.CredentialBlob,
                credential.CredentialBlobSize,
            )
            return encoded.decode("utf-16-le")
        except UnicodeDecodeError as exc:
            raise CredentialStoreError("stored credential has an invalid encoding") from exc
        finally:
            library.CredFree(credential_pointer)

    def delete(self, reference: CredentialReference) -> bool:
        if reference.kind is not CredentialReferenceKind.WINDOWS_CREDENTIAL:
            return False
        target = self._owned_reference_target(reference)
        library = self._require_windows()
        if library.CredDeleteW(target, self._CRED_TYPE_GENERIC, 0):
            return True
        code = ctypes.get_last_error()
        if code == self._ERROR_NOT_FOUND:
            return False
        raise CredentialStoreError(f"credential store deletion failed (Windows error {code})")
