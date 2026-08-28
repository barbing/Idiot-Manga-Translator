"""Platform-selected secure credential persistence and resolution."""

from __future__ import annotations

from typing import Any

from app.config.credential_store import (
    CompositeCredentialResolver,
    CredentialResolver,
    CredentialStore,
    CredentialStoreError,
    EnvironmentCredentialResolver,
    WindowsCredentialStore,
)
from app.config.settings_contracts import (
    CredentialReference,
    CredentialReferenceKind,
)

from .contracts import OperatingSystem, PlatformIdentity


def canonical_credential_target(value: str) -> str:
    if not isinstance(value, str):
        raise CredentialStoreError("credential target must be a string")
    target = value.strip().replace("\\", "/")
    parts = target.split("/")
    if (
        not target
        or any(ord(character) < 32 for character in target)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise CredentialStoreError("credential target is invalid")
    return "/".join(parts)


def system_keyring_reference(
    target_name: str,
    *,
    label: str = "macOS Keychain",
) -> CredentialReference:
    return CredentialReference(
        kind=CredentialReferenceKind.SYSTEM_KEYRING,
        reference=canonical_credential_target(target_name),
        label=label,
    )


class KeyringCredentialStore:
    """Store opaque provider credentials in the active system keyring."""

    SERVICE = "YomiFrame"

    def __init__(self, *, backend: Any | None = None) -> None:
        if backend is None:
            import keyring as backend
        self._backend = backend

    @staticmethod
    def _require_secret(secret: str) -> str:
        if not isinstance(secret, str) or not secret:
            raise CredentialStoreError("credential value must not be empty")
        return secret

    def store(self, target_name: str, secret: str) -> CredentialReference:
        target = canonical_credential_target(target_name)
        value = self._require_secret(secret)
        try:
            self._backend.set_password(self.SERVICE, target, value)
        except Exception as exc:
            raise CredentialStoreError(
                f"system credential store write failed ({type(exc).__name__})"
            ) from exc
        return system_keyring_reference(target)

    def resolve(self, reference: CredentialReference) -> str | None:
        if reference.kind is not CredentialReferenceKind.SYSTEM_KEYRING:
            return None
        try:
            return self._backend.get_password(self.SERVICE, reference.reference)
        except Exception as exc:
            raise CredentialStoreError(
                f"system credential store read failed ({type(exc).__name__})"
            ) from exc

    def delete(self, reference: CredentialReference) -> bool:
        if reference.kind is not CredentialReferenceKind.SYSTEM_KEYRING:
            return False
        delete_error = getattr(
            getattr(self._backend, "errors", object()),
            "PasswordDeleteError",
            (),
        )
        try:
            self._backend.delete_password(self.SERVICE, reference.reference)
        except Exception as exc:
            if delete_error and isinstance(exc, delete_error):
                return False
            raise CredentialStoreError(
                f"system credential store deletion failed ({type(exc).__name__})"
            ) from exc
        return True


def build_credential_store(
    identity: PlatformIdentity | None = None,
    *,
    keyring_backend: Any | None = None,
) -> CredentialStore:
    selected = identity or PlatformIdentity.detect()
    if selected.os is OperatingSystem.WINDOWS:
        return WindowsCredentialStore()
    if selected.os is OperatingSystem.MACOS:
        return KeyringCredentialStore(backend=keyring_backend)
    raise CredentialStoreError(
        f"credential persistence is unsupported on {selected.os.value}"
    )


def build_credential_resolver(
    identity: PlatformIdentity | None = None,
    *,
    keyring_backend: Any | None = None,
) -> CredentialResolver:
    selected = identity or PlatformIdentity.detect()
    return CompositeCredentialResolver(
        environment=EnvironmentCredentialResolver(),
        windows=WindowsCredentialStore(),
        system=(
            KeyringCredentialStore(backend=keyring_backend)
            if selected.os is OperatingSystem.MACOS
            else None
        ),
    )


def credential_store_label(identity: PlatformIdentity | None = None) -> str:
    selected = identity or PlatformIdentity.detect()
    if selected.os is OperatingSystem.WINDOWS:
        return "Windows Credential Manager"
    if selected.os is OperatingSystem.MACOS:
        return "macOS Keychain"
    return "system credential store"


__all__ = [
    "KeyringCredentialStore",
    "build_credential_resolver",
    "build_credential_store",
    "canonical_credential_target",
    "credential_store_label",
    "system_keyring_reference",
]
