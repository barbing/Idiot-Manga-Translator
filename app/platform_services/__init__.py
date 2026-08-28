"""Cross-platform services shared by YomiFrame application owners."""

from dataclasses import dataclass
from typing import Any

from app.config.credential_store import CredentialResolver, CredentialStore

from .contracts import ComputeBackend, OperatingSystem, PlatformIdentity
from .credentials import (
    KeyringCredentialStore,
    build_credential_resolver,
    build_credential_store,
    credential_store_label,
)
from .compute import (
    ComputeCapabilitySnapshot,
    MpsMemoryFacts,
    OnnxProviderSelection,
    TorchDeviceSelection,
    detect_compute_capabilities,
    llama_backend_from_device_listing,
    llama_cpp_backend_from_capability,
    llama_server_is_launchable,
    probe_llama_cpp_python_backend,
    probe_llama_server_backend,
    release_torch_memory,
    resolve_llama_server,
    select_onnx_providers,
    select_torch_device,
)
from .paths import PlatformPaths, StandardRoots, qt_platform_paths
from .runtime_assets import (
    RuntimeAssetSpec,
    RuntimeAssetTarget,
    paddle_targets,
    runtime_asset_catalog,
    runtime_asset_spec,
)


@dataclass(frozen=True, slots=True)
class PlatformServices:
    identity: PlatformIdentity
    paths: PlatformPaths
    credential_store: CredentialStore
    credential_resolver: CredentialResolver
    compute: ComputeCapabilitySnapshot
    runtime_assets: tuple[RuntimeAssetSpec, ...]


def build_platform_services(
    identity: PlatformIdentity | None = None,
    *,
    roots: StandardRoots | None = None,
    compute: ComputeCapabilitySnapshot | None = None,
    keyring: Any | None = None,
) -> PlatformServices:
    selected = identity or PlatformIdentity.detect()
    if not isinstance(selected, PlatformIdentity):
        raise TypeError("identity must be PlatformIdentity")
    if roots is not None and not isinstance(roots, StandardRoots):
        raise TypeError("roots must be StandardRoots")
    if compute is not None and not isinstance(compute, ComputeCapabilitySnapshot):
        raise TypeError("compute must be ComputeCapabilitySnapshot")
    paths = PlatformPaths.from_roots(roots) if roots is not None else qt_platform_paths()
    return PlatformServices(
        identity=selected,
        paths=paths,
        credential_store=build_credential_store(
            selected,
            keyring_backend=keyring,
        ),
        credential_resolver=build_credential_resolver(
            selected,
            keyring_backend=keyring,
        ),
        compute=compute or detect_compute_capabilities(selected),
        runtime_assets=runtime_asset_catalog(selected),
    )

__all__ = [
    "ComputeBackend",
    "ComputeCapabilitySnapshot",
    "OperatingSystem",
    "KeyringCredentialStore",
    "MpsMemoryFacts",
    "OnnxProviderSelection",
    "PlatformIdentity",
    "PlatformPaths",
    "PlatformServices",
    "RuntimeAssetSpec",
    "RuntimeAssetTarget",
    "StandardRoots",
    "TorchDeviceSelection",
    "build_credential_resolver",
    "build_credential_store",
    "build_platform_services",
    "credential_store_label",
    "detect_compute_capabilities",
    "llama_backend_from_device_listing",
    "llama_cpp_backend_from_capability",
    "llama_server_is_launchable",
    "release_torch_memory",
    "qt_platform_paths",
    "paddle_targets",
    "probe_llama_cpp_python_backend",
    "probe_llama_server_backend",
    "resolve_llama_server",
    "runtime_asset_catalog",
    "runtime_asset_spec",
    "select_onnx_providers",
    "select_torch_device",
]
