"""Cross-platform compute capability and executable selection policy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

from .contracts import ComputeBackend, OperatingSystem, PlatformIdentity


@dataclass(frozen=True, slots=True)
class TorchDeviceSelection:
    backend: ComputeBackend
    device: str
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class OnnxProviderSelection:
    backend: ComputeBackend
    providers: tuple[str, ...]
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class MpsMemoryFacts:
    recommended_max_bytes: int
    driver_allocated_bytes: int
    tensor_allocated_bytes: int

    @property
    def available_bytes(self) -> int:
        return max(0, self.recommended_max_bytes - self.driver_allocated_bytes)


@dataclass(frozen=True, slots=True)
class ComputeCapabilitySnapshot:
    torch: TorchDeviceSelection
    onnx: OnnxProviderSelection
    mps_memory: MpsMemoryFacts | None
    llama_server: Path | None
    llama_server_backend: ComputeBackend = ComputeBackend.CPU

    @property
    def torch_backend(self) -> ComputeBackend:
        return self.torch.backend

    @property
    def onnx_backend(self) -> ComputeBackend:
        return self.onnx.backend


def select_torch_device(
    allow_acceleration: bool,
    *,
    cuda: bool | None = None,
    mps: bool | None = None,
) -> TorchDeviceSelection:
    if cuda is None or mps is None:
        try:
            import torch

            if cuda is None:
                cuda = bool(torch.cuda.is_available())
            if mps is None:
                mps = bool(
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                )
        except Exception:
            cuda = bool(cuda)
            mps = bool(mps)
    if not allow_acceleration:
        return TorchDeviceSelection(ComputeBackend.CPU, "cpu")
    if cuda:
        return TorchDeviceSelection(ComputeBackend.CUDA, "cuda")
    if mps:
        return TorchDeviceSelection(ComputeBackend.MPS, "mps")
    return TorchDeviceSelection(
        ComputeBackend.CPU,
        "cpu",
        "accelerator_unavailable",
    )


def select_onnx_providers(
    allow_acceleration: bool,
    available_providers: Sequence[str] | None = None,
) -> OnnxProviderSelection:
    if available_providers is None:
        try:
            import onnxruntime as ort

            available_providers = tuple(str(item) for item in ort.get_available_providers())
        except Exception:
            available_providers = ()
    available = tuple(dict.fromkeys(str(item) for item in available_providers))
    if allow_acceleration and "CUDAExecutionProvider" in available:
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        return OnnxProviderSelection(ComputeBackend.CUDA, providers)
    if allow_acceleration and "CoreMLExecutionProvider" in available:
        providers = ("CoreMLExecutionProvider", "CPUExecutionProvider")
        return OnnxProviderSelection(ComputeBackend.COREML, providers)
    reason = "" if not allow_acceleration else "accelerator_provider_unavailable"
    return OnnxProviderSelection(
        ComputeBackend.CPU,
        ("CPUExecutionProvider",),
        reason,
    )


def probe_mps_memory(torch_module=None) -> MpsMemoryFacts | None:
    try:
        torch = torch_module
        if torch is None:
            import torch
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return None
        return MpsMemoryFacts(
            recommended_max_bytes=max(0, int(torch.mps.recommended_max_memory())),
            driver_allocated_bytes=max(0, int(torch.mps.driver_allocated_memory())),
            tensor_allocated_bytes=max(0, int(torch.mps.current_allocated_memory())),
        )
    except Exception:
        return None


def release_torch_memory(
    torch_module=None,
    *,
    synchronize: bool = False,
) -> ComputeBackend:
    """Release the active Torch accelerator cache without assuming CUDA."""

    try:
        torch = torch_module
        if torch is None:
            import torch
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()
            return ComputeBackend.CUDA
        if bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            empty_cache = getattr(torch.mps, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            if synchronize:
                synchronize_mps = getattr(torch.mps, "synchronize", None)
                if callable(synchronize_mps):
                    synchronize_mps()
            return ComputeBackend.MPS
    except Exception:
        return ComputeBackend.CPU
    return ComputeBackend.CPU


def llama_backend_from_device_listing(output: str) -> ComputeBackend:
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip().casefold()
        device_id = line.split(":", 1)[0]
        if re.fullmatch(r"cuda\d+", device_id):
            return ComputeBackend.CUDA
        if re.fullmatch(r"mtl\d+", device_id):
            return ComputeBackend.METAL
    return ComputeBackend.CPU


def probe_llama_server_backend(
    executable: str | os.PathLike[str] | None,
    *,
    runner=subprocess.run,
) -> ComputeBackend:
    if not executable:
        return ComputeBackend.CPU
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = runner(
            [str(executable), "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return ComputeBackend.CPU
    if int(getattr(result, "returncode", 1)) != 0:
        return ComputeBackend.CPU
    return llama_backend_from_device_listing(
        f"{result.stdout or ''}\n{result.stderr or ''}"
    )


def llama_cpp_backend_from_capability(
    supports_gpu_offload: bool,
    identity: PlatformIdentity,
    *,
    cuda_available: bool,
) -> ComputeBackend:
    if not supports_gpu_offload:
        return ComputeBackend.CPU
    if identity.os is OperatingSystem.MACOS:
        return ComputeBackend.METAL
    if identity.os is OperatingSystem.WINDOWS and cuda_available:
        return ComputeBackend.CUDA
    return ComputeBackend.CPU


def probe_llama_cpp_python_backend(
    identity: PlatformIdentity | None = None,
    *,
    cuda_available: bool = False,
    module=None,
) -> ComputeBackend:
    selected = identity or PlatformIdentity.detect()
    try:
        llama_cpp = module
        if llama_cpp is None:
            import llama_cpp
        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        return llama_cpp_backend_from_capability(
            bool(callable(supports) and supports()),
            selected,
            cuda_available=cuda_available,
        )
    except Exception:
        return ComputeBackend.CPU


def llama_server_is_launchable(
    executable: str | os.PathLike[str] | None,
    *,
    runner=subprocess.run,
) -> bool:
    if not executable:
        return False
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = runner(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{getattr(result, 'stdout', '') or ''}\n{getattr(result, 'stderr', '') or ''}"
    return int(getattr(result, "returncode", 1)) == 0 and bool(output.strip())


def _candidate_is_executable(path: Path, identity: PlatformIdentity) -> bool:
    if not path.is_file():
        return False
    if identity.os is OperatingSystem.WINDOWS:
        return path.suffix.casefold() == ".exe"
    return os.access(path, os.X_OK)


def _llama_names(identity: PlatformIdentity) -> tuple[str, ...]:
    if identity.os is OperatingSystem.WINDOWS:
        return ("llama-server.exe",)
    if identity.os is OperatingSystem.MACOS:
        return ("llama-server",)
    return ("llama-server", "llama-server.exe")


def resolve_llama_server(
    *,
    identity: PlatformIdentity | None = None,
    override: str | os.PathLike[str] | None = None,
    search_roots: Iterable[str | os.PathLike[str]] = (),
    path_environment: str | None = None,
    environment_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> Path | None:
    selected = identity or PlatformIdentity.detect()
    if override:
        candidate = Path(override).expanduser().resolve()
        if _candidate_is_executable(candidate, selected):
            return candidate

    environment_path = os.environ.get("PATH", "") if path_environment is None else path_environment
    for name in _llama_names(selected):
        resolved = shutil.which(name, path=environment_path)
        if resolved:
            candidate = Path(resolved).resolve()
            if _candidate_is_executable(candidate, selected):
                return candidate

    base_roots = (
        (
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "Scripts",
        )
        if environment_roots is None
        else tuple(Path(root).expanduser() for root in environment_roots)
    )
    roots = (*base_roots, *(Path(root).expanduser() for root in search_roots))
    names = set(_llama_names(selected))
    for root in roots:
        if not root.is_dir():
            continue
        candidates: list[Path] = []
        for candidate in root.rglob("*"):
            if candidate.name in names and _candidate_is_executable(candidate, selected):
                candidates.append(candidate.resolve())
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                "cuda" not in item.as_posix().casefold()
                if selected.os is OperatingSystem.WINDOWS
                else False,
                len(item.parts),
                item.as_posix().casefold(),
            )
        )
        return candidates[0]
    return None


def detect_compute_capabilities(
    identity: PlatformIdentity | None = None,
) -> ComputeCapabilitySnapshot:
    selected = identity or PlatformIdentity.detect()
    torch_selection = select_torch_device(True)
    onnx_selection = select_onnx_providers(True)
    llama_server = resolve_llama_server(identity=selected)
    return ComputeCapabilitySnapshot(
        torch=torch_selection,
        onnx=onnx_selection,
        mps_memory=probe_mps_memory(),
        llama_server=llama_server,
        llama_server_backend=probe_llama_server_backend(llama_server),
    )


__all__ = [
    "ComputeCapabilitySnapshot",
    "MpsMemoryFacts",
    "OnnxProviderSelection",
    "TorchDeviceSelection",
    "detect_compute_capabilities",
    "llama_backend_from_device_listing",
    "llama_cpp_backend_from_capability",
    "llama_server_is_launchable",
    "probe_mps_memory",
    "probe_llama_cpp_python_backend",
    "probe_llama_server_backend",
    "release_torch_memory",
    "resolve_llama_server",
    "select_onnx_providers",
    "select_torch_device",
]
