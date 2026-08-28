"""Immutable platform identity and backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import platform
import sys


class OperatingSystem(str, Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    OTHER = "other"


class ComputeBackend(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    COREML = "coreml"
    METAL = "metal"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class PlatformIdentity:
    os: OperatingSystem
    architecture: str
    frozen: bool

    @classmethod
    def from_values(
        cls,
        sys_platform: str,
        architecture: str,
        frozen: bool,
    ) -> "PlatformIdentity":
        operating_system = {
            "win32": OperatingSystem.WINDOWS,
            "darwin": OperatingSystem.MACOS,
        }.get(str(sys_platform or "").strip().lower(), OperatingSystem.OTHER)
        normalized_architecture = str(architecture or "").strip() or "unknown"
        return cls(
            os=operating_system,
            architecture=normalized_architecture,
            frozen=bool(frozen),
        )

    @classmethod
    def detect(cls) -> "PlatformIdentity":
        return cls.from_values(
            sys.platform,
            platform.machine(),
            bool(getattr(sys, "frozen", False)),
        )


__all__ = ["ComputeBackend", "OperatingSystem", "PlatformIdentity"]
