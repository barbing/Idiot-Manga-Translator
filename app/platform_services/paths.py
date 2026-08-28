"""Platform-correct application paths for source and Conda execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StandardRoots:
    data: Path
    cache: Path
    documents: Path


@dataclass(frozen=True, slots=True)
class PlatformPaths:
    data_root: Path
    config_root: Path
    runtime_root: Path
    cache_root: Path
    default_project_root: Path

    @classmethod
    def from_roots(cls, roots: StandardRoots) -> "PlatformPaths":
        data_root = Path(roots.data).expanduser() / "YomiFrame"
        return cls(
            data_root=data_root,
            config_root=data_root / "config",
            runtime_root=data_root / "runtime",
            cache_root=Path(roots.cache).expanduser() / "YomiFrame",
            default_project_root=(
                Path(roots.documents).expanduser() / "YomiFrame Projects"
            ),
        )


def qt_standard_roots() -> StandardRoots:
    from PySide6 import QtCore

    location = QtCore.QStandardPaths.StandardLocation

    def required_path(kind: QtCore.QStandardPaths.StandardLocation) -> Path:
        value = QtCore.QStandardPaths.writableLocation(kind)
        if not value:
            raise RuntimeError(f"Qt standard location is unavailable: {kind.name}")
        return Path(value)

    return StandardRoots(
        data=required_path(location.GenericDataLocation),
        cache=required_path(location.GenericCacheLocation),
        documents=required_path(location.DocumentsLocation),
    )


def qt_platform_paths() -> PlatformPaths:
    return PlatformPaths.from_roots(qt_standard_roots())


__all__ = [
    "PlatformPaths",
    "StandardRoots",
    "qt_platform_paths",
    "qt_standard_roots",
]
