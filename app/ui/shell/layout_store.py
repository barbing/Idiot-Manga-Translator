# -*- coding: utf-8 -*-
"""Versioned native layout persistence with responsive clamping."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from app.ui.ui_contract import (
    ACTIVITY_FACET_IDS,
    clamp_activity_dock_height,
    resolve_activity_dock_bounds,
    resolve_layout_mode,
)


LAYOUT_SCHEMA_VERSION = 1
_ACTIVITY_TABS = ("overview", "history", "warnings", "cleanup")


def _positive_sizes(values: Iterable[object], field_name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} values must be integers")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} values must be integers") from exc
        if number < 0 or number > 32768:
            raise ValueError(f"{field_name} values must be between 0 and 32768")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class WorkspaceLayoutState:
    """Portable part of the shell layout; Qt geometry bytes remain optional."""

    theme: str = "dark"
    density: str = "comfortable"
    font_scale: int = 100
    reduced_motion: bool = True
    editor_splitter_sizes: tuple[int, ...] = (184, 820, 368)
    editor_vertical_sizes: tuple[int, ...] = (580, 320)
    activity_height: int = 324
    activity_tab: str = "overview"
    activity_expanded: bool = True
    inspector_tab: str = "text"
    canvas_mode: str = "original"
    window_geometry: bytes = b""
    window_state: bytes = b""
    schema_version: int = LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LAYOUT_SCHEMA_VERSION:
            raise ValueError("unsupported workspace layout schema")
        if self.theme not in {"dark", "light"}:
            raise ValueError(f"unsupported theme: {self.theme!r}")
        if self.density not in {"comfortable", "compact"}:
            raise ValueError(f"unsupported density: {self.density!r}")
        if not 100 <= self.font_scale <= 200 or self.font_scale % 5:
            raise ValueError(f"unsupported font scale: {self.font_scale!r}")
        if not isinstance(self.reduced_motion, bool):
            raise TypeError("reduced_motion must be bool")
        object.__setattr__(
            self,
            "editor_splitter_sizes",
            _positive_sizes(self.editor_splitter_sizes, "editor_splitter_sizes"),
        )
        object.__setattr__(
            self,
            "editor_vertical_sizes",
            _positive_sizes(self.editor_vertical_sizes, "editor_vertical_sizes"),
        )
        if isinstance(self.activity_height, bool) or not isinstance(
            self.activity_height, int
        ):
            raise TypeError("activity_height must be int")
        if self.activity_tab not in _ACTIVITY_TABS:
            raise ValueError(f"unsupported Activity tab: {self.activity_tab!r}")
        if not isinstance(self.activity_expanded, bool):
            raise TypeError("activity_expanded must be bool")
        if self.inspector_tab not in {"text", "style", "layout", "cleanup", "history"}:
            raise ValueError(f"unsupported inspector tab: {self.inspector_tab!r}")
        if self.canvas_mode not in {"original", "cleaned", "final", "compare"}:
            raise ValueError(f"unsupported canvas mode: {self.canvas_mode!r}")
        for field_name in ("window_geometry", "window_state"):
            value = getattr(self, field_name)
            if not isinstance(value, bytes):
                raise TypeError(f"{field_name} must be bytes")

    def normalized(self, *, width: int, height: int) -> "WorkspaceLayoutState":
        mode = resolve_layout_mode(
            width=width,
            height=height,
            font_scale=self.font_scale,
            density=self.density,
        )
        clamped = clamp_activity_dock_height(self.activity_height, mode)
        return replace(self, activity_height=clamped)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "theme": self.theme,
            "density": self.density,
            "font_scale": self.font_scale,
            "reduced_motion": self.reduced_motion,
            "editor_splitter_sizes": list(self.editor_splitter_sizes),
            "editor_vertical_sizes": list(self.editor_vertical_sizes),
            "activity_height": self.activity_height,
            "activity_tab": self.activity_tab,
            "activity_expanded": self.activity_expanded,
            "inspector_tab": self.inspector_tab,
            "canvas_mode": self.canvas_mode,
            "window_geometry": self.window_geometry,
            "window_state": self.window_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WorkspaceLayoutState":
        if not isinstance(payload, dict):
            raise TypeError("layout payload must be a dict")
        allowed = {
            "schema_version",
            "theme",
            "density",
            "font_scale",
            "reduced_motion",
            "editor_splitter_sizes",
            "editor_vertical_sizes",
            "activity_height",
            "activity_tab",
            "activity_expanded",
            "inspector_tab",
            "canvas_mode",
            "window_geometry",
            "window_state",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown workspace layout fields: {sorted(unknown)}")
        kwargs = dict(payload)
        for field_name in ("window_geometry", "window_state"):
            value = kwargs.get(field_name, b"")
            if hasattr(value, "data"):
                value = bytes(value.data())
            elif isinstance(value, bytearray):
                value = bytes(value)
            kwargs[field_name] = value
        return cls(**kwargs)  # type: ignore[arg-type]


class QtLayoutStore:
    """Small QSettings adapter; no feature state or project data is persisted."""

    def __init__(self, settings: object | None = None) -> None:
        if settings is None:
            from PySide6 import QtCore

            settings = QtCore.QSettings("YomiFrame", "Translator")
        for method in ("beginGroup", "endGroup", "value", "setValue", "sync"):
            if not hasattr(settings, method):
                raise TypeError("settings must provide the QSettings interface")
        self._settings = settings

    def load(self) -> WorkspaceLayoutState:
        self._settings.beginGroup("gui5/layout")
        try:
            schema = int(self._settings.value("schema_version", LAYOUT_SCHEMA_VERSION))
            if schema != LAYOUT_SCHEMA_VERSION:
                return WorkspaceLayoutState()
            payload = {
                "schema_version": schema,
                "theme": str(self._settings.value("theme", "dark")),
                "density": str(self._settings.value("density", "comfortable")),
                "font_scale": int(self._settings.value("font_scale", 100)),
                "reduced_motion": self._bool("reduced_motion", True),
                "editor_splitter_sizes": self._list(
                    "editor_splitter_sizes", (184, 820, 368)
                ),
                "editor_vertical_sizes": self._list(
                    "editor_vertical_sizes", (580, 320)
                ),
                "activity_height": int(self._settings.value("activity_height", 324)),
                "activity_tab": str(self._settings.value("activity_tab", "overview")),
                "activity_expanded": self._bool("activity_expanded", True),
                "inspector_tab": str(self._settings.value("inspector_tab", "text")),
                "canvas_mode": str(self._settings.value("canvas_mode", "original")),
                "window_geometry": self._bytes("window_geometry"),
                "window_state": self._bytes("window_state"),
            }
            try:
                return WorkspaceLayoutState.from_dict(payload)
            except (TypeError, ValueError):
                return WorkspaceLayoutState()
        finally:
            self._settings.endGroup()

    def save(self, state: WorkspaceLayoutState) -> None:
        if not isinstance(state, WorkspaceLayoutState):
            raise TypeError("state must be WorkspaceLayoutState")
        self._settings.beginGroup("gui5/layout")
        try:
            for key, value in state.to_dict().items():
                self._settings.setValue(key, value)
        finally:
            self._settings.endGroup()
        self._settings.sync()

    def reset(self) -> WorkspaceLayoutState:
        state = WorkspaceLayoutState()
        self.save(state)
        return state

    def _bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().casefold() in {"1", "true", "yes", "on"}

    def _list(self, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
        value = self._settings.value(key, list(default))
        if isinstance(value, str):
            value = [part for part in value.split(",") if part.strip()]
        if not isinstance(value, (list, tuple)):
            return default
        try:
            return _positive_sizes(value, key)
        except (TypeError, ValueError):
            return default

    def _bytes(self, key: str) -> bytes:
        value = self._settings.value(key, b"")
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if hasattr(value, "data"):
            return bytes(value.data())
        return b""


def clamp_window_rect(
    rect: tuple[int, int, int, int],
    available_rects: Iterable[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Return a visible window rect while preserving useful size."""

    x, y, width, height = (int(value) for value in rect)
    width = max(960, width)
    height = max(640, height)
    screens = tuple(available_rects)
    if not screens:
        return (x, y, width, height)
    for sx, sy, sw, sh in screens:
        if x + width > sx and y + height > sy and x < sx + sw and y < sy + sh:
            return (
                min(max(x, sx), sx + max(0, sw - width)),
                min(max(y, sy), sy + max(0, sh - height)),
                min(width, sw),
                min(height, sh),
            )
    sx, sy, sw, sh = screens[0]
    return (sx, sy, min(width, sw), min(height, sh))
