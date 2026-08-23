# -*- coding: utf-8 -*-
"""Hybrid Pro action icons and the canonical YomiFrame application mark."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6 import QtGui
import qtawesome as qta

from .tokens import theme_token


_ICON_NAMES = {
    "hub": "fa5s.home",
    "workspace": "fa5s.briefcase",
    "editor": "fa5s.pen",
    "settings": "fa5s.cog",
    "undo": "fa5s.undo-alt",
    "redo": "fa5s.redo-alt",
    "theme": "fa5s.sun",
    "theme-dark": "fa5s.moon",
    "minimize": "fa5s.minus",
    "maximize": "fa5s.square",
    "close": "fa5s.times",
    "open": "fa5s.folder-open",
    "new": "fa5s.plus",
    "search": "fa5s.search",
    "warning": "fa5s.exclamation-triangle",
    "success": "fa5s.check-circle",
    "provider": "fa5s.database",
    "provider-gguf": "fa5s.microchip",
    "provider-ollama": "fa5s.server",
    "provider-cloud": "fa5s.cloud",
    "provider-compatible": "fa5s.plug",
    "general": "fa5s.cog",
    "appearance": "fa5s.palette",
    "providers": "fa5s.key",
    "modules": "fa5s.sliders-h",
    "runtime": "fa5s.microchip",
    "glossary": "fa5s.book-open",
    "shortcuts": "fa5s.keyboard",
    "caret-right": "fa5s.chevron-right",
    "caret-left": "fa5s.chevron-left",
    "caret-down": "fa5s.chevron-down",
    "caret-up": "fa5s.chevron-up",
    "grid": "fa5s.th-large",
    "list": "fa5s.list-ul",
    "history": "fa5s.history",
    "cleanup": "fa5s.paint-brush",
    "project-scope": "fa5s.database",
    "status-muted": "fa5s.circle",
    "status-ready": "fa5s.circle",
    "status-editing": "fa5s.circle",
    "status-queued": "fa5s.circle",
    "shield": "fa5s.shield-alt",
    "play": "fa5s.play",
    "stop": "fa5s.stop-circle",
    "check": "fa5s.check",
    "file-text": "fa5s.file-alt",
    "translate": "fa5s.language",
    "clock": "fa5s.clock",
    "arrow-right": "fa5s.arrow-right",
    "filter": "fa5s.filter",
    "more": "fa5s.ellipsis-h",
    "select": "fa5s.crop-alt",
    "lasso": "fa5s.draw-polygon",
    "rectangle": "fa5s.vector-square",
    "eraser": "fa5s.eraser",
    "pan": "fa5s.hand-paper",
    "zoom-out": "fa5s.search-minus",
    "zoom-in": "fa5s.search-plus",
    "overlays": "fa5s.vector-square",
    "fullscreen": "fa5s.expand-arrows-alt",
    "sidebar": "fa5s.columns",
    "eye": "fa5s.eye",
}

_BRAND_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "branding"
_BRAND_ICON_PATH = _BRAND_ASSET_DIR / "yomiframe.ico"


def brand_icon_path() -> Path:
    """Return the source and frozen-runtime path to the canonical Windows icon."""

    return _BRAND_ICON_PATH


@lru_cache(maxsize=1)
def brand_icon() -> QtGui.QIcon:
    """Load the reviewed multi-size YomiFrame icon without recoloring it."""

    icon = QtGui.QIcon(str(brand_icon_path()))
    if icon.isNull():
        raise RuntimeError(
            f"Missing or unreadable YomiFrame application icon: {brand_icon_path()}"
        )
    return icon


@lru_cache(maxsize=64)
def hybrid_icon(
    name: str,
    theme: str = "dark",
    *,
    active: bool = False,
    accent: bool = False,
    secondary: bool = False,
) -> QtGui.QIcon:
    """Return one real library icon tinted from the Hybrid Pro palette."""

    if str(name) == "brand":
        return brand_icon()
    icon_name = _ICON_NAMES.get(str(name))
    if icon_name is None:
        raise ValueError(f"Unknown Hybrid Pro icon: {name!r}")
    if sum((bool(active), bool(accent), bool(secondary))) > 1:
        raise ValueError("Only one Hybrid Pro icon color state may be selected")
    semantic_role = {
        "success": "success-text",
        "shield": "success-text",
        "status-ready": "success-text",
        "status-editing": "accent-primary",
        "status-queued": "content-muted",
        "stop": "status-danger",
    }.get(str(name), "content-muted")
    color_role = (
        "accent-primary"
        if accent
        else "content-inverse"
        if active
        else "content-secondary"
        if secondary
        else semantic_role
    )
    color = theme_token(theme, color_role)
    disabled = theme_token(theme, "content-disabled")
    return qta.icon(icon_name, color=color, color_disabled=disabled)


__all__ = ["brand_icon", "brand_icon_path", "hybrid_icon"]
