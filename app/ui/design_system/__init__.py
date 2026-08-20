# -*- coding: utf-8 -*-
"""Semantic native design-system foundation for the GUI-5 shell."""

from .components import (
    CommandButton,
    EmptyState,
    SectionHeader,
    SemanticProperties,
    StatusPill,
    apply_semantic_properties,
)
from .dialogs import (
    HybridConfirmDialog,
    HybridDialog,
    HybridDialogHeader,
    HybridTextInputDialog,
)
from .theme import (
    ThemeOptions,
    apply_application_theme,
    build_application_stylesheet,
    build_qpalette,
)


def __getattr__(name: str):
    """Load Qt-backed delegates only when a caller explicitly requests one."""

    if name in {"PageRailDelegate", "ProjectCardDelegate"}:
        from .delegates import PageRailDelegate, ProjectCardDelegate

        value = {
            "PageRailDelegate": PageRailDelegate,
            "ProjectCardDelegate": ProjectCardDelegate,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(name)
from .tokens import (
    OVERLAY_ROLE_IDS,
    OVERLAY_TOKEN_ROLES,
    THEME_IDS,
    THEME_TOKENS,
    UI_METRIC_TOKENS,
    ThemeTokens,
    metric_pixels,
    metric_token,
    overlay_token,
    resolve_theme,
    theme_token,
)

__all__ = [
    "CommandButton",
    "EmptyState",
    "HybridConfirmDialog",
    "HybridDialog",
    "HybridDialogHeader",
    "HybridTextInputDialog",
    "OVERLAY_ROLE_IDS",
    "OVERLAY_TOKEN_ROLES",
    "SectionHeader",
    "PageRailDelegate",
    "ProjectCardDelegate",
    "SemanticProperties",
    "StatusPill",
    "THEME_IDS",
    "THEME_TOKENS",
    "UI_METRIC_TOKENS",
    "ThemeOptions",
    "ThemeTokens",
    "apply_application_theme",
    "apply_semantic_properties",
    "build_application_stylesheet",
    "build_qpalette",
    "metric_pixels",
    "metric_token",
    "overlay_token",
    "resolve_theme",
    "theme_token",
]
