# -*- coding: utf-8 -*-
"""Portable Hybrid Pro theme and metric tokens.

``THEME_TOKENS`` and ``UI_METRIC_TOKENS`` intentionally mirror the accepted
JavaScript tables key-for-key.  Native-only overlay paint consumes semantic
aliases into that frozen palette rather than adding feature-local colors.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping


def _frozen(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(values))


_SHARED_ACCENT_TOKENS = {
    "accent-primary": "#2587f6",
    "accent-primary-hover": "#3c96fa",
    "status-warning": "#f1ad2f",
    "status-success": "#2fc284",
    "status-danger": "#f45a63",
    "status-cleanup": "#27c4b1",
    "violet": "#8b7cf6",
    "focus-ring": "#67adff",
}

_DARK_TOKENS = {
    "surface-app": "#0c121a",
    "surface-shell": "#111722",
    "surface-header": "#111a25",
    "surface-panel": "#17202c",
    "surface-panel-raised": "#1b2533",
    "surface-control": "#121a24",
    "surface-hover": "#202b3a",
    "surface-selected": "#1a3857",
    "surface-canvas": "#070b10",
    "surface-dock": "#0a1119",
    "surface-dock-bar": "#0f1721",
    "surface-facet": "#121c27",
    "surface-facet-accent": "#142130",
    "border-subtle": "#263545",
    "border-default": "#334253",
    "border-strong": "#46586d",
    "content-primary": "#f0f4f9",
    "content-secondary": "#c6ced9",
    "content-muted": "#94a1b2",
    "content-disabled": "#8391a3",
    "content-inverse": "#ffffff",
    "accent-primary-surface": "#102f55",
    "accent-primary-border": "#2a78c9",
    "accent-text": "#8fc7ff",
    "status-warning-surface": "#33230c",
    "status-warning-border": "#8f6517",
    "status-success-surface": "#123326",
    "status-success-border": "#25765a",
    "success-text": "#7be0b7",
    "status-danger-surface": "#3b171d",
    "status-danger-border": "#8a313a",
    "status-cleanup-surface": "#103732",
    "status-cleanup-border": "#237f74",
    "status-edit-surface": "#24220f",
    "status-edit-border": "#c18a06",
    "status-effective-surface": "#122b27",
    "status-effective-border": "#2d9b7d",
    "shadow-card": "0 7px 22px rgba(0, 0, 0, 0.16)",
    "shadow-float": "0 18px 48px rgba(0, 0, 0, 0.38)",
    **_SHARED_ACCENT_TOKENS,
}

_LIGHT_TOKENS = {
    "surface-app": "#eef2f6",
    "surface-shell": "#f4f7fa",
    "surface-header": "#f8fafc",
    "surface-panel": "#ffffff",
    "surface-panel-raised": "#f4f7fa",
    "surface-control": "#f8fafc",
    "surface-hover": "#eaf1f8",
    "surface-selected": "#dceeff",
    "surface-canvas": "#080d13",
    "surface-dock": "#e9eff5",
    "surface-dock-bar": "#f8fafc",
    "surface-facet": "#ffffff",
    "surface-facet-accent": "#f2f8ff",
    "border-subtle": "#e0e6ed",
    "border-default": "#d2dbe5",
    "border-strong": "#aebdcb",
    "content-primary": "#142235",
    "content-secondary": "#33475c",
    "content-muted": "#5b6f84",
    "content-disabled": "#78899b",
    "content-inverse": "#ffffff",
    "accent-primary-surface": "#dcecff",
    "accent-primary-border": "#5596d8",
    "accent-text": "#155f9d",
    "status-warning-surface": "#fff3d4",
    "status-warning-border": "#c58c20",
    "status-success-surface": "#e0f5eb",
    "status-success-border": "#57a887",
    "success-text": "#176b4d",
    "status-danger-surface": "#fde7e9",
    "status-danger-border": "#c96c74",
    "status-cleanup-surface": "#dcf5f1",
    "status-cleanup-border": "#4aa99e",
    "status-edit-surface": "#fff7df",
    "status-edit-border": "#c18a06",
    "status-effective-surface": "#e7f7f1",
    "status-effective-border": "#49a98b",
    "shadow-card": "0 6px 18px rgba(23, 42, 62, 0.08)",
    "shadow-float": "0 18px 48px rgba(23, 42, 62, 0.2)",
    **_SHARED_ACCENT_TOKENS,
}

THEME_IDS = ("dark", "light")
THEME_TOKENS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "dark": _frozen(_DARK_TOKENS),
        "light": _frozen(_LIGHT_TOKENS),
    }
)

UI_METRIC_TOKENS: Mapping[str, str] = _frozen(
    {
        "type-caption": "11px",
        "type-label": "12px",
        "type-body": "13px",
        "type-title-sm": "14px",
        "type-title": "16px",
        "target-compact": "32px",
        "target-default": "36px",
        "target-primary": "40px",
        "space-1": "4px",
        "space-2": "6px",
        "space-3": "8px",
        "space-4": "12px",
        "space-5": "16px",
        "radius-sm": "5px",
        "radius": "8px",
        "radius-lg": "12px",
    }
)

# Overlay paint is deliberately not a second palette.  Each role aliases an
# accepted semantic color so native canvas code never introduces local
# literals while the portable THEME_TOKENS table stays mechanically exact.
OVERLAY_TOKEN_ROLES: Mapping[str, str] = MappingProxyType(
    {
        "parent": "accent-primary",
        "render": "status-cleanup",
        "source": "status-warning",
        "baseline": "violet",
        "cleanup": "status-cleanup",
        "protected": "status-warning",
        "proof": "status-cleanup",
        "selection": "accent-primary",
    }
)
OVERLAY_ROLE_IDS = tuple(OVERLAY_TOKEN_ROLES)


def theme_token(theme: str, role: str) -> str:
    values = THEME_TOKENS.get(theme)
    if values is None:
        raise ValueError(f"Unknown theme: {theme}")
    if role not in values:
        raise ValueError(f"Unknown theme role: {role}")
    return values[role]


def metric_token(role: str) -> str:
    if role not in UI_METRIC_TOKENS:
        raise ValueError(f"Unknown UI metric role: {role}")
    return UI_METRIC_TOKENS[role]


_PIXEL_VALUE = re.compile(r"^(?P<value>\d+)px$")


def metric_pixels(role: str) -> int:
    """Return one integer logical-pixel metric, rejecting non-pixel tokens."""

    value = metric_token(role)
    match = _PIXEL_VALUE.fullmatch(value)
    if match is None:
        raise ValueError(f"UI metric is not a logical-pixel value: {role}")
    return int(match.group("value"))


def overlay_token(theme: str, role: str) -> str:
    token_role = OVERLAY_TOKEN_ROLES.get(role)
    if token_role is None:
        raise ValueError(f"Unknown overlay role: {role}")
    return theme_token(theme, token_role)


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    """Immutable value object consumed by native theme and canvas adapters."""

    theme: str
    colors: Mapping[str, str]
    metrics: Mapping[str, str] = UI_METRIC_TOKENS

    def token(self, role: str) -> str:
        return theme_token(self.theme, role)

    def metric(self, role: str) -> str:
        return metric_token(role)

    def overlay(self, role: str) -> str:
        return overlay_token(self.theme, role)


def resolve_theme(theme: str) -> ThemeTokens:
    if theme not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme}")
    return ThemeTokens(theme=theme, colors=THEME_TOKENS[theme])


__all__ = [
    "OVERLAY_ROLE_IDS",
    "OVERLAY_TOKEN_ROLES",
    "THEME_IDS",
    "THEME_TOKENS",
    "UI_METRIC_TOKENS",
    "ThemeTokens",
    "metric_pixels",
    "metric_token",
    "overlay_token",
    "resolve_theme",
    "theme_token",
]
