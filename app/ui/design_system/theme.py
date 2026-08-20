# -*- coding: utf-8 -*-
"""Lazy PySide6 adapter for the semantic Hybrid Pro design tokens."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from app.ui.ui_contract import DENSITY_IDS

from .tokens import metric_pixels, metric_token, resolve_theme, theme_token


_PIXEL_FONT_SIZE = re.compile(
    r"(?P<prefix>font-size:\s*)(?P<pixels>[0-9]+(?:\.[0-9]+)?)px\b"
)


def _font_sizes_in_points(stylesheet: str) -> str:
    """Keep QSS typography sized without producing pixel-font QFont objects."""

    def replace(match: re.Match[str]) -> str:
        points = float(match.group("pixels")) * 0.75
        return f"{match.group('prefix')}{points:g}pt"

    return _PIXEL_FONT_SIZE.sub(replace, stylesheet)


def _scaled_type_metric(role: str, font_scale: int) -> str:
    """Scale a frozen ``Npx`` typography token for the application setting."""

    value = metric_token(role)
    if not value.endswith("px"):
        raise ValueError(f"Typography token {role!r} must use px units")
    try:
        pixels = float(value[:-2])
    except ValueError as exc:
        raise ValueError(f"Typography token {role!r} is invalid") from exc
    return f"{max(1, round(pixels * font_scale / 100.0))}px"


@dataclass(frozen=True, slots=True)
class ThemeOptions:
    theme: str = "dark"
    density: str = "comfortable"
    font_scale: int = 100
    reduced_motion: bool = True

    def __post_init__(self) -> None:
        resolve_theme(self.theme)
        if self.density not in DENSITY_IDS:
            raise ValueError(f"Unknown density: {self.density}")
        if isinstance(self.font_scale, bool) or not isinstance(
            self.font_scale, int
        ):
            raise TypeError("font_scale must be an integer percentage")
        if not 100 <= self.font_scale <= 200 or self.font_scale % 5:
            raise ValueError("font_scale must be between 100 and 200 in steps of 5")
        if not isinstance(self.reduced_motion, bool):
            raise TypeError("reduced_motion must be a boolean")


def build_qpalette(theme: str) -> Any:
    """Build a ``QPalette`` without importing PySide6 at module import time."""

    from PySide6.QtGui import QColor, QPalette

    tokens = resolve_theme(theme)
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "surface-app",
        QPalette.ColorRole.WindowText: "content-primary",
        QPalette.ColorRole.Base: "surface-control",
        QPalette.ColorRole.AlternateBase: "surface-panel-raised",
        QPalette.ColorRole.ToolTipBase: "surface-panel-raised",
        QPalette.ColorRole.ToolTipText: "content-primary",
        QPalette.ColorRole.Text: "content-primary",
        QPalette.ColorRole.Button: "surface-control",
        QPalette.ColorRole.ButtonText: "content-primary",
        QPalette.ColorRole.BrightText: "content-inverse",
        QPalette.ColorRole.Highlight: "accent-primary",
        QPalette.ColorRole.HighlightedText: "content-inverse",
        QPalette.ColorRole.PlaceholderText: "content-muted",
    }
    for role, token_role in roles.items():
        palette.setColor(role, QColor(tokens.token(token_role)))
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(tokens.token("content-disabled")),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(tokens.token("content-disabled")),
    )
    return palette


def build_application_stylesheet(options: ThemeOptions) -> str:
    """Return compact semantic QSS for roles Qt does not expose via palette."""

    if not isinstance(options, ThemeOptions):
        raise TypeError("options must be ThemeOptions")
    theme = options.theme
    target = metric_pixels(
        "target-compact" if options.density == "compact" else "target-default"
    )
    primary_target = metric_pixels("target-primary")
    radius_sm = metric_pixels("radius-sm")
    radius = metric_pixels("radius")
    space_2 = metric_pixels("space-2")
    space_3 = metric_pixels("space-3")
    space_4 = metric_pixels("space-4")
    space_5 = metric_pixels("space-5")
    radius_lg = metric_pixels("radius-lg")
    # Role selectors override the application font in QSS, so they must carry
    # the same user scale explicitly instead of freezing labels at 100%.
    type_caption = _scaled_type_metric("type-caption", options.font_scale)
    type_label = _scaled_type_metric("type-label", options.font_scale)
    type_body = _scaled_type_metric("type-body", options.font_scale)
    type_title = _scaled_type_metric("type-title", options.font_scale)
    c = lambda role: theme_token(theme, role)
    workspace_run_background = (
        "#17253a" if theme == "dark" else c("accent-primary-surface")
    )
    workspace_run_border = (
        "#365d89" if theme == "dark" else c("accent-primary-border")
    )
    info_callout_background = (
        "#102239" if theme == "dark" else c("accent-primary-surface")
    )
    info_callout_border = (
        "#284c76" if theme == "dark" else c("accent-primary-border")
    )
    info_callout_text = "#b9d9fb" if theme == "dark" else c("accent-text")
    appearance_preview_canvas = "#080d13" if theme == "dark" else "#d9e3ec"
    appearance_preview_canvas_text = (
        c("content-muted") if theme == "dark" else "#496176"
    )

    stylesheet = f"""
QMainWindow, QDialog {{ background: {c('surface-app')}; color: {c('content-primary')}; }}
QWidget[role="shell"] {{ background: {c('surface-shell')}; color: {c('content-primary')}; }}
QWidget[role="header"] {{ background: {c('surface-header')}; border-bottom: 1px solid {c('border-subtle')}; }}
QWidget[role="panel"] {{ background: {c('surface-panel')}; border: 1px solid {c('border-default')}; border-radius: {radius}px; }}
QWidget[role="panel-raised"] {{ background: {c('surface-panel-raised')}; border: 1px solid {c('border-default')}; border-radius: {radius}px; }}
QFrame#authorityCard[authority="user"] {{ background: {c('status-edit-surface')}; border-color: {c('status-edit-border')}; }}
QFrame#authorityCard[authority="effective"] {{ background: {c('status-effective-surface')}; border-color: {c('status-effective-border')}; }}
QFrame#authorityCard[authority="user"] QLabel[role="eyebrow"] {{ color: {c('status-warning')}; }}
QFrame#authorityCard[authority="effective"] QLabel[role="eyebrow"] {{ color: {c('status-success')}; }}
QFrame#inspectorFooter {{ border-top: 1px solid {c('border-default')}; background: {c('surface-header')}; }}
QWidget[role="dock"] {{ background: {c('surface-dock')}; border-top: 1px solid {c('border-default')}; }}
QWidget[role="dock-bar"] {{ background: {c('surface-dock-bar')}; border-bottom: 1px solid {c('border-subtle')}; }}
QWidget[role="facet"] {{ background: {c('surface-facet')}; border: 1px solid {c('border-default')}; border-radius: {radius}px; }}
QWidget[role="facet-accent"] {{ background: {c('surface-facet-accent')}; border: 1px solid {c('accent-primary-border')}; border-radius: {radius}px; }}
QWidget[role="canvas"] {{ background: {c('surface-canvas')}; }}
QFrame[role="status-banner"], QLabel[role="status-banner"] {{ padding: {space_2}px; border: 1px solid {info_callout_border}; border-radius: {radius_sm}px; background: {info_callout_background}; color: {info_callout_text}; }}
QFrame[role="status-banner"][tone="warning"], QLabel[role="status-banner"][tone="warning"] {{ border-color: {c('status-warning-border')}; background: {c('status-warning-surface')}; color: {c('content-primary')}; }}
QFrame[role="status-banner"][tone="error"], QLabel[role="status-banner"][tone="error"] {{ border-color: {c('status-danger-border')}; background: {c('status-danger-surface')}; color: {c('status-danger')}; }}
QFrame[role="status-banner"][tone="ready"], QLabel[role="status-banner"][tone="ready"] {{ border-color: {c('status-success-border')}; background: {c('status-success-surface')}; color: {c('success-text')}; }}
QFrame[role="status-banner"][tone="muted"], QLabel[role="status-banner"][tone="muted"] {{ border-color: {c('border-default')}; background: {c('surface-panel-raised')}; color: {c('content-secondary')}; }}
QFrame[role="state-callout"] {{ padding: {space_2}px; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; background: {c('surface-panel-raised')}; color: {c('content-primary')}; }}
QFrame[role="state-callout"][tone="ready"] {{ border-color: {c('status-success-border')}; background: {c('status-success-surface')}; color: {c('success-text')}; }}
QFrame[role="state-callout"][tone="warning"] {{ border-color: {c('status-warning-border')}; background: {c('status-warning-surface')}; color: {c('content-primary')}; }}
QFrame[role="state-callout"][tone="error"] {{ border-color: {c('status-danger-border')}; background: {c('status-danger-surface')}; color: {c('status-danger')}; }}
QLabel[role="secondary"] {{ color: {c('content-secondary')}; }}
QLabel[role="muted"] {{ color: {c('content-muted')}; }}
QLabel[role="title"], QLabel#surfaceTitle {{ color: {c('content-primary')}; font-size: {type_title}; font-weight: 700; }}
QLabel[role="eyebrow"] {{ color: {c('content-muted')}; font-size: {type_caption}; font-weight: 700; }}
QLabel[role="section"] {{ color: {c('content-primary')}; font-size: {type_label}; font-weight: 700; }}
QLabel[role="metric"] {{ color: {c('content-primary')}; font-size: {type_body}; font-weight: 800; }}
QFrame[role="status-pill"], QLabel[role="status-pill"] {{ min-height: 24px; padding: 0 {space_2}px; border-radius: 12px; }}
QLabel[role="status-pill"] {{ max-height: 24px; }}
QToolButton[role="status-pill"] {{ min-height: 24px; padding: 0 {space_2}px; border-radius: 12px; }}
QFrame[role="status-pill"][tone="ready"], QLabel[role="status-pill"][tone="ready"] {{ background: {c('status-success-surface')}; border: 1px solid {c('status-success-border')}; color: {c('success-text')}; }}
QToolButton[role="status-pill"][tone="ready"] {{ background: {c('status-success-surface')}; border: 1px solid {c('status-success-border')}; color: {c('success-text')}; }}
QFrame[role="status-pill"][tone="editing"], QFrame[role="status-pill"][tone="info"], QLabel[role="status-pill"][tone="editing"], QLabel[role="status-pill"][tone="info"] {{ background: {c('accent-primary-surface')}; border: 1px solid {c('accent-primary-border')}; color: {c('accent-text')}; }}
QFrame[role="status-pill"][tone="warning"], QLabel[role="status-pill"][tone="warning"] {{ background: {c('status-warning-surface')}; border: 1px solid {c('status-warning-border')}; color: {c('content-primary')}; }}
QToolButton[role="status-pill"][tone="warning"] {{ background: {c('status-warning-surface')}; border: 1px solid {c('status-warning-border')}; color: {c('content-primary')}; }}
QFrame[role="status-pill"][tone="error"], QLabel[role="status-pill"][tone="error"] {{ background: {c('status-danger-surface')}; border: 1px solid {c('status-danger-border')}; color: {c('status-danger')}; }}
QToolButton[role="status-pill"][tone="error"] {{ background: {c('status-danger-surface')}; border: 1px solid {c('status-danger-border')}; color: {c('status-danger')}; }}
QFrame[role="status-pill"][tone="muted"], QFrame[role="status-pill"][tone="queued"], QLabel[role="status-pill"][tone="muted"], QLabel[role="status-pill"][tone="queued"] {{ background: {c('surface-control')}; border: 1px solid {c('border-default')}; color: {c('content-secondary')}; }}
QToolButton[role="status-pill"][tone="muted"], QToolButton[role="status-pill"][tone="queued"] {{ background: {c('surface-control')}; border: 1px solid {c('border-default')}; color: {c('content-secondary')}; }}
QAbstractButton[role="command"] {{ min-height: {target}px; padding: 0 {space_3}px; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; background: {c('surface-control')}; color: {c('content-primary')}; }}
QAbstractButton[role="command"]:hover {{ background: {c('surface-hover')}; border-color: {c('border-strong')}; }}
QAbstractButton[role="command"]:focus {{ border: 2px solid {c('focus-ring')}; }}
QAbstractButton[role="command"]:checked {{ background: {c('surface-selected')}; border-color: {c('accent-primary-border')}; color: {c('accent-text')}; }}
QAbstractButton[role="command"][variant="primary"] {{ min-height: {primary_target}px; background: {c('accent-primary')}; border-color: {c('accent-primary')}; color: {c('content-inverse')}; font-weight: 700; }}
QAbstractButton[role="command"][variant="quiet"] {{ background: transparent; border-color: transparent; color: {c('accent-text')}; }}
QAbstractButton[role="command"][tone="danger"] {{ background: {c('status-danger-surface')}; border-color: {c('status-danger-border')}; color: {c('status-danger')}; }}
QAbstractButton[role="command"]:disabled {{ color: {c('content-disabled')}; background: {c('surface-control')}; border-color: {c('border-subtle')}; }}
#sourceRerunLink, #targetRerunLink {{ min-height: 26px; max-height: 26px; padding: 0 4px; border: 0; background: transparent; color: {c('accent-text')}; }}
#sourceRerunLink:hover, #targetRerunLink:hover {{ border: 0; background: transparent; color: {c('content-primary')}; }}
QWidget[role="empty-state"] {{ padding: {space_4}px; border: 1px dashed {c('border-default')}; border-radius: {radius}px; background: {c('surface-panel-raised')}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{ min-height: {target}px; padding: 0 {space_2}px; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; background: {c('surface-control')}; color: {c('content-primary')}; selection-background-color: {c('surface-selected')}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 2px solid {c('focus-ring')}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border: 0; background: transparent; }}
QComboBox[hybridChevron="true"]::down-arrow {{ image: none; width: 0px; height: 0px; }}
QListView, QTreeView, QTableView, QListWidget, QTreeWidget {{ background: {c('surface-panel')}; alternate-background-color: {c('surface-panel-raised')}; color: {c('content-primary')}; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; outline: none; padding: {space_2}px; }}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{ background: {c('surface-selected')}; color: {c('content-primary')}; }}
QHeaderView::section {{ background: {c('surface-panel-raised')}; color: {c('content-secondary')}; border: 0; border-bottom: 1px solid {c('border-default')}; padding: {space_2}px; font-weight: 700; }}
QTableWidget[role="runtime-grid"] {{ padding: 0; font-size: {type_caption}; }}
QTabWidget::pane {{ border: 1px solid {c('border-default')}; background: {c('surface-panel')}; }}
QTabBar::tab {{ min-height: {target}px; padding: 0 {space_4}px; color: {c('content-secondary')}; background: {c('surface-panel')}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {c('accent-text')}; border-bottom-color: {c('accent-primary')}; background: {c('surface-panel-raised')}; }}
QProgressBar {{ min-height: 8px; max-height: 8px; border: 0; border-radius: 4px; background: {c('surface-control')}; color: transparent; }}
QProgressBar::chunk {{ border-radius: 4px; background: {c('accent-primary')}; }}
QSplitter::handle {{ background: {c('border-subtle')}; }}
QSplitter::handle:hover {{ background: {c('accent-primary-border')}; }}
QScrollBar:horizontal {{ height: 10px; margin: 1px; background: transparent; }}
QScrollBar:vertical {{ width: 10px; margin: 1px; background: transparent; }}
QScrollBar::handle {{ min-width: 28px; min-height: 28px; border-radius: 4px; background: {c('border-strong')}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QMenu {{ background: {c('surface-panel-raised')}; color: {c('content-primary')}; border: 1px solid {c('border-default')}; padding: {space_2}px; }}
QMenu::item {{ min-height: {target}px; padding: 0 {space_5}px; border-radius: {radius_sm}px; }}
QMenu::item:selected {{ background: {c('surface-selected')}; }}
QStatusBar {{ background: {c('surface-header')}; color: {c('content-muted')}; border-top: 1px solid {c('border-subtle')}; }}
QLabel#stageStep {{ min-height: {target}px; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; background: {c('surface-control')}; color: {c('content-muted')}; }}
QLabel#stageStep[state="active"] {{ background: {c('accent-primary-surface')}; border-color: {c('accent-primary-border')}; color: {c('accent-text')}; font-weight: 700; }}
QLabel#stageStep[state="complete"] {{ background: {c('status-success-surface')}; border-color: {c('status-success-border')}; color: {c('success-text')}; }}
QPushButton#projectActionCard {{ min-height: 84px; border-radius: {radius_lg}px; text-align: left; padding: {space_4}px {space_5}px; }}
#projectHub {{ background: {c('surface-app')}; }}
#projectHubScroll, #projectHubScrollContent, #workspaceScroll, #workspaceScrollContent {{ background: {c('surface-app')}; border: 0; }}
#projectHubScroll QScrollBar:vertical, #workspaceScroll QScrollBar:vertical {{ width: 16px; margin: 1px; }}
#projectHub QLabel#surfaceTitle {{ font-size: 34px; }}
#hubCallout {{ border: 1px solid {c('border-strong')}; border-radius: {radius_lg}px; background: {c('surface-panel')}; }}
#hubCalloutCopy {{ background: {c('surface-panel')}; }}
#hubProjectTitle {{ color: {c('content-primary')}; font-size: 27px; font-weight: 700; }}
#hubHealth {{ border: 0; border-left: 1px solid {c('border-default')}; border-radius: 0 {radius_lg}px {radius_lg}px 0; background: {c('surface-panel-raised')}; }}
#hubHealthCheck {{ font-size: {type_caption}; }}
#hubHealthCheck:disabled {{ color: {c('content-disabled')}; }}
#recentProjects {{ border: 0; border-radius: 0; background: transparent; padding: 0; }}
#translationWorkspace {{ background: {c('surface-app')}; }}
#translationWorkspace QLabel#surfaceTitle {{ font-size: 34px; }}
#workspaceHeadingActions {{ background: transparent; }}
#workspaceRecovery {{ background: {c('status-warning-surface')}; border: 1px solid {c('status-warning-border')}; border-radius: {radius}px; }}
#workspaceRunCard {{ background: {workspace_run_background}; border: 1px solid {workspace_run_border}; border-radius: {radius}px; }}
#workspaceRunTitle {{ color: {c('content-primary')}; font-size: {type_title}; font-weight: 700; }}
#workspaceStageNote {{ background: {c('accent-primary-surface')}; border: 1px solid {c('accent-primary-border')}; border-radius: {radius_sm}px; }}
QToolTip {{ background: {c('surface-panel-raised')}; color: {c('content-primary')}; border: 1px solid {c('border-strong')}; padding: {space_2}px; }}
#applicationHeader {{ min-height: 54px; max-height: 54px; background: {c('surface-header')}; }}
#applicationIdentity {{ background: transparent; }}
#applicationMark {{ background: {c('accent-primary')}; border: 0; border-radius: 4px; }}
#applicationBrand {{ color: {c('content-primary')}; font-size: 19px; font-weight: 720; }}
#applicationRouteContext {{ color: {c('content-muted')}; font-size: {type_label}; padding-left: 14px; border-left: 1px solid {c('border-subtle')}; }}
#productNavigation {{ min-height: 54px; max-height: 54px; background: transparent; border: 0; border-radius: 0; }}
#productNavigation QToolButton {{ min-width: 110px; min-height: 54px; max-height: 54px; padding: 0 8px; border: 0; border-radius: 0; background: transparent; color: {c('content-secondary')}; font-size: {type_label}; font-weight: 650; }}
#productNavigation QToolButton:hover {{ color: {c('content-primary')}; background: {c('surface-panel')}; }}
#productNavigation QToolButton:checked {{ color: {c('content-primary')}; background: transparent; border-bottom: 3px solid {c('accent-primary')}; }}
#headerActions {{ background: transparent; }}
#headerActions > QToolButton {{ min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px; padding: 0; border: 1px solid {c('border-default')}; border-radius: 5px; background: {c('surface-panel')}; }}
#headerActions > QToolButton:hover {{ background: {c('surface-hover')}; border-color: {c('border-strong')}; }}
#headerActions > QToolButton:disabled {{ background: transparent; border-color: {c('border-subtle')}; }}
#windowControls {{ background: transparent; border-left: 1px solid {c('border-subtle')}; }}
#windowControls QToolButton {{ min-width: 31px; max-width: 31px; min-height: 31px; max-height: 31px; padding: 0; border: 0; border-radius: 4px; background: transparent; }}
#windowControls QToolButton:hover {{ background: {c('surface-hover')}; }}
#windowClose:hover {{ background: {c('status-danger-surface')}; }}
#pageRail {{ border-right: 1px solid {c('border-default')}; background: {c('surface-panel')}; }}
#pageRailHeading {{ min-height: 40px; max-height: 40px; border-bottom: 1px solid {c('border-subtle')}; background: transparent; }}
#pageRailProjectLabel {{ color: {c('content-muted')}; font-size: {type_caption}; font-weight: 500; }}
#pageRailProjectName {{ color: {c('content-primary')}; font-size: {type_label}; font-weight: 700; }}
#pageRailToggle {{ min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; padding: 0; border: 0; border-radius: 4px; background: transparent; }}
#pageRailToggle:hover {{ background: {c('surface-hover')}; }}
#pageRailSearchBand {{ min-height: 36px; max-height: 36px; background: transparent; }}
#pageRailSearchBand QLineEdit {{ min-height: 29px; max-height: 29px; padding: 0 8px; border: 1px solid {c('border-default')}; border-radius: 4px; background: {c('surface-control')}; }}
#editorPageList {{ border: 0; border-radius: 0; padding: 3px 7px 4px; background: transparent; outline: 0; }}
#pageRailFooter {{ min-height: 37px; max-height: 37px; border-top: 1px solid {c('border-default')}; background: transparent; }}
#pageRailFooter QToolButton {{ min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; padding: 0; border: 0; border-radius: 4px; background: transparent; }}
#pageRailFooter QToolButton:checked {{ color: {c('accent-primary')}; background: {c('accent-primary-surface')}; }}
#canvasModeStrip {{ min-height: 34px; max-height: 34px; border: 1px solid {c('border-default')}; border-radius: 5px; background: {c('surface-control')}; }}
#canvasModeStrip QToolButton {{ min-width: 70px; max-width: 70px; min-height: 28px; max-height: 28px; padding: 0; border: 0; border-radius: 3px; background: transparent; color: {c('content-secondary')}; }}
#canvasModeStrip QToolButton:checked {{ color: {c('content-primary')}; background: {c('accent-primary-surface')}; border-bottom: 2px solid {c('accent-primary')}; }}
#canvasModeStrip QToolButton[canvasMode="compare"] {{ min-width: 90px; max-width: 90px; color: {c('content-primary')}; background: {c('accent-primary-surface')}; }}
#canvasToolbarDivider {{ background: {c('border-strong')}; }}
#canvasToolbar > QToolButton {{ min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; padding: 0; border: 1px solid transparent; border-radius: 4px; background: transparent; }}
#canvasToolbar > QToolButton:hover, #canvasToolbar > QToolButton:checked {{ background: {c('surface-hover')}; border-color: {c('border-strong')}; }}
#canvasToolbar > #canvasFitButton {{ min-width: 54px; max-width: 54px; }}
#canvasToolbar > #canvasHoldButton {{ min-width: 94px; max-width: 94px; }}
#canvasOverlayButton::menu-indicator {{ image: none; width: 0px; height: 0px; }}
#settingsNewProviderButton::menu-indicator {{ image: none; width: 0px; height: 0px; }}
#badgeTextEditFrame {{ background: transparent; border: 0; }}
#textAuthorityBadge {{ color: {c('content-secondary')}; background: {c('surface-control')}; border: 1px solid {c('border-default')}; border-radius: 9px; padding: 2px 8px; font-size: {type_caption}; font-weight: 650; }}
#textAuthorityBadge[authority="user"] {{ color: {c('status-warning')}; border-color: {c('status-warning-border')}; background: {c('status-warning-surface')}; }}
#textAuthorityBadge[authority="unavailable"] {{ color: {c('content-disabled')}; }}
#activityBar {{ min-height: 50px; max-height: 50px; background: {c('surface-dock-bar')}; border-bottom: 1px solid {c('border-subtle')}; }}
#activityRunMonitor {{ min-width: 300px; max-width: 340px; min-height: 34px; max-height: 34px; background: {c('surface-panel')}; border: 1px solid {c('border-default')}; border-radius: 6px; }}
#activityRunProject {{ color: {c('content-secondary')}; font-size: {type_caption}; font-weight: 700; }}
#activityRunEta, #activityPageIdentity {{ color: {c('content-muted')}; font-size: {type_caption}; }}
#activityPageIdentity {{ max-width: 220px; }}
#activityTabs {{ min-height: 32px; max-height: 32px; background: {c('surface-dock')}; border: 1px solid {c('border-default')}; border-radius: 6px; }}
#activityTabs QToolButton {{ min-height: 26px; max-height: 26px; padding: 0 7px; border: 1px solid transparent; border-radius: 4px; background: transparent; color: {c('content-muted')}; font-size: {type_caption}; font-weight: 700; }}
#activityTabs QToolButton:hover {{ background: {c('surface-hover')}; color: {c('content-primary')}; }}
#activityTabs QToolButton:checked {{ background: {c('surface-panel-raised')}; border-color: {c('border-default')}; color: {c('content-primary')}; }}
#activityAuthorityStrip {{ min-height: 26px; max-height: 26px; border: 1px solid {c('border-subtle')}; border-radius: 4px; background: {c('surface-control')}; }}
#activityAuthorityStrip QLabel {{ color: {c('content-muted')}; font-size: {type_caption}; border-right: 1px solid {c('border-subtle')}; }}
#activityAuthorityStrip QLabel[authority="effective"] {{ color: {c('success-text')}; background: {c('status-effective-surface')}; border-right: 0; font-weight: 700; }}
#translationWorkspace #commandBar {{ border: 1px solid {c('border-subtle')}; border-radius: {radius}px; }}
#workspacePageSearch {{ min-width: 100px; max-width: 180px; min-height: 30px; max-height: 30px; padding: 0 8px; border: 1px solid {c('border-default')}; border-radius: 5px; background: {c('surface-control')}; color: {c('content-primary')}; }}
#workspacePageFilter {{ min-width: 108px; max-width: 160px; min-height: 34px; max-height: 34px; padding: 0 28px 0 9px; border: 1px solid {c('border-default')}; border-radius: 5px; background: {c('surface-control')}; color: {c('content-secondary')}; }}
#workspacePageFilter::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 28px; border: 0; background: transparent; }}
#workspaceQueueColumns {{ min-height: 34px; max-height: 34px; background: {c('surface-panel-raised')}; border-bottom: 1px solid {c('border-default')}; }}
#workspacePageList {{ padding: 0; border: 0; border-radius: 0; }}
#workspaceStageActivity QFrame#stageStep {{ min-height: 48px; border: 0; border-radius: {radius_sm}px; background: transparent; }}
#workspaceStageActivity QFrame#stageStep[state="active"] {{ background: {c('accent-primary-surface')}; border-left: 2px solid {c('accent-primary')}; }}
#workspaceStageActivity QLabel#stageStepNumber {{ color: {c('content-muted')}; border: 1px solid {c('border-default')}; border-radius: 11px; }}
#workspaceStageActivity QLabel#stageStepTitle {{ color: {c('content-secondary')}; font-weight: 700; }}
#workspaceStageActivity QLabel#stageStepDetail {{ color: {c('content-muted')}; font-size: {type_caption}; }}
#workspaceStageActivity QFrame#stageStep[state="complete"] QLabel#stageStepTitle {{ color: {c('content-secondary')}; }}
#workspaceStageActivity QFrame#stageStep[state="active"] QLabel#stageStepTitle {{ color: {c('accent-text')}; }}
#settingsSidebar {{ border: 0; border-right: 1px solid {c('border-default')}; border-radius: 0; }}
#settingsSidebar QLineEdit {{ min-height: 32px; }}
#settingsCategories {{ border: 0; background: transparent; padding: 0; }}
#settingsCategories::item {{ min-height: 40px; margin: 1px 0; padding: 0; border: 1px solid transparent; border-radius: {radius_sm}px; }}
#settingsCategories::item:hover {{ background: {c('surface-hover')}; }}
#settingsCategories::item:selected {{ background: {c('surface-panel-raised')}; border-color: {c('accent-primary-border')}; color: {c('content-primary')}; }}
#settingsCategories QWidget[settingsCategory="true"] {{ background: transparent; }}
#settingsCategories QWidget[settingsCategory="true"] QLabel {{ color: {c('content-muted')}; }}
#settingsCategories QWidget[settingsCategory="true"][active="true"] QLabel {{ color: {c('content-primary')}; }}
#settingsProjectScope {{ background: {c('surface-panel')}; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; }}
#settingsPageTitle {{ font-size: 25px; }}
#settingsPageHeading {{ background: transparent; }}
QFrame[role="effective-run-value"] {{ background: {c('surface-control')}; border: 1px solid {c('border-default')}; border-radius: {radius_sm}px; }}
QToolButton[role="theme-choice"] {{ min-height: 112px; padding: 0; border: 1px solid {c('border-default')}; border-radius: {radius}px; background: {c('surface-control')}; color: {c('content-primary')}; font-weight: 650; text-align: left; }}
QToolButton[role="theme-choice"]:hover {{ background: {c('surface-hover')}; border-color: {c('border-strong')}; }}
QToolButton[role="theme-choice"]:checked {{ background: {c('surface-selected')}; border: 2px solid {c('accent-primary')}; color: {c('content-primary')}; }}
#appearanceThemeSwatch {{ border: 1px solid #526174; border-radius: 4px; }}
#appearanceThemeSwatch[theme="dark"] {{ background: #111a28; }}
#appearanceThemeSwatch[theme="light"] {{ background: #f7f9fc; }}
#appearanceThemeChoiceTitle, #appearanceThemeChoiceDetail {{ background: transparent; }}
#appearancePreviewSample {{ min-height: 120px; border: 1px solid {c('border-default')}; border-radius: {radius}px; background: {c('surface-app')}; }}
#appearancePreviewNav {{ background: {c('surface-header')}; color: {c('content-primary')}; font-weight: 700; border-right: 1px solid {c('border-default')}; border-radius: {radius}px 0 0 {radius}px; }}
#appearancePreviewCanvas {{ background: {appearance_preview_canvas}; color: {appearance_preview_canvas_text}; }}
#appearancePreviewInspector {{ background: {c('surface-panel-raised')}; border-left: 1px solid {c('border-default')}; border-radius: 0 {radius}px {radius}px 0; }}
#runtimeAssetRow QLabel[role="runtime-icon"] {{ background: {c('accent-primary-surface')}; border-radius: 6px; }}
#shortcutBindingRow {{ min-height: 52px; background: transparent; border: 0; border-bottom: 1px solid {c('border-default')}; border-radius: 0; }}
#providerHeading QLabel#surfaceTitle {{ font-size: 25px; }}
#providersWorkspace {{ border: 1px solid {c('border-default')}; border-radius: {radius_lg}px; background: {c('surface-panel')}; }}
#providerProfilesPanel {{ border: 0; border-right: 1px solid {c('border-default')}; border-radius: {radius_lg}px 0 0 {radius_lg}px; background: {c('surface-panel-raised')}; }}
#providerProfileList {{ border: 0; background: transparent; padding: 8px; }}
#providerProfileList::item {{ min-height: 60px; margin: 2px; padding: 8px 10px; border: 1px solid transparent; border-radius: {radius}px; }}
#providerProfileList::item:hover {{ background: {c('surface-hover')}; }}
#providerProfileList::item:selected {{ background: {c('surface-selected')}; border-color: {c('accent-primary-border')}; color: {c('content-primary')}; }}
#providerProfileList QWidget[providerProfileRow="true"] {{ background: transparent; }}
#providerListMark {{ background: {c('accent-primary')}; border-radius: {radius_sm}px; }}
#providerListName {{ color: {c('content-primary')}; font-size: {type_label}; font-weight: 600; }}
#providerListDetail {{ color: {c('content-muted')}; font-size: {type_caption}; }}
#providerEditorCard {{ border: 0; border-radius: 0 {radius_lg}px {radius_lg}px 0; }}
#providerEditorHeader {{ border-bottom: 1px solid {c('border-default')}; background: {c('surface-panel')}; }}
#providerEditorBody, #providerEditorBodyContent {{ border: 0; background: transparent; }}
#providerEmptyState {{ border: 0; background: transparent; }}
#providerEmptyStateIcon {{ min-width: 48px; min-height: 48px; background: {c('accent-primary-surface')}; border-radius: {radius}px; }}
#providerSummaryIcon {{ background: {c('accent-primary')}; border-radius: {radius}px; }}
#newProjectDialogIcon {{ background: {c('accent-primary')}; border-radius: {radius}px; }}
QDialog[hybridDialog="true"] {{ background: transparent; border: 0; }}
#hybridDialogHeader {{ border: 0; background: transparent; }}
#hybridDialogHeaderIcon {{ background: {c('accent-primary')}; border-radius: {radius}px; }}
#hybridDialogClose {{ min-width: 32px; max-width: 32px; min-height: 32px; max-height: 32px; padding: 0; border: 1px solid transparent; border-radius: 5px; background: transparent; }}
#hybridDialogClose:hover {{ background: {c('status-danger-surface')}; border-color: {c('status-danger-border')}; }}
#hybridDialogMessage {{ padding: 10px 12px; background: {c('surface-control')}; border: 1px solid {c('border-default')}; border-radius: {radius}px; }}
#hybridDialogTextInput {{ min-height: 38px; }}
#providerForm QLabel[role="field-label"] {{ color: {c('content-muted')}; font-size: {type_caption}; }}
#providerForm QLineEdit, #providerForm QComboBox {{ min-height: 34px; max-height: 34px; }}
#secureCredentialReference {{ color: {c('content-muted')}; background: {c('surface-control')}; }}
#providerCommitBar {{ min-height: 61px; border-top: 1px solid {c('border-default')}; border-bottom: 0; background: {c('surface-panel-raised')}; }}
#providerSafetyCallout {{ background: {c('status-success-surface')}; border: 1px solid {c('status-success-border')}; border-radius: {radius}px; }}
#providerSafetyIcon {{ color: {c('success-text')}; }}
""".strip()
    return _font_sizes_in_points(stylesheet)


def apply_application_theme(application: Any, options: ThemeOptions) -> None:
    """Apply palette, semantic QSS, and user font scale to a Qt application."""

    if not isinstance(options, ThemeOptions):
        raise TypeError("options must be ThemeOptions")
    palette = build_qpalette(options.theme)
    application.setPalette(palette)
    application.setStyleSheet(build_application_stylesheet(options))

    font = application.font()
    property_name = "yomiframeBasePointSize"
    base_size = application.property(property_name)
    try:
        base_size = float(base_size)
    except (TypeError, ValueError):
        base_size = float(font.pointSizeF())
        if not math.isfinite(base_size) or base_size <= 0:
            base_size = 9.0
        application.setProperty(property_name, base_size)
    scaled_size = max(1.0, base_size * options.font_scale / 100.0)
    font.setPointSizeF(scaled_size)
    application.setFont(font)
    application.setProperty("yomiframeTheme", options.theme)
    application.setProperty("yomiframeDensity", options.density)
    application.setProperty("yomiframeFontScale", options.font_scale)
    application.setProperty("yomiframeReducedMotion", options.reduced_motion)


__all__ = [
    "ThemeOptions",
    "apply_application_theme",
    "build_application_stylesheet",
    "build_qpalette",
]
