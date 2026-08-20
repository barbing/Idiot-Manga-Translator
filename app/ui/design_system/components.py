# -*- coding: utf-8 -*-
"""Small reusable semantic PySide6 widget factories.

PySide6 is imported only when a factory is called.  Pure contracts, tests, and
view models can import this module without initializing the Qt runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.ui.ui_contract import PRESENTATION_TONE_IDS


_HYBRID_COMBO_CLASS: Any | None = None


@dataclass(frozen=True, slots=True)
class SemanticProperties:
    role: str | None = None
    tone: str | None = None
    authority: str | None = None
    state: str | None = None
    variant: str | None = None
    accessible_name: str | None = None
    accessible_description: str | None = None


def apply_semantic_properties(
    widget: Any,
    properties: SemanticProperties | None = None,
    *,
    role: str | None = None,
    tone: str | None = None,
    authority: str | None = None,
    state: str | None = None,
    variant: str | None = None,
    accessible_name: str | None = None,
    accessible_description: str | None = None,
) -> Any:
    """Apply dynamic style and accessibility properties to a Qt widget."""

    base = properties or SemanticProperties()
    values = {
        "role": role if role is not None else base.role,
        "tone": tone if tone is not None else base.tone,
        "authority": authority if authority is not None else base.authority,
        "state": state if state is not None else base.state,
        "variant": variant if variant is not None else base.variant,
    }
    for name, value in values.items():
        if value is not None:
            widget.setProperty(name, value)
    name = accessible_name if accessible_name is not None else base.accessible_name
    description = (
        accessible_description
        if accessible_description is not None
        else base.accessible_description
    )
    if name is not None:
        widget.setAccessibleName(name)
    if description is not None:
        widget.setAccessibleDescription(description)

    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()
    return widget


def _widgets() -> Any:
    from PySide6 import QtWidgets

    return QtWidgets


def StatusPill(
    text: str,
    *,
    tone: str = "muted",
    icon: Any = None,
    parent: Any = None,
) -> Any:
    """Create a non-interactive status pill with text and optional ``QIcon``."""

    if tone not in PRESENTATION_TONE_IDS:
        raise ValueError(f"Unknown presentation tone: {tone}")
    QtWidgets = _widgets()
    frame = QtWidgets.QFrame(parent)
    layout = QtWidgets.QHBoxLayout(frame)
    layout.setContentsMargins(7, 2, 7, 2)
    layout.setSpacing(5)
    if icon is not None:
        from PySide6.QtCore import Qt

        icon_label = QtWidgets.QLabel(frame)
        icon_label.setPixmap(icon.pixmap(14, 14))
        icon_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        layout.addWidget(icon_label)
    label = QtWidgets.QLabel(text, frame)
    label.setProperty("role", "status-text")
    layout.addWidget(label)
    apply_semantic_properties(
        frame,
        role="status-pill",
        tone=tone,
        accessible_name=text,
    )
    return frame


def SectionHeader(
    title: str,
    *,
    subtitle: str = "",
    action: Any = None,
    parent: Any = None,
) -> Any:
    """Create a title/subtitle group with an optional trailing action widget."""

    QtWidgets = _widgets()
    frame = QtWidgets.QFrame(parent)
    outer = QtWidgets.QHBoxLayout(frame)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(12)
    copy = QtWidgets.QWidget(frame)
    copy_layout = QtWidgets.QVBoxLayout(copy)
    copy_layout.setContentsMargins(0, 0, 0, 0)
    copy_layout.setSpacing(2)
    title_label = QtWidgets.QLabel(title, copy)
    title_label.setProperty("role", "title")
    copy_layout.addWidget(title_label)
    if subtitle:
        subtitle_label = QtWidgets.QLabel(subtitle, copy)
        subtitle_label.setProperty("role", "muted")
        subtitle_label.setWordWrap(True)
        copy_layout.addWidget(subtitle_label)
    outer.addWidget(copy, 1)
    if action is not None:
        outer.addWidget(action, 0)
    apply_semantic_properties(
        frame,
        role="section-header",
        accessible_name=title,
        accessible_description=subtitle or None,
    )
    return frame


def EmptyState(
    title: str,
    *,
    detail: str = "",
    action_text: str = "",
    on_action: Callable[[], None] | None = None,
    parent: Any = None,
) -> Any:
    """Create a compact empty state with an optional explicit command."""

    QtWidgets = _widgets()
    frame = QtWidgets.QFrame(parent)
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(6)
    title_label = QtWidgets.QLabel(title, frame)
    title_label.setProperty("role", "title")
    layout.addWidget(title_label)
    if detail:
        detail_label = QtWidgets.QLabel(detail, frame)
        detail_label.setProperty("role", "muted")
        detail_label.setWordWrap(True)
        layout.addWidget(detail_label)
    if action_text:
        button = CommandButton(
            action_text,
            command_id="empty-state-action",
            variant="secondary",
            parent=frame,
        )
        if on_action is not None:
            button.clicked.connect(on_action)
        layout.addWidget(button, 0)
    apply_semantic_properties(
        frame,
        role="empty-state",
        accessible_name=title,
        accessible_description=detail or None,
    )
    return frame


def CommandButton(
    text: str,
    *,
    command_id: str,
    variant: str = "secondary",
    tone: str | None = None,
    checkable: bool = False,
    parent: Any = None,
) -> Any:
    """Create a semantic command button; callers own command dispatch."""

    if not isinstance(command_id, str) or not command_id.strip():
        raise ValueError("command_id is required")
    if variant not in {"primary", "secondary", "quiet"}:
        raise ValueError(f"Unknown command variant: {variant}")
    if tone is not None and tone not in {*PRESENTATION_TONE_IDS, "danger"}:
        raise ValueError(f"Unknown command tone: {tone}")
    QtWidgets = _widgets()
    button = QtWidgets.QPushButton(text, parent)
    button.setCheckable(checkable)
    button.setProperty("commandId", command_id.strip())
    apply_semantic_properties(
        button,
        role="command",
        tone=tone,
        variant=variant,
        accessible_name=text,
        accessible_description=f"Command: {command_id.strip()}",
    )
    return button


def HybridComboBox(*, parent: Any = None) -> Any:
    """Create a combo box with the prototype's clean library chevron.

    Qt's platform combo indicator includes a boxed native drop-down surface on
    Windows.  Hybrid Pro uses the ordinary control border plus one quiet
    chevron, so the reusable widget paints that real QtAwesome glyph after the
    standard combo contents instead of approximating it with text or CSS art.
    """

    global _HYBRID_COMBO_CLASS
    if _HYBRID_COMBO_CLASS is None:
        from PySide6 import QtCore, QtGui, QtWidgets

        from app.ui.design_system.icons import hybrid_icon

        class _HybridComboBox(QtWidgets.QComboBox):
            def __init__(self, widget_parent: Any = None) -> None:
                super().__init__(widget_parent)
                self.setProperty("hybridChevron", True)

            def paintEvent(self, event: Any) -> None:  # noqa: N802
                super().paintEvent(event)
                palette_color = self.palette().color(
                    QtGui.QPalette.ColorRole.Window
                )
                theme = "dark" if palette_color.lightnessF() < 0.5 else "light"
                icon = hybrid_icon("caret-down", theme)
                size = 9
                indicator = QtCore.QRect(
                    max(0, self.width() - 19),
                    max(0, (self.height() - size) // 2),
                    size,
                    size,
                )
                painter = QtGui.QPainter(self)
                mode = (
                    QtGui.QIcon.Mode.Normal
                    if self.isEnabled()
                    else QtGui.QIcon.Mode.Disabled
                )
                icon.paint(
                    painter,
                    indicator,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    mode,
                )

        _HYBRID_COMBO_CLASS = _HybridComboBox
    return _HYBRID_COMBO_CLASS(parent)


__all__ = [
    "CommandButton",
    "EmptyState",
    "HybridComboBox",
    "SectionHeader",
    "SemanticProperties",
    "StatusPill",
    "apply_semantic_properties",
]
