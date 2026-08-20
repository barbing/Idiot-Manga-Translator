# -*- coding: utf-8 -*-
"""Hybrid Pro modal chrome and reusable app-owned decision dialogs."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .icons import hybrid_icon
from .tokens import theme_token


def _current_theme() -> str:
    application = QtWidgets.QApplication.instance()
    value = application.property("yomiframeTheme") if application is not None else None
    return "light" if str(value or "dark").strip().casefold() == "light" else "dark"


class HybridDialog(QtWidgets.QDialog):
    """Frameless app-owned modal with visible, bounded custom chrome."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setProperty("hybridDialog", True)
        self.dialog_header: HybridDialogHeader | None = None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 10.0, 10.0)
        theme = _current_theme()
        painter.fillPath(path, QtGui.QColor(theme_token(theme, "surface-panel")))
        painter.setPen(
            QtGui.QPen(QtGui.QColor(theme_token(theme, "border-strong")), 1.0)
        )
        painter.drawPath(path)
        painter.end()
        event.accept()

    def create_dialog_header(
        self,
        *,
        title: str,
        subtitle: str = "",
        icon_name: str | None = None,
        close_accessible_name: str = "Close dialog",
    ) -> "HybridDialogHeader":
        header = HybridDialogHeader(
            title=title,
            subtitle=subtitle,
            icon_name=icon_name,
            close_accessible_name=close_accessible_name,
            parent=self,
        )
        header.close_requested.connect(self.reject)
        self.dialog_header = header
        return header


class HybridDialogHeader(QtWidgets.QFrame):
    """Visible title/drag region; no resize or invisible hit-test extension."""

    close_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        title: str,
        subtitle: str = "",
        icon_name: str | None = None,
        close_accessible_name: str = "Close dialog",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("hybridDialogHeader")
        self.setProperty("role", "dialog-header")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        if icon_name:
            icon = QtWidgets.QLabel()
            icon.setObjectName("hybridDialogHeaderIcon")
            icon.setProperty("dialogIcon", True)
            icon.setFixedSize(36, 36)
            icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            icon.setPixmap(
                hybrid_icon(icon_name, _current_theme(), active=True).pixmap(20, 20)
            )
            layout.addWidget(icon, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        copy = QtWidgets.QVBoxLayout()
        copy.setSpacing(3)
        self.title = QtWidgets.QLabel(str(title or "").strip())
        self.title.setObjectName("hybridDialogTitle")
        self.title.setProperty("role", "title")
        copy.addWidget(self.title)
        self.subtitle = QtWidgets.QLabel(str(subtitle or "").strip())
        self.subtitle.setObjectName("hybridDialogSubtitle")
        self.subtitle.setProperty("role", "secondary")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(self.subtitle.text()))
        copy.addWidget(self.subtitle)
        layout.addLayout(copy, 1)

        self.trailing_layout = QtWidgets.QHBoxLayout()
        self.trailing_layout.setContentsMargins(0, 0, 0, 0)
        self.trailing_layout.setSpacing(8)
        layout.addLayout(self.trailing_layout)
        self.close_button = QtWidgets.QToolButton()
        self.close_button.setObjectName("hybridDialogClose")
        self.close_button.setIcon(hybrid_icon("close", _current_theme(), secondary=True))
        self.close_button.setAccessibleName(close_accessible_name)
        self.close_button.setAccessibleDescription(
            "Close this app-owned dialog without applying a pending decision."
        )
        self.close_button.setToolTip(close_accessible_name)
        self.close_button.clicked.connect(self.close_requested)
        self.trailing_layout.addWidget(self.close_button)

    def add_trailing_widget(self, widget: QtWidgets.QWidget) -> None:
        self.trailing_layout.insertWidget(
            max(0, self.trailing_layout.count() - 1),
            widget,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            child = self.childAt(event.position().toPoint())
            if child is not self.close_button:
                handle = self.window().windowHandle()
                if handle is not None and handle.startSystemMove():
                    event.accept()
                    return
        super().mousePressEvent(event)


class HybridConfirmDialog(HybridDialog):
    """Themed replacement for app-owned QMessageBox decisions."""

    def __init__(
        self,
        *,
        title: str,
        message: str,
        confirm_text: str,
        cancel_text: str = "Cancel",
        destructive: bool = False,
        icon_name: str = "warning",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("hybridConfirmDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setAccessibleName(title)
        self.setAccessibleDescription(message)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(16)
        root.addWidget(
            self.create_dialog_header(
                title=title,
                icon_name=icon_name,
                close_accessible_name="Cancel and close",
            )
        )
        self.message = QtWidgets.QLabel(str(message or "").strip())
        self.message.setObjectName("hybridDialogMessage")
        self.message.setWordWrap(True)
        self.message.setProperty("role", "secondary")
        root.addWidget(self.message)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton(cancel_text)
        self.cancel_button.setProperty("role", "command")
        self.cancel_button.setProperty("variant", "quiet")
        self.cancel_button.setAccessibleName(cancel_text)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button = QtWidgets.QPushButton(confirm_text)
        self.confirm_button.setProperty("role", "command")
        self.confirm_button.setProperty("variant", "primary")
        if destructive:
            self.confirm_button.setProperty("tone", "danger")
        self.confirm_button.setAccessibleName(confirm_text)
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setDefault(True)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.confirm_button)
        root.addLayout(footer)

        QtWidgets.QWidget.setTabOrder(self.cancel_button, self.confirm_button)
        if self.dialog_header is not None:
            QtWidgets.QWidget.setTabOrder(
                self.confirm_button,
                self.dialog_header.close_button,
            )

    @classmethod
    def ask(
        cls,
        parent: QtWidgets.QWidget | None,
        *,
        title: str,
        message: str,
        confirm_text: str,
        cancel_text: str = "Cancel",
        destructive: bool = False,
        icon_name: str = "warning",
    ) -> bool:
        dialog = cls(
            parent=parent,
            title=title,
            message=message,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            destructive=destructive,
            icon_name=icon_name,
        )
        result = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        dialog.deleteLater()
        return result


class HybridTextInputDialog(HybridDialog):
    """Themed secure single-value prompt for provider credentials."""

    def __init__(
        self,
        *,
        title: str,
        prompt: str,
        confirm_text: str,
        cancel_text: str = "Cancel",
        password: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("hybridTextInputDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setAccessibleName(title)
        self.setAccessibleDescription(prompt)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(16)
        root.addWidget(
            self.create_dialog_header(
                title=title,
                subtitle=(
                    "The secret stays transient unless a successful test is "
                    "followed by explicit secure-save approval."
                ),
                icon_name="providers",
                close_accessible_name="Cancel credential entry",
            )
        )
        field = QtWidgets.QWidget()
        field_layout = QtWidgets.QVBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(6)
        label = QtWidgets.QLabel(prompt)
        label.setProperty("role", "field-label")
        field_layout.addWidget(label)
        self.input = QtWidgets.QLineEdit()
        self.input.setObjectName("hybridDialogTextInput")
        self.input.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password
            if password
            else QtWidgets.QLineEdit.EchoMode.Normal
        )
        self.input.setAccessibleName(prompt)
        self.input.setAccessibleDescription(
            "Sensitive value; characters are masked and are not copied to project data."
        )
        field_layout.addWidget(self.input)
        root.addWidget(field)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton(cancel_text)
        self.cancel_button.setProperty("role", "command")
        self.cancel_button.setProperty("variant", "quiet")
        self.cancel_button.setAccessibleName(cancel_text)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button = QtWidgets.QPushButton(confirm_text)
        self.confirm_button.setProperty("role", "command")
        self.confirm_button.setProperty("variant", "primary")
        self.confirm_button.setAccessibleName(confirm_text)
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setDefault(True)
        self.input.textChanged.connect(
            lambda value: self.confirm_button.setEnabled(bool(str(value).strip()))
        )
        self.confirm_button.setEnabled(False)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.confirm_button)
        root.addLayout(footer)

        QtWidgets.QWidget.setTabOrder(self.input, self.cancel_button)
        QtWidgets.QWidget.setTabOrder(self.cancel_button, self.confirm_button)
        if self.dialog_header is not None:
            QtWidgets.QWidget.setTabOrder(
                self.confirm_button,
                self.dialog_header.close_button,
            )
        self.input.setFocus()

    @property
    def value(self) -> str:
        return self.input.text()

    @classmethod
    def get_text(
        cls,
        parent: QtWidgets.QWidget | None,
        *,
        title: str,
        prompt: str,
        confirm_text: str,
        password: bool = True,
    ) -> tuple[str, bool]:
        dialog = cls(
            parent=parent,
            title=title,
            prompt=prompt,
            confirm_text=confirm_text,
            password=password,
        )
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        value = dialog.value if accepted else ""
        dialog.deleteLater()
        return value, accepted


__all__ = [
    "HybridConfirmDialog",
    "HybridDialog",
    "HybridDialogHeader",
    "HybridTextInputDialog",
]
