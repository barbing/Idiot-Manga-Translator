# -*- coding: utf-8 -*-
"""Hybrid Pro runtime-download progress dialog."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.ui.design_system.dialogs import HybridDialog


class DownloadDialog(HybridDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Downloading Models",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("downloadDialog")
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setAccessibleName(title)
        self.setAccessibleDescription(
            "Managed runtime download progress with an explicit cancellation action."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(16)
        layout.addWidget(
            self.create_dialog_header(
                title=title,
                subtitle=(
                    "YomiFrame keeps this asset in its managed runtime folder. "
                    "Cancel leaves the current installed state unchanged."
                ),
                icon_name="runtime",
                close_accessible_name="Cancel download",
            )
        )

        self.status_label = QtWidgets.QLabel("Preparing download…")
        self.status_label.setObjectName("downloadStatus")
        self.status_label.setProperty("role", "secondary")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Download status")
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("downloadProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAccessibleName("Runtime asset download progress")
        layout.addWidget(self.progress_bar)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel download")
        self.cancel_btn.setProperty("role", "command")
        self.cancel_btn.setProperty("variant", "quiet")
        self.cancel_btn.setAccessibleName("Cancel runtime asset download")
        self.cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self.cancel_btn)
        layout.addLayout(footer)

        if self.dialog_header is not None:
            QtWidgets.QWidget.setTabOrder(
                self.cancel_btn,
                self.dialog_header.close_button,
            )
        self._downloader = None

    def set_downloader(self, downloader: QtCore.QObject) -> None:
        """Connect the owned downloader without changing worker semantics."""

        self._downloader = downloader
        self._downloader.progress_changed.connect(self.progress_bar.setValue)
        self._downloader.status_changed.connect(self.status_label.setText)
        terminal = getattr(self._downloader, "completed", self._downloader.finished)
        terminal.connect(self._on_finished)
        self.rejected.connect(
            self._downloader.request_cancel,
            QtCore.Qt.ConnectionType.DirectConnection,
        )

    def _on_finished(self, success: bool, message: str) -> None:
        self.status_label.setText(str(message or "Download finished."))
        if success:
            self.accept()
        else:
            self.reject()


__all__ = ["DownloadDialog"]
