# -*- coding: utf-8 -*-
"""Download progress dialog."""
from PySide6 import QtWidgets, QtCore

class DownloadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Downloading Models"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(400)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint) # Disable close button to enforce wait or cancel

        layout = QtWidgets.QVBoxLayout(self)
        
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        
        self._downloader = None

    def set_downloader(self, downloader):
        """Connect downloader signals."""
        self._downloader = downloader
        self._downloader.progress_changed.connect(self.progress_bar.setValue)
        self._downloader.status_changed.connect(self.status_label.setText)
        self._downloader.finished.connect(self._on_finished)
        # Connect cancel
        self.rejected.connect(self._downloader.request_cancel)

    def _on_finished(self, success: bool, message: str):
        if success:
            self.accept()
        else:
            if message == "Cancelled":
                pass # Already rejected
            else:
                QtWidgets.QMessageBox.critical(self, "Download Error", message)
                self.reject()
