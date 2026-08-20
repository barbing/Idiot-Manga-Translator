# -*- coding: utf-8 -*-
"""Split view page review dialog."""
from __future__ import annotations
import os
from typing import Any, Dict
from PySide6 import QtCore, QtGui, QtWidgets
from app.ui.page_rerender_worker import (
    PageRerenderWorker,
    discard_preview_lease,
)
from app.ui.viewmodels.page_rerender_model import (
    PageRerenderCommand,
    PageRerenderFailure,
    PageRerenderPreviewLease,
    PageRerenderViewModel,
    PageRerenderViewState,
)


class ResizableLabel(QtWidgets.QLabel):
    """QLabel that scales its pixmap to fill the available space."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._pixmap = None

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        super().setPixmap(self._scaled_pixmap())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap())

    def _scaled_pixmap(self) -> QtGui.QPixmap:
        if not self._pixmap or self._pixmap.isNull():
            return self._pixmap
        return self._pixmap.scaled(
            self.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )

class PageReviewDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        page_record: Dict[str, Any] | None = None,
        json_path: str = "",
        use_gpu: bool = True,
        pipeline_idle: bool = True,
        pipeline_block_reason: str = "",
    ) -> None:
        super().__init__(parent)
        if type(use_gpu) is not bool or type(pipeline_idle) is not bool:
            raise TypeError("use_gpu and pipeline_idle must be bool values")
        self.setWindowTitle("Review Page")
        self.resize(1320, 880)
        self.setMinimumSize(1200, 820)
        self._page = page_record or {}
        self._json_path = json_path
        self._use_gpu = use_gpu
        self._pipeline_idle = pipeline_idle
        self._pipeline_block_reason = str(pipeline_block_reason or "").strip()
        self._rerender_model = PageRerenderViewModel()
        self._rerender_thread: QtCore.QThread | None = None
        self._rerender_worker: PageRerenderWorker | None = None
        self._close_after_rerender = False
        self._preview_lease: PageRerenderPreviewLease | None = None
        self._setup_ui()
        self._load_page()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout.addWidget(split, 10)

        left = QtWidgets.QWidget(split)
        left_layout = QtWidgets.QVBoxLayout(left)
        self.original_label = QtWidgets.QLabel("Original")
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_view = ResizableLabel()
        left_layout.addWidget(self.original_label)
        left_layout.addWidget(self.original_view, 1)

        right = QtWidgets.QWidget(split)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.translated_label = QtWidgets.QLabel("Translated")
        self.translated_label.setAlignment(QtCore.Qt.AlignCenter)
        self.translated_view = ResizableLabel()
        right_layout.addWidget(self.translated_label)
        right_layout.addWidget(self.translated_view, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Region", "OCR", "Translation", "Needs Review"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 2)

        rerender_status = QtWidgets.QHBoxLayout()
        self.rerender_status = QtWidgets.QLabel(
            "Preview uses the current persisted project edits."
        )
        self.rerender_status.setWordWrap(True)
        self.rerender_progress = QtWidgets.QProgressBar()
        self.rerender_progress.setRange(0, 1)
        self.rerender_progress.setValue(0)
        self.rerender_progress.setTextVisible(False)
        self.rerender_progress.setMaximumWidth(220)
        self.cancel_render_btn = QtWidgets.QPushButton("Cancel preview")
        self.cancel_render_btn.setEnabled(False)
        rerender_status.addWidget(self.rerender_status, 1)
        rerender_status.addWidget(self.rerender_progress)
        rerender_status.addWidget(self.cancel_render_btn)
        layout.addLayout(rerender_status)

        footer = QtWidgets.QHBoxLayout()
        self.manual_cleanup_btn = QtWidgets.QPushButton("Manual cleanup...")
        self.manual_cleanup_btn.setObjectName("pageReviewManualCleanupButton")
        self.render_btn = QtWidgets.QPushButton("Preview current project")
        self.close_btn = QtWidgets.QPushButton("Close")
        footer.addWidget(self.manual_cleanup_btn)
        footer.addWidget(self.render_btn)
        footer.addStretch(1)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

        self.render_btn.clicked.connect(self._rerender)
        self.manual_cleanup_btn.clicked.connect(self._open_manual_cleanup)
        self.cancel_render_btn.clicked.connect(self._cancel_rerender)
        self.close_btn.clicked.connect(self._request_close)

    def _load_page(self) -> None:
        image_path = self._page.get("image_path", "")
        output_path = self._page.get("output_path", "")
        if image_path and os.path.isfile(image_path):
            pixmap = QtGui.QPixmap(image_path)
            self.original_view.setPixmap(pixmap)
        if output_path and os.path.isfile(output_path):
            pixmap = QtGui.QPixmap(output_path)
            self.translated_view.setPixmap(pixmap)
        self._populate_table()
        self._update_manual_cleanup_action()

    def _update_manual_cleanup_action(self) -> None:
        project_ready = bool(self._json_path and os.path.isfile(self._json_path))
        page_ready = bool(str(self._page.get("page_id") or "").strip())
        rerender_idle = self._rerender_thread is None
        enabled = (
            self._pipeline_idle
            and project_ready
            and page_ready
            and rerender_idle
        )
        self.manual_cleanup_btn.setEnabled(enabled)
        if not self._pipeline_idle:
            self.manual_cleanup_btn.setToolTip(
                self._pipeline_block_reason
                or "Manual cleanup is unavailable while the forward pipeline is running."
            )
        elif not project_ready:
            self.manual_cleanup_btn.setToolTip(
                "Open or save this project before editing its clean base."
            )
        elif not page_ready:
            self.manual_cleanup_btn.setToolTip(
                "This page has no stable identity for manual cleanup."
            )
        else:
            self.manual_cleanup_btn.setToolTip(
                "Create a user-authored cleanup revision for this page."
            )

    def _open_manual_cleanup(self) -> None:
        if not self.manual_cleanup_btn.isEnabled():
            return
        page_id = str(self._page.get("page_id") or "").strip()
        from app.ui.manual_cleanup_dialog import ManualCleanupDialog

        dialog = ManualCleanupDialog(
            self,
            project_path=self._json_path,
            page_id=page_id,
            use_gpu=self._use_gpu,
            pipeline_idle=self._pipeline_idle,
            pipeline_block_reason=self._pipeline_block_reason,
        )
        dialog.exec()
        if dialog.committed_receipt is None:
            return
        self.rerender_status.setText(
            "Manual cleanup revision committed. Refreshing the effective preview..."
        )
        self._reload_effective_page()
        self._rerender()

    def _reload_effective_page(self) -> None:
        """Refresh immutable page evidence before the worker reloads projection.

        Manual cleanup changes only the adjacent edit/artifact state, not this
        automated page record.  ``_rerender`` immediately asks the typed worker
        to reload the hydrated project and its effective clean-base revision.
        """

        self._load_page()

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        for region in self._page.get("regions", []):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(region.get("region_id", ""))))
            ocr_item = QtWidgets.QTableWidgetItem(str(region.get("ocr_text", "")))
            ocr_item.setFlags(ocr_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 1, ocr_item)
            trans_item = QtWidgets.QTableWidgetItem(str(region.get("translation", "")))
            trans_item.setFlags(trans_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 2, trans_item)
            check_item = QtWidgets.QTableWidgetItem("")
            check_item.setFlags(check_item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.Checked if region.get("flags", {}).get("needs_review") else QtCore.Qt.Unchecked)
            self.table.setItem(row, 3, check_item)

    def _rerender(self) -> None:
        if self._rerender_thread is not None:
            return
        page_id = str(self._page.get("page_id") or "").strip()
        project_path = str(self._json_path or "").strip()
        if not project_path or not os.path.isfile(project_path):
            self.rerender_status.setText(
                "Save or open this project before requesting a preview."
            )
            return
        if not page_id:
            self.rerender_status.setText(
                "This page has no stable page identity and cannot be previewed."
            )
            return
        try:
            command = PageRerenderCommand(project_path, page_id)
            state = self._rerender_model.begin(command)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.rerender_status.setText(str(exc))
            return

        thread = QtCore.QThread(self)
        worker = PageRerenderWorker(command)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.preflight.connect(self._on_rerender_preflight)
        worker.progress.connect(self._on_rerender_progress)
        worker.preview_lease.connect(self._on_rerender_preview_lease)
        worker.receipt.connect(self._on_rerender_receipt)
        worker.failure.connect(self._on_rerender_failure)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_rerender_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._rerender_thread = thread
        self._rerender_worker = worker
        self._apply_rerender_state(state)
        thread.start()

    @QtCore.Slot(object)
    def _on_rerender_preflight(self, preflight: object) -> None:
        try:
            state = self._rerender_model.accept_preflight(preflight)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.rerender_status.setText(str(exc))
            return
        self._apply_rerender_state(state)

    @QtCore.Slot(object)
    def _on_rerender_progress(self, progress: object) -> None:
        try:
            state = self._rerender_model.accept_progress(progress)
        except (RuntimeError, TypeError, ValueError):
            return
        self._apply_rerender_state(state)

    @QtCore.Slot(object)
    def _on_rerender_preview_lease(self, lease: object) -> None:
        if not isinstance(lease, PageRerenderPreviewLease):
            return
        previous = self._preview_lease
        self._preview_lease = lease
        if previous is not None and previous != lease:
            discard_preview_lease(previous)

    @QtCore.Slot(object)
    def _on_rerender_receipt(self, receipt: object) -> None:
        try:
            state = self._rerender_model.accept_receipt(receipt)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.rerender_status.setText(str(exc))
            return
        output_path = str(getattr(receipt, "output_path", "") or "")
        if output_path and os.path.isfile(output_path):
            pixmap = QtGui.QPixmap(output_path)
            if not pixmap.isNull():
                self.translated_view.setPixmap(pixmap)
                self.translated_label.setText("Effective preview")
        self._apply_rerender_state(state)

    @QtCore.Slot(object)
    def _on_rerender_failure(self, failure: object) -> None:
        if not isinstance(failure, PageRerenderFailure):
            return
        try:
            state = self._rerender_model.accept_failure(failure)
        except (RuntimeError, TypeError, ValueError):
            self.rerender_status.setText(failure.message)
            return
        self._apply_rerender_state(state)

    @QtCore.Slot()
    def _on_rerender_thread_finished(self) -> None:
        self._rerender_worker = None
        self._rerender_thread = None
        self._apply_rerender_state(self._rerender_model.state)
        if self._close_after_rerender:
            self._close_after_rerender = False
            self._discard_preview_output()
            QtWidgets.QDialog.accept(self)

    def _apply_rerender_state(self, state: PageRerenderViewState) -> None:
        worker_running = self._rerender_thread is not None
        cancellable = worker_running and state.cancel_enabled
        self.render_btn.setEnabled(state.preview_enabled and not worker_running)
        self.cancel_render_btn.setEnabled(cancellable)
        self.table.setEnabled(not worker_running)
        self._update_manual_cleanup_action()
        self.rerender_status.setText(state.message or "Ready to preview.")
        progress = state.progress
        if worker_running and (progress is None or progress.total_parents <= 0):
            self.rerender_progress.setRange(0, 0)
        else:
            total = max(1, int(getattr(progress, "total_parents", 1) or 1))
            completed = min(
                total,
                max(0, int(getattr(progress, "completed_parents", 0) or 0)),
            )
            self.rerender_progress.setRange(0, total)
            self.rerender_progress.setValue(completed)

    def _cancel_rerender(self) -> None:
        worker = self._rerender_worker
        if worker is None:
            return
        worker.request_cancel()
        self.cancel_render_btn.setEnabled(False)
        self.rerender_status.setText("Cancelling preview at the next safe point...")

    def _request_close(self) -> None:
        if self._rerender_thread is not None:
            self._close_after_rerender = True
            self._cancel_rerender()
            return
        self._discard_preview_output()
        self.accept()

    def reject(self) -> None:
        if self._rerender_thread is not None:
            self._close_after_rerender = True
            self._cancel_rerender()
            return
        self._discard_preview_output()
        super().reject()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._rerender_thread is not None:
            self._close_after_rerender = True
            self._cancel_rerender()
            event.ignore()
            return
        self._discard_preview_output()
        super().closeEvent(event)

    def _discard_preview_output(self) -> None:
        lease = self._preview_lease
        self._preview_lease = None
        if lease is None:
            return
        try:
            discard_preview_lease(lease)
        except (OSError, TypeError, ValueError):
            pass
