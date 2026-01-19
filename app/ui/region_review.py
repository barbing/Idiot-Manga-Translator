# -*- coding: utf-8 -*-
"""Region review dialog for quick ignore toggles."""
from __future__ import annotations
from typing import Any, Dict, Tuple
from PySide6 import QtCore, QtWidgets
from app.io.project import load_project, save_project


class RegionReviewDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Region Review")
        self.resize(900, 600)
        self._path = ""
        self._data: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open JSON")
        self.save_btn = QtWidgets.QPushButton("Save")
        header.addWidget(self.open_btn)
        header.addWidget(self.save_btn)
        header.addStretch(1)
        layout.addLayout(header)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Page",
            "Region",
            "OCR",
            "Translation",
            "Ignore",
            "Needs Review",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        footer = QtWidgets.QHBoxLayout()
        self.close_btn = QtWidgets.QPushButton("Close")
        footer.addStretch(1)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

        self.open_btn.clicked.connect(self._open)
        self.save_btn.clicked.connect(self._save)
        self.close_btn.clicked.connect(self.accept)

    def set_path(self, path: str) -> None:
        self._path = path
        if path:
            self._load(path)

    def _open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Project JSON", filter="JSON Files (*.json)")
        if not path:
            return
        self._path = path
        self._load(path)

    def _load(self, path: str) -> None:
        self._data = load_project(path)
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        pages = self._data.get("pages", [])
        for page_index, page in enumerate(pages):
            page_id = page.get("page_id", str(page_index))
            for region_index, region in enumerate(page.get("regions", [])):
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._set_text(row, 0, str(page_id))
                self._set_text(row, 1, str(region.get("region_id", region_index)))
                self._set_text(row, 2, str(region.get("ocr_text", "")))
                self._set_text(row, 3, str(region.get("translation", "")))
                self._set_check(row, 4, bool(region.get("flags", {}).get("ignore")))
                self._set_check(row, 5, bool(region.get("flags", {}).get("needs_review")))
                self.table.item(row, 0).setData(QtCore.Qt.UserRole, (page_index, region_index))

    def _save(self) -> None:
        if not self._path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project JSON", filter="JSON Files (*.json)")
            if not path:
                return
            self._path = path
        self._apply_changes()
        save_project(self._path, self._data)

    def _apply_changes(self) -> None:
        pages = self._data.get("pages", [])
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item:
                continue
            page_index, region_index = item.data(QtCore.Qt.UserRole)
            region = pages[page_index]["regions"][region_index]
            flags = region.get("flags", {})
            flags["ignore"] = self._check_state(row, 4)
            flags["needs_review"] = self._check_state(row, 5)
            region["flags"] = flags

    def _set_text(self, row: int, col: int, text: str) -> None:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(row, col, item)

    def _set_check(self, row: int, col: int, checked: bool) -> None:
        item = QtWidgets.QTableWidgetItem("")
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self.table.setItem(row, col, item)

    def _check_state(self, row: int, col: int) -> bool:
        item = self.table.item(row, col)
        if not item:
            return False
        return item.checkState() == QtCore.Qt.Checked
