# -*- coding: utf-8 -*-
"""Style guide editor dialog."""
from __future__ import annotations
from typing import Any, Dict
from PySide6 import QtWidgets
from app.io.style_guide import default_style_guide, load_style_guide, save_style_guide


class StyleGuideEditor(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Style Guide Editor")
        self.resize(720, 520)
        self._path = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.tone_combo = QtWidgets.QComboBox()
        self.tone_combo.addItems(["neutral", "formal", "casual", "classic", "modern"])
        form.addRow("Tone", self.tone_combo)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("World/Setting Notes"))
        self.notes_edit = QtWidgets.QPlainTextEdit()
        layout.addWidget(self.notes_edit)

        layout.addWidget(QtWidgets.QLabel("Glossary"))
        self.glossary_table = QtWidgets.QTableWidget(0, 4)
        self.glossary_table.setHorizontalHeaderLabels(["Source", "Target", "Notes", "Priority"])
        self.glossary_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.glossary_table)

        glossary_btns = QtWidgets.QHBoxLayout()
        self.add_glossary = QtWidgets.QPushButton("Add")
        self.remove_glossary = QtWidgets.QPushButton("Remove")
        glossary_btns.addWidget(self.add_glossary)
        glossary_btns.addWidget(self.remove_glossary)
        glossary_btns.addStretch(1)
        layout.addLayout(glossary_btns)

        list_layout = QtWidgets.QHBoxLayout()
        left_box = QtWidgets.QVBoxLayout()
        left_box.addWidget(QtWidgets.QLabel("Required Terms (one per line)"))
        self.required_edit = QtWidgets.QPlainTextEdit()
        left_box.addWidget(self.required_edit)
        right_box = QtWidgets.QVBoxLayout()
        right_box.addWidget(QtWidgets.QLabel("Forbidden Terms (one per line)"))
        self.forbidden_edit = QtWidgets.QPlainTextEdit()
        right_box.addWidget(self.forbidden_edit)
        list_layout.addLayout(left_box)
        list_layout.addLayout(right_box)
        layout.addLayout(list_layout)

        footer = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.close_btn = QtWidgets.QPushButton("Close")
        footer.addWidget(self.load_btn)
        footer.addWidget(self.save_btn)
        footer.addStretch(1)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

        self.add_glossary.clicked.connect(self._add_glossary_row)
        self.remove_glossary.clicked.connect(self._remove_glossary_row)
        self.load_btn.clicked.connect(self._load_from_file)
        self.save_btn.clicked.connect(self._save_to_file)
        self.close_btn.clicked.connect(self.accept)

    def set_path(self, path: str) -> None:
        self._path = path

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.tone_combo.setCurrentText(data.get("tone", "neutral"))
        self.notes_edit.setPlainText(data.get("notes", ""))

        self.glossary_table.setRowCount(0)
        for entry in data.get("glossary", []):
            self._add_glossary_row(entry)

        self.required_edit.setPlainText("\n".join(data.get("required_terms", [])))
        self.forbidden_edit.setPlainText("\n".join(data.get("forbidden_terms", [])))

    def load_from_path(self, path: str) -> None:
        data = load_style_guide(path)
        self._path = path
        self.load_from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data = default_style_guide()
        data["tone"] = self.tone_combo.currentText()
        data["notes"] = self.notes_edit.toPlainText().strip()

        glossary = []
        for row in range(self.glossary_table.rowCount()):
            glossary.append(
                {
                    "source": self._table_text(row, 0),
                    "target": self._table_text(row, 1),
                    "notes": self._table_text(row, 2),
                    "priority": self._table_text(row, 3) or "soft",
                }
            )
        data["glossary"] = glossary

        data["required_terms"] = self._lines(self.required_edit)
        data["forbidden_terms"] = self._lines(self.forbidden_edit)
        return data

    def _add_glossary_row(self, entry: Dict[str, Any] | None = None) -> None:
        row = self.glossary_table.rowCount()
        self.glossary_table.insertRow(row)
        values = ["", "", "", "soft"]
        if entry:
            values = [
                str(entry.get("source", "")),
                str(entry.get("target", "")),
                str(entry.get("notes", "")),
                str(entry.get("priority", "soft")),
            ]
        for col, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            self.glossary_table.setItem(row, col, item)

    def _remove_glossary_row(self) -> None:
        row = self.glossary_table.currentRow()
        if row >= 0:
            self.glossary_table.removeRow(row)

    def _load_from_file(self) -> None:
        path = self._path
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Style Guide", filter="JSON Files (*.json)")
        if not path:
            return
        self.load_from_path(path)

    def _save_to_file(self) -> None:
        path = self._path
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Style Guide", filter="JSON Files (*.json)")
        if not path:
            return
        save_style_guide(path, self.to_dict())
        self._path = path

    def _table_text(self, row: int, col: int) -> str:
        item = self.glossary_table.item(row, col)
        return item.text().strip() if item else ""

    def _lines(self, edit: QtWidgets.QPlainTextEdit) -> list[str]:
        return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]
