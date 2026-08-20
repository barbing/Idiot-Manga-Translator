# -*- coding: utf-8 -*-
"""Prototype-faithful, GUI-owned new-project naming dialog."""
from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtWidgets

from app.ui.design_system.dialogs import HybridDialog
from app.ui.design_system.icons import hybrid_icon


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def named_project_filename(project_name: str) -> str:
    """Return one safe Windows filename while retaining the visible name."""

    value = str(project_name or "").strip()
    if not value:
        raise ValueError("project name must not be empty")
    value = _INVALID_FILENAME.sub("_", value).rstrip(" .")
    if not value:
        raise ValueError("project name contains no usable filename characters")
    if value.casefold().endswith(".yomiframe.json"):
        value = value[: -len(".yomiframe.json")].rstrip(" .")
    if value.upper() in _RESERVED_FILENAMES:
        value = f"_{value}"
    value = value[:160].rstrip(" .")
    if not value:
        raise ValueError("project name contains no usable filename characters")
    return f"{value}.yomiframe.json"


def named_project_display_name(project_path: str) -> str:
    """Recover a GUI-authored name from its durable named-file convention."""

    filename = Path(str(project_path or "")).name
    suffix = ".yomiframe.json"
    if filename.casefold().endswith(suffix):
        value = filename[: -len(suffix)].strip()
        if value:
            return value
    return Path(filename).stem or "Project"


class NewProjectDialog(HybridDialog):
    """Collect the user-facing project identity before filesystem selection."""

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("newProjectDialog")
        self.setWindowTitle("Create project")
        self.setModal(True)
        self.setMinimumWidth(470)
        self.setAccessibleName("Create project")
        self.setAccessibleDescription(
            "Name the local project before selecting source and output folders."
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(16)

        root.addWidget(
            self.create_dialog_header(
                title="Create project",
                subtitle=(
                    "Name the project now; source and output folders are selected "
                    "next. Languages can be changed later in Settings > General."
                ),
                icon_name="new",
                close_accessible_name="Cancel project creation",
            )
        )

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.project_name = QtWidgets.QLineEdit("Untitled manga")
        self.project_name.setObjectName("newProjectName")
        self.project_name.setAccessibleName("Project name")
        self.project_name.setAccessibleDescription(
            "Required name used by the Workspace, Editor, recent projects, and project file."
        )
        self.project_name.textChanged.connect(self._refresh_create_state)
        form.addRow("Project name", self.project_name)
        root.addLayout(form)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setProperty("role", "command")
        self.cancel_button.setProperty("variant", "quiet")
        self.cancel_button.clicked.connect(self.reject)
        self.create_button = QtWidgets.QPushButton("Create project")
        self.create_button.setProperty("role", "command")
        self.create_button.setProperty("variant", "primary")
        self.create_button.setIcon(hybrid_icon("new", active=True))
        self.create_button.setAccessibleName("Create project")
        self.create_button.setAccessibleDescription(
            "Continue to source and output folder selection for this named project."
        )
        self.create_button.clicked.connect(self.accept)
        self.create_button.setDefault(True)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.create_button)
        root.addLayout(footer)

        QtWidgets.QWidget.setTabOrder(self.project_name, self.cancel_button)
        QtWidgets.QWidget.setTabOrder(self.cancel_button, self.create_button)
        if self.dialog_header is not None:
            QtWidgets.QWidget.setTabOrder(
                self.create_button,
                self.dialog_header.close_button,
            )
        self._refresh_create_state()
        self.project_name.selectAll()

    @property
    def selected_project_name(self) -> str:
        return self.project_name.text().strip()

    def _refresh_create_state(self, *_args: object) -> None:
        try:
            named_project_filename(self.selected_project_name)
        except ValueError:
            ready = False
        else:
            ready = True
        self.create_button.setEnabled(ready)


__all__ = [
    "NewProjectDialog",
    "named_project_display_name",
    "named_project_filename",
]
