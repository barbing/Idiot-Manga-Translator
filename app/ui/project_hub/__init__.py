"""Project Hub surface for the GUI-5 native shell."""

from .view import ProjectHubView
from .new_project_dialog import (
    NewProjectDialog,
    named_project_display_name,
    named_project_filename,
)

__all__ = [
    "NewProjectDialog",
    "ProjectHubView",
    "named_project_display_name",
    "named_project_filename",
]
