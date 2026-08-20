"""Page Editor surfaces for the GUI-5 native shell."""

from .activity_dock import ActivityDock
from .canvas import CanvasArtifactSet, OverlayShape, PageCanvasView
from .view import PageEditorView

__all__ = [
    "ActivityDock",
    "CanvasArtifactSet",
    "OverlayShape",
    "PageCanvasView",
    "PageEditorView",
]
