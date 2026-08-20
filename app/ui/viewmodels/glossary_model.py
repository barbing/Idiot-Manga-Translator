"""Pure UI-thread model for the project glossary Settings surface."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import os
import unicodedata
import uuid
from typing import TYPE_CHECKING, Any

from app.project_edits.glossary_commands import GlossaryEntryV1

if TYPE_CHECKING:
    from app.ui.shell.project_projection import EditHistoryReference, ProjectUiProjection


class GlossaryEditorPhase(str, Enum):
    READY = "ready"
    DIRTY = "dirty"
    BUSY = "busy"
    COMMITTED = "committed"
    EXPORTED = "exported"
    CANCELLED = "cancelled"
    STALE = "stale"
    FAILED = "failed"


class GlossaryWorkerOperation(str, Enum):
    SET_ENTRY = "set_entry"
    REMOVE_ENTRY = "remove_entry"
    IMPORT_FILE = "import_file"
    EXPORT_FILE = "export_file"
    HISTORY_REVOKE = "history_revoke"
    HISTORY_REAPPLY = "history_reapply"


class GlossaryWorkerStage(str, Enum):
    LOADING_PROJECT = "loading_project"
    OPENING_EDIT_STORE = "opening_edit_store"
    READING_SNAPSHOT = "reading_snapshot"
    READING_FILE = "reading_file"
    PREPARING_COMMAND = "preparing_command"
    PERSISTING = "persisting"
    MATERIALIZING_PROJECT = "materializing_project"
    BUILDING_UI_PROJECTION = "building_ui_projection"
    WRITING_FILE = "writing_file"
    CLOSING_EDIT_STORE = "closing_edit_store"
    COMPLETE = "complete"


class GlossaryWorkerFailureCode(str, Enum):
    PROJECT_LOAD_FAILED = "project_load_failed"
    PROJECT_INVALID = "project_invalid"
    EDIT_STORE_FAILED = "edit_store_failed"
    FILE_INVALID = "file_invalid"
    FILE_WRITE_FAILED = "file_write_failed"
    COMMAND_REJECTED = "command_rejected"
    SNAPSHOT_STALE = "snapshot_stale"
    DUPLICATE_TERM = "duplicate_term"
    HISTORY_REJECTED = "history_rejected"
    POST_COMMIT_PROJECTION_FAILED = "post_commit_projection_failed"
    WORKER_REUSED = "worker_reused"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _required_sha256(value: object, field_name: str) -> str:
    normalized = _required_text(value, field_name).lower()
    if len(normalized) != 64 or any(value not in "0123456789abcdef" for value in normalized):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return normalized


def _term_key(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip()).casefold()


@dataclass(frozen=True, slots=True)
class GlossarySelection:
    project_path: str
    project_id: str
    anchor_page_id: str
    anchor_page_fingerprint: str
    entries: tuple[GlossaryEntryV1, ...]
    glossary_fingerprint: str
    history: tuple["EditHistoryReference", ...]
    stale_page_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        path = os.path.abspath(_required_text(self.project_path, "project_path"))
        object.__setattr__(self, "project_path", path)
        for field_name in ("project_id", "anchor_page_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        for field_name in ("anchor_page_fingerprint", "glossary_fingerprint"):
            object.__setattr__(self, field_name, _required_sha256(getattr(self, field_name), field_name))
        if any(not isinstance(entry, GlossaryEntryV1) for entry in self.entries):
            raise TypeError("entries must contain GlossaryEntryV1 values")
        if any(
            not all(hasattr(item, name) for name in ("record_id", "domain", "operation", "active", "is_control"))
            for item in self.history
        ):
            raise TypeError("history must contain glossary History references")
        if len(self.stale_page_ids) != len(set(self.stale_page_ids)):
            raise ValueError("stale page identities must be unique")


@dataclass(frozen=True, slots=True)
class GlossaryDraft:
    entry_id: str
    source: str
    target: str
    notes: str
    aliases: tuple[str, ...]
    priority: str

    @classmethod
    def from_entry(cls, entry: GlossaryEntryV1) -> "GlossaryDraft":
        return cls(**entry.to_dict() | {"aliases": entry.aliases})

    def to_entry(self) -> GlossaryEntryV1:
        return GlossaryEntryV1(
            entry_id=self.entry_id,
            source=self.source,
            target=self.target,
            notes=self.notes,
            aliases=self.aliases,
            priority=self.priority,
        )


@dataclass(frozen=True, slots=True)
class GlossaryWorkerCommand:
    project_path: str
    project_id: str
    anchor_page_id: str
    operation: GlossaryWorkerOperation
    expected_glossary_fingerprint: str
    expected_anchor_page_fingerprint: str
    entries: tuple[GlossaryEntryV1, ...] = ()
    entry_ids: tuple[str, ...] = ()
    history_edit_id: str = ""
    file_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_path", os.path.abspath(_required_text(self.project_path, "project_path")))
        for field_name in ("project_id", "anchor_page_id"):
            object.__setattr__(self, field_name, _required_text(getattr(self, field_name), field_name))
        object.__setattr__(self, "operation", GlossaryWorkerOperation(self.operation))
        for field_name in ("expected_glossary_fingerprint", "expected_anchor_page_fingerprint"):
            object.__setattr__(self, field_name, _required_sha256(getattr(self, field_name), field_name))
        if any(not isinstance(entry, GlossaryEntryV1) for entry in self.entries):
            raise TypeError("entries must contain GlossaryEntryV1 values")
        if self.operation is GlossaryWorkerOperation.SET_ENTRY and len(self.entries) != 1:
            raise ValueError("set_entry requires exactly one entry")
        if self.operation is GlossaryWorkerOperation.REMOVE_ENTRY and len(self.entry_ids) != 1:
            raise ValueError("remove_entry requires exactly one entry ID")
        if self.operation is GlossaryWorkerOperation.IMPORT_FILE and not self.file_path:
            raise ValueError("import_file requires a path")
        if self.operation is GlossaryWorkerOperation.EXPORT_FILE and not self.file_path:
            raise ValueError("export_file requires a path")
        if self.operation in {GlossaryWorkerOperation.HISTORY_REVOKE, GlossaryWorkerOperation.HISTORY_REAPPLY} and not self.history_edit_id:
            raise ValueError("history operation requires an edit identity")


@dataclass(frozen=True, slots=True)
class GlossaryWorkerBusyState:
    stage: GlossaryWorkerStage
    busy: bool
    cancellation_enabled: bool
    persistence_started: bool
    message: str


@dataclass(frozen=True, slots=True)
class GlossaryWorkerFailure:
    code: GlossaryWorkerFailureCode
    stage: GlossaryWorkerStage
    command: GlossaryWorkerCommand
    message: str
    exception_type: str = ""
    persistence_committed: bool = False
    command_receipt: object | None = None

    @property
    def stale(self) -> bool:
        return self.code in {GlossaryWorkerFailureCode.SNAPSHOT_STALE, GlossaryWorkerFailureCode.POST_COMMIT_PROJECTION_FAILED} or self.persistence_committed


@dataclass(frozen=True, slots=True)
class GlossaryCancelledReceipt:
    command: GlossaryWorkerCommand
    stage: GlossaryWorkerStage
    message: str = "Glossary action cancelled before persistence."


@dataclass(frozen=True, slots=True)
class GlossaryWorkerReceipt:
    command: GlossaryWorkerCommand
    command_receipt: object
    project: dict
    projection: "ProjectUiProjection"


@dataclass(frozen=True, slots=True)
class GlossaryExportReceipt:
    command: GlossaryWorkerCommand
    exported_path: str
    entry_count: int


@dataclass(frozen=True, slots=True)
class GlossaryEditorState:
    selection: GlossarySelection
    phase: GlossaryEditorPhase
    selected_entry_id: str = ""
    draft: GlossaryDraft | None = None
    selected_stale_page_ids: tuple[str, ...] = ()
    message: str = ""
    worker_command: GlossaryWorkerCommand | None = None

    @property
    def busy(self) -> bool:
        return self.phase is GlossaryEditorPhase.BUSY

    @property
    def dirty(self) -> bool:
        return self.phase is GlossaryEditorPhase.DIRTY


def glossary_selection_from_projection(
    project_path: str,
    projection: "ProjectUiProjection",
) -> GlossarySelection:
    required = (
        "metadata",
        "pages",
        "glossary_entries",
        "glossary_fingerprint",
        "glossary_history",
        "glossary_stale_page_ids",
    )
    if any(not hasattr(projection, name) for name in required):
        raise TypeError("projection must expose the project glossary UI contract")
    if not projection.pages:
        raise ValueError("project projection has no glossary anchor page")
    anchor = projection.pages[0]
    return GlossarySelection(
        project_path=project_path,
        project_id=projection.metadata.project_id,
        anchor_page_id=anchor.effective.page_id,
        anchor_page_fingerprint=anchor.effective.effective_fingerprint,
        entries=projection.glossary_entries,
        glossary_fingerprint=projection.glossary_fingerprint,
        history=projection.glossary_history,
        stale_page_ids=projection.glossary_stale_page_ids,
    )


class GlossaryEditorModel:
    def __init__(self, selection: GlossarySelection) -> None:
        if not isinstance(selection, GlossarySelection):
            raise TypeError("selection must be a GlossarySelection")
        self._state = GlossaryEditorState(
            selection=selection,
            phase=GlossaryEditorPhase.READY,
            selected_stale_page_ids=(),
            message=self._ready_message(selection),
        )

    @property
    def state(self) -> GlossaryEditorState:
        return self._state

    def filtered_entries(self, query: str) -> tuple[GlossaryEntryV1, ...]:
        needle = _term_key(str(query or ""))
        entries = tuple(
            sorted(
                self._state.selection.entries,
                key=lambda entry: (
                    0 if entry.priority == "hard" else 1,
                    _term_key(entry.source),
                    entry.entry_id,
                ),
            )
        )
        if not needle:
            return entries
        return tuple(
            entry
            for entry in entries
            if needle in _term_key(" ".join((entry.source, entry.target, entry.notes, *entry.aliases)))
        )

    def select_entry(self, entry_id: str) -> GlossaryEditorState:
        if self._state.busy or self._state.dirty:
            raise RuntimeError("finish or cancel the current glossary draft first")
        matches = tuple(entry for entry in self._state.selection.entries if entry.entry_id == entry_id)
        if len(matches) != 1:
            raise KeyError(f"glossary entry is missing: {entry_id}")
        self._state = replace(
            self._state,
            selected_entry_id=entry_id,
            draft=GlossaryDraft.from_entry(matches[0]),
            phase=GlossaryEditorPhase.READY,
            message="Selected project glossary entry.",
        )
        return self._state

    def begin_new(self, *, entry_id: str | None = None) -> GlossaryEditorState:
        if self._state.busy or self._state.dirty:
            raise RuntimeError("finish or cancel the current glossary draft first")
        identity = entry_id or f"glossary-entry-{uuid.uuid4().hex}"
        self._state = replace(
            self._state,
            selected_entry_id=identity,
            draft=GlossaryDraft(identity, "", "", "", (), "soft"),
            phase=GlossaryEditorPhase.DIRTY,
            message="Enter the source and required target, then save.",
        )
        return self._state

    def update_draft(
        self,
        *,
        source: str,
        target: str,
        notes: str,
        aliases: tuple[str, ...],
        priority: str,
    ) -> GlossaryEditorState:
        if self._state.busy or self._state.draft is None:
            raise RuntimeError("no glossary draft is editable")
        draft = replace(
            self._state.draft,
            source=str(source),
            target=str(target),
            notes=str(notes),
            aliases=tuple(aliases),
            priority=str(priority),
        )
        self._state = replace(self._state, draft=draft, phase=GlossaryEditorPhase.DIRTY, message="Glossary draft has unsaved changes.")
        return self._state

    def draft_problem(self) -> str:
        draft = self._state.draft
        if draft is None:
            return "Select or create a glossary entry."
        try:
            entry = draft.to_entry()
        except (TypeError, ValueError) as exc:
            return str(exc)
        owner_by_key = {
            _term_key(value): existing
            for existing in self._state.selection.entries
            if existing.entry_id != entry.entry_id
            for value in (existing.source, *existing.aliases)
        }
        for value in (entry.source, *entry.aliases):
            owner = owner_by_key.get(_term_key(value))
            if owner is not None:
                return (
                    f'“{value}” overlaps the existing {owner.source} alias group. '
                    "Resolve the duplicate before export."
                )
        return ""

    def cancel_draft(self) -> GlossaryEditorState:
        if self._state.busy:
            raise RuntimeError("cannot cancel the draft while a worker is active")
        selected = next((entry for entry in self._state.selection.entries if entry.entry_id == self._state.selected_entry_id), None)
        self._state = replace(
            self._state,
            draft=(GlossaryDraft.from_entry(selected) if selected is not None else None),
            selected_entry_id=(selected.entry_id if selected is not None else ""),
            phase=GlossaryEditorPhase.CANCELLED,
            message="Glossary draft cancelled.",
        )
        return self._state

    def _base_command(self, operation: GlossaryWorkerOperation, **kwargs) -> GlossaryWorkerCommand:
        selection = self._state.selection
        return GlossaryWorkerCommand(
            project_path=selection.project_path,
            project_id=selection.project_id,
            anchor_page_id=selection.anchor_page_id,
            operation=operation,
            expected_glossary_fingerprint=selection.glossary_fingerprint,
            expected_anchor_page_fingerprint=selection.anchor_page_fingerprint,
            **kwargs,
        )

    def begin_save(self) -> GlossaryWorkerCommand:
        if self._state.busy or self._state.draft is None:
            raise RuntimeError("no glossary draft can be saved")
        problem = self.draft_problem()
        if problem:
            raise ValueError(problem)
        return self._begin(self._base_command(GlossaryWorkerOperation.SET_ENTRY, entries=(self._state.draft.to_entry(),)))

    def begin_remove(self) -> GlossaryWorkerCommand:
        if self._state.busy or self._state.dirty or not any(entry.entry_id == self._state.selected_entry_id for entry in self._state.selection.entries):
            raise RuntimeError("select a persisted glossary entry to remove")
        return self._begin(self._base_command(GlossaryWorkerOperation.REMOVE_ENTRY, entry_ids=(self._state.selected_entry_id,)))

    def begin_import(self, path: str) -> GlossaryWorkerCommand:
        if self._state.dirty:
            raise RuntimeError("save or cancel the glossary draft before importing")
        return self._begin(self._base_command(GlossaryWorkerOperation.IMPORT_FILE, file_path=os.path.abspath(path)))

    def begin_export(self, path: str) -> GlossaryWorkerCommand:
        if self._state.dirty:
            raise RuntimeError("save or cancel the glossary draft before exporting")
        return self._begin(self._base_command(GlossaryWorkerOperation.EXPORT_FILE, file_path=os.path.abspath(path)))

    def begin_history(self, edit_id: str) -> GlossaryWorkerCommand:
        if self._state.dirty:
            raise RuntimeError("save or cancel the glossary draft before using History")
        matches = tuple(item for item in self._state.selection.history if item.record_id == edit_id and not item.is_control)
        if len(matches) != 1 or matches[0].domain != "glossary":
            raise RuntimeError("select one project glossary history edit")
        operation = GlossaryWorkerOperation.HISTORY_REVOKE if matches[0].active else GlossaryWorkerOperation.HISTORY_REAPPLY
        return self._begin(self._base_command(operation, history_edit_id=edit_id))

    def set_stale_page_selected(self, page_id: str, selected: bool) -> GlossaryEditorState:
        if page_id not in self._state.selection.stale_page_ids:
            raise KeyError(f"stale glossary page is missing: {page_id}")
        current = list(self._state.selected_stale_page_ids)
        if selected and page_id not in current:
            current.append(page_id)
        elif not selected and page_id in current:
            current.remove(page_id)
        ordered = tuple(page for page in self._state.selection.stale_page_ids if page in current)
        self._state = replace(self._state, selected_stale_page_ids=ordered)
        return self._state

    def accept_success(self, message: str) -> GlossaryEditorState:
        self._state = replace(self._state, phase=GlossaryEditorPhase.COMMITTED, message=message, worker_command=None)
        return self._state

    def accept_failure(self, failure: GlossaryWorkerFailure) -> GlossaryEditorState:
        self._state = replace(self._state, phase=(GlossaryEditorPhase.STALE if failure.stale else GlossaryEditorPhase.FAILED), message=failure.message, worker_command=None)
        return self._state

    def accept_cancelled(self, receipt: GlossaryCancelledReceipt) -> GlossaryEditorState:
        self._state = replace(self._state, phase=GlossaryEditorPhase.CANCELLED, message=receipt.message, worker_command=None)
        return self._state

    def _begin(self, command: GlossaryWorkerCommand) -> GlossaryWorkerCommand:
        if self._state.busy:
            raise RuntimeError("a glossary worker is already active")
        self._state = replace(self._state, phase=GlossaryEditorPhase.BUSY, message="Applying project glossary action...", worker_command=command)
        return command

    @staticmethod
    def _ready_message(selection: GlossarySelection) -> str:
        if selection.entries:
            return f"{len(selection.entries)} project glossary entries."
        return "No project glossary entries. Add or import one to begin."


__all__ = [
    "GlossaryCancelledReceipt",
    "GlossaryDraft",
    "GlossaryEditorModel",
    "GlossaryEditorPhase",
    "GlossaryEditorState",
    "GlossaryExportReceipt",
    "GlossarySelection",
    "GlossaryWorkerBusyState",
    "GlossaryWorkerCommand",
    "GlossaryWorkerFailure",
    "GlossaryWorkerFailureCode",
    "GlossaryWorkerOperation",
    "GlossaryWorkerReceipt",
    "GlossaryWorkerStage",
    "glossary_selection_from_projection",
]
