"""Project-scoped glossary edit commands for the native GUI.

This module owns durable edit-layer state only.  It never invokes translation,
automatic glossary discovery, a provider, or any other pipeline owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from app.io.project_edit_store import (
    ProjectEditCommitReceipt,
    ProjectEditReadSnapshot,
    ProjectEditStore,
    StalePageEditHeadError,
    StaleProjectEditHeadError,
)

from .contracts import (
    EditDomain,
    EditTarget,
    EditTargetKind,
    ProjectEdit,
    create_project_edit,
)
from .fingerprints import canonical_sha256, project_id_for
from .invalidation import InvalidationResult, invalidation_for_edit
from .ledger import ProjectEditLedger
from .projection import (
    EffectivePageSnapshot,
    ProjectionIssue,
    automatic_page_fingerprint,
    field_base_fingerprint,
    project_effective_page,
)


_PATH_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PROJECT_ID = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,127}|project:[A-Za-z0-9][A-Za-z0-9._:-]{0,127})"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _required_identity(value: object, field_name: str) -> str:
    normalized = _required_text(value, field_name)
    if _PATH_SAFE_ID.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be path-safe")
    return normalized


def _required_project_identity(value: object) -> str:
    """Accept repository-canonical project IDs without widening slot IDs."""

    normalized = _required_text(value, "project_id")
    if _PROJECT_ID.fullmatch(normalized) is None:
        raise ValueError("project_id must be a canonical project identity")
    return normalized


def _required_sha256(value: object, field_name: str) -> str:
    normalized = _required_text(value, field_name).lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return normalized


def _term_key(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip()).casefold()


@dataclass(frozen=True, slots=True)
class GlossaryEntryV1:
    """One strict user-facing glossary entry.

    ``soft`` and ``hard`` are the exact priority values already understood by
    the existing translation owner.  Duplicate-key normalization is validation
    only; it does not change runtime matching semantics.
    """

    entry_id: str
    source: str
    target: str
    notes: str = ""
    aliases: tuple[str, ...] = ()
    priority: str = "soft"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entry_id",
            _required_identity(self.entry_id, "entry_id"),
        )
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        object.__setattr__(self, "target", _required_text(self.target, "target"))
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        object.__setattr__(self, "notes", self.notes.strip())
        if not isinstance(self.aliases, tuple):
            raise TypeError("aliases must be an exact tuple")
        aliases = tuple(_required_text(value, "alias") for value in self.aliases)
        keys = tuple(_term_key(value) for value in (self.source, *aliases))
        if len(keys) != len(set(keys)):
            raise ValueError("source and aliases contain a duplicate term")
        object.__setattr__(self, "aliases", aliases)
        if self.priority not in {"soft", "hard"}:
            raise ValueError("priority must be exactly 'soft' or 'hard'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source": self.source,
            "target": self.target,
            "notes": self.notes,
            "aliases": list(self.aliases),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlossaryEntryV1":
        if not isinstance(value, Mapping):
            raise TypeError("glossary entry must be a mapping")
        required = {"entry_id", "source", "target"}
        optional = {"notes", "aliases", "priority"}
        missing = required - set(value)
        unknown = set(value) - required - optional
        if missing:
            raise ValueError(f"glossary entry is missing fields: {sorted(missing)}")
        if unknown:
            raise ValueError(
                f"glossary entry has unsupported fields: {sorted(unknown)}"
            )
        aliases = value.get("aliases", ())
        if (
            not isinstance(aliases, Sequence)
            or isinstance(aliases, (str, bytes, bytearray))
        ):
            raise TypeError("aliases must be a sequence of strings")
        priority = value.get("priority", "soft")
        if isinstance(priority, bool):
            raise TypeError("priority must not be a boolean")
        if isinstance(priority, int):
            priority = "hard" if priority > 0 else "soft"
        return cls(
            entry_id=str(value.get("entry_id") or ""),
            source=str(value.get("source") or ""),
            target=str(value.get("target") or ""),
            notes=str(value.get("notes") or ""),
            aliases=tuple(aliases),
            priority=str(priority),
        )


class GlossaryOperation(str, Enum):
    SET_ENTRY = "set_entry"
    REMOVE_ENTRY = "remove_entry"
    IMPORT_ENTRIES = "import_entries"


class GlossaryCommandErrorCode(str, Enum):
    STORE_IDENTITY_MISMATCH = "store_identity_mismatch"
    PROJECT_IDENTITY_MISMATCH = "project_identity_mismatch"
    ANCHOR_PAGE_MISMATCH = "anchor_page_mismatch"
    PROJECT_GLOSSARY_INVALID = "project_glossary_invalid"
    STALE_GLOSSARY = "stale_glossary"
    STALE_PAGE_HEAD = "stale_page_head"
    STALE_GLOBAL_HEAD = "stale_global_head"
    DUPLICATE_COMMAND = "duplicate_command"
    DUPLICATE_TERM = "duplicate_term"
    SLOT_CONFLICT = "slot_conflict"
    NO_OP = "no_op"
    PROJECTION_REJECTED = "projection_rejected"
    INVALIDATION_REJECTED = "invalidation_rejected"


class GlossaryCommandError(RuntimeError):
    def __init__(self, code: GlossaryCommandErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = GlossaryCommandErrorCode(code)


@dataclass(frozen=True, slots=True)
class ProjectGlossarySnapshot:
    project_id: str
    anchor_page_id: str
    page_ids: tuple[str, ...]
    entries: tuple[GlossaryEntryV1, ...]
    fingerprint: str
    effective_pages: tuple[EffectivePageSnapshot, ...]

    def __post_init__(self) -> None:
        _required_project_identity(self.project_id)
        _required_identity(self.anchor_page_id, "anchor_page_id")
        if not self.page_ids or self.page_ids[0] != self.anchor_page_id:
            raise ValueError("anchor page must be the first project page")
        if len(self.page_ids) != len(set(self.page_ids)):
            raise ValueError("project page identities must be unique")
        if tuple(entry.entry_id for entry in self.entries) != tuple(
            sorted(entry.entry_id for entry in self.entries)
        ):
            raise ValueError("glossary entries must be sorted by entry identity")
        _required_sha256(self.fingerprint, "fingerprint")


def _project_page_ids(project: Mapping[str, Any]) -> tuple[str, ...]:
    pages = project.get("pages")
    if not isinstance(pages, (list, tuple)) or not pages:
        raise ValueError("project pages are unavailable")
    page_ids = tuple(
        _required_identity(
            page.get("page_id") if isinstance(page, Mapping) else None,
            "page.page_id",
        )
        for page in pages
    )
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("project page identities are duplicated")
    return page_ids


def _normalized_entries(
    glossary: Iterable[tuple[str, Any]],
) -> tuple[GlossaryEntryV1, ...]:
    result: list[GlossaryEntryV1] = []
    for entry_id, value in glossary:
        if not isinstance(value, Mapping):
            raise ValueError("effective glossary entries must be mappings")
        data = dict(value)
        data.setdefault("entry_id", str(entry_id))
        if str(data.get("entry_id") or "") != str(entry_id):
            raise ValueError("effective glossary entry identity is inconsistent")
        result.append(GlossaryEntryV1.from_dict(data))
    result.sort(key=lambda entry: entry.entry_id)
    if len({entry.entry_id for entry in result}) != len(result):
        raise ValueError("effective glossary entry identities are duplicated")
    return tuple(result)


def glossary_entries_fingerprint(entries: Iterable[GlossaryEntryV1]) -> str:
    values = tuple(entries)
    if any(not isinstance(entry, GlossaryEntryV1) for entry in values):
        raise TypeError("entries must contain GlossaryEntryV1 values")
    return canonical_sha256([entry.to_dict() for entry in values])


def project_glossary_snapshot(
    project: Mapping[str, Any],
    ledger: ProjectEditLedger,
) -> ProjectGlossarySnapshot:
    """Project one exact glossary and prove every page observes it equally."""

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    if not isinstance(ledger, ProjectEditLedger):
        raise TypeError("ledger must be a ProjectEditLedger")
    project_id = project_id_for(project)
    if ledger.project_id != project_id:
        raise ValueError("project and ledger identities differ")
    page_ids = _project_page_ids(project)
    pages = tuple(
        project_effective_page(project, ledger, page_id=page_id)
        for page_id in page_ids
    )
    glossary_values = tuple(page.effective_glossary for page in pages)
    if any(value != glossary_values[0] for value in glossary_values[1:]):
        raise ValueError("effective glossary differs between project pages")
    glossary_issues = tuple(
        issue
        for page in pages
        for issue in page.issues
        if issue.domain == EditDomain.GLOSSARY.value
    )
    if glossary_issues:
        raise ValueError("effective glossary has unresolved projection issues")
    entries = _normalized_entries(glossary_values[0])
    return ProjectGlossarySnapshot(
        project_id=project_id,
        anchor_page_id=page_ids[0],
        page_ids=page_ids,
        entries=entries,
        fingerprint=glossary_entries_fingerprint(entries),
        effective_pages=pages,
    )


@dataclass(frozen=True, slots=True)
class GlossaryCommand:
    command_id: str
    project_id: str
    anchor_page_id: str
    operation: GlossaryOperation
    entries: tuple[GlossaryEntryV1, ...] = ()
    entry_ids: tuple[str, ...] = ()
    expected_glossary_fingerprint: str = ""
    expected_page_head_sha256: str = ""
    expected_global_head_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_id",
            _required_identity(self.command_id, "command_id"),
        )
        object.__setattr__(
            self,
            "project_id",
            _required_project_identity(self.project_id),
        )
        object.__setattr__(
            self,
            "anchor_page_id",
            _required_identity(self.anchor_page_id, "anchor_page_id"),
        )
        object.__setattr__(self, "operation", GlossaryOperation(self.operation))
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, GlossaryEntryV1) for entry in self.entries
        ):
            raise TypeError("entries must be an exact GlossaryEntryV1 tuple")
        if not isinstance(self.entry_ids, tuple):
            raise TypeError("entry_ids must be an exact tuple")
        entry_ids = tuple(
            _required_identity(value, "entry_id") for value in self.entry_ids
        )
        object.__setattr__(self, "entry_ids", entry_ids)
        if self.operation is GlossaryOperation.SET_ENTRY:
            if len(self.entries) != 1 or self.entry_ids:
                raise ValueError("set_entry requires exactly one entry")
        elif self.operation is GlossaryOperation.REMOVE_ENTRY:
            if self.entries or not self.entry_ids:
                raise ValueError("remove_entry requires at least one entry ID")
        elif self.operation is GlossaryOperation.IMPORT_ENTRIES:
            if not self.entries or self.entry_ids:
                raise ValueError("import_entries requires at least one entry")
        requested_ids = tuple(entry.entry_id for entry in self.entries) or entry_ids
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("glossary command entry identities are duplicated")
        for field_name in (
            "expected_glossary_fingerprint",
            "expected_page_head_sha256",
            "expected_global_head_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class GlossaryCommandReceipt:
    command_id: str
    operation: GlossaryOperation
    edits: tuple[ProjectEdit, ...]
    superseded_edit_ids: tuple[str | None, ...]
    before_entries: tuple[GlossaryEntryV1, ...]
    after_entries: tuple[GlossaryEntryV1, ...]
    before_glossary_fingerprint: str
    after_glossary_fingerprint: str
    stale_page_ids: tuple[str, ...]
    before_issues: tuple[ProjectionIssue, ...]
    after_issues: tuple[ProjectionIssue, ...]
    invalidation: InvalidationResult
    effective_pages: tuple[EffectivePageSnapshot, ...]
    commit_receipt: ProjectEditCommitReceipt


def _validate_term_conflicts(entries: Iterable[GlossaryEntryV1]) -> None:
    owner_by_key: dict[str, str] = {}
    for entry in entries:
        for value in (entry.source, *entry.aliases):
            key = _term_key(value)
            owner = owner_by_key.get(key)
            if owner is not None and owner != entry.entry_id:
                raise GlossaryCommandError(
                    GlossaryCommandErrorCode.DUPLICATE_TERM,
                    "A source or alias is already owned by another glossary entry.",
                )
            owner_by_key[key] = entry.entry_id


def _entry_id_for_edit(edit: ProjectEdit) -> str:
    if edit.domain is not EditDomain.GLOSSARY:
        return ""
    if edit.operation == GlossaryOperation.SET_ENTRY.value:
        entry = edit.payload.get("entry")
        return str(entry.get("entry_id") or "") if isinstance(entry, Mapping) else ""
    if edit.operation == GlossaryOperation.REMOVE_ENTRY.value:
        return str(edit.payload.get("entry_id") or "")
    return ""


def _active_slot_head(
    ledger: ProjectEditLedger,
    *,
    project_id: str,
    entry_id: str,
) -> ProjectEdit | None:
    candidates = tuple(
        edit
        for edit in ledger.active_edits()
        if edit.domain is EditDomain.GLOSSARY
        and edit.target.kind is EditTargetKind.PROJECT
        and edit.project_id == project_id
        and _entry_id_for_edit(edit) == entry_id
    )
    if not candidates:
        return None
    candidate_ids = {edit.edit_id for edit in candidates}
    superseded = {
        edit.supersedes_edit_id
        for edit in candidates
        if edit.supersedes_edit_id in candidate_ids
    }
    heads = tuple(edit for edit in candidates if edit.edit_id not in superseded)
    if len(heads) != 1:
        raise GlossaryCommandError(
            GlossaryCommandErrorCode.SLOT_CONFLICT,
            "The glossary entry has competing active edits.",
        )
    return heads[0]


def _project_page(project: Mapping[str, Any], page_id: str) -> Mapping[str, Any]:
    pages = project.get("pages") or ()
    matches = tuple(
        page
        for page in pages
        if isinstance(page, Mapping)
        and str(page.get("page_id") or "").strip() == page_id
    )
    if len(matches) != 1:
        raise GlossaryCommandError(
            GlossaryCommandErrorCode.ANCHOR_PAGE_MISMATCH,
            "The canonical glossary anchor page is unavailable.",
        )
    return matches[0]


def _edit_ids(command: GlossaryCommand, count: int) -> tuple[str, ...]:
    if count == 1:
        return (command.command_id,)
    return tuple(
        _required_identity(f"{command.command_id}-{index + 1}", "edit_id")
        for index in range(count)
    )


def _stale_page_ids(snapshot: ProjectGlossarySnapshot) -> tuple[str, ...]:
    return tuple(
        page.page_id
        for page in snapshot.effective_pages
        if any(parent.target_text.strip() for parent in page.parents if not parent.excluded)
    )


def _assert_only_glossary_changed(
    before: ProjectGlossarySnapshot,
    after: ProjectGlossarySnapshot,
) -> None:
    if before.page_ids != after.page_ids:
        raise GlossaryCommandError(
            GlossaryCommandErrorCode.PROJECTION_REJECTED,
            "Project pages changed while applying the glossary edit.",
        )
    for before_page, after_page in zip(
        before.effective_pages,
        after.effective_pages,
        strict=True,
    ):
        if (
            before_page.automatic_fingerprint != after_page.automatic_fingerprint
            or before_page.base_revision_id != after_page.base_revision_id
            or before_page.cleaned_base_revision_id
            != after_page.cleaned_base_revision_id
            or before_page.cleaned_page_base != after_page.cleaned_page_base
            or before_page.cleaned_base_provenance
            != after_page.cleaned_base_provenance
            or before_page.hierarchy != after_page.hierarchy
            or before_page.parents != after_page.parents
            or before_page.stage_requirements != after_page.stage_requirements
        ):
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.PROJECTION_REJECTED,
                "A glossary edit changed unrelated effective page state.",
            )


class GlossaryCommandService:
    """Persist project glossary edits under one canonical page/global CAS."""

    def __init__(self, *, edit_store: ProjectEditStore) -> None:
        if not isinstance(edit_store, ProjectEditStore):
            raise TypeError("edit_store must be a ProjectEditStore")
        self._edit_store = edit_store

    def execute(
        self,
        *,
        project: Mapping[str, Any],
        command: GlossaryCommand,
    ) -> GlossaryCommandReceipt:
        if not isinstance(project, Mapping):
            raise TypeError("project must be a mapping")
        snapshot = self._edit_store.materialize_project_snapshot(
            project,
            page_id=command.anchor_page_id,
        )
        return self.execute_materialized(snapshot=snapshot, command=command)

    def execute_materialized(
        self,
        *,
        snapshot: ProjectEditReadSnapshot,
        command: GlossaryCommand,
    ) -> GlossaryCommandReceipt:
        if not isinstance(snapshot, ProjectEditReadSnapshot):
            raise TypeError("snapshot must be a ProjectEditReadSnapshot")
        if not isinstance(command, GlossaryCommand):
            raise TypeError("command must be a GlossaryCommand")
        if self._edit_store.project_id != command.project_id:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STORE_IDENTITY_MISMATCH,
                "Project edit store identity does not match the glossary command.",
            )
        if project_id_for(snapshot.project) != command.project_id:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.PROJECT_IDENTITY_MISMATCH,
                "Project identity does not match the glossary command.",
            )
        try:
            before = project_glossary_snapshot(snapshot.project, snapshot.ledger)
        except (KeyError, TypeError, ValueError) as exc:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.PROJECT_GLOSSARY_INVALID,
                "The current project glossary cannot be projected.",
            ) from exc
        if command.anchor_page_id != before.anchor_page_id:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.ANCHOR_PAGE_MISMATCH,
                "Glossary commands must use the first project page as anchor.",
            )
        if snapshot.page_head_sha256 != command.expected_page_head_sha256:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STALE_PAGE_HEAD,
                "The glossary anchor page changed after preparation.",
            )
        if snapshot.global_head_sha256 != command.expected_global_head_sha256:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed after glossary preparation.",
            )
        if before.fingerprint != command.expected_glossary_fingerprint:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STALE_GLOSSARY,
                "The effective project glossary changed after preparation.",
            )

        current = {entry.entry_id: entry for entry in before.entries}
        requested: list[tuple[str, GlossaryEntryV1 | None]] = []
        if command.operation in {
            GlossaryOperation.SET_ENTRY,
            GlossaryOperation.IMPORT_ENTRIES,
        }:
            for entry in command.entries:
                if current.get(entry.entry_id) != entry:
                    requested.append((entry.entry_id, entry))
                    current[entry.entry_id] = entry
        else:
            for entry_id in command.entry_ids:
                if entry_id in current:
                    requested.append((entry_id, None))
                    current.pop(entry_id)
        if not requested:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.NO_OP,
                "The glossary command makes no effective change.",
            )
        _validate_term_conflicts(current.values())

        edit_ids = _edit_ids(command, len(requested))
        if any(snapshot.ledger.get(edit_id) is not None for edit_id in edit_ids):
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.DUPLICATE_COMMAND,
                "The glossary command was already recorded.",
            )
        page = _project_page(snapshot.project, command.anchor_page_id)
        target = EditTarget(EditTargetKind.PROJECT)
        records: list[ProjectEdit] = []
        superseded: list[str | None] = []
        for edit_id, (entry_id, entry) in zip(edit_ids, requested, strict=True):
            operation = (
                GlossaryOperation.SET_ENTRY.value
                if entry is not None
                else GlossaryOperation.REMOVE_ENTRY.value
            )
            payload: dict[str, Any] = (
                {"entry": entry.to_dict()}
                if entry is not None
                else {"entry_id": entry_id}
            )
            base = field_base_fingerprint(
                project=snapshot.project,
                page=page,
                target=target,
                domain=EditDomain.GLOSSARY,
                operation=operation,
                payload=payload,
            )
            if base is None:
                raise GlossaryCommandError(
                    GlossaryCommandErrorCode.PROJECT_GLOSSARY_INVALID,
                    "The automatic glossary base is unavailable.",
                )
            head = _active_slot_head(
                snapshot.ledger,
                project_id=command.project_id,
                entry_id=entry_id,
            )
            superseded.append(head.edit_id if head is not None else None)
            records.append(
                create_project_edit(
                    project_id=command.project_id,
                    page_id=command.anchor_page_id,
                    target=target,
                    domain=EditDomain.GLOSSARY,
                    operation=operation,
                    payload=payload,
                    base_revision_id=f"automatic-project:{command.project_id}",
                    base_fingerprint=base,
                    supersedes_edit_id=(head.edit_id if head is not None else None),
                    edit_id=edit_id,
                )
            )

        candidate = snapshot.ledger
        try:
            for record in records:
                candidate = candidate.append(record)
            after = project_glossary_snapshot(snapshot.project, candidate)
        except GlossaryCommandError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.PROJECTION_REJECTED,
                "The candidate project glossary cannot be projected.",
            ) from exc
        expected_entries = tuple(sorted(current.values(), key=lambda value: value.entry_id))
        if after.entries != expected_entries or after.fingerprint == before.fingerprint:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.PROJECTION_REJECTED,
                "The candidate glossary does not match the requested state.",
            )
        _assert_only_glossary_changed(before, after)
        invalidation = invalidation_for_edit(records[0])
        if invalidation.unresolved_facts:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.INVALIDATION_REJECTED,
                "Glossary invalidation has unresolved dependency facts.",
            )
        try:
            commit_receipt = self._edit_store.commit_page_edits(
                tuple(records),
                automatic_page_sha256=automatic_page_fingerprint(page),
                expected_page_head_sha256=snapshot.page_head_sha256,
                expected_global_head_sha256=snapshot.global_head_sha256,
                transaction_id=command.command_id,
            )
        except StalePageEditHeadError as exc:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STALE_PAGE_HEAD,
                "The glossary anchor page changed before commit.",
            ) from exc
        except StaleProjectEditHeadError as exc:
            raise GlossaryCommandError(
                GlossaryCommandErrorCode.STALE_GLOBAL_HEAD,
                "Project edits changed before glossary commit.",
            ) from exc
        return GlossaryCommandReceipt(
            command_id=command.command_id,
            operation=command.operation,
            edits=tuple(records),
            superseded_edit_ids=tuple(superseded),
            before_entries=before.entries,
            after_entries=after.entries,
            before_glossary_fingerprint=before.fingerprint,
            after_glossary_fingerprint=after.fingerprint,
            stale_page_ids=_stale_page_ids(after),
            before_issues=tuple(
                issue for page_snapshot in before.effective_pages for issue in page_snapshot.issues
            ),
            after_issues=tuple(
                issue for page_snapshot in after.effective_pages for issue in page_snapshot.issues
            ),
            invalidation=invalidation,
            effective_pages=after.effective_pages,
            commit_receipt=commit_receipt,
        )


__all__ = [
    "GlossaryCommand",
    "GlossaryCommandError",
    "GlossaryCommandErrorCode",
    "GlossaryCommandReceipt",
    "GlossaryCommandService",
    "GlossaryEntryV1",
    "GlossaryOperation",
    "ProjectGlossarySnapshot",
    "glossary_entries_fingerprint",
    "project_glossary_snapshot",
]
