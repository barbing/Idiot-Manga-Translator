# -*- coding: utf-8 -*-
"""Append-only Project Edit Ledger."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping
import uuid

from .contracts import (
    LEDGER_SCHEMA_VERSION,
    EditDomain,
    EditTarget,
    EditTargetKind,
    ProjectEdit,
    create_project_edit,
    utc_now,
)


def _slot_key(record: ProjectEdit) -> tuple[Any, ...]:
    """Return the field-local identity used by explicit supersession."""

    if record.domain in {
        EditDomain.RENDER_STYLE,
        EditDomain.RENDER_LAYOUT,
        EditDomain.REVIEW_METADATA,
    }:
        fields = record.payload.get("fields")
        if isinstance(fields, Mapping):
            field_ids = tuple(sorted(str(field) for field in fields))
        elif isinstance(fields, tuple):
            field_ids = tuple(sorted(str(field) for field in fields))
        else:
            field_ids = ("*",)
    elif record.domain is EditDomain.STRUCTURAL:
        field_ids = {
            "add_user_parent": ("add_user_parent",),
            "split_user_parent": ("split_user_parent",),
            "exclude": ("excluded",),
            "restore": ("excluded",),
            "set_geometry": ("geometry",),
            "set_reading_order": ("reading_order",),
            "set_role": ("role",),
        }.get(record.operation, (record.operation,))
    elif record.domain is EditDomain.GLOSSARY:
        entry = record.payload.get("entry")
        if isinstance(entry, Mapping):
            field_ids = (str(entry.get("entry_id") or ""),)
        else:
            field_ids = (str(record.payload.get("entry_id") or ""),)
    else:
        field_ids = (record.domain.value,)
    scope_id = (
        record.project_id
        if record.domain is EditDomain.GLOSSARY
        else record.page_id
    )
    return (
        record.project_id,
        scope_id,
        tuple(sorted(record.target.to_dict().items())),
        record.domain.value,
        field_ids,
    )


def _canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LedgerState:
    active_edit_ids: tuple[str, ...]
    revoked_edit_ids: tuple[str, ...]


class ProjectEditLedger:
    """An immutable sequence of edit and state-transition records."""

    def __init__(
        self,
        edits: Iterable[ProjectEdit] = (),
        *,
        project_id: str = "",
        schema_version: str = LEDGER_SCHEMA_VERSION,
    ) -> None:
        if schema_version != LEDGER_SCHEMA_VERSION:
            raise ValueError(f"unsupported edit ledger schema: {schema_version}")
        records = tuple(edits)
        seen: set[str] = set()
        base_edits: set[str] = set()
        records_by_id: dict[str, ProjectEdit] = {}
        project_ids: set[str] = set()
        for record in records:
            if not isinstance(record, ProjectEdit):
                raise TypeError("ledger entries must be ProjectEdit records")
            if record.edit_id in seen:
                raise ValueError(f"duplicate edit ID: {record.edit_id}")
            seen.add(record.edit_id)
            project_ids.add(record.project_id)
            if record.is_control:
                target_edit_id = record.target.edit_id
                if target_edit_id not in base_edits:
                    raise ValueError(
                        f"ledger control targets an unknown or later edit: {target_edit_id}"
                    )
            else:
                base_edits.add(record.edit_id)
                if record.supersedes_edit_id and record.supersedes_edit_id not in base_edits:
                    raise ValueError(
                        "supersedes_edit_id must reference an earlier edit"
                    )
                if record.supersedes_edit_id:
                    superseded = records_by_id[record.supersedes_edit_id]
                    if _slot_key(record) != _slot_key(superseded):
                        raise ValueError(
                            "supersession must remain in the same field-local edit slot"
                        )
                records_by_id[record.edit_id] = record
        if len(project_ids) > 1:
            raise ValueError("one ledger cannot contain multiple project IDs")
        bound_project_id = str(project_id or "").strip()
        if project_ids:
            edit_project_id = next(iter(project_ids))
            if bound_project_id and bound_project_id != edit_project_id:
                raise ValueError("ledger project ID does not match its edit records")
            bound_project_id = edit_project_id
        self._edits = records
        self._schema_version = schema_version
        self._project_id = bound_project_id

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def edits(self) -> tuple[ProjectEdit, ...]:
        return self._edits

    @property
    def project_id(self) -> str:
        return self._project_id

    def append(self, edit: ProjectEdit) -> "ProjectEditLedger":
        if self.project_id and edit.project_id != self.project_id:
            raise ValueError("edit project ID does not match ledger project ID")
        return ProjectEditLedger(
            (*self._edits, edit),
            project_id=self._project_id or edit.project_id,
        )

    def _control(
        self,
        edit_id: str,
        *,
        operation: str,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> "ProjectEditLedger":
        target = self.get(edit_id)
        if target is None or target.is_control:
            raise ValueError(f"cannot {operation} unknown edit: {edit_id}")
        is_active = edit_id in set(self.state().active_edit_ids)
        if operation == "revoke" and not is_active:
            raise ValueError(f"edit is already revoked: {edit_id}")
        if operation == "reapply" and is_active:
            raise ValueError(f"edit is already active: {edit_id}")
        if operation == "revoke":
            dependants = self.dependent_active_edit_ids(edit_id)
            if dependants:
                raise ValueError(
                    "cannot revoke structural topology while later active edits "
                    f"depend on its parent/root identities: {list(dependants)}"
                )
        control = create_project_edit(
            project_id=target.project_id,
            page_id=target.page_id,
            target=EditTarget(EditTargetKind.EDIT, edit_id=target.edit_id),
            domain=EditDomain.LEDGER_CONTROL,
            operation=operation,
            payload={"edit_id": target.edit_id},
            base_revision_id=f"ledger:{self.fingerprint()}",
            base_fingerprint=self.fingerprint(),
            provenance="user",
            edit_id=event_id or str(uuid.uuid4()),
            created_at=created_at or utc_now(),
        )
        return self.append(control)

    def revoke(
        self,
        edit_id: str,
        *,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> "ProjectEditLedger":
        return self._control(
            edit_id,
            operation="revoke",
            event_id=event_id,
            created_at=created_at,
        )

    def reapply(
        self,
        edit_id: str,
        *,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> "ProjectEditLedger":
        return self._control(
            edit_id,
            operation="reapply",
            event_id=event_id,
            created_at=created_at,
        )

    def get(self, edit_id: str) -> ProjectEdit | None:
        for record in self._edits:
            if record.edit_id == edit_id:
                return record
        return None

    def dependent_active_edit_ids(self, edit_id: str) -> tuple[str, ...]:
        """Return later active records that reference created parent/root IDs.

        Dependency safety is intentionally structural and exact.  It does not
        infer identity from geometry and it includes records that are still
        active ledger events even when a later supersession masks their field.
        """

        target = self.get(edit_id)
        if target is None or target.is_control:
            return ()
        try:
            target_index = next(
                index
                for index, record in enumerate(self._edits)
                if record.edit_id == target.edit_id
            )
        except StopIteration:
            return ()
        active_ids = set(self.state().active_edit_ids)
        if (
            target.domain is EditDomain.STRUCTURAL
            and target.operation == "add_user_parent"
            and target.target.kind is EditTargetKind.PARENT
        ):
            identities = {
                target.target.parent_id,
                str(target.payload.get("root_id") or ""),
            }
        elif (
            target.domain is EditDomain.STRUCTURAL
            and target.operation == "split_user_parent"
            and target.target.kind is EditTargetKind.PARENT
        ):
            identities = {
                str(value)
                for value in (
                    *(target.payload.get("child_parent_ids") or ()),
                    *(target.payload.get("child_root_ids") or ()),
                )
                if str(value)
            }
        elif (
            target.domain is EditDomain.STRUCTURAL
            and target.operation == "merge_pipeline_parents"
            and target.target.kind is EditTargetKind.PARENT
        ):
            identities = {
                target.target.parent_id,
                str(target.payload.get("merged_root_id") or ""),
            }
        else:
            result: list[str] = []
            for record in self._edits[target_index + 1 :]:
                if (
                    record.is_control
                    or record.edit_id not in active_ids
                    or record.domain is not EditDomain.STRUCTURAL
                    or record.target.kind is not EditTargetKind.PARENT
                ):
                    continue
                if record.operation == "split_user_parent":
                    source_identities = {record.target.parent_id}
                elif record.operation == "merge_pipeline_parents":
                    source_identities = {
                        str(value)
                        for value in record.payload.get("source_parent_ids") or ()
                        if str(value)
                    }
                else:
                    continue
                if (
                    target.target.kind is EditTargetKind.PARENT
                    and target.target.parent_id in source_identities
                ) or _json_references_any(target.payload, source_identities):
                    result.append(record.edit_id)
            return tuple(result)
        result: list[str] = []
        for record in self._edits[target_index + 1 :]:
            if record.is_control or record.edit_id not in active_ids:
                continue
            target_values = {
                record.target.parent_id,
                record.target.artifact_id,
                record.target.edit_id,
            }
            if identities.intersection(target_values) or _json_references_any(
                record.payload,
                identities,
            ):
                result.append(record.edit_id)
        return tuple(result)

    def state(self) -> LedgerState:
        active: dict[str, bool] = {}
        order: list[str] = []
        for record in self._edits:
            if record.is_control:
                active[record.target.edit_id] = record.operation == "reapply"
                continue
            active[record.edit_id] = record.active
            order.append(record.edit_id)
        return LedgerState(
            active_edit_ids=tuple(edit_id for edit_id in order if active[edit_id]),
            revoked_edit_ids=tuple(edit_id for edit_id in order if not active[edit_id]),
        )

    def active_edits(self, *, page_id: str | None = None) -> tuple[ProjectEdit, ...]:
        active_ids = set(self.state().active_edit_ids)
        return tuple(
            record
            for record in self._edits
            if not record.is_control
            and record.edit_id in active_ids
            and (page_id is None or record.page_id == page_id)
        )

    def records_for_page(self, page_id: str) -> tuple[ProjectEdit, ...]:
        page_id = str(page_id or "").strip()
        if not page_id:
            raise ValueError("page_id is required")
        return tuple(record for record in self._edits if record.page_id == page_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self._schema_version,
            "project_id": self._project_id,
            "edits": [record.to_dict() for record in self._edits],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectEditLedger":
        if not isinstance(value, Mapping):
            raise TypeError("edit ledger must be a mapping")
        unknown = frozenset(value) - {"schema_version", "project_id", "edits"}
        if unknown:
            raise ValueError(f"edit ledger has unsupported fields: {sorted(unknown)}")
        edits = value.get("edits")
        if not isinstance(edits, list):
            raise ValueError("edit ledger edits must be a list")
        return cls(
            (ProjectEdit.from_persisted_dict(item) for item in edits),
            project_id=str(value.get("project_id") or ""),
            schema_version=str(value.get("schema_version") or ""),
        )

    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def _json_references_any(value: Any, identities: set[str]) -> bool:
    if isinstance(value, str):
        return value in identities
    if isinstance(value, Mapping):
        return any(_json_references_any(item, identities) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_json_references_any(item, identities) for item in value)
    return False
